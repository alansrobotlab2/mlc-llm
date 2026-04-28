"""Run the Qwen3.5 MTP head in pure PyTorch to determine the correct
position convention for self-spec decode.

Loads the HF model + the `mtp.*` weights separately, runs target greedy
generation, then for each generation step probes:
  - candidate A: input (h_n, embed(T_{n+1})) → predicts T_{n+2}
  - candidate B: input (h_n, embed(T_n))     → predicts T_{n+1}
  - candidate C: input (h_{n-1}, embed(T_n)) → predicts T_{n+1}

Prints, per step, which (if any) candidate's argmax matches the actual
target-greedy token. The convention with the highest hit-rate is the
correct one to wire into MLC.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_PATH = "/home/alfie/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"


class RMSNorm(nn.Module):
    """Matches HF Qwen3_5RMSNorm: output * (1.0 + weight)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        f = x.float()
        rms = (f.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return ((f * rms) * (1.0 + self.weight.float())).to(x.dtype)


class MTPHead(nn.Module):
    """Self-spec MTP head: (h, e) → fc → 1× DecoderLayer → norm.

    No lm_head: target's tied lm_head is applied externally on the output.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, rms_eps, rotary_dim, rope_theta, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size

        # MTP fuse layer
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, rms_eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, rms_eps)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        # Single decoder layer
        self.input_layernorm = RMSNorm(hidden_size, rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_eps)

        # attn (with output gate, head_dim 256, 8 heads, 2 kv_heads for 0.8B)
        self.q_proj = nn.Linear(hidden_size, 2 * num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_eps)
        self.k_norm = RMSNorm(head_dim, rms_eps)

        # MLP
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Final norm
        self.final_norm = RMSNorm(hidden_size, rms_eps)

    def _rope(self, x, position_ids):
        # Apply rotary on first `rotary_dim` channels of head dim.
        b, h, s, d = x.shape
        rot, rest = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        # cos/sin
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=x.device)
            / self.rotary_dim))
        positions = position_ids.float()  # (s,)
        freqs = positions[:, None] * inv_freq[None, :]  # (s, rotary_dim/2)
        cos = freqs.cos().to(x.dtype)[None, None, :, :]
        sin = freqs.sin().to(x.dtype)[None, None, :, :]
        # rotate_half on rot
        x1 = rot[..., :self.rotary_dim // 2]
        x2 = rot[..., self.rotary_dim // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        # broadcast cos/sin to rotary_dim
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)
        rot_out = rot * cos_full + rotated * sin_full
        return torch.cat([rot_out, rest], dim=-1)

    def forward_batch(self, h_seq, e_seq):
        """Batch forward over a full sequence.

        h_seq: (B, S, hidden) — "prev_hidden" inputs at each position.
        e_seq: (B, S, hidden) — "prev_embed" inputs at each position.
        position_ids run 0..S-1 (RoPE positions).

        Returns (B, S, hidden) — MTP output at each position. Causal attention.
        """
        b, s, _ = h_seq.shape
        device = h_seq.device

        # Fuse step
        h_norm = self.pre_fc_norm_hidden(h_seq)
        e_norm = self.pre_fc_norm_embedding(e_seq)
        x = self.fc(torch.cat([h_norm, e_norm], dim=-1))  # (B, S, hidden)

        # Decoder layer
        residual = x
        n = self.input_layernorm(x)

        # attn projections
        q_gate = self.q_proj(n).reshape(b, s, self.num_heads, 2 * self.head_dim)
        q, gate = q_gate.split([self.head_dim, self.head_dim], dim=-1)
        gate = gate.reshape(b, s, self.num_heads * self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # (B, H, S, D)

        k = self.k_proj(n).reshape(b, s, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (B, KV, S, D)
        v = self.v_proj(n).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # MTP at MTP-pos n consumes (h_n, e_{n+1}) — the "virtual current token"
        # is at sequence position n+1, so RoPE positions should start at 1.
        position_ids = torch.arange(s, device=device) + getattr(self, "rope_offset", 1)
        q = self._rope(q, position_ids)
        k = self._rope(k, position_ids)

        # GQA
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Causal attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, S, S)
        causal = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        out = self.o_proj(out)
        x = residual + out

        # MLP
        residual = x
        n = self.post_attention_layernorm(x)
        gate = F.silu(self.gate_proj(n))
        up = self.up_proj(n)
        x = residual + self.down_proj(gate * up)

        # Final MTP norm
        return self.final_norm(x)


def load_mtp_weights(mtp_head, base_path):
    """Load mtp.* weights from HF safetensors into our PyTorch MTPHead."""
    import json
    idx = json.load(open(os.path.join(base_path, "model.safetensors.index.json")))["weight_map"]

    name_map = {
        "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
        "fc.weight": "mtp.fc.weight",
        "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
        "post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
        "k_proj.weight": "mtp.layers.0.self_attn.k_proj.weight",
        "v_proj.weight": "mtp.layers.0.self_attn.v_proj.weight",
        "o_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        "q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "gate_proj.weight": "mtp.layers.0.mlp.gate_proj.weight",
        "up_proj.weight": "mtp.layers.0.mlp.up_proj.weight",
        "down_proj.weight": "mtp.layers.0.mlp.down_proj.weight",
        "final_norm.weight": "mtp.norm.weight",
    }
    state = {}
    for mlc_name, hf_name in name_map.items():
        fname = idx[hf_name]
        with safe_open(os.path.join(base_path, fname), framework="pt") as f:
            state[mlc_name] = f.get_tensor(hf_name).float()
    mtp_head.load_state_dict(state, strict=True)
    print(f"[mtp] loaded {len(state)} weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    print(f"[mtp] loading HF target on {device}")
    tok = AutoTokenizer.from_pretrained(HF_PATH)
    target = AutoModelForCausalLM.from_pretrained(HF_PATH, dtype=torch.float16, device_map=device)
    target = target.half()  # transformers 5.6 ignores dtype= kwarg, force fp16
    target.eval()

    cfg = target.config.text_config if hasattr(target.config, "text_config") else target.config
    mtp = MTPHead(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_eps=cfg.rms_norm_eps,
        rotary_dim=int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"]),
        rope_theta=cfg.rope_parameters["rope_theta"],
        vocab_size=cfg.vocab_size,
    ).float()
    load_mtp_weights(mtp, HF_PATH)
    mtp = mtp.half().to(device).eval()

    # Tokenize and run target greedy with hidden_states output
    chat = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                    tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt").input_ids.to(device)
    print(f"[mtp] prompt tokens: {ids.shape[1]}")

    # Get target's hidden states for the prompt
    with torch.no_grad():
        out = target(ids, output_hidden_states=True)
        # out.hidden_states is a tuple of (num_layers+1) tensors, each (1, S, hidden)
        # The LAST one is post-final-norm? Or pre-final-norm? Let me check.
        # Actually HF returns the output of EACH layer; the final norm is applied by
        # LM head after model.forward. So the last hidden_state IS post-each-layer-norm
        # but PRE-final-norm. The model also has a final norm at the end (model.norm).
        # We need the post-final-norm hidden, which is what the model returns as
        # "last_hidden_state" (out[0] for CausalLM is logits; need .hidden_states[-1] which is
        # the OUTPUT of the model.norm).
        last_hidden = out.hidden_states[-1]  # (1, S, hidden) post-final-norm
        # Get target's greedy next tokens
        logits = out.logits  # (1, S, V) — based on lm_head(post-final-norm)
        target_tokens = logits[0].argmax(dim=-1)  # (S,) — at position i, the predicted next token

    print(f"[mtp] hidden state shape: {tuple(last_hidden.shape)}")
    print(f"[mtp] target argmax tokens (predictions): {target_tokens.tolist()[:10]}...")

    # Continue greedy generation for n_tokens, capturing hidden states
    generated_ids = ids[0].tolist()
    hidden_states_per_pos = list(last_hidden[0])  # list of (hidden,) tensors

    for step in range(args.n_tokens):
        # Greedy: sample argmax of last position's logits
        next_tok = target_tokens[-1].item()
        generated_ids.append(next_tok)
        # Re-run target with the appended token to get its hidden state
        ids = torch.tensor([generated_ids], device=device)
        with torch.no_grad():
            out = target(ids, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
            target_tokens = out.logits[0].argmax(dim=-1)
        hidden_states_per_pos = list(last_hidden[0])

    print(f"[mtp] full token sequence: {generated_ids[-args.n_tokens-1:]}")
    print(f"[mtp] decoded: {tok.decode(generated_ids[-args.n_tokens-1:])!r}")

    # Now probe MTP head with three position conventions.
    # We have hidden[0..N-1] and tokens[0..N-1] (where N = len(generated_ids))
    embed = target.get_input_embeddings()
    ground_truth = generated_ids  # T_n at index n

    print()
    print("=" * 80)
    print("Per-position MTP batch probe (causal, halved precision)")
    print("=" * 80)

    # Stack hiddens and embeddings as full tensors over the generated sequence.
    # tokens[0..L-1], hiddens[0..L-1] (where hiddens[i] = post-final-norm output after target processed token i)
    L = len(generated_ids)
    hiddens = torch.stack([h.detach() for h in hidden_states_per_pos[:L]]).unsqueeze(0)  # (1, L, h)
    tokens_t = torch.tensor(generated_ids, device=device).unsqueeze(0)  # (1, L)
    embeds = embed(tokens_t).detach()  # (1, L, h)

    # Convention A: (h_seq, e_seq=embeds_shifted_left_by_1) — i.e. at pos n, MTP gets (h_n, e_{n+1})
    #   Predicts: should be T_{n+2}.
    # Convention B: (h_seq, e_seq=embeds) — same-position pair (h_n, e_n).
    #   Predicts: target convention says T_{n+1}.
    # Convention C: (h_seq_shifted_left_by_1, e_seq=embeds[1:]) — at pos n', MTP gets (h_{n'+0}, e_{n'+1}) where n'=n-1.
    #   Equivalent to: at MTP pos n, input (h_{n-1}, e_n), predicts target convention T_{n+1}.

    def probe(h_seq, e_seq, label):
        with torch.no_grad():
            out = mtp.forward_batch(h_seq, e_seq)
            logits = target.lm_head(out)  # (1, S, V)
        return logits

    # We use hiddens[:L-1] paired with various embed shifts to ensure same-shape tensors.
    h_full = hiddens[:, :L-1, :]  # positions 0..L-2
    e_same = embeds[:, :L-1, :]   # positions 0..L-2 (matched)
    e_shift = embeds[:, 1:L, :]    # positions 1..L-1 (shifted left by 1)

    logits_B = probe(h_full, e_same, "B")     # at MTP pos n: (h_n, e_n)
    logits_A = probe(h_full, e_shift, "A")    # at MTP pos n: (h_n, e_{n+1})
    h_shift = hiddens[:, 0:L-1, :]            # positions 0..L-2 (same as h_full, but interpret differently)
    # For C: at MTP pos n, want (h_{n-1}, e_n). So we use h_seq=hiddens[:, :L-2] (positions 0..L-3) and
    # e_seq=embeds[:, 1:L-1] (positions 1..L-2). MTP pos i in this batch corresponds to original position i+1.
    h_C = hiddens[:, :L-2, :]
    e_C = embeds[:, 1:L-1, :]
    logits_C = probe(h_C, e_C, "C")

    # Now compare argmax at each MTP position against the actual T_{n+1} or T_{n+2}.
    pred_A = logits_A.argmax(dim=-1)[0].tolist()  # length L-1, at MTP pos n is prediction
    pred_B = logits_B.argmax(dim=-1)[0].tolist()
    pred_C = logits_C.argmax(dim=-1)[0].tolist()

    # Target's pure next-token prediction at each position: lm_head(h_n) → T_{n+1}_pred.
    with torch.no_grad():
        target_pure_logits = target.lm_head(hiddens[0])  # (L, V)
        target_pure_pred = target_pure_logits.argmax(dim=-1).tolist()

    print(f"{'n':>4} {'T_n':>10} {'T_{n+1}':>10} {'T_{n+2}':>10} {'tgt_pure':>10} | "
          f"{'A pred':>10} {'==T_{n+2}?':>11} {'==T_{n+1}?':>11} | "
          f"{'B pred':>10} {'==T_{n+1}?':>11} | "
          f"{'C pred':>10} {'==T_{n+1}?':>11}")

    hits_A_np2, hits_A_np1, hits_B_np1, hits_C_np1 = 0, 0, 0, 0
    total = 0
    start_n = max(1, L - args.n_tokens - 4)
    end_n = L - 2
    for n in range(start_n, end_n):
        t_n = ground_truth[n]
        t_np1 = ground_truth[n + 1]
        t_np2 = ground_truth[n + 2] if n + 2 < L else -1
        # MTP convention A is at pos n, predicts T_{n+2}
        # logits_A[0, n] is the prediction at MTP pos n where input was (h_n, e_{n+1})
        a = pred_A[n]
        b = pred_B[n]
        # For C: at MTP pos i (in the C batch), input was (h_i, e_{i+1}), predicting in convention C: T_{i+2}
        # If we wanted (h_{n-1}, e_n) for original n, that's MTP pos n-1 in C batch.
        c = pred_C[n - 1] if n - 1 < len(pred_C) else -1

        tgt_pure = target_pure_pred[n]
        match_A_np2 = "Y" if a == t_np2 else "N"
        match_A_np1 = "Y" if a == t_np1 else "N"
        match_B_np1 = "Y" if b == t_np1 else "N"
        match_C_np1 = "Y" if c == t_np1 else "N"
        hits_A_np2 += match_A_np2 == "Y"
        hits_A_np1 += match_A_np1 == "Y"
        hits_B_np1 += match_B_np1 == "Y"
        hits_C_np1 += match_C_np1 == "Y"
        total += 1
        print(f"{n:>4} {repr(tok.decode([t_n]))[:10]:>10} {repr(tok.decode([t_np1]))[:10]:>10} "
              f"{repr(tok.decode([t_np2])) if t_np2!=-1 else '-':>10} "
              f"{repr(tok.decode([tgt_pure]))[:10]:>10} | "
              f"{repr(tok.decode([a]))[:10]:>10} {match_A_np2:>11} {match_A_np1:>11} | "
              f"{repr(tok.decode([b]))[:10]:>10} {match_B_np1:>11} | "
              f"{repr(tok.decode([c])) if c!=-1 else '-':>10} {match_C_np1:>11}")

    print()
    print(f"Hit rates over {total} positions:")
    print(f"  A (h_n, e_{{n+1}}) pred matches T_{{n+2}} (DeepSeek MTP convention): {hits_A_np2}/{total}")
    print(f"  A (h_n, e_{{n+1}}) pred matches T_{{n+1}} (1-step shift):            {hits_A_np1}/{total}")
    print(f"  B (h_n, e_n)       pred matches T_{{n+1}} (EAGLE-1 convention):    {hits_B_np1}/{total}")
    print(f"  C (h_{{n-1}}, e_n) pred matches T_{{n+1}} (EAGLE-2/inverted):       {hits_C_np1}/{total}")

    # Hypothesis: MTP_output(h_n, e_{n+1}) ≈ target's h_{n+1}.
    # If true, MTP serves as a cheap replacement for running target on T_{n+1}.
    # We can chain MTP autoregressively to draft multiple tokens.
    print()
    print("=" * 80)
    print("Hypothesis: does MTP_out(h_n, e_{n+1}) ≈ target's h_{n+1}?")
    print("=" * 80)
    # logits_A is at MTP pos n: (h_n, e_{n+1}) → out
    # We want to compare mtp.forward_batch(h_n, e_{n+1}) hidden output to hiddens[n+1].
    with torch.no_grad():
        mtp_out = mtp.forward_batch(h_full, e_shift)  # at MTP pos n (idx n), output should ≈ h_{n+1}
    # Compare per-position cosine similarity
    target_next = hiddens[:, 1:L, :]  # h_{n+1} for n=0..L-2
    cos = F.cosine_similarity(mtp_out, target_next, dim=-1)[0]  # (L-1,)
    diff_norm = (mtp_out - target_next).norm(dim=-1)[0] / target_next.norm(dim=-1)[0]
    print(f"{'n':>4} {'cos_sim(mtp, h_{n+1})':>22} {'rel_diff':>10}")
    for n in range(start_n, end_n):
        print(f"{n:>4} {cos[n].item():>22.4f} {diff_norm[n].item():>10.4f}")


if __name__ == "__main__":
    main()
