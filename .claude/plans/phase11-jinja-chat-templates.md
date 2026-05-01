# Phase 11 — Native Jinja chat-template evaluation (vendor llama.cpp's engine)

> ## 🔖 Session resume notes (read first)
>
> **Status as of 2026-05-01 (kickoff):** Plan written. Stage 0 audit complete in-conversation — see "What's already in tree" and "Reference target" sections below. No code changes yet. Parallel/independent of Phase 10 ([phase10-vision-input.md](phase10-vision-input.md)); the only intersection is the `qwen3_5_vl` template stub (Phase 10 Stage 5) — if Phase 11 ships first, that stub deletes itself because the model's own `chat_template` handles vision-token slots natively.
>
> **What this is:** vendor [common/jinja/](file:///home/alfie/llama.cpp/common/jinja/) from llama.cpp into `3rdparty/jinja-llamacpp/`, replace MLC's hand-written `Conversation::CreatePrompt()` ([cpp/json_ffi/conv_template.cc](../../cpp/json_ffi/conv_template.cc)) with native Jinja evaluation of the model's `tokenizer_config.json` chat_template, and expose `enable_thinking` / `chat_template_kwargs` / `reasoning_content` through the OpenAI-compat API to match llama.cpp/vLLM/SGLang.
>
> **What this is NOT:** option B (the API-surface-only patch that swaps prebuilt templates per request). Option B was rejected in favor of the structural fix because every future model that ships a Jinja chat_template (i.e. all of them) works on day one once C-1 lands.
>
> **Headline gate:** Qwen3.5-0.8B end-to-end greedy-decode parity vs HF transformers with `enable_thinking={true,false}` toggled per request, ≥ 48/50 token match in both modes. Same bar as CLAUDE.md Stage 5.

---

**Date opened:** 2026-05-01
**Predecessor:** None directly — this is structural cleanup motivated by user request to expose Qwen3.5/3.6 thinking-mode toggle. Runs in parallel with [phase10-vision-input.md](phase10-vision-input.md).
**Conversation reference:** in-session research on llama.cpp Jinja engine, Qwen team's `enable_thinking` spec, and MLC's `Conversation` blast radius (this transcript, 2026-05-01).

**Goal:** evaluate the model's own `tokenizer_config.json` chat_template natively in C++ at request time, so MLC tracks every upstream chat-template change for free and exposes the same per-request controls (`enable_thinking`, `chat_template_kwargs`, `reasoning_content` split) as llama.cpp/vLLM.

**Why now:**
- User-visible: Qwen3.5/3.6 thinking-mode toggle is currently build-time only ([python/mlc_llm/conversation_template/qwen3_5.py](../../python/mlc_llm/conversation_template/qwen3_5.py)) — clients can't toggle per request, can't see `reasoning_content` separately, history handling is wrong (`strip_reasoning_in_history=False` is opposite of Qwen's rolling-checkpoint spec).
- Structural: 30 hand-written templates in [python/mlc_llm/conversation_template/](../../python/mlc_llm/conversation_template/) (~1.3K LOC) are continual maintenance debt; every new model family needs a hand-port. Native Jinja eliminates the class.
- Strategic: every other serious inference engine (llama.cpp, vLLM, SGLang) evaluates Jinja directly. Staying off-path means we keep losing parity races.

**Non-goals:**
- Input-marking security feature (llama.cpp's `jinja::string::is_input` flag for special-token injection defense). Useful, but additive — schedule as Stage 8 / post-v1.
- Generalized `<think>` parser for non-Qwen reasoning models (DeepSeek-R1, Hermes-3 Reasoning, Apriel-Thinker etc.). v1 ships Qwen-only `reasoning_content` extraction; generalize when a second model needs it.
- Migrating off TVM FFI JSON. nlohmann::json gets vendored only as a boundary adapter for the Jinja runtime; MLC's own JSON path stays on TVM FFI.
- Deleting the 30 Python templates in v1. Stage 7 deprecates them on a release boundary; v1 keeps them as a fallback path for old `mlc-chat-config.json` artifacts in the wild.

---

## What's already in tree

Audit (2026-05-01) of MLC's chat-template plumbing:

| component | location | state |
|---|---|---|
| C++ Conversation struct + `CreatePrompt()` | [cpp/json_ffi/conv_template.h](../../cpp/json_ffi/conv_template.h), [cpp/json_ffi/conv_template.cc:224](../../cpp/json_ffi/conv_template.cc#L224) | ~620 LOC; the entire request-time chat-template path. Replace, but keep alive behind a feature gate for legacy configs |
| C++ engine call site | [cpp/json_ffi/json_ffi_engine.cc:81](../../cpp/json_ffi/json_ffi_engine.cc#L81) | one call into `CreatePrompt()`; needs to route to Jinja path when `chat_template` is present, legacy path otherwise |
| MLC's JSON layer | [cpp/support/json_parser.h](../../cpp/support/json_parser.h) | uses TVM FFI JSON (`tvm/ffi/extra/json.h`); **not** nlohmann. Adapter required at the Jinja boundary |
| Python protocol | [python/mlc_llm/protocol/conversation_protocol.py](../../python/mlc_llm/protocol/conversation_protocol.py) | 278 LOC; `Conversation.as_prompt` (~90 LOC at [:120](../../python/mlc_llm/protocol/conversation_protocol.py#L120)) is the Python-side mirror of `CreatePrompt`. Becomes legacy/passthrough |
| Conversation templates | [python/mlc_llm/conversation_template/](../../python/mlc_llm/conversation_template/) | 30 files, ~1.3K LOC across qwen3, qwen3_5, qwen2, deepseek (×5), llama, mistral, gemma, phi, glm, cohere, … . Stage 7 deletion target |
| Engine `as_prompt` consumer | [python/mlc_llm/serve/engine_base.py](../../python/mlc_llm/serve/engine_base.py) | 1278 LOC; reads `mlc_chat_config.conv_template`, calls `as_prompt()`, uses `stop_token_ids` / `system_prefix_token_ids`. ~300 LOC of rewiring for the Jinja path |
| `gen_config` template embedding | [python/mlc_llm/interface/gen_config.py](../../python/mlc_llm/interface/gen_config.py) | embeds full `Conversation` struct into `mlc-chat-config.json` at build time. Stage 4 changes it to emit `chat_template` string + `chat_template_kwargs` defaults instead |
| MLCChatConfig schema | [python/mlc_llm/protocol/mlc_chat_config.py:54](../../python/mlc_llm/protocol/mlc_chat_config.py#L54) | `conv_template: Conversation` field. Stage 4 makes optional, adds `chat_template: Optional[str]` and `chat_template_kwargs: Optional[Dict]` siblings |
| OpenAI API protocol | [python/mlc_llm/protocol/openai_api_protocol.py](../../python/mlc_llm/protocol/openai_api_protocol.py) | no `enable_thinking`, no `chat_template_kwargs`, no `reasoning_content`. Stage 5 adds all three |
| C++ Conversation tests | [tests/cpp/conv_template_unittest.cc](../../tests/cpp/conv_template_unittest.cc) | 50+ cases of `FromJSON()` and `CreatePrompt()`. Stage 6 ports to Jinja path |
| Python template tests | [tests/python/conversation_template/](../../tests/python/conversation_template/) | parametrized over the 30 templates. Stage 6 retains for legacy-fallback regression coverage; deleted with templates in Stage 7 |
| Mobile/web bindings | [android/](../../android/), [ios/](../../ios/), [web/](../../web/) | no `conv_template` references — all chat-template work happens in the C++ engine. **Free ride** for the refactor; only impact is binary-size bump from added Jinja sources |

**No existing C++ Jinja in tree.** [3rdparty/](../../3rdparty/) has tvm, tokenizers-cpp, xgrammar, argparse, googletest, stb — nothing template-shaped. nlohmann::json absent.

---

## Reference target — llama.cpp's in-tree Jinja engine

Source: [/home/alfie/llama.cpp/common/jinja/](file:///home/alfie/llama.cpp/common/jinja/), introduced in PR#18462 ([README.md:3](file:///home/alfie/llama.cpp/common/jinja/README.md#L3)).

| file | LOC | role |
|---|---|---|
| `caps.{h,cpp}` | 32 + 480 | chat-template capability detection (used to drive `reasoning_format` heuristics) |
| `lexer.{h,cpp}` | 157 + 341 | source → token stream; predictive parser, no preprocessing (preserves source-line info for errors) |
| `parser.{h,cpp}` | 21 + 602 | tokens → `jinja::program` AST |
| `runtime.{h,cpp}` | 646 + 906 | recursive `execute(ctx)` interpreter over the AST |
| `value.{h,cpp}` | 756 + 1480 | primitive types (int/float/bool/string/array/object/none/undefined) + builtins; `shared_ptr` wrapping for sharing |
| `string.{h,cpp}` | 61 + 213 | `jinja::string` with `is_input` flag for injection defense (off by default) |
| `utils.h` | 149 | helpers |
| **total** | **5,932** | — |

**License:** MIT, identical to MLC ([llama.cpp/LICENSE](file:///home/alfie/llama.cpp/LICENSE), "Copyright (c) 2023-2026 The ggml authors"). Vendoring is friction-free.

**External deps inside the engine** (the integration surface to MLC):
- `nlohmann/json.hpp` — used in [caps.cpp:8](file:///home/alfie/llama.cpp/common/jinja/caps.cpp#L8) and [value.cpp:6](file:///home/alfie/llama.cpp/common/jinja/value.cpp#L6) for `global_from_json` and capability JSON. README.md states explicitly: "Decoupled from `nlohmann::json`: this dependency is only used for JSON-to-internal type translation and is **completely optional**." Two paths: (a) vendor `nlohmann/json.hpp` (single header, MIT, ~24K LOC) into `3rdparty/nlohmann/`; (b) write a TVM-FFI-JSON → `jinja::value` adapter and skip nlohmann entirely (~150 LOC). **Decision: option (a) for v1** — minimal code surface; nlohmann is a single header so the maintenance cost is zero. Revisit if the build-size impact on mobile becomes painful (see R-6).
- `unicode.h` — used in [value.cpp:2](file:///home/alfie/llama.cpp/common/jinja/value.cpp#L2). llama.cpp's unicode helper. Vendor a minimal subset (case-folding for `lower`/`upper` filters) into `3rdparty/jinja-llamacpp/shim/unicode.h`. Most Jinja templates only use ASCII for the case filters; a 200-LOC shim covers it.
- `log.h` — used in [caps.cpp:1](file:///home/alfie/llama.cpp/common/jinja/caps.cpp#L1). One-line shim mapping llama.cpp's `LOG_*` macros to TVM's `LOG()` / `DLOG()`.

**Qwen3.5 coverage confirmed:**
- Test fixtures shipped: [Qwen-Qwen3-0.6B.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja), [Qwen3.5-4B.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen3.5-4B.jinja), [Qwen3-Coder.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen3-Coder.jinja), [Qwen-Qwen2.5-7B-Instruct.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen-Qwen2.5-7B-Instruct.jinja), [Qwen-QwQ-32B.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen-QwQ-32B.jinja).
- Test suite: [tests/test-jinja.cpp](file:///home/alfie/llama.cpp/tests/test-jinja.cpp) (2,514 LOC unit) + [tests/test-chat-template.cpp](file:///home/alfie/llama.cpp/tests/test-chat-template.cpp) (712 LOC integration).
- Qwen3's rolling-checkpoint history strategy (`loop.index0 > ns.last_query_index`) is in the Jinja template itself ([Qwen-Qwen3-0.6B.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja) ~L31-50) — runs natively against this engine, no special MLC handling needed. This obsoletes the `strip_reasoning_in_history` flag MLC carries today.

**Integration pattern to mirror** ([common/chat.cpp:762-794](file:///home/alfie/llama.cpp/common/chat.cpp#L762-L794)):
```cpp
jinja::context ctx(tmpl.source());
jinja::global_from_json(ctx, inp, inputs.mark_input);
jinja::runtime runtime(ctx);
const jinja::value results = runtime.execute(tmpl.prog);
auto parts = jinja::runtime::gather_string_parts(results);
```
The `tmpl.prog` AST is parsed once at model-load and cached on the `Conversation` (or its successor `JinjaTemplate` struct) for the model's lifetime. Per-request work is just `execute()` + `gather_string_parts()`.

---

## Stages — gates & exit criteria

### Stage 0 — Audit & plan (this session)

- [x] Confirmed llama.cpp moved off vendored `minja.hpp` to in-tree `common/jinja/` engine; license + dep surface mapped.
- [x] Confirmed Qwen3.5 templates (including thinking-mode rolling-checkpoint history) covered by their test fixtures.
- [x] Mapped MLC blast radius: C++ runtime, Python protocol, gen_config, engine consumers, tests, mobile/web (no impact).
- [x] Plan doc written ([this file](phase11-jinja-chat-templates.md)).
- [ ] Worklog kickoff entry.

**Gate:** plan committed; worklog has Phase 11 entry.

### Stage 1 — Vendor the Jinja engine, get it building

1. Copy `common/jinja/{caps,lexer,parser,runtime,value,string,utils}.{h,cpp}` (drop `README.md`) into `3rdparty/jinja-llamacpp/` preserving the directory layout.
2. Vendor `nlohmann/json.hpp` (single header from upstream nlohmann/json release tag) into `3rdparty/nlohmann/`.
3. Write shims:
   - `3rdparty/jinja-llamacpp/shim/log.h` — map `LOG_*` macros to TVM `LOG()`.
   - `3rdparty/jinja-llamacpp/shim/unicode.h` — minimal case-folding (`tolower`/`toupper` UTF-8 aware enough for the case filters; ASCII-only fallback acceptable for v1, document the limitation).
4. Add the new sources to [cpp/CMakeLists.txt](../../cpp/CMakeLists.txt) — new static lib `jinja_llamacpp`, linked into the existing `mlc_llm` shared lib.
5. Verify it builds standalone on Orin (CUDA 12, GCC 11) and on the macOS dev box. No symbol collisions with existing code.

**Gate (1):** `cmake --build build --target jinja_llamacpp` succeeds on Orin and macOS. No-op smoke binary that includes `<jinja/runtime.h>` and exits 0 links cleanly. CI matrix unchanged.

### Stage 2 — Standalone smoke: render Qwen3.5 chat_template byte-equal to llama.cpp

New unit test [tests/cpp/jinja_smoke_unittest.cc](../../tests/cpp/):
1. Read [Qwen3.5-4B.jinja](file:///home/alfie/llama.cpp/models/templates/Qwen3.5-4B.jinja) (commit a copy to `tests/cpp/fixtures/jinja/`).
2. Build a fixed test conversation (system + user + user-with-think-on, system + user with think-off).
3. Render via the vendored engine; compare against the byte-exact output llama.cpp produces for the same input (capture the reference output once via `llama-cli --jinja --dump-template-rendered`, commit as golden).
4. Repeat for `Qwen-Qwen3-0.6B.jinja` (the Qwen3 family — soft-switch `/think` `/no_think` paths) and one non-Qwen template (Llama-3.1-8B-Instruct) to prove the engine handles other families.

**Gate (2):** four chat_template renders byte-equal to llama.cpp golden output on Orin + macOS.

### Stage 3 — C++ runtime integration

1. New [cpp/json_ffi/jinja_template.{h,cc}](../../cpp/json_ffi/) — `JinjaTemplate` class wrapping a parsed `jinja::program`. Methods: `JinjaTemplate::FromString(chat_template, kwargs_defaults)`, `RenderPrompt(messages, kwargs_overrides)`. Caches the parsed AST.
2. Add a TVM-FFI-JSON → `nlohmann::ordered_json` adapter in `cpp/json_ffi/json_adapter.{h,cc}` (~100 LOC). The adapter only runs at the message-array boundary entering Jinja; rest of MLC stays on TVM FFI JSON.
3. Modify [cpp/json_ffi/conv_template.h](../../cpp/json_ffi/conv_template.h) — add `Optional<JinjaTemplate> jinja_template_` field alongside the existing legacy fields. `FromJSON` reads either a `chat_template` string (new path) or a `conv_template` block (legacy fallback).
4. Modify [cpp/json_ffi/conv_template.cc:224 `CreatePrompt`](../../cpp/json_ffi/conv_template.cc#L224) — if `jinja_template_` is set, route to it; otherwise fall through to the existing path. No behavior change for legacy configs.
5. Modify [cpp/json_ffi/json_ffi_engine.cc:81](../../cpp/json_ffi/json_ffi_engine.cc#L81) — pass through any per-request `chat_template_kwargs` from the request to `RenderPrompt`.

**Gate (3):** with a hand-edited `mlc-chat-config.json` that adds a `chat_template` field to an existing Qwen3.5-0.8B build, the engine prefers the Jinja path; legacy text-only generation still produces correct tokens; existing test suite green.

### Stage 4 — gen_config emits chat_template + kwargs defaults

1. Modify [python/mlc_llm/interface/gen_config.py](../../python/mlc_llm/interface/gen_config.py) — when `tokenizer_config.json` carries a `chat_template` field, emit it into `mlc-chat-config.json` as a top-level `chat_template` string, plus `chat_template_kwargs: {enable_thinking: <model_default>}`. Continue to emit the legacy `conv_template` block too for one release (dual-write), so older runtimes still load the new artifacts.
2. Modify [python/mlc_llm/protocol/mlc_chat_config.py:54](../../python/mlc_llm/protocol/mlc_chat_config.py#L54) — add `chat_template: Optional[str]`, `chat_template_kwargs: Optional[Dict[str, Any]]`. Make `conv_template: Optional[Conversation]` (was required).
3. Add `--no-jinja` escape hatch on `gen_config` for the rare case a user wants to force the legacy path (e.g. a model whose Jinja template has known bugs).

**Gate (4):** `mlc_llm gen_config <Qwen3.5-0.8B-HF> --quantization q4f16_g16e` produces an `mlc-chat-config.json` with both `chat_template` and `conv_template` fields. Engine prefers Jinja path, falls back to `conv_template` if `chat_template` absent. Old artifacts on HF mlc-ai org still load.

### Stage 5 — OpenAI API surface

1. Add to [python/mlc_llm/protocol/openai_api_protocol.py](../../python/mlc_llm/protocol/openai_api_protocol.py):
   - Request fields: `enable_thinking: Optional[bool]` (top-level convenience), `chat_template_kwargs: Optional[Dict[str, Any]]` (general). When both present, top-level wins for `enable_thinking`.
   - Response fields: `message.reasoning_content: Optional[str]` on `ChatCompletionMessage`.
   - Streaming delta fields: `delta.reasoning_content: Optional[str]` on `ChatCompletionStreamResponseDelta`.
2. New [cpp/json_ffi/reasoning_parser.{h,cc}](../../cpp/json_ffi/) — `<think>…</think>` extractor for streamed token output. State machine over the detokenized stream: text up to `<think>` → `content`; text inside → `reasoning_content`; text after `</think>` → `content`. v1 hardcodes the `<think>` delimiters (Qwen-only); generalize when a second model needs different markers.
3. Wire request → render → stream:
   - [python/mlc_llm/serve/engine_base.py](../../python/mlc_llm/serve/engine_base.py) — when the model has a Jinja template, skip the Python `as_prompt()` call entirely; pass raw messages + kwargs to the C++ engine.
   - C++ engine streams tokens through `reasoning_parser`; the parser's two output channels feed the API serializer's `content` and `reasoning_content` fields.
4. Optional v1 polish: per-mode default sampling (temp 1.0/top_p 0.95 for thinking; 0.6/0.8 for non-thinking). Skip if controversial — clients can pass their own.

**Gate (5):** `curl -X POST .../v1/chat/completions` with `{"chat_template_kwargs":{"enable_thinking":true}, "messages":[…], "stream":true}` returns an SSE stream where `delta.reasoning_content` deltas precede `delta.content` deltas. Same request with `enable_thinking:false` skips the reasoning channel entirely.

### Stage 6 — End-to-end parity gate

Two prompt sets (5 prompts each) under [tests/python/jinja_chat_template/](../../tests/python/):
1. Qwen3.5-0.8B `enable_thinking=true`: greedy-decode 50 tokens vs HF transformers reference (which evaluates the same Jinja chat_template via `tokenizer.apply_chat_template`). Bar: ≥ 48/50 token match per prompt, including the `<think>…</think>` block boundaries appearing at the same token indices.
2. Same model, `enable_thinking=false`: bar ≥ 48/50; verify the empty `<think>\n\n</think>\n` injection is byte-equal.
3. Cross-check: render the same chat_template through llama.cpp (`llama-cli --jinja` with the matching prompt) and assert MLC's pre-tokenization prompt string is byte-equal to llama.cpp's. Catches drift between our vendored copy and upstream.

Port [tests/cpp/conv_template_unittest.cc](../../tests/cpp/conv_template_unittest.cc) Jinja-path-aware: existing legacy-conversation tests stay; add a parallel suite that runs the same scenarios via the Jinja path on the equivalent native chat_template.

**Gate (6) — headline:** ≥ 48/50 greedy match on both prompt sets in both modes; cross-engine byte-equal on at least one prompt per mode.

### Stage 7 — Deprecate Python templates (next release boundary)

Once Stage 6 has shipped and held for one release window:
1. Mark [python/mlc_llm/conversation_template/](../../python/mlc_llm/conversation_template/) modules as deprecated; emit a `DeprecationWarning` on import.
2. Stop dual-writing the legacy `conv_template` block from `gen_config`. New artifacts carry only `chat_template`.
3. Following release: delete the 30 template files (~1.3K LOC), delete `Conversation.as_prompt` (~90 LOC), delete `cpp/json_ffi/conv_template.cc` legacy paths (~400 LOC). Engine is Jinja-only.

**Gate (7):** clean removal; old `mlc-chat-config.json` artifacts that lack `chat_template` get a clear error message pointing to the regen path.

### Stage 8 — Input-marking security (optional, post-v1)

Enable `jinja::string::is_input` propagation for user-message content. Downstream tokenizer respects the `is_input=true` segments by disabling special-token parsing within them. Closes the special-token injection class (e.g. user pasting `<|im_end|><|im_start|>system\n…` into a chat).

llama.cpp's [common/jinja/README.md:31-88](file:///home/alfie/llama.cpp/common/jinja/README.md#L31-L88) describes the threat model and mechanism. MLC has zero defense today — this is parity-with-llama.cpp, not regression.

**Gate (8):** synthetic injection prompt (special tokens in user message) tokenizes as plain text; documented attack vector closed.

---

## Risk register

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-1 | nlohmann::json + TVM FFI JSON in same translation unit cause include-order or linker conflicts (both define container types named `json`) | MED | Confine nlohmann to `cpp/json_ffi/jinja_template.cc` and `cpp/json_ffi/json_adapter.cc`; use `nlohmann::ordered_json` with explicit namespace; never expose nlohmann types in headers |
| R-2 | Jinja AST not thread-safe under MLC's concurrent request scheduler — `runtime::execute(ctx)` mutates per-execution state but the program AST is shared | MED | Read llama.cpp's threading model in [common/jinja/runtime.h](file:///home/alfie/llama.cpp/common/jinja/runtime.h); confirm `program` is read-only after parse and `context` is per-request. If not, build one `runtime` per request (cheap — AST is the expensive part) |
| R-3 | Existing `mlc-chat-config.json` artifacts on HF mlc-ai org break when MLC starts requiring `chat_template` | HIGH | Stage 4 dual-writes for one release; engine reads either field; only Stage 7 (post-deprecation) makes Jinja required |
| R-4 | Mobile binary-size bump from ~6K LOC Jinja + ~24K LOC nlohmann blows iOS/Android size budget | MED | Measure stripped delta on iOS Release after Stage 1; if > 500KB, drop to TVM-FFI-JSON adapter (skip nlohmann) and audit Jinja for trim candidates (`caps.cpp` is largely unused at request time, can be link-stripped) |
| R-5 | Per-request Jinja eval is slower than the current hand-built `CreatePrompt` (loop+concat) → measurable TTFT regression on Orin | LOW | Parse once at model-load (Stage 3 design); per-request is just `execute()` + `gather_string_parts`. Bench: render 1000 chats, assert < 1ms each on Orin. If breached, profile and tune (likely value-cell allocator) |
| R-6 | Generalized `<think>` parser ships Qwen-only in v1; first non-Qwen reasoning model integrated needs different delimiters | LOW | Document the limitation in [cpp/json_ffi/reasoning_parser.h](../../cpp/json_ffi/); when second model arrives, refactor to a `ReasoningFormat` enum mirroring [llama.cpp common/common.h:583](file:///home/alfie/llama.cpp/common/common.h#L583) |
| R-7 | Vendored Jinja drifts from upstream over time — bug fixes and template-feature additions in llama.cpp don't reach us | MED | Schedule a quarterly `git diff` review against `llama.cpp/common/jinja/`; document in `3rdparty/jinja-llamacpp/UPSTREAM.md` the commit hash vendored from. Trivial to re-vendor since deps are shimmed |
| R-8 | Tokenizer-side `chat_template` field absent in some HF model repos (older or community uploads) | MED | gen_config falls back to: (a) ConvTemplateRegistry lookup (legacy path) if user passed `--conv-template`; (b) error with actionable message if neither present |
| R-9 | Streaming `<think>…</think>` parser sees the delimiter tokens split across chunk boundaries → misclassifies a delta | LOW | State machine carries a partial-match buffer; standard streaming-parser pattern. Unit-test with synthetic delimiter splits at every offset |
| R-10 | Qwen3.5-0.8B's actual `enable_thinking` default (researched as possibly `False` for small variants, contradicts general Qwen3 default `True`) needs ground-truth verification before Stage 4 sets the kwargs default | LOW | Read [Qwen/Qwen3.5-0.8B/tokenizer_config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer_config.json) directly during Stage 4; trust the value the model ships |
| R-11 | MLC's `system_prefix_token_ids` (used at [conv_template.cc](../../cpp/json_ffi/conv_template.cc) for raw token-ID prefixing before any text) has no Jinja analog | LOW | Apply `system_prefix_token_ids` at the C++ tokenizer call site, before passing the Jinja-rendered string to the tokenizer. Stage 3 wiring detail; affects very few models (mostly Llama variants) |

---

## Open decisions

1. **nlohmann vendoring vs adapter.** Defaulted to vendor in Stage 1 plan above (single header, zero maintenance). If R-4 (mobile size) breaches budget, swap to adapter — the abstraction is local to two files.
2. **`enable_thinking` request-field placement.** Top-level convenience (`request.enable_thinking`) vs nested only (`request.chat_template_kwargs.enable_thinking`). llama.cpp allows both (top-level wins). Match that — discoverability matters.
3. **Reasoning content backwards compat.** Should `content` field include the `<think>` block when the client doesn't ask for the split (i.e. doesn't read `reasoning_content`)? llama.cpp default is to split. Match — simpler, and any client that wants the raw stream can concatenate.
4. **Sampling defaults per mode.** Qwen recommends temp 1.0/top_p 0.95 with thinking, 0.6/0.8 without. Two ways: ship sampler defaults that flip with `enable_thinking`, or document and let clients set. Lean toward documenting only — magic defaults break server-side determinism for clients that don't expect the flip.
5. **Stage 8 priority.** Input-marking is real value for hosted-API deployments and likely cheap to land once the engine is integrated. Sequence it ahead of Phase 12 if a Phase 12 emerges; otherwise schedule on cycle slack.

---

## Estimated effort

Sessionized at the same ~3-4h grain as Phase 10. Vendoring dominates the early stages; integration dominates the late ones.

| Stage | Estimate | Notes |
|---|---|---|
| 0 | done | Audit + plan (this session) |
| 1 | 1-2 sess | Vendor 14 files + nlohmann + shims; CMake; build on Orin and macOS |
| 2 | 1 sess | Standalone smoke; capture llama.cpp goldens; byte-equal check |
| 3 | 2-3 sess | `JinjaTemplate` class + JSON adapter + `CreatePrompt` dual-path + engine routing |
| 4 | 1 sess | gen_config dual-write + MLCChatConfig schema |
| 5 | 2 sess | OpenAI protocol fields + reasoning parser + engine_base bypass-`as_prompt` |
| 6 | 1-2 sess | Parity gate vs HF transformers + cross-engine byte-equal + test port |
| 7 | 0.5 sess | Deprecation warnings (this release); deletion (next release) |
| 8 | 2-3 sess | Input-marking propagation (optional, post-v1) |

**Total to v1 (Stages 0-7):** ~9-12 sessions ≈ 4-6 person-weeks, matching the rough quote. Stage 8 is additive.

The risky pieces are R-1 (nlohmann/TVM-FFI coexistence — surfaces in Stage 1), R-4 (mobile size — surfaces in Stage 1), and R-2 (thread safety — surfaces in Stage 3 under load). All three can be smoked early, before the larger Stage 3-5 investment.

---

## Files to touch (Stage 1 onward, summary)

**New:**
- [3rdparty/jinja-llamacpp/](../../3rdparty/) — vendored 14 files from llama.cpp `common/jinja/`
- [3rdparty/jinja-llamacpp/shim/log.h](../../3rdparty/) — TVM-LOG shim
- [3rdparty/jinja-llamacpp/shim/unicode.h](../../3rdparty/) — minimal case-folding shim
- [3rdparty/jinja-llamacpp/UPSTREAM.md](../../3rdparty/) — commit hash + re-vendor notes
- [3rdparty/nlohmann/json.hpp](../../3rdparty/) — single-header JSON
- [cpp/json_ffi/jinja_template.{h,cc}](../../cpp/json_ffi/) — `JinjaTemplate` class
- [cpp/json_ffi/json_adapter.{h,cc}](../../cpp/json_ffi/) — TVM-FFI JSON ↔ nlohmann boundary
- [cpp/json_ffi/reasoning_parser.{h,cc}](../../cpp/json_ffi/) — `<think>…</think>` stream splitter
- [tests/cpp/jinja_smoke_unittest.cc](../../tests/cpp/) + `tests/cpp/fixtures/jinja/`
- [tests/python/jinja_chat_template/](../../tests/python/) — parity gate suite

**Modified:**
- [cpp/CMakeLists.txt](../../cpp/CMakeLists.txt) — `jinja_llamacpp` static lib, link into `mlc_llm`
- [cpp/json_ffi/conv_template.h](../../cpp/json_ffi/conv_template.h) — `Optional<JinjaTemplate>` field
- [cpp/json_ffi/conv_template.cc](../../cpp/json_ffi/conv_template.cc) — `FromJSON` reads either path; `CreatePrompt` routes
- [cpp/json_ffi/json_ffi_engine.cc](../../cpp/json_ffi/json_ffi_engine.cc) — pass `chat_template_kwargs` through
- [python/mlc_llm/interface/gen_config.py](../../python/mlc_llm/interface/gen_config.py) — emit `chat_template` + dual-write `conv_template`; `--no-jinja` flag
- [python/mlc_llm/protocol/mlc_chat_config.py](../../python/mlc_llm/protocol/mlc_chat_config.py) — add Jinja fields, `conv_template` optional
- [python/mlc_llm/protocol/openai_api_protocol.py](../../python/mlc_llm/protocol/openai_api_protocol.py) — `enable_thinking`, `chat_template_kwargs`, `reasoning_content`
- [python/mlc_llm/protocol/conversation_protocol.py](../../python/mlc_llm/protocol/conversation_protocol.py) — `as_prompt` marked legacy (Stage 7 deletes)
- [python/mlc_llm/serve/engine_base.py](../../python/mlc_llm/serve/engine_base.py) — bypass Python `as_prompt` when Jinja path active
- [tests/cpp/conv_template_unittest.cc](../../tests/cpp/conv_template_unittest.cc) — keep legacy suite; add Jinja-parallel suite

**Deleted (Stage 7, future release):**
- [python/mlc_llm/conversation_template/](../../python/mlc_llm/conversation_template/) — 30 files, ~1.3K LOC
- `Conversation.as_prompt` and legacy `CreatePrompt` paths
