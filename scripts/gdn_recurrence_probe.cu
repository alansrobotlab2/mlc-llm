// gdn_recurrence_probe.cu — why `gdn_func_history_inplace` runs at ~3.8% of peak,
// and which of the two candidate fixes is worth building in TIR (workplan §9 item 0c.1).
//
// §15.6 measured the shipped kernel at 95.6 ms against 19.3 ms for the next kernel and
// attributed it to being "latency-bound on a dependency chain". This probe separates
// three candidate causes, because they call for different fixes:
//
//   1. the FMA accumulator chain. ptxas emits `FFMA R124, R125, R###, R124` for all 128
//      rows — ONE accumulator, so each FMA waits ~4 cycles on the previous. Two such
//      chains per timestep (dot_sk, dot_sq) = a 256-deep serial chain per position.
//   2. register spill. `float state_local[128]` needs 128 registers for state alone;
//      ptxas lands on the 255 ceiling and still spills 192 bytes/thread to local memory,
//      so ~47 of the 128 rows are re-loaded and re-stored from L1 every timestep.
//      §15's claim that the fusion left "no bandwidth to reclaim" covered DRAM only.
//   3. occupancy. The grid is (n_kh, batch) = 16 blocks at batch=1 on a 16-SM GPU, so
//      one 128-thread block per SM: 4 warps of the 48 an SM can hold.
//
// Variants, in increasing order of how much they change:
//   base   — as TVM emits it today, reproduced FMA for FMA.
//   acc4   — 4 partial accumulators per dot. Breaks (1) only. Intra-thread, ~10 lines
//            of TIR. Changes summation order, so it is not bit-exact with the copy path.
//   ksplit — K split 2 ways across adjacent lanes (tid = v*2 + half), reduced with
//            __shfl_xor_sync. Fixes (1) partly, (2) fully (64 floats/thread), and (3)
//            2x. Needs a real new kernel, and the shuffle reduction runs per timestep.
//
// The arithmetic is identical to qwen35_model.py's `create_gated_delta_net_func_*`:
// per position, S *= gate; dot_sk = <S, k>; S += k (x) beta*(v - dot_sk); out = <S, q>*scale.
// Only the ORDER of the K-reduction differs between variants, which is the point.
//
// Scope, so the numbers are not over-read: this measures the RECURRENCE ONLY. The
// per-position scatter into the history ring (`storage_buf[...] = state_local[row]`,
// guarded by `seq_len <= t + max_history`) is deliberately absent — it touches at most
// the last `max_history` positions of a chunk, it is pure stores, and including it here
// would blend a bandwidth term into a latency measurement. A speedup seen here is an
// upper bound on what the shipped kernel would gain.
//
// ⚠️ THE GRID HERE IS THE 0.8B's, NOT THE 35B's — do not quote these ratios against a 35B
// number. `grid(n_kh, batch)` below launches 16 blocks, but the real kernel binds
// blockIdx.x to `num_value_heads`: 16 on the 0.8B (so this is faithful) and **32 on the
// 35B**, where `base` therefore fits 2 blocks/SM and starts at 8 warps/SM instead of 4.
// Measured in TIR at the real geometry (workplan §16.5), ksplit4 is 1.94x on the 0.8B and
// only 1.21x on the 35B at seq_len=512, and ksplit2 is a 0.96x REGRESSION on the 35B. Use
// `scripts/gdn_kernel_bench.py` for anything that has to be true of a shipped kernel.
//
//   nvcc -arch=sm_87 -O3 -o gdn_probe scripts/gdn_recurrence_probe.cu && ./gdn_probe
//
// Prints per-variant ms and the implied speedup. Correctness against `base` is checked
// in fp64-ish terms (fp32 reference on host is too slow at these sizes; the variants are
// compared to each other with a tolerance, since the reduction order legitimately differs).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const int K = 128;   // key head dim
static const int V = 128;   // value head dim
static const float SCALE = 0.08838834764831845f;  // 1/sqrt(128)

// ─────────────────────────────────────────────────────────────────────────────
// base — one accumulator per dot, 128 floats/thread. What ships today.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(128) void gdn_base(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    const float* __restrict__ gate, const float* __restrict__ beta,
    const float* __restrict__ state_in, float* __restrict__ out, int64_t seq_len) {
  float state[K];
  const int kh = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int n_kh = gridDim.x;
#pragma unroll 1
  for (int row = 0; row < K; ++row)
    state[row] = state_in[(((int64_t)b * n_kh + kh) * K + row) * V + tid];

  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t hbase = (((int64_t)b * seq_len + t) * n_kh + kh);
    const float g = gate[hbase], bt = beta[hbase];
    const int64_t vb = hbase * V;
    const float v_val = __half2float(v[vb + tid]);
    float dot_sk = 0.f;
    for (int row = 0; row < K; ++row) {
      state[row] = state[row] * g;
      dot_sk = dot_sk + state[row] * __half2float(k[vb + row]);
    }
    const float coef = bt * (v_val - dot_sk);
    float dot_sq = 0.f;
    for (int row = 0; row < K; ++row) {
      state[row] = state[row] + __half2float(k[vb + row]) * coef;
      dot_sq = dot_sq + state[row] * __half2float(q[vb + row]);
    }
    out[vb + tid] = dot_sq * SCALE;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// acc4 — 4 independent partial sums per dot. Same data layout, same registers,
// only the reduction tree changes. Isolates cause (1).
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(128) void gdn_acc4(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    const float* __restrict__ gate, const float* __restrict__ beta,
    const float* __restrict__ state_in, float* __restrict__ out, int64_t seq_len) {
  float state[K];
  const int kh = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int n_kh = gridDim.x;
#pragma unroll 1
  for (int row = 0; row < K; ++row)
    state[row] = state_in[(((int64_t)b * n_kh + kh) * K + row) * V + tid];

  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t hbase = (((int64_t)b * seq_len + t) * n_kh + kh);
    const float g = gate[hbase], bt = beta[hbase];
    const int64_t vb = hbase * V;
    const float v_val = __half2float(v[vb + tid]);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int row = 0; row < K; row += 4) {
      state[row + 0] *= g; a0 += state[row + 0] * __half2float(k[vb + row + 0]);
      state[row + 1] *= g; a1 += state[row + 1] * __half2float(k[vb + row + 1]);
      state[row + 2] *= g; a2 += state[row + 2] * __half2float(k[vb + row + 2]);
      state[row + 3] *= g; a3 += state[row + 3] * __half2float(k[vb + row + 3]);
    }
    const float coef = bt * (v_val - ((a0 + a1) + (a2 + a3)));
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int row = 0; row < K; row += 4) {
      state[row + 0] += __half2float(k[vb + row + 0]) * coef;
      c0 += state[row + 0] * __half2float(q[vb + row + 0]);
      state[row + 1] += __half2float(k[vb + row + 1]) * coef;
      c1 += state[row + 1] * __half2float(q[vb + row + 1]);
      state[row + 2] += __half2float(k[vb + row + 2]) * coef;
      c2 += state[row + 2] * __half2float(q[vb + row + 2]);
      state[row + 3] += __half2float(k[vb + row + 3]) * coef;
      c3 += state[row + 3] * __half2float(q[vb + row + 3]);
    }
    out[vb + tid] = ((c0 + c1) + (c2 + c3)) * SCALE;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ksplit — K halved across ADJACENT lanes so the cross-thread reduction is a single
// __shfl_xor_sync with no __syncthreads. tid = v*2 + half, 256 threads/block.
// 64 floats/thread, which is what takes the kernel off the 255-register ceiling.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256) void gdn_ksplit(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    const float* __restrict__ gate, const float* __restrict__ beta,
    const float* __restrict__ state_in, float* __restrict__ out, int64_t seq_len) {
  const int KH = K / 2;
  float state[KH];
  const int kh = blockIdx.x, b = blockIdx.y;
  const int n_kh = gridDim.x;
  const int lane_v = threadIdx.x >> 1;        // 0..127, the value-dim index
  const int half_id = threadIdx.x & 1;        // which K half this lane owns
  const int row0 = half_id * KH;

#pragma unroll 1
  for (int row = 0; row < KH; ++row)
    state[row] = state_in[(((int64_t)b * n_kh + kh) * K + row0 + row) * V + lane_v];

  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t hbase = (((int64_t)b * seq_len + t) * n_kh + kh);
    const float g = gate[hbase], bt = beta[hbase];
    const int64_t vb = hbase * V;
    const float v_val = __half2float(v[vb + lane_v]);
    float a0 = 0.f, a1 = 0.f;
    for (int row = 0; row < KH; row += 2) {
      state[row + 0] *= g; a0 += state[row + 0] * __half2float(k[vb + row0 + row + 0]);
      state[row + 1] *= g; a1 += state[row + 1] * __half2float(k[vb + row0 + row + 1]);
    }
    float dot_sk = a0 + a1;
    dot_sk += __shfl_xor_sync(0xffffffffu, dot_sk, 1);   // combine the two K halves
    const float coef = bt * (v_val - dot_sk);
    float c0 = 0.f, c1 = 0.f;
    for (int row = 0; row < KH; row += 2) {
      state[row + 0] += __half2float(k[vb + row0 + row + 0]) * coef;
      c0 += state[row + 0] * __half2float(q[vb + row0 + row + 0]);
      state[row + 1] += __half2float(k[vb + row0 + row + 1]) * coef;
      c1 += state[row + 1] * __half2float(q[vb + row0 + row + 1]);
    }
    float dot_sq = c0 + c1;
    dot_sq += __shfl_xor_sync(0xffffffffu, dot_sq, 1);
    if (half_id == 0) out[vb + lane_v] = dot_sq * SCALE;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ksplit4 — the same idea taken to 4 lanes per v (tid = v*4 + quarter), 512 threads,
// 32 floats/thread. Two shuffle steps instead of one. If ksplit wins because of
// occupancy this should win more; if it wins because the spill went away, this adds
// reduction cost for nothing. That is the discriminator between causes (2) and (3).
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(512) void gdn_ksplit4(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    const float* __restrict__ gate, const float* __restrict__ beta,
    const float* __restrict__ state_in, float* __restrict__ out, int64_t seq_len) {
  const int KQ = K / 4;
  float state[KQ];
  const int kh = blockIdx.x, b = blockIdx.y;
  const int n_kh = gridDim.x;
  const int lane_v = threadIdx.x >> 2;
  const int quarter = threadIdx.x & 3;
  const int row0 = quarter * KQ;

#pragma unroll 1
  for (int row = 0; row < KQ; ++row)
    state[row] = state_in[(((int64_t)b * n_kh + kh) * K + row0 + row) * V + lane_v];

  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t hbase = (((int64_t)b * seq_len + t) * n_kh + kh);
    const float g = gate[hbase], bt = beta[hbase];
    const int64_t vb = hbase * V;
    const float v_val = __half2float(v[vb + lane_v]);
    float a0 = 0.f, a1 = 0.f;
    for (int row = 0; row < KQ; row += 2) {
      state[row + 0] *= g; a0 += state[row + 0] * __half2float(k[vb + row0 + row + 0]);
      state[row + 1] *= g; a1 += state[row + 1] * __half2float(k[vb + row0 + row + 1]);
    }
    float dot_sk = a0 + a1;
    dot_sk += __shfl_xor_sync(0xffffffffu, dot_sk, 1);
    dot_sk += __shfl_xor_sync(0xffffffffu, dot_sk, 2);
    const float coef = bt * (v_val - dot_sk);
    float c0 = 0.f, c1 = 0.f;
    for (int row = 0; row < KQ; row += 2) {
      state[row + 0] += __half2float(k[vb + row0 + row + 0]) * coef;
      c0 += state[row + 0] * __half2float(q[vb + row0 + row + 0]);
      state[row + 1] += __half2float(k[vb + row0 + row + 1]) * coef;
      c1 += state[row + 1] * __half2float(q[vb + row0 + row + 1]);
    }
    float dot_sq = c0 + c1;
    dot_sq += __shfl_xor_sync(0xffffffffu, dot_sq, 1);
    dot_sq += __shfl_xor_sync(0xffffffffu, dot_sq, 2);
    if (quarter == 0) out[vb + lane_v] = dot_sq * SCALE;
  }
}

// ─────────────────────────────────────────────────────────────────────────────

static float frand() { return (float)rand() / (float)RAND_MAX - 0.5f; }

struct Bufs {
  half *q, *k, *v;
  float *gate, *beta, *state, *out;
};

static Bufs alloc(int batch, int n_kh, int64_t seq_len) {
  Bufs b{};
  size_t nqkv = (size_t)batch * seq_len * n_kh * K;
  size_t nh = (size_t)batch * seq_len * n_kh;
  size_t nst = (size_t)batch * n_kh * K * V;
  CHECK(cudaMalloc(&b.q, nqkv * sizeof(half)));
  CHECK(cudaMalloc(&b.k, nqkv * sizeof(half)));
  CHECK(cudaMalloc(&b.v, nqkv * sizeof(half)));
  CHECK(cudaMalloc(&b.gate, nh * sizeof(float)));
  CHECK(cudaMalloc(&b.beta, nh * sizeof(float)));
  CHECK(cudaMalloc(&b.state, nst * sizeof(float)));
  CHECK(cudaMalloc(&b.out, nqkv * sizeof(float)));

  half* h = (half*)malloc(nqkv * sizeof(half));
  for (size_t i = 0; i < nqkv; ++i) h[i] = __float2half(frand() * 0.6f);
  CHECK(cudaMemcpy(b.q, h, nqkv * sizeof(half), cudaMemcpyHostToDevice));
  for (size_t i = 0; i < nqkv; ++i) h[i] = __float2half(frand() * 0.6f);
  CHECK(cudaMemcpy(b.k, h, nqkv * sizeof(half), cudaMemcpyHostToDevice));
  for (size_t i = 0; i < nqkv; ++i) h[i] = __float2half(frand());
  CHECK(cudaMemcpy(b.v, h, nqkv * sizeof(half), cudaMemcpyHostToDevice));
  free(h);

  // Sized for the LARGEST of the three float arrays, not for nqkv: at seq_len=1 the
  // state (batch*n_kh*K*V) is two orders of magnitude bigger than q/k/v.
  size_t nf = nqkv;
  if (nh > nf) nf = nh;
  if (nst > nf) nf = nst;
  float* f = (float*)malloc(nf * sizeof(float));
  // gate = exp(-u*0.1) in (0,1] and beta = sigmoid(): the ranges the model produces,
  // so the recurrence decays instead of blowing up and the comparison stays meaningful.
  for (size_t i = 0; i < nh; ++i) f[i] = expf(-fabsf(frand()) * 0.1f);
  CHECK(cudaMemcpy(b.gate, f, nh * sizeof(float), cudaMemcpyHostToDevice));
  for (size_t i = 0; i < nh; ++i) f[i] = 1.f / (1.f + expf(-frand()));
  CHECK(cudaMemcpy(b.beta, f, nh * sizeof(float), cudaMemcpyHostToDevice));
  for (size_t i = 0; i < nst; ++i) f[i] = frand() * 0.4f;
  CHECK(cudaMemcpy(b.state, f, nst * sizeof(float), cudaMemcpyHostToDevice));
  free(f);
  return b;
}

typedef void (*KernFn)(const half*, const half*, const half*, const float*,
                       const float*, const float*, float*, int64_t);

static float time_kernel(KernFn fn, int threads, Bufs& b, int batch, int n_kh,
                         int64_t seq_len, int iters, float* host_out) {
  dim3 grid(n_kh, batch), block(threads);
  fn<<<grid, block>>>(b.q, b.k, b.v, b.gate, b.beta, b.state, b.out, seq_len);
  CHECK(cudaDeviceSynchronize());
  if (host_out) {
    size_t n = (size_t)batch * seq_len * n_kh * K;
    CHECK(cudaMemcpy(host_out, b.out, n * sizeof(float), cudaMemcpyDeviceToHost));
  }
  cudaEvent_t ev_start, ev_stop;
  CHECK(cudaEventCreate(&ev_start)); CHECK(cudaEventCreate(&ev_stop));
  CHECK(cudaEventRecord(ev_start));
  for (int i = 0; i < iters; ++i)
    fn<<<grid, block>>>(b.q, b.k, b.v, b.gate, b.beta, b.state, b.out, seq_len);
  CHECK(cudaEventRecord(ev_stop));
  CHECK(cudaEventSynchronize(ev_stop));
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
  CHECK(cudaEventDestroy(ev_start)); CHECK(cudaEventDestroy(ev_stop));
  return ms / iters;
}

int main(int argc, char** argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 1;
  int iters = argc > 2 ? atoi(argv[2]) : 20;
  srand(1234);

  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  printf("device: %s  SMs=%d  regs/SM=%d  maxThreads/SM=%d\n\n",
         p.name, p.multiProcessorCount, p.regsPerMultiprocessor,
         p.maxThreadsPerMultiProcessor);

  const int n_kh = 16;  // both models: 16 key heads
  const int64_t seq_lens[] = {1, 128, 512, 2048};

  printf("batch=%d n_kh=%d (grid = n_kh x batch = %d blocks on %d SMs)\n",
         batch, n_kh, n_kh * batch, p.multiProcessorCount);
  printf("%8s %10s %10s %10s %10s %8s %8s %8s\n",
         "seq_len", "base ms", "acc4 ms", "ksplit2", "ksplit4",
         "acc4 x", "ks2 x", "ks4 x");

  for (int64_t sl : seq_lens) {
    Bufs b = alloc(batch, n_kh, sl);
    size_t n = (size_t)batch * sl * n_kh * K;
    float* o_base = (float*)malloc(n * sizeof(float));
    float* o_a4 = (float*)malloc(n * sizeof(float));
    float* o_ks = (float*)malloc(n * sizeof(float));
    float* o_k4 = (float*)malloc(n * sizeof(float));

    float t0 = time_kernel(gdn_base, 128, b, batch, n_kh, sl, iters, o_base);
    float t1 = time_kernel(gdn_acc4, 128, b, batch, n_kh, sl, iters, o_a4);
    float t2 = time_kernel(gdn_ksplit, 256, b, batch, n_kh, sl, iters, o_ks);
    float t3 = time_kernel(gdn_ksplit4, 512, b, batch, n_kh, sl, iters, o_k4);

    // Reduction order differs by construction, so compare relative to the output scale
    // rather than demanding bit-exactness.
    double scale = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      scale = fmax(scale, fabs((double)o_base[i]));
      e1 = fmax(e1, fabs((double)o_a4[i] - (double)o_base[i]));
      e2 = fmax(e2, fabs((double)o_ks[i] - (double)o_base[i]));
      e3 = fmax(e3, fabs((double)o_k4[i] - (double)o_base[i]));
    }
    if (scale < 1e-9) scale = 1e-9;

    printf("%8lld %10.4f %10.4f %10.4f %10.4f %8.2f %8.2f %8.2f\n",
           (long long)sl, t0, t1, t2, t3, t0 / t1, t0 / t2, t0 / t3);
    printf("%8s rel_err vs base: acc4=%.2e ksplit2=%.2e ksplit4=%.2e\n",
           "", e1 / scale, e2 / scale, e3 / scale);

    free(o_base); free(o_a4); free(o_ks); free(o_k4);
    cudaFree(b.q); cudaFree(b.k); cudaFree(b.v);
    cudaFree(b.gate); cudaFree(b.beta); cudaFree(b.state); cudaFree(b.out);
  }
  printf("\nrel_err is vs `base`; a nonzero value is the reduction order changing,\n"
         "not a wrong answer. `scripts/gdn_kernel_check.py` is the correctness gate.\n");
  return 0;
}
