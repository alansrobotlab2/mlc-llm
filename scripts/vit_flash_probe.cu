// vit_flash_probe.cu — item 0r's gating measurement, before any TIR is written.
//
// The question
// ------------
// §20.9 prices flash attention for the Qwen3.5-VL tower at ~7.3 ms/layer against the
// 14.71 ms the three score-matrix kernels cost, and it gets that number from "cuBLAS's
// demonstrated 51% of fp32 peak". **A hand-written kernel is not cuBLAS.** The same
// section records the *generated* fp32 matmul at 1.03 TFLOP/s — 19% of peak — and at
// that rate the same 19.5 GFLOP/layer takes 18.9 ms and flash attention is a
// regression. The whole item turns on a rate nobody has measured on this box: what
// does a hand-written fp32 tiled kernel actually reach at these shapes?
//
// Two kernels, one process, one clock state:
//
//   gemm_lowk   C[2520,2520] = A[2520,64] @ B[2520,64]^T, the QK^T shape. cuBLAS does
//               this at 2.69 TFLOP/s (§20.5) and dlight at 1.03. A hand-tiled kernel
//               landing near the former says the gap is schedule, not silicon.
//   flash       the real thing: per (head, m-tile), stream K/V tiles, keep the scores
//               in shared, online softmax, accumulate O. Never materialises the
//               (12,2520,2520) fp32 = 305 MB score matrix.
//
// `flash` is the answer; `gemm_lowk` is the control that says whether a disappointing
// `flash` is flash's fault or the GPU's. Correctness is checked against a CPU
// reference first, because a fast wrong kernel answers nothing.
//
// This is a *probe*, in the sense scripts/gdn_recurrence_probe.cu is: it decides
// whether a TIR kernel is worth writing. Nothing here ships.
//
//   nvcc -arch=sm_87 -O3 -o /tmp/vit_flash_probe scripts/vit_flash_probe.cu
//   /tmp/vit_flash_probe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// The cat fixture at the tower's native patch count, and the tower's own config.
static const int H = 12, S = 2520, D = 64, LAYERS = 12;

// ---------------------------------------------------------------------------
// control: hand-tiled fp32 GEMM at the QK^T shape (M=N=2520, K=64).
// 64x64 block tile, 256 threads, 4x4 register tile, K=64 in one shared stage.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void gemm_lowk(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C, int M, int N) {
  __shared__ float As[64][D + 1];
  __shared__ float Bs[64][D + 1];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  for (int i = threadIdx.x; i < 64 * D; i += 256) {
    int r = i / D, c = i % D;
    As[r][c] = (m0 + r < M) ? A[(size_t)(m0 + r) * D + c] : 0.f;
    Bs[r][c] = (n0 + r < N) ? B[(size_t)(n0 + r) * D + c] : 0.f;
  }
  __syncthreads();
  float acc[4][4] = {};
  for (int k = 0; k < D; ++k) {
    float a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][k];
#pragma unroll
    for (int j = 0; j < 4; ++j) b[j] = Bs[tx * 4 + j][k];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int r = m0 + ty * 4 + i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c = n0 + tx * 4 + j;
      if (c < N) C[(size_t)r * N + c] = acc[i][j];
    }
  }
}

// ---------------------------------------------------------------------------
// flash attention, fp32, online softmax, templated on the tile so the probe reports
// the *best* hand-written configuration rather than the first one guessed. Each thread
// owns RM=BM/TY query rows, RN=BN/TX score columns and RO=D/TX output columns.
//
// `seq` is a runtime argument, not a constant, because the real model's sequence length
// is symbolic (§20.8) — pinning it here would flatter the probe.
//
// Padding is per-buffer and deliberate: Qs/Ks are read down a column (stride D+1 keeps
// the TX readers on distinct banks) while Vs is read along a row as float4 (no padding,
// so the 16 B lanes tile the 32 banks exactly).
// ---------------------------------------------------------------------------
template <int BM, int BN, int TX, int TY>
__global__ __launch_bounds__(TX * TY) void flash(const float* __restrict__ Q,
                                                 const float* __restrict__ K,
                                                 const float* __restrict__ V,
                                                 float* __restrict__ O,
                                                 int seq, float scale) {
  constexpr int NT = TX * TY, RM = BM / TY, RN = BN / TX, RO = D / TX;
  constexpr int PARTS = NT / BM;            // threads cooperating on one softmax row
  __shared__ float Qs[BM][D + 1];
  __shared__ float Ks[BN][D + 1];
  __shared__ float Vs[BN][D];
  __shared__ float Ss[BM][BN + 1];
  __shared__ float rowmax[BM], rowsum[BM], rowcorr[BM];

  const int h = blockIdx.y;
  const int m0 = blockIdx.x * BM;
  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  const size_t base = (size_t)h * seq * D;

  for (int i = threadIdx.x; i < BM * D; i += NT) {
    int r = i / D, c = i % D;
    Qs[r][c] = (m0 + r < seq) ? Q[base + (size_t)(m0 + r) * D + c] * scale : 0.f;
  }
  if (threadIdx.x < BM) { rowmax[threadIdx.x] = -INFINITY; rowsum[threadIdx.x] = 0.f; }
  float acc[RM][RO] = {};
  __syncthreads();

  for (int n0 = 0; n0 < seq; n0 += BN) {
    for (int i = threadIdx.x; i < BN * D; i += NT) {
      int r = i / D, c = i % D;
      bool ok = (n0 + r) < seq;
      Ks[r][c] = ok ? K[base + (size_t)(n0 + r) * D + c] : 0.f;
      Vs[r][c] = ok ? V[base + (size_t)(n0 + r) * D + c] : 0.f;
    }
    __syncthreads();

    // --- S = (Q*scale) @ K^T for this tile ---
    float s[RM][RN] = {};
    for (int k = 0; k < D; ++k) {
      float a[RM], b[RN];
#pragma unroll
      for (int i = 0; i < RM; ++i) a[i] = Qs[ty * RM + i][k];
#pragma unroll
      for (int j = 0; j < RN; ++j) b[j] = Ks[tx * RN + j][k];
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) s[i][j] = fmaf(a[i], b[j], s[i][j]);
    }
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
      for (int j = 0; j < RN; ++j)
        Ss[ty * RM + i][tx * RN + j] =
            ((n0 + tx * RN + j) < seq) ? s[i][j] : -INFINITY;
    __syncthreads();

    // --- online softmax: PARTS threads per row. The partners are adjacent lanes of
    //     one warp for every configuration swept here, so a butterfly shuffle works. ---
    {
      const int row = threadIdx.x / PARTS, part = threadIdx.x % PARTS;
      float pmax = -INFINITY;
      for (int c = part; c < BN; c += PARTS) pmax = fmaxf(pmax, Ss[row][c]);
#pragma unroll
      for (int off = 1; off < PARTS; off <<= 1)
        pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffff, pmax, off));
      const float mnew = fmaxf(rowmax[row], pmax);
      float psum = 0.f;
      for (int c = part; c < BN; c += PARTS) {
        float e = __expf(Ss[row][c] - mnew);
        Ss[row][c] = e;
        psum += e;
      }
#pragma unroll
      for (int off = 1; off < PARTS; off <<= 1)
        psum += __shfl_xor_sync(0xffffffff, psum, off);
      if (part == 0) {
        // expf(-inf - mnew) = 0 on the first tile, which is what zeroes `acc`.
        const float corr = __expf(rowmax[row] - mnew);
        rowcorr[row] = corr;
        rowsum[row] = rowsum[row] * corr + psum;
        rowmax[row] = mnew;
      }
    }
    __syncthreads();

    // --- O = O*corr + P @ V ---
#pragma unroll
    for (int i = 0; i < RM; ++i) {
      const float corr = rowcorr[ty * RM + i];
#pragma unroll
      for (int j = 0; j < RO; ++j) acc[i][j] *= corr;
    }
    for (int k = 0; k < BN; ++k) {
      float a[RM];
#pragma unroll
      for (int i = 0; i < RM; ++i) a[i] = Ss[ty * RM + i][k];
#pragma unroll
      for (int j4 = 0; j4 < RO; j4 += 4) {
        const float4 b = *(const float4*)&Vs[k][tx * RO + j4];
#pragma unroll
        for (int i = 0; i < RM; ++i) {
          acc[i][j4 + 0] = fmaf(a[i], b.x, acc[i][j4 + 0]);
          acc[i][j4 + 1] = fmaf(a[i], b.y, acc[i][j4 + 1]);
          acc[i][j4 + 2] = fmaf(a[i], b.z, acc[i][j4 + 2]);
          acc[i][j4 + 3] = fmaf(a[i], b.w, acc[i][j4 + 3]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < RM; ++i) {
    const int r = m0 + ty * RM + i;
    if (r >= seq) continue;
    const float inv = 1.f / rowsum[ty * RM + i];
#pragma unroll
    for (int j4 = 0; j4 < RO; j4 += 4)
      *(float4*)&O[base + (size_t)r * D + tx * RO + j4] =
          make_float4(acc[i][j4 + 0] * inv, acc[i][j4 + 1] * inv,
                      acc[i][j4 + 2] * inv, acc[i][j4 + 3] * inv);
  }
}

// ---------------------------------------------------------------------------
static float *dQ, *dK, *dV, *dO, *dC;
static int g_seq = S;

typedef void (*Launcher)(int);

template <int BM, int BN, int TX, int TY>
static void launch_cfg(int seq) {
  flash<BM, BN, TX, TY><<<dim3((seq + BM - 1) / BM, H), TX * TY>>>(
      dQ, dK, dV, dO, seq, 1.f / sqrtf((float)D));
}

struct Cfg { const char* name; Launcher fn; int shared; };

// Shared bytes per CTA, so the table can show what each config costs in occupancy.
template <int BM, int BN>
constexpr int shmem() {
  return (BM * (D + 1) + BN * (D + 1) + BN * D + BM * (BN + 1) + 3 * BM) * 4;
}

// Constrained by three things, all checked at compile time by the template:
//   - static __shared__ caps at 48 kB, which rules out every BN=64 tile here (67 kB at
//     BM=64). Reaching those needs `extern __shared__` plus a carveout opt-in; left out
//     deliberately, and noted, rather than silently dropped.
//   - RO = D/TX must be a multiple of 4 for the float4 reads of Vs, so TX <= 16.
//   - PARTS = NT/BM threads cooperate per softmax row and must be adjacent lanes of one
//     warp for the butterfly shuffle, so BM >= NT/32.
static Cfg CFGS[] = {
  {"BM64 BN32 16x16", launch_cfg<64, 32, 16, 16>, shmem<64, 32>()},
  {"BM64 BN32  8x32", launch_cfg<64, 32, 8, 32>,  shmem<64, 32>()},
  {"BM64 BN16 16x16", launch_cfg<64, 16, 16, 16>, shmem<64, 16>()},
  {"BM32 BN32 16x8 ", launch_cfg<32, 32, 16, 8>,  shmem<32, 32>()},
  {"BM32 BN32  8x16", launch_cfg<32, 32, 8, 16>,  shmem<32, 32>()},
  {"BM32 BN16 16x8 ", launch_cfg<32, 16, 16, 8>,  shmem<32, 16>()},
};

static void launch_gemm() {
  gemm_lowk<<<dim3((S + 63) / 64, (S + 63) / 64), 256>>>(dQ, dK, dC, S, S);
}

static float time_gemm(int iters) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));  CHECK(cudaEventCreate(&b));
  launch_gemm();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) launch_gemm();
  CHECK(cudaEventRecord(b));
  CHECK(cudaEventSynchronize(b));
  float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
  CHECK(cudaGetLastError());
  return ms / iters;
}

static float time_flash(Launcher fn, int iters) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));  CHECK(cudaEventCreate(&b));
  fn(g_seq);
  if (cudaDeviceSynchronize() != cudaSuccess) return -1.f;
  CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) fn(g_seq);
  CHECK(cudaEventRecord(b));
  CHECK(cudaEventSynchronize(b));
  float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
  if (cudaGetLastError() != cudaSuccess) return -1.f;
  return ms / iters;
}

// Reference check on the shipping config, at sequence lengths that are not multiples
// of BM or BN so the partial-tile paths are exercised. A fast wrong kernel answers
// nothing.
static void verify(int s) {
  std::vector<float> q(s * D), k(s * D), v(s * D), o(s * D), ref(s * D);
  for (int i = 0; i < s * D; ++i) {
    q[i] = (float)drand48() - 0.5f;
    k[i] = (float)drand48() - 0.5f;
    v[i] = (float)drand48() - 0.5f;
  }
  float *gq, *gk, *gv, *go;
  CHECK(cudaMalloc(&gq, s * D * 4));  CHECK(cudaMalloc(&gk, s * D * 4));
  CHECK(cudaMalloc(&gv, s * D * 4));  CHECK(cudaMalloc(&go, s * D * 4));
  CHECK(cudaMemcpy(gq, q.data(), s * D * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gk, k.data(), s * D * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gv, v.data(), s * D * 4, cudaMemcpyHostToDevice));
  const float scale = 1.f / sqrtf((float)D);
  for (int i = 0; i < s; ++i) {
    std::vector<float> sc(s);
    float mx = -INFINITY;
    for (int j = 0; j < s; ++j) {
      float a = 0;
      for (int d = 0; d < D; ++d) a += q[i * D + d] * scale * k[j * D + d];
      sc[j] = a;
      mx = std::max(mx, a);
    }
    float sum = 0;
    for (int j = 0; j < s; ++j) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
    for (int d = 0; d < D; ++d) {
      float a = 0;
      for (int j = 0; j < s; ++j) a += sc[j] * v[j * D + d];
      ref[i * D + d] = a / sum;
    }
  }
  flash<64, 32, 16, 16><<<dim3((s + 63) / 64, 1), 256>>>(gq, gk, gv, go, s, scale);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(o.data(), go, s * D * 4, cudaMemcpyDeviceToHost));
  double worst = 0;
  for (int i = 0; i < s * D; ++i) worst = std::max(worst, (double)fabsf(o[i] - ref[i]));
  printf("[verify] seq=%-5d (partial m- and n-tiles) max abs diff %.3e  %s\n",
         s, worst, worst < 2e-5 ? "OK" : "*** MISMATCH ***");
  if (!(worst < 2e-5)) exit(1);
  cudaFree(gq); cudaFree(gk); cudaFree(gv); cudaFree(go);
}

int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  // CUDA 13 removed `clockRate` from cudaDeviceProp; the attribute query survives.
  int clk_khz = 0;
  CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  // GA10B has 128 fp32 lanes/SM. This basis reproduces §20.5's "cuBLAS at 51% of peak".
  const double peak = p.multiProcessorCount * 128.0 * 2.0 * clk_khz * 1e3;
  printf("[probe] %s  %d SMs  %.1f MHz  fp32 peak %.2f TFLOP/s\n",
         p.name, p.multiProcessorCount, clk_khz / 1e3, peak / 1e12);

  CHECK(cudaMalloc(&dQ, (size_t)H * S * D * 4));
  CHECK(cudaMalloc(&dK, (size_t)H * S * D * 4));
  CHECK(cudaMalloc(&dV, (size_t)H * S * D * 4));
  CHECK(cudaMalloc(&dO, (size_t)H * S * D * 4));
  CHECK(cudaMalloc(&dC, (size_t)S * S * 4));
  std::vector<float> host((size_t)H * S * D);
  for (auto& x : host) x = (float)drand48() - 0.5f;
  CHECK(cudaMemcpy(dQ, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dK, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dV, host.data(), host.size() * 4, cudaMemcpyHostToDevice));

  verify(200);
  verify(1000);

  const double gemm_flop = 2.0 * (double)S * S * D;             // one head of QK^T
  const double flash_flop = 2.0 * 2.0 * H * (double)S * S * D;  // QK^T + P@V, 12 heads

  const float g = time_gemm(20);
  printf("\ncontrol — hand-tiled fp32 GEMM at the QK^T shape (the rate question)\n");
  printf("  1 head   %6.2f ms   %5.2f TFLOP/s (%4.1f%% of peak)\n",
         g, gemm_flop / (g * 1e-3) / 1e12, 100.0 * gemm_flop / (g * 1e-3) / peak);
  printf("  x12      %6.2f ms   vs cuBLAS 3.63 (2.69 TFLOP/s) and dlight 9.46 (1.03)\n",
         g * H);

  printf("\nflash attention, one layer (12 heads), fp32, online softmax\n");
  printf("  %-20s %7s %8s %8s %10s %8s\n",
         "config", "shared", "CTAs/SM", "ms/layer", "TFLOP/s", "vs 14.71");
  float best = 1e30f; const char* bestname = "";
  for (auto& c : CFGS) {
    float ms = time_flash(c.fn, 20);
    if (ms < 0) { printf("  %-20s %6.1fk   (launch failed)\n", c.name, c.shared / 1024.f); continue; }
    int ctas = (164 * 1024) / (c.shared + 1024);   // CC 8.x reserves 1 kB/block (§21.2)
    printf("  %-20s %6.1fk %8d %8.2f %10.2f %7.2fx\n", c.name, c.shared / 1024.f,
           ctas, ms, flash_flop / (ms * 1e-3) / 1e12, ms / 14.71f);
    if (ms < best) { best = ms; bestname = c.name; }
  }

  printf("\n§20.9's three kernels: QK^T 3.63 + softmax 4.05 + P@V 7.03 = 14.71 ms/layer,"
         " 176.5 ms/iter\n");
  printf("best: %s at %.2f ms/layer = %.1f ms/image_embed (%.2fx)\n",
         bestname, best, best * LAYERS, best / 14.71f);
  printf("VERDICT: %s\n", best < 14.71f
         ? "a WIN — saves (14.71 - best) x 12 ms of image_embed; worth building in TIR"
         : "a REGRESSION — item 0r is dead");
  return 0;
}
