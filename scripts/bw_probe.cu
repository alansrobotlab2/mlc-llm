// Achievable streaming-read bandwidth on sm_87, read-dominated (the access
// shape that matters for a weight-streaming dequant-GEMV decode roofline).
// 128-bit vectorized loads (float4 = 8x fp16), grid-stride, no writes in the
// inner loop so the measurement is pure read traffic.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void sum_kernel(const float4* __restrict__ a, size_t n4, float* out) {
    float acc = 0.f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // keep the compiler honest without adding meaningful write traffic
    if (acc == 1234.5678f) out[0] = acc;
}

int main() {
    size_t bytes = 2ull << 30;  // 2 GiB, far beyond the 4 MB L2
    size_t n4 = bytes / sizeof(float4);
    bytes = n4 * sizeof(float4);

    float4* d = nullptr;
    float* out = nullptr;
    cudaMalloc(&d, bytes);
    cudaMalloc(&out, sizeof(float));
    cudaMemset(d, 0, bytes);

    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    int threads = 256;
    int blocks = prop.multiProcessorCount * 16;

    // warmup
    for (int i = 0; i < 3; ++i) sum_kernel<<<blocks, threads>>>(d, n4, out);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    const int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) sum_kernel<<<blocks, threads>>>(d, n4, out);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    double sec = ms / 1e3 / iters;
    double gbps = bytes / sec / 1e9;

    printf("device            : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);
    printf("buffer            : %.2f GiB\n", bytes / (double)(1 << 30));
    printf("per-iter time     : %.3f ms\n", sec * 1e3);
    printf("ACHIEVABLE READ BW: %.1f GB/s\n", gbps);
    printf("vs 204.8 spec peak: %.1f%%\n", gbps / 204.8 * 100.0);

    const double ACTIVE_GB = 1.600;  // 35B-A3B active bytes/token @ 4.345 bits/param
    double tps[] = {54.13, 54.46};
    for (double t : tps) {
        double eff = ACTIVE_GB * t;
        printf("\n35B @ %.2f tps -> %.1f GB/s = %.1f%% of achievable, %.1f%% of spec\n",
               t, eff, eff / gbps * 100.0, eff / 204.8 * 100.0);
    }
    printf("\nroofline @ achievable: %.1f tps\n", gbps / ACTIVE_GB);
    return 0;
}
