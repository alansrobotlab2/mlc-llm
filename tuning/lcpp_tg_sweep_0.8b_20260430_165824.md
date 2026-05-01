| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |           pp512 |     4538.48 ± 171.43 |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |           tg128 |        100.23 ± 0.18 |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |     pp512+tg512 |        196.17 ± 0.05 |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |    pp512+tg1024 |        148.52 ± 0.05 |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |    pp512+tg2048 |        123.96 ± 0.03 |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |    pp512+tg4096 |        109.94 ± 0.47 |
| qwen35 0.8B Q4_K - Medium      | 522.43 MiB |   752.39 M | CUDA       |  99 |  1 |    pp512+tg8192 |        102.36 ± 0.71 |

build: 8bc492e (1)
