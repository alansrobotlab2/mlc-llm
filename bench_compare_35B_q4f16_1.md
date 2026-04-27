### 35B-A3B — apples-to-apples (Orin AGX)

| ctx | backend | pp_tps | tg_tps | tg vs llama.cpp |
|---:|---|---:|---:|---:|
| 128 | llama.cpp Q4_K_S | 345.98 | 29.59 | 1.000× |
| 128 | MLC q4f16_1 | 149.88 | 10.12 | 0.342× |
| 1024 | llama.cpp Q4_K_S | 628.38 | 29.24 | 1.000× |
| 1024 | MLC q4f16_1 | 187.78 | 9.66 | 0.330× |
| 4096 | llama.cpp Q4_K_S | 603.96 | 28.49 | 1.000× |
| 4096 | MLC q4f16_1 | 165.10 | 8.36 | 0.293× |
