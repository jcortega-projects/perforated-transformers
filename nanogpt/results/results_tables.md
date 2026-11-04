# Midterm results tables

## Quality (2,000 steps, fixed validation batches)

| Model | Params used | Val loss seed 1337 | Val loss seed 2024 | Mean | Perplexity (mean) | Train time (s) | Time to val 3.0 (s) |
|---|---|---|---|---|---|---|---|
| Vanilla (16.1M) | 16,058,624 | 2.9484 | 2.9426 | 2.9455 | 19.02 | 237 | 187 |
| Vanilla, matched params (17.1M) | 17,111,296 | 2.9414 | 2.9316 | 2.9365 | 18.85 | 205 | 159 |
| Dendrites c_fc, default LR | 17,111,296 | 2.9482 | - | 2.9482 | 19.07 | 511 | 408 |
| Dendrites c_fc, 10x dendrite LR | 17,111,296 | 2.9308 | 2.9278 | 2.9293 | 18.71 | 486 | 403 |
| Dendrites attn.c_attn | 16,848,128 | 2.9478 | - | 2.9478 | 19.06 | 539 | 426 |
| Dendrites c_fc, block 0 only | 16,321,792 | 2.9484 | - | 2.9484 | 19.08 | 388 | 308 |
| Dendrites c_fc, block 3 only | 16,321,792 | 2.9483 | - | 2.9483 | 19.07 | 391 | 314 |

## Inference cost (RTX 4050 Laptop)

| Model | Prefill B32xT128 bf16 (tok/s) | Prefill B32xT128 fp32 (tok/s) | Decode B1 bf16 (tok/s) | Decode B1 fp32 (tok/s) | TTFT B1xT64 fp32 (ms) | Peak memory (MiB) |
|---|---|---|---|---|---|---|
| Vanilla (16.1M) | 371,804 | 204,360 | 166.7 | 233.5 | 4.4 ± 0.1 | 118 |
| Vanilla, matched params (17.1M) | 300,028 | 165,014 | 166.0 | 219.2 | 5.1 ± 0.3 | 138 |
| Dendrites c_fc, default LR | 122,413 | 75,586 | 48.4 | 23.8 | 115.4 ± 5.2 | 167 |
| Dendrites c_fc, 10x dendrite LR | 123,339 | 91,817 | 48.9 | 51.5 | 35.3 ± 24.1 | 167 |
| Dendrites attn.c_attn | 116,556 | 89,797 | 36.0 | 52.4 | 80.2 ± 3.7 | 145 |
| Dendrites c_fc, block 0 only | 243,314 | 134,902 | 115.7 | 130.4 | 23.7 ± 5.3 | 161 |
| Dendrites c_fc, block 3 only | 226,212 | 138,966 | 82.5 | 108.5 | 10.8 ± 0.7 | 161 |

## Profiler, per forward pass (B32xT128 bf16)

| Model | Wall (ms) | GPU busy (ms) | GPU idle | Kernels | GPU->CPU syncs | dtype casts (GPU ms) |
|---|---|---|---|---|---|---|
| Vanilla (16.1M) | 11.2 | 8.0 | 28% | 78 | 0 | 27 (0.69) |
| Vanilla, matched params (17.1M) | 15.2 | 11.0 | 28% | 74 | 0 | 27 (0.72) |
| Dendrites c_fc, block 3 only | 19.0 | 9.5 | 50% | 97 | 6 | 30 (0.91) |
| Dendrites c_fc, 10x dendrite LR | 40.4 | 14.4 | 64% | 154 | 24 | 39 (1.57) |
