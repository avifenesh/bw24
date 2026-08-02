---
license: apache-2.0
base_model: stepfun-ai/Step-3.7-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - quantized
  - imatrix
  - moe
  - agent
  - tool-calling
  - reasoning
  - vision
  - multimodal
language:
  - en
  - zh
  - ja
  - ko
  - ar
  - hi
  - de
  - fr
  - es
  - ru
---

**[ModelPage]**: https://static.stepfun.com/blog/step-3.7-flash/

## 1. Introduction

GGUF quantizations of [`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash).

Step-3.7-Flash is a 198B-parameter sparse Mixture-of-Experts vision-language model from StepFun-ai, activating ~11B parameters per token for up to 400 t/s throughput. It pairs a 196B-parameter language backbone with a 1.8B-parameter vision encoder for native image understanding, supports a 256K context window, and offers three selectable reasoning levels (low / medium / high) to balance speed, cost, and depth. Built for agentic workloads — tool calling, multi-step reasoning, code, and math — with native multilingual coverage.

A separate `mmproj` projector ships alongside the language quants for multimodal inference. With 128 GB of unified memory (Mac Studio, DGX Spark, Ryzen AI Max+ 395, etc.), you can privately host Step-3.7-Flash: Q4 quants and below run at full 256K context with high precision.

## 2. Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `Step-3.7-flash-BF16.gguf` | BF16 | 394 GB | Full-precision reference.|
| `Step-3.7-flash-Q8_0.gguf` | Q8_0 | 209 GB | Near-lossless. Does **not** use imatrix.|
| `Step-3.7-flash-Q4_K_S.gguf` | Q4_K_S | 112 GB | imatrix-calibrated. Balanced quality / size.|
| `Step-3.7-flash-IQ4_XS.gguf` | IQ4_XS | 105 GB | imatrix-calibrated. Slightly smaller than Q4_K_S, comparable quality. |
| `Step-3.7-flash-Q3_K_L.gguf` | Q3_K_L | 103 GB | imatrix-calibrated. Aggressive size reduction. |
| `Step-3.7-flash-Q3_K_M.gguf` | Q3_K_M | 94 GB | imatrix-calibrated. Use when you need to fit on a single 64-96 GB device; expect modest quality loss at low bit-widths. |
| `Step-3.7-flash-IQ3_XXS.gguf` | IQ3_XXS | 76 GB | imatrix-calibrated. Recommended only when memory is the primary constraint; offers the smallest footprint among the provided quantizations. |
| `mmproj-Step-3.7-flash-f16.gguf` | F16 | 4 GB | Vision projector. Pair with any of the language quants above for image input. |

## 3. Quickstart

Build llama.cpp and run:

```bash
# 1. Clone and build
git clone https://github.com/stepfun-ai/llama.cpp.git
cd llama.cpp
git checkout -b step3.7 origin/step3.7
cmake -B build -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release -j$(nproc)

# 2. Test performance (benchmark)
./build/bin/llama-batched-bench \
  -m Step-3.7-flash-Q4_K_S.gguf \
  -c 32768 -b 2048 -ub 2048 \
  -npp 0,2048,8192,16384,32768 -ntg 128 -npl 1

# 3. Text-only inference
./build/bin/llama-cli \
  -m Step-3.7-flash-Q4_K_S.gguf \
  -c 32768 -ngl 99 -fa on \
  -p "Write a Python function to compute the n-th Fibonacci number."

# 4. With vision (image + text)
./build/bin/llama-mtmd-cli \
  -m Step-3.7-flash-Q4_K_S.gguf \
  --mmproj mmproj-Step-3.7-flash-f16.gguf \
  -c 32768 -ngl 99 -fa on \
  --image path/to/image.jpg \
  -p "Describe this image."

# 5. OpenAI-compatible server (text + vision)
./build/bin/llama-server \
  -m Step-3.7-flash-Q4_K_S.gguf \
  --mmproj mmproj-Step-3.7-flash-f16.gguf \
  -c 32768 -ngl 99 -fa on \
  --host 0.0.0.0 --port 8080
```

For full CLI / server options, see the [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

## 4. Performance

### Apple Mac Studio (M4 max, 128 GB unified memory)

**Step-3.7-flash-Q4_K_S**

```
./llama-batched-bench -m Step-3.7-flash-Q4_K_S.gguf -c 262150 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536,131072,262144 -ntg 128 -npl 1
```

| PP | TG | PL | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
|---:|---:|---:|-----:|-------:|---------:|-------:|---------:|----:|------:|
| 0 | 128 | 1 | 128 | 0.000 | 0.00 | 2.500 | 51.20 | 2.500 | 51.20 |
| 2048 | 128 | 1 | 2176 | 4.873 | 420.28 | 2.639 | 48.51 | 7.512 | 289.68 |
| 8192 | 128 | 1 | 8320 | 20.292 | 403.70 | 2.757 | 46.43 | 23.049 | 360.97 |
| 16384 | 128 | 1 | 16512 | 42.854 | 382.32 | 2.924 | 43.77 | 45.779 | 360.69 |
| 32768 | 128 | 1 | 32896 | 95.168 | 344.32 | 3.223 | 39.72 | 98.391 | 334.34 |
| 65536 | 128 | 1 | 65664 | 233.885 | 280.21 | 3.909 | 32.74 | 237.794 | 276.14 |
| 131072 | 128 | 1 | 131200 | 635.499 | 206.25 | 5.759 | 22.23 | 641.258 | 204.60 |
| 262144 | 128 | 1 | 262272 | 2362.488 | 110.96 | 13.188 | 9.71 | 2375.677 | 110.40 |

**Step-3.7-flash-IQ4_XS**

```
./llama-batched-bench -m Step-3.7-flash-IQ4_XS.gguf -c 262150 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536,131072,262144 -ntg 128 -npl 1
```

| PP | TG | PL | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
|---:|---:|---:|-----:|-------:|---------:|-------:|---------:|----:|------:|
| 0 | 128 | 1 | 128 | 0.000 | 0.00 | 2.582 | 49.58 | 2.582 | 49.58 |
| 2048 | 128 | 1 | 2176 | 4.835 | 423.56 | 2.679 | 47.78 | 7.514 | 289.60 |
| 8192 | 128 | 1 | 8320 | 19.954 | 410.55 | 2.803 | 45.66 | 22.757 | 365.60 |
| 16384 | 128 | 1 | 16512 | 42.142 | 388.78 | 2.957 | 43.29 | 45.098 | 366.13 |
| 32768 | 128 | 1 | 32896 | 93.489 | 350.50 | 3.288 | 38.93 | 96.777 | 339.91 |
| 65536 | 128 | 1 | 65664 | 227.088 | 288.59 | 3.945 | 32.44 | 231.033 | 284.22 |
| 131072 | 128 | 1 | 131200 | 635.047 | 206.40 | 5.791 | 22.10 | 640.838 | 204.73 |
| 262144 | 128 | 1 | 262272 | 2170.271 | 120.79 | 13.070 | 9.79 | 2183.342 | 120.12 |

**Step-3.7-flash-Q3_K_L**

```
./llama-batched-bench -m Step-3.7-flash-Q3_K_L.gguf -c 262272 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536,131072,262144 -ntg 128 -npl 1
```

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     0 |    128 |    1 |    128 |    0.000 |     0.00 |    3.590 |    35.66 |    3.590 |    35.66 |
|  2048 |    128 |    1 |   2176 |    5.263 |   389.15 |    3.702 |    34.57 |    8.965 |   242.72 |
|  8192 |    128 |    1 |   8320 |   21.789 |   375.97 |    3.817 |    33.53 |   25.606 |   324.92 |
| 16384 |    128 |    1 |  16512 |   45.819 |   357.58 |    3.977 |    32.18 |   49.796 |   331.59 |
| 32768 |    128 |    1 |  32896 |  100.827 |   324.99 |    4.308 |    29.71 |  105.135 |   312.89 |
| 65536 |    128 |    1 |  65664 |  242.172 |   270.62 |    4.977 |    25.72 |  247.149 |   265.69 |
|131072 |    128 |    1 | 131200 |  659.645 |   198.70 |    6.764 |    18.92 |  666.409 |   196.88 |
|262144 |    128 |    1 | 262272 | 2200.370 |   119.14 |   14.008 |     9.14 | 2214.378 |   118.44 |

### NVIDIA DGX Spark (GB10, 128 GB unified memory)

**Step-3.7-flash-Q4_K_S**

```
./llama-batched-bench -m Step-3.7-flash-Q4_K_S.gguf -c 131300 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536,131072 -ntg 128 -npl 1
```

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     0 |    128 |    1 |    128 |    0.000 |     0.00 |    5.157 |    24.82 |    5.157 |    24.82 |
|  2048 |    128 |    1 |   2176 |    8.021 |   255.33 |    4.907 |    26.08 |   12.929 |   168.31 |
|  8192 |    128 |    1 |   8320 |   10.866 |   753.89 |    5.169 |    24.76 |   16.035 |   518.86 |
| 16384 |    128 |    1 |  16512 |   29.389 |   557.49 |    6.215 |    20.60 |   35.603 |   463.78 |
| 32768 |    128 |    1 |  32896 |   52.501 |   624.14 |    6.931 |    18.47 |   59.432 |   553.50 |
| 65536 |    128 |    1 |  65664 |  112.321 |   583.47 |    7.769 |    16.48 |  120.090 |   546.79 |
|131072 |    128 |    1 | 131200 |  281.479 |   465.66 |    9.834 |    13.02 |  291.313 |   450.37 |

**Step-3.7-flash-IQ4_XS**

```
./llama-batched-bench -m Step-3.7-flash-IQ4_XS.gguf -c 262272 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536,131072,262144 -ntg 128 -npl 1
```

| PP | TG | PL | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
|---:|---:|---:|-----:|-------:|---------:|-------:|---------:|----:|------:|
| 0 | 128 | 1 | 128 | 0.000 | 0.00 | 5.368 | 23.85 | 5.368 | 23.85 |
| 2048 | 128 | 1 | 2176 | 4.250 | 481.87 | 5.311 | 24.10 | 9.561 | 227.58 |
| 8192 | 128 | 1 | 8320 | 12.531 | 653.73 | 5.817 | 22.01 | 18.348 | 453.46 |
| 16384 | 128 | 1 | 16512 | 24.474 | 669.44 | 5.915 | 21.64 | 30.389 | 543.35 |
| 32768 | 128 | 1 | 32896 | 51.976 | 630.44 | 6.531 | 19.60 | 58.508 | 562.25 |
| 65536 | 128 | 1 | 65664 | 116.305 | 563.48 | 7.934 | 16.13 | 124.239 | 528.53 |
| 131072 | 128 | 1 | 131200 | 298.746 | 438.74 | 10.263 | 12.47 | 309.009 | 424.58 |
| 262144 | 128 | 1 | 262272 | 924.872 | 283.44 | 14.862 | 8.61 | 939.734 | 279.09 |

**Step-3.7-flash-Q3_K_L**

```
./llama-batched-bench -m Step-3.7-flash-Q3_K_L.gguf -c 262272 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536,131072,262144 -ntg 128 -npl 1
```

| PP | TG | PL | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
|---:|---:|---:|-----:|-------:|---------:|-------:|---------:|----:|------:|
| 0 | 128 | 1 | 128 | 0.000 | 0.00 | 5.947 | 21.52 | 5.947 | 21.52 |
| 2048 | 128 | 1 | 2176 | 4.145 | 494.08 | 5.623 | 22.76 | 9.768 | 222.77 |
| 8192 | 128 | 1 | 8320 | 14.889 | 550.20 | 5.799 | 22.07 | 20.688 | 402.17 |
| 16384 | 128 | 1 | 16512 | 29.374 | 557.78 | 6.140 | 20.85 | 35.513 | 464.95 |
| 32768 | 128 | 1 | 32896 | 54.957 | 596.25 | 6.744 | 18.98 | 61.702 | 533.15 |
| 65536 | 128 | 1 | 65664 | 129.827 | 504.79 | 8.347 | 15.33 | 138.174 | 475.23 |
| 131072 | 128 | 1 | 131200 | 315.402 | 415.57 | 10.780 | 11.87 | 326.182 | 402.23 |
| 262144 | 128 | 1 | 262272 | 910.215 | 288.00 | 15.568 | 8.22 | 925.783 | 283.30 |


### AMD Ryzen AI Max+ 395 (Strix Halo, 128 GB unified memory)

**Step-3.7-flash-Q4_K_S**

```
llama-batched-bench.exe -m Step-3.7-flash-Q4_K_S.gguf -c 65664 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536 -ntg 128 -npl 1
```

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     0 |    128 |    1 |    128 |    0.000 |     0.00 |    4.878 |    26.24 |    4.878 |    26.24 |
|  2048 |    128 |    1 |   2176 |    9.367 |   218.63 |    5.134 |    24.93 |   14.501 |   150.06 |
|  8192 |    128 |    1 |   8320 |   43.540 |   188.15 |    5.508 |    23.24 |   49.048 |   169.63 |
| 16384 |    128 |    1 |  16512 |  111.814 |   146.53 |    5.947 |    21.53 |  117.761 |   140.22 |
| 32768 |    128 |    1 |  32896 |  357.819 |    91.58 |    6.779 |    18.88 |  364.598 |    90.23 |
| 65536 |    128 |    1 |  65664 | 1342.501 |    48.82 |    8.495 |    15.07 | 1350.996 |    48.60 |

**Step-3.7-flash-IQ4_XS**

```
./llama-batched-bench -m Step-3.7-flash-IQ4_XS.gguf -c 65664 -b 2048 -ub 1024 -npp 0,2048,8192,16384,32768,65536 -ntg 128 -npl 1
```

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     0 |    128 |    1 |    128 |    0.000 |     0.00 |    5.931 |    21.58 |    5.931 |    21.58 |
|  2048 |    128 |    1 |   2176 |    8.143 |   251.50 |    6.194 |    20.67 |   14.337 |   151.78 |
|  8192 |    128 |    1 |   8320 |   39.899 |   205.32 |    6.521 |    19.63 |   46.420 |   179.23 |
| 16384 |    128 |    1 |  16512 |  105.098 |   155.89 |    6.891 |    18.57 |  111.989 |   147.44 |
| 32768 |    128 |    1 |  32896 |  338.645 |    96.76 |    7.793 |    16.42 |  346.439 |    94.95 |
| 65536 |    128 |    1 |  65664 | 1310.820 |    50.00 |    9.489 |    13.49 | 1320.309 |    49.73 |

**Step-3.7-flash-Q3_K_L**

```
./llama-batched-bench -m Step-3.7-flash-Q3_K_L.gguf -c 262272 -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 -npp 0,2048,8192,16384,32768,65536,131072,262144 -ntg 128 -npl 1
```

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     0 |    128 |    1 |    128 |    0.000 |     0.00 |    5.015 |    25.53 |    5.015 |    25.53 |
|  2048 |    128 |    1 |   2176 |   10.246 |   199.88 |    5.073 |    25.23 |   15.319 |   142.04 |
|  8192 |    128 |    1 |   8320 |   37.229 |   220.05 |    5.341 |    23.96 |   42.570 |   195.44 |
| 16384 |    128 |    1 |  16512 |   79.234 |   206.78 |    5.489 |    23.32 |   84.723 |   194.89 |
| 32768 |    128 |    1 |  32896 |  179.697 |   182.35 |    5.810 |    22.03 |  185.507 |   177.33 |
| 65536 |    128 |    1 |  65664 |  436.593 |   150.11 |    6.577 |    19.46 |  443.169 |   148.17 |
|131072 |    128 |    1 | 131200 | 1262.377 |   103.83 |    9.124 |    14.03 | 1271.501 |   103.19 |
|262144 |    128 |    1 | 262272 | 3487.921 |    75.16 |   11.391 |    11.24 | 3499.312 |    74.95 |

## 5. Acknowledgments

This release stands on the work of the following authors and communities:

- **[bartowski](https://huggingface.co/bartowski)** &mdash; for [`calibration_datav5`](https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d),
  the community-standard imatrix calibration anchor used by countless GGUF releases.
  Used for calibration purposes only; no license has been verified for this resource.
- **[eaddario](https://huggingface.co/eaddario)** &mdash; for the [`imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration)
  dataset (MIT), providing multilingual / code / math splits that form the backbone of this release's domain balance
- **[NousResearch](https://huggingface.co/NousResearch)** &mdash; for [`hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
  (Apache-2.0), used for agent / tool-call calibration coverage
- **[ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp)** &mdash; for the entire quantization and inference toolchain (MIT)

## 6. License

The GGUF quantization files in this repository are derivative works of
[`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash)
and are released under the same **Apache 2.0** license.

| Component | License |
|---|---|
| Base model weights ([stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash)) | Apache-2.0 |
| Calibration dataset ([eaddario/imatrix-calibration](https://huggingface.co/datasets/eaddario/imatrix-calibration)) | MIT |
| Calibration dataset ([NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)) | Apache-2.0 |
| Quantization toolchain ([llama.cpp](https://github.com/ggml-org/llama.cpp)) | MIT |

All calibration datasets retain their original licenses and are used strictly for quantization calibration purposes only.
