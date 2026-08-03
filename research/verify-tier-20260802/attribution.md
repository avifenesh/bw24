## Cost curve — decode(T=1) vs verify(T=1..9), probe medians (N per json)

### q27 (ctx=2048, N=50), decode T=1 = 21.042ms

| T | verify ms | x decode | premium ms | us/extra-column (vs T=1) | marginal us/col (T-1 -> T) |
|---|---|---|---|---|---|
| 1 | 21.681 | 1.030x | +0.639 | 0 | nan |
| 2 | 23.297 | 1.107x | +2.256 | 1616 | 1616 |
| 3 | 24.400 | 1.160x | +3.358 | 1360 | 1103 |
| 4 | 26.223 | 1.246x | +5.181 | 1514 | 1823 |
| 5 | 32.654 | 1.552x | +11.612 | 2743 | 6431 |
| 6 | 34.894 | 1.658x | +13.852 | 2643 | 2240 |
| 7 | 37.464 | 1.780x | +16.422 | 2631 | 2570 |
| 8 | 40.308 | 1.916x | +19.267 | 2661 | 2844 |
| 9 | 94.853 | 4.508x | +73.812 | 9147 | 54545 |

### q9 (ctx=2048, N=50), decode T=1 = 7.336ms

| T | verify ms | x decode | premium ms | us/extra-column (vs T=1) | marginal us/col (T-1 -> T) |
|---|---|---|---|---|---|
| 1 | 7.682 | 1.047x | +0.345 | 0 | nan |
| 2 | 8.852 | 1.207x | +1.515 | 1170 | 1170 |
| 3 | 9.207 | 1.255x | +1.870 | 763 | 355 |
| 4 | 9.793 | 1.335x | +2.457 | 704 | 587 |
| 5 | 11.896 | 1.622x | +4.560 | 1054 | 2103 |
| 6 | 12.995 | 1.771x | +5.658 | 1063 | 1099 |
| 7 | 13.959 | 1.903x | +6.623 | 1046 | 964 |
| 8 | 14.886 | 2.029x | +7.550 | 1029 | 927 |
| 9 | 29.174 | 3.977x | +21.838 | 2687 | 14288 |

## Per-arm kernel attribution (nsys kern_sum; per-pass = total/18)

### q27 decode_h — probe wall 21.04ms/pass

| m=1 kernel | ms/step | launches/step | avg us |
|---|---|---|---|
| qmatvec_nvfp4_mmvq_mr2_rp | 9.243 | 272 | 34.0 |
| qmatvec_nvfp4_mmvq_dual_mr2_rp | 8.214 | 112 | 73.3 |
| qmatvec_q5_K_mmvq_il | 1.076 | 1 | 1037.2 |
| qmatvec_nvfp4_dp4a_rp | 0.950 | 4 | 267.2 |
### q27 verify_t2 — probe wall 23.30ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_b2_rp | 7.854 | 33.7% | 64 | 122.7 |
| qmatvec_nvfp4_mmvq_b2_rpr2 | 5.513 | 23.7% | 128 | 43.1 |
| qmatvec_nvfp4_mmvq_b2_rp | 4.573 | 19.6% | 240 | 19.1 |
| qmatvec_q5_K_mmvq_b2_r2 | 1.089 | 4.7% | 1 | 1089.5 |
| **sum b-tier** | **19.030** | **81.7%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 1.4ms/pass = 6% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_dp4a_rp | 1.310 | 265.5 |
| qmatvec_q5_K_mmvq_il | 0.056 | 1050.6 |

### q27 verify_t3 — probe wall 24.40ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_b4_rpr2 | 8.252 | 33.8% | 64 | 128.9 |
| qmatvec_nvfp4_mmvq_b4_rpr2w8 | 7.581 | 31.1% | 176 | 43.1 |
| qmatvec_nvfp4_mmvq_b4_rp | 2.408 | 9.9% | 176 | 13.7 |
| qmatvec_q5_K_mmvq_b4_r2 | 1.133 | 4.6% | 1 | 1133.1 |
| qmatvec_nvfp4_mmvq_b4_rpr2 | 0.762 | 3.1% | 16 | 47.6 |
| **sum b-tier** | **20.136** | **82.5%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 1.3ms/pass = 5% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_dp4a_rp | 1.264 | 266.7 |
| qmatvec_q5_K_mmvq_il | 0.053 | 1045.8 |

### q27 verify_t4 — probe wall 26.22ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_b4_rpr2 | 8.737 | 33.3% | 64 | 136.5 |
| qmatvec_nvfp4_mmvq_b4_rpr2w8 | 7.887 | 30.1% | 176 | 44.8 |
| qmatvec_nvfp4_mmvq_b4_rp | 2.835 | 10.8% | 176 | 16.1 |
| qmatvec_q5_K_mmvq_b4_r2 | 1.200 | 4.6% | 1 | 1200.1 |
| qmatvec_nvfp4_mmvq_b4_rpr2 | 0.815 | 3.1% | 16 | 50.9 |
| **sum b-tier** | **21.475** | **81.9%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 1.3ms/pass = 5% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_dp4a_rp | 1.199 | 264.5 |
| qmatvec_q5_K_mmvq_il | 0.051 | 1043.9 |

### q27 verify_t5 — probe wall 32.65ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_b8_rpsc | 25.699 | 78.7% | 496 | 51.8 |
| qmatvec_q5_K_mmvq_b8_r2 | 1.341 | 4.1% | 1 | 1340.5 |
| **sum b-tier** | **27.040** | **82.8%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 1.2ms/pass = 4% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_dp4a_rp | 1.151 | 265.2 |

### q27 verify_t6 — probe wall 34.89ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_b8_rpsc | 28.005 | 80.3% | 496 | 56.5 |
| qmatvec_q5_K_mmvq_b8_r2 | 1.444 | 4.1% | 1 | 1443.5 |
| **sum b-tier** | **29.449** | **84.4%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 1.1ms/pass = 3% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_dp4a_rp | 1.094 | 264.5 |

### q27 verify_t8 — probe wall 40.31ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_b8_rpsc | 32.795 | 81.4% | 496 | 66.1 |
| qmatvec_q5_K_mmvq_b8_r2 | 1.640 | 4.1% | 1 | 1640.0 |
| **sum b-tier** | **34.435** | **85.4%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 1.0ms/pass = 2% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_dp4a_rp | 0.990 | 264.9 |

### q27 verify_t9 — probe wall 94.85ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| **sum b-tier** | **0.000** | **0.0%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 91.9ms/pass = 97% of wall  <- THE OFF-TIER PASS

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_mmvq_rp | 81.778 | 164.9 |
| qmatvec_q5_K_mmvq | 9.232 | 9232.3 |
| qmatvec_nvfp4_dp4a_rp | 0.932 | 263.8 |

### q9 decode_h — probe wall 7.34ms/pass

| m=1 kernel | ms/step | launches/step | avg us |
|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_mr2_rp | 1.988 | 29 | 68.6 |
| qmatvec_nvfp4_mmvq_mr2_rp | 1.621 | 55 | 29.5 |
| qmatvec_q6_K_mmvq | 1.006 | 1 | 970.3 |
| qmatvec_q4_K_mmvq | 0.878 | 43 | 20.4 |
| qmatvec_q5_K_mmvq_mr2_il | 0.627 | 44 | 14.2 |
| qmatvec_q8_0_dp4a | 0.273 | 2 | 153.6 |
### q9 verify_t2 — probe wall 8.85ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_b2_rp | 2.013 | 22.7% | 29 | 69.4 |
| qmatvec_nvfp4_mmvq_b2_rpr2 | 1.183 | 13.4% | 31 | 38.2 |
| qmatvec_q6_K_mmvq_b2_r2 | 1.032 | 11.7% | 1 | 1031.6 |
| qmatvec_q4_K_mmvq_b2_r2 | 0.883 | 10.0% | 43 | 20.5 |
| qmatvec_q5_K_mmvq_b2 | 0.808 | 9.1% | 44 | 18.4 |
| qmatvec_nvfp4_mmvq_b2_rp | 0.519 | 5.9% | 24 | 21.6 |
| **sum b-tier** | **6.438** | **72.7%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 0.4ms/pass = 5% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_q8_0_dp4a | 0.375 | 152.1 |
| qmatvec_q6_K_mmvq | 0.052 | 980.9 |

### q9 verify_t3 — probe wall 9.21ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_b4_rpr2 | 2.117 | 23.0% | 29 | 73.0 |
| qmatvec_nvfp4_mmvq_b4_rpr2 | 1.467 | 15.9% | 45 | 32.6 |
| qmatvec_q6_K_mmvq_b4_r2 | 1.026 | 11.1% | 1 | 1026.2 |
| qmatvec_q4_K_mmvq_b4_r2 | 0.941 | 10.2% | 43 | 21.9 |
| qmatvec_q5_K_mmvq_b4 | 0.898 | 9.8% | 44 | 20.4 |
| qmatvec_nvfp4_mmvq_b4_rp | 0.275 | 3.0% | 10 | 27.5 |
| **sum b-tier** | **6.725** | **73.0%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 0.4ms/pass = 4% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_q8_0_dp4a | 0.359 | 151.8 |
| qmatvec_q6_K_mmvq | 0.051 | 982.5 |

### q9 verify_t4 — probe wall 9.79ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_dual_b4_rpr2 | 2.227 | 22.7% | 29 | 76.8 |
| qmatvec_nvfp4_mmvq_b4_rpr2 | 1.525 | 15.6% | 45 | 33.9 |
| qmatvec_q6_K_mmvq_b4_r2 | 1.055 | 10.8% | 1 | 1055.1 |
| qmatvec_q5_K_mmvq_b4 | 1.044 | 10.7% | 44 | 23.7 |
| qmatvec_q4_K_mmvq_b4_r2 | 1.018 | 10.4% | 43 | 23.7 |
| qmatvec_nvfp4_mmvq_b4_rp | 0.305 | 3.1% | 10 | 30.5 |
| **sum b-tier** | **7.174** | **73.2%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 0.3ms/pass = 3% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_q8_0_dp4a | 0.343 | 151.2 |

### q9 verify_t5 — probe wall 11.90ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_b8_rpsc | 4.865 | 40.9% | 113 | 43.1 |
| qmatvec_q4_K_mmvq_b8_r2 | 1.394 | 11.7% | 43 | 32.4 |
| qmatvec_q6_K_mmvq_b8_r2 | 1.231 | 10.3% | 1 | 1231.2 |
| qmatvec_q5_K_mmvq_b8 | 1.207 | 10.1% | 44 | 27.4 |
| qmatvec_q8_0_mmvq_b8 | 0.348 | 2.9% | 48 | 7.3 |
| **sum b-tier** | **9.046** | **76.0%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 0.3ms/pass = 3% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_q8_0_dp4a | 0.330 | 152.1 |

### q9 verify_t8 — probe wall 14.89ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_nvfp4_mmvq_b8_rpsc | 6.084 | 40.9% | 113 | 53.8 |
| qmatvec_q4_K_mmvq_b8_r2 | 1.891 | 12.7% | 43 | 44.0 |
| qmatvec_q5_K_mmvq_b8 | 1.612 | 10.8% | 44 | 36.6 |
| qmatvec_q6_K_mmvq_b8_r2 | 1.479 | 9.9% | 1 | 1478.5 |
| qmatvec_q8_0_mmvq_b8 | 0.447 | 3.0% | 48 | 9.3 |
| **sum b-tier** | **11.512** | **77.3%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 0.3ms/pass = 2% of wall

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_q8_0_dp4a | 0.284 | 151.9 |

### q9 verify_t9 — probe wall 29.17ms/pass

| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |
|---|---|---|---|---|
| qmatvec_q6_K_mmvq_b16 | 1.766 | 6.1% | 1 | 1765.8 |
| **sum b-tier** | **1.766** | **6.1%** | | |

m=1-class kernels WITH per-pass residency (continuation-corrected vs the decode_h profile): 26.3ms/pass = 90% of wall  <- THE OFF-TIER PASS

| kernel | ms/pass (corrected) | avg us/launch |
|---|---|---|
| qmatvec_nvfp4_mmvq_rp | 17.646 | 156.2 |
| qmatvec_q5_K_mmvq | 4.196 | 95.4 |
| qmatvec_q4_K_mmvq | 4.022 | 69.2 |
| qmatvec_q8_0_dp4a | 0.268 | 151.5 |
| qmatvec_q8_0_mmvq | 0.146 | 3.1 |


