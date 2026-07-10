# Five-arm artifact sizes — G7e staging, 2026-07-10

All five strictly validated artifacts use the same 24,999,514,624-byte non-expert payload. The
current research artifacts are sparse bw24 repack overlays sourced from the pinned BF16
safetensors checkpoint; they are not standalone GGUF files. `Logical bytes` adds that common
payload once to the expert overlay. Final standalone GGUF sizes can differ when common tensors are
exported and quantized.

| arm | expert overlay bytes | staged directory bytes | overlay GiB | logical bytes | logical GiB |
|---|---:|---:|---:|---:|---:|
| `plain_quant` | 161,036,107,776 | 161,050,318,253 | 149.977 | 186,035,622,400 | 173.259 |
| `plain_reap_quant` | 80,518,053,888 | 80,525,342,350 | 74.988 | 105,517,568,512 | 98.271 |
| `plain_reap_mix_quant` | 63,743,459,328 | 63,750,744,562 | 59.366 | 88,742,973,952 | 82.648 |
| `mix_quant` | 110,220,705,792 | 110,233,241,054 | 102.651 | 135,220,220,416 | 125.934 |
| `mix_quant_prune25` | 94,496,882,688 | 94,507,626,343 | 88.007 | 119,496,397,312 | 111.290 |

Counts below are layer-expert slots across 79 MoE layers; every retained slot has three expert
projections in its assigned encoding.

| arm | NVFP4 | Q3_K | Q2_K | pruned |
|---|---:|---:|---:|---:|
| `plain_quant` | 15,168 | 0 | 0 | 0 |
| `plain_reap_quant` | 7,584 | 0 | 0 | 7,584 |
| `plain_reap_mix_quant` | 3,792 | 0 | 3,792 | 7,584 |
| `mix_quant` | 3,363 | 6,620 | 3,363 | 1,822 |
| `mix_quant_prune25` | 3,792 | 3,792 | 3,792 | 3,792 |

`mix_quant_prune25` is exactly 48 NVFP4, 48 Q3_K, 48 Q2_K, and 48 pruned experts in every layer.
Its frozen plan SHA-256 is
`344367993f53a99043a71e0ec00ea608d0a54fb2bab15c0054ad5b3627c11bba`.
