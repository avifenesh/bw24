
### agg tok/s  (median of N reps)

| c | B-batched | G-gated | M-mixed | Q-bounded | N |
|---|---|---|---|---|---|
| 6 | 456.1 | 445.6 | 393.9 | 390.4 | 5-5 |

### TTFT p50  (median of N reps)

| c | B-batched | G-gated | M-mixed | Q-bounded | N |
|---|---|---|---|---|---|
| 6 | 0.013 | 0.014 | 0.014 | 0.014 | 5-5 |

### TTFT p95  (median of N reps)

| c | B-batched | G-gated | M-mixed | Q-bounded | N |
|---|---|---|---|---|---|
| 6 | 0.079 | 0.518 | 0.517 | 0.365 | 5-5 |

### per-stream p50  (median of N reps)

| c | B-batched | G-gated | M-mixed | Q-bounded | N |
|---|---|---|---|---|---|
| 6 | 6.728 | 6.753 | 6.752 | 6.839 | 5-5 |

### per-stream p95  (median of N reps)

| c | B-batched | G-gated | M-mixed | Q-bounded | N |
|---|---|---|---|---|---|
| 6 | 6.767 | 7.249 | 11.243 | 11.075 | 5-5 |

### run health (totals across reps)

| arm | c | n_ok | n_err | n_shed | errors |
|---|---|---|---|---|---|
| B-batched | 6 | 90 | 0 | 0 |  |
| G-gated | 6 | 90 | 0 | 0 |  |
| M-mixed | 6 | 90 | 0 | 0 |  |
| Q-bounded | 6 | 90 | 0 | 0 |  |

### per-rep agg tok/s (spread check)

| arm | c | reps | min | median | max | spread |
|---|---|---|---|---|---|---|
| B-batched | 6 | 5 | 455.1 | 456.1 | 467.8 | 2.8% |
| G-gated | 6 | 5 | 444.8 | 445.6 | 448.2 | 0.8% |
| M-mixed | 6 | 5 | 393.4 | 393.9 | 399.7 | 1.6% |
| Q-bounded | 6 | 5 | 389.2 | 390.4 | 393.8 | 1.2% |
