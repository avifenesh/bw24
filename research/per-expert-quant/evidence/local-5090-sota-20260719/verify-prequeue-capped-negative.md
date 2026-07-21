# Frozen verification prequeue: capped negative

Responsive local envelope: systemd `CPUQuota=200%`, `CPUWeight=10`, four OpenMP workers pinned to
CPUs 0-3, 20 GiB normal-RAM expert cache, `MemoryHigh=34G`, and `MemoryMax=38G`. Both arms used the
same 32-token K=3 discarded profile, N=16 generation, K=1 scoring, prompt, model, and scalar CPU
companion. These are paired single runs with no intentional cooldown.

| arm | plain | K=1 | acceptance | exactness |
|---|---:|---:|---:|---|
| scalar control | 5.54 tok/s | 5.79 tok/s | 87.5% | PASS |
| verification prequeue | 5.49 tok/s | 5.40 tok/s | 87.5% | PASS |

The matched plain arms differ by 0.9%, while prequeue regresses K=1 by 6.7% and verify-issue time
rises from 2,338.8 ms to 2,529.2 ms. Under a shared CPU quota, advancing the second CPU row early
spends quota that the owner thread needs to issue and accumulate GPU work. The candidate and its
flag were removed; the raw logs remain as the negative record.
