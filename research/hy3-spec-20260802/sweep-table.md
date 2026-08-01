# Hy3 MTP spec K-sweep — per-run table (machine-parsed)

Generated: `python3 parse-sweep.py logs/sweep-r1.log logs/sweep-r2.log logs/sweep-r3.log`
Protocol/provenance: SUMMARY.md. Raw logs in logs/. N=3 full K=1..8 batteries,
board-d1736 raw prompt (1818 tok), MEMRA_NGEN=128, Mumbai H100, build 2b9a6aa6 (sm_90a).
Run windows (UTC 2026-08-01): r1 14:53:16-15:44:10, r2 16:00:11-16:49:18, r3 17:12:57-18:02:04.

| log | plain gen tok/s | prime s |
|---|---|---|
| sweep-r1 | 2.38 | 187.4 | (battery: PASS)
| sweep-r2 | 2.49 | 152.0 | (battery: PASS)
| sweep-r3 | 2.49 | 152.3 | (battery: PASS)

| K | self-consistency | acceptance %% (acc/drafted) | spec tok/s | ratio vs plain | N |
|---|---|---|---|---|---|
| 1 | PASS | 8.5% (10/117 10/117 10/117) | 2.08 | 0.84x | 3 |
|   |  | per-run acc: 8.5 8.5 8.5 | per-run: 2.06 2.08 2.08 | per-run: 0.87 0.84 0.84 | |
| 2 | PASS | 4.3% (10/234 10/234 10/234) | 1.44 | 0.58x | 3 |
|   |  | per-run acc: 4.3 4.3 4.3 | per-run: 1.43 1.44 1.44 | per-run: 0.60 0.58 0.58 | |
| 3 | PASS | 2.8% (10/351 10/351 10/351) | 1.11 | 0.45x | 3 |
|   |  | per-run acc: 2.8 2.8 2.8 | per-run: 1.11 1.11 1.11 | per-run: 0.47 0.45 0.45 | |
| 4 | PASS | 2.1% (10/468 10/468 10/468) | 0.91 | 0.37x | 3 |
|   |  | per-run acc: 2.1 2.1 2.1 | per-run: 0.87 0.91 0.91 | per-run: 0.36 0.37 0.37 | |
| 5 | PASS | 1.7% (10/585 10/585 10/585) | 0.80 | 0.32x | 3 |
|   |  | per-run acc: 1.7 1.7 1.7 | per-run: 0.80 0.80 0.80 | per-run: 0.34 0.32 0.32 | |
| 6 | PASS | 1.4% (10/702 10/702 10/702) | 0.67 | 0.27x | 3 |
|   |  | per-run acc: 1.4 1.4 1.4 | per-run: 0.68 0.67 0.67 | per-run: 0.28 0.27 0.27 | |
| 7 | PASS | 1.2% (10/819 10/819 10/819) | 0.61 | 0.25x | 3 |
|   |  | per-run acc: 1.2 1.2 1.2 | per-run: 0.61 0.61 0.61 | per-run: 0.26 0.25 0.25 | |
| 8 | PASS | 1.1% (10/936 10/936 10/936) | 0.57 | 0.23x | 3 |
|   |  | per-run acc: 1.1 1.1 1.1 | per-run: 0.57 0.57 0.57 | per-run: 0.24 0.23 0.23 | |
