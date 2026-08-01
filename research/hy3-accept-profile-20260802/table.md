## Per-rep rows

| class | rep | prompt tok | plain tok/s | spec tok/s | spec/plain | accepted/drafted | acceptance = rounds-accept% | gate |
|---|---|---|---|---|---|---|---|---|
| chat-qa-short | r1 | 25 | 0.43 | 2.07 | 4.80x (COLD, excluded) | 40/87 | 46.0% | PASS |
| chat-qa-short | r2 | 25 | 4.04 | 2.96 | 0.73x | 40/87 | 46.0% | PASS |
| chat-qa-short | r3 | 25 | 1.65 | 2.58 | 1.56x | 40/87 | 46.0% | PASS |
| chat-qa-short | r4 | 25 | 4.04 | 2.96 | 0.73x | 40/87 | 46.0% | PASS |
| chat-qa-short | r5 | 25 | 4.03 | 2.96 | 0.73x | 40/87 | 46.0% | PASS |
| chat-prose-medium | r1 | 38 | 1.92 | 2.88 | 1.50x (COLD, excluded) | 39/89 | 43.8% | PASS |
| chat-prose-medium | r2 | 38 | 2.83 | 3.03 | 1.07x | 39/89 | 43.8% | PASS |
| chat-prose-medium | r3 | 38 | 2.53 | 3.01 | 1.19x | 39/89 | 43.8% | PASS |
| code-gen-short | r1 | 42 | 2.09 | 3.26 | 1.56x (COLD, excluded) | 55/73 | 75.3% | PASS |
| code-gen-short | r2 | 42 | 3.72 | 3.34 | 0.90x | 55/73 | 75.3% | PASS |
| code-gen-short | r3 | 42 | 3.39 | 3.33 | 0.98x | 55/73 | 75.3% | PASS |
| code-review-medium | r1 | 1799 | 2.46 | 3.07 | 1.25x | 50/77 | 64.9% | PASS |
| code-review-medium | r2 | 1799 | 2.46 | 2.40 | 0.97x | 50/77 | 64.9% | PASS |
| code-review-medium | r3 | 1799 | 2.46 | 3.08 | 1.25x | 50/77 | 64.9% | PASS |
| agentic-tool | r1 | 655 | 2.49 | 3.02 | 1.21x | 50/77 | 64.9% | PASS |
| agentic-tool | r2 | 655 | 2.48 | 3.01 | 1.21x | 50/77 | 64.9% | PASS |
| agentic-tool | r3 | 655 | 2.49 | 3.02 | 1.21x | 50/77 | 64.9% | PASS |
| summarize-medium | r1 | 2096 | 2.46 | 2.63 | 1.07x | 39/88 | 44.3% | PASS |
| summarize-medium | r2 | 2096 | 2.46 | 2.64 | 1.07x | 39/88 | 44.3% | PASS |
| summarize-medium | r3 | 2096 | 2.46 | 2.63 | 1.07x | 39/88 | 44.3% | PASS |

## Per-class medians (K=1, NGEN=128, greedy, chat-templated; warm-storage reps only for tok/s)

| class | prompt tok | N(acc)/N(warm) | acceptance = rounds-accept% | plain tok/s | spec tok/s | spec/plain @floor | PP-2 ceiling 1+r | PP-2 est 1+r/2 |
|---|---|---|---|---|---|---|---|---|
| chat-qa-short | 25 | 5/4 | 46.0% | 4.04 | 2.96 | 0.73x | 1.46x | 1.23x |
| chat-prose-medium | 38 | 3/2 | 43.8% | 2.68 | 3.02 | 1.13x | 1.44x | 1.22x |
| code-gen-short | 42 | 3/2 | 75.3% | 3.56 | 3.33 | 0.94x | 1.75x | 1.38x |
| code-review-medium | 1799 | 3/3 | 64.9% | 2.46 | 3.07 | 1.25x | 1.65x | 1.32x |
| agentic-tool | 655 | 3/3 | 64.9% | 2.49 | 3.02 | 1.21x | 1.65x | 1.32x |
| summarize-medium | 2096 | 3/3 | 44.3% | 2.46 | 2.63 | 1.07x | 1.44x | 1.22x |
