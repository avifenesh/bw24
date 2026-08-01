### ornith9b run-spec K=1..8 self-consistency: PASS (8/8 identical, acceptance>0)
per-K acceptance: K1:89.6% K2:80.6% K3:71.5% K4:58.6% K5:52.8% K6:47.5% K7:40.7% K8:35.6%

### ornith9b acceptance table (greedy, ngen 256, board prompts)

| K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|
| 2 | 71.4% / 1.98x | 58.0% / 1.72x | 58.9% / 1.67x |
| 3 | 61.1% / 2.14x | 47.0% / 1.76x | 47.8% / 1.70x |
| 4 | 51.8% / 2.11x | 39.1% / 1.67x | 40.2% / 1.64x |

### ornith9b e2e spec vs plain @K=3 (interleaved in-process, x3)

| class | rep1 | rep2 | rep3 | median ratio | median acc |
|---|---|---|---|---|---|
| p1-code-short | 2.16x | 2.16x | 2.16x | 2.16x | 61.1% |
| p2-code-medium | 1.77x | 1.77x | 1.76x | 1.77x | 47.0% |
| p3-agentic-long | 1.70x | 1.70x | 1.70x | 1.70x | 47.8% |
