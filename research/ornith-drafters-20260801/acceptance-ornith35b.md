### ornith35b run-spec K=1..8 self-consistency: PASS (8/8 identical, acceptance>0)
per-K acceptance: K1:78.9% K2:67.6% K3:56.7% K4:47.2% K5:37.7% K6:31.4% K7:26.9% K8:23.6%

### ornith35b acceptance table (greedy, ngen 256, board prompts)

| K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|
| 2 | 65.9% / 1.39x | 63.8% / 1.11x | 63.8% / 1.00x |
| 3 | 54.1% / 1.28x | 56.5% / 1.62x | 52.5% / 0.98x |
| 4 | 45.8% / 1.17x | 46.1% / 0.98x | 42.8% / 0.89x |

### ornith35b e2e spec vs plain @K=2 (interleaved in-process, x3)

| class | rep1 | rep2 | rep3 | median ratio | median acc |
|---|---|---|---|---|---|
| p1-code-short | 1.38x | 1.45x | 1.38x | 1.38x | 65.9% |
| p2-code-medium | 1.09x | 1.09x | 1.10x | 1.09x | 63.8% |
| p3-agentic-long | 1.04x | 1.05x | 1.05x | 1.05x | 63.8% |
