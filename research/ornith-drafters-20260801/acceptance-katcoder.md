### katcoder run-spec K=1..8 self-consistency: PASS (8/8 identical, acceptance>0)
per-K acceptance: K1:92.4% K2:83.3% K3:75.2% K4:65.7% K5:58.8% K6:51.0% K7:44.2% K8:39.1%

### katcoder acceptance table (greedy, ngen 256, board prompts)

| K | p1-code-short | p2-code-medium | p3-agentic-long |
|---|---|---|---|
| 2 | 82.5% / 1.09x | 61.7% / 0.91x | 55.4% / 0.86x |
| 3 | 70.3% / 1.01x | 51.5% / 0.82x | 39.3% / 0.69x |
| 4 | 64.8% / 0.96x | 40.6% / 0.69x | 31.9% / 0.59x |

### katcoder e2e spec vs plain @K=2 (interleaved in-process, x3)

| class | rep1 | rep2 | rep3 | median ratio | median acc |
|---|---|---|---|---|---|
| p1-code-short | 1.09x | 1.09x | 1.09x | 1.09x | 82.5% |
| p2-code-medium | 0.91x | 0.91x | 0.92x | 0.91x | 61.7% |
| p3-agentic-long | 0.84x | 0.85x | 0.86x | 0.85x | 55.4% |
