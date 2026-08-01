# Lane status — COMPLETE (see RESULTS.md)

| stage | ornith9b | ornith35b | katcoder |
|---|---|---|---|
| own-gen corpus (254 prompts) | DONE 127,390 tok | DONE 128,617 tok | DONE 119,273 tok |
| ranks artifact (owngen-ranks-32768) | DONE sha d40d7f4f | DONE sha a423317a | DONE sha e2aed7f6 |
| drafter built (donor block + own trim) | DONE sha 5f2de011 | DONE sha 78ff4bfb | DONE sha b7ff069a |
| run-spec K=1..8 self-consistency | PASS 8/8 | PASS 8/8 | PASS 8/8 |
| acceptance table K=2..4 (p1/p2/p3) | DONE | DONE | DONE (+K=1 probe) |
| e2e spec/plain x3 (serving K) | 2.16/1.77/1.70x — ADOPT | 1.38/1.09/1.05x — ADOPT | 1.09/0.91/0.85x — NO ADOPT |

Rig notes: corpora chunked (64 prompts/chunk, `gen-corpus-chunk.sh`) under
`flock /tmp/gpu5090.lock`, interleaved with two sibling lanes; one transient
CUDA_ERROR_LAUNCH_TIMEOUT during the 9B corpus (captured in `corpus/ornith9b-owngen.log`,
did not reproduce on resume — contention-class). The Ornith-35B corpus crash under the
default gen-graph door was root-caused and fixed (b5a08d23).
