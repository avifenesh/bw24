# Audit-fix lane 2026-08-05 — Q2 / Q3 / Q7 battery receipts

Source briefs: `research/sweep-audits-20260805/AUDIT.md` (ranked fix queue items 1-3).
Branch `lane/audit-fixes` off `restructure/public-split` @ b7f09a36. All GPU runs on the
local 5090 under the shared `/tmp/gpu5090.lock` flock, single runs (correctness gates,
not perf claims — no perf claim is made in this lane; the Q3 timing smoke is a CI bound,
not a benchmark).

## Fixes

| Brief | Commit | What |
|---|---|---|
| Q7 | `dd42b995` | 64-bit `offset_dst` in all 11 vendored MMQ launchers (+ 2 latent quantizer thread-id widenings in qmatvec.cu) |
| Q2 | `c487b1ca` | Loud once-per-flip draft-graph fallback, reset-on-pool-resume, honest GraphSession::step error surfacing |
| Q3 | `92ad6c12` | PrefixCache recency-index eviction — O(log E)/victim, policy-identical timestamp-LRU |

## Gates

| Gate | Verdict | Receipt |
|---|---|---|
| kernel-check | **ALL GREEN** ("kernels match CPU reference") — Q7 bit-identity holds, no STOP condition | `kernel-check.log` |
| run-gen argmax (9B NVFP4 MTP) | **MATCH** (prefill argmax == decode argmax == 268) | `run-gen-9b.log` |
| run-spec K=1..8 (9B NVFP4 MTP) | **PASS** all 8 K values ("identical to generate"), SELF-CONSISTENCY PASS | `run-spec-9b.log` |
| serve-smoke | **0 failed** (plain + spec + truncation matrix + affinity resume) | `serve-smoke.log` |
| cargo test -p memra-server | 74/74 pass (incl. 3 new Q3 eviction tests: old-policy same-victim property over a 400-step recorded pattern, touch-rescues-victim, 10k-entry flush smoke) | CI-reproducible |
| cargo test -p memra-engine --lib | 35 pass / 1 ignored (incl. 3 new Q2 fallback tests: flip-once, reset-on-resume, silent shape-clears) | CI-reproducible |

## Semantic notes (deviations examined, none traded away)

- **Q3 eviction policy**: the old loop's victim = global strict-< `last_use` minimum
  (timestamp-LRU). The BTreeMap index picks the same minimum. The ONLY divergence class is
  ties on equal `Instant`s, where the old code was HashMap-iteration-order nondeterministic;
  the new `(last_use, id)` key breaks ties by insertion order — a determinization, not a
  policy change. The property test therefore drives distinct timestamps (the only regime
  where the old choice was well-defined).
- **Q3 pool order**: `remove_at` uses `swap_remove`, so a pool's Vec order changes on
  eviction. Verified order-independent for every probe: `lookup` (longest-match; ties
  impossible under exact-key dedupe), `best_lcp` (max), `has_covering`/`has_key` (any).
- **Q2 MaxNew carve-out**: `GraphSession::step` has one benign error cause — generation
  budget exhaustion (`pos + 1 >= bucket_max`). The worker checks that same bound before
  stepping; only that case keeps the (honest) MaxNew stop. Every other step error now logs
  and sends `Event::Error` instead of a clean MaxNew.
- **Q2 no-fallback path**: byte-identical behavior when no capture fails — same memoization
  (doomed captures still attempted once per shape, not per burst), same replay path; only
  the log lines, the resume-time flag reset, and the error event differ.
- **Q7 small shapes**: the widened multiply is value-identical below 2^31; kernel-check's
  bit-identity gates (incl. fp8-blk Q8_0 bit-parity and the MMQ paths) all GREEN, satisfying
  the brief's "if kernel-check shows any bit change, STOP" condition with no stop.
