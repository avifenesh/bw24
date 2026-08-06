# serve-hardening — gate verdicts

Lane: `lane/serve-hardening` off `origin/restructure/public-split`. Design + mechanism:
`DESIGN.md`. Every verdict below is a tail of a committed raw log in `logs/` — the log is the
evidence, this file is the index.

Rig: local RTX 5090 Laptop, driver 595.84, CUDA 13.1, `nvcc` 13.1, warm (back-to-back gate
battery under one `flock /tmp/memra-5090.lock`, so nothing else held the GPU). Server crate only
— **no CUDA kernel changes**, so `kernel-check` / `run-gen` / `run-spec` numbers cannot move and
the lane's obligation is the *serving* battery below.

Battery re-run in full **after the rebase onto `origin/restructure/public-split`** (the base had
advanced 18 commits, incl. the `lane/spec-scaling` and `lane/pp2-spec` merges that move
`worker.rs` spec paths *underneath* this lane's code). The logs below are that post-rebase run at
lane head `7c988bb7`; every verdict was re-earned, not carried over from the pre-rebase build,
and the pre-rebase numbers matched to within run-to-run noise. The last code commit on the lane
(the handler-layer G6 fix, which touches the auth/lane refusal path) re-ran the whole battery
again rather than only the tests it looked like it affected — the logs here are that final run.

| gate | command | verdict | log |
|---|---|---|---|
| unit tests | `cargo test -p memra-server --release` | **92 passed, 0 failed** (82 before this lane) | `logs/cargo-test-memra-server.txt` |
| serve-smoke | `tools/serve-smoke.sh` | **0 failed** (`SMOKE_RC=0`) — 16 checks incl. spec-vs-plain greedy identity, the 4-arm sampled truncation matrix (every arm `bangs=0 <= baseline 0`), session-affinity resume (`affinity fired (3 rewind(s))`, `no failed rewinds`) | `logs/serve-smoke.txt`, `logs/serve-smoke-server.log` |
| api-key auth | `tools/apikeys-gate.sh` | **0 failed / 18 gates** (`APIKEYS_RC=0`) | `logs/apikeys-gate.txt`, `logs/apikeys-gate.jsonl`, `logs/apikeys-gate-server.log` |
| serve-stress c=64 | `tools/serve-stress-gate.sh` | **ALL GREEN** — `completed 64/64; wall p50=46.2s p95=53.3s max=54.0s; ttfb p50=0.51s p95=5.00s` (ttfb informational), streams well-formed, worker alive, log clean (`STRESS_RC=0`) | `logs/serve-stress-c64.txt`, `logs/serve-stress-c64-server.log` |
| accept-gate (smoke arm) | `tools/accept-gate.sh` | **1 pass, 0 fail** — `q27-p1: PASS (rounds=42 drafted=126 accepted=85 accept=0.6746, 128 tok text sha-identical)` (`ACCEPT_RC=0`) | `logs/accept-gate-smoke.txt` |

Live wire verification (not a pass/fail gate — payload receipts; see `DESIGN.md §7` for the
per-arm table and the two assumptions the wire corrected):

| probe | what it captures | log |
|---|---|---|
| `probe-endpoints.sh` | `/health` `/livez` `/readyz` across load / ready / drain, the reachable G6 arms, streaming intact, exit code, G24 startup lines | `logs/endpoints-live.txt`, `logs/endpoints-live-server.log` |
| `probe-worker-death.sh` | the G5 ladder on a real CUDA worker: panic → 503 with the quoted payload → respawn → `generation:1` → serves; and `MEMRA_WORKER_RESPAWN=0` → exit 70 | `logs/worker-death.txt`, `logs/worker-death-respawn-server.log`, `logs/worker-death-exit70-server.log` |

## Notes an operator should read before trusting these

* **The c=64 stress gate was the one at risk from this lane, and it was verified by running it,
  not by reasoning.** The new 429 class could in principle have turned stress load into sheds
  and broken its `completed n/n` assertion (its Python client uses `urllib.request.urlopen`, so
  any 429/503 raises `HTTPError` into `row["error"]` and fails the gate). It does not: the
  interactive lane queues FIFO and never sheds, so nothing on the stress path can reach the
  dark-lane shed. 64/64 both before and after.
* **serve-smoke caught a real bug on this lane** — the supervisor's first version read the
  worker's load verdict after `catch_unwind` returned, so `main` blocked forever in
  `ready_rx.recv()` and never bound the socket: a box that loaded the model and answered
  nothing, which is exactly the failure class this lane exists to remove. Fixed in `4f64bf62`;
  the verdict is now relayed on a short-lived thread.
* **`ttfb p95=5.00s` in the stress gate is expected and informational**, not a regression: c=64
  arrivals are staggered and queue FIFO behind the interactive cap. The gate asserts completion
  and stream well-formedness, not latency.
* **`tools/apikeys-gate.sh` writes into another lane's committed evidence directory.** Its `OUT`
  defaults to `research/apikeys-20260805` (`tools/apikeys-gate.sh:13`), and `apikey_gate.py` lives
  there too, so it cannot simply be redirected — running the gate rewrites
  `apikeys-20260805/{apikey-gates.jsonl,meter-lines-sample.log,server-apikeys.log}` with the
  rerunner's timestamps. This lane's rule: run it in place, **copy** the resulting jsonl + server
  log into `logs/apikeys-gate.jsonl` / `logs/apikeys-gate-server.log`, then
  `git checkout <merge-base> -- research/apikeys-20260805/` to hand the other lane's receipts back
  byte-identical. Anyone reusing this gate should do the same, or teach the script to take an
  `--out` that doesn't have to also hold the harness.
* **No perf-board surfaces move.** This lane publishes no benchmark numbers, so
  `tools/update-perf-board.py` has nothing to regenerate (`current-board.json` untouched). The
  `pre-push` hook's `--check` therefore passes unchanged.
