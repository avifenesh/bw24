# lane/cx-fleet-metering - fleet-through-memra receipt instrument

## State

Tooling, including the agent-shaped replay driver, is complete. Live accumulation and the
required five-minute replay did not start during verification because the owner-critical
`127.0.0.1:8002` endpoint was down. No server process was started, stopped, or reconfigured,
and no fabricated row was written to the production ledger. Replay verification is
fixture-only.

| commit | contents |
|---|---|
| `e5a9fe04` | `fleet-meter.sh`, `fleet-report.py`, restart fixture, and focused tests |
| `fc1c1261` | serving docs, systemd service/timer, and initial verification receipts |
| `4467bf3c` | `fleet-replay.py` and mock-server tests |
| this docs commit | replay operating notes and refreshed verification receipts |

## Delivered

- `tools/fleet-meter.sh`: one-shot scrape by default, optional 30-minute foreground loop,
  locked JSONL append, unchanged-snapshot suppression, safe down-server skip, and restart
  marking on cumulative counter regression.
- `tools/fleet-report.py`: UTC daily deltas across process segments, hit-ratio trend,
  0.25-to-1.0 cache-billing revenue band, and tick-seg window share. Economics and
  histogram-window formulas are imported from `tools/cache_economics.py`.
- `tools/fleet-replay.py`: low-rate exponential arrivals across carried synthetic sessions,
  89.5:1 prompt:completion budgeting by default, eight 1k-4k-token tool-schema prefixes,
  2-4-turn session bursts, tenant-scoped replay salts, fixed `replay-calibrated` receipt
  labeling, and SIGTERM-graceful shutdown.
- `deploy/systemd/memra-fleet-meter.{service,timer}`: site-adjustable one-shot service and
  persistent half-hour calendar timer.
- `docs/SERVING.md`: operating and interpretation notes for the pre-listing receipt and
  controlled replay workload.

## Receipts

Raw logs are under `raw/`.

- `live-meter.log`: at `2026-08-07T21:32:02Z`, curl returned
  `Failed to connect to 127.0.0.1 port 8002`; the meter logged `skip`, exited successfully,
  and `research/fleet-meter/rig5090-fleet.jsonl` remained absent.
- `replay-live-check.log`: at `2026-08-07T23:25:52Z`, the required pre-run curl returned
  exit 7, `Failed to connect to 127.0.0.1 port 8002`; the five-minute replay was not run.
- `tests.log`: 9/9 discovered tool unittests pass. The three replay cases cover the eight
  template sizes, request ratio and tenant salt shape, per-session assistant-reply carry,
  and graceful SIGTERM during an exponential wait. The meter/report and NVMe safety cases
  stay green.
- `fixture-report.log`: the reset day closes at 1,200 prompt tokens, 550 cached, 650
  computed, 45.8% hit-token ratio, `+5.83pp` trend, `1.2115x..1.8462x` revenue band,
  50.0% tick-seg share, and one restart.
- `static-checks.log`: bash syntax, Python compilation, shellcheck, and
  `systemd-analyze verify` all pass. Unit verification used a temporary copy with the
  site-specific `/opt/memra` and `memra` account placeholders replaced by this worktree
  and user.
- `replay-static-checks.log`: replay Python compilation, CLI help, whitespace, and generated
  perf-board drift checks pass.

## Live handoff

When the existing port-8002 deployment is available in a dev-idle window, bracket the
five-minute controlled workload with meter snapshots:

```bash
tools/fleet-meter.sh --once
MEMRA_API_KEY=<local-key> tools/fleet-replay.py --duration 300
tools/fleet-meter.sh --once
```

Then confirm `/metrics` prompt and cached counters moved and report that interval only as
`replay-calibrated`. After the first real row, enable the adjusted timer or run the
foreground loop. The default ledger is `research/fleet-meter/rig5090-fleet.jsonl`.
