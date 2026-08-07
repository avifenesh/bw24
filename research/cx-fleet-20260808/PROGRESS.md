# lane/cx-fleet-metering - fleet-through-memra receipt instrument

## State

Tooling is complete. Live accumulation did not start during verification because the
owner-critical `127.0.0.1:8002` endpoint was down. No server process was started, stopped,
or reconfigured, and no fabricated row was written to the production ledger.

| commit | contents |
|---|---|
| `e5a9fe04` | `fleet-meter.sh`, `fleet-report.py`, restart fixture, and focused tests |
| second lane commit | serving docs, systemd service/timer, raw verification logs, this receipt |

## Delivered

- `tools/fleet-meter.sh`: one-shot scrape by default, optional 30-minute foreground loop,
  locked JSONL append, unchanged-snapshot suppression, safe down-server skip, and restart
  marking on cumulative counter regression.
- `tools/fleet-report.py`: UTC daily deltas across process segments, hit-ratio trend,
  0.25-to-1.0 cache-billing revenue band, and tick-seg window share. Economics and
  histogram-window formulas are imported from `tools/cache_economics.py`.
- `deploy/systemd/memra-fleet-meter.{service,timer}`: site-adjustable one-shot service and
  persistent half-hour calendar timer.
- `docs/SERVING.md`: operating and interpretation notes for the pre-listing receipt.

## Receipts

Raw logs are under `raw/`.

- `live-meter.log`: at `2026-08-07T21:32:02Z`, curl returned
  `Failed to connect to 127.0.0.1 port 8002`; the meter logged `skip`, exited successfully,
  and `research/fleet-meter/rig5090-fleet.jsonl` remained absent.
- `tests.log`: 6/6 discovered tool unittests pass. The three fleet cases cover restart
  marking, duplicate suppression, safe scrape failure, restart-aware daily deltas,
  economics reuse, and report rendering; the existing NVMe builder safety tests stay green.
- `fixture-report.log`: the reset day closes at 1,200 prompt tokens, 550 cached, 650
  computed, 45.8% hit-token ratio, `+5.83pp` trend, `1.2115x..1.8462x` revenue band,
  50.0% tick-seg share, and one restart.
- `static-checks.log`: bash syntax, Python compilation, shellcheck, and
  `systemd-analyze verify` all pass. Unit verification used a temporary copy with the
  site-specific `/opt/memra` and `memra` account placeholders replaced by this worktree
  and user.

## Live handoff

When the existing port-8002 deployment is available, one read-only command starts the real
receipt without touching the server lifecycle:

```bash
tools/fleet-meter.sh --once
```

After that first real row, enable the adjusted timer or run the foreground loop. The default
ledger is `research/fleet-meter/rig5090-fleet.jsonl`.
