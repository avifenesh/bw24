# cx-serve-glue progress - 2026-08-08

## Status

All six serving-trial glue deliverables are implemented and dry-runnable on
branch `lane/cx-serve-glue`. No pod exists in this lane, so this receipt proves
offline construction and validation, not live Step loading, public routing,
marketplace traffic, or target-rig correctness.

The requested steering file
`/home/avifenesh/.lanectl/inbox/cx-glue.md` was absent when checked on
2026-08-08. The inbox contained `cx-fleet.md`, but no substitute steering file
was assumed or used.

## Deliverables

| Deliverable | Commit | Result |
|---|---|---|
| TLS/tunnel glue | `240895f4` | Pinned cloudflared installer, dashboard DNS step, live `--check`, RunPod proxy fallback |
| AI Horde bridge | `6912caa7` | Pinned OpenAI-compatible text bridge, one-thread/context/output caps, offline renderer and validator |
| Poe bot glue | `25adf74f` | `fastapi-poe` shim, keyed streaming translation, registration guide, mock-backed tests |
| Bridge secret loading correction | `b5430f22` | Systemd-loaded secrets no longer require an unprivileged service to reopen root-only environment files |
| Key issue/revoke flow | `c6351c8c` | Serialized permission-safe wrapper and one-slot tenant guidance |
| One-command launcher/down path | `08e3332a` | RunPod composition, ordered startup, ten-row health matrix, maintenance/drain shutdown |
| Per-key cap enforcement | `779daae5` | Atomic pre-worker 429 gate for configured tenant concurrency caps |
| Measurement and harvest runbook | this commit | Observation cadence, raw-first receipts, error taxonomy, shutdown harvest |

The launcher composes with the sibling RunPod lane's existing contract:
`/etc/memra/runpod.env`, `/etc/memra/keys.toml`,
`memra-server.service`, `memra-fleet-meter.timer`, staged model bytes, and the
PP-2 CUDA device assignment. It does not duplicate provisioning.

## Current-source checks

Source audit date: 2026-08-08.

- Cloudflare Tunnel setup and release pin are recorded in
  `deploy/glue/TLS.md`.
- AI Horde candidate revisions, dates, limitations, and the selected pinned
  revision are recorded in `deploy/glue/horde-worker/UPSTREAM.md`.
- Poe protocol/package references and the pinned `fastapi-poe 0.0.83` surface
  are recorded in `deploy/glue/poe-bot/REGISTRATION.md`.
- The AI Horde worker API schema and public lookup were checked against the
  live `aihorde.net/api/v2` surface.

The Horde selection is an operational fit, not a claim of mature maintenance:
the selected bridge is a pinned one-commit project with no releases. The
official worker remains the reference implementation but cannot send the
required Bearer authorization to memra's keyed local OpenAI backend.

## Offline verification completed

- All eight glue shell files passed `bash -n`.
- All eight glue shell files passed ShellCheck with no findings.
- `cloudflared-setup.sh --dry-run` passed without a connector token.
- Horde setup and runtime dry runs passed without installing or connecting.
- The rendered Horde YAML passed the lane validator and the pinned bridge's
  own configuration parser.
- Poe setup and runtime dry runs passed without installed host state.
- Five Poe tests passed against an `httpx` mock and the real FastAPI/Poe route;
  the two warnings are deprecations inside pinned `fastapi-poe`.
- Key mint and revoke dry runs passed without a server binary or keyring.
- `trial-up.sh --dry-run` and `trial-down.sh --dry-run` passed.
- Trial model metadata passed the server's TOML/parser contract.
- Every health-matrix JSON predicate passed against offline fixtures.
- `cargo test -p memra-server` passed: 117 tests, 0 failures.
- The cap-one concurrency regression proves exactly one of two simultaneous
  arrivals acquires a tenant slot.
- `python3 tools/update-perf-board.py --check` passed; generated performance
  surfaces remain unchanged and current.

No GPU command, `nsys`, `rustup`, origin push, tag, or release action was run.

## Live-only gates

The following cannot be proved without the provisioned two-card pod and owner
accounts:

1. Target-rig `kernel-check`, `run-gen` argmax, and `run-spec` K=1..8.
2. Byte-identical model staging, cold Step load, PP-2 peer access, and real
   authorized generation.
3. The Cloudflare published-hostname routes, DNS propagation, connector, and
   public TLS.
4. AI Horde account ownership, first worker registration, online polling,
   stranger jobs, and kudos deltas.
5. Poe server-bot registration, private smoke, public transition, and real Poe
   protocol traffic.
6. Organic cache economics, tenant-cap 429 rates, dark-lane sheds, error
   distribution, thermal regime, and final receipt harvest.

Do not merge, tag, or report the trial live until those gates produce raw logs
and the harvest sequence in `deploy/glue/TRIAL-RUNBOOK.md` completes.
