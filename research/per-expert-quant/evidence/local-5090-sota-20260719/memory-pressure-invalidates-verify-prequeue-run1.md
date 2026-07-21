# Verification prequeue run 1 invalidation

The first `BW24_CPU_EXPERT_VERIFY_PREQUEUE=1` run was interrupted after its plain `t=1` control
window fell to 0.39 tok/s, before the candidate's K=1 window could be interpreted. The candidate is
inactive for `t=1`, so this was not a prequeue comparison.

Live probe at 2026-07-20T17:06:59+03:00:

- RAM: 59 GiB / 60 GiB used, 1.1 GiB available.
- Swap: 8.0 GiB / 8.0 GiB used.
- `run-spec`: 46,336,988 KiB RSS.
- Load average: 35.74 / 23.57 / 18.75.
- `vmstat`: 4.2-4.7 GiB/s block input, 19-29% I/O wait, 3-8 blocked tasks.
- The unrelated rsync had completed before launch; the pressure came from the 36 GiB expert cache
  plus the live desktop/process footprint after swap was reset.

The run was stopped with SIGINT (exit 130) to avoid further system-wide swap thrash. Re-test with a
bounded normal-RAM expert cache and record the effective headroom.
