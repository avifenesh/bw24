# Power-outage window log (lane/ladder-3072, 2026-08-02)

Owner announced a wall-power cut (laptop -> battery -> hard GPU clock throttle,
then a fresh thermal/power regime on re-plug).

- Pre-outage regime A cells (VALID, all completed before 16:34:50Z):
  kernel-sweep-split.log (fa_deep_bench forced-split wall timings, 16:29:10Z-16:29:12Z headers,
  runs serialized under flock) and nsys-sp{8,32,64}-d{1024,2048,3072,4096}.txt
  (last cell sp64 d4096 finished ~16:34:40Z, temps 56-60C).
- Outage window: see below (ADP0 online transitions polled).
- Regime B (post-outage): ALL e2e run-gen / llama-bench cells. No timed cell spans the window;
  arms are interleaved within regime B only. Kernel-vs-e2e comparisons across regimes are
  directional only, never a published ratio.

Resolution (coordinator-verified): outage window fell between ~16:34:50Z and 16:37:25Z
(before the ADP0 poller started; poller logged no transitions). AC online=1 confirmed
post-outage, GPU idle 180MHz/55C. NO timed cell ran during or spanned the window —
regime A ended ~16:34:40Z, regime B (e2e sweep) started after re-plug verification.
Zero quarantined reps expected from this event; the sweep harness still carries the
per-rep ADP0 guard in case of a second cut.
