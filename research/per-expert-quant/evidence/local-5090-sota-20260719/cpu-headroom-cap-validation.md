# CPU expert RAM-headroom cap: retain

`BW24_CPU_EXPERT_RESERVE_GB` caps the requested normal-RAM expert cache to live
`MemAvailable - reserve`. With the flag absent, the previous `BW24_CPU_EXPERT_CACHE_GB` behavior is
unchanged.

Validation on the local 5090 rig:

- Current source SHA-256: `971008fc779c4c1614d5752dd0a92aa2bcdb438c85ef4322b573f0aa3cac74c6`.
- A verification build ran at 50% of one CPU, 4 GiB hard memory, 215 MiB peak RSS, and zero swaps.
- Its binary was byte-identical to the retained artifact: SHA-256
  `8bf23e565f64baec94f9b40f1848d179e17e42b87784db1c1d3def5ea89fa031`.
- A 25%-CPU ABI smoke requested 36 GiB with an 8 GiB reserve; 40.84 GiB was available and the
  effective cache ceiling was 32.84 GiB. ABI version 1 and zeroed cache counters were returned.
- A 50%-CPU synthetic 4-expert/100-iteration compute smoke returned the established
  `0b35230a73c63d1f` hash with zero swaps. Its throttled timing is not performance evidence.

The active N=64 A/B runner now enforces `CPUQuota=200%`, `CPUWeight=10`, `MemoryHigh=34G`,
`MemoryMax=38G`, four CPU/IO workers, cleanup traps, active-process refusal, and a 23,000 MiB free
VRAM preflight. Two setup attempts parsed before this hardening were deliberately terminated and
are marked invalid separately.

Raw evidence:

- `build-cpu-experts-headroom-capped-verify.log`
- `cpu-expert-headroom-capped-abi-smoke.log`
- `cpu-moe-headroom-capped-compute-smoke-n4-t8-n100.log`
