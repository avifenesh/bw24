# CPU expert cache policy and fused-stage decision

The shipped CPU expert companion keeps the non-fused LRU path. The remaining experimental cache
and compute branches did not beat the established sustained result and were removed rather than
left behind as dormant flags.

| Arm | Effective CPU cache | Plain N=64 | K=1 N=64 | Result |
|---|---:|---:|---:|---|
| non-fused LRU reference | 36.00 GiB | 7.54 tok/s | 7.60 tok/s | retain |
| fused LRU run 7 | 36.00 GiB | 6.86 tok/s | 6.90 tok/s | reject |
| fused least-stale run 1 | 30.13 GiB | 6.99 tok/s | 7.04 tok/s | reject; capacity differs |

The two later capacity-lock attempts produced no comparison: least-stale run 10 was stopped when
its live cap resolved to 30.27 GiB, and LRU run 12 was stopped when its live cap resolved to
22.17 GiB. Neither produced a measured generation window. No fused or least-stale run exceeded
the non-fused LRU reference, so there is no basis for shipping their extra dispatch, metadata ABI,
or policy state.

The winners-only companion rebuilt successfully as
`/tmp/libbw24-cpu-experts-winner.so`. Its four-expert compute smoke retained the exact established
hash `0b35230a73c63d1f`; the raw build and smoke logs are
`cpu-experts-winners-only-build.log` and `cpu-moe-winners-only-compute-smoke-n4-t8-n100.log`.

Reference logs:

- `hy3-mtp-k1to8-postcleanup-maxprofile-ngen64-run1.log`
- `hy3-mtp-k1-headroom6-wholepack-control-clean-ngen64-run7.log`
- `hy3-mtp-k1-headroom6-wholepack-control-clean-ngen64-run1.log`
- `fused-least-stale-run10-unscored.md`
- `fused-lru-run12-unscored.md`

## Pinned release validation

The historical 36 GiB reference was not reproduced under the final release load because the live
desktop had only 28-34 GiB available before model startup. The exact product commit
`4641033f0fee83d0cc4fc77bbbffa0f9d3adc8d0` instead produced two current single-run checks:

| Profile | HBM fraction | Effective CPU cache | Plain N=64 | Interpretation |
|---|---:|---:|---:|---|
| safe launcher equivalent | 0.85 | 24.37 GiB | 5.92 tok/s | release fallback; argmax MATCH |
| maximum supported allocation | 0.95 | 29.71 GiB | 5.09 tok/s | reject; live swap pressure erased the cache gain |

The maximum-allocation run still passed K=1 through K=8 self-consistency. Its K=1 result was 3.42
tok/s, so speculative decoding is not promoted as the throughput winner for this memory regime.
The attempted `0.97` fraction was correctly rejected by the validated `0.10..=0.95` parser and
stopped before scoring.

Pinned validation logs:

- `release-gate-provenance-4641033.log`
- `release-gate-kernel-check-4641033.log`
- `release-gate-run-gen-safe85-4641033.log`
- `release-gate-run-spec-k1to8-max95-4641033.log`
