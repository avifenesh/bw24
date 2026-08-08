THESE NUMBERS ARE INVALID — kept deliberately, not deleted.

This is the FIRST run of run-ppspec-perf.sh, taken before the zero-emit session-tail underflow
fix (spec.rs:5218, `out.len().saturating_sub(1)`). Every c=8 point in it is worthless because
31 of 32 requests died with HTTP 500 "worker closed stream" behind this worker panic:

  thread 'memra-gpu-worker' (50489) panicked at crates/memra-engine/src/spec.rs:5218:49:
  range end index 18446744073709551615 out of range for slice of length 0

That hit EVERY spec arm, including the door-shut single-card denominator — which is what
identified the bug as pre-existing (b4aea184) rather than anything to do with the PP split.

The c=1 points in this file are not affected by the panic (0 errors on every arm) and agree with
the post-fix run to within the per-rep spread, but the post-fix run in ../perf/ is the receipt of
record for BOTH concurrencies. This directory exists so the claim "the pre-fix c=8 numbers were
invalid and were re-measured" is checkable rather than asserted.
