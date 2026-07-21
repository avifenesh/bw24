# Global whole-expert pack: rejected

Raw run: `hy3-mtp-k1-wholepack-packed-clean-ngen64-run1.log`

- Clean-swap, single run, N=64.
- The unconstrained global planner changed frozen MTP residency from the profiled 21 complete
  experts to 13 complete experts (39 blocks).
- Plain generation measured 63 tokens in 59.930 seconds, or 1.05 tok/s. Prompt prime was
  654.254 seconds.
- `/proc/<pid>/io` reported 797,234,528,256 physical read bytes at the decision boundary.
- The run was deliberately interrupted after this measured window at about 20:15 elapsed. The
  K=1 self-consistency repeat was not allowed to spend another pathological prime because this
  placement was already an unambiguous performance loser.

This result rejects unconstrained global packing. It does not reject whole-expert packing with a
hard floor that retains the already-profiled complete MTP set.
