# Whole-expert pack run 5: not started

The clean-swap reset completed, but another target-rig `run-gen` began at 18:29 before the
whole-expert runner could launch. That made the intended cold-start boundary and GPU/CPU resource
regime non-exclusive, so run 5 was abandoned before starting the candidate process.

There is no throughput result for run 5. The swap-reset record is
`clean-swap-before-headroom6-wholepack-pack-run5.log`; the next valid attempt uses a persistent
user-systemd launcher and a new run number.
