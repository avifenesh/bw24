# Whole-expert pack run 6: not started

The exact swap reset ran from 18:53:48 to 18:54:31 and completed successfully. During that reset,
the separate mixed-q8 K=1..8 exactness battery started at 18:53:57. The runner's post-reset process
interlock detected its `run-spec` and refused to launch the performance candidate.

There is no throughput result for run 6. The successful but subsequently contaminated reset is
recorded in `clean-swap-before-headroom6-wholepack-pack-run6.log`; the mixed-q8 validation is in
`hy3-mixed-q8-k1to8-safe-ngen8.log`. A later run number must establish a new exclusive reset.
