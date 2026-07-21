# Headroom-6 whole-pack run 3: invalid

Run 3 was stopped during setup after its runner disclosed that the inference command would execute
with `CPUQuota=200%`, `CPUWeight=10`, `MemoryHigh=34G`, `MemoryMax=38G`, four compute threads, and
four I/O threads. Those limits cannot accommodate the known approximately 46 GiB runtime and do
not represent the local 8-P-core target. No model inference or candidate code executed.

The corrected runner creates a sibling user-systemd scope with `cpu.max=max`, `CPUWeight=100`,
`memory.high=max`, and `memory.max=max`, restores the 8-thread P-core configuration, and prints the
live in-scope cgroup files before executing the runtime.

Raw launcher log: `headroom6-wholepack-pack-run3-launcher.log`.
