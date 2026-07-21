# Headroom-6 whole-pack run 2: invalid

The `pack` arm's attached PTY exited during model loading, before a runtime banner, inference,
or candidate execution. The raw log ends after preflight telemetry and contains neither an error
nor `/usr/bin/time`/runner exit telemetry. No kernel OOM or crash event was logged. This second
automatic retry was deliberately terminated because it had parsed the old uncapped runner before
the runner was cgroup-hardened. It is not performance evidence.

The runner now uses `CPUQuota=200%`, `CPUWeight=10`, `MemoryHigh=34G`, and `MemoryMax=38G`, with
four CPU/IO workers. Renaming the executable is not an acceptable way to bypass the guard.

Raw log: `hy3-mtp-k1-headroom6-wholepack-pack-clean-ngen64-run2.log`.
