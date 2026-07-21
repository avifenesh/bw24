# Fused LRU RAM36 run 4 — invalid (host reboot)

This run produced no throughput or correctness result. The raw log stops during the 128-token
discarded warmup after the CPU companion initialized its 36.00 GiB cache. The previous host boot
ended at `2026-07-20 20:38:42 IDT`, and the next boot began at
`2026-07-20 20:38:58 IDT` (`journalctl --list-boots`). The run therefore did not reach residency
freeze, generation, self-consistency, or the runner's exit record.

Do not classify the interruption as an OOM: neither the raw log nor the retained kernel journal
contains an OOM diagnostic for this process. The measured outcome is only: **invalid — host
rebooted during warmup; cause not captured**.
