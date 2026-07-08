# Environment flags

Interim stub — the full audited catalog (every flag: default, purpose, provenance) lands with the
flag-audit lane. Until then, the doctrine (CLAUDE.md): winners are defaults, naked commands run
the tuned path; flags exist only for runtime parameters (`BW24_PROMPT`, `BW24_NGEN`, `BW24_CHAT`,
`BW24_SPEC_K`, `BW24_SPEC_PMIN`, `BW24_SPEC_PMIN0`, `BW24_SPEC_HPOST`, `BW24_FRSPEC_TRIM`,
`BW24_PP_ONLY`, `BW24_PP_REPS`, `BW24_PRINT_TEXT`), machine config (`BW24_PP_FP8` +
`BW24_PP_FP8_BUDGET_MB`, `BW24_KV_K`/`BW24_KV_V`, `BW24_SPILL_DISK`, `BW24_MOE_PINNED`,
`BW24_NVCC`), rollback seams (`BW24_FAST=0` f32 oracle, `BW24_MMVQ=0`, `BW24_MOE_CACHE=0`,
`BW24_FA_V2=0`, `BW24_SPEC_LEAN=0`, `BW24_NO_FA_VEC`), diagnostics (`BW24_MOE_STATS`), and
experimental doors (`BW24_MMQ` W4A4 — exactness-blocked, see tune-data).
