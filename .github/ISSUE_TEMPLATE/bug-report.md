---
name: Bug report
about: Wrong output, crash, gate failure, or performance regression
---

<!--
Read CONTRIBUTING.md first. This isn't red tape — bw24 has no CI on sm_120a hardware, so a
report without the details below usually can't be reproduced or actioned at all, and will
sit or get closed rather than fixed. For a security vulnerability, do NOT use this template —
see SECURITY.md instead.
-->

## What happened

<!-- What you expected vs what you observed. Be specific: "wrong output" is not actionable;
     "argmax MISMATCH at token 12" or "run-gen panics with <exact message>" is. -->

## Repro

```
<exact command you ran, with flags/env vars>
```

- Model: <exact model + quant, e.g. Qwen3.6-27B-NVFP4, or the HF/GGUF path>
- Was this a fresh checkout of `main`, or a specific commit/branch? <commit hash>

## Gate output at the time of the bug (if applicable)

```
<kernel-check / run-gen / run-spec output showing the gate result — paste actual output, not "gates failed">
```

## Your hardware

- GPU: <model — this project targets sm_120a specifically; if you're on different hardware, say so>
- Driver / CUDA version:
- OS:

## What you've already checked

- [ ] I searched [`research/tune-data/`](../research/tune-data/) and existing issues for this
- [ ] This reproduces on a clean `main` checkout, not just a local branch
