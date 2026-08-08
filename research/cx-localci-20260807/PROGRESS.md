# local-ci run-spec gate receipt — 2026-08-07

Branch: `lane/cx-localci-spec`

Train tip: `ac51498737d12e83562729a4f63af558434cf20e`

Script commit: `a61ef5fbcf06fc00cfe82563a8ee7c886485a6c3`

## Change

`tools/local-ci.sh` now runs the standing `run-spec` K=1..8 self-consistency gate:

- target: `/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
- external MTP draft:
  `/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf`
- prompt: `tools/fast-gate/prompts/probe.txt`
- generation window: 32 tokens
- timeout: 900 seconds
- raw log before parsing: `/tmp/local-ci-run-spec.log`
- skip knob: `MEMRA_CI_RUNSPEC=0`

The gate clears `MEMRA_SPEC_K`, `MEMRA_PROMPT_DIR`, and `MEMRA_GEN_ONLY` before launch so an
inherited single-K or alternate run mode cannot silently narrow the sweep. Success requires all
eight per-K `self-consistency: PASS` lines, eight K headers, and the final
`=== SELF-CONSISTENCY PASS ===` marker.

On a red self-consistency result, the wrapper parses the raw log and prints both the failing K and
the exact `FIRST DIVERGENCE at index N` position before quoting the log tail.

## Target choice

The owner-call wording named the standing 31B cell. The actual 31B cell is Gemma-4 and cannot run
through `run-spec`: it uses the separate `GemmaDraft` path selected by `MEMRA_DRAFT`, while
`run-spec` requires an embedded or `MEMRA_MTP_DRAFT` NextN head. The existing Gemma-4 31B
`gemma-gate` stream-agreement 64/64 check remains unchanged.

The new `run-spec` sweep therefore uses Q35, which `local-ci` already requires for `prime-gate`,
plus its standing external MTP draft. This closes the documented binary-level `run-spec` K=1..8
gap rather than relabeling the single-K Gemma check.

## Verification

Static and build checks:

```text
bash -n tools/local-ci.sh
PASS

shellcheck tools/local-ci.sh
PASS

git diff --check -- tools/local-ci.sh
PASS

cargo build --release -p memra-engine --bin run-spec
Finished release profile in 2m 25s
```

Artifact preflight:

```text
Gemma-4 31B target: present, 17G
Gemma-4 31B draft: present, 333M
Qwen 35B target: present, 17G
Qwen 35B external MTP draft: present, 901M
```

Dry logic receipts:

```text
known-green parser: passes=8 ks=8 summary=1
known-green parser: PASS

synthetic failure parser: K=4 position=19
synthetic failure parser: PASS
```

The known-green input was the committed raw K=1..8 log at
`research/graph-warmups-5090-20260805/logs/gate-runspec-q27.txt`. The synthetic red exercised the
same `awk` parser used by `local-ci`.

## Live-run status

The model files exist, but the isolated GPU run was not started. `nvidia-smi` was checked twice,
including immediately before the live-run decision, and both checks reported these compute apps:

```text
144655, llama-server, 332 MiB
7600, hermes gateway Python, 394 MiB
```

The `llama-server` is the allowed CPU-bound `--embedding -ngl 0` service. The Hermes gateway is a
separate compute context and makes the window non-clean under `local-ci`'s own filter. Per the lane
instruction, the live stage was skipped rather than contending with another lane. No live
K=1..8 PASS is claimed.
