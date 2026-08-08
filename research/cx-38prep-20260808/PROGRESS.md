# Qwen 3.8 27B FP8-ST day-one preparation

Date: 2026-08-08
Branch: `lane/cx-38-runbook`
Train tip: `9aebdb3e`

## State

The day-one runbook and its release-independent tooling are ready. The production direction is
the official Qwen FP8 safetensors artifact loaded directly by memra. No Qwen 3.8 Q8_0, GGUF,
NVFP4, local requant, or community-requant bridge is part of the plan.

The requested steering file, `~/.lanectl/inbox/cx-38prep.md`, was absent during this work. The
preflight records that as `WAIT`; no alternate inbox file was treated as lane authority.

The live Hugging Face Qwen namespace did not expose an official `Qwen3.8-27B` model at the final
preflight on 2026-08-08. The default candidate name remains
`Qwen/Qwen3.8-27B-FP8`, but the runbook searches the official namespace and binds the exact
published repo plus immutable revision before downloading.

## Delivered

- `docs/qwen38-bringup-runbook.md`: hour-by-hour runbook from official repo discovery through
  config STOPs, immutable download, direct FP8 classification, model-backed `kernel-check`,
  naked residency and FP8-MMQ proof, exact HF token parity, chunk invariance, conditional MTP
  `run-spec`, thinking-surface verification, own-generation ranking, and drafter A/B.
- `tools/preflight-38.sh`: release-independent environment, disk, authentication, frozen
  baseline, direct-FP8, dependency, and release-binary checks. Release absence is `WAIT`;
  local deficiencies are `FAIL`.
- `tools/inspect-fp8-st.py`: header-only safetensors class census. It accepts per-tensor and
  exact block-128 E4M3 weights, rejects per-row/unknown layouts, and reports known packed-U8
  auxiliary scale planes separately.
- `tools/hf-greedy-reference.py` and `tools/compare-greedy-tokens.py`: release-day greedy HF
  reference plus exact prompt-token and generated-token comparison.
- `research/qwen38-prep-20260803/arch-diff-fields.py`: mechanized same-architecture classifier
  with hard STOPs for model type, heads, GDN/layer cycle, RoPE, attention/gating, MoE, and MTP
  contract changes.
- `concat-prime-probe` plus `tools/chunk-invariance-gate.sh`: an HF safetensors directory can
  now run the same chunk-invariance assertion as a GGUF model, with a caller-selected raw-log
  path.
- HF-directory tokenizer loading now recognizes Qwen's shipped `pretokenize_regex` as the
  existing exact `qwen35` split class instead of silently using the generic fallback.

## Frozen Qwen 3.6 baseline

The preflight pins these local A/B and oracle artifacts:

- FP8-ST: `/data/ai-ml/hf-models/qwen36-27b-blk128fp8`
- architecture/tokenizer reference: `/data/ai-ml/hf-models/qwen36-27b-hf-min`
- model-backed GGUF oracle:
  `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
- frozen own-trim drafter:
  `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf`

The official `Qwen/Qwen3.6-27B-FP8` config still matches the frozen direct contract:
`quant_method=fp8`, `fmt=e4m3`, `weight_block_size=[128,128]`, and
`activation_scheme=dynamic`.

The exact architecture STOP/GO matrix is in the runbook. Key frozen values include 64 layers,
hidden size 5120, 24 attention heads, 4 KV heads, head dimension 256, full-attention interval 4,
the `linear,linear,linear,full` cycle, and the existing Qwen 3.5 RoPE/GDN contract. A changed
model type, head contract, attention cycle, GDN shape, RoPE scheme, or MTP interface is a
bring-up lane, not a runbook continuation.

## Receipts

`preflight-20260808.log`:

```text
summary: PASS=54 WAIT=3 FAIL=0
PREFLIGHT-38: READY-WITH-WAITS
```

The three waits are:

1. requested steering file absent;
2. official Qwen 3.8 FP8 repo not visible yet;
3. target artifact directory absent because weights are not released.

`fp8-header-q36-baseline.log` proves the header classifier against the frozen direct artifact:

```text
2D F8_E4M3 weights: 208
  per-tensor :    0 tensors
  block-128  :  208 tensors
  per-row    :    0 tensors
  unsupported:    0 tensors
packed-U8 E4M3 scale planes: 193
header-only direct-path verdict: PASS
```

`chunkinv-q36-st-smoke.raw.log` and `chunkinv-q36-st-smoke.gate.log` are a real model smoke
through the new safetensors-directory probe. Both pinned prompts were bit-identical across
chunks 64 and 32; the gate passed and emitted no tokenizer-fallback warning. The driver
deliberately inherited `MEMRA_PRIME_F32CHUNK0=1`; the wrapper cleared it for the default arm,
proving an ambient rollback seam cannot falsify the naked result.

Additional focused receipts:

- `config-diff-q36-selftest.log`: zero architecture diffs and zero hard stops.
- `config-diff-q36-official-fp8.log`: official Qwen 3.6 FP8 metadata PASS.
- `config-diff-classifier-smoke.log`: size/context changes take the GO-with-gates path, while
  head, RoPE, and layer-count mismatches return the hard-STOP exit.
- `q36-tokenizer-recognition.log`: Qwen FP8-ST prompt encoded with the recognized Qwen split.
- `validation-20260808.log`: focused build, loader/config, tokenizer, thinking, and static gates.

## Prior art applied

- `research/fp8st-20260803/P1-VERDICT.md`
- `research/fp8st-20260803/armb/RESULTS.md`
- `research/fp8st-20260804/mmq/SLICE3-MODEL-VERDICT.md`
- `research/fp8blk-20260805/VERDICT.md`
- `research/rp-on-st-20260806/VERDICT.md`
- `research/qwen38-prep-20260803/{AUDIT,DRYRUN-20260804,WATCH}.md`
- `docs/{DRAFT-REGIME,SERVING,TESTING,RELEASING}.md`

The RP-mirror result is reflected as one production trunk artifact. The full Qwen 3.6 GGUF is
only an oracle/A-B reference and a possible byte-verbatim donor for a small external drafter
when every trunk interface field matches. The existing trim builder accepts a GGUF donor, so it
must not be used to manufacture a full Qwen 3.8 bridge.

## Release-dependent work

No Qwen 3.8 model gate has been claimed. On release day the operator must bind the official
revision, run the config/tokenizer STOP gates, hash every indexed shard, prove tensor classes,
and execute every model-backed gate in runbook order.

`run-spec` remains conditional on an embedded compatible MTP layer or an external drafter that
passes the exact donor-interface check and the own-generation trim regime. Absence of a drafter
is `WAIT`, not permission to publish an unverified speculative path.

No merge, tag, or origin push was performed.
