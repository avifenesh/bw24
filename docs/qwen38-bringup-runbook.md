# Qwen 3.8 27B FP8-ST day-one runbook

Prepared 2026-08-08 for the expected Qwen 3.8 drop. This is an executable same-architecture
runbook, not a substitute for a bring-up lane.

The production model artifact is the official Qwen FP8 safetensors directory. There is no
Q8_0, GGUF, NVFP4, or community-requant bridge for Qwen 3.8. The frozen Qwen 3.6 GGUF appears
below only as:

- the real-weight oracle for `kernel-check`, whose model-backed sections are GGUF-specific;
- an A/B performance reference; and
- a byte-verbatim MTP donor for a small external drafter, and only when the trunk interface is
  exactly compatible.

The shipped defaults are the desired path: `MEMRA_ST_E4M3` and `MEMRA_ST_E4M3_BLK` are on,
and block-128 prefill uses the native `MEMRA_FP8_MMQ` source. Do not set
`MEMRA_ST_E4M3=1`, `MEMRA_ST_E4M3_BLK=1`, or `MEMRA_FP8_MMQ=1`; naked behavior is the
production behavior. `MEMRA_FP8_MMQ=1` additionally admits a duplicate stash source and is
not this path.

## Before release

Run from the dedicated worktree in Bash; the snippets use Bash arrays and `PIPESTATUS`. Every GPU
command on the local 5090 uses the shared lock. Raw logs are written first and parsed second.

```bash
set -euo pipefail
cd /home/avifenesh/projects/wt-cx-38prep

export Q38_REPO=Qwen/Qwen3.8-27B-FP8
export Q38_DIR=/data/ai-ml/hf-models/qwen38-27b-fp8
export Q38_DERIVED_ROOT=/data/ai-ml/hf-models/qwen38-27b-derived
export Q38_VENV_ROOT=/data/ai-ml/venvs
export Q36_REF=/data/ai-ml/hf-models/qwen36-27b-hf-min
export Q36_ST=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
export Q36_GGUF=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
export Q36_DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
export R=research/qwen38-bringup-$(date +%Y%m%d)
mkdir -p "$R"

# A copied shell must not turn a default-path gate into an inherited experiment.
unset MEMRA_ST_E4M3 MEMRA_ST_E4M3_BLK MEMRA_FP8_MMQ MEMRA_FP8_FOLD
unset MEMRA_FP8_BLK_GPU MEMRA_PP_FP8 MEMRA_MTP_DRAFT MEMRA_FRSPEC_TRIM
unset MEMRA_SPEC_K

tools/preflight-38.sh 2>&1 | tee "$R/preflight.log"
```

`PREFLIGHT-38: READY-WITH-WAITS` is green before release. A missing official repository and
target directory are expected `WAIT`s. Any `FAIL` is fixed before day one.

## Day-one clock

| Time | Work | Exit condition |
|---|---|---|
| H+00:00 | Bind the exact official repo/revision; fetch metadata | official FP8 repo and immutable revision recorded |
| H+00:15 | Config and tokenizer diff | same-architecture verdict; otherwise STOP |
| H+00:30 | Download FP8 weights; create manifest | every indexed shard present and hashed |
| H+01:00 | Build binaries and current HF reference environment | release binaries and reference imports succeed |
| H+01:30 | FP8 header census | every F8 weight is per-tensor or block-128 |
| H+02:00 | Model-backed `kernel-check` | final line `ALL GREEN` |
| H+02:30 | Naked memra first light and residency proof | native FP8 residency, no silent class fallback |
| H+03:15 | HF greedy-reference comparison | identical generated token ids |
| H+04:00 | Chunk invariance | invariant across the pinned chunk sizes |
| H+04:30 | Embedded-MTP `run-spec` | K=1..8 self-consistency passes |
| H+05:15 | ST serve battery and thinking surface | serve gate green; Qwen thinking switch detected and exercised |
| H+06:30 | Own-generation ranks and trim validation | corpus floor met; in-model trim A/B recorded |
| H+09:00 | External drafter build and gates | donor compatible, draft built, K=1..8 passes, e2e verdict recorded |
| H+10:00 | Close receipt | every publish gate green or an explicit STOP recorded |

Downloads, the HF reference environment, and the model-backed kernel battery can overlap.
Do not overlap two GPU jobs.

## H+00:00 - bind the official artifact

Search the official namespace instead of assuming the release name. An official BF16 repo is
source/reference material; the official FP8 sibling is the only production candidate.

```bash
curl -fsS \
  'https://huggingface.co/api/models?author=Qwen&search=Qwen3.8-27B&limit=100' \
  | tee "$R/hf-search.json"
jq -r '.[] | (.id // .modelId)' "$R/hf-search.json"

# Set this to the exact official FP8 id printed above if the release spelling differs.
export Q38_REPO=Qwen/Qwen3.8-27B-FP8
jq -e --arg repo "$Q38_REPO" \
  'any(.[]; (.id // .modelId) == $repo)' "$R/hf-search.json"

curl -fsS "https://huggingface.co/api/models/$Q38_REPO" \
  | tee "$R/hf-model.json"
export Q38_REV
Q38_REV=$(jq -er '.sha' "$R/hf-model.json")
printf 'repo=%s\nrevision=%s\n' "$Q38_REPO" "$Q38_REV" \
  | tee "$R/artifact-source.txt"
export A="$Q38_DERIVED_ROOT/$Q38_REV"
mkdir -p "$A" "$Q38_VENV_ROOT"
printf 'derived_artifacts=%s\n' "$A" | tee -a "$R/artifact-source.txt"
```

STOP if the FP8 sibling is absent. Do not fill the gap with Q8_0, a local conversion, or a
community FP8 requant.

Fetch the small files first:

```bash
mapfile -t META_FILES < <(
  jq -r '.siblings[].rfilename' "$R/hf-model.json" \
  | grep -E '(^README\.md$|\.json$|chat_template.*\.jinja$)'
)
hf download "$Q38_REPO" "${META_FILES[@]}" \
  --revision "$Q38_REV" --local-dir "$Q38_DIR" \
  2>&1 | tee "$R/hf-metadata-download.log"
```

## H+00:15 - same-architecture gate

Run the mechanized diff before downloading the weight shards:

```bash
set +e
python3 research/qwen38-prep-20260803/arch-diff-fields.py --expect-fp8 \
  "$Q38_DIR/config.json" "$Q36_REF/config.json" \
  2>&1 | tee "$R/config-diff.log"
CONFIG_RC=${PIPESTATUS[0]}
set -e

case "$CONFIG_RC" in
  0) echo "same-architecture config; continue" ;;
  2) echo "architecture matches; FP8 metadata waits for tensor-header proof" ;;
  *) echo "STOP: architecture/config contract changed"; exit 1 ;;
esac
```

Verify the tokenizer and thinking dialect. Qwen 3.8 must remain the Qwen ChatML class:

```bash
python3 - "$Q38_DIR/tokenizer_config.json" "$Q36_REF/tokenizer_config.json" <<'PY' \
  2>&1 | tee "$R/tokenizer-contract.log"
import json
import sys

new = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
hard = ("tokenizer_class", "pretokenize_regex", "add_bos_token")
bad = [(key, ref.get(key), new.get(key)) for key in hard if new.get(key) != ref.get(key)]
template = new.get("chat_template") or ""
markers = ("enable_thinking", "add_generation_prompt", "<think>")
missing = [marker for marker in markers if marker not in template]
print("hard-field diffs:", bad)
print("missing Qwen thinking markers:", missing)
if bad or missing:
    raise SystemExit(1)
PY

jq -S '.model.type, .pre_tokenizer' "$Q38_DIR/tokenizer.json" \
  > "$R/tokenizer38-structure.json"
jq -S '.model.type, .pre_tokenizer' "$Q36_REF/tokenizer.json" \
  > "$R/tokenizer36-structure.json"
diff -u "$R/tokenizer36-structure.json" "$R/tokenizer38-structure.json" \
  | tee "$R/tokenizer-structure.diff"
test "${PIPESTATUS[0]}" -eq 0
```

Any tokenizer-class, pre-tokenizer, regex, or thinking-template change is a bring-up lane,
not a runbook edit.

## Config-diff checklist

These are the frozen Qwen 3.6 values and memra's current interpretation. “GO with gates” means
the parser consumes the value generically; it does not waive model-backed, HF-reference, or
serve gates.

| Config field | Frozen 3.6 value | Day-one classification |
|---|---|---|
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` | any change: STOP; current HF mapping does not recognize a new Qwen 3.8 class |
| top/text `model_type` | `qwen3_5` / `qwen3_5_text` | any change: STOP; only these names map to the Qwen35 engine |
| `num_hidden_layers` | `64` | GO with gates; parsed size |
| `hidden_size` | `5120` | GO with gates; parsed size, but a change forbids the 3.6 MTP donor |
| `intermediate_size` | `17408` | GO with gates; parsed size |
| `num_attention_heads` | `24` | any change: STOP |
| `num_key_value_heads` | `4` | any change: STOP |
| `head_dim` | `256` | any change: STOP; FA dispatch and MTP interface are head-dimension keyed |
| `full_attention_interval` | `4` | any change: STOP |
| `layer_types` | 64 entries repeating `linear, linear, linear, full` | new type, changed cycle, interval mismatch, or count mismatch with `num_hidden_layers`: STOP |
| `linear_num_key_heads` | `16` | any change: STOP |
| `linear_num_value_heads` | `48` | any change: STOP |
| `linear_key_head_dim` / `linear_value_head_dim` | `128` / `128` | any change: STOP |
| `linear_conv_kernel_dim` | `4` | any change: STOP |
| `attention_bias` / `attention_dropout` | `false` / `0.0` | any change: STOP; these are not generic runtime knobs |
| `attn_output_gate` / `output_gate_type` | `true` / `swish` | any change: STOP; the Qwen35 forward assumes this gate contract |
| `hidden_act` | `silu` | any change: STOP |
| `tie_word_embeddings` | `false` | any change: STOP; tensor ownership and output-head loading change |
| `rope_parameters.rope_type` | `default` | any change, including YaRN: STOP |
| `rope_parameters.rope_theta` | `10000000` | any change: STOP |
| `rope_parameters.partial_rotary_factor` | `0.25` | any change: STOP |
| `rope_parameters.mrope_interleaved` | `true` | any change: STOP |
| `rope_parameters.mrope_section` | `[11,11,10]` | any change: STOP |
| `vocab_size` | `248320` | GO for plain loading if tokenizer gates pass; change forbids the 3.6 MTP donor |
| `bos_token_id` / `eos_token_id` | `248044` / `248044` | review chat stops; tokenizer/template gates must explain any change |
| `image_token_id` / `video_token_id` | `248056` / `248057` | text-only load may continue if tokenizer contract is unchanged |
| `mtp_num_hidden_layers` | `1` | any change: publish/spec STOP |
| `mtp_use_dedicated_embeddings` | `false` | any change: publish/spec STOP |
| `max_position_embeddings` | `262144` | GO with gates; parsed limit |
| `rms_norm_eps` | `1e-6` | GO with gates; parsed numeric value |
| `num_experts` | absent | any value: STOP; this is no longer the dense-hybrid architecture |
| `quant_method` / `fmt` | `fp8` / `e4m3` | explicit different value: STOP |
| `weight_block_size` | `[128,128]` | block-128 direct class; a different explicit block: STOP |
| `activation_scheme` | `dynamic` | explicit different value: STOP |

The config is not authoritative for scale granularity. The tensor sibling shape is:

- one scale value: `MEMRA_ST_E4M3`, per-tensor direct;
- `[out,1]` or `out` values: per-row, no direct kernel, STOP;
- `[ceil(out/128),ceil(in/128)]`: `MEMRA_ST_E4M3_BLK`, block-128 direct;
- anything else: unsupported, STOP.

## H+00:30 - full download and immutable manifest

```bash
hf download "$Q38_REPO" --revision "$Q38_REV" --local-dir "$Q38_DIR" \
  2>&1 | tee "$R/hf-full-download.log"

while read -r shard; do
  test -f "$Q38_DIR/$shard" || {
    echo "missing indexed shard: $shard"
    exit 1
  }
done < <(jq -r '.weight_map[]' "$Q38_DIR/model.safetensors.index.json" | sort -u)

find "$Q38_DIR" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$R/q38-files.sha256"
du -sh "$Q38_DIR" | tee "$R/q38-size.txt"
```

Do not mutate the downloaded config to make a diff pass. A RoPE override creates a different
model configuration and belongs in a separate lane.

## H+01:00 - build and HF reference environment

```bash
cargo build --release \
  --bin kernel-check \
  --bin run-gen \
  --bin run-spec \
  --bin concat-prime-probe \
  --bin frspec-owngen \
  --bin memra-server \
  2>&1 | tee "$R/build.log"

export HFV="$Q38_VENV_ROOT/q38-hf-$Q38_REV"
uv venv "$HFV"
uv pip install --python "$HFV/bin/python" --upgrade \
  torch transformers accelerate safetensors \
  2>&1 | tee "$R/hf-reference-install.log"
"$HFV/bin/python" -c \
  'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__)' \
  | tee "$R/hf-reference-versions.txt"
```

If the released model card names a newer minimum or an additional package, install that exact
requirement and record it. Do not use an older environment merely because it imports.

## H+01:30 - FP8 artifact classification

```bash
python3 tools/inspect-fp8-st.py "$Q38_DIR" --require-direct \
  2>&1 | tee "$R/fp8-header-census.log"
```

This check is header-only. It must report at least one FP8 weight and zero per-row/unsupported
weights. Runtime still has to prove finite scales, no E4M3 NaN refusal, supported transforms,
native residency, and native prefill dispatch.

## H+02:00 - model-backed kernel battery

`kernel-check` cannot use an ST directory as its real-weight oracle. Pass the frozen Qwen 3.6
GGUF so the 27B shape sections are model-backed, while the synthetic FP8 cells cover per-tensor
and block-128 arithmetic:

```bash
flock /tmp/gpu5090.lock \
  target/release/kernel-check "$Q36_GGUF" \
  2>&1 | tee "$R/kernel-check.log"
grep -q 'ALL GREEN' "$R/kernel-check.log"
```

A new Qwen 3.8 shape that is not represented by the existing synthetic cells is a kernel lane,
not a reason to accept a skipped section.

## H+02:30 - naked direct-path proof

Use a prompt longer than the prefill GEMM threshold so block-128 MMQ can dispatch:

```bash
export PROBE=tools/fast-gate/prompts/probe.txt
FP8_DEFAULT_ENV=(
  -u MEMRA_ST_E4M3
  -u MEMRA_ST_E4M3_BLK
  -u MEMRA_FP8_MMQ
  -u MEMRA_FP8_FOLD
  -u MEMRA_FP8_BLK_GPU
  -u MEMRA_PP_FP8
)

flock /tmp/gpu5090.lock env "${FP8_DEFAULT_ENV[@]}" \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=32 \
  MEMRA_PROMPT_FILE="$PROBE" \
  MEMRA_RESIDENCY_CENSUS=1 \
  target/release/run-gen "$Q38_DIR" \
  2>&1 | tee "$R/run-gen-naked.log"

grep -q 'argmax=.*MATCH' "$R/run-gen-naked.log"
grep -Eq 'F8_E4M3(_BLK)?:[[:space:]]+[1-9]' "$R/run-gen-naked.log"
if grep -Eq 'block-128[[:space:]]*:[[:space:]]*[1-9]' "$R/fp8-header-census.log"; then
  flock /tmp/gpu5090.lock env "${FP8_DEFAULT_ENV[@]}" \
    MEMRA_CHAT=1 \
    MEMRA_PROMPT_FILE="$PROBE" \
    MEMRA_PP_ONLY=1 \
    MEMRA_PP_WARMUP=1 \
    MEMRA_PP_REPS=1 \
    target/release/run-gen "$Q38_DIR" \
    2>&1 | tee "$R/run-gen-pp-direct.log"
  grep -Eq 'fp8-mmq dispatches: [1-9]' "$R/run-gen-pp-direct.log"
fi
```

Run one diagnostic rollback census. This is proof of the seam, not a production arm:

```bash
flock /tmp/gpu5090.lock env "${FP8_DEFAULT_ENV[@]}" \
  MEMRA_ST_E4M3=0 \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=1 \
  MEMRA_PROMPT_FILE="$PROBE" \
  MEMRA_RESIDENCY_CENSUS=1 \
  target/release/run-gen "$Q38_DIR" \
  2>&1 | tee "$R/run-gen-q8-rollback-census.log"
! grep -Eq 'F8_E4M3(_BLK)?:[[:space:]]+[1-9]' "$R/run-gen-q8-rollback-census.log"
grep -Eq 'Q8_0:[[:space:]]+[1-9]' "$R/run-gen-q8-rollback-census.log"
```

The naked log must contain native `F8_E4M3` and/or `F8_E4M3_BLK` residency. The rollback log
must remove those native bytes and increase Q8_0 residency. Residual small Q8_0 tensors are
allowed; routing the FP8 projection bank through Q8_0 is not. Any native decline, NaN refusal,
bad grid, or zero block-MMQ dispatch is STOP.

Never set `MEMRA_FP8_FOLD=1`; it is lossy. Never set `MEMRA_FP8_MMQ=0`; that reintroduces
dequant-per-call. Never use `MEMRA_FP8_MMQ=1`; that enables the duplicate stash source.

## H+03:15 - Hugging Face greedy-reference gate

Generate the reference from the same official FP8 directory, same prompt, same chat template,
thinking enabled, and greedy decoding. The comparator checks the rendered prompt ids before it
checks the generated ids:

```bash
flock /tmp/gpu5090.lock \
  "$HFV/bin/python" tools/hf-greedy-reference.py "$Q38_DIR" \
  --prompt-file "$PROBE" \
  --tokens-out "$R/hf-greedy.json" \
  --max-new-tokens 32 \
  2>&1 | tee "$R/hf-greedy.log"

python3 tools/compare-greedy-tokens.py \
  "$R/hf-greedy.json" "$R/run-gen-naked.log" \
  2>&1 | tee "$R/hf-vs-memra.log"
```

The gate is exact prompt-token and generated-token identity. A mismatch is not waived as “close
logits”; prompt-token failure is tokenizer/template work, while a generated-token failure after
prompt parity is model arithmetic.

## H+04:00 - chunk invariance

The probe and wrapper accept an HF safetensors directory:

```bash
MEMRA_CHUNKINV_LOG="$R/chunk-invariance.raw.log" \
flock /tmp/gpu5090.lock \
  tools/chunk-invariance-gate.sh "$Q38_DIR" \
  --chunks 2048,64,32 --steps 48 --expect-invariant \
  2>&1 | tee "$R/chunk-invariance.gate.log"
grep -q 'chunk-invariance-gate: PASS' "$R/chunk-invariance.gate.log"
```

Any chunk-dependent result is STOP. Do not tune the chunk list or switch the expectation to make
the release pass.

## H+04:30 - embedded MTP gate

If the checkpoint carries its own MTP tensors, run without `MEMRA_SPEC_K` so the binary executes
its K=1..8 self-consistency battery:

```bash
if jq -e 'any(.weight_map | keys[]; startswith("mtp."))' \
  "$Q38_DIR/model.safetensors.index.json" >/dev/null; then
  flock /tmp/gpu5090.lock env \
    -u MEMRA_SPEC_K \
    -u MEMRA_MTP_DRAFT \
    -u MEMRA_FRSPEC_TRIM \
    MEMRA_CHAT=1 \
    MEMRA_NGEN=64 \
    MEMRA_PROMPT_FILE="$PROBE" \
    target/release/run-spec "$Q38_DIR" \
    2>&1 | tee "$R/run-spec-embedded.log"
  grep -q 'SELF-CONSISTENCY PASS' "$R/run-spec-embedded.log"
else
  printf 'WAIT: checkpoint carries no embedded MTP tensors; run-spec waits for a compatible draft\n' \
    | tee "$R/run-spec-embedded.WAIT"
fi
```

Plain first-light evidence may continue without a head, but publication does not: `run-spec`
waits until either the embedded head or the compatible external draft below exists.

## H+05:15 - serving and thinking surface

Run the existing ST battery:

```bash
flock /tmp/gpu5090.lock \
  tools/serve-st-gate.sh "$Q38_DIR" \
  2>&1 | tee "$R/serve-st-gate.log"
grep -q 'serve-st-gate: 0 failed' "$R/serve-st-gate.log"
```

Pin the per-architecture thinking mapping in unit tests:

```bash
cargo test -p memra-tokenizer qwen_think_mode_covers_all_three_directions \
  2>&1 | tee "$R/test-qwen-thinking.log"
cargo test -p memra-server reasoning_effort_maps_to_think_switch \
  2>&1 | tee "$R/test-reasoning-effort.log"
```

Then exercise the actual Qwen 3.8 template:

```bash
export ADDR=127.0.0.1:8188
export BASE=http://$ADDR
flock -F /tmp/gpu5090.lock env MEMRA_COMPAT=openai MEMRA_SERVE_SPEC=0 \
  MEMRA_MODELS="q38=$Q38_DIR" MEMRA_ADDR="$ADDR" \
  target/release/memra-server > "$R/thinking-server.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true' EXIT
for _ in $(seq 600); do
  curl -sf "$BASE/health" >/dev/null && break
  sleep 2
done
curl -sf "$BASE/health" >/dev/null
grep -Eq 'q38: template caps .*think=true think_switch=true chat_ok=true' \
  "$R/thinking-server.log"

for effort in absent none high; do
  extra=
  [ "$effort" != absent ] && extra=",\"reasoning_effort\":\"$effort\""
  curl -fsS "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"q38\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue?\"}],
         \"max_tokens\":64,\"temperature\":0$extra}" \
    > "$R/thinking-$effort.json"
done

python3 - "$R" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = {
    name: json.load(open(root / f"thinking-{name}.json"))["choices"][0]["message"]
    for name in ("absent", "none", "high")
}
for name, message in records.items():
    emitted = (message.get("reasoning") or "") + (message.get("content") or "")
    assert emitted.strip(), f"{name}: empty response"
assert not (records["none"].get("reasoning") or ""), "none did not close thinking"
assert (records["absent"].get("reasoning") or ""), "default did not use Qwen thinking"
assert (records["high"].get("reasoning") or ""), "high did not enable Qwen thinking"
print("THINKING-SURFACE PASS: default/high on, none off")
PY

kill "$SPID"
wait "$SPID" || true
trap - EXIT
```

For the Qwen binary switch, `low`, `medium`, and `high` all mean thinking on; `none` and
`minimal` mean off; absent preserves the template default, which is on.

## H+06:30 - own-generation trim regime

Ranks always come from Qwen 3.8's own generations with chat templating on. Use bounded chunks
so the shared GPU lock is released between batches:

```bash
export PACK=research/gemma4-bringup/corpus-prompts
export RANKS="$A/q38-own-ranks-32768.gguf"
export CORPUS="$A/q38-own-corpus-ids.txt"
test "$(find "$PACK" -type f -name '*.txt' | wc -l)" -eq 254

while [ ! -f "$RANKS" ]; do
  flock /tmp/gpu5090.lock \
    target/release/frspec-owngen "$Q38_DIR" "$RANKS" 32768 \
    --ngen 1024 \
    --corpus-out "$CORPUS" \
    --limit 32 \
    --validate \
    "$PACK" \
    2>&1 | tee -a "$R/frspec-owngen.log"
done

awk '
  { total += NF }
  END {
    print "own-generated tokens:", total
    if (total < 131072) exit 1
  }
' "$CORPUS" | tee "$R/own-corpus-count.txt"
sha256sum "$CORPUS" "$RANKS" "$RANKS.txt" | tee "$R/own-ranks.sha256"
```

This is the daily-drafter regime's canonical 254-prompt pack, greedy decoding, chat template on,
and bounded 32-prompt lock holds. The 1,024-token window is deliberately larger than the older
512-token receipts so the current four-times-top-N floor is enforceable rather than theoretical.
The floor is at least 131,072 own-generated tokens for a 32,768-row head. If the count misses it,
STOP and expand the frozen prompt pack; do not build from the warning-sized corpus. The final
`--validate` run records embedded-MTP baseline versus runtime-trimmed e2e throughput. Acceptance
explains the result; e2e tokens/s decides it.

## H+09:00 - build the external trimmed drafter

The current standalone builder extracts MTP bytes from a GGUF donor. It does not extract from
the ST directory. The approved donor variant is executable only when the Qwen 3.8 trunk/MTP
interface is exactly the frozen Qwen 3.6 interface:

```bash
python3 - "$Q38_DIR/config.json" "$Q36_REF/config.json" <<'PY' \
  2>&1 | tee "$R/donor-interface.log"
import json
import sys

new = json.load(open(sys.argv[1])).get("text_config")
ref = json.load(open(sys.argv[2])).get("text_config")
keys = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
)
bad = [(key, ref.get(key), new.get(key)) for key in keys if new.get(key) != ref.get(key)]
print("donor interface diffs:", bad)
if bad:
    raise SystemExit(1)
PY

export DRAFT="$A/draft-q38-owntrim-nvfp4head-q4blk.gguf"
tools/make-trimmed-draft.sh \
  "$Q36_GGUF" "$RANKS.txt" "$DRAFT" 32768 \
  2>&1 | tee "$R/make-trimmed-draft.log"
sha256sum "$DRAFT" "$RANKS" "$RANKS.txt" \
  | tee "$R/drafter.sha256"

flock /tmp/gpu5090.lock env \
  -u MEMRA_SPEC_K \
  -u MEMRA_MTP_DRAFT \
  -u MEMRA_FRSPEC_TRIM \
  MEMRA_MTP_DRAFT="$DRAFT" \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=64 \
  MEMRA_PROMPT_FILE="$PROBE" \
  target/release/run-spec "$Q38_DIR" \
  2>&1 | tee "$R/run-spec-owntrim.log"
grep -q 'SELF-CONSISTENCY PASS' "$R/run-spec-owntrim.log"

mkdir -p "$R/drafter-prompts"
cp "$PROBE" "$R/drafter-prompts/probe.txt"
cp research/e2e/prompts/p3-agentic-long.txt "$R/drafter-prompts/p3-agentic-long.txt"

run_draft_arm() {
  local arm=$1 rep=$2 draft=${3:-}
  local envs=(
    MEMRA_SPEC_K=3
    MEMRA_NGEN=256
    MEMRA_CHAT=1
    MEMRA_PROMPT_DIR="$R/drafter-prompts"
  )
  [ -n "$draft" ] && envs+=(MEMRA_MTP_DRAFT="$draft")
  flock /tmp/gpu5090.lock env \
    -u MEMRA_MTP_DRAFT \
    -u MEMRA_FRSPEC_TRIM \
    "${envs[@]}" \
    target/release/run-spec "$Q38_DIR" \
    2>&1 | tee "$R/drafter-ab-$arm-r$rep.log"
}

# Adjacent, alternating order: embedded/trimmed, then trimmed/embedded.
run_draft_arm embedded 1
run_draft_arm owntrim 1 "$DRAFT"
run_draft_arm owntrim 2 "$DRAFT"
run_draft_arm embedded 2

grep -hE '^\[SWEEP\]|SELF-CONSISTENCY' "$R"/drafter-ab-*.log \
  | tee "$R/drafter-ab-summary.txt"

flock /tmp/gpu5090.lock env \
  MEMRA_MTP_DRAFT="$Q36_DRAFT" \
  MEMRA_SPEC_K=3 \
  MEMRA_NGEN=256 \
  MEMRA_CHAT=1 \
  MEMRA_PROMPT_DIR="$R/drafter-prompts" \
  target/release/run-spec "$Q36_GGUF" \
  2>&1 | tee "$R/q36-daily-reference.log"
```

The donor supplies only byte-verbatim NextN/MTP block bytes; ranks are from Qwen 3.8, and the
serving model supplies token embeddings. Verify-based speculation remains exact, but donor drift
can reduce acceptance. The four A/B runs are one adjacent, alternating N=2 session over short and
agentic-long prompt classes. Adopt the external draft only when both per-prompt evidence and the
aggregate e2e tokens/s justify it; acceptance is diagnostic.

The Qwen 3.6 daily run is a frozen operational reference, not a Qwen 3.8 acceptance threshold.
Keep its number separate from the embedded-versus-owntrim Qwen 3.8 decision.

If the donor-interface check fails, do not create a full-model GGUF bridge. Keep the embedded ST
MTP path and open a narrowly scoped ST-MTP extraction lane before publishing an external draft.

## Completion and STOP matrix

Day one is complete only when the receipt directory contains:

- official repo id and immutable revision;
- full local file hashes;
- config and tokenizer verdicts;
- FP8 header census;
- model-backed `kernel-check` raw log;
- naked and rollback residency logs;
- HF and memra token streams plus exact comparison;
- chunk-invariance raw log;
- embedded and external-draft `run-spec` logs;
- ST serve battery;
- thinking default/off/on responses;
- own-generation corpus/rank/draft paths under `/data`, their hashes, and the e2e verdict.

Hard architecture STOP, open a bring-up lane:

- new `architectures`/`model_type`;
- changed full-attention cycle, attention/KV head counts, head dimension, attention gate/bias,
  GDN geometry, RoPE contract, tokenizer class/pre-tokenizer, or dense-to-MoE change;
- new tensor names or shapes the current Qwen35 mapping cannot resolve.

Artifact/direct-path STOP:

- no official FP8 release;
- explicit non-E4M3 or non-128 block metadata;
- per-row or unsupported FP8 scale layouts;
- native FP8 residency absent, block-MMQ dispatch absent for a block artifact, or any FP8 bank
  silently landing in Q8_0;
- HF greedy token mismatch.

Correctness/publish STOP:

- `kernel-check`, internal argmax, chunk invariance, or K=1..8 self-consistency failure;
- no embedded or interface-compatible external MTP head that passes `run-spec`;
- missing `enable_thinking`, `think_switch=false`, or live default/off/on mapping failure;
- external donor interface mismatch or a trimmed draft that loses e2e throughput.

Do not merge, tag, or publish support from this runbook until every applicable gate is green.
