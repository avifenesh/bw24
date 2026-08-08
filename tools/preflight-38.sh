#!/usr/bin/env bash
# Qwen3.8-27B day-one readiness. WAIT is release-dependent and does not fail the run.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PASS=0
WAIT=0
FAIL=0

pass() { PASS=$((PASS + 1)); printf '[PASS] %s\n' "$*"; }
wait_for() { WAIT=$((WAIT + 1)); printf '[WAIT] %s\n' "$*"; }
fail() { FAIL=$((FAIL + 1)); printf '[FAIL] %s\n' "$*"; }

check_file() {
    local label=$1 path=$2
    if [ -f "$path" ]; then
        pass "$label: $path"
    else
        fail "$label missing: $path"
    fi
}

check_dir() {
    local label=$1 path=$2
    if [ -d "$path" ]; then
        pass "$label: $path"
    else
        fail "$label missing: $path"
    fi
}

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/data/ai-ml/hf-models}
Q38_REPO=${Q38_REPO:-Qwen/Qwen3.8-27B-FP8}
Q36_OFFICIAL_REPO=${Q36_OFFICIAL_REPO:-Qwen/Qwen3.6-27B-FP8}
Q38_DIR=${Q38_DIR:-$ARTIFACT_ROOT/qwen38-27b-fp8}
Q36_ST=${Q36_ST:-$ARTIFACT_ROOT/qwen36-27b-blk128fp8}
Q36_BASE=${Q36_BASE:-$ARTIFACT_ROOT/qwen36-27b-hf-min}
Q36_GGUF_DIR=${Q36_GGUF_DIR:-$ARTIFACT_ROOT/qwen36-27b-nvfp4-mtp}
Q36_GGUF=${Q36_GGUF:-$Q36_GGUF_DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
Q36_DRAFT=${Q36_DRAFT:-$Q36_GGUF_DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf}
MIN_FREE_GB=${MIN_FREE_GB:-96}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-lane/cx-38-runbook}
TRAIN_TIP=${TRAIN_TIP:-9aebdb3e}
STEERING=${STEERING:-$HOME/.lanectl/inbox/cx-38prep.md}

printf 'Qwen3.8-27B FP8-ST preflight\n'
printf 'timestamp: %s\n' "$(date --iso-8601=seconds)"
printf 'repo: %s\n' "$(pwd)"
printf 'target repo: %s\n' "$Q38_REPO"
printf 'target dir: %s\n\n' "$Q38_DIR"

branch=$(git branch --show-current 2>/dev/null || true)
if [ "$branch" = "$EXPECTED_BRANCH" ]; then
    pass "dedicated branch: $branch"
elif [ -n "$branch" ] && [ "$branch" != "main" ]; then
    wait_for "branch is isolated but differs from expected: $branch"
else
    fail "feature work must not run on branch ${branch:-<detached>}"
fi
if git merge-base --is-ancestor "$TRAIN_TIP" HEAD >/dev/null 2>&1; then
    pass "train tip $TRAIN_TIP is an ancestor of HEAD"
else
    fail "HEAD does not contain train tip $TRAIN_TIP"
fi

if [ -f "$STEERING" ]; then
    pass "steering file present: $STEERING"
else
    wait_for "steering file absent: $STEERING"
fi

for tool in git curl jq python3 uv hf cargo nvcc nvidia-smi flock sha256sum; do
    if command -v "$tool" >/dev/null 2>&1; then
        pass "tool available: $tool"
    else
        fail "tool missing: $tool"
    fi
done
if gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) \
    && [ -n "$gpu_names" ]; then
    pass "NVIDIA GPU visible: $(printf '%s' "$gpu_names" | paste -sd ';' -)"
else
    fail "nvidia-smi cannot see an NVIDIA GPU"
fi

if hf auth whoami >/dev/null 2>&1; then
    pass "Hugging Face authentication works"
elif [ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]; then
    wait_for "HF token is present but hf auth whoami failed"
else
    fail "no working Hugging Face authentication"
fi

check_dir "artifact root" "$ARTIFACT_ROOT"
if [ -w "$ARTIFACT_ROOT" ]; then
    pass "artifact root is writable"
else
    fail "artifact root is not writable: $ARTIFACT_ROOT"
fi
free_gb=$(df -Pk "$ARTIFACT_ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
    pass "artifact filesystem free space: ${free_gb} GiB (minimum ${MIN_FREE_GB})"
else
    fail "artifact filesystem has ${free_gb} GiB free; need at least ${MIN_FREE_GB}"
fi

check_dir "3.6 FP8-ST A/B baseline" "$Q36_ST"
check_file "3.6 FP8-ST config" "$Q36_ST/config.json"
check_file "3.6 FP8-ST index" "$Q36_ST/model.safetensors.index.json"
check_file "3.6 FP8-ST tokenizer config" "$Q36_ST/tokenizer_config.json"
check_file "3.6 FP8-ST tokenizer" "$Q36_ST/tokenizer.json"
check_file "3.6 architecture reference config" "$Q36_BASE/config.json"
check_file "3.6 tokenizer reference" "$Q36_BASE/tokenizer_config.json"
check_file "3.6 tokenizer structure reference" "$Q36_BASE/tokenizer.json"
check_file "3.6 frozen GGUF comparison baseline" "$Q36_GGUF"
check_file "3.6 frozen own-trim drafter" "$Q36_DRAFT"
check_file "day-one probe prompt" tools/fast-gate/prompts/probe.txt
check_file "drafter long prompt" research/e2e/prompts/p3-agentic-long.txt
check_dir "canonical own-generation prompt pack" research/gemma4-bringup/corpus-prompts
prompt_count=$(find research/gemma4-bringup/corpus-prompts -type f -name '*.txt' | wc -l)
if [ "$prompt_count" -eq 254 ]; then
    pass "canonical own-generation prompt count: $prompt_count"
else
    fail "canonical own-generation prompt count is $prompt_count; expected 254"
fi
check_dir "llama.cpp gguf-py" /data/projects/llama.cpp/gguf-py
check_file "MTP extractor" tools/extract_mtp_draft.py
check_file "draft-head trimmer" tools/trim_draft_head.py
if [ -x tools/make-trimmed-draft.sh ]; then
    pass "trimmed-draft builder is executable"
else
    fail "trimmed-draft builder missing or not executable"
fi
if [ -x /data/projects/llama.cpp/build/bin/llama-quantize ]; then
    pass "llama-quantize is executable"
else
    fail "llama-quantize missing or not executable"
fi
for helper in \
    tools/serve-st-gate.sh \
    tools/chunk-invariance-gate.sh \
    tools/inspect-fp8-st.py \
    tools/hf-greedy-reference.py \
    tools/compare-greedy-tokens.py; do
    if [ -x "$helper" ]; then
        pass "day-one helper is executable: $helper"
    else
        fail "day-one helper missing or not executable: $helper"
    fi
done
if python3 -c 'import numpy' >/dev/null 2>&1; then
    pass "Python NumPy dependency is available"
else
    fail "Python NumPy dependency is missing"
fi

if python3 - "$Q36_ST/tokenizer_config.json" "$Q36_BASE/tokenizer_config.json" <<'PY'
import json
import sys

candidate = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
fields = ("tokenizer_class", "pretokenize_regex", "add_bos_token")
raise SystemExit(any(candidate.get(field) != reference.get(field) for field in fields))
PY
then
    pass "3.6 ST tokenizer matches the frozen Qwen tokenizer contract"
else
    fail "3.6 ST tokenizer differs from the frozen Qwen tokenizer contract"
fi

if [ -d "$Q36_ST" ]; then
    if python3 tools/inspect-fp8-st.py "$Q36_ST" --require-direct >/tmp/memra-preflight-38-fp8.log 2>&1; then
        pass "3.6 FP8-ST header census satisfies the direct-path contract"
    else
        fail "3.6 FP8-ST header census failed (see /tmp/memra-preflight-38-fp8.log)"
        tail -20 /tmp/memra-preflight-38-fp8.log
    fi
fi

if [ -f "$Q36_BASE/config.json" ]; then
    if python3 research/qwen38-prep-20260803/arch-diff-fields.py \
        "$Q36_BASE/config.json" "$Q36_BASE/config.json" >/tmp/memra-preflight-38-config.log 2>&1; then
        pass "config-diff helper self-test"
    else
        fail "config-diff helper self-test failed"
        tail -20 /tmp/memra-preflight-38-config.log
    fi
fi

if curl -fsSL --retry 2 \
    "https://huggingface.co/$Q36_OFFICIAL_REPO/resolve/main/config.json" \
    -o /tmp/memra-preflight-38-q36-official-config.json \
    && python3 research/qwen38-prep-20260803/arch-diff-fields.py --expect-fp8 \
        /tmp/memra-preflight-38-q36-official-config.json \
        "$Q36_BASE/config.json" >/tmp/memra-preflight-38-q36-official.log 2>&1; then
    pass "official 3.6 FP8 config matches the frozen architecture/block-128 contract"
else
    fail "official 3.6 FP8 config no longer matches the frozen contract"
    tail -20 /tmp/memra-preflight-38-q36-official.log 2>/dev/null || true
fi

repo_status=$(curl -sS -o /tmp/memra-preflight-38-hf.json -w '%{http_code}' \
    'https://huggingface.co/api/models?author=Qwen&search=Qwen3.8-27B&limit=100' || true)
if [ "$repo_status" = "200" ] && jq -e --arg repo "$Q38_REPO" \
    'any(.[]; (.id // .modelId) == $repo)' /tmp/memra-preflight-38-hf.json >/dev/null; then
    pass "official target repo is visible: $Q38_REPO"
elif [ "$repo_status" = "200" ]; then
    wait_for "official target repo not visible in the Qwen namespace: $Q38_REPO"
else
    fail "official Qwen namespace lookup failed (HTTP ${repo_status:-network-error})"
fi

if [ -d "$Q38_DIR" ]; then
    check_file "target config" "$Q38_DIR/config.json"
    check_file "target safetensors index" "$Q38_DIR/model.safetensors.index.json"
    check_file "target tokenizer config" "$Q38_DIR/tokenizer_config.json"
    check_file "target tokenizer" "$Q38_DIR/tokenizer.json"
    if [ -f "$Q38_DIR/config.json" ] && [ -f "$Q36_BASE/config.json" ]; then
        python3 research/qwen38-prep-20260803/arch-diff-fields.py --expect-fp8 \
            "$Q38_DIR/config.json" "$Q36_BASE/config.json"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            pass "target config is same-architecture and FP8 metadata is known-direct"
        elif [ "$rc" -eq 2 ]; then
            wait_for "target architecture matches, but FP8 metadata needs tensor inspection"
        else
            fail "target config is not a runbook fast-path"
        fi
    fi
    if compgen -G "$Q38_DIR/*.safetensors" >/dev/null; then
        if python3 tools/inspect-fp8-st.py "$Q38_DIR" --require-direct; then
            pass "target FP8 tensor headers satisfy the direct-path contract"
        else
            fail "target FP8 tensor headers require a non-direct path"
        fi
    else
        wait_for "target safetensors have not been downloaded"
    fi
else
    wait_for "target artifact directory does not exist yet: $Q38_DIR"
fi

build_log=/tmp/memra-preflight-38-build.log
if cargo build --release --quiet \
    --bin kernel-check \
    --bin run-gen \
    --bin run-spec \
    --bin concat-prime-probe \
    --bin frspec-owngen \
    --bin memra-server >"$build_log" 2>&1; then
    pass "release binaries build"
else
    fail "release binary build failed (see $build_log)"
    tail -40 "$build_log"
fi
for binary in kernel-check run-gen run-spec concat-prime-probe frspec-owngen memra-server; do
    if [ -x "target/release/$binary" ]; then
        pass "binary ready: target/release/$binary"
    else
        fail "binary missing after build: target/release/$binary"
    fi
done

printf '\nsummary: PASS=%d WAIT=%d FAIL=%d\n' "$PASS" "$WAIT" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
    printf 'PREFLIGHT-38: FAIL\n'
    exit 1
fi
printf 'PREFLIGHT-38: READY-WITH-WAITS\n'
