#!/usr/bin/env bash
# lane/q8-argmax ARM BATTERY — the 27B Q8_0 board-2048 prefill-vs-decode argmax MISMATCH.
# Rig: RunPod RTX PRO 6000 Blackwell WS, 188 SM, driver 570.211.01 (community pod; exactness
# rows only, no perf claimed). Raw log tee'd FIRST, parsed second.
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=/root/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24-q8a
LOGD=/root/bw24-q8a/research/q8-argmax-20260806/pod; mkdir -p "$LOGD"
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
P=/root/bw24-q8a/research/e2e/prompts/board-2048.txt
BIN=/root/bw24-q8a/target/release/run-gen

wait_free() { local n=0 u; while [ $n -lt 240 ]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 4000 ] && { echo "[wait] clean ${u}MiB"; return 0; }; sleep 15; n=$((n+1)); done
    echo "[wait] TIMEOUT ${u}MiB"; return 1; }

arm() { local name="$1"; shift
    wait_free || { echo "ARM $name SKIPPED gpu-busy"; return 2; }
    echo "=== ARM $name  $(date -Is)  :: $* ==="
    ( "$@" ) > "$LOGD/arm-$name.log" 2>&1; local rc=$?
    echo "  EXIT=$rc"
    grep -E 'prefill argmax|batched-prime|\[gate\]|panicked|Error' "$LOGD/arm-$name.log" | head -6
    echo
}

echo "########## q8-argmax ARMS $(date -Is) commit $(cat /root/bw24-q8a/BOX-COMMIT.txt) ##########"
echo "run-gen md5: $(md5sum $BIN)"
echo "artifact md5: $(md5sum $Q8)"
echo "prompt md5: $(md5sum $P)"
echo

# A. REPRO on current train tip (mission item 1)
arm A1-repro-tip           env MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8
arm A2-repro-tip-rerun     env MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8

# B. the k27 rung test (mission item 2) — re-run on THIS tip, board-2048 prompt (the
#    v071 triage ran split arms on the DEFAULT short prompt, not board-2048).
arm B1-fa-split8   env MEMRA_FA_SPLIT=8  MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8
arm B2-fa-split16  env MEMRA_FA_SPLIT=16 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8
arm B3-fa-split1   env MEMRA_FA_SPLIT=1  MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8

# C. the other SM-keyed / Q8-specific doors
arm C1-q80g2-off   env MEMRA_Q80_G2=0    MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8
arm C2-fast0       env MEMRA_FAST=0      MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8
arm C3-prime-chunk-off env MEMRA_PRIME_CHUNK=100000 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $Q8

# D. controls: the SAME prompt on the NVFP4 artifact (was MATCH), and the q9 Q8_0 class
arm D1-nv-same-prompt env MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN $NV
arm D2-q9q8-same-prompt env MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P $BIN /root/pilot/q9base-Q8_0.gguf

echo "########## ARMS DONE $(date -Is) ##########"
