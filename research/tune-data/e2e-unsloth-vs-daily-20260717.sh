#!/usr/bin/env bash
# Board protocol run 2026-07-17: unsloth 27B artifact (+own trimmed draft) vs the
# nvidia daily, both engines, p1/p2/p3, N=2 interleaved rounds, gpu-full-power on.
# bw24-new config: BW24_MTP_DRAFT=<owntrim-nvfp4head-q4blk> BW24_SPEC_HPOST=1 K=3
# (probe 2026-07-17: hpost 86.1 > plain 81.7 on p2; pmin flat).
# Raw log: research/tune-data/e2e-unsloth-vs-daily-20260717.log
set -uo pipefail
cd /home/avifenesh/projects/bw24
D=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
U=/data/ai-ml/hf-models/unsloth-qwen36-27b-nvfp4-gguf
PDIR=research/e2e/prompts
RS=./target/release/run-spec

bw_arm() { # name model env...
    local name=$1 model=$2; shift 2
    for P in p1-code-short p2-code-medium p3-agentic-long; do
        echo "=== bw24 $name $P ==="
        env "$@" BW24_PROMPT="$(cat $PDIR/$P.txt)" BW24_SPEC_K=3 BW24_NGEN=256 \
            timeout 900 $RS "$model" 2>/dev/null | grep -E "K=3\]|acceptance" | head -2
    done
}

ll_arm() { # name model draft
    local name=$1 model=$2 draft=$3
    echo "=== llama $name boot ==="
    /data/projects/llama.cpp/build/bin/llama-server -m "$model" --ctx-size 16384 -ngl 999 \
        -fa on --cache-type-k q8_0 --cache-type-v q5_1 --host 127.0.0.1 --port 8899 --parallel 1 \
        --model-draft "$draft" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 -ngld 999 \
        > /tmp/e2e-ll-$name.log 2>&1 &
    local pid=$!
    for i in $(seq 240); do curl -sf http://127.0.0.1:8899/health >/dev/null 2>&1 && break; sleep 2; done
    for P in p1-code-short p2-code-medium p3-agentic-long; do
        echo "=== llama $name $P ==="
        python3 - "$PDIR/$P.txt" << 'PY'
import json,sys,urllib.request
prompt=open(sys.argv[1]).read()
req=urllib.request.Request('http://127.0.0.1:8899/completion',
  data=json.dumps({'prompt':prompt,'n_predict':256,'temperature':0,'cache_prompt':False}).encode(),
  headers={'Content-Type':'application/json'})
r=json.loads(urllib.request.urlopen(req, timeout=900).read())
t=r['timings']
print(f"gen: {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} tok/s (prompt {t['prompt_n']} @ {t['prompt_per_second']:.0f})")
PY
    done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
    sleep 3
}

for round in 1 2; do
    echo "########## ROUND $round ##########"
    bw_arm old "$D/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf" \
        BW24_FRSPEC_TRIM=$D/mtp-Qwen3.6-27B-Q4_K_M-frspec-code75-32768.gguf \
        BW24_SPEC_PMIN=0.15 BW24_SPEC_HPOST=1
    bw_arm new "$U/Qwen3.6-27B-unsloth-NVFP4-w4attn-im-q5h-mtp.gguf" \
        BW24_MTP_DRAFT=$U/draft-unsloth-owntrim-nvfp4head-q4blk.gguf BW24_SPEC_HPOST=1
    ll_arm old "$D/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf" "$D/mtp-Qwen3.6-27B-Q4_K_M-frspec-code75-32768.gguf"
    ll_arm new "$U/Qwen3.6-27B-unsloth-NVFP4-w4attn-im-q5h-mtp.gguf" "$U/draft-unsloth-owntrim-nvfp4head-q4blk.gguf"
done
echo "########## DONE ##########"
