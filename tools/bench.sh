#!/usr/bin/env bash
# bw24 vs llama.cpp vs vLLM vs SGLang — FAIR single-stream prefill+decode, N=5 medians.
# Protocol is defined in COMPETITOR-SETUP.md section 5 and MUST stay identical across all 4 engines:
#   same prompt (P=512), same gen length (N_GEN=128, greedy/temp0), single stream, 1 warmup,
#   N timed runs -> MEDIAN, gpu-full-power on, GPU otherwise idle (SERIAL).
#
# SERIAL by design: each engine runs ALONE. vLLM/SGLang GPU-memory profiling RACES if other GPU
# procs allocate/free during init. NEVER run two engines (or a bench + bw24) concurrently.
# This script auto-runs ONLY llama.cpp (llama-bench, llama-server+MTP) and bw24 (both release the GPU
# fully between steps). vLLM/SGLang are PRINTED as exact serial commands (their init race is the reason).
set -euo pipefail
export PATH=/usr/local/cuda-13.1/bin:$PATH
export GGML_CUDA_GRAPH_OPT=1
gpu-full-power on >/dev/null 2>&1 || true

# ---- protocol knobs (identical across engines) ----
N=${N:-5}            # timed repetitions -> median
P=${P:-512}          # prefill prompt length (tokens)
NGEN=${NGEN:-128}    # generated tokens, greedy / temp 0

# ---- paths (verified on this box) ----
LB=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
LS=/home/avifenesh/projects/llama.cpp/build/bin/llama-server
M9_Q8=/home/avifenesh/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf
M9_NVFP4=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27_NVFP4=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
M27_MTP_DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-NVFP4.gguf
VLLM_VENV=/data/projects/bench-engines/vllm-venv
SGLANG_VENV=/data/projects/bench-engines/sglang-venv

# ---- serial guard: refuse to run if another compute proc holds GPU mem (init/bench would race) ----
others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | awk -F, '$2+0 > 500' || true)
if [ -n "$others" ]; then
  echo "ABORT: other GPU compute procs hold >500MiB — protocol requires the GPU be idle (SERIAL):"
  echo "$others"
  exit 1
fi

# median of stdin numbers (one per line)
median() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "NA";exit} m=int((NR+1)/2); print (NR%2)? a[m] : (a[m]+a[m+1])/2 }'; }

echo "=================================================================="
echo " FAIR BENCH  P=$P  NGEN=$NGEN  N=$N (median)  greedy/temp0  single-stream"
echo "=================================================================="

# ------------------------------------------------------------------ #
# 1. llama.cpp llama-bench — RAW pp512 / tg128 (no-spec, kernel-fair)  #
#    llama-bench does its own warmup + repetitions (-r N) internally.  #
# ------------------------------------------------------------------ #
run_llama_bench() {  # $1=label  $2=model
  echo "### llama.cpp $1 (pp$P / tg$NGEN, -r $N) ###"
  if [ ! -f "$2" ]; then echo "  SKIP: model missing: $2"; return; fi
  "$LB" -m "$2" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p "$P" -n "$NGEN" -r "$N" 2>/dev/null \
    | grep -E "pp$P|tg$NGEN" || echo "  llama-bench failed"
}
run_llama_bench "9B NVFP4"  "$M9_NVFP4"
run_llama_bench "9B Q8_0"   "$M9_Q8"
run_llama_bench "27B NVFP4" "$M27_NVFP4"

# ------------------------------------------------------------------ #
# 2. llama.cpp llama-server + MTP — PEAK 27B decode (spec-decode on)   #
#    llama-bench CANNOT do MTP; only the server shows the spec uplift. #
#    NOTE: the 27B "-mtp.gguf" has NO embedded NextN -> external -md.  #
# ------------------------------------------------------------------ #
echo "### llama.cpp 27B NVFP4 + MTP (server, PEAK decode) ###"
if [ -f "$M27_NVFP4" ] && [ -f "$M27_MTP_DRAFT" ]; then
  echo "  Run manually (server is long-lived; release GPU after):"
  echo "    $LS -m $M27_NVFP4 \\"
  echo "      -md $M27_MTP_DRAFT \\"
  echo "      -ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 65536 --parallel 1 \\"
  echo "      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.2 \\"
  echo "      --jinja --temp 0 --host 127.0.0.1 --port 8099"
  echo "  Then drive $N greedy completions of $NGEN tokens on a $P-token prompt; read the"
  echo "  server's 'eval time ... tokens per second' (decode) line; take the median of $N."
  echo "  MEASURED reference on this box: 66.6 tok/s (MTP) vs 42.1 raw, accept 0.762."
else
  echo "  SKIP: 27B NVFP4 model or external MTP draft missing."
fi

# ------------------------------------------------------------------ #
# 3. bw24 run-gen — decode tok/s (no-spec; prefill timing N/A yet)     #
#    1 warmup discarded, then N timed; median printed.                #
#    Prompt = fixed P token ids of the SAME content for both models.  #
#    NOTE: run_gen.rs times DECODE only (loops decode_step over the    #
#    prompt to warm the cache, then times NGEN greedy steps). It does  #
#    NOT print a batched-prefill tok/s — reported N/A per protocol 5.3.#
# ------------------------------------------------------------------ #
# Fixed 512-id prompt (repeat a deterministic id pattern to length P).
PROMPT_IDS=$(awk -v n="$P" 'BEGIN{for(i=0;i<n;i++){printf "%d ", 100+(i*7)%900}}')

run_bw24() {  # $1=label  $2=model
  echo "### bw24 $1 decode (warmup+$N timed, median; prefill=N/A) ###"
  if [ ! -f "$2" ]; then echo "  SKIP: model missing: $2"; return; fi
  # warmup (discarded)
  BW24_NGEN="$NGEN" cargo run -q --release -p bw24-engine --bin run-gen -- "$2" $PROMPT_IDS >/dev/null 2>&1 || \
    { echo "  bw24 run-gen failed (model may use a dtype not yet validated — see ROADMAP debt)"; return; }
  for i in $(seq 1 "$N"); do
    BW24_NGEN="$NGEN" cargo run -q --release -p bw24-engine --bin run-gen -- "$2" $PROMPT_IDS 2>/dev/null \
      | grep -oE "[0-9.]+ tok/s" | grep -oE "[0-9.]+" | head -1
  done | median | awk '{print "  bw24 decode median: " $1 " tok/s (no-spec, native KV)"}'
}
# 9B Q8_0 is the validated daily path today (NVFP4 dtype gate is ROADMAP debt).
run_bw24 "9B Q8_0" "$M9_Q8"
# Uncomment once the NVFP4 dequant validation gate passes (ROADMAP next workflow):
# run_bw24 "9B NVFP4"  "$M9_NVFP4"
# run_bw24 "27B NVFP4" "$M27_NVFP4"

# ------------------------------------------------------------------ #
# 4. vLLM + 5. SGLang — RUN SERIALLY (printed; not auto-run: init race)#
# ------------------------------------------------------------------ #
cat <<EOF
### vLLM 0.23.0 — run SEPARATELY, GPU idle (serial). See COMPETITOR-SETUP.md section 2. ###
  source $VLLM_VENV/bin/activate
  # 27B PEAK (NVFP4 modelopt + FP8 KV + FlashInfer + MTP n=3):
  VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \\
    --quantization modelopt --language-model-only --trust-remote-code \\
    --tensor-parallel-size 1 --max-model-len 65536 --max-num-seqs 1 --max-num-batched-tokens 8192 \\
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 --mamba-cache-mode align \\
    --max-cudagraph-capture-size 32 --enable-prefix-caching --enable-chunked-prefill \\
    --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
  # Bench: P=$P prompt, SamplingParams(max_tokens=$NGEN, temperature=0, ignore_eos=True), N=$N median.
  # Report BOTH: no-spec (drop --speculative-config) AND MTP-on decode tok/s.

### SGLang 0.5.9 — run SEPARATELY, GPU idle (serial). See COMPETITOR-SETUP.md section 3. ###
  source $SGLANG_VENV/bin/activate
  # 27B (NVFP4 modelopt_fp4 — FP8 27B does NOT fit 24GB), triton backend (hybrid-GDN), NEXTN MTP:
  SGLANG_USE_CUTEDSL_GDN_DECODE=1 python -m sglang.launch_server \\
    --model-path <Qwen3.6-27B NVFP4 modelopt_fp4 HF repo> --quantization modelopt_fp4 \\
    --tp-size 1 --attention-backend triton --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.80 \\
    --context-length 32768 --cuda-graph-max-bs 4 --chunked-prefill-size 2048 --reasoning-parser qwen3 \\
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \\
    --speculative-num-draft-tokens 4 --max-running-requests 1 --host 127.0.0.1 --port 30000
  # Bench: python -m sglang.bench_one_batch_server --batch-size 1 --input-len $P --output-len $NGEN
  #        --attention-backend triton  (and once without NEXTN for the no-spec number), N=$N median.
EOF

echo "=================================================================="
echo " Headline bars to beat (24GB sm_120, single-stream, MEASURED llama.cpp):"
echo "   9B decode 126.6 t/s | 27B decode no-spec 42.1 / peak(MTP) 66.6 t/s"
echo "   9B prefill 6220 pp$P | 27B prefill 1980 pp$P"
echo " bw24 now: 9B Q8_0 decode 59.6 t/s (no-spec). Ranking table in COMPETITOR-SETUP.md section 6."
echo "=================================================================="
