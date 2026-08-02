#!/bin/bash
# q35-spec-repair: re-measure the board q35 speculative row under the new sm_120a naked
# default (AUTO-KQUANT f16g mode 3, merged cf8a9358) with the EXACT published-row protocol.
#
# Row provenance (research/tune-data/rig5090.jsonl:345, 2026-07-18 REGIME ROLLOUT +
# research/tune-data/rebaseline2-2026-07-09.md protocol):
#   memra arm: run-spec, MEMRA_MTP_DRAFT=draft-35b-owntrim-nvfp4head-q4blk.gguf,
#     MEMRA_SPEC_K=2 (fixed-K is the qwen default), MEMRA_NGEN=256, naked otherwise.
#     p1 = p1-code-short.txt   greedy temp-0
#     p2 = p2-code-medium.txt  greedy temp-0
#     p3 = p3-agentic-long-v3.txt, MEMRA_CHAT=1 (chat-templated -> 5420 tok),
#          SAMPLED MEMRA_SPEC_TEMP=0.7 MEMRA_SEED=42
#   llama arm (its swept best, 24GB working self-MTP config): llama-server, embedded
#     NextN (NO -md; -md same-file OOMs), --spec-type draft-mtp --spec-draft-n-max 2
#     --spec-draft-p-min 0.1, -ngl 999 -fa on -ctk q8_0 -ctv q5_1, GGML_CUDA_GRAPH_OPT=1.
#     p1 /completion temp 0 ignore_eos=true (raw p1 EOSes at 1 tok — 2026-07-08 row)
#     p2 /completion temp 0
#     p3 /v1/chat/completions temperature 0.7 seed 42 max_tokens 256
#   N=3 per arm per cell, both engines interleaved per rep in one session; idle-gated;
#   every GPU run under flock /tmp/gpu5090.lock; co-resident llama-server --embedding
#   (port 8181, -ngl 0) is allowlisted and untouched.
#
# usage: run-row-repair.sh [nreps]
set -u
N=${1:-3}
W=/home/avifenesh/projects/wt-q35-spec-repair
R=$W/research/q35-spec-repair-20260802
PDIR=$W/research/e2e/prompts
OUT=$R/row-repair.jsonl
MODEL=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
LS=/home/avifenesh/projects/llama.cpp/build/bin/llama-server
PORT=8099
export GGML_CUDA_GRAPH_OPT=1

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
LLAMA_VER=$("$LS" --version 2>&1 | head -1)
gpu-full-power on >/dev/null 2>&1 || true

busy_procs() { # GPU compute apps minus the allowlisted --embedding co-resident
  local n=0 pid
  while IFS=, read -r pid _; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding" || n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  echo $n
}
wait_idle() {
  local n=0 busy clk
  while true; do
    busy=$(busy_procs)
    clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "$busy" -eq 0 ] && [ "${clk:-2000}" -lt 1200 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle: 10min timeout (busy=$busy clk=$clk)"; break; }
  done
}
gputemp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits; }
row() { # arm cell metric value rep extra_json
  printf '{"ts":"%s","git":"%s","llama_build":"%s","arm":"%s","cell":"%s","metric":"%s","value":%s,"rep":%s,"temp_c":%s,"profile":"%s"%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$LLAMA_VER" "$1" "$2" "$3" "$4" "$5" "$(gputemp)" "$PROFILE" "${6:-}" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}

memra_cell() { # cell promptfile rep extra_envs...
  local cell=$1 pf=$2 rep=$3; shift 3
  local log="$R/memra-$cell-rep$rep.log"
  wait_idle
  env "$@" MEMRA_MTP_DRAFT="$DRAFT" MEMRA_SPEC_K=2 MEMRA_SPEC_STATS=1 MEMRA_NGEN=256 \
    MEMRA_PROMPT_FILE="$PDIR/$pf" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-spec" "$MODEL" > "$log" 2>&1
  local gen spec acc cons ntok
  ntok=$(grep -aoE "text prompt .* -> [0-9]+ tokens" "$log" | grep -oE "[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)
  gen=$(grep -aoE "^\[generate\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  spec=$(grep -aoE "\[generate_spec K=2\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  acc=$(grep -aoE "acceptance: [0-9]+/[0-9]+ = [0-9.]+%" "$log" | tail -1 | grep -oE "[0-9.]+%" | tr -d '%')
  cons=$(grep -ac "self-consistency: PASS" "$log")
  row memra "$cell" prompt_ntok "${ntok:-null}" "$rep"
  row memra "$cell" gen_toks "${gen:-null}" "$rep"
  row memra "$cell" spec_k2_toks "${spec:-null}" "$rep"
  row memra "$cell" acceptance_pct "${acc:-null}" "$rep"
  row memra "$cell" self_consistency_pass "${cons:-0}" "$rep"
}

llama_rep() { # rep — one server bring-up, all three classes, lock held across the block
  local rep=$1
  local slog="$R/llama-server-rep$rep.log"
  wait_idle
  exec 9>/tmp/gpu5090.lock
  flock 9
  "$LS" -m "$MODEL" -ngl 999 -fa on -c 16384 --parallel 1 \
    --host 127.0.0.1 --port $PORT -ctk q8_0 -ctv q5_1 \
    --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.1 > "$slog" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 240); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 2
  done
  if [ "$up" -ne 1 ]; then
    echo "LLAMA SERVER FAILED rep$rep (see $slog)"; kill $spid 2>/dev/null
    flock -u 9; exec 9>&-; return 1
  fi
  # p1 greedy ignore_eos / p2 greedy — /completion, raw prompt text, n_predict 256
  local cls pf ieos
  for cls in p1 p2; do
    if [ "$cls" = p1 ]; then pf=p1-code-short.txt; ieos=true; else pf=p2-code-medium.txt; ieos=false; fi
    python3 - "$PDIR/$pf" "$ieos" "$R/llama-$cls-rep$rep.json" <<'PY'
import json, sys, urllib.request
pf, ieos, out = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
req = urllib.request.Request('http://127.0.0.1:8099/completion',
  data=json.dumps({'prompt': open(pf).read(), 'n_predict': 256, 'temperature': 0,
                   'cache_prompt': False, 'ignore_eos': ieos}).encode(),
  headers={'Content-Type': 'application/json'})
r = json.loads(urllib.request.urlopen(req, timeout=900).read())
json.dump(r, open(out, 'w'))
t = r['timings']
acc = t.get('draft_n_accepted', 0) / max(1, t.get('draft_n', 0) or 1)
print(f"{t['predicted_per_second']:.2f} {t.get('predicted_n',0)} {t.get('draft_n',0)} {acc:.3f}")
PY
  done > "$R/llama-completion-rep$rep.out" 2>&1
  # p3 sampled — /v1/chat/completions, raw v3 text as the user turn (server-side template),
  # temperature 0.7 seed 42 max_tokens 256 (rebaseline2 protocol line)
  python3 - "$PDIR/p3-agentic-long-v3.txt" "$R/llama-p3-rep$rep.json" <<'PY' >> "$R/llama-completion-rep$rep.out" 2>&1
import json, sys, urllib.request
pf, out = sys.argv[1], sys.argv[2]
req = urllib.request.Request('http://127.0.0.1:8099/v1/chat/completions',
  data=json.dumps({'messages': [{'role': 'user', 'content': open(pf).read()}],
                   'temperature': 0.7, 'seed': 42, 'max_tokens': 256,
                   'cache_prompt': False}).encode(),
  headers={'Content-Type': 'application/json'})
r = json.loads(urllib.request.urlopen(req, timeout=900).read())
json.dump(r, open(out, 'w'))
t = r.get('timings') or {}
if not t:
    print("NO-TIMINGS-IN-RESPONSE"); sys.exit(0)
acc = t.get('draft_n_accepted', 0) / max(1, t.get('draft_n', 0) or 1)
print(f"{t['predicted_per_second']:.2f} {t.get('predicted_n',0)} {t.get('draft_n',0)} {acc:.3f}")
PY
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  flock -u 9; exec 9>&-
  # emit rows from the captured .out (3 lines: p1, p2, p3)
  local i=0 line
  for cls in p1-code-short p2-code-medium p3-agentic-long; do
    i=$((i+1)); line=$(sed -n "${i}p" "$R/llama-completion-rep$rep.out")
    local tps ntok dn acc
    tps=$(echo "$line" | awk '{print $1}'); ntok=$(echo "$line" | awk '{print $2}')
    dn=$(echo "$line" | awk '{print $3}'); acc=$(echo "$line" | awk '{print $4}')
    case "$tps" in ''|*[!0-9.]*) tps=null;; esac
    row llama "$cls" spec_toks "${tps:-null}" "$rep" ",\"predicted_n\":${ntok:-0},\"draft_n\":${dn:-0},\"draft_accept\":${acc:-0}"
  done
  return 0
}

echo "=== Q35 SPEC ROW REPAIR x$N $TS git=$GIT_SHA profile=$PROFILE llama=[$LLAMA_VER] ===" | tee -a "$R/console.log"
{
  for rep in $(seq 1 "$N"); do
    echo "--- rep $rep: llama arm ---"
    llama_rep "$rep"
    echo "--- rep $rep: memra arm ---"
    memra_cell p1-code-short  p1-code-short.txt  "$rep"
    memra_cell p2-code-medium p2-code-medium.txt "$rep"
    memra_cell p3-agentic-long p3-agentic-long-v3.txt "$rep" MEMRA_CHAT=1 MEMRA_SPEC_TEMP=0.7 MEMRA_SEED=42
  done
  echo "ROW-REPAIR-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/console.log"
