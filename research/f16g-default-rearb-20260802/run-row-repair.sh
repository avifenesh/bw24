#!/bin/bash
# f16g-default-rearb: board q35 speculative row re-pair under the mode-2 naked default
# (this lane's flip). EXACT published-row protocol from research/q35-spec-repair-20260802/
# (provenance rig5090.jsonl:345 2026-07-18 + rebaseline2-2026-07-09.md), with llama at its
# per-class --spec-draft-n-max optimum (p1@3 / p2@2 / p3@4 — re-swept same-day same-build
# 2026-08-02 by q35-spec-repair, llama-nmax-sweep.out; build unchanged since).
#
#   memra: run-spec, MEMRA_MTP_DRAFT=owntrim drafter, MEMRA_SPEC_K=2, MEMRA_NGEN=256,
#     naked otherwise (naked = the mode-2 flip candidate).
#     p1 p1-code-short.txt greedy / p2 p2-code-medium.txt greedy /
#     p3 p3-agentic-long-v3.txt MEMRA_CHAT=1 SAMPLED MEMRA_SPEC_TEMP=0.7 MEMRA_SEED=42
#   llama: llama-server embedded NextN self-MTP, --spec-type draft-mtp
#     --spec-draft-p-min 0.1, -ngl 999 -fa on -c 16384 --parallel 1 -ctk q8_0 -ctv q5_1,
#     GGML_CUDA_GRAPH_OPT=1; per-class n-max (3/2/4). p1 /completion temp 0 ignore_eos /
#     p2 /completion temp 0 / p3 /v1/chat/completions temp 0.7 seed 42 max_tokens 256.
#   N=3 per arm per cell, engines interleaved per rep in one session; idle-gated; every
#   GPU run under flock /tmp/gpu5090.lock; co-resident llama-server --embedding untouched.
# usage: run-row-repair.sh [nreps]
set -u
N=${1:-3}
W=/home/avifenesh/projects/wt-f16g-rearb
R=$W/research/f16g-default-rearb-20260802
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

busy_procs() {
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
  row memra "$cell" spec_k2_toks "${spec:-null}" "$rep"
  row memra "$cell" acceptance_pct "${acc:-null}" "$rep"
  row memra "$cell" self_consistency_pass "${cons:-0}" "$rep"
}

llama_class() { # rep cls nmax — one server bring-up per (rep, class) at that class's n-max
  local rep=$1 cls=$2 nmax=$3
  local slog="$R/llama-server-$cls-nmax$nmax-rep$rep.log"
  wait_idle
  exec 9>/tmp/gpu5090.lock
  flock 9
  "$LS" -m "$MODEL" -ngl 999 -fa on -c 16384 --parallel 1 \
    --host 127.0.0.1 --port $PORT -ctk q8_0 -ctv q5_1 \
    --spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-p-min 0.1 > "$slog" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 240); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 2
  done
  if [ "$up" -ne 1 ]; then
    echo "LLAMA SERVER FAILED $cls rep$rep (see $slog)"; kill $spid 2>/dev/null
    flock -u 9; exec 9>&-; return 1
  fi
  local line
  if [ "$cls" = p3-agentic-long ]; then
    line=$(python3 - "$PDIR/p3-agentic-long-v3.txt" "$R/llama-p3-rep$rep.json" <<'PY'
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
)
  else
    local pf ieos
    if [ "$cls" = p1-code-short ]; then pf=p1-code-short.txt; ieos=true; else pf=p2-code-medium.txt; ieos=false; fi
    line=$(python3 - "$PDIR/$pf" "$ieos" "$R/llama-${cls%%-*}-rep$rep.json" <<'PY'
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
)
  fi
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  flock -u 9; exec 9>&-
  echo "$line" >> "$R/llama-completion-rep$rep.out"
  local tps ntok dn acc
  tps=$(echo "$line" | awk '{print $1}'); ntok=$(echo "$line" | awk '{print $2}')
  dn=$(echo "$line" | awk '{print $3}'); acc=$(echo "$line" | awk '{print $4}')
  case "$tps" in ''|*[!0-9.]*) tps=null;; esac
  row llama "$cls" spec_toks "${tps:-null}" "$rep" ",\"nmax\":$nmax,\"predicted_n\":${ntok:-0},\"draft_n\":${dn:-0},\"draft_accept\":${acc:-0}"
  return 0
}

echo "=== Q35 SPEC ROW RE-PAIR (mode-2 naked) x$N $TS git=$GIT_SHA profile=$PROFILE llama=[$LLAMA_VER] ===" | tee -a "$R/row-repair-console.log"
{
  for rep in $(seq 1 "$N"); do
    echo "--- rep $rep: llama arm (per-class n-max 3/2/4) ---"
    llama_class "$rep" p1-code-short 3
    llama_class "$rep" p2-code-medium 2
    llama_class "$rep" p3-agentic-long 4
    echo "--- rep $rep: memra arm (naked mode-2 default) ---"
    memra_cell p1-code-short  p1-code-short.txt  "$rep"
    memra_cell p2-code-medium p2-code-medium.txt "$rep"
    memra_cell p3-agentic-long p3-agentic-long-v3.txt "$rep" MEMRA_CHAT=1 MEMRA_SPEC_TEMP=0.7 MEMRA_SEED=42
  done
  echo "ROW-REPAIR-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/row-repair-console.log"
