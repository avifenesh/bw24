#!/bin/bash
# h100-v064-verify extension: (a) board2048 decode/prefill reps 4-7 with ARM ORDER FLIPPED
# (iqfast0 first — kills the order-bias hypothesis on the rep1-3 naked-first 2.2% decode
# median delta), (b) a second decode shape (p2-code-medium) clean of the pp512 near-tie.
# Same jsonl, same parsers as run-iqfast-h100.sh. Every GPU run under flock /tmp/gpu-h100.lock.
set -u
W=/home/ubuntu/memra
R=$W/research/h100-v064-verify-20260802
Q35=/home/ubuntu/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
OUT=$R/iqfast-sweep.jsonl
GIT_SHA=1576d8b3
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

busy_procs() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true
}
row() { # cell arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"%s","arm":"%s","metric":"%s","value":%s,"rep":%s,"temp_c":%s,"busy_procs":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" \
    "$(busy_procs)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}
run_arm() { # cell promptfile arm rep
  local cell=$1 pf=$2 arm=$3 rep=$4 log="$R/$1-$3-rep$4.log"
  local -a env_extra=()
  case "$arm" in
    iqfast0) env_extra=(MEMRA_IQ_FAST=0) ;;
    naked)   ;;
  esac
  env "${env_extra[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$pf" \
    flock /tmp/gpu-h100.lock timeout 1200 "$W/target/release/run-gen" "$Q35" > "$log" 2>&1
  local rc=$?
  local pp tg match mism thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "MATCH" "$log")
  mism=$(grep -cE "MISMATCH|FAIL|panic" "$log")
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  row "$cell" "$arm" prefill_toks "${pp:-null}" "$rep"
  row "$cell" "$arm" decode_toks "${tg:-null}" "$rep"
  row "$cell" "$arm" match_lines "${match:-0}" "$rep"
  row "$cell" "$arm" mismatch_lines "${mism:-0}" "$rep"
  row "$cell" "$arm" token_sha "\"$thash\"" "$rep"
  [ $rc -ne 0 ] && echo "  [$cell/$arm rep$rep] NONZERO EXIT rc=$rc — see $log"
}

for rep in 4 5 6 7; do
  echo "=== ext rep $rep (iqfast0 FIRST) ==="
  run_arm q35-board2048 "$W/research/e2e/prompts/board-2048.txt" iqfast0 "$rep"
  run_arm q35-board2048 "$W/research/e2e/prompts/board-2048.txt" naked   "$rep"
done
for rep in 1 2 3; do
  echo "=== p2 rep $rep ==="
  if [ $((rep % 2)) -eq 1 ]; then
    run_arm q35-p2 "$W/research/e2e/prompts/p2-code-medium.txt" naked   "$rep"
    run_arm q35-p2 "$W/research/e2e/prompts/p2-code-medium.txt" iqfast0 "$rep"
  else
    run_arm q35-p2 "$W/research/e2e/prompts/p2-code-medium.txt" iqfast0 "$rep"
    run_arm q35-p2 "$W/research/e2e/prompts/p2-code-medium.txt" naked   "$rep"
  fi
done

echo "=== combined medians + bit-identity ==="
python3 - "$OUT" <<'EOF'
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1])]
cells = {}
shas = {}
for r in rows:
    k = (r["cell"], r["arm"])
    if r["metric"] in ("prefill_toks", "decode_toks") and r["value"] is not None:
        cells.setdefault((r["cell"], r["arm"], r["metric"]), []).append(float(r["value"]))
    if r["metric"] == "token_sha":
        shas.setdefault(k, []).append(r["value"])
for (c, a, m), v in sorted(cells.items()):
    print(f"{c:14s} {a:8s} {m:13s} N={len(v)} median={statistics.median(v):.1f} range=[{min(v):.1f},{max(v):.1f}]")
for c in ("q35-pp512", "q35-board2048", "q35-p2"):
    sn = shas.get((c, "naked"), []); s0 = shas.get((c, "iqfast0"), [])
    ident = len(set(sn)) == 1 and set(sn) == set(s0)
    print(f"{c}: naked={sn} iqfast0={s0} -> {'BIT-IDENTICAL' if ident else 'DIVERGED'}")
EOF
echo IQFAST-EXT-DONE
