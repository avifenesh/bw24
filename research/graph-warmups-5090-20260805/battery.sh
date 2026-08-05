#!/bin/bash
# Post-flip battery (MEMRA_GRAPH_WARMUPS default now 1): kernel-check, graph-decode +
# graph-session gates both models, run-gen argmax both models, run-spec K=1..8 (q27 arm),
# serve-smoke. NAKED env everywhere — the default under test is the shipped one.
# Every GPU run under flock /tmp/gpu5090.lock (short holds). tee-first, parse-second.
set -uo pipefail
cd /home/avifenesh/projects/wt-warmups
D=research/graph-warmups-5090-20260805/logs
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
L() { flock -w 10800 /tmp/gpu5090.lock "$@"; }
FAILS=0
echo "battery start $(date -Is) commit $(git rev-parse HEAD)"

L target/release/kernel-check > "$D/gate-kernelcheck.txt" 2>&1
tail -1 "$D/gate-kernelcheck.txt" | grep -q "ALL GREEN" \
  && echo "kernel-check: GREEN" || { echo "kernel-check: FAIL"; FAILS=$((FAILS+1)); }

for M in "$Q9" "$Q27"; do
  n=$(basename "$M" .gguf)
  L target/release/graph-decode-gate "$M" > "$D/gate-graphdecode-$n.txt" 2>&1
  grep -q "Phase-3 gate PASS" "$D/gate-graphdecode-$n.txt" \
    && echo "graph-decode-gate $n: PASS" || { echo "graph-decode-gate $n: FAIL"; FAILS=$((FAILS+1)); }
  L target/release/graph-session-gate "$M" > "$D/gate-graphsession-$n.txt" 2>&1
  grep -q "ALL GREEN" "$D/gate-graphsession-$n.txt" \
    && echo "graph-session-gate $n: PASS" || { echo "graph-session-gate $n: FAIL"; FAILS=$((FAILS+1)); }
  L env MEMRA_NGEN=20 MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
    target/release/run-gen "$M" > "$D/gate-rungen-$n.txt" 2>&1
  grep -q "MATCH" "$D/gate-rungen-$n.txt" && ! grep -q "MISMATCH-STRUCTURED" "$D/gate-rungen-$n.txt" \
    && echo "run-gen argmax $n: MATCH" || { echo "run-gen argmax $n: FAIL"; FAILS=$((FAILS+1)); }
done

# run-spec K=1..8 (one arm: q27 NVFP4+MTP — naked = the full sweep)
L target/release/run-spec "$Q27" > "$D/gate-runspec-q27.txt" 2>&1
grep -q "SELF-CONSISTENCY PASS" "$D/gate-runspec-q27.txt" \
  && echo "run-spec K=1..8 q27: PASS" || { echo "run-spec q27: FAIL"; FAILS=$((FAILS+1)); }

# serve-smoke (holds the lock for the full battery — the server binds a port; still the
# shortest correct hold)
L tools/serve-smoke.sh > "$D/gate-servesmoke.txt" 2>&1
if grep -qE "0 failed|failed: 0" "$D/gate-servesmoke.txt" || ! grep -q "FAIL" "$D/gate-servesmoke.txt"; then
  echo "serve-smoke: PASS ($(grep -cE '^  ok:' "$D/gate-servesmoke.txt") ok)"
else
  echo "serve-smoke: FAIL"; FAILS=$((FAILS+1))
fi

echo "battery end $(date -Is): $FAILS fail(s)"
exit "$FAILS"
