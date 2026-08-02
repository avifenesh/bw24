#!/usr/bin/env bash
# serve-tools gates (2026-08-02), RTX 5090. Three flock holds, one server boot each:
#   hold 1: BASELINE binary (restructure/public-split build) + q35 -> greedy c1 refs + c1-vs-c16
#   hold 2: NEW binary + q35 -> c1-vs-c16 (fresh refs) + c16-vs-BASELINE-refs (cross-binary
#           isolation) + the tools round-trip battery (roundtrip_gate.py)
#   hold 3: NEW binary + AgentWorld -> tools round-trip battery
# Laws: literals only (workflow-args law); server killed by PID, never pkill (co-resident
# llama-server must survive); every GPU run inside `flock /tmp/gpu5090.lock`.
set -uo pipefail
cd /home/avifenesh/projects/wt-serve-tools
OUT=research/serve-tools-20260802
PORT=8127
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
AW=/data/ai-ml/hf-models/agentworld-35b-gguf/Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf
TOKCHECK=/home/avifenesh/projects/wt-serve-tools/target/release/tok-check

boot() { # $1=binary $2=models-spec $3=log
  MEMRA_ADDR="127.0.0.1:${PORT}" MEMRA_MODELS="$2" "$1" > "$3" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 240); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then return 0; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "server died early: see $3"; return 1; fi
  done
  echo "server never became healthy: see $3"; return 1
}
stop() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; sleep 3; }

hold1() {
  echo "=== hold 1: baseline binary, q35 refs + c1-vs-c16 ==="
  boot /tmp/memra-server-baseline "q35=${Q35}" "${OUT}/server-baseline-q35.log" || return 1
  python3 tools/check-batch-exact.py --base "http://127.0.0.1:${PORT}" --model q35 \
    --n 16 --max-tokens 96 --label baseline-q35 \
    --out "${OUT}/greedy-hash.jsonl" --ref "${OUT}/greedy-refs-baseline-q35.json" \
    2>&1 | tee "${OUT}/greedy-hash-baseline-q35.log"
  stop
}

hold2() {
  echo "=== hold 2: new binary, q35 — isolation + tools battery ==="
  boot ./target/release/memra-server "q35=${Q35}" "${OUT}/server-new-q35.log" || return 1
  python3 tools/check-batch-exact.py --base "http://127.0.0.1:${PORT}" --model q35 \
    --n 16 --max-tokens 96 --label new-q35-c16 \
    --out "${OUT}/greedy-hash.jsonl" --ref "${OUT}/greedy-refs-new-q35.json" \
    2>&1 | tee "${OUT}/greedy-hash-new-q35.log"
  python3 tools/check-batch-exact.py --base "http://127.0.0.1:${PORT}" --model q35 \
    --n 16 --max-tokens 96 --label new-q35-vs-baseline-refs \
    --out "${OUT}/greedy-hash.jsonl" --ref "${OUT}/greedy-refs-baseline-q35.json" \
    2>&1 | tee "${OUT}/greedy-hash-new-q35-vs-baseline.log"
  python3 - <<'EOF'
import json
a = json.load(open("research/serve-tools-20260802/greedy-refs-baseline-q35.json"))
b = json.load(open("research/serve-tools-20260802/greedy-refs-new-q35.json"))
same = a == b
print(json.dumps({"gate": "cross-binary-c1-refs", "identical": same,
                  "n": len(a), "verdict": "PASS" if same else "FAIL"}))
open("research/serve-tools-20260802/cross-binary-refs-verdict.json", "w").write(
    json.dumps({"identical": same, "n": len(a)}))
EOF
  python3 "${OUT}/roundtrip_gate.py" --base "http://127.0.0.1:${PORT}" --model q35 \
    --gguf "${Q35}" --tok-check "${TOKCHECK}" --out "${OUT}" --tag q35 \
    2>&1 | tee "${OUT}/roundtrip-q35-console.log"
  stop
}

hold3() {
  echo "=== hold 3: new binary, AgentWorld — tools battery ==="
  boot ./target/release/memra-server "aw=${AW}" "${OUT}/server-new-aw.log" || return 1
  python3 "${OUT}/roundtrip_gate.py" --base "http://127.0.0.1:${PORT}" --model aw \
    --gguf "${AW}" --tok-check "${TOKCHECK}" --out "${OUT}" --tag aw \
    2>&1 | tee "${OUT}/roundtrip-aw-console.log"
  stop
}

case "${1:-all}" in
  hold1) flock /tmp/gpu5090.lock bash -c "$(declare -f boot stop hold1); cd /home/avifenesh/projects/wt-serve-tools; OUT='${OUT}'; PORT=${PORT}; Q35='${Q35}'; hold1" ;;
  hold2) flock /tmp/gpu5090.lock bash -c "$(declare -f boot stop hold2); cd /home/avifenesh/projects/wt-serve-tools; OUT='${OUT}'; PORT=${PORT}; Q35='${Q35}'; TOKCHECK='${TOKCHECK}'; hold2" ;;
  hold3) flock /tmp/gpu5090.lock bash -c "$(declare -f boot stop hold3); cd /home/avifenesh/projects/wt-serve-tools; OUT='${OUT}'; PORT=${PORT}; AW='${AW}'; TOKCHECK='${TOKCHECK}'; hold3" ;;
  all) "$0" hold1 && "$0" hold2 && "$0" hold3 ;;
esac
