#!/usr/bin/env bash
# graph-warmup-stress gate (lane/graph-warmups, 2026-08-05): the pool-growth adversarial
# battery the MEMRA_GRAPH_WARMUPS=1 default owes. Two arms:
#
#   1. STRESS (the gate): N cycles of large-session-boot/retire <-> small-session
#      capture-over-freed-blocks, both directions, + the overlap arm (two live graphs,
#      shared engine pools, mid-flight drop + forced recapture). Every stream must be
#      BIT-IDENTICAL to eager decode_step and no CUDA fault may propagate. One reproduced
#      stale-address divergence = the warmups=1 default is REFUTED (keep =2, keep the door).
#   2. CANARY (the teeth proof): same battery with an injected mid-stream clobber of a
#      graph-referenced buffer (token_d) — the gate must CATCH it. A comparator that cannot
#      see injected graph-memory corruption proves nothing about arm 1. (A true cross-
#      allocation alias is not deterministically constructible from user code — the async
#      pool exposes no placement control — so the canary corrupts graph-read memory
#      directly; see graph_warmup_stress.rs header.)
#
# Runs under MEMRA_GRAPH_WARMUPS=1 explicitly (the value under test, independent of the
# shipped default). Usage: tools/graph-warmup-stress-gate.sh [model.gguf [cycles]]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
CYCLES="${2:-10}"
[ -f "$MODEL" ] || { echo "graph-warmup-stress-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x target/release/graph-warmup-stress ] || \
  cargo build --release -p memra-engine --bin graph-warmup-stress || exit 1

echo "== graph-warmup-stress gate: $MODEL cycles=$CYCLES warmups=1 =="
MEMRA_GRAPH_WARMUPS=1 target/release/graph-warmup-stress "$MODEL" --cycles "$CYCLES" \
  || { echo "GATE FAIL: stress battery (warmups=1 stale-address class reproduced or fault)"; exit 1; }

echo "== canary arm: injected graph-memory corruption must be CAUGHT =="
MEMRA_GRAPH_WARMUPS=1 target/release/graph-warmup-stress "$MODEL" --cycles 1 --canary \
  || { echo "GATE FAIL: canary not caught (comparator blind — arm 1 proves nothing)"; exit 1; }

echo "ALL GREEN: graph-warmup-stress gate ($CYCLES cycles + canary)"
