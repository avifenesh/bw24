#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
LOCK=${LOCK:-$ROOT/research/per-expert-quant/layer-balanced-bridge.lock.json}
VALIDATOR=${VALIDATOR:-$ROOT/research/per-expert-quant/validate_layer_balanced_bridge_lock.py}

export ARMS_CSV=layer_balanced120,layer_balanced137
export TARGET_BYTES_CSV=120000000000,137459192320
export WEIGHTS_CSV=0:0:0,0:0:0
export MIP_REL_GAP=1e-6
export SENSITIVITY_COMPLETE=/data/logs/hy3-quant-sensitivity-53de6ca/complete
export SENSITIVITY=/data/calibration/hy3-quant-iq3-iq4-q4-99f3dc3/seven-format-sensitivity.json
export RETENTION=/data/calibration/hy3-100gb-5f02c37/expert-retention-scores.json
export CONFIDENCE=/data/calibration/hy3-100gb-5f02c37/confidence-expert-scores-13a4d92.json
export REFERENCE_PLAN=/data/plans/per-expert-quant-100gb-5f02c37/traffic-nvfp4-53-q2-139-exact100gb.json
export REFERENCE_RECEIPTS=/data/heal/per-expert-quant-iq3-iq4-q4-pareto-6c5c5ea/smart100_iq3_iq4_q4_pareto/receipts
export LAYER_CONSTRAINTS=/data/plans/per-expert-quant-layer-balanced100-3db293f/layer-balanced100.constraints.json
export SOURCE=/opt/dlami/nvme/models/hy3-source
export TRACE_LOCK=/data/calibration/hy3-100gb-5f02c37/moe-inputs.lock.json
export PLAN_ROOT=/data/plans/per-expert-quant-layer-balanced-bridge
export HEAL_ROOT=/data/heal/per-expert-quant-layer-balanced-bridge
export ARTIFACT_ROOT=/data/artifacts/per-expert-quant-layer-balanced-bridge
export SCRATCH_ROOT=/scratch/bw24-artifacts-layer-balanced-bridge
export LOG_ROOT=/data/logs/layer-balanced-bridge-build
export SELECTION_LOCK=$LOCK
export SELECTION_LOCK_VALIDATOR=$VALIDATOR

exec "$ROOT/research/per-expert-quant/run_hy3_smart100_build_transition.sh"
