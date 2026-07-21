#!/usr/bin/env bash
# Weekly upstream mining (2026-07-15, owner directive): llama.cpp commits + vLLM/SGLang
# releases since the last sweep, filtered for batch-1 dense-decode mechanisms. Appends a
# dated section to research/upstream-sweeps.md. Two weeks of upstream yielded exactly one
# testable item (PDL-for-FA, probed flat) — the vein is thin but real; the sweep exists so
# it never silently goes stale again (the 06-30 freeze hid llama's E4B-MTP fix and the
# fa=auto behavior for two weeks).
#
# Install (weekly, Monday 09:17):
#   (crontab -l; echo '17 9 * * 1 cd ~/projects/bw24 && tools/upstream-sweep.sh >> /tmp/bw24-upstream-sweep.log 2>&1') | crontab -
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=research/upstream-sweeps.md
STATE=research/.upstream-sweep-since
SINCE=$(cat "$STATE" 2>/dev/null || echo "2026-07-15T00:00:00Z")
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

command -v gh >/dev/null || { echo "upstream-sweep: gh required"; exit 2; }

FILTER='cuda|mmvq|mmq|moe|gemma|qwen|flash|fattn|kv|graph|fusion|fuse|decode|gemv|dp4a|nvfp4|fp4|latency|pdl|batch.?1|bs=1'

{
    echo ""
    echo "## Sweep $NOW (since $SINCE)"
    echo ""
    echo "### llama.cpp commits (decode-relevant, CUDA)"
    for page in 1 2 3; do
        gh api "repos/ggml-org/llama.cpp/commits?since=$SINCE&per_page=100&page=$page" \
            --jq '.[].commit.message | split("\n")[0]' 2>/dev/null
    done | grep -iE "$FILTER" | grep -ivE "opencl|sycl|vulkan|hip|rocm|musa|webgpu|cann" | sort -u \
        | sed 's/^/- /' || echo "- (none)"
    echo ""
    for repo in vllm-project/vllm sgl-project/sglang; do
        echo "### $repo releases"
        gh api "repos/$repo/releases?per_page=5" \
            --jq ".[] | select(.published_at > \"$SINCE\") | \"#### \" + .tag_name + \" (\" + .published_at + \")\n\" + .body" 2>/dev/null \
            | grep -iE "$FILTER|^####" | head -40 | sed 's/^\([^#]\)/- \1/' || echo "- (none)"
        echo ""
    done
    echo "_Review protocol: anything testable gets ported behind a seam + A/B'd per the_"
    echo "_flags doctrine; parity items get a one-line note; the jsonl is the record._"
} >> "$OUT"

echo "$NOW" > "$STATE"
echo "upstream-sweep: appended to $OUT (since $SINCE)"
