#!/usr/bin/env bash
# Draft release notes from conventional-commit history: [FROM_TAG] [TO_REF]
# Uses shuck (tools/changelog.shk) when available; otherwise the bash fallback below.
# CI runners don't carry shuck — v0.34.3's release run died on exit 127 here — so the
# fallback is what actually publishes releases. Keep the two implementations in sync.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BW24_ROOT="$(cd "$ROOT/.." && pwd)"
export BW24_TOOLS="$ROOT"
SHUCK="${SHUCK:-$HOME/.local/bin/shuck}"
[[ -x "$SHUCK" ]] || SHUCK="$(command -v shuck || true)"
if [[ -n "$SHUCK" && -x "$SHUCK" ]]; then
    exec "$SHUCK" "$ROOT/changelog.shk" "$@"
fi

# ---- bash fallback (mirrors changelog.shk) ----
FROM="${1:-}"
TO="${2:-HEAD}"
if [[ -z "$FROM" ]]; then
    FROM="$(git describe --tags --abbrev=0 "$TO^" 2>/dev/null \
        || git rev-list --max-parents=0 HEAD)"
fi

section() {
    local title=$1 prefix=$2 body
    body="$(git log --no-merges --format='- %s' "$FROM..$TO" \
        | grep -E "^- $prefix(\([^)]*\))?!?: " \
        | sed -E "s/^- $prefix(\([^)]*\))?!?: /- /" || true)"
    if [[ -n "$body" ]]; then
        printf '## %s\n%s\n\n' "$title" "$body"
    fi
}

echo "Changes since $FROM:"
echo ""
section "Performance" perf
section "Features" feat
section "Fixes" fix
section "Configuration" config
section "Documentation" docs

other="$(git log --no-merges --format='- %s' "$FROM..$TO" \
    | grep -vE '^- (perf|feat|fix|config|docs|data|chore|wip|probe)(\([^)]*\))?!?:' || true)"
if [[ -n "$other" ]]; then
    printf '## Other\n%s\n\n' "$other"
fi

echo "Boards + reproduction artifacts: https://huggingface.co/Avifenesh/bw24-bench · full experiment log in research/tune-data/"
