#!/bin/bash
# Draft release notes from conventional-commit history: changelog.sh [FROM_TAG] [TO_REF]
# Defaults: FROM = previous tag, TO = HEAD. Groups by prefix; data()/chore() and merge
# commits are dropped (research log rows and plumbing are not user-facing changes).
set -euo pipefail
TO=${2:-HEAD}
FROM=${1:-$(git describe --tags --abbrev=0 "$TO"^ 2>/dev/null || git rev-list --max-parents=0 HEAD | tail -1)}

section() { # section <title> <grep-prefix-regex>
  local body
  body=$(git log --no-merges --format='- %s' "$FROM..$TO" | grep -E "^- $2" \
         | sed -E "s/^- $2(\([^)]*\))?!?: /- /" || true)
  [ -n "$body" ] && printf '## %s\n%s\n\n' "$1" "$body"
  return 0   # an empty section is not an error (set -e)
}

echo "Changes since ${FROM}:"
echo
section "Performance"    "perf"
section "Features"       "feat"
section "Fixes"          "fix"
section "Configuration"  "config"
section "Documentation"  "docs"
# anything not matching a known prefix (and not data/chore) lands under Other
other=$(git log --no-merges --format='- %s' "$FROM..$TO" \
        | grep -vE '^- (perf|feat|fix|config|docs|data|chore|wip|probe)(\([^)]*\))?!?:' || true)
[ -n "$other" ] && printf '## Other\n%s\n\n' "$other" || true

echo "Boards + reproduction artifacts: https://huggingface.co/Avifenesh/bw24-bench · full experiment log in research/tune-data/"
