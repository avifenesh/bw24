#!/usr/bin/env bash
# Thin launcher — logic in acceptance_battery.shk (shuck).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BW24_ROOT="$(cd "$ROOT/.." && pwd)"
export BW24_TOOLS="$ROOT"
SHUCK="${SHUCK:-$HOME/.local/bin/shuck}"
[[ -x "$SHUCK" ]] || SHUCK="$(command -v shuck || true)"
[[ -n "$SHUCK" && -x "$SHUCK" ]] || { echo "shuck not found" >&2; exit 127; }
exec "$SHUCK" "$ROOT/acceptance_battery.shk" "$@"
