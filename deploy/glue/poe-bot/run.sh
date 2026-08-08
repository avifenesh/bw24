#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MEMRA_POE_VENV:-/opt/memra-poe-venv}"
ENV_FILE="${MEMRA_POE_ENV_FILE:-/etc/memra/poe-bot.env}"
mode="run"
mode_selected=0

usage() {
    cat <<'EOF'
Usage: run.sh [--dry-run | --check] [--env-file PATH]

  --dry-run  Validate defaults with placeholder secrets. No connections.
  --check    Validate the installed environment and real secrets. No connections.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

load_env_file() {
    local path="$1"
    local line key value
    [[ -f "$path" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]] ||
            die "invalid environment line in $path"
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        [[ "$key" == "POE_ACCESS_KEY" || "$key" == MEMRA_POE_* ]] ||
            die "unexpected variable in $path: $key"
        if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi
        export "$key=$value"
    done <"$path"
}

verify_venv() {
    [[ -x "${VENV}/bin/python" ]] || die "Poe virtualenv is missing: $VENV"
    "${VENV}/bin/python" - <<'PY'
from importlib.metadata import version
expected = {
    "fastapi-poe": "0.0.83",
    "httpx": "0.28.1",
    "uvicorn": "0.52.1",
}
for package, wanted in expected.items():
    found = version(package)
    if found != wanted:
        raise SystemExit(f"{package} version mismatch: expected {wanted}, found {found}")
print("PASS: pinned Poe runtime packages are installed")
PY
}

validate_bind() {
    local bind="${MEMRA_POE_BIND:-127.0.0.1}"
    local port="${MEMRA_POE_PORT:-8081}"
    [[ "$bind" == "127.0.0.1" || "$bind" == "0.0.0.0" ]] ||
        die "MEMRA_POE_BIND must be 127.0.0.1 or 0.0.0.0"
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
        die "MEMRA_POE_PORT must be between 1 and 65535"
    fi
}

while (($#)); do
    case "$1" in
        --dry-run)
            ((mode_selected == 0)) || die "choose only one of --dry-run or --check"
            mode="dry-run"
            mode_selected=1
            ;;
        --check)
            ((mode_selected == 0)) || die "choose only one of --dry-run or --check"
            mode="check"
            mode_selected=1
            ;;
        --env-file)
            shift
            (($#)) || die "--env-file requires a path"
            ENV_FILE="$1"
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

if [[ "$mode" == "dry-run" ]]; then
    python3 "$SCRIPT_DIR/poe_config.py" --dry-run
    validate_bind
    if [[ -x "${VENV}/bin/python" ]]; then
        verify_venv
    else
        printf 'INFO: Poe virtualenv is absent; setup.sh will install it on the host\n'
    fi
    printf 'DRY RUN PASS: no Poe or memra connection was attempted\n'
    exit 0
fi

if [[ -z "${POE_ACCESS_KEY:-}" || -z "${MEMRA_POE_BACKEND_KEY:-}" ]]; then
    load_env_file "$ENV_FILE"
fi
validate_bind

if [[ "$mode" == "check" ]]; then
    verify_venv
    "${VENV}/bin/python" "$SCRIPT_DIR/poe_config.py"
    printf 'CHECK PASS: Poe runtime and config are valid; no connection was attempted\n'
    exit 0
fi

verify_venv
cd "$SCRIPT_DIR"
exec "${VENV}/bin/uvicorn" server:app \
    --host "${MEMRA_POE_BIND:-127.0.0.1}" \
    --port "${MEMRA_POE_PORT:-8081}" \
    --no-access-log
