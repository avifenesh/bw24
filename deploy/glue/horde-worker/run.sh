#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURE="${SCRIPT_DIR}/configure.py"
BRIDGE_DIR="${MEMRA_HORDE_BRIDGE_DIR:-/opt/memra-horde-bridge}"
BRIDGE_COMMIT="36659f95cb9cbe3caf847a36ecbb41d08fb913f7"
NODE_HOME="${MEMRA_HORDE_NODE_HOME:-/opt/node-v22.23.2-linux-x64}"
STATE_DIR="${MEMRA_HORDE_STATE_DIR:-/var/lib/memra-horde-worker}"
CONFIG_PATH="${STATE_DIR}/config.yaml"
ENV_FILE="${MEMRA_HORDE_ENV_FILE:-/etc/memra/horde-worker.env}"

mode="run"
mode_selected=0

usage() {
    cat <<'EOF'
Usage: run.sh [--dry-run | --check] [--env-file PATH]

Run the pinned AI Horde bridge against memra.

  --dry-run  Validate the template and conservative defaults with no network.
             Missing installation paths and secrets are allowed.
  --check    Validate the installed bridge, configured secrets, and rendered
             config without contacting memra or AI Horde.
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
        [[ "$key" == "AI_HORDE_API_KEY" || "$key" == MEMRA_HORDE_* ]] ||
            die "unexpected variable in $path: $key"
        if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi
        export "$key=$value"
    done <"$path"
}

verify_install() {
    [[ -x "${NODE_HOME}/bin/node" ]] || die "pinned Node is not installed: ${NODE_HOME}"
    [[ "$("${NODE_HOME}/bin/node" --version)" == "v22.23.2" ]] ||
        die "unexpected Node version under ${NODE_HOME}"
    [[ -f "${BRIDGE_DIR}/index.js" ]] || die "bridge is not installed: ${BRIDGE_DIR}"
    [[ -d "${BRIDGE_DIR}/node_modules/axios" ]] ||
        die "bridge dependencies are not installed: run setup.sh"
    [[ -d "${BRIDGE_DIR}/.git" ]] || die "bridge checkout metadata is missing"

    local installed_commit
    installed_commit="$(git -C "$BRIDGE_DIR" rev-parse HEAD)"
    [[ "$installed_commit" == "$BRIDGE_COMMIT" ]] ||
        die "bridge commit mismatch: expected ${BRIDGE_COMMIT}, found ${installed_commit}"
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
    "$CONFIGURE" --dry-run
    if [[ -x "${NODE_HOME}/bin/node" && -f "${BRIDGE_DIR}/index.js" ]]; then
        verify_install
        printf 'PASS: pinned bridge installation is present\n'
    else
        printf 'INFO: bridge installation is absent; setup.sh will install it on the pod\n'
    fi
    printf 'DRY RUN PASS: no backend or AI Horde connection was attempted\n'
    exit 0
fi

if [[ -z "${AI_HORDE_API_KEY:-}" || -z "${MEMRA_HORDE_BACKEND_KEY:-}" ]]; then
    load_env_file "$ENV_FILE"
fi

if [[ "$mode" == "check" ]]; then
    verify_install
    "$CONFIGURE" --validate
    if [[ -f "$CONFIG_PATH" ]]; then
        "$CONFIGURE" --check "$CONFIG_PATH"
    else
        printf 'INFO: rendered config is absent; the service will create %s at first start\n' \
            "$CONFIG_PATH"
    fi
    printf 'CHECK PASS: installation and config are valid; no network connection was attempted\n'
    exit 0
fi

verify_install
mkdir -p "$STATE_DIR"
"$CONFIGURE" --output "$CONFIG_PATH"
cd "$STATE_DIR"
exec "${NODE_HOME}/bin/node" "${BRIDGE_DIR}/index.js"
