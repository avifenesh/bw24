#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODE_VERSION="22.23.2"
NODE_ARCHIVE="node-v${NODE_VERSION}-linux-x64.tar.xz"
NODE_SHA256="d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307"
NODE_HOME="/opt/node-v${NODE_VERSION}-linux-x64"
BRIDGE_REPO="https://github.com/Belarrius1/belarrius-ai-horde-bridge.git"
BRIDGE_COMMIT="36659f95cb9cbe3caf847a36ecbb41d08fb913f7"
BRIDGE_DIR="/opt/memra-horde-bridge"
UNIT_NAME="memra-horde-worker.service"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}"
ENV_PATH="/etc/memra/horde-worker.env"

mode="install"
dry_run=0
tmp_dir=""

usage() {
    cat <<'EOF'
Usage: setup.sh [--dry-run | --check]

Install the pinned Node runtime, pinned AI Horde bridge checkout, dependencies,
and memra-horde-worker.service. The service is enabled but not started.

  --dry-run  Validate repo-owned configuration and print actions only.
  --check    Verify the installed files and pins without network access.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "$tmp_dir" && -d "$tmp_dir" ]]; then
        rm -r -- "$tmp_dir"
    fi
}

trap cleanup EXIT

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    if ((dry_run)); then
        print_command "$@"
        return 0
    fi
    "$@"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

verify_repo_config() {
    python3 "$SCRIPT_DIR/configure.py" --dry-run
}

verify_install() {
    [[ -x "${NODE_HOME}/bin/node" ]] || die "pinned Node is missing"
    [[ "$("${NODE_HOME}/bin/node" --version)" == "v${NODE_VERSION}" ]] ||
        die "pinned Node version mismatch"
    [[ -d "${BRIDGE_DIR}/.git" ]] || die "bridge checkout is missing"
    [[ "$(git -C "$BRIDGE_DIR" rev-parse HEAD)" == "$BRIDGE_COMMIT" ]] ||
        die "bridge commit mismatch"
    [[ -d "${BRIDGE_DIR}/node_modules/axios" ]] ||
        die "bridge npm dependencies are missing"
    [[ -f "$ENV_PATH" ]] || die "worker environment file is missing"
    local env_mode
    env_mode="$(stat -c '%a' "$ENV_PATH")"
    ((((8#$env_mode) & 077) == 0)) ||
        die "worker environment file must not be group/world-readable"
    cmp --silent "$SCRIPT_DIR/$UNIT_NAME" "$UNIT_PATH" ||
        die "installed systemd unit differs from the repo copy"
    printf 'CHECK PASS: pinned Horde worker installation is complete\n'
}

install_node() {
    if [[ -x "${NODE_HOME}/bin/node" ]]; then
        [[ "$("${NODE_HOME}/bin/node" --version)" == "v${NODE_VERSION}" ]] ||
            die "unexpected binary already exists under ${NODE_HOME}"
        printf 'Node %s already installed\n' "$NODE_VERSION"
        return 0
    fi

    tmp_dir="$(mktemp -d)"
    local archive="${tmp_dir}/${NODE_ARCHIVE}"
    run curl --fail --location --silent --show-error \
        --output "$archive" \
        "https://nodejs.org/dist/v${NODE_VERSION}/${NODE_ARCHIVE}"
    if ((dry_run)); then
        printf '+ verify sha256 %s  %s\n' "$NODE_SHA256" "$archive"
    else
        printf '%s  %s\n' "$NODE_SHA256" "$archive" | sha256sum --check -
    fi
    run tar --extract --xz --file "$archive" --directory /opt
}

install_bridge() {
    if [[ -e "$BRIDGE_DIR" ]]; then
        [[ -d "${BRIDGE_DIR}/.git" ]] ||
            die "${BRIDGE_DIR} exists but is not the expected git checkout"
        [[ "$(git -C "$BRIDGE_DIR" remote get-url origin)" == "$BRIDGE_REPO" ]] ||
            die "unexpected bridge origin under ${BRIDGE_DIR}"
        [[ -z "$(git -C "$BRIDGE_DIR" status --short --untracked-files=no)" ]] ||
            die "bridge checkout has local tracked changes"
        if [[ "$(git -C "$BRIDGE_DIR" rev-parse HEAD)" != "$BRIDGE_COMMIT" ]]; then
            run git -C "$BRIDGE_DIR" fetch --depth 1 origin "$BRIDGE_COMMIT"
            run git -C "$BRIDGE_DIR" checkout --detach "$BRIDGE_COMMIT"
        fi
    else
        run git clone --filter=blob:none "$BRIDGE_REPO" "$BRIDGE_DIR"
        run git -C "$BRIDGE_DIR" checkout --detach "$BRIDGE_COMMIT"
    fi

    run env "PATH=${NODE_HOME}/bin:/usr/bin:/bin" \
        "${NODE_HOME}/bin/npm" ci --omit=dev --prefix "$BRIDGE_DIR"
}

install_service() {
    if ((dry_run)); then
        printf '+ verify provisioner-created memra user\n'
    else
        getent passwd memra >/dev/null ||
            die "memra user is missing; run the RunPod provisioner first"
    fi
    run install -d -m 0755 /etc/memra
    run install -d -o memra -g memra -m 0700 /var/lib/memra-horde-worker
    if [[ ! -e "$ENV_PATH" ]]; then
        run install -m 0600 "$SCRIPT_DIR/horde-worker.env.example" "$ENV_PATH"
    else
        printf 'Preserving existing %s\n' "$ENV_PATH"
    fi
    run install -m 0644 "$SCRIPT_DIR/$UNIT_NAME" "$UNIT_PATH"
    run systemctl daemon-reload
    run systemctl enable "$UNIT_NAME"
}

while (($#)); do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --check)
            mode="check"
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

((dry_run == 0)) || [[ "$mode" == "install" ]] ||
    die "--dry-run and --check are mutually exclusive"

require_command python3
verify_repo_config

if [[ "$mode" == "check" ]]; then
    require_command git
    require_command stat
    require_command cmp
    verify_install
    exit 0
fi

if ((dry_run == 0)); then
    [[ "$EUID" -eq 0 ]] || die "installation must run as root"
fi

run apt-get update
run apt-get install --yes ca-certificates curl git xz-utils

require_command curl
require_command git
require_command sha256sum
require_command tar
require_command systemctl

install_node
install_bridge
install_service

if ((dry_run)); then
    printf 'DRY RUN PASS: install plan validated; no files, packages, or services changed\n'
else
    verify_install
    printf 'Setup complete. Configure %s, then start with trial-up.sh.\n' "$ENV_PATH"
fi
