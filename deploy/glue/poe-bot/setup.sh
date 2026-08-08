#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="/opt/memra-poe-venv"
ENV_PATH="/etc/memra/poe-bot.env"
UNIT_NAME="memra-poe-bot.service"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}"
mode="install"
dry_run=0

usage() {
    cat <<'EOF'
Usage: setup.sh [--dry-run | --check]

Install the pinned Poe Python environment, dedicated service user, environment
template, and systemd unit. The unit is enabled but not started.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

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

verify_install() {
    [[ -x "${VENV}/bin/python" ]] || die "Poe virtualenv is missing"
    [[ -f "$ENV_PATH" ]] || die "Poe environment file is missing"
    local env_mode
    env_mode="$(stat -c '%a' "$ENV_PATH")"
    ((((8#$env_mode) & 077) == 0)) ||
        die "Poe environment file must not be group/world-readable"
    cmp --silent "$SCRIPT_DIR/$UNIT_NAME" "$UNIT_PATH" ||
        die "installed Poe unit differs from the repo copy"
    MEMRA_POE_VENV="$VENV" "$SCRIPT_DIR/run.sh" --dry-run
    printf 'CHECK PASS: pinned Poe bot installation is complete\n'
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

python3 "$SCRIPT_DIR/poe_config.py" --dry-run

if [[ "$mode" == "check" ]]; then
    verify_install
    exit 0
fi

if ((dry_run == 0)); then
    [[ "$EUID" -eq 0 ]] || die "installation must run as root"
fi

run apt-get update
run apt-get install --yes python3 python3-venv ca-certificates

if ((dry_run)); then
    printf '+ create system user memra-poe when absent\n'
else
    if ! getent passwd memra-poe >/dev/null; then
        useradd --system --home-dir /var/lib/memra-poe --create-home \
            --shell /usr/sbin/nologin memra-poe
    fi
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    run python3 -m venv "$VENV"
fi
run "${VENV}/bin/pip" install --disable-pip-version-check \
    --requirement "$SCRIPT_DIR/requirements.txt"

run install -d -m 0755 /etc/memra
if [[ ! -e "$ENV_PATH" ]]; then
    run install -m 0600 "$SCRIPT_DIR/poe-bot.env.example" "$ENV_PATH"
else
    printf 'Preserving existing %s\n' "$ENV_PATH"
fi
run install -m 0644 "$SCRIPT_DIR/$UNIT_NAME" "$UNIT_PATH"
run systemctl daemon-reload
run systemctl enable "$UNIT_NAME"

if ((dry_run)); then
    printf 'DRY RUN PASS: Poe install plan validated; no host changes were made\n'
else
    verify_install
    printf 'Setup complete. Configure %s, then start with trial-up.sh.\n' "$ENV_PATH"
fi
