#!/usr/bin/env bash
set -Eeuo pipefail

CLOUDFLARED_VERSION="2026.7.3"
CLOUDFLARED_AMD64_SHA256="049777d30f9bf93da6df8bbe31383460eb2aa51a832c6551824d56f9fcc55974"

mode="install"
dry_run=0
hostname="${MEMRA_TUNNEL_HOSTNAME:-api.tiyuvta.ai}"
origin="${MEMRA_TUNNEL_ORIGIN:-http://127.0.0.1:8002}"
token_file="${CLOUDFLARED_TOKEN_FILE:-}"
tmp_dir=""

usage() {
    cat <<'EOF'
Usage: cloudflared-setup.sh [--check] [--dry-run]
                            [--hostname HOST] [--origin URL]
                            [--token-file PATH]

Install and run a remotely managed Cloudflare Tunnel connector for memra.

Configuration:
  CLOUDFLARED_TOKEN       Tunnel token. Never pass it as a command-line flag.
  CLOUDFLARED_TOKEN_FILE  Root-readable file containing the tunnel token.
  MEMRA_TUNNEL_HOSTNAME   Public hostname (default: api.tiyuvta.ai).
  MEMRA_TUNNEL_ORIGIN     Loopback memra origin (default: http://127.0.0.1:8002).

Modes:
  --dry-run  Validate inputs and print redacted actions without changing state.
  --check    Verify the connector, DNS, origin health, and public TLS endpoint.

One manual DNS step is required in the Cloudflare dashboard:
  Open the remotely managed tunnel used by this connector and add a Published
  application route for the chosen hostname to the origin URL. Cloudflare
  writes the proxied DNS record. Copy that tunnel's connector token into
  CLOUDFLARED_TOKEN or a 0600 token file before running this script.

The tunnel route must not add a path suffix: memra serves /v1, /readyz, and
/metrics from the origin root.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '%s\n' "$*"
}

cleanup() {
    if [[ -n "$tmp_dir" && -d "$tmp_dir" ]]; then
        rm -rf "$tmp_dir"
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

run_token_command() {
    if ((dry_run)); then
        printf '+ cloudflared service install <redacted-token>\n'
        return 0
    fi
    cloudflared service install "$1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_inputs() {
    [[ "$hostname" =~ ^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)+$ ]] ||
        die "invalid hostname: $hostname"
    [[ "$hostname" == *.tiyuvta.ai ]] ||
        die "hostname must be a tiyuvta.ai subdomain: $hostname"
    [[ "$origin" =~ ^http://(127\.0\.0\.1|localhost):[0-9]+$ ]] ||
        die "origin must be a loopback HTTP URL with an explicit port"
}

read_token() {
    local token="${CLOUDFLARED_TOKEN:-}"

    if [[ -n "$token_file" ]]; then
        [[ -f "$token_file" ]] || die "token file does not exist: $token_file"
        if ((dry_run == 0)); then
            local mode_bits mode_value
            mode_bits="$(stat -c '%a' "$token_file")"
            mode_value=$((8#$mode_bits))
            (((mode_value & 077) == 0)) ||
                die "token file must not be group/world-readable: $token_file ($mode_bits)"
        fi
        token="$(<"$token_file")"
    fi

    token="${token//$'\r'/}"
    token="${token//$'\n'/}"
    if [[ -z "$token" ]] && ((dry_run)); then
        token="<dry-run-token>"
    fi
    [[ -n "$token" ]] || die \
        "set CLOUDFLARED_TOKEN or CLOUDFLARED_TOKEN_FILE for first-time service installation"
    printf '%s' "$token"
}

install_cloudflared() {
    if command -v cloudflared >/dev/null 2>&1; then
        note "cloudflared already installed: $(cloudflared --version 2>&1 | head -n 1)"
        return 0
    fi

    require_command curl
    require_command sha256sum
    require_command dpkg
    require_command apt-get

    [[ "$(dpkg --print-architecture)" == "amd64" ]] ||
        die "the pinned installer currently supports RunPod amd64 hosts only"

    local deb
    if ((dry_run)); then
        tmp_dir="/tmp/cloudflared-${CLOUDFLARED_VERSION}.dry-run"
    else
        tmp_dir="$(mktemp -d)"
    fi
    deb="${tmp_dir}/cloudflared-linux-amd64.deb"

    run curl --fail --location --silent --show-error \
        --output "$deb" \
        "https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64.deb"
    if ((dry_run)); then
        note "+ verify sha256 ${CLOUDFLARED_AMD64_SHA256}  ${deb}"
    else
        printf '%s  %s\n' "$CLOUDFLARED_AMD64_SHA256" "$deb" | sha256sum --check -
    fi
    run apt-get install --yes "$deb"
    if ((dry_run == 0)); then
        rm -rf "$tmp_dir"
        tmp_dir=""
    fi
}

service_exists() {
    systemctl cat cloudflared.service >/dev/null 2>&1
}

check_url() {
    local label="$1"
    local url="$2"
    curl --fail --silent --show-error --max-time 15 --output /dev/null "$url"
    printf 'PASS  %-18s %s\n' "$label" "$url"
}

check_tunnel() {
    require_command cloudflared
    require_command curl
    require_command getent
    require_command systemctl

    printf 'INFO  %-18s %s\n' "cloudflared" "$(cloudflared --version 2>&1 | head -n 1)"
    systemctl is-active --quiet cloudflared.service ||
        die "cloudflared.service is not active"
    printf 'PASS  %-18s %s\n' "connector service" "active"

    check_url "loopback origin" "${origin}/readyz"

    getent ahosts "$hostname" >/dev/null ||
        die "DNS does not resolve for $hostname; complete the dashboard route step"
    printf 'PASS  %-18s %s\n' "public DNS" "$hostname"

    check_url "public TLS" "https://${hostname}/readyz"
    check_url "OpenAI endpoint" "https://${hostname}/v1/models"
}

while (($#)); do
    case "$1" in
        --check)
            mode="check"
            ;;
        --dry-run)
            dry_run=1
            ;;
        --hostname)
            shift
            (($#)) || die "--hostname requires a value"
            hostname="$1"
            ;;
        --origin)
            shift
            (($#)) || die "--origin requires a value"
            origin="${1%/}"
            ;;
        --token-file)
            shift
            (($#)) || die "--token-file requires a value"
            token_file="$1"
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

validate_inputs

if [[ "$mode" == "check" ]]; then
    ((dry_run == 0)) || die "--check and --dry-run are mutually exclusive"
    check_tunnel
    exit 0
fi

if ((dry_run == 0)); then
    [[ "${EUID}" -eq 0 ]] || die "installation must run as root"
fi

note "Cloudflare tunnel connector"
note "  hostname: ${hostname}"
note "  origin:   ${origin}"
note "  version:  ${CLOUDFLARED_VERSION} (new installs)"

install_cloudflared
require_command systemctl

if service_exists; then
    note "cloudflared.service already exists; preserving its configured tunnel token"
else
    tunnel_token="$(read_token)"
    run_token_command "$tunnel_token"
    unset tunnel_token
fi

run systemctl enable --now cloudflared.service

if ((dry_run)); then
    note "DRY RUN PASS: inputs and install path validated; no network or service changes were made."
    note "After the dashboard route exists, run: $0 --check --hostname ${hostname} --origin ${origin}"
else
    check_tunnel
fi
