#!/usr/bin/env bash
set -Eeuo pipefail

server_bin="${MEMRA_SERVER_BIN:-/usr/local/bin/memra-server}"
keys_file="${MEMRA_KEYS_FILE:-${MEMRA_API_KEYS:-/etc/memra/keys.toml}}"
dry_run=0
command_name=""

usage() {
    cat <<'EOF'
Usage:
  keyctl.sh [--dry-run] [--keys FILE] [--bin PATH] \
    mint TENANT [--lane interactive|batch] [--rate-limit N]
  keyctl.sh [--dry-run] [--keys FILE] [--bin PATH] revoke KEY_PREFIX

Trial defaults: lane=interactive, rate-limit=1.

KEY_PREFIX must be the safe `prefix` value stored in keys.toml
(`mk-<tenant>-<12 hex>`), never the full plaintext key.
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

secure_keyring() {
    if [[ "$EUID" -eq 0 ]] && getent group memra >/dev/null 2>&1; then
        chown root:memra "$keys_file"
        chmod 0640 "$keys_file"
    else
        chmod 0600 "$keys_file"
    fi
}

validate_live_target() {
    [[ -x "$server_bin" ]] || die "memra-server is not executable: $server_bin"
    [[ ! -L "$keys_file" ]] || die "refusing symlink keyring: $keys_file"
    [[ -d "$(dirname -- "$keys_file")" ]] ||
        die "keyring parent directory does not exist: $(dirname -- "$keys_file")"
    if [[ -e "$keys_file" ]]; then
        [[ -f "$keys_file" ]] || die "keyring is not a regular file: $keys_file"
        local mode
        mode="$(stat -c '%a' "$keys_file")"
        [[ "$mode" == "600" || "$mode" == "640" ]] ||
            die "keyring mode must be 0600 or 0640 before mutation (found $mode)"
    fi
    command -v flock >/dev/null 2>&1 || die "required command not found: flock"
}

while (($#)); do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --keys)
            shift
            (($#)) || die "--keys requires a path"
            keys_file="$1"
            ;;
        --bin)
            shift
            (($#)) || die "--bin requires a path"
            server_bin="$1"
            ;;
        mint | revoke)
            command_name="$1"
            shift
            break
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument before command: $1"
            ;;
    esac
    shift
done

[[ -n "$command_name" ]] || {
    usage >&2
    exit 2
}

if [[ "$command_name" == "mint" ]]; then
    (($#)) || die "mint requires a tenant"
    tenant="$1"
    shift
    lane="interactive"
    rate_limit=1

    while (($#)); do
        case "$1" in
            --lane)
                shift
                (($#)) || die "--lane requires a value"
                lane="$1"
                ;;
            --rate-limit)
                shift
                (($#)) || die "--rate-limit requires a value"
                rate_limit="$1"
                ;;
            *)
                die "unknown mint argument: $1"
                ;;
        esac
        shift
    done

    [[ "$tenant" =~ ^[A-Za-z0-9_-]+$ ]] ||
        die "tenant must match [A-Za-z0-9_-]+"
    [[ "$lane" == "interactive" || "$lane" == "batch" ]] ||
        die "lane must be interactive or batch"
    if [[ ! "$rate_limit" =~ ^[0-9]+$ ]] ||
        ((rate_limit < 1 || rate_limit > 16)); then
        die "rate-limit must be an integer from 1 to 16"
    fi

    command=(
        "$server_bin" --gen-key "$tenant"
        --lane "$lane"
        --rate-limit "$rate_limit"
        --keys "$keys_file"
    )
else
    (($# == 1)) || die "revoke requires exactly one stored key prefix"
    prefix="$1"
    [[ "$prefix" =~ ^mk-[A-Za-z0-9_-]+-[0-9a-fA-F]{12}$ ]] ||
        die "revoke requires the stored mk-<tenant>-<12 hex> prefix, not a full key"
    command=("$server_bin" --revoke-key "$prefix" --keys "$keys_file")
fi

if ((dry_run)); then
    printf 'keyring: %s\n' "$keys_file"
    print_command "${command[@]}"
    printf 'DRY RUN PASS: arguments validated; no key was minted or revoked\n'
    exit 0
fi

umask 077
validate_live_target
exec {lock_fd}>"${keys_file}.lock"
flock "$lock_fd"

"${command[@]}"
secure_keyring
