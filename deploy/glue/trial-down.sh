#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

HORDE_ENV="${MEMRA_HORDE_ENV_FILE:-/etc/memra/horde-worker.env}"
FLEET_LEDGER="${MEMRA_FLEET_LEDGER:-/var/lib/memra/receipts/fleet.jsonl}"
dry_run=0
failures=0

usage() {
    cat <<'EOF'
Usage: trial-down.sh [--dry-run]

Stop the serving trial in traffic-safe order. Before the live command, set the
Poe bot private in the Poe creator UI and run the receipts-harvest sequence in
TRIAL-RUNBOOK.md. This script then:

  1. places the AI Horde worker in maintenance;
  2. disables and stops the Horde worker and Poe shim;
  3. disables the fleet timer and takes one final metrics snapshot;
  4. disables and drains memra-server through SIGTERM;
  5. disables and stops cloudflared.

The services remain installed but disabled so a pod reboot cannot reopen the
trial. A later trial-up explicitly enables each unit again.
EOF
}

note() {
    printf '[trial-down] %s\n' "$*"
}

warn() {
    printf '[trial-down] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[trial-down] ERROR: %s\n' "$*" >&2
    exit 1
}

env_value() {
    local path="$1" wanted="$2"
    awk -v wanted="$wanted" '
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        {
            equals = index($0, "=")
            if (equals == 0) next
            key = substr($0, 1, equals - 1)
            if (key == wanted) {
                found += 1
                value = substr($0, equals + 1)
            }
        }
        END {
            if (found != 1) exit(found == 0 ? 1 : 2)
            if ((substr(value, 1, 1) == "\"" && substr(value, length(value), 1) == "\"") ||
                (substr(value, 1, 1) == "\047" && substr(value, length(value), 1) == "\047")) {
                value = substr(value, 2, length(value) - 2)
            }
            print value
        }
    ' "$path"
}

service_exists() {
    systemctl cat "$1" >/dev/null 2>&1
}

disable_service() {
    local unit="$1"
    if ! service_exists "$unit"; then
        note "$unit is not installed"
        return 0
    fi
    note "disabling and stopping $unit"
    if ! systemctl disable --now "$unit"; then
        warn "failed to disable and stop $unit"
        failures=$((failures + 1))
    fi
}

put_horde_in_maintenance() {
    [[ -f "$HORDE_ENV" ]] || {
        warn "$HORDE_ENV is absent; cannot set Horde maintenance"
        return 1
    }

    local cluster worker_name horde_key encoded details worker_id
    cluster="$(env_value "$HORDE_ENV" MEMRA_HORDE_CLUSTER_URL)" || return 1
    worker_name="$(env_value "$HORDE_ENV" MEMRA_HORDE_WORKER_NAME)" || return 1
    horde_key="$(env_value "$HORDE_ENV" AI_HORDE_API_KEY)" || return 1
    encoded="$(jq -nr --arg value "$worker_name" '$value | @uri')"
    details="$(
        curl --fail --silent --show-error --max-time 15 \
            "${cluster%/}/api/v2/workers/name/${encoded}"
    )" || return 1
    worker_id="$(jq -er '.id | select(type == "string" and length > 0)' <<<"$details")" ||
        return 1

    curl --fail --silent --show-error --max-time 20 \
        --request PUT \
        --header @- \
        --header 'Content-Type: application/json' \
        --header 'Client-Agent: memra-trial-glue:1:https://github.com/avifenesh/memra' \
        --data '{"maintenance":true,"maintenance_msg":"memra research preview is offline"}' \
        "${cluster%/}/api/v2/workers/${worker_id}" \
        >/dev/null <<<"apikey: ${horde_key}"
    unset horde_key
}

while (($#)); do
    case "$1" in
        --dry-run)
            dry_run=1
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

if ((dry_run)); then
    cat <<EOF
[trial-down] DRY RUN - no network or service state will change

Required manual step:
  Set the Poe bot private before shutdown.

Planned sequence:
  PUT AI Horde worker maintenance=true using ${HORDE_ENV}
  systemctl disable --now memra-horde-worker.service
  systemctl disable --now memra-poe-bot.service
  systemctl disable --now memra-fleet-meter.timer
  systemctl start memra-fleet-meter.service  # final ${FLEET_LEDGER} row
  systemctl disable --now memra-server.service  # graceful SIGTERM drain
  systemctl disable --now cloudflared.service

DRY RUN PASS: no API calls, files, or services changed.
EOF
    exit 0
fi

[[ "$EUID" -eq 0 ]] || die "live shutdown must run as root"
[[ -d /run/systemd/system ]] || die "systemd is not running"
for command in awk curl jq systemctl; do
    command -v "$command" >/dev/null 2>&1 ||
        die "required command not found: $command"
done

warn "confirm the Poe bot is private; this cannot be automated by the pod"
if put_horde_in_maintenance; then
    note "AI Horde worker is in maintenance"
else
    warn "could not set AI Horde maintenance; continuing emergency-safe shutdown"
    failures=$((failures + 1))
fi

disable_service memra-horde-worker.service
disable_service memra-poe-bot.service
disable_service memra-fleet-meter.timer

if service_exists memra-fleet-meter.service &&
    systemctl is-active --quiet memra-server.service; then
    note "taking the final fleet-meter snapshot"
    if ! systemctl start memra-fleet-meter.service; then
        warn "final fleet-meter snapshot failed"
        failures=$((failures + 1))
    elif [[ ! -s "$FLEET_LEDGER" ]]; then
        warn "fleet ledger is missing or empty: $FLEET_LEDGER"
        failures=$((failures + 1))
    fi
fi

disable_service memra-server.service
disable_service cloudflared.service

if ((failures > 0)); then
    die "shutdown completed with $failures warning(s); inspect the journal and receipts"
fi
note "trial services are disabled and stopped; installed state and receipts were preserved"
