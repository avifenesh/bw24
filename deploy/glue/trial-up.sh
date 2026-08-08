#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNPOD_ENV="${MEMRA_RUNPOD_ENV:-/etc/memra/runpod.env}"
HORDE_ENV="${MEMRA_HORDE_ENV_FILE:-/etc/memra/horde-worker.env}"
POE_ENV="${MEMRA_POE_ENV_FILE:-/etc/memra/poe-bot.env}"
MODEL_METADATA="${SCRIPT_DIR}/trial-models.toml"
MODEL_ALIAS="stepfun/step-3.7-flash"
FLEET_LEDGER="${MEMRA_FLEET_LEDGER:-/var/lib/memra/receipts/fleet.jsonl}"
DROPIN_DIR="/etc/systemd/system/memra-server.service.d"
DROPIN_PATH="${DROPIN_DIR}/trial-glue.conf"

mode="up"
mode_selected=0
hostname="${MEMRA_TUNNEL_HOSTNAME:-api.tiyuvta.ai}"
origin_override="${MEMRA_TUNNEL_ORIGIN:-}"
token_file="${CLOUDFLARED_TOKEN_FILE:-}"
ready_timeout="${MEMRA_TRIAL_READY_TIMEOUT:-1800}"
horde_timeout="${MEMRA_TRIAL_HORDE_TIMEOUT:-120}"

LOCAL_ROOT=""
PUBLIC_ROOT=""
KEYS_FILE=""
SERVER_PORT=""
HORDE_CLUSTER_URL="https://aihorde.net"
HORDE_WORKER_NAME="memra-research-preview-runpod"
POE_PATH="/poe"
POE_PORT="8081"
HEALTH_FAILURES=0
tmp_file=""

usage() {
    cat <<'EOF'
Usage: trial-up.sh [--dry-run | --check]
                   [--hostname HOST] [--origin URL] [--token-file PATH]
                   [--ready-timeout SECONDS] [--horde-timeout SECONDS]

Start the complete Step-3.7-Flash serving trial on a provisioned RunPod host.

  --dry-run  Validate repo-owned glue and print the live sequence. No host,
             service, secret, network, or package state is changed.
  --check    Do not start anything; print the live health-check matrix.

Live prerequisites:
  deploy/runpod/provision.sh has staged the model, installed memra-server,
  written /etc/memra/runpod.env and the keyring, and installed fleet metering.
  The Horde and Poe environment files must contain real dedicated backend keys.
  The Cloudflare dashboard must contain both the direct catch-all route and the
  more-specific Poe /poe route documented under deploy/glue/.
EOF
}

note() {
    printf '[trial-up] %s\n' "$*"
}

warn() {
    printf '[trial-up] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[trial-up] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "$tmp_file" && -f "$tmp_file" ]]; then
        rm -f -- "$tmp_file"
    fi
}

trap cleanup EXIT

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
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

optional_env_value() {
    local path="$1" wanted="$2" value
    if value="$(env_value "$path" "$wanted" 2>/dev/null)"; then
        printf '%s' "$value"
    fi
}

validate_positive_seconds() {
    local name="$1" value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
    ((value <= 86400)) || die "$name must be at most 86400 seconds"
}

validate_hostname() {
    [[ "$hostname" =~ ^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)+$ ]] ||
        die "invalid tunnel hostname: $hostname"
    [[ "$hostname" == *.tiyuvta.ai ]] ||
        die "tunnel hostname must be a tiyuvta.ai subdomain"
}

validate_metadata() {
    python3 - "$MODEL_METADATA" "$MODEL_ALIAS" <<'PY'
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
alias = sys.argv[2]
with path.open("rb") as handle:
    data = tomllib.load(handle)
model = data.get("models", {}).get(alias)
if not isinstance(model, dict):
    raise SystemExit(f"missing metadata for {alias}")
pricing = model.get("pricing", {})
expected = {
    "prompt": "0.000000234",
    "cached_prompt": "0.0000000585",
    "completion": "0.000001872",
}
if pricing != expected:
    raise SystemExit(f"unexpected trial pricing: {pricing!r}")
if model.get("is_ready") is not True:
    raise SystemExit("trial model metadata must set is_ready=true")
print("PASS: trial model metadata and paper prices are valid")
PY
}

validate_runpod_contract() {
    [[ -f "$RUNPOD_ENV" ]] || die \
        "missing $RUNPOD_ENV; run deploy/runpod/provision.sh first"
    [[ -x /usr/local/bin/memra-server ]] ||
        die "missing /usr/local/bin/memra-server"
    [[ -f /etc/systemd/system/memra-server.service ]] ||
        die "memra-server.service is not installed"
    [[ -f /etc/systemd/system/memra-fleet-meter.service ]] ||
        die "memra-fleet-meter.service is not installed"
    [[ -f /etc/systemd/system/memra-fleet-meter.timer ]] ||
        die "memra-fleet-meter.timer is not installed"

    local compat models pp_stages pp_devices address disabled
    compat="$(env_value "$RUNPOD_ENV" MEMRA_COMPAT)" ||
        die "MEMRA_COMPAT must appear exactly once in $RUNPOD_ENV"
    models="$(env_value "$RUNPOD_ENV" MEMRA_MODELS)" ||
        die "MEMRA_MODELS must appear exactly once in $RUNPOD_ENV"
    KEYS_FILE="$(env_value "$RUNPOD_ENV" MEMRA_API_KEYS)" ||
        die "MEMRA_API_KEYS must appear exactly once in $RUNPOD_ENV"
    pp_stages="$(env_value "$RUNPOD_ENV" MEMRA_PP_STAGES)" ||
        die "MEMRA_PP_STAGES must appear exactly once in $RUNPOD_ENV"
    pp_devices="$(env_value "$RUNPOD_ENV" MEMRA_PP_DEVICES)" ||
        die "MEMRA_PP_DEVICES must appear exactly once in $RUNPOD_ENV"
    address="$(env_value "$RUNPOD_ENV" MEMRA_ADDR)" ||
        die "MEMRA_ADDR must appear exactly once in $RUNPOD_ENV"

    [[ "$compat" == "openai" ]] || die "MEMRA_COMPAT must be openai"
    [[ "$models" == "${MODEL_ALIAS}="*+* ]] ||
        die "MEMRA_MODELS must bind $MODEL_ALIAS to trunk+drafter"
    [[ "$pp_stages" == "2" && "$pp_devices" == "0,1" ]] ||
        die "the trial requires PP-2 on CUDA devices 0,1"
    [[ "$address" =~ ^127\.0\.0\.1:([1-9][0-9]{0,4})$ ]] ||
        die "the Cloudflare trial requires loopback MEMRA_ADDR, found $address"
    SERVER_PORT="${BASH_REMATCH[1]}"
    ((SERVER_PORT <= 65535)) || die "invalid memra port in MEMRA_ADDR"
    LOCAL_ROOT="http://127.0.0.1:${SERVER_PORT}"
    PUBLIC_ROOT="https://${hostname}"

    [[ -f "$KEYS_FILE" && ! -L "$KEYS_FILE" ]] ||
        die "MEMRA_API_KEYS is not a regular non-symlink file: $KEYS_FILE"
    local key_mode
    key_mode="$(stat -c '%a' "$KEYS_FILE")"
    [[ "$key_mode" == "600" || "$key_mode" == "640" ]] ||
        die "keyring mode must be 0600 or 0640, found $key_mode"

    for disabled in MEMRA_SERVE_BATCH MEMRA_ADMIT_YIELD; do
        if [[ "$(optional_env_value "$RUNPOD_ENV" "$disabled")" == "0" ]]; then
            die "$disabled=0 disables a required trial admission path"
        fi
    done

    if [[ -n "$origin_override" ]]; then
        [[ "$origin_override" == "$LOCAL_ROOT" ]] ||
            die "--origin must match the provisioned loopback origin $LOCAL_ROOT"
    else
        origin_override="$LOCAL_ROOT"
    fi
}

load_service_coordinates() {
    local value
    if [[ -f "$HORDE_ENV" ]]; then
        value="$(optional_env_value "$HORDE_ENV" MEMRA_HORDE_CLUSTER_URL)"
        HORDE_CLUSTER_URL="${value:-$HORDE_CLUSTER_URL}"
        value="$(optional_env_value "$HORDE_ENV" MEMRA_HORDE_WORKER_NAME)"
        HORDE_WORKER_NAME="${value:-$HORDE_WORKER_NAME}"
    fi
    if [[ -f "$POE_ENV" ]]; then
        value="$(optional_env_value "$POE_ENV" MEMRA_POE_PATH)"
        POE_PATH="${value:-$POE_PATH}"
        value="$(optional_env_value "$POE_ENV" MEMRA_POE_PORT)"
        POE_PORT="${value:-$POE_PORT}"
    fi
}

install_server_metadata() {
    note "installing the trial model metadata drop-in"
    install -d -m 0755 "$DROPIN_DIR"
    tmp_file="$(mktemp)"
    {
        printf '[Service]\n'
        printf 'Environment=MEMRA_MODEL_METADATA=%s\n' "$MODEL_METADATA"
    } >"$tmp_file"
    install -m 0644 "$tmp_file" "$DROPIN_PATH"
    rm -f "$tmp_file"
    tmp_file=""
    systemctl daemon-reload
}

prepare_glue() {
    if systemctl cat memra-horde-worker.service >/dev/null 2>&1; then
        "$SCRIPT_DIR/horde-worker/setup.sh" --check
    else
        note "installing the pinned AI Horde bridge"
        "$SCRIPT_DIR/horde-worker/setup.sh"
    fi

    if systemctl cat memra-poe-bot.service >/dev/null 2>&1; then
        "$SCRIPT_DIR/poe-bot/setup.sh" --check
    else
        note "installing the pinned Poe bot runtime"
        "$SCRIPT_DIR/poe-bot/setup.sh"
    fi

    "$SCRIPT_DIR/horde-worker/run.sh" --check
    "$SCRIPT_DIR/poe-bot/run.sh" --check
    systemd-analyze verify \
        /etc/systemd/system/memra-server.service \
        /etc/systemd/system/memra-fleet-meter.service \
        /etc/systemd/system/memra-fleet-meter.timer \
        /etc/systemd/system/memra-horde-worker.service \
        /etc/systemd/system/memra-poe-bot.service
}

wait_for_url() {
    local url="$1" timeout="$2" label="$3"
    local deadline=$((SECONDS + timeout))
    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error --max-time 5 \
            "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    warn "$label did not become ready within ${timeout}s"
    return 1
}

horde_worker_url() {
    local encoded
    encoded="$(jq -nr --arg value "$HORDE_WORKER_NAME" '$value | @uri')"
    printf '%s/api/v2/workers/name/%s' "${HORDE_CLUSTER_URL%/}" "$encoded"
}

clear_horde_maintenance() {
    local details_file details_code details worker_id horde_key
    details_file="$(mktemp)"
    details_code="$(
        curl --silent --show-error --max-time 15 \
            --output "$details_file" --write-out '%{http_code}' \
            "$(horde_worker_url)"
    )" || {
        rm -f "$details_file"
        die "could not reach AI Horde while checking maintenance state"
    }
    if [[ "$details_code" == "404" ]]; then
        rm -f "$details_file"
        note "Horde worker is not registered yet; first poll will create it"
        return 0
    fi
    [[ "$details_code" == "200" ]] || {
        rm -f "$details_file"
        die "AI Horde worker lookup returned HTTP $details_code"
    }
    details="$(<"$details_file")"
    rm -f "$details_file"
    worker_id="$(jq -er '.id | select(type == "string" and length > 0)' <<<"$details")" ||
        die "AI Horde returned worker details without an id"
    horde_key="$(env_value "$HORDE_ENV" AI_HORDE_API_KEY)" ||
        die "AI_HORDE_API_KEY must appear exactly once in $HORDE_ENV"

    note "clearing AI Horde maintenance mode"
    curl --fail --silent --show-error --max-time 20 \
        --request PUT \
        --header @- \
        --header 'Content-Type: application/json' \
        --header 'Client-Agent: memra-trial-glue:1:https://github.com/avifenesh/memra' \
        --data '{"maintenance":false}' \
        "${HORDE_CLUSTER_URL%/}/api/v2/workers/${worker_id}" \
        >/dev/null <<<"apikey: ${horde_key}"
    unset horde_key
}

check_local_ready() {
    curl --fail --silent --show-error --max-time 10 "$LOCAL_ROOT/readyz" |
        jq -e '.status == "ready"' >/dev/null
}

check_public_tls() {
    systemctl is-active --quiet cloudflared.service &&
        curl --fail --silent --show-error --max-time 15 \
            "$PUBLIC_ROOT/readyz" |
            jq -e '.status == "ready"' >/dev/null
}

check_openrouter_schema() {
    curl --fail --silent --show-error --max-time 15 \
        "$PUBLIC_ROOT/models?schema=openrouter" |
        jq -e --arg model "$MODEL_ALIAS" '
            any(.data[];
                .id == $model
                and .schema_version == "2.4"
                and .is_ready == true
                and any(.input_modalities[]?;
                    .type == "text"
                    and any(.pricing[]?;
                        .type == "prompt"
                        and .cost_usd == "0.000000234")
                    and any(.pricing[]?;
                        .type == "cached_prompt"
                        and .cost_usd == "0.0000000585"))
                and any(.output_modalities[]?;
                    .type == "text"
                    and .streaming == true
                    and any(.pricing[]?;
                        .type == "completion"
                        and .cost_usd == "0.000001872")))
        ' >/dev/null
}

check_metrics() {
    curl --fail --silent --show-error --max-time 10 "$LOCAL_ROOT/metrics" |
        jq -e '
            (.admitted | type == "number")
            and (.completed | type == "number")
            and (.tokens_out | type == "number")
            and (.step_p50_ms | type == "number")
            and (.step_p99_ms | type == "number")
            and (.prompt_tokens_in | type == "number")
            and (.cached_tokens_in | type == "number")
            and (.computed_tokens_in
                == (.prompt_tokens_in - .cached_tokens_in))
            and (.cache_hit_token_ratio | type == "number")
            and (.prefix_cache_hits | type == "number")
            and (.prefix_cache_misses | type == "number")
            and (.lcp_histogram.edges | type == "array")
            and (.lcp_histogram.counts | type == "array")
            and (.serve_idle_seconds | type == "number")
        ' >/dev/null
}

check_admission() {
    curl --fail --silent --show-error --max-time 10 \
        "$LOCAL_ROOT/yield/metrics" |
        jq -e '
            .lanes
            | all(.interactive, .judge, .harvest;
                (.admitted | type == "number")
                and (.shed | type == "number")
                and (.completed | type == "number")
                and (.tokens_out | type == "number"))
        ' >/dev/null
}

check_auth() {
    local code
    code="$(
        curl --silent --show-error --max-time 15 \
            --output /dev/null --write-out '%{http_code}' \
            --header 'Content-Type: application/json' \
            --data "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"auth check\"}],\"max_tokens\":1}" \
            "$LOCAL_ROOT/v1/chat/completions"
    )" || return 1
    [[ "$code" == "401" ]]
}

check_fleet_meter() {
    systemctl is-active --quiet memra-fleet-meter.timer &&
        [[ -s "$FLEET_LEDGER" ]]
}

check_horde_polling() {
    systemctl is-active --quiet memra-horde-worker.service &&
        curl --fail --silent --show-error --max-time 15 \
            "$(horde_worker_url)" |
            jq -e --arg worker "$HORDE_WORKER_NAME" '
                .name == $worker
                and .type == "text"
                and .online == true
                and .maintenance_mode == false
                and (.models | index("memra-research-preview") != null)
            ' >/dev/null
}

wait_for_horde_polling() {
    local timeout="$1"
    local deadline=$((SECONDS + timeout))
    while ((SECONDS < deadline)); do
        if check_horde_polling; then
            return 0
        fi
        sleep 5
    done
    return 1
}

check_poe_local() {
    systemctl is-active --quiet memra-poe-bot.service &&
        curl --fail --silent --show-error --max-time 10 \
            "http://127.0.0.1:${POE_PORT}${POE_PATH}/livez" >/dev/null
}

check_poe_public() {
    curl --fail --silent --show-error --max-time 15 \
        "${PUBLIC_ROOT}${POE_PATH}/livez" >/dev/null
}

matrix_row() {
    local label="$1" check_function="$2"
    if "$check_function"; then
        printf 'PASS  %-30s\n' "$label"
    else
        printf 'FAIL  %-30s\n' "$label"
        HEALTH_FAILURES=$((HEALTH_FAILURES + 1))
    fi
}

health_matrix() {
    HEALTH_FAILURES=0
    printf '\nTrial health matrix\n'
    printf '%-6s%-30s\n' "STATE" "CHECK"
    matrix_row "local endpoint ready" check_local_ready
    matrix_row "public TLS endpoint ready" check_public_tls
    matrix_row "OpenRouter 2.4 + prices" check_openrouter_schema
    matrix_row "cache metering live" check_metrics
    matrix_row "admission lanes live" check_admission
    matrix_row "missing key rejected" check_auth
    matrix_row "fleet-meter timer + ledger" check_fleet_meter
    matrix_row "Horde worker polling" check_horde_polling
    matrix_row "Poe shim local" check_poe_local
    matrix_row "Poe route public" check_poe_public
    printf '\n'
    ((HEALTH_FAILURES == 0))
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
        --hostname)
            shift
            (($#)) || die "--hostname requires a value"
            hostname="$1"
            ;;
        --origin)
            shift
            (($#)) || die "--origin requires a value"
            origin_override="${1%/}"
            ;;
        --token-file)
            shift
            (($#)) || die "--token-file requires a path"
            token_file="$1"
            ;;
        --ready-timeout)
            shift
            (($#)) || die "--ready-timeout requires seconds"
            ready_timeout="$1"
            ;;
        --horde-timeout)
            shift
            (($#)) || die "--horde-timeout requires seconds"
            horde_timeout="$1"
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

validate_positive_seconds MEMRA_TRIAL_READY_TIMEOUT "$ready_timeout"
validate_positive_seconds MEMRA_TRIAL_HORDE_TIMEOUT "$horde_timeout"
validate_hostname
require_command python3
validate_metadata

if [[ "$mode" == "dry-run" ]]; then
    note "DRY RUN - validating every repo-owned setup and runtime path"
    "$SCRIPT_DIR/cloudflared-setup.sh" --dry-run \
        --hostname "$hostname" \
        --origin "${origin_override:-http://127.0.0.1:8002}"
    "$SCRIPT_DIR/horde-worker/setup.sh" --dry-run
    "$SCRIPT_DIR/horde-worker/run.sh" --dry-run
    "$SCRIPT_DIR/poe-bot/setup.sh" --dry-run
    "$SCRIPT_DIR/poe-bot/run.sh" --dry-run
    cat <<EOF

[trial-up] Live sequence after deploy/runpod/provision.sh:
  1. Verify PP-2 Step model, keyring, loopback bind, admission defaults, and units.
  2. Install/check the pinned Horde and Poe runtimes; validate real secrets offline.
  3. Attach ${MODEL_METADATA} to memra-server and restart through systemd.
  4. Wait up to ${ready_timeout}s for the cold model load.
  5. Take a fleet-meter snapshot and arm its timer.
  6. Install/check cloudflared for https://${hostname}.
  7. Clear Horde maintenance, start its one-thread worker, and wait up to ${horde_timeout}s
     for the public worker record to report online.
  8. Start the Poe shim and verify its local and public /poe route.
  9. Print the ten-row trial health matrix.

DRY RUN PASS: no pod, files, packages, services, secrets, or network resources changed.
EOF
    exit 0
fi

[[ "$EUID" -eq 0 ]] || die "live modes must run as root"
[[ -d /run/systemd/system ]] || die "systemd is not running"
for command in awk curl install jq stat systemctl systemd-analyze; do
    require_command "$command"
done
validate_runpod_contract
load_service_coordinates

if [[ "$mode" == "check" ]]; then
    "$SCRIPT_DIR/horde-worker/run.sh" --check
    "$SCRIPT_DIR/poe-bot/run.sh" --check
    health_matrix || die "$HEALTH_FAILURES health checks failed"
    note "all trial checks pass"
    exit 0
fi

install_server_metadata
prepare_glue

note "restarting memra-server with trial metadata; cold load may take several minutes"
systemctl enable memra-server.service
systemctl reset-failed memra-server.service >/dev/null 2>&1 || true
if ! systemctl restart memra-server.service; then
    journalctl -u memra-server.service --no-pager -n 200 >&2 || true
    die "memra-server failed to restart"
fi
if ! wait_for_url "$LOCAL_ROOT/readyz" "$ready_timeout" "memra-server"; then
    journalctl -u memra-server.service --no-pager -n 200 >&2 || true
    die "memra-server did not become ready"
fi

note "taking the first fleet receipt and arming the timer"
systemctl start memra-fleet-meter.service
systemctl enable --now memra-fleet-meter.timer
[[ -s "$FLEET_LEDGER" ]] ||
    die "fleet meter did not create $FLEET_LEDGER"

cloudflare_args=(
    --hostname "$hostname"
    --origin "$origin_override"
)
if [[ -n "$token_file" ]]; then
    cloudflare_args+=(--token-file "$token_file")
elif [[ -f /etc/memra/cloudflared-token ]]; then
    cloudflare_args+=(--token-file /etc/memra/cloudflared-token)
fi
"$SCRIPT_DIR/cloudflared-setup.sh" "${cloudflare_args[@]}"

clear_horde_maintenance
note "starting the AI Horde worker"
systemctl enable memra-horde-worker.service
systemctl reset-failed memra-horde-worker.service >/dev/null 2>&1 || true
systemctl restart memra-horde-worker.service
if ! wait_for_horde_polling "$horde_timeout"; then
    journalctl -u memra-horde-worker.service --no-pager -n 200 >&2 || true
    die "Horde worker did not become online and polling within ${horde_timeout}s"
fi

note "starting the Poe bot shim"
systemctl enable memra-poe-bot.service
systemctl reset-failed memra-poe-bot.service >/dev/null 2>&1 || true
systemctl restart memra-poe-bot.service
if ! wait_for_url \
    "http://127.0.0.1:${POE_PORT}${POE_PATH}/livez" 60 "Poe shim"; then
    journalctl -u memra-poe-bot.service --no-pager -n 200 >&2 || true
    die "Poe shim did not become ready"
fi

health_matrix || die "$HEALTH_FAILURES health checks failed"
note "trial is live at ${PUBLIC_ROOT}/v1"
note "Poe protocol endpoint: ${PUBLIC_ROOT}${POE_PATH}"
note "Horde worker: ${HORDE_WORKER_NAME}"
note "fleet ledger: ${FLEET_LEDGER}"
