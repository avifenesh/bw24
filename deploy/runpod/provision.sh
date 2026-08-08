#!/usr/bin/env bash
# Provision a systemd-capable RunPod 2x RTX PRO 6000 pod for Step-3.7-Flash.

set -Eeuo pipefail
umask 027

DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: deploy/runpod/provision.sh [--dry-run]

Required for a live source build:
  MEMRA_REF                 owner-approved git commit (7-40 hex characters)

Required for a live release install:
  MEMRA_INSTALL_MODE=release
  MEMRA_VERSION             owner-approved release tag (for example v0.72.0)
  MEMRA_REF                 optional; defaults to MEMRA_VERSION

Deployment inputs:
  MEMRA_MODEL_SOURCE        hf (default), rsync, or existing
  MEMRA_MODEL_DIR           pod-local model directory
                            (default /scratch/models/step-3.7-flash)
  MEMRA_RSYNC_SOURCE        rsync source root when source=rsync; it must contain
                            IQ4_XS/ and Step3.7-flash-mtp-Q8_0.gguf
  MEMRA_EXPOSURE            cloudflare (default) or runpod-proxy
  MEMRA_PUBLIC_URL          public origin, with or without trailing /v1
  CLOUDFLARED_TOKEN         remotely-managed tunnel token; required when
                            installing cloudflared for the first time

Optional:
  MEMRA_REPO_URL            default https://github.com/avifenesh/memra.git
  MEMRA_REPO_DIR            default /opt/memra
  MEMRA_TENANT              initial key tenant, default owner
  MEMRA_KEY_RATE_LIMIT      initial key concurrency cap, default 4
  MEMRA_SMOKE_API_KEY       existing key used for the local authorized smoke
  MEMRA_VERIFY_MODEL        1 (default) verifies the four pinned SHA-256 values
  MEMRA_CUDA_ROOT           explicit CUDA toolkit root
  MEMRA_CUDA_COMPAT_DIR     explicit CUDA compatibility-library directory
  RUNPOD_POD_ID             used to derive the runpod-proxy URL

Run the live path as root (or through sudo with the listed environment preserved).
It requires apt, systemd as PID 1, exactly two idle sm_120
GPUs, and the CUDA compatibility libcuda.so.1 used by the RunPod receipts.
It never installs Rust or runs rustup.
EOF
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'provision.sh: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

note() {
    printf '[runpod] %s\n' "$*"
}

warn() {
    printf '[runpod] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[runpod] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

require_no_whitespace() {
    local name="$1" value="$2"
    [[ -n "$value" ]] || die "$name must not be empty"
    [[ "$value" != *[$' \t\r\n']* ]] || die "$name must not contain whitespace"
}

REPO_URL="${MEMRA_REPO_URL:-https://github.com/avifenesh/memra.git}"
REPO_DIR="${MEMRA_REPO_DIR:-/opt/memra}"
INSTALL_MODE="${MEMRA_INSTALL_MODE:-source}"
MEMRA_REF="${MEMRA_REF:-}"
MEMRA_VERSION="${MEMRA_VERSION:-}"
MODEL_SOURCE="${MEMRA_MODEL_SOURCE:-hf}"
MODEL_DIR="${MEMRA_MODEL_DIR:-/scratch/models/step-3.7-flash}"
RSYNC_SOURCE="${MEMRA_RSYNC_SOURCE:-}"
VERIFY_MODEL="${MEMRA_VERIFY_MODEL:-1}"
EXPOSURE="${MEMRA_EXPOSURE:-cloudflare}"
PUBLIC_URL="${MEMRA_PUBLIC_URL:-}"
TENANT="${MEMRA_TENANT:-owner}"
KEY_RATE_LIMIT="${MEMRA_KEY_RATE_LIMIT:-4}"
KEYS_FILE="/etc/memra/keys.toml"
SERVER_PORT="${MEMRA_SERVER_PORT:-8002}"
MODEL_ALIAS="stepfun/step-3.7-flash"
HF_REPO="stepfun-ai/Step-3.7-Flash-GGUF"
HF_REVISION="0b69336d2fd2adfdef9c66e425f7778196c31482"
HF_VENV="${MEMRA_HF_VENV:-/opt/memra-hf}"
RECEIPT_DIR="/var/lib/memra/receipts"
FLEET_LEDGER="${RECEIPT_DIR}/fleet.jsonl"

MODEL_FILES=(
    "IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"
    "IQ4_XS/Step-3.7-flash-IQ4_XS-00002-of-00003.gguf"
    "IQ4_XS/Step-3.7-flash-IQ4_XS-00003-of-00003.gguf"
    "Step3.7-flash-mtp-Q8_0.gguf"
)
MODEL_SHA256=(
    "b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259"
    "e7e0caaaf0057fabc8bf9b71cbe41322f9945a44df7240bb58e6b7c375e7ffec"
    "ccbd3df81b4f4cb8e73d899734944bcbdefcf436faec9203353419c6750c0590"
    "469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57"
)
MODEL_SIZES=(
    "46483327296"
    "46999941600"
    "11510293728"
    "3707276416"
)

case "$INSTALL_MODE" in
    source|release) ;;
    *) die "MEMRA_INSTALL_MODE must be source or release" ;;
esac
case "$MODEL_SOURCE" in
    hf|rsync|existing) ;;
    *) die "MEMRA_MODEL_SOURCE must be hf, rsync, or existing" ;;
esac
case "$VERIFY_MODEL" in
    0|1) ;;
    *) die "MEMRA_VERIFY_MODEL must be 0 or 1" ;;
esac
case "$EXPOSURE" in
    cloudflare|runpod-proxy) ;;
    *) die "MEMRA_EXPOSURE must be cloudflare or runpod-proxy" ;;
esac
[[ "$SERVER_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || die "MEMRA_SERVER_PORT must be a valid port"
((SERVER_PORT <= 65535)) || die "MEMRA_SERVER_PORT must be <= 65535"
[[ "$TENANT" =~ ^[A-Za-z0-9_-]+$ ]] || die "MEMRA_TENANT must match [A-Za-z0-9_-]+"
[[ "$KEY_RATE_LIMIT" =~ ^[1-9][0-9]*$ ]] ||
    die "MEMRA_KEY_RATE_LIMIT must be a positive integer"
require_no_whitespace MEMRA_REPO_DIR "$REPO_DIR"
require_no_whitespace MEMRA_MODEL_DIR "$MODEL_DIR"
[[ "$REPO_DIR" == /* ]] || die "MEMRA_REPO_DIR must be an absolute path"
[[ "$MODEL_DIR" == /* ]] || die "MEMRA_MODEL_DIR must be an absolute path"
[[ "$HF_VENV" == /* ]] || die "MEMRA_HF_VENV must be an absolute path"
[[ "$MODEL_DIR" != *[,+]* ]] ||
    die "MEMRA_MODEL_DIR must not contain ',' or '+' (MEMRA_MODELS delimiters)"

if ((DRY_RUN)); then
    ref_display="$MEMRA_REF"
    if [[ -z "$ref_display" && "$INSTALL_MODE" == release ]]; then
        ref_display="$MEMRA_VERSION"
    fi
    ref_display="${ref_display:-<owner-approved-ref-required-live>}"
    version_display="${MEMRA_VERSION:-<owner-approved-release-required-in-release-mode>}"
    public_display="$PUBLIC_URL"
    if [[ -z "$public_display" && "$EXPOSURE" == runpod-proxy ]] &&
        [[ -n "${RUNPOD_POD_ID:-}" ]]; then
        public_display="https://${RUNPOD_POD_ID}-${SERVER_PORT}.proxy.runpod.net"
    fi
    public_display="${public_display:-<public-origin-required-live>}"
    cat <<EOF
[runpod] DRY RUN - no files, packages, services, keys, or network resources will change

Install:
  mode:              ${INSTALL_MODE}
  repository:        ${REPO_URL}
  checkout:          ${ref_display}
  release:           ${version_display}
  destination:       /usr/local/bin/memra-server

Model staging:
  source:            ${MODEL_SOURCE}
  destination:       ${MODEL_DIR}
  Hugging Face repo: ${HF_REPO}
  pinned revision:   ${HF_REVISION}
  bytes:             four pinned files (108700839040 total; SHA-256 checked live)

Systemd launch contract:
  CUDA_VISIBLE_DEVICES=0,1
  LD_LIBRARY_PATH=<detected CUDA compat>:<detected CUDA lib64>
  MEMRA_ADDR=<127.0.0.1 for cloudflare; 0.0.0.0 for runpod-proxy>:${SERVER_PORT}
  MEMRA_COMPAT=openai
  MEMRA_MODELS=${MODEL_ALIAS}=${MODEL_DIR}/${MODEL_FILES[0]}+${MODEL_DIR}/${MODEL_FILES[3]}
  MEMRA_API_KEYS=${KEYS_FILE}
  MEMRA_PP_STAGES=2
  MEMRA_PP_DEVICES=0,1
  MEMRA_CTX=131072
  MEMRA_SERVE_SPEC=<unset; owner-approved train default>
  MEMRA_SPEC_K=<unset; owner-approved train default>
  MEMRA_SERVE_BATCH=<unset; owner-approved train default>

Operations:
  install hardened memra-server unit and 30-minute fleet-meter timer
  create the first SHA-only API key if ${KEYS_FILE} does not exist
  wait for /readyz, validate /metrics, 401 auth, and one authorized request
  exposure:          ${EXPOSURE}
  public origin:     ${public_display}
  validate public /v1/models

Live-only checks:
  systemd is PID 1; exactly two idle 96 GB sm_120 GPUs; CUDA compat libcuda.so.1;
  local-NVMe capacity and model checksums; source/release availability; model load;
  cloudflared or RunPod proxy routing; public smoke traffic.
EOF
    exit 0
fi

[[ "$EUID" -eq 0 ]] || die "run the live provisioner as root"
[[ -d /run/systemd/system ]] ||
    die "systemd is not running; select a RunPod template that boots systemd as PID 1"
require_command systemctl
require_command apt-get
require_command nvidia-smi

if [[ "$INSTALL_MODE" == source ]]; then
    [[ -n "$MEMRA_REF" ]] ||
        die "MEMRA_REF is required; use the owner-approved post-Step-dance commit"
    [[ "$MEMRA_REF" =~ ^[0-9A-Fa-f]{7,40}$ ]] ||
        die "source-mode MEMRA_REF must be an immutable 7-40 character git commit"
else
    [[ -n "$MEMRA_VERSION" ]] ||
        die "MEMRA_VERSION is required for release installs"
    if [[ -n "$MEMRA_REF" && "$MEMRA_REF" != "$MEMRA_VERSION" ]]; then
        die "release mode requires MEMRA_REF to match MEMRA_VERSION"
    fi
    MEMRA_REF="$MEMRA_VERSION"
fi

require_no_whitespace MEMRA_REF "$MEMRA_REF"
if [[ "$MODEL_SOURCE" == rsync ]]; then
    [[ -n "$RSYNC_SOURCE" ]] || die "MEMRA_RSYNC_SOURCE is required for rsync staging"
fi

if [[ -z "$PUBLIC_URL" && "$EXPOSURE" == runpod-proxy ]]; then
    [[ -n "${RUNPOD_POD_ID:-}" ]] ||
        die "set MEMRA_PUBLIC_URL or RUNPOD_POD_ID for runpod-proxy exposure"
    PUBLIC_URL="https://${RUNPOD_POD_ID}-${SERVER_PORT}.proxy.runpod.net"
fi
[[ -n "$PUBLIC_URL" ]] || die "MEMRA_PUBLIC_URL is required for the live public check"
require_no_whitespace MEMRA_PUBLIC_URL "$PUBLIC_URL"
[[ "$PUBLIC_URL" == https://* ]] || die "MEMRA_PUBLIC_URL must use https://"
if [[ "$EXPOSURE" == cloudflare ]] &&
    ! systemctl cat cloudflared.service >/dev/null 2>&1 &&
    [[ -z "${CLOUDFLARED_TOKEN:-}" ]]; then
    die "CLOUDFLARED_TOKEN is required when cloudflared is not already installed"
fi

note "installing base packages (Rust is intentionally not installed)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    build-essential ca-certificates curl git jq pkg-config python3 python3-venv \
    rsync tar util-linux

require_command sha256sum
require_command systemd-analyze

if ! id memra >/dev/null 2>&1; then
    useradd --system --home-dir /var/lib/memra --create-home \
        --shell /usr/sbin/nologin memra
fi
for group in video render; do
    if getent group "$group" >/dev/null 2>&1; then
        usermod -a -G "$group" memra
    fi
done
install -d -m 0750 -o root -g memra /etc/memra
install -d -m 0750 -o memra -g memra "$RECEIPT_DIR"

note "checking out the explicitly approved memra ref"
if [[ ! -e "$REPO_DIR" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
elif [[ ! -d "$REPO_DIR/.git" ]]; then
    die "$REPO_DIR exists but is not a git checkout"
fi
if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    die "$REPO_DIR has local changes; preserve or remove them before provisioning"
fi
git -C "$REPO_DIR" fetch --prune --tags origin
APPROVED_COMMIT=""
for candidate in "$MEMRA_REF" "origin/$MEMRA_REF"; do
    if git -C "$REPO_DIR" rev-parse --verify "${candidate}^{commit}" >/dev/null 2>&1; then
        APPROVED_COMMIT="$(
            git -C "$REPO_DIR" rev-parse "${candidate}^{commit}"
        )"
        break
    fi
done
if [[ -z "$APPROVED_COMMIT" ]]; then
    git -C "$REPO_DIR" fetch origin "$MEMRA_REF"
    APPROVED_COMMIT="$(git -C "$REPO_DIR" rev-parse 'FETCH_HEAD^{commit}')"
fi
git -C "$REPO_DIR" checkout --detach "$APPROVED_COMMIT"
[[ -z "$(git -C "$REPO_DIR" status --porcelain)" ]] ||
    die "checkout became dirty unexpectedly"
note "approved commit: $APPROVED_COMMIT"

detect_cuda() {
    local candidate
    local -a roots=()
    if [[ -n "${MEMRA_CUDA_ROOT:-}" ]]; then
        roots+=("$MEMRA_CUDA_ROOT")
    fi
    roots+=(/usr/local/cuda /usr/local/cuda-13.2 /usr/local/cuda-13.1)

    CUDA_ROOT=""
    for candidate in "${roots[@]}"; do
        [[ -d "$candidate" ]] || continue
        if [[ -x "$candidate/bin/nvcc" ]] ||
            compgen -G "$candidate/lib64/libcudart.so*" >/dev/null; then
            CUDA_ROOT="$(readlink -f "$candidate")"
            break
        fi
    done
    [[ -n "$CUDA_ROOT" ]] || die "no CUDA 13 toolkit/runtime found under /usr/local"
    if [[ -x "$CUDA_ROOT/bin/nvcc" ]]; then
        CUDA_VERSION="$(
            "$CUDA_ROOT/bin/nvcc" --version |
                sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' |
                head -1
        )"
        [[ "$CUDA_VERSION" == 13.* ]] ||
            die "CUDA 13 is required; $CUDA_ROOT reports ${CUDA_VERSION:-unknown}"
    elif compgen -G "$CUDA_ROOT/lib64/libcudart.so.13*" >/dev/null; then
        CUDA_VERSION="13-runtime"
    else
        die "CUDA 13 is required; no nvcc or libcudart.so.13 found under $CUDA_ROOT"
    fi

    local -a compat_candidates=()
    if [[ -n "${MEMRA_CUDA_COMPAT_DIR:-}" ]]; then
        compat_candidates+=("$MEMRA_CUDA_COMPAT_DIR")
    fi
    compat_candidates+=(
        "$CUDA_ROOT/compat"
        /usr/local/cuda/compat
        /usr/local/cuda-13.2/compat
        /usr/local/cuda-13.1/compat
    )

    CUDA_COMPAT_DIR=""
    for candidate in "${compat_candidates[@]}"; do
        [[ -d "$candidate" ]] || continue
        if compgen -G "$candidate/libcuda.so.1*" >/dev/null; then
            CUDA_COMPAT_DIR="$(readlink -f "$candidate")"
            break
        fi
    done
    [[ -n "$CUDA_COMPAT_DIR" ]] ||
        die "CUDA compat libcuda.so.1 not found; the RunPod receipts fail CUBLAS init without it"
    compgen -G "$CUDA_ROOT/lib64/libcublas.so*" >/dev/null ||
        die "libcublas was not found under $CUDA_ROOT/lib64"

    local -a ld_parts=("$CUDA_COMPAT_DIR" "$CUDA_ROOT/lib64")
    for candidate in /usr/local/nvidia/lib64 /usr/local/nvidia/lib; do
        [[ -d "$candidate" ]] && ld_parts+=("$candidate")
    done
    CUDA_LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${ld_parts[*]}")"
    export LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export CUDA_HOME="$CUDA_ROOT"
    export MEMRA_NVCC="$CUDA_ROOT/bin/nvcc"
    export MEMRA_CUDA_ARCH=120a
}

detect_cuda
note "CUDA root: $CUDA_ROOT ($CUDA_VERSION)"
note "CUDA compatibility path (first in LD_LIBRARY_PATH): $CUDA_COMPAT_DIR"

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//'
)
mapfile -t GPU_CAPS < <(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits |
        sed 's/[[:space:]]//g'
)
mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
        sed 's/[[:space:]]//g'
)
mapfile -t GPU_POWER < <(
    nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits |
        sed 's/[[:space:]]//g'
)
[[ "${#GPU_NAMES[@]}" -eq 2 ]] ||
    die "expected exactly two visible GPUs, found ${#GPU_NAMES[@]}"
[[ "${#GPU_CAPS[@]}" -eq 2 && "${#GPU_MEMORY[@]}" -eq 2 &&
    "${#GPU_POWER[@]}" -eq 2 ]] ||
    die "nvidia-smi returned incomplete two-GPU metadata"
for i in 0 1; do
    [[ "${GPU_CAPS[$i]}" == 12.* ]] ||
        die "GPU $i has compute capability ${GPU_CAPS[$i]}, expected sm_120"
    awk -v value="${GPU_MEMORY[$i]}" 'BEGIN { exit !(value >= 90000) }' ||
        die "GPU $i has ${GPU_MEMORY[$i]} MiB, expected a 96 GB-class card"
    if [[ "${GPU_NAMES[$i]}" != *"RTX PRO 6000"* ]]; then
        warn "GPU $i is ${GPU_NAMES[$i]}, not an RTX PRO 6000"
    fi
    note "GPU $i: ${GPU_NAMES[$i]}, cc=${GPU_CAPS[$i]}, ${GPU_MEMORY[$i]} MiB, power limit ${GPU_POWER[$i]} W"
done
mapfile -t GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
        awk '$1 ~ /^[0-9]+$/ { print $1 }'
)
[[ "${#GPU_PIDS[@]}" -eq 0 ]] ||
    die "GPU compute processes are already active: ${GPU_PIDS[*]}"
if awk -v a="${GPU_POWER[0]}" -v b="${GPU_POWER[1]}" \
    'BEGIN { exit !((a <= 520) || (b <= 520)) }'; then
    warn "a roughly 510 W community-pod cap is a known absolute-throughput caveat"
fi
nvidia-smi topo -m || warn "nvidia-smi topo -m failed; PP-2 startup will still check peer access"

case "$INSTALL_MODE" in
    source)
        require_command cargo
        [[ -x "$MEMRA_NVCC" ]] ||
            die "source mode requires nvcc at $MEMRA_NVCC; use a CUDA devel template"
        note "building memra-server from $APPROVED_COMMIT"
        (
            cd "$REPO_DIR"
            env \
                CUDA_HOME="$CUDA_ROOT" \
                LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
                MEMRA_CUDA_ARCH=120a \
                MEMRA_NVCC="$MEMRA_NVCC" \
                cargo build --release -p memra-server --bin memra-server
        )
        install -m 0755 "$REPO_DIR/target/release/memra-server" \
            /usr/local/bin/memra-server
        ;;
    release)
        note "installing release $MEMRA_VERSION for sm_120a"
        env \
            MEMRA_VERSION="$MEMRA_VERSION" \
            MEMRA_CUDA_ARCH=120a \
            MEMRA_INSTALL_DIR=/usr/local/bin \
            sh "$REPO_DIR/tools/install.sh"
        ;;
esac

[[ -x /usr/local/bin/memra-server ]] || die "memra-server was not installed"

missing_bytes=0
for i in "${!MODEL_FILES[@]}"; do
    path="${MODEL_DIR}/${MODEL_FILES[$i]}"
    size=0
    [[ -f "$path" ]] && size="$(stat -c '%s' "$path")"
    if [[ "$size" != "${MODEL_SIZES[$i]}" ]]; then
        ((missing_bytes += MODEL_SIZES[i]))
    fi
done
install -d -m 0750 -o memra -g memra "$MODEL_DIR"
free_bytes="$(df --output=avail -B1 "$MODEL_DIR" | awk 'NR == 2 { print $1 }')"
reserve_bytes=$((16 * 1024 * 1024 * 1024))
required_bytes=$((missing_bytes + reserve_bytes))
((free_bytes >= required_bytes)) ||
    die "$MODEL_DIR has $free_bytes bytes free; staging needs $required_bytes"
note "model filesystem: $(findmnt -no SOURCE,FSTYPE,TARGET --target "$MODEL_DIR" || true)"
note "model staging requires $missing_bytes missing bytes plus 16 GiB working headroom"

case "$MODEL_SOURCE" in
    hf)
        note "installing the current Hugging Face CLI into $HF_VENV"
        if [[ ! -x "$HF_VENV/bin/python" ]]; then
            python3 -m venv "$HF_VENV"
        fi
        "$HF_VENV/bin/python" -m pip install --upgrade pip
        "$HF_VENV/bin/python" -m pip install --upgrade huggingface_hub hf_xet
        note "downloading the pinned Step artifact revision"
        HF_XET_HIGH_PERFORMANCE=1 "$HF_VENV/bin/hf" download \
            "$HF_REPO" \
            "${MODEL_FILES[@]}" \
            --revision "$HF_REVISION" \
            --local-dir "$MODEL_DIR"
        ;;
    rsync)
        note "rsyncing model bytes from $RSYNC_SOURCE"
        rsync -a --partial --append-verify \
            "${RSYNC_SOURCE%/}/" "$MODEL_DIR/"
        ;;
    existing)
        note "using pre-staged model files in $MODEL_DIR"
        ;;
esac

note "validating model file sizes"
for i in "${!MODEL_FILES[@]}"; do
    path="${MODEL_DIR}/${MODEL_FILES[$i]}"
    [[ -f "$path" ]] || die "missing model file: $path"
    actual_size="$(stat -c '%s' "$path")"
    [[ "$actual_size" == "${MODEL_SIZES[$i]}" ]] ||
        die "wrong size for $path: got $actual_size, expected ${MODEL_SIZES[$i]}"
done

if [[ "$VERIFY_MODEL" == 1 ]]; then
    note "verifying all four pinned model SHA-256 values"
    for i in "${!MODEL_FILES[@]}"; do
        path="${MODEL_DIR}/${MODEL_FILES[$i]}"
        actual_sha="$(sha256sum "$path" | awk '{ print $1 }')"
        [[ "$actual_sha" == "${MODEL_SHA256[$i]}" ]] ||
            die "SHA-256 mismatch for $path: got $actual_sha"
    done
else
    warn "MEMRA_VERIFY_MODEL=0: full model SHA-256 verification was skipped"
fi
chown -R memra:memra "$MODEL_DIR"
chmod -R u=rwX,g=rX,o= "$MODEL_DIR"

model_manifest_tmp="$(mktemp)"
{
    printf 'hf_repo=%s\n' "$HF_REPO"
    printf 'hf_revision=%s\n' "$HF_REVISION"
    printf 'model_dir=%s\n' "$MODEL_DIR"
    printf 'sha256_verified=%s\n' "$VERIFY_MODEL"
    for i in "${!MODEL_FILES[@]}"; do
        printf '%s  %s\n' "${MODEL_SHA256[$i]}" "${MODEL_FILES[$i]}"
    done
} >"$model_manifest_tmp"
install -m 0640 -o memra -g memra \
    "$model_manifest_tmp" "$RECEIPT_DIR/step-model-manifest.txt"
sha256sum "$RECEIPT_DIR/step-model-manifest.txt" |
    awk '{ print $1 }' >"$RECEIPT_DIR/step-model-manifest.sha256"
chown memra:memra "$RECEIPT_DIR/step-model-manifest.sha256"
chmod 0640 "$RECEIPT_DIR/step-model-manifest.sha256"
rm -f "$model_manifest_tmp"

NEW_API_KEY=""
if [[ ! -e "$KEYS_FILE" ]]; then
    note "creating the first API key for tenant $TENANT"
    NEW_API_KEY="$(
        /usr/local/bin/memra-server \
            --gen-key "$TENANT" \
            --rate-limit "$KEY_RATE_LIMIT" \
            --keys "$KEYS_FILE"
    )"
    chown root:memra "$KEYS_FILE"
    chmod 0640 "$KEYS_FILE"
    printf '\n[runpod] NEW API KEY (shown once; plaintext is not stored):\n%s\n\n' \
        "$NEW_API_KEY"
else
    [[ -f "$KEYS_FILE" ]] || die "$KEYS_FILE exists but is not a regular file"
    chown root:memra "$KEYS_FILE"
    chmod 0640 "$KEYS_FILE"
    note "preserving the existing keyring at $KEYS_FILE"
fi

if [[ "$EXPOSURE" == cloudflare ]]; then
    LISTEN_HOST=127.0.0.1
else
    LISTEN_HOST=0.0.0.0
fi
TRUNK="${MODEL_DIR}/${MODEL_FILES[0]}"
DRAFTER="${MODEL_DIR}/${MODEL_FILES[3]}"

note "installing the hardened server and fleet-meter units"
install -m 0644 "$REPO_DIR/deploy/systemd/memra-server.service" \
    /etc/systemd/system/memra-server.service
systemd_version="$(systemctl --version | awk 'NR == 1 { print $2 }')"
[[ "$systemd_version" =~ ^[0-9]+$ ]] || die "could not parse the systemd version"
if ((systemd_version < 254)); then
    sed -i '/^RestartSteps=/d; /^RestartMaxDelaySec=/d' \
        /etc/systemd/system/memra-server.service
    warn "systemd $systemd_version: using flat RestartSec=10 backoff"
fi

env_tmp="$(mktemp)"
cat >"$env_tmp" <<EOF
HOME=/var/lib/memra
LD_LIBRARY_PATH=${CUDA_LD_LIBRARY_PATH}
CUDA_VISIBLE_DEVICES=0,1
MEMRA_ADDR=${LISTEN_HOST}:${SERVER_PORT}
MEMRA_COMPAT=openai
MEMRA_MODELS=${MODEL_ALIAS}=${TRUNK}+${DRAFTER}
MEMRA_API_KEYS=${KEYS_FILE}
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_CTX=131072
EOF
install -m 0640 -o root -g memra "$env_tmp" /etc/memra/runpod.env
rm -f "$env_tmp"

install -d -m 0755 /etc/systemd/system/memra-server.service.d
server_dropin_tmp="$(mktemp)"
{
    printf '[Service]\n'
    printf 'EnvironmentFile=/etc/memra/runpod.env\n'
    printf 'WorkingDirectory=%s\n' "$REPO_DIR"
    printf 'TimeoutStartSec=1800\n'
    supplementary=()
    for group in video render; do
        getent group "$group" >/dev/null 2>&1 && supplementary+=("$group")
    done
    if ((${#supplementary[@]})); then
        printf 'SupplementaryGroups=%s\n' "${supplementary[*]}"
    fi
} >"$server_dropin_tmp"
install -m 0644 "$server_dropin_tmp" \
    /etc/systemd/system/memra-server.service.d/runpod.conf
rm -f "$server_dropin_tmp"

install -m 0644 "$REPO_DIR/deploy/systemd/memra-fleet-meter.service" \
    /etc/systemd/system/memra-fleet-meter.service
install -m 0644 "$REPO_DIR/deploy/systemd/memra-fleet-meter.timer" \
    /etc/systemd/system/memra-fleet-meter.timer
install -d -m 0755 /etc/systemd/system/memra-fleet-meter.service.d
meter_dropin_tmp="$(mktemp)"
cat >"$meter_dropin_tmp" <<EOF
[Service]
WorkingDirectory=${REPO_DIR}
Environment=FLEET_METRICS_URL=http://127.0.0.1:${SERVER_PORT}/metrics
Environment=FLEET_LEDGER=${FLEET_LEDGER}
EOF
install -m 0644 "$meter_dropin_tmp" \
    /etc/systemd/system/memra-fleet-meter.service.d/runpod.conf
rm -f "$meter_dropin_tmp"

systemd-analyze verify \
    /etc/systemd/system/memra-server.service \
    /etc/systemd/system/memra-fleet-meter.service \
    /etc/systemd/system/memra-fleet-meter.timer
systemctl daemon-reload
systemctl enable memra-server.service
systemctl reset-failed memra-server.service >/dev/null 2>&1 || true

deployment_tmp="$(mktemp)"
cat >"$deployment_tmp" <<EOF
installed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
memra_ref=${MEMRA_REF}
memra_commit=${APPROVED_COMMIT}
install_mode=${INSTALL_MODE}
release_version=${MEMRA_VERSION}
cuda_root=${CUDA_ROOT}
cuda_version=${CUDA_VERSION}
cuda_compat_dir=${CUDA_COMPAT_DIR}
model_manifest_sha256=$(cat "$RECEIPT_DIR/step-model-manifest.sha256")
model_source=${MODEL_SOURCE}
model_dir=${MODEL_DIR}
exposure=${EXPOSURE}
public_url=${PUBLIC_URL}
spec_policy=owner-approved-train-default
batch_policy=owner-approved-train-default
EOF
install -m 0640 -o memra -g memra \
    "$deployment_tmp" "$RECEIPT_DIR/deployment.env"
rm -f "$deployment_tmp"

note "starting memra-server; the first Step load may take several minutes"
if ! systemctl restart memra-server.service; then
    journalctl -u memra-server.service --no-pager -n 200 >&2 || true
    die "memra-server failed to start"
fi

LOCAL_ROOT="http://127.0.0.1:${SERVER_PORT}"
ready=0
for _ in $(seq 1 360); do
    if curl --fail --silent --show-error --max-time 5 \
        "$LOCAL_ROOT/readyz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
done
if ((ready == 0)); then
    journalctl -u memra-server.service --no-pager -n 200 >&2 || true
    die "memra-server did not become ready within 30 minutes"
fi

note "validating local model listing and metrics"
curl --fail --silent --show-error "$LOCAL_ROOT/v1/models" |
    jq -e --arg model "$MODEL_ALIAS" \
        '.data | type == "array" and any(.id == $model)' >/dev/null
curl --fail --silent --show-error "$LOCAL_ROOT/metrics" |
    jq -e '
        type == "object"
        and (.admitted | type == "number")
        and (.completed | type == "number")
        and (.prompt_tokens_in | type == "number")
        and (.cached_tokens_in | type == "number")
        and (.computed_tokens_in | type == "number")
        and (.lcp_histogram | type == "object")
    ' >/dev/null

unauthorized_body="$(mktemp)"
if ! unauthorized_code="$(
    curl --silent --show-error --max-time 15 \
        -o "$unauthorized_body" -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"auth check\"}],\"max_tokens\":1}" \
        "$LOCAL_ROOT/v1/chat/completions"
)"; then
    rm -f "$unauthorized_body"
    die "missing-key auth check could not reach the local server"
fi
rm -f "$unauthorized_body"
[[ "$unauthorized_code" == 401 ]] ||
    die "missing-key request returned HTTP $unauthorized_code, expected 401"

SMOKE_KEY="${MEMRA_SMOKE_API_KEY:-$NEW_API_KEY}"
if [[ -n "$SMOKE_KEY" ]]; then
    note "running one authorized local inference request"
    authorized_body="$(mktemp)"
    if ! authorized_code="$(
        curl --silent --show-error --max-time 600 \
            -o "$authorized_body" -w '%{http_code}' \
            -H @- \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":1,\"temperature\":0}" \
            "$LOCAL_ROOT/v1/chat/completions" \
            <<<"Authorization: Bearer ${SMOKE_KEY}"
    )"; then
        rm -f "$authorized_body"
        die "authorized local inference request could not reach the server"
    fi
    if [[ "$authorized_code" != 200 ]] ||
        ! jq -e '
            .usage.prompt_tokens >= 0
            and .usage.completion_tokens >= 0
            and .usage.total_tokens
                == (.usage.prompt_tokens + .usage.completion_tokens)
            and .usage.prompt_tokens_details.cached_tokens >= 0
        ' "$authorized_body" >/dev/null; then
        cat "$authorized_body" >&2
        rm -f "$authorized_body"
        die "authorized local inference smoke failed (HTTP $authorized_code)"
    fi
    rm -f "$authorized_body"
else
    warn "no plaintext key was available; set MEMRA_SMOKE_API_KEY to run authorized inference"
fi

note "arming the 30-minute fleet-meter timer"
systemctl start memra-fleet-meter.service
systemctl enable --now memra-fleet-meter.timer
[[ -s "$FLEET_LEDGER" ]] ||
    die "fleet meter did not create its first receipt at $FLEET_LEDGER"

if [[ "$EXPOSURE" == cloudflare ]]; then
    if ! command -v cloudflared >/dev/null 2>&1; then
        [[ "$(dpkg --print-architecture)" == amd64 ]] ||
            die "automatic cloudflared install currently supports amd64 only"
        cloudflared_deb="$(mktemp --suffix=.deb)"
        curl --fail --location --show-error \
            -o "$cloudflared_deb" \
            https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        apt-get install -y "$cloudflared_deb"
        rm -f "$cloudflared_deb"
    fi
    if ! systemctl cat cloudflared.service >/dev/null 2>&1; then
        [[ -n "${CLOUDFLARED_TOKEN:-}" ]] ||
            die "CLOUDFLARED_TOKEN is required to install the tunnel service"
        note "installing the remotely-managed cloudflared service (token redacted)"
        cloudflared service install "$CLOUDFLARED_TOKEN"
    else
        note "reusing the existing cloudflared systemd service"
    fi
    systemctl enable --now cloudflared.service
else
    warn "runpod-proxy binds memra to 0.0.0.0; ensure HTTP port $SERVER_PORT is exposed in the pod template"
fi

PUBLIC_ROOT="${PUBLIC_URL%/}"
if [[ "$PUBLIC_ROOT" == */v1 ]]; then
    PUBLIC_ROOT="${PUBLIC_ROOT%/v1}"
fi
public_ready=0
for _ in $(seq 1 60); do
    if curl --fail --silent --show-error --max-time 10 \
        "$PUBLIC_ROOT/v1/models" >/dev/null 2>&1; then
        public_ready=1
        break
    fi
    sleep 2
done
((public_ready == 1)) ||
    die "public /v1/models did not become reachable at $PUBLIC_ROOT"

note "serving: $PUBLIC_ROOT/v1"
note "local metrics: $LOCAL_ROOT/metrics"
note "fleet receipts: $FLEET_LEDGER"
note "service logs: journalctl -u memra-server -f"
if [[ -z "$NEW_API_KEY" ]]; then
    note "issue another key with: memra-server --gen-key TENANT --rate-limit N --keys $KEYS_FILE"
fi
note "run $REPO_DIR/deploy/runpod/smoke.sh from a separate machine before sending traffic"
