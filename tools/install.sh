#!/bin/sh
# memra installer — downloads the prebuilt self-contained binaries from the latest
# GitHub release (or $MEMRA_VERSION, e.g. MEMRA_VERSION=v0.69.0) into $MEMRA_INSTALL_DIR
# (default ~/.local/bin) and verifies the sha256 against the release's SHA256SUMS.
#
#   curl -fsSL https://raw.githubusercontent.com/avifenesh/memra/main/tools/install.sh | sh
#
# Arch selection: sm_120a (RTX 50-series, default) / sm_90a (Hopper) / sm_89 (Ada,
# portable build) — auto-detected from nvidia-smi when present, override with
# MEMRA_CUDA_ARCH. Requirements at RUN time: Linux x86_64, NVIDIA driver >= 580
# (CUDA 13 runtime support), CUDA 13 runtime libraries (cudart, cublas, cublasLt),
# glibc >= 2.35. Model weights are NOT bundled — run-gen/memra-server auto-download
# from Hugging Face via hf:owner/repo:QUANT specs.
set -eu

REPO="avifenesh/memra"
INSTALL_DIR="${MEMRA_INSTALL_DIR:-$HOME/.local/bin}"
BINS="memra-server run-gen run-spec kernel-check"

err() { echo "install.sh: $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || err "curl is required"
command -v tar  >/dev/null 2>&1 || err "tar is required"
[ "$(uname -s)" = "Linux" ]   || err "prebuilt binaries are Linux-only; build from source: cargo install memra-server"
[ "$(uname -m)" = "x86_64" ]  || err "prebuilt binaries are x86_64-only; build from source: cargo install memra-server"

# Resolve version: explicit MEMRA_VERSION or the latest release tag.
VERSION="${MEMRA_VERSION:-}"
if [ -z "$VERSION" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -m1 '"tag_name"' | cut -d'"' -f4) || err "could not resolve latest release"
fi

# Resolve CUDA arch: explicit MEMRA_CUDA_ARCH, else nvidia-smi compute cap, else 120a.
ARCH="${MEMRA_CUDA_ARCH:-}"
if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ') || cap=""
    case "$cap" in
        12.0|12.1) ARCH=120a ;;
        9.0)       ARCH=90a  ;;
        8.9)       ARCH=89   ;;
    esac
fi
ARCH="${ARCH:-120a}"

# glibc floor: pick the 2.35 build (ubuntu-22.04) — runs on 2.35+.
PKG="memra-$VERSION-linux-x86_64-glibc2.35-sm$ARCH"
BASE="https://github.com/$REPO/releases/download/$VERSION"

echo "memra $VERSION (sm_$ARCH) -> $INSTALL_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fsSL -o "$TMP/$PKG.tar.gz" "$BASE/$PKG.tar.gz" \
    || err "download failed: $BASE/$PKG.tar.gz (release may predate the sm_$ARCH matrix)"
curl -fsSL -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" \
    || err "download failed: $BASE/SHA256SUMS"
( cd "$TMP" && grep " $PKG.tar.gz\$" SHA256SUMS | sha256sum -c - >/dev/null ) \
    || err "sha256 verification FAILED for $PKG.tar.gz"

tar -C "$TMP" -xzf "$TMP/$PKG.tar.gz"
mkdir -p "$INSTALL_DIR"
for b in $BINS; do
    install -m 755 "$TMP/$PKG/$b" "$INSTALL_DIR/$b"
done

echo "installed: $BINS"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo "NOTE: $INSTALL_DIR is not on your PATH" ;;
esac
echo "verify:   $INSTALL_DIR/kernel-check   # expect: ALL GREEN"
