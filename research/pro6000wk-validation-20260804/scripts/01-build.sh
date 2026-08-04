#!/usr/bin/env bash
# pro6000wk-validation: build stage — sm_120a auto-detect + release wall-clock
set -uo pipefail
cd /root/bw24
export PATH=/usr/local/cuda-13.1/bin:$HOME/.cargo/bin:$PATH
R=/root/receipts
mkdir -p "$R"

echo "=== env ===" | tee "$R/build.log"
nvcc --version 2>&1 | tail -2 | tee -a "$R/build.log"
rustc --version | tee -a "$R/build.log"
nvidia-smi --query-gpu=name,compute_cap,driver_version,power.limit --format=csv | tee -a "$R/build.log"

echo "=== cargo build --release (wall-clock) ===" | tee -a "$R/build.log"
t0=$(date +%s)
cargo build --release >> "$R/build.log" 2>&1
rc=$?
t1=$(date +%s)
echo "BUILD rc=$rc wall=$((t1-t0))s" | tee -a "$R/build.log"
grep -iE "sm_120|120a|arch|MEMRA_CUDA_ARCH" "$R/build.log" | grep -v warning | head -10 | tee "$R/build-arch-banner.log"
ls -la target/release/kernel-check target/release/run-gen target/release/run-spec target/release/memra-server 2>&1 | tee -a "$R/build.log"
exit $rc
