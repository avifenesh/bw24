#!/usr/bin/env bash
# Provision the darklanes-8x box (p5.48xlarge, 8xH100) from a fresh DLAMI — round 47 scale-up.
# Run ON the new box. Assumes the darklanes-bench key can reach the old box for the toolkit.
#
#   bash provision-8x.sh <old-box-ip>
#
# GPU DISCIPLINE (the whole point of scale-up):
#   GPU 0  = BENCHMARK ONLY. Nothing runs there except battery/board runs.
#   GPU 1-7 = dev lanes (CUDA_VISIBLE_DEVICES=<n> per lane; one lane per GPU).
# Lanes must export CUDA_VISIBLE_DEVICES explicitly — a naked run grabs GPU 0 and poisons
# the bench regime. Check with `nvidia-smi` before benching: GPU 0 must be idle.
set -euo pipefail
OLD=${1:?usage: provision-8x.sh <old-box-ip>}
KEY=~/.ssh/darklanes-bench.pem

# 1. DLAMI pre-mounts the ephemeral NVMe set as one LVM at /opt/dlami/nvme (28T on the 48x).
SCRATCH=/opt/dlami/nvme
sudo chown ubuntu $SCRATCH 2>/dev/null || true
mkdir -p $SCRATCH/models $SCRATCH/hf
export HF_HOME=$SCRATCH/hf

# 2. rust + repo + cuda-13.3.1 toolkit (rsync'd from the old box) + models.
command -v cargo >/dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)
source ~/.cargo/env
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no" ubuntu@$OLD:~/cuda-13.3.1/ ~/cuda-13.3.1/ &
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no" --exclude 'target*' ubuntu@$OLD:~/memra/ ~/memra/ &
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no" ubuntu@$OLD:~/models/ $SCRATCH/models/ &
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no" ubuntu@$OLD:/opt/dlami/nvme/models/ $SCRATCH/models/ &
wait
ln -sfn $SCRATCH/models ~/models

# 3. build (sm_90a auto-detected).
cd ~/memra
export PATH=$HOME/cuda-13.3.1/bin:$PATH
cargo build --release 2>&1 | tail -1

# 4. sanity: 8 GPUs visible, kernel-check on a DEV gpu (not GPU 0).
nvidia-smi --query-gpu=index,name --format=csv,noheader
CUDA_VISIBLE_DEVICES=7 ./target/release/kernel-check | tail -1
echo "PROVISIONED — GPU 0 reserved for benchmarking; lanes use CUDA_VISIBLE_DEVICES=1..7"
