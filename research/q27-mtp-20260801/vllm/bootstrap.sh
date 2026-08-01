#!/bin/bash
set -x
V=$HOME/vllm-env
LOGD=$HOME/lane1/research/q27-mtp-20260801/vllm
python3 -m venv $V
$V/bin/pip install -U pip wheel > $LOGD/pip-base.log 2>&1
$V/bin/pip install "huggingface_hub[hf_transfer]" >> $LOGD/pip-base.log 2>&1
sudo mkdir -p /opt/dlami/nvme/hf && sudo chown ubuntu:ubuntu /opt/dlami/nvme/hf
HF_HOME=/opt/dlami/nvme/hf HF_HUB_ENABLE_HF_TRANSFER=1 $V/bin/hf download Qwen/Qwen3.6-27B-FP8 > $LOGD/dl-fp8.log 2>&1 &
DLPID=$!
$V/bin/pip install vllm > $LOGD/pip-vllm.log 2>&1
echo "vllm install rc=$?"
wait $DLPID
echo "download rc=$?"
$V/bin/pip list 2>/dev/null | grep -iE "^(vllm|torch) " 
echo BOOTSTRAP-DONE
