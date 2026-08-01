#!/bin/bash
V=$HOME/vllm-env
LOGD=$HOME/lane1/research/q27-mtp-20260801/vllm
cd $HOME/lane1
export HF_HOME=/opt/dlami/nvme/hf
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export PATH=$V/bin:/usr/local/cuda/bin:$PATH
for K in 0 1 2 3; do
  # wait until GPU 1 has >70GB free (orphaned EngineCores die slowly)
  for i in $(seq 1 30); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    [ "$FREE" -gt 71000 ] && break
    sleep 10
  done
  $V/bin/python3 bench_vllm.py --model Qwen/Qwen3.6-27B-FP8 --runs 5 --spec-k $K \
    --out $LOGD/vllm-fp8-spec$K.json > $LOGD/bench-spec$K.log 2>&1
  echo "spec$K rc=$?"
done
echo VLLM-BENCH-DONE
