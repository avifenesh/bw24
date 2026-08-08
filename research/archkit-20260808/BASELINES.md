# Architecture kit pre-refactor baselines

Source commit: `e6a9842664f541a1a3180e12202df147d92870cd`

These are the before-images for the geometry-table migration. The post-change
runs must keep both `MATCH` lines and reproduce the recorded generated token
arrays exactly.

## Qwen 3.5

- Rig: local RTX 5090 Laptop GPU, CUDA 13.1, `sm_120a`
- Artifact:
  `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`
- Command:

  ```sh
  flock -w 3600 /tmp/gpu5090.lock env \
    MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
    MEMRA_NGEN=32 \
    target/release/run-gen \
    /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
  ```

- Result: prefill/decode `MATCH`; batched-prime/tokenwise `MATCH`
- Generated tokens:

  ```text
  [760, 5638, 4534, 11147, 8232, 14855, 10480, 6009, 13914, 494, 23683, 7575, 1472, 46120, 3357, 26502, 888, 87567, 1083, 3545, 4779, 11, 1332, 9966, 1452, 5344, 3947, 61369, 23304, 1070, 13, 8618]
  ```

- Raw log: `raw/baseline-qwen35-run-gen.log`

## Step35

- Rig: Box 1, two RTX PRO 6000 Blackwell Server Edition GPUs, CUDA 13.2,
  `sm_120a`, PP2 on devices 0 and 1
- Artifact:
  `/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf`
- Command:

  ```sh
  flock -w 3600 /tmp/memra-gpu.lock env \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
    MEMRA_NGEN=32 \
    /home/ubuntu/archkit-target/release/run-gen \
    /home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
  ```

- Result: prefill/decode `MATCH`; batched-prime/tokenwise `MATCH`
- Generated tokens:

  ```text
  [128799, 201, 795, 8850, 10531, 75405, 30872, 73815, 295, 260, 21857, 10672, 9934, 15, 123047, 343, 28608, 39, 11, 5629, 14880, 271, 19, 16, 2619, 14907, 99761, 666, 2143, 262, 565, 455]
  ```

- Raw log: `raw/baseline-step35-run-gen-box1.log`

## Build receipts

- Local build: `raw/baseline-build.log`
- Box 1 isolated build: `raw/baseline-build-box1.log`
