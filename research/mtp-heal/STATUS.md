# MTP-HEAL MEASUREMENT RUN STATUS

**Started**: 2026-07-09 07:18 UTC  
**Branch**: lane/mtprun (worktree /home/avifenesh/projects/bw24-mtprun)  
**Binaries**: /home/avifenesh/projects/bw24/target/release/run-spec (main checkout)

## Configuration

### ARM A: bf16 full-precision CEILING
- Model: /data/ai-ml/hf-models/qwen35-9b-hf
- Env: `FULL_PREC=1` (BF16 resident + f32 oracle path)
- Expected: ~8 tok/s (SLOW)
- Output: research/mtp-heal/out-bf16.jsonl
- Log: research/mtp-heal/arm_a.log

### ARM B: NVFP4 quantized
- Model: /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
- Env: default fast path
- Expected: ~100+ tok/s
- Output: research/mtp-heal/out-nvfp4.jsonl
- Log: research/mtp-heal/arm_b.log

## Battery Protocol

- Prompts: p1 (code-short), p2 (code-medium), p3 (agentic-long)
- K values: 1, 2, 3, 4
- N (runs per cell): 3
- NGEN: 128 tokens
- Agent loop: 8 turns, accumulative, K=3, NGEN=256

Total cells per arm: (3 prompts × 4 K × 3 runs) + (8 agent-loop turns) = 36 + 8 = 44 rows

## Progress (as of 07:19)

### ARM A (bf16)
- Status: RUNNING (first run p1/K=1)
- Rows completed: 0/44
- Process: PID 2105867 (CPU 97%, MEM 6.9GB)
- ETA: ~6-8 hours (based on ~8 tok/s × 128 tokens × 36 spec runs + agent loop)

### ARM B (nvfp4)
- Status: RUNNING (p2/K=1 starting)
- Rows completed: 12/44 (p1 complete)
- ETA: ~1-2 hours

## Monitoring

Check progress:
```bash
wc -l research/mtp-heal/out-*.jsonl
tail -5 research/mtp-heal/arm_*.log
```

Check running processes:
```bash
ps aux | grep run-spec | grep -v grep
nvidia-smi
```

## Next Steps (once both complete)

1. Verify row counts (44 rows each)
2. Run delta analysis:
   ```bash
   python3 tools/acceptance_delta.py \
     research/mtp-heal/out-bf16.jsonl \
     research/mtp-heal/out-nvfp4.jsonl \
     > research/mtp-heal/delta-9b-bf16-vs-nvfp4.md
   ```
3. Write JSONL rows to research/tune-data/rig5090.jsonl
4. Commit results on lane/mtprun
5. Report findings

## Notes

- ARM A is EXPECTED to be slow (~8 tok/s) — full-precision path uses bf16-resident weights + f32 compute
- GPU memory usage: ~9GB (bf16 model loaded)
- Both arms running in parallel (separate processes)
- No TRIM (natural MTP heads for both models per protocol)
