"""vLLM side of the engine decision bench. Single-sequence, N runs, GGUF-first.

Usage: python3 bench_vllm.py [--model PATH_OR_ID] [--runs 5] [--out vllm.json]
Defaults to the boot-session GGUF path; falls back to a w8a8 repo id via --model.
"""
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3.5-9B-Q8_0.gguf")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--ngen", type=int, default=512)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="vllm-single.json")
    # --spec-k K: enable vLLM's MTP speculative decoding (the model's own NextN head;
    # safetensors-only — the GGUF plugin has no nextn mapping). 0 = plain decode.
    ap.add_argument("--spec-k", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = {}
    if args.model.endswith(".gguf"):
        kwargs["tokenizer"] = args.tokenizer
    if args.spec_k > 0:
        kwargs["speculative_config"] = {"method": "mtp",
                                        "num_speculative_tokens": args.spec_k}
    # max_num_seqs: the protocol is single-stream; hybrid (Mamba/GDN) models cap decode
    # sequences by Mamba cache blocks (27B FP8: 784 < the 1024 default -> engine init fail).
    llm = LLM(model=args.model, max_model_len=args.prompt_tokens + args.ngen + 64,
              gpu_memory_utilization=0.90, enable_prefix_caching=False, max_num_seqs=8,
              **kwargs)
    tok = llm.get_tokenizer()
    # REAL-TEXT prompt (round 45): the fox-repeat prompt is the ledgered degenerate class —
    # its flat next-token distribution flips argmax across numeric arms (g26 prefill-vs-
    # decode MISMATCH at maxdiff ~11 on EVERY dispatch arm; real prompts MATCH). Both board
    # arms read the same file; ids truncate to prompt_tokens.
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "research/e2e/prompts/board-2048.txt")).read()
    ids = tok(text * 4)["input_ids"]
    prompt = tok.decode(ids[:args.prompt_tokens])

    rows = []
    for r in range(args.runs + 1):  # +1 warmup
        sp_pre = SamplingParams(max_tokens=1, temperature=0.0)
        t0 = time.perf_counter()
        llm.generate([prompt], sp_pre, use_tqdm=False)
        t_prefill = time.perf_counter() - t0

        sp = SamplingParams(max_tokens=args.ngen, temperature=0.0,
                            ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        t_total = time.perf_counter() - t0

        decode_tps = (args.ngen - 1) / max(t_total - t_prefill, 1e-9)
        prefill_tps = args.prompt_tokens / t_prefill
        if r == 0:
            continue
        rows.append({"run": r, "prefill_tps": round(prefill_tps, 1),
                     "decode_tps": round(decode_tps, 2),
                     "ttft_s": round(t_prefill, 4)})
        print(rows[-1], flush=True)

    med = sorted(x["decode_tps"] for x in rows)[len(rows) // 2]
    pre_med = sorted(x["prefill_tps"] for x in rows)[len(rows) // 2]
    out = {"engine": "vllm", "model": args.model, "ngen": args.ngen,
           "prompt_tokens": args.prompt_tokens, "spec_k": args.spec_k,
           "runs": rows, "decode_tps_median": med, "prefill_tps_median": pre_med,
           "e2e_tps_median": round(args.ngen / (args.ngen / med +
                                                args.prompt_tokens / pre_med), 2)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("median decode tok/s:", med)


if __name__ == "__main__":
    main()
