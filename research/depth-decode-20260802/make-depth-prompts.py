#!/usr/bin/env python3
"""Cut per-model depth prompts from ONE document (p4-16k.txt, code-class) so the
four depth points share content class and differ only in length. Cut criterion:
llama-tokenize (the model's own gguf vocab) count == target exactly (binary search
on the char prefix; BOS included in the count, matching what both engines feed).
Writes depth-{512,2048,4096,6144}-{kat,q35,o35b}.txt + a manifest jsonl."""
import json, subprocess, sys, tempfile, os

TOK = "/home/avifenesh/projects/llama.cpp/build/bin/llama-tokenize"
SRC = "/home/avifenesh/projects/wt-depth-decode/research/e2e/prompts/p4-16k.txt"
OUT = "/home/avifenesh/projects/wt-depth-decode/research/depth-decode-20260802"
MODELS = {
    "kat": "/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf",
    "q35": "/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    "o35b": "/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf",
}
TARGETS = [512, 2048, 4096, 6144]

text = open(SRC, encoding="utf-8").read()

def ntok(model, chars):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text[:chars]); tmp = f.name
    try:
        out = subprocess.run([TOK, "-m", model, "-f", tmp, "--ids"],
                             capture_output=True, text=True, timeout=120)
        ids = json.loads(out.stdout.strip().splitlines()[-1])
        return len(ids)
    finally:
        os.unlink(tmp)

man = open(f"{OUT}/depth-prompts-manifest.jsonl", "w")
for name, model in MODELS.items():
    for tgt in TARGETS:
        lo, hi = 1, len(text)          # find max chars with ntok <= tgt
        # seed with the global ratio to cut iterations
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ntok(model, mid) <= tgt: lo = mid
            else: hi = mid - 1
        n = ntok(model, lo)
        path = f"{OUT}/depth-{tgt}-{name}.txt"
        open(path, "w", encoding="utf-8").write(text[:lo])
        row = {"model": name, "target": tgt, "chars": lo, "llama_tok_count": n, "file": os.path.basename(path)}
        man.write(json.dumps(row) + "\n"); man.flush()
        print(row, file=sys.stderr)
man.close()
