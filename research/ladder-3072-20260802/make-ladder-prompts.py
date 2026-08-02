#!/usr/bin/env python3
"""lane/ladder-3072: cut d1024/d3072 prompts for kat+q35 from the SAME document
(p4-16k.txt) with the depth-decode lane's exact criterion (model-tokenizer count
== target, binary search on char prefix, BOS included)."""
import json, subprocess, sys, tempfile, os

TOK = "/home/avifenesh/projects/llama.cpp/build/bin/llama-tokenize"
SRC = "/home/avifenesh/projects/wt-ladder-3072/research/e2e/prompts/p4-16k.txt"
OUT = "/home/avifenesh/projects/wt-ladder-3072/research/ladder-3072-20260802"
MODELS = {
    "kat": "/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf",
    "q35": "/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
}
TARGETS = [1024, 3072]

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

man = open(f"{OUT}/ladder-prompts-manifest.jsonl", "w")
for name, model in MODELS.items():
    for tgt in TARGETS:
        lo, hi = 1, len(text)
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
