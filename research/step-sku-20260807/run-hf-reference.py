#!/usr/bin/env python3
"""HF-reference side of the step35 tokenizer byte-parity gate.

Loads the REAL `stepfun-ai/Step-3.7-Flash` fast tokenizer (tokenizer.json, sha-pinned in
raw/hf-ref-sha256.txt) with the `tokenizers` library — the same engine HF transformers
delegates to — and encodes every corpus case both with and without special-token handling:

  ids_special  = encode(text, add_special_tokens=True)   # TemplateProcessing prepends BOS 0
  ids_plain    = encode(text, add_special_tokens=False)

Output ref-ids.tsv: `<name>\t<ids_special csv>\t<ids_plain csv>` (empty field = empty ids).
The Rust side (`tok-parity` in memra-tokenizer) compares
  memra encode(text, add_special=true)  vs ids_special
  memra encode(text, add_special=false) vs ids_plain
token-for-token against the GGUF-built tokenizer on the box artifact.

Run: python3 research/step-sku-20260807/run-hf-reference.py <hf_ref_dir>
"""
import pathlib
import sys

from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus.tsv"
OUT = HERE / "ref-ids.tsv"

ref_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/step37-hf-ref")
tok = Tokenizer.from_file(str(ref_dir / "tokenizer.json"))

rows = []
for line in CORPUS.read_text().splitlines():
    name, hexbytes = line.split("\t")
    text = bytes.fromhex(hexbytes).decode("utf-8")
    ids_special = tok.encode(text, add_special_tokens=True).ids
    ids_plain = tok.encode(text, add_special_tokens=False).ids
    rows.append((name, ids_special, ids_plain))

with OUT.open("w") as f:
    for name, s, p in rows:
        f.write(f"{name}\t{','.join(map(str, s))}\t{','.join(map(str, p))}\n")

n_tok = sum(len(s) for _, s, _ in rows)
print(f"wrote {OUT}: {len(rows)} cases, {n_tok} special-mode tokens total")
# a couple of eyeball receipts for the log
for probe in ("digits-14", "special-im-start-end", "cjk-sentence"):
    for name, s, _ in rows:
        if name == probe:
            print(f"  {name}: {s}")
