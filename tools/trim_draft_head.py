#!/usr/bin/env python3
"""Bake an FR-Spec vocab trim into a standalone MTP draft GGUF.

The draft's dominant cost is its lm_head projection over the full vocab; a
frequency-ranked top-N subset cuts that read without touching correctness
(verification runs the full vocab on the target — lossless). This is the
PRODUCER for the trimmed-draft artifact class (`*-frspec<N>.gguf`): it gathers
the ranked rows of the draft's `output.weight` at the BYTE level (quantized
rows are independent — zero requant, any row-aligned quant type) and embeds
the `d2t` (draft-to-target id) tensor the loaders consume.

Ranks come from frspec-owngen (the model's OWN generations — rank files are
vocab+distribution artifacts; foreign ranks measured -12 acceptance pts on the
unsloth 27B) or frspec-rank (corpus tokenize).

usage: trim_draft_head.py <draft.gguf> <ranks.txt> <out.gguf> [topN]
"""
import sys

import numpy as np

sys.path.insert(0, "/data/projects/llama.cpp/gguf-py")
import gguf  # noqa: E402


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    draft_path, ranks_path, out_path = sys.argv[1:4]
    top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 32768

    ids = np.array([int(l) for l in open(ranks_path) if l.strip()][:top_n], dtype=np.int32)
    assert len(ids) == top_n, f"ranks file has {len(ids)} ids < topN {top_n}"

    reader = gguf.GGUFReader(draft_path)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(out_path, arch)

    for field in reader.fields.values():
        # the writer emits its own header/arch fields
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # trimmed output.weight: byte-level row gather (rows are independent in every
    # row-aligned quant; same gather the engine's BW24_FRSPEC_TRIM loader does).
    out_t = next(t for t in reader.tensors if t.name == "output.weight")
    n_vocab = int(out_t.shape[-1]) if len(out_t.shape) > 1 else int(out_t.shape[0])
    row_bytes = out_t.n_bytes // n_vocab
    assert out_t.n_bytes % n_vocab == 0, "output.weight rows not byte-aligned"
    flat = np.ascontiguousarray(out_t.data).view(np.uint8).reshape(n_vocab, row_bytes)
    trimmed = np.ascontiguousarray(flat[ids])
    # byte shape for add_tensor_info: same per-row byte layout, top_n rows
    trimmed_shape = (top_n, out_t.data.shape[-1]) if out_t.data.ndim > 1 else (trimmed.nbytes,)
    trimmed = trimmed.reshape(trimmed_shape).astype(out_t.data.dtype, copy=False) \
        if out_t.data.ndim > 1 else trimmed.reshape(-1).view(out_t.data.dtype)
    print(f"output.weight: {n_vocab} -> {top_n} rows ({out_t.tensor_type.name}, "
          f"{row_bytes}B/row, {out_t.n_bytes >> 20}MB -> {trimmed.nbytes >> 20}MB)")

    d2t = ids  # i32 [top_n]

    for t in reader.tensors:
        if t.name == "output.weight":
            writer.add_tensor_info(t.name, trimmed.shape, trimmed.dtype, trimmed.nbytes,
                                   t.tensor_type)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes,
                                   t.tensor_type)
    writer.add_tensor_info("d2t", d2t.shape, d2t.dtype, d2t.nbytes,
                           gguf.GGMLQuantizationType.I32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(trimmed if t.name == "output.weight" else t.data,
                                 tensor_endianess=reader.endianess)
    writer.write_tensor_data(d2t, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {out_path} (d2t {top_n} ids embedded)")


if __name__ == "__main__":
    main()
