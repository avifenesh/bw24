#!/usr/bin/env python3
"""Extract a standalone MTP draft GGUF from a full model GGUF.

The draft carries the NextN block (blk.<n_trunk>.*), the lm head + output_norm,
and token_embd — byte-verbatim from the model file (zero requant), so the
external draft is numerically IDENTICAL to the embedded head it mirrors.
Trim + head-requant happen downstream (tools/trim_draft_head.py, llama-quantize).

usage: extract_mtp_draft.py <model.gguf> <out-draft.gguf>
"""
import sys

sys.path.insert(0, "/data/projects/llama.cpp/gguf-py")
import gguf  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_path, out_path = sys.argv[1:3]

    reader = gguf.GGUFReader(model_path)
    fields = {f.name: f for f in reader.fields.values()}
    arch = fields["general.architecture"].contents()
    n_layer = fields[f"{arch}.block_count"].contents()
    nextn_key = f"{arch}.nextn_predict_layers"
    assert nextn_key in fields, f"{model_path} has no {nextn_key} — no MTP block to extract"
    nextn = fields[nextn_key].contents()
    n_trunk = n_layer - nextn
    keep_prefix = tuple(f"blk.{il}." for il in range(n_trunk, n_layer))
    keep_exact = {"output.weight", "output_norm.weight", "token_embd.weight"}

    writer = gguf.GGUFWriter(out_path, arch)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    tensors = [t for t in reader.tensors
               if t.name in keep_exact or t.name.startswith(keep_prefix)]
    assert any(".nextn." in t.name for t in tensors), "no nextn glue tensors found"
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes,
                               t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    total = sum(t.n_bytes for t in tensors)
    print(f"extracted {len(tensors)} tensors (blk.{n_trunk}..{n_layer - 1} + head/embd), "
          f"{total >> 20}MB -> {out_path}")


if __name__ == "__main__":
    main()
