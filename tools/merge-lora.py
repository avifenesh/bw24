#!/usr/bin/env python3
"""merge-lora.py — LoRA adapter -> standalone merged BF16 HF checkpoint (loop gap #1,
research/training-readiness-20260805/DESIGN.md §2.1).

memra has zero LoRA support and GGUF adapter files don't ride our conversion pipeline:
merge-then-convert is the only path from a finetune to a servable artifact. This tool is
the scripted pre-conversion step:

    PEFT merge_and_unload() -> bf16 safetensors shards + index
    + tokenizer / chat template / generation & preprocessor configs copied
    + SIDECAR TENSOR CARRYOVER: any base-checkpoint tensor the transformers model class
      does not materialize (Qwen3.5/3.6: the whole `mtp.*` MTP block) is copied verbatim
      from the base shards into an extra shard and re-indexed. Without this the merged
      dir silently loses the MTP head and the converted GGUF has no nextn block — no
      spec serving, no drafter (found on the train-loop pilot, 2026-08-05).

Weights are written from `model.state_dict()` directly (manual shards + index), NOT via
`save_pretrained` — transformers 5.5 `save_pretrained` on Qwen3_5ForConditionalGeneration
re-prefixed every LM tensor to `model.language_model.language_model.language_model.*`
(triple-nested), so the converter would have read base weights via carryover and silently
dropped the finetune (train-loop pilot find #2, 2026-08-05).

usage: merge-lora.py <base_hf_dir> <adapter_dir> <out_dir>

The merge runs on CPU (bf16); no GPU lock needed. LoRA never touches the carried-over
tensors, so byte-verbatim copy is exact by construction.
"""
import json
import shutil
import sys
from pathlib import Path

SHARD_BYTES = 5 * 10**9


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    base_dir, adapter_dir, out_dir = (Path(p) for p in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.loads((base_dir / "config.json").read_text())
    arch = cfg["architectures"][0]
    model_cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
    print(f"[merge-lora] base={base_dir} class={getattr(model_cls, '__name__', model_cls)}")
    model = model_cls.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None
    )
    print(f"[merge-lora] adapter={adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    # --- manual bf16 shard write with the module-tree tensor names ---
    sd = model.state_dict()
    # tied embeddings: state_dict may alias lm_head to the embed table; keep base's key set
    def index_keys(d: Path) -> dict:
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            return json.loads(idx.read_text())["weight_map"]
        single = d / "model.safetensors"
        with safe_open(single, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}

    base_map = index_keys(base_dir)
    shards: list[dict] = [{}]
    sizes = [0]
    for k, t in sd.items():
        t = t.contiguous()
        b = t.numel() * t.element_size()
        if sizes[-1] + b > SHARD_BYTES and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = t
        sizes[-1] += b

    # carryover: base tensors the model class never materialized (e.g. mtp.*)
    missing = sorted(k for k in base_map if k not in sd)
    if missing:
        prefixes = sorted({k.split(".")[0] for k in missing})
        print(f"[merge-lora] carryover: {len(missing)} base tensors not in the model class "
              f"(prefixes: {prefixes}) — copying verbatim")
        by_shard: dict = {}
        for k in missing:
            by_shard.setdefault(base_map[k], []).append(k)
        for shard_file, keys in by_shard.items():
            with safe_open(base_dir / shard_file, framework="pt") as f:
                for k in keys:
                    t = f.get_tensor(k)
                    b = t.numel() * t.element_size()
                    if sizes[-1] + b > SHARD_BYTES and shards[-1]:
                        shards.append({})
                        sizes.append(0)
                    shards[-1][k] = t
                    sizes[-1] += b

    n = len(shards)
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, out_dir / fname, metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        print(f"[merge-lora] wrote {fname} ({len(shard)} tensors, {sizes[i-1]/2**30:.1f} GiB)")
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map}, indent=2))

    # sanity: merged key set must equal the base key set (same arch, same tensors)
    extra = sorted(set(weight_map) - set(base_map))
    lost = sorted(set(base_map) - set(weight_map))
    if extra or lost:
        print(f"[merge-lora] FATAL: key-set drift vs base — extra={extra[:5]} lost={lost[:5]}")
        return 1

    model.config.save_pretrained(out_dir)
    tok = transformers.AutoTokenizer.from_pretrained(base_dir)
    tok.save_pretrained(out_dir)
    for aux in (
        "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json", "merges.txt", "vocab.json",
    ):
        src = base_dir / aux
        if src.exists() and not (out_dir / aux).exists():
            shutil.copy2(src, out_dir / aux)

    print(f"[merge-lora] DONE: {out_dir} ({len(weight_map)} tensors, key-set == base)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
