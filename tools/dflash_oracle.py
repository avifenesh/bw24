#!/usr/bin/env python3
"""DFlash draft-forward oracle (bring-up step 2, DFLASH-BRINGUP-PLAN.md).

Runs the z-lab reference DFlashDraftModel on the real backbone-only checkpoint with
FIXED-SEED synthetic inputs (target_hidden + noise_embedding stand-ins), dumping every
layer's intermediates. The bw24 loader+forward must reproduce the final hidden states
(and intermediates, for bisecting) bit-close (f32 rel ~1e-3 class vs bf16 reference).

Synthetic inputs isolate the DRAFT math from target plumbing: target taps / embed /
lm_head are bw24-native pieces gated separately.

Usage (in ~/.venvs/torch):
  python tools/dflash_oracle.py /data/ai-ml/hf-models/dspark-gemma4-31b-draft/backbone-only \
      /data/cache/dflash-oracle.npz
"""
import sys, json, numpy as np, torch

ckpt_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/data/projects/dflash")
from dflash.model import DFlashDraftModel  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

cfg = json.load(open(f"{ckpt_dir}/config.json"))
config = Qwen3Config(**{k: v for k, v in cfg.items() if k != "architectures"})
config.num_target_layers = 62  # gemma-4-31B layer count (only used if target_layer_ids absent)
config._attn_implementation = "eager"

torch.manual_seed(0)
model = DFlashDraftModel(config)
from safetensors.torch import load_file  # noqa: E402
sd = load_file(f"{ckpt_dir}/model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)
model = model.to(torch.float32).eval()   # f32 reference (bw24 math is f32 accumulate)

H = config.hidden_size
NT = len(model.target_layer_ids)
CTX, BLK = 8, config.block_size
g = torch.Generator().manual_seed(42)
# scale ~1.0 like real residual-stream features
target_hidden = torch.randn(1, CTX, NT * H, generator=g, dtype=torch.float32)
noise_embedding = torch.randn(1, BLK, H, generator=g, dtype=torch.float32)
position_ids = torch.arange(0, CTX + BLK).unsqueeze(0)

with torch.inference_mode():
    # replicate DFlashDraftModel.forward but keep intermediates
    ctx = model.hidden_norm(model.fc(target_hidden))
    pos_emb = model.rotary_emb(noise_embedding, position_ids)
    inter = {"ctx_features": ctx.numpy()}
    h = noise_embedding
    from transformers.cache_utils import DynamicCache
    kv = DynamicCache()
    for i, layer in enumerate(model.layers):
        h = layer(hidden_states=h, target_hidden=ctx, attention_mask=None,
                  position_ids=position_ids, past_key_value=kv, use_cache=True,
                  cache_position=None, position_embeddings=pos_emb, is_causal=False)
        inter[f"layer{i}_out"] = h.numpy()
    out = model.norm(h)
    inter["final"] = out.numpy()

np.savez(out_path,
         target_hidden=target_hidden.numpy(),
         noise_embedding=noise_embedding.numpy(),
         position_ids=position_ids.numpy(),
         **inter)
print(f"saved {out_path}: final shape {out.shape}, |final| mean {out.abs().mean():.4f}")

# flat f32 dumps for the Rust parity bin (row-major, little-endian)
import os
d = os.path.dirname(out_path) or "."
for name, arr in [("target_hidden", target_hidden), ("noise_embedding", noise_embedding),
                  ("ctx_features", torch.from_numpy(inter["ctx_features"])),
                  ("final", out)]:
    a = (arr.numpy() if hasattr(arr, "numpy") else arr).astype(np.float32)
    a.tofile(f"{d}/dflash-{name}.f32")
for i in range(len(model.layers)):
    inter[f"layer{i}_out"].astype(np.float32).tofile(f"{d}/dflash-layer{i}_out.f32")
print("flat dumps written to", d)

# ---- layer-0 sub-stage dumps (parity bisect) ----
import torch.nn.functional as F
import os
from dflash.model import apply_rotary_pos_emb
l0 = model.layers[0]
with torch.inference_mode():
    hs = noise_embedding
    ctx_t = torch.from_numpy(inter["ctx_features"])
    xn = l0.input_layernorm(hs)
    xn.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_xn.f32")
    a = l0.self_attn
    bsz, q_len = xn.shape[:-1]; ctx_len = ctx_t.shape[1]
    q0 = a.q_proj(xn)
    q0.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_q0.f32")
    q = q0.view(bsz, q_len, -1, a.head_dim)
    q = a.q_norm(q)
    q.reshape(1, q_len, -1).numpy().astype(np.float32).tofile(f"{d}/dflash-l0_qn.f32")
    q = q.transpose(1, 2)
    k_ctx = a.k_proj(ctx_t); k_noise = a.k_proj(xn)
    v_ctx = a.v_proj(ctx_t); v_noise = a.v_proj(xn)
    k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, a.head_dim)
    v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, a.head_dim)
    k = a.k_norm(k).transpose(1, 2); v = v.transpose(1, 2)
    if not os.environ.get("DFLASH_NOROPE"):
        cos, sin = model.rotary_emb(hs, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q.transpose(1,2).reshape(1, q_len, -1).numpy().astype(np.float32).tofile(f"{d}/dflash-l0_q.f32")
    k.transpose(1,2).reshape(1, ctx_len+q_len, -1).numpy().astype(np.float32).tofile(f"{d}/dflash-l0_k.f32")
    # eager GQA attention, non-causal, no mask
    kk = k.repeat_interleave(a.num_key_value_groups, dim=1)
    vv = v.repeat_interleave(a.num_key_value_groups, dim=1)
    att = torch.softmax((q @ kk.transpose(-1, -2)) * a.scaling, dim=-1)
    ao = (att @ vv).transpose(1, 2).reshape(bsz, q_len, -1)
    ao.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_attn.f32")
    o = a.o_proj(ao)
    x1 = hs + o
    x1.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_x1.f32")
print("layer0 sub-stages dumped")
