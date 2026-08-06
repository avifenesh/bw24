#!/usr/bin/env python3
"""Q1 prefix-duplication census — analytic KV/session accounting for q9 (Qwen3.5-9B NVFP4).

Mirrors the EXACT allocation math in the runtime (verified against source):
  - crates/memra-kv/src/lib.rs Cache::new_inner — per full-attn layer
      k_tok_bytes = (kv_dim_k/32)*34  (q8_0),  v_tok_bytes = (kv_dim_v/32)*24 (q5_1)
      allocated at max_ctx up front (+8B tail pad, ignored here);
    per linear-attn layer: conv_state f32[conv_dim*(d_conv-1)] +
      ssm_state f32[d_state^2*num_v] + ssm_state_alt (same) — FIXED, ctx-independent.
  - crates/memra-engine/src/spec.rs MtpScratch::new — ONE KvLayer at cap (draft KV),
      same per-token bytes (draft head_count_kv=4, head_dim 256 — same as trunk).
  - crates/memra-server/src/worker.rs ctx_cap = max(prompt+max_new+8, MEMRA_CTX floor)
      (explicit max_tokens), or the serving ctx when omitted.

q9 geometry (GGUF metadata, Qwen3.5-9B-NVFP4-MTP-GGUF.gguf):
  block_count=33 (incl. 1 NextN), full_attention_interval=4 -> 8 full-attn trunk layers,
  25 linear layers (Cache loop covers il=0..32; il=32 classifies linear).
  head_count_kv=4, key_length=value_length=256 -> kv_dim = 1024 both planes.
  ssm: d_state=128, group_count=16, time_step_rank=32, conv_kernel=4
    -> conv_dim = 128*16*2 + 128*32 = 8192.
"""

MIB = 1 << 20

class Model:
    """Per-model geometry -> per-token KV bytes + fixed recurrent bytes."""
    def __init__(self, name, n_layer, nextn, kv_heads, head_dim, num_k, num_v):
        self.name = name
        kv_dim = head_dim * kv_heads
        k_tok = (kv_dim // 32) * 34          # q8_0
        v_tok = (kv_dim // 32) * 24          # q5_1
        # (il+1)%4==0 over il in 0..n_layer (Cache::new_inner walks ALL cfg.n_layer
        # incl. the NextN block; NextN classifies by the same interval rule)
        full = sum(1 for il in range(n_layer) if (il + 1) % 4 == 0)
        linear = n_layer - full
        self.trunk_tok = full * (k_tok + v_tok)
        self.draft_tok = k_tok + v_tok       # MtpScratch single KvLayer, same geometry
        self.per_tok = self.trunk_tok + self.draft_tok
        conv_dim = 128 * num_k * 2 + 128 * num_v
        d_conv = 4
        recur = 4 * (conv_dim * (d_conv - 1) + 2 * 128 * 128 * num_v)  # f32, incl alt
        self.recur_fixed = linear * recur
        self.full, self.linear, self.recur_per = full, linear, recur

    def banner(self):
        print(f"== {self.name}: {self.full} full-attn / {self.linear} linear layers ==")
        print(f"  per-token KV (trunk):        {self.trunk_tok} B/tok")
        print(f"  per-token KV (draft scratch): {self.draft_tok} B/tok")
        print(f"  per-token KV (spec session):  {self.per_tok} B/tok = {self.per_tok/1024:.2f} KiB/tok")
        print(f"  fixed recurrent per session:  {self.recur_fixed/MIB:.1f} MiB "
              f"({self.linear} x {self.recur_per/MIB:.2f} MiB)")
        print()

# q9  = Qwen3.5-9B NVFP4 (GGUF metadata): 33 blocks (32 trunk + 1 NextN), interval 4
Q9 = Model("q9  Qwen3.5-9B", 33, 1, 4, 256, 16, 32)
# k27 = Qwen3.6-27B NVFP4 (the daily/deployment model): 65 blocks (64+1 NextN), interval 4
K27 = Model("k27 Qwen3.6-27B", 65, 1, 4, 256, 16, 48)

def census(m, prefix_tok, tail_tok, gen_tok, ctx_alloc, c, card_mib, label):
    used = prefix_tok + tail_tok + gen_tok
    per_sess_ctx = ctx_alloc * m.per_tok
    per_sess_total = per_sess_ctx + m.recur_fixed
    total = c * per_sess_total
    # sealed-prefix-sharing ideal: prefix KV once, tails+gen+recurrent per session
    prefix_bytes = prefix_tok * m.per_tok
    dup = (c - 1) * prefix_bytes                       # what sharing would free
    slack = c * (ctx_alloc - used) * m.per_tok         # ladder/right-size stranding
    print(f"--- {label}: prefix={prefix_tok} tail={tail_tok} gen={gen_tok} "
          f"ctx_alloc={ctx_alloc} c={c} card={card_mib} MiB")
    print(f"  per-session alloc: {per_sess_total/MIB:.1f} MiB "
          f"(ctx KV {per_sess_ctx/MIB:.1f} + recurrent {m.recur_fixed/MIB:.1f})")
    print(f"  total KV alloc (c={c}):        {total/MIB:8.1f} MiB = {100*total/(card_mib*MIB):5.2f}% of card")
    print(f"  prefix duplication (c-1 copies): {dup/MIB:6.1f} MiB = {100*dup/(card_mib*MIB):5.2f}% of card")
    print(f"  ladder slack (alloc-used):     {slack/MIB:8.1f} MiB = {100*slack/(card_mib*MIB):5.2f}% of card")
    print()
    return dup, slack

CARD_5090 = 24463
CARD_96G = 96 * 1024

Q9.banner()
# ---- the pi/coding-agent shape (dogfood ctx4k trace: ~3.6-3.9k system+log prefix) ----
# tails ~120 tok unique question + nonce; gen 192-256 (explicit max_tokens).
# ctx_cap = max(prompt+max_new+8, 8192) = 8192 at 4k prefix.
# MEASURED anchor (logs/q1-*): c=8 served, prompt_tokens 3623 each, cached 0 — every
# session re-prefilled + re-stored the identical prefix; worker's own observed session
# VRAM cost 235MB(SI)=224MiB vs analytic 232.8MiB (-3.8%, free-delta granularity).
census(Q9, 3500, 123, 192, 8192, 8, CARD_5090,
       "q9 c=8 / MEASURED shape (3623 prompt, 192 gen) / MEMRA_CTX=8192 / 5090-24GB")
for c in (16, 32):
    census(Q9, 3500, 123, 192, 8192, c, CARD_96G,
           f"q9 c={c} / MEASURED 4k shape / MEMRA_CTX=8192 / 96GB")
# 8k prefix: ctx_cap = 8100+120+256+8 = 8484
census(Q9, 8100, 120, 256, 8484, 8, CARD_5090, "q9 c=8 / 8k prefix / right-sized / 5090-24GB")

# ---- deployment card extrapolation (96GB), same trace shape ----
for c in (16, 32):
    census(Q9, 8100, 120, 256, 8484, c, CARD_96G, f"q9 c={c} / 8k prefix / right-sized / 96GB")
# max_tokens omitted at MEMRA_CTX=32768 -> ctx_alloc=32768 (the omitted-max-tokens stranding)
for c in (16, 32):
    census(Q9, 8100, 120, 256, 32768, c, CARD_96G,
           f"q9 c={c} / 8k prefix / MEMRA_CTX=32768 omitted-max / 96GB")
# long-prefix upper shape: 32k sealed system+repo context (the RAG/agent-farm shape)
census(Q9, 32000, 120, 256, 33000, 16, CARD_96G, "q9 c=16 / 32k prefix / right-sized / 96GB")
census(Q9, 32000, 120, 256, 33000, 32, CARD_96G, "q9 c=32 / 32k prefix / right-sized / 96GB")

K27.banner()
# the deployment daily model on the 96GB card, same trace shapes
for c in (16, 32):
    census(K27, 3500, 123, 192, 8192, c, CARD_96G,
           f"k27 c={c} / MEASURED 4k shape / MEMRA_CTX=8192 / 96GB")
for c in (16, 32):
    census(K27, 8100, 120, 256, 8484, c, CARD_96G, f"k27 c={c} / 8k prefix / right-sized / 96GB")
census(K27, 32000, 120, 256, 33000, 16, CARD_96G, "k27 c=16 / 32k prefix / right-sized / 96GB")
census(K27, 32000, 120, 256, 33000, 32, CARD_96G, "k27 c=32 / 32k prefix / right-sized / 96GB")
