#!/usr/bin/env python3
"""Step-3.7-Flash size + fit math from committed receipts (no invented numbers).

Inputs: raw/hf-files-*-20260802.json (HF API ?blobs=true, pulled 2026-08-02) and
raw/config.json (stepfun-ai/Step-3.7-Flash, pulled 2026-08-02).
Run from research/step37-bringup-20260802/: python3 sizes-math.py
"""
import json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
GB = 1e9
GIB = 2**30

def blobs(name):
    d = json.load(open(os.path.join(R, name)))
    return {s["rfilename"]: s["size"] for s in d["siblings"]}

def tot(files, pred):
    return sum(sz for fn, sz in files.items() if pred(fn))

print("=== 1. Repo byte totals (HF API blobs, 2026-08-02) ===")
bf16 = blobs("hf-files-stepfun-ai-Step-3p7-Flash-20260802.json")
text = tot(bf16, lambda f: f.startswith("model-0") and f.endswith(".safetensors"))
vit = tot(bf16, lambda f: f.startswith("model-vit"))
print(f"BF16 safetensors text shards (incl. MTP shard 24): {text:,} B = {text/GB:.1f} GB")
print(f"BF16 safetensors vision shards:                    {vit:,} B = {vit/GB:.2f} GB")
print(f"BF16 safetensors total:                            {(text+vit)/GB:.1f} GB")

fp8 = blobs("hf-files-stepfun-ai-Step-3p7-Flash-FP8-20260802.json")
t8 = tot(fp8, lambda f: f.endswith(".safetensors"))
print(f"FP8 repo safetensors total (text+vit, MTP shard stays BF16): {t8/GB:.1f} GB")

nv = blobs("hf-files-stepfun-ai-Step-3p7-Flash-NVFP4-20260802.json")
tn = tot(nv, lambda f: f.endswith(".safetensors"))
print(f"NVFP4 repo safetensors total: {tn/GB:.1f} GB")

g = blobs("hf-files-stepfun-ai-Step-3p7-Flash-GGUF-20260802.json")
for q in ["BF16", "IQ4_XS", "Q4_K_S", "Q3_K_L", "Q3_K_M", "IQ3_XXS", "Q8_0"]:
    s = tot(g, lambda f, q=q: f.startswith(q + "/"))
    print(f"official GGUF {q:8s}: {s:,} B = {s/GB:.2f} GB = {s/GIB:.2f} GiB")
for f in ["Step3.7-flash-mtp-BF16.gguf", "Step3.7-flash-mtp-Q8_0.gguf",
          "mmproj-step3.7-flash-f16.gguf", "Step-3.7.imatrix.gguf"]:
    print(f"official {f}: {g[f]:,} B = {g[f]/GB:.2f} GB")

u = blobs("hf-files-unsloth-Step-3p7-Flash-GGUF-20260802.json")
for q in ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q3_K_XL", "UD-IQ3_XXS"]:
    s = tot(u, lambda f, q=q: f.startswith(q + "/"))
    print(f"unsloth GGUF {q:10s}: {s:,} B = {s/GB:.2f} GB = {s/GIB:.2f} GiB")

print("\n=== 2. Param count from config.json shapes (text model) ===")
cfg = json.load(open(os.path.join(R, "config.json")))["text_config"]
h = cfg["hidden_size"]                      # 4096
V = cfg["vocab_size"]                       # 128896
hd = cfg["head_dim"]                        # 128
nkv = cfg["num_attention_groups"]           # 8
nh_full = cfg["num_attention_heads"]        # 64
nh_swa = cfg["attention_other_setting"]["num_attention_heads"]  # 96
ffn = cfg["intermediate_size"]              # 11264
E = cfg["moe_num_experts"]                  # 288
eff = cfg["moe_intermediate_size"]          # 1280
sh = cfg["share_expert_dim"]                # 1280
L = cfg["num_hidden_layers"]                # 45
nmtp = cfg["num_nextn_predict_layers"]      # 3
lt = cfg["layer_types"]                     # 48 entries (45 main + 3 MTP)
assert len(lt) == L + nmtp
n_full = sum(1 for t in lt[:L] if t == "full_attention")
n_swa = L - n_full
moe_layers = len(cfg["moe_layers_enum"].split(","))  # 42
dense_layers = L - moe_layers                        # 3
print(f"main layers: {L} = {n_full} full + {n_swa} SWA(512); MTP layers: {nmtp} (all SWA per layer_types)")

def attn(nh):  # q,o: h*(nh*hd) each; k,v: h*(nkv*hd) each; gate: h*nh; norms tiny
    return 2*h*nh*hd + 2*h*nkv*hd + h*nh + 2*hd + 2*h

a = n_full*attn(nh_full) + n_swa*attn(nh_swa)
dense = dense_layers * 3*h*ffn
experts = moe_layers * E * 3*h*eff
shexp = moe_layers * 3*h*sh
router = moe_layers * (h*E + E)
emb = V*h; head = V*h
main = a + dense + experts + shexp + router + emb + head
mtp = nmtp * (attn(nh_swa) + 3*h*ffn + 2*h*h + V*h + 5*h)  # eh_proj [2h,h], shared head, norms
print(f"attention {a/1e9:.2f}B | dense FFN {dense/1e9:.2f}B | routed experts {experts/1e9:.2f}B "
      f"| shared experts {shexp/1e9:.2f}B | router {router/1e9:.3f}B | emb+head {(emb+head)/1e9:.2f}B")
print(f"main text model ~{main/1e9:.1f}B params; MTP ~{mtp/1e9:.2f}B; text total ~{(main+mtp)/1e9:.1f}B")
bf16_gguf = tot(g, lambda f: f.startswith("BF16/"))
print(f"cross-check: official BF16 GGUF {bf16_gguf/GB:.1f} GB / 2 B-per-param -> {bf16_gguf/2/1e9:.1f}B params (main only)")

print("\n=== 3. IQ4_XS honesty math ===")
iq4 = tot(g, lambda f: f.startswith("IQ4_XS/"))
ud4 = tot(u, lambda f: f.startswith("UD-IQ4_XS/"))
print(f"official IQ4_XS effective bpw: {iq4*8/main:.2f} (uniform IQ4_XS experts per header receipt)")
print(f"unsloth UD-IQ4_XS effective bpw: {ud4*8/main:.2f} "
      f"(gate/up experts IQ3_S=3.44bpw, down IQ4_XS, attn/shexp Q8_0 per header receipts)")
pred = (experts*4.25 + (a+shexp+dense)*4.5 + emb*4.25 + head*6.56 + router*32) / 8
print(f"predicted uniform-IQ4_XS size from shapes: {pred/GB:.1f} GB (actual official: {iq4/GB:.2f} GB)")

print("\n=== 4. KV-cache budget (8 KV heads x 128 head_dim, K+V) ===")
per_tok = nkv*hd*2  # elems per token per layer
for name, b in [("FP16", 2), ("FP8/q8", 1)]:
    ptl = per_tok*b
    print(f"-- {name}: {ptl} B/token/layer")
    for ctx in [32768, 131072, 262144]:
        full_kv = n_full*ctx*ptl
        swa_ring = n_swa*(512)*ptl  # window-sized (ring) allocation
        swa_maxctx = n_swa*ctx*ptl  # memra's CURRENT allocation policy (max_ctx-sized)
        print(f"   ctx {ctx//1024:>3}K: full-attn {full_kv/GIB:6.2f} GiB | SWA ring {swa_ring/GIB:6.3f} GiB "
              f"| SWA at max_ctx (memra today) {swa_maxctx/GIB:6.2f} GiB | total today {(full_kv+swa_maxctx)/GIB:6.2f} GiB")

print("\n=== 5. 4x RTX 5090 fit (128 GiB = 137.4 GB total VRAM) ===")
vram = 4*32*GIB
mtp_q8 = g["Step3.7-flash-mtp-Q8_0.gguf"]
for label, w in [("official IQ4_XS + MTP Q8_0", iq4+mtp_q8), ("unsloth UD-IQ4_XS + MTP Q8_0", ud4+mtp_q8)]:
    print(f"{label}: weights {w/GB:.1f} GB = {w/GIB:.1f} GiB -> headroom {(vram-w)/GIB:.1f} GiB "
          f"before KV/activations/runtime")

print("\n=== 6. 8-bit / NVFP4 card-count math on 32 GiB 5090s (serving-quant standard) ===")
import math
q8_gguf = tot(g, lambda f: f.startswith("Q8_0/"))
kv128_fp8 = 11.25*GIB   # from section 4: 128K ctx, FP8 KV, memra's max_ctx SWA allocation
runtime = 7*GB          # StepFun's own llama.cpp "Runtime Overhead: ~7 GB" (GGUF README)
for label, w in [("FP8 safetensors (official repo)", t8),
                 ("Q8_0 official GGUF", q8_gguf),
                 ("NVFP4 safetensors (official repo)", tn)]:
    wg = w/GIB
    n_w = math.ceil(wg/32)
    need = w + kv128_fp8 + runtime
    n_all = math.ceil(need/GIB/32)
    print(f"{label}: {w/GB:.1f} GB = {wg:.1f} GiB -> {n_w} cards weights-only "
          f"(headroom {n_w*32-wg:.1f} GiB); + 128K FP8-KV + ~7 GB runtime = "
          f"{need/GIB:.0f} GiB -> {n_all} cards (next card up for real headroom: {n_all+1})")
