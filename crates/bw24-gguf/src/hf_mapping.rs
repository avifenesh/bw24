//! HF (transformers) tensor-name -> ggml tensor-name mapping.
//!
//! The bw24 engine ONLY ever asks for ggml-style names (e.g. `blk.3.attn_q.weight`,
//! hardcoded in model.rs / hybrid.rs). The safetensors source resolves on demand by
//! translating a requested **ggml name** into the **HF name** to look up in the file.
//!
//! Direction is ggml -> HF (one lookup per requested tensor; no full-table build needed).
//!
//! Coverage:
//!   * dense transformer (Qwen3 / Llama / OLMoE-attention style) — `ggml_to_hf`, zero-copy.
//!   * MoE stacked experts — per-expert ggml name `blk.{il}.ffn_{proj}_exps.{e}.weight` ->
//!     `model.layers.{il}.mlp.experts.{e}.{proj}_proj.weight` (gathered by HostExps::load_from_source).
//!   * hybrid (qwen35) linear-attn SSM tensors — `resolve_ggml`, with the value transforms
//!     (`-exp(A_log)`, norm `+1`, conv1d squeeze, V-reorder) materialized as owned buffers.
//!
//! The SSM name map + transforms are cited against llama.cpp's converter (conversion/qwen.py,
//! gguf-py/gguf/tensor_mapping.py) — see `resolve_ggml` and `TransformKind`.

use crate::config::{Arch, ModelConfig};

/// Translate a requested ggml tensor name into the HF safetensors name (DENSE tensors only —
/// a plain rename with no value transform). Returns `None` if the ggml name has no known HF
/// equivalent for this arch (e.g. SSM tensors, which go through `resolve_ggml`).
pub fn ggml_to_hf(ggml: &str, arch: &Arch) -> Option<String> {
    // Top-level tensors (arch-independent across Llama/Qwen dense).
    match ggml {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }

    // Per-layer: "blk.{il}.{suffix}".
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;

    // MiniMax-M3 MoE-side names live under `block_sparse_moe.` (Mixtral-style), with DeepSeek-V3
    // routing extras (e_score_correction_bias) and `shared_experts.{p}_proj` (plural, no gate_inp).
    // Dense FFN layers (moe_layer_freq==0) keep the standard `mlp.{p}_proj` names below.
    if arch.is_minimax() {
        let m3_suffix: Option<&str> = match suffix {
            "ffn_gate_inp.weight" => Some("block_sparse_moe.gate.weight"),
            // llama.cpp ggml name for the DeepSeek-V3 selection bias is `exp_probs_b.bias`.
            "exp_probs_b.bias" => Some("block_sparse_moe.e_score_correction_bias"),
            "ffn_gate_shexp.weight" => Some("block_sparse_moe.shared_experts.gate_proj.weight"),
            "ffn_up_shexp.weight" => Some("block_sparse_moe.shared_experts.up_proj.weight"),
            "ffn_down_shexp.weight" => Some("block_sparse_moe.shared_experts.down_proj.weight"),
            _ => None,
        };
        if let Some(s) = m3_suffix {
            return Some(format!("model.layers.{il}.{s}"));
        }
    }

    // Hy3 REAP tensors use dense layer-0 `mlp.{gate,up,down}_proj`, then routed MoE layers with
    // `mlp.router.*`, `mlp.shared_mlp.*`, and stacked `mlp.switch_mlp.*` projections.
    if arch.is_hy3() {
        let hy3_suffix: Option<&str> = match suffix {
            "ffn_gate_inp.weight" => Some("mlp.router.gate.weight"),
            // Current tencent/Hy3 checkpoints store the selection-only correction beside the
            // router module. Preview checkpoints used `mlp.router.expert_bias`; the safetensors
            // source keeps that older spelling as a lookup fallback.
            "exp_probs_b.bias" => Some("mlp.expert_bias"),
            "ffn_gate_shexp.weight" => Some("mlp.shared_mlp.gate_proj.weight"),
            "ffn_up_shexp.weight" => Some("mlp.shared_mlp.up_proj.weight"),
            "ffn_down_shexp.weight" => Some("mlp.shared_mlp.down_proj.weight"),
            "ffn_gate_exps.weight" => Some("mlp.switch_mlp.gate_proj.weight"),
            "ffn_up_exps.weight" => Some("mlp.switch_mlp.up_proj.weight"),
            "ffn_down_exps.weight" => Some("mlp.switch_mlp.down_proj.weight"),
            _ => None,
        };
        if let Some(s) = hy3_suffix {
            return Some(format!("model.layers.{il}.{s}"));
        }
    }

    let hf_suffix: &str = match suffix {
        // attention block
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight", // qwen3 / qwen3moe
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        // some HF checkpoints carry attention biases (Qwen2 style) — map them too.
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        // FFN (dense SwiGLU)
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        // MoE router + shared expert (qwen3_moe dense-side tensors; experts handled separately).
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
        "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
        _ => return None,
    };
    Some(format!("model.layers.{il}.{hf_suffix}"))
}

/// HF name for a single expert's projection weight (for the MoE gather+concat path).
/// ggml stacks all experts into one 3D tensor; HF stores them separately. `proj` is one of
/// "gate", "up", "down". Two layouts:
///   * qwen3moe / olmoe: `mlp.experts.{e}.{gate,up,down}_proj.weight`
///   * MiniMax-M3 (Mixtral-style): `block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`
///     (w1=gate, w2=down, w3=up — the Mixtral convention)
pub fn hf_expert_name(il: u32, e: u32, proj: &str, arch: &Arch) -> String {
    if arch.is_minimax() {
        let w = match proj {
            "gate" => "w1",
            "down" => "w2",
            "up" => "w3",
            other => panic!("unknown expert proj {other}"),
        };
        return format!("model.layers.{il}.block_sparse_moe.experts.{e}.{w}.weight");
    }
    let p = match proj {
        "gate" => "gate_proj",
        "up" => "up_proj",
        "down" => "down_proj",
        other => panic!("unknown expert proj {other}"),
    };
    format!("model.layers.{il}.mlp.experts.{e}.{p}.weight")
}

/// Resolved HF target for a requested ggml name.
pub enum HfTarget {
    /// A plain rename — borrow the on-disk bytes zero-copy.
    Plain(String),
    /// A rename PLUS a value transform that must materialize an owned f32 buffer (§2.1).
    Transform { hf: String, kind: TransformKind },
}

/// The value transforms the qwen35 HF->GGUF converter applies (llama.cpp conversion/qwen.py).
/// All operate on the dequantized-to-f32 tensor and return the GGUF-equivalent owned bytes.
#[derive(Clone, Copy)]
pub enum TransformKind {
    /// `ssm_a` <- `A_log`: elementwise `-exp(x)`, then V-head reorder (qwen.py:296-297, 496-503).
    NegExpReorderHeads,
    /// `ssm_dt.bias` <- `dt_bias`: V-head reorder only (qwen.py:496-503).
    ReorderHeads,
    /// `ssm_norm.weight` <- `linear_attn.norm.weight`: NO +1, NO reorder (qwen.py:302-303 carve-out).
    /// Materialized only to keep a single owned-arm code path; value is unchanged.
    Identity,
    /// qwen35 norm `+1` (every `*norm.weight` EXCEPT linear_attn.norm) (qwen.py:302-303).
    NormPlusOne,
    /// `ssm_conv1d.weight` <- `conv1d.weight`: squeeze `[C,1,K]->[C,K]`, V-channel reorder (qwen.py:300-301,505-512).
    Conv1dSqueezeReorder,
    /// `attn_qkv.weight` <- `in_proj_qkv.weight`: reorder ONLY the V row-block (rows 4096:8192) (qwen.py:478-486).
    QkvVReorderRows,
    /// `attn_gate.weight` <- `in_proj_z.weight`: reorder ALL rows (qwen.py:488-490).
    ZReorderRows,
    /// `ssm_alpha`/`ssm_beta` <- `in_proj_a`/`in_proj_b`: reorder ALL rows, head_dim=1 (qwen.py:492-494).
    AbReorderRows,
    /// `ssm_out.weight` <- `out_proj.weight`: reorder COLUMNS (in-dim), head_dim=128 (qwen.py:514-516).
    OutReorderCols,
}

impl TransformKind {
    /// Apply the transform to `data` (dequantized f32, row-major matching the HF shape whose ggml
    /// `ne` is `ne_in`). Returns the new ggml `ne` and the owned little-endian f32 bytes.
    pub fn apply(&self, data: &mut [f32], ne_in: Vec<u64>, cfg: &ModelConfig) -> (Vec<u64>, Vec<u8>) {
        let (nk, nv, hk, hv) = head_params(cfg);
        match self {
            TransformKind::Identity => (ne_in, f32_to_le(data)),
            TransformKind::NormPlusOne => {
                for x in data.iter_mut() { *x += 1.0; }
                (ne_in, f32_to_le(data))
            }
            TransformKind::NegExpReorderHeads => {
                for x in data.iter_mut() { *x = -x.exp(); }
                // 1D [nv]: per-head reorder, head_dim=1.
                let out = reorder_v_heads_axis0(data, nv, nk, 1);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::ReorderHeads => {
                let out = reorder_v_heads_axis0(data, nv, nk, 1);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::AbReorderRows => {
                // in_proj_a/b: HF [out=nv, in=hidden] -> ne [hidden, nv]. Reorder the `nv` out-rows
                // (head_dim=1). data is row-major [nv][hidden]; row index = V-head.
                let in_f = ne_in[0] as usize;
                let out = reorder_rows_v(data, nv, nk, 1, in_f, 0, nv);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::ZReorderRows => {
                // in_proj_z: HF [out=value_dim, in=hidden] -> ne [hidden, value_dim]. Reorder ALL
                // value_dim out-rows, head_dim=hv. data row-major [value_dim][hidden].
                let in_f = ne_in[0] as usize;
                let total_out = ne_in[1] as usize;       // value_dim = nv*hv
                let out = reorder_rows_v(data, nv, nk, hv, in_f, 0, total_out);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::QkvVReorderRows => {
                // in_proj_qkv: HF [out=conv_dim, in=hidden] -> ne [hidden, conv_dim].
                // q rows [0, qk), k rows [qk, 2qk) UNTOUCHED; V rows [2qk, conv_dim) reordered (head_dim=hv).
                let in_f = ne_in[0] as usize;
                let conv_dim = ne_in[1] as usize;
                let qk = nk * hk;                          // q_dim = k_dim = num_k_heads*head_k_dim
                let out = reorder_rows_v(data, nv, nk, hv, in_f, 2 * qk, conv_dim);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::Conv1dSqueezeReorder => {
                // conv1d HF [C, 1, K] -> ne_in reversed = [K, 1, C]. Squeeze -> ggml [K, C] (ne[K,C]).
                // data row-major [C][1][K] == [C][K]; channels [0,2qk) untouched, V channels reordered (hv).
                let k = ne_in[0] as usize;                 // kernel taps (4)
                // ne_in could be [K,1,C] or [K,C]; channel count is the product of the rest.
                let c: usize = ne_in[1..].iter().map(|&d| d as usize).product();
                let qk = nk * hk;
                let out = reorder_rows_v(data, nv, nk, hv, k, 2 * qk, c);
                (vec![k as u64, c as u64], f32_to_le(&out))
            }
            TransformKind::OutReorderCols => {
                // out_proj: HF [out=hidden, in=value_dim] -> ne [value_dim, hidden]. Reorder the
                // `value_dim` INPUT columns (head_dim=hv). data row-major [hidden][value_dim].
                let value_dim = ne_in[0] as usize;          // in-features
                let hidden = ne_in[1] as usize;             // out-features
                let out = reorder_cols_v(data, hidden, value_dim, nv, nk, hv);
                (ne_in, f32_to_le(&out))
            }
        }
    }
}

impl TransformKind {
    /// Apply this transform to a REPACKED GGUF NVFP4 weight WITHOUT dequantizing, when it is a pure
    /// structural V-head permutation (qkv/z/a/b out-row reorder, out_proj in-column reorder). Keeps
    /// the weight NVFP4 (no ~8x f32 blow-up). Returns `Some((ne, packed_bytes))` for the permutable
    /// kinds, `None` for the value transforms (`-exp`, `+1`, conv1d squeeze, identity) which must go
    /// through f32. `packed`/`out_f`/`in_f` describe the repacked weight (`ne = [in_f, out_f]`).
    pub fn apply_nvfp4(&self, packed: &[u8], out_f: usize, in_f: usize, cfg: &ModelConfig)
                       -> Option<(Vec<u64>, Vec<u8>)> {
        use crate::nvfp4_repack::{reorder_cols_nvfp4, reorder_rows_nvfp4};
        let (nk, nv, hk, hv) = head_params(cfg);
        let row_bytes = (in_f / 64) * 36;
        let ne = vec![in_f as u64, out_f as u64];
        match self {
            TransformKind::QkvVReorderRows => {
                // V band = out-rows [2*qk, conv_dim); q/k rows untouched (copied through).
                let qk = nk * hk;
                Some((ne, reorder_rows_nvfp4(packed, out_f, row_bytes, nv, nk, hv, 2 * qk, out_f)))
            }
            TransformKind::ZReorderRows => {
                // all value_dim=out_f out-rows reordered (head_dim=hv).
                Some((ne, reorder_rows_nvfp4(packed, out_f, row_bytes, nv, nk, hv, 0, out_f)))
            }
            TransformKind::AbReorderRows => {
                // out-rows reordered, head_dim=1 (a/b: out_f == nv).
                Some((ne, reorder_rows_nvfp4(packed, out_f, row_bytes, nv, nk, 1, 0, out_f)))
            }
            TransformKind::OutReorderCols => {
                // in-columns (value_dim) reordered, head_dim=hv (block-aligned: hv % 64 == 0).
                Some((ne, reorder_cols_nvfp4(packed, out_f, in_f, nv, nk, hv)))
            }
            // value transforms (operate on tiny BF16 tensors) — no NVFP4 fast path.
            _ => None,
        }
    }
}

/// (num_k_heads, num_v_heads, head_k_dim, head_v_dim) from cfg.ssm / cfg fields (qwen35).
fn head_params(cfg: &ModelConfig) -> (usize, usize, usize, usize) {
    let ssm = cfg.ssm.as_ref().expect("ssm config for hybrid transform");
    let nk = ssm.group_count as usize;       // linear_num_key_heads = 16
    let nv = ssm.time_step_rank as usize;    // linear_num_value_heads = 32 (stored here, see config.rs)
    let hk = ssm.state_size as usize;        // linear_key_head_dim = 128
    let hv = hk;                              // linear_value_head_dim == key (qwen35: both 128)
    (nk, nv, hk, hv)
}

/// Core V-head permutation primitive for a flat axis of `num_v_heads*head_dim` elements
/// (qwen.py:366-376 `_reorder_v_heads`). Output enumerates `j` (within-K-group) OUTER, `g`
/// (K-group) INNER: output V-head slot `j*num_k_heads + g` <- source HF V-head `g*num_v_per_k + j`.
/// `block` is one contiguous V axis (length num_v_heads*head_dim); returns the permuted copy.
fn reorder_v_axis(block: &[f32], num_v_heads: usize, num_k_heads: usize, head_dim: usize) -> Vec<f32> {
    let num_v_per_k = num_v_heads / num_k_heads;
    debug_assert_eq!(block.len(), num_v_heads * head_dim);
    let mut out = vec![0f32; block.len()];
    for j in 0..num_v_per_k {
        for g in 0..num_k_heads {
            let dst_head = j * num_k_heads + g;
            let src_head = g * num_v_per_k + j;
            let dst = dst_head * head_dim;
            let src = src_head * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&block[src..src + head_dim]);
        }
    }
    out
}

/// 1D tensor [num_v_heads*head_dim] (here head_dim is the per-head element count). Thin wrapper.
fn reorder_v_heads_axis0(data: &[f32], num_v_heads: usize, num_k_heads: usize, head_dim: usize) -> Vec<f32> {
    reorder_v_axis(data, num_v_heads, num_k_heads, head_dim)
}

/// Reorder the V-head rows of a row-major [out_rows][in_f] matrix in-place over the row-band
/// `[row_lo, row_hi)` (which must span `num_v_heads*head_dim` rows). Rows outside the band copy
/// through unchanged. Used for in_proj_qkv (V band), in_proj_z / a / b (whole band), conv1d (V band).
fn reorder_rows_v(data: &[f32], num_v_heads: usize, num_k_heads: usize, head_dim: usize,
                  in_f: usize, row_lo: usize, row_hi: usize) -> Vec<f32> {
    let num_v_per_k = num_v_heads / num_k_heads;
    let mut out = data.to_vec();
    debug_assert_eq!(row_hi - row_lo, num_v_heads * head_dim);
    for j in 0..num_v_per_k {
        for g in 0..num_k_heads {
            let dst_head = j * num_k_heads + g;
            let src_head = g * num_v_per_k + j;
            for d in 0..head_dim {
                let dst_row = row_lo + dst_head * head_dim + d;
                let src_row = row_lo + src_head * head_dim + d;
                out[dst_row * in_f..dst_row * in_f + in_f]
                    .copy_from_slice(&data[src_row * in_f..src_row * in_f + in_f]);
            }
        }
    }
    out
}

/// Reorder the V-head COLUMNS of a row-major [out_rows][in_f=value_dim] matrix (out_proj, dim=1).
/// Column index = V-head*head_dim + d over the whole `value_dim` input axis.
fn reorder_cols_v(data: &[f32], out_rows: usize, in_f: usize,
                  num_v_heads: usize, num_k_heads: usize, head_dim: usize) -> Vec<f32> {
    let num_v_per_k = num_v_heads / num_k_heads;
    debug_assert_eq!(in_f, num_v_heads * head_dim);
    let mut out = data.to_vec();
    for r in 0..out_rows {
        let base = r * in_f;
        for j in 0..num_v_per_k {
            for g in 0..num_k_heads {
                let dst_head = j * num_k_heads + g;
                let src_head = g * num_v_per_k + j;
                for d in 0..head_dim {
                    out[base + dst_head * head_dim + d] = data[base + src_head * head_dim + d];
                }
            }
        }
    }
    out
}

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v { out.extend_from_slice(&f.to_le_bytes()); }
    out
}

/// Resolve a requested ggml name to an HF target (+ optional transform), arch-aware.
/// Order: per-expert MoE name -> dense plain map -> qwen35 SSM map / norm `+1`.
pub fn resolve_ggml(ggml: &str, cfg: &ModelConfig) -> Option<HfTarget> {
    // 1. Per-expert MoE: blk.{il}.ffn_{gate,up,down}_exps.{e}.weight (gathered one expert at a time).
    if let Some(rest) = ggml.strip_prefix("blk.") {
        if let Some((il, suffix)) = rest.split_once('.') {
            for (tag, proj) in [("ffn_gate_exps.", "gate"), ("ffn_up_exps.", "up"), ("ffn_down_exps.", "down")] {
                if let Some(e_part) = suffix.strip_prefix(tag) {
                    if let Some(e) = e_part.strip_suffix(".weight") {
                        if let Ok(eid) = e.parse::<u32>() {
                            if let Ok(ilid) = il.parse::<u32>() {
                                // Qwen stores appended MTP blocks under a separate `mtp.layers`
                                // namespace. Hy3 appends its MTP block to `model.layers` instead.
                                let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
                                if cfg.nextn_predict_layers > 0
                                    && ilid >= n_trunk
                                    && matches!(cfg.arch, Arch::Qwen35 | Arch::Qwen35Moe)
                                {
                                    let mtp_il = ilid - n_trunk;
                                    let projection = match proj {
                                        "gate" => "gate_proj",
                                        "up" => "up_proj",
                                        "down" => "down_proj",
                                        _ => unreachable!(),
                                    };
                                    return Some(HfTarget::Plain(format!(
                                        "mtp.layers.{mtp_il}.mlp.experts.{eid}.{projection}.weight"
                                    )));
                                }
                                return Some(HfTarget::Plain(hf_expert_name(
                                    ilid, eid, proj, &cfg.arch,
                                )));
                            }
                        }
                    }
                }
            }
        }
    }

    // Hy3 appends its MTP block as model.layers.{n_trunk}, in the same namespace and tensor
    // layout as the trunk. Only the four glue tensors need special ggml names; ordinary attention,
    // FFN, router, shared-expert, and per-expert names continue through the standard Hy3 maps.
    if cfg.arch.is_hy3() && cfg.nextn_predict_layers > 0 {
        let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
        if let Some((il, suffix)) = ggml
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
        {
            if il.parse::<u32>().ok().is_some_and(|il| il >= n_trunk) {
                let hf_suffix = match suffix {
                    "nextn.enorm.weight" => Some("enorm.weight"),
                    "nextn.hnorm.weight" => Some("hnorm.weight"),
                    "nextn.eh_proj.weight" => Some("eh_proj.weight"),
                    "nextn.shared_head_norm.weight" => Some("final_layernorm.weight"),
                    _ => None,
                };
                if let Some(hf_suffix) = hf_suffix {
                    return Some(HfTarget::Plain(format!("model.layers.{il}.{hf_suffix}")));
                }
            }
        }
    }

    // 2. qwen35 norm +1: every `*norm.weight` EXCEPT linear_attn (ssm_norm). The dense map below
    //    renames them; for hybrid we must additionally add +1 (qwen.py:302-303). Catch the dense
    //    norm names here so they take the Transform arm.
    // `is_hybrid()` also includes the dense-attention MoE architectures that reuse HybridModel
    // (MiniMax-M3 and Hy3). The MTP/SSM maps and +1 norm convention here are Qwen3.5-specific.
    // MiniMax applies its independently configured Gemma-norm fold in the next arm; Hy3 uses the
    // checkpoint norm weights verbatim.
    if matches!(cfg.arch, Arch::Qwen35 | Arch::Qwen35Moe) {
        // MTP/NextN block FIRST: blk.{n_trunk}.* maps into the HF `mtp.*` namespace (NVIDIA 27B /
        // qwen3.6 text ckpts), NOT model.layers.{n_trunk}.* (which does not exist — HF
        // num_hidden_layers excludes the MTP block). Must precede the SSM/dense arms.
        if let Some(t) = resolve_mtp_block(ggml, cfg) { return Some(t); }
        if let Some(t) = resolve_hybrid_ssm(ggml, cfg) { return Some(t); }
        // norm +1 on the dense-mapped norms (attn_norm / ffn_norm / q_norm / k_norm / output_norm).
        if is_plusone_norm(ggml) {
            if let Some(hf) = ggml_to_hf(ggml, &cfg.arch) {
                return Some(HfTarget::Transform { hf, kind: TransformKind::NormPlusOne });
            }
        }
    }

    // 3. MiniMax-M3 gemma-norm: out = normed*(1+w) — fold the +1 into the weight at load so the
    //    engine's plain RMSNorm runs unchanged (exact; same fold the qwen35 converter applies).
    //    Applies to EVERY norm in the text model (input/post_attention layernorm, q/k_norm,
    //    model.norm — and index_q/k_norm when the MSA arc lands).
    if cfg.m3.as_ref().is_some_and(|m| m.use_gemma_norm) && is_plusone_norm(ggml) {
        if let Some(hf) = ggml_to_hf(ggml, &cfg.arch) {
            return Some(HfTarget::Transform { hf, kind: TransformKind::NormPlusOne });
        }
    }

    // 4. Dense plain rename (zero-copy).
    ggml_to_hf(ggml, &cfg.arch).map(HfTarget::Plain)
}

/// qwen3.5/3.6 MTP (NextN) block map: the engine asks for `blk.{n_trunk}.*` (GGUF numbering, where
/// block_count includes the MTP block) but the HF checkpoint stores the head under a separate
/// `mtp.*` namespace (NVIDIA 27B + qwen3.6 text ckpts; HF num_hidden_layers EXCLUDES the block):
///   nextn.enorm  -> mtp.pre_fc_norm_embedding   (+1 fold, verified vs GGUF twin blk.64)
///   nextn.hnorm  -> mtp.pre_fc_norm_hidden      (+1)
///   nextn.eh_proj -> mtp.fc                     (plain, [2*n_embd, n_embd])
///   nextn.shared_head_norm -> mtp.norm          (+1)
///   attn/ffn/norm tensors -> mtp.layers.{il-n_trunk}.{hf dense names}  (norms +1, matrices plain)
/// `nextn.shared_head.weight` has no mtp.* equivalent (head reuses the model's lm_head) -> None.
fn resolve_mtp_block(ggml: &str, cfg: &ModelConfig) -> Option<HfTarget> {
    if cfg.nextn_predict_layers == 0 { return None; }
    let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;
    let il: u32 = il.parse().ok()?;
    if il < n_trunk { return None; }
    let mtp_il = il - n_trunk; // mtp.layers.{k} index (27B: always 0)
    // NextN glue tensors (top-level mtp.* names).
    let glue: Option<(&str, TransformKind)> = match suffix {
        "nextn.enorm.weight" => Some(("mtp.pre_fc_norm_embedding.weight", TransformKind::NormPlusOne)),
        "nextn.hnorm.weight" => Some(("mtp.pre_fc_norm_hidden.weight", TransformKind::NormPlusOne)),
        "nextn.eh_proj.weight" => Some(("mtp.fc.weight", TransformKind::Identity)),
        "nextn.shared_head_norm.weight" => Some(("mtp.norm.weight", TransformKind::NormPlusOne)),
        _ => None,
    };
    if let Some((hf, kind)) = glue {
        return Some(match kind {
            TransformKind::Identity => HfTarget::Plain(hf.into()),
            k => HfTarget::Transform { hf: hf.into(), kind: k },
        });
    }
    // Full transformer block tensors under mtp.layers.{k}.*. Reuse the ordinary Qwen layer map so
    // the MoE router and shared-expert tensors cannot silently remain in model.layers.{n_trunk}.
    let hf_suffix = if suffix == "post_attention_norm.weight" {
        "post_attention_layernorm.weight".to_string()
    } else {
        let ordinary = ggml_to_hf(ggml, &cfg.arch)?;
        ordinary
            .strip_prefix(&format!("model.layers.{il}."))?
            .to_string()
    };
    let hf = format!("mtp.layers.{mtp_il}.{hf_suffix}");
    Some(if is_plusone_norm(ggml) {
        HfTarget::Transform {
            hf,
            kind: TransformKind::NormPlusOne,
        }
    } else {
        HfTarget::Plain(hf)
    })
}

/// True for the ggml norm names that get qwen35 `+1` (all norms except ssm_norm/linear_attn.norm).
fn is_plusone_norm(ggml: &str) -> bool {
    match ggml {
        "output_norm.weight" => true,
        _ => {
            let Some(rest) = ggml.strip_prefix("blk.") else { return false };
            let Some((_, suffix)) = rest.split_once('.') else { return false };
            matches!(suffix,
                "attn_norm.weight" | "ffn_norm.weight"
                | "attn_q_norm.weight" | "attn_k_norm.weight"
                | "post_attention_norm.weight")
        }
    }
}

/// qwen35 linear-attn SSM tensor map (the `blk.{il}.{ssm/attn_qkv/attn_gate}` family). HF prefix is
/// `model.layers.{il}.linear_attn.` (the `model.language_model.` wrapper is handled by the source's
/// prefix-fallback). Cited against llama.cpp gguf-py/gguf/tensor_mapping.py + conversion/qwen.py.
fn resolve_hybrid_ssm(ggml: &str, _cfg: &ModelConfig) -> Option<HfTarget> {
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;
    let la = |t: &str| format!("model.layers.{il}.linear_attn.{t}");
    let (hf, kind) = match suffix {
        // matrices with a V-reorder transform (owned f32 buffer)
        "attn_qkv.weight"   => (la("in_proj_qkv.weight"), TransformKind::QkvVReorderRows),  // tensor_mapping.py:251
        "attn_gate.weight"  => (la("in_proj_z.weight"),   TransformKind::ZReorderRows),     // :386
        "ssm_beta.weight"   => (la("in_proj_b.weight"),   TransformKind::AbReorderRows),    // :909
        "ssm_alpha.weight"  => (la("in_proj_a.weight"),   TransformKind::AbReorderRows),    // :885
        "ssm_a"             => (la("A_log"),              TransformKind::NegExpReorderHeads), // :846
        "ssm_dt.bias"       => (la("dt_bias"),            TransformKind::ReorderHeads),      // :831
        "ssm_conv1d.weight" => (la("conv1d.weight"),      TransformKind::Conv1dSqueezeReorder), // :816
        "ssm_norm.weight"   => (la("norm.weight"),        TransformKind::Identity),          // :871 (NO +1)
        "ssm_out.weight"    => (la("out_proj.weight"),    TransformKind::OutReorderCols),    // :880
        _ => return None,
    };
    Some(HfTarget::Transform { hf, kind })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Arch;

    #[test]
    fn top_level_names() {
        let a = Arch::Qwen3;
        assert_eq!(ggml_to_hf("token_embd.weight", &a).unwrap(), "model.embed_tokens.weight");
        assert_eq!(ggml_to_hf("output_norm.weight", &a).unwrap(), "model.norm.weight");
        assert_eq!(ggml_to_hf("output.weight", &a).unwrap(), "lm_head.weight");
    }

    #[test]
    fn per_layer_dense() {
        let a = Arch::Qwen3;
        assert_eq!(ggml_to_hf("blk.0.attn_q.weight", &a).unwrap(), "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(ggml_to_hf("blk.7.attn_k.weight", &a).unwrap(), "model.layers.7.self_attn.k_proj.weight");
        assert_eq!(ggml_to_hf("blk.3.attn_output.weight", &a).unwrap(), "model.layers.3.self_attn.o_proj.weight");
        assert_eq!(ggml_to_hf("blk.0.attn_norm.weight", &a).unwrap(), "model.layers.0.input_layernorm.weight");
        assert_eq!(ggml_to_hf("blk.0.ffn_norm.weight", &a).unwrap(), "model.layers.0.post_attention_layernorm.weight");
        assert_eq!(ggml_to_hf("blk.0.ffn_gate.weight", &a).unwrap(), "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(ggml_to_hf("blk.0.ffn_up.weight", &a).unwrap(), "model.layers.0.mlp.up_proj.weight");
        assert_eq!(ggml_to_hf("blk.0.ffn_down.weight", &a).unwrap(), "model.layers.0.mlp.down_proj.weight");
        assert_eq!(ggml_to_hf("blk.5.attn_q_norm.weight", &a).unwrap(), "model.layers.5.self_attn.q_norm.weight");
    }

    #[test]
    fn unmapped_returns_none() {
        let a = Arch::Qwen35; // SSM tensors are NOT in the plain dense map (they go through resolve_ggml)
        assert!(ggml_to_hf("blk.0.ssm_a", &a).is_none());
        assert!(ggml_to_hf("blk.0.attn_qkv.weight", &a).is_none());
        assert!(ggml_to_hf("totally.unknown", &a).is_none());
    }

    #[test]
    fn expert_names() {
        assert_eq!(hf_expert_name(2, 13, "gate", &Arch::Qwen3Moe),
                   "model.layers.2.mlp.experts.13.gate_proj.weight");
        assert_eq!(hf_expert_name(0, 0, "down", &Arch::Qwen3Moe),
                   "model.layers.0.mlp.experts.0.down_proj.weight");
        // MiniMax-M3 Mixtral-style layout: w1=gate, w2=down, w3=up.
        assert_eq!(hf_expert_name(5, 63, "gate", &Arch::MinimaxM3),
                   "model.layers.5.block_sparse_moe.experts.63.w1.weight");
        assert_eq!(hf_expert_name(5, 63, "down", &Arch::MinimaxM3),
                   "model.layers.5.block_sparse_moe.experts.63.w2.weight");
        assert_eq!(hf_expert_name(5, 63, "up", &Arch::MinimaxM3),
                   "model.layers.5.block_sparse_moe.experts.63.w3.weight");
    }

    #[test]
    fn hy3_moe_names() {
        let a = Arch::Hy3;
        assert_eq!(
            ggml_to_hf("blk.0.ffn_gate.weight", &a).unwrap(),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_gate_inp.weight", &a).unwrap(),
            "model.layers.1.mlp.router.gate.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.exp_probs_b.bias", &a).unwrap(),
            "model.layers.1.mlp.expert_bias"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_gate_shexp.weight", &a).unwrap(),
            "model.layers.1.mlp.shared_mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_up_shexp.weight", &a).unwrap(),
            "model.layers.1.mlp.shared_mlp.up_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_down_shexp.weight", &a).unwrap(),
            "model.layers.1.mlp.shared_mlp.down_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_gate_exps.weight", &a).unwrap(),
            "model.layers.1.mlp.switch_mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_up_exps.weight", &a).unwrap(),
            "model.layers.1.mlp.switch_mlp.up_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_down_exps.weight", &a).unwrap(),
            "model.layers.1.mlp.switch_mlp.down_proj.weight"
        );
    }

    #[test]
    fn hy3_appended_mtp_names_use_layer_namespace() {
        let json = r#"{
            "model_type":"hy_v3",
            "num_hidden_layers":80,
            "num_nextn_predict_layers":1,
            "hidden_size":256,
            "num_attention_heads":8,
            "num_key_value_heads":2,
            "intermediate_size":512,
            "vocab_size":1024,
            "max_position_embeddings":2048
        }"#;
        let cfg = ModelConfig::from_hf(&crate::config::HfConfig::parse(json));
        assert_eq!(cfg.n_layer, 81);
        for (ggml, expected) in [
            ("blk.80.nextn.enorm.weight", "model.layers.80.enorm.weight"),
            ("blk.80.nextn.hnorm.weight", "model.layers.80.hnorm.weight"),
            (
                "blk.80.nextn.eh_proj.weight",
                "model.layers.80.eh_proj.weight",
            ),
            (
                "blk.80.nextn.shared_head_norm.weight",
                "model.layers.80.final_layernorm.weight",
            ),
            (
                "blk.80.ffn_gate_exps.7.weight",
                "model.layers.80.mlp.experts.7.gate_proj.weight",
            ),
            (
                "blk.80.attn_q.weight",
                "model.layers.80.self_attn.q_proj.weight",
            ),
        ] {
            match resolve_ggml(ggml, &cfg) {
                Some(HfTarget::Plain(hf)) => assert_eq!(hf, expected),
                _ => panic!("Hy3 MTP tensor {ggml} did not resolve as a plain layer tensor"),
            }
        }
    }

    #[test]
    fn qwen35_moe_appended_mtp_names_use_mtp_namespace() {
        let json = r#"{
            "model_type":"qwen3_5_moe",
            "num_hidden_layers":2,
            "num_nextn_predict_layers":1,
            "hidden_size":256,
            "num_attention_heads":8,
            "num_key_value_heads":2,
            "intermediate_size":512,
            "vocab_size":1024,
            "max_position_embeddings":2048
        }"#;
        let cfg = ModelConfig::from_hf(&crate::config::HfConfig::parse(json));
        assert_eq!(cfg.n_layer, 3);
        for (ggml, expected) in [
            (
                "blk.2.ffn_gate_exps.7.weight",
                "mtp.layers.0.mlp.experts.7.gate_proj.weight",
            ),
            ("blk.2.ffn_gate_inp.weight", "mtp.layers.0.mlp.gate.weight"),
            (
                "blk.2.ffn_gate_shexp.weight",
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            ),
            (
                "blk.2.ffn_gate_inp_shexp.weight",
                "mtp.layers.0.mlp.shared_expert_gate.weight",
            ),
            (
                "blk.2.attn_q.weight",
                "mtp.layers.0.self_attn.q_proj.weight",
            ),
        ] {
            match resolve_ggml(ggml, &cfg) {
                Some(HfTarget::Plain(hf)) => assert_eq!(hf, expected),
                _ => panic!("Qwen3.5 MTP tensor {ggml} did not resolve in the mtp namespace"),
            }
        }
        match resolve_ggml("blk.2.attn_norm.weight", &cfg) {
            Some(HfTarget::Transform {
                hf,
                kind: TransformKind::NormPlusOne,
            }) => assert_eq!(hf, "mtp.layers.0.input_layernorm.weight"),
            _ => panic!("Qwen3.5 MTP input norm lost its mtp namespace or +1 transform"),
        }
    }

    fn mapping_config(model_type: &str) -> ModelConfig {
        let json = format!(r#"{{
            "model_type":"{model_type}",
            "num_hidden_layers":2,
            "hidden_size":256,
            "num_attention_heads":8,
            "num_key_value_heads":2,
            "intermediate_size":512,
            "vocab_size":1024,
            "max_position_embeddings":2048
        }}"#);
        ModelConfig::from_hf(&crate::config::HfConfig::parse(&json))
    }

    #[test]
    fn norm_transform_is_arch_specific() {
        let qwen35 = mapping_config("qwen3_5");
        match resolve_ggml("blk.0.attn_norm.weight", &qwen35) {
            Some(HfTarget::Transform { kind: TransformKind::NormPlusOne, .. }) => {}
            _ => panic!("Qwen3.5 attn norm lost its required +1 transform"),
        }

        let hy3 = mapping_config("hy_v3");
        for ggml in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "output_norm.weight",
        ] {
            match resolve_ggml(ggml, &hy3) {
                Some(HfTarget::Plain(_)) => {}
                _ => panic!("Hy3 norm {ggml} must use the raw checkpoint weight"),
            }
        }
    }

    /// The V-head reorder permutation: HF grouped [0,1, 2,3, ...] -> ggml tiled [0,2,...,1,3,...].
    /// For nv=4, nk=2, head_dim=1: src heads [0,1,2,3] (grouped g*2+j) -> dst slots j*2+g.
    /// dst0=src0(g0,j0), dst1=src2(g1,j0), dst2=src1(g0,j1), dst3=src3(g1,j1) => [v0,v2,v1,v3].
    #[test]
    fn v_reorder_permutation() {
        let block = vec![10.0f32, 11.0, 12.0, 13.0]; // 4 heads, head_dim=1
        let out = reorder_v_axis(&block, 4, 2, 1);
        assert_eq!(out, vec![10.0, 12.0, 11.0, 13.0]);
        // identity when nv == nk (num_v_per_k == 1): order unchanged.
        let id = reorder_v_axis(&block, 4, 4, 1);
        assert_eq!(id, block);
    }

    /// reorder_rows_v over a band: a [4 rows x 2 cols] matrix, reorder all 4 rows (nv=4,nk=2,hd=1).
    #[test]
    fn v_reorder_rows() {
        // rows: r0=[0,1] r1=[2,3] r2=[4,5] r3=[6,7]; expect dst order [r0,r2,r1,r3].
        let data = vec![0.0f32,1.0, 2.0,3.0, 4.0,5.0, 6.0,7.0];
        let out = reorder_rows_v(&data, 4, 2, 1, 2, 0, 4);
        assert_eq!(out, vec![0.0,1.0, 4.0,5.0, 2.0,3.0, 6.0,7.0]);
    }
}
