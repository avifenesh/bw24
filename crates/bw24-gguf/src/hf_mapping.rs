//! HF (transformers) tensor-name -> ggml tensor-name mapping.
//!
//! The bw24 engine ONLY ever asks for ggml-style names (e.g. `blk.3.attn_q.weight`,
//! hardcoded in model.rs / hybrid.rs). The safetensors source resolves on demand by
//! translating a requested **ggml name** into the **HF name** to look up in the file.
//!
//! Direction is ggml -> HF (one lookup per requested tensor; no full-table build needed).
//!
//! Coverage: dense transformer (Qwen3 / Llama / Gemma style). MoE expert stacking and
//! hybrid (qwen35) SSM tensors are NOT mapped here — see the TODOs below — because no
//! safetensors test model for those was available to validate against.

use crate::config::Arch;

/// Translate a requested ggml tensor name into the HF safetensors name.
/// Returns `None` if the ggml name has no known HF equivalent for this arch.
pub fn ggml_to_hf(ggml: &str, _arch: &Arch) -> Option<String> {
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
        // NOT MAPPED (no safetensors test model to validate against):
        //   * stacked MoE experts: ggml "blk.N.ffn_{gate,up,down}_exps.weight" is ONE 3D tensor,
        //     HF stores them as N separate 2D tensors "model.layers.N.mlp.experts.{e}.*_proj.weight".
        //     Requires a gather+concat, not a single-name lookup — see hf_expert_name() below.
        //   * hybrid/SSM: "attn_qkv", "attn_gate", "ssm_*" (qwen35) have no validated HF name map.
        _ => return None,
    };
    Some(format!("model.layers.{il}.{hf_suffix}"))
}

/// HF name for a single expert's projection weight (for the MoE gather+concat path).
/// ggml stacks all experts into one 3D tensor; HF stores them separately. `proj` is one of
/// "gate", "up", "down". NOT yet wired into the loader (no MoE safetensors test model available),
/// but provided so the expert-staging code has the canonical name to gather from.
pub fn hf_expert_name(il: u32, e: u32, proj: &str) -> String {
    let p = match proj {
        "gate" => "gate_proj",
        "up" => "up_proj",
        "down" => "down_proj",
        other => panic!("unknown expert proj {other}"),
    };
    format!("model.layers.{il}.mlp.experts.{e}.{p}.weight")
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
        let a = Arch::Qwen35; // hybrid SSM tensors intentionally unmapped
        assert!(ggml_to_hf("blk.0.ssm_a", &a).is_none());
        assert!(ggml_to_hf("blk.0.attn_qkv.weight", &a).is_none());
        assert!(ggml_to_hf("totally.unknown", &a).is_none());
    }

    #[test]
    fn expert_names() {
        assert_eq!(hf_expert_name(2, 13, "gate"), "model.layers.2.mlp.experts.13.gate_proj.weight");
        assert_eq!(hf_expert_name(0, 0, "down"), "model.layers.0.mlp.experts.0.down_proj.weight");
    }
}
