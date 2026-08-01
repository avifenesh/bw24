//! Increment 2, deliverable 3 receipt: the glm-dsa loader arm END TO END on a real device —
//! micro fixture (generated at test time) -> GgufFile -> HybridModel::load -> Mixer::Mla
//! device buffers + latent-cache geometry. No forward (increment 4).
//!
//! GPU-gated: `#[ignore]` by default (CI is compile-only / GPU-less). Run on the rig:
//!   flock /tmp/gpu5090.lock cargo test -p memra-engine --test mla_fixture_load_gpu -- --ignored

use memra_engine::hybrid::{Ffn, HybridModel, Mixer};
use memra_engine::Engine;
use memra_gguf::micro_gguf::write_glm_dsa_micro;
use memra_gguf::GgufFile;

#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/gpu5090.lock"]
fn gpu_load_glm_dsa_micro_fixture() {
    let p = std::env::temp_dir().join(format!("memra-mla-gpu-load-{}.gguf", std::process::id()));
    let d = write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
    let g = GgufFile::open(&p).unwrap();
    let e = Engine::new(0).expect("CUDA device 0");
    let model = HybridModel::load(&e, &g).expect("glm-dsa micro fixture loads");
    std::fs::remove_file(&p).ok();

    // trunk = block_count - nextn; the MTP block is the separate head.
    assert_eq!(model.layers.len(), d.n_trunk as usize);
    assert_eq!(model.cfg.n_layer, d.block_count() as u32);
    let mla_cfg = model.cfg.mla.as_ref().expect("cfg.mla parsed");

    for (il, layer) in model.layers.iter().enumerate() {
        let mla = match &layer.mixer {
            Mixer::Mla(m) => m,
            other => panic!(
                "layer {il}: expected Mixer::Mla, got {}",
                match other { Mixer::Full(_) => "Full", Mixer::Linear(_) => "Linear", _ => "?" }
            ),
        };
        // latent-cache geometry resolved from metadata, cross-checked against tensor shapes
        assert_eq!(mla.geom.n_head, d.n_head as usize);
        assert_eq!(mla.geom.d_nope, d.d_nope as usize);
        assert_eq!(mla.geom.d_rope, d.d_rope as usize);
        assert_eq!(mla.geom.d_v, d.d_v as usize);
        assert_eq!(mla.geom.kv_rank, d.kv_lora as usize);
        assert_eq!(mla.geom.latent_dim, d.latent_dim() as usize);
        assert!((mla.geom.scale - mla_cfg.scale()).abs() < 1e-9);
        // device-resident projections with the conversion-split shapes
        assert_eq!(mla.wq_a.ne(), &[d.n_embd, d.q_lora]);
        assert_eq!(mla.wq_b.ne(), &[d.q_lora, d.n_head * (d.d_nope + d.d_rope)]);
        assert_eq!(mla.wkv_a.ne(), &[d.n_embd, d.kv_lora + d.d_rope]);
        assert_eq!(mla.wk_b.ne(), &[d.d_nope, d.kv_lora, d.n_head]);
        assert_eq!(mla.wv_b.ne(), &[d.kv_lora, d.d_v, d.n_head]);
        assert_eq!(mla.wo.ne(), &[d.n_head * d.d_v, d.n_embd]);
        // FFN split: leading_dense_block_count = 1
        match (&layer.ffn, il) {
            (Ffn::Dense { .. }, 0) => {}
            (Ffn::Moe(m), _) if il > 0 => {
                assert_eq!(m.gate_exps.n_expert, d.n_expert as usize);
                assert!(m.exp_probs_b.is_some(), "noaux_tc selection bias loads");
            }
            _ => panic!("layer {il}: wrong FFN arm"),
        }
    }

    // MTP head: dense-MLA NextN block, head falls back (no shared_head_head in the artifact set)
    let mtp = model.mtp.as_ref().expect("nextn=1 loads the MTP head");
    assert!(matches!(mtp.mixer, Mixer::Mla(_)), "MTP block is MLA on glm-dsa");
    assert!(matches!(mtp.ffn, Ffn::Moe(_)));
    assert!(mtp.shared_head_norm.is_some());
    assert!(mtp.shared_head_head.is_none(), "fixture matches the real artifact: no nextn head");

    eprintln!("[mla-gpu-load] OK: {} trunk layers + MTP, latent_dim {}, scale {}",
              model.layers.len(), mla_cfg.latent_dim(), mla_cfg.scale());
}
