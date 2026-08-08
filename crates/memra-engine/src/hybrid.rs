//! Qwen3.5/3.6 hybrid model: linear-attention (Gated DeltaNet) layers + periodic full-attention
//! layers + SwiGLU FFN. Loads weights, runs the forward, dual cache. Builds on the validated
//! conv1d + gdn_scan kernels (M2/M3) and the dense full-attn path (M0).

use crate::model::{EmbedHost, GpuTensor, HostExps};
use crate::Engine;
use memra_gguf::config::{LayerKind, MlaConfig, ModelConfig};
use memra_gguf::source::{GgufSource, TensorSource};
use memra_gguf::{GgmlType, GgufFile};
use cudarc::driver::CudaSlice;
use std::collections::HashMap;

// Source-agnostic load helpers (GGUF or safetensors). The GGUF wrappers below keep `load()`
// byte-identical; only the source object differs.
fn load_t(
    e: &Engine,
    src: &dyn TensorSource,
    name: &str,
) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    GpuTensor::load_from_source(e, src, name)
}
fn load_opt(
    e: &Engine,
    src: &dyn TensorSource,
    name: &str,
) -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
    GpuTensor::load_opt_from_source(e, src, name)
}

struct ResidencyBytes {
    experts: HashMap<usize, usize>,
    rest: usize,
    saw_experts: bool,
}

fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn residency_bytes_by_device<'a>(
    tensors: impl IntoIterator<Item = (&'a str, usize)>,
    layer_devices: &[usize],
    primary_device: usize,
) -> ResidencyBytes {
    let mut out = ResidencyBytes {
        experts: HashMap::new(),
        rest: 0,
        saw_experts: false,
    };
    for (name, bytes) in tensors {
        if name.starts_with("blk.") && name.contains("_exps.") {
            let device = block_index(name)
                .and_then(|il| layer_devices.get(il).copied())
                .unwrap_or(primary_device);
            *out.experts.entry(device).or_default() += bytes;
            out.saw_experts = true;
        } else {
            out.rest += bytes;
        }
    }
    out
}

/// Load-local resident-expert capacity decisions. PP stages on distinct devices are charged only
/// for their own layer slices; co-located stages share a device key and are charged together.
pub(crate) struct ResidentPlan {
    primary_device: usize,
    layer_devices: Vec<usize>,
    layer_counts: HashMap<usize, usize>,
    exact_expert_bytes: Option<HashMap<usize, usize>>,
    trunk_bytes: usize,
    decisions: HashMap<usize, bool>,
    pp: bool,
}

impl ResidentPlan {
    fn from_layout(
        src: &dyn TensorSource,
        primary_device: usize,
        layer_devices: Vec<usize>,
        pp: bool,
    ) -> Self {
        let mut layer_counts = HashMap::new();
        for &device in &layer_devices {
            *layer_counts.entry(device).or_default() += 1;
        }
        let (exact_expert_bytes, trunk_bytes) = match src.gguf() {
            Some(g) => {
                let bytes = residency_bytes_by_device(
                    g.tensors.iter().map(|t| (t.name.as_str(), t.n_bytes as usize)),
                    &layer_devices,
                    primary_device,
                );
                if bytes.saw_experts {
                    (Some(bytes.experts), bytes.rest)
                } else {
                    (None, 0)
                }
            }
            None => (None, 0),
        };
        Self {
            primary_device,
            layer_devices,
            layer_counts,
            exact_expert_bytes,
            trunk_bytes,
            decisions: HashMap::new(),
            pp,
        }
    }

    pub(crate) fn unsharded(e: &Engine, src: &dyn TensorSource, cfg: &ModelConfig) -> Self {
        let device = e.ctx().ordinal();
        Self::from_layout(src, device, vec![device; cfg.n_layer as usize], false)
    }

    pub(crate) fn pp(
        e: &Engine,
        src: &dyn TensorSource,
        cfg: &ModelConfig,
        n_trunk: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let primary = e.ctx().ordinal();
        let Some(_fence) = crate::pp::pp_cuts(n_trunk) else {
            return Ok(Self::unsharded(e, src, cfg));
        };
        let mut layer_devices = vec![primary; cfg.n_layer as usize];
        for (il, device) in layer_devices.iter_mut().take(n_trunk).enumerate() {
            *device = crate::pp::layer_engine(e, n_trunk, il)?.ctx().ordinal();
        }
        Ok(Self::from_layout(src, primary, layer_devices, true))
    }

    fn should_reside(&mut self, e: &Engine, il: usize, per_layer: usize) -> bool {
        let device = self.layer_devices.get(il).copied().unwrap_or(self.primary_device);
        debug_assert_eq!(e.ctx().ordinal(), device);
        if let Some(&decision) = self.decisions.get(&device) {
            return decision;
        }
        if std::env::var("MEMRA_MOE_RESIDENT").as_deref() == Ok("0") {
            self.decisions.insert(device, false);
            return false;
        }
        let (free, _total) = match e.ctx().mem_get_info() {
            Ok(v) => v,
            Err(_) => {
                self.decisions.insert(device, false);
                return false;
            }
        };
        let projected = self.exact_expert_bytes.as_ref()
            .map(|bytes| bytes.get(&device).copied().unwrap_or(0))
            .unwrap_or(per_layer * self.layer_counts.get(&device).copied().unwrap_or(1));
        let budget = std::env::var("MEMRA_MOE_RESIDENT_GB").ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|gb| (gb * 1e9) as usize)
            .unwrap_or_else(|| {
                let reserve = std::env::var("MEMRA_MOE_RESIDENT_HEADROOM_GB").ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|gb| (gb * 1e9) as usize)
                    .unwrap_or(2_000_000_000);
                (free as usize).saturating_sub(self.trunk_bytes + reserve)
            });
        let ok = projected <= budget;
        eprintln!("[moe] resident-experts decision ({}dev{}): experts {:.2}GB + trunk {:.2}GB vs free {:.2}GB (expert budget {:.2}GB) -> {}",
                  if self.pp { "PP " } else { "" }, device, projected as f64 / 1e9,
                  self.trunk_bytes as f64 / 1e9, free as f64 / 1e9, budget as f64 / 1e9,
                  if ok { "RESIDENT" } else { "SLRU cache" });
        self.decisions.insert(device, ok);
        ok
    }
}

/// Load the mixer (full-attn, linear-attn, or MLA) for block `il`. Shared by the trunk loop and
/// the MTP head. `kind` overrides cfg.layer_kind (the MTP/NextN block is ALWAYS full-attn
/// regardless of the periodic interval — its GGUF carries attn_q/k/v, not ssm_*/attn_qkv).
/// `mla` is the Arch gate: `Some` only for glm-dsa (cfg.mla) — every layer of an MLA model,
/// INCLUDING its NextN/MTP block (dense MLA, no indexer), takes the Mla arm.
///
/// `sep_gate` = `ModelConfig::attn_gate_separate()`: load step35's separate head-wise
/// `attn_gate.weight` onto the full-attn arm. It is passed in rather than read off a cfg so the
/// MTP/draft call sites (which build a synthetic cfg) opt in explicitly.
fn load_mixer_kind(
    e: &Engine,
    src: &dyn TensorSource,
    il: u32,
    kind: LayerKind,
    mla: Option<&MlaConfig>,
    sep_gate: bool,
) -> Result<Mixer, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    if let Some(m) = mla {
        assert_eq!(kind, LayerKind::FullAttention, "MLA layers are full-attention class");
        return Ok(Mixer::Mla(MlaAttnLayer::load(e, src, il, m)?));
    }
    Ok(match kind {
        LayerKind::FullAttention => Mixer::Full(FullAttnLayer {
            wq: load_t(e, src, &p("attn_q.weight"))?,
            wk: load_t(e, src, &p("attn_k.weight"))?,
            // gemma4 global layers ship NO v_proj (attention_k_eq_v): V = the K projection
            // output pre-rope (llama gemma4.cpp: `Vcur = wv ? mm(wv,cur) : Kcur`). Loading
            // wv := wk reproduces that exactly with zero forward changes; the gemma forward
            // adds the weightless V rms_norm (R7 part 2).
            wv: match load_opt(e, src, &p("attn_v.weight"))? {
                Some(v) => v,
                None => load_t(e, src, &p("attn_k.weight"))?,
            },
            wo: load_t(e, src, &p("attn_output.weight"))?,
            q_norm: load_t(e, src, &p("attn_q_norm.weight"))?,
            k_norm: load_t(e, src, &p("attn_k_norm.weight"))?,
            // step35: REQUIRED when the arch says so — a missing gate would silently drop the
            // per-head sigmoid and produce plausible-but-wrong logits, so this is load_t not
            // load_opt. Step-3.7-Flash ships it on all 45 blocks (width = that layer's n_head).
            attn_gate: if sep_gate {
                Some(load_t(e, src, &p("attn_gate.weight"))?)
            } else {
                None
            },
        }),
        LayerKind::LinearAttention => Mixer::Linear(LinearAttnLayer {
            wqkv: load_t(e, src, &p("attn_qkv.weight"))?,
            wqkv_gate: load_t(e, src, &p("attn_gate.weight"))?,
            ssm_beta: load_t(e, src, &p("ssm_beta.weight"))?,
            ssm_alpha: load_t(e, src, &p("ssm_alpha.weight"))?,
            ssm_a: load_t(e, src, &p("ssm_a"))?,
            ssm_dt: load_t(e, src, &p("ssm_dt.bias"))?,
            ssm_conv1d: load_t(e, src, &p("ssm_conv1d.weight"))?,
            ssm_norm: load_t(e, src, &p("ssm_norm.weight"))?,
            ssm_out: load_t(e, src, &p("ssm_out.weight"))?,
        }),
    })
}

/// Load the FFN (dense SwiGLU or routed MoE) for block `il`. Source-agnostic (GGUF or safetensors
/// via `TensorSource`); shared by the hybrid trunk/MTP loops AND the dense-attention MoE path (OLMoE).
/// Shared-expert tensors are OPTIONAL (`load_opt`): qwen35moe has them, OLMoE/vanilla-MoE do not.
/// When `spill` is `Some` (MEMRA_SPILL_DISK on) AND the source is the GGUF on disk, MoE experts load
/// through the per-expert tier split (`HostExps::load_tiered`: hottest pinned, rest mmap'd from disk);
/// otherwise experts take the all-host / gather path. Spill tiering is GGUF-only (needs the file mmap).
pub(crate) fn load_ffn(
    e: &Engine,
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    il: u32,
    spill: Option<(&GgufFile, &mut crate::spill::SpillCtx)>,
    resident: &mut ResidentPlan,
) -> Result<Ffn, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    // MiniMax-M3: moe_layer_freq[il]==0 -> this layer is a DENSE-FFN layer (layers 0..2) even
    // though the arch is MoE; force the Dense arm (its mlp.{p}_proj names map via ggml_to_hf).
    // Hy3: `first_k_dense_replace` leading layers are dense-FFN (REAP50: layer 0 only).
    let dense_override = cfg.m3.as_ref()
        .is_some_and(|m| m.moe_layer_freq.get(il as usize).copied() == Some(0))
        || cfg.hy3.as_ref().is_some_and(|h| il < h.first_k_dense_replace)
        // glm-dsa: leading_dense_block_count layers (GLM-5.2: 3) are dense-FFN
        || cfg.mla.as_ref().is_some_and(|m| il < m.first_k_dense_replace)
        // step35: leading_dense_block_count (Step-3.7-Flash: 3) — blocks 0-2 ship
        // ffn_gate/up/down and NO ffn_gate_inp, so the MoE arm's load_t would fail.
        || cfg.step35.as_ref().is_some_and(|s| il < s.first_k_dense_replace)
        // gemma4 DENSE variants (31B/E4B): the arch is MoE-capable but the file ships no
        // expert tensors at all — tensor presence decides.
        || (cfg.gemma4.is_some() && !src.has(&p("ffn_gate_exps.weight"))
            && !src.has(&p("ffn_gate_up_exps.weight")))
        // A NextN/MTP BLOCK can be DENSE inside a MoE trunk. Step-3.7-Flash's standalone drafter
        // (Step3.7-flash-mtp-Q8_0.gguf) ships blk.45/46/47 with `ffn_gate/up/down.weight` and NO
        // `ffn_gate_inp`/`ffn_*_exps`, while the same file's config declares expert_count=288 (it
        // carries the TRUNK's hparams) — so the MoE arm's `load_t("ffn_gate_exps.weight")` would
        // fail on a perfectly well-formed file. Scoped to `il >= n_trunk` and gated on tensor
        // presence: only an MTP block can take this door, so no trunk MoE layer's dispatch can
        // shift. (qwen35's MTP block IS MoE and keeps the MoE arm — it ships the expert slabs.)
        || (cfg.nextn_predict_layers > 0
            && il >= cfg.n_layer.saturating_sub(cfg.nextn_predict_layers)
            && src.has(&p("ffn_gate.weight"))
            && !src.has(&p("ffn_gate_exps.weight"))
            && !src.has(&p("ffn_gate_up_exps.weight")));
    Ok(
        if let Some(moe) = cfg.moe.as_ref().filter(|_| !dense_override) {
            let n_expert = moe.expert_count as usize;
            // Expert loader. `spill` carries an optional (GgufFile, SpillCtx) — only the GGUF on-disk
            // path can tier (it needs the file mmap); safetensors always gathers/stacks all-host.
            //  - spill Some -> per-expert tier split (hottest pinned, rest mmap'd from the GGUF).
            //  - GGUF 3D stacked name resolves -> load_stacked_from_source (all-host).
            //  - else (safetensors) -> gather N separate 2D expert tensors.
            let (gate_exps, up_exps, down_exps) = match spill {
                Some((g, ctx)) => (
                    HostExps::load_tiered(e, g, &p("ffn_gate_exps.weight"), ctx)?,
                    HostExps::load_tiered(e, g, &p("ffn_up_exps.weight"), ctx)?,
                    HostExps::load_tiered(e, g, &p("ffn_down_exps.weight"), ctx)?,
                ),
                None => {
                    let exps =
                        |e: &Engine, n: &str| -> Result<HostExps, Box<dyn std::error::Error>> {
                            if src.has(n) {
                                HostExps::load_stacked_from_source(e, src, n)
                            } else {
                                HostExps::load_from_source(e, src, n, n_expert)
                            }
                        };
                    // gemma4: gate+up ship FUSED (ffn_gate_up_exps, gate rows first) — split at load.
                    let fused = p("ffn_gate_up_exps.weight");
                    if !src.has(&p("ffn_gate_exps.weight")) && src.has(&fused) {
                        let ff = moe.expert_ff_length as usize;
                        (
                            HostExps::load_stacked_split_from_source(e, src, &fused, 0, ff)?,
                            HostExps::load_stacked_split_from_source(e, src, &fused, ff, 2 * ff)?,
                            exps(e, &p("ffn_down_exps.weight"))?,
                        )
                    } else {
                        (
                            exps(e, &p("ffn_gate_exps.weight"))?,
                            exps(e, &p("ffn_up_exps.weight"))?,
                            exps(e, &p("ffn_down_exps.weight"))?,
                        )
                    }
                }
            };
            // FITS-VRAM RESIDENT EXPERTS: upload this layer's 3 expert slabs when the owning
            // device's budget (MEMRA_MOE_RESIDENT_GB override; default = free VRAM minus the file's
            // non-expert bytes minus a measured headroom reserve) covers the expert bytes assigned
            // to that device, summed exactly from the GGUF header. Decision is made once per device
            // (first MoE layer there). Failure to fit => None => the SLRU spill machinery.
            let dev_exps =
                build_dev_exps(e, resident, il as usize, &gate_exps, &up_exps, &down_exps)?;
            // Device macro row [3*n_expert]: gate, up, down (ones when the artifact carries none).
            let mut macro_row = vec![1.0f32; 3 * n_expert];
            for (slot, exps) in [(0usize, &gate_exps), (1, &up_exps), (2, &down_exps)] {
                if let Some(ms) = exps.macros.as_ref() {
                    macro_row[slot * n_expert..(slot + 1) * n_expert].copy_from_slice(ms);
                }
            }
            let has_macros = macro_row.iter().any(|&m| m != 1.0);
            let dev_macros = e.htod(&macro_row)?;
            // e_score_correction_bias (M3 sigmoid routing): tiny [n_expert] f32, host-side.
            let exp_probs_b = src
                .find(&p("exp_probs_b.bias"))
                .map(|v| memra_gguf::dequant::dequantize(v.ggml_type, &v.bytes, n_expert));
            let active_experts = src.active_experts(il).map(<[bool]>::to_vec);
            Ffn::Moe(MoeWeights {
                gate_inp: load_t(e, src, &p("ffn_gate_inp.weight"))?,
                gate_inp_shexp: load_opt(e, src, &p("ffn_gate_inp_shexp.weight"))?,
                exp_probs_b,
                active_experts,
                gate_exps,
                up_exps,
                down_exps,
                gate_shexp: load_opt(e, src, &p("ffn_gate_shexp.weight"))?,
                up_shexp: load_opt(e, src, &p("ffn_up_shexp.weight"))?,
                down_shexp: load_opt(e, src, &p("ffn_down_shexp.weight"))?,
                dev_exps,
                dev_macros,
                has_macros,
            })
        } else {
            Ffn::Dense {
                ffn_gate: load_t(e, src, &p("ffn_gate.weight"))?,
                ffn_up: load_t(e, src, &p("ffn_up.weight"))?,
                ffn_down: load_t(e, src, &p("ffn_down.weight"))?,
            }
        }
    )
}

/// Decide + build the resident expert slabs for one layer. Budget check runs once per device,
/// RESIDENT-IF-FITS (2026-08-02, research/residency-cap-20260802/): the bank is resident when
/// its EXACT byte total (summed from the GGUF header — UD-quants make per-layer bytes
/// non-uniform, Ornith-35B blk.0 is +7% over the mean, so first-layer x n_layer misprojects)
/// plus the file's non-expert bytes plus a measured headroom reserve fits free VRAM. The old
/// default (0.80 x free vs first-layer x n_layer) reserved 20% of the card (4.8GB on 24GB)
/// and spilled the Ornith-35B bank that fits — a priced -33% decode / -54% prefill. Measured
/// need beside the weights at board shape is ~1.7GB (CUDA ctx + KV + workspace); reserve
/// default 2.0GB, machine-specific override `MEMRA_MOE_RESIDENT_HEADROOM_GB` (VRAM-budget
/// class). `MEMRA_MOE_RESIDENT_GB` stays the absolute expert-budget override;
/// MEMRA_MOE_RESIDENT=0 forces the SLRU path. Fits => every subsequent layer on that device
/// uploads too.
fn build_dev_exps(
    e: &Engine,
    resident: &mut ResidentPlan,
    il: usize,
    gate: &HostExps,
    up: &HostExps,
    down: &HostExps,
) -> Result<Option<crate::hybrid::DevExps>, Box<dyn std::error::Error>> {
    // The resident pointer-table kernels take one qtype/row stride per projection. Mixed-expert
    // layers stay on the metadata-aware staged/SLRU paths until those kernels group by layout.
    if !gate.is_uniform_layout() || !up.is_uniform_layout() || !down.is_uniform_layout() {
        return Ok(None);
    }
    let per_layer =
        gate.bytes.as_bytes().len() + up.bytes.as_bytes().len() + down.bytes.as_bytes().len();
    if gate.tiers.is_some() {
        return Ok(None); // tiered/spill loads keep the cache path
    }
    let fits = resident.should_reside(e, il, per_layer);
    if !fits {
        return Ok(None);
    }
    use cudarc::driver::DevicePtr;
    let gu_il = std::env::var("MEMRA_MOE_GU_IL").as_deref() == Ok("1")
        && gate.out_f == up.out_f
        && gate.in_f == up.in_f;
    let n_expert = gate.n_expert;
    let (g, u) = if gu_il {
        // interleave gate/up rows: [ex][row o] = gate-row-o bytes ++ up-row-o bytes.
        let (rbg, rbu) = (gate.row_bytes, up.row_bytes);
        let n_rows = gate.out_f;
        let gb = gate.bytes.as_bytes();
        let ub = up.bytes.as_bytes();
        let mut il = vec![0u8; n_expert * n_rows * (rbg + rbu)];
        for ex in 0..n_expert {
            for o in 0..n_rows {
                let dst = (ex * n_rows + o) * (rbg + rbu);
                let sg = ex * gate.expert_stride + o * rbg;
                let su = ex * up.expert_stride + o * rbu;
                il[dst..dst + rbg].copy_from_slice(&gb[sg..sg + rbg]);
                il[dst + rbg..dst + rbg + rbu].copy_from_slice(&ub[su..su + rbu]);
            }
        }
        let ild = e.htod_bytes_padded(&il, 8)?;
        // `up` slot points into the same buffer via ptr math; keep a tiny placeholder alloc so
        // the struct shape is unchanged (the table below carries the real pointers).
        (ild, e.htod_bytes(&[0u8; 16])?)
    } else {
        (
            e.htod_bytes_padded(gate.bytes.as_bytes(), 8)?,
            e.htod_bytes_padded(up.bytes.as_bytes(), 8)?,
        )
    };
    // 144B tail slack (2026-07-31, g26 prefill lever): the ragged-k expert MMA walks
    // whole 256-val superblocks — the LAST row's final partial superblock overreads up
    // to 144B past the slab (harmless bytes: the act's zero-padded k-range multiplies
    // every overread weight to zero; the slack only prevents the OOB fault).
    let d = e.htod_bytes_padded(down.bytes.as_bytes(), 144)?;
    let mut host = vec![0u64; 3 * n_expert];
    let (pg, pu, pd) = {
        let __s_e0 = e.stream();
        let (pg, _e0) = g.device_ptr(&__s_e0);
        let __s_e1 = e.stream();
        let (pu, _e1) = u.device_ptr(&__s_e1);
        let __s_e2 = e.stream();
        let (pd, _e2) = d.device_ptr(&__s_e2);
        (pg as u64, pu as u64, pd as u64)
    };
    for ex in 0..n_expert {
        if gu_il {
            let stride = gate.out_f * (gate.row_bytes + up.row_bytes);
            host[ex] = pg + (ex * stride) as u64;
            host[n_expert + ex] = pg + (ex * stride + gate.row_bytes) as u64;
        } else {
            host[ex] = pg + (ex * gate.expert_stride) as u64;
            host[n_expert + ex] = pu + (ex * up.expert_stride) as u64;
        }
        host[2 * n_expert + ex] = pd + (ex * down.expert_stride) as u64;
    }
    if gu_il {
        eprintln!("[moe] gate/up dev slab INTERLEAVED (MEMRA_MOE_GU_IL)");
    }
    let ptr_row = e.htod_u64(&host)?;
    Ok(Some(crate::hybrid::DevExps {
        gate: g,
        up: u,
        down: d,
        ptr_row,
        gu_il,
        dev: e.ctx().ordinal(),
    }))
}

pub struct FullAttnLayer {
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// step35-class SEPARATE head-wise attention gate: `blk.N.attn_gate.weight [n_embd, n_head_l]`
    /// where `n_head_l` is this layer's query-head count (64 full / 96 SWA on Step-3.7-Flash, so
    /// the width VARIES per layer). Produces one pre-sigmoid scalar per head from the
    /// post-attn_norm hidden state; the forward broadcasts sigmoid(gate) over head_dim and
    /// multiplies attn_out before wo (upstream `step35.cpp:267-285`).
    ///
    /// `None` for every other arch. Do NOT confuse with `LinearAttnLayer::wqkv_gate`, which reads
    /// the SAME tensor name on qwen35's SSM layers but is a different mechanism (a full-width
    /// z-gate, not a per-head scalar), nor with the qwen35 FUSED gate packed inside wq that
    /// `ModelConfig::attn_out_gate()` / `q_gate_split` handle.
    pub attn_gate: Option<GpuTensor>,
}

/// Latent-KV geometry for one MLA layer, resolved at load from `MlaConfig` (glm-dsa). The KV
/// cache stores ONE `latent_dim`-wide row per token per layer: [rmsnorm(c_kv) | rope(k_pe)];
/// V is the first `kv_rank` elements of the SAME row (no V plane). All heads stream it (MQA).
#[derive(Clone, Copy, Debug)]
pub struct MlaGeom {
    pub n_head: usize,     // 64  — query heads; n_head_kv semantics = 1
    pub d_nope: usize,     // 192 — qk nope head dim (absorb GEMM K)
    pub d_rope: usize,     // 64  — decoupled rope width (q_pe / k_pe)
    pub d_v: usize,        // 256 — v head dim after wv_b decompression
    pub kv_rank: usize,    // 512 — latent rank (absorbed qk dim, AV accumulator width)
    pub latent_dim: usize, // 576 = kv_rank + d_rope — the cache row / K width
    pub scale: f32,        // 1/sqrt(d_nope + d_rope) = 1/16 — NOT 1/sqrt(latent_dim)
}

/// GLM-5.2 MLA attention block (DESIGN.md §3.1 mapping). INCREMENT 2: loader-only — the
/// projections + latent-cache geometry land on device; forward arms (prefill/decode/dc/graph)
/// are increment 4. The CPU oracle for those arms is `crate::mla` (naive ≡ absorbed, proven).
pub struct MlaAttnLayer {
    pub wq_a: GpuTensor,      // attn_q_a.weight      [H -> Lq] (q down-projection)
    pub q_a_norm: GpuTensor,  // attn_q_a_norm.weight [Lq]
    pub wq_b: GpuTensor,      // attn_q_b.weight      [Lq -> N*(nope+rope)] (q up, per head [nope|rope])
    pub wkv_a: GpuTensor,     // attn_kv_a_mqa.weight [H -> Lkv+rope] (latent row producer)
    pub kv_a_norm: GpuTensor, // attn_kv_a_norm.weight [Lkv] (c_kv rms; k_pe is NOT normed)
    pub wk_b: GpuTensor,      // attn_k_b.weight      [nope, Lkv, N] 3D — TRANSPOSED nope slice of
                              //   kv_b (conversion split): the per-head absorb GEMM operand
    pub wv_b: GpuTensor,      // attn_v_b.weight      [Lkv, V, N] 3D — the post-softmax decompress
    pub wo: GpuTensor,        // attn_output.weight   [N*V -> H]
    pub geom: MlaGeom,
}

impl MlaAttnLayer {
    /// Load one MLA attention block to device. `attn_kv_b` (the unsplit tensor, when present)
    /// is intentionally NOT loaded — v1 runs absorbed-form everywhere; the MHA-prefill arm that
    /// would consume it is a later arc (DESIGN.md §3.1 "unused v1").
    ///
    /// NOTE (increment-3+): wk_b/wv_b are 3D. The F32 fixture rides the Float path (exact, full
    /// ne kept). Quantized 3D tensors would mis-derive `row_bytes` in the generic 2D Quant arm
    /// (out_f = ne[1] only) — the real-weights loader must split per head or flatten ne[1]*ne[2]
    /// before the batched-GEMM kernels consume them. Guarded by the assert below.
    pub fn load(
        e: &Engine,
        src: &dyn TensorSource,
        il: u32,
        m: &MlaConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let p = |s: &str| format!("blk.{il}.{s}");
        let geom = MlaGeom {
            n_head: 0, // patched below from wq_b's out width (metadata cross-check)
            d_nope: m.qk_nope_head_dim as usize,
            d_rope: m.qk_rope_head_dim as usize,
            d_v: m.v_head_dim as usize,
            kv_rank: m.kv_lora_rank as usize,
            latent_dim: m.latent_dim() as usize,
            scale: m.scale(),
        };
        let wq_a = load_t(e, src, &p("attn_q_a.weight"))?;
        let wq_b = load_t(e, src, &p("attn_q_b.weight"))?;
        let wkv_a = load_t(e, src, &p("attn_kv_a_mqa.weight"))?;
        let wk_b = load_t(e, src, &p("attn_k_b.weight"))?;
        let wv_b = load_t(e, src, &p("attn_v_b.weight"))?;
        let wo = load_t(e, src, &p("attn_output.weight"))?;
        // shape audit at load (fail loudly, not as garbage activations later):
        let n_head = wq_b.out_features() / (geom.d_nope + geom.d_rope);
        assert_eq!(wq_b.out_features(), n_head * (geom.d_nope + geom.d_rope),
                   "wq_b out {} not a multiple of qk_head_dim {}", wq_b.out_features(),
                   geom.d_nope + geom.d_rope);
        assert_eq!(wq_a.in_features() , wkv_a.in_features(), "q_a/kv_a hidden mismatch");
        assert_eq!(wq_b.in_features(), m.q_lora_rank as usize, "wq_b in != q_lora_rank");
        assert_eq!(wkv_a.out_features(), geom.latent_dim, "wkv_a out != kv_lora_rank + rope");
        assert_eq!(wk_b.ne(), &[geom.d_nope as u64, geom.kv_rank as u64, n_head as u64],
                   "attn_k_b must be the TRANSPOSED (nope, kv_rank, head) conversion split");
        assert_eq!(wv_b.ne(), &[geom.kv_rank as u64, geom.d_v as u64, n_head as u64],
                   "attn_v_b must be the (kv_rank, v, head) conversion split");
        assert_eq!(wo.in_features(), n_head * geom.d_v, "wo in != n_head * v_head_dim");
        Ok(MlaAttnLayer {
            wq_a,
            q_a_norm: load_t(e, src, &p("attn_q_a_norm.weight"))?,
            wq_b,
            wkv_a,
            kv_a_norm: load_t(e, src, &p("attn_kv_a_norm.weight"))?,
            wk_b,
            wv_b,
            wo,
            geom: MlaGeom { n_head, ..geom },
        })
    }
}

/// Increment-2 guard: every forward-path `match` on `Mixer` routes Mla here until increment 4
/// lands the MLA kernels. Loading a glm-dsa model works; running it panics with THIS message
/// instead of garbage math. Zero behavior change for Full/Linear arches (arm never taken).
#[track_caller]
pub(crate) fn mla_forward_unimplemented() -> ! {
    panic!("Mixer::Mla has no forward arm yet — glm-dsa is loader-only in increment 2; \
            the CUDA forward lands in increment 4 (research/mla-bringup-20260801/DESIGN.md §4)")
}

pub struct LinearAttnLayer {
    pub wqkv: GpuTensor,       // [n_embd, conv_dim] -> qkv_mixed
    pub wqkv_gate: GpuTensor,  // [n_embd, value_dim] -> z
    pub ssm_beta: GpuTensor,   // [n_embd, num_v_heads]
    pub ssm_alpha: GpuTensor,  // [n_embd, num_v_heads]
    pub ssm_a: GpuTensor,      // [num_v_heads] (pre-negated -exp(A_log))
    pub ssm_dt: GpuTensor,     // [num_v_heads] bias
    pub ssm_conv1d: GpuTensor, // [d_conv, conv_dim]
    pub ssm_norm: GpuTensor,   // [head_v_dim]
    pub ssm_out: GpuTensor,    // [value_dim, n_embd]
}

pub enum Mixer {
    Full(FullAttnLayer),
    Linear(LinearAttnLayer),
    /// glm-dsa MLA block (loader-only in increment 2; forward = increment 4).
    Mla(MlaAttnLayer),
}

/// MoE weights for one layer. Router + shared expert stay GPU-RESIDENT (tiny); the routed
/// experts stay HOST-RESIDENT (HostExps) and are staged per-token (EDGE-1).
///
/// The shared-expert fields are `Option`: qwen35moe carries a shared expert, but OLMoE (and most
/// vanilla MoE) have none (`shared_expert_intermediate_size` absent) — those layers `load_opt` the
/// shexp tensors to `None` (ST-MOE-PLAN §1.3, §3.2). When `None` the shared-expert branch is skipped.
pub struct MoeWeights {
    pub gate_inp: GpuTensor, // F32 [n_embd, n_expert] router  (GPU resident, Float)
    pub gate_inp_shexp: Option<GpuTensor>, // F32 [n_embd] 1-D shared gate dot (qwen35moe only)
    /// DeepSeek-V3/MiniMax-M3 `e_score_correction_bias` [n_expert]: added to the sigmoid scores
    /// for expert SELECTION only; the routing weights use the un-biased scores. Kept host-side —
    /// routing's top-k is a host loop and this is n_expert floats.
    pub exp_probs_b: Option<Vec<f32>>,
    /// Original-width router mask for physically pruned expert overlays. Inactive ids never enter
    /// top-k, so their absent weight files cannot be dispatched.
    pub active_experts: Option<Vec<bool>>,
    pub gate_exps: HostExps, // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub up_exps: HostExps,   // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub down_exps: HostExps, // [n_ff_exp, n_embd, n_expert] TRANSPOSED (HOST)
    pub gate_shexp: Option<GpuTensor>,
    pub up_shexp: Option<GpuTensor>,
    pub down_shexp: Option<GpuTensor>,
    /// FITS-VRAM RESIDENT EXPERTS (2026-07-06): when the WHOLE model's expert bytes fit the VRAM
    /// budget, each (proj) slab is uploaded once as a contiguous device buffer and the fused
    /// _dev kernels take base+ex*stride pointers — no SLRU, no dispatch, no residency checks
    /// (llama's full-offload regime; measured 169.55 vs memra's cache path 28.5 on the local 35B).
    /// None => the SLRU host-expert machinery (the spill regime, where it WINS vs llama's
    /// CPU-offload degradation). Decided at load in `load_ffn` (MEMRA_MOE_RESIDENT=0 forces off).
    pub dev_exps: Option<DevExps>,
    /// Per-expert post-matmul macro-scales on DEVICE: [3*n_expert] f32 in (gate, up, down)
    /// order — all 1.0 unless the checkpoint carries compressed-tensors NVFP4 global scales
    /// (unsloth qwen3.6 class). The _dev gate_up epilogues multiply unconditionally (x*1.0f
    /// is bit-exact — zero change for macro-free artifacts); the down fold is one
    /// moe_w_scale_by_expert launch gated on `has_macros`.
    pub dev_macros: cudarc::driver::CudaSlice<f32>,
    pub has_macros: bool,
}

impl MoeWeights {
    #[inline]
    pub fn has_uniform_expert_layout(&self) -> bool {
        self.gate_exps.is_uniform_layout()
            && self.up_exps.is_uniform_layout()
            && self.down_exps.is_uniform_layout()
    }
}

/// Device-resident expert slabs for one layer (gate/up/down) + the prebuilt [3, n_expert]
/// pointer row the _dev kernels consume.
pub struct DevExps {
    pub gate: CudaSlice<u8>,
    pub up: CudaSlice<u8>,
    pub down: CudaSlice<u8>,
    /// [3*n_expert] u64 device row: gate ptrs, up ptrs, down ptrs (proj-major like layer_dev_row).
    pub ptr_row: CudaSlice<u64>,
    /// The CUDA device ordinal these slabs live on (the OWNING stage's device under the PP
    /// sharded loader — cx-503b sizes and `layer_engine` places per device). Consumers that
    /// dispatch from a DIFFERENT device must NOT dereference the slabs: an m=1 qmatvec over
    /// peer-read expert bytes is the measured 34-150x slow class (research/pp-prefill-20260807
    /// anatomy), strictly worse than SLRU staging. The sequential arm's slab-locality gate
    /// (lane/pp-leverb) keys on this field; the per-stage prime walker makes every layer's
    /// slab local by construction.
    pub dev: usize,
    /// WALL-GAP ARC (MEMRA_MOE_GU_IL=1): gate/up rows INTERLEAVED in one slab — row o of gate at
    /// base + o*(rb_g+rb_u), up at +rb_g. Consumers on the dev path must use (rb_g+rb_u) as the
    /// row stride for BOTH projections (see MoeWeights::dev_rb_gu). One contiguous 1760B stream
    /// per (expert,row) instead of two scattered 880B streams — the measured 56%-of-wall fix
    /// candidate. Kernels unchanged (stride is already a parameter everywhere).
    pub gu_il: bool,
}

/// Per-layer FFN: dense SwiGLU (qwen35) or 256-expert MoE (qwen35moe).
pub enum Ffn {
    Dense {
        ffn_gate: GpuTensor,
        ffn_up: GpuTensor,
        ffn_down: GpuTensor,
    },
    Moe(MoeWeights),
}

pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor, // "post_attention_norm" = PRE-FFN norm
    pub mixer: Mixer,
    pub ffn: Ffn,
    pub gemma4: Option<Gemma4LayerBits>,
}

/// Gemma-4 per-layer extras (R8 wiring, HANDOVER "R8 VERIFIED WIRING"): the parallel shared
/// FFN branch, the four extra norms, the router prologue scale vector, per-expert output
/// scales, and the layer output scalar.
pub struct Gemma4LayerBits {
    pub ffn_norm: GpuTensor, // ffn pre-norm (dense: THE ffn norm; moe: shared branch)
    pub post_ffw_norm: GpuTensor, // combined post (before the attn_out residual)
    /// MoE-layer extras (None on the dense gemma4 variants — 31B/E4B): the parallel shared
    /// branch norms + tensors, the router prologue vector, per-expert output scales.
    pub moe_bits: Option<Gemma4MoeBits>,
    pub layer_scale: f32, // layer_output_scale [1]
    /// E4B extras (None on 26B/31B): the per-layer-embedding tail block + KV-share target.
    pub e4b: Option<Gemma4E4bLayer>,
}

/// gemma-4 E4B per-layer bits (see research/gemma4-bringup/e4b-arch-map.md):
/// tail block  cur += rms_norm(proj . (gelu(inp_gate . cur) * inp_pl[il]), post_norm)
/// and the KV-share map — layers il >= n_layer-shared_kv_layers have NO own k/v projections
/// and attend the cache of layer (n_layer-shared) - (swa ? 2 : 1) with their own Q.
pub struct Gemma4E4bLayer {
    pub inp_gate: GpuTensor,           // blk.N.inp_gate  [n_embd, n_epl]
    pub proj: GpuTensor,               // blk.N.proj      [n_epl, n_embd]
    pub post_norm: GpuTensor,          // blk.N.post_norm [n_embd]
    /// wave-4b: wq|wk|wv concatenated along OUT (one Q4_0 matvec at t=1 instead of the
    /// fused3 3-subgrid launch). Built at the mirror hook from the GPU byte planes (rows
    /// are independent in Q4_0, so an out-dim concat is a byte concat); own-KV layers only.
    pub qkv_cat: Option<GpuTensor>,
    /// Some(target_layer) on KV-shared layers (wk/wv here are the TARGET layer's tensors,
    /// loaded for shape symmetry only — the forward must skip k/v compute + append and read
    /// the target's cache; TODO dedupe the duplicate weight upload ~63MB).
    pub kv_share: Option<u32>,
}

/// gemma-4 E4B model-level per-layer-embedding tensors (prologue inputs). The token table
/// stays HOST-side raw GGUF bytes at load (Q6_K [n_epl*n_layer, n_vocab], ~2.3GB VRAM when
/// uploaded — the forward arc decides resident-vs-gather placement).
pub struct Gemma4E4bModel {
    /// device copy of the per-layer token table, uploaded on first use (the 26B embd_gpu
    /// pattern — keeps the ~2.3GB off load-critical paths that never decode).
    pub tok_tbl_gpu: std::sync::OnceLock<CudaSlice<u8>>,
    pub tok_embd_bytes: Vec<u8>,
    pub tok_embd_qt: i32,
    pub tok_embd_row_bytes: usize,
    pub model_proj: GpuTensor, // per_layer_model_proj [n_embd, n_epl*n_layer] F16
    pub proj_norm: GpuTensor,  // per_layer_proj_norm [n_epl]
    pub n_epl: usize,
}

pub struct Gemma4MoeBits {
    pub post_ffw_norm_1: GpuTensor, // shared-branch post
    pub pre_ffw_norm_2: GpuTensor,  // moe-branch pre
    pub post_ffw_norm_2: GpuTensor, // moe-branch post
    pub shared_gate: GpuTensor,
    pub shared_up: GpuTensor,
    pub shared_down: GpuTensor,
    /// ffn_gate_inp.scale [n_embd] PRE-multiplied by 1/sqrt(n_embd) at load: the router
    /// prologue (weightless rms_norm x 1/sqrt(n_embd) x scale-vec) collapses to ONE rms_norm
    /// with this as the norm weight (x_hat * (v*s) vs llama's (x_hat*s)*v — one reassociation;
    /// the argmax gate arbitrates).
    pub router_scale_pre: CudaSlice<f32>,
    pub per_expert_scale: Vec<f32>, // ffn_down_exps.scale [n_expert] (host)
    pub per_expert_scale_d: CudaSlice<f32>, // device copy (router-weight fold kernel)
}

/// Qwen3.5 NextN/MTP head: a full transformer block (attn+FFN, same tensors as a trunk layer)
/// plus the MTP glue (enorm/hnorm/eh_proj that fold the next-token embedding into the trunk
/// hidden, and an optional shared_head_norm/head). Loaded from blk.{n_trunk}.* — the block the
/// trunk loop drops. Used for speculative decode (drafts 1 token per call). See research/mtp/MTP-PLAN.md.
pub struct MtpHead {
    pub enorm: GpuTensor, // blk.N.nextn.enorm   — RMSNorm of the next-token embedding
    pub hnorm: GpuTensor, // blk.N.nextn.hnorm   — RMSNorm of the trunk hidden
    pub eh_proj: GpuTensor, // blk.N.nextn.eh_proj [2*n_embd, n_embd]: [e_norm; h_norm] -> n_embd
    pub attn_norm: GpuTensor, // blk.N.attn_norm
    pub post_attn_norm: GpuTensor, // blk.N.post_attention_norm (pre-FFN)
    pub mixer: Mixer,     // full-attn block (qwen35 MTP block is full-attn)
    pub ffn: Ffn,         // Dense or Moe, same loader as trunk
    pub shared_head_norm: Option<GpuTensor>, // blk.N.nextn.shared_head_norm (else reuse output_norm)
    pub shared_head_head: Option<GpuTensor>, // blk.N.nextn.shared_head      (else reuse output)
    /// FR-Spec draft->target vocab map: the draft lm_head is TRIMMED to the highest-frequency
    /// tokens (e.g. 32768 rows of the full 248320-row head); `d2t[draft_idx]` = the target vocab
    /// token id of trimmed row `draft_idx`. `None` for a full-vocab head (identity map). Host-side:
    /// the draft argmax already lands on host as one u32, so the map is a single Vec index.
    pub d2t: Option<Vec<u32>>,
    /// DISTILLED-STUDENT geometry (None = the natural NextN block at trunk shape). A distilled
    /// draft (StudentSV) runs the same block structure at a narrower inner width with fewer
    /// heads, then up-projects back to n_embd (`out_up`) — the chain carrier and the head input
    /// stay at n_embd, so the trunk/verify interface is unchanged. Selected by the presence of
    /// `blk.N.nextn.out_up.weight` in a MEMRA_MTP_DRAFT file.
    pub geom: Option<DraftGeom>,
    /// step35: the DRAFT BLOCK's RESOLVED per-layer geometry (`None` for every arch whose
    /// geometry is uniform). Without it the head forward would use the trunk's max-derived
    /// scalars and compute wrong attention — and the failure mode is plausible-but-wrong drafts
    /// (tanked acceptance, correct output), exactly what the exactness gates cannot see.
    pub step35: Option<Step35MtpGeom>,
}

/// step35 MTP-block geometry, RESOLVED at load time from the file that actually carries the
/// block's own `Step35Config` arrays.
///
/// Why resolved and not "look it up per forward from the model's cfg": Step-3.7-Flash ships MTP
/// as a SEPARATE GGUF, and the two files disagree about which layers exist. The trunk artifact
/// declares `block_count=45` / `nextn_predict_layers=0`, so its per-layer arrays hold 45 entries
/// (0..=44) and `Step35Config::n_head(45)` falls off the end into the `.last()` fallback — index
/// 44, which is a FULL-attn layer at 64 heads. The draft file declares `block_count=48` /
/// `nextn=3` and its arrays' index 45 is the truth: SWA, 96 heads (matching that file's
/// `blk.45.attn_q.weight [4096, 12288]` = 96*128 and `blk.45.attn_gate.weight [4096, 96]`).
/// Receipt: `research/step37-bringup-20260802/raw/gguf-header-stepfun-mtp-q8-20260802.txt` plus
/// the tail dump in `research/step37-p2-20260806/raw/` — `head_count[43..48] = [96, 64, 96, 96,
/// 96]`, `sliding_window_pattern[43..48] = [True, False, True, True, True]`.
#[derive(Debug, Clone)]
pub struct Step35MtpGeom {
    /// Block index inside the file that carries it (45 for Step-3.7-Flash). Diagnostics only.
    pub il: u32,
    pub n_head: usize,    // 96 on Step-3.7-Flash's MTP block (SWA-type)
    pub n_head_kv: usize, // 8
    pub n_rot: usize,     // 128 (SWA keeps the unhalved rotary width)
    pub rope_base: f32,   // 1e4 (SWA base, not the trunk's 5e6 global)
    pub swa: bool,        // true
    pub window: usize,    // 512
    /// This block's `swiglu_clamp_shexp` limit. The MTP block's FFN is a DENSE SwiGLU, and
    /// upstream's one `build_ffn` serves both the dense MLP and the shared expert off the
    /// SHEXP array (llama-graph.cpp:1751) — so a dense MTP block keys off shexp, not exp.
    /// 0.0 (`None`) on Step-3.7-Flash's block 45; live (16.0) only on trunk layers 43-44.
    pub clamp_shexp: Option<f32>,
}

impl Step35MtpGeom {
    /// Resolve block `il`'s geometry from the `Step35Config` of the file that OWNS that block.
    pub fn resolve(s: &memra_gguf::config::Step35Config, il: u32) -> Self {
        Step35MtpGeom {
            il,
            n_head: s.n_head(il) as usize,
            n_head_kv: s.n_head_kv(il) as usize,
            n_rot: s.n_rot(il) as usize,
            rope_base: s.rope_base(il),
            swa: s.is_swa(il),
            window: s.sliding_window as usize,
            clamp_shexp: s.clamp_shexp(il),
        }
    }
}

/// Draft-head geometry override for a distilled (narrower) student block.
pub struct DraftGeom {
    pub d_inner: usize, // block inner width (eh_proj out / attn / ffn), e.g. 2048
    pub n_head: usize,  // draft attention heads (head_dim = main head_dim)
    pub n_head_kv: usize,
    pub out_up: GpuTensor, // [d_inner -> n_embd]: carrier + head input up-projection
}

/// Which tensor is the DRAFT lm_head, for a standalone NextN/MTP draft GGUF whose block index is
/// `n`. Preference order is the artifact's, not ours — upstream step35.cpp:553 is
/// `layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output`.
///
/// Split out of `MtpHead::load_draft` purely so it is unit-testable: the loader needs a CUDA
/// device and a multi-GB file, while the failure this guards is invisible to every exactness gate
/// (a wrong head still produces CORRECT output — the verify arbitrates — it just accepts nothing).
/// `has` is the tensor-presence predicate (`src.has`).
pub fn draft_head_tensor(has: impl Fn(&str) -> bool, n: u32) -> String {
    let own = format!("blk.{n}.nextn.shared_head_head.weight");
    if has(&own) {
        return own;
    }
    // Legacy name kept as a probe so anything that ever matched it still does; no shipped
    // artifact or upstream mapping uses it (see the `load_draft` note).
    let legacy = format!("blk.{n}.nextn.shared_head.weight");
    if has(&legacy) {
        return legacy;
    }
    // FR-Spec / tied-head drafts: the file-level head IS the draft head.
    "output.weight".to_string()
}

impl MtpHead {
    /// Load an MTP/NextN head from a STANDALONE draft GGUF (MEMRA_MTP_DRAFT override). The draft
    /// file carries ONLY the NextN block (blk.N.nextn.* glue + attn/ffn) plus its own lm_head
    /// (`output.weight`) — which for an FR-Spec draft is TRIMMED to the top-frequency rows, with
    /// a `d2t` (i32/i64) tensor mapping trimmed-row index -> target vocab token id. Draft-token
    /// embedding still uses the MAIN model's token_embd (identical weights, saves VRAM), so the
    /// draft file's full-vocab token_embd copy is ignored.
    pub fn load_draft(
        e: &Engine,
        g: &GgufFile,
        main_cfg: &ModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let src = GgufSource(g);
        let dcfg = src.config();
        // NextN block index INSIDE THE DRAFT FILE (its block_count includes the trunk numbering).
        // Graceful error, not assert: the server's `+draft` attach path surfaces this to the
        // user (a gemma-assistant draft or any non-NextN GGUF lands here; a panic killed the
        // whole worker — serve-smoke find, 2026-07-30).
        if dcfg.nextn_predict_layers == 0 {
            return Err(format!(
                "draft GGUF has no nextn_predict_layers (arch {:?}) — not a NextN/MTP regime \
                 draft; gemma assistant drafters attach via MEMRA_DRAFT, not '+draft'",
                g.arch()).into());
        }
        let n = dcfg.n_layer - dcfg.nextn_predict_layers;
        let p = |s: &str| format!("blk.{n}.{s}");

        // Distilled student (narrow block + out_up) vs natural NextN clone. The interface dims
        // (n_embd in/out, head_dim for the shared rope kernel) must match the main model; a
        // student may shrink the inner width and head counts.
        let student = src.has(&p("nextn.out_up.weight"));
        assert_eq!(dcfg.n_embd, main_cfg.n_embd, "draft n_embd != model n_embd");
        assert_eq!(
            dcfg.head_dim_k, main_cfg.head_dim_k,
            "draft head_dim != model head_dim"
        );
        // step35: geometry is PER-LAYER, so "same shape as the trunk" is the wrong question — the
        // draft block at il=45 is an SWA-type block (96 q heads, 128 rotary dims, rope base 1e4)
        // while the trunk's full-attn layers are 64/64/5e6. Resolve the block's geometry from the
        // DRAFT FILE's own arrays (the trunk artifact's arrays stop at index 44 — see
        // `Step35MtpGeom`'s note) and verify it against the block's real tensor shapes. The dims
        // that must still agree with the trunk are the INTERFACE ones (n_embd, head_dim, KV width).
        let step35 = match (main_cfg.step35.as_ref(), dcfg.step35.as_ref()) {
            (Some(_), Some(s)) => {
                let g = Step35MtpGeom::resolve(s, n);
                // ne is inner-fastest: ne[0] = in_features, ne[1] = out_features for a [in, out] 2D.
                let out_f = |t: &str| -> Option<usize> {
                    src.find(&p(t)).and_then(|v| v.ne.get(1).copied()).map(|x| x as usize)
                };
                let hd = dcfg.head_dim_k as usize;
                let wq_out = out_f("attn_q.weight")
                    .ok_or("step35 draft block has no attn_q.weight")?;
                assert_eq!(
                    wq_out, g.n_head * hd,
                    "step35 draft blk.{n}: attn_q out {wq_out} != n_head({}) * head_dim({hd}) — \
                     the draft file's head_count array disagrees with its own tensors",
                    g.n_head
                );
                // The SEPARATE head-wise gate is [n_embd, n_head_l] — one scalar per head. Its
                // width is the second independent witness of this block's head count.
                let wg_out = out_f("attn_gate.weight")
                    .ok_or("step35 draft block has no attn_gate.weight (head-wise gate)")?;
                assert_eq!(
                    wg_out, g.n_head,
                    "step35 draft blk.{n}: attn_gate out {wg_out} != n_head({})", g.n_head
                );
                // The draft attends its OWN scratch, but `MtpScratch::new` sizes those rows from
                // the TRUNK cfg's `n_head_kv` (for step35, the max over its per-layer array).
                // Compare against exactly that value, not a per-layer accessor.
                assert_eq!(
                    g.n_head_kv, main_cfg.n_head_kv as usize,
                    "step35 draft blk.{n} KV heads {} != trunk n_head_kv {} — the MTP scratch \
                     rows are sized from the trunk cfg, so a differing draft KV width would \
                     write past the row",
                    g.n_head_kv, main_cfg.n_head_kv
                );
                eprintln!(
                    "[mtp-draft] step35 MTP geometry blk.{n}: n_head={} n_head_kv={} n_rot={} \
                     rope_base={:.0} swa={} window={}",
                    g.n_head, g.n_head_kv, g.n_rot, g.rope_base, g.swa, g.window
                );
                Some(g)
            }
            (Some(_), None) => {
                return Err(format!(
                    "MEMRA_MTP_DRAFT points at a non-step35 GGUF (arch {:?}) but the model is \
                     step35 — the draft block's per-layer geometry is unknowable from the trunk \
                     config (its arrays stop at the trunk's last layer)",
                    g.arch()
                ).into())
            }
            (None, Some(_)) => {
                return Err("MEMRA_MTP_DRAFT is a step35 draft but the model is not step35".into())
            }
            (None, None) => None,
        };
        if step35.is_none() && !student {
            // The head forward runs with the MAIN model's cfg — the draft block must be the
            // same shape or the forward is garbage.
            assert_eq!(dcfg.n_head, main_cfg.n_head, "draft n_head != model n_head");
            assert_eq!(
                dcfg.n_head_kv, main_cfg.n_head_kv,
                "draft n_head_kv != model n_head_kv"
            );
        }

        // Draft lm_head. PREFERENCE ORDER IS THE ARTIFACT'S, NOT OURS (upstream step35.cpp:553
        // `layer.nextn.shared_head_head ? ... : model.output`): a NextN block owns its OWN head,
        // and only a file that omits it falls back to the file-level `output.weight`.
        //
        // MEASURED ON THE SHIPPED ARTIFACT (Step3.7-flash-mtp-Q8_0.gguf, byte hashes in
        // research/step37-p2-20260806/raw/draft-head-tensor-hashes-20260807.txt): the file carries
        // BOTH, they are DIFFERENT matrices, and the three MTP blocks' heads differ from each
        // other too —
        //     output.weight                        sha 3eec5831…  <- the TRUNK lm_head, re-quantized
        //     blk.45.nextn.shared_head_head.weight sha c90b907b…  <- block 45's own head
        //     blk.46 …                             sha a22d2957…
        //     blk.47 …                             sha 4b21e137…
        // The tell: this file's top-level `output_norm.weight` is BYTE-IDENTICAL to the trunk
        // artifact's (both sha d7526f44…), i.e. the top level is a copy of the trunk's output
        // stack, present so the draft gguf stands alone. Reading it as the draft head projects
        // the MTP block's hidden through the TRUNK's head — coherent-looking drafts the verify
        // never accepts. Receipt: acceptance 0/248 across K=1..8 with self-consistency PASS
        // (raw/mtp-draft-20260806T212902Z.log) — the exact failure class run_spec.rs's
        // "acceptance == 0 with identical output" WARNING exists to catch.
        //
        // FR-Spec drafts (trimmed [n_embd, draft_vocab] + d2t) publish the trimmed head as the
        // file-level `output.weight` and carry no `nextn.shared_head_head`, so they keep the
        // fallback — hence preference, not replacement.
        // Name choice is factored into `draft_head_tensor` so it is testable WITHOUT a GPU or a
        // 3.5 GB artifact (this whole function needs both). Getting it wrong is invisible to
        // every exactness gate, so the choice itself is pinned by a unit test.
        let head_name = draft_head_tensor(|t| src.has(t), n);
        let head = load_t(e, &src, &head_name)?;
        let head_norm = match load_opt(e, &src, &p("nextn.shared_head_norm.weight"))? {
            Some(t) => Some(t),
            None => load_opt(e, &src, "output_norm.weight")?,
        };

        // d2t: draft-row -> target-token-id map (absolute ids, verified against the tokenizer).
        let d2t: Option<Vec<u32>> = g.find("d2t").map(|t| {
            let bytes = g.tensor_data(t);
            match t.ggml_type {
                GgmlType::I32 => bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                    .collect(),
                GgmlType::I64 => bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
                    .collect(),
                other => panic!("d2t must be I32/I64, got {other:?}"),
            }
        });
        if let Some(map) = &d2t {
            assert_eq!(
                map.len(),
                head.out_features(),
                "d2t len {} != draft head rows {}",
                map.len(),
                head.out_features()
            );
            let n_vocab = main_cfg.n_vocab as u64;
            assert!(
                map.iter().all(|&t| (t as u64) < n_vocab),
                "d2t contains token id >= model n_vocab {n_vocab}"
            );
        }
        let eh_proj = load_t(e, &src, &p("nextn.eh_proj.weight"))?;
        // defensive load gates (review feedback): a malformed student gguf fails HERE with a
        // named assert, not later as garbage drafts. eh_proj consumes concat(e_norm, h_norm).
        assert_eq!(
            eh_proj.in_features(),
            2 * main_cfg.n_embd as usize,
            "eh_proj in dim != 2*n_embd"
        );
        let geom = if student {
            let out_up = load_t(e, &src, &p("nextn.out_up.weight"))?;
            let d_inner = eh_proj.out_features();
            assert_eq!(
                out_up.out_features(),
                main_cfg.n_embd as usize,
                "out_up out dim != n_embd"
            );
            assert_eq!(
                out_up.in_features(),
                d_inner,
                "out_up in dim != eh_proj out dim (d_inner)"
            );
            assert!(
                dcfg.n_head >= 1 && dcfg.n_head_kv >= 1 && dcfg.n_head % dcfg.n_head_kv == 0,
                "student head counts malformed ({}/{})",
                dcfg.n_head,
                dcfg.n_head_kv
            );
            Some(DraftGeom {
                d_inner,
                n_head: dcfg.n_head as usize,
                n_head_kv: dcfg.n_head_kv as usize,
                out_up,
            })
        } else {
            None
        };
        // Log the name WITHOUT the blk.{n}. prefix (already printed) so the line reads
        // `source=nextn.shared_head_head` vs `source=output.weight` — the one-glance receipt
        // that the head choice went the right way on this artifact.
        let blk_prefix = format!("blk.{n}.");
        let head_src = head_name.strip_prefix(&blk_prefix).unwrap_or(&head_name);
        eprintln!(
            "[mtp-draft] external draft head: blk.{n}, source={}, head_vocab={}{}{}",
            head_src,
            head.out_features(),
            if d2t.is_some() {
                " (trimmed, d2t map)"
            } else {
                " (full)"
            },
            match &geom {
                Some(g) => format!(
                    " (student d_inner={} heads={}/{})",
                    g.d_inner, g.n_head, g.n_head_kv
                ),
                None => String::new(),
            }
        );

        let mut resident = ResidentPlan::unsharded(e, &src, &dcfg);
        Ok(MtpHead {
            enorm: load_t(e, &src, &p("nextn.enorm.weight"))?,
            hnorm: load_t(e, &src, &p("nextn.hnorm.weight"))?,
            eh_proj,
            attn_norm: load_t(e, &src, &p("attn_norm.weight"))?,
            post_attn_norm: load_opt(e, &src, &p("post_attention_norm.weight"))?
                .or(load_opt(e, &src, &p("ffn_norm.weight"))?)
                .expect("draft NextN block needs post_attention_norm or ffn_norm"),
            mixer: load_mixer_kind(e, &src, n, LayerKind::FullAttention, dcfg.mla.as_ref(),
                                   dcfg.attn_gate_separate())?,
            ffn: load_ffn(e, &src, &dcfg, n, None, &mut resident)?,
            shared_head_norm: head_norm,
            shared_head_head: Some(head),
            d2t,
            geom,
            step35,
        })
    }
}

/// gemma4 model-level auxiliaries.
pub struct GemmaAux {
    /// rope_freqs.weight [hd_global/2] freq factors — global layers' RoPE (R9).
    pub rope_freqs: Option<CudaSlice<f32>>,
    /// all-ones norm weight [512] (max head_dim) — the weightless rms_norms (R7 V-norm).
    pub ones: CudaSlice<f32>,
    /// tokenizer suppress_tokens uploaded once (None when the model ships none) — masked to
    /// -inf on every logits row before argmax/sampling (12B QAT ships two control ids).
    pub suppress_d: Option<(CudaSlice<i32>, usize)>,
    /// E4B per-layer-embedding model tensors (None on 26B/31B).
    pub e4b: Option<Gemma4E4bModel>,
}

/// step35 model-level auxiliaries. Deliberately NOT folded into `GemmaAux`: every gemma4 path
/// does `gemma4_aux.as_ref().unwrap()` and would then also fire on a step35 model.
pub struct Step35Aux {
    /// `rope_freqs.weight [n_rot_full/2]` llama3-style freq factors. Upstream applies them to
    /// FULL-attention layers ONLY (`rope_factors = is_swa ? nullptr : get_rope_factors(...)`,
    /// step35.cpp:246) — the SWA layers pass a null factor pointer. Step-3.7-Flash ships [64] F32.
    pub rope_freqs: Option<CudaSlice<f32>>,
}

pub struct HybridModel {
    pub cfg: ModelConfig,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<HybridLayer>,
    pub mtp: Option<MtpHead>, // NextN spec-decode head (None if nextn_predict_layers == 0)
    /// Lazily-uploaded DEVICE copy of the raw embed table (spec/graph hot loops gather rows
    /// on-device instead of host-dequant + htod). ~0.5GB; uploaded once on first use.
    pub embd_gpu: std::sync::OnceLock<cudarc::driver::CudaSlice<u8>>,
    pub gemma4_aux: Option<GemmaAux>,
    /// step35 (Step-3.7-Flash) model auxiliaries — `Some` iff `cfg.step35.is_some()`.
    pub step35_aux: Option<Step35Aux>,
    /// PRIME ACTIVATION SLABS (piecewise-graph foundation, 2026-07-26): the layer loop's
    /// seven trunk transients live in RESIDENT per-model buffers instead of per-call pool
    /// allocs — kills ~224 alloc/free API calls per prime AND freezes the Lt GEMM operand
    /// addresses (nvjet's alignment-variant kernels become run-to-run stable once their
    /// pointers stop moving). Sized on first prime to the largest T seen; Mutex = lazy init
    /// only (single GPU worker).
    pub prime_slabs: std::sync::Mutex<Option<crate::hybrid_forward::PrimeSlabs>>,
}

impl HybridModel {
    /// Load a hybrid (qwen35) model from GGUF. Thin byte-identical wrapper over `load_from_source`.
    pub fn load(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source(e, &GgufSource(g))
    }

    /// Plain-generation loader. `run-gen` never calls the optional draft head, so avoid loading
    /// its weights and expert bank while preserving the model config and all trunk semantics.
    pub fn load_without_mtp(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source_impl(e, &GgufSource(g), false)
    }

    /// Load a hybrid model from any `TensorSource` (GGUF or a safetensors HF checkpoint). The whole
    /// loop speaks ggml names; the source maps them (and, for safetensors, applies the SSM value
    /// transforms via the owned-buffer seam). The forward graph is untouched.
    pub fn load_from_source(
        e: &Engine,
        src: &dyn TensorSource,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source_impl(e, src, true)
    }

    /// Source-backed twin of `load_without_mtp`, used by the safetensors/repack `run-gen` path.
    pub fn load_from_source_without_mtp(
        e: &Engine,
        src: &dyn TensorSource,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source_impl(e, src, false)
    }

    fn load_from_source_impl(
        e: &Engine,
        src: &dyn TensorSource,
        load_mtp: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = src.config();
        assert!(cfg.arch.is_hybrid(), "not a hybrid arch");
        // SPEC-SERVING stream-k key, per model, set at LOAD so it governs the PRIME too
        // (2026-07-27; explicit MEMRA_MMQ_SK wins): the sk autotune's per-process kernel
        // coin flips knife-edge prime shapes between kernels run-to-run — the 12B depth
        // spec cell was BIMODAL (205 @ 0.756 / 260 @ 0.943 identical invocations; tiling
        // x6 = stable 263-269 @ 0.953, chat +3%; 31B neutral). The 26B is opposite: its
        // drafter accepts BETTER under sk's fold order (depth 328 @ 0.826 vs 293 @ 0.750).
        // Big dense (n_embd >= 3500) forces tiling under spec intent; MoE/small keep sk.
        // An earlier attempt set this in generate_spec_gemma — too late, the prime's
        // GEMMs had already autotuned.
        if std::env::var("MEMRA_DRAFT").is_ok() && std::env::var("MEMRA_MMQ_SK").is_err() {
            let force = if cfg.n_embd >= 3500 { 0i8 } else { -1i8 };
            crate::MMQ_SK_FORCE.store(force, std::sync::atomic::Ordering::Relaxed);
        }
        // FP8-KV door: OFF for every hybrid-path model (35B: fp8 format-gates its v3
        // dp4a lane, −2% measured 2026-07-12; gemma keys its KV formats independently
        // of this flag). The 9B dense loader is the only ON site.
        crate::KV_FP8_FORCE.store(0, std::sync::atomic::Ordering::Relaxed);

        // B0 FIX (hoisted): cfg.n_layer == block_count INCLUDES the MTP/NextN block(s)
        // (41 for the 35B-MoE); the trunk is n_layer - nextn. Computed before any tensor
        // upload because the M2 sharded loader (crate::pp::layer_engine) places tensors
        // by the trunk stage map.
        let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
        let embd = EmbedHost::from_source(src, "token_embd.weight");
        // M2 increment 2 (weight sharding): output_norm + lm head upload through the LAST
        // stage's engine — the stage that runs them (outside the pp door / MEMRA_PP_SHARD=0
        // this is the primary engine, byte-identical to the M1 loader).
        let e_head = crate::pp::layer_engine(e, n_trunk, n_trunk - 1)?;
        let output_norm = load_t(e_head, src, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent.
        let mut output = if src.has("output.weight") {
            load_t(e_head, src, "output.weight")?
        } else {
            load_t(e_head, src, "token_embd.weight")?
        };
        let mut resident = ResidentPlan::pp(e, src, &cfg, n_trunk)?;

        // SPILLING-PLAN §2: build the tiered-spill context ONCE, before loading any experts, but
        // only for a MoE model with the disk tier forced on (`MEMRA_SPILL_DISK`). It probes free VRAM
        // + host RAM at runtime (never hardcoded) and opens one shared GGUF mmap; all expert tensors
        // draw down its single pinned-RAM budget (hottest pinned, the rest mmap'd from disk). When
        // unset/dense this stays `None` and the load takes the byte-identical all-host path.
        // Disk spill is GGUF-only (needs the on-disk file mmap); src.gguf() is None for safetensors.
        let gguf: Option<&GgufFile> = src.gguf();
        // expert_count > 0: Arch::Gemma4 carries cfg.moe = Some on its DENSE variants too
        // (the 2026-07-14 discriminator-bug class) — a dense 31B/E4B under the spill env
        // would otherwise probe budgets + open an expert mmap it never consumes.
        let mut spill: Option<crate::spill::SpillCtx> =
            if cfg.moe.as_ref().is_some_and(|m| m.expert_count > 0)
                && crate::spill::disk_tier_enabled() && gguf.is_some() {
                let budget = crate::spill::MemBudget::probe(e)?;
                let ctx = crate::spill::SpillCtx::open(gguf.unwrap(), &budget)?;
                eprintln!("[spill] disk tier ON: free_vram={} MiB  pinnable_ram={} MiB (MemAvailable*frac)",
                          budget.free_vram >> 20, budget.free_pinnable_ram >> 20);
                Some(ctx)
            } else { None };

        // Running the MTP block as a trunk layer is wrong; iterate only the trunk layers
        // (n_trunk hoisted above). 9B (nextn=0): n_trunk = 32. 35B-MoE (nextn=1): 40.
        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk as u32 {
            let p = |s: &str| format!("blk.{il}.{s}");
            // M2 weight sharding: this layer's tensors upload through the OWNING stage's
            // engine (shadowed `e`) — the bring-up remote peer-read placement dies here.
            // Door shut / MEMRA_PP_SHARD=0: `layer_engine` returns the primary (no change).
            let e = crate::pp::layer_engine(e, n_trunk, il as usize)?;
            // attn_norm always; post_attention_norm is the pre-FFN norm in qwen35
            layers.push(HybridLayer {
                attn_norm: load_t(e, src, &p("attn_norm.weight"))?,
                post_attn_norm: load_opt(e, src, &p("post_attention_norm.weight"))?
                    .or(load_opt(e, src, &p("ffn_norm.weight"))?)
                    .expect("need post_attention_norm or ffn_norm"),
                mixer: {
                    // E4B KV-shared layers ship NO attn_k/attn_v — load the SHARE TARGET's
                    // k/v tensors for shape symmetry (forward skips k/v compute there and
                    // reads the target layer's cache; see Gemma4E4bLayer::kv_share).
                    let g4_shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
                    let kv_from = n_trunk as u32 - g4_shared;
                    if g4_shared > 0
                        && il >= kv_from
                        && !src.has(&format!("blk.{il}.attn_k.weight"))
                    {
                        let g4 = cfg.gemma4.as_ref().unwrap();
                        let swa = g4.swa_pattern.get(il as usize).copied().unwrap_or(true);
                        let tgt = kv_from - if swa { 2 } else { 1 };
                        let tp = |s: &str| format!("blk.{tgt}.{s}");
                        Mixer::Full(FullAttnLayer {
                            wq: load_t(e, src, &p("attn_q.weight"))?,
                            wk: load_t(e, src, &tp("attn_k.weight"))?,
                            wv: load_t(e, src, &tp("attn_v.weight"))?,
                            wo: load_t(e, src, &p("attn_output.weight"))?,
                            q_norm: load_t(e, src, &p("attn_q_norm.weight"))?,
                            k_norm: load_t(e, src, &tp("attn_k_norm.weight"))?,
                            attn_gate: None, // gemma4 has no separate head-wise gate
                        })
                    } else {
                        load_mixer_kind(e, src, il, cfg.layer_kind(il), cfg.mla.as_ref(),
                                        cfg.attn_gate_separate())?
                    }
                },
                ffn: load_ffn(e, src, &cfg, il,
                              spill.as_mut().map(|c| (gguf.unwrap(), c)), &mut resident)?,
                gemma4: if cfg.gemma4.is_some() {
                    let scalar = |n: &str| -> f32 {
                        let t = src.find(&p(n)).unwrap_or_else(|| panic!("missing {n}"));
                        memra_gguf::dequant::dequantize(t.ggml_type, &t.bytes, 1)[0]
                    };
                    let vecf = |n: &str| -> Vec<f32> {
                        let t = src.find(&p(n)).unwrap_or_else(|| panic!("missing {n}"));
                        memra_gguf::dequant::dequantize(
                            t.ggml_type,
                            &t.bytes,
                            t.ne.iter().product::<u64>() as usize,
                        )
                    };
                    let moe_bits = if src.find(&p("ffn_gate_inp.scale")).is_some() {
                        Some(crate::hybrid::Gemma4MoeBits {
                            post_ffw_norm_1: load_t(e, src, &p("post_ffw_norm_1.weight"))?,
                            pre_ffw_norm_2: load_t(e, src, &p("pre_ffw_norm_2.weight"))?,
                            post_ffw_norm_2: load_t(e, src, &p("post_ffw_norm_2.weight"))?,
                            shared_gate: load_t(e, src, &p("ffn_gate.weight"))?,
                            shared_up: load_t(e, src, &p("ffn_up.weight"))?,
                            shared_down: load_t(e, src, &p("ffn_down.weight"))?,
                            router_scale_pre: {
                                let inv = 1.0 / (cfg.n_embd as f32).sqrt();
                                let v: Vec<f32> =
                                    vecf("ffn_gate_inp.scale").iter().map(|x| x * inv).collect();
                                e.htod(&v)?
                            },
                            per_expert_scale: vecf("ffn_down_exps.scale"),
                            per_expert_scale_d: e.htod(&vecf("ffn_down_exps.scale"))?,
                        })
                    } else {
                        None
                    };
                    // E4B extras (tensor-presence: blk.N.inp_gate only exists on E4B)
                    let e4b = if src.has(&p("inp_gate.weight")) {
                        let g4 = cfg.gemma4.as_ref().unwrap();
                        let kv_from = n_trunk as u32 - g4.shared_kv_layers;
                        let kv_share = if g4.shared_kv_layers > 0 && il >= kv_from {
                            let swa = g4.swa_pattern.get(il as usize).copied().unwrap_or(true);
                            Some(kv_from - if swa { 2 } else { 1 })
                        } else {
                            None
                        };
                        Some(crate::hybrid::Gemma4E4bLayer {
                            inp_gate: load_t(e, src, &p("inp_gate.weight"))?,
                            proj: load_t(e, src, &p("proj.weight"))?,
                            post_norm: load_t(e, src, &p("post_norm.weight"))?,
                            kv_share,
                            qkv_cat: None,   // built at the mirror hook (wave 4b)
                        })
                    } else {
                        None
                    };
                    Some(Gemma4LayerBits {
                        ffn_norm: load_t(e, src, &p("ffn_norm.weight"))?,
                        post_ffw_norm: load_t(e, src, &p("post_ffw_norm.weight"))?,
                        moe_bits,
                        layer_scale: scalar("layer_output_scale.weight"),
                        e4b,
                    })
                } else {
                    None
                },
            });
        }

        // MTP/NextN head: load the block the trunk loop drops (il = n_trunk). It is a full
        // transformer block PLUS the nextn.{enorm,hnorm,eh_proj} glue. Only when nextn>0 and the
        // eh_proj tensor actually exists in the file (some MTP GGUFs ship the draft separately).
        let mtp = if load_mtp && cfg.nextn_predict_layers > 0 {
            let n = n_trunk as u32;
            let p = |s: &str| format!("blk.{n}.{s}");
            match src.has(&p("nextn.eh_proj.weight")) {
                true => Some(MtpHead {
                    enorm: load_t(e, src, &p("nextn.enorm.weight"))?,
                    hnorm: load_t(e, src, &p("nextn.hnorm.weight"))?,
                    eh_proj: load_t(e, src, &p("nextn.eh_proj.weight"))?,
                    attn_norm: load_t(e, src, &p("attn_norm.weight"))?,
                    post_attn_norm: load_opt(e, src, &p("post_attention_norm.weight"))?
                        .or(load_opt(e, src, &p("ffn_norm.weight"))?)
                        .expect("MTP block needs post_attention_norm or ffn_norm"),
                    mixer: load_mixer_kind(e, src, n, LayerKind::FullAttention, cfg.mla.as_ref(),
                                           cfg.attn_gate_separate())?,
                    ffn: load_ffn(e, src, &cfg, n,
                                  spill.as_mut().map(|c| (gguf.unwrap(), c)), &mut resident)?,
                    shared_head_norm: load_opt(e, src, &p("nextn.shared_head_norm.weight"))?,
                    // `nextn.shared_head_head` is the name the convert script and upstream both
                    // use (LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD -> "blk.%d.nextn.shared_head_head");
                    // `nextn.shared_head` is a name no shipped artifact carries, so this arm was
                    // silently always-None and every embedded-MTP model fell back to the trunk
                    // `self.output` in `mtp_head_forward_dev` op 12. Harmless for qwen35-family
                    // heads that genuinely tie to the trunk head; wrong for any artifact that
                    // ships its own — which the StepFun step35 drafter does (see `load_draft`).
                    // Keep the old name as a fallback so nothing that did match still does.
                    shared_head_head: load_opt(e, src, &p("nextn.shared_head_head.weight"))?
                        .or(load_opt(e, src, &p("nextn.shared_head.weight"))?),
                    d2t: None,
                    geom: None,
                    // EMBEDDED MTP block: same file, so its own arrays cover index `n`.
                    step35: cfg.step35.as_ref().map(|s| Step35MtpGeom::resolve(s, n)),
                }),
                false => None, // nextn>0 but no embedded eh_proj (external draft GGUF) -> no head
            }
        } else {
            None
        };

        // MEMRA_MTP_DRAFT=<path.gguf>: REPLACE the MTP head with one loaded from a standalone
        // draft GGUF (e.g. an FR-Spec trimmed-vocab draft). Verify-based spec decode stays exact
        // regardless of the draft — a different draft only changes WHICH tokens get proposed.
        let mtp = if load_mtp {
            match std::env::var("MEMRA_MTP_DRAFT") {
                Ok(path) if !path.is_empty() => {
                    eprintln!("[mtp-draft] loading external MTP draft: {path}");
                    let dg = GgufFile::open(&path)?;
                    Some(MtpHead::load_draft(e, &dg, &cfg)?)
                }
                _ => mtp,
            }
        } else {
            None
        };

        // MEMRA_FRSPEC_TRIM=<frspec.gguf>: SELF-TRIMMED draft head. Reads ONLY the d2t ranked-token
        // list from the given file and gathers those rows from the MAIN model's own output.weight
        // bytes (quantized rows are independent — a byte-level row gather, zero requant). The MTP
        // block, norms, and head quant all stay main-model, so there is no cross-file quality
        // mismatch (the external Q4_K draft file measured -15pts acceptance vs the native block).
        // Draft lm_head reads drop vocab/32768-fold; verify stays full-vocab -> exactness unchanged.
        // FULL_PREC (MTP-heal ceiling): the self-trim gathers rows into `from_quant_bytes` (Quant
        // only) and, more to the point, the full-precision ceiling wants the model's NATURAL full
        // head — trimming the draft vocab is a speed lever, not part of the exactness measurement.
        // Disable trim under the flag (documented resolution, §item 2).
        let trim_env = if load_mtp {
            std::env::var("MEMRA_FRSPEC_TRIM")
        } else {
            Err(std::env::VarError::NotPresent)
        };
        if crate::model::full_prec_enabled()
            && trim_env.as_deref().map(|p| !p.is_empty()).unwrap_or(false)
        {
            eprintln!(
                "[frspec-trim] DISABLED under MEMRA_FULL_PREC — using the natural full MTP head"
            );
        }
        let mtp = match (
            if crate::model::full_prec_enabled() {
                Err(std::env::VarError::NotPresent)
            } else {
                trim_env
            },
            mtp,
        ) {
            (Ok(path), Some(mut head)) if !path.is_empty() => {
                let tg = GgufFile::open(&path)?;
                let d2t_t = tg
                    .find("d2t")
                    .expect("MEMRA_FRSPEC_TRIM file has no d2t tensor");
                let d2t_bytes = tg.tensor_data(d2t_t);
                let d2t: Vec<u32> = match d2t_t.ggml_type {
                    GgmlType::I32 => d2t_bytes
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                        .collect(),
                    GgmlType::I64 => d2t_bytes
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
                        .collect(),
                    other => panic!("d2t must be I32/I64, got {other:?}"),
                };
                let v = src
                    .find("output.weight")
                    .or_else(|| src.find("token_embd.weight"))
                    .expect("model has no output.weight for FR-Spec trim");
                let out_f = v.ne[1] as usize;
                let row_bytes = v.bytes.len() / out_f;
                assert!(
                    d2t.iter().all(|&t| (t as usize) < out_f),
                    "d2t token id >= lm_head rows {out_f}"
                );
                let mut gathered = Vec::with_capacity(d2t.len() * row_bytes);
                for &t in &d2t {
                    let off = t as usize * row_bytes;
                    gathered.extend_from_slice(&v.bytes[off..off + row_bytes]);
                }
                let trimmed = GpuTensor::from_quant_bytes(
                    e,
                    &gathered,
                    v.ggml_type,
                    v.ne[0],
                    d2t.len() as u64,
                    /*nvfp4 macro-scale*/
                    match src.find("output.scale") {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    },
                )?;
                eprintln!(
                    "[frspec-trim] self-trimmed head: {} rows of main output.weight ({:?})",
                    d2t.len(),
                    v.ggml_type
                );
                head.shared_head_head = Some(trimmed);
                head.d2t = Some(d2t);
                Some(head)
            }
            (_, m) => m,
        };

        if let Some(ctx) = spill.as_ref() {
            eprintln!(
                "[spill] experts placed: {} pinned (Tier 1), {} mmap'd from disk (Tier 2, {} MiB)",
                ctx.n_pinned,
                ctx.n_mmap,
                ctx.mmap_bytes >> 20
            );
        }

        // FA v4 GQA CAPACITY GUARD (2026-08-06, lane/122b-bringup): fa_v4_smem sizes its
        // per-warp Q arrays q_ints[8][64]/q_d[8][8] for gqa<=8 — every model before the
        // 122B-A10B (32 Q heads / 2 KV heads = gqa 16) fit. At gqa>8 the (32,gqa,1) block's
        // warps 8..15 write q_ints[wy] PAST the array into the k_ints/k_d K tile, corrupting
        // scores -> all-NaN decode logits (receipts: research/122b-bringup-20260806/, arm
        // battery: v4/deep MISMATCH+NaN, v3/v2/smem/reg/scalar all MATCH). The hd512 lane
        // already carries its own capacity guard at dispatch ("gqa <= 16 = fa_v4_smem_512's
        // q-array capacity"); hd256 v4 never got one. Key FA_V4_MAX_DEFAULT=0 at load so
        // EVERY v4 dispatch site (eager, rows-verify, dc, rows_dc, windowed, seqs) flips to
        // the v3 lane together — decode/verify stay kernel-family-identical (the parity law).
        // Explicit MEMRA_FA_V4_MAX env still wins (diagnostic seam). The real v4 gqa16
        // extension is a kernel change gated on its own battery + perf receipts (fix brief
        // in research/122b-bringup-20260806/VERDICT.md).
        if cfg.n_head_kv > 0 && cfg.n_head / cfg.n_head_kv > 8 {
            crate::FA_V4_MAX_DEFAULT.store(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[fa] v4 decode family disabled: gqa {} > fa_v4_smem capacity 8 (v3 lane serves)",
                cfg.n_head / cfg.n_head_kv
            );
        }

        if cfg.gemma4.is_some() {
            // gemma4 fa-vec crossover default (measured sweep 2026-07-10; env overrides).
            crate::FA_VEC_MIN_DEFAULT.store(1, std::sync::atomic::Ordering::Relaxed);
            // windowed split per gemma variant (2026-07-12 sweeps): MoE 26B = 32 (grid-limited
            // t=1 under the raw-e4m3 sV ceiling), dense 31B = 64 (37.13 vs 36.87 at 1.7k, N=2).
            // DISCRIMINATOR FIX (2026-07-14): Arch::Gemma4 is in is_moe(), so cfg.moe is
            // Some (expert_count 0) on the DENSE 31B/E4B too — `cfg.moe.is_some()` keyed
            // every "per-variant" default to the 26B values and the dense arms of the
            // 2026-07-12 sweeps (SPW 64, SP512 32) never actually reached the 31B. Key on
            // expert_count instead.
            let real_moe = cfg.moe.as_ref().is_some_and(|m| m.expert_count > 0);
            crate::FA_SPW_DEFAULT.store(if real_moe { 32 } else { 64 },
                                        std::sync::atomic::Ordering::Relaxed);
            // hd512 global split per variant (26B=16 landed 2026-07-11; 31B=32 swept 2026-07-12).
            crate::FA_SP512_DEFAULT.store(if real_moe { 16 } else { 32 },
                                          std::sync::atomic::Ordering::Relaxed);
            // gemma4 router w8 RE-ARBITRATED 2026-08-01 (g26 decode dig): the 2026-07-31
            // knife-edge that stored false here was single-synthetic-prompt roulette — on 6
            // real prompts the w8 twin's gate outcome is IDENTICAL to the lone-warp form
            // (5 MATCH/5 MATCH; the one MISMATCH prompt fails both arms with the same
            // argmax pair, router-independent). w8 = +13% g26 decode (182->206 tok/s x3
            // interleaved, H100). Receipts: research/g26-decode-20260801/. gemma4 now rides
            // the global default (true); MEMRA_ROUTER_V2=0 is the rollback seam.
            // fused t=1 pair/triple mr1 per variant (2026-07-14 DRAM-duty arc: dense +1.1%
            // short / +0.6% depth on 31B; MoE 26B −1.2% — stays mr2).
            crate::FUSED_MR1_DEFAULT.store(!real_moe,
                                           std::sync::atomic::Ordering::Relaxed);
            // gemma4 rms_norm block 1024 (single-row 2816-col norms; battery-arbitrated per model).
            crate::RMS_BLOCK_DEFAULT.store(1024, std::sync::atomic::Ordering::Relaxed);
            // gemma4 fa split ladder (d1736 sweep; see fa_split_keys).
            crate::FA_SP_GEMMA.store(true, std::sync::atomic::Ordering::Relaxed);
            // depth fa: PARITY LAW (2026-07-10) — decode and verify share the rows_w/rows_dpl16
            // kernel symbols (decode t=1), so lane choice is freely tunable; v4 measured the
            // depth winner. Seams: MEMRA_FA_V4_MAX / MEMRA_FA_SMEM_TKV / MEMRA_GEMMA_ROWS_W.
        }
        // gemma4: the dc serving loop + spec draft gather read the device embed table every
        // step — upload it AT LOAD (OnceLock init) so first-use cost never lands in a timed span.
        let force_embd_gpu = cfg.gemma4.is_some();
        let gemma4_aux = if cfg.gemma4.is_some() {
            let rope_freqs = match src.find("rope_freqs.weight") {
                Some(t) => Some(e.htod(&memra_gguf::dequant::dequantize(
                    t.ggml_type,
                    &t.bytes,
                    t.ne.iter().product::<u64>() as usize,
                ))?),
                None => None,
            };
            // E4B per-layer-embedding model tensors (tensor-presence gated).
            let e4b = match src.find("per_layer_token_embd.weight") {
                Some(t) => {
                    let n_epl = cfg
                        .gemma4
                        .as_ref()
                        .map(|g| g.n_embd_per_layer as usize)
                        .unwrap_or(0);
                    let row = t.ne[0] as usize; // n_epl * n_layer
                    let row_bytes = t.bytes.len() / (t.ne[1] as usize);
                    eprintln!(
                        "[gemma4-e4b] per-layer-embed model detected (n_epl={n_epl}, row {row}) — \
                               first-light forward (eager decode + prime); dc/graph/spec unwired \
                               (HANDOVER-E4B.md)"
                    );
                    Some(crate::hybrid::Gemma4E4bModel {
                        tok_tbl_gpu: std::sync::OnceLock::new(),
                        tok_embd_bytes: t.bytes.to_vec(),
                        tok_embd_qt: match t.ggml_type {
                            memra_gguf::GgmlType::Q6_K => crate::QT_Q6_K,
                            memra_gguf::GgmlType::Q8_0 => crate::QT_Q8_0,
                            other => panic!("e4b per-layer tok embd: unhandled dtype {other:?}"),
                        },
                        tok_embd_row_bytes: row_bytes,
                        model_proj: load_t(e, src, "per_layer_model_proj.weight")?,
                        proj_norm: load_t(e, src, "per_layer_proj_norm.weight")?,
                        n_epl,
                    })
                }
                None => None,
            };
            let suppress_d = {
                let sup = &cfg.gemma4.as_ref().unwrap().suppress_tokens;
                if sup.is_empty() { None } else {
                    let ids: Vec<i32> = sup.iter().map(|&x| x as i32).collect();
                    eprintln!("[gemma4] suppress_tokens: {} ids masked at sampling", ids.len());
                    Some((e.htod_i32(&ids)?, ids.len()))
                }
            };
            Some(GemmaAux {
                rope_freqs,
                ones: e.htod(&[1.0f32; 512])?,
                suppress_d,
                e4b,
            })
        } else {
            None
        };
        // step35: rope_freqs.weight [n_rot_full/2] — FULL-attn layers only (SWA passes null).
        // Loaded by tensor presence, not required: the key is absent on a sibling without
        // llama3-style scaling, and `None` is the correct "no factors" signal for rope_neox2.
        let step35_aux = if cfg.step35.is_some() {
            let rope_freqs = match src.find("rope_freqs.weight") {
                Some(t) => Some(e.htod(&memra_gguf::dequant::dequantize(
                    t.ggml_type,
                    &t.bytes,
                    t.ne.iter().product::<u64>() as usize,
                ))?),
                None => None,
            };
            Some(Step35Aux { rope_freqs })
        } else {
            None
        };
        let mut layers = layers;
        // Q8_0 SPLIT-PLANE DECODE MIRRORS (2026-07-26, the H100 lane): Q8_0-trunk models
        // (Qwen3.5-9B class) stream their whole weight mass through the 34B-stride GGUF
        // layout — ncu on H100 held Max Bandwidth at 41-46% (Mem Busy 66-76%) from sector
        // overfetch. Mirrors route the m<=16 mmvq/batched decode family to the aligned-16B
        // `_rp` twins (bit-identical). VRAM cost == the mirrored trunk (~model size), so
        // DEFAULT ON only on the Hopper lane (80GB); MEMRA_Q8RP=1/0 overrides either way.
        {
            let q8rp_on = match std::env::var("MEMRA_Q8RP").as_deref() {
                Ok("0") => false,
                Ok(_) => true,
                Err(_) => cfg!(memra_hopper_mma),
            };
            // K-quant split-plane mirrors (q4_K/q6_K, 2026-08-01 H100 coalescing fix) ride
            // the same trunk walk under their own seam (MEMRA_KQRP, default = hopper lane).
            let kqrp_on = crate::Engine::kqrp_enabled();
            if q8rp_on || kqrp_on {
                // f16 prefill mirrors, PER-MODEL argmax-gate arbitration (round 45): on the
                // qwen Q8_0 dense class the f16-prefill-vs-int8-decode gap (maxdiff ~0.67)
                // flips the run-gen argmax gate on real prompts (board-2048: 485 vs 332,
                // deterministic x5) — gate-violating defaults don't ship. gemma (Q4_0) and
                // the MoE hybrids hold MATCH on the same prompt and keep their mirrors.
                // MEMRA_PP_F16=1 forces (diagnostic seam); =0 still kills everywhere.
                let f16_model_ok = cfg.gemma4.is_some() || cfg.moe.is_some()
                    || std::env::var("MEMRA_PP_F16").as_deref() == Ok("1");
                let mut nmir = 0usize;
                // M2 weight sharding: mirrors are the DECODE weights on these paths — each
                // builds through its layer's OWNING stage engine (`e_ref` param), so the
                // mirror lands on the device that dereferences it.
                let mut mir = |e_ref: &crate::Engine, w: &mut crate::model::GpuTensor| -> Result<(), Box<dyn std::error::Error>> {
                    let before = matches!(w, crate::model::GpuTensor::Quant { rp4: Some(_), .. });
                    if q8rp_on { e_ref.build_q8_rp4(w)?; }
                    if kqrp_on {
                        e_ref.build_q4k_rp4(w)?;
                        e_ref.build_q6k_rp4(w)?;
                    }
                    // Q6_K mirrors are model-CLASS-agnostic (round 47): no MMQ arm exists for
                    // Q6_K — the fallback dequant-GEMM is ~10x the f16 lane (q27's prefill
                    // wall). The qwen-dense argmax-flip evidence (round 45) was the Q8_0
                    // mirror specifically; Q6_K admission is arbitrated by its own gate runs.
                    let q6k = matches!(w, crate::model::GpuTensor::Quant { qtype, .. }
                                       if *qtype == crate::QT_Q6_K);
                    if q8rp_on && crate::f16_ffi::pp_f16_enabled() && (f16_model_ok || q6k) {
                        e_ref.build_q8_f16(w)?;
                    }
                    if !before && matches!(w, crate::model::GpuTensor::Quant { rp4: Some(_), .. }) {
                        nmir += 1;
                    }
                    Ok(())
                };
                for (il, layer) in layers.iter_mut().enumerate() {
                    let el = crate::pp::layer_engine(e, n_trunk, il)?;
                    match &mut layer.mixer {
                        Mixer::Full(fa) => {
                            for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] { mir(el, w)?; }
                        }
                        Mixer::Linear(la) => {
                            for w in [&mut la.wqkv, &mut la.wqkv_gate, &mut la.ssm_beta,
                                      &mut la.ssm_alpha, &mut la.ssm_out] { mir(el, w)?; }
                        }
                        // MLA: no decode mirrors in increment 2 (its kernels arrive in inc 4;
                        // mirror admission is arbitrated there with measurements).
                        Mixer::Mla(_) => {}
                    }
                    if let Ffn::Dense { ffn_gate, ffn_up, ffn_down } = &mut layer.ffn {
                        for w in [ffn_gate, ffn_up, ffn_down] { mir(el, w)?; }
                    }
                }
                mir(e_head, &mut output)?;
                if nmir > 0 {
                    eprintln!("[q8rp] split-plane decode mirrors built: {nmir} tensors");
                }
                // Q4_K f16 prefill mirrors (round 49): Q4_K joins the q6k carve-out —
                // model-class-agnostic admission, arbitrated by per-model argmax gates
                // (the round-45 flip evidence was the Q8_0 mirror on qwen-dense; the q27
                // Q4_K bulk rides mul_mat_q_q45k int8-MMA, which the Lt f16 lane beats at
                // large m — campaign-A precedent). SECOND pass over the trunk so the shared
                // MEMRA_PP_F16_BUDGET_MB keeps FULL Q6_K coverage as its floor: Q6_K mirrors
                // replace a ~10x dequant-GEMM (no MMQ arm exists), Q4_K mirrors upgrade a
                // working int8-MMA arm — a joint walk would evict late-layer Q6_K mirrors
                // for the weaker lever. Layer-order prefix within the Q4_K class.
                // Round 49b: Q5_K (q27's 48 ssm_out — the last mul_mat_q_q45k class) rides
                // a THIRD pass strictly after all Q4_K, so the default-budget composition
                // (and its banked gates) stays byte-identical: the 32GB default is exhausted
                // by the Q4_K pass; Q5_K mirrors only light up under a raised
                // MEMRA_PP_F16_BUDGET_MB (machine-specific config).
                if q8rp_on && crate::f16_ffi::pp_f16_enabled() {
                    for (want, tag) in [(crate::QT_Q4_K, "q4kf16"), (crate::QT_Q5_K, "q5kf16")] {
                        let (mut n4, mut b4) = (0usize, 0usize);
                        let mut mirk = |e_ref: &crate::Engine, w: &mut crate::model::GpuTensor|
                                       -> Result<(), Box<dyn std::error::Error>> {
                            if matches!(w, crate::model::GpuTensor::Quant { qtype, f16: None, .. }
                                        if *qtype == want) {
                                e_ref.build_q8_f16(w)?;
                                if let crate::model::GpuTensor::Quant { f16: Some(m), .. } = w {
                                    n4 += 1;
                                    b4 += m.len();
                                }
                            }
                            Ok(())
                        };
                        for (il, layer) in layers.iter_mut().enumerate() {
                            let el = crate::pp::layer_engine(e, n_trunk, il)?;
                            match &mut layer.mixer {
                                Mixer::Full(fa) => {
                                    for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] { mirk(el, w)?; }
                                }
                                Mixer::Linear(la) => {
                                    for w in [&mut la.wqkv, &mut la.wqkv_gate, &mut la.ssm_beta,
                                              &mut la.ssm_alpha, &mut la.ssm_out] { mirk(el, w)?; }
                                }
                                Mixer::Mla(_) => {} // no mirrors in increment 2 (see above)
                            }
                            if let Ffn::Dense { ffn_gate, ffn_up, ffn_down } = &mut layer.ffn {
                                for w in [ffn_gate, ffn_up, ffn_down] { mirk(el, w)?; }
                            }
                        }
                        mirk(e_head, &mut output)?;
                        if n4 > 0 {
                            eprintln!("[{tag}] prefill fp16 mirrors built: {n4} tensors \
                                       ({} MB)", b4 >> 20);
                        }
                    }
                }
            }
        }
        // Q4_0 SPLIT-PLANE DECODE MIRRORS (2026-07-10, MEMRA_Q4RP seam): gemma-4 MoE-class trunk
        // (26B — attn wq/wk/wv/wo + the parallel shared FFN triple). The 18B GGUF block stride
        // costs ~25-35% decode bandwidth in sector overfetch (rp_q4_probe: m=1 1.34x, m=3 1.17x,
        // bitwise); the mirror (~0.7GB for the 26B) fixes the m<=8 mmvq/batched/fused family.
        // Dense 31B is NOT mirrored (its 15GB trunk mirror does not fit 24GB — the full layout
        // swap is the follow-up arc); raw bytes stay for prefill/gemm/Stage-A either way.
        if cfg.gemma4.is_some() && crate::Engine::q4rp_enabled() {
            let mut nmir = 0usize;
            for (il, layer) in layers.iter_mut().enumerate() {
                // M2 weight sharding: mirrors/concats build through the owning stage engine.
                let e = crate::pp::layer_engine(e, n_trunk, il)?;
                // 26B MoE-class trunk (moe_bits) OR the E4B dense trunk (e4b bits). E4B mirror
                // arithmetic: attn ~7.5MB/layer (shared layers skip wk/wv via build's no-op on
                // duplicate mirrors is NOT automatic — they alias the target's tensors as
                // separate GpuTensors, so their mirrors double ~1.5MB/shared-layer; acceptable)
                // + dense ffn 3 x 2560x10240 Q4_0 ~44MB + inp_gate/proj ~0.75MB => ~2.2GB for
                // the 5.2GB model; 24GB card holds model+mirror+KV with >14GB headroom.
                // Dense 31B stays unmirrored (15GB mirror does not fit) — its arm is the
                // layout-swap follow-up.
                let is_moe26 = layer.gemma4.as_ref().is_some_and(|g| g.moe_bits.is_some());
                let is_e4b = layer.gemma4.as_ref().is_some_and(|g| g.e4b.is_some());
                if !(is_moe26 || is_e4b) {
                    continue;
                }
                if let Mixer::Full(fa) = &mut layer.mixer {
                    for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                        e.build_q4_rp4(w)?;
                        nmir += 1;
                    }
                }
                if is_e4b {
                    // wave-4b: own-KV layers get the wq|wk|wv OUT-concat (one matvec at t=1).
                    let own_kv = layer.gemma4.as_ref().unwrap().e4b.as_ref()
                        .is_some_and(|e4| e4.kv_share.is_none());
                    if own_kv {
                        if let Mixer::Full(fa) = &layer.mixer {
                            if let Some(mut cat) = e.build_q4_out_concat3(&fa.wq, &fa.wk, &fa.wv)? {
                                e.build_q4_rp4(&mut cat)?; nmir += 1;
                                layer.gemma4.as_mut().unwrap().e4b.as_mut().unwrap()
                                    .qkv_cat = Some(cat);
                            }
                        }
                    }
                    if let Ffn::Dense { ffn_gate, ffn_up, ffn_down } = &mut layer.ffn {
                        for w in [ffn_gate, ffn_up, ffn_down] {
                            e.build_q4_rp4(w)?;
                            nmir += 1;
                        }
                    }
                    let e4 = layer.gemma4.as_mut().unwrap().e4b.as_mut().unwrap();
                    for w in [&mut e4.inp_gate, &mut e4.proj] {
                        e.build_q4_rp4(w)?;
                        nmir += 1;
                    }
                }
                if let Some(mb) = layer.gemma4.as_mut().unwrap().moe_bits.as_mut() {
                    for w in [&mut mb.shared_gate, &mut mb.shared_up, &mut mb.shared_down] {
                        e.build_q4_rp4(w)?;
                        nmir += 1;
                    }
                }
            }
            if nmir > 0 {
                eprintln!("[q4rp] split-plane decode mirrors built: {nmir} trunk tensors");
            }
            // DENSE gemma (31B / E4B trunks): the trunk is too big to MIRROR on 24GB, so the
            // split layout replaces the GGUF bytes IN PLACE (zero steady-state VRAM; the 31B
            // profile put 76% of decode on the non-rp q4_0 matvecs). Every consumer routes
            // off the tensor's rp flag: mmvq/batched `_rp` twins + qmatvec_gemm_q4_0_rp
            // prefill. The Stage-A f32 oracle reads GGUF layout, so the swap is gated on the
            // fast path being active (MEMRA_FAST=0 keeps GGUF bytes end to end — exact oracle).
            let fast_on = std::env::var("MEMRA_FAST").as_deref() != Ok("0");
            if fast_on {
                let mut nswap = 0usize;
                let mut nf16 = 0usize;
                // f16 prefill mirrors (campaign A, 2026-07-31): built from the GGUF Q4_0
                // bytes BEFORE the in-place rp swap destroys that layout. Same Lt lane and
                // budget env as the qwen Q8_0 mirrors (MEMRA_PP_F16 / MEMRA_PP_F16_BUDGET_MB;
                // Hopper default ON, sm_120a default OFF — the 24GB card can't carry them).
                // Per-model (battery-keyed, 2026-07-31, REAL-prompt gates — the fox-repeat
                // family is layout-lottery degenerate and was retired from campaign gates):
                // 12B pp1736 8.3k -> 17.1k MATCH; 31B pp1736 4.8k -> 7.6k MATCH but ONLY
                // with the full-trunk mirror (420 tensors ~53GB — set
                // MEMRA_PP_F16_BUDGET_MB=57344 on 80GB boxes; the default 32GB partial
                // mirror measured FLAT there). MEMRA_Q4F16=1|0 forces either way.
                let q4f16_model_ok = matches!(cfg.n_embd, 3840 | 5376); // 12B | 31B geometry
                let f16_on = match std::env::var("MEMRA_Q4F16").as_deref() {
                    Ok("1") => crate::f16_ffi::pp_f16_enabled(),
                    Ok("0") => false,
                    _ => crate::f16_ffi::pp_f16_enabled() && q4f16_model_ok,
                };
                for (il, layer) in layers.iter_mut().enumerate() {
                    // M2 weight sharding: swap/mirror through the owning stage engine.
                    let e = crate::pp::layer_engine(e, n_trunk, il)?;
                    let dense_gemma = layer.gemma4.as_ref().is_some_and(|g| g.moe_bits.is_none());
                    if !dense_gemma {
                        continue;
                    }
                    if let Mixer::Full(fa) = &mut layer.mixer {
                        for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                            if f16_on {
                                e.build_q8_f16(w)?;
                                if matches!(w, crate::model::GpuTensor::Quant { f16: Some(_), .. }) {
                                    nf16 += 1;
                                }
                            }
                            if e.build_q4_rp_swap(w)? {
                                nswap += 1;
                            }
                        }
                    }
                    if let Ffn::Dense {
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                    } = &mut layer.ffn
                    {
                        for w in [ffn_gate, ffn_up, ffn_down] {
                            if f16_on {
                                e.build_q8_f16(w)?;
                                if matches!(w, crate::model::GpuTensor::Quant { f16: Some(_), .. }) {
                                    nf16 += 1;
                                }
                            }
                            if e.build_q4_rp_swap(w)? {
                                nswap += 1;
                            }
                        }
                    }
                }
                if nswap > 0 {
                    eprintln!("[q4rp] split-plane IN-PLACE swap: {nswap} dense trunk tensors");
                }
                if nf16 > 0 {
                    eprintln!("[q4f16] prefill fp16 mirrors built: {nf16} dense trunk tensors");
                }
            }
        }
        let model = HybridModel {
            cfg,
            embd,
            output_norm,
            output,
            layers,
            mtp,
            embd_gpu: std::sync::OnceLock::new(),
            gemma4_aux,
            step35_aux,
            prime_slabs: std::sync::Mutex::new(None),
        };
        e.configure_moe_cache_layout(model.moe_cache_block_sizes());
        if force_embd_gpu {
            let _ = model
                .embd_gpu
                .get_or_init(|| e.upload_u8(&model.embd.raw).expect("embed table upload"));
        }
        // M2 LOAD BARRIER (pp door open at load): uploads + mirror builds above ran on
        // the loading engines' worker streams; the first decode consumer runs on OTHER
        // streams with no event between them. Synchronize every stage context once so
        // no consumer can ever read a half-built tensor (the 2026-08-02 split5 ref=0.0
        // head-mirror find). No-op with the door shut.
        crate::pp::sync_stages_after_load(e, n_trunk)?;
        Ok(model)
    }

    /// Force the device embed table resident, FALLIBLY (F5 right-size ladder,
    /// 2026-08-05). The lazy `embd_gpu.get_or_init(.. expect ..)` sites panic the
    /// GPU worker on OOM; on a VRAM-tight rig a right-sized spec session that
    /// "fits" can leave too little for this ~hundreds-of-MB upload and die on its
    /// first prefill (observed: research/specpool-20260804/server-ladder-miss.log).
    /// The server calls this after each ladder landing so the biggest lazy
    /// transient surfaces as a catchable Err (shrink further / fall back) instead
    /// of a panic. No-op when the host-gather door (MEMRA_EMBED_DEV=0) is open or
    /// the table is already resident.
    pub fn ensure_embed_resident(&self, e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
        if std::env::var("MEMRA_EMBED_DEV").as_deref() == Ok("0") {
            return Ok(());
        }
        if self.embd_gpu.get().is_none() {
            let buf = e.upload_u8(&self.embd.raw)?;
            let _ = self.embd_gpu.set(buf); // racing set = already resident; fine
        }
        Ok(())
    }

    pub fn embed(
        &self,
        e: &Engine,
        tokens: &[u32],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        // DEVICE embed gather (round 30; the gemma4 machinery adopted for every model):
        // resident quantized table + gather kernel — replaces the CPU row gather + 31MB
        // pageable HtoD (2.2ms at T=2048, the lane's largest host stall). Same d*q
        // dequant math as the CPU gather; the greedy-stream A/B arbitrates.
        // MEMRA_EMBED_DEV=0 reverts.
        if std::env::var("MEMRA_EMBED_DEV").as_deref() != Ok("0") {
            let tbl = self
                .embd_gpu
                .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload"));
            let tok_d = e.htod_u32_v(tokens)?;
            let (qt, rb) = self.embd.qt_and_row_bytes(n_embd);
            return e.embed_gather_device_td(tbl, &tok_d, tokens.len(), n_embd, qt, rb);
        }
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}

#[cfg(test)]
mod residency_tests {
    use super::residency_bytes_by_device;

    #[test]
    fn pp_residency_counts_only_each_devices_expert_slice() {
        let tensors = [
            ("blk.0.ffn_gate_exps.weight", 10usize),
            ("blk.0.ffn_up_exps.weight", 20),
            ("blk.1.ffn_down_exps.weight", 30),
            ("blk.2.ffn_gate_exps.weight", 40),
            ("blk.3.ffn_up_exps.weight", 50),
            ("blk.0.attn_q.weight", 7),
            ("output.weight", 11),
        ];
        let bytes = residency_bytes_by_device(tensors, &[0, 0, 1, 1], 0);
        assert_eq!(bytes.experts.get(&0), Some(&60));
        assert_eq!(bytes.experts.get(&1), Some(&90));
        assert_eq!(bytes.rest, 18);
        assert!(bytes.saw_experts);
    }

    #[test]
    fn pp_residency_combines_stages_that_share_one_device() {
        let tensors = [
            ("blk.0.ffn_gate_exps.weight", 10usize),
            ("blk.1.ffn_gate_exps.weight", 20),
            ("blk.2.ffn_gate_exps.weight", 30),
            ("blk.3.ffn_gate_exps.weight", 40),
        ];
        let bytes = residency_bytes_by_device(tensors, &[0, 0, 0, 0], 0);
        assert_eq!(bytes.experts.get(&0), Some(&100));
        assert_eq!(bytes.experts.len(), 1);
    }
}

#[cfg(test)]
mod draft_head_tests {
    use super::draft_head_tensor;

    /// Names present in the real Step-3.7-Flash MTP drafter (Step3.7-flash-mtp-Q8_0.gguf), as
    /// enumerated by the on-disk byte probe in
    /// research/step37-p2-20260806/raw/draft-head-tensor-hashes-20260807.txt.
    /// Both candidate heads exist in that file with IDENTICAL [4096, 128896] Q8_0 shape, so no
    /// shape or dtype check can distinguish them — only the sha256 of the payload could, and it
    /// showed them to be different matrices (blk.45 head c90b907b… vs output.weight 3eec5831…).
    const STEP37_DRAFTER: &[&str] = &[
        "output.weight",
        "output_norm.weight",
        "token_embd.weight",
        "blk.45.nextn.shared_head_norm.weight",
        "blk.45.nextn.shared_head_head.weight",
        "blk.46.nextn.shared_head_head.weight",
        "blk.47.nextn.shared_head_head.weight",
    ];

    fn present(names: &'static [&'static str]) -> impl Fn(&str) -> bool {
        move |t: &str| names.contains(&t)
    }

    /// THE REGRESSION. Reading `output.weight` off this drafter cost acceptance 0/248 across
    /// K=1..8 with self-consistency PASS at every K — correct output, dead speculation, no gate
    /// red (raw/mtp-draft-20260806T212902Z.log). The drafter's top-level output stack is a
    /// re-quantized COPY OF THE TRUNK'S (its output_norm is byte-identical to the trunk's,
    /// d7526f44…), so it is the standalone-decode head, not the MTP head. Preferring
    /// blk.45.nextn.shared_head_head took K=1 to 14/18 = 77.8%
    /// (raw/mtp-draft-PASS-20260806T215132Z.log).
    #[test]
    fn step37_drafter_prefers_the_blocks_own_nextn_head_over_file_level_output() {
        assert_eq!(
            draft_head_tensor(present(STEP37_DRAFTER), 45),
            "blk.45.nextn.shared_head_head.weight"
        );
    }

    /// Each NextN block owns a DIFFERENT head (c90b907b / a22d2957 / 4b21e137 — a shared head
    /// would have collided), so the name must be built from the block index, never hardcoded.
    /// This is what multi-block chaining (45->46->47) will index when it lands.
    #[test]
    fn each_nextn_block_selects_its_own_head() {
        for n in 45..=47u32 {
            assert_eq!(
                draft_head_tensor(present(STEP37_DRAFTER), n),
                format!("blk.{n}.nextn.shared_head_head.weight")
            );
        }
    }

    /// FR-Spec / tied-head drafts publish the (possibly vocab-trimmed) head as the file-level
    /// `output.weight` and ship no nextn head. They must keep working — hence preference, not
    /// replacement.
    #[test]
    fn draft_without_a_nextn_head_falls_back_to_file_level_output() {
        let fr_spec: &[&str] = &["output.weight", "output_norm.weight", "d2t.weight"];
        assert_eq!(draft_head_tensor(present(fr_spec), 45), "output.weight");
    }

    /// The legacy `nextn.shared_head` probe sits between the two: no shipped artifact and no
    /// upstream mapping uses it (upstream is LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD ->
    /// "blk.%d.nextn.shared_head_head"), but anything that ever matched it still must, and it
    /// must never win over the real name.
    #[test]
    fn legacy_shared_head_is_probed_but_loses_to_shared_head_head() {
        let legacy_only: &[&str] = &["output.weight", "blk.45.nextn.shared_head.weight"];
        assert_eq!(
            draft_head_tensor(present(legacy_only), 45),
            "blk.45.nextn.shared_head.weight"
        );

        let both: &[&str] = &[
            "output.weight",
            "blk.45.nextn.shared_head.weight",
            "blk.45.nextn.shared_head_head.weight",
        ];
        assert_eq!(
            draft_head_tensor(present(both), 45),
            "blk.45.nextn.shared_head_head.weight"
        );
    }

    /// A drafter whose nextn head belongs to a DIFFERENT block must not be borrowed: asking for
    /// block 45 in a file that only carries 46/47 falls back rather than silently mismatching
    /// the geometry the trunk verified against.
    #[test]
    fn a_different_blocks_nextn_head_is_never_borrowed() {
        let wrong_block: &[&str] = &[
            "output.weight",
            "blk.46.nextn.shared_head_head.weight",
            "blk.47.nextn.shared_head_head.weight",
        ];
        assert_eq!(draft_head_tensor(present(wrong_block), 45), "output.weight");
    }
}
