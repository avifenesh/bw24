//! gemma4 MTP spec-decode: the "gemma4-assistant" drafter (4-layer, Q-only attention over the
//! MAIN model's KV cache — no draft KV, no trims) + the greedy draft/verify loop.
//!
//! Wiring verified from llama gemma4-assistant.cpp + llama-model.cpp:2162 (HANDOVER "GEMMA4 MTP
//! DRAFTER — VERIFIED WIRING"): per draft token, x = MAIN tok_embd(token) * sqrt(2816);
//! xh = concat(x, h[2816]) -> pre_proj [5632->1024]; 4 gemma-style blocks whose attention
//! projects Q ONLY and attends the main cache (SWA layers 0..2 -> main layer n-2 = 28 windowed;
//! global layer 3 -> main layer n-1 = 29 full); dense GELU_PAR ffn; final output_norm ->
//! TIED 1024-dim head (no softcap); h_next = post_proj [1024->2816].

use crate::cache::Cache;
use crate::hybrid::HybridModel;
use crate::model::GpuTensor;
use crate::Engine;
use bw24_gguf::source::{GgufSource, TensorSource};
use bw24_gguf::GgufFile;
use cudarc::driver::CudaSlice;

pub struct GemmaDraftLayer {
    pub attn_norm: GpuTensor,
    pub wq: GpuTensor,
    pub wo: GpuTensor,
    pub q_norm: GpuTensor,
    pub post_attn_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn_gate: GpuTensor,
    pub ffn_up: GpuTensor,
    pub ffn_down: GpuTensor,
    pub ffn_post_norm: GpuTensor,
    pub out_scale: f32,
    pub swa: bool,
    pub hd: usize,
    pub nh: usize,
}

pub struct GemmaDraft {
    pub layers: Vec<GemmaDraftLayer>,
    pub pre_proj: GpuTensor,  // [5632 -> 1024]
    pub post_proj: GpuTensor, // [1024 -> 2816]
    pub output_norm: GpuTensor,
    pub head: GpuTensor, // tied drafter token_embd [1024, n_vocab] (or FR-trimmed rows)
    /// FR-Spec trim map: draft-row index -> target token id (None = full head, identity).
    pub d2t: Option<Vec<u32>>,
    /// Device copy of `d2t` — the async round translates each drafted trim-idx in place
    /// (u32_map_k) before it seeds the next draft step or meets the verify argmax.
    pub d2t_dev: Option<CudaSlice<u32>>,
    /// Adaptive trim (coverage escapes are the entire trim cost — oracle-proven +2% on the
    /// cell the static trim lost by 17%, jsonl 2026-07-19): spare head slots learned at
    /// serve time from the prompt's own ids and verify-correction tokens.
    pub trim_adapt: Option<TrimAdapt>,
    pub rope_freqs: CudaSlice<f32>,
    pub ones: CudaSlice<f32>, // weightless-norm weight (max hd 512)
    pub n_embd: usize,        // 1024
    pub n_backbone: usize,    // 2816
    pub rope_base_global: f32,
    pub rope_base_swa: f32,
    pub sliding_window: usize,
}

/// Serve-time adaptive trim (BW24_GEMMA_TRIM_ADAPT=<spare slots>): the static FR trim's whole
/// loss is coverage escapes — tokens the base emits that the trim can't propose (guaranteed
/// rejections; the oracle control that injected the exact escapees flipped a -17% cell to +2%
/// at identical acceptance, jsonl 2026-07-19). Every escape self-identifies at serve time: it
/// arrives as a verify CORRECTION token (and its cousins ride in with the prompt), so the head
/// keeps `n_spare` extra rows and learns them — prompt ids up front, corrections as they land.
/// First miss pays one rejected round; every recurrence after is proposable. Rows are written
/// into the existing device buffers (no realloc — captured graphs keep their baked addresses).
pub struct TrimAdapt {
    /// full-vocab head rows (host copy) — the gather source for learned rows.
    src_rows: Vec<u8>,
    row_bytes: usize,
    n_vocab: usize,
    /// trim-set membership by token id (ranked + learned).
    present: Vec<bool>,
    /// spare slots live at [spare_base, spare_base + n_spare) in the gathered head.
    spare_base: usize,
    n_spare: usize,
    used: usize,
    logged_full: bool,
}

impl TrimAdapt {
    /// Add `tok`'s head row to the trim set if absent and a spare slot is free.
    fn maybe_add(&mut self, e: &Engine, tok: u32, head: &mut GpuTensor,
                 d2t: &mut [u32], d2t_dev: &mut CudaSlice<u32>)
                 -> Result<bool, Box<dyn std::error::Error>> {
        let t = tok as usize;
        if t >= self.n_vocab || self.present[t] { return Ok(false); }
        if self.used == self.n_spare {
            if !self.logged_full {
                self.logged_full = true;
                eprintln!("[trim-adapt] spare slots exhausted ({}) — later escapes stay unproposable",
                          self.n_spare);
            }
            return Ok(false);
        }
        let slot = self.spare_base + self.used;
        self.used += 1;
        self.present[t] = true;
        if let GpuTensor::Quant { bytes, .. } = head {
            e.htod_u8_into(bytes, slot * self.row_bytes,
                           &self.src_rows[t * self.row_bytes..(t + 1) * self.row_bytes])?;
        }
        d2t[slot] = tok;
        e.u32_set_k(d2t_dev, tok, slot)?;
        Ok(true)
    }
}

/// Union `toks` into the adaptive trim set (no-op when the draft has no adaptive state).
/// Split-borrow helper: the fields move together or not at all.
fn trim_adapt_learn(e: &Engine, d: &mut GemmaDraft, toks: &[u32])
                    -> Result<(), Box<dyn std::error::Error>> {
    let GemmaDraft { trim_adapt, head, d2t, d2t_dev, .. } = d;
    let (Some(ta), Some(d2t), Some(d2t_dev)) =
        (trim_adapt.as_mut(), d2t.as_mut(), d2t_dev.as_mut()) else { return Ok(()) };
    for &tok in toks {
        ta.maybe_add(e, tok, head, d2t, d2t_dev)?;
    }
    Ok(())
}

impl GemmaDraft {
    /// Adaptive-trim stats: (slots used, slot budget). None when adaptation is off.
    pub fn trim_adapt_stats(&self) -> Option<(usize, usize)> {
        self.trim_adapt.as_ref().map(|ta| (ta.used, ta.n_spare))
    }

    /// Persist the learned trim rows: append ids not yet in the sidecar to
    /// `<ranks>.learned` (the load path pre-fills spare slots from it, so a distribution's
    /// escapes pay their first-miss round ONCE across the serve lifetime, not per request).
    pub fn trim_adapt_save(&self) -> std::io::Result<usize> {
        let (Some(ta), Some(d2t), Some(path)) =
            (self.trim_adapt.as_ref(), self.d2t.as_ref(), self.trim_learned_path()) else {
            return Ok(0);
        };
        let prior: std::collections::HashSet<u32> = std::fs::read_to_string(&path)
            .map(|t| t.lines().filter_map(|l| l.trim().parse().ok()).collect())
            .unwrap_or_default();
        let fresh: Vec<u32> = d2t[ta.spare_base..ta.spare_base + ta.used].iter()
            .copied().filter(|id| !prior.contains(id)).collect();
        if !fresh.is_empty() {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
            for id in &fresh {
                writeln!(f, "{id}")?;
            }
        }
        Ok(fresh.len())
    }

    fn trim_learned_path(&self) -> Option<String> {
        std::env::var("BW24_GEMMA_DRAFT_RANKS").ok().map(|p| format!("{p}.learned"))
    }
}

fn load_t(e: &Engine, src: &dyn TensorSource, name: &str)
          -> Result<GpuTensor, Box<dyn std::error::Error>> {
    GpuTensor::load_from_source(e, src, name)
}

impl GemmaDraft {
    pub fn load(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        // two published spellings of the same arch: the 26B/31B drafters ship
        // "gemma4-assistant", the E4B assistant ships "gemma4_assistant" — the metadata
        // key prefix follows the arch string verbatim.
        let arch = match g.arch() {
            Some(a @ ("gemma4-assistant" | "gemma4_assistant")) => a.to_string(),
            other => panic!("not a gemma4-assistant drafter (arch {other:?})"),
        };
        let src = GgufSource(g);
        let meta_u = |k: &str| -> u32 {
            g.metadata.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).unwrap_or(0) as u32
        };
        let meta_f = |k: &str, d: f32| -> f32 {
            match g.metadata.get(&format!("{arch}.{k}")) {
                Some(bw24_gguf::MetaValue::F32(v)) => *v,
                Some(bw24_gguf::MetaValue::F64(v)) => *v as f32,
                _ => d,
            }
        };
        let n_layer = meta_u("block_count") as usize;
        let n_embd = meta_u("embedding_length") as usize;
        // 26B/31B carry the target width as embedding_length_out; the E4B assistant as
        // n_embd_backbone.
        let n_backbone = match meta_u("embedding_length_out") as usize {
            0 => meta_u("n_embd_backbone") as usize,
            v => v,
        };
        let hd_g = meta_u("attention.key_length") as usize;
        let hd_s = meta_u("attention.key_length_swa") as usize;
        let swa_pat: Vec<bool> = match g.metadata.get(&format!("{arch}.attention.sliding_window_pattern")) {
            Some(bw24_gguf::MetaValue::Array(a)) =>
                a.iter().filter_map(|v| v.as_u64().map(|x| x != 0)).collect(),
            _ => return Err("drafter missing sliding_window_pattern".into()),
        };

        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let p = |n: &str| format!("blk.{il}.{n}");
            let swa = swa_pat[il];
            let out_scale = {
                let t = src
                    .find(&p("layer_output_scale.weight"))
                    .ok_or("missing layer_output_scale")?;
                bw24_gguf::dequant::dequantize(t.ggml_type, &t.bytes, 1)[0]
            };
            let hd = if swa { hd_s } else { hd_g };
            let wq = load_t(e, &src, &p("attn_q.weight"))?;
            // heads per layer from the projection shape (the E4B assistant keeps 4 heads on
            // BOTH classes — hd differs — while 26B/31B are uniform; the shape is the truth).
            let nh = wq.out_features() / hd;
            layers.push(GemmaDraftLayer {
                attn_norm: load_t(e, &src, &p("attn_norm.weight"))?,
                wq,
                wo: load_t(e, &src, &p("attn_output.weight"))?,
                q_norm: load_t(e, &src, &p("attn_q_norm.weight"))?,
                post_attn_norm: load_t(e, &src, &p("post_attention_norm.weight"))?,
                ffn_norm: load_t(e, &src, &p("ffn_norm.weight"))?,
                ffn_gate: load_t(e, &src, &p("ffn_gate.weight"))?,
                ffn_up: load_t(e, &src, &p("ffn_up.weight"))?,
                ffn_down: load_t(e, &src, &p("ffn_down.weight"))?,
                ffn_post_norm: load_t(e, &src, &p("post_ffw_norm.weight"))?,
                out_scale,
                swa,
                hd,
                nh,
            });
        }
        let rope_freqs = {
            let t = src
                .find("rope_freqs.weight")
                .ok_or("drafter missing rope_freqs")?;
            e.htod(&bw24_gguf::dequant::dequantize(
                t.ggml_type,
                &t.bytes,
                t.ne.iter().product::<u64>() as usize,
            ))?
        };
        // FR-Spec head trim (BW24_GEMMA_DRAFT_RANKS=<ids file, rank order>): gather the ranked
        // rows of the drafter head + d2t map. (Top-N-IDS truncation measured NEGATIVE — id
        // order is not frequency; the CORPUS-ranked gather is the real FR-Spec.)
        // BW24_GEMMA_TRIM_ADAPT=<n> (default 512 when ranks are set, 0 = off) appends n spare
        // rows the serve loop fills from prompt ids + verify corrections (see TrimAdapt).
        let (head, d2t, trim_adapt) = {
            let t = src.find("token_embd.weight").ok_or("drafter missing token_embd")?;
            let in_f = t.ne[0] as usize;
            let n_vocab = t.ne[1] as usize;
            match std::env::var("BW24_GEMMA_DRAFT_RANKS").ok() {
                Some(path) => {
                    // row gather is layout-agnostic given the per-row byte stride: Q4_0 (26B
                    // drafter) and Q8_0 (31B drafter) both ship 32-elem blocks row-major.
                    // (qtype, elems/block, bytes/block) — the gather is stride-agnostic.
                    let (qtype, blk_e, blk_b) = match t.ggml_type {
                        bw24_gguf::GgmlType::Q4_0 => (crate::QT_Q4_0, 32, 18),
                        bw24_gguf::GgmlType::Q8_0 => (crate::QT_Q8_0, 32, 34),
                        bw24_gguf::GgmlType::Q6_K => (crate::QT_Q6_K, 256, 210),
                        other => panic!("drafter head trim: unsupported head type {other:?}"),
                    };
                    let ids: Vec<u32> = std::fs::read_to_string(&path)?
                        .lines().filter_map(|l| l.trim().parse().ok())
                        .filter(|&id| (id as usize) < n_vocab).collect();
                    let n_spare: usize = std::env::var("BW24_GEMMA_TRIM_ADAPT").ok()
                        .and_then(|v| v.parse().ok()).unwrap_or(512);
                    let row_bytes = in_f / blk_e * blk_b;
                    let mut gathered = Vec::with_capacity((ids.len() + n_spare) * row_bytes);
                    for &id in &ids {
                        let off = id as usize * row_bytes;
                        gathered.extend_from_slice(&t.bytes[off..off + row_bytes]);
                    }
                    // spare slots start as copies of row ids[0] mapping to ids[0] — a real,
                    // already-present token, so however the argmax resolves the duplicate-
                    // logit tie, the d2t translation lands on the same token id.
                    for _ in 0..n_spare {
                        let off = ids[0] as usize * row_bytes;
                        gathered.extend_from_slice(&t.bytes[off..off + row_bytes]);
                    }
                    eprintln!("[gemma-draft] FR head trim: {} rows + {} adaptive ({} MB vs {} MB full)",
                              ids.len(), n_spare,
                              (ids.len() + n_spare) * row_bytes / 1_000_000,
                              n_vocab * row_bytes / 1_000_000);
                    let mut trim_adapt = (n_spare > 0).then(|| {
                        let mut present = vec![false; n_vocab];
                        for &id in &ids { present[id as usize] = true; }
                        TrimAdapt {
                            src_rows: t.bytes.to_vec(),
                            row_bytes, n_vocab, present,
                            spare_base: ids.len(), n_spare, used: 0, logged_full: false,
                        }
                    });
                    let mut d2t = ids;
                    let spare_fill = d2t[0];
                    d2t.extend(std::iter::repeat_n(spare_fill, n_spare));
                    // pre-fill spare slots from the learned sidecar (trim_adapt_save):
                    // prior serves' escapes are proposable from round 1 of THIS serve.
                    if let Some(ta) = trim_adapt.as_mut() {
                        let learned: Vec<u32> = std::fs::read_to_string(format!("{path}.learned"))
                            .map(|t| t.lines().filter_map(|l| l.trim().parse().ok()).collect())
                            .unwrap_or_default();
                        let mut n_pre = 0usize;
                        for id in learned {
                            let i = id as usize;
                            if i < n_vocab && !ta.present[i] && ta.used < ta.n_spare {
                                let slot = ta.spare_base + ta.used;
                                ta.used += 1;
                                ta.present[i] = true;
                                let off = i * row_bytes;
                                gathered[slot * row_bytes..(slot + 1) * row_bytes]
                                    .copy_from_slice(&t.bytes[off..off + row_bytes]);
                                d2t[slot] = id;
                                n_pre += 1;
                            }
                        }
                        if n_pre > 0 {
                            eprintln!("[trim-adapt] {n_pre} learned rows pre-filled from {path}.learned");
                        }
                    }
                    // upload AFTER the sidecar pre-fill wrote its rows into `gathered`.
                    let bytes = e.htod_bytes(&gathered)?;
                    (GpuTensor::Quant {
                        bytes, qtype, row_bytes,
                        ne: vec![in_f as u64, d2t.len() as u64], scale: 1.0, rp: false,
                        #[cfg(bw24_cutlass)]
                        cutlass: None,
                        fp8: None, rp4: None,
                    }, Some(d2t), trim_adapt)
                }
                None => (load_t(e, &src, "token_embd.weight")?, None, None),
            }
        };
        // Q4_0 split-plane decode mirrors (BW24_Q4RP, same as the main trunk — see hybrid.rs):
        // the draft chain is 3 serial mmvq trips/round; the head alone is ~137MB/draft.
        // projection tensor prefix: 26B/31B "nextn.", the E4B assistant "mtp.".
        let proj_prefix = if src.find("nextn.pre_projection.weight").is_some() { "nextn" }
                          else { "mtp" };
        let (mut pre_proj, mut post_proj) =
            (load_t(e, &src, &format!("{proj_prefix}.pre_projection.weight"))?,
             load_t(e, &src, &format!("{proj_prefix}.post_projection.weight"))?);
        let mut head = head;
        let mut layers = layers;
        if crate::Engine::q4rp_enabled() {
            // adaptive-trim heads skip the split-plane mirror: the mmvq _rp twins read the
            // MIRROR, so an in-place row learn on `bytes` would be invisible to the matmul.
            let head_ws: &mut [&mut GpuTensor] = if trim_adapt.is_some() {
                &mut [&mut pre_proj, &mut post_proj]
            } else {
                &mut [&mut pre_proj, &mut post_proj, &mut head]
            };
            for w in head_ws.iter_mut() { e.build_q4_rp4(w)?; }
            for l in layers.iter_mut() {
                for w in [
                    &mut l.wq,
                    &mut l.wo,
                    &mut l.ffn_gate,
                    &mut l.ffn_up,
                    &mut l.ffn_down,
                ] {
                    e.build_q4_rp4(w)?;
                }
            }
        }
        let d2t_dev = match &d2t {
            Some(m) => Some(e.stream().clone_htod(&m[..])?),
            None => None,
        };
        Ok(GemmaDraft {
            layers,
            pre_proj,
            post_proj,
            output_norm: load_t(e, &src, "output_norm.weight")?,
            head,
            d2t,
            d2t_dev,
            trim_adapt,
            rope_freqs,
            ones: e.htod(&[1.0f32; 512])?,
            n_embd,
            n_backbone,
            rope_base_global: meta_f("rope.freq_base", 1e6),
            rope_base_swa: meta_f("rope.freq_base_swa", 1e4),
            sliding_window: meta_u("attention.sliding_window") as usize,
        })
    }
}

impl HybridModel {
    /// The MAIN layer whose KV cache a drafter layer attends (llama-model.cpp:2139):
    /// the last OWN-KV layer of the class — `boundary - 2` windowed / `boundary - 1`
    /// global, where boundary = n_layer - shared_kv_layers. Shared across every
    /// gemma4-assistant drafter (26B/31B: boundary = n_layer; E4B: 24).
    pub(crate) fn gemma4_draft_kv_target(&self, swa: bool) -> usize {
        let shared = self.cfg.gemma4.as_ref().map(|g| g.shared_kv_layers as usize).unwrap_or(0);
        let boundary = self.layers.len() - shared;
        boundary - if swa { 2 } else { 1 }
    }

    /// One drafter step: (token, h[2816 device]) at absolute position `pos` over the FROZEN main
    /// cache. Returns (draft logits host [n_vocab], h_next [2816 device]).
    pub fn gemma4_draft_step(
        &self,
        e: &Engine,
        d: &GemmaDraft,
        token: u32,
        h: &CudaSlice<f32>,
        pos: usize,
        cache: &Cache,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (hn, h_next) = self.gemma4_draft_trunk(e, d, token, h, pos, cache)?;
        let logits = e.dtoh(&e.matmul(&d.head, &hn, 1)?)?;
        Ok((logits, h_next))
    }

    /// Drafter trunk with the token in DEVICE memory (a 1-elem view of the round's batch
    /// buffer) — zero host traffic.
    fn gemma4_draft_trunk_dev(
        &self,
        e: &Engine,
        d: &GemmaDraft,
        tok_v: &cudarc::driver::CudaView<u32>,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        cache: &Cache,
        dc_bucket: Option<usize>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let nb = d.n_backbone;
        let embd_gpu = self
            .embd_gpu
            .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload"));
        let (qt, rb) = self.embd.qt_and_row_bytes(nb);
        let mut xs = e.embed_gather_device_tv(embd_gpu, tok_v, 1, nb, qt, rb)?;
        e.scale_inplace(&mut xs, (nb as f32).sqrt(), nb)?;
        self.gemma4_draft_trunk_from_x(e, d, &xs, h, pos_d, cache, dc_bucket)
    }

    /// Drafter trunk: returns (post-output_norm hidden [1024], h_next [2816]).
    fn gemma4_draft_trunk(
        &self,
        e: &Engine,
        d: &GemmaDraft,
        token: u32,
        h: &CudaSlice<f32>,
        pos: usize,
        cache: &Cache,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let nb = d.n_backbone;
        let mut xs = e.htod(&self.embd.gather(nb, &[token]))?;
        e.scale_inplace(&mut xs, (nb as f32).sqrt(), nb)?;
        let pos_d = e.htod_i32(&[pos as i32])?;
        return self.gemma4_draft_trunk_from_x(e, d, &xs, h, &pos_d, cache, None);
    }

    /// Trunk body from the pre-scaled main-embed row.
    fn gemma4_draft_trunk_from_x(
        &self,
        e: &Engine,
        d: &GemmaDraft,
        xs: &CudaSlice<f32>,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        cache: &Cache,
        dc_bucket: Option<usize>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // pos rides a DEVICE slot (burst-arc step a, 2026-07-12): the round fills persistent
        // slots via set_i32_one (kernel-arg stores — no per-step htod/alloc) and the chain
        // becomes graph-capturable (an in-graph i32_copy_add can feed the slots later).
        let eps = self.cfg.rms_eps;
        let ne = d.n_embd;

        // xh = concat(x, h) [2*n_backbone]
        let nb = d.n_backbone;
        let mut xh = e.uninit(2 * nb)?;
        e.copy_into(&mut xh, 0, xs, nb)?;
        e.copy_into(&mut xh, nb, h, nb)?;

        let mut cur = e.matmul(&d.pre_proj, &xh, 1)?; // [1024]

        for (_il, dl) in d.layers.iter().enumerate() {
            // attention over the shared MAIN KV: swa -> the last OWN-KV windowed layer,
            // global -> the last OWN-KV global layer (llama-model.cpp:2139 rule). Plain
            // 26B/31B trunks have no shared tail, so this is n-2 / n-1 there; E4B's 18
            // KV-shared tail layers move the boundary to 24 -> targets 22 (swa) / 23.
            let main_il = self.gemma4_draft_kv_target(dl.swa);
            let kvl = cache.kv[main_il].as_ref().unwrap();
            let (hd, nhh) = (dl.hd, dl.nh);
            let nkv = kvl.kv_dim_k / hd;
            let base = if dl.swa {
                d.rope_base_swa
            } else {
                d.rope_base_global
            };

            let mut hn = e.uninit(ne)?;
            e.rms_norm(&cur, dl.attn_norm.float_data(), &mut hn, ne, 1, eps)?;
            let q0 = e.matmul(&dl.wq, &hn, 1)?;
            let mut q = e.uninit(nhh * hd)?;
            e.rms_norm(&q0, dl.q_norm.float_data(), &mut q, hd, nhh, eps)?;
            if dl.swa {
                e.rope_neox(&mut q, pos_d, hd, hd, nhh, 1, base, 1.0)?;
            } else {
                e.rope_neox_ff(&mut q, pos_d, hd, hd, nhh, 1, base, 1.0, &d.rope_freqs)?;
            }
            let avail = kvl.len;
            let win = d.sliding_window;
            let mut attn = e.uninit(nhh * hd)?;
            // drafter attends the MAIN cache — its format follows the main layer's class
            // (windowed L28 = wkv arm, global L29 = gkv arm; gkv routing is hd-keyed inside).
            // DEVICE-LEN arms (burst arc): the length rides the main layer's len_d counter
            // so the chain is replay-correct across rounds. dc_bucket = the RUNG the round
            // derived (power-of-2, shared by eager and captured replays — same n_splits,
            // same combine order; the main graph arc's bucket lesson). None = host-len arm.
            if let Some(bucket) = dc_bucket {
                let k_view = e.view_u8(&kvl.k, kvl.k.len());
                let v_view = e.view_u8(&kvl.v, kvl.v.len());
                if dl.swa && avail > win {
                    e.fa_decode_rows_w(
                        &q,
                        &k_view,
                        &v_view,
                        &mut attn,
                        hd,
                        nhh,
                        nkv,
                        &kvl.len_d,
                        -1,
                        1,
                        1.0,
                        win,
                        kvl.k_tok_bytes,
                        kvl.v_tok_bytes,
                    )?;
                } else {
                    e.fa_decode_dc(
                        &q,
                        &k_view,
                        &v_view,
                        &mut attn,
                        hd,
                        nhh,
                        nkv,
                        &kvl.len_d,
                        bucket,
                        1.0,
                        kvl.k_tok_bytes,
                        kvl.v_tok_bytes,
                        dl.swa && crate::Engine::wkv_on(),
                    )?;
                }
            } else {
                let (off_tok, t_kv) = if dl.swa && avail > win {
                    (avail - win, win)
                } else {
                    (0, avail)
                };
                let k_view = e.view_u8_range(
                    &kvl.k,
                    off_tok * kvl.k_tok_bytes,
                    (off_tok + t_kv) * kvl.k_tok_bytes,
                );
                let v_view = e.view_u8_range(
                    &kvl.v,
                    off_tok * kvl.v_tok_bytes,
                    (off_tok + t_kv) * kvl.v_tok_bytes,
                );
                e.fa_decode_kvmod(
                    &q,
                    &k_view,
                    &v_view,
                    &mut attn,
                    hd,
                    nhh,
                    nkv,
                    t_kv,
                    1.0,
                    kvl.k_tok_bytes,
                    kvl.v_tok_bytes,
                    dl.swa && crate::Engine::wkv_on(),
                )?;
            }
            let o = e.matmul(&dl.wo, &attn, 1)?;

            let mut post = e.uninit(ne)?;
            e.rms_norm(&o, dl.post_attn_norm.float_data(), &mut post, ne, 1, eps)?;
            let mut attn_out = e.uninit(ne)?;
            e.add(&post, &cur, &mut attn_out, ne)?;

            let mut z = e.uninit(ne)?;
            e.rms_norm(&attn_out, dl.ffn_norm.float_data(), &mut z, ne, 1, eps)?;
            let n_ff = dl.ffn_gate.out_features();
            let gate = e.matmul(&dl.ffn_gate, &z, 1)?;
            let up = e.matmul(&dl.ffn_up, &z, 1)?;
            let mut act = e.uninit(n_ff)?;
            e.gelu_tanh_mul(&gate, &up, &mut act, n_ff)?;
            let f0 = e.matmul(&dl.ffn_down, &act, 1)?;
            let mut fpost = e.uninit(ne)?;
            e.rms_norm(&f0, dl.ffn_post_norm.float_data(), &mut fpost, ne, 1, eps)?;
            let mut xn = e.uninit(ne)?;
            e.add_scale(&fpost, &attn_out, dl.out_scale, &mut xn, ne)?;
            cur = xn;
        }

        let mut hn = e.uninit(ne)?;
        e.rms_norm(&cur, d.output_norm.float_data(), &mut hn, ne, 1, eps)?;
        let h_next = e.matmul(&d.post_proj, &hn, 1)?; // [2816]; head applied by callers (NO softcap)
        Ok((hn, h_next))
    }

    /// Greedy draft step: like gemma4_draft_step but the token argmax stays on device —
    /// host sees 4 bytes (no 1MB logits dtoh per draft). Returns (token, h_next).
    pub fn gemma4_draft_step_greedy(
        &self,
        e: &Engine,
        d: &GemmaDraft,
        token: u32,
        h: &CudaSlice<f32>,
        pos: usize,
        cache: &Cache,
    ) -> Result<(u32, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (hn, h_next) = self.gemma4_draft_trunk(e, d, token, h, pos, cache)?;
        let ld = e.matmul(&d.head, &hn, 1)?;
        let tok_d = e.argmax_token_device(&ld, d.head.out_features())?;
        let idx = e.dtoh_u32(&tok_d)?[0];
        let tok = match &d.d2t {
            Some(map) => map[idx as usize],
            None => idx,
        };
        Ok((tok, h_next))
    }
}

impl HybridModel {
    /// gemma4 MTP greedy spec loop: prime the prompt, then rounds of (chained K-token draft
    /// over the frozen main cache) + (ONE batched verify) + longest-prefix accept + KV rollback.
    /// Returns generated tokens; prints acceptance stats.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_spec_gemma(&self, e: &Engine, d: &mut GemmaDraft, prompt: &[u32],
                               max_new: usize, k: usize, eos: &[u32])
                               -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let mut cache = Cache::new(e, &self.cfg, prompt.len() + max_new + k + 8)?;

        // Adaptive trim, learn point 1: the PROMPT's own ids — the measured escapees are the
        // prompt's domain content words echoed back (▁oceans, clouds, Explain...), so the
        // prompt is the cheapest predictor of what the trim is about to miss.
        trim_adapt_learn(e, d, prompt)?;

        let t_prime = std::time::Instant::now();
        // short prompts fall below prime_cache's T floor — the batched verify IS a prime.
        let (pl, h_seed) = if prompt.len() >= crate::hybrid_forward::PRIME_MIN_T {
            let (l, hs, _hh) = self.prime_cache(e, prompt, &mut cache)?;
            (l, hs)
        } else if self.is_gemma4_e4b() {
            // E4B short-prompt prime: TOKENWISE — the batched e4b trunk at base_len==0
            // rides the PRIME-FA f32 arm (a different numerics class from the plain arm's
            // tokenwise prime), and the class skew flipped near-tie streams (3/64,
            // 2026-07-13). decode_step_h is the same chain the plain arm primes with.
            let n_embd_ = self.cfg.n_embd as usize;
            let mut ll = Vec::new();
            let mut hx = e.zeros(n_embd_)?;
            for &tok in prompt {
                let (l, hh) = self.gemma4_e4b_decode_step_h(e, tok, &mut cache)?;
                ll = l; hx = hh;
            }
            // decode_step_h returns the PRE-output_norm hidden; the short-prompt arm's
            // h convention below is POST-norm — norm here.
            let mut hp = e.uninit(n_embd_)?;
            e.rms_norm(&hx, self.output_norm.float_data(), &mut hp, n_embd_, 1, eps)?;
            (ll, hp)
        } else {
            let n_vocab = self.output.out_features();
            let (lv, hv) = self.gemma4_decode_step_t_h(e, prompt, 0, &mut cache)?;
            let t = prompt.len();
            let last = lv[(t - 1) * n_vocab..t * n_vocab].to_vec();
            // NOTE hv rows are POST-output_norm; h_seed convention below expects PRE-norm and
            // re-norms — so recover a pre-norm-free path: use the post-norm row DIRECTLY.
            let hvv = e.view(&hv, t * n_embd);
            let row = hvv.slice((t - 1) * n_embd..t * n_embd);
            let mut hrow = e.uninit(n_embd)?;
            e.copy_view_into(&mut hrow, 0, &row, n_embd)?;
            // mark: already post-norm — skip the re-norm below via the flag
            (last, hrow)
        };
        e.stream().synchronize()?;
        crate::PRIME_NANOS.store(
            t_prime.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // drafter h = POST-output_norm hidden (llama h_nextn); prime returns PRE-norm h_seed,
        // the short-prompt verify path already returns post-norm rows.
        let mut h = if prompt.len() >= crate::hybrid_forward::PRIME_MIN_T {
            let mut hh = e.uninit(n_embd)?;
            e.rms_norm(
                &h_seed,
                self.output_norm.float_data(),
                &mut hh,
                n_embd,
                1,
                eps,
            )?;
            hh
        } else {
            h_seed
        };

        let mut last = crate::forward::argmax(&pl) as u32;
                // BW24_PROFILE_SPEC=2: capture starts at the ROUND LOOP (prime excluded) — pair
        // with `nsys -c cudaProfilerApi` (the qwen loop's pattern, spec.rs).
        if std::env::var("BW24_PROFILE_SPEC").as_deref() == Ok("2") {
            unsafe extern "C" { fn cudaProfilerStart() -> i32; }
            unsafe { cudaProfilerStart(); }
        }
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let (mut drafted, mut accepted, mut rounds) = (0usize, 0usize, 0usize);
        // per-position accept histogram (BW24_SPEC_STATS): [attempted, accepted] per slot —
        // the depth-K policy statistic (deep slots' marginal accept decides fixed-cap vs deep).
        let mut pos_att = [0usize; 16];
        let mut pos_acc = [0usize; 16];

        // ASYNC ROUND v2 (dc class): the whole draft chain + verify enqueue with ZERO host
        // syncs — token seeds via kernel-arg store (u32_set_k, no host-memory transfer), draft
        // argmaxes land in the batch buffer, verify argmaxes in vam_d; ONE pack + ONE dtoh of
        // (k drafts + k+1 vam) closes the round. (v1 with memcpy_htod seeding measured
        // NEGATIVE — the pageable-copy sync; this is the retry with the sync removed.)
        let mut batch_d = e.stream().alloc_zeros::<u32>(k + 1)?;
        let mut packed = e.stream().alloc_zeros::<u32>(2 * k + 1)?;
        // confidence-adaptive depth (BW24_SPEC_PMIN, default 0 = off): per-draft probs.
        let pmin: f32 = std::env::var("BW24_SPEC_PMIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let mut p_d = e.stream().alloc_zeros::<f32>(k.max(1))?;

        // ADAPTIVE DRAFT LENGTH (default ON 2026-07-10; BW24_SPEC_ADAPT=0 reverts): llama's
        // draft-mtp reaches 0.64-0.70 acceptance on the SAME drafter (ours fixed-K: 0.52) by
        // drafting fewer tokens when unconfident (p-min gate). Zero-sync host proxy: next
        // round's depth = last round's accepted run + 1, clamped to [floor=1, k] — rounds
        // after a miss shrink, streaks re-deepen. The round's ONE dtoh already carries the
        // acceptance; no new syncs. Policy sweep (short chat, N=1 each): floor1/cap3 239.2
        // vs fixed-K3 231.1 (+3.5%, accept .52->.58); floor2 and cap4/5 all worse.
        let adapt = std::env::var("BW24_SPEC_ADAPT").as_deref() != Ok("0");
        // cap ceiling 7 by default; BW24_SPEC_CAPMAX opens the b16 verify tier (t=9..16).
        // The historical cap>=8 "crash" was two host bugs, both fixed 2026-07-12: round 1
        // ran UNCLAMPED (`kc = k` — verify t=K+1 entered the b16 tier while it was gated)
        // and the b16 dispatch requested _r2 twins that were never compiled (mcols==16 now
        // forces the base variant). Stream gates arbitrate any raised cap.
        let cap_max: usize = std::env::var("BW24_SPEC_CAPMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7);
        let k_cap = k.min(cap_max).max(1);
        // DRAFT-CHAIN GRAPHS (burst-arc step c, BW24_GEMMA_DRAFT_GRAPH=1): the whole k-step
        // draft chain replays as ONE captured graph — pos slots fill in-graph from pos_base,
        // the seed hidden rides the persistent g_seed buffer, KV lengths ride len_d (step b).
        // Keyed on (kr, rung, over_win): a new depth/rung/window regime captures lazily.
        let graph_on = std::env::var("BW24_GEMMA_DRAFT_GRAPH").as_deref() == Ok("1");
        let mut draft_graphs: std::collections::HashMap<
            (usize, usize, bool),
            (
                cudarc::driver::CudaGraph,
                Vec<Box<dyn std::any::Any + Send>>,
            ),
        > = Default::default();
        let mut g_seed = e.zeros(n_embd)?;
        let mut pos_base = e.htod_i32(&[0])?;
        // seed len_d before round 1 (prime went through the host-len path).
        for kvl in cache.kv.iter_mut().flatten() {
            e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
        }
        // persistent per-step rope-pos slots (device; filled by set_i32_one kernel-arg stores).
        let mut pos_slots: Vec<CudaSlice<i32>> = (0..k_cap.max(1))
            .map(|_| e.htod_i32(&[0]))
            .collect::<Result<_, _>>()?;
        // clamp round 1 too (the leak above).
        let mut kc = k_cap;
        // BURST (BW24_GEMMA_SPEC_BURST=M, default off): pre-issue M full rounds — draft-graph
        // replay + verify-stream + device accept/seed/rollback/ring-commit — with ONE host
        // sync per M rounds (the ring drain). The draft(N+1)-overlapping-verify(N) window this
        // opens is the burst arc's whole prize (~14% of a round; launch tax alone is hidden
        // at 96.7% busy). Requires the draft graphs (step c) and a regime-stable horizon.
        let burst_m: usize = std::env::var("BW24_GEMMA_SPEC_BURST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut burst_state: Option<(
            crate::round_stream::StreamBufs,
            CudaSlice<f32>,
            CudaSlice<u64>,
            crate::hybrid_forward::VerifyStreamScratch,
        )> = None;
        let win_main = self
            .cfg
            .gemma4
            .as_ref()
            .map(|g| g.sliding_window as usize)
            .unwrap_or(0);
        let g4_shared = self
            .cfg
            .gemma4
            .as_ref()
            .map(|g| g.shared_kv_layers)
            .unwrap_or(0);
        'outer: while out.len() < max_new {
            // burst gate first (see the BURST ARM below): a burst round drafts at FULL depth
            // (kr = k_cap — the captured chain replays a fixed K; adaptation is host logic).
            let horizon = burst_m * (k_cap + 1);
            let burst_ok = burst_m >= 1 && pmin == 0.0 && g4_shared == 0
                && (cache.pos + horizon + k_cap + 4 < win_main || cache.pos > win_main)
                // fa512 crossover: the whole horizon on one side (the stream verify's global
                // arm picks per-row-dc vs rows by hint; straddling rounds stay eager).
                && (cache.pos + horizon + k_cap + 4 < crate::fa512_min_tkv()
                    || cache.pos + 1 >= crate::fa512_min_tkv())
                && e.fa_rows_eligible(cache.pos, 256)
                && cache.pos + horizon + k_cap + 2 <= cache.max_ctx
                && out.len() + horizon <= max_new;
            let kr = if burst_ok {
                k_cap
            } else if adapt {
                kc
            } else {
                k_cap
            };
            // power-of-2 rung bucket for the dc arms (shared by eager and captured replays);
            // BW24_GEMMA_DRAFT_DC=0 reverts to the host-len kvmod arm.
            let dc_bucket: Option<usize> = {
                static DC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if *DC.get_or_init(|| std::env::var("BW24_GEMMA_DRAFT_DC").as_deref() != Ok("0")) {
                    let ml = cache
                        .kv
                        .iter()
                        .flatten()
                        .map(|kv| kv.len)
                        .max()
                        .unwrap_or(1);
                    // burst rounds size the rung for the WHOLE horizon: the captured chain
                    // replays M rounds between host looks, so the grid must cover the last
                    // round's len too (a per-round rung undersizes past its pow2 boundary).
                    let slack = if burst_ok { horizon } else { 0 };
                    Some((ml + slack + k_cap + 2).next_power_of_two().max(512))
                } else {
                    None
                }
            };
            e.u32_set_k(&mut batch_d, last, 0)?;
            e.copy_into(&mut g_seed, 0, &h, n_embd)?;
            // the draft chain, step j: reads g_seed via the hc chain, pos from pos_slots[j]
            // (eager: host-filled; graph: filled in-graph from pos_base).
            let run_chain = |e: &Engine, d: &GemmaDraft, batch_d: &mut CudaSlice<u32>,
                             p_d: &mut CudaSlice<f32>, g_seed: &CudaSlice<f32>,
                             pos_slots: &Vec<CudaSlice<i32>>|
             -> Result<(), Box<dyn std::error::Error>> {
                // uninit+copy (NOT clone_dtod): clone_dtod's internal alloc bypasses the
                // capture-retain hooks — its address got pool-reused between replays and the
                // replayed chain read a corrupted seed (accept 0.52 vs 0.76).
                let mut hc = e.uninit(n_embd)?;
                e.copy_into(&mut hc, 0, g_seed, n_embd)?;
                for j in 0..kr {
                    let tv = batch_d.slice(j..j + 1);
                    let (hn, h_next) = self.gemma4_draft_trunk_dev(
                        e,
                        d,
                        &tv,
                        &hc,
                        &pos_slots[j],
                        &cache,
                        dc_bucket,
                    )?;
                    let ld = e.matmul(&d.head, &hn, 1)?;
                    e.argmax_token_device_col(&ld, 0, d.head.out_features(), batch_d, j + 1)?;
                    // confidence-adaptive depth (BW24_SPEC_PMIN): TRIM-space prob before d2t.
                    if pmin > 0.0 {
                        e.prob_of_token_device_col(
                            &ld,
                            batch_d,
                            j + 1,
                            p_d,
                            j,
                            d.head.out_features(),
                        )?;
                    }
                    // FR-trimmed head: translate the trim-space argmax to the vocab id.
                    if let Some(map) = &d.d2t_dev {
                        e.u32_map_k(batch_d, map, j + 1)?;
                    }
                    hc = h_next;
                }
                Ok(())
            };
            let over_win = {
                let win = d.sliding_window;
                d.layers.iter().any(|dl| dl.swa
                    && cache.kv[self.gemma4_draft_kv_target(true)].as_ref()
                        .is_some_and(|kv| kv.len > win))
            };
            // ---- ROUND-GRAPH ARM ---- (BW24_GEMMA_ROUND_GRAPH=1): the WHOLE round —
            // draft chain + stream verify + device accept/seed/rollback/commit + the
            // device adaptive-depth update — captured ONCE per (k_cap, rung, over_win)
            // regime and replayed as ONE graph launch per round (the llama round-cost
            // mechanism: ~600 per-round enqueues collapse to 1). The round is SELF-FEEDING
            // (pos_ctr/pend/brk/g_seed all advance in-graph), so the capture warmups are
            // simply two SERVED rounds — their tokens land in the ring and drain normally
            // (no snapshot/rollback needed, unlike the E4B token door).
            // Adaptive K rides brk[0] via spec_adapt_k: drafts always run k_cap deep (the
            // drafter is cheap) but the accept walk depth follows the host policy exactly.
            let round_graph_on = std::env::var("BW24_GEMMA_ROUND_GRAPH").as_deref() == Ok("1");
            if round_graph_on && burst_m == 0 && dc_bucket.is_some() && pmin == 0.0
                && g4_shared == 0 && !self.is_gemma4_e4b()
                && (cache.pos + 2 * (k_cap + 1) + k_cap + 4 < win_main || cache.pos > win_main)
                && (cache.pos + 2 * (k_cap + 1) + k_cap + 4 < crate::fa512_min_tkv()
                    || cache.pos + 1 >= crate::fa512_min_tkv())
                && e.fa_rows_eligible(cache.pos, 256)
                && cache.pos + 2 * (k_cap + 1) + k_cap + 2 <= cache.max_ctx
            {
                if burst_state.is_none() {
                    // ring sized for the capture warmups (2 rounds) + the live round.
                    let bufs = crate::round_stream::StreamBufs::new(e, k_cap, 3)?;
                    let fill_dummy = e.zeros(n_embd)?;
                    let ptrs = crate::round_stream::kv_len_ptr_table(e, &cache,
                                                                     Some(&bufs.pos_ctr))?;
                    let scr = self.verify_stream_scratch(e, k_cap + 1)?;
                    burst_state = Some((bufs, fill_dummy, ptrs, scr));
                }
                let adapt_floor: usize = std::env::var("BW24_SPEC_ADAPT_FLOOR").ok()
                    .and_then(|v| v.parse().ok()).unwrap_or(1);
                // entry: `last` is the pending token (emitted at drain), h is the seed.
                let (bufs, fill_dummy, ptrs, scr) = burst_state.as_mut().unwrap();
                let n_rows = cache.kv.len() + 1;
                e.set_i32_one(&mut bufs.pos_ctr, cache.pos as i32)?;
                e.u32_set_k(&mut bufs.ring_d, 0, 0)?;
                e.u32_set_k(&mut bufs.pend_d, last, 0)?;
                e.u32_set_k(&mut bufs.brk_d, (if adapt { kc } else { k_cap }) as u32, 0)?;
                e.u32_set_k(&mut bufs.brk_d, 1, 1)?;
                e.copy_into(&mut g_seed, 0, &h, n_embd)?;
                // entry pend is emitted host-side (the ring only carries accepted drafts
                // + bonuses — the burst-arm contract).
                out.push(last);
                if eos.contains(&last) { break 'outer; }
                if out.len() >= max_new { break 'outer; }
                let key = (usize::MAX - k_cap, dc_bucket.unwrap(), over_win);
                let mut fresh_rounds = 1usize;   // rounds executed by this iteration
                // `hint` is the verify stream's ARM-GATING upper bound — it must sit on
                // the SAME side of every crossover as the live lengths this capture
                // serves, INCLUDING the arms' own margins (`hint + t < f512` gates the
                // global scalar arm; `hint + 1 >= win` gates rows_w), or the captured
                // verify bakes a different kernel class than the eager reference
                // (107-vs-106 / 4-64 drifts; the regime gate above guarantees the live
                // side with the same margins).
                let hint = if cache.pos > win_main {
                    dc_bucket.unwrap() + k_cap + 2          // over-window: rows_w regime
                } else if cache.pos + 1 >= crate::fa512_min_tkv() {
                    win_main - 2                             // above f512, under window
                } else {
                    crate::fa512_min_tkv().saturating_sub(k_cap + 5)   // under both
                };
                let bufs_ptr: *mut crate::round_stream::StreamBufs = &mut *bufs;
                let scr_ptr: *mut crate::hybrid_forward::VerifyStreamScratch = &mut *scr;
                let cache_ptr: *mut Cache = &mut cache;
                let batch_ptr: *mut CudaSlice<u32> = &mut batch_d;
                let seed_ptr: *mut CudaSlice<f32> = &mut g_seed;
                let slots_ptr: *mut Vec<CudaSlice<i32>> = &mut pos_slots;
                let mut round_body = |e: &Engine| -> Result<(), Box<dyn std::error::Error>> {
                    // SAFETY: single-threaded round body; the raw pointers alias the outer
                    // &mut only within this closure (no overlapping borrows).
                    let (bufs, scr, cache, batch_d, g_seed, pos_slots) = unsafe {
                        (&mut *bufs_ptr, &mut *scr_ptr, &mut *cache_ptr,
                         &mut *batch_ptr, &mut *seed_ptr, &mut *slots_ptr) };
                    e.i32_copy_add(&bufs.pos_ctr, &mut bufs.pos_start_d, 0)?;
                    e.u32_copy(&bufs.pend_d, batch_d)?;
                    for (j, slot) in pos_slots.iter_mut().take(k_cap).enumerate() {
                        e.i32_copy_add(&bufs.pos_ctr, slot, j as i32)?;
                    }
                    let mut hc = e.uninit(n_embd)?;
                    e.copy_into(&mut hc, 0, g_seed, n_embd)?;
                    for j in 0..k_cap {
                        let tv = batch_d.slice(j..j + 1);
                        let (hn, h_next) = self.gemma4_draft_trunk_dev(
                            e, d, &tv, &hc, &pos_slots[j], cache, dc_bucket)?;
                        let ld = e.matmul(&d.head, &hn, 1)?;
                        e.argmax_token_device_col(&ld, 0, d.head.out_features(),
                                                  batch_d, j + 1)?;
                        if let Some(map) = &d.d2t_dev {
                            e.u32_map_k(batch_d, map, j + 1)?;
                        }
                        hc = h_next;
                    }
                    let (vam_d, vh) = self.gemma4_verify_t_am_stream(
                        e, batch_d, k_cap + 1, &bufs.pos_ctr, hint, cache, scr)?;
                    e.spec_accept_greedy_dc(&vam_d, batch_d, &bufs.last_pred_d,
                                            &bufs.brk_d, &mut bufs.acc_d)?;
                    if std::env::var("BW24_DEBUG_SPEC").as_deref() == Ok("1")
                        && std::env::var("BW24_ROUND_GRAPH_CHECK").as_deref() == Ok("1") {
                        let vhh = e.dtoh(&vh)?;
                        let nrm = |r: usize| vhh[r * n_embd..(r + 1) * n_embd].iter()
                            .map(|x| x * x).sum::<f32>().sqrt();
                        let vamh = e.dtoh_u32(&vam_d)?;
                        eprintln!("[rg-vh] |row0|={:.3} |row1|={:.3} |row2|={:.3} vam={:?}",
                                  nrm(0), nrm(1), nrm(2), &vamh[..(k_cap + 1).min(7)]);
                    }
                    e.spec_seed_gather(&vh, fill_dummy, &bufs.acc_d, g_seed, 1, n_embd)?;
                    e.spec_rollback_stream(ptrs, &bufs.pos_start_d, &bufs.acc_d, 1, n_rows)?;
                    e.spec_ring_commit(batch_d, &bufs.acc_d, &bufs.brk_d,
                                       &mut bufs.ring_d, &mut bufs.pend_d)?;
                    e.spec_adapt_k(&bufs.acc_d, &mut bufs.brk_d, adapt_floor, k_cap)?;
                    Ok(())
                };
                // BW24_ROUND_GRAPH_CHECK=1: run the body EAGERLY (no capture/replay) —
                // splits "body semantics wrong" from "replay mechanics wrong".
                let body_check = std::env::var("BW24_ROUND_GRAPH_CHECK").as_deref() == Ok("1");
                if body_check {
                    round_body(e)?;
                    if std::env::var("BW24_DEBUG_SPEC").as_deref() == Ok("1") {
                        let acc = e.dtoh_u32(&bufs.acc_d)?;
                        let brk = e.dtoh_u32(&bufs.brk_d)?;
                        let bt = e.dtoh_u32(&batch_d)?;
                        let tgt = self.gemma4_draft_kv_target(true);
                        let ld = e.dtoh_i32(&cache.kv[tgt].as_ref().unwrap().len_d)?[0];
                        let gs = e.dtoh(&g_seed)?;
                        let gn: f32 = gs.iter().map(|x| x * x).sum::<f32>().sqrt();
                        eprintln!("[rg-check] pos0={} batch={bt:?} n_acc={} bonus={} brk_next={:?} len_d[L{tgt}]={ld} |g_seed|={gn:.3}",
                                  cache.pos, acc[0], acc[1], brk);
                    }
                } else {
                    if !draft_graphs.contains_key(&key) {
                        let g = e.capture_graph_retained(&mut round_body)?;
                        draft_graphs.insert(key, g);
                        fresh_rounds += 2;   // the capture warmups were served rounds
                    }
                    draft_graphs.get(&key).unwrap().0.launch()?;
                }
                // drain: ONE host sync per iteration (warmup rounds included on capture).
                let toks = bufs.drain_ring(e)?;
                let posh = e.dtoh_i32(&bufs.pos_ctr)?[0] as usize;
                if std::env::var("BW24_DEBUG_SPEC").as_deref() == Ok("1") {
                    eprintln!("[round-graph] fresh={fresh_rounds} drained={} posh={posh} toks={:?}",
                              toks.len(), &toks[..toks.len().min(12)]);
                }
                drafted += fresh_rounds * k_cap;
                rounds += fresh_rounds;
                accepted += toks.len().saturating_sub(fresh_rounds);
                let mut ended = false;
                for &tk in &toks[..toks.len() - 1] {
                    out.push(tk);
                    if eos.contains(&tk) || out.len() >= max_new { ended = true; break; }
                }
                last = *toks.last().unwrap();
                cache.pos = posh;
                for kvl in cache.kv.iter_mut().flatten() { kvl.len = posh; }
                // NO allocation between replays: a pool alloc here can land on a baked
                // transient address and corrupt the next replay (the draft-graph lesson).
                // g_seed already holds the next seed (in-graph gather); copy INTO the
                // existing h buffer for the (possible) eager-arm handoff.
                e.copy_into(&mut h, 0, &g_seed, n_embd)?;
                kc = k_cap;   // device brk owns the walk depth; host kc only seeds entry
                // learn point 2 (round-graph drain): ring = accepted drafts + bonuses; only
                // bonuses can be escapes, and the present-bitmap check skips the rest cheap.
                trim_adapt_learn(e, d, &toks)?;
                if ended { break 'outer; }
                continue 'outer;
            }
            // ---- BURST ARM ---- (gate computed at the loop top; needs dc arms too)
            if burst_ok && dc_bucket.is_some() {
                if burst_state.is_none() {
                    let bufs = crate::round_stream::StreamBufs::new(e, k_cap, burst_m)?;
                    let fill_dummy = e.zeros(n_embd)?; // spec_seed_gather j>=1 always: unread
                    let ptrs =
                        crate::round_stream::kv_len_ptr_table(e, &cache, Some(&bufs.pos_ctr))?;
                    let scr = self.verify_stream_scratch(e, k_cap + 1)?;
                    burst_state = Some((bufs, fill_dummy, ptrs, scr));
                }
                // the loop-top dc_bucket already carries the horizon slack on burst rounds,
                // so the key below matches the rung the captured chain actually launches with.
                let key = (k_cap, dc_bucket.unwrap(), over_win);
                if std::env::var("BW24_GEMMA_BURST_GRAPH").as_deref() == Ok("1")
                    && !draft_graphs.contains_key(&key)
                {
                    let g = e.capture_graph_retained(|e| {
                        run_chain(e, d, &mut batch_d, &mut p_d, &g_seed, &pos_slots)
                    })?;
                    draft_graphs.insert(key, g);
                }
                // entry: `last` is the not-yet-emitted pending token (the ring only ever
                // carries accepted drafts + bonuses; the entry pend is emitted host-side).
                out.push(last);
                if eos.contains(&last) {
                    break 'outer;
                }
                if out.len() >= max_new {
                    break 'outer;
                }
                let (bufs, fill_dummy, ptrs, scr) = burst_state.as_mut().unwrap();
                let n_rows = cache.kv.len() + 1; // + the pos counter row
                e.set_i32_one(&mut bufs.pos_ctr, cache.pos as i32)?;
                e.u32_set_k(&mut bufs.ring_d, 0, 0)?;
                e.u32_set_k(&mut bufs.pend_d, last, 0)?;
                e.u32_set_k(&mut bufs.brk_d, k_cap as u32, 0)?; // k_used = K (no p-min cut)
                e.u32_set_k(&mut bufs.brk_d, 1, 1)?; // base = 1 (pend always set)
                e.copy_into(&mut g_seed, 0, &h, n_embd)?;
                let pos0 = cache.pos;
                for r in 0..burst_m {
                    // every op below is ENQUEUED; nothing reads back until the drain.
                    e.i32_copy_add(&bufs.pos_ctr, &mut bufs.pos_start_d, 0)?;
                    e.u32_copy(&bufs.pend_d, &mut batch_d)?; // batch_d[0] <- pend
                    for (j, slot) in pos_slots.iter_mut().take(k_cap).enumerate() {
                        e.i32_copy_add(&bufs.pos_ctr, slot, j as i32)?;
                    }
                    // the chain enqueues ZERO-SYNC with device pos slots — the captured-graph
                    // replay is measured EXPENSIVE (26B eager 379 -> 253 with replay), so the
                    // burst runs the chain eagerly by default; BW24_GEMMA_BURST_GRAPH=1 keeps
                    // the replay door for A/B.
                    if std::env::var("BW24_GEMMA_BURST_GRAPH").as_deref() == Ok("1") {
                        draft_graphs.get(&key).unwrap().0.launch()?;
                    } else {
                        // run_chain's body inlined: the closure holds &cache for the loop's
                        // lifetime and collides with the verify's &mut cache borrow.
                        let mut hc = e.uninit(n_embd)?;
                        e.copy_into(&mut hc, 0, &g_seed, n_embd)?;
                        for j in 0..k_cap {
                            let tv = batch_d.slice(j..j + 1);
                            let (hn, h_next) = self.gemma4_draft_trunk_dev(
                                e,
                                d,
                                &tv,
                                &hc,
                                &pos_slots[j],
                                &cache,
                                dc_bucket,
                            )?;
                            let ld = e.matmul(&d.head, &hn, 1)?;
                            e.argmax_token_device_col(
                                &ld,
                                0,
                                d.head.out_features(),
                                &mut batch_d,
                                j + 1,
                            )?;
                            if let Some(map) = &d.d2t_dev {
                                e.u32_map_k(&mut batch_d, map, j + 1)?;
                            }
                            hc = h_next;
                        }
                    }
                    // host UPPER bound on this round's base (full-accept growth): sizes the
                    // stream verify's splits + window-arm gate; device len is the true bound.
                    let hint = pos0 + (r + 1) * (k_cap + 1) + 2;
                    let (vam_d, vh) = self.gemma4_verify_t_am_stream(
                        e,
                        &batch_d,
                        k_cap + 1,
                        &bufs.pos_ctr,
                        hint,
                        &mut cache,
                        scr,
                    )?;
                    e.spec_accept_greedy_dc(
                        &vam_d,
                        &batch_d,
                        &bufs.last_pred_d,
                        &bufs.brk_d,
                        &mut bufs.acc_d,
                    )?;
                    e.spec_seed_gather(&vh, fill_dummy, &bufs.acc_d, &mut g_seed, 1, n_embd)?;
                    e.spec_rollback_stream(ptrs, &bufs.pos_start_d, &bufs.acc_d, 1, n_rows)?;
                    e.spec_ring_commit(
                        &batch_d,
                        &bufs.acc_d,
                        &bufs.brk_d,
                        &mut bufs.ring_d,
                        &mut bufs.pend_d,
                    )?;
                }
                // drain: THE one sync per M rounds. Ring = [acc..., bonus] per round; the
                // final element is the next pending token (eager pushes it next round).
                let toks = bufs.drain_ring(e)?;
                let posh = e.dtoh_i32(&bufs.pos_ctr)?[0] as usize;
                drafted += burst_m * k_cap;
                rounds += burst_m;
                accepted += toks.len().saturating_sub(burst_m); // each round adds n_acc + 1
                let mut ended = false;
                for &tk in &toks[..toks.len() - 1] {
                    out.push(tk);
                    if eos.contains(&tk) || out.len() >= max_new {
                        ended = true;
                        break;
                    }
                }
                last = *toks.last().unwrap();
                // host mirrors re-sync (device counters are already correct from rollback).
                cache.pos = posh;
                for kvl in cache.kv.iter_mut().flatten() {
                    kvl.len = posh;
                }
                // next seed hidden = g_seed (the final round's device gather).
                let mut hrow = e.uninit(n_embd)?;
                e.copy_into(&mut hrow, 0, &g_seed, n_embd)?;
                h = hrow;
                kc = k_cap;
                // learn point 2 (burst drain): same contract as the round-graph drain.
                trim_adapt_learn(e, d, &toks)?;
                if ended { break 'outer; }
                continue 'outer;
            }
            if graph_on && dc_bucket.is_some() {
                let key = (kr, dc_bucket.unwrap(), over_win);
                if !draft_graphs.contains_key(&key) {
                    // chain-only capture; pos slots are graph INPUTS (filled eagerly before
                    // each launch, like g_seed — the in-graph copy_add fills replayed one
                    // round stale, see jsonl).
                    let g = e.capture_graph_retained(|e| {
                        run_chain(e, d, &mut batch_d, &mut p_d, &g_seed, &pos_slots)
                    })?;
                    draft_graphs.insert(key, g);
                }
                for (j, slot) in pos_slots.iter_mut().take(kr).enumerate() {
                    e.set_i32_one(slot, (cache.pos + j) as i32)?;
                }
                draft_graphs.get(&key).unwrap().0.launch()?;
                // BW24_DRAFT_GRAPH_CHECK=1: re-run the chain eagerly from the same state and
                // diff the drafted slots (replay-vs-eager divergence bisect).
                if std::env::var("BW24_DRAFT_GRAPH_CHECK").as_deref() == Ok("1") {
                    // NON-DESTRUCTIVE: compare, then restore the graph's tokens so the round
                    // proceeds exactly as it would without the check.
                    let gtoks = e.dtoh_u32(&batch_d)?;
                    for (j, slot) in pos_slots.iter_mut().take(kr).enumerate() {
                        e.set_i32_one(slot, (cache.pos + j) as i32)?;
                    }
                    run_chain(e, d, &mut batch_d, &mut p_d, &g_seed, &pos_slots)?;
                    let etoks = e.dtoh_u32(&batch_d)?;
                    if gtoks[..=kr] != etoks[..=kr] {
                        eprintln!(
                            "[draft-graph] DIVERGE round={rounds} graph={:?} eager={:?}",
                            &gtoks[..=kr],
                            &etoks[..=kr]
                        );
                    }
                    for (j, &t) in gtoks.iter().enumerate().take(kr + 1) {
                        e.u32_set_k(&mut batch_d, t, j)?;
                    }
                }
            } else {
                for (j, slot) in pos_slots.iter_mut().take(kr).enumerate() {
                    e.set_i32_one(slot, (cache.pos + j) as i32)?;
                }
                run_chain(e, d, &mut batch_d, &mut p_d, &g_seed, &pos_slots)?;
            }
            drafted += kr;
            rounds += 1;
            let pos0 = cache.pos;
            // BW24_BURST_VCHECK=1: run the STREAM verify first on the same batch/state and
            // diff its argmaxes against the eager verify (bisect harness — the stream append
            // writes the same rows the eager append then overwrites, so state is untouched).
            let vcheck = std::env::var("BW24_BURST_VCHECK").as_deref() == Ok("1");
            let kvsum = |e: &Engine,
                         cache: &Cache|
             -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
                let mut out = Vec::new();
                for kvl in cache.kv.iter().flatten() {
                    let kb = e.dtoh_u8(&kvl.k)?;
                    let vb = e.dtoh_u8(&kvl.v)?;
                    let lo = pos0 * kvl.k_tok_bytes;
                    let hi = (pos0 + kr + 1) * kvl.k_tok_bytes;
                    let lov = pos0 * kvl.v_tok_bytes;
                    let hiv = (pos0 + kr + 1) * kvl.v_tok_bytes;
                    out.push((
                        kb[lo..hi].iter().map(|&b| b as u64).sum(),
                        vb[lov..hiv].iter().map(|&b| b as u64).sum(),
                    ));
                }
                Ok(out)
            };
            let vam_s = if vcheck && !self.is_gemma4_e4b() {
                let mut ctr = e.htod_i32(&[pos0 as i32])?;
                e.set_i32_one(&mut ctr, pos0 as i32)?;
                let mut scr0 = self.verify_stream_scratch(e, kr + 1)?;
                let (vs, vhs) = self.gemma4_verify_t_am_stream(e, &batch_d, kr + 1, &ctr,
                                                               pos0 + kr + 3, &mut cache,
                                                               &mut scr0)?;
                let ss = kvsum(e, &cache)?;
                Some((e.dtoh_u32(&vs)?, ss, e.dtoh(&vhs)?))
            } else { None };
            let (vam_d, vh) = if self.is_gemma4_e4b() {
                self.gemma4_e4b_decode_step_t_am_dev(e, &batch_d, kr + 1, pos0, &mut cache)?
            } else {
                self.gemma4_decode_step_t_am_dev(e, &batch_d, kr + 1, pos0, &mut cache)?
            };
            if let Some((vs, ss, vhs)) = vam_s {
                let vhe = e.dtoh(&vh)?;
                for r in 0..kr + 1 {
                    let md = vhs[r * n_embd..(r + 1) * n_embd].iter()
                        .zip(&vhe[r * n_embd..(r + 1) * n_embd])
                        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    if md > 1e-3 {
                        eprintln!("[vcheck-vh] round={rounds} row={r} maxdiff={md:.3e}");
                    }
                }
                let se = kvsum(e, &cache)?;
                for (il, (a, b)) in ss.iter().zip(&se).enumerate() {
                    if a != b {
                        eprintln!("[vcheck-kv] round={rounds} il={il} stream={a:?} eager={b:?}");
                    }
                }
                let ve = e.dtoh_u32(&vam_d)?;
                if vs[..kr + 1] != ve[..kr + 1] {
                    eprintln!(
                        "[vcheck] DIVERGE round={rounds} pos0={pos0} stream={:?} eager={:?}",
                        &vs[..kr + 1],
                        &ve[..kr + 1]
                    );
                } else {
                    eprintln!("[vcheck] match round={rounds} pos0={pos0}");
                }
            }
            e.u32_pack2(&batch_d, 1, kr, &vam_d, kr + 1, &mut packed)?;
            let host = e.dtoh_u32(&packed)?; // the round's ONE sync
            let k = kr;
            let dtoks: Vec<u32> = host[..k].to_vec();
            let vam: Vec<u32> = host[k..2 * k + 1].to_vec();
            // longest accepted prefix: d_i accepted iff d_i == argmax(verify[i-1])
            // (trimmed heads: batch_d slots were d2t-translated in the draft loop, so dtoks
            // are full-vocab ids here — the 2026-07-10 async rewrite silently dropped this
            // and the trim probes read accept=0.000 through it.)
            let mut m = 0usize;
            while m < k {
                if dtoks[m] == vam[m] {
                    m += 1;
                } else {
                    break;
                }
            }
            if std::env::var("BW24_DEBUG_SPEC").as_deref() == Ok("1") {
                let l0 = cache.kv.iter().flatten().next().map(|kv| kv.len).unwrap_or(0);
                let hh = e.dtoh(&h)?;
                let hn: f32 = hh.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("[round {rounds}] pos0={pos0} post_pos={} kv0_len={l0} last={last} dtoks={dtoks:?} vam={vam:?} m={m} |h_in|={hn:.3}",
                          cache.pos);
            }
            accepted += m;
            for j in 0..k.min(16) {
                pos_att[j] += 1;
                if j < m { pos_acc[j] += 1; }
            }
            // emit last + accepted drafts; the correction token comes from verify row m.
            out.push(last);
            if eos.contains(&last) {
                break 'outer;
            }
            for &dt in &dtoks[..m] {
                out.push(dt);
                if eos.contains(&dt) {
                    break 'outer;
                }
                if out.len() >= max_new {
                    break 'outer;
                }
            }
            let next = vam[m];
            // roll back rejected rows: batch appended k+1 rows; keep m+1 (positions of
            // last + accepted drafts). SWA layers cap t_kv by the window view, so a plain
            // len rewind is safe for every layer.
            let keep = m + 1;
            for kvl in cache.kv.iter_mut().flatten() {
                kvl.len -= (k + 1) - keep;
                // keep len_d in lockstep: the drafter's device-len attention arms read it
                // (the gemma round appends via the HOST-len path, which doesn't maintain
                // the counter — stale len_d gutted acceptance to 0.059 on the dc probe).
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            cache.pos -= (k + 1) - keep;
            // h for the next round = main hidden at the LAST KEPT position (verify row m).
            let hv = e.view(&vh, (k + 1) * n_embd);
            let row = hv.slice(m * n_embd..(m + 1) * n_embd);
            let mut hrow = e.uninit(n_embd)?;
            e.copy_view_into(&mut hrow, 0, &row, n_embd)?;
            h = hrow;
            last = next;
            // Adaptive trim, learn point 2: ALL verify argmaxes — vam[m] is the emitted
            // correction (the only emitted token that can sit outside the trim set; accepted
            // drafts are trim members by construction), and vam[i>m] are main-model
            // predictions for positions never reached this round: next round usually wants
            // exactly those tokens, so learning them here lets the draft propose them
            // BEFORE any miss is paid (prose escapes are first-occurrence-dominated —
            // corrections-only learning measured +0.5 acceptance pts, jsonl 2026-07-19).
            trim_adapt_learn(e, d, &vam)?;
            if adapt {
                let floor: usize = std::env::var("BW24_SPEC_ADAPT_FLOOR")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1);
                kc = (m + 1).clamp(floor.min(k_cap), k_cap);
                // confidence cut (BW24_SPEC_PMIN > 0): next round drafts no deeper than one
                // past the first low-confidence draft of THIS round (llama's p-min class,
                // one round late — the zero-sync enqueue stays intact). One extra tiny dtoh.
                if pmin > 0.0 {
                    let ph = e.dtoh(&p_d)?;
                    if let Some(fl) = ph[..kr].iter().position(|&p| p < pmin) {
                        kc = kc.min((fl + 1).max(floor.min(k_cap)));
                    }
                }
            }
        }
        eprintln!("[gemma-spec] rounds={rounds} drafted={drafted} accepted={accepted}                    accept-rate={:.3} tok/round={:.2}",
                  accepted as f64 / drafted.max(1) as f64,
                  out.len() as f64 / rounds.max(1) as f64);
        if let Some((used, budget)) = d.trim_adapt_stats() {
            eprintln!("[trim-adapt] {used}/{budget} spare slots learned");
            match d.trim_adapt_save() {
                Ok(n) if n > 0 => eprintln!("[trim-adapt] {n} new ids appended to the .learned sidecar"),
                Ok(_) => {}
                Err(err) => eprintln!("[trim-adapt] sidecar save failed: {err}"),
            }
        }
        if std::env::var("BW24_SPEC_STATS").as_deref() == Ok("1") {
            let hist: Vec<String> = (0..16).filter(|&j| pos_att[j] > 0)
                .map(|j| format!("p{j}:{}/{}", pos_acc[j], pos_att[j])).collect();
            eprintln!("[gemma-spec] per-position accept: {}", hist.join(" "));
        }
        Ok(out)
    }
}
