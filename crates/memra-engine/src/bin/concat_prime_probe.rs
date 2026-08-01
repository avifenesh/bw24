//! concat-prime-probe (lane/concat-prime-exact): solo-vs-concat batch-prime differential.
//!
//! The serve lane (research/ornith-serve-20260801 §2) pinned greedy c1-vs-c16 divergence on
//! Ornith-35B/KAT to the batch-prime concat prefill (prime_cache_batch). This probe separates
//! (a) the m=sum_T concat-GEMM FP-reduction class from (b) an indexing/masking/state defect
//! in the concat path, and measures the near-tie margins that decide whether the FP class
//! flips greedy argmax.
//!
//! modes (all greedy, all engine-level — no server):
//!   repro   <model> repro   --prompt-a <txt> --prompt-b <txt> [--steps N] [--chat]
//!           prime A solo vs A in concat [A,B]; lockstep greedy decode from both caches;
//!           first-divergence step, top-2 margins both sides, full-vocab logit maxdiff.
//!   posdiff <model> posdiff --prompt-a <txt> --prompt-b <txt> [--order ab|ba] [--chat]
//!           per-POSITION prefill hidden + logit diff, solo vs concat (both sides' logits
//!           computed by the SAME m=1 epilogue so the diff isolates the trunk). A defect
//!           shows structured position/boundary-dependent divergence; FP noise scatters.
//!   content <model> content --prompt-a <txt> --prompt-b <txt> --prompt-c <txt> [--chat]
//!           leakage razor: B and C truncated to equal token length; A's concat outputs
//!           must be BIT-IDENTICAL across co-batch content [A,B] vs [A,C] (row-independent
//!           GEMMs + per-seq cores => only shapes may matter). Also determinism ([A,B] x2)
//!           and offset variant ([B,A] vs [C,A]).
//!   margins <model> margins --prompts-file <f> [--steps N] [--chat] [--jsonl <out>]
//!           per-prompt greedy top1-top2 logit-gap distribution (prefill + every decode
//!           step) — the near-tie density that converts FP perturbation into argmax flips.

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            i2 = i1; v2 = v1; i1 = i; v1 = v;
        } else if v > v2 {
            i2 = i; v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn arg(rest: &[String], key: &str) -> Option<String> {
    rest.iter().position(|a| a == key).and_then(|i| rest.get(i + 1)).cloned()
}

fn encode_prompt(tok: &Tokenizer, text: &str, chat: bool) -> Vec<u32> {
    // exactly the server's chat arm (worker.rs:850): template + encode(parse_special)
    if chat {
        let rendered = tok.apply_chat_template(&[("user", text)], true);
        tok.encode(&rendered, true)
    } else {
        tok.encode(text, true)
    }
}

struct Ctx {
    e: Engine,
    model: HybridModel,
    tok: Tokenizer,
    ctx_len: usize,
}

impl Ctx {
    /// prime A solo; greedy-decode `steps`; return (streams, per-step margins, prefill logits)
    fn solo_stream(&self, toks: &[u32], steps: usize)
                   -> Result<(Vec<u32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let mut c = Cache::new(&self.e, &self.model.cfg, self.ctx_len)?;
        let (logits, _, _) = self.model.prime_cache(&self.e, toks, &mut c)?;
        let mut t = argmax(&logits) as u32;
        let (_, v1, _, v2) = top2(&logits);
        let mut margins = vec![v1 - v2];
        let mut stream = vec![t];
        for _ in 0..steps {
            let (l, _) = self.model.decode_step_h(&self.e, t, &mut c)?;
            t = argmax(&l) as u32;
            let (_, v1, _, v2) = top2(&l);
            margins.push(v1 - v2);
            stream.push(t);
        }
        Ok((stream, margins, logits))
    }
}

fn load(path: &str) -> Result<Ctx, Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let g = GgufFile::open(path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
    eprintln!("loaded {} ({} layers)", g.arch().unwrap_or("?"), model.layers.len());
    Ok(Ctx { e, model, tok, ctx_len: 2048 })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: concat-prime-probe <model.gguf> <mode> [opts]");
    let mode = args.next().expect("mode: repro|posdiff|content|margins");
    let rest: Vec<String> = args.collect();
    let chat = rest.iter().any(|a| a == "--chat");
    let cx = load(&path)?;

    match mode.as_str() {
        "repro" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let steps: usize = arg(&rest, "--steps").and_then(|v| v.parse().ok()).unwrap_or(96);
            let slot: usize = arg(&rest, "--slot").and_then(|v| v.parse().ok()).unwrap_or(0);
            let ta = encode_prompt(&cx.tok, &pa, chat);
            // co-arrivals: --co-file (one prompt per line) or --prompt-b (single)
            let co_texts: Vec<String> = if let Some(cf) = arg(&rest, "--co-file") {
                std::fs::read_to_string(&cf)?
                    .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
            } else {
                vec![arg(&rest, "--prompt-b").expect("--prompt-b or --co-file")]
            };
            let co_toks: Vec<Vec<u32>> = co_texts.iter()
                .map(|t| encode_prompt(&cx.tok, t, chat)).collect();
            let b = co_toks.len() + 1;
            assert!(slot < b, "--slot must be < batch size {b}");
            println!("repro: T_a={} b={b} slot={slot} co_T={:?} steps={steps} chat={chat}",
                     ta.len(), co_toks.iter().map(|t| t.len()).collect::<Vec<_>>());

            // solo reference for A
            let (stream_solo, _m_solo, logits_solo) = cx.solo_stream(&ta, steps)?;

            // concat prime with A at `slot`; decode A's cache greedily, lockstep vs solo
            let mut batch_toks: Vec<&[u32]> = co_toks.iter().map(|t| t.as_slice()).collect();
            batch_toks.insert(slot, &ta);
            let mut caches: Vec<Cache> = (0..b)
                .map(|_| Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len))
                .collect::<Result<_, _>>()?;
            let logits_batch = {
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let mut outs = cx.model.prime_cache_batch(&cx.e, &batch_toks, &mut cache_refs)?;
                outs.remove(slot).0
            };
            let mut ca = caches.remove(slot);
            let (s1, sv1, _, sv2) = top2(&logits_solo);
            let (b1, bv1, _, bv2) = top2(&logits_batch);
            println!("prefill: solo argmax={s1} margin={:.6}  batch argmax={b1} margin={:.6}  \
                      logit maxdiff={:.6e}  {}",
                     sv1 - sv2, bv1 - bv2, maxdiff(&logits_solo, &logits_batch),
                     if s1 == b1 { "MATCH" } else { "ARGMAX FLIP" });

            let mut t_batch = b1 as u32;
            let mut stream_batch = vec![t_batch];
            let mut first_div: Option<usize> = None;
            if stream_solo[0] != t_batch { first_div = Some(0); }
            let mut prev_solo_logits = logits_solo.clone();
            let mut prev_batch_logits = logits_batch.clone();
            // replay solo stream against a re-primed solo cache in lockstep with the batch
            // cache so per-step logit maxdiff is observable until divergence.
            let mut c_solo = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let _ = cx.model.prime_cache(&cx.e, &ta, &mut c_solo)?;
            let mut t_solo = stream_solo[0];
            for step in 1..=steps {
                let (ls, _) = cx.model.decode_step_h(&cx.e, t_solo, &mut c_solo)?;
                let (lb, _) = cx.model.decode_step_h(&cx.e, t_batch, &mut ca)?;
                let ns = argmax(&ls) as u32;
                let nb = argmax(&lb) as u32;
                if first_div.is_none() {
                    let md = maxdiff(&ls, &lb);
                    if ns != nb {
                        let (_, sv1, _, sv2) = top2(&ls);
                        let (_, bv1, _, bv2) = top2(&lb);
                        println!("FIRST DIVERGENCE at decode step {step}: solo tok={ns} \
                                  (margin {:.6}) batch tok={nb} (margin {:.6}) logit maxdiff={:.6e}",
                                 sv1 - sv2, bv1 - bv2, md);
                        first_div = Some(step);
                    } else if step <= 8 || step % 16 == 0 {
                        let (_, v1, _, v2) = top2(&ls);
                        println!("  step {step}: agree tok={ns} solo-margin={:.6} maxdiff={:.6e}",
                                 v1 - v2, md);
                    }
                }
                t_solo = ns;
                t_batch = nb;
                stream_batch.push(nb);
                prev_solo_logits = ls;
                prev_batch_logits = lb;
            }
            let _ = (prev_solo_logits, prev_batch_logits);
            match first_div {
                Some(0) => println!("verdict: DIVERGED at prefill argmax (step 0)"),
                Some(s) => println!("verdict: DIVERGED at decode step {s}"),
                None => println!("verdict: streams MATCH for {steps} steps"),
            }
            println!("solo : {}", cx.tok.decode(&stream_solo));
            println!("batch: {}", cx.tok.decode(&stream_batch));
        }

        "posdiff" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let pb = arg(&rest, "--prompt-b").expect("--prompt-b");
            let order = arg(&rest, "--order").unwrap_or_else(|| "ab".into());
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let tb = encode_prompt(&cx.tok, &pb, chat);
            let n_embd = cx.model.cfg.n_embd as usize;
            let eps = cx.model.cfg.rms_eps;
            println!("posdiff: T_a={} T_b={} order={order} chat={chat}", ta.len(), tb.len());

            // solo hidden stack for A (pre-output-norm [T, n_embd])
            let mut c = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let (_, _, hid_solo) = cx.model.prime_cache(&cx.e, &ta, &mut c)?;
            let h_solo = cx.e.dtoh(&hid_solo)?;

            // concat hidden stack for A
            let mut c1 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let mut c2 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let (prompts, a_idx): (Vec<&[u32]>, usize) = match order.as_str() {
                "ba" => (vec![&tb, &ta], 1),
                _ => (vec![&ta, &tb], 0),
            };
            let mut caches: Vec<&mut Cache> = vec![&mut c1, &mut c2];
            let mut outs = cx.model.prime_cache_batch(&cx.e, &prompts, &mut caches)?;
            let (_, _, hid_batch) = outs.remove(a_idx);
            let h_batch = cx.e.dtoh(&hid_batch)?;
            assert_eq!(h_solo.len(), ta.len() * n_embd);
            assert_eq!(h_batch.len(), ta.len() * n_embd);

            // identical m=1 epilogue on BOTH sides: rms_norm row + lm_head matvec
            let logits_row = |host: &[f32], p: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let row = &host[p * n_embd..(p + 1) * n_embd];
                let d = cx.e.htod(row)?;
                let mut hn = cx.e.uninit(n_embd)?;
                cx.e.rms_norm(&d, cx.model.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
                Ok(cx.e.dtoh(&cx.e.matmul(&cx.model.output, &hn, 1)?)?)
            };
            println!("pos | hid_maxdiff | hid_relrms | argmax s/b | margin_solo | logit_maxdiff");
            let mut flips = 0usize;
            for p in 0..ta.len() {
                let rs = &h_solo[p * n_embd..(p + 1) * n_embd];
                let rb = &h_batch[p * n_embd..(p + 1) * n_embd];
                let md = maxdiff(rs, rb);
                let (mut se, mut de) = (0f64, 0f64);
                for (x, y) in rs.iter().zip(rb) {
                    se += ((x - y) as f64).powi(2);
                    de += (*x as f64).powi(2);
                }
                let relrms = (se / de.max(1e-30)).sqrt();
                let ls = logits_row(&h_solo, p)?;
                let lb = logits_row(&h_batch, p)?;
                let (s1, sv1, _, sv2) = top2(&ls);
                let (b1, _, _, _) = top2(&lb);
                let flip = s1 != b1;
                if flip { flips += 1; }
                println!("{p:4} | {md:.6e} | {relrms:.6e} | {s1}/{b1}{} | {:.6} | {:.6e}",
                         if flip { " FLIP" } else { "" }, sv1 - sv2, maxdiff(&ls, &lb));
            }
            println!("posdiff summary: {}/{} per-position argmax flips", flips, ta.len());
        }

        "content" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let pb = arg(&rest, "--prompt-b").expect("--prompt-b");
            let pc = arg(&rest, "--prompt-c").expect("--prompt-c");
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let mut tb = encode_prompt(&cx.tok, &pb, chat);
            let mut tc = encode_prompt(&cx.tok, &pc, chat);
            let l = tb.len().min(tc.len());
            assert!(l >= 16, "co-prompts must be >= 16 tokens after truncation");
            tb.truncate(l);
            tc.truncate(l);
            println!("content: T_a={} T_co={} chat={chat}", ta.len(), l);

            let run = |first: &[u32], second: &[u32], want: usize|
                       -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
                let mut c1 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let mut c2 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let prompts: Vec<&[u32]> = vec![first, second];
                let mut caches: Vec<&mut Cache> = vec![&mut c1, &mut c2];
                let mut outs = cx.model.prime_cache_batch(&cx.e, &prompts, &mut caches)?;
                let (logits, _, hid) = outs.remove(want);
                Ok((logits, cx.e.dtoh(&hid)?))
            };
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let verdict = |name: &str, x: (&[f32], &[f32]), y: (&[f32], &[f32])| {
                let li = bits(x.0) == bits(y.0);
                let hi = bits(x.1) == bits(y.1);
                println!("{name}: logits {} (maxdiff {:.6e}), hidden {} (maxdiff {:.6e})",
                         if li { "BIT-IDENTICAL" } else { "DIFFER" }, maxdiff(x.0, y.0),
                         if hi { "BIT-IDENTICAL" } else { "DIFFER" }, maxdiff(x.1, y.1));
            };

            let (l1, h1) = run(&ta, &tb, 0)?;   // [A,B] -> A
            let (l1r, h1r) = run(&ta, &tb, 0)?; // determinism
            let (l2, h2) = run(&ta, &tc, 0)?;   // [A,C] -> A
            verdict("determinism [A,B] x2      ", (&l1, &h1), (&l1r, &h1r));
            verdict("content [A,B] vs [A,C] -> A", (&l1, &h1), (&l2, &h2));
            let (l3, h3) = run(&tb, &ta, 1)?;   // [B,A] -> A (offset)
            let (l4, h4) = run(&tc, &ta, 1)?;   // [C,A] -> A
            verdict("content [B,A] vs [C,A] -> A", (&l3, &h3), (&l4, &h4));
        }

        "margins" => {
            let pf = arg(&rest, "--prompts-file").expect("--prompts-file");
            let steps: usize = arg(&rest, "--steps").and_then(|v| v.parse().ok()).unwrap_or(96);
            let jsonl = arg(&rest, "--jsonl");
            let prompts: Vec<String> = std::fs::read_to_string(&pf)?
                .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
            let mut out = jsonl.as_ref().map(|p| std::fs::File::create(p)).transpose()?;
            let mut all: Vec<f32> = Vec::new();
            for (i, p) in prompts.iter().enumerate() {
                let toks = encode_prompt(&cx.tok, p, chat);
                let (_, margins, _) = cx.solo_stream(&toks, steps)?;
                let mut sorted = margins.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let pn = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
                println!("prompt {i:2}: prefill_margin={:.6} min={:.6} p10={:.6} p50={:.6} (T={} steps={})",
                         margins[0], sorted[0], pn(0.10), pn(0.50), toks.len(), steps);
                if let Some(f) = out.as_mut() {
                    use std::io::Write as _;
                    let ms: Vec<String> = margins.iter().map(|m| format!("{m:.6}")).collect();
                    writeln!(f, "{{\"i\":{i},\"t\":{},\"prefill_margin\":{:.6},\"min\":{:.6},\"p10\":{:.6},\"p50\":{:.6},\"margins\":[{}]}}",
                             toks.len(), margins[0], sorted[0], pn(0.10), pn(0.50), ms.join(","))?;
                }
                all.extend_from_slice(&margins);
            }
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pn = |q: f64| all[((all.len() - 1) as f64 * q) as usize];
            println!("ALL ({} margins): min={:.6} p1={:.6} p5={:.6} p10={:.6} p50={:.6}",
                     all.len(), all[0], pn(0.01), pn(0.05), pn(0.10), pn(0.50));
        }

        // (b)-vs-(a) RAZOR: shape-vs-content dependence of A's concat prime outputs.
        //   r1 b=1 batch vs solo            : the batch code path at m=T_a (no concat)
        //   r2 [A,A] slot0 vs slot1         : OFFSET invariance at identical content/shape
        //   r3 [A,B] vs [A,C], len(B)==len(C): CO-BATCH CONTENT dependence at fixed shapes
        //   r4 [A,B] x2                     : determinism
        //   r5 [A,B] vs [A,B'] len(B')!=len(B): SHAPE dependence (the (a) knob)
        // A defect (b) fails r2 or r3; the FP class (a) fails only r5 (and r1's m change).
        "razor" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let pb = arg(&rest, "--prompt-b").expect("--prompt-b");
            let pc = arg(&rest, "--prompt-c").expect("--prompt-c");
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let mut tb = encode_prompt(&cx.tok, &pb, chat);
            let mut tc = encode_prompt(&cx.tok, &pc, chat);
            let l = tb.len().min(tc.len());
            assert!(l >= 16, "co-prompts must be >= 16 tokens after truncation");
            tb.truncate(l);
            tc.truncate(l);
            let tb2: Vec<u32> = tb[..l - 1].to_vec();   // same content, T-1 (shape knob)
            println!("razor: T_a={} T_co={} chat={chat}", ta.len(), l);

            // returns (logits, hidden-stack) for the sequence at `want`
            let batch = |seqs: &[&[u32]], want: usize|
                         -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
                let mut cs: Vec<Cache> = (0..seqs.len())
                    .map(|_| Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len))
                    .collect::<Result<_, _>>()?;
                let mut refs: Vec<&mut Cache> = cs.iter_mut().collect();
                let mut outs = cx.model.prime_cache_batch(&cx.e, seqs, &mut refs)?;
                let (logits, _, hid) = outs.remove(want);
                Ok((logits, cx.e.dtoh(&hid)?))
            };
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let mut defect = 0usize;
            let mut cmp = |name: &str, x: &(Vec<f32>, Vec<f32>), y: &(Vec<f32>, Vec<f32>),
                           must_be_exact: bool| {
                let li = bits(&x.0) == bits(&y.0);
                let hi = x.1.len() == y.1.len() && bits(&x.1) == bits(&y.1);
                let (a1, ..) = top2(&x.0);
                let (b1, ..) = top2(&y.0);
                let tag = if li && hi { "BIT-IDENTICAL" }
                          else if must_be_exact { defect += 1; "*** DIFFER (DEFECT) ***" }
                          else { "DIFFER (expected: numeric config change)" };
                println!("{name}: {tag}  logit_maxdiff={:.6e} hid_maxdiff={:.6e} argmax {a1} vs {b1}{}",
                         maxdiff(&x.0, &y.0),
                         if x.1.len() == y.1.len() { maxdiff(&x.1, &y.1) } else { f32::NAN },
                         if a1 == b1 { "" } else { " FLIP" });
            };

            let mut c_solo = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let (l_solo, _, h_solo) = cx.model.prime_cache(&cx.e, &ta, &mut c_solo)?;
            let solo = (l_solo, cx.e.dtoh(&h_solo)?);
            let b1 = batch(&[&ta], 0)?;
            let ab_a = batch(&[&ta, &tb], 0)?;
            let ab_a_rep = batch(&[&ta, &tb], 0)?;
            let ac_a = batch(&[&ta, &tc], 0)?;
            let aa_s0 = batch(&[&ta, &ta], 0)?;
            let aa_s1 = batch(&[&ta, &ta], 1)?;
            let ab2_a = batch(&[&ta, &tb2], 0)?;
            let ba_a = batch(&[&tb, &ta], 1)?;
            let ca_a = batch(&[&tc, &ta], 1)?;

            cmp("r4 determinism   [A,B] x2      ", &ab_a, &ab_a_rep, true);
            cmp("r2 offset        [A,A] s0 vs s1", &aa_s0, &aa_s1, true);
            cmp("r3 co-content    [A,B] vs [A,C]", &ab_a, &ac_a, true);
            cmp("r3b co-content   [B,A] vs [C,A]", &ba_a, &ca_a, true);
            cmp("r5 co-SHAPE      [A,B] vs [A,B-1]", &ab_a, &ab2_a, false);
            cmp("r1 batch-path    solo vs b=1   ", &solo, &b1, false);
            cmp("r6 concat        solo vs [A,B] ", &solo, &ab_a, false);
            println!("razor verdict: {}",
                     if defect == 0 { "NO DEFECT — outputs depend on SHAPES only, not co-batch content or offset" }
                     else { "*** DEFECT: content/offset/determinism dependence found ***" });
        }

        // B-SWEEP + per-B invariance razors. Co-arrivals are truncated to a COMMON length so
        // every variant at a given B has an IDENTICAL shape multiset; only content/offset move.
        //   perm : [A, co...] vs [A, reverse(co)...]   -> co-batch CONTENT invariance
        //   tail : [A, co...] vs [co..., A]            -> A's OFFSET invariance
        // Any DIFFER in perm/tail = defect (b). DIFFER only vs solo, growing with total m,
        // with perm/tail exact = the m-dependent concat-GEMM FP class (a).
        "sweep" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let cf = arg(&rest, "--co-file").expect("--co-file");
            let bmax: usize = arg(&rest, "--bmax").and_then(|v| v.parse().ok()).unwrap_or(6);
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let co_all: Vec<Vec<u32>> = std::fs::read_to_string(&cf)?
                .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty())
                .map(|t| encode_prompt(&cx.tok, &t, chat)).collect();
            let lmin = co_all.iter().map(|t| t.len()).min().unwrap();
            let co: Vec<Vec<u32>> = co_all.iter().map(|t| t[..lmin].to_vec()).collect();
            println!("sweep: T_a={} co_n={} co_T={lmin} bmax={bmax} chat={chat}",
                     ta.len(), co.len(), );

            let mut c_solo = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let (l_solo, _, _) = cx.model.prime_cache(&cx.e, &ta, &mut c_solo)?;
            let (a_solo, sv1, _, sv2) = top2(&l_solo);
            println!("solo: argmax={a_solo} margin={:.6}", sv1 - sv2);

            let batch = |seqs: &[&[u32]], want: usize|
                         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let mut cs: Vec<Cache> = (0..seqs.len())
                    .map(|_| Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len))
                    .collect::<Result<_, _>>()?;
                let mut refs: Vec<&mut Cache> = cs.iter_mut().collect();
                let mut outs = cx.model.prime_cache_batch(&cx.e, seqs, &mut refs)?;
                Ok(outs.remove(want).0)
            };
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let mut defects = 0usize;
            println!(" B | total_m | argmax | flip | margin  | maxdiff_vs_solo | perm | tail");
            for b in 2..=bmax.min(co.len() + 1) {
                let cu: Vec<&[u32]> = co[..b - 1].iter().map(|t| t.as_slice()).collect();
                let total = ta.len() + cu.iter().map(|s| s.len()).sum::<usize>();
                let mut v1: Vec<&[u32]> = vec![&ta]; v1.extend(cu.iter().copied());
                let mut v2: Vec<&[u32]> = vec![&ta];
                v2.extend(cu.iter().rev().copied());
                let mut v3: Vec<&[u32]> = cu.iter().copied().collect(); v3.push(&ta);
                let o1 = batch(&v1, 0)?;
                let o2 = batch(&v2, 0)?;
                let o3 = batch(&v3, b - 1)?;
                let perm_ok = bits(&o1) == bits(&o2);
                let tail_ok = bits(&o1) == bits(&o3);
                if !perm_ok || !tail_ok { defects += 1; }
                let (a1, v1t, _, v2t) = top2(&o1);
                println!("{b:2} | {total:7} | {a1:6} | {:4} | {:.6} | {:.9e} | {} | {}",
                         if a1 == a_solo { "-" } else { "YES" }, v1t - v2t,
                         maxdiff(&l_solo, &o1),
                         if perm_ok { "EXACT" } else { "DIFFER(defect)" },
                         if tail_ok { "EXACT" } else { "DIFFER(defect)" });
            }
            println!("sweep verdict: {}",
                     if defects == 0 { "content/offset INVARIANT at every B (no indexing defect); \
                                        solo-vs-concat differences are shape/m-driven" }
                     else { "*** DEFECT: content or offset dependence ***" });
        }

        // m-BISECT at FIXED B=2: only the CO-SEQUENCE LENGTH moves, so b, dispatch arms and
        // A's own content/offset are constant — every difference is a function of total m
        // (= T_a + L). Locates the exact m where the trunk's GEMM reduction shape changes.
        "mscan" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let pb = arg(&rest, "--prompt-b").expect("--prompt-b");
            let lmin: usize = arg(&rest, "--lmin").and_then(|v| v.parse().ok()).unwrap_or(16);
            let lmax: usize = arg(&rest, "--lmax").and_then(|v| v.parse().ok()).unwrap_or(80);
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let tb_full = encode_prompt(&cx.tok, &pb, chat);
            let pad = arg(&rest, "--pad-token").and_then(|v| v.parse::<u32>().ok());
            let mut tb = tb_full.clone();
            if let Some(p) = pad {
                while tb.len() < lmax { tb.push(p); }
            }
            assert!(tb.len() >= lmax, "co prompt too short ({}) for --lmax {lmax}; use --pad-token", tb.len());
            let mut c_solo = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
            let (l_solo, _, _) = cx.model.prime_cache(&cx.e, &ta, &mut c_solo)?;
            let (a_solo, sv1, _, sv2) = top2(&l_solo);
            println!("mscan: T_a={} co_max={} L={lmin}..{lmax} solo_argmax={a_solo} solo_margin={:.6}",
                     ta.len(), tb.len(), sv1 - sv2);
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let sb = bits(&l_solo);
            let mut prev: Option<Vec<u32>> = None;
            // --desc: descending L. If the threshold sits at the SAME total_m in both
            // directions it is m-driven; if it moves with iteration count it is evolving
            // process state (SLRU residency / scratch growth), not the concat shape.
            let ls: Vec<usize> = if rest.iter().any(|a| a == "--desc") {
                (lmin..=lmax).rev().collect()
            } else {
                (lmin..=lmax).collect()
            };
            println!("  L | total_m | argmax | exact_vs_solo | maxdiff_vs_solo | vs_prev_L");
            for l in ls {
                let co = &tb[..l];
                let mut c1 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let mut c2 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let logits = {
                    let seqs: Vec<&[u32]> = vec![&ta, co];
                    let mut refs: Vec<&mut Cache> = vec![&mut c1, &mut c2];
                    cx.model.prime_cache_batch(&cx.e, &seqs, &mut refs)?.remove(0).0
                };
                let (a1, ..) = top2(&logits);
                let cur = bits(&logits);
                let vp = match &prev {
                    None => "-".to_string(),
                    Some(p) => if *p == cur { "same".into() } else { "CHANGED".to_string() },
                };
                println!("{l:3} | {:7} | {a1:6} | {} | {:.9e} | {vp}", ta.len() + l,
                         if cur == sb { "EXACT" } else { "differs" },
                         maxdiff(&l_solo, &logits));
                prev = Some(cur);
            }
        }

        // KERNEL-LEVEL razor: the SAME activation rows are fed to matmul at m = T_a (solo
        // shape) and as the first T_a rows of a taller m = T_a + L batch. If rows [0,T_a)
        // of the tall GEMM differ from the m=T_a GEMM, the prefill GEMM is m-dependent —
        // the concat prime's divergence is inherited from the GEMM, not from the batching
        // logic. Weight = layer 0's wq (the first GEMM every prime executes).
        "gemm" => {
            let lmin: usize = arg(&rest, "--lmin").and_then(|v| v.parse().ok()).unwrap_or(16);
            let lmax: usize = arg(&rest, "--lmax").and_then(|v| v.parse().ok()).unwrap_or(80);
            let ta: usize = arg(&rest, "--ta").and_then(|v| v.parse().ok()).unwrap_or(19);
            let n_embd = cx.model.cfg.n_embd as usize;
            // --weight router|head|wq : which prefill GEMM to probe. `router` = the MoE
            // ffn_gate_inp (F32 -> cuBLASLt, the arm hybrid_forward.rs:2100 documents as
            // n-DEPENDENT); head/wq are quantized weights on the MMQ/f16 lanes.
            let which_w = arg(&rest, "--weight").unwrap_or_else(|| "head".into());
            let mut il_probe = 0usize;
            let w = match which_w.as_str() {
                "router" => {
                    let mut found = None;
                    for (i, layer) in cx.model.layers.iter().enumerate() {
                        if let memra_engine::hybrid::Ffn::Moe(m) = &layer.ffn {
                            found = Some(&m.gate_inp);
                            il_probe = i;
                            break;
                        }
                    }
                    found.expect("no MoE layer (router probe needs an MoE model)")
                }
                // first FULL-attn layer's wq (hybrid stacks put Linear mixers at layer 0)
                "wq" => {
                    let mut found = None;
                    for (i, layer) in cx.model.layers.iter().enumerate() {
                        if let memra_engine::hybrid::Mixer::Full(fa) = &layer.mixer {
                            found = Some(&fa.wq);
                            il_probe = i;
                            break;
                        }
                    }
                    found.expect("no full-attn layer")
                }
                // first Linear (GDN) mixer's fused qkv projection
                "wqkv" => {
                    let mut found = None;
                    for (i, layer) in cx.model.layers.iter().enumerate() {
                        if let memra_engine::hybrid::Mixer::Linear(la) = &layer.mixer {
                            found = Some(&la.wqkv);
                            il_probe = i;
                            break;
                        }
                    }
                    found.expect("no linear-attn layer")
                }
                // shared-expert FFN gate (the MoE layer's dense side)
                "shexp" => {
                    let mut found = None;
                    for (i, layer) in cx.model.layers.iter().enumerate() {
                        if let memra_engine::hybrid::Ffn::Moe(mm) = &layer.ffn {
                            if let Some(g) = mm.gate_shexp.as_ref() {
                                found = Some(g);
                                il_probe = i;
                                break;
                            }
                        }
                    }
                    found.expect("no shared-expert gate")
                }
                _ => &cx.model.output,
            };
            println!("gemm probe weight={which_w} (il={il_probe})");
            let out_f = w.out_features();
            // deterministic pseudo-random activations
            let tot = (ta + lmax) * n_embd;
            let mut xs = Vec::with_capacity(tot);
            let mut s = 0x2545F4914F6CDD1Du64;
            for _ in 0..tot {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                xs.push(((s >> 40) as f32 / 8192.0) - 1.5);
            }
            let xd = cx.e.htod(&xs)?;
            println!("gemm razor: weight out_f={out_f} in_f={n_embd} base_m={ta} L={lmin}..{lmax}");
            println!("  m | rows[0,{ta}) vs m={ta} | maxdiff");
            // --gemv: probe the in-house router GEMV instead of matmul (the candidate
            // m-INVARIANT replacement: one block per (expert,row), fixed per-row FP order).
            let gemv = rest.iter().any(|a| a == "--gemv");
            let run = |m: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                if gemv {
                    let y = cx.e.router_gemv(w.float_data(), &xd, n_embd, out_f, m)?;
                    cx.e.dtoh(&y).map_err(Into::into)
                } else {
                    cx.e.dtoh(&cx.e.matmul(w, &xd, m)?).map_err(Into::into)
                }
            };
            let base = run(ta)?;
            let mut first_change = None;
            for l in lmin..=lmax {
                let m = ta + l;
                let y = run(m)?;
                let head = &y[..ta * out_f];
                let same = head.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits());
                let md = maxdiff(head, &base);
                if !same && first_change.is_none() { first_change = Some(m); }
                println!("{m:4} | {} | {md:.6e}", if same { "BIT-IDENTICAL" } else { "DIFFER" });
            }
            match first_change {
                Some(m) => println!("gemm verdict: prefill GEMM is m-DEPENDENT (first change at m={m}) \
                                     — existing rows' values move when the batch grows"),
                None => println!("gemm verdict: prefill GEMM rows are m-INVARIANT over this range"),
            }
        }

        // ROUTE mode: prime ONE configuration and exit, so an external MEMRA_MOE_TRACE /
        // MEMRA_MOE_WEIGHT_TRACE file captures exactly that prime's router selections.
        //   --which solo             : single prime of A            (rows = A's tokens)
        //   --which batch --colen L  : concat prime [A, co[..L]]    (rows [0,T_a) = A's tokens)
        // Comparing A's rows across the two traces shows whether the concat changes MoE
        // expert SELECTION for A's own tokens (a top-k discontinuity), vs only weights.
        "route" => {
            let pa = arg(&rest, "--prompt-a").expect("--prompt-a");
            let pb = arg(&rest, "--prompt-b").unwrap_or_default();
            let which = arg(&rest, "--which").unwrap_or_else(|| "solo".into());
            let colen: usize = arg(&rest, "--colen").and_then(|v| v.parse().ok()).unwrap_or(56);
            let ta = encode_prompt(&cx.tok, &pa, chat);
            println!("route: which={which} T_a={} colen={colen}", ta.len());
            if which == "solo" {
                let mut c = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let (l, _, _) = cx.model.prime_cache(&cx.e, &ta, &mut c)?;
                let (a1, v1, _, v2) = top2(&l);
                println!("solo argmax={a1} margin={:.6}", v1 - v2);
            } else {
                let tb = encode_prompt(&cx.tok, &pb, chat);
                assert!(tb.len() >= colen, "co prompt shorter than --colen");
                let co = &tb[..colen];
                let mut c1 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let mut c2 = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len)?;
                let seqs: Vec<&[u32]> = vec![&ta, co];
                let mut refs: Vec<&mut Cache> = vec![&mut c1, &mut c2];
                let l = cx.model.prime_cache_batch(&cx.e, &seqs, &mut refs)?.remove(0).0;
                let (a1, v1, _, v2) = top2(&l);
                println!("batch argmax={a1} margin={:.6} (total_m={})", v1 - v2, ta.len() + colen);
            }
        }

        // ALL-WEIGHTS m-invariance census: for every distinct GEMM weight in layer `--il`
        // (plus the lm_head and the router), feed identical activation rows at m=m0 and m=m1
        // and report whether rows [0,m0) move. Names every m-DEPENDENT GEMM in the trunk.
        "allw" => {
            let m0: usize = arg(&rest, "--m0").and_then(|v| v.parse().ok()).unwrap_or(74);
            let m1: usize = arg(&rest, "--m1").and_then(|v| v.parse().ok()).unwrap_or(75);
            let base_rows: usize = arg(&rest, "--rows").and_then(|v| v.parse().ok()).unwrap_or(19);
            let nl: usize = arg(&rest, "--layers").and_then(|v| v.parse().ok()).unwrap_or(4);
            println!("allw: m0={m0} m1={m1} compare rows[0,{base_rows}) over first {nl} layers");
            let mut rng = 0x9E3779B97F4A7C15u64;
            let mut probe = |name: &str, w: &memra_engine::model::GpuTensor|
                             -> Result<(), Box<dyn std::error::Error>> {
                let in_f = w.in_features();
                let out_f = w.out_features();
                let mut xs = Vec::with_capacity(m1 * in_f);
                for _ in 0..m1 * in_f {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    xs.push(((rng >> 40) as f32 / 8192.0) - 1.5);
                }
                let xd = cx.e.htod(&xs)?;
                let y0 = cx.e.dtoh(&cx.e.matmul(w, &xd, m0)?)?;
                let y1 = cx.e.dtoh(&cx.e.matmul(w, &xd, m1)?)?;
                let n = base_rows * out_f;
                let same = y0[..n].iter().zip(&y1[..n]).all(|(a, b)| a.to_bits() == b.to_bits());
                println!("{:<28} in={in_f:6} out={out_f:7} {} maxdiff={:.6e}", name,
                         if same { "m-INVARIANT" } else { "*** m-DEPENDENT ***" },
                         maxdiff(&y0[..n], &y1[..n]));
                Ok(())
            };
            probe("lm_head", &cx.model.output)?;
            // The shexp GATE is not a GpuTensor matmul but a raw cuBLASLt `linear` (out_f=1)
            // at prefill / a fused sigmoid-dot at small t — probe both forms explicitly.
            if let Some(memra_engine::hybrid::Ffn::Moe(mm)) =
                cx.model.layers.iter().find_map(|l| match &l.ffn {
                    f @ memra_engine::hybrid::Ffn::Moe(_) => Some(f),
                    _ => None,
                })
            {
                if let Some(gi) = mm.gate_inp_shexp.as_ref() {
                    let in_f = cx.model.cfg.n_embd as usize;
                    let mut xs = Vec::with_capacity(m1 * in_f);
                    let mut r2 = 0xD1B54A32D192ED03u64;
                    for _ in 0..m1 * in_f {
                        r2 ^= r2 << 13; r2 ^= r2 >> 7; r2 ^= r2 << 17;
                        xs.push(((r2 >> 40) as f32 / 8192.0) - 1.5);
                    }
                    let xd = cx.e.htod(&xs)?;
                    let a = cx.e.dtoh(&cx.e.linear(&xd, gi.float_data(), m0, in_f, 1)?)?;
                    let b = cx.e.dtoh(&cx.e.linear(&xd, gi.float_data(), m1, in_f, 1)?)?;
                    let same = a[..base_rows].iter().zip(&b[..base_rows])
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                    println!("{:<28} in={in_f:6} out={:7} {} maxdiff={:.6e}",
                             "shexp_gate linear(cuBLASLt)", 1,
                             if same { "m-INVARIANT" } else { "*** m-DEPENDENT ***" },
                             maxdiff(&a[..base_rows], &b[..base_rows]));
                    let a2 = cx.e.dtoh(&cx.e.sigmoid_dot_rows(&xd, gi.float_data(), in_f, m0)?)?;
                    let b2 = cx.e.dtoh(&cx.e.sigmoid_dot_rows(&xd, gi.float_data(), in_f, m1)?)?;
                    let same2 = a2[..base_rows].iter().zip(&b2[..base_rows])
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                    println!("{:<28} in={in_f:6} out={:7} {} maxdiff={:.6e}",
                             "shexp_gate sigmoid_dot_rows", 1,
                             if same2 { "m-INVARIANT" } else { "*** m-DEPENDENT ***" },
                             maxdiff(&a2[..base_rows], &b2[..base_rows]));
                }
            }
            for (i, layer) in cx.model.layers.iter().enumerate().take(nl) {
                match &layer.mixer {
                    memra_engine::hybrid::Mixer::Full(fa) => {
                        probe(&format!("l{i}.attn.wq"), &fa.wq)?;
                        probe(&format!("l{i}.attn.wk"), &fa.wk)?;
                        probe(&format!("l{i}.attn.wv"), &fa.wv)?;
                        probe(&format!("l{i}.attn.wo"), &fa.wo)?;
                    }
                    memra_engine::hybrid::Mixer::Linear(la) => {
                        probe(&format!("l{i}.gdn.wqkv"), &la.wqkv)?;
                        probe(&format!("l{i}.gdn.wqkv_gate"), &la.wqkv_gate)?;
                        probe(&format!("l{i}.gdn.ssm_beta"), &la.ssm_beta)?;
                        probe(&format!("l{i}.gdn.ssm_alpha"), &la.ssm_alpha)?;
                        probe(&format!("l{i}.gdn.ssm_out"), &la.ssm_out)?;
                    }
                    memra_engine::hybrid::Mixer::Mla(_) => {}
                }
                match &layer.ffn {
                    memra_engine::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                        probe(&format!("l{i}.ffn.gate"), ffn_gate)?;
                        probe(&format!("l{i}.ffn.up"), ffn_up)?;
                        probe(&format!("l{i}.ffn.down"), ffn_down)?;
                    }
                    memra_engine::hybrid::Ffn::Moe(mm) => {
                        probe(&format!("l{i}.moe.router(gate_inp)"), &mm.gate_inp)?;
                        if let Some(g) = mm.gate_shexp.as_ref() {
                            probe(&format!("l{i}.moe.shexp_gate"), g)?;
                        }
                        if let Some(u) = mm.up_shexp.as_ref() {
                            probe(&format!("l{i}.moe.shexp_up"), u)?;
                        }
                        if let Some(d) = mm.down_shexp.as_ref() {
                            probe(&format!("l{i}.moe.shexp_down"), d)?;
                        }
                    }
                }
            }
        }

        m => return Err(format!("unknown mode {m}").into()),
    }
    Ok(())
}
