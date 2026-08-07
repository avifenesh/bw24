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
//!   twpos   <model> twpos --prompt-a <txt|@file> [--chat] [--every N]
//!           SOLO batched-vs-tokenwise prime, per-POSITION logit diff/flip profile
//!           (gap #46 differential — scattered near-tie flips = FP class, boundary or
//!           wide-margin structure = defect).
//!   causal  <model> causal --prompt-a <txt|@file> --suffix <txt|@file> [--chat]
//!           chunk-boundary content razor: prime(P) vs prime(P+S) rows of P must be
//!           BIT-IDENTICAL when the chunk boundary sits at |P|.
//!   chunkinv <model> chunkinv --prompt-a <txt|@file> [--chunks 2048,64,32] [--steps N]
//!           chunk-ORDER invariance: the same prompt primed at several MEMRA_PRIME_CHUNK
//!           values (zero reuse) must give bit-identical prefill logits. Reports the first
//!           diverging hidden-stack ROW so a boundary-localized leak is distinguishable
//!           from a global one. Engine-level twin of the server-side chunk-order-probe.py.
//!   tickinv <model> tickinv --prompt-a <txt|@file> [--budgets 0,1024,256,64] [--steps N]
//!                           [--splits 64,256,512]
//!           the SECOND segmentation axis, one level ABOVE chunkinv. `chunkinv` varies
//!           MEMRA_PRIME_CHUNK *inside one* prime_cache call; serve additionally splits a
//!           prompt across SEVERAL prime_cache CALLS — one per scheduler tick, `take` tokens
//!           each (worker.rs:3555 / :3111, budget = MEMRA_PREFILL_TICK 1024 interactive,
//!           MEMRA_PREFILL_JUDGE/HARVEST 256 dark-lane). Each CALL sees its own cache.pos, so
//!           any per-call quantity (e.g. step35's seq_end arm predicate) can differ between
//!           tick budgets even when every call is internally chunk-invariant. This mode
//!           replicates that loop faithfully — including the tail-merge that keeps the last
//!           chunk >= PRIME_MIN_T — and asserts the resulting logits/hiddens are
//!           budget-independent. budget 0 = one monolithic call (the chunkinv regime).
//!           --splits adds OFF-GRID-RESUME arms (vLLM #51113's second hole, upstream-sweeps
//!           08-07): prime [0,L) then [L,T) as TWO calls — serve's prefix-cache LCP-split
//!           shape, where the first call stops exactly at the snapshot boundary L regardless
//!           of budget (worker.rs prefill_tick bound_rem) and the second call RESUMES at the
//!           unaligned position L. Any LCP in [64, win=512] reproduced the FA-prefix defect
//!           on an interactive request. Rows print as `sp<L>`.

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

/// `--prompt-x` values starting with '@' name a FILE whose whole (multi-line) content is
/// the prompt — the pp512-class probe prompts don't fit on a CLI line.
fn text_arg(rest: &[String], key: &str) -> Option<String> {
    let v = arg(rest, key)?;
    match v.strip_prefix('@') {
        Some(path) => Some(std::fs::read_to_string(path).expect("prompt file unreadable")),
        None => Some(v),
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

        // SOLO batched-vs-tokenwise per-POSITION differential (gap #46, prime-path
        // FP-composition family): the SAME prompt primed (1) tokenwise (decode_step loop,
        // m=1 — the oracle-stream config) and (2) batched (prime_cache, prefill GEMMs).
        // Per position: logit maxdiff + argmax flip + tokenwise margin, both sides through
        // the SAME m=1 epilogue class (decode's rms_norm row + lm_head matvec). A defect
        // shows structured position/boundary-dependent divergence (e.g. jumps at
        // MEMRA_PRIME_CHUNK boundaries); the FP class scatters and flips only near-ties.
        "twpos" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let every: usize = arg(&rest, "--every").and_then(|v| v.parse().ok()).unwrap_or(32);
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let t = ta.len();
            let n_embd = cx.model.cfg.n_embd as usize;
            let eps = cx.model.cfg.rms_eps;
            println!("twpos: T={t} chat={chat} chunk_env={:?}",
                     std::env::var("MEMRA_PRIME_CHUNK").ok());

            // batched prime -> full pre-output-norm hidden stack
            let mut cb = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len.max(t + 8))?;
            let (_, _, hid) = cx.model.prime_cache(&cx.e, &ta, &mut cb)?;
            let h_batch = cx.e.dtoh(&hid)?;
            assert_eq!(h_batch.len(), t * n_embd);
            let logits_row = |host: &[f32], p: usize|
                              -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let d = cx.e.htod(&host[p * n_embd..(p + 1) * n_embd])?;
                let mut hn = cx.e.uninit(n_embd)?;
                cx.e.rms_norm(&d, cx.model.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
                Ok(cx.e.dtoh(&cx.e.matmul(&cx.model.output, &hn, 1)?)?)
            };

            // tokenwise loop, comparing on the fly (position p = logits after token p)
            let mut ct = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len.max(t + 8))?;
            let mut flips: Vec<usize> = Vec::new();
            let (mut md_max, mut md_max_pos) = (0.0f32, 0usize);
            println!(" pos | maxdiff | argmax tw/bp | tw_margin");
            for (p, &tk) in ta.iter().enumerate() {
                let ltw = cx.model.decode_step(&cx.e, tk, &mut ct)?;
                let lbp = logits_row(&h_batch, p)?;
                let md = maxdiff(&ltw, &lbp);
                let (a_tw, v1, _, v2) = top2(&ltw);
                let (a_bp, ..) = top2(&lbp);
                let flip = a_tw != a_bp;
                if flip { flips.push(p); }
                if md > md_max { md_max = md; md_max_pos = p; }
                if flip || p % every == 0 || p + 1 == t {
                    println!("{p:4} | {md:.4e} | {a_tw}/{a_bp}{} | {:.6}",
                             if flip { " FLIP" } else { "" }, v1 - v2);
                }
            }
            println!("twpos summary: {}/{t} argmax flips at positions {:?}",
                     flips.len(), flips);
            println!("twpos summary: max maxdiff {md_max:.4e} at pos {md_max_pos} \
                      (scattered small + near-tie-only flips = FP class; boundary-clustered \
                      or wide-margin flips = structured)");
        }

        // SOLO CONTENT/CAUSALITY razor for the chunked prime: rows of a prefix P must be
        // BIT-IDENTICAL between prime(P) and prime(P+S) when a chunk boundary falls exactly
        // at |P| (chunk 0 processes P at identical m in both runs; S is later content and
        // must be invisible backwards). The monolithic arm (one chunk over P+S) legally
        // DIFFERs (m changes — numeric-config knob, the concat lane's r5 analog).
        // QWEN-STACK ONLY: gemma4_prime ignores MEMRA_PRIME_CHUNK (monolithic v0), so the
        // c1 arm's bit-identity demand does not apply there.
        //   causal <model> causal --prompt-a <txt|@f> --suffix <txt|@f> [--chat]
        "causal" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let ps = text_arg(&rest, "--suffix").expect("--suffix");
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let ts_ = cx.tok.encode(&ps, false);
            assert!(ts_.len() >= 16, "suffix must be >= 16 tokens (chunker merges shorter tails)");
            let t = ta.len();
            let n_embd = cx.model.cfg.n_embd as usize;
            let eps = cx.model.cfg.rms_eps;
            let mut cat = ta.clone();
            cat.extend_from_slice(&ts_);
            println!("causal: T_p={t} T_s={} chat={chat}", ts_.len());

            let prime_hid = |toks: &[u32]| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let mut c = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len.max(toks.len() + 8))?;
                let (_, _, hid) = cx.model.prime_cache(&cx.e, toks, &mut c)?;
                Ok(cx.e.dtoh(&hid)?)
            };
            let logits_row = |host: &[f32], p: usize|
                              -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let d = cx.e.htod(&host[p * n_embd..(p + 1) * n_embd])?;
                let mut hn = cx.e.uninit(n_embd)?;
                cx.e.rms_norm(&d, cx.model.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
                Ok(cx.e.dtoh(&cx.e.matmul(&cx.model.output, &hn, 1)?)?)
            };
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let mut defect = 0usize;
            let mut run_arm = |name: &str, chunk: &str, must_be_exact: bool|
                               -> Result<(), Box<dyn std::error::Error>> {
                unsafe { std::env::set_var("MEMRA_PRIME_CHUNK", chunk); }
                let h_p = prime_hid(&ta)?;
                let h_ps = prime_hid(&cat)?;
                let head = &h_ps[..t * n_embd];
                let hid_same = bits(head) == bits(&h_p);
                let lp = logits_row(&h_p, t - 1)?;
                let lps = logits_row(&h_ps, t - 1)?;
                let log_same = bits(&lp) == bits(&lps);
                let (a1, ..) = top2(&lp);
                let (a2, ..) = top2(&lps);
                let tag = if hid_same && log_same { "BIT-IDENTICAL" }
                          else if must_be_exact { defect += 1; "*** DIFFER (DEFECT) ***" }
                          else { "DIFFER (expected: numeric config change)" };
                println!("{name}: {tag}  hid_maxdiff={:.4e} lastP_logit_maxdiff={:.4e} \
                          argmax {a1} vs {a2}{}",
                         maxdiff(head, &h_p), maxdiff(&lp, &lps),
                         if a1 == a2 { "" } else { " FLIP" });
                Ok(())
            };
            // chunk boundary exactly at |P|: P's rows/KV computed at identical m -> exact.
            run_arm("c1 chunk@|P|  prime(P) vs prime(P+S) rows[0,|P|)", &t.to_string(), true)?;
            // monolithic: P's rows inside an m=|P|+|S| pass — the legal m knob.
            run_arm("c2 monolithic prime(P) vs prime(P+S) rows[0,|P|)", "0", false)?;
            println!("causal verdict: {}",
                     if defect == 0 { "NO DEFECT — later content invisible across chunk \
                                       boundary; only the GEMM m moves rows" }
                     else { "*** DEFECT: suffix content leaked backwards across a chunk \
                             boundary ***" });
        }

        // CHUNK-ORDER INVARIANCE (lane/chunk-invariance, 2026-08-05): the SAME prompt primed
        // at several MEMRA_PRIME_CHUNK values with ZERO reuse. Reports, per chunk value vs the
        // reference: prefill-logit bit-identity, the hidden stack's FIRST diverging position
        // (which localizes the leak to a chunk boundary vs everywhere), argmax flip, and the
        // greedy stream's first diverging step. This is the engine-level twin of
        // research/session-affinity-20260805/chunk-order-probe.py (which needed a live server).
        //   chunkinv <model> chunkinv --prompt-a <txt|@f> [--chunks 2048,64,32] [--steps N] [--chat]
        "chunkinv" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let steps: usize = arg(&rest, "--steps").and_then(|v| v.parse().ok()).unwrap_or(48);
            let chunks: Vec<String> = arg(&rest, "--chunks")
                .unwrap_or_else(|| "2048,64,32".into())
                .split(',').map(|s| s.trim().to_string()).collect();
            let jsonl = arg(&rest, "--jsonl");
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let t = ta.len();
            let n_embd = cx.model.cfg.n_embd as usize;
            println!("chunkinv: T={t} chat={chat} chunks={chunks:?} steps={steps}");

            // one arm = set MEMRA_PRIME_CHUNK, prime cold, greedy-decode `steps`.
            // prime_cache reads the env var per call, so in-process switching is honest.
            let arm = |cv: &str| -> Result<(Vec<f32>, Vec<f32>, Vec<u32>), Box<dyn std::error::Error>> {
                // single-threaded probe main; no other thread reads the environment here.
                unsafe { std::env::set_var("MEMRA_PRIME_CHUNK", cv) };
                let mut c = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len.max(t + steps + 8))?;
                let (logits, _, hid) = cx.model.prime_cache(&cx.e, &ta, &mut c)?;
                let h = cx.e.dtoh(&hid)?;
                let mut tk = argmax(&logits) as u32;
                let mut stream = vec![tk];
                for _ in 0..steps {
                    let (l, _) = cx.model.decode_step_h(&cx.e, tk, &mut c)?;
                    tk = argmax(&l) as u32;
                    stream.push(tk);
                }
                Ok((logits, h, stream))
            };
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let (l_ref, h_ref, s_ref) = arm(&chunks[0])?;
            let (a_ref, r1, _, r2) = top2(&l_ref);
            println!("ref chunk={} argmax={a_ref} margin={:.6}", chunks[0], r1 - r2);
            let mut out = jsonl.as_ref().map(std::fs::File::create).transpose()?;
            let mut defects = 0usize;
            println!("  chunk | logits | first_div_pos | maxdiff   | argmax | stream_div");
            for cv in &chunks[1..] {
                let (l, h, s) = arm(cv)?;
                let log_same = bits(&l) == bits(&l_ref);
                // first diverging ROW of the hidden stack: a boundary-localized leak shows
                // its first divergence at the first chunk boundary, not at row 0.
                let mut first_div: i64 = -1;
                for p in 0..t {
                    let (x, y) = (&h[p * n_embd..(p + 1) * n_embd],
                                  &h_ref[p * n_embd..(p + 1) * n_embd]);
                    if x.iter().zip(y).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        first_div = p as i64;
                        break;
                    }
                }
                let (a, ..) = top2(&l);
                let sd = s.iter().zip(&s_ref).position(|(a, b)| a != b);
                if !log_same { defects += 1; }
                // MECHANISM RAZOR: per-row maxdiff profile. A pure GEMM-m reduction-order
                // effect is a flat small band across ALL rows; a PRECISION-CLASS change at a
                // chunk boundary shows an order-of-magnitude STEP at that boundary (rows past
                // the first boundary read the quantized cache instead of f32 K/V).
                if rest.iter().any(|x| x == "--profile") {
                    let cv_n: usize = cv.parse().unwrap_or(0);
                    let rowmd: Vec<f32> = (0..t).map(|p| {
                        maxdiff(&h[p * n_embd..(p + 1) * n_embd],
                                &h_ref[p * n_embd..(p + 1) * n_embd])
                    }).collect();
                    let pre: f32 = rowmd[..cv_n.min(t)].iter().cloned().fold(0.0, f32::max);
                    let post: f32 = rowmd[cv_n.min(t)..].iter().cloned().fold(0.0, f32::max);
                    println!("   profile chunk={cv}: rows[0,{cv_n}) maxdiff={pre:.3e} | \
                              rows[{cv_n},{t}) maxdiff={post:.3e} | step={:.1}x",
                             if pre > 0.0 { post / pre } else { f32::INFINITY });
                    let buckets: Vec<String> = rowmd.chunks(8)
                        .map(|c| format!("{:.1e}", c.iter().cloned().fold(0.0, f32::max)))
                        .collect();
                    println!("   per-8-row maxdiff: {}", buckets.join(" "));
                }
                println!("{cv:>7} | {} | {:13} | {:.3e} | {} | {}",
                         if log_same { "EXACT" } else { "DIFFER" },
                         first_div, maxdiff(&l, &l_ref),
                         if a == a_ref { "-" } else { "FLIP" },
                         match sd { None => "identical".to_string(), Some(i) => format!("step {i}") });
                if let Some(f) = out.as_mut() {
                    use std::io::Write as _;
                    writeln!(f, "{{\"chunk\":\"{cv}\",\"ref_chunk\":\"{}\",\"T\":{t},\
                                 \"logits_exact\":{log_same},\"first_div_pos\":{first_div},\
                                 \"logit_maxdiff\":{:.6e},\"argmax\":{a},\"argmax_ref\":{a_ref},\
                                 \"stream_div_step\":{}}}",
                             chunks[0], maxdiff(&l, &l_ref),
                             match sd { None => "null".to_string(), Some(i) => i.to_string() })?;
                }
            }
            println!("chunkinv verdict: {}",
                     if defects == 0 { "CHUNK-INVARIANT — prefill logits bit-identical at every \
                                        chunk size" }
                     else { "*** CHUNK-DEPENDENT: prefill logits move with MEMRA_PRIME_CHUNK ***" });
        }

        // TICK-BUDGET INVARIANCE (lane/step35-chunkfix, 2026-08-07): the segmentation axis one
        // level ABOVE chunkinv. chunkinv varies the split INSIDE one prime_cache call; serve also
        // splits a prompt across SEVERAL calls, one per scheduler tick. Each call has its own
        // cache.pos, so a per-CALL quantity is free to differ between budgets even when every
        // call is internally chunk-invariant — which is exactly the shape of the step35 defect,
        // just one level out. This mode exists so that axis has a MEASURED receipt instead of an
        // enumeration argument.
        //   tickinv <model> tickinv --prompt-a <txt|@f> [--budgets 0,1024,256,64] [--steps N]
        //                           [--splits 64,256,512]
        "tickinv" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let steps: usize = arg(&rest, "--steps").and_then(|v| v.parse().ok()).unwrap_or(24);
            let budgets: Vec<usize> = arg(&rest, "--budgets")
                .unwrap_or_else(|| "0,1024,256,64".into())
                .split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let splits: Vec<usize> = arg(&rest, "--splits").unwrap_or_default()
                .split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let ta = encode_prompt(&cx.tok, &pa, chat);
            let t = ta.len();
            let n_embd = cx.model.cfg.n_embd as usize;
            let min_t = memra_engine::hybrid_forward::PRIME_MIN_T;
            println!("tickinv: T={t} chat={chat} budgets={budgets:?} splits={splits:?} \
                      steps={steps} PRIME_MIN_T={min_t} (budget 0 = single monolithic call)");

            // An arm is a SEGMENTATION of [0,T): either the worker's budget loop, or an
            // explicit two-call split at L (the prefix-cache LCP shape — off-grid RESUME,
            // vLLM #51113's second hole: call 2 starts at the unaligned position L).
            enum Seg { Budget(usize), Split(usize) }
            // FAITHFUL replica of the worker's prefill tick loop (worker.rs:3551-3568): take
            // min(queue, budget) per call, and if the remainder would fall below PRIME_MIN_T
            // take the whole rest instead (the tail merge). Each `take` is ONE prime_cache call
            // on the SAME cache, so cache.pos advances across calls exactly as it does in serve.
            let arm = |seg: &Seg| -> Result<(Vec<f32>, Vec<f32>, Vec<u32>, usize), Box<dyn std::error::Error>> {
                let mut c = Cache::new(&cx.e, &cx.model.cfg, cx.ctx_len.max(t + steps + 8))?;
                let mut hid_all: Vec<f32> = Vec::with_capacity(t * n_embd);
                let mut logits = Vec::new();
                let mut fed = 0usize;
                let mut calls = 0usize;
                while fed < t {
                    let q = t - fed;
                    let mut take = match *seg {
                        Seg::Budget(0) => q,
                        Seg::Budget(b) => q.min(b),
                        // LCP split: first call stops EXACTLY at L (worker.rs prefill_tick
                        // bound_rem — the snapshot boundary overrides the budget), the second
                        // call resumes at pos=L and takes the whole rest.
                        Seg::Split(l) => if fed == 0 { l.min(q) } else { q },
                    };
                    if q - take > 0 && q - take < min_t { take = q; }
                    let (l, _, hid) = cx.model.prime_cache(&cx.e, &ta[fed..fed + take], &mut c)?;
                    hid_all.extend_from_slice(&cx.e.dtoh(&hid)?);
                    logits = l;
                    fed += take;
                    calls += 1;
                }
                let mut tk = argmax(&logits) as u32;
                let mut stream = vec![tk];
                for _ in 0..steps {
                    let (l, _) = cx.model.decode_step_h(&cx.e, tk, &mut c)?;
                    tk = argmax(&l) as u32;
                    stream.push(tk);
                }
                Ok((logits, hid_all, stream, calls))
            };
            let bits = |a: &[f32]| -> Vec<u32> { a.iter().map(|v| v.to_bits()).collect() };
            let (l_ref, h_ref, s_ref, c_ref) = arm(&Seg::Budget(budgets[0]))?;
            println!("ref budget={} calls={c_ref} argmax={}", budgets[0], argmax(&l_ref));
            let mut defects = 0usize;
            println!(" budget | calls | logits | first_div_row | maxdiff   | argmax | stream_div");
            let arms: Vec<(String, Seg)> = budgets[1..].iter()
                .map(|&b| (format!("{b}"), Seg::Budget(b)))
                .chain(splits.iter().filter(|&&l| l >= min_t && l + min_t <= t)
                       .map(|&l| (format!("sp{l}"), Seg::Split(l))))
                .collect();
            for (name, seg) in &arms {
                let (l, h, s, calls) = arm(seg)?;
                let log_same = bits(&l) == bits(&l_ref);
                let mut first_div: i64 = -1;
                for p in 0..t.min(h.len() / n_embd).min(h_ref.len() / n_embd) {
                    let (x, y) = (&h[p * n_embd..(p + 1) * n_embd],
                                  &h_ref[p * n_embd..(p + 1) * n_embd]);
                    if x.iter().zip(y).any(|(a, bb)| a.to_bits() != bb.to_bits()) {
                        first_div = p as i64;
                        break;
                    }
                }
                if !log_same { defects += 1; }
                let sd = s.iter().zip(&s_ref).position(|(a, bb)| a != bb);
                println!("{name:>7} | {calls:>5} | {} | {:13} | {:.3e} | {} | {}",
                         if log_same { "EXACT" } else { "DIFFER" },
                         first_div, maxdiff(&l, &l_ref),
                         if argmax(&l) == argmax(&l_ref) { "-" } else { "FLIP" },
                         match sd { None => "identical".to_string(), Some(i) => format!("step {i}") });
            }
            println!("tickinv verdict: {}",
                     if defects == 0 { "TICK-INVARIANT — prefill logits bit-identical at every \
                                        per-tick prefill budget" }
                     else { "*** TICK-DEPENDENT: prefill logits move with the per-tick prefill \
                             budget (MEMRA_PREFILL_TICK / _JUDGE / _HARVEST) ***" });
        }

        // NLL WINDOW THROUGH THE SERVING PRIME (lane/chunkinv-flip, 2026-08-05): mean token
        // NLL over a frozen text window, computed from prime_cache's OWN hidden stack (the
        // pass the grain-free fix changes). forward()/fp8_mmq_stream ride full_attn (fresh
        // f32 prefill) and CANNOT see this change — this mode is the quality instrument for
        // anything that moves prime arithmetic. Env decides the arm (MEMRA_PRIME_F32CHUNK0,
        // MEMRA_PRIME_CHUNK); the mode itself is arm-neutral.
        //   nllwin <model> nllwin --prompt-a <txt|@f> [--window 1024] [--chunk <c>]
        "nllwin" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let window: usize = arg(&rest, "--window").and_then(|v| v.parse().ok()).unwrap_or(1024);
            if let Some(cv) = arg(&rest, "--chunk") {
                unsafe { std::env::set_var("MEMRA_PRIME_CHUNK", &cv) };
            }
            let mut ids = cx.tok.encode(&pa, true);
            ids.truncate(window.max(2));
            let t = ids.len();
            let n_embd = cx.model.cfg.n_embd as usize;
            let n_vocab = cx.model.output.out_features();
            let mut c = Cache::new(&cx.e, &cx.model.cfg, t + 8)?;
            let (_, _, hid) = cx.model.prime_cache(&cx.e, &ids, &mut c)?;
            // hid = [T, n_embd] pre-output-norm hiddens; lm_head each row like forward()
            let mut hn = cx.e.uninit(t * n_embd)?;
            cx.e.rms_norm(&hid, cx.model.output_norm.float_data(), &mut hn, n_embd, t,
                          cx.model.cfg.rms_eps)?;
            let logits = cx.e.matmul(&cx.model.output, &hn, t)?;
            let all = cx.e.dtoh(&logits)?;
            let mut sum = 0.0f64;
            for p in 1..t {
                let row = &all[(p - 1) * n_vocab..p * n_vocab];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + row.iter().map(|&v| ((v as f64) - mx).exp()).sum::<f64>().ln();
                sum += lse - row[ids[p] as usize] as f64;
            }
            let nll = sum / (t - 1) as f64;
            println!("nllwin: tokens={t} chunk={} f32chunk0={} mean_nll={nll:.6} ppl={:.6}",
                     std::env::var("MEMRA_PRIME_CHUNK").unwrap_or_else(|_| "4096(default)".into()),
                     std::env::var("MEMRA_PRIME_F32CHUNK0").unwrap_or_else(|_| "0".into()),
                     nll.exp());
        }

        // TEACHER-FORCED ARM COMPARISON (lane/chunkinv-flip; the mmq-v2 flip protocol):
        // prime the SAME window under the grain-free default and under the legacy seam
        // (MEMRA_PRIME_F32CHUNK0=1), lm_head every row of both hidden stacks, and report the
        // per-position argmax disagreement count + each flip's LEGACY-arm margin against the
        // legacy margin distribution (median/percentile) — near-tie flips sit far below the
        // median. Teacher-forced by construction: every row is conditioned on the true prefix.
        //   tfcmp <model> tfcmp --prompt-a <txt|@f> [--window 1024] [--chunk <c>]
        "tfcmp" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let window: usize = arg(&rest, "--window").and_then(|v| v.parse().ok()).unwrap_or(1024);
            if let Some(cv) = arg(&rest, "--chunk") {
                unsafe { std::env::set_var("MEMRA_PRIME_CHUNK", &cv) };
            }
            let mut ids = cx.tok.encode(&pa, true);
            ids.truncate(window.max(2));
            let t = ids.len();
            let n_embd = cx.model.cfg.n_embd as usize;
            let n_vocab = cx.model.output.out_features();
            let run_arm = |seam: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                unsafe { std::env::set_var("MEMRA_PRIME_F32CHUNK0", seam) };
                let mut c = Cache::new(&cx.e, &cx.model.cfg, t + 8)?;
                let (_, _, hid) = cx.model.prime_cache(&cx.e, &ids, &mut c)?;
                let mut hn = cx.e.uninit(t * n_embd)?;
                cx.e.rms_norm(&hid, cx.model.output_norm.float_data(), &mut hn, n_embd, t,
                              cx.model.cfg.rms_eps)?;
                let logits = cx.e.matmul(&cx.model.output, &hn, t)?;
                Ok(cx.e.dtoh(&logits)?)
            };
            let l_new = run_arm("0")?;      // grain-free default
            let l_old = run_arm("1")?;      // legacy f32-chunk0 arithmetic
            unsafe { std::env::remove_var("MEMRA_PRIME_F32CHUNK0") };
            let mut legacy_margins: Vec<f32> = Vec::with_capacity(t);
            let mut flips: Vec<(usize, f32)> = Vec::new();
            for p in 0..t {
                let ro = &l_old[p * n_vocab..(p + 1) * n_vocab];
                let rn = &l_new[p * n_vocab..(p + 1) * n_vocab];
                let (ao, v1, _, v2) = top2(ro);
                let (an, ..) = top2(rn);
                legacy_margins.push(v1 - v2);
                if ao != an { flips.push((p, v1 - v2)); }
            }
            let mut sorted = legacy_margins.clone();
            sorted.sort_by(f32::total_cmp);
            let med = sorted[sorted.len() / 2];
            println!("tfcmp: window={t} disagreements={} of {t} | legacy margin median={med:.4}",
                     flips.len());
            for (p, m) in &flips {
                let pct = sorted.iter().filter(|&&v| v < *m).count() as f64
                    / sorted.len() as f64 * 100.0;
                println!("  flip @pos {p}: legacy margin {m:.6} = {:.3}x median ({pct:.1}th pctile)",
                         m / med);
            }
        }

        // PRIME-ONLY THROUGHPUT (lane/chunkinv-flip): timed prime_cache reps, fresh cache per
        // rep, median tok/s — the SERVING prefill pass. run-gen's GGUF MEMRA_PP_ONLY times
        // forward_last (fresh f32 attention, prime-dispatch-blind); this mode times the pass
        // the grain-free fix actually changes. Env (MEMRA_PRIME_F32CHUNK0 / MEMRA_PRIME_CHUNK)
        // selects the arm.
        //   ppprime <model> ppprime --prompt-a <txt|@f> [--reps 3] [--warmup 1]
        "ppprime" => {
            let pa = text_arg(&rest, "--prompt-a").expect("--prompt-a");
            let reps: usize = arg(&rest, "--reps").and_then(|v| v.parse().ok()).unwrap_or(3);
            let warmup: usize = arg(&rest, "--warmup").and_then(|v| v.parse().ok()).unwrap_or(1);
            let ids = cx.tok.encode(&pa, true);
            let t = ids.len();
            for _ in 0..warmup {
                let mut c = Cache::new(&cx.e, &cx.model.cfg, t + 8)?;
                let _ = cx.model.prime_cache(&cx.e, &ids, &mut c)?;
            }
            cx.e.stream().synchronize()?;
            let mut times = Vec::with_capacity(reps);
            for r in 0..reps {
                let mut c = Cache::new(&cx.e, &cx.model.cfg, t + 8)?;
                let t0 = std::time::Instant::now();
                let _ = cx.model.prime_cache(&cx.e, &ids, &mut c)?;
                cx.e.stream().synchronize()?;
                let dt = t0.elapsed().as_secs_f64();
                println!("ppprime rep {r}: {t} tok in {dt:.4}s = {:.1} tok/s", t as f64 / dt);
                times.push(dt);
            }
            times.sort_by(f64::total_cmp);
            let med = times[times.len() / 2];
            println!("ppprime MEDIAN: {t} tok in {med:.4}s = {:.1} tok/s (chunk={} f32chunk0={})",
                     t as f64 / med,
                     std::env::var("MEMRA_PRIME_CHUNK").unwrap_or_else(|_| "4096(default)".into()),
                     std::env::var("MEMRA_PRIME_F32CHUNK0").unwrap_or_else(|_| "0".into()));
        }

        m => return Err(format!("unknown mode {m}").into()),
    }
    Ok(())
}
