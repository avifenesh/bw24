//! graph-warmup-stress (lane/graph-warmups, 2026-08-05): the pool-growth adversarial gate
//! the MEMRA_GRAPH_WARMUPS default flip owes.
//!
//! MECHANISM UNDER TEST: `capture_graph` runs the step eagerly `warmups` times before
//! capturing. Warmup 1's allocations may GROW/map the async pool (lazy engine pools —
//! fa_part_pool, argmax partials, scratch — plus per-step transients); warmup 2 re-walks
//! the same alloc/free sequence over the settled pool so the captured run bakes
//! steady-state addresses. Dropping to warmups=1 captures on the run whose pool walk may
//! differ from every later replay's — the #68 stale-baked-address class (a captured graph
//! address returns to the pool, a live alloc lands there, the next replay writes over it:
//! stream corruption, usually WITHOUT a CUDA fault). The arbiter is therefore per-token
//! BIT-IDENTITY vs the eager decode_step stream, plus fault propagation.
//!
//! ADVERSARIAL ARMS (each cycle, both directions):
//!   large->small: a big session (big cache + big graph) boots, generates across several
//!     kernel-class recaptures, is dropped — its buffers return to the async pool as freed
//!     blocks. A small session then boots: its capture warms up OVER the freed blocks.
//!   small->large: a small session boots/retires first, then the large one — the large
//!     session's warmup 1 must GROW the pool (small steady state can't hold it): the exact
//!     "warmup 1 grows/maps" scenario warmup 2 exists to absorb.
//!   overlap (once, after the cycles): two live sessions interleave steps — the large
//!     boot grows the SHARED engine fa_part_pool while the small session's graph holds
//!     baked pointers to the pre-grow buffers (retire-on-grow is the guard under test);
//!     the small session is then dropped mid-flight, and the survivor takes a FORCED
//!     recapture over the freed blocks + keeps generating (the F5-adjacent
//!     capture-over-existing-cache path — worker park/resume promotes through the same
//!     graph_capture_segment).
//!   Every arm also takes one forced mid-stream recapture (capture from a live cache over
//!   a churned pool) in addition to the natural class-boundary recaptures.
//!
//! Default-pool RESERVED/USED bytes are printed at phase boundaries so the receipt shows
//! the pool actually grew and actually held freed blocks (adversarial preconditions met,
//! not assumed).
//!
//! --canary: mid-stream, clobber the session's graph-referenced token_d buffer (the
//! observable surface of the stale-address class: a graph-consumed buffer whose contents a
//! stray write changed) and require the bit-identity check to CATCH it. A real cross-
//! allocation alias cannot be forced deterministically from user code (the allocator
//! exposes no placement control), so the canary corrupts the graph's INPUT memory directly
//! — it proves the comparator + plumbing detect graph-memory corruption end-to-end, not
//! merely that a label flips (the chunkinv-canary trap).
//!
//! usage: graph-warmup-stress <model.gguf> [--cycles 10] [--large-steps 160]
//!                            [--small-steps 90] [--canary]
//! exit 0 = every cycle bit-identical + no fault (canary mode: corruption detected).

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn arg_val(rest: &[String], key: &str) -> Option<String> {
    rest.iter().position(|a| a == key).and_then(|i| rest.get(i + 1)).cloned()
}

/// Default async-pool occupancy (reserved = mapped from the OS, used = live allocations).
/// reserved > used == freed blocks parked in the pool (the adversarial precondition).
fn pool_stats() -> (u64, u64) {
    use cudarc::driver::sys;
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, 0) != sys::CUresult::CUDA_SUCCESS {
            return (0, 0);
        }
        let (mut reserved, mut used) = (0u64, 0u64);
        let _ = sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut core::ffi::c_void,
        );
        let _ = sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            &mut used as *mut u64 as *mut core::ffi::c_void,
        );
        (reserved, used)
    }
}

fn mb(b: u64) -> f64 { b as f64 / (1024.0 * 1024.0) }

/// Ground truth: token-wise eager decode_step stream (same emission convention as the
/// graph gates: token 0 = argmax of the last prime step). Greedy = deterministic.
fn eager_ref(e: &Engine, m: &HybridModel, prompt: &[u32], n: usize)
             -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut cache = Cache::new(e, &m.cfg, prompt.len() + n + 8)?;
    let mut ll = Vec::new();
    for &t in prompt { ll = m.decode_step(e, t, &mut cache)?; }
    let mut tok = argmax(&ll) as u32;
    let mut out = Vec::with_capacity(n);
    out.push(tok);
    for _ in 1..n {
        ll = m.decode_step(e, tok, &mut cache)?;
        tok = argmax(&ll) as u32;
        out.push(tok);
    }
    Ok(out)
}

/// Boot a GraphSession, generate n tokens (forced recapture at `recap_at`, canary clobber
/// at `canary_at`), return the stream. Any Err = propagated CUDA fault = gate FAIL.
fn session_run(e: &Engine, m: &HybridModel, prompt: &[u32], budget: usize, n: usize,
               recap_at: Option<usize>, canary_at: Option<usize>)
               -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let (mut sess, first) = m.graph_session_new(e, prompt, budget)?;
    let mut out = Vec::with_capacity(n);
    out.push(first);
    for i in 1..n {
        if recap_at == Some(i) { m.graph_session_recapture_pub(e, &mut sess)?; }
        if canary_at == Some(i) {
            // stray write into a graph-referenced buffer (see header). 1234 is in-vocab
            // for every gated family and never the greedy continuation here.
            e.set_u32_one(&mut sess.gs.token_d, 1234)?;
        }
        out.push(sess.step(e, m)?);
    }
    Ok(out)
}

/// First mismatch index, if any.
fn diff(a: &[u32], b: &[u32]) -> Option<usize> {
    if a.len() != b.len() { return Some(a.len().min(b.len())); }
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: graph-warmup-stress <model.gguf> [--cycles N] [--canary]");
    let rest: Vec<String> = args.collect();
    let cycles: usize = arg_val(&rest, "--cycles").and_then(|v| v.parse().ok()).unwrap_or(10);
    let large_n: usize = arg_val(&rest, "--large-steps").and_then(|v| v.parse().ok()).unwrap_or(160);
    let small_n: usize = arg_val(&rest, "--small-steps").and_then(|v| v.parse().ok()).unwrap_or(90);
    let canary = rest.iter().any(|a| a == "--canary");

    // Size classes: the small session's budget keeps its cache tiny; the large budget
    // inflates cache/scratch so its boot MUST grow the pool past small steady state.
    let large_budget = 4096usize;
    let small_budget = small_n + 6;

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load_without_mtp(&e, &g)?;
    let prompt: Vec<u32> = (0..48u32).map(|j| 55 + j * 31).collect();
    let warm = std::env::var("MEMRA_GRAPH_WARMUPS").unwrap_or_else(|_| "(default)".into());
    println!("model {path} arch {} ({} layers) cycles={cycles} large={large_n}tok/@{large_budget} \
              small={small_n}tok/@{small_budget} MEMRA_GRAPH_WARMUPS={warm} canary={canary}",
             g.arch().unwrap_or("?"), m.layers.len());
    let (r0, u0) = pool_stats();
    println!("[pool] post-load: reserved {:.0} MB used {:.0} MB", mb(r0), mb(u0));

    // ground truth once per size class (greedy determinism; the serving gates pin it).
    let ref_small = eager_ref(&e, &m, &prompt, small_n)?;
    let ref_large = eager_ref(&e, &m, &prompt, large_n)?;
    let (r1, u1) = pool_stats();
    println!("[pool] post-eager-ref: reserved {:.0} MB used {:.0} MB", mb(r1), mb(u1));

    let mut fails = 0usize;
    let mut canary_caught = false;

    for c in 1..=cycles {
        // direction 1: LARGE boots (pool grows), retires (freed blocks), SMALL captures over them.
        let out_l = session_run(&e, &m, &prompt, large_budget, large_n, Some(large_n / 2), None)?;
        let (ra, ua) = pool_stats();
        let out_s = session_run(&e, &m, &prompt, small_budget, small_n, Some(small_n / 2),
                                if canary && c == 1 { Some(small_n / 3) } else { None })?;
        let (rb, ub) = pool_stats();
        // direction 2: SMALL first, then LARGE — the large warmup 1 must grow the pool.
        let out_s2 = session_run(&e, &m, &prompt, small_budget, small_n, Some(small_n / 2), None)?;
        let out_l2 = session_run(&e, &m, &prompt, large_budget, large_n, Some(large_n / 2), None)?;

        let mut cycle_ok = true;
        for (label, out, r) in [("L->", &out_l, &ref_large), ("->S", &out_s, &ref_small),
                                ("S->", &out_s2, &ref_small), ("->L", &out_l2, &ref_large)] {
            let canary_arm = canary && c == 1 && label == "->S";
            match diff(out, r) {
                None if canary_arm => {
                    println!("cycle {c} {label}: CANARY NOT CAUGHT (corrupted stream still matched — comparator blind)");
                    cycle_ok = false;
                }
                None => {}
                Some(i) if canary_arm => {
                    println!("cycle {c} {label}: canary caught at token {i} (expected — comparator has teeth)");
                    canary_caught = true;
                }
                Some(i) => {
                    println!("cycle {c} {label}: MISMATCH at token {i} (graph {} vs eager {})",
                             out.get(i).copied().unwrap_or(0), r.get(i).copied().unwrap_or(0));
                    cycle_ok = false;
                }
            }
        }
        println!("cycle {c}: {}  [pool after L-drop: {:.0}/{:.0} MB, after S: {:.0}/{:.0} MB rsv/used]",
                 if cycle_ok { "OK" } else { "FAIL" }, mb(ra), mb(ua), mb(rb), mb(ub));
        if !cycle_ok { fails += 1; }
    }

    // OVERLAP arm (once): two live graphs share the engine pools; the large boot grows
    // fa_part_pool under the small session's baked pointers (retire-on-grow under test),
    // the small session dies mid-flight, the survivor recaptures over its freed blocks.
    {
        let (mut sa, fa) = m.graph_session_new(&e, &prompt, small_budget)?;
        let mut out_a = vec![fa];
        for _ in 1..40 { out_a.push(sa.step(&e, &m)?); }
        let (mut sb, fb) = m.graph_session_new(&e, &prompt, large_budget)?;
        let mut out_b = vec![fb];
        for _ in 1..40 { out_b.push(sb.step(&e, &m)?); }
        for _ in 40..small_n { out_a.push(sa.step(&e, &m)?); }   // A replays after B grew the pools
        drop(sa);                                                 // A's cache -> freed blocks under B
        m.graph_session_recapture_pub(&e, &mut sb)?;              // fresh capture over them
        for _ in 40..large_n { out_b.push(sb.step(&e, &m)?); }
        let da = diff(&out_a, &ref_small);
        let db = diff(&out_b, &ref_large);
        if da.is_none() && db.is_none() {
            println!("overlap arm: OK (A survived B's pool growth; B survived A's free + recapture)");
        } else {
            println!("overlap arm: MISMATCH (A diff {da:?}, B diff {db:?})");
            fails += 1;
        }
    }

    let (rz, uz) = pool_stats();
    println!("[pool] end: reserved {:.0} MB used {:.0} MB", mb(rz), mb(uz));

    if canary {
        if canary_caught && fails == 0 {
            println!("CANARY GATE PASS: injected graph-memory corruption was detected; all clean arms held");
            Ok(())
        } else {
            Err(format!("CANARY GATE FAIL: caught={canary_caught} other_fails={fails}").into())
        }
    } else if fails == 0 {
        println!("ALL GREEN: graph-warmup-stress ({cycles} cycles x 4 arms + overlap, bit-identical, no fault)");
        Ok(())
    } else {
        Err(format!("graph-warmup-stress FAILED: {fails} failing cycle(s)").into())
    }
}
