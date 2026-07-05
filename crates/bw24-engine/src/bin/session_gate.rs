//! SESSION EXACTNESS GATE: multi-turn generate_spec_session must produce, at every turn, the
//! IDENTICAL tokens plain greedy generate() produces when primed with the session's full
//! committed history. Verify guarantees per-turn exactness; this gate pins the turn BOUNDARY
//! (suffix continuation prime + cross-turn draft-KV/pairing state).
//!
//! usage: session-gate <model.gguf>   (env: BW24_SPEC_K, BW24_NGEN per turn)
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: session-gate <model.gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let k: usize = std::env::var("BW24_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let n_new: usize = std::env::var("BW24_NGEN").ok().and_then(|v| v.parse().ok()).unwrap_or(24);

    // Three turns of real-ish token suffixes (numeric ids are fine — the gate is exactness,
    // not acceptance).
    let turn1: Vec<u32> = (101..229).collect();
    let turn2: Vec<u32> = (350..415).collect();
    let turn3: Vec<u32> = (517..551).collect();

    let mut sess = model.new_session(&e, 4096)?;
    let mut ok = true;
    for (i, suffix) in [turn1, turn2, turn3].iter().enumerate() {
        let (out, _d, _a) = model.generate_spec_session(&e, &mut sess, suffix, n_new, k)?;
        // reference: plain greedy over the FULL committed prefix (history includes this turn's
        // suffix + generated). generate() re-primes from scratch — the independent oracle.
        let hist_end = sess.committed.len() - out.len();
        let prefix: Vec<u32> = sess.committed[..hist_end].to_vec();
        let reference = model.generate(&e, &prefix, out.len())?;
        let m = if out == reference[..out.len().min(reference.len())] { "MATCH" } else { "MISMATCH" };
        if m == "MISMATCH" { ok = false; }
        println!("turn {}: {} tok generated, committed={} -> {m}", i + 1, out.len(),
                 sess.committed.len());
        if m == "MISMATCH" {
            println!("  session:   {:?}", &out[..out.len().min(12)]);
            println!("  reference: {:?}", &reference[..reference.len().min(12)]);
        }
    }
    println!("{}", if ok { "SESSION GATE: ALL TURNS MATCH" } else { "SESSION GATE: FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
