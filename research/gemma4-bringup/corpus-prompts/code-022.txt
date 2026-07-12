//! tok-trim: cut a text file to a prefix that encodes to EXACTLY n tokens under the model's own
//! tokenizer (bench-harness tool: builds the pp512/pp2048 prompt files from a longer prompt so
//! prefill benches hit exact token counts). HF-dir sources only (tokenizer.json).
//! usage: tok-trim <hf_dir> <in.txt> <n_tokens> <out.txt>

use bw24_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let (dir, inp, n, out) = (
        a.next().expect("usage: tok-trim <hf_dir> <in.txt> <n_tokens> <out.txt>"),
        a.next().expect("in.txt"),
        a.next().expect("n_tokens").parse::<usize>()?,
        a.next().expect("out.txt"),
    );
    let tok = Tokenizer::from_hf_dir(std::path::Path::new(&dir))?;
    let text = std::fs::read_to_string(&inp)?;
    let total = tok.encode(&text, true).len();
    assert!(total >= n, "input encodes to {total} < requested {n} tokens");

    // Binary search the longest char-boundary prefix with encode-len <= n, then walk to exact.
    // (BPE token count is monotone non-decreasing in prefix length at char granularity here;
    // the final assert guards any edge case.)
    let idx: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
    let (mut lo, mut hi) = (0usize, idx.len() - 1); // lo: <= n tokens, hi: > n or end
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if tok.encode(&text[..idx[mid]], true).len() <= n { lo = mid } else { hi = mid }
    }
    let mut cut = lo;
    while cut > 0 && tok.encode(&text[..idx[cut]], true).len() > n { cut -= 1; }
    let got = tok.encode(&text[..idx[cut]], true).len();
    assert_eq!(got, n, "could not hit exactly {n} tokens (closest prefix = {got})");
    std::fs::write(&out, &text[..idx[cut]])?;
    println!("wrote {out}: {} chars = exactly {n} tokens", idx[cut]);
    Ok(())
}
