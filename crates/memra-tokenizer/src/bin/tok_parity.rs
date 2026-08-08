//! tok-parity: byte-exact token-id parity gate vs a reference-ids file.
//!
//! The listing gate for a new SKU's tokenizer (first user: step35 / Step-3.7-Flash):
//! memra's GGUF-built tokenizer must produce IDENTICAL ids to the HF reference tokenizer
//! on an adversarial corpus. The reference side is produced offline by
//! `research/step-sku-20260807/run-hf-reference.py` (the `tokenizers` library over the
//! sha-pinned `tokenizer.json`); this bin is the memra side plus the comparison.
//!
//! usage: tok-parity <model.gguf> <corpus.tsv> <ref-ids.tsv>
//!
//!   corpus.tsv   `<name>\t<hex of utf-8 bytes>` — hex transport because the corpus is
//!                full of newlines/tabs/control-token literals.
//!   ref-ids.tsv  `<name>\t<ids add_special=true csv>\t<ids add_special=false csv>`
//!
//! Compares BOTH encode modes per case. On mismatch prints the first diverging index,
//! both id streams, and each side's decode of its own ids (so the split point is visible
//! in text, not just numbers). Exit 0 = all cases, both modes, token-for-token identical.

use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("bad hex"))
        .collect()
}

fn parse_ids(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').map(|x| x.parse().expect("bad id")).collect()
}

fn first_div(a: &[u32], b: &[u32]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: tok-parity <model.gguf> <corpus.tsv> <ref-ids.tsv>";
    let model = args.next().expect(usage);
    let corpus_path = args.next().expect(usage);
    let ref_path = args.next().expect(usage);

    let g = GgufFile::open(&model)?;
    let tok = Tokenizer::from_gguf(&g)?;
    println!(
        "tok-parity: model={model} pre={} vocab={} bos={:?} add_bos matches encode(add_special)",
        tok.pre(),
        tok.vocab_size(),
        tok.bos_id()
    );

    // name -> (ids_special, ids_plain)
    let mut refs = std::collections::HashMap::new();
    for line in std::fs::read_to_string(&ref_path)?.lines() {
        let mut f = line.splitn(3, '\t');
        let name = f.next().expect("ref name").to_string();
        let spec = parse_ids(f.next().expect("ref special ids"));
        let plain = parse_ids(f.next().expect("ref plain ids"));
        refs.insert(name, (spec, plain));
    }

    let corpus = std::fs::read_to_string(&corpus_path)?;
    let (mut n_cases, mut n_fail) = (0usize, 0usize);
    for line in corpus.lines() {
        let mut f = line.splitn(2, '\t');
        let name = f.next().expect("corpus name");
        let text = String::from_utf8(unhex(f.next().expect("corpus hex")))?;
        let (ref_spec, ref_plain) = refs
            .get(name)
            .unwrap_or_else(|| panic!("no reference ids for case {name:?}"));
        n_cases += 1;

        let mut case_ok = true;
        for (mode, add_special, want) in
            [("special", true, ref_spec), ("plain", false, ref_plain)]
        {
            let got = tok.encode(&text, add_special);
            if &got != want {
                case_ok = false;
                let d = first_div(&got, want);
                println!("MISMATCH {name} [{mode}] first_div={d} memra_len={} ref_len={}",
                         got.len(), want.len());
                println!("  text bytes: {:?}", text);
                println!("  memra ids: {got:?}");
                println!("  ref   ids: {want:?}");
                let ctx = d.saturating_sub(2);
                println!("  memra tail decode from {ctx}: {:?}",
                         tok.decode_special(&got[ctx..], true));
                println!("  ref   tail decode from {ctx}: {:?}",
                         tok.decode_special(&want[ctx..], true));
            }
        }
        if case_ok {
            println!("OK {name}");
        } else {
            n_fail += 1;
        }
    }

    println!("tok-parity: {}/{n_cases} cases identical in BOTH modes", n_cases - n_fail);
    if n_fail > 0 {
        println!("tok-parity: FAIL ({n_fail} mismatching case(s))");
        std::process::exit(1);
    }
    println!("tok-parity: PASS");
    Ok(())
}
