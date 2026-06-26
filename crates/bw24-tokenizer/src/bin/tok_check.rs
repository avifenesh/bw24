//! tok-check: quick encode/decode sanity tool for the bw24 tokenizer.
//! usage: tok-check <model.gguf> "<text>"

use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: tok-check <model.gguf> \"<text>\"");
    let text = std::env::args().nth(2).unwrap_or_else(|| "Hello, world!".to_string());

    let g = GgufFile::open(&path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    println!("pre={} vocab={} eos={} bos={:?}", tok.pre(), tok.vocab_size(), tok.eos_id(), tok.bos_id());

    let ids = tok.encode(&text, true);
    let ids_json: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    println!("encode({text:?}) = [{}]", ids_json.join(", "));

    let back = tok.decode(&ids);
    println!("decode = {back:?}");
    println!("round-trip {}", if back == text { "OK" } else { "DIFFERS" });

    Ok(())
}
