// Find the token id whose token_embd row matches a target prefix (to align with llama.cpp dumps).
use bw24_gguf::{GgufFile, dequant};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let target: Vec<f32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let g = GgufFile::open(&path).unwrap();
    let t = g.find("token_embd.weight").unwrap();
    let n_embd = t.ne[0] as usize;
    let n_vocab = t.ne[1] as usize;
    let (blk, tsize) = t.ggml_type.block_and_type_size();
    let row_bytes = (n_embd as u64 / blk * tsize) as usize;
    let raw = g.tensor_data(t);
    let mut best = (0usize, f32::INFINITY);
    for tok in 0..n_vocab {
        let row = dequant::dequantize(t.ggml_type, &raw[tok*row_bytes..tok*row_bytes+row_bytes], n_embd);
        let mut err = 0f32;
        for (i, &tv) in target.iter().enumerate() { err += (row[i]-tv).abs(); }
        if err < best.1 { best = (tok, err); }
    }
    let row = dequant::dequantize(t.ggml_type, &raw[best.0*row_bytes..best.0*row_bytes+row_bytes], n_embd);
    println!("best token = {} (err {:.5})  first8: {:?}", best.0, best.1, &row[..8]);
}
