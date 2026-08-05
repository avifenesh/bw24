import sys
edits = [
 ("/root/bw24/crates/memra-engine/cu/qmatvec.cu",
  "    q8_0_mmvq_fused2_b<4>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);\n}\ntemplate<int MCOLS>\n__device__ __forceinline__ void q8_0_mmvq_fused3_b(",
  """    q8_0_mmvq_fused2_b<4>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
// b8 wrapper (lane/q27-deepdive, 2026-08-05): the SERVING tier. Same template, same
// q8_0_mmvq_batched_row body, MCOLS=8 -> BIT-IDENTICAL per (tensor,token,row) to the two
// qmatvec_q8_0_mmvq_b8 launches, one shared q8_1 activation instead of two re-quantizes.
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2_b8(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q8_0_mmvq_fused2_b<8>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_fused3_b("""),

 ("/root/bw24/crates/memra-engine/src/lib.rs",
  '        let f = self.func(if Self::batched_mcols(m) == 2 { "qmatvec_q8_0_mmvq_fused2_b2" }\n                          else { "qmatvec_q8_0_mmvq_fused2_b4" });',
  '''        let f = self.func(match Self::batched_mcols(m) {
            2 => "qmatvec_q8_0_mmvq_fused2_b2",
            4 => "qmatvec_q8_0_mmvq_fused2_b4",
            _ => "qmatvec_q8_0_mmvq_fused2_b8",   // b8 = the SERVING tier (c=5..8)
        });'''),

 ("/root/bw24/crates/memra-engine/src/lib.rs",
  '        if !(2..=4).contains(&m) || std::env::var("MEMRA_NO_BATCHED").is_ok() { return Ok(None); }\n        let Some([p0, p1]) = self.q8_fused_params(&[w0, w1]) else { return Ok(None) };\n        Ok(Some(self.q8_fused2_t_core(p0.0, p1.0, aq, ad, m, w0.in_features(), p0.1, p1.1, p0.2)?))',
  '''        // m<=8 (lane/q27-deepdive): the serving mcols-8 tier now has its fused2_b8 wrapper.
        if !(2..=8).contains(&m) || std::env::var("MEMRA_NO_BATCHED").is_ok() { return Ok(None); }
        let Some([p0, p1]) = self.q8_fused_params(&[w0, w1]) else { return Ok(None) };
        Ok(Some(self.q8_fused2_t_core(p0.0, p1.0, aq, ad, m, w0.in_features(), p0.1, p1.1, p0.2)?))'''),

 ("/root/bw24/crates/memra-engine/src/lib.rs",
  "    fn q8_ffn_fuse2_on(&self) -> bool {",
  "    pub fn q8_ffn_fuse2_on(&self) -> bool {"),

 ("/root/bw24/crates/memra-engine/src/decode_batch.rs",
  """                    let n_ff = ffn_gate.out_features();
                    let (zq, zd) = e.quantize_q8_1(&z, b_n, n_embd)?;
                    let g = e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?;
                    let u = e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?;""",
  """                    let n_ff = ffn_gate.out_features();
                    let (zq, zd) = e.quantize_q8_1(&z, b_n, n_embd)?;
                    // BATCHED FFN gate+up LAUNCH FUSION (lane/q27-deepdive, 2026-08-05): the
                    // serve tick ran these as two `matmul_pre` -> two qmatvec_q8_0_mmvq_b8
                    // launches per dense layer per tick (73.2% of the c=8 tick is the b8 class).
                    // matmul_q8_fused2_t is bit-identical per (tensor,token,row) to the pair and
                    // shares one q8_1 quantize. Seam: MEMRA_Q8_FFN_FUSE2=0 / MEMRA_NO_BATCHED.
                    let fused = if e.q8_ffn_fuse2_on() {
                        e.matmul_q8_fused2_t(ffn_gate, ffn_up, &zq, &zd, b_n)?
                    } else {
                        None
                    };
                    let (g, u) = match fused {
                        Some(gu) => gu,
                        None => (e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?,
                                 e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?),
                    };"""),
]
for path, old, new in edits:
    s = open(path).read()
    if new.strip() and new in s:
        print("SKIP already:", path); continue
    n = s.count(old)
    if n != 1:
        print("ANCHOR-FAIL count=%d in %s: %r" % (n, path, old[:70])); sys.exit(1)
    open(path, "w").write(s.replace(old, new))
    print("OK", path)
