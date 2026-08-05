import sys
p = "/root/bw24/crates/memra-engine/src/decode_batch.rs"
s = open(p).read()
old = """                    // BATCHED FFN gate+up LAUNCH FUSION (lane/q27-deepdive, 2026-08-05): the
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
                    };"""
new = """                    // REFUTED ARM (lane/q27-deepdive, 2026-08-05): fusing this pair into
                    // matmul_q8_fused2_t (fused2_b8) measured FLAT-TO-NEGATIVE at the serving
                    // tick (bench c=8 sign-flipping, serve c=8 paired mean -0.20%/N=3). The
                    // c=8 tick is 73.2% one weight-bound class with launch cost already hidden.
                    // The m=1 arm in matmul_pre_dual_noscale (+0.94%) stays.
                    let g = e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?;
                    let u = e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?;"""
if new in s: print("SKIP already"); sys.exit(0)
n = s.count(old)
if n != 1: print("ANCHOR-FAIL count=%d" % n); sys.exit(1)
open(p, "w").write(s.replace(old, new)); print("OK reverted lever3 call site")
