import sys, hashlib
p = "/root/bw24/crates/memra-engine/src/lib.rs"
s = open(p).read()
if "q8_ffn_fuse2_on" in s:
    print("ALREADY-APPLIED"); sys.exit(0)

a1 = "        if w1.in_features() != in_f || w1.out_features() != out_f { return Ok(None); }\n"
ins1 = '''        // Q8_0 ARM (lane/q27-deepdive, 2026-08-05): the dense-FFN gate+up pair on a Q8_0 trunk fell
        // through this NVFP4-only gate to two `matmul_pre_noscale` launches -- measured 128 of the
        // 1015 launches/token on q27-Q8_0 decode, the single largest un-fused class in the tick.
        // `q8_fused2_core` already serves the same pair shape for the shared-expert gate/up, and its
        // kernel body is `qmatvec_q8_0_mmvq` VERBATIM per (tensor,row) -> BIT-IDENTICAL to the two
        // separate launches. Q8_0 carries no macro-scale (q8_fused_params requires scale==1.0), so
        // the noscale contract is satisfied by returning 1.0 for both. Seam: MEMRA_Q8_FFN_FUSE2=0.
        // rp4 guard: under MEMRA_Q8RP the singles route to the `_rp` split-plane twin; fused2 has no
        // `_rp` form, so fusing there would swap dispatch families mid-model -> bail.
        let no_mirror = |w: &crate::model::GpuTensor| {
            !matches!(w, GpuTensor::Quant { rp4: Some(_), .. })
        };
        if self.q8_ffn_fuse2_on()
            && no_mirror(w0) && no_mirror(w1)
            && let Some([p0, p1]) = self.q8_fused_params(&[w0, w1])
        {
            let (y0, y1) = self.q8_fused2_core(p0.0, p1.0, aq, ad, in_f, p0.1, p1.1, p0.2)?;
            return Ok(Some(((y0, 1.0), (y1, 1.0))));
        }
'''
a2 = "    /// Eligibility + param extraction for the fused q8_0 launches:"
ins2 = '''    /// Rollback seam for the Q8_0 dense-FFN gate+up fusion arm in `matmul_pre_dual_noscale`
    /// (lane/q27-deepdive, 2026-08-05). Default ON; `MEMRA_Q8_FFN_FUSE2=0` restores the
    /// two-`matmul_pre_noscale` pair. Read once -- the dispatch must not vary within a run.
    fn q8_ffn_fuse2_on(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("MEMRA_Q8_FFN_FUSE2").as_deref() != Ok("0"))
    }

'''
for a in (a1, a2):
    n = s.count(a)
    if n != 1:
        print(f"ANCHOR-FAIL count={n} for {a[:60]!r}"); sys.exit(1)
s = s.replace(a1, a1 + ins1).replace(a2, ins2 + a2)
open(p, "w").write(s)
print("APPLIED sha256", hashlib.sha256(s.encode()).hexdigest()[:16])
