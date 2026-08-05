#!/usr/bin/env python3
"""Re-derive achieved DRAM bandwidth per Q8_0 decode kernel class from the committed nsys trace.

WHY THIS EXISTS: ncu is HARD-BLOCKED on the pro6000wk-runpod-community pod (ERR_NVGPUCTRPERM;
/proc/driver/nvidia/params has RmProfilingAdminOnly:1, and the container has neither cap_perfmon
nor cap_sys_admin), so the phase-1 deliverable's "achieved BW + occupancy per top-5 kernel" cannot
be measured there. Occupancy is simply unavailable. Achieved BW *can* be derived, because at m=1
every one of these kernels is weight-streaming: it reads each Q8_0 weight byte exactly once, and
the activation read + output write are ~4 orders of magnitude smaller than the weight traffic. So

    achieved GB/s  ~  weight_bytes / kernel_duration

This script recomputes every number in the RESULTS.md derived-BW table straight from the committed
trace CSV, so the table is reproducible rather than asserted. Numbers are DERIVED, not ncu-measured,
and every table quoting them says so.

Kernel classes are identified by GRID DIMENSION (GrdX = number of output rows, one warp per row),
not by name, because several distinct tensor shapes share the `qmatvec_q8_0_mmvq` symbol. The
grd=1280 bucket is bimodal (two shapes, 64 launches/token each) and is split on duration.

Tensor shapes are authoritative, read from the GGUF header (gguf-inspect + a header walk):
  ffn_down      Q8_0 ne=[17408, 5120]   x64 layers
  ffn_gate/up   Q8_0 ne=[5120, 17408]   x64 layers each
  attn_output   Q8_0 ne=[6144, 5120]    x16 full-attn layers
  ssm_out       Q8_0 ne=[6144, 5120]    x48 GDN layers
  attn_qkv      Q8_0 ne=[5120, 10240]   x48
  output.weight Q8_0 ne=[5120, 248320]  (lm_head)
Q8_0 row layout: row_bytes = (in_features / 32) * 34  (32 int8 weights + one fp16 scale).

Usage: python3 scripts/derive-bw.py [nsys/nsys-q8-decode-c1-trace_cuda_gpu_trace.csv]
"""
import collections
import csv
import statistics as st
import sys

BOARD_EFFECTIVE_GBS = 1711.0  # drooped mem clock 13365/14001 -> effective, community board


def row_bytes(in_f: int) -> int:
    assert in_f % 32 == 0, in_f
    return (in_f // 32) * 34


def wbytes(in_f: int, out_f: int) -> int:
    return row_bytes(in_f) * out_f


# GrdX identifies the launch: it is (total output rows / 4), one warp per row, 4 warps per block.
# A fused launch's row count is the SUM over the tensors it emits, which is how fused2/fused3 are
# told apart from the plain per-tensor launches that share the symbol name.
BY_GRID = {
    # GrdX -> (label, in_features, TOTAL output rows produced by the launch)
    "4352": ("ffn gate+up (separate launches)", 5120, 17408),
    # fused2 on the GDN layers emits attn_qkv (10240) + attn_gate (6144) = 16384 rows in one launch
    "4096": ("fused2 (qkv+gate, 48 GDN layers)", 5120, 16384),
    # fused3 on the full-attn layers emits q (12288) + k (1024) + v (1024) = 14336 rows
    "3584": ("fused3 (q/k/v, 16 full-attn layers)", 5120, 14336),
    "62080": ("lm_head", 5120, 248320),
    # tiny per-layer fused2 (ssm_alpha+ssm_beta, 48+48 rows)
    "24": ("ssm alpha+beta (fused2, tiny)", 5120, 96),
}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "nsys/nsys-q8-decode-c1-trace_cuda_gpu_trace.csv"
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        name = r["Name"].split("(")[0]
        if "qmatvec_q8_0_mmvq" not in name:
            continue
        by[(name, r["GrdX"])].append(int(r["Duration (ns)"]))

    print(f"source: {path}")
    print("DERIVED from nsys per-launch durations x exact Q8_0 byte footprints.")
    print("ncu is BLOCKED on this pod => these are NOT ncu-measured; occupancy UNAVAILABLE.\n")
    print(f"{'class':<34}{'launch/tok':>11}{'med ns':>10}{'MB':>9}{'GB/s':>9}")

    n_tok = len(by.get(("qmatvec_q8_0_mmvq", "62080"), [])) or 128  # 1 lm_head launch per token
    rows = []
    tot_bytes = tot_ns = 0
    for (name, grd), durs in by.items():
        if grd not in BY_GRID:
            continue
        label, in_f, out_f = BY_GRID[grd]
        med = st.median(durs)
        nb = wbytes(in_f, out_f)
        rows.append((label, len(durs) / n_tok, med, nb, nb / med))
        tot_bytes += nb * len(durs)
        tot_ns += sum(durs)

    # grd=1280 carries TWO shapes at 64 launches/token each: ffn_down (in=17408) and the
    # attn_output / ssm_out class (in=6144). Split on duration — the gap is 22us vs 60us.
    d1280 = by.get(("qmatvec_q8_0_mmvq", "1280"), [])
    if d1280:
        small = [d for d in d1280 if d < 40000]
        big = [d for d in d1280 if d >= 40000]
        for label, in_f, durs in (
            ("attn_out / ssm_out", 6144, small),
            ("ffn_down", 17408, big),
        ):
            if not durs:
                continue
            med = st.median(durs)
            nb = wbytes(in_f, 5120)
            rows.append((label, len(durs) / n_tok, med, nb, nb / med))
            tot_bytes += nb * len(durs)
            tot_ns += sum(durs)

    for label, per_tok, med, nb, gbs in sorted(rows, key=lambda x: -x[4]):
        print(f"{label:<34}{per_tok:>11.0f}{med:>10.0f}{nb/1e6:>9.2f}{gbs:>9.1f}")

    agg = tot_bytes / tot_ns
    print(f"{'WEIGHTED AGGREGATE (family)':<34}{'':>11}{'':>10}"
          f"{tot_bytes/n_tok/1e9:>8.3f}G{agg:>9.1f}")
    print(f"\nmatvec family moves {tot_bytes/n_tok/1e9:.3f} GB/token in "
          f"{tot_ns/n_tok/1e6:.3f} ms  (n_tok={n_tok})")
    print(f"board effective (drooped) bandwidth: ~{BOARD_EFFECTIVE_GBS:.0f} GB/s"
          f"  => aggregate is {100*agg/BOARD_EFFECTIVE_GBS:.1f}% of achievable")
    print("=> the big weight-streaming classes run at ~92% of the drooped board bandwidth, while")
    print("   the small class (attn_out/ssm_out, in=6144) is the laggard — that gap is hypothesis")
    print("   H4 in RESULTS.md, and closing it needs ncu occupancy on prod-class silicon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
