import json, statistics, sys, collections
pts = collections.defaultdict(list)
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    lab = d["label"]                        # arm-cN-pM
    arm, c, p = lab.split("-")
    pts[(arm, c)].append((p, d["agg_tok_s"], d["lat_p50_s"], d["n_err"], d["n_shed"]))
for key in sorted(pts):
    rows = sorted(pts[key])
    aggs = [r[1] for r in rows]
    print("%-8s %-4s N=%d  agg median %.2f  (%s)  p50 median %.3f  err=%d shed=%d" % (
        key[0], key[1], len(aggs), statistics.median(aggs),
        " ".join("%.2f" % a for a in aggs),
        statistics.median([r[2] for r in rows]),
        sum(r[3] for r in rows), sum(r[4] for r in rows)))
# paired deltas per (c, pass)
by = {}
for (arm, c), rows in pts.items():
    for p, agg, *_ in rows:
        by[(c, p, arm)] = agg
print("\npaired deltas (fuse ON vs OFF, per pass):")
for c in ("c1", "c8"):
    ds = []
    for p in ("p1", "p2", "p3"):
        b, n = by.get((c, p, "base")), by.get((c, p, "nofuse"))
        if b and n:
            d = (b / n - 1) * 100
            ds.append(d); print("  %s %s: base %.2f vs nofuse %.2f = %+.2f%%" % (c, p, b, n, d))
    if ds: print("  %s MEAN %+.2f%%  (median %+.2f%%)" % (c, sum(ds)/len(ds), statistics.median(ds)))
