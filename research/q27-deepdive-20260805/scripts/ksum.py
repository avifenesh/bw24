import csv, sys
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    tot = sum(float(r["Total Time (ns)"]) for r in rows)
    n = sum(int(r["Instances"]) for r in rows)
    print("== %s  kernels=%d launches=%d tot_ms=%.2f" % (path.split("/")[-1], len(rows), n, tot/1e6))
    for r in rows[:10]:
        pct = float(r["Total Time (ns)"])/tot*100
        print("   %6.2f%% %7s  %s" % (pct, r["Instances"], r["Name"][:64]))
    for r in rows:
        if "fused" in r["Name"]:
            print("   FUSED %7s  %s" % (r["Instances"], r["Name"][:64]))
