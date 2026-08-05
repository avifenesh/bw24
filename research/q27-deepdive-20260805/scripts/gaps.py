import csv, sys
for path in sys.argv[1:]:
    st = []
    for r in csv.DictReader(open(path)):
        try:
            s = float(r["Start (ns)"]); d = float(r["Duration (ns)"])
        except (KeyError, ValueError):
            continue
        st.append((s, s + d))
    if not st:
        print("NO-ROWS", path); continue
    st.sort()
    span = st[-1][1] - st[0][0]
    busy = 0.0; gap = 0.0; ngap = 0; gaps = []
    cur_s, cur_e = st[0]
    for s, e in st[1:]:
        if s > cur_e:
            busy += cur_e - cur_s; g = s - cur_e; gap += g; ngap += 1; gaps.append(g)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    gaps.sort()
    med = gaps[len(gaps)//2] if gaps else 0
    print("%-46s launches=%d span=%.2fms busy=%.2f%% gap=%.2f%% (%d gaps, %.2fms, median %.0fns)" % (
        path.split("/")[-1], len(st), span/1e6, busy/span*100, gap/span*100, ngap, gap/1e6, med))
