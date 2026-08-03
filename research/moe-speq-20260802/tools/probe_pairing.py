import sys
sys.path.insert(0, "/home/ubuntu/receipts/moe-speq")
from moespeq_inproc import (read_route_lines, read_miss_lines, lockstep,
                            parse_spec_rounds, TRUNK_LAYERS, N_USED)

D = "/home/ubuntu/receipts/moe-speq"
tag = sys.argv[1] if len(sys.argv) > 1 else "chatprose"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 1

route_lines = read_route_lines(f"{D}/route3-{tag}-k{k}.txt")
rounds = parse_spec_rounds(f"{D}/spec3-{tag}-k{k}.log")

# oracle sweeps + batches, replicating the analyzer's timeline logic minus misses
t1_runs, cur_run, cur_r, prev_il = [], None, {}, None
for i, (il, t, ids) in enumerate(route_lines):
    if il >= TRUNK_LAYERS or t != 1:
        prev_il = None
        if cur_r and cur_run is not None:
            cur_run[1].append(cur_r)
            cur_r = {}
        if cur_run is not None:
            t1_runs.append(cur_run)
            cur_run = None
        continue
    if cur_run is None:
        cur_run = (i, [])
    if prev_il is not None and il <= prev_il and cur_r:
        cur_run[1].append(cur_r)
        cur_r = {}
    cur_r[il] = ids[:N_USED]
    prev_il = il
if cur_r and cur_run is not None:
    cur_run[1].append(cur_r)
if cur_run is not None:
    t1_runs.append(cur_run)
oracle_run = max(t1_runs, key=lambda r: len(r[1]))
oracle = oracle_run[1]
end_line = oracle_run[0] + sum(len(s) for s in oracle)

batches, cur_b, prev_il = [], {}, None
for i in range(end_line, len(route_lines)):
    il, t, ids = route_lines[i]
    if il >= TRUNK_LAYERS or t != k + 1:
        continue
    if prev_il is not None and il <= prev_il and cur_b:
        batches.append(cur_b)
        cur_b = {}
    cur_b[il] = [ids[c * N_USED:(c + 1) * N_USED] for c in range(t)]
    prev_il = il
if cur_b:
    batches.append(cur_b)

print(f"oracle={len(oracle)} batches={len(batches)} rounds={len(rounds)}")


def score(shift, delta):
    ok = tot = 0
    for bi, b in enumerate(batches):
        ri = bi + shift
        if ri < 0 or ri >= len(rounds):
            continue
        r = rounds[ri]
        for j in range(1, r["n_acc"] + 1):
            p = r["out_len"] - 1 + j + delta
            if p < 0 or p >= len(oracle):
                continue
            for il, cols in b.items():
                if len(cols) <= j or il not in oracle[p]:
                    continue
                tot += 1
                if cols[j] == oracle[p][il]:
                    ok += 1
    return ok, tot


for shift in [-1, 0, 1, 2]:
    row = []
    for delta in range(-2, 3):
        ok, tot = score(shift, delta)
        row.append(f"d{delta}:{ok}/{tot}")
    print(f"shift {shift}: " + "  ".join(row))

# per-round alignment at the best cell, to see WHICH rounds misalign
best = max(((score(s, d), s, d) for s in [-1, 0, 1, 2] for d in range(-2, 3)),
           key=lambda x: x[0][0])
(ok, tot), s, d = best
print(f"best shift={s} delta={d}: {ok}/{tot}")
mis = []
for bi, b in enumerate(batches):
    ri = bi + s
    if ri < 0 or ri >= len(rounds):
        continue
    r = rounds[ri]
    for j in range(1, r["n_acc"] + 1):
        p = r["out_len"] - 1 + j + d
        if p < 0 or p >= len(oracle):
            continue
        okr = sum(1 for il, cols in b.items()
                  if len(cols) > j and il in oracle[p] and cols[j] == oracle[p][il])
        mis.append((ri, p, okr))
print("per accepted col (round, pos, exact-layers/79):")
print(mis[:30])
print("...")
print(mis[-10:])
