#!/usr/bin/env python3
"""MoE-SpeQ oracle-ceiling analyzer, IN-PROCESS pairing (moe-speq lane, 2026-08-03).

One patched run-spec process per (class, K) emits three artifacts from the SAME forwards:
  spec3-<tag>-k<K>.log   [generate] oracle + generate_spec + [R<i>] MEMRA_DEBUG_SPEC rounds
  route3-<tag>-k<K>.txt  MEMRA_MOE_TRACE   "<layer> <t> <id,...>" per (layer, forward)
  miss3-<tag>-k<K>.txt   MEMRA_MOE_MISS_TRACE "<layer> <proj> <expert> <H|M>" per SLRU lookup

Why in-process: run-gen and run-spec's oracle are BOTH greedy but numerically distinct code
paths; on this artifact they diverge textually at a near-tie token (~position 31 on
chat-prose), so cross-process position alignment is invalid (measured: cross-process routing
similarity collapses 0.95 -> 0.33 after the divergence). Within one process, run-spec's
self-consistency gate asserts the spec phase commits EXACTLY the oracle's tokens, so
positions correspond and the drafter's verify-column routing can be compared to the
oracle decode forward for the same position.

Structure of the process (route trace order):
  [oracle prime: one t=n_prompt sweep]  [oracle decode: n_gen t=1 sweeps]
  [spec prime: t=n_prompt sweep]        [draft-KV fill etc: il=65535, filtered]
  [rounds: t=K+1 verify sweeps interleaved with t=1 MTP-head forwards (il=65535, filtered)]

Denominator: misses in the oracle-decode sweeps (the real plain-decode miss pattern; the
miss trace is walked in lockstep with the route trace, so oracle sweeps are identified
positionally between the two prime sweeps).

Numerator: verify batch b (pairs with round b+shift), column j >= 1 = the drafter's
lookahead-j routing for generated position p = out_len-1+j (+delta, resolved empirically by
maximizing accepted-column routing equality; expected delta=0 in-process).
  strict  = miss (l,e) at p is in an ACCEPTED column's predicted set for (p, l).
  partial = same over ALL columns targeting p (rejected drafts still prefetch).
"""
import argparse
import json
import re
import sys
from collections import defaultdict

N_USED = 8
TRUNK_LAYERS = 80


def read_route_lines(path):
    out = []
    for line in open(path):
        parts = line.split()
        if len(parts) != 3:
            continue
        il, t = int(parts[0]), int(parts[1])
        ids = [int(x) for x in parts[2].split(",")]
        out.append((il, t, ids))  # keep MTP lines for lockstep, filter later
    return out


def read_miss_lines(path):
    out = []
    for line in open(path):
        parts = line.split()
        if len(parts) != 4:
            continue
        out.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return out


def lockstep(route_lines, miss_lines):
    """Pair each route line with its contiguous block of miss lines (same layer).

    Both traces come from the same forwards in the same order. A forward at layer il
    yields ONE route line and (on cache-dispatch paths) a run of miss lines at il.
    Forwards that bypass the cache (none in observe mode) or layers with no lookups
    yield no miss lines; the walk only consumes miss lines whose il matches.
    Returns per-route-line: dict (l,e) -> [missed_any, n_lookups, n_missed].
    """
    per_line = []
    mi = 0
    for il, t, ids in route_lines:
        rec = {}
        while mi < len(miss_lines) and miss_lines[mi][0] == il:
            _, proj, ex, hm = miss_lines[mi]
            r = rec.setdefault((il, ex), [False, 0, 0])
            r[1] += 1
            if hm == "M":
                r[0] = True
                r[2] += 1
            mi += 1
        per_line.append(rec)
    return per_line, mi


ROUND_RX = re.compile(
    r"\[R(\d+)\] pos=(\d+) out_len=(\d+) last_tok=\d+ draft=\[([0-9, ]*)\] n_acc=(\d+) bonus=(\d+)")


def parse_spec_rounds(log_path):
    rounds = []
    for line in open(log_path, errors="replace"):
        m = ROUND_RX.search(line)
        if m:
            draft = [int(x) for x in m.group(4).split(",")] if m.group(4).strip() else []
            rounds.append(dict(idx=int(m.group(1)), pos=int(m.group(2)),
                               out_len=int(m.group(3)), draft=draft,
                               n_acc=int(m.group(5))))
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--miss", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    k = args.k

    route_lines = read_route_lines(args.route)
    miss_lines = read_miss_lines(args.miss)
    per_line, consumed = lockstep(route_lines, miss_lines)
    rounds = parse_spec_rounds(args.log)

    # Process timeline (measured, chatprose-k1): [warmup prime t=n_p + 1 decode sweep]
    # [oracle prime t=n_p + n_gen t=1 sweeps] [spec prime + setup] [rounds: t=k+1 verify
    # sweeps interleaved with MTP-head lines]. Oracle decode = the LONGEST contiguous run
    # of t==1 trunk sweeps; verify batches = all t==k+1 trunk sweeps after that run.
    # Contiguous means: consecutive full 79-layer t==1 sweeps with nothing but t==1 trunk
    # lines between them (an intervening prime/MTP/verify line breaks the run).
    runs = []   # (kind, start_line, sweep list) where kind is "t1" or other
    cur_r, cur_m, prev_il = {}, {}, None
    t1_runs = []      # list of (first_line_idx, [(sweep, miss)])
    cur_run = None
    for i, (il, t, ids) in enumerate(route_lines):
        if il >= TRUNK_LAYERS:
            prev_il = None
            if cur_r:
                if cur_run is not None:
                    cur_run[1].append((cur_r, cur_m))
                cur_r, cur_m = {}, {}
            if cur_run is not None:
                t1_runs.append(cur_run)
                cur_run = None
            continue
        if t != 1:
            prev_il = None
            if cur_r:
                if cur_run is not None:
                    cur_run[1].append((cur_r, cur_m))
                cur_r, cur_m = {}, {}
            if cur_run is not None:
                t1_runs.append(cur_run)
                cur_run = None
            continue
        if cur_run is None:
            cur_run = (i, [])
        if prev_il is not None and il <= prev_il and cur_r:
            cur_run[1].append((cur_r, cur_m))
            cur_r, cur_m = {}, {}
        cur_r[il] = ids[:N_USED]
        cur_m.update(per_line[i])
        prev_il = il
    if cur_r and cur_run is not None:
        cur_run[1].append((cur_r, cur_m))
    if cur_run is not None:
        t1_runs.append(cur_run)

    oracle_run = max(t1_runs, key=lambda r: len(r[1]))
    oracle = oracle_run[1]
    oracle_end_line = oracle_run[0]  # start line; end approximated below
    # find the line index where the oracle run ends: next non-t1 line after its start
    # (sufficient: batches are collected only after the oracle run START + its length in lines)
    oracle_end_line = oracle_run[0] + sum(len(s[0]) for s in oracle)

    # Verify batches: t==k+1 trunk sweeps after the oracle run.
    batches = []
    cur_b, prev_il = {}, None
    for i in range(oracle_end_line, len(route_lines)):
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

    n_oracle = len(oracle)
    total_disp = sum(len(m) for _, m in oracle)
    total_miss = sum(1 for _, m in oracle for rec in m.values() if rec[0])
    block_miss = sum(rec[2] for _, m in oracle for rec in m.values())
    block_look = sum(rec[1] for _, m in oracle for rec in m.values())
    # Gate 1: lockstep sanity — missed (l,e) must be routed in its own sweep.
    bad = 0
    for r, m in oracle:
        for (il, ex) in m:
            if il in r and ex not in r[il]:
                bad += 1
    print(f"[{args.tag}-k{k}] oracle sweeps={n_oracle} verify batches={len(batches)} "
          f"rounds={len(rounds)} miss-lines consumed={consumed}/{len(miss_lines)}")
    print(f"[{args.tag}-k{k}] oracle expert-dispatches={total_disp} missed={total_miss} "
          f"({total_miss / max(total_disp, 1) * 100:.1f}%) block-lookups={block_look} "
          f"block-misses={block_miss} lockstep-disagreements={bad}")

    # batch <-> round pairing: batches[bi] pairs with rounds[bi + shift]; position of
    # column j is p = out_len - 1 + j + delta. Both offsets are resolved EMPIRICALLY by
    # maximizing accepted-column routing equality with the oracle sweeps (self-consistency
    # makes the true cell exact; measured chatprose-k1: shift=1, delta=0 -> 3081/3081).
    def alignment(shift, delta):
        ok = tot = 0
        for bi, b in enumerate(batches):
            ri = bi + shift
            if ri < 0 or ri >= len(rounds):
                continue
            r = rounds[ri]
            for j in range(1, r["n_acc"] + 1):
                p = r["out_len"] - 1 + j + delta
                if p < 0 or p >= n_oracle:
                    continue
                orr = oracle[p][0]
                for il, cols in b.items():
                    if len(cols) <= j or il not in orr:
                        continue
                    tot += 1
                    if cols[j] == orr[il]:
                        ok += 1
        return ok, tot

    best = (0, -1, 0, 0)   # (ok, tot? ...) -> track (ok, tot, shift, delta)
    best = None
    for s in range(-1, 3):
        for d in range(-3, 4):
            ok, tot = alignment(s, d)
            if tot and (best is None or ok > best[0]):
                best = (ok, tot, s, d)
    if best is None:
        sys.exit(f"[{args.tag}-k{k}] no accepted columns to align")
    align_ok, align_tot, shift, delta = best
    pairs = []
    for bi, b in enumerate(batches):
        ri = bi + shift
        if 0 <= ri < len(rounds):
            pairs.append((rounds[ri], b))
    skipped = len(rounds) - len(pairs)

    acc_pred = defaultdict(lambda: defaultdict(set))
    any_pred = defaultdict(lambda: defaultdict(set))
    depth_of = {}
    for r, b in pairs:
        for j in range(1, len(r["draft"]) + 1):
            p = r["out_len"] - 1 + j + delta
            if p < 0 or p >= n_oracle:
                continue
            accepted = j <= r["n_acc"]
            for il, cols in b.items():
                if len(cols) <= j:
                    continue
                any_pred[p][il].update(cols[j])
                if accepted:
                    acc_pred[p][il].update(cols[j])
            if accepted and (p not in depth_of or depth_of[p] > j):
                depth_of[p] = j

    acc_pred = dict(acc_pred)
    any_pred = dict(any_pred)
    hidden_strict = hidden_partial = denom = 0
    by_depth = defaultdict(int)
    miss_at_uncovered = 0
    for p in range(n_oracle):
        for (il, ex), rec in oracle[p][1].items():
            if not rec[0]:
                continue
            denom += 1
            if p in acc_pred and ex in acc_pred[p].get(il, ()):
                hidden_strict += 1
                by_depth[depth_of.get(p, 0)] += 1
            if p in any_pred and ex in any_pred[p].get(il, ()):
                hidden_partial += 1
            else:
                if p not in any_pred:
                    miss_at_uncovered += 1

    res = dict(
        k=k, oracle_sweeps=n_oracle, batches=len(batches), rounds=len(rounds),
        skipped_rounds=skipped, shift=shift, delta=delta,
        align_ok=align_ok, align_tot=align_tot,
        dispatches=total_disp, missed=denom,
        block_lookups=block_look, block_misses=block_miss,
        lockstep_disagreements=bad,
        strict=hidden_strict / denom if denom else 0.0,
        partial=hidden_partial / denom if denom else 0.0,
        acc_pos=len(acc_pred), any_pos=len(any_pred),
        miss_at_uncovered=miss_at_uncovered,
        by_depth=dict(sorted(by_depth.items())),
    )
    print(f"  shift={shift} delta={delta} align={align_ok}/{align_tot} "
          f"({align_ok / max(align_tot, 1) * 100:.1f}%) accepted-pos={len(acc_pred)}/{n_oracle} "
          f"targeted-pos={len(any_pred)}/{n_oracle} misses-at-untargeted={miss_at_uncovered}")
    print(f"  STRICT miss-hiding = {res['strict'] * 100:.2f}%   "
          f"PARTIAL = {res['partial'] * 100:.2f}%   (denom {denom}; strict by depth {res['by_depth']})")
    print(f"RESULT {args.tag} K={k} strict {res['strict'] * 100:.1f}% "
          f"partial {res['partial'] * 100:.1f}%")
    if args.json_out:
        json.dump(dict(tag=args.tag, **res), open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
