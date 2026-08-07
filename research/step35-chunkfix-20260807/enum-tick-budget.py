# Does serve's OUTER per-tick segmentation re-open the arm-selection door?
# Serve calls prime_cache once per tick with `take` tokens; prime_cache then splits that call
# into MEMRA_PRIME_CHUNK sub-chunks. Post-fix the sub-chunk split is invariant (seq_end fixed
# per CALL) -- but seq_end differs per CALL, so the TICK BUDGET is a second segmentation axis.
WIN = 512
PRIME_MIN_T = 16

def prime_calls(T, budget):
    """serve's tick loop: chunks of `budget`, tail merged to keep >= PRIME_MIN_T (worker.rs:3555)"""
    calls, q, pos = [], T, 0
    while q > 0:
        if q < max(PRIME_MIN_T, 2):
            calls.append((pos, q, 'tokenwise')); pos += q; q = 0; continue
        take = min(q, budget)
        if 0 < q - take < PRIME_MIN_T:
            take = q
        calls.append((pos, take, 'prime')); pos += take; q -= take
    return calls

def arms_post(T, budget, chunk=4096):
    """post-fix: per CALL seq_end = pos+t; SWA arm = naive_w iff seq_end > WIN"""
    out = []
    for pos, t, kind in prime_calls(T, budget):
        if kind == 'tokenwise':
            out.append(('decode', pos, t)); continue
        seq_end = pos + t
        arm = 'naive_w' if seq_end > WIN else 'FA'
        # sub-chunk split inside the call: same arm for every sub-chunk (that IS the fix)
        out.append((arm, pos, t))
    return out

def fa_rows_post(T, budget):
    """set of absolute rows computed by the FA arm"""
    r = set()
    for arm, pos, t in arms_post(T, budget):
        if arm == 'FA':
            r |= set(range(pos, pos + t))
    return r

# 1) is the DEFAULT interactive budget (1024) identical to a monolithic prime?
bad_int = [T for T in range(2, 40000) if fa_rows_post(T, 1024) != fa_rows_post(T, 10**9)]
print("interactive budget=1024 vs monolithic: differing T count =", len(bad_int),
      "first few:", bad_int[:8])

# 2) dark lanes (judge/harvest budget 256)
bad_dark = [T for T in range(2, 40000) if fa_rows_post(T, 256) != fa_rows_post(T, 10**9)]
print("judge/harvest budget=256 vs monolithic: differing T count =", len(bad_dark),
      "range:", (min(bad_dark), max(bad_dark)) if bad_dark else None)
if bad_dark:
    T = bad_dark[0]
    print("  e.g. T=%d: budget256 FA rows=%s  monolithic FA rows=%s"
          % (T, sorted(fa_rows_post(T,256))[:3] + ['..'] + sorted(fa_rows_post(T,256))[-1:],
             sorted(fa_rows_post(T,10**9))[:3] or 'none'))
    print("  max |FA rows| under budget=256 over all T:", max(len(fa_rows_post(t,256)) for t in bad_dark))

# 3) which budgets are immune? (budget > WIN => first tick already crosses)
print("\nbudget -> immune over T in [2,40000)?")
for b in (16, 64, 128, 256, 384, 512, 513, 600, 1024, 2048):
    diff = any(fa_rows_post(T, b) != fa_rows_post(T, 10**9) for T in range(2, 40000))
    print("  budget=%5d  %s" % (b, "DIVERGES" if diff else "immune"))
