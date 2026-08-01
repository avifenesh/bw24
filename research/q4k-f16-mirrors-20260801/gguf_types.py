#!/usr/bin/env python3
# Minimal GGUF v3 tensor-table dump: per-ggml-type tensor count + elem count (2D only flagged).
import struct, sys
from collections import defaultdict

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
         10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
         16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
         22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",
         30:"BF16",34:"NVFP4",35:"F8E4M3"}

def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_val(f, t):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8: rd_str(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n): skip_val(f, et)
    else: f.read(sizes[t])

path = sys.argv[1]
f = open(path, "rb")
magic = f.read(4)
assert magic == b"GGUF", magic
ver = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
for _ in range(n_kv):
    k = rd_str(f)
    t = struct.unpack("<I", f.read(4))[0]
    skip_val(f, t)
agg = defaultdict(lambda: [0, 0])   # type -> [n_tensors, n_elems]
rows = []
for _ in range(n_tensors):
    name = rd_str(f)
    nd = struct.unpack("<I", f.read(4))[0]
    ne = struct.unpack("<%dQ" % nd, f.read(8 * nd))
    ty = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    el = 1
    for d in ne: el *= d
    tn = TYPES.get(ty, str(ty))
    agg[tn][0] += 1
    agg[tn][1] += el
    rows.append((name, tn, ne, el))
print(path)
for tn, (cnt, el) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print("  %-8s %4d tensors  %12d elems  (%.2f GB @2B/w)" % (tn, cnt, el, el * 2 / 1e9))
if len(sys.argv) > 2 and sys.argv[2] == "-v":
    for name, tn, ne, el in rows:
        if tn in ("Q4_K", "Q6_K"):
            print("  %-40s %-6s %s" % (name, tn, ne))
