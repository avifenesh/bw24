#!/usr/bin/env python3
"""Minimal GGUF header parser: prints KV metadata + tensor names/shapes/types.
Tolerates truncated files (range-downloaded heads)."""
import sys, struct

TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
GGML = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16'}

class R:
    def __init__(s, f): s.f=f
    def read(s, n):
        b = s.f.read(n)
        if len(b) < n: raise EOFError
        return b
    def u32(s): return struct.unpack('<I', s.read(4))[0]
    def u64(s): return struct.unpack('<Q', s.read(8))[0]
    def i32(s): return struct.unpack('<i', s.read(4))[0]
    def i64(s): return struct.unpack('<q', s.read(8))[0]
    def f32(s): return struct.unpack('<f', s.read(4))[0]
    def f64(s): return struct.unpack('<d', s.read(8))[0]
    def s_(s):
        n = s.u64(); return s.read(n).decode('utf-8', 'replace')
    def val(s, t):
        if t==0: return s.read(1)[0]
        if t==1: return struct.unpack('<b', s.read(1))[0]
        if t==2: return struct.unpack('<H', s.read(2))[0]
        if t==3: return struct.unpack('<h', s.read(2))[0]
        if t==4: return s.u32()
        if t==5: return s.i32()
        if t==6: return s.f32()
        if t==7: return bool(s.read(1)[0])
        if t==8: return s.s_()
        if t==9:
            et = s.u32(); n = s.u64()
            out = [s.val(et) for _ in range(n)]
            return out
        if t==10: return s.u64()
        if t==11: return s.i64()
        if t==12: return s.f64()
        raise ValueError(f'type {t}')

def main(path):
    f = open(path, 'rb'); r = R(f)
    magic = r.read(4)
    assert magic == b'GGUF', magic
    ver = r.u32(); n_tensors = r.u64(); n_kv = r.u64()
    print(f'# {path}: GGUF v{ver}, tensors={n_tensors}, kv={n_kv}')
    try:
        for _ in range(n_kv):
            k = r.s_(); t = r.u32(); v = r.val(t)
            if isinstance(v, list):
                if len(v) > 24 and k not in ('step3p5.rope.dimension_count',):
                    show = f'[{len(v)} items] first24={v[:24]}'
                else:
                    show = v
            elif isinstance(v, str) and len(v) > 400:
                show = v[:400] + f'...({len(v)} chars)'
            else:
                show = v
            print(f'KV {k} ({TYPES.get(t,t)}) = {show}')
        print('--- tensors ---')
        for _ in range(n_tensors):
            name = r.s_(); nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            gt = r.u32(); off = r.u64()
            print(f'T {name} {dims} {GGML.get(gt, gt)}')
    except EOFError:
        print('[truncated head — stopped here]')

if __name__ == '__main__':
    main(sys.argv[1])
