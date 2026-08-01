#!/usr/bin/env python3
"""Parse GGUF metadata KVs from a (possibly truncated) file. Stops at first
struct error (truncation) and prints what it got. Usage: gguf_meta.py <file> [key-filter...]"""
import struct, sys, json

TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}

class R:
    def __init__(self, f): self.f=f
    def read(self, n):
        b=self.f.read(n)
        if len(b)<n: raise EOFError
        return b
    def scalar(self, t):
        fmt=FMT[t]; return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]
    def string(self):
        n=struct.unpack('<Q', self.read(8))[0]
        return self.read(n).decode('utf-8', errors='replace')
    def value(self, t):
        if t==8: return self.string()
        if t==9:
            et=struct.unpack('<I', self.read(4))[0]
            n=struct.unpack('<Q', self.read(8))[0]
            vals=[self.value(et) for _ in range(n)]
            return vals
        return self.scalar(t)

def main():
    path=sys.argv[1]
    filters=sys.argv[2:]
    out={}
    with open(path,'rb') as f:
        r=R(f)
        magic=r.read(4)
        assert magic==b'GGUF', magic
        ver=struct.unpack('<I', r.read(4))[0]
        n_tensors=struct.unpack('<Q', r.read(8))[0]
        n_kv=struct.unpack('<Q', r.read(8))[0]
        out['__version']=ver; out['__n_tensors']=n_tensors; out['__n_kv']=n_kv
        try:
            for _ in range(n_kv):
                k=r.string()
                t=struct.unpack('<I', r.read(4))[0]
                v=r.value(t)
                if isinstance(v,list) and len(v)>16:
                    v=f'<array len={len(v)} head={v[:4]}>'
                out[k]=v
        except (EOFError, struct.error):
            out['__truncated']=True
    for k,v in out.items():
        if filters and not any(fl in k for fl in filters) and not k.startswith('__'): continue
        if isinstance(v,str) and len(v)>400: v=v[:400]+f'...<{len(v)} chars>'
        print(f'{k} = {v!r}')

main()
