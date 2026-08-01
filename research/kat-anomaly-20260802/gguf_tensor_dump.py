#!/usr/bin/env python3
"""Dump GGUF tensor infos (name, dims, ggml_type, bytes) from the header only.
Usage: gguf_tensor_dump.py <file> [--summary]
--summary groups by (tensor-class, qtype) with counts + total bytes."""
import struct, sys
from collections import defaultdict

GGML = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
        10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',
        16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',
        22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',
        30:'BF16',34:'F8_E4M3',39:'NVFP4'}
# block size (elems), type size (bytes) for byte math
BLK = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),
       10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),
       16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),
       22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),29:(256,56),
       30:(1,2),34:(1,1),39:(16,9)}

def rstr(f):
    n=struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def skip_val(f, t):
    if t==8: rstr(f); return
    if t==9:
        et=struct.unpack('<I', f.read(4))[0]
        n=struct.unpack('<Q', f.read(8))[0]
        for _ in range(n): skip_val(f, et)
        return
    sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
    f.read(sz)

def main():
    path=sys.argv[1]; summary='--summary' in sys.argv[2:]
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'
        struct.unpack('<I', f.read(4))
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        for _ in range(n_kv):
            rstr(f); t=struct.unpack('<I', f.read(4))[0]; skip_val(f, t)
        rows=[]
        for _ in range(n_tensors):
            name=rstr(f)
            nd=struct.unpack('<I', f.read(4))[0]
            dims=struct.unpack(f'<{nd}Q', f.read(8*nd))
            ty=struct.unpack('<I', f.read(4))[0]
            struct.unpack('<Q', f.read(8))  # offset
            elems=1
            for d in dims: elems*=d
            bs,tb=BLK.get(ty,(1,0))
            nbytes=elems//bs*tb
            rows.append((name, dims, GGML.get(ty,f'?{ty}'), nbytes))
    if summary:
        g=defaultdict(lambda:[0,0])
        for name,dims,ty,nb in rows:
            if name.startswith('blk.'):
                cls='.'.join(name.split('.')[2:])
            else:
                cls=name
            g[(cls,ty)][0]+=1; g[(cls,ty)][1]+=nb
        tot=0
        for (cls,ty),(n,nb) in sorted(g.items(), key=lambda kv:-kv[1][1]):
            print(f'{cls:42s} {ty:8s} x{n:4d} {nb/1e9:9.3f} GB')
            tot+=nb
        exps=sum(nb for (cls,ty),(n,nb) in g.items() if '_exps.' in cls)
        print(f'{"TOTAL":42s} {"":8s} {"":5s} {tot/1e9:9.3f} GB   (experts {exps/1e9:.3f} GB, non-expert {(tot-exps)/1e9:.3f} GB)')
    else:
        for name,dims,ty,nb in rows:
            print(f'{name:60s} {str(list(dims)):24s} {ty:8s} {nb:>12d}')

main()
