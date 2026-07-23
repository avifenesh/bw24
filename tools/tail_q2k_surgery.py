#!/usr/bin/env python3
"""Tail-Q2_K overlay surgery: rebuild the Hy3 Layer-103.5 runtime with the frozen demote set
re-quantized to Q2_K from the compact BF16 store; every kept expert's bytes are copied
verbatim from the existing overlay. Output: a new runtime dir (manifest + experts/), plan
updated, hashes recomputed. No imatrix (sidecars are G7e-only) — documented; the 115-question
screen is the quality gate."""
import ctypes, json, os, struct, sys, hashlib
import numpy as np

SRC_RT = '/home/avifenesh/.local/share/bw24-models/hy3-layer103p5-dual-nvme'
OUT_RT = '/data/ai-ml/hf-models/hy3-tailq2k-runtime'
BF16 = '/data/ai-ml/hf-models/hy3-tail-bf16'
DEMOTE = '/home/avifenesh/projects/bw24-hy3lane/research/per-expert-quant/tail-q2k-demote-set.json'
GGML = os.path.expanduser('~/projects/llama.cpp/build/bin/libggml-base.so')
GGML_TYPE_Q2_K = 10

lib = ctypes.CDLL(GGML)
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
lib.ggml_quantize_init(GGML_TYPE_Q2_K)

demote = {tuple(x) for x in json.load(open(DEMOTE))}
manifest = json.load(open(f'{SRC_RT}/manifest.json'))
tensors = manifest['tensors']

# BF16 tensor lookup: name -> (compact file, header meta)
bf16_index = {}
for fn in sorted(os.listdir(BF16)):
    if not fn.endswith('.safetensors'):
        continue
    path = os.path.join(BF16, fn)
    with open(path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen))
    for name, meta in header.items():
        bf16_index[name] = (path, 8 + hlen, meta)
print(f'bf16 tensors indexed: {len(bf16_index)}', flush=True)

def quantize_q2k(name, ne):
    path, base, meta = bf16_index[name]
    o0, o1 = meta['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + o0)
        raw = f.read(o1 - o0)
    bf = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    f32 = np.ascontiguousarray(bf.view(np.float32))
    n_per_row, nrows = ne[0], ne[1]
    assert f32.size == n_per_row * nrows, (name, f32.size, ne)
    row_bytes = n_per_row // 256 * 84
    out = ctypes.create_string_buffer(row_bytes * nrows)
    written = lib.ggml_quantize_chunk(
        GGML_TYPE_Q2_K,
        f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out, 0, nrows, n_per_row, None)
    assert written == row_bytes * nrows, (name, written, row_bytes * nrows)
    return bytes(out.raw), row_bytes

os.makedirs(f'{OUT_RT}/experts', exist_ok=True)
# group tensors by output file, preserving offset order
by_file = {}
for name, meta in tensors.items():
    by_file.setdefault(meta['file'], []).append((meta['offset'], name))

proj_map = {'ffn_gate_exps': 'gate_proj', 'ffn_up_exps': 'up_proj', 'ffn_down_exps': 'down_proj'}
new_tensors = {}
requantized = 0
for file_name in sorted(by_file):
    entries = sorted(by_file[file_name])
    src_path = os.path.join(SRC_RT, file_name)
    out_path = os.path.join(OUT_RT, file_name)
    cursor = 0
    with open(src_path, 'rb') as src, open(out_path, 'wb') as out:
        for _, name in entries:
            meta = dict(tensors[name])
            parts = name.split('.')  # blk.L.ffn_X_exps.E.weight
            layer, kind, expert = int(parts[1]), parts[2], int(parts[3])
            if (layer, expert) in demote and meta['qtype'] != 'Q2_K':
                hf_name = f'model.layers.{layer}.mlp.experts.{expert}.{proj_map[kind]}.weight'
                data, row_bytes = quantize_q2k(hf_name, meta['ne'])
                meta['qtype'] = 'Q2_K'
                meta['row_bytes'] = row_bytes
                meta['bytes'] = len(data)
                requantized += 1
            else:
                src.seek(meta['offset'])
                data = src.read(meta['bytes'])
            meta['offset'] = cursor
            out.write(data)
            cursor += len(data)
            new_tensors[name] = meta
    print(f'{file_name}: {cursor/1e9:.2f} GB', flush=True)

manifest['tensors'] = new_tensors
# plan: rewrite assignments so every demoted expert projection reads Q2_K
plan = manifest['plan']
new_assignments = []
for entry in plan['assignments']:
    layer = entry['layer']
    keep, moved = [], []
    for expert in entry['experts']:
        (moved if (layer, expert) in demote and entry['qtype'] != 'Q2_K' else keep).append(expert)
    if keep:
        new_assignments.append({**entry, 'experts': keep})
    if moved:
        new_assignments.append({'layer': layer, 'projections': entry['projections'],
                                'qtype': 'Q2_K', 'experts': moved})
plan['assignments'] = new_assignments
plan['recipe'] = plan.get('recipe', '') + ' + tail-q2k-demotion-v1 (frequency-cold non-resident experts -> Q2_K, no imatrix)'
canonical = json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()
manifest['plan_canonical_sha256'] = hashlib.sha256(canonical).hexdigest()
manifest['payload_bytes'] = sum(t['bytes'] for t in new_tensors.values())
json.dump(manifest, open(f'{OUT_RT}/manifest.json', 'w'))
# carry non-expert runtime pieces
for extra in os.listdir(SRC_RT):
    src = os.path.join(SRC_RT, extra)
    dst = os.path.join(OUT_RT, extra)
    if extra in ('manifest.json', 'experts') or os.path.exists(dst):
        continue
    if os.path.isfile(src):
        os.link(src, dst) if os.stat(src).st_dev == os.stat(OUT_RT).st_dev else __import__('shutil').copy2(src, dst)
print(f'DONE requantized={requantized} payload={manifest["payload_bytes"]/1e9:.2f} GB', flush=True)
