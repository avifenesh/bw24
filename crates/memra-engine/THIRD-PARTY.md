# Third-party notices — memra-engine

Some CUDA kernels under `cu/` are hand-ports ("vendored" per the mmq playbook — see each
file's header) of kernels from llama.cpp's ggml-cuda backend
(https://github.com/ggml-org/llama.cpp, snapshot c818263f2), restructured to be
ggml-decoupled and self-contained. Affected translation units:

- `cu/mmq_q8_0.cu`, `cu/mmq_q4_0.cu`, `cu/mmq_q45k.cu`, `cu/mmq_iq_experts.cu`,
  `cu/mmq_fp4.cu`, `cu/mmq_nvfp4_w4a8.cu`, `cu/mmq_nvfp4_f8f4.cu`, `cu/mmq_fp8_blk.cu`
  (mul_mat_q tile/MMA structure)
- `cu/fattn_vendor.cu` (flash_attn_ext MMA-f16 structure)

llama.cpp is distributed under the MIT license:

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

memra itself is MIT-licensed (repository-root LICENSE, Copyright (c) 2026 Avi Fenesh).
