# Security Policy

bw24 loads GGUF and safetensors model files by memory-mapping them and parsing headers/tensor
metadata directly — a malicious or corrupted model file is untrusted input to a memory-unsafe
surface (CUDA kernels, raw byte repacking, mmap'd tensor reads). Treat any parser bug reachable
from a model file as a security issue, not just a correctness bug.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Instead:

- Use [GitHub's private vulnerability reporting](https://github.com/avifenesh/bw24/security/advisories/new) for this repo, or
- Email the maintainer directly (see the GitHub profile for contact) with a clear repro (a minimal
  crafted GGUF/safetensors file or a description of the malformed field, and the crash/UB
  observed).

Include:
- The specific file/loader path involved (`bw24-gguf`, the safetensors loader in `bw24-engine`, etc.)
- A minimal reproducing input if possible
- What you observed (crash, OOB read/write, panic, incorrect output) and how you triggered it

## Scope

In scope: memory-safety and parsing issues triggerable by a crafted model file (GGUF header
fields, tensor metadata, safetensors JSON header, NVFP4/quant block layouts). Also in scope:
`bw24-server`'s HTTP-facing request handling.

Out of scope: this is a single-user, single-machine research engine with no built-in
authentication or multi-tenant isolation — running `bw24-server` exposed to untrusted network
clients without your own auth/network controls in front of it is a deployment choice, not a bw24
vulnerability.

## Response

This is a small research project without a dedicated security team — expect an acknowledgment
within a reasonable window, not an SLA. Fixes land as normal commits once triaged; there is no
separate security-release channel at this project's current size.
