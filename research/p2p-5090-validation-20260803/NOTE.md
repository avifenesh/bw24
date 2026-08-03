# RTX 5090 PCIe P2P — claim validation (2026-08-03)

Claim under test: "NVIDIA disables P2P (cuDeviceCanAccessPeer) on GeForce RTX 5090 at the
driver level — like the 4090 — so NO consumer 5090 host will ever support P2P regardless of
motherboard/BIOS/IOMMU; only pro cards (RTX PRO 6000 Blackwell etc.) get P2P."

Our measurement: cuDeviceCanAccessPeer=0 both directions, vast.ai 2x5090 (Japan, EPYC,
topo NODE, IOMMU unknown), stock driver.

## VERDICT (three-way)

**(a) universal + (c) patchable.** On the STOCK driver, cuDeviceCanAccessPeer=0 on GeForce
5090 is a universal, deliberate driver-policy decision — NOT host-configuration-dependent
(IOMMU/ACS/BIOS do not change the advertisement) and NOT card-variant-dependent (all GB202
GeForce SKUs including 5090 D). No stock-driver 5090 host anywhere reports P2P=1; hunting
for one is futile. However, the claim's "NO consumer 5090 host will EVER support direct
peer-to-peer copies" is too strong: the hardware path works, and a maintained open-kernel-
module fork (aikitoria, descended from tinygrad/geohot) force-enables BAR1 P2P on 5090s
today, with multiple independent working reports since Aug 2025. It is unsupported and
unsuitable as a product dependency (details in §6).

---

## 1. Official NVIDIA policy statements

- **4090 (Ada), Feb 2023** — NVIDIA employee on the dev forums (quoted by Tom's Hardware,
  2023-02-22): "Feedback from Engineering is that **Peer to Peer is not supported on 4090.
  The applications/driver should not report this configuration as peer to peer capable. The
  reporting is being fixed and future drivers will report the following instead.**"
  Early 4090 drivers erroneously reported P2P capable and actually produced corrupted copies
  (Puget Systems repro); the "fix" was to report 0. Source:
  https://www.tomshardware.com/news/nvidia-confirms-geforce-cards-lack-p2p-support
  (original thread: forums.developer.nvidia.com/t/standard-nvidia-cuda-tests-fail-with-dual-rtx-4090-linux-box/233202).
- **5090 (Blackwell GB202 consumer), Mar 2025** — NVIDIA moderator (Yuki Ni) in
  "P2P issue using two RTX 5090 GPUs" (forums.developer.nvidia.com/t/326776, driver
  570.124.06, CUDA 12.8): "**RTX 50-series are GeForce GPU series, and p2p is not supported
  for GPU<->GPU P2P. For GPU lists that can support p2p, there are mainly 2 series, Quadro
  RTX [now RTX PRO] and Data Center GPUs.**"
- No GeForce driver release note grants P2P to any 40/50-series card; the policy has been
  stable since Turing lost PCIe P2P (Turing had it only via NVLink bridge; Ampere/Ada/
  Blackwell GeForce: none).

**Sub-verdict: confirmed. Driver-level product segmentation, applies to the whole GeForce
line including all 5090s.**

## 2. The tinygrad/geohot patch saga — and its 5090 successor

What NVIDIA "blocks": GeForce silicon lacks/disables the legacy MAILBOXP2P hardware
interface that the driver's P2P path uses, and the driver refuses to advertise P2P. What
the patch does (tinygrad/open-gpu-kernel-modules, branch 550.90.07-p2p README, Apr 2024):
force-enable **BAR1P2P** — the H100-era mode that maps the entire VRAM into a large BAR1
and does peer copies by DMA-writing the other GPU's physical BAR addresses — by bypassing
the HAL and calling GH100 methods (`kbusEnableStaticBar1Mapping_GH100`) directly, rewriting
`GMMU_APERTURE_PEER` -> `GMMU_APERTURE_SYS_NONCOH` and putting the BAR1 base in
`fabricBaseAddress`. Quote: "This is not a hack, this is using PCIe according to the spec."
Requirements: **large BAR (resizable BAR) + IOMMU off (or passthrough `iommu=pt`)**; on
some boards also ACS off. Measured on 6x4090: ~50 GB/s bidir per pair, NCCL-compatible.
Source: https://github.com/tinygrad/open-gpu-kernel-modules/tree/550.90.07-p2p

**5090 state (current, 2026):**
- tinygrad branches exist through `570.148.08-p2p` ("allows using P2P on 4090/5090"), but
  multiple users hit `cudaErrorMapBufferObjectFailed` on 5090 pairs even with IOMMU+ACS off
  (tinygrad issues #35 (2025-03-27), #44 (2025-06-15) — note in #44 canAccessPeer returned
  1/1 under the patched driver; the *enable* failed, i.e., 570-era tinygrad branch was
  broken for 5090, not the concept).
- **aikitoria/open-gpu-kernel-modules** (fork of the tinygrad work) fixed 5090 support and
  is actively maintained: p2p branches for 570.86/570.124/570.133/570.153, 575.51/575.64,
  580.76/580.82/580.95/580.105, 590.44/**590.48.01**, 595.45/595.58/595.71, and
  **610.43.03** (newest, 2026). 590.48.01-p2p README: "NVIDIA driver 590.48.01 with P2P for
  4090 **and 5090** … modifies the kernel driver to force enable BAR1 P2P mode … IOMMU
  virtualization must be disabled to use the patch … On some systems you might additionally
  need to disable ACS. On some systems resizable BAR might be unavailable. **4090s come up
  with large BAR by default, but 5090s don't**" (i.e., 5090 needs Resizable BAR/Above-4G
  enabled in BIOS to get the 32 GiB BAR1; the silicon supports it — stock 5090s already show
  `bar size (MiB): 32768` in NVIDIA's own forum GDS dump).
  Source: https://github.com/aikitoria/open-gpu-kernel-modules
- First public multi-5090 success: r/LocalLLaMA 2025-08-30 (panchovix), "Patched P2P NVIDIA
  driver now works with multiple 5090s (and possibly blackwell 2.0 in general). Also works
  for 4090/3090" — p2pBandwidthLatencyTest passing with each 5090 at PCIe 5.0 x8/x8, using
  the aikitoria fork after the tinygrad 570.148.08 branch failed. Same user later ran a
  7-GPU mixed rig (2x5090 + 2x4090 + A6000/A40/3090) on the P2P driver (r/LocalLLaMA,
  2026-01-18). A cross-arch caveat from the multi-GPU community (r/LocalLLaMA 2026-02):
  RTX 6000->5090 direction showed corruption on current drivers and had to be masked.
- Write-up with velocity numbers: smcleod.net 2026-02-25 "Patching NVIDIA's driver and vLLM
  to enable P2P on consumer GPUs" — patched 590.48.01, `iommu=pt`, dual 3090: 10–30%
  end-to-end throughput gain in vLLM TP-2; also documents that vLLM's pynvml-based
  capability check must be bypassed because userspace still reports "unsupported."

**Sub-verdict: the 4090-era patch exists, evolved, and works on 5090s in 2026 via the
aikitoria fork (driver 570→610 branches). Requires open kernel modules, Resizable BAR
enabled (5090 is not large-BAR by default), IOMMU off/pt, sometimes ACS off.**

## 3. Measured evidence on 2x/4x 5090 rigs (stock driver)

- forums.developer.nvidia.com/t/326776 (Mar 2025): 2x5090, TRX50 AI TOP, Threadripper
  7960X, driver 570.124.06, **IOMMU: disabled**, BAR1 32 GiB each — P2P unsupported; NVIDIA
  staff confirms policy (quote in §1). Key: IOMMU already off, ReBAR already giving 32G
  BAR1, still no P2P → kills hypothesis (b) host-configuration-dependent.
- discuss.vllm.ai (Nov 2025): 2x5090 on EPYC — vLLM: "**custom allreduce is disabled
  because your platform lacks GPU P2P capability or P2P test failed.**"
- vllm-project/vllm#14628 + NVIDIA/nccl#1637 (Mar 2025): 2x5090 TP-2 failures; the NCCL
  crash itself was an NCCL bug fixed in NCCL 2.26.2 (per issue resolution) — but P2P stayed
  unavailable; NCCL just falls back cleanly.
- Our vast.ai Japan EPYC measurement (canAccessPeer=0/0) matches every public stock-driver
  report. **Zero** public reports of stock-driver 5090 P2P=1 were found (searched NVIDIA
  forums, GitHub, Reddit, vast.ai mentions).
- Bounce-path context: with P2P disabled, 5090-pair D2D through host measures ~21–25 GB/s
  unidirectional on Gen4/Gen5-x8-ish topologies (tinygrad #35: 21.2/21.4; #44: 24.6/24.7
  GB/s), vs ~50+ GB/s with patched P2P on Gen5.

**Sub-verdict: (a) confirmed empirically; no host, BIOS, or IOMMU combination flips the
stock-driver answer.**

## 4. What llama.cpp / vLLM / NCCL actually do on consumer 5090s

- **NCCL**: auto-detects peer capability; on GeForce 5090 pairs it selects the **SHM
  transport (staged through host shared memory)**, no env var needed. `NCCL_P2P_DISABLE=1`
  is only a manual override. Early NCCL 2.25 had a 5090 crash (nccl#1637), fixed in 2.26.2.
- **vLLM**: on P2P-less platforms it disables its custom allreduce (warning quoted above)
  and uses NCCL, which host-stages. Even with the patched driver, vLLM's pynvml check says
  "unsupported" and must be short-circuited (smcleod's sed hack on
  `vllm/platforms/cuda.py`).
- **llama.cpp** (docs/multi-gpu.md, current master): default is layer (pipeline) split;
  `-sm tensor` uses NCCL when built `-DGGML_CUDA_NCCL=ON`. Direct CUDA P2P is **opt-in**
  via `GGML_CUDA_P2P=1` and the docs warn: "P2P requires driver support (**usually
  restricted to workstation/datacenter GPUs**) and may cause crashes or corrupted outputs
  on some motherboards or BIOS configurations (e.g. when IOMMU is enabled)."
- **exllama**: no P2P dependency; per-GPU layer split over host.

**Sub-verdict: every mainstream engine already runs consumer 5090 multi-GPU through
host-staged paths (NCCL SHM / D2H→H2D). P2P is a fast-path they enable only when the
driver advertises it.**

## 5. RTX PRO 6000 Blackwell (GB202 pro variant)

- Corroborated: PyTorch `torch.cuda.can_device_access_peer()` returns **True both
  directions** on 2x RTX PRO 6000 Blackwell Max-Q, WRX90, driver 580.95.05, CUDA 13.0
  (Level1Techs, Dec 2025). Matches our G7e/AWS experience (RTX PRO 6000 Blackwell Server
  Edition, P2P/GPUDirect working). AWS G7e launch materials advertise multi-GPU inference
  on these parts.
- Caveats even on pro silicon: that same L1T rig hit an NCCL 2.28.9 P2P hang (config/NCCL
  issue, not capability); and under ESXi passthrough the card doesn't advertise PCIe ATS,
  blocking the hypervisor's P2P path (NVIDIA forum t/362222, Mar 2026) — bare metal is the
  safe assumption for pro-card P2P.

**Sub-verdict: confirmed — the pro GB202 advertises and does P2P on stock drivers; the
GeForce GB202 does not. Same silicon, driver/firmware segmentation.**

## 6. Production 2x5090 box we control: is there a path to P2P=1?

Yes — unsupported: **aikitoria/open-gpu-kernel-modules**, branch matching the installed
driver (currently up to 610.43.03-p2p; 590.48.01-p2p is the widely-replicated one).
Recipe: install matching .run driver with open kernel modules → `./install.sh` from the
fork → BIOS: Resizable BAR + Above-4G ON (mandatory on 5090; not large-BAR by default),
ACS off; kernel: `iommu=pt` (or off). Verify with `nvidia-smi topo -p2p r` (OK vs NS) and
simpleP2P/p2pBandwidthLatencyTest.

Costs/risks, honestly stated:
1. **Security**: IOMMU pt/off means unrestricted DMA between devices — fine for a closed
   appliance, unacceptable multi-tenant.
2. **Maintenance treadmill**: kernel-module fork must be rebuilt/rebased per driver bump;
   you are pinned to versions the fork tracks (it does track actively — 570→610 through
   2026 — but it's one volunteer lineage).
3. **Userspace fights you**: NVML/pynvml still report "unsupported" → vLLM (and anything
   using capability checks) needs patching; our own runtime would read
   cuDeviceCanAccessPeer=1 correctly since the check happens in the driver.
4. **Correctness fragility**: tinygrad-era cache-flush uncertainty ("not sure all the cache
   flushes are right"); 5090-specific enable bugs existed for months (issues #35/#44);
   cross-arch pairs showed one-directional corruption reports. Any patched-P2P deployment
   needs our own byte-exactness gate (kernel-check/argmax battery) run under the patched
   driver before trust.
5. **No vendor support**, no warranty coverage for the config.

## Bottom line for PP-2 transport

- The vast.ai result is the universal stock-driver truth, not a host quirk. **Stop hunting
  for P2P-capable 5090 hosts — they do not exist on stock drivers.** Rented/vast.ai boxes
  can never be assumed patchable (no BIOS/kernel control).
- **Build the host-staged (D2H→H2D pinned-bounce) PP-2 transport as the product default.**
  That is exactly what NCCL/vLLM/llama.cpp ship on this hardware; budget ~20–25 GB/s
  effective pair bandwidth on typical topologies (better on clean Gen5 x16 with pinned
  double-buffering) and hide it behind PP overlap — PP-N already tolerates thin links far
  better than TP.
- **Keep patched P2P as an optional, flag-gated fast path for owned/appliance boxes only**
  (aikitoria fork + ReBAR + iommu=pt), promoted only after our correctness battery passes
  under the patched driver. Never make product function depend on it.
- Do not abandon PP-2: host-staged PP-2 is viable; the pro-card path (RTX PRO 6000
  Blackwell) remains the supported P2P option where the BOM allows.

## Source index

1. Tom's Hardware, 2023-02-22 — NVIDIA engineering statement, 4090 P2P unsupported.
   https://www.tomshardware.com/news/nvidia-confirms-geforce-cards-lack-p2p-support
2. NVIDIA dev forums t/326776, Mar 2025 — 5090 pair, driver 570.124.06, NVIDIA staff: 50-series GeForce = no P2P; IOMMU disabled on that host.
   https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776
3. tinygrad/open-gpu-kernel-modules 550.90.07-p2p README — BAR1P2P mechanism, large-BAR + IOMMU-off requirement, 4090 numbers.
   https://github.com/tinygrad/open-gpu-kernel-modules/tree/550.90.07-p2p
4. aikitoria/open-gpu-kernel-modules — maintained 4090+5090 P2P branches through 610.43.03 (2026); 590.48.01-p2p README quoted.
   https://github.com/aikitoria/open-gpu-kernel-modules
5. r/LocalLLaMA 2025-08-30 (panchovix) — first multi-5090 patched-P2P success, Gen5 x8/x8.
   https://www.reddit.com/r/LocalLLaMA/comments/1n3qcqn/
6. tinygrad issues #35 (2025-03-27), #44 (2025-06-15) — 5090 enable failures on tinygrad 570 branches; bounce-path 21–25 GB/s numbers.
   https://github.com/tinygrad/open-gpu-kernel-modules/issues/44
7. smcleod.net 2026-02-25 — patched 590.48.01 + vLLM bypass, 10–30% TP-2 gain, iommu=pt recipe.
   https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/
8. NVIDIA/nccl#1637 + vllm-project/vllm#14628 (Mar 2025) — NCCL 2.25 5090 bug (fixed 2.26.2); vLLM custom-allreduce disabled without P2P.
   https://github.com/NVIDIA/nccl/issues/1637
9. llama.cpp docs/multi-gpu.md (master) — GGML_CUDA_P2P opt-in, "usually restricted to workstation/datacenter GPUs", NCCL tensor mode.
   https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
10. Level1Techs, Dec 2025 — 2x RTX PRO 6000 Blackwell Max-Q: can_device_access_peer True/True on driver 580.95.05.
    https://forum.level1techs.com/t/dual-rtx-pro-6000-blackwell-max-q-how-to-make-p2p-nccl-work/242403
11. NVIDIA dev forums t/362222, Mar 2026 — PRO 6000 Blackwell lacks ATS advertisement, blocks ESXi P2P path (virtualization caveat).
    https://forums.developer.nvidia.com/t/rtx-pro-6000-blackwell-does-not-advertise-pcie-ats-blocking-esxis-p2p-path/362222
