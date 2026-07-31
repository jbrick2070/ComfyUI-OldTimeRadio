# Low-VRAM video under ComfyUI -- what an 8 GB (and 6 GB) tier can actually do

Operator question, 2026-07-31: *"deep research on what 8gb model for video we can
support -- maybe that means we load 3gb of such and such, save that, offload
them, load 6gb of that, but all while remaining under 8gb... how do we support
GPUs at 8gb and under under comfy, and replace wan with that? Or maybe our
research is wrong."*

**Short answer: the operator's instinct is right, our estimator is structurally
wrong, and we should KEEP Wan 2.2 TI2V-5B while changing how we run it. The
single most consequential finding is that our GGUF stack silently opts OUT of
the exact ComfyUI mechanism that makes 8 GB work.**

Three independent analyses converged on the estimator finding: this window's own
reasoning, a live web-research pass (2026-07-31), and a ChatGPT pass the operator
ran in parallel. Sources are cited inline; where evidence is thin it says so.

---

## 1. The estimator is the wrong SHAPE, not the wrong constant

Ours (`motion_common.py:263-265`):

    vram = overhead(7000 MB) + per_frame(185 MB) * frames

That is a **co-resident** model -- it assumes everything is loaded at once, so
costs ADD. The real low-VRAM technique is staged: text-encode, release, sample,
release, decode. Peak is the **maximum over stages**, not the sum.

The correct shape:

    peak_job = max over stages(
        resident_weights
      + live_latents_and_outputs
      + activations(H, W, frames, batch)
      + attention/backend workspace
      + cast / dequantization scratch
      + allocator slack
    ) + safety_reserve

Minimally, for our pipeline:

    peak_job = max(text_encode, image_encode, sample, vae_decode) + reserve

### The operational form (adopted from the operator's second ChatGPT pass)

Per stage `s`, sum ONLY that stage's residency:

    peak_job = max over s( W_s + L_s + A_s + WS_s + S_s + D_s ) + R

| term | meaning |
|---|---|
| `W_s` | resident weights actually on GPU in that stage |
| `L_s` | live latents / feature maps / outputs (HxWxframes, batch) |
| `A_s` | activations + scheduler buffers for the current step |
| `WS_s` | backend workspace (cuDNN/Triton scratch, attention workspace) |
| `S_s` | cast / dequantization scratch (GGUF -> fp16, fp8 staging) |
| `D_s` | allocator slack and fragmentation not yet reclaimed |
| `R` | safety reserve -- 0.5-1.0 GB on a consumer card |

This is the shape to implement. It maps cleanly onto our config knobs:
resolution and frame count drive `L_s`/`A_s`; quantization level and device
placement drive `W_s`/`S_s`; VRAM mode and reserve drive `D_s`/`R`.

Note `S_s` is not decorative for us: **GGUF dequant scratch is a real per-stage
cost we currently model as zero**, and it is one plausible home for part of the
unexplained gap.

**The proof that addition is wrong:** ComfyUI's own Wan 2.2 workflow totals
roughly 18 GB of model files and its official docs state *"The Wan2.2 5B version
should fit well on 8GB vram with the ComfyUI native offloading"*
(https://docs.comfy.org/tutorials/video/wan/wan2_2). 18 GB of weights running in
8 GB is only possible if costs do not add.

This also explains the ~2 GB gap this window flagged yesterday between the
modelled 7,000 MB overhead and the ~4,980 MB of resident weights. At
1280x704x121 the decoded RGB frames alone are ~1.22 GiB as float32 before VAE
workspace, allocator caching, temporary casts and dequant buffers. `per_frame *
frames` is a real term -- but it belongs to ONE stage (decode), not to the whole
job.

---

## 2. THE FINDING THAT MATTERS MOST: our GGUF opts out of Dynamic VRAM

ComfyUI v0.16.0 (5 Mar 2026) made **Dynamic VRAM** (`comfy-aimdo`) default-on for
NVIDIA/non-WSL: a custom PyTorch allocator that streams weights JIT from pinned
host memory, so total weight size stops being a hard gate
(https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local).

**GGUF does not participate.** `ComfyUI-GGUF` defines
`class GGUFModelPatcher(comfy.model_patcher.ModelPatcher)` -- NOT
`ModelPatcherDynamic` -- and `clone()` force-reassigns `__class__` back. Confirmed
as a known limitation in Comfy-Org/ComfyUI#13953 (18 May 2026): *"custom GGUF
loaders using GGUFModelPatcher(ModelPatcher) do not use ModelPatcherDynamic, so
GGUF UNet/CLIP remain on legacy lowvram behavior."*

**We ship `Wan2.2-TI2V-5B-Q5_K_M.gguf` + `umt5-xxl-encoder-Q5_K_M.gguf`. Our
entire 8 GB tier is on the 2025 code path**, while the official 8 GB workflow
that "fits well" uses **fp8 scaled safetensors** -- which do get the dynamic
patcher. This inverts the usual "GGUF is smaller so GGUF is better at low VRAM"
intuition on ComfyUI >= 0.16.

Corroborating symptom, measured: on an RTX 3070 8 GB, GGUF occupancy fell from
7.4/8.0 GB to ~5 GB with a matching slowdown after an async-offload change,
because GGUF lacks the pinning path (Comfy-Org/ComfyUI#11081).

**This is a hypothesis with a verified mechanism but no isolating benchmark.** No
one has published GGUF-vs-fp8 under Dynamic VRAM on 8 GB. It is a one-afternoon
A/B and it may be our single biggest structural win.

**Also: `--lowvram` is inert.** ComfyUI's own `cli_args.py`: *"Doesn't do anything
if dynamic vram is enabled."* `text_encoder_device()` short-circuits on
`aimdo_enabled` and returns the GPU regardless. Every 2026 guide still telling
8 GB users to add `--lowvram` is repeating stale advice.

---

## 3. The free money: cache the text embeddings

`umt5-xxl-encoder-Q5_K_M` is **3.861 GiB -- larger than the 3.549 GiB UNET.**

kijai's `WanVideoTextEncodeCached` already solves this, and `use_disk_cache`
defaults to **True** on that node. Its docstring: *"This node loads and completely
unloads the T5 after done, leaving no VRAM or RAM imprint. If prompts have been
cached before T5 is not loaded at all."*

Measured encode cost being avoided: on an RTX 3070 8 GB, umt5 Q4 GGUF text
encoding took **~3.5 minutes** in the good case and regressed to 27+ minutes in a
bad one (Comfy-Org/ComfyUI#11081). Per leg. For text we have already encoded.

**OTR is close to the ideal case for this** -- a bounded, highly repetitive prompt
set across episodes. Cache once and 3.9 GiB plus minutes per leg simply leave the
budget. Caveat: this is a wrapper-level node, not ComfyUI core; there is an open
request for path-addressable save/load (WanVideoWrapper#1794).

---

## 4. Keep Wan 2.2 TI2V-5B. Nothing has displaced it.

**Wan 2.5/2.6/2.7 are not open weights.** Verified against the `Wan-AI` HF org
directly: the newest repos are Wan-Dancer-14B (17 Jul 2026) and Wan2.2-Animate.
Every "Wan 2.6/2.7 local" article traces to SEO farms; treat as fiction.

**Do NOT switch to LTX-2.3.** It is 22B with a **Gemma-3-12B** text encoder --
~20 GB of weights minimum (DiT Q2_K 8.28 GB + encoder UD-Q4_K_XL 7.43 GB +
connectors 2.31 + VAEs 1.81). Community floor is 12 GB; Lightricks' own ComfyUI
prerequisites say 32 GB+. And ComfyUI's own activation estimator rates it
**LTXV 5.5 vs WAN21_T2V ~1.38** -- roughly 4x more activation-hungry per latent.
Wan's 16x16x4 VAE is precisely why it is cheap.

Other 2026 candidates, all rejected for now: HunyuanVideo-1.5 (VAE decode OOMs a
12 GB card at 121 frames), Motif-Video 2B (great Apache-2.0 numbers but its own
card says ComfyUI *"currently requires High VRAM mode"*), MobileWan (a pruned,
3-step-distilled TI2V-5B aimed at exactly our problem -- but BF16 only, no GGUF,
no ComfyUI node, and a Qualcomm Responsible-AI licence unsuited to a shipped
pack). **MobileWan is the one to watch.**

**The honest case against the 5B:** the one head-to-head found on 8 GB hardware
(RTX 4060, lilting.ch, 6 Mar 2026) had an offloaded **14B** beating it on both
quality and wall clock -- the 5B rows read "distorted", "minimal motion". The
reason is structural: **there is no Lightning/step-distill LoRA for the 5B** --
every `lightx2v/Wan2.2-Lightning` variant is A14B. So the 5B pays full 20-30
steps while 14B users run 4-6.

Realistic 8 GB envelope for the 5B: 480x480-832x480, 33-81 frames, ~2-5 s of
video, ~95-180 s per clip on a 4060-class card. 720p is the ceiling and is
generally not reached.

---

## 5. 6 GB

Viable, miserable, and there is exactly ONE measured data point: RTX 3050 6 GB,
Wan 2.2 14B Q3_K/Q4_K + umt5 Q3_K_M, 512x512, **16 frames**, 4 steps, ~9 GB
offloaded to RAM -> **~4.6 minutes** (Cordux/ComfyUI-Wan2.2-workflow). That is
roughly 4.6 min per second of output.

Our own stack at Q3_K_M would be 2.55 + 3.06 + 1.41 = 7.0 GB of weights -- over
6 GB before activations, so 6 GB *forces* the caching architecture rather than
merely rewarding it. **With cached embeddings it drops to 3.96 GB and fits.**

Verdict: document 6 GB as a degraded mode gated on 32 GB system RAM and an NVMe.
Do not promise it as a tier.

**Proof that tiled decode converts a VRAM wall into a time cost:** a **GTX 970,
4 GB** decoded 704x704x121 through the Wan 2.2 VAE at ~1,290 MB peak in 766 s
(hum-ma/ComfyUI-TiledVaeLite). Our `tiled_vae: True` is right; we simply never
measured it.

---

## 6. What to change, in order

1. **Replace the estimator with the max-over-stages shape** (section 1). Ours
   cannot express staging, so it cannot model the technique that makes 8 GB work.
   This is why it predicts 9,442 MB for a 17-frame render an 8 GB card should
   handle.
2. **Cache text embeddings.** Removes 3.9 GiB and minutes per leg. Highest
   benefit-to-effort by a wide margin, and it is what makes 6 GB conceivable.
3. **A/B fp8_e4m3fn_scaled safetensors against our GGUF** (section 2). Verified
   mechanism, unmeasured consequence. Possibly the biggest structural win.
4. **Declare `render_canvas` on the engine** -- still true and still unfixed; we
   currently render at 1472x832, 3.07x the intended pixels. Recommend 768x432.
5. **Add a `t5_device` knob** defaulting to CPU (ltx measured this as decisive).
   Note: with embedding caching in place this matters much less.
6. **Fix the flags.** `--lowvram` is inert. Use `--reserve-vram` /
   `--vram-headroom`, or `--disable-dynamic-vram` to opt the GGUF stack cleanly
   onto the legacy path instead of sitting in the hybrid.
7. **Consider a second 8 GB tier: Wan 2.2 14B Q4 + Lightning LoRA at 4-6 steps**,
   offloading ~9-11 GB to system RAM. Two independent 8 GB reports say it beats
   the 5B on both axes. Requires 32 GB system RAM.
8. Decode tiled deliberately rather than waiting for ComfyUI's OOM auto-retry.

---

## 6b. Where the parallel ChatGPT passes DISAGREE with source-verified findings

The operator ran two ChatGPT passes alongside this research. Their estimator
reasoning is excellent and section 1 adopts it. Two of their CONFIGURATION
claims do not survive grounding, and one is the crux of our biggest win. Judged
here rather than merged silently.

**1. "VRAM mode set to `auto` or `lowvram`" -- REJECTED.** Verified in ComfyUI's
own `comfy/cli_args.py`: *"Doesn't do anything if dynamic vram is enabled. If
dynamic vram isn't being used this option makes the text encoders run on the
CPU."* `text_encoder_device()` short-circuits on `aimdo_enabled` and returns the
GPU regardless (`comfy/model_management.py` L1164-1173). Source beats blog: the
`--lowvram` advice is stale across essentially the entire 2026 guide ecosystem,
which is exactly why it keeps getting repeated back to us.

**2. "the official workflow fits 8 GB using GGUF UNet quantization" -- REJECTED,
and this one matters most.** The official Wan 2.2 workflow uses **fp8 scaled
safetensors**, not GGUF. The distinction is load-bearing because
`ComfyUI-GGUF` defines `GGUFModelPatcher(ModelPatcher)` -- NOT
`ModelPatcherDynamic` -- and `clone()` force-reassigns `__class__` back
(https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/nodes.py L35),
confirmed as a known limitation in Comfy-Org/ComfyUI#13953. **GGUF does not get
Dynamic VRAM.** So citing the official 8 GB workflow as evidence that *our GGUF
stack* will stage the same way is precisely backwards: it fits BECAUSE it is
safetensors. ChatGPT's citation for the GGUF claim is a HuggingFace forum
thread; ours is the loader source and the tracking issue.

**Consequence: the "18 GB runs on 8 GB" proof point is real, but it is NOT
currently a proof point about US.** It becomes one only after the fp8-vs-GGUF
A/B in section 6 item 3. That single experiment is now the highest-value thing
on the list, above even embedding caching, because it decides whether staging is
available to us at all.

Minor: several ChatGPT citations do not support their attached claims (the
float32 frame-buffer arithmetic is cited to a blog that does not state it; a
dailymotion video and civitai pages back workflow claims). The arithmetic is
correct on its own terms -- it just is not sourced.

## 7. Honest gaps -- read before acting

- **Reddit was unreachable** during the research pass. r/comfyui and
  r/StableDiffusion are where 8 GB consensus actually forms; the community
  characterizations above are second-hand via GitHub, forums and blogs.
- **No rigorous peak-VRAM figure exists for Wan 2.2 TI2V-5B on an 8 GB card
  anywhere.** Not one source reported `max_memory_allocated`. The widely-quoted
  "4.6 GB loaded / 11 GB offloaded" is a ComfyUI log line, not a peak. **We are
  the best-positioned party to measure and publish this.**
- **Block swap has no published cost curve** -- neither kijai's docs nor
  DisTorch2 publish "N blocks swapped -> X GB saved, Y% slower". DisTorch2's
  "10% faster" is unbenchmarked vendor copy.
- **The GGUF-vs-fp8-under-Dynamic-VRAM question is open** -- mechanism verified
  in source, performance consequence inferred.
- **Flagged as marketing, not measurement:** NVIDIA's LTX-2 guide (16 GB
  recommendation, zero timings) and the `willitrunai` / `localaimaster` /
  `wan27.org` cluster, which is where most "Wan 2.6/2.7 on 8 GB" claims
  originate.

**Net: our research was not wrong about Wan. It was wrong about ADDITION, and it
missed that our own quantization choice disables the mechanism the whole 8 GB
story depends on.**
