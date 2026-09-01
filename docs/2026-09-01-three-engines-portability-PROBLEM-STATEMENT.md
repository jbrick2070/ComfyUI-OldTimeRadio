# Three engines, and no answer for anyone who is not us

> **RESOLVED / SUPERSEDED 2026-09-01.** This file preserves the original
> problem statement, not current installation truth. The controlling answer is
> [`2026-09-01-three-engines-portability-PLAN.md`](2026-09-01-three-engines-portability-PLAN.md).
> HuMo now has a complete pinned five-file lane; LTX 2.5 has a pinned in-repo
> Gemma-4/BF16 GGUF patch plus a complete gated manual recipe; and H3 uses node
> classes from current ComfyUI core. Comfy-Org states its H3 NVFP4 encoder does
> not require Blackwell. Legal H3 124-model / 129-canvas-frame receipts measured
> 6,315 MB FL2VA and 6,678 MB REF2VA absolute VRAM on the RTX 5080, with host RAM
> unmeasured. The separate physical RTX 4060 lab has receipt-bearing 90-frame
> H3 runs and exported 124-frame Ref2VA A/V artifacts, but no canonical H3
> episode. Physical-4060 LTX 2.5 remains unqualified: its exact plan is staged
> but unrun. The earlier 15.47-15.60 GiB observations were 5080 reserve/clamp
> pressure tests, not a physical-8GB surrogate.
> The explicit H3 lane pins five files totaling 63,440,965,087 bytes (59.084
> GiB) but is never publicly auto-selected. The optional Larry Turbo pack is not
> OTR's node source.

<details>
<summary>Original problem statement (historical evidence snapshot)</summary>

**Problem statement, 2026-09-01. Open-ended by intent -- it states what is known
and what is not, and does not propose a fix.**

Every number below was measured on named hardware or read out of the real files.
Where something is unknown it is marked UNKNOWN rather than estimated.

---

## The question

**HuMo, LTX 2.5 and MiniMax H3 each produce published episodes on exactly one
machine. What would it take to run each of them under ComfyUI on the platforms
people actually have -- and for which of them is the honest answer "you cannot"?**

The three fail for three unrelated reasons, which is why they are stated together:
one is a missing download, one is a missing upstream capability, one is a licence.
A single strategy will not cover them, and treating them as one problem has
already produced one wrong answer.

## What "the major platforms" means here

| platform | hardware in evidence | status |
|---|---|---|
| Windows + NVIDIA Blackwell | RTX 5080 Laptop, 16 GB, sm_120 | the only machine all three have ever run on |
| Windows + NVIDIA Ada, 8 GB | RTX 4060 Laptop, 7.99 GB, 31.70 GB host RAM | **SUPERSEDED:** isolated H3 is now lab-proven; physical LTX 2.5 remains UNKNOWN/unqualified |
| Linux + NVIDIA Ampere | rented RTX A4500, 19.6 GB, sm_86, container | LTX 2.5 blocked at the loader; h3 and HuMo never attempted |
| Linux + AMD (ROCm) | none | UNKNOWN -- never attempted |
| macOS + Apple Silicon (MPS) | none | UNKNOWN -- never attempted |

**Two axes decide everything and only one of them is usually reported.** VRAM is
the axis everyone quotes; **host RAM** is the one that has actually killed runs
here. And inside a container, `free` reports the HOST, not the container -- the
rented pod showed 251 GiB total / 193 GiB available while its real ceiling was
**57.7 GiB** (`cgroup memory.max = 61999996928`). ComfyUI was SIGKILLed crossing
it, with no CUDA OOM and no error in any log.

A third axis has bitten once and is easy to forget: **quantisation format is
hardware, not a size setting.** nvfp4 requires hardware fp4 (sm_120), so a card
handed an nvfp4 file picks the one weight it cannot execute.

---

## Engine 1 -- HuMo: nothing is broken, and nobody can install it

**Status: every node class resolves. There is no fetch lane.**

On the rented Ampere pod, all of HuMo's node classes resolved with no patching.
Nothing in the code is missing. What is missing is roughly **27 GiB of weights**
and any supported way to obtain them:

    Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors   16.66 GiB   default UNET
    umt5_xxl_fp8_e4m3fn_scaled.safetensors              6.27 GiB
    whisper_large_v3_fp16.safetensors                   2.88 GiB
    lightx2v_I2V_14B_480p_cfg_step_distill_rank64       0.69 GiB
    wan_2.1_vae.safetensors                             0.24 GiB

`scripts/otr_fetch_lane_weights.py` has no `humo` lane, so there is no command a
stranger can run.

**Proven:** 32 published episodes on the 5080 across four engine variants
(`humo_14B_169` x11, `humo_1.7B_169` x9, `humo_1.7B` x7, `humo` x5), recorded in
`meta.render_engines.per_clip[].delivered_engine`. Bench receipt
`humo_14b_diet_landscape_832x480_f97.json`: PASS, peak **13.06 GiB VRAM /
27.53 GiB host RAM**.

**UNKNOWN:** whether 13.06 GiB VRAM and 27.5 GiB host RAM hold on Ampere or Ada;
whether the fp8 scaled weights execute correctly off Blackwell; anything at all
about ROCm or MPS; whether the 1.7B variants lower the floor enough to matter on
smaller cards, and by how much.

**The open question:** is HuMo's 27 GiB simply a download nobody has paid for, or
does the fp8/Blackwell provenance of these specific files hide a portability
problem that only appears once someone tries it elsewhere?

---

## Engine 2 -- LTX 2.5: upstream does not support the encoder

**Status: fails at model load on a stock install, everywhere except one machine.**

LTX 2.5's text encoder is `gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf`. A clean
ComfyUI-GGUF refuses it outright:

    Error: Unexpected text model architecture type in GGUF file: 'gemma4'

Upstream `city96/ComfyUI-GGUF` was cloned fresh on 2026-09-01: HEAD `6ea2651`,
dated 2026-01-12. The repository publishes **no GitHub Releases**, so that is
`main` HEAD and not a release tag. `gemma4` is not in its architecture list.

The reference machine's copy is modified, and the delta splits along a line worth
stating precisely:

    loader.py   17 lines   "gemma4" added to the architecture allowlist, plus a
                           BF16 dequant path for raw LTXAV parameters that GGMLOps
                           does not handle. GENERIC -- nothing OTR-specific in it.
    nodes.py    76 lines   CLIPLoaderGGUFCPU: a text-encoder loader whose patcher
                           is pinned to CPU across all three placements, because
                           initial_device alone is not a residency guarantee and
                           a 12B encoder must stay off a 16 GB card.
                           OTR-SPECIFIC -- our VRAM problem, not upstream's bug.

Copying those two files to the pod **did** clear the loader error and the run
proceeded into the two-stage decode. So the patch is correct. It is simply not
obtainable by anyone else, and the modified copy on the reference machine has no
`.git`, so its own provenance is UNKNOWN -- published fork, Manager install, or
hand patch, we cannot tell from the filesystem.

**Clearing the loader is not the same as rendering.** With the patch applied, the
pod still died: SIGKILL at the 57.7 GiB cgroup limit during the two-stage decode
at 1664x960. No episode has ever been published by LTX 2.5 on hardware other
than the 5080 (where it has 13 `ltx25_video`, 7 `ltx25_foley_plus` and 3
`ltx25_mime` episodes).

**SUPERSEDED interpretation:** sixteen 5080 reserve/clamp runs stayed near
**15.47-15.60 GiB**. They are pressure evidence, not a physical-8GB surrogate;
the prepared physical-4060 LTX 2.5 plan remains unrun and UNKNOWN/unqualified.

**UNKNOWN:** the host-RAM envelope the two-stage decode actually needs at each
canvas; whether a smaller canvas clears the cgroup ceiling and at what quality
cost; whether upstream would accept the 17 generic lines, and on what timescale
for a repository whose last commit was eight months ago; ROCm and MPS entirely.

**The open question:** what is the right relationship to an upstream that lacks a
capability we have already implemented in 17 lines -- and separately, what is the
true host-RAM cost of the two-stage decode, given that the loader was never the
thing that killed the run?

---

## Engine 3 -- MiniMax H3: two independent blocks, either one sufficient

**SUPERSEDED status:** current ComfyUI core supplies OTR's H3 node classes and
the explicit five-file lane is complete. This operator's signed offline/owned-
hardware policy still applies; third parties must establish their own authority.

`minimax_h3_video` needs `MiniMaxH3ImageToVideo`, supplied by a node pack whose
git origin on the reference machine is a local path:

    C:\ComfyUI-Models\quarantine\h3-turbo-larry-v4-step600-ema\source\ComfyUI-MiniMax-H3-Turbo

There is no URL any provisioner could clone. (`MiniMaxH3ReferenceToVideo` belongs
to the separate audio-conditioned lane, not this one.)

Two blocks, and they are independent:

1. **Licence.** `docs/H3_LICENSE_ATTESTATION.md` is signed and closed: H3
   inference runs only on the operator's own hardware, offline; no hosted or
   shared endpoint; weights never redistributed, republished, mirrored or bundled
   "in any form, quantized included"; commercial scope not established in writing.
2. **SUPERSEDED silicon inference.** Comfy-Org documents the supplied H3 NVFP4
   encoder as not requiring Blackwell, and the exact encoder loaded in the
   physical Ada RTX 4060 lab.

**Proven:** 3 published episodes on the 5080.

**The 8 GB claim needs retiring or restating.** H3 has been described as the only
engine with a path onto an 8 GB card, from a receipt of **7.28 GiB VRAM /
27.56 GiB host RAM** at 864x480, 90 model frames. But OTR's shortest legal H3 clip
is **124 model frames**, 38% longer, and a newer video-only cold run of the same
lane reached **33.34 GiB host RAM** -- 1.64 GiB more than the 4060 physically has.
The peaks at 124 frames were never measured. Both 8 GB H3 profiles remain `draft`.

**UNKNOWN:** the peaks at 124 model frames; whether a non-nvfp4 encoder exists or
could be produced; whether the node pack was ever published anywhere; whether the
authorization permits a third party to run it at all, which is a question for the
operator and not for a coder.

**The open question:** given that the licence alone forecloses redistribution and
the encoder alone forecloses non-Blackwell silicon, is there any version of "run
H3 elsewhere" that is both legal and technically possible -- and if not, what
should the project say publicly about an engine it can demonstrate but nobody
else can run?

---

## What a good answer would have to include

* **Both peaks, always.** A VRAM figure with no host-RAM figure cannot be acted
  on; two of the three ceilings here are host RAM.
* **The container distinction.** Any Linux number must say whether it was read
  against the cgroup or against `free`, because those differed by 135 GiB on the
  one machine tested.
* **Model frames and canvas frames stated separately.** A 24 fps model against a
  25 fps timeline is exactly the confusion that produced the unmeasured H3 claim.
* **Where it failed, not just that it failed** -- load, sample, or decode. LTX
  2.5's real ceiling turned out to be in the decode, long past the error everyone
  was looking at.
* **An honest "no."** A lane that fits only under settings nobody would ship is a
  worse answer than a measured refusal. The matrix records failures as readily as
  successes.

</details>
