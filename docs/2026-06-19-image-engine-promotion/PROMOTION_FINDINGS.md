# Image-engine verify-and-promote -- qwen_image / hidream_i1 FINDINGS (2026-06-19)

> **UPDATE 2026-06-19 (CPU build phase, operator directed the Phase-0 code edits):**
> After the audit below, the operator directed the CPU-side code/registry/tests
> edits now (GPU held by the still-only z_image soak). Ground-truth recipes were
> captured from the INSTALLED ComfyUI templates (no /object_info needed; CPU-only):
> `image_qwen_image_2512_with_2steps_lora.json` (Qwen) + `hidream_i1_fast.json`
> (HiDream). Decisions:
>
> - **qwen_image: BUILT (CPU-complete, GPU-smoke-PENDING).** Real `render_image` +
>   the native GGUF recipe (UnetLoaderGGUF + CLIPLoader[type=qwen_image] qwen-2.5-vl
>   TE + VAELoader -> ModelSamplingAuraFlow -> KSampler -> VAEDecode) -- a near-twin
>   of `lumina_image`. Graph-build CPU tests added; graduated from the stub matrix.
>   STILL HIDDEN (not in VALIDATED_ENGINES) until a GPU smoke proves it renders AND
>   measures the per-quant resident peak <= 14.5 GB. Single qwen-2.5-vl TE (~5 GB
>   fp8) offloads before sampling -> the GGUF diffusion (Q3/Q4 ~8-11 GB) + VAE has a
>   REAL chance of fitting. Worth the smoke.
> - **hidream_i1: NOT BUILT -- high-confidence ceiling NO-GO (math below).** The
>   official HiDream-Fast recipe needs a `QuadrupleCLIPLoader` = clip_l + clip_g +
>   **t5xxl (~5 GB fp8) + llama-3.1-8B (~8 GB fp8)** ~= **~14 GB of text encoders
>   ALONE**, before the HiDream UNet. Even with offload-before-sampling, the encode-
>   phase transient almost certainly breaches the 14.5 GB single-resident ceiling.
>   Speculatively building its full 4-encoder recipe (4 new TE env vars + plumbing)
>   for a model that very likely CANNOT ship is the wrong call -- this is the "over
>   ceiling -> log + skip" stop condition, predicted from the encoder math rather
>   than measured. The grounded recipe is captured below for a future build IF the
>   operator accepts a reduced-encoder config or a higher ceiling.
>
> Validated IMAGE set remains UNCHANGED + undisturbed (`flux_gen1` /
> `z_image_turbo` / `flux2_klein` / `lumina_image`); no validated engine touched;
> z_image left entirely alone during its soak.

**Task:** verify-and-promote `qwen_image` and `hidream_i1` to `VALIDATED_ENGINES`
using the exact playbook that landed `lumina_image` (commits `ed560a0`/`631d0b0`),
*only if* they pass a GPU smoke under the 14.5 GB resident ceiling.

**Verdict: BOTH SKIPPED -- they are NOT in the Lumina state. No GPU smoke was run
(none is possible yet).** This is the directive's "weights missing -> document +
surface, skip" branch, plus a deeper gap: the render recipe was never built.

The validated IMAGE set is UNCHANGED and undisturbed: `flux_gen1` (default) +
`z_image_turbo` + `flux2_klein` + `lumina_image`.

---

## Why Lumina could be promoted but these two cannot (the gap analysis)

`lumina_image` was promotable because it was **code-complete**: a real
`render_image` driving a full declarative recipe (`_node_candidates` +
`_build_lumina_graph`), only hidden behind the tested-only dropdown gate pending a
GPU smoke. The smoke just had to *prove the existing recipe renders*.

`qwen_image` and `hidream_i1` are a different animal -- **stub peers by design**:

- `render_image` is an explicit `raise NotImplementedError("... the in-stack GGUF
  GPU/operator smoke; download ..., set <ENV>, and run the verify-on-5080
  checklist first")`. There is **no recipe** -- no `_node_candidates`, no
  `_build_*_graph`, no sampler chain.
- The CPU contract suite **asserts** this:
  `tests/test_image_engine_matrix_peers.py::test_peer_render_is_operator_gpu_smoke`
  expects `NotImplementedError`. Lumina and flux2_klein each *graduated out* of
  that matrix when they gained a real `render_image`; qwen/hidream never did.

So there is nothing to "verify-and-promote." Promoting either would require
**building the engine** (the real GGUF recipe), which is outside the authorized
"clearly bounded, already-decided verify-and-promote" scope -- and it carries real
14.5 GB ceiling risk (below).

## Weights: NOT on disk (the explicit skip condition)

Scanned `C:\ComfyUI-Models` (incl. `diffusion_models/`, `unet/`). Present GGUFs:
HuMo-17b, Wan2.2-TI2V-5B, flux-2-klein-4b, ltx-2.3-22b. **Absent:** any
Qwen-Image GGUF, any HiDream-I1 GGUF. Their `assert_usable` correctly fails closed
(`MISSING_MODEL`) -- ABSENT/greyed, never a stub (BUG-046).

## VRAM: both are HEAVY and at-ceiling (the promotion GATE is the risk)

The hard promotion gate is resident peak <= 14.5 GB. Both engines are flagged in
their own docstrings + CAPABILITIES rows as TIGHT:

| engine       | CAPABILITIES        | size / quant            | ceiling risk |
|--------------|---------------------|-------------------------|--------------|
| qwen_image   | heavy / 14000 MB    | 20B, GGUF Q4 (~12-15 GB)| HIGH -- 20B core, near the cap before TE/VAE |
| hidream_i1   | heavy / 14000 MB    | GGUF, ~13-15 GB         | HIGH -- needs a 4-encoder TE stack (CLIP-L + CLIP-G + T5-XXL + Llama-3.1-8B); the TE residency is what makes it tight |

Contrast: `lumina_image` is medium/~7 GB (measured resident peak ~12.2 GB with the
TE held), `z_image_turbo` ~5 GB. qwen/hidream have no headroom margin -- a real
smoke could easily breach 14.5 GB depending on quant + whether the TE stack
offloads before sampling. Per-quant VRAM MUST be measured before any promotion.

---

## To actually ship either engine (operator decisions required first)

This is a BUILD, not a promotion. Per engine it needs: (1) operator confirms the
weights repo + license + exact quant; (2) download the GGUF + companions; (3) build
the real `render_image` recipe from the official ComfyUI workflow; (4) GPU-measure
per-quant resident peak and pick a quant that fits <= 14.5 GB (LOUD fail-closed if
none does); (5) then the Lumina verify-and-promote playbook applies.

**qwen_image (Apache-2.0, commercial-clean):**
- Recipe: official ComfyUI Qwen-Image GGUF workflow = `UnetLoaderGGUF`
  (qwen-image GGUF) + `CLIPLoader` (qwen-2.5-vl text encoder) + `VAELoader`
  (qwen-image VAE) -> sampler -> decode. Mirrors the flux2_klein GGUF build
  pattern already in the repo.
- Weights (operator to confirm provenance/quant): the standard community GGUF
  requant repos on Hugging Face are `city96/Qwen-Image-gguf` and
  `QuantStack/Qwen-Image-GGUF`. Pick the largest Q that measures <= 14.5 GB
  (likely Q3_K_M / Q4_K_S given the 20B core + the qwen-2.5-vl TE).

**hidream_i1 (MIT, commercial-clean):**
- Recipe: official ComfyUI HiDream workflow needs the HiDream UNet (GGUF) +
  FOUR text encoders (CLIP-L, CLIP-G, T5-XXL, Llama-3.1-8B-Instruct) + the HiDream
  VAE. The 4-encoder stack is the VRAM pressure point -- staging/offload order
  matters for the ceiling.
- Use the **Fast** variant (fewest steps) per the C2 matrix.
- Weights (operator to confirm): community GGUF requants exist under `city96/`
  on Hugging Face; the 4 encoders are the standard ComfyUI HiDream text-encoder set.

**Recommendation:** treat each as its own bounded build sprint *after* the operator
confirms weights/license/quant -- NOT a same-session verify-and-promote. The
ceiling risk means a roundtable or at least a careful per-quant probe is warranted
for whether they fit OTR's single-resident 14.5 GB budget at all (qwen's 20B core
and hidream's 4-encoder stack are both genuinely tight, unlike Lumina's 2.6B).

## What was NOT touched

No edits to any validated engine (`flux_gen1`/`z_image_turbo`/`flux2_klein`/
`lumina_image`), no registry promotion, no workflow-JSON change, no GPU server
booted (zero contention with the story-quality soak on :8011). CPU audit only.

---

## Appendix: grounded recipes (captured from installed ComfyUI templates, 2026-06-19)

CPU-verified node classes (installed): `UnetLoaderGGUF`, `CLIPLoaderGGUF`,
`QuadrupleCLIPLoaderGGUF` (ComfyUI-GGUF); core `CLIPLoader` supports
`type in {qwen_image, hidream, lumina2, ...}` (comfy/sd.py CLIPType).

**qwen_image (BUILT)** -- from `image_qwen_image_2512_with_2steps_lora.json`
(minus the 2-step turbo LoRA + ConditioningZeroOut; standard cfg path restored):
- `UnetLoaderGGUF{unet_name}` -> MODEL  (OTR_QWEN_IMAGE_GGUF; 20B Q-quant)
- `CLIPLoader{clip_name, type:"qwen_image"}` -> CLIP  (qwen_2.5_vl_7b_fp8_scaled; OTR_QWEN_IMAGE_CLIP)
- `VAELoader{vae_name}` -> VAE  (qwen_image_vae; OTR_QWEN_IMAGE_VAE)
- `ModelSamplingAuraFlow{model, shift:3.0}`
- `CLIPTextEncode` pos/neg ; `EmptySD3LatentImage{w,h,1}`
- `KSampler{steps:20, cfg:2.5, euler, simple, denoise:1}` ; `VAEDecode` -> still

**hidream_i1 Fast (NOT built -- ceiling NO-GO; recipe for a future call)** -- from
`hidream_i1_fast.json`:
- `UNETLoader`/`UnetLoaderGGUF{unet_name}`  (hidream_i1_fast_fp8 / a GGUF quant)
- `QuadrupleCLIPLoader{clip_l_hidream, clip_g_hidream, t5xxl_fp8_e4m3fn_scaled,
  llama_3.1_8b_instruct_fp8_scaled}`  <- the ~14 GB encoder stack (the NO-GO driver)
- `VAELoader{ae.safetensors}`  (the Flux ae VAE)
- `ModelSamplingSD3{model, shift:3}`
- `CLIPTextEncode` pos/neg ; `EmptySD3LatentImage{1024,1024,1}`
- `KSampler{steps:16, cfg:1.0, lcm, normal, denoise:1}` ; `VAEDecode` -> still

To make hidream viable under 14.5 GB the encoder stack must shrink (e.g. GGUF-quant
t5xxl + llama, or a sequential-encode-then-fully-evict step before the UNet loads) --
an operator/roundtable call, not a same-session build.
