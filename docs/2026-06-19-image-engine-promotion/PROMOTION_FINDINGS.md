# Image-engine verify-and-promote -- qwen_image / hidream_i1 FINDINGS (2026-06-19)

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
