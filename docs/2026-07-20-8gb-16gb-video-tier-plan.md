# OTR Video Tiers — Final Plan (8GB build + 16GB append-only)

**Date:** 2026-07-20
**Status:** DRAFT for operator review-panel confirmation
**Scope:** Add an 8GB video tier; label the existing (already-working) 16GB tier. No rebuilds. Upscaling explicitly deferred.

---

## 1. Decision summary

| Tier | Action | Engines |
| --- | --- | --- |
| **8GB** | **BUILD** (two new selectable rows) | `ltx_8gb` (new) + `wan_8gb` (alias of existing `wan_ti2v`) |
| **16GB** | **LEAVE AS IS** — append `16gb` labels only | `ltx_video` -> `ltx23_16gb_video`; `ltx_audio_in` -> `ltx23_16gb_audio_in` |
| **Upscaling** | **DEFERRED** — separate future conversation | (rebuild upscale plans later; out of scope here) |

**Governing guardrails (operator directives):**

- **Additive-first. No rips until the new 8GB paths test green.** Then rip the unneeded ones.
- **HuMo is untouched** throughout.
- **Minimal:** no new gates, no new profiles, no scaffolding beyond what is needed to work. Engines are selectable dropdown rows ("registry IS the menu").
- **Honest naming:** the 8GB LTX is the 2B `v0.9` model, never labelled `2.3`.
- UTF-8 no BOM, SFW. Commit + push each green chunk to `v2.0-alpha`.

---

## 2. Why these choices (rationale for the panel)

- **A true 8GB LTX-2.3 does not exist.** The smallest 2.3 quant is `ltx-2.3-22b-distilled-1.1-Q2_K.gguf` = **7.94GB of transformer weights alone** — over an 8GB card's budget before the encoder/VAE/latents/desktop reserve. So the 8GB LTX row must be the **2B `v0.9`** model (the only sub-22B LTX). It runs on any 8GB card and is already on disk.
- **`wan_8gb`** gives 8GB users a higher-quality option: Wan 2.2 TI2V-5B, whose engine is **already built** (`eng_wan_ti2v`) and whose GGUF is **already on disk** — so it is a rename + test, not a build.
- **The 16GB tier is already built.** The two proposed 16GB recipes map 1:1 onto existing, proven engines running the shared `ltx-2.3-22b-dev-Q3_K_M.gguf`. Nothing to build; only labels to append.

---

## 3. 8GB tier (NEW)

> Both 8GB engines stay **single-pass / no upscale** to fit the 8GB budget. Any 8GB upscaling belongs to the deferred *additional-upscaler* discussion (section 5), not this build.

### 3.1 `ltx_8gb`
- **Model:** `ltx-video-2b-v0.9.safetensors` (already on disk, 8.73GB). Zero download.
- **Open item (smoke decides):** the installed `ComfyUI-LTXVideo` is a modern 2.3-era build (Gemma / ICLoRA / Q8 / `low_vram_loaders.py`, `LowVRAMCheckpointLoader`) whose version widget **defaults to `0.9.8`**. The standalone smoke confirms whether the original `v0.9` loads cleanly, or whether we point `ltx_8gb` at `0.9.8` instead — same engine, one loader argument either way.
- **Behavior:** silent i2v (condition on the beat's scene still), audio muxed separately (matches the existing `8gb_lite` pattern).
- **Build:** mirror the `eng_ltx_video` / `wrapper_bridge` i2v patterns; reuse the existing `gpu_residency` reclaim. **No enable-flag gate** (`requires_flag=None`). Add its `CAPABILITIES` row.

### 3.2 `wan_8gb`
- **Model:** Wan 2.2 TI2V-5B — `Wan2.2-TI2V-5B-Q5_K_M.gguf` (already on disk, 3.55GB).
- **Action:** additive **alias** of the existing `wan_ti2v` engine (keep `wan_ti2v` working via a legacy alias — no rip until tested). Register `wan_8gb` as a selectable 8GB row; smoke-test it renders.
- **Behavior:** i2v, silent, audio muxed separately. No new scaffolding.

### 3.3 8GB clip calculation (both engines)
- Compute a **short per-clip length (3–5s = what 8GB handles)** but render/loop **enough clips to cover the FULL beat window** — never a single short clip frozen-held for the remainder. Extends the existing LTX beat-fill / boomerang logic (rendered coverage >= beat target).

### 3.4 8GB wiring (additive)
- Add `ltx_8gb` + `wan_8gb` as selectable `OTR_VideoDirector` dropdown rows in `workflows/otr_canonical.json` (same change as the code, per repo rule §0).
- `8gb_lite` profile: **keep `still_motion` as the default until both new paths test green**; `ltx_8gb` + `wan_8gb` are selectable. No new profile.
- Re-validate: `OTR_WorkflowValidator` + JSON round-trip + link/widget audit.

### 3.5 8GB validation
- Standalone smoke each engine on the 5080 (renders correctly + asset lands on disk).
- **Real 8GB-card validation is the operator's later test on true 8GB hardware** — not simulated here.
- Regression suite + Bug Bible green; commit + push to `v2.0-alpha`.

---

## 4. 16GB tier (ALREADY BUILT — append labels only)

Both proposed recipes already exist as proven engines on the shared `ltx-2.3-22b-dev-Q3_K_M.gguf`:

| Proposed recipe | Existing engine | Verified behavior |
| --- | --- | --- |
| `ltx23_16gb_audio_in` | `ltx_audio_in` (`eng_ltx_av`, IA2V) | Image + frozen per-beat OTR audio -> video. **Silent by contract:** the audio-VAE-decode branch is never wired; `OTR_MasterAudioMux` muxes the master audio. Matches "do not audio-VAE-decode the supplied audio at the end." |
| `ltx23_16gb_video` | `ltx_video` (`eng_ltx_video`, two-stage HQ) | Q3 dev unet: motion (half canvas) -> x2 latent upscale -> refine with the distilled LoRA. Silent by contract. **Cleaner than the spec assumes — the HQ port deletes the whole audio lane, so it never samples an audio latent.** |

**Shared assets — all confirmed on disk:**
`ltx-2.3-22b-dev-Q3_K_M.gguf` (10.0GB) · `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7.08GB) · `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (0.93GB) · Gemma-3 12B encoder · text/audio connectors · LTX-2.3 video VAE · LTX-2.3 audio VAE.

**Peak:** ~14.8GB measured on the RTX 5080 (existing baseline).

**Action:** leave the working engines untouched; **append the `ltx23_16gb_*` display labels** where appropriate (alias only, no code/engine-id rip). The **internal x2 spatial upscaler stays ACTIVE** — it is part of the existing two-stage HQ recipe graph (`LTXVLatentUpsampler` + `LatentUpscaleModelLoader`, already-installed nodes, **no additional nodes**). The whole-path VRAM discipline is already substantially in place (reclaim barrier, `OTR_LTX_AV_RESERVE_VRAM_GB` default 4.0 = ComfyUI `--reserve-vram`, NVML ceiling guard, tiled decode) — fold the panel's cleanup checklist in as an optional **verification pass**, not new code.

---

## 5. Deferred / out of scope

- **ADDITIONAL / new upscaler nodes** — separate future conversation; rebuild later. This is *only* about adding extra upscaler nodes/passes or different upscaler models beyond what a recipe already runs internally. It does **NOT** include the **internal x2 spatial upscaler already inside the 16GB two-stage HQ recipe** (`LTXVLatentUpsampler` + `LatentUpscaleModelLoader`, already-installed nodes) — that **stays active**, no additional nodes needed.

  **Intended shape (operator, 2026-07-20):** a model-agnostic **multi-bank upscaler registry** — a fourth parallel namespace alongside the existing video / image / audio engine registries and the LLM banks, on the same `engine_registry_base` pattern (pluggable upscaler adapters + `CAPABILITIES` rows, "registry IS the menu", no gating). The current internal x2 upscaler becomes the default entry; the bank makes upscalers swappable and extensible. Scoped and built in that separate conversation — not this build.
- **Post-green cleanup** — after `ltx_8gb` + `wan_8gb` test green: rip the now-unneeded old paths/names (the `wan_ti2v` legacy alias, and `still_motion` as the `8gb_lite` default if superseded). Exact list decided post-test.

---

## 6. Execution order

1. Smoke `ltx_8gb` (LTX-2B; confirm `v0.9` vs `0.9.8` loads) on the box.
2. Build `eng_ltx_8gb` adapter + `CAPABILITIES` row (no gate).
3. Add `wan_8gb` alias of `wan_ti2v`; smoke it.
4. Add the 8GB beat-fill clip calc to both engines.
5. Wire both rows into `otr_canonical.json` (additive; `still_motion` stays the `8gb_lite` default).
6. Append `ltx23_16gb_*` labels to the two existing 16GB engines (alias only).
7. Regression suite + Bug Bible green.
8. Commit + push to `v2.0-alpha` (verify HEAD == origin, no BOM, AST parse).
9. Full `8gb_lite` leg smoke on the 5080 (RESULT SUCCESS + obs_publish + asset on disk).
10. (Deferred) Operator validates on a real 8GB card; then post-green cleanup.

---

## 7. Open questions for the review

1. `ltx_8gb` checkpoint: original **`v0.9`** or the install-default **`0.9.8`**? (Smoke can decide, or reviewer preference.)
2. After green, the `8gb_lite` **default** engine: stay `still_motion`, or promote `ltx_8gb` (lightest) or `wan_8gb` (higher quality)?
3. 16GB labels: **alias-only display rename** (recommended, non-breaking) vs. leaving the engine ids entirely untouched with no new labels?
