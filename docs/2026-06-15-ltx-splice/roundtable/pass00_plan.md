# LTX clean-break splice plan — production video LTX → MIT-friendly GGUF/Goofer recipe

**Status:** DRAFT for roundtable (2026-06-15). Planner window. Coding gated on operator + roundtable.

## 0. Goal (operator directive)
Clean-break **replace the production VIDEO LTX recipe** with the proven, license-safe
MIT-friendly bookend recipe — for **every** LTX video shot (radio bookend AND per-character /
per-scene clips), each keeping its own per-shot prompt. No shim, no runtime gate, no mish-mash:
rip out the old recipe wholesale, drop in the new one. (Matches the cleanbreak rule: no
back-compat naming, no runtime gates inside the cleanbreak sprint.)

**Phase 0 (rip-out) comes FIRST (operator directive 2026-06-15):** before splicing, delete the
defunct LTX surface — `ltx_orbit` (niche orbit preset) and the dead 2B/T5/VAEDecode recipe — so the
LTX surface is exactly **three uses (announcer / music / per-beat) on ONE engine**. KEEP the dark
audio-input lanes (`ltx_av_talk` / `ltx_av_music`) parked — they're a separate, not-defunct feature
to test + splice later. THEN (Phase 1) drop the GGUF/Goofer recipe into that single engine.

## 1. Complete LTX render-path inventory (verified 2026-06-15)
All LTX rendering lives in `nodes/_otr_video_engines/`. The production workflow JSON
(`workflows/otr_scifi_16gb_full.json`) contains **zero** LTX ComfyUI nodes — it is 23 `OTR_*`
policy nodes; the LTX ComfyUI graph is built in Python at runtime and run via `wrapper_bridge`.

| engine id | class / file | role | disposition |
|---|---|---|---|
| `ltx_video` | `LtxVideoEngine` `eng_ltx_video.py:222` | text->video + i2v; serves all 3 uses (announcer / music / per-beat) | **REPLACE (Phase 1 clean break)** |
| `ltx_orbit` | `LtxOrbitEngine(LtxVideoEngine)` `eng_ltx_video.py:888` | niche orbit-camera preset; NOT one of the 3 uses | **RIP (Phase 0)** |
| `ltx_av_talk` | `eng_ltx_av.py` | LTX-2.3 audio-INPUT talk (A2V) | **KEEP dark** (separate audio-input feature; test + splice later) |
| `ltx_av_music` | `eng_ltx_av.py` | LTX-2.3 audio-INPUT music | **KEEP dark** (separate audio-input feature; test + splice later) |

No LTX render code outside `_otr_video_engines/` (only `_otr_shared/av_dims.py`, a dims helper).

## 2. The target recipe (frozen, proven, license-clean)
From `workflows/ltx_bookend_mini_repro_gguf_mit.json` (rendered 87s, good quality, no OOM):
- model: `UnetLoaderGGUF(ltx-2.3-22b-dev-Q4_K_S.gguf)` + `LoraLoaderModelOnly(ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors, 0.70)`
- vae: `VAELoader(ltx-2.3-22b-dev_video_vae.safetensors)` (GGUF unet has no VAE)
- text: `LTXAVTextEncoderLoader(gemma_3_12B_it_fp4_mixed, ltx-2.3-22b-dev)` -> CLIPTextEncode(pos/neg) -> LTXVConditioning(25fps)
- i2v: `EmptyLTXVLatentVideo(832x480xN)` -> `LTXVImgToVideoConditionOnly(strength 0.75)`
- sampling (Goofer/v0_9): `KSamplerSelect(euler_cfg_pp)` + `ManualSigmas(8-step distilled)` + `RandomNoise` + `CFGGuider(cfg 1.0)` + `SamplerCustomAdvanced`
- decode: `VAEDecodeTiled(512/64/4096/8)`
- loop: OTR's own ffmpeg boomerang (NOT VHS); per-engine (`ltx_video` loops, `ltx_orbit` does not)
- License: ComfyUI core + ComfyUI-GGUF (Apache) + ComfyUI-LTXVideo (LTX-2 Community). **NO RES4LYF (AGPL), NO VHS (GPL).**

## 3. Clean-break changes (file by file, grounded)

### 3.0 PHASE 0 — rip out the defunct LTX (do FIRST, separate commit)
- Delete `LtxOrbitEngine` (`eng_ltx_video.py:887-920+`, the `@register` class `name="ltx_orbit"`) and
  its registry row `"ltx_orbit"` (`registry.py:155`). Grep the repo for `ltx_orbit` / `LtxOrbitEngine`
  and remove every reference (dropdowns, profiles, docs pointers, tests) so nothing dangles.
- Delete the dead 2B recipe scaffolding inside `LtxVideoEngine` that Phase 1 won't use: the
  `CheckpointLoaderSimple`/`CLIPLoader`(T5) candidate branches and the `ltx-video-2b` constant
  (`registry.py:150` `model_requirements`) — these go when Phase 1 swaps in GGUF/Gemma.
- KEEP `eng_ltx_av.py` (`ltx_av_talk`/`ltx_av_music`) untouched and dark.
- Result: ONE LTX video engine (`LtxVideoEngine`) serving the three uses. Run the suite + Bug Bible;
  commit Phase 0 green BEFORE starting Phase 1 (so the rip-out is auditable on its own).

### 3A. `nodes/_otr_video_engines/eng_ltx_video.py` (PHASE 1 PRIMARY — the one LTX video engine)
- `_node_candidates()` (~:488): `CheckpointLoaderSimple` -> `UnetLoaderGGUF`; add a `VAELoader`
  entry; `CLIPLoader`(T5) -> `LTXAVTextEncoderLoader`(Gemma). Remove the T5/2B candidates (clean break).
- `_build_graph()` (~:622) + `_build_graph_i2v()` (~:420): rewire GGUF-unet->model, VAELoader->vae,
  Gemma->clip; wire `LoraLoaderModelOnly`(distilled @0.70) **unconditionally** (drop the
  `"22b" in ckpt_name` gate at ~:561, or make `_ckpt_name()` return the GGUF name so the gate passes);
  swap terminal `VAEDecode` -> `VAEDecodeTiled(512/64/4096/8)` (~:349,:718).
- `_sampler_mode()` (~:506): default -> `distilled` (8-step Goofer). `_sampler_name()` already `euler_cfg_pp`.
- `assert_usable()` (~:315-346): switch the MISSING_MODEL check from "2B ckpt + t5xxl" to
  "GGUF unet + Gemma + distilled LoRA + video VAE present". Resolve the **default-OFF/dark** gate:
  make the GGUF recipe the real production default (remove/flip `OTR_ENABLE_LTX_VIDEO` dark-gate) so
  LTX actually executes instead of failing closed to the `still_kenburns` floor (BUG-413 R1).
- Canvas 832x480, i2v strength 0.75, boomerang defaults already correct.
- `ltx_orbit` (`LtxOrbitEngine`): no change needed — inherits everything; keep its no-auto-loop override.

### 3B. `nodes/_otr_video_engines/render_driver.py` (MINIMAL)
- Per-shot prompt sourcing stays AS-IS: `build_request_from_shot()` + `_LTX_MOTION_PROMPT_BY_ROLE`
  (announcer/music_open/music_close/music_inter/sfx) already vary per shot/role; the i2v still
  carries the look, the role prompt carries motion. **Do not touch the prompt logic.**
- BUG-413 R2 (i2v seed): ensure the FLUX `radio_bookend` still is generated/stamped so `_use_i2v`
  doesn't fall to murky text2vid (`radio_bookend_path` must be non-None). Verify the
  ImageGenDispatcher (node 91) seed-4242 scene_open still is wired.
- `check_ltx_open_health()` (BUG-413 guard, ~:1506) stays; update `_LTX_OPEN_ENGINES` if engine ids change (they don't).

### 3C. `workflows/otr_scifi_16gb_full.json` (SOURCE OF TRUTH — append-only if needed)
- No LTX nodes to rewire. If the recipe exposes a new knob (e.g., GGUF quant pick), **APPEND**
  a positional widget at the END of `OTR_VideoDirector`(id 87) `widgets_values` — never insert
  mid-list (BUG-LOCAL-097). Re-validate with `OTR_WorkflowValidator` + JSON round-trip + widget audit.
- Any node/widget change ships IN THIS JSON in the SAME commit as the code (else it's dead).

### 3D. `eng_ltx_av.py` — NOT TOUCHED (separate audio-input lanes, already clean; reference pattern).

## 4. BUG-413 resolution (clean-break, not a guard)
BUG-413 = LTX never executes -> falls to procgen floor. Two roots: (R1) `assert_usable` fails
closed (flag/Sage/missing-2B-model); (R2) missing i2v seed. The clean break fixes R1 structurally
(GGUF recipe is present + default-on, so assert_usable passes) and R2 by guaranteeing the bookend
still. The existing LOUD `check_ltx_open_health` guard stays as a tripwire.

## 5. License clearance (must hold)
Current production `ltx_video` path is already clean (no RES4LYF, no VHS, no GGUF). The splice
ADDS only `UnetLoaderGGUF` (ComfyUI-GGUF, Apache) + keeps stock LTXV + Goofer stock samplers.
It must NOT introduce `ClownSampler_Beta`/`MultimodalGuider` (RES4LYF AGPL) or `VHS_VideoCombine` (GPL).

## 6. Validation / test gate (per CLAUDE.md)
- After edit: `OTR_WorkflowValidator` + JSON round-trip + link/widget audit on the JSON.
- Regression suite + Bug Bible (cd survival-guide repo, relative path) after every code change.
- GPU smoke: reset (CIM kill by CommandLine, :8000 free, VRAM baseline) -> headless render a real
  episode leg -> confirm an LTX clip lands (not the floor), VRAM <= 14.5GB single-resident, ~70-90s/clip.
- Commit AND push per green chunk on `v2.0-alpha`; verify HEAD==origin, no BOM, AST parse.

## 7. Open questions for the roundtable (pressure-test these)
1. Default-on vs keep a flag: cleanbreak says no runtime gate — is making GGUF-LTX the hard default safe on a 16GB box, or do we keep `OTR_ENABLE_LTX_VIDEO` as a kill switch only?
2. GGUF Q4 vs Q3 default for the per-shot batch (many shots/episode) — VRAM headroom vs fidelity across a full episode.
3. Distilled LoRA unconditional wiring — any role (orbit, scene_broll) where the distilled LoRA hurts?
4. assert_usable model-presence: exact file checks for GGUF + Gemma + LoRA + VAE (names/paths).
5. The Gemma encoder reads connector weights from the 46GB dev ckpt — is that a per-episode load cost across many shots, or cached? Should we point ckpt_name at the lighter embeddings_connectors?
6. ltx_av lanes: leave fully untouched, or align their sampler to the same Goofer recipe later?
