# LTX clean-break splice plan — production video LTX → frozen GGUF mini recipe (CONSISTENT REWRITE, pass03-hardened)

Coder window executes; planner does not write production code. Grounded vs `eng_ltx_video.py` / `registry.py` /
`eng_ltx_av.py` across roundtable passes 01-03 (2026-06-15).

> **#1 HARD INVARIANT (operator).** The frozen mini json `workflows/ltx_bookend_mini_repro_gguf_mit.json` is the
> VERIFIED-WORKING render path. The splice MUST reproduce that **graph + values EXACTLY** — and the sampler is
> part of it: GGUF unet + distilled LoRA @0.70 + Gemma encoder + the **distilled** chain (`KSamplerSelect
> euler_cfg_pp` + `ManualSigmas` 8-step + `RandomNoise` + `CFGGuider cfg 1.0` + `SamplerCustomAdvanced`) +
> `VAEDecodeTiled 512/64/4096/8` + `LTXVImgToVideoConditionOnly strength 0.75` + 832x480. The wiring BEFORE/AFTER
> it (resolvers, init-image sourcing, VRAM keep-set, registry, per-shot prompt) is FLEXIBLE; the inner graph is
> not. Every smoke compares against the mini's known-good output, not "a clip landed".

## 1. SCOPE + decisions
- **Model/encoder/decode ONLY.** Shipped infra STAYS, untouched: boomerang `loop_via_reverse` (`e8a74da`),
  native 832x480 (`4fc4268`), full-frame-flux-not-portrait (`977801a`). Swap ONLY the loaders + decode + sampler-mode-default.
- **(A) Sampler = `distilled` GLOBAL default** (the mini). NOT a per-role fallback — `_sampler_mode()` is global and
  `load()` caches the resolved classes, so per-request sampler switching is not supported and is OUT of this splice.
  The §6 smoke must prove distilled clears the motion floor on EVERY production role (announcer/music/scene, t2v + i2v).
  If music_open t2v gates on the 22B, that is a SEPARATE future "per-role sampler selector" ticket — not this splice.
  (`ksampler` stays a global env knob `OTR_LTX_SAMPLER=ksampler` for manual A/B only.)
- **(B) per-beat = scene/b-roll.** `LtxVideoEngine.roles` already covers announcer_visual/music_visual/scene_broll.
  Do NOT add `character_video` (HuMo keeps faces).

## 2. LTX inventory
`ltx_video` (`LtxVideoEngine` eng_ltx_video.py:222) → **REPLACE (Phase 1)**. `ltx_orbit` (`LtxOrbitEngine`:888,
subclass) → **RIP (Phase 0)**. `ltx_av_talk`/`ltx_av_music` (eng_ltx_av.py) → **KEEP dark** (the proven GGUF/Gemma
pattern to MIRROR for resolvers + floors + node-class gate). No LTX render outside `_otr_video_engines/`.

## 3. PHASE 0 — rip `ltx_orbit` ONLY (own commit, green before Phase 1)
Do NOT delete the 2B loaders here (they are the live graph). Remove ONLY orbit:
- Delete class `LtxOrbitEngine` (:887-928) + its `@register`; drop from `__all__`.
- Delete registry row `"ltx_orbit"` (registry.py:155) + its `CAPABILITIES` entry.
- Grep & remove EVERY `ltx_orbit`/`LtxOrbitEngine`/`OTR_ENABLE_LTX_ORBIT` ref incl. the shared comments in
  `_LOOP_VIA_REVERSE_DEFAULT` + `render_clip`, dropdowns, profiles, docs, tests.
- Suite + Bug Bible green → commit. No JSON change.

## 4. PHASE 1 — swap the recipe inside `LtxVideoEngine` (ONE commit: old out + new in together)
**4A. Resolvers (add; do NOT reuse `_ckpt_name()` for two files).**
`_unet_name()`/`OTR_LTX_VIDEO_UNET`→`ltx-2.3-22b-dev-Q4_K_S.gguf`; `_projection_ckpt()`/`OTR_LTX_VIDEO_PROJECTION_CKPT`
→`ltx-2.3-22b-dev.safetensors`; `_video_vae_name()`/`OTR_LTX_VIDEO_VAE`→`ltx-2.3-22b-dev_video_vae.safetensors`;
`_encoder_name()`/`OTR_LTX_VIDEO_TEXT_ENCODER`→`gemma_3_12B_it_fp4_mixed.safetensors`; LoRA `_distilled_lora_file()`
→`ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors`. (Use exact filenames WITH extensions.)

**4B. `_node_candidates()` (~:488) + i2v candidate set:** drop `"checkpoint"`/`"encoder"`; add
`"unet":("UnetLoaderGGUF",)`, `"videovae":("VAELoader",)`, `"te":("LTXAVTextEncoderLoader",)`,
`"lora":("LoraLoaderModelOnly",)` — in BOTH the base and the i2v candidate dicts. `"vaedecode":("VAEDecodeTiled",)`.
Delete `_use_distilled_lora()` (or hard-return True); `"lora"` is ALWAYS present in both sampler modes.

**4C. `_build_graph()` (~:622) + `_build_graph_i2v()` (~:420) — exact wiring (no self-reference):**
- `unet`: `UnetLoaderGGUF{unet_name:_unet_name()}`
- `lora`: `LoraLoaderModelOnly{model: W("unet",0), lora_name:_distilled_lora_file(), strength_model:0.70}`
- `videovae`: `VAELoader{vae_name:_video_vae_name()}`
- `te`: `LTXAVTextEncoderLoader{text_encoder:_encoder_name(), ckpt_name:_projection_ckpt(), device:"default"}`
- **downstream model inputs** (guider/sampler) = `W("lora",0)` — NEVER `W("lora",0)` into lora's own model.
- **all vae inputs** (`img2vid.vae` + `vaedecode.vae`) = `W("videovae",0)`. **Zero `W("checkpoint",*)` left anywhere.**
- decode node `VAEDecodeTiled{samples, vae:W("videovae",0), tile_size:512, overlap:64, temporal_size:4096, temporal_overlap:8}`.

**4D. `render_clip` VRAM reclaim (:805 `keep={"checkpoint",...}`, :809 `results.get("checkpoint")`):** the resident
patcher is now the LoRA-wrapped unet. A unit/smoke test must assert the retained key under `free_after_use=True`;
set `keep={"unet","lora",self._TERMINAL}` and `results.get(<proven key>,(None,))[0]` accordingly; NVML drain asserts no leak.

**4E. Canvas + sampler + usability + comments:**
- `_LTX_DEFAULT_W=832`, `_LTX_DEFAULT_H=480` (engine defaults at :168-169, currently 768/512).
- `_sampler_mode()` default `distilled` (per #1 invariant; `ksampler` env-only).
- `assert_usable()` gains a **node-class gate** (mirror eng_ltx_av.py) for `UnetLoaderGGUF/VAELoader/
  LTXAVTextEncoderLoader/LoraLoaderModelOnly/VAEDecodeTiled` BEFORE declaring usable; `_installed()`/`_weight_paths()`
  validate ALL FIVE artifacts with **min-size floors** (define `_FLOOR_*` like eng_ltx_av.py: GGUF unet ~10GB,
  Gemma ~5GB, projection ckpt ~30GB, video VAE ~1GB, LoRA ~5GB — ballpark, confirm vs actual). Update the
  `MISSING_MODEL` message + the stale v0.9/2B/T5/1472x832/`VAEDecode` docstrings + comments.
- Init-time fail-closed: these run LAZILY at dispatch; missing model → `EngineUnusable` LOUD → fallback, never a
  module-load raise.

**4F. registry.py.** `CAPABILITIES["ltx_video"].model_requirements` 2B → the 5-artifact tuple; keep `heavy`;
re-measure `vram_estimate_mb` post-smoke (mini ~13GB). **`commercial_clean` (eng_ltx_video.py:244=False): DECIDE in
this commit** — read how the profile filter consumes it; the GGUF recipe is license-clean (Apache GGUF + LTX-2
Community model, no AGPL/GPL), so set it to reflect "usable in commercial profiles" if that is the field's meaning,
else leave + a one-line ticket. No open guess.

**4G. render_driver.py — prompt logic UNTOUCHED.** `build_request_from_shot` + `_LTX_MOTION_PROMPT_BY_ROLE` already
vary per shot/role. Confirm canvas=832x480 supplied per request. (Verify, do not rewrite.)

**4H. `otr_scifi_16gb_full.json` — NO node change** (no LTX nodes in it). Re-run `OTR_WorkflowValidator` + JSON
round-trip ONLY to PROVE the shell is untouched. (Cut the GGUF-quant-widget discussion — dead weight.)

## 5. BUG-413 + the i2v init image
`OTR_ENABLE_LTX_VIDEO` already defaults ON (:326) — opt-out kill switch, keep it. BUG-413 = model-presence/Sage
gate + missing i2v seed, not the flag; the GGUF recipe passing `assert_usable` + a guaranteed init image fixes it.
**LTX i2v init = FULL-FRAME FLUX scene still, NEVER a portrait (operator).** `_init_image_path()` reads `asset_refs`
in order `still → init_image → image`; ensure render_driver puts the FULL-FRAME scene still in `asset_refs["still"]`
for LTX (or add an LTX-specific resolver that prefers the scene key over a portrait). Regression: with BOTH a
portrait ref and a scene ref present, the LTX init resolves to the FULL-FRAME scene (open the file, assert aspect),
and never the portraits dir.

## 6. Test gate + motion-acceptance smoke
- Suite + Bug Bible after EVERY change; Phase 0 then Phase 1 each green before commit+push.
- **Structured anti-regression unit test** (not a raw grep): assert no candidate/graph class EQUALS `"VAEDecode"`/
  `"CheckpointLoaderSimple"`/`"CLIPLoader"` and none equals `"ClownSampler_Beta"`/`"MultimodalGuider"`/`"VHS_VideoCombine"`;
  assert the built graph contains `unet→lora→guider.model`, `videovae→{img2vid.vae,vaedecode.vae}`, `te→clip`, and
  `"lora"` present in BOTH sampler modes.
- **Motion smoke (the §1 invariant gate):** (1) reproduce the mini i2v bookend and assert it MATCHES the mini's
  known-good output; (2) render ≥1 announcer + ≥1 music + ≥1 scene/per-beat with `distilled` on the 22B and assert
  **frame-diff ≥ a stated floor** (define numerically — the 2B static read 0.84, dynamic 7.85; require e.g. ≥2.0 per
  role, tied to the `otr_ltx_motion_smoke` baseline). If a t2v role gates, FILE the per-role-sampler ticket; do not
  hand-hack it into this splice.
- Reset GPU before each smoke (CIM kill by CommandLine, :8000 free, VRAM baseline); single resident ≤14.5GB.
- Commit AND push per green chunk on `v2.0-alpha`; verify HEAD==origin, no BOM, AST parse.

## 7. WIRING + REGISTRATION + INIT (operator focus)
- `@register` stays on `LtxVideoEngine`; orbit's is deleted; no double-register/orphan name.
- `__init__.py` guarded import still succeeds post-orbit-delete (grep `__init__`/`__all__` for orbit).
- registry-consistency invariant holds (every registered engine ↔ exactly one CAPABILITIES row).
- `role_compat`/schemas still resolve for the 3 roles with `required_inputs=("text_prompt",)`; none reference orbit
  (the "zod/schema-init" concern).
- NO DORMANT CODE: every new candidate is REFERENCED in the live graph (the §6 graph-shape test proves it).
- Every new node class resolves via `wrapper_bridge` NODE_CLASS_MAPPINGS at render (fail LOUD if absent).

## 8. License clearance (must hold)
Adds only `UnetLoaderGGUF` (Apache) + keeps stock LTXV + stock distilled samplers. MUST NOT introduce RES4LYF
(AGPL) or VHS (GPL). The §6 banned-class test enforces it.

## 9. VERIFY-AT-BUILD (perf only; never silent deviations from the mini)
Gemma `device="cpu"` + lighter projection-ckpt (av-lane pattern) ONLY if a full episode shows many-shot VRAM
pressure; `temporal_size` 4096 (mini/b001-proven) → 64 only if a longer-than-tested shot OOMs; `ModelSamplingLTXV`
stays omitted (mini-proven) unless a look-test regresses.

## 10. GO-FORWARD hygiene (operator — no old-LTX patching)
Mark `LTX-REGR` (GO_FORWARD §5) **SUPERSEDED BY this splice** — it planned to bake the **2B** recipe into
`eng_ltx_video.py`; the splice does the 22B GGUF evolution instead, so the coder must NOT do the 2B bake. Remove
`ltx_orbit` from the "Ship defaults" selectable list (§5). Keep BUG-411 (flux) — it feeds the full-frame i2v init,
adjacent-supportive, not LTX.
