# LTX clean-break splice plan (HARDENED — pass 01) — production video LTX → frozen GGUF/Goofer recipe

**Status:** roundtable-hardened (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4, 2026-06-15, grounded by Claude).
Two operator decisions resolved (below). Coder window executes; planner does not write production code.

## 0. Goal + SOURCE OF TRUTH
Clean-break replace the production VIDEO LTX recipe with the frozen, proven, license-safe
**`workflows/ltx_bookend_mini_repro_gguf_mit.json`** — that JSON is the **recipe source of truth**.
The splice replicates its exact node graph + values inside the engine for every LTX video shot
(announcer + music + scene/b-roll per-beat), each keeping its own per-shot prompt. No shim, no dark
gate. **SCOPE = MODEL/ENCODER/DECODE ONLY.** The recent shipped LTX infra STAYS, NOT re-touched: boomerang
`loop_via_reverse` (`e8a74da`), native 832x480 render (`4fc4268`), the `_sampler_mode` ksampler/distilled
knob (`5bcee2c`/`21bfe7a`), full-frame-flux-not-portrait (`977801a`, BUG-407). Swap ONLY loaders + decode.
Operator decisions (2026-06-15): **(A) recipe = the mini JSON (source of truth) but MODEL-ONLY — keep the
shipped `ksampler` default; distilled is the mini/i2v recipe, gated on a t2v motion smoke before any default
flip;** **(B) per-beat = scene/b-roll** (HuMo keeps character faces — no role change).

## 1. LTX inventory (verified)
- `ltx_video` `LtxVideoEngine` (eng_ltx_video.py:222) → **REPLACE (Phase 1)**; serves the 3 uses.
  Its `roles=("scene_broll","background_abstract","music_visual","announcer_visual")` (line 235)
  ALREADY covers announcer+music+scene-per-beat. Do NOT add `character_video` (decision B).
- `ltx_orbit` `LtxOrbitEngine` (eng_ltx_video.py:888) → **RIP (Phase 0)**.
- `ltx_av_talk` / `ltx_av_music` (eng_ltx_av.py) → **KEEP dark** (the proven GGUF/Gemma pattern to mirror).
- No LTX render outside `_otr_video_engines/`.

## 2. The recipe (verbatim from the frozen mini JSON — the source of truth)
`UnetLoaderGGUF(ltx-2.3-22b-dev-Q4_K_S.gguf)` → `LoraLoaderModelOnly(ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors, 0.70)`;
`VAELoader(ltx-2.3-22b-dev_video_vae.safetensors)`; `LTXAVTextEncoderLoader(gemma_3_12B_it_fp4_mixed, ltx-2.3-22b-dev, device="default")`
→ `CLIPTextEncode` pos/neg → `LTXVConditioning(25fps)`; `EmptyLTXVLatentVideo(832x480xN)` →
`LTXVImgToVideoConditionOnly(strength 0.75)`; **distilled** chain `KSamplerSelect(euler_cfg_pp)` +
`ManualSigmas(1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0)` + `RandomNoise` +
`CFGGuider(cfg 1.0)` + `SamplerCustomAdvanced`; `VAEDecodeTiled(512/64/4096/8)`. **No** `ModelSamplingLTXV`.
Boomerang = OTR's own ffmpeg (NOT VHS). License: ComfyUI core + ComfyUI-GGUF (Apache) + ComfyUI-LTXVideo
(LTX-2 Community). NO RES4LYF (AGPL), NO VHS (GPL).

## 3. PHASE 0 — rip `ltx_orbit` ONLY (separate commit, stays green)
**Do NOT delete the 2B/T5/VAEDecode loaders here** — they are the live `ltx_video` graph; deleting them
before Phase 1 bricks the engine (all 3 panelists). Phase 0 = remove `ltx_orbit` only:
- Delete class `LtxOrbitEngine` (eng_ltx_video.py:887-928) + drop it from `__all__`.
- Delete registry row `"ltx_orbit"` (registry.py:155) + its `CAPABILITIES` entry.
- Grep & remove every `ltx_orbit` / `LtxOrbitEngine` / `OTR_ENABLE_LTX_ORBIT` reference (dropdowns, profiles, docs, tests).
- Suite + Bug Bible green → commit Phase 0 alone (auditable). No JSON change → skip the workflow round-trip here.

## 4. PHASE 1 — swap the recipe inside `LtxVideoEngine` (one commit: delete old + add new together)
### 4A. eng_ltx_video.py — graph + loaders
- `_node_candidates()` (~:488): replace `"checkpoint":("CheckpointLoaderSimple",)` with
  `"unet":("UnetLoaderGGUF",)` + `"videovae":("VAELoader",)`; replace `"encoder":("CLIPLoader",)` with
  `"te":("LTXAVTextEncoderLoader",)`. Add `"lora":("LoraLoaderModelOnly",)` (always present, decision below).
- `_build_graph()` (~:622) + `_build_graph_i2v()` (~:420): wire `unet→lora(model,0.70)→model`,
  `videovae→vae`, `te→clip`. **Replace EVERY** `W("checkpoint",0)` with `W("lora",0)` and **every**
  `W("checkpoint",2)` with `W("videovae",0)` in BOTH builders (txt2vid + i2v). Inputs:
  `UnetLoaderGGUF{unet_name}`, `VAELoader{vae_name}`, `LTXAVTextEncoderLoader{text_encoder,ckpt_name,device:"default"}`.
- Terminal decode (~:349,:718): `"vaedecode":("VAEDecodeTiled",)` with `{samples, vae:W("videovae",0),
  tile_size:512, overlap:64, temporal_size:4096, temporal_overlap:8}` in both paths. No `W("checkpoint",2)` left.
- `render_clip` VRAM reclaim (line 805 `keep={"checkpoint",...}`, line 809 `results.get("checkpoint")`):
  **update the node key** to `"lora"` (the resident model patcher after LoRA) — else the patcher is dropped
  and VRAM leaks (Gemini, grounded). Verify which key holds the live model and keep THAT one.
- LoRA: remove the `"22b" in _ckpt_name()` gate (~:561); always include `"lora"`; require the LoRA file in
  `_installed()`/`assert_usable()`; fail closed if absent (no silent skip).
- Canvas: set `_LTX_DEFAULT_W=832`, `_LTX_DEFAULT_H=480` (lines 168-169 are 768/512 — the engine default,
  not just the per-request value).
- Sampler: **MODEL-ONLY swap — do NOT flip the sampler default.** Production shipped `ksampler` as default
  (`5bcee2c`) because distilled GATED t2v motion on the 2B (0.84 vs 7.85 framediff, music_open). The mini's
  distilled recipe is proven on the **i2v** path only; whether the 22B+distilled-LoRA clears the motion floor
  on **t2v** (music_open) is UNKNOWN until the §7 smoke runs BOTH samplers on the 22B. Keep `_sampler_mode`
  logic + both selectable (`OTR_LTX_SAMPLER`); the splice changes the loaders/decode, not the sampler default.
  Flipping the default to distilled (if the 22B clears the floor on all roles) is a SEPARATE follow-up commit.
- Path/usability rewrite: `_ckpt_path()`, `_text_encoder_name()`, `_installed()`, `load()` error string, and
  `assert_usable()` MISSING_MODEL → GGUF-aware (GGUF unet + Gemma + LoRA + video VAE present), mirroring
  `eng_ltx_av.py`'s resolution. Update the stale v0.9/2B/T5 docstrings + top-of-file comments.

### 4B. registry.py — capability declaration (not just code)
- `CAPABILITIES["ltx_video"]`: `model_requirements` 2B → GGUF unet + Gemma + distilled LoRA + video VAE; keep
  `vram_class:"heavy"` and re-measure `vram_estimate_mb` after the smoke (the mini held ~13GB).
- `commercial_clean` (eng_ltx_video.py:244 = False): verify-at-build whether profile filtering reads it; flip if so.

### 4C. render_driver.py — leave prompt logic ALONE
Per-shot prompts (`build_request_from_shot` + `_LTX_MOTION_PROMPT_BY_ROLE`) already vary per shot/role —
do not touch. Confirm every production LTX request supplies `canvas=832x480` (render_driver:840-846). 

### 4D. otr_scifi_16gb_full.json — no node change
No LTX nodes in the JSON. Do NOT append a GGUF-quant widget (panel: over-engineering; env/default is enough).
Only touch the JSON if a knob is truly wired through — then APPEND-only on `OTR_VideoDirector` (BUG-LOCAL-097).

## 5. BUG-413 + the i2v init image (corrected by grounding + operator)
`OTR_ENABLE_LTX_VIDEO` already DEFAULTS ON (eng_ltx_video.py:326 `os.getenv(...,"1")`) — it's an opt-out kill
switch, NOT a dark gate. So BUG-413's "LTX never executes" is the model-presence/Sage gates + the missing i2v
seed, not the flag. Fix = the GGUF recipe passes `assert_usable` (models present) + guarantee the i2v init exists.
Keep the flag as the kill switch. The LOUD `check_ltx_open_health` guard stays.

**LTX i2v init = FULL-FRAME FLUX still, NEVER a portrait (operator directive 2026-06-15):** every LTX shot's
i2v init image must be a full-frame FLUX-generated scene image (like the `radio_bookend` `scene_open` still),
NOT a character portrait — portraits are HuMo's face crops; an LTX shot fed a portrait yields a cropped-face
video instead of a full scene. Build requirement: confirm `build_request_from_shot`'s init_image resolution
(render_driver.py:722-745) routes LTX shots to the full-frame scene-still pool, and that FLUX actually
generates a full-frame scene still per LTX shot (announcer/music/per-beat). Regression assert: the LTX
init_image path exists AND is a full-frame still (not the portraits dir) for each LTX role; if a shot would
fall back to a portrait, that is a BUG — generate/route a full-frame still instead.

## 6. License clearance (must hold)
Adds only `UnetLoaderGGUF` (Apache). Keeps stock LTXV + stock Goofer samplers. MUST NOT introduce
`ClownSampler_Beta`/`MultimodalGuider` (RES4LYF AGPL) or `VHS_VideoCombine` (GPL). Add a grep test for those
3 banned class names + for residual `CheckpointLoaderSimple`/`CLIPLoader`/`VAEDecode`/`ltx-video-2b` in the engine.

## 7. Test gate (per CLAUDE.md) + the motion acceptance smoke
- After edit: regression suite + Bug Bible after EVERY change. Phase 0 + Phase 1 each green before commit+push.
- **Motion-acceptance smoke (decision A gate):** GPU smoke must render ≥1 announcer, ≥1 music, ≥1 scene/per-beat
  clip and assert a **frame-diff above a motion floor** (not just "a clip landed") — proves distilled isn't static
  on the 22B+LoRA recipe. If any role gates motion, fall back to `ksampler` for that path before shipping.
- BUG-413 R2 regression: assert `build_request_from_shot()` yields an `init_image` path that EXISTS for the
  bookend so `_use_i2v` can't drop to txt2vid.
- Reset GPU (CIM kill by CommandLine, :8000 free, VRAM baseline) before each smoke; single resident ≤14.5GB.
- Commit AND push per green chunk on `v2.0-alpha`; verify HEAD==origin, no BOM, AST parse.

## 8. Verify-at-build (perf tweaks ONLY if a full episode shows trouble — never silent deviations)
- Gemma `device`: mini uses `"default"` (proven 1-shot). If many-shot VRAM pressure, switch to `"cpu"` + the
  lighter `_projection_ckpt` (embeddings_connectors) like `eng_ltx_av.py` — measured, not assumed.
- `VAEDecodeTiled temporal_size`: mini/b001 use **4096** (production-proven at episode frame counts). Only drop
  to 64 (the av lane's value) if a longer-than-tested shot OOMs.
- `ModelSamplingLTXV`: mini omits it and looks good → omit. Add only if a look-test regresses.
- `commercial_clean` flip pending the profile-filter check (§4B).

## 9. PASS-02 HARDENING (grounded folds — these supersede any looser wording above)

**9.1 Resolve the model-vs-sampler contradiction (the core one).** The mini JSON is the source of truth for
the **model/encoder/decode GRAPH ONLY** — the loaders (`UnetLoaderGGUF`+`VAELoader`+`LTXAVTextEncoderLoader`+
`LoraLoaderModelOnly`) and `VAEDecodeTiled` + the distilled sigma VALUES. The **SAMPLER is an orthogonal,
pre-existing engine knob** (`_sampler_mode`, `OTR_LTX_SAMPLER`): the shipped default stays `ksampler`. "Exact
mini graph" therefore means the model/decode nodes, NOT the sampler choice. The §7 smoke decides the ship
default per role. Delete any "replicate exact node graph (incl. sampler)" wording — it's model/decode only.

**9.2 LoRA wiring — NEVER self-reference.** `graph["lora"].inputs.model = W("unet", 0)`. The guider/sampler
`model` input = `W("lora", 0)`. Do NOT blanket-replace `W("checkpoint",0)`→`W("lora",0)`: the LoRA node's own
model input is `W("unet",0)`; only the DOWNSTREAM model consumers read `W("lora",0)`.

**9.3 Separate file resolvers (do not reuse `_ckpt_name()` for both).** Add: `_unet_name()`/`OTR_LTX_VIDEO_UNET`
(→ `ltx-2.3-22b-dev-Q4_K_S.gguf`), `_projection_ckpt()`/`OTR_LTX_VIDEO_PROJECTION_CKPT` (→ `ltx-2.3-22b-dev.safetensors`,
the `LTXAVTextEncoderLoader.ckpt_name`), `_video_vae_name()`/`OTR_LTX_VIDEO_VAE`, `_encoder_name()`/`OTR_LTX_VIDEO_TEXT_ENCODER`.
`UnetLoaderGGUF{unet_name:_unet_name()}`; `LTXAVTextEncoderLoader{text_encoder:_encoder_name(), ckpt_name:_projection_ckpt(), device:"default"}`.
ALL FOUR files (GGUF unet, Gemma encoder, projection ckpt, video VAE) + the LoRA go in `_installed()`/`assert_usable()`/
`CAPABILITIES` with min-size floors, mirroring `eng_ltx_av.py`.

**9.4 Sigmas shim is allowed-internal.** The engine's distilled path injects sigmas via the internal
`_SigmasFromValues` shim (NOT a literal `ManualSigmas` node). That is FINE and stays — the mini's `ManualSigmas`
maps to it; only the sigma VALUES must match (they already do). This is the one sanctioned internal shim; it does
not violate "no shim" (which is about back-compat/dead-code shims, not the sigma injector).

**9.5 `render_clip` VRAM keep-set under `free_after_use=True`.** The resident model after the swap is the
LoRA-wrapped UNET. Verify whether `keep={"lora", self._TERMINAL}` retains the underlying UNET patcher or whether
`free_after_use` evicts it — if eviction, use `keep={"unet","lora",self._TERMINAL}`. `model = results.get(<resident key>,(None,))[0]`.
Prove no VRAM leak via the NVML assert in the smoke (BUG-291 reclaim must still drain between beats).

**9.6 `_use_distilled_lora()` — neuter, don't gate.** Remove the stale `"22b" in _ckpt_name()` logic entirely;
always include `"lora"` in the candidates for BOTH sampler modes; require the LoRA file in `_installed()`/`assert_usable()`;
fail closed if absent. (Note Grok's key-mismatch worry: confirm the distilled LoRA applies cleanly on the ksampler
path too, or the smoke catches it.)

**9.7 Structured anti-regression test (not a raw grep).** Assert no node-candidate tuple or graph class EQUALS
`"VAEDecode"`/`"CheckpointLoaderSimple"`/`"CLIPLoader"` (exact, so `VAEDecodeTiled` does not false-match) and no
`("ClownSampler_Beta",)`/`("MultimodalGuider",)`/`("VHS_VideoCombine",)` anywhere. Drop the raw-substring grep.

**9.8 §7 motion smoke = a MATRIX.** {`ksampler`,`distilled`} × {t2v, i2v} × {announcer, music, scene/per-beat},
each above a frame-diff motion floor on the GGUF graph. **Shipping `ksampler` default REQUIRES the ksampler rows
pass on the GGUF** (currently unproven); flipping to distilled requires the distilled rows pass. Until proven,
neither default is assumed.

**9.9 `commercial_clean` — decide now.** Read how the profile filter consumes `commercial_clean` (registry
protocol does read it); set the correct value for the GGUF recipe in THIS splice, or explicitly scope profile-filter
behavior OUT with a one-line ticket. Do not leave a runtime policy bit as an open guess.

**9.10 CUTS (pass-02 consensus).** Drop the §4D GGUF-quant-widget warning entirely (no LTX node in the JSON).
Drop the "No ModelSamplingLTXV" callout from required steps (already omitted in the mini; verify-at-build only).
The Gemma `device=cpu`/projection-ckpt/temporal-64 tweaks stay §8 post-smoke perf notes, not build steps.

**9.11 GO-FORWARD hygiene (operator 2026-06-15).** Mark `LTX-REGR` (GO_FORWARD §5) **SUPERSEDED BY this splice**
(it planned to bake the *2B* recipe into `eng_ltx_video.py`; the splice does the 22B GGUF evolution instead — the
coder must NOT do the 2B bake). Remove `ltx_orbit` from the "Ship defaults" selectable list (§5). Keep BUG-411
(flux) — it's the i2v-init pipeline, adjacent-supportive, not LTX.
