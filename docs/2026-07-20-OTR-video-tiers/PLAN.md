# OTR Video Tiers -- Code-Ready Hardened Plan (kibitz input)

**Date:** 2026-07-20
**Source directive:** `docs/2026-07-20-OTR-video-tiers-FINAL-lead-coder.md` (operator FINAL)
**This doc:** the code-grounded, code-ready hardening of that directive. Every
"grounded" fact below was verified against the REAL Windows files / disk on
2026-07-20. Feeds the `/kibitz` arc (r1 arc -> r2 coding -> r3 wiring -> r4
convergence).

---

## 0. The four final selectable names (unchanged from the directive)

| Final name | Implementation | Audio | Upscale | Status |
| --- | --- | --- | --- | --- |
| `ltx_8gb` | NEW: LTX-Video 0.9.8 distilled 2B (T5 family), own adapter | silent, mux after | none, single-pass | NEW engine |
| `wan_8gb` | existing `eng_wan_ti2v` (Wan 2.2 TI2V-5B Q5_K_M) | silent, mux after | none, single-pass | display alias of `wan_ti2v` |
| `ltx23_16gb_audio_in` | existing `eng_ltx_av` (`ltx_audio_in`) IA2V | audio conditions video; master preserved | unchanged | display alias of `ltx_audio_in` |
| `ltx23_16gb_video` | existing `eng_ltx_video` (`ltx_video`) two-stage HQ | silent, mux after | internal x2 latent upscaler stays active | display alias of `ltx_video` |

User-facing labels (from first implementation, no status words):
`LTX 0.9.8 2B - 8GB`, `Wan 2.2 TI2V 5B - 8GB`, `LTX 2.3 - 16GB Audio In`,
`LTX 2.3 - 16GB Video`.

---

## 1. Grounded facts (verified 2026-07-20 -- re-verify, do not rebuild)

### Registry / wiring mechanism (grounded)
- Video registry: `nodes/_otr_video_engines/registry.py`. Adapters self-register
  via `@register` (their `name` attr). `CAPABILITIES` is a parallel table;
  `nodes/_otr_shared/capability_profiles.py` DERIVES the per-profile enable-set
  from it. Registry-consistency invariant: a selectable engine needs BOTH an
  `@register`ed adapter AND a `CAPABILITIES` row that fits the active profile
  (`capability_profiles.validate_declaration` + `enabled_engines`).
- The dropdown is built from `registry.all_engine_names()` -> labels
  (`otr_video_director._video_model_combo` / `_label_for`). There is NO
  validated-subset filter (C4): every registered engine is selectable.
- Saved pick -> engine id via `otr_video_director._engine_id_from_pick`: strips
  the ` (suffix)`, then maps through `_LEGACY_ENGINE_ALIASES` (OLD -> current id).
- All three existing target engines already carry `requires_flag = None`
  (vestigial). `OTR_ENABLE_WAN_TI2V` / `OTR_ENABLE_LTX_AV` / `OTR_ENABLE_LTX_VIDEO`
  appear only in docstrings / error strings -- NONE gate selection or `assert_usable`.
  => "un-gated, no VRAM gate" is ALREADY the architecture. We must not ADD a gate.

### 16GB engines (grounded -- preserve, do not touch behavior)
- `eng_ltx_video.py` (`name="ltx_video"`, family `text_to_video`): `_recipe()`
  auto-detects `hq_two_stage` on the dev unet. The internal x2 upscaler is LIVE in
  `_node_candidates_hq` / `_build_graph_hq`: `LatentUpscaleModelLoader`
  (`upscale_loader`) + `LTXVLatentUpsampler` (`upscaler`) + default
  `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (`_ltx_hq_upscale_model_name`).
- `eng_ltx_av.py` (`name="ltx_audio_in"`, family `audio_conditioned_video`):
  V-1 silent -- graph terminates at `LTXVSeparateAVLatent -> video_latent ->
  VAEDecodeTiled`; audio branch (`LTXVAudioVAEDecode`) never wired;
  `OTR_MasterAudioMux` supplies audio. NVML REQUIRED (`assert_usable` fails closed
  w/o NVML) -> this is the existing `requires_vendor:"nvidia"` CAPABILITIES row.
  This is an EXISTING behavior of this engine; the directive says keep it. The
  display alias inherits it (same engine).

### wan_ti2v (grounded -- reuse for wan_8gb)
- `eng_wan_ti2v.py` (`name="wan_ti2v"`, family `image_to_video`, `render_aspect
  "wide"`): CORE Wan 2.2 TI2V-5B graph, GGUF default `Wan2.2-TI2V-5B-Q5_K_M.gguf`,
  umt5 GGUF encoder, REQUIRES `wan2.2_vae.safetensors` (M8 fail-closed allow-list).
  Silent clip (M7 ffprobe contract) + CLIP-FILL ping-pong extend to the beat
  window (no freeze-hold). `requires_flag=None`. Already the 8GB engine today.

### Disk state (grounded via probe 2026-07-20)
- `ltxv-2b-0.9.8-distilled.safetensors` -- **MISS** (must download).
- Forbidden `ltx-video-2b-v0.9.safetensors` (8.73 GB) -- PRESENT (do NOT use).
- 16GB stack all PRESENT: `ltx-2.3-22b-dev-Q3_K_M.gguf` (10.03 GB), dev+distilled
  video/audio VAEs, projection ckpt `ltx-2.3-22b-dev.safetensors` (42.98 GB),
  distilled LoRAs 384 + 384-1.1, and `latent_upscale_models/
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (0.93 GB, the internal x2 upscaler).
- Text encoders present: `t5xxl_fp16.safetensors` (9.12 GB), `t5-base.safetensors`
  (0.83 GB), umt5 GGUF + fp8. `t5xxl_fp8_e4m3fn.safetensors` MISS.
- Model root: `C:\ComfyUI-Models` (via extra_model_paths.yaml).

### HF pin (grounded via HF Hub 2026-07-20)
- Repo `Lightricks/LTX-Video` contains:
  - `ltxv-2b-0.9.8-distilled.safetensors` -- 6,340,744,492 B (~5.9 GB) <- USE THIS
  - `ltxv-2b-0.9.8-distilled-fp8.safetensors` -- ~4.15 GB (fallback if VRAM needs it)
  - `ltxv-spatial-upscaler-0.9.8.safetensors`, `ltxv-temporal-upscaler-0.9.8.safetensors`
    -- OUT OF SCOPE (8GB routes are single-pass/no-upscale).

### Rename blast radius (grounded)
- The three ids `wan_ti2v` / `ltx_video` / `ltx_audio_in` appear **420x across
  82+ .py/.json files** (render_driver 46, ~15 test files, every
  `config/profiles/*.json` + `workflows/variants/*.json`). A literal rename of the
  engine `name` attrs would be a sweeping, non-additive change touching routing,
  tests, and profiles. => favor a DISPLAY-ALIAS layer (below), not a rename.

---

## 2. `ltx_8gb` -- the only NEW engine

### 2.1 Assets to fetch (offline-first; download once, Test-Path after)
- Checkpoint: `Lightricks/LTX-Video/ltxv-2b-0.9.8-distilled.safetensors` ->
  `C:\ComfyUI-Models\checkpoints\` (all-in-one 0.9.x file: DiT + VAE embedded;
  T5 is SEPARATE). VERIFY-AT-SMOKE whether the VAE is embedded (0.9.x convention)
  or needs a separate `ltxv` VAE.
- Text encoder: reuse on-disk `t5xxl_fp16.safetensors` (T5-XXL, the 0.9.x family
  encoder). No new download expected. (fp8 variant is a VRAM fallback, currently
  MISS.)
- NO LoRA, NO projection ckpt, NO Gemma (those are the LTX-2.3 family, not 0.9.x).

### 2.2 Recipe (0.9.x family -- its OWN adapter, NOT the 2.3 graph)
Hypothesis to confirm at the standalone smoke (task: 0.9.8 smoke, go/no-go):
- Loader: `CheckpointLoaderSimple(ltxv-2b-0.9.8-distilled.safetensors)` -> MODEL,
  (VAE if embedded). If VAE not embedded, add the 0.9.x `VAELoader`.
- Text: `CLIPLoader(t5xxl_fp16.safetensors, type="ltxv")` -> CLIP -> `CLIPTextEncode`
  x2 (pos/neg).
- Latent: `EmptyLTXVLatentVideo(width,height,length,batch_size=1)`.
- Conditioning: `LTXVConditioning(frame_rate)`; distilled schedule (0.9.8 distilled
  is few-step: ~8 steps, cfg ~1.0 -- confirm the exact distilled sampler at smoke).
- I2V (OTR default): `LTXVImgToVideo` (or the ConditionOnly variant the installed
  pack exposes) on the beat's selected still.
- Decode: `VAEDecode` / `VAEDecodeTiled`.
- Installed `ComfyUI-LTXVideo` pack + CORE ComfyUI LTXV nodes provide these; the
  pack's `low_vram_loaders.py` is available if the smoke needs offload. The kickoff
  notes a "version" widget defaulting to `0.9.8` -- CONFIRM which node exposes it.

### 2.3 Adapter design (`nodes/_otr_video_engines/eng_ltx_8gb.py`)
- New module, cold-import clean (V-12): stdlib + dep-free shared helpers +
  registry only; torch / wrapper nodes lazy inside `load` / `render_clip`.
- Mirror the proven adapter shape (MotionEngineBase + wrapper_bridge declarative
  graph + silent bt709 encode + M7 silent-clip contract + CLIP-FILL beat coverage).
- `name = "ltx_8gb"`, `family = "image_to_video"` (OTR default is silent I2V on
  the still; supports T2V where nodes permit), `render_aspect` per the model's
  legal canvas (confirm at smoke; likely wide), `required_inputs = ("init_image",)`
  for the I2V default, `requires_flag = None`, `commercial_clean` = True (LTX-Video
  0.9.x is OpenRail-M / LTX open weights -- CONFIRM license classification).
- `assert_usable`: ordinary asset/node preflight ONLY -- checkpoint present + above
  a sanity floor, t5xxl present, required node classes resolve. NO VRAM query, NO
  vendor gate, NO fallback. (Do NOT copy eng_ltx_av's NVML gate.)
- `render_clip`: batch 1; encode prompt then offload/CPU the T5 before diffusion
  where the graph supports it; decode; return silent asset; master audio muxed by
  the existing `OTR_MasterAudioMux` path. Single-pass; NO upscaler node.
- Beat coverage: short clips loop/boomerang (reuse the shared helpers) to span the
  FULL beat window; never freeze-hold. Frame/canvas/step numbers come from the
  smoke + leg tests, NOT hardcoded "3-5s".

### 2.4 `CAPABILITIES` row for `ltx_8gb` (registry.py)
Model on `wan_ti2v` (cuda, no vendor gate, no fp8/fp4):
```
"ltx_8gb": {
    "required_toolchain": None, "requires_sidecar": False,
    "device_backends": ["cuda"], "requires_vendor": None,
    "needs_fp8_te": False, "needs_fp4_te": False,
    "practical_without_gpu": False, "sidecar_conditional": False,
    "model_requirements": ["ltxv-2b-0.9.8-distilled"]},
```
Must land in the SAME change as the `@register` (registry-consistency invariant).
Confirm it lands in the box's active-profile enable-set (nv50 / 16gb_full).

---

## 3. Wiring: display aliases for the 3 existing engines (r3 -- the key fork)

### Recommendation (anchor): DISPLAY-ALIAS layer, NOT an engine rename
Rationale: the directive says "final user-facing ALIASES only; do not fork or
rebuild their engines" and "alias of the existing adapter"; and a real rename
touches 420 references / ~15 tests / every profile (non-additive, high risk).
Keep the engine `name` attrs (`ltx_video` / `ltx_audio_in` / `wan_ti2v`) and their
CAPABILITIES keys UNCHANGED (all internal routing, tests, profiles keep working),
and add a thin presentation layer in `otr_video_director.py`:

1. Forward resolve map (new display id -> existing engine id):
   ```
   _DISPLAY_ALIASES = {
       "ltx23_16gb_video": "ltx_video",
       "ltx23_16gb_audio_in": "ltx_audio_in",
       "wan_8gb": "wan_ti2v",
   }
   ```
   Merge into `_engine_id_from_pick` (after the ` (suffix)` strip, before/with the
   legacy-alias lookup) so a saved `wan_8gb` resolves to `wan_ti2v`. Old raw ids
   still resolve (they ARE the registered ids -> bare passthrough).
2. Menu presentation: `_video_model_combo` shows the final display name for these
   three engines (relabel) so exactly the four final names appear as rows; the new
   `ltx_8gb` appears by ordinary registration. The old ids do NOT double-appear.
3. Saved value in the 4 preset JSONs = the final display names (resolve via
   `_DISPLAY_ALIASES`).

### Must-verify before committing to display-alias (r3 grounding)
- `render_driver.py` (46 refs) and every consumer must resolve the director's
  picked value through `_engine_id_from_pick` / `direct()` BEFORE using it as an
  engine id -- confirm no consumer reads the raw menu string and expects a bare
  engine id (a display name would fail there). If a consumer bypasses the resolver,
  route it through the resolver in the same change (still additive).
- Confirm the label round-trips: `_engine_id_from_pick(_label_for("wan_ti2v"))`
  path yields the engine, and picking the new label yields the engine too.

### Alternative (rejected unless the panel finds a blocker): full rename
Rename `name` attrs + CAPABILITIES keys + all 420 refs + `_LEGACY_ENGINE_ALIASES`
old->new + every profile/variant JSON + tests, with a machine-checkable stale-ID
audit (the keep-6 bank-rename pattern). Clean end-state id == display name, but
non-additive and high-risk; only adopt if display-alias has a real integration
blocker the panel proves.

---

## 4. Canonical JSON wiring (`workflows/otr_canonical.json`)
- The `OTR_VideoDirector` menu is DYNAMIC (built from `all_engine_names()` at
  `INPUT_TYPES`), so registering `ltx_8gb` + the display-alias relabels makes all
  four final names appear WITHOUT enumerating them in the JSON.
- Canonical JSON currently carries NO `ltx_*` / `wan_ti2v` engine string in a
  director widget (default is a non-LTX engine; leave the current default while
  tuning, per directive s10).
- Required JSON work IN THE SAME GREEN CHANGE as the code: keep the file valid +
  additive; re-run `OTR_WorkflowValidator` + JSON round-trip + link/widget audit so
  the new registry state round-trips cleanly. If a director widget's saved value
  must change to exercise a route, do it here (not a post-hoc edit).
- After operator testing: save 4 ordinary preset JSONs (NOT gated profiles) with
  the tested res/frames/fps/steps/decoder + selected engine:
  `workflows/otr_8gb_ltx.json`, `otr_8gb_wan.json`, `otr_16gb_ltx_audio_in.json`,
  `otr_16gb_ltx_video.json`.

---

## 5. Pre-video cleanup / ledger contract (operator "no hole in the ledger")
- Reuse the EXISTING cleanup path (directive s8): finish LLM/TTS/image -> unload
  LLM, TTS, image diffusion, prior CLIP/T5/Gemma encoders + VAEs + video models ->
  invoke ComfyUI free-memory -> confirm baseline before video load.
- `ltx_8gb` writes EVERY ledger field the existing video engines write (out_path,
  frame_count, silent-clip contract fields, recipe/vram stamps consumed by TTS
  slicing / shot direction / captions / credits / obs_publish). No new engine may
  leave a downstream ledger field unowned. Verify against the `wan_ti2v` /
  `ltx_video` canonicalize() output shape.

---

## 6. Clip sizing / beat coverage
- Do NOT hardcode "8GB = 3-5s". Tune each preset's resolution / legal frame count
  (model's Nn+1 rule) / fps / step schedule via testing. Progression A 512x288x49
  -> B 640x384x49 -> C 640x384x81, adjusted to the model's legal dims.
- Short clips loop/boomerang to span the full beat window; never freeze-hold.

---

## 7. Validation + commit order (directive s12)
1. Confirm assets/nodes for LTX 0.9.8 (download + Test-Path).
2. Standalone LTX 0.9.8 smoke on the RTX 5080 -- GO/NO-GO gate before wiring.
   (Reset box first: selective CIM kill, never a blanket python kill.)
3. Implement + register `ltx_8gb` (+ CAPABILITIES row).
4. Add `wan_8gb` display alias (un-gated; not dependent on OTR_ENABLE_WAN_TI2V).
5. Wire both 8GB rows into canonical JSON (menu is dynamic; validate round-trip).
6. Add `ltx23_16gb_audio_in` + `ltx23_16gb_video` display aliases (no engine code
   change).
7. Verify the 16GB internal x2 upscaler still connected/used.
8. `OTR_WorkflowValidator` + JSON round-trip + link audit + widget audit + AST
   parse + no-BOM + regression suite + Bug Bible after EACH code change.
9. Full OTR leg per route after real LLM/TTS/image + confirmed cleanup; renders ->
   `otr\episodes\<ep>\`, final -> `otr\obs\`; Test-Path asset before success.
10. Tune presets to reliable; 11. save the 4 preset JSONs; 12. commit AND push each
    green chunk to `v2.0-alpha`; verify HEAD==origin, no BOM, AST parse.

Commit chunking (each green + pushed):
- C1: `eng_ltx_8gb` + registry row + `ltx_8gb` reg (unit-green; smoke separately).
- C2: display-alias layer (`_DISPLAY_ALIASES` + combo relabel) for all 4 names.
- C3: canonical JSON round-trip validation + any widget default.
- C4..C7: per-route leg proofs + the 4 preset JSONs (as each goes green).

---

## 8. Open questions for the kibitz arc
- r1 (arc): confirm 0.9.8 2B distilled is the right 8GB pick and its recipe family
  (T5, not Gemma); confirm VAE embedded vs separate; confirm the installed pack's
  0.9.8 support + the "version" widget owner.
- r2 (coding): eng_ltx_8gb shape -- I2V default, T5 offload, distilled sampler
  values, license/commercial_clean classification, sanity floors.
- r3 (wiring): display-alias vs rename (verify no raw-value consumer bypasses the
  resolver); CAPABILITIES row fits the active profile; canonical round-trip.
- r4 (convergence): no new must-fix; ledger completeness; clip-fill coverage.

---

## 9. Non-goals (directive s13) -- do NOT do
experimental/candidate labels; later rename pass; VRAM gates; auto-detect that
hides rows; auto downgrade/fallback; new feature flags for the four rows; GPU
vendor/arch whitelists; new low-VRAM profile system; upscaler-bank scaffold; new
8GB upscaling; FP8/NVFP4/Q8 forks; HuMo changes; deletion of legacy aliases.
