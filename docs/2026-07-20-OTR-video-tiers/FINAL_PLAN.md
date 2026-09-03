# OTR Video Tiers -- FINAL code-ready plan (kibitz arc converged 2026-07-20)

Converged output of the r1->r4 /kibitz arc (codex gpt-5.6-sol + antigravity
Gemini 3.5 Flash High; Claude anchor + judge). Supersedes PLAN.md and all r*_plan.md.
Every claim grounded against the real files / installed pack / disk on 2026-07-20.
Implements docs/2026-07-20-OTR-video-tiers-FINAL-lead-coder.md (operator directive,
which wins on any conflict).

## 0. What ships
Four selectable video rows, additive over the registry-is-the-menu superstructure,
no rename of the ~420 internal id refs, no new gate/profile/fp8 fork, HuMo untouched:
`ltx_8gb` (NEW engine, LTX-Video 0.9.8 distilled 2B, I2V this sprint) |
`wan_8gb` -> internal `wan_ti2v` | `ltx23_16gb_audio_in` -> internal `ltx_audio_in` |
`ltx23_16gb_video` -> internal `ltx_video`.

## 1. Public-name resolver -- `nodes/_otr_shared/public_engines.py` (NEW, stdlib-only, cold-import clean)
```
_PUBLIC_ENGINES = {"ltx_8gb":"ltx_8gb","wan_8gb":"wan_ti2v",
                   "ltx23_16gb_audio_in":"ltx_audio_in","ltx23_16gb_video":"ltx_video"}
_LEGACY_ENGINE_ALIASES = {"flat_still":"still_flat","flux_still":"still_pan",
                          "still_kenburns":"still_motion","visualizer":"viz_green"}  # MOVED here
_INTERNAL_TO_PUBLIC = {v:k for k,v in _PUBLIC_ENGINES.items()}
_PUBLIC_LABEL = {"ltx_8gb":"LTX 0.9.8 2B - 8GB","wan_8gb":"Wan 2.2 TI2V 5B - 8GB",
                 "ltx23_16gb_audio_in":"LTX 2.3 - 16GB Audio In",
                 "ltx23_16gb_video":"LTX 2.3 - 16GB Video"}  # TOOLTIP/DOCS ONLY
assert len(_PUBLIC_ENGINES)==len(_INTERNAL_TO_PUBLIC)  # bijection (unique internals)

def resolve_engine_id(value):
    bare = str(value or "").split(" (",1)[0]
    resolved = _PUBLIC_ENGINES.get(bare, bare)          # public -> internal
    return _LEGACY_ENGINE_ALIASES.get(resolved, resolved)  # then legacy -> current
```
Menu representation (DECISION, resolves the label question): the combo VALUE = short
public id + existing aspect/descriptor suffix (`wan_8gb (16:9)`), consistent with the
current menu (id tokens + suffix). Friendly `_PUBLIC_LABEL` -> static widget tooltip +
docs only. (Operator eyeball: flip to friendly-prose dropdown strings = a one-line
label-prefix swap + resolver keys on the label.)

### Boundaries calling resolve_engine_id (all grounded; import from the dep-free module)
1. `otr_video_director._engine_id_from_pick` -> wrapper over resolve_engine_id.
   `otr_video_director` imports `_LEGACY_ENGINE_ALIASES` FROM public_engines (single source).
2. `_aspect_suffix`/`_descriptor_suffix`: resolve before `_vreg.get_engine`.
3. `_label_for(internal)` = `_INTERNAL_TO_PUBLIC.get(internal,internal) + _aspect_suffix(internal)
   + _descriptor_suffix(internal)`. `_video_model_combo`: dedup by public id, pass the
   INTERNAL id to `_label_for`. Preserves `_engine_id_from_pick(_label_for("wan_ti2v"))=="wan_ti2v"`.
4. `exact_menu_option_for(internal_id)` (NEW): the unique live-combo option whose
   resolve==internal_id (fail on 0/>1); used by the applier + build_variants for ALL
   FOUR public engines.
5. `_otr_workflow_apply._is_engine_director_admissible` (:234): `resolve_engine_id(bare)
   in all_engine_names()`. Writing an OTR_VideoDirector role widget -> write
   `exact_menu_option_for(internal)`; leave OTR_VideoRenderBatch.engine internal.
6. `render_driver.parse_engine_override` (:2490): resolve before is_registered/ENGINE_FAMILY.
7. `capability_profiles.cross_validate_profile`: resolve before CAPABILITIES membership.
NOT a boundary: `role_slots._engine_id_of` (shot_lock gets the Director's RESOLVED
`policy["video_models"]`; a resolve there would mask an upstream break -- proven by the
ShotLock internal-only test).

## 2. C0 -- discovery + assets + smoke (blocking go/no-go; NO engine code)
### 2a. Live discovery receipt -> `docs/2026-07-20-OTR-video-tiers/ltx_8gb_discovery.{md,json}`
In-process `/object_info` probe on the 5080 (record raw + normalized hash + resolved
class names): the 0.9.x loader class + its `folder_paths` model CATEGORY; T5 CLIPLoader
class + `type` token + device support; conditioning/sampler/scheduler/cfg/steps; i2v
node; decode (tiled vs untiled, LTXVTiledVAEDecode contract); legal dims (step 32) +
frame rule (8n+1, min 9, discovered max). EXCLUDE `LTXQ8Patch` (owns the `0.9.8`
widget; NOT the loader). Confirm VAE embedded in the all-in-one checkpoint; if separate,
fetch the 0.9.x VAE from Lightricks/LTX-Video (assert NO 2.3 VAE enters the graph).
Record the LTX 0.9.x license -> set `commercial_clean` from evidence.
### 2b. Asset
`scripts/download_ltx_0_9_8.ps1` via an EXTENDED `hf_download_driver.py` (accept
`revision`; verify byte length + SHA-256; reject mismatch): repo `Lightricks/LTX-Video`,
file `ltxv-2b-0.9.8-distilled.safetensors`, size 6340744492, SHA256
76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f (VERIFY vs the actual
download), dest = the discovered loader category, T5 dep t5xxl_fp16 (on disk). Pin the
exact commit SHA. NO fp8 fork. `Test-Path` after.
### 2c. Standalone functional smoke (operator directive s12.2 -- BEFORE the adapter)
Throwaway in-process node-class probe (wrapper_bridge-style class calls, NOT an ad-hoc
API JSON graph -- honors s0) rendering one legal-minimum 0.9.8 I2V clip to
`otr/episodes/<smoke_ep>/` (reset box first: selective CIM kill). Proves the discovered
graph renders. Delete the probe. Failure = STOP + report (no fp8 drift).

## 3. C1 -- eng_ltx_8gb + registration (ONE commit: registry + JSON consistent)
`nodes/_otr_video_engines/eng_ltx_8gb.py`, cold-import clean; MotionEngineBase +
wrapper_bridge graph + silent bt709 encode + M7 contract + shared CLIP-FILL (loop/
boomerang to the full beat, no freeze-hold).
- `name="ltx_8gb"`, `family="image_to_video"`, `required_inputs=("init_image",)`,
  `render_aspect` per discovery, `requires_flag=None`, `commercial_clean` from the license.
- `assert_usable`: ordinary preflight ONLY (checkpoint + T5 present + sanity floors
  [ckpt>=4GiB]; node classes resolve). NO NVML/vendor/VRAM gate, NO fallback.
- frames: LTX 8n+1 quantizer (mirror eng_ltx_video `((L-1)//8)*8+1`, min 9, discovered
  max) -- NOT wan's 4n+1 `compute_real_frame_budget`.
- T5 offload: use the ONE route settled at discovery (CPU-device CLIPLoader OR a
  two-graph encode/diffuse split with a tracked patcher detach); ADOPT only if the
  offload-OFF-vs-ON measurement shows peak reduction past tolerance without RAM spill
  (prior LTX offload was measured ineffective). Keep MODEL, register patcher in
  `prepared["patchers"]`, verify teardown. Decode tiled if the 8GB peak needs it;
  single-pass (NO upscaler node). Silent; master audio via OTR_MasterAudioMux.
- output: follow the proven wan/ltx pattern (render_driver places the final asset in
  otr/episodes/<ep>/; Test-Path verifies at leg-test).
- `_clip_from_raw`: CanonicalClip dict at parity with wan/ltx + top-level receipt keys
  `recipe` (defined string), `render_canvas` (= REQUESTED dims), `vram_peak_mb` (THREAD
  the measured VramPeakProbe peak). NO CanonicalClip schema change (render_driver
  reads the dict via `clip.get(...)`, not the extra=forbid model -- grounded
  render_driver.py:2801-2825).
- `CAPABILITIES["ltx_8gb"]` modeled on wan_ti2v (cuda, no vendor gate, no fp8/fp4),
  `model_requirements=["ltxv-2b-0.9.8-distilled"]`. Registration = guarded
  `from . import eng_ltx_8gb` in `__init__.py`, SAME commit.
- ALSO in C1 (or a tiny sibling): thread `render_peak` into wan_ti2v's returned raw so
  wan_8gb's manifest receipt is populated (NEWBUG-1; required for wan 8GB acceptance).
- Validation gate (full, s0): OTR_WorkflowValidator + JSON round-trip + link referential
  integrity + wired-input-name audit + widget-count/live-INPUT_TYPES audit + AST parse
  + no-BOM + regression suite + Bug Bible. Canonical stores NO aliased id (grep=0) so
  no node-value edit -- validate the round-trip only.

## 4. C2 -- public_engines + boundaries + variant regen + tests
public_engines.py + boundaries 1-7 + `exact_menu_option_for` + menu relabel. REGENERATE
`workflows/variants/{otr_amd16_rocm,otr_amd8_rocm,otr_nv40_12gb}.json` via
build_variants (they carry wan_ti2v from their profiles -> now write the exact public
option) + refresh master hashes so `build_variants.py --check` stays green. Tests:
resolve round-trips (each public id, each legacy alias, each internal id, each suffixed
label); menu uniqueness (4 public once, no aliased duplicate, ADD_CUSTOM); canonical
stored values in the live choices; profile apply -> director public option -> direct()
-> resolved internal; ShotLock internal-only (public option -> direct -> build_execution
_plan -> only internal ids in groups/shots); forced-engine parse; cross_validate; cold
import; downloader revision+integrity (reject wrong size/SHA/missing/partial). UPDATE
test_still_aspect_and_labels.py siblings + test_workflow_apply.py assertions that pin
exact labels/written values for the 3 aliased ids.

## 5. C3+ -- presets + per-route legs (each green + pushed)
### Preset spec (generated via build_variants; NOT hand copies)
| preset JSON | base profile | Director option (all 3 slots) | RenderBatch engine | canvas | fps | frame budget | required_models |
| --- | --- | --- | --- | --- | --- | --- | --- |
| otr_8gb_ltx.json | 8gb_lite | `ltx_8gb (<aspect>)` | ltx_8gb | discovery | disc | disc | ltxv-2b-0.9.8-distilled(+T5) |
| otr_8gb_wan.json | 8gb_lite | `wan_8gb (16:9)` | wan_ti2v | 832x480 | 25 | tuned | wan2.2-ti2v-5b(+umt5+vae) |
| otr_16gb_ltx_audio_in.json | 16gb_full | `ltx23_16gb_audio_in (16:9)` | ltx_audio_in | 832x480 | 25 | tuned | ltx-2.3 stack |
| otr_16gb_ltx_video.json | 16gb_full | `ltx23_16gb_video (16:9)` | ltx_video | 832x480 | 25 | tuned | ltx-2.3 stack |
Each preset: build_variants emits the JSON (role slots = the tier's exact public option;
canvas/fps via the widget mapping) + stamps `OTR_WorkflowValidator.workflow_json_path`
to itself + a paired `<variant>.env.json` (allowlisted recipe-knob keys the widgets do
not own: steps/decode-tiling/i2v-strength) with a workflow/hash binding. required_models
filled from the real asset tuple. Locked acceptance: ALL THREE video slots use the tier
engine (no mixed-route this sprint).
### Env + server hygiene
A launcher loads the matching `<variant>.env.json` (UTF-8, CLEAR the full prior-tier
key set first) before boot; ONE env per workflow. A distinct tier env => a FRESH
reset/boot (a running server keeps its boot env); only cold+3-warm of the SAME tier
share a server.
### Live legs (s0 -- NEVER load a preset as the source workflow)
For each route: reset box (selective CIM kill), boot with the tier env, load
`workflows/otr_canonical.json`, apply the tier profile IN MEMORY (otr_api profile seam),
run the full LLM/TTS/image->video path after confirmed cleanup; renders ->
otr/episodes/<ep>/, final -> otr/obs/; Test-Path the asset; RESULT SUCCESS + obs_publish
OK. cold + 3 warm per route.
### 8GB acceptance (testing telemetry, NEVER a code gate)
budget_mib = the operator's target-card budget (e.g. 8192) minus headroom_mib (e.g.
512 desktop baseline); telemetry = NVML TOTAL GPU memory (not process delta); PASS =
EVERY cold+3-warm whole-path peak <= (budget_mib - headroom_mib); sampling ~0.1s
(VramPeakProbe); telemetry absent -> record "unmeasured" (never a silent pass). Adopt
T5 offload / tiled decode only when measured reduction meets tolerance + preserves
output + meets budget + no RAM spill. Record peak/RAM/pagefile/render+decode time.

## 6. Commit order (each green + pushed to v2.0-alpha; verify HEAD==origin, no BOM, AST)
C0 (discovery+download+smoke receipts) -> C1 (eng_ltx_8gb + row + __init__ + wan-peak
thread + full validation) -> C2 (public_engines + boundaries + variant regen + tests)
-> C3+ (presets via build_variants + per-route legs + env, each green). Regression
suite + Bug Bible after every code change.

## 7. Non-goals + CUTS (directive s13)
No experimental/candidate labels, later rename, VRAM gates, auto-hide/downgrade/
fallback, new flags, vendor/arch whitelists, low-VRAM profile system, upscaler-bank
scaffold, new 8GB upscaling, fp8/NVFP4/Q8 forks, HuMo changes, alias deletion. CUT
this sprint: fp8 fallback; the 420-ref rename; the defensive role_slots resolve;
_PUBLIC_LABEL in the executable path; mixed-route presets.

## 8. Incidental existing bugs surfaced (see NEWBUG-video-tier-sweep.md)
(1) wan_ti2v discards its measured peak -> manifest vram_peak_mb always None
(eng_wan_ti2v.py:450-485) -- NOW IN SCOPE (fold the one-line thread in C1). (2)
hf_download_driver verifies neither revision nor checksum -- folded into C0 (2b).
