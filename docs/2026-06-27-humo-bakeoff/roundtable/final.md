# HuMo quality + VRAM-fit -- FINAL build-ready plan (kibitz/roundtable r1->r4 CONVERGED)

Panel arc: r1 kibitz (Codex + Claude), r2 roundtable (GPT-5.5 + Gemini-3.1-pro + Claude),
r3 kibitz (Codex + Claude), r4 kibitz (Codex + Claude). Antigravity stalled every local
round (0-byte sessions; not credits) -- Codex carried the local rounds. Claude grounded
every claim against the real files; r4 verdict = yes-with-fixes (spec-precision only) =
CONVERGED. DIAGNOSTIC harness only; ALL promotion operator-gated. Spend: r2 ~$0.12; r1/r3/r4
local (free).

## Goal
Get the operator-preferred HuMo-14B LOOK to (1) FIT with real headroom and (2) a more
realistic mouth/teeth, on a 16GB RTX 5080, without touching production until the operator
eyeballs clips. The just-run bakeoff showed 14B rides ~15.8-16 GB (NVML) and the two-stage
encoder evict shaved only ~217 MB.

## Build order (GPU legs SERIAL on one :8000 + resident VRAM; CPU/build prep may overlap)
A allocator-honest-meter probe -> [kill-gate] -> B GGUF feasibility -> matrix ;
C no-LoRA/steps mouth-ceiling (independent track) ; D model-swap dep-probe (only if needed).

## Step A -- honest VRAM meter + allocator A/B (cheapest; settles true-vs-cached)
Two-node in-process contract in the sibling `otr_bakeoff_helper` pack (NOT eng_humo):
- a LATENT passthrough RESET node spliced after `WanHuMoImageToVideo`/`OTR_BakeoffReclaim`
  and BEFORE `KSampler`, that calls `torch.cuda.reset_peak_memory_stats()` (lazy torch
  import, always-dirty IS_CHANGED);
- an IMAGE passthrough READ node after `VAEDecode` and before `SaveImage` that logs
  `max_memory_allocated()` + `memory_reserved()`.
Sentinel: HuMo is submitted second, so the reset node naturally runs after the LTX render.
Inject `PYTORCH_CUDA_ALLOC_CONF` (from a new `OTR_BAKEOFF_ALLOC_CONF` knob) in
`run_humo_bakeoff.boot_server`'s env; run leg ii A/B.
- **PROMOTABLE GATE (grounded): all frame-matrix + sentinel cells must have true
  `max_memory_allocated` <= 13500 MB AND NVML peak <= 14500 MB** (the resident ceiling
  `wrapper_bridge.VRAM_CEILING_MB=14500`) under the SAME allocator env -- otherwise a
  production wrapper re-run with that env is required before promotion. KILL-GATE: if both
  hold for the fp8 14B, it is promotable as-is and Step B is unnecessary.

## Step B -- GGUF 14B (only if A insufficient): explicit harness leg schema
Builder emits a harness-only GGUF leg (do NOT just "mirror eng_wan_i2v" -- spec it):
`loader_mode=gguf`, `loader_class=UnetLoaderGGUF`, `loader_param=unet_name`,
`unet=<resolved .gguf basename>`; gguf forced by the leg (not only .gguf inference). Emit
`{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":<gguf>}}` -- NO `gguf_name`, NO
`weight_dtype` (ComfyUI-GGUF contract; eng_wan_i2v.py:215-218). Make the runner
loader-agnostic: add `loader_class`/`loader_param` to `meta` so `assert_checkpoints` +
`build_manifest` stop hardcoding `UNETLoader`. Feasibility gate (in order): live
`/object_info` has `UnetLoaderGGUF`(unet_name) -> a HuMo-14B GGUF resolves via folder_paths
-> 33-frame MIN smoke (`_HUMO_MIN_FRAMES=33`, NOT "1 frame") proves `WanHuMoImageToVideo`
accepts the GGUF model WITH audio conditioning AND that lightx2v `LoraLoaderModelOnly`
merges onto it (else run the GGUF leg LoRA-free at higher steps) -> then the frame matrix.

## Step C -- mouth/teeth ceiling (independent; ENV-driven; operator-rubric gate)
Per-leg ENV (the runner sets these at build time, builds, restores -- NOT literal patches,
because no-LoRA deletes the LoraLoaderModelOnly node + rewires ModelSamplingSD3.model<-unet,
which `_build_graph` already does on skip_lora). PER-TIER ENV TABLE (grounded eng_humo.py):
- 14B (`humo` / `humo_14B_169`): `OTR_HUMO_LORA_NAME`, `OTR_HUMO_STEPS`, `OTR_HUMO_CFG`
  (+ `OTR_HUMO_14B_169_CFG` for the wide tier).
- 1.7B (`humo_1.7B` / `humo_1.7B_169`): `OTR_HUMO_17B_LORA_NAME`, `OTR_HUMO_17B_STEPS`,
  `OTR_HUMO_17B_CFG` (+ `OTR_HUMO_17B_169_CFG` for the wide tier).
Setting the wrong tier's var silently no-ops -> the manifest MUST assert the lora/steps/cfg
+ tier ACTUALLY loaded. Mouth acceptance = fixed plosive/vowel clip(s) + the side-by-side
montage (reuse the built ffmpeg hstack) + an OPERATOR rubric gate; do NOT invent an
automated teeth metric (no libs). A no-LoRA/higher-step result is only a "mouth win" after
that gate.

## Step D -- model-swap dep probe ONLY (no adapter yet)
Windows/Blackwell sm_120/torch-2.10/offline dependency-probe script for the candidate
lip-sync models (LatentSync/MuseTalk mouth-pass; Sonic/Hallo2/EchoMimic swap). The adapter
is a separate project gated on it passing AND on mapping into the in-process ALWAYS-SILENT
wrapper path.

## Frame matrix
frames = [49, 97, 177] (49 current; 97 ~3.9 s; 177 = `_HUMO_MAX_FRAMES`, the production worst
case). Runner generates `(leg x frames)`; thread the count into label / manifest /
frames_prefix / out_clip / gates. ASSERT `result.frame_count == manifest.length` per cell
(today only `frame_count>0`). 177f 14B may hard-OOM -> record as a RESULT ("no fit at max
beat"), confirm teardown leaves :8000 clean for the next cell.

## Promotion (DEFERRED; operator-gated) -- the exact wiring
Production pins `humo_1.7B` (PORTRAIT), NOT `humo_1.7B_169`:
`config/profiles/16gb_full.json` `role_overrides.other_beats_visual="humo_1.7B"` +
`slot_overrides.video_render_engine="humo_1.7B"`. Promote via the mapping authority
`config/profiles/widget_mapping.json` (role_overrides.other_beats_visual ->
`OTR_VideoDirector.other_beats_video_model`), NOT raw node-id patching; node 92
(`OTR_VideoRenderBatch.engine`) is `slot_overrides.video_render_engine` and is IGNORED in
episode mode (renders from the ShotLock ledger, otr_video_render_batch.py:127-134). Don't
move announcer off `ltx_audio_in`. The winner must be re-expressed through the in-process
`wrapper_bridge.run_graph` path (the harness is HTTP /prompt) and land in
`workflows/otr_scifi_16gb_full.json` + the profile in the SAME change; then validator +
JSON round-trip + link/widget audit + suite + Bug Bible + B7, commit v2.0-alpha.

## Control caveat
The bakeoff control was `humo_1.7B_169` (wide), an aspect-matched A/B vs the wide 14B (what
the operator eyeballed). The shipping engine is `humo_1.7B` (portrait). Adding a portrait
`humo_1.7B` control leg is OPTIONAL -- only do it for a literal production-control baseline;
it is apples-to-oranges against the wide 14B, and the operator goal is the 14B look.

## VERIFY-AT-BUILD checklist (Codex r4)
1. `/object_info` confirms `UnetLoaderGGUF` exists, required input `unet_name`, no
   `gguf_name`/`weight_dtype`. 2. GGUF leg manifest proves resolved loader class/param/file
   match the built prompt. 3. GGUF 33f min-smoke: WanHuMo accepts the GGUF model + audio +
   LoRA-merge. 4. Frame matrix [49,97,177] for viability legs; assert frame_count==length.
   5. Step A records allocated/reserved peaks after a same-prompt reset + NVML peak +
   effective PYTORCH_CUDA_ALLOC_CONF. 6. Sentinel resets CUDA peak after LTX, before HuMo.
   7. No-LoRA/steps legs assert actual lora absence + steps + cfg + tier in the manifest.
   8. Mouth review uses fixed plosive/vowel clips + operator side-by-side rubric. 9. Any
   promotion re-expresses through run_graph, edits workflow + profile in one change, then
   validator/round-trip/audit/suite/Bug Bible/B7.

## Invariants (guarded throughout)
Single resident <= 14.5 GB (target <= 13.5 true-allocated); in-process always-silent path;
cold-import clean (probe nodes lazy-import torch); harness diagnostic-only; 100% local;
UTF-8 no BOM; SFW; commit per green chunk to v2.0-alpha; prod/main + tags GATED.

## Convergence
r4 raised only spec-precision (yes-with-fixes), no new architecture -> CONVERGED. Stop at
r4. This plan is build-ready for a coder window. Manifest of raw reviews: kibitz-runs/
2026-06-27-humo-quality/{r1,r3,r4}/ + docs/2026-06-27-humo-bakeoff/roundtable/pass02/.
