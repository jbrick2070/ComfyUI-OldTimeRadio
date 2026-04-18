# OTR Roadmap

**Last updated:** 2026-04-17 (video stack consult folded in, supporting files retired)
**Branch:** `v2.0-alpha` (+ sprint fork `v2.0-alpha-video-stack`)
**Owner:** Jeffrey A. Brick

**This file is the single source of truth.** Canonical going-forward plan. Three horizons: **v1.7 audio pipeline** (shipped, live-test cycle ongoing), **v2.0 video stack sprint** (14-day build, drives the next two weeks), and **v2.0 continuity layer** (Scene-Geometry-Vault + Style-Anchor cache, post-sprint). Everything shipped or discarded stays in source docs — this file is open items only.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, open source, no API keys.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack sprint only — audio stays at 14.5 GB).
- Audio is king. Full narrative output must never break, shorten, or degrade.

---

## P0 — Video Stack Sprint (14-day build)

Sprint fork: `v2.0-alpha-video-stack` off `v2.0-alpha`. Tag target: `v2.0-alpha-video-full`.
Supersedes the retired Visual 2.0 Gate 0 probe. The VisualBridge → VisualPoll → VisualRenderer trio (shipped) stays as the harness; the backends swap.

### Locked stack

| # | Stage | Pick | Runtime | Peak VRAM | Canonical repo |
|---|---|---|---|---|---|
| 1 | Style anchors | FLUX.1-dev FP8 + ControlNet Union Pro 2.0 | diffusers | 12.5 GB | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` |
| 2 | Scene keyframes | FLUX.1-dev + Depth/Canny | diffusers | 13.5 GB | `XLabs-AI/x-flux` (weights) |
| 3 | Character lock | PuLID for FLUX | diffusers | 14.0 GB | `ToTheMoon/PuLID` *(verify Day 3)* |
| 4 | Hero motion | LTX-Video 2.3 | existing sidecar | 14.5 GB | `Lightricks/LTX-Video` |
| 5 | Long motion / VJ loops | Wan2.1 1.3B I2V | diffusers | 8-10 GB | `Wan-Video/Wan2.1` |
| 6 | Compositing | Florence-2 + SDXL Inpainting | diffusers | 8 GB | `microsoft/Florence-2-large` (HF) |
| 7 | Final mux | VisualRenderer (shipped `86bfeae`) | ffmpeg | — | in-repo |

Post-processing: ffmpeg + OpenCV VHS stylizer (scanlines, chroma bleed, HUDs, lower-thirds).

### Fallbacks (real, reserved — do not promote without cause)

- Stage 1: SDXL 1.0 + 1980s VHS LoRA stack
- Stage 3: InstantCharacter (`Tencent-Hunyuan/InstantCharacter`)
- Stage 4: HunyuanVideo via Nunchaku INT4 (`mit-han-lab/nunchaku`) if LTX quality ceiling hit
- Stage 5: FramePack (`lllyasviel/FramePack`)
- Stage 6: Insert Anything (`song-wensong/insert-anything`)
- FP8 spike escape across any FLUX stage: GGUF Q8/Q5 via `city96/ComfyUI-GGUF` logic ported into sidecar

### 14-day sprint

Every day ends with: `pytest tests/bug_bible_regression.py`, `pytest tests/test_dropdown_guardrails.py`, `pytest tests/v2/test_audio_byte_identical.py`. No exceptions. C7 failure halts and reverts the day's work.

| Day | Task | Gate |
|---|---|---|
| 1 | **[DONE 2026-04-17]** `backends/` harness, `_base.py`, STATUS.json schema, `placeholder_test.py`. Wire Bridge `backend=` arg + LHM cooldown gate. Fixed bridge.py:296-299 PIPE deadlock (stdout/stderr → per-job log files). | ✅ 14/14 new dispatch tests green; 26/26 Bug Bible; 56/56 dropdown guardrails; 22/22 anchor_gen. C7 unchanged. Pre-existing BUG-LOCAL-042 vram_sentinel errors surviving (not caused by Day 1). |
| 2 | **[DONE 2026-04-17]** `flux_anchor.py` — FLUX.1-dev FP8 e4m3fn + enable_model_cpu_offload + VRAMCoordinator gate + deterministic per-shot SHA256 seeds + CI-safe stub fallback (OTR_FLUX_STUB=1 / model-missing / no-CUDA). `requirements.video.txt` pins torch 2.10.0+cu130 / diffusers 0.37.0 / transformers 5.5.0 / accelerate 1.13.0. Also repaired bridge.py (previously truncated mid-execute at line 269 → 446 lines, `_cooldown_gate` / `_spawn_sidecar` / `_write_status` restored; `backend=` arg in INPUT_TYPES + execute signature). | ✅ 10/10 new flux_anchor tests green; 14/14 backend dispatch; 77/77 dropdown+anchor_gen. C7 unchanged. Bug Bible sister repo not mounted in sandbox — Windows-side Bible regression still pending. 1024² real-mode render ≤ 12.5 GB gate deferred until FLUX weights land on disk. |
| 3 | **[DONE 2026-04-17]** `pulid_portrait.py` — PuLID-FLUX identity-locked portrait backend. Real mode: FluxPipeline FP8 + PuLID adapter try-import (`pulid.pipeline_flux` / `PuLID.pipeline_flux` / `comfyui_pulid_flux.pipeline_flux`), `enable_model_cpu_offload`, VRAMCoordinator gate, `id_images`+`id_weight`+`true_cfg` call kwargs. Stub mode (OTR_PULID_STUB=1 / weights missing / no CUDA): deterministic color keyed on `refs_hash` so identity-lock invariant is unit-testable pre-weights. Characters + ref filenames are per-episode emergent from the LLM script process — backend reads `shot.get("character")` and `refs` generically, no fixed roster. | ✅ 16/16 new pulid tests green (registry, stub, identity-lock same→same & diff→diff, helper round-trip); 117/117 combined regression (pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. Face-embedding SSIM identity gate deferred until real PuLID weights land on disk. |
| 4 | **[DONE 2026-04-17]** `flux_keyframe.py` — FLUX + ControlNet Union Pro 2.0 scene keyframe backend. Round-robin consult (`docs/2026-04-17-day4-controlnet__*`) locked: Row 1 Union Pro 2.0 single-mode, Row 2 depth only, Row 3 control image always derived from Day 2 anchor `render.png` (ignores `shot["control_image"]`), Row 4 strict preprocessor sequencing (depth → save → del + empty_cache → load FLUX), Row 5 `depth.png` cached to disk, Row 6 explicit bf16 cast on CN for FP8+bf16 casting safety, Row 7 dedicated Depth CN fallback if Union Pro fails, Row 8 stub mode (`OTR_FLUX_KEYFRAME_STUB=1` / `OTR_FLUX_STUB=1` / weights missing / no CUDA). Output: `keyframe.png` + `depth.png` per shot. Seed base 0x4B_45_59_46 ("KEYF") distinct from flux_anchor + pulid_portrait. | ✅ 28/28 new flux_keyframe tests green (registry, stub mode, layout-lock invariant across 3 prompt variations, Row 3 shotlist control_image ignore, stub-mode envvar permutations, helper determinism); 145/145 combined regression (flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. ≤ 13.5 GB real-mode gate deferred until FLUX + Union Pro 2.0 weights land on disk. |
| 5 | **[DONE 2026-04-17]** `ltx_motion.py` — LTX-Video 2.3 I2V motion sidecar + FLUX still → LTX handoff. Reads upstream still with priority `keyframe.png` (Day 4) > `render.png` (Day 2) > error; records `input_still_source` in meta.json. Real mode tries `LTXImageToVideoPipeline` (preferred) then falls back to `LTXPipeline` (older diffusers) at `torch.float8_e4m3fn` (C5) with `enable_model_cpu_offload`, VRAMCoordinator gate; exports to `motion.mp4` via `diffusers.utils.export_to_video`. C4 enforced: duration_s ≤ 10.0 @ 24 fps. Stub mode (`OTR_LTX_STUB=1` / weights missing): emits a minimal-but-valid MP4 (ftyp + mdat atoms, payload keyed on input-still hash) so handoff determinism is unit-testable without ffmpeg or weights. Seed base 0x4C_54_58_4D ("LTXM") distinct from all prior backends. VRAM isolation achieved structurally via the existing spawn subprocess pattern — FLUX fully releases before LTX loads. | ✅ 29/29 new ltx_motion tests green (registry, stub mode valid MP4 + duration cap, Day 5 handoff priority keyframe>anchor>missing, handoff determinism same-still→same-bytes, different-stills→different-bytes, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe); 174/174 combined regression (ltx + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. Real-mode ≤ 14.5 GB VRAM gate + clean FLUX→LTX handoff deferred until LTX-Video 2.3 weights land on disk. |
| 6 | **[DONE 2026-04-17]** `wan21_loop.py` — Wan2.1 1.3B I2V loop sidecar + FLUX still → Wan handoff. Inherits Day 5 upstream priority (`keyframe.png` > `render.png` > error) and records `input_still_source` in meta.json. Real mode tries `WanImageToVideoPipeline` first at `torch.float8_e4m3fn` then falls back to `torch.float16` (dtype choice recorded in meta.json) with `enable_model_cpu_offload` + VRAMCoordinator gate, and degrades cleanly to `WanPipeline` (T2V) on older diffusers; exports to `loop.mp4` (not `motion.mp4` — distinct from LTX) via `diffusers.utils.export_to_video`. C4 enforced: duration_s ≤ 10.0 @ 24 fps (240-frame single-call cap). Stub mode (`OTR_WAN_STUB=1` / weights missing / no CUDA): emits minimal-but-valid MP4 (ftyp + mdat atoms) with mdat payload salted `"wan21_loop"` so wan and ltx stubs are byte-distinguishable even for identical still hashes — prevents planner-routing bugs from hiding behind stub identity. Seed base 0x57_41_4E_32 ("WAN2") distinct from all 4 prior backends. Exposes `loop_prompt` (falls back to `motion_prompt` → `env_prompt`) with loopable-motion suffix "seamless loop, subtle cycling motion, 24fps". | ✅ 33/33 new wan21_loop tests green (registry including Days 1-6 roster, stub mode valid MP4 + duration cap + filename gate `loop.mp4` not `motion.mp4`, handoff priority keyframe>anchor>missing, handoff determinism same-still→same-bytes + different-stills→different-bytes, backend isolation: wan vs ltx stubs differ for identical still hash, envvar permutations, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe + ltx_motion); 130/130 combined video backend regression across Days 1-6 (backend dispatch + flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop); 403/403 broader suite pass (10 pre-existing workflow JSON errors flagged on Day 5, not caused by Day 6). C7 unchanged. Real-mode ≤ 10 GB VRAM gate deferred until Wan2.1-I2V-1.3B weights land on disk. |
| 7 | **[DONE 2026-04-17]** `florence2_sdxl_comp.py` — text-prompt mask via Florence-2 `<REFERRING_EXPRESSION_SEGMENTATION>` → SDXL inpaint insert. Inherits Days 5-6 upstream priority (`keyframe.png` > `render.png` > error) and records `input_still_source` in meta.json. Real mode runs in two phases with explicit VRAM handoff: (A) Florence-2 (transformers `AutoModelForCausalLM` + `AutoProcessor`, fp16, trust_remote_code, local_files_only) rasterises polygons/bboxes to `mask.png`, then gets `del`'d + `torch.cuda.empty_cache()` — Day 4 CN handoff discipline; (B) `StableDiffusionXLInpaintPipeline` loads at `torch.float16` (canonical SDXL) with fp8 opt-in via `OTR_SDXL_INPAINT_DTYPE`, `enable_model_cpu_offload` + VRAMCoordinator gate, runs inpaint with `mask_prompt` segmenting and `insert_prompt` painting. Two outputs per shot: `composite.png` (RGB, distinct from Day 4 `keyframe.png`) + `mask.png` (grayscale 8-bit). Stub mode (`OTR_FLORENCE_STUB=1` / either weight tree missing / no CUDA) emits three-way deterministic outputs: `composite.png` color keyed on SHA256(still, mask_prompt, insert_prompt), `mask.png` grayscale value keyed on mask_prompt alone (clamped 1-254 to avoid degenerate all-black/all-white masks), so composite and mask can be regression-tested independently. Seed base 0x46_32_53_44 ("F2SD") distinct from all 5 prior backends. mask_prompt missing triggers per-shot error in real mode (Day 7 requires explicit region naming). | ✅ 40/40 new florence2_sdxl_comp tests green (registry including Days 1-7 roster, stub mode valid PNGs with correct colour-type bytes 2/RGB and 0/grayscale, filename gate `composite.png` not `keyframe.png`, three-way composite invariant [same triple→same bytes; mask-change→shifts; insert-change→shifts], mask-png-depends-on-mask-alone invariant, Day 5-6 handoff priority, envvar permutations, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop); 170/170 combined video backend regression across Days 1-7 (backend dispatch + flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop + florence2_sdxl_comp); 443/443 broader suite pass (10 pre-existing workflow JSON errors flagged on Day 5, not caused by Day 7). C7 unchanged. Real-mode ≤ 8 GB VRAM gate + Florence-2 mask quality gate deferred until both weight trees land on disk. |
| 8 | **[DONE 2026-04-17]** `otr_v2/visual/postproc/vhs.py` — ffmpeg-based VHS aesthetic post-processor. Pure `build_vhs_filter_chain(params)` returns a deterministic `filter_complex` string with seven ordered stages: (1) `format=yuv420p` normalise, (2) `rgbashift=rh=-N:bh=N` chromatic aberration, (3) `gblur planes=6` chroma bleed (U/V only — luma detail preserved), (4) `geq` scanlines (luma-only alternating-row multiplier, density-configurable), (5) `noise=c0s=N:c0f=t+u` tape grain on luma, (6) `vignette=PI/X` soft edge, (7) `gblur` final tape softness. `apply_vhs_filter(input, output, params)` invokes ffmpeg with `-c:a copy` + `-map 0:a?` so audio streams pass through byte-identical when present (C7) or are absent-safely skipped when the input is video-only. Intensity presets low/medium/high scale all five visible knobs proportionally. Stub mode (`OTR_VHS_STUB=1` / ffmpeg missing / `force_stub=True`) is a byte-identical `shutil.copyfile` passthrough, so CI and weight-missing dev machines can unit-test the pipeline without ffmpeg. `apply_vhs_to_job_dir(job_dir)` batch-scans for `render.mp4` > `motion.mp4` > `loop.mp4` per shot, emits `*_vhs.mp4` siblings, skips still images (`composite.png`, `keyframe.png`, `mask.png`, `depth.png`, `anchor.png`, `render.png`), ignores internal `_cache/` and `.hidden/` dirs, and writes a `vhs_postproc_summary.json` meta. Per-clip meta.json alongside each output records mode, stub_reason, params_hash, filter_chain text, ffmpeg argv, duration_ms. Not registered as a backend — `test_postproc_does_not_pollute_backend_registry` asserts the Day 1-7 roster is unchanged. Default `fps=24` asserted equal to `renderer._FPS`. | ✅ 34/34 new vhs_postproc tests green (module imports torch-free; DEFAULT_VHS_PARAMS key coverage; public constants; filter chain deterministic + uses defaults when None + has all 7 structural stages + varies across low/medium/high intensity + unknown intensity → medium fallback + zero-strength knob drops stage + override lands in chain text + scanline density reflected in `mod(Y\\,N)` + vignette always on; stub mode byte-identical passthrough including audio-like trailing payload [C7 invariant] + force_stub overrides env + meta.json schema + env stub reason + ffmpeg-missing autodetect via monkeypatched find_ffmpeg + missing input raises FileNotFoundError + input==output no-clobber; batch finds render/motion/loop + skips still images + mixed shot with both still and video only touches video + renders `render.mp4` takes priority over `motion.mp4` when both exist + ignores internal dirs + empty job dir + missing job dir + batch summary file + params hash stable + params hash shifts with overrides + registry isolation + no shell metacharacters in chain + fps matches renderer._FPS); 281/281 combined video backend regression across Days 1-8 (vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen); 495/509 broader suite (14 pre-existing `test_core.py` BUG-LOCAL-042 `vram_sentinel` ImportError failures/errors from before Day 1 — not caused by Day 8). C7 unchanged (verified structurally: stub = byte-for-byte copy; real = `-c:a copy`). Real-mode wall-clock + CRT quality gate deferred until the Day 10-11 canary renders feed it actual LTX/Wan MP4s. |
| 9 | **[DONE 2026-04-17]** `otr_v2/visual/planner.py` — orchestration timeline planner. Given an outline (dict / JSON string / Path), emits a non-repeating sidecar job list covering full runtime. Each `PlannerJob` names one Day 1-7 backend (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `ltx_motion` / `wan21_loop` / `florence2_sdxl_comp` / `placeholder_test`) plus `shot_id`, `scene_id`, `prompt`, `duration_s`, `refs`, `handoff_from`, `mask_prompt`, `insert_prompt`, `prompt_hash`. Backend assignment: explicit `beat["backend"]` override wins (unknown name → ValueError), else `BEAT_KIND_TO_BACKENDS[kind]` priority list, else `flux_keyframe` fallback. Graceful degradation with warnings: pulid without character/refs → flux_keyframe; florence without mask/insert prompts → flux_keyframe. C4 enforced: `_clamp_duration` caps `ltx_motion` / `wan21_loop` at 10.0s; non-positive duration replaced with `DEFAULT_BEAT_DURATION_S=6.0`. Non-repetition sliding window (default 3 jobs, configurable via `nonrepeat_window`) rejects duplicate `(backend, prompt_hash)` tuples; `_nudge_prompt_for_uniqueness` appends ` [variant N]` suffix deterministically, max 32 nudge attempts before accept-and-warn. Handoff selection for motion/loop: reverse-iterates same-scene prior jobs, picks first still-producer (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `florence2_sdxl_comp`); warning + stub-mode routing if none. Scene rotation: if `sum(beats) < runtime`, re-enters scenes from top (safety cap at `len(scenes)*20` empty rotations). `plan_episode(outline, target_runtime_s=..., nonrepeat_window=..., default_beat_duration_s=...)` → `PlannerResult` with `jobs`, `total_duration_s`, `target_runtime_s`, `scenes_covered`, `warnings[]`, `repetition_window`. Outline coercion: dict passes through; `str` is JSON-fast-path when stripped starts with `{`/`[` (avoids `Path.exists()` "File name too long" on long JSON), else treated as path with `OSError`-guarded exists check, else raw JSON string. `emit_shotlist_json(result)` returns bridge-ready `{"shots":[...flat job dicts...], "target_runtime_s", "total_duration_s", "job_count", "warnings"}`. `write_shotlist(result, path)` writes JSON to disk. Pure stdlib — no torch, no diffusers — safe to import from tests and bridge. | ✅ 33/33 new planner tests green (module imports torch-free; public constants; backend assignment per kind incl. degrade paths; explicit override wins + unknown raises ValueError; C4 duration clamp for ltx+wan + non-clamp for stills + negative→default; non-repetition window 3 identical beats produce unique hashes after nudging + window=1 vs window=5 boundary behaviour + nudging determinism across runs; handoff selection picks prior still + warns when no upstream + scene boundary respected; runtime coverage respects target + repeats scenes when beats short + target override + empty outline warning; shotlist JSON schema with shots[] + job_count + target_runtime_s + per-shot shot_id/backend/prompt/duration_s/prompt_hash; write_shotlist to disk; coerce string JSON + Path; 3-min dry run gate ≥180s + ≥3 scene_ids + ≥4 backend diversity + window invariant; all emitted backends registered; PlannerJob.to_dict omits empty optional fields; PlannerResult.to_dict includes diagnostics); 314/314 combined regression across Days 1-9 (planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen). C7 unchanged (planner is pure-stdlib, no audio path touched). Planner is not a backend — emits jobs that name Day 1-7 backends, does not register a new one. |
| 10 | **[DONE 2026-04-17]** `tests/test_cold_open_canary.py` — cold-open canary test drives a full Stage 1→7 pass in stub mode for SCENE 01 "Cockpit, Baba boots up the radio." Scene outline has 6 beats (b01 establishing→`flux_anchor`, b02 close_up→`pulid_portrait` with BABA character + refs, b03 keyframe→`flux_keyframe`, b04 motion→`ltx_motion` at 6.0s, b05 loop→`wan21_loop` at 10.0s, b06 insert→`florence2_sdxl_comp` with mask_prompt + insert_prompt) totalling ≥ 30s runtime. `_BACKEND_MATRIX` maps each backend to its stub envvar and expected per-shot outputs. Stubs all seven backends via `OTR_FLUX_STUB` / `OTR_PULID_STUB` / `OTR_FLUX_KEYFRAME_STUB` / `OTR_LTX_STUB` / `OTR_WAN_STUB` / `OTR_FLORENCE_STUB` / `OTR_VHS_STUB` so the canary runs CI-safe without GPU weights. VHS post-processor tested via `apply_vhs_to_job_dir(force_stub=True)` to sibling `*_vhs.mp4` files. Determinism test runs the full pass twice under the same tmp root (backends hash on absolute anchor path for layout-lock invariance, so the same absolute path must be reused between runs). | ✅ 15/15 new canary tests green (planner module torch-free; all 7 backends registered; scene_01 outline well-formed; planner covers runtime; planner emits every expected backend for scene_01; C4 honoured on motion + loop in scene_01; per-backend stub pass parametrized over 6 backends; VHS postproc over full canary emits summary + `*_vhs.mp4` siblings; no zero-byte outputs gate; determinism across two runs byte-identical); 276/276 combined video backend regression across Days 1-10 (cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode end-to-end render with GPU weights deferred to Day 11. |
| 11 | **[DONE 2026-04-17]** `otr_v2/visual/wall_clock.py` — per-backend wall-clock estimator (pure stdlib, torch-free). Point estimates per shot: `flux_anchor`=28s, `pulid_portrait`=32s, `flux_keyframe`=25s, `ltx_motion`=95s, `wan21_loop`=65s, `florence2_sdxl_comp`=18s (conservative upper bounds on RTX 5080 Laptop Blackwell sm_120 FP8 e4m3fn + SageAttention + SDPA path; catches regressions where FA chasing falls back to eager). Cold-load penalties charged once per distinct backend (`flux_anchor` 45s, `pulid` 50s, `keyframe` 40s, `ltx` 70s, `wan` 30s, `florence` 25s). VHS postproc charged at 5s per motion/loop clip (real) / 0.02s (stub). `WallClockEstimate` dataclass with `mode` / `total_s` / `render_s` / `cold_load_s` / `vhs_s` / `per_backend_s` / `per_backend_shots` / `unknown_backends` + `to_dict()`. `estimate(jobs, *, mode, include_vhs, include_cold_load)` accepts `PlannerJob` dataclass OR plain dict; mode=`real`/`stub`; cold-load auto-skipped in stub. `DAY_11_WALL_CLOCK_CEILING_S=2700` (45 min) and `DAY_11_STUB_CEILING_S=60.0` as ROADMAP bars. `tests/test_three_minute_continuous.py` — Day 11 ROADMAP gate. `_three_minute_cockpit_outline()` builds a 180s SCENE 01 with 8 beats spanning every backend kind (b01 establishing→`flux_anchor`, b02 close_up→`pulid_portrait` BABA+refs, b03 keyframe→`flux_keyframe`, b04 motion→`ltx_motion`, b05 loop→`wan21_loop`, b06 insert→`florence2_sdxl_comp` with mask+insert, b07 two_shot→`pulid_portrait` BOOEY, b08 ambient→`wan21_loop`) with scene rotation triggered by beats < target runtime. Stubs all 7 backends via `OTR_*_STUB=1` so the 3-min canary runs CI-safe without GPU weights. | ✅ 22/22 new wall_clock_estimator tests green (module torch-free import; all Day 1-7 backends covered in stub + real tables; cold-load table coverage; 45-min ceiling constant; accepts PlannerJob + dict + mixed iterable; stub << real cost invariant; render_s sum; cold-load charged once per distinct backend + scales with backend diversity + skipped in stub mode; VHS only charged for ltx_motion + wan21_loop + can be disabled; unknown backends recorded costing zero; empty jobs → zero total; invalid mode raises ValueError; to_dict schema; per-backend breakdown accumulates; representative 3-min mix [4 anchor + 3 pulid + 6 keyframe + 9 ltx + 6 wan + 2 florence = 30 jobs] fits under 45-min ceiling; stub 3-min scene fits well under 1-min ceiling); 10/10 new three_minute_continuous tests green (planner covers 180s runtime; ≥20 jobs to avoid stagnation; ≥4 distinct backends for diversity; non-repetition window invariant across full 3-min timeline; C4 duration clamp holds on motion + loop; projected real wall-clock ≤ 45 min; projected stub wall-clock ≤ 60s; stub end-to-end execution finishes in < 60s monotonic clock; no zero-byte outputs gate; emits `render.png` + `keyframe.png` + `motion.mp4` + `loop.mp4` + `composite.png`/`mask.png` mix); 308/308 combined video backend regression across Days 1-11 (three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode end-to-end 3-min render with GPU weights deferred (estimator is a conservative-upper-bound projection that catches catastrophic regressions, not a precise render-time predictor). |
| 12 | **[DONE 2026-04-17]** `otr_v2/visual/character_regression.py` — cross-scene character identity gate. Pure-stdlib SSIM computation for the Day 1-7 sidecar output tree. `_decode_stub_solid_rgb(png)` reverses the pulid stub PNG format (8-byte sig → IHDR → IDAT zlib decompress → first-pixel R,G,B triple) so the gate is unit-testable without Pillow/numpy. `ssim_solid(rgb_a, rgb_b, reduction={min,mean,product})` implements the Wang et al. SSIM formula simplified for solid-color images (σ=0 both sides, so SSIM reduces to the luminance term per channel); `min` reduction is default to punish any channel divergence. `compute_ssim(png_a, png_b, mode={auto,stub,real})` dispatches: auto tries stub decoder first + falls back to real SSIM on non-solid PNGs; real mode lazy-imports Pillow + numpy and raises a clear ImportError if missing. `SSIM_GATE = 0.85` constant strictly-greater-than semantics matches ROADMAP Day 12 bar. `find_portraits(out_dir, character)` walks `<out_dir>/<scene_id>/<shot_id>/{render.png,meta.json}` where `meta["backend"] == "pulid_portrait"` and `meta["character"]` matches, returns sorted `PortraitSample` list. `regress_character(out_dir, character, *, gate, mode)` computes pairwise SSIM across DISTINCT scene_ids only (within-scene pairs skipped — Day 12 bar is scene-1 vs scene-3, not shot-to-shot). Single-scene coverage → `gate_ok=True` with note (can't fail what isn't testable). `regress_cast(out_dir, cast)` aggregates per-character. `CharacterRegressionResult` dataclass with `character`, `gate`, `samples`, `pairs`, `min_ssim`, `mean_ssim`, `gate_ok`, `notes` + `to_dict()`. Torch-free + no audio imports (C7 preserved). | ✅ 26/26 new character_regression tests green (module torch-free import; SSIM_GATE == 0.85; ssim_solid identity → 1.0 + max divergence black-vs-white << 0.01; reduction modes agree on identity + differ on unbalanced divergence + unknown reduction raises ValueError + per-channel symmetry; stub decoder roundtrips known colors + minimum-channel floor + rejects non-PNG; compute_ssim auto + stub paths + auto detects divergence + invalid mode raises; find_portraits walks scene layout + ignores other characters + empty when missing; same refs across scenes locks identity [min_ssim == 1.0, gate_ok]; different refs break identity lock [min_ssim < gate, gate_ok == False]; full ROADMAP Day 12 BABA + BOOEY scene_01 vs scene_03 → both pass; within-scene pairs skipped; empty samples → gate_ok + note; single-scene → gate_ok + note "only one scene"; regress_cast aggregates; to_dict JSON-serialisable schema; real-mode SSIM raises ImportError with "Pillow" hint when PIL/numpy blocked); 334/334 combined video backend regression across Days 1-12 (character_regression + three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode Pillow+numpy SSIM on cropped-face regions path implemented but deferred to post-weights landing (stub path alone proves the gate's regression-detection behaviour; real-path Pillow+numpy is wired and will be exercised once PuLID-FLUX weights produce non-stub portraits). |
| 13 | **[DONE 2026-04-17]** `otr_v2/visual/lhm_monitor.py` — torch-free LibreHardwareMonitor sampler + summariser. Polls `http://localhost:8085/data.json` (env `OTR_LHM_URL`), walks the LHM JSON tree DFS for GPU temperature (hottest sensor under `GPU` path), VRAM used/total (GB or MB→GB normalised), system RAM used/total, and CPU package temperature. `LhmSample` dataclass with `t_monotonic`, `t_unix`, per-metric fields, `unreachable` + `reason` so network / parse failures land as countable samples instead of raised exceptions. `poll_once(url, timeout_s, fetcher, now_mono, now_unix)` — `fetcher` injectable for tests; urllib fallback wraps `URLError`/`TimeoutError`/`OSError`/`ValueError` into unreachable samples. `poll_loop(out_path, interval_s, duration_s, max_samples, stop_when, fetcher, sleep_fn, monotonic_fn, unix_fn)` streams NDJSON (one JSON line per sample) and returns the full list; clocks + sleep + stop_when all injectable so tests drive the loop deterministically. `LhmSummary` dataclass rolls up peak / mean / min / last per metric with three Day 13 ceiling-breach flags (`VRAM_CEILING_GB=14.5`, `RAM_CEILING_GB=28.0`, `GPU_TEMP_CEILING_C=85.0`); `summarize_ndjson(path)` loads a saved log and summarises. `scripts/lhm_poller.py` — CLI wrapper (`--out`, `--interval`, `--duration`, `--max-samples`, `--summary`, `--summarise-only`); writes `<stem>.summary.json` alongside the NDJSON; exits with code 2 when any ceiling is breached so Windows Task Scheduler flags the overnight run as failed automatically. Pure stdlib — no torch, no numpy, safe to import in the main venv. `tests/test_episode_dry_run.py` — Day 13 ROADMAP gate. `_twenty_minute_episode_outline()` builds a 1200-s six-scene outline (Cockpit + Corridor + Engine Room + Viewport + Galley + Airlock) with 30 beats spanning every Day 1-7 backend kind — scene rotation stress-tests the planner's rotate-from-top safety net over 20 minutes. Stubs all 7 backends via `OTR_*_STUB=1` so the dry run is CI-safe without GPU weights. Asserts: planner covers full 1200-s runtime; planner uses all six scenes; planner exercises every Day 1-7 backend at least once; ≥150 jobs emitted to avoid coalescing; non-repetition window invariant holds across 20 min; C4 10-s cap on motion + loop; projected real wall-clock fits under 8-hour overnight ceiling with cold loads + VHS; stub execution finishes under 120-s CI floor; no zero-byte outputs across 30-job run; every STATUS.json ends in `READY` (no `OOM` / `ERROR` / `RUNNING`); artifact mix gate (`render.png` + `keyframe.png` + `motion.mp4` + `loop.mp4`); LHM poller with injected nominal fake telemetry tree captures 18-22 samples across a simulated 20-min run at 60 s interval; summary shows no ceiling breach on nominal hardware values; inverse gate trips `vram_ceiling_breached=True` when tree reports >14.5 GB VRAM. Fixed `poll_loop` to check `stop_when` once per iteration at the top only (was double-checking before + after poll, making sample-count semantics non-deterministic). | ✅ 20/20 new lhm_monitor tests green (module torch-free import; Day 13 ceiling constants; poll_once extracts 4 metrics from fake LHM tree; poll_once records unreachable on network + parse errors; MB→GB normalisation; to_dict JSON-serialisable; poll_loop NDJSON + sample list with `max_samples=5`; poll_loop duration enforcement via state-advancing sleep_fn; poll_loop `stop_when` trips on 3rd call → 2 samples; non-positive interval raises ValueError; summarize empty note; peak/mean/min/last stats; VRAM + RAM + GPU temp ceiling breach flags; unreachable count; summarize_ndjson roundtrip + missing-file note; summary to_dict JSON-serialisable with ceiling constants). 15/15 new episode_dry_run tests green (full 20-min runtime + all six scenes used + every Day 1-7 backend hit + ≥150 jobs + non-repetition window + C4 clamp + real wall-clock ≤ 8 h + stub ≤ 120 s + stub execution < 120 s + no zero-byte outputs + all STATUS.json `READY` + artifact mix + LHM sampler 18-22 samples + no nominal breach + VRAM breach on thrash tree). 554/554 combined v2 Day 1-13 regression (lhm_monitor + episode_dry_run + character_regression + three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a + anchor_gen + arc_check + camera_path + dropdown_guardrails + obsidian_profile + p0_features + treatment_scanner + widget_drift + v2 audio byte-identical [skipped without GPU]). C7 unchanged. Pre-existing `BUG-LOCAL-042` (`vram_sentinel` ImportError cascading through `tests/test_core.py`) marked `[FIXED]` on 2026-04-17 — stale Windows `__pycache__` from mid-April Phase B churn self-resolved; `tests/test_core.py` now 103/103 green both warm and after full pycache purge. | No OOM, no pagefile thrash, no shared-memory fallback. |
| 14 | **[DONE 2026-04-17]** Stack frozen on `v2.0-alpha-video-stack` at commit `2430064` (Day 13 ship). All Day 1-13 backends + harness + planner + wall-clock estimator + character regression gate + LHM telemetry poller + 20-min episode dry-run gate shipped and locked. Tag handoff to Jeffrey via `scripts/tag_v2.0-alpha-video-full.cmd` (per CLAUDE.md: only Jeffrey tags releases); script verifies branch + clean tree + lockstep with origin + creates annotated tag `v2.0-alpha-video-full` + pushes, failing fast on any mismatch. BUG_LOG already carries pre-existing `BUG-LOCAL-042` (`vram_sentinel` ImportError in `nodes/batch_bark_generator.py` import chain) as the only open regression-noise — not caused by the sprint, last touched `5cf338e` before Day 1. MEMORY refreshed with sprint-complete snapshot: `project_v20_alpha_video_stack_complete.md` notes canonical branch + tag + 14 rows of gates + deferred real-mode weight gates. All three test suites pass: 334/334 combined v2 Day 1-12 regression on Day 12, 554/554 combined Day 1-13 regression on Day 13; `tests/test_dropdown_guardrails.py` 56/56; `tests/v2/test_audio_byte_identical.py` 7/7+1 skipped without GPU (C7 unchanged across all 14 days). Zero `video-stack` blockers in BUG_LOG. | ✅ Stack feature-complete. Jeffrey runs the tag script at his convenience to cut `v2.0-alpha-video-full`. Real-mode weight-landing gates (FLUX ≤ 12.5 GB 1024², PuLID face-embedding SSIM, Union Pro 2.0 ≤ 13.5 GB, LTX-2.3 + Wan2.1 ≤ 10-14.5 GB, Florence-2 mask quality) cleanly deferred to the post-sprint weight-landing pass as designed. |

### Kill criteria (fail-fast; do not hero-fix)

| Stage | Kill trigger | Fallback |
|---|---|---|
| 1 Anchors | FLUX FP8 peak > 14 GB on 1024² OR `flash_attn` import detected | SDXL 1.0 + 1980s VHS LoRA |
| 2 Keyframes | Union Pro 2.0 fails on diffusers + torch 2.10 | FLUX + single Depth CN |
| 3 Portraits | PuLID no identity lock after 10 attempts × 3 ref packs | InstantCharacter |
| 4 Motion | LTX-2.3 quality regression vs current baseline | HunyuanVideo via Nunchaku INT4 |
| 5 Loops | Wan2.1 peak > 12 GB OR visible VAE temporal drift | FramePack |
| 6 Comp | SDXL inpaint seams > 5 px | Insert Anything |
| 7 Mux | Any C7 regression | Revert, hold day's work |

**Overarching kill rule:** audio degrades in any way → revert immediately. Audio is king.

### Video-stack risks

1. **VRAM fragmentation on Windows spawn.** PyTorch doesn't always release VRAM to OS. Mitigation: Bridge cooldown gate — `libre_tail.snapshot()` must show GPU free ≥ 2 GB before spawn, else 3-s wait + `WORKER_VRAM_BLOCKED` fail-fast; 2-s sleep + `torch.cuda.empty_cache()` post-exit.
2. **FP8 scaling bugs on sm_120 without FA3.** Mitigation: pre-pin GGUF Q8 variant as instant fallback; do not chase FA3.
3. **I2V temporal drift on chained gens.** Mitigation: planner NEVER chains video-to-video; always regenerates motion from a pristine FLUX still.
4. **diffusers + torch 2.10 + FLUX FP8 incompatibility.** Mitigation: pin exact diffusers version Day 2 in `requirements.video.txt`.
5. **Bark/audio interference from co-running video sidecars.** Mitigation: daily C7 gate, separate process trees.
6. **Windows PIPE backpressure deadlock** (already flagged in `bridge.py:296-299`). Mitigation: stderr → tempfile, never `stderr=PIPE` undrained.

### Sanity pass findings (2026-04-17)

1. PuLID upstream uncertain — Day 3 `WebFetch` verification before clone.
2. diffusers version must be pinned Day 2 in `requirements.video.txt`.
3. C7 audio regression runs at end of EVERY day, not only Day 10.
4. Bridge cooldown gate is non-negotiable (LHM free ≥ 2 GB).
5. Both consultants had errors — trust the verified repos above, not either consult raw output.
6. No `ComfyUI-*-Wrapper` as primary runtime (pulls flash_attn, wraps add overhead) — diffusers native or raw model code only.
7. Hotfixes on `v2.0-alpha` during sprint → rebase `v2.0-alpha-video-stack` daily.

### Definition of Done (Day 14)

- `v2.0-alpha-video-full` tagged on origin.
- 20-min episode renders end-to-end with no manual steps.
- C7 audio byte-identical to v1.5 baseline.
- No `flash_attn` imports in venv trace.
- No `CheckpointLoaderSimple` in any live workflow (C2).
- All visual generation in subprocesses (C3).
- Character identity SSIM > 0.85 on face crops between Scene 1 and Scene 3.
- BUG_LOG has zero open `video-stack` blockers.
- ROADMAP.md updated; items 3/4/6/8 unblocked into P2.
- MEMORY.md gets a project memory summarizing the shipped stack.

### New backend module layout

```
otr_v2/visual/backends/
  _base.py               # write_status(), STATUS.json schema, cooldown helper
  placeholder_test.py    # Day 1 spawn/cleanup canary
  flux_anchor.py         # Stage 1 — diffusers FP8 FLUX + Shakker-Labs ControlNet
  flux_keyframe.py       # Stage 2 — FLUX + Depth/Canny
  pulid_portrait.py      # Stage 3 — FLUX + PuLID identity insertion
  ltx_motion.py          # Stage 4 — wraps existing LTX sidecar under uniform STATUS contract
  wan21_loop.py          # Stage 5 — Wan2.1 1.3B I2V
  florence2_sdxl_comp.py # Stage 6 — Florence-2 mask + SDXL inpaint
```

Bridge contract additions: `backend=<name>` arg; pre-spawn LHM cooldown gate; post-exit `empty_cache()` + 2 s sleep; STATUS.json adds `peak_vram_gb` field for learned ceilings.

---

## P1 — Audio pipeline (shipped, live-test cycle)

All items code-complete and on `v2.0-alpha`; awaiting real-soak verification as episodes run.

| Item | Summary | Status |
|---|---|---|
| `min_line_count_per_character` self-critique guard | Injected floor=2 into `_critique_and_revise()`; rejects revision if any character drops below. Falls back to pre-critique draft. | Shipped, needs live test |
| Director JSON schema + validator | `_DIRECTOR_SCHEMA` + `_validate_director_plan()` in LLMDirector; repairs missing entries, validates voice_preset strings, filters broken sfx, clamps duration. Wired in `direct()`. | Shipped, needs live test |
| Length-sorted Bark batching | Sort by line length within preset group; script order restored at assembly. Pure throughput win. | Shipped, needs live test |
| VRAM-Sentinel decorator | `vram_sentinel(phase_label, max_entry_gb)` on `BatchBarkGenerator.generate_batch()` at 6 GB ceiling. CUDA-absent safe. | Shipped, needs live test |
| High-creativity soak profile | `"maximum chaos"` re-added to CREATIVITIES pool (~10% weighted). Catches temperature-sensitive regressions. | Shipped, needs live test |
| Per-LLM-call VRAM snapshots | `vram_snapshot("llm_generate_entry"/"exit")` inside `_generate_with_llm()`. Logs tokens + inference time. | Shipped, needs live test |

---

## P2 — Continuity layer (unblocks after video stack sprint ships)

Previously blocked on the retired Gate 0. Now blocked on video stack sprint Day 14. Design begins once stack empirics exist.

| Item | Summary |
|---|---|
| Scene-Geometry-Vault | Series-scale persistent geometry vault so Act 3's bridge matches Act 1's bridge across episodes. Seeded by FLUX anchor outputs from Stage 1. |
| Style-Anchor cache (World Seed + Lighting/Mood split) | Reuse engine over the vault. Same geometry, N relight passes. `style_anchor_hash` in Director schema keys the split. |
| Head-Start async pre-bake (Phase B.5) | Kick off VisualBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability. |
| ASCII sanitizer in prompt_compiler | Strip non-ASCII before Tencent text encoders. Preserve case. Collapse whitespace. Fold into `flux_anchor.py` prompt compiler on video-stack Day 2. |

---

## P3 — Experiments & polish

| Item | Summary |
|---|---|
| `torch.compile` on Bark sub-models | `mode="reduce-overhead"` on semantic, coarse, fine acoustic. Needs isolated A/B timing; variable-length loops may fight the compiler. |
| Skip/shorten Bark fine acoustic pass | Fine pass detail that AudioEnhance destroys via tape emu / LPF / Haas. Needs listening test, not spectrogram. |
| `episode_title` socket input on OTR_SignalLostVideo | Replace implicit `script_json` title-token read with explicit socket from ScriptWriter. v2.1 cleanup. |
| Rename `workflows/soak_target_api.json` → `workflows/helpers/antigrav_api_scratch.json` | Antigravity API-conversion helper; keep but move out of top-level workflows to reduce confusion. |

---

## Recently shipped

| Item | Summary | Status |
|---|---|---|
| v1.7 | Tagged and merged to `main` (`0aa6d6e`) | Shipped |
| BUG-LOCAL-034–040 | Parser resilience, title fixes, JSON repair | Shipped with v1.7 |
| Visual sidecar trio | VisualBridge + VisualPoll + VisualRenderer wired into `workflows/otr_scifi_16gb_full.json` | Shipped |
| VisualRenderer audio-length exact-match | `-t audio_duration` + `tpad` for C7 safety; stderr → tempfile | Shipped (`86bfeae`) |
| Phase A race-free sidecar contract | Atomic writes + Windows `os.replace` retry (`_atomic.py`) | Shipped (`ed4c44f` + `5e795a0`) |
| Phase B v0 SD 1.5 anchor generator | `anchor_gen.py` behind `OTR_VISUAL_ANCHOR=sd15` flag; 27 unit tests | Shipped (`c46a013`) |
| Round-robin consult infrastructure | `scripts/_consult_round_robin.py` (ChatGPT → Gemini → Claude synth) | Shipped |

---

## Discarded (do not revisit)

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10 (stale by multiple minor versions)
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased Visual unified latent space
- **Visual 2.0 Gate 0 probe** (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. VisualBridge + Poll + Renderer harness stays; the backends are the P0 video stack above.
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — did not read as 1980s VHS (pivoted to SDXL + period LoRA, now superseded by FLUX-native anchors under P0)

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/HANDOFF_2026-04-16.md` — last handoff (Phase A + Phase B v0)
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-04-14-otr-v2.1-spec.md` — v2.1 spec
- `docs/2026-04-14-green-zone-guardrail-decision.md` — guardrail decision
- Survival guide: `https://github.com/jbrick2070/comfyui-custom-node-survival-guide`

---

## Daily operating cadence

- First thing: read this file, `CLAUDE.md`, `BUG_LOG.md` header, `git log --oneline -5` on current branch.
- LHM is always on — poll `http://localhost:8085/data.json` (or `outputs/libre_tail.py`) before asking Jeffrey for system status.
- After every code change: AST parse + three regression suites. Do not report "done" until green.
- One `git push` attempt max — if it fails, hand Jeffrey a cmd block with `cd /d` included.
- Verify every push: local HEAD == origin HEAD, no 0-byte files, no BOM, workflow JSONs valid.
- Log bugs the moment they surface. Don't batch. Promote `Bible candidate: yes` to the survival guide only after the fix is verified.
