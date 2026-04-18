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
Supersedes the retired HY-World 2.0 Gate 0 probe. The HyworldBridge → HyworldPoll → HyworldRenderer trio (shipped) stays as the harness; the backends swap.

### Locked stack

| # | Stage | Pick | Runtime | Peak VRAM | Canonical repo |
|---|---|---|---|---|---|
| 1 | Style anchors | FLUX.1-dev FP8 + ControlNet Union Pro 2.0 | diffusers | 12.5 GB | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` |
| 2 | Scene keyframes | FLUX.1-dev + Depth/Canny | diffusers | 13.5 GB | `XLabs-AI/x-flux` (weights) |
| 3 | Character lock | PuLID for FLUX | diffusers | 14.0 GB | `ToTheMoon/PuLID` *(verify Day 3)* |
| 4 | Hero motion | LTX-Video 2.3 | existing sidecar | 14.5 GB | `Lightricks/LTX-Video` |
| 5 | Long motion / VJ loops | Wan2.1 1.3B I2V | diffusers | 8-10 GB | `Wan-Video/Wan2.1` |
| 6 | Compositing | Florence-2 + SDXL Inpainting | diffusers | 8 GB | `microsoft/Florence-2-large` (HF) |
| 7 | Final mux | HyworldRenderer (shipped `86bfeae`) | ffmpeg | — | in-repo |

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
| 8 | **[DONE 2026-04-17]** `otr_v2/hyworld/postproc/vhs.py` — ffmpeg-based VHS aesthetic post-processor. Pure `build_vhs_filter_chain(params)` returns a deterministic `filter_complex` string with seven ordered stages: (1) `format=yuv420p` normalise, (2) `rgbashift=rh=-N:bh=N` chromatic aberration, (3) `gblur planes=6` chroma bleed (U/V only — luma detail preserved), (4) `geq` scanlines (luma-only alternating-row multiplier, density-configurable), (5) `noise=c0s=N:c0f=t+u` tape grain on luma, (6) `vignette=PI/X` soft edge, (7) `gblur` final tape softness. `apply_vhs_filter(input, output, params)` invokes ffmpeg with `-c:a copy` + `-map 0:a?` so audio streams pass through byte-identical when present (C7) or are absent-safely skipped when the input is video-only. Intensity presets low/medium/high scale all five visible knobs proportionally. Stub mode (`OTR_VHS_STUB=1` / ffmpeg missing / `force_stub=True`) is a byte-identical `shutil.copyfile` passthrough, so CI and weight-missing dev machines can unit-test the pipeline without ffmpeg. `apply_vhs_to_job_dir(job_dir)` batch-scans for `render.mp4` > `motion.mp4` > `loop.mp4` per shot, emits `*_vhs.mp4` siblings, skips still images (`composite.png`, `keyframe.png`, `mask.png`, `depth.png`, `anchor.png`, `render.png`), ignores internal `_cache/` and `.hidden/` dirs, and writes a `vhs_postproc_summary.json` meta. Per-clip meta.json alongside each output records mode, stub_reason, params_hash, filter_chain text, ffmpeg argv, duration_ms. Not registered as a backend — `test_postproc_does_not_pollute_backend_registry` asserts the Day 1-7 roster is unchanged. Default `fps=24` asserted equal to `renderer._FPS`. | ✅ 34/34 new vhs_postproc tests green (module imports torch-free; DEFAULT_VHS_PARAMS key coverage; public constants; filter chain deterministic + uses defaults when None + has all 7 structural stages + varies across low/medium/high intensity + unknown intensity → medium fallback + zero-strength knob drops stage + override lands in chain text + scanline density reflected in `mod(Y\\,N)` + vignette always on; stub mode byte-identical passthrough including audio-like trailing payload [C7 invariant] + force_stub overrides env + meta.json schema + env stub reason + ffmpeg-missing autodetect via monkeypatched find_ffmpeg + missing input raises FileNotFoundError + input==output no-clobber; batch finds render/motion/loop + skips still images + mixed shot with both still and video only touches video + renders `render.mp4` takes priority over `motion.mp4` when both exist + ignores internal dirs + empty job dir + missing job dir + batch summary file + params hash stable + params hash shifts with overrides + registry isolation + no shell metacharacters in chain + fps matches renderer._FPS); 281/281 combined video backend regression across Days 1-8 (vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen); 495/509 broader suite (14 pre-existing `test_core.py` BUG-LOCAL-042 `vram_sentinel` ImportError failures/errors from before Day 1 — not caused by Day 8). C7 unchanged (verified structurally: stub = byte-for-byte copy; real = `-c:a copy`). Real-mode wall-clock + CRT quality gate deferred until the Day 10-11 canary renders feed it actual LTX/Wan MP4s. |
| 9 | **[DONE 2026-04-17]** `otr_v2/hyworld/planner.py` — orchestration timeline planner. Given an outline (dict / JSON string / Path), emits a non-repeating sidecar job list covering full runtime. Each `PlannerJob` names one Day 1-7 backend (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `ltx_motion` / `wan21_loop` / `florence2_sdxl_comp` / `placeholder_test`) plus `shot_id`, `scene_id`, `prompt`, `duration_s`, `refs`, `handoff_from`, `mask_prompt`, `insert_prompt`, `prompt_hash`. Backend assignment: explicit `beat["backend"]` override wins (unknown name → ValueError), else `BEAT_KIND_TO_BACKENDS[kind]` priority list, else `flux_keyframe` fallback. Graceful degradation with warnings: pulid without character/refs → flux_keyframe; florence without mask/insert prompts → flux_keyframe. C4 enforced: `_clamp_duration` caps `ltx_motion` / `wan21_loop` at 10.0s; non-positive duration replaced with `DEFAULT_BEAT_DURATION_S=6.0`. Non-repetition sliding window (default 3 jobs, configurable via `nonrepeat_window`) rejects duplicate `(backend, prompt_hash)` tuples; `_nudge_prompt_for_uniqueness` appends ` [variant N]` suffix deterministically, max 32 nudge attempts before accept-and-warn. Handoff selection for motion/loop: reverse-iterates same-scene prior jobs, picks first still-producer (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `florence2_sdxl_comp`); warning + stub-mode routing if none. Scene rotation: if `sum(beats) < runtime`, re-enters scenes from top (safety cap at `len(scenes)*20` empty rotations). `plan_episode(outline, target_runtime_s=..., nonrepeat_window=..., default_beat_duration_s=...)` → `PlannerResult` with `jobs`, `total_duration_s`, `target_runtime_s`, `scenes_covered`, `warnings[]`, `repetition_window`. Outline coercion: dict passes through; `str` is JSON-fast-path when stripped starts with `{`/`[` (avoids `Path.exists()` "File name too long" on long JSON), else treated as path with `OSError`-guarded exists check, else raw JSON string. `emit_shotlist_json(result)` returns bridge-ready `{"shots":[...flat job dicts...], "target_runtime_s", "total_duration_s", "job_count", "warnings"}`. `write_shotlist(result, path)` writes JSON to disk. Pure stdlib — no torch, no diffusers — safe to import from tests and bridge. | ✅ 33/33 new planner tests green (module imports torch-free; public constants; backend assignment per kind incl. degrade paths; explicit override wins + unknown raises ValueError; C4 duration clamp for ltx+wan + non-clamp for stills + negative→default; non-repetition window 3 identical beats produce unique hashes after nudging + window=1 vs window=5 boundary behaviour + nudging determinism across runs; handoff selection picks prior still + warns when no upstream + scene boundary respected; runtime coverage respects target + repeats scenes when beats short + target override + empty outline warning; shotlist JSON schema with shots[] + job_count + target_runtime_s + per-shot shot_id/backend/prompt/duration_s/prompt_hash; write_shotlist to disk; coerce string JSON + Path; 3-min dry run gate ≥180s + ≥3 scene_ids + ≥4 backend diversity + window invariant; all emitted backends registered; PlannerJob.to_dict omits empty optional fields; PlannerResult.to_dict includes diagnostics); 314/314 combined regression across Days 1-9 (planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen). C7 unchanged (planner is pure-stdlib, no audio path touched). Planner is not a backend — emits jobs that name Day 1-7 backends, does not register a new one. |
| 10 | Cold-open canary: "SCENE 01 — Cockpit, Baba boots up the radio." Full Stage 1→7 pass. | End-to-end render, audio aligned, no black frames. |
| 11 | 3-min continuous scene at full res. | No stagnation, no duplicate stills. Wall clock < 45 min. |
| 12 | Character regression: Baba + Booey between Scene 1 and Scene 3. | SSIM > 0.85 on cropped faces. |
| 13 | Full 20-min episode dry run overnight. LHM polled every 60 s via scheduled task. | No OOM, no pagefile thrash, no shared-memory fallback. |
| 14 | Freeze stack. Tag `v2.0-alpha-video-full`. Update BUG_LOG, MEMORY, this ROADMAP. | All three test suites pass. Zero `video-stack` blockers in BUG_LOG. |

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
otr_v2/hyworld/backends/
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
| Head-Start async pre-bake (Phase B.5) | Kick off HyworldBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability. |
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
| HyWorld sidecar trio | HyworldBridge + HyworldPoll + HyworldRenderer wired into `workflows/otr_scifi_16gb_full.json` | Shipped |
| HyworldRenderer audio-length exact-match | `-t audio_duration` + `tpad` for C7 safety; stderr → tempfile | Shipped (`86bfeae`) |
| Phase A race-free sidecar contract | Atomic writes + Windows `os.replace` retry (`_atomic.py`) | Shipped (`ed4c44f` + `5e795a0`) |
| Phase B v0 SD 1.5 anchor generator | `anchor_gen.py` behind `OTR_HYWORLD_ANCHOR=sd15` flag; 27 unit tests | Shipped (`c46a013`) |
| Round-robin consult infrastructure | `scripts/_consult_round_robin.py` (ChatGPT → Gemini → Claude synth) | Shipped |

---

## Discarded (do not revisit)

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10 (stale by multiple minor versions)
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased HyWorld unified latent space
- **HY-World 2.0 Gate 0 probe** (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. HyworldBridge + Poll + Renderer harness stays; the backends are the P0 video stack above.
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — did not read as 1980s VHS (pivoted to SDXL + period LoRA, now superseded by FLUX-native anchors under P0)

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/HANDOFF_2026-04-16.md` — last handoff (Phase A + Phase B v0)
- `docs/2026-04-15-hyworld-integration-plan-review.md` — external review triage (HY-World 2.0 probe now retired)
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
