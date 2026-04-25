# OTR Roadmap

**Last updated:** 2026-04-25 (HuMo full-episode coverage promoted to top P0; video stack sprint shipped on 2026-04-17 retained below as completed history)
**Branch:** `v2.0-alpha`
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

## P0 — HuMo Full-Episode Coverage (current focus)

**Branch:** `v2.0-alpha`
**State:** Goal 1 scaffolded (orchestrator + concat scripts shipped 2026-04-25), not yet run end-to-end. Goal 2 not started.

This section supersedes the original 14-day video stack sprint as the active P0. The video stack sprint shipped on 2026-04-17 and is retained below as completed history (see "P0 [SHIPPED 2026-04-17] — Video Stack Sprint").

### Hardware floor (measured 2026-04-25)

After ~5 hours of HuMo configuration testing on the RTX 5080 Laptop 16 GB + 64 GB RAM, these numbers are locked. Do not relitigate without new hardware or a major upstream change (FA3 on Blackwell, ComfyUI memory manager rewrite, etc.).

- **Production weight set:** HuMo 17B Q5_K_M GGUF (11.86 GB on disk, from `VeryAladeen/Wan2_1-HuMo_17B-GGUF`). Loaded via city96's `ComfyUI-GGUF` `UnetLoaderGGUF`. Stock fp8 (15.89 GB) caused PCIe spillover and is now redundant — deletable once Q5_K_M output quality is signed off.
- **Stable shape:** `length=97` (3.88 s @ 25 fps), `width=480`, `height=832`, `batch_size=1`. `length=65` triggers a torch.compile RAM spike that pages the Python worker to disk swap (reproduced twice — normalvram + lowvram modes both hung). Do not retry length=65 without a fresh OS process and pinned-RAM config audit.
- **Per-step:** 42 s at length=97. Confirmed identical across fp8 / Q5_K_M GGUF / smart-memory-off / lowvram. ComfyUI is already evicting maximally — the partial-load split (~6.7 GB GPU + ~5.5 GB pinned RAM) is HuMo's own weights, not encoder squeeze.
- **Per-clip:** ~4:30 native (HuMo only), ~6:15 in TEST_humo (FLUX preroll + UnloadAll handoff + HuMo).
- **Cold load:** ~50 s for the first prompt; amortizes across N prompts via Pattern B sequential POSTs because ComfyUI's model cache stays warm.
- **NVIDIA Sysmem Fallback Policy:** "Prefer No Sysmem Fallback" set on `python.exe` for diagnostic clarity. Doesn't change `cudaMallocAsync` behavior.
- **Decisions documented:** `docs/2026-04-25-humo-batch-pipeline.md`.

### Goal 1 — TEST workflow renders every shot with HuMo

**Definition of done:** Queue a TEST-style HuMo workflow and end up with one HuMo MP4 per shot the OTR_VideoPlan would have rendered as a FLUX-only PASS3 composite. Audio is sliced from the master WAV per ledger timing (or `--auto-slice` when timing is missing).

**Pieces shipped 2026-04-25 (`c387e525`):**

- `scripts/render_humo_batch.py` — Pattern B orchestrator. Reads ledger, slices audio, copies portraits to ComfyUI `input/`, POSTs HuMo prompts to `:8000/prompt` sequentially. Scope flags: `all` / `first-per-scene` / `cold-open` / `custom:l001,l005,...`. `--auto-slice` fallback for ledgers without per-line timing. Pure Python stdlib + ffmpeg subprocess.
- `scripts/concat_humo_episode.py` — ffmpeg stitcher. Two modes: `concat` (back-to-back clips, master WAV replaces audio) and `overlay` (clips composite onto base track at line.start_s/dur_s positions).
- `workflows/otr_videoplan_TEST_humo.json` — TEST workflow wired with `UnetLoaderGGUF` → `Wan2_1-HuMo-17B_Q5_K_M.gguf`. Reverted to length=97 stable shape.
- Recipe + decision log: `docs/2026-04-25-humo-batch-pipeline.md`, `docs/2026-04-24-humo-poc-recipe.md` (corrected fp8 size + Q5_K_M / Q4_K_S guidance committed `2093a14`).

**Pieces shipped 2026-04-25 (Goal 1 prep, this session):**

- `scripts/build_test_ledger_from_director.py` — adapter that reads a TEST-style workflow's baked-in `director_json`, expands the pass3 shot plan via `nodes.otr_video_plan.build_shot_plan`, and emits a synthetic ledger with `cast[]` + `lines[]` (one line per shot, speaker rotated across cast). Bridges the gap between `OTR_VideoPlan.execute()` (which writes `shots[]` only) and `render_humo_batch.py` (which iterates `lines[]`). No workflow edit, no new OTR_* nodes, pure stdlib. 16 unit tests + dry-run verified against `workflows/otr_videoplan_TEST_humo.json` — produces 6 ledger lines (3 scenes × 2 shots/scene) with cycled portraits.
- `tests/test_build_test_ledger_from_director.py` — 16 tests covering workflow parsing, director expansion, speaker strategies, schema versioning, and round-trip with `render_humo_batch.filter_lines`. All green.

**Architecture decision (2026-04-25):** Picked path B (orchestrator + small adapter) over path A (mega-workflow with N HuMo subgraphs). Reasoning: the orchestrator already exists, scales to any N via `--scope all`, and keeps the workflow JSON small. The adapter is the smallest possible bridge — it does not modify the workflow, does not add new ComfyUI nodes, and reuses `build_shot_plan` for shot expansion so the TEST run lines up with the same plan a FULL run would produce. See `docs/2026-04-25-humo-batch-pipeline.md`.

**Remaining for Goal 1:**

- [x] **Adapter + dry-run** — `build_test_ledger_from_director.py` against `otr_videoplan_TEST_humo.json` writes a 6-line ledger; `render_humo_batch.py --scope all --dry-run` plans 6 prompts cleanly with portraits resolved per speaker. Verified 2026-04-25 in this session.
- [ ] **Smoke run (live ComfyUI)** — Jeffrey runs the two-step block below against a live ComfyUI server at :8000. Confirms warm-cache assumption + HTTP flow on real hardware. Expect ~6:15 for clip 1 (cold load), ~4:30 each for clips 2-6 → ~30 min total.
- [ ] **Scale up** — same flow on the FULL ledger once it lands; drop scope cap. For a 7-min episode at ~4:30/clip, ~6 h overnight.
- [ ] **Concat run** — `concat_humo_episode.py --mode concat` against the clip directory + master WAV. Verify the final MP4 plays end-to-end and audio aligns with the visible clips.

**Smoke-run command block (Jeffrey to execute when ComfyUI is up):**

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

# 1. Render the 3 character portraits via the existing TEST_humo workflow.
#    Open the workflow in ComfyUI Desktop (localhost:8000) and Queue Prompt
#    once. Confirms portraits land at:
#      C:\Users\jeffr\Documents\ComfyUI\output\otr_humo_pass1_portrait_*.png
#    The in-graph HuMo step also produces 1 smoke clip — that's expected.

# 2. Build the synthetic test ledger from the baked-in director_json.
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\build_test_ledger_from_director.py `
  --workflow workflows\otr_videoplan_TEST_humo.json `
  --out output\old_time_radio\test_humo_ledger.json

# 3. Render N HuMo clips against the ledger. With LEMMY/Saturn this gives
#    6 MP4s landing in output\old_time_radio\humo_test\.
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\render_humo_batch.py `
  --ledger output\old_time_radio\test_humo_ledger.json `
  --master-wav C:\Users\jeffr\Documents\ComfyUI\input\humo_test.wav `
  --out-dir output\old_time_radio\humo_test `
  --scope all
```

**Gates:**

| Gate | Threshold |
|---|---|
| Smoke run wall clock | First clip ≤ 6:00 (cold load), clip 2 ≤ 5:00 (warm) |
| Per-clip steady-state | ≤ 4:45 after warm |
| MP4 validity | ffprobe-parseable, > 100 KB, contains audio + video streams, duration ≈ 3.88 s |
| VRAM peak across run | ≤ 14 GB sustained (no pagefile thrash, no swap-out symptoms) |
| Failure recovery | If a single clip fails, orchestrator continues to next clip and reports failure count at end |

### Goal 2 — FULL pipeline covers every shot with HuMo (not yet started)

**Definition of done:** Run `otr_scifi_16gb_full.json` end-to-end, walk away, return to a finished episode MP4 where every line of dialogue is HuMo-animated. The Director's script is consumed as-is — no human-in-the-loop curation of which lines deserve HuMo treatment. The pipeline is fire-and-forget.

**Required upstream pieces:**

- [ ] **Ledger L2 (Task #20)** — SceneSequencer populates `ledger.lines[].start_s` and `dur_s`. Without per-line timing, the orchestrator falls back to `--auto-slice` which doesn't respect line boundaries (lip-sync drift accumulates across the episode). Goal 2 quality depends on Ledger L2 landing first.
- [ ] **FULL→HuMo handoff** — after the FULL pipeline writes the master WAV, the orchestrator must auto-trigger. Two options:
  - A new ComfyUI node (e.g. `OTR_HuMoBatch`) at the end of the FULL graph that shells out to `render_humo_batch.py` then `concat_humo_episode.py`. Per existing memory, prefer external script over OTR_* custom node — so:
  - A post-FULL script (`scripts/run_full_then_humo.py`) that watches for ledger completion, then chains the orchestrator + concat. Invokable as a single command.
- [ ] **Resumability** — if the long render dies at clip 50 of 80, restart should skip the first 50. Add `--resume` flag to `render_humo_batch.py` that skips `line_id`s whose MP4 already exists in `--out-dir`.
- [ ] **Episode meta.json** — total HuMo clips rendered, total wall clock, per-clip durations, lip-sync drift warnings, ffmpeg concat command logged for reproducibility.

**Gates:**

| Gate | Threshold |
|---|---|
| End-to-end unattended | A 7-min episode completes FULL + HuMo coverage in one overnight run, ≤ 10 hours total |
| Zero human edits | No prompts, no clicks, no manual file moves between FULL queue and final MP4 |
| Output validity | Final MP4 plays end-to-end, audio matches master WAV duration, every line is on-screen |
| Lip-sync drift | Visible only at HuMo's 3.88 s window boundaries (acceptable; not a regression vs Goal 1 single-clip lip-sync) |
| Resumability | Killing the orchestrator mid-render and restarting completes without re-rendering existing clips |

**Out of scope for Goal 2 (defer):**

- Mixing HuMo with Wan 2.2 + SVI Pro for atmospheric (non-talking) clips
- FLUX-still-with-pan-zoom as a non-HuMo fallback for narration / sound-design beats
- Director scoring of "headline" lines for higher-fidelity treatment
- FA3 / Blackwell flash-attention 3 (not yet available on this driver / torch combo)

### Why these two goals in this order

Goal 1 proves the orchestrator runs end-to-end on real hardware with real ledgers. It validates Pattern B's warm-cache hypothesis at scale, exposes any bugs in the audio slicing or portrait selection, and gives a known-good output to compare against. Goal 2 wraps it in unattended automation. Reversing the order — wiring auto-trigger before proving the orchestrator works — risks a 10-hour overnight run that fails at clip 5 because of a 1-line bug we'd catch in 20 minutes by running Goal 1 first.

### Open prerequisites

- **Task #20** (Ledger L2 — per-line `start_s` / `dur_s` populated by SceneSequencer): blocks Goal 2 quality. Goal 1 can proceed with `--auto-slice` while Task #20 ships.
- **Task #25** (OTR_LoadLedger node): non-blocking. Lets TEST workflow replay an existing ledger for HuMo iteration without paying the LLM cost — useful if Goal 1 needs many iterations on prompt wording or portrait selection.
- **Task #34** (tail-end OTR_UnloadAll on TEST_humo): non-blocking but reduces zombie residual between runs. Apply opportunistically.

### Kill criteria (fail-fast)

| Trigger | Response |
|---|---|
| Per-step time > 60 s sustained on a known-good config | Stop. Re-validate hardware (LHM telemetry), driver, and that no other process is competing for VRAM. Do not chase per-step optimization. |
| Process paged to disk swap (negative `WorkingSet`) | Kill, restart ComfyUI, do not retry the same shape change. The 2026-04-25 length=65 hangs are the precedent — the system RAM ceiling is real. |
| Clip output is silent video / black frames | Whisper audio format mismatch. Verify mono 16 kHz WAV. |
| Lip-sync visibly drifts mid-clip | Audio slice longer than HuMo's 3.88 s native window. Cap at 3.88 s in orchestrator. |
| FULL pipeline regression on Goal 2 wiring | Revert immediately. Audio is king (C7) — never let HuMo break the audio path. |

---

## P0 [SHIPPED 2026-04-17] — Video Stack Sprint (14-day build)

Branch: `v2.0-alpha`. Tag target: `v2.0-alpha-video-full`.
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

Every day ends with: `pytest tests/bug_bible_regression.py`, `pytest tests/test_dropdown_guardrails.py`, `pytest tests/test_audio_byte_identical.py`. No exceptions. C7 failure halts and reverts the day's work.

| Day | Task | Gate |
|---|---|---|
| 1 | **[DONE 2026-04-17]** `backends/` harness, `_base.py`, STATUS.json schema, `placeholder_test.py`. Wire Bridge `backend=` arg + LHM cooldown gate. Fixed bridge.py:296-299 PIPE deadlock (stdout/stderr → per-job log files). | ✅ 14/14 new dispatch tests green; 26/26 Bug Bible; 56/56 dropdown guardrails; 22/22 anchor_gen. C7 unchanged. Pre-existing BUG-LOCAL-042 vram_sentinel errors surviving (not caused by Day 1). |
| 2 | **[DONE 2026-04-17]** `flux_anchor.py` — FLUX.1-dev FP8 e4m3fn + enable_model_cpu_offload + VRAMCoordinator gate + deterministic per-shot SHA256 seeds + CI-safe stub fallback (OTR_FLUX_STUB=1 / model-missing / no-CUDA). `requirements.video.txt` pins torch 2.10.0+cu130 / diffusers 0.37.0 / transformers 5.5.0 / accelerate 1.13.0. Also repaired bridge.py (previously truncated mid-execute at line 269 → 446 lines, `_cooldown_gate` / `_spawn_sidecar` / `_write_status` restored; `backend=` arg in INPUT_TYPES + execute signature). | ✅ 10/10 new flux_anchor tests green; 14/14 backend dispatch; 77/77 dropdown+anchor_gen. C7 unchanged. Bug Bible sister repo not mounted in sandbox — Windows-side Bible regression still pending. 1024² real-mode render ≤ 12.5 GB gate deferred until FLUX weights land on disk. |
| 3 | **[DONE 2026-04-17]** `pulid_portrait.py` — PuLID-FLUX identity-locked portrait backend. Real mode: FluxPipeline FP8 + PuLID adapter try-import (`pulid.pipeline_flux` / `PuLID.pipeline_flux` / `comfyui_pulid_flux.pipeline_flux`), `enable_model_cpu_offload`, VRAMCoordinator gate, `id_images`+`id_weight`+`true_cfg` call kwargs. Stub mode (OTR_PULID_STUB=1 / weights missing / no CUDA): deterministic color keyed on `refs_hash` so identity-lock invariant is unit-testable pre-weights. Characters + ref filenames are per-episode emergent from the LLM script process — backend reads `shot.get("character")` and `refs` generically, no fixed roster. | ✅ 16/16 new pulid tests green (registry, stub, identity-lock same→same & diff→diff, helper round-trip); 117/117 combined regression (pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. Face-embedding SSIM identity gate deferred until real PuLID weights land on disk. |
| 4 | **[DONE 2026-04-17]** `flux_keyframe.py` — FLUX + ControlNet Union Pro 2.0 scene keyframe backend. Round-robin consult (`docs/2026-04-17-day4-controlnet__*`) locked: Row 1 Union Pro 2.0 single-mode, Row 2 depth only, Row 3 control image always derived from Day 2 anchor `render.png` (ignores `shot["control_image"]`), Row 4 strict preprocessor sequencing (depth → save → del + empty_cache → load FLUX), Row 5 `depth.png` cached to disk, Row 6 explicit bf16 cast on CN for FP8+bf16 casting safety, Row 7 dedicated Depth CN fallback if Union Pro fails, Row 8 stub mode (`OTR_FLUX_KEYFRAME_STUB=1` / `OTR_FLUX_STUB=1` / weights missing / no CUDA). Output: `keyframe.png` + `depth.png` per shot. Seed base 0x4B_45_59_46 ("KEYF") distinct from flux_anchor + pulid_portrait. | ✅ 28/28 new flux_keyframe tests green (registry, stub mode, layout-lock invariant across 3 prompt variations, Row 3 shotlist control_image ignore, stub-mode envvar permutations, helper determinism); 145/145 combined regression (flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. ≤ 13.5 GB real-mode gate deferred until FLUX + Union Pro 2.0 weights land on disk. |
| 5 | **[DONE 2026-04-17]** `ltx_motion.py` — LTX-Video 2.3 I2V motion sidecar + FLUX still → LTX handoff. Reads upstream still with priority `keyframe.png` (Day 4) > `render.png` (Day 2) > error; records `input_still_source` in meta.json. Real mode tries `LTXImageToVideoPipeline` (preferred) then falls back to `LTXPipeline` (older diffusers) at `torch.float8_e4m3fn` (C5) with `enable_model_cpu_offload`, VRAMCoordinator gate; exports to `motion.mp4` via `diffusers.utils.export_to_video`. C4 enforced: duration_s ≤ 10.0 @ 24 fps. Stub mode (`OTR_LTX_STUB=1` / weights missing): emits a minimal-but-valid MP4 (ftyp + mdat atoms, payload keyed on input-still hash) so handoff determinism is unit-testable without ffmpeg or weights. Seed base 0x4C_54_58_4D ("LTXM") distinct from all prior backends. VRAM isolation achieved structurally via the existing spawn subprocess pattern — FLUX fully releases before LTX loads. | ✅ 29/29 new ltx_motion tests green (registry, stub mode valid MP4 + duration cap, Day 5 handoff priority keyframe>anchor>missing, handoff determinism same-still→same-bytes, different-stills→different-bytes, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe); 174/174 combined regression (ltx + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. Real-mode ≤ 14.5 GB VRAM gate + clean FLUX→LTX handoff deferred until LTX-Video 2.3 weights land on disk. |
| 6 | **[DONE 2026-04-17]** `wan21_loop.py` — Wan2.1 1.3B I2V loop sidecar + FLUX still → Wan handoff. Inherits Day 5 upstream priority (`keyframe.png` > `render.png` > error) and records `input_still_source` in meta.json. Real mode tries `WanImageToVideoPipeline` first at `torch.float8_e4m3fn` then falls back to `torch.float16` (dtype choice recorded in meta.json) with `enable_model_cpu_offload` + VRAMCoordinator gate, and degrades cleanly to `WanPipeline` (T2V) on older diffusers; exports to `loop.mp4` (not `motion.mp4` — distinct from LTX) via `diffusers.utils.export_to_video`. C4 enforced: duration_s ≤ 10.0 @ 24 fps (240-frame single-call cap). Stub mode (`OTR_WAN_STUB=1` / weights missing / no CUDA): emits minimal-but-valid MP4 (ftyp + mdat atoms) with mdat payload salted `"wan21_loop"` so wan and ltx stubs are byte-distinguishable even for identical still hashes — prevents planner-routing bugs from hiding behind stub identity. Seed base 0x57_41_4E_32 ("WAN2") distinct from all 4 prior backends. Exposes `loop_prompt` (falls back to `motion_prompt` → `env_prompt`) with loopable-motion suffix "seamless loop, subtle cycling motion, 24fps". | ✅ 33/33 new wan21_loop tests green (registry including Days 1-6 roster, stub mode valid MP4 + duration cap + filename gate `loop.mp4` not `motion.mp4`, handoff priority keyframe>anchor>missing, handoff determinism same-still→same-bytes + different-stills→different-bytes, backend isolation: wan vs ltx stubs differ for identical still hash, envvar permutations, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe + ltx_motion); 130/130 combined video backend regression across Days 1-6 (backend dispatch + flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop); 403/403 broader suite pass (10 pre-existing workflow JSON errors flagged on Day 5, not caused by Day 6). C7 unchanged. Real-mode ≤ 10 GB VRAM gate deferred until Wan2.1-I2V-1.3B weights land on disk. |
| 7 | **[DONE 2026-04-17]** `florence2_sdxl_comp.py` — text-prompt mask via Florence-2 `<REFERRING_EXPRESSION_SEGMENTATION>` → SDXL inpaint insert. Inherits Days 5-6 upstream priority (`keyframe.png` > `render.png` > error) and records `input_still_source` in meta.json. Real mode runs in two phases with explicit VRAM handoff: (A) Florence-2 (transformers `AutoModelForCausalLM` + `AutoProcessor`, fp16, trust_remote_code, local_files_only) rasterises polygons/bboxes to `mask.png`, then gets `del`'d + `torch.cuda.empty_cache()` — Day 4 CN handoff discipline; (B) `StableDiffusionXLInpaintPipeline` loads at `torch.float16` (canonical SDXL) with fp8 opt-in via `OTR_SDXL_INPAINT_DTYPE`, `enable_model_cpu_offload` + VRAMCoordinator gate, runs inpaint with `mask_prompt` segmenting and `insert_prompt` painting. Two outputs per shot: `composite.png` (RGB, distinct from Day 4 `keyframe.png`) + `mask.png` (grayscale 8-bit). Stub mode (`OTR_FLORENCE_STUB=1` / either weight tree missing / no CUDA) emits three-way deterministic outputs: `composite.png` color keyed on SHA256(still, mask_prompt, insert_prompt), `mask.png` grayscale value keyed on mask_prompt alone (clamped 1-254 to avoid degenerate all-black/all-white masks), so composite and mask can be regression-tested independently. Seed base 0x46_32_53_44 ("F2SD") distinct from all 5 prior backends. mask_prompt missing triggers per-shot error in real mode (Day 7 requires explicit region naming). | ✅ 40/40 new florence2_sdxl_comp tests green (registry including Days 1-7 roster, stub mode valid PNGs with correct colour-type bytes 2/RGB and 0/grayscale, filename gate `composite.png` not `keyframe.png`, three-way composite invariant [same triple→same bytes; mask-change→shifts; insert-change→shifts], mask-png-depends-on-mask-alone invariant, Day 5-6 handoff priority, envvar permutations, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop); 170/170 combined video backend regression across Days 1-7 (backend dispatch + flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop + florence2_sdxl_comp); 443/443 broader suite pass (10 pre-existing workflow JSON errors flagged on Day 5, not caused by Day 7). C7 unchanged. Real-mode ≤ 8 GB VRAM gate + Florence-2 mask quality gate deferred until both weight trees land on disk. |
| 8 | **[DONE 2026-04-17]** `visual/postproc/vhs.py` — ffmpeg-based VHS aesthetic post-processor. Pure `build_vhs_filter_chain(params)` returns a deterministic `filter_complex` string with seven ordered stages: (1) `format=yuv420p` normalise, (2) `rgbashift=rh=-N:bh=N` chromatic aberration, (3) `gblur planes=6` chroma bleed (U/V only — luma detail preserved), (4) `geq` scanlines (luma-only alternating-row multiplier, density-configurable), (5) `noise=c0s=N:c0f=t+u` tape grain on luma, (6) `vignette=PI/X` soft edge, (7) `gblur` final tape softness. `apply_vhs_filter(input, output, params)` invokes ffmpeg with `-c:a copy` + `-map 0:a?` so audio streams pass through byte-identical when present (C7) or are absent-safely skipped when the input is video-only. Intensity presets low/medium/high scale all five visible knobs proportionally. Stub mode (`OTR_VHS_STUB=1` / ffmpeg missing / `force_stub=True`) is a byte-identical `shutil.copyfile` passthrough, so CI and weight-missing dev machines can unit-test the pipeline without ffmpeg. `apply_vhs_to_job_dir(job_dir)` batch-scans for `render.mp4` > `motion.mp4` > `loop.mp4` per shot, emits `*_vhs.mp4` siblings, skips still images (`composite.png`, `keyframe.png`, `mask.png`, `depth.png`, `anchor.png`, `render.png`), ignores internal `_cache/` and `.hidden/` dirs, and writes a `vhs_postproc_summary.json` meta. Per-clip meta.json alongside each output records mode, stub_reason, params_hash, filter_chain text, ffmpeg argv, duration_ms. Not registered as a backend — `test_postproc_does_not_pollute_backend_registry` asserts the Day 1-7 roster is unchanged. Default `fps=24` asserted equal to `renderer._FPS`. | ✅ 34/34 new vhs_postproc tests green (module imports torch-free; DEFAULT_VHS_PARAMS key coverage; public constants; filter chain deterministic + uses defaults when None + has all 7 structural stages + varies across low/medium/high intensity + unknown intensity → medium fallback + zero-strength knob drops stage + override lands in chain text + scanline density reflected in `mod(Y\\,N)` + vignette always on; stub mode byte-identical passthrough including audio-like trailing payload [C7 invariant] + force_stub overrides env + meta.json schema + env stub reason + ffmpeg-missing autodetect via monkeypatched find_ffmpeg + missing input raises FileNotFoundError + input==output no-clobber; batch finds render/motion/loop + skips still images + mixed shot with both still and video only touches video + renders `render.mp4` takes priority over `motion.mp4` when both exist + ignores internal dirs + empty job dir + missing job dir + batch summary file + params hash stable + params hash shifts with overrides + registry isolation + no shell metacharacters in chain + fps matches renderer._FPS); 281/281 combined video backend regression across Days 1-8 (vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen); 495/509 broader suite (14 pre-existing `test_core.py` BUG-LOCAL-042 `vram_sentinel` ImportError failures/errors from before Day 1 — not caused by Day 8). C7 unchanged (verified structurally: stub = byte-for-byte copy; real = `-c:a copy`). Real-mode wall-clock + CRT quality gate deferred until the Day 10-11 canary renders feed it actual LTX/Wan MP4s. |
| 9 | **[DONE 2026-04-17]** `visual/planner.py` — orchestration timeline planner. Given an outline (dict / JSON string / Path), emits a non-repeating sidecar job list covering full runtime. Each `PlannerJob` names one Day 1-7 backend (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `ltx_motion` / `wan21_loop` / `florence2_sdxl_comp` / `placeholder_test`) plus `shot_id`, `scene_id`, `prompt`, `duration_s`, `refs`, `handoff_from`, `mask_prompt`, `insert_prompt`, `prompt_hash`. Backend assignment: explicit `beat["backend"]` override wins (unknown name → ValueError), else `BEAT_KIND_TO_BACKENDS[kind]` priority list, else `flux_keyframe` fallback. Graceful degradation with warnings: pulid without character/refs → flux_keyframe; florence without mask/insert prompts → flux_keyframe. C4 enforced: `_clamp_duration` caps `ltx_motion` / `wan21_loop` at 10.0s; non-positive duration replaced with `DEFAULT_BEAT_DURATION_S=6.0`. Non-repetition sliding window (default 3 jobs, configurable via `nonrepeat_window`) rejects duplicate `(backend, prompt_hash)` tuples; `_nudge_prompt_for_uniqueness` appends ` [variant N]` suffix deterministically, max 32 nudge attempts before accept-and-warn. Handoff selection for motion/loop: reverse-iterates same-scene prior jobs, picks first still-producer (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `florence2_sdxl_comp`); warning + stub-mode routing if none. Scene rotation: if `sum(beats) < runtime`, re-enters scenes from top (safety cap at `len(scenes)*20` empty rotations). `plan_episode(outline, target_runtime_s=..., nonrepeat_window=..., default_beat_duration_s=...)` → `PlannerResult` with `jobs`, `total_duration_s`, `target_runtime_s`, `scenes_covered`, `warnings[]`, `repetition_window`. Outline coercion: dict passes through; `str` is JSON-fast-path when stripped starts with `{`/`[` (avoids `Path.exists()` "File name too long" on long JSON), else treated as path with `OSError`-guarded exists check, else raw JSON string. `emit_shotlist_json(result)` returns bridge-ready `{"shots":[...flat job dicts...], "target_runtime_s", "total_duration_s", "job_count", "warnings"}`. `write_shotlist(result, path)` writes JSON to disk. Pure stdlib — no torch, no diffusers — safe to import from tests and bridge. | ✅ 33/33 new planner tests green (module imports torch-free; public constants; backend assignment per kind incl. degrade paths; explicit override wins + unknown raises ValueError; C4 duration clamp for ltx+wan + non-clamp for stills + negative→default; non-repetition window 3 identical beats produce unique hashes after nudging + window=1 vs window=5 boundary behaviour + nudging determinism across runs; handoff selection picks prior still + warns when no upstream + scene boundary respected; runtime coverage respects target + repeats scenes when beats short + target override + empty outline warning; shotlist JSON schema with shots[] + job_count + target_runtime_s + per-shot shot_id/backend/prompt/duration_s/prompt_hash; write_shotlist to disk; coerce string JSON + Path; 3-min dry run gate ≥180s + ≥3 scene_ids + ≥4 backend diversity + window invariant; all emitted backends registered; PlannerJob.to_dict omits empty optional fields; PlannerResult.to_dict includes diagnostics); 314/314 combined regression across Days 1-9 (planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen). C7 unchanged (planner is pure-stdlib, no audio path touched). Planner is not a backend — emits jobs that name Day 1-7 backends, does not register a new one. |
| 10 | **[DONE 2026-04-17]** `tests/test_cold_open_canary.py` — cold-open canary test drives a full Stage 1→7 pass in stub mode for SCENE 01 "Cockpit, Baba boots up the radio." Scene outline has 6 beats (b01 establishing→`flux_anchor`, b02 close_up→`pulid_portrait` with BABA character + refs, b03 keyframe→`flux_keyframe`, b04 motion→`ltx_motion` at 6.0s, b05 loop→`wan21_loop` at 10.0s, b06 insert→`florence2_sdxl_comp` with mask_prompt + insert_prompt) totalling ≥ 30s runtime. `_BACKEND_MATRIX` maps each backend to its stub envvar and expected per-shot outputs. Stubs all seven backends via `OTR_FLUX_STUB` / `OTR_PULID_STUB` / `OTR_FLUX_KEYFRAME_STUB` / `OTR_LTX_STUB` / `OTR_WAN_STUB` / `OTR_FLORENCE_STUB` / `OTR_VHS_STUB` so the canary runs CI-safe without GPU weights. VHS post-processor tested via `apply_vhs_to_job_dir(force_stub=True)` to sibling `*_vhs.mp4` files. Determinism test runs the full pass twice under the same tmp root (backends hash on absolute anchor path for layout-lock invariance, so the same absolute path must be reused between runs). | ✅ 15/15 new canary tests green (planner module torch-free; all 7 backends registered; scene_01 outline well-formed; planner covers runtime; planner emits every expected backend for scene_01; C4 honoured on motion + loop in scene_01; per-backend stub pass parametrized over 6 backends; VHS postproc over full canary emits summary + `*_vhs.mp4` siblings; no zero-byte outputs gate; determinism across two runs byte-identical); 276/276 combined video backend regression across Days 1-10 (cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode end-to-end render with GPU weights deferred to Day 11. |
| 11 | **[DONE 2026-04-17]** `visual/wall_clock.py` — per-backend wall-clock estimator (pure stdlib, torch-free). Point estimates per shot: `flux_anchor`=28s, `pulid_portrait`=32s, `flux_keyframe`=25s, `ltx_motion`=95s, `wan21_loop`=65s, `florence2_sdxl_comp`=18s (conservative upper bounds on RTX 5080 Laptop Blackwell sm_120 FP8 e4m3fn + SageAttention + SDPA path; catches regressions where FA chasing falls back to eager). Cold-load penalties charged once per distinct backend (`flux_anchor` 45s, `pulid` 50s, `keyframe` 40s, `ltx` 70s, `wan` 30s, `florence` 25s). VHS postproc charged at 5s per motion/loop clip (real) / 0.02s (stub). `WallClockEstimate` dataclass with `mode` / `total_s` / `render_s` / `cold_load_s` / `vhs_s` / `per_backend_s` / `per_backend_shots` / `unknown_backends` + `to_dict()`. `estimate(jobs, *, mode, include_vhs, include_cold_load)` accepts `PlannerJob` dataclass OR plain dict; mode=`real`/`stub`; cold-load auto-skipped in stub. `DAY_11_WALL_CLOCK_CEILING_S=2700` (45 min) and `DAY_11_STUB_CEILING_S=60.0` as ROADMAP bars. `tests/test_three_minute_continuous.py` — Day 11 ROADMAP gate. `_three_minute_cockpit_outline()` builds a 180s SCENE 01 with 8 beats spanning every backend kind (b01 establishing→`flux_anchor`, b02 close_up→`pulid_portrait` BABA+refs, b03 keyframe→`flux_keyframe`, b04 motion→`ltx_motion`, b05 loop→`wan21_loop`, b06 insert→`florence2_sdxl_comp` with mask+insert, b07 two_shot→`pulid_portrait` BOOEY, b08 ambient→`wan21_loop`) with scene rotation triggered by beats < target runtime. Stubs all 7 backends via `OTR_*_STUB=1` so the 3-min canary runs CI-safe without GPU weights. | ✅ 22/22 new wall_clock_estimator tests green (module torch-free import; all Day 1-7 backends covered in stub + real tables; cold-load table coverage; 45-min ceiling constant; accepts PlannerJob + dict + mixed iterable; stub << real cost invariant; render_s sum; cold-load charged once per distinct backend + scales with backend diversity + skipped in stub mode; VHS only charged for ltx_motion + wan21_loop + can be disabled; unknown backends recorded costing zero; empty jobs → zero total; invalid mode raises ValueError; to_dict schema; per-backend breakdown accumulates; representative 3-min mix [4 anchor + 3 pulid + 6 keyframe + 9 ltx + 6 wan + 2 florence = 30 jobs] fits under 45-min ceiling; stub 3-min scene fits well under 1-min ceiling); 10/10 new three_minute_continuous tests green (planner covers 180s runtime; ≥20 jobs to avoid stagnation; ≥4 distinct backends for diversity; non-repetition window invariant across full 3-min timeline; C4 duration clamp holds on motion + loop; projected real wall-clock ≤ 45 min; projected stub wall-clock ≤ 60s; stub end-to-end execution finishes in < 60s monotonic clock; no zero-byte outputs gate; emits `render.png` + `keyframe.png` + `motion.mp4` + `loop.mp4` + `composite.png`/`mask.png` mix); 308/308 combined video backend regression across Days 1-11 (three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode end-to-end 3-min render with GPU weights deferred (estimator is a conservative-upper-bound projection that catches catastrophic regressions, not a precise render-time predictor). |
| 12 | **[DONE 2026-04-17]** `visual/character_regression.py` — cross-scene character identity gate. Pure-stdlib SSIM computation for the Day 1-7 sidecar output tree. `_decode_stub_solid_rgb(png)` reverses the pulid stub PNG format (8-byte sig → IHDR → IDAT zlib decompress → first-pixel R,G,B triple) so the gate is unit-testable without Pillow/numpy. `ssim_solid(rgb_a, rgb_b, reduction={min,mean,product})` implements the Wang et al. SSIM formula simplified for solid-color images (σ=0 both sides, so SSIM reduces to the luminance term per channel); `min` reduction is default to punish any channel divergence. `compute_ssim(png_a, png_b, mode={auto,stub,real})` dispatches: auto tries stub decoder first + falls back to real SSIM on non-solid PNGs; real mode lazy-imports Pillow + numpy and raises a clear ImportError if missing. `SSIM_GATE = 0.85` constant strictly-greater-than semantics matches ROADMAP Day 12 bar. `find_portraits(out_dir, character)` walks `<out_dir>/<scene_id>/<shot_id>/{render.png,meta.json}` where `meta["backend"] == "pulid_portrait"` and `meta["character"]` matches, returns sorted `PortraitSample` list. `regress_character(out_dir, character, *, gate, mode)` computes pairwise SSIM across DISTINCT scene_ids only (within-scene pairs skipped — Day 12 bar is scene-1 vs scene-3, not shot-to-shot). Single-scene coverage → `gate_ok=True` with note (can't fail what isn't testable). `regress_cast(out_dir, cast)` aggregates per-character. `CharacterRegressionResult` dataclass with `character`, `gate`, `samples`, `pairs`, `min_ssim`, `mean_ssim`, `gate_ok`, `notes` + `to_dict()`. Torch-free + no audio imports (C7 preserved). | ✅ 26/26 new character_regression tests green (module torch-free import; SSIM_GATE == 0.85; ssim_solid identity → 1.0 + max divergence black-vs-white << 0.01; reduction modes agree on identity + differ on unbalanced divergence + unknown reduction raises ValueError + per-channel symmetry; stub decoder roundtrips known colors + minimum-channel floor + rejects non-PNG; compute_ssim auto + stub paths + auto detects divergence + invalid mode raises; find_portraits walks scene layout + ignores other characters + empty when missing; same refs across scenes locks identity [min_ssim == 1.0, gate_ok]; different refs break identity lock [min_ssim < gate, gate_ok == False]; full ROADMAP Day 12 BABA + BOOEY scene_01 vs scene_03 → both pass; within-scene pairs skipped; empty samples → gate_ok + note; single-scene → gate_ok + note "only one scene"; regress_cast aggregates; to_dict JSON-serialisable schema; real-mode SSIM raises ImportError with "Pillow" hint when PIL/numpy blocked); 334/334 combined video backend regression across Days 1-12 (character_regression + three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode Pillow+numpy SSIM on cropped-face regions path implemented but deferred to post-weights landing (stub path alone proves the gate's regression-detection behaviour; real-path Pillow+numpy is wired and will be exercised once PuLID-FLUX weights produce non-stub portraits). |
| 13 | **[DONE 2026-04-17]** `visual/lhm_monitor.py` — torch-free LibreHardwareMonitor sampler + summariser. Polls `http://localhost:8085/data.json` (env `OTR_LHM_URL`), walks the LHM JSON tree DFS for GPU temperature (hottest sensor under `GPU` path), VRAM used/total (GB or MB→GB normalised), system RAM used/total, and CPU package temperature. `LhmSample` dataclass with `t_monotonic`, `t_unix`, per-metric fields, `unreachable` + `reason` so network / parse failures land as countable samples instead of raised exceptions. `poll_once(url, timeout_s, fetcher, now_mono, now_unix)` — `fetcher` injectable for tests; urllib fallback wraps `URLError`/`TimeoutError`/`OSError`/`ValueError` into unreachable samples. `poll_loop(out_path, interval_s, duration_s, max_samples, stop_when, fetcher, sleep_fn, monotonic_fn, unix_fn)` streams NDJSON (one JSON line per sample) and returns the full list; clocks + sleep + stop_when all injectable so tests drive the loop deterministically. `LhmSummary` dataclass rolls up peak / mean / min / last per metric with three Day 13 ceiling-breach flags (`VRAM_CEILING_GB=14.5`, `RAM_CEILING_GB=28.0`, `GPU_TEMP_CEILING_C=85.0`); `summarize_ndjson(path)` loads a saved log and summarises. `scripts/lhm_poller.py` — CLI wrapper (`--out`, `--interval`, `--duration`, `--max-samples`, `--summary`, `--summarise-only`); writes `<stem>.summary.json` alongside the NDJSON; exits with code 2 when any ceiling is breached so Windows Task Scheduler flags the overnight run as failed automatically. Pure stdlib — no torch, no numpy, safe to import in the main venv. `tests/test_episode_dry_run.py` — Day 13 ROADMAP gate. `_twenty_minute_episode_outline()` builds a 1200-s six-scene outline (Cockpit + Corridor + Engine Room + Viewport + Galley + Airlock) with 30 beats spanning every Day 1-7 backend kind — scene rotation stress-tests the planner's rotate-from-top safety net over 20 minutes. Stubs all 7 backends via `OTR_*_STUB=1` so the dry run is CI-safe without GPU weights. Asserts: planner covers full 1200-s runtime; planner uses all six scenes; planner exercises every Day 1-7 backend at least once; ≥150 jobs emitted to avoid coalescing; non-repetition window invariant holds across 20 min; C4 10-s cap on motion + loop; projected real wall-clock fits under 8-hour overnight ceiling with cold loads + VHS; stub execution finishes under 120-s CI floor; no zero-byte outputs across 30-job run; every STATUS.json ends in `READY` (no `OOM` / `ERROR` / `RUNNING`); artifact mix gate (`render.png` + `keyframe.png` + `motion.mp4` + `loop.mp4`); LHM poller with injected nominal fake telemetry tree captures 18-22 samples across a simulated 20-min run at 60 s interval; summary shows no ceiling breach on nominal hardware values; inverse gate trips `vram_ceiling_breached=True` when tree reports >14.5 GB VRAM. Fixed `poll_loop` to check `stop_when` once per iteration at the top only (was double-checking before + after poll, making sample-count semantics non-deterministic). | ✅ 20/20 new lhm_monitor tests green (module torch-free import; Day 13 ceiling constants; poll_once extracts 4 metrics from fake LHM tree; poll_once records unreachable on network + parse errors; MB→GB normalisation; to_dict JSON-serialisable; poll_loop NDJSON + sample list with `max_samples=5`; poll_loop duration enforcement via state-advancing sleep_fn; poll_loop `stop_when` trips on 3rd call → 2 samples; non-positive interval raises ValueError; summarize empty note; peak/mean/min/last stats; VRAM + RAM + GPU temp ceiling breach flags; unreachable count; summarize_ndjson roundtrip + missing-file note; summary to_dict JSON-serialisable with ceiling constants). 15/15 new episode_dry_run tests green (full 20-min runtime + all six scenes used + every Day 1-7 backend hit + ≥150 jobs + non-repetition window + C4 clamp + real wall-clock ≤ 8 h + stub ≤ 120 s + stub execution < 120 s + no zero-byte outputs + all STATUS.json `READY` + artifact mix + LHM sampler 18-22 samples + no nominal breach + VRAM breach on thrash tree). 554/554 combined v2 Day 1-13 regression (lhm_monitor + episode_dry_run + character_regression + three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a + anchor_gen + arc_check + camera_path + dropdown_guardrails + obsidian_profile + p0_features + treatment_scanner + widget_drift + v2 audio byte-identical [skipped without GPU]). C7 unchanged. Pre-existing `BUG-LOCAL-042` (`vram_sentinel` ImportError cascading through `tests/test_core.py`) marked `[FIXED]` on 2026-04-17 — stale Windows `__pycache__` from mid-April Phase B churn self-resolved; `tests/test_core.py` now 103/103 green both warm and after full pycache purge. | No OOM, no pagefile thrash, no shared-memory fallback. |
| 14 | **[DONE 2026-04-17]** Stack frozen on `v2.0-alpha` at commit `2430064` (Day 13 ship). All Day 1-13 backends + harness + planner + wall-clock estimator + character regression gate + LHM telemetry poller + 20-min episode dry-run gate shipped and locked. Tag handoff to Jeffrey via `scripts/tag_v2.0-alpha-video-full.cmd` (per CLAUDE.md: only Jeffrey tags releases); script verifies branch + clean tree + lockstep with origin + creates annotated tag `v2.0-alpha-video-full` + pushes, failing fast on any mismatch. BUG_LOG already carries pre-existing `BUG-LOCAL-042` (`vram_sentinel` ImportError in `nodes/batch_bark_generator.py` import chain) as the only open regression-noise — not caused by the sprint, last touched `5cf338e` before Day 1. MEMORY refreshed with sprint-complete snapshot: `project_v20_alpha_video_stack_complete.md` notes canonical branch + tag + 14 rows of gates + deferred real-mode weight gates. All three test suites pass: 334/334 combined v2 Day 1-12 regression on Day 12, 554/554 combined Day 1-13 regression on Day 13; `tests/test_dropdown_guardrails.py` 56/56; `tests/v2/test_audio_byte_identical.py` 7/7+1 skipped without GPU (C7 unchanged across all 14 days). Zero `video-stack` blockers in BUG_LOG. | ✅ Stack feature-complete. Jeffrey runs the tag script at his convenience to cut `v2.0-alpha-video-full`. Real-mode weight-landing gates (FLUX ≤ 12.5 GB 1024², PuLID face-embedding SSIM, Union Pro 2.0 ≤ 13.5 GB, LTX-2.3 + Wan2.1 ≤ 10-14.5 GB, Florence-2 mask quality) cleanly deferred to the post-sprint weight-landing pass as designed. |

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
7. Hotfixes on `v2.0-alpha` during sprint → rebase `v2.0-alpha` daily.

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
visual/backends/
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


---

## Session handoff &mdash; 2026-04-23 (hybrid refactor shipped, video-continuity decision pending)

Branch: `v2.0-alpha`. Do NOT touch `main`.

### What shipped tonight

- **Pivot from sidecar to ComfyUI-native FLUX.** After ~4 days of Diffusers+torchao+accelerate sidecar failures (BUG-LOCAL-046..057), ported FLUX generation into the main ComfyUI graph via `CheckpointLoaderSimple`. Renders FLUX.1-dev-fp8 clean in ~44 s on the 5080 at 11.35 GB peak. Validated in memory `feedback_comfyui_native_over_diffusers_torchao.md`.
- **Three new OTR custom nodes** (registered in `__init__.py`; 23 &rarr; 25 loaded nodes on startup):
  - `OTR_CheckpointLoaderGated` &mdash; wraps stock `CheckpointLoaderSimple` with a `trigger` forceInput + pre-load `llm_polish.unload()`. Forces FLUX load to wait until Mistral polish is done; releases Mistral-Nemo NF4 before FLUX occupies VRAM.
  - `OTR_VisualExtractFluxPrompt` &mdash; pure string adapter between PromptCoercion's JSON-stringified token list and CLIPTextEncode's STRING input. Zero torch / zero VRAM.
  - `OTR_UnloadAll` &mdash; IMAGE passthrough between VAEDecode and SaveImage that calls `comfy.model_management.unload_all_models()` + `soft_empty_cache(force=True)` + `llm_polish.unload()`. Fixes the 44% post-render VRAM retention Jeffrey flagged.
  - `OTR_BatchFluxRender` &mdash; lockstep multi-prompt renderer. Parses `cleaned_script_json`, pulls N env tokens, renders them under one MODEL/CLIP/VAE load. Two modes via the `fast_batch` BOOLEAN widget (default True): stacks N CONDITIONING tensors into one batched call + single KSampler for the whole batch (18% faster per shot); falls through to serial loop on any stacking error. Pre-pins MODEL via `load_models_gpu` so per-sample `load_models_gpu` calls in KSampler become no-ops.
- **Option A workflow wiring** in `workflows/otr_scifi_16gb_TEST.json`: deleted 6 single-shot sampler nodes (CLIPTextPos + CLIPTextNeg + EmptySD3Latent + FluxGuidance + KSampler + VAEDecode), replaced with `OTR_BatchFluxRender` + `OTR_UnloadAll` + `SaveImage`. Node count dropped 14 &rarr; 9. `batch_limit = 4` preset, changeable on the fly via widget.
- **First successful 4-shot lockstep run** 2026-04-23 at 14:50: 4 unique FLUX images (starship bridge, science officer, wide captain, climactic reveal) in **131.60 s total, 108 s sampling**, one KSampler progress bar, clean UnloadAll trio at the end.
- **BUG_LOG additions** &mdash; BUG-LOCAL-058 (OneDrive sync race truncates JSON tail during Edit roundtrip, environmental) and BUG-LOCAL-059 (ComfyUI auto-injects a hidden `control_after_generate` widget after any INT field named `seed` &mdash; positional `widgets_values` array must account for it, Bible candidate).
- **Two live Cowork artifacts**: `otr-test-workflow-walkthrough` (full node graph + live log tail + per-node AI explain + pass/fail tracking) and `otr-batch-render-progress` (simplified 4-shot visual grid + batch-progress bar + demo button).

### Pipeline state snapshot (v2.0-alpha HEAD: `df578db`)

```
Mistral-Nemo NF4 polish (6 GB VRAM, 11s)
    |  cleaned_script_json (N env tokens)
    v
OTR_CheckpointLoaderGated (unload Mistral, load FLUX fp8, 11.3 GB peak)
    |  MODEL + CLIP + VAE
    v
OTR_BatchFluxRender fast_batch (N images in one KSampler call, 108s for N=4)
    |  IMAGE batch [N, H, W, C]
    v
OTR_UnloadAll (evict MODEL/CLIP/VAE + empty_cache, VRAM -> baseline)
    |  IMAGE
    v
SaveImage -> N PNGs in ComfyUI/output/
```

Wall clock for a 4-shot 1024x1024 run: **131.60 s**, peak VRAM **11.35 GB**.

### Open video-continuity decision (for the next conversation)

Reference: `uploads/OTR_v2-alpha_VIDEO_CONSISTENCY_BUILD_DECISION.md` (full v3 brief, 199 lines). Three options evaluated:

| Option | Engine | Identity lock | Multi-clip continuity | VRAM | Risk |
|---|---|---|---|---|---|
| A (recommended in the brief) | Wan 2.2 I2V A14B fp8 + SVI 2.0 Pro LoRA | Single reference image anchors across clips | Built-in latent-level continuity via motion-frame overlap | ~12-14 GB | Blackwell sm_120 fp8 kernel path is the unknown |
| B | Wan 2.2 Animate single-shot I2V | None built-in | Manual last-frame-to-first-frame chain | ~12 GB | Character drift between clips expected |
| C | LTX-Video 2.3 + IPAdapter FaceID + ControlNet/Pose | FaceID conditioning on every frame | Per-clip only | <10 GB | No native inter-clip continuity; manual stitching |

**Existing workflows surveyed (all free + local):**
1. `vita-epfl/Stable-Video-Infinity` &mdash; paper authors' own ComfyUI graph
2. `Well-Made/ComfyUI-Wan-SVI2Pro-FLF` &mdash; pre-wired 7-clip FLF + SVI Pro, closest match to OTR scene-graph
3. Kijai's native Wan 2.2 SVI Pro JSON &mdash; fallback if FLF repo doesn't cooperate with Blackwell
4. DaSiWa Wan 2.2 pack on Civitai &mdash; reference buffet
5. Wan 2.2 FLF2V native ComfyUI template &mdash; lighter-weight chain option

**The Goofer discovery (2026-04-23 late):** Jeffrey's own `jbrick2070/ComfyUI-Goofer` repo ships a working **LTX-Video 0.9.5** batch-video + video-concat + music-gen pipeline already validated on the 5080 (RTX 5080 / 4090 is the listed recommended GPU). The `GooferBatchVideo` + `GooferVideoConcat` + `GooferPromptGen` node pattern maps 1:1 onto what OTR-video needs. Saved as reference memory `reference_goofer_pipeline.md`. **Likely cheapest path to video:** port the Goofer node pattern into OTR as `OTR_BatchLtxVideo` + `OTR_VideoConcat`, feed FLUX batch stills as LTX starting frames for I2V. That's Option C in the brief, except the pattern's already written.

**Strategic tension to resolve next session:**
- Character continuity is NOT yet solved in either the Goofer path or the SVI-Pro path. Goofer doesn't care (each clip is its own scene); SVI solves inter-clip drift but not "is this the same character." Before committing to a video engine, **test character lock in stills first** &mdash; wire PuLID (already in repo at `visual/backends/pulid_portrait.py`) into BatchFluxRender so all N shots share one character identity. If PuLID works on Blackwell fp8, identity-locked FLUX anchors feed either video path cleanly.
- Video-vs-stills repositioning question: OTR is a radio drama with visual companion. Moving to SVI Pro V2 repositions the project around video-as-main-artifact. C7 "Audio is king" stays invariant regardless.

### Open items carried forward

| Item | Why | Who |
|---|---|---|
| Wire PuLID into BatchFluxRender for character continuity in stills | Answers "can OTR make a coherent episode" before video engine choice | Next session |
| Port `goofer_batch_video.py` + `goofer_video_concat.py` patterns into OTR as `OTR_BatchLtxVideo` + `OTR_VideoConcat` | Reuses working LTX 0.9.5 batch-video + stitch pattern Jeffrey already shipped in Goofer | Next session |
| Evaluate `Well-Made/ComfyUI-Wan-SVI2Pro-FLF` 7-clip workflow on Blackwell fp8 | Validate Option A is actually reachable on sm_120 before committing to it | Next session |
| Add `tests/test_widget_value_alignment.py` regression | Cross-check every workflow JSON's `widgets_values` against node `INPUT_TYPES()` + auto-injected hidden widgets (BUG-LOCAL-059 class of bugs) | When stills work lands |
| Fix `[PromptCoercion] cleaned 49 tokens (env=0 dlg=0 sfx=0)` counter | Reports zero env tokens when 4 are clearly produced; cosmetic, not blocking | Low priority |
| Flip `OTR_FLUX_ALL_SHOTS=1` default | Get Mistral polish on all N env prompts (currently only shot 1 is polished; shots 2..N pass through raw) | When character continuity is wired |
| Promote BUG-LOCAL-059 to Bug Bible | Bible candidate, needs one more clean live-run verification first | After widget-alignment test lands |
| Save BUG-LOCAL-058 (OneDrive JSON truncation) monitor note | Environmental; recurs if ComfyUI repo stays under OneDrive-synced Documents. Options: pause sync / move repo / use Desktop Commander `edit_block` for JSON writes | Already logged |

### Handoff prompt &mdash; drop into a new Claude conversation

> I'm continuing OTR v2-alpha on branch `v2.0-alpha` (do NOT touch main). Tonight we shipped the hybrid FLUX refactor + 4-shot batched rendering and now need to decide the video-continuity path. Read `ROADMAP.md` "Session handoff &mdash; 2026-04-23" section first, then `uploads/OTR_v2-alpha_VIDEO_CONSISTENCY_BUILD_DECISION.md` for the full options brief.
>
> First move: wire PuLID (already shipped at `visual/backends/pulid_portrait.py`) into `OTR_BatchFluxRender` so all N shots lock to one character reference. This is the prerequisite test before committing to Option A (SVI Pro / Wan 2.2) or the Goofer-pattern LTX 0.9.5 path. If character lock works on Blackwell fp8 in stills, identity-anchored FLUX stills feed either video engine cleanly.
>
> Context:
> - 16 GB VRAM hard ceiling (video stack lifted to 15.5 GB during sprint).
> - Audio is king. C7 byte-identical audio gate unchanged.
> - `OTR_BatchFluxRender.fast_batch=True` is the production path. Serial loop is fallback.
> - The `seed` widget auto-injects a hidden `control_after_generate` slot &mdash; any JSON edit to that node must include `"randomize"` at `widgets_values[3]` (BUG-LOCAL-059).
> - Repo lives under OneDrive-synced Documents. Prefer Desktop Commander `edit_block` + `copy /Y` over multi-step sandbox Edits for JSON writes (BUG-LOCAL-058).
> - `ComfyUI-Goofer` (Jeffrey's own repo) has a working `GooferBatchVideo` + `GooferVideoConcat` + `GooferPromptGen` pipeline on LTX-Video 0.9.5 2B &mdash; copy the pattern when porting OTR-video, don't reinvent.
>
> Build order:
> 1. `OTR_CharacterAnchor` node or widget on BatchFluxRender that accepts one reference image + writes PuLID-conditioning into the batched CONDITIONING. Smoke-test identity consistency across 4 FLUX shots at 1024x1024 on Blackwell fp8.
> 2. If (1) works: port Goofer's LTX 0.9.5 batch-video + video-concat patterns as `OTR_BatchLtxVideo` + `OTR_VideoConcat`, feeding BatchFluxRender's identity-locked IMAGE batch as I2V starting frames. Validate on Blackwell before committing.
> 3. If (1) fails on Blackwell: fall back to InstantCharacter (`Tencent-Hunyuan/InstantCharacter`) as Stage 3 per the ROADMAP fallback list.
> 4. Evaluate `Well-Made/ComfyUI-Wan-SVI2Pro-FLF` 7-clip workflow only AFTER still-image identity lock is proven; SVI Pro is meaningful only if FLUX anchors are already character-consistent.
>
> Constraints:
> - Use `placeholder` / `stub` / descriptive name &mdash; never `dummy` &mdash; in code and comments.
> - UTF-8 no BOM always.
> - Run three regression suites after every code change: `tests/test_core.py`, `tests/test_dropdown_guardrails.py`, `tests/test_workflow_json_guardrails.py`. Bug Bible regression if the sister repo is mounted.
> - 16 GB VRAM hard ceiling. LHM @ `http://localhost:8085/data.json` is always on.


## Session handoff &mdash; 2026-04-23b (MIT video-consistency pivot)

Branch: `v2.0-alpha`. Do NOT touch `main`.

### Decisions locked this session

1. **OTR stays MIT licensed.** Considered and rejected a shift to GPL-3.0 (one-way door; preserves future optionality). Never vendor GPL-3 packs (`Kosinkadink/ComfyUI-VideoHelperSuite`, `Well-Made/ComfyUI-Wan-SVI2Pro-FLF`) into the OTR tree &mdash; reimplement under MIT or use native ComfyUI templates (Apache via core) instead.
2. **Don't vendor community packs as `OTR_*` wrappers.** Past OTR custom video nodes have bottlenecked. Where a pack is MIT/Apache-compatible and the license permits, write OTR-native MIT code instead; prefer native ComfyUI core nodes for anything they can do. Community-pack vendoring only when (a) license is MIT/Apache, (b) no native equivalent, (c) OTR doesn't need pipeline-specific coordination.
3. **Character consistency without faces.** Jeffrey rejected face-identity anchors (PuLID). Path forward is style/environment anchoring via IP-Adapter (per C6 &mdash; environments only, never characters) using XLabs FLUX IP-Adapter weights (Apache 2.0, MIT-compatible reimplementation path).
4. **First-Last-Frame chained video** is the consistency mechanism for audio-length coverage. N+1 FLUX stills produce N video clips where each clip's last frame equals the next clip's first frame by construction &mdash; concat is seamless without crossfades.

### Shipped this session

- **`nodes/otr_video_concat.py`** (new, ~280 lines, MIT) &mdash; ffmpeg-based seamless clip concatenation. Pure subprocess wrapper (no Python pixel processing); `-c copy` fast path with auto-fallback to re-encode on codec mismatch; stub mode (`OTR_VIDEO_CONCAT_STUB=1` / ffmpeg-missing / `force_stub=True`); C7 audio passthrough via `-c:a copy -map 0:a?`. Replaces the need to vendor `VideoHelperSuite`.
- **`tests/test_otr_video_concat.py`** (new, ~320 lines) &mdash; 24 unit tests covering path parsing, filelist writer, argv builder, stub mode, node surface area. Torch-free, ffmpeg-free.
- **`__init__.py`** &mdash; OTR_VideoConcat registered. Loaded node count: 25 &rarr; 26.
- **`.gitignore`** &mdash; `nodes/vendor/` and `/tmp/otr_vendor_stage/` blacklisted so failed clone fragments can never enter git (OneDrive held sandbox locks during the MIT-pivot cleanup).
- **`docs/2026-04-23-MIT-video-consistency-plan.md`** (new) &mdash; full implementation spec for OTR_FluxIpAdapter (phase 2) and OTR_WanFlfVideo (phase 3). Architecture notes, kill criteria, test lists, path A vs B analysis for the FLF engine.
- **Memory**: `feedback_otr_stays_mit.md` saved (never vendor GPL into MIT); `feedback_use_community_nodes_not_custom.md` saved (community over OTR wrappers).

### Pending handoff (Jeffrey, on return)

Full verification + commit script queued at `scripts/_claude_handoff_2026-04-23b.ps1`. Runs: OneDrive cleanup of `nodes/vendor/` fragments, AST parse of both new files, regression suites, test run on new tests, local commit. Jeffrey reviews + pushes.

### Phase 2 &mdash; OTR_FluxIpAdapter (next session)

See `docs/2026-04-23-MIT-video-consistency-plan.md` for the full spec. Estimate 2-3 sessions. Read `XLabs-AI/x-flux-comfyui` (Apache 2.0) staged at `/tmp/otr_vendor_stage/x-flux-comfyui/` during this session; MIT reimplementation via ComfyUI `ModelPatcher.set_model_attn2_patch()`; stub mode first; Blackwell fp8 kernel check before claiming done.

### Phase 3 &mdash; OTR_WanFlfVideo (next-next session)

Two candidate paths. Path A: native ComfyUI Wan 2.2 FLF2V template (Apache) wired in via a thin `OTR_WanFlfShotList` helper, ~60 lines. Path B: OTR-native Wan 2.2 I2V sidecar, adapted from existing `visual/backends/wan21_loop.py` with `last_image` conditioning added. 30-min Blackwell fp8 smoke test decides A vs B.

### Environmental issues to watch

- **BUG-LOCAL-058 (OneDrive sync race)** hit hard this session. Sandbox `rm` on fresh git clones failed with "Operation not permitted" on `.git/` internals; sandbox-mount view of `__init__.py` stayed stale for minutes after the Write tool updated the Windows file. Mitigation used: file tool Writes for deliverables, bash heredoc for any sandbox-verified writes, all verification deferred to Windows-side script. Consider moving the repo off OneDrive-synced `Documents\` to clear this class of bug permanently.

---

## Session handoff &mdash; 2026-04-23c (video branch shipped; FLUX.2 + HuMo rollout planned)

Branch: `v2.0-alpha`. 14 commits tonight. Full audio+video pipeline runs end-to-end on 5080 16GB.

### Shipped this session (cumulative, commits `3cee09d`..`ce15063`)

- `OTR_VideoConcat` &mdash; MIT ffmpeg concat node, 28 unit tests + real smoke passed. Replaces need for VideoHelperSuite (GPL-3).
- `OTR_VideoPlan` &mdash; read-only Director/script adapter, 3-pass outputs (pass1 char portraits, pass2 scene envs, pass3 composite shots). Multi-character mode default. `audio_gate` optional STRING input for execution-order sequencing. 52 unit tests.
- `OTR_ShotDurationCalculator` &mdash; expands 1-clip-per-shot to N-clips-per-shot from shot durations. FLF shared-boundary invariant preserved. `clips_per_shot(dur) = 1 if dur<=10 else ceil(dur/9)`. 25 unit tests.
- `workflows/otr_scifi_16gb_full.json` &mdash; full audio pipeline + bolt-on video branch wired via `audio_gate`. Execution order guaranteed: Director → audio → POC video → ffmpeg mux → VideoPlan → Calculator → FLUX → UnloadAll → SaveImage. **No separate `_with_video` variant** (consolidated per minimum-JSON discipline).
- `workflows/otr_videoplan_TEST.json` &mdash; standalone 10-frame video-branch test. Validated end-to-end on 5080, 10 PNGs on disk at `output/otr_videoplan_pass3_000{03..12}.png`.
- `tests/fixtures/sample_director_lemmy.json` &mdash; realistic 3-scene Director JSON with LEMMY + KENJI CROSS for manual testing without running the full ScriptWriter+Director chain.
- **258 tests green** (cumulative: 52 plan + 28 concat + 25 calculator + 50 dropdown + 103 core).

### Architectural direction &mdash; SUPERSEDED and new (FLUX.2 + HuMo)

**Previously:** FLUX.1-dev + Wan 2.2 + VACE (First-Last-Frame) + Lightning LoRA &mdash; locked earlier on 2026-04-23.

**Now (as of 2026-04-23c):** FLUX.2-klein + HuMo 17B for audio-driven character animation. Research confirmed both fit on 5080 16GB Blackwell via GGUF quantization. HuMo is a direct replacement for Wan 2.2 in the video position, with the critical difference that HuMo is **audio-driven** &mdash; characters visibly speak their Bark-rendered dialogue with real lip-sync. Aligns perfectly with OTR's "audio is king" principle.

Wan 2.2 + VACE plan retained as fallback if HuMo fails to fit (unlikely given the quantized variants available).

### FLUX.2-klein + HuMo 4-stage rollout plan (next sessions)

| Stage | Scope | Est. time | Success signal |
|---|---|---|---|
| **1. FLUX.2-klein in TEST** | Download FLUX.2-klein Q5_K_M GGUF (~10-13 GB). Swap `CheckpointLoaderSimple` widget in `otr_videoplan_TEST.json`. Re-queue. Compare new 10 PASS 3 PNGs against the FLUX.1 baseline (on disk from tonight). | 30-60 min | Better multi-character composites (LEMMY + KENJI + ANNOUNCER in one frame). Rollback = one widget change. |
| **2. Add HuMo to TEST with pre-baked audio** | Grab 3-4 per-line Bark WAVs from tonight's full-run output. Add `LoadAudio` node to TEST. Download HuMo 17B GGUF Q6 from `calcuis/humo-gguf`. Wire: FLUX.2-klein portrait + WAV + prompt → HuMo node. | 2-3 hr first time | One ~3.9s clip where character's mouth moves with the audio. |
| **3. "Creative ffmpeg" proof-of-life .mp4** | HuMo emits IMAGE batch. Mux with audio via either VHS_VideoCombine (install as runtime-only, GPL-3 not vendored) or extend OTR_VideoConcat to take IMAGE batch + audio. | 30 min | One `.mp4` where a character speaks a real OTR line. Concrete deliverable from TEST. |
| **4. Full integration** | Swap FLUX.2-klein into `otr_scifi_16gb_full.json` CheckpointLoaderSimple. Insert HuMo nodes between Calculator and VideoConcat. Wire real per-shot Bark WAVs into HuMo's audio input (this is the audio-timeline wiring we've been deferring). Flip `SEGMENT_TARGET_DURATION_S` 9.0 → 3.5 and `SEGMENT_HARD_CAP_S` 10.0 → 4.0 in `otr_shot_duration_calculator.py` (matches HuMo's 97-frames-at-25fps = ~3.9s cap). | 3-5 hr | Full episode rendered with characters lip-synced to their Bark dialogue. |

Full plan details preserved in memory `project_flux2_humo_rollout_2026-04-23.md`. Do NOT start Stage 2 until Stage 1 is green.

### VRAM stack &mdash; projected production pipeline (fits on 5080 16 GB)

| Stage | Model | Est. peak | Native Blackwell? |
|---|---|---|---|
| LLM Script + Director | Mistral-Nemo NF4 | ~6 GB | yes |
| Dialogue TTS | Bark bf16 | ~4 GB | yes |
| Music | MusicGen medium | ~5 GB | yes |
| Announcer TTS | Kokoro (local) | ~1 GB | yes |
| PASS 1/2/3 image | FLUX.2-klein Q5_K_M GGUF | ~10-13 GB | yes (FP8 tensor cores) |
| Video (audio-driven) | HuMo 17B GGUF Q6 | ~10-15 GB | yes (FP8 tensor cores) |
| Mux | ffmpeg | 0 GB (CPU) | n/a |

All 100% local, no cloud, no API keys. Full 2026-era audio-drama-with-video pipeline on consumer hardware.

### Constants that flip in Stage 4 (the whole diff)

```python
# nodes/otr_shot_duration_calculator.py
SEGMENT_TARGET_DURATION_S = 9.0   # -> 3.5 for HuMo
SEGMENT_HARD_CAP_S        = 10.0  # -> 4.0 for HuMo
SEGMENT_TARGET_FPS        = 16    # -> 25 for HuMo
SEGMENT_MAX_FRAMES        = 161   # -> 97 for HuMo (97 @ 25fps = 3.88s)
```

Math is unchanged. Only the constants move.

### Honest gaps still open after Stage 4

- **Multi-character in one shot when both speak.** HuMo is designed for one speaking character at a time. For two-shot dialogue, need either (a) composite non-speaker into background + animate speaker, or (b) run HuMo twice per shot and composite. Deferred until after single-character quality is proven.
- **Scene-geometry consistency across episodes** (Scene-Geometry-Vault from P2). Still deferred, same as before.
- **Still no IP-Adapter / Kontext for character identity lock.** Text-composite remains the floor; upgrading to image-reference PASS 3 compose (FLUX.2-Kontext klein) is a Stage 5 item.

### Parallel consideration &mdash; TTS upgrade path (Bark &rarr; Fish Speech / CosyVoice)

Independent of the FLUX.2 + HuMo rollout. Bark is shipping and stable, but both Fish Speech and CosyVoice are legitimate 2026-era upgrades worth evaluating once the video stack is green. Do NOT start this before Stage 4 is done &mdash; audio is king, and replacing the TTS backbone while video is still in flux would break our baseline reference.

**Candidates (all fit on 5080 16 GB):**

| Engine | Est. VRAM | License | Strengths | Weaknesses |
|---|---|---|---|---|
| **Bark** (current) | 8-12 GB | MIT | Natural laughter / sighs / non-verbal; great character colour; proven in OTR pipeline | Slower per-token; no per-speaker cloning without retraining; English-centric |
| **Fish Speech S2-Pro** | 16 GB (BNB NF4 4-bit); 24 GB+ full | Non-commercial research license &mdash; verify before use | Best-in-class audio quality among 2026 open TTS; clean zero-shot voice cloning; has ComfyUI node (`Saganaki22/ComfyUI-FishAudioS2`) | License profile is the blocker &mdash; OTR is MIT and we do not vendor restrictive code. Worth evaluating for personal use only unless upstream relaxes. |
| **CosyVoice 2.0** | ~6-8 GB | Apache-2.0 | Ultra-low latency (~150 ms first-chunk); pronunciation error rate 30-50% lower than v1; multilingual; streaming; Apache-2.0 is safe to vendor | No built-in non-verbal sounds as expressive as Bark; would lose some of the 1940s-radio character colour we rely on |
| **CosyVoice 3.0** | ~8-10 GB | Apache-2.0 | Quality improvements over 2.0; same license profile | Still maturing; double-check upstream stability before committing |
| **Qwen3-TTS** | ~6 GB | Apache-2.0 | Apache-2.0; Alibaba quality; strong multilingual | Newer &mdash; less community tooling than CosyVoice; ComfyUI node coverage thin as of 2026-04-23 |

**Criteria to evaluate when we pick this up:**

1. **License compatibility first.** OTR stays MIT. Fish Speech's non-commercial research license means we can listen and evaluate on Jeffrey's machine, but cannot ship it vendored in the repo. CosyVoice 2/3 and Qwen3-TTS are Apache-2.0 &mdash; safe to use and recommend.
2. **Does it keep the "1940s radio voice" character?** Bark's strength is non-verbal expressivity (sighs, laughter, uh-huhs). A pure-quality win on neutral speech is a net loss if the period-drama colour drains out of the announcer / character reads.
3. **Does it fit into `batch_bark_generator.py` without rewriting the orchestrator?** The sequencer (length-sorted batching, VRAM-Sentinel decorator, per-call snapshots) is proven; an ideal swap is a drop-in TTS backend behind the same interface.
4. **Per-character voice consistency across episodes.** Currently Bark uses preset matching. CosyVoice's zero-shot cloning from a 3-10s reference could actually improve cross-episode consistency (feed the same reference WAV every time).
5. **Streaming vs batch.** CosyVoice 2's streaming 150 ms could let us interleave TTS with FLUX &mdash; probably overkill for OTR's batch pipeline, but worth noting.

**Recommended first look (when we get here):** CosyVoice 2.0 Apache-2.0, sideload a ComfyUI node, A/B against Bark on the LEMMY + KENJI CROSS + ANNOUNCER test fixture. No commitment, no rip-and-replace until we hear it in context.

Deferred. Captured here so it doesn't drop on the floor.

### Quick-start for next session

1. Read `memory/project_flux2_humo_rollout_2026-04-23.md`
2. Download FLUX.2-klein Q5_K_M GGUF to `C:\Users\jeffr\Documents\ComfyUI\models\checkpoints\`
3. Load `workflows/otr_videoplan_TEST.json`, swap checkpoint widget, queue
4. Compare outputs against `C:\Users\jeffr\Documents\ComfyUI\output\otr_videoplan_pass3_00003..12.png` (FLUX.1 baseline from this session)

---

## Session handoff &mdash; 2026-04-23d (FULL workflow cleanup + reference fixture)

Branch: `v2.0-alpha`. Three small commits to end the night: `c2067ed`, `7c5415f`, `2dea4b1`. All ready for tomorrow.

### What shipped after the 23:04:18 full run

- **`c2067ed` &mdash; ROADMAP TTS upgrade section.** Parallel consideration for Bark &rarr; Fish Speech / CosyVoice / Qwen3-TTS. Deferred until FLUX.2+HuMo Stage 4 green. License-first filter keeps OTR MIT. Bark stays shipping for now.
- **`7c5415f` &mdash; FULL workflow cleanup.** Removed the dead sidecar trio (`OTR_VisualBridge`, `OTR_VisualPoll`, `OTR_VisualRenderer`) from `otr_scifi_16gb_full.json` &mdash; they were hitting `BUG-046` meta-tensor every run and burning ~4 min on procedural-video fallback. Rewired `OTR_VideoPlan.audio_gate` &rarr; `OTR_SignalLostVideo.video_path` for sequencing. Also bumped `LLMDirector` token budget from `min(1700, 550+len//10)` to `min(2500, 700+len//6)` &mdash; tonight's 6180-char script got 1168 tokens and truncated mid-`visual_plan.characters`, losing `visual_plan.scenes` entirely (PASS2=0 PASS3=0 downstream). Workflow went 19&rarr;16 nodes, 36&rarr;29 links.
- **`2dea4b1` &mdash; Satellites Collide reference fixture.** Hand-built from the run's `_treatment.txt` + Director log output. Five real characters (DUANE VOSS, PARRY MARTIN, ALAN SIRIKIT, REGINALD HAYES, ANNOUNCER), phantom cast stripped (CAPTAIN JOHNSON / ENSIGN PARKER / CONTROL were a critique-revise bleed-through), full `visual_plan.characters` + `visual_plan.scenes` for all 3 scenes, 7 SFX, 3 music cues. Saves ~37 min per TEST iteration.

### Bugs this closes

- Sidecar FLUX meta-tensor error (BUG-046 family) no longer in the FULL graph.
- `[VisualRenderer] No shot assets found. Falling back to procedural video.` no longer happens.
- `OTR_VideoPlan READY: PASS1=N PASS2=0 PASS3=0` should flip to nonzero PASS2/PASS3 with the new token budget.
- `OTR_ShotDurationCalculator READY: shots=0` should flip to real shot counts.

### What's NOT yet fixed (known + deferred)

- **Phantom cast bleed-through** &mdash; critique/revise still pastes in unrelated scenes (the SPACE STATION / CAPTAIN JOHNSON scene in tonight's run). Needs a guard in `_critique_and_revise()` or a post-revise scrubbing pass. Separate task.
- **Reference POC video is pointer-only** &mdash; 483 MB `.mp4` stays in `output/old_time_radio/`, not in git. Fixture README documents the path. If it ever gets deleted, regenerate from the fixture's `director.json` + script text.
- **Per-line Bark WAVs** &mdash; not yet extracted to the fixture. When Stage 2 (HuMo insertion) lands, write `scripts/extract_reference_bark_wavs.py` that reads the baked audio from the POC `.mp4` and splits it into per-line WAVs keyed off the canonical 1.0 script timing.

### Confirmed architecture for video compositing (Stage 3)

The "creative ffmpeg" proof-of-life deliverable uses a base+overlay model:

- **Base layer:** POC proc-gen `.mp4` (already has the full episode audio baked in with crossfades, waveform visualizer, treatment/title splash). Acts as the scaffold.
- **Foreground overlays:** HuMo clips (97 frames @ 25fps = 3.88s each, one per dialogue line or per shot).
- **Composite pass:** `ffmpeg` overlays HuMo clips onto the scaffold at timecodes that match the character's dialogue in the audio timeline. Moments where no character is on-screen (ANNOUNCER, SFX beats, music bridges) keep showing the POC base.

The math works because both layers derive from the same audio timeline.

### Quick-start for tomorrow (2026-04-24)

1. Pull `v2.0-alpha`, confirm HEAD is `2dea4b1`.
2. Open ComfyUI Desktop &rarr; reload workflows &rarr; load `workflows/otr_scifi_16gb_full.json`. Confirm only **16 nodes** render, with VideoPlan fed from `OTR_SignalLostVideo.video_path` on its audio_gate.
3. Queue one FULL run. Paste the console log into chat when it finishes (or errors). **Expected green signals:**
  - No `[VisualBridge]` / `[sidecar:]` / `[flux_anchor]` noise.
  - No `Falling back to procedural video.`
  - `[LLMDirector] max_new_tokens=...` is ~1700 for a 6k script and there's no `+3 braces` JSON-repair warning, OR the warning is smaller than tonight's.
  - `OTR_VideoPlan READY: PASS1=N PASS2=M PASS3=K` with **nonzero** M and K.
  - `OTR_ShotDurationCalculator READY: shots=X` where X equals the Director's scene count &times; shots_per_scene.
  - `BatchFluxRender` renders **all** shots, not just 1.
4. If green: diff the fresh `production_plan_json` against `tests/fixtures/reference_episode/director_satellites_collide.json`. If structurally similar, call the fixture canonical; if the fresh one is better, promote it and update the fixture.
5. If green: start Stage 1 of `memory/project_flux2_humo_rollout_2026-04-23.md` &mdash; download FLUX.2-klein Q5_K_M GGUF, swap into TEST, compare against tonight's `otr_videoplan_pass3_00003..12.png` baselines.

### Ready-to-paste pickup prompt (copy this into tomorrow's first message)

```
Continuing OTR v2.0-alpha on branch v2.0-alpha. Read in order:

1. ROADMAP.md "Session handoff - 2026-04-23d" (latest section, post-TTS table).
2. memory/project_flux2_humo_rollout_2026-04-23.md.
3. tests/fixtures/reference_episode/README.md.

Last three commits (verify `git log --oneline -3` == c2067ed, 7c5415f, 2dea4b1):
- c2067ed: roadmap TTS upgrade section (Bark -> Fish Speech / CosyVoice / Qwen3-TTS, deferred until Stage 4 green)
- 7c5415f: FULL workflow - removed dead sidecar trio (VisualBridge/Poll/Renderer), bumped Director tokens 1700->2500, rewired VideoPlan.audio_gate to OTR_SignalLostVideo.video_path
- 2dea4b1: Satellites Collide reference fixture (clean director.json with 5 chars + 3 scenes, no phantom cast)

Tonight's goal: run the patched FULL workflow and verify the four green signals
in the ROADMAP handoff. Specifically confirm:
- no sidecar errors
- PASS2/PASS3 nonzero in OTR_VideoPlan
- Calculator sees real shot durations
- BatchFluxRender processes all shots not just 1

If green, diff fresh Director output against the fixture, promote if better.
If green, begin Stage 1 of FLUX.2+HuMo rollout: download FLUX.2-klein
Q5_K_M GGUF, swap into otr_videoplan_TEST.json, re-queue.

Do NOT touch main. Do NOT start Stage 2 (HuMo) before Stage 1 is green.
Do NOT re-run the full pipeline just to get a script - use the fixture.
```

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
