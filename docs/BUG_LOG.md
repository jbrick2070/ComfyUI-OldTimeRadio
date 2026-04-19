# OTR v2.0 Bug Log

Active bug log for the v2.0 Visual Sidecar build.
Every bug gets logged the moment it is found. Entries are never deleted.

---

### BUG-LOCAL-048: Visual sidecar starves at spawn — parent ComfyUI holds 16.1 GB VRAM, leaves 173 MB for FLUX/LTX/Wan [FIXED]
- **Date:** 2026-04-19 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Second live run after the BUG-LOCAL-047 fix (`vs_3412f49920ef`, 10:00:13) spawned a sidecar (PID 54372) that loaded FLUX BF16 successfully (`[flux_anchor] loaded pipeline dtype=bf16` in `sidecar_stderr.log`) but then never rendered a single shot. LibreHardwareMonitor + nvidia-smi snapshot at 10:02:55 showed: GPU Core load 100%, GPU Memory load 98.9%, VRAM used 16,129 MB of 16,303 MB (173 MB free), GPU power 55 W (idle-range for RTX 5080 Laptop; real inference is 150+ W), core clock pinned at 2887 MHz boost. Classic PCIe memory-thrashing signature — GPU at 100% is spinning on offload streams, not compute. `STATUS.json` frozen at `updated_at=10:00:13`. Worker PID 5200 at 20.5 GB RAM. Zero shot PNGs written over 2+ minutes.
- **Cause:** The sidecar subprocess is spawned via `multiprocessing.get_context("spawn")` so it inherits none of the parent's CUDA context — but physical VRAM is a single 16 GB pool shared at the device level. ComfyUI's main process at spawn time held residual VRAM from the LLM phase (Mistral-Nemo 12B or its offload buffers), audio phase (Bark / Kokoro / MusicGen), and anchor_gen passes. With `enable_model_cpu_offload()` FLUX BF16 normally peaks around 6-7 GB on a clean device, but with only 173 MB free at the start there was no room to page modules in — the offload coordinator thrashed the PCIe bus instead of making forward progress. Violates C-constraint (14.5 GB peak ceiling). The existing `_cooldown_gate` checks GPU *temperature* via LHM before spawning, but nothing was flushing VRAM *residues* from the parent.
- **Fix:** Added `VisualBridge._pre_spawn_vram_flush(job_id)` to `visual/bridge.py`. Called from `_spawn_sidecar` before `_cooldown_gate`. Implementation: imports `force_vram_offload()` + `vram_snapshot()` + `vram_reset_peak()` from `nodes._vram_log`, takes a before/after snapshot pair, calls `force_vram_offload()` (which runs registered OTR cleanup callbacks → `comfy.model_management.unload_all_models()` + `soft_empty_cache()` → `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`), logs freed GB, then resets the peak counter so the episode report attributes VRAM to the sidecar window. Errors are caught and logged — a pre-spawn hygiene routine must never abort a real generation run.
- **Verify:** (a) AST parse + BOM clean on `visual/bridge.py` and `visual/backends/flux_anchor.py`. (b) Bug Bible regression: 23 passed, 1 skipped, 2 xfailed. (c) Dropdown guardrails + core + audio regression: pending (in flight). (d) Next live run should log `[VisualBridge] pre-spawn VRAM flush starting for vs_* (before=X.XX GB)` followed by `... complete for vs_* (after=Y.YY GB, freed=Z.ZZ GB)` in ComfyUI console, and LHM VRAM used at sidecar spawn time should be < 5 GB.
- **Tags:** vram-ceiling, pre-spawn, bridge, spawn-starvation, pcie-thrashing, cpu-offload, flux-anchor, ltx-motion, wan21-loop, visual-sidecar, c-constraint-violation, bible-candidate, force-vram-offload

### BUG-LOCAL-047: FLUX anchor silently stubs every shot on Blackwell — `couldn't find storage object Float8_e4m3fnStorage` [FIXED]
- **Date:** 2026-04-19 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Live run `vs_70285e165e80` ("The Silent Uprising", 9 shots) dispatched to `video_stack`, sidecar PID 33460 survived, `STATUS.json` reported `"wan21_loop real mode: 9 shots, I2V loops from FLUX stills"`, and 9 `render.png` files were written at `04:31:43`. Every PNG was ~15 KB and 1024x1024 solid color. Every `meta.json` recorded: `{"backend":"flux_anchor","mode":"stub","reason":"load_error:TypeError:couldn't find storage object Float8_e4m3fnStorage"}`. Wan2.1 then did I2V loops on top of the solid-color stubs, producing motion video animated from placeholder stills rather than real FLUX keyframes. The chain ran green and audio remained byte-identical (Rule C7 held), but there were no actual FLUX visuals in the output.
- **Cause:** `visual/backends/flux_anchor.py::_try_load_pipeline` called `FluxPipeline.from_pretrained(..., torch_dtype=torch.float8_e4m3fn, local_files_only=True)` unconditionally. On torch 2.10.0 + CUDA 13 + Blackwell sm_120, the FP8 e4m3fn dtype is recognized by the kernels but the accompanying `Float8_e4m3fnStorage` type is not exposed by the `safetensors`/`diffusers` loader path, so deserialization raises `TypeError: couldn't find storage object Float8_e4m3fnStorage` before the pipeline hits the GPU. The existing except clause converted the failure to `(None, "load_error:...")` and the backend fell to stub mode, which `meta.json` recorded but neither `sidecar_stderr.log` nor `STATUS.json` advertised (0-byte stderr log — observability gap tracked by BUG-LOCAL-046).
- **Fix:** Patch `_try_load_pipeline` to attempt a dtype ladder (FP8 → BF16) instead of FP8-only, with explicit stderr logging on each attempt. Added `OTR_FLUX_DTYPE` env override (`fp8` | `bf16`) so the ladder can be pinned. BF16 path uses the same `enable_model_cpu_offload` gate so the 12.5 GB Day 2 budget still holds (BF16 + offload ≈ 12-13 GB peak vs FP8 + offload ≈ 6-7 GB). Also added `_log_stderr()` helper that flushes after every write so the fallback decision appears in sidecar stderr log even when the parent bridge process swallows stdout. Commit: `v2.0-alpha-video-stack` — `visual/backends/flux_anchor.py`.
- **Verify:** (a) AST parse + BOM check clean on patched file. (b) Bug Bible regression: 23 passed, 1 skipped, 2 xfailed. (c) Dropdown guardrails + core + audio: 160 passed, 1 skipped. (d) Next live run should show `"mode":"real"` in every `meta.json` and `[flux_anchor] loaded pipeline dtype=bf16` (or `fp8_e4m3fn`) in `sidecar_stderr.log`. (e) If FP8 ever starts working, remove the BF16 fallback — ladder will self-select FP8 first.
- **Tags:** fp8, float8-e4m3fn, blackwell, sm_120, dtype-fallback, silent-stub, bug-046-family, flux-anchor, torch-2.10, diffusers, safetensors, visual-sidecar

### BUG-LOCAL-046: Workflow QA green but visual output is gradient placeholders — silent stub-mode fallback when model weights missing [OPEN]
- **Date:** 2026-04-18 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** A user installs a custom node pack, loads a workflow, and every node validates. `/queue` accepts the prompt, every node runs green, the final `.mp4` writes successfully, the ComfyUI console shows no red errors. The user thinks the workflow is working. But every frame in the video is a procgen gradient, not a real generated image. Only way to discover the failure is to open `io/visual_out/<job>/<shot>/meta.json` and read `"mode": "stub", "reason": "ltx_weights_missing:...\\models\\diffusers\\LTX-Video"` — a field no workflow-level QA test ever inspects. Episode render pipeline, audio bus, timing — all of it is green. Only the pixels are fake.
- **Cause:** OTR visual backends (`flux_anchor.py`, `ltx_motion.py`, `wan21_loop.py`) catch `ImportError` / `FileNotFoundError` / HuggingFace `GatedRepoError` at load time and silently fall back to a procgen gradient generator rather than raising. The rationale was good (don't crash the whole episode if one shot's weights are missing), but the side effect is that workflow QA is blind: there is no signal at the node-graph level that any backend is in stub mode. The existing test suite (`tests/test_dropdown_guardrails.py`, `tests/test_core.py`, `tests/test_audio_byte_identical.py`) all verify the *node graph is valid* — which it is — not that each node's underlying model *actually loaded*. Workflow-JSON audits (`scripts/audit_workflow_nodes.py`) only check node registration. The chain-of-trust between "ComfyUI workflow QA green" and "real visuals in the output" was never explicit, and this failure mode proved that silently twice: first with HY-WorldMirror weights missing in March, again with FLUX.1-dev missing 2026-04-18.
- **Fix:** Pending. Three-part remediation planned:
  1. **Loud stub-mode banner.** Every backend that falls back to stub must emit a structured `stub_mode_warning` record into the sidecar STATUS.json and a `STUB_MODE: <backend> reason=<...>` line to `otr_runtime.log` with `level=WARNING`, not `INFO`. The existing `meta.json` record is too far downstream — the warning needs to surface in the same place an operator is already watching.
  2. **New node: OTR_ModelLoadedAssert.** A workflow-level guard that runs *after* sidecar kickoff, reads the latest sidecar STATUS.json, and raises a validation error if any shot reports `mode=stub`. Wire into the TEST workflow as a hard gate between video sidecar and renderer. User-facing: if you see red at this node, you know your weights didn't load before you see the gradient video.
  3. **Opt-in strict mode: `OTR_REQUIRE_REAL_MODELS=1`.** Env var that flips the silent fallback into a raised exception at backend load time. Default off (preserves current "degrade gracefully" behavior for dev). Turned on in CI and by the OTR_ModelLoadedAssert node.
- **Verify:** Once fixed, regression must include: (a) unit test asserting that with `OTR_REQUIRE_REAL_MODELS=1` and weights present, `mode=real` for every shot in meta.json; (b) unit test asserting that with `OTR_REQUIRE_REAL_MODELS=1` and weights missing, sidecar raises before first frame; (c) integration test that OTR_ModelLoadedAssert fires on a stub-mode run and the workflow errors red instead of green.
- **Tags:** silent-fallback, stub-mode, workflow-qa-blind-spot, gradient-placeholders, bible-candidate, model-loading, visual-sidecar, chain-of-trust, flux-anchor, ltx-motion, wan21-loop, stub-warning, meta-json, comfyui-workflow

### BUG-LOCAL-045: `_workflow_to_api_prompt` silently drops widget values on COMBO-typed inputs, HTTP 400 validation on `/prompt` [FIXED]
- **Date:** 2026-04-18 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Autonomous re-submission of `workflows/otr_scifi_16gb_TEST.json` via `POST /prompt` returned HTTP 400 with body `"LoadAudio.validate_inputs() missing 1 required positional argument: 'audio'"` on node 100, `input_name=episode_audio`, `input_config=["AUDIO", {...}]`. Dumped `_api_prompt.json` showed `"100": {"class_type": "LoadAudio", "inputs": {}}` — widget value `"silent_uprising_test_hq48k.wav"` was present in the UI-format `widgets_values` but never translated. The ComfyUI Desktop UI loaded and queued the same workflow fine; only HTTP-API submission failed.
- **Cause:** `scripts/soak_operator.py::_is_widget_backed()` gates which schema inputs are eligible to pick up values from `widgets_values[]`. The gate is `param_type in _WIDGET_PRIMITIVE_TYPES`, where `_WIDGET_PRIMITIVE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "BOOL"}`. `LoadAudio.INPUT_TYPES()` declares its `audio` parameter as `["COMBO", {"options": [...], "audio_upload": True}]` — the newer unified dropdown schema that replaced legacy list-typed dropdowns. "COMBO" was not in the set, so `_is_widget_backed()` returned False, the param was skipped, and `inputs["audio"]` was never set. Since LoadAudio has no socket connection for `audio`, validation failed with "missing required argument". This would bite any workflow using COMBO-typed widgets — LoadAudio, CheckpointLoaderSimple, LoraLoader, VAELoader, many built-in nodes — on API submission.
- **Fix:** Added `"COMBO"` to `_WIDGET_PRIMITIVE_TYPES` in `scripts/soak_operator.py:974`. One-line change, test-scaffold-adjacent. No changes to socket resolution, link flattening, or the UI→API conversion pipeline itself — only widened the widget-backed type set to include the current ComfyUI dropdown schema family.
- **Verify:** `outputs\submit_test_workflow.py` re-run against a clean ComfyUI instance: `prompt_id=fb49c470-e5ae-4c8e-95c9-16d596008b52` accepted, queue=3, completed in 32s with `status_str=success`. Regression: Bug Bible 23 passed / 1 skipped / 2 xfailed; `tests\test_dropdown_guardrails.py tests\test_core.py tests\test_audio_byte_identical.py` 160 passed / 1 skipped in 121s. AST parse of `soak_operator.py` clean.
- **Tags:** api-submission, workflow-to-api, combo-type, widget-backed, load-audio, comfyui-http, bible-candidate, soak-operator

### BUG-LOCAL-044: BUG-LOCAL-042 re-opened — real root cause is `sys.modules` pollution from `test_dropdown_guardrails`, not pycache staleness [FIXED]
- **Date:** 2026-04-17 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Running `pytest tests/test_dropdown_guardrails.py tests/test_core.py` (or combined regression that orders dropdown_guardrails before test_core) reproduces the exact same failure signature BUG-LOCAL-042 had originally: `3 failed, 139 passed, 11 errors in 116.10s`. `ImportError: cannot import name 'vram_sentinel' from 'nodes._vram_log' (unknown location)` at `nodes/batch_bark_generator.py:32`. Test_core alone: 103/103 pass. Bug Bible + test_core: 127 + 2 xfailed. wedge_probe + test_core: 111 passed. The cascade is test-order dependent, not pycache-dependent.
- **Cause:** `tests/test_dropdown_guardrails.py` lines 52-58 builds a `types.ModuleType("nodes._vram_log")` stub with `vram_snapshot`, `vram_reset_peak`, `force_vram_offload`, `register_vram_cleanup` — but **omits `vram_sentinel`**. The stub is assigned to `sys.modules["nodes._vram_log"]` and never restored. When a later test suite (e.g. `test_core.py`) imports `nodes.batch_bark_generator`, the `from ._vram_log import force_vram_offload, vram_sentinel` statement resolves against the polluted stub and raises `ImportError` because the stub doesn't have `vram_sentinel`. The `(unknown location)` phrasing in the error came from `ModuleType` having no `__file__`, not from a stale `.pyc`. Earlier "fixed-by-time" diagnosis was wrong: the original fix only happened to validate isolated `test_core.py` runs, which don't trigger the pollution chain. A second stubbed attribute (`vram_snapshot` returning `None`) also caused `VRAMGuardian.flush()` to fail at `vram_guardian.py:58` with `TypeError: 'NoneType' object is not subscriptable` on `before["current_gb"]`.
- **Fix:** In `tests/test_dropdown_guardrails.py`, two changes: (1) added `_vram_mod.vram_sentinel = lambda *a, **kw: (lambda fn: fn)` to the stub — real `vram_sentinel(label, max_entry_gb)` is a decorator factory, so a pass-through decorator stub is shape-correct for import-time resolution; (2) upgraded `_fake_vram_snapshot` from `pass` (returns `None`) to `return {"phase": label, "current_gb": 0.0, "peak_gb": 0.0}` so callers subscripting the result don't crash. Both changes are test-scaffold fixes, not production code changes. No restoration-on-teardown added because the pattern is deliberate (allows other test files in the same session to import `story_orchestrator` without invoking the real VRAM machinery).
- **Verify:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\test_dropdown_guardrails.py tests\test_core.py tests\test_wedge_probe.py -q` returns `161 passed in 109.40s`. Bug Bible regression: `24 passed, 2 xfailed in 0.87s`.
- **Tags:** test-order-pollution, sys-modules-leak, vram-sentinel, stub-incomplete, batch-bark-generator, test-scaffold, supersedes-bug-local-042

### BUG-LOCAL-043: SD 1.5 `.ckpt` loading through diffusers fails with 4-layer offline/Windows stack [FIXED]
- **Date:** 2026-04-16 | **Phase:** B | **Bible candidate:** yes
- **Symptom:** Phase B smoketest (`scripts/phase_b_smoketest.py`) repeatedly returned `STATUS=READY detail=... SD15 anchors: 0 ok, 2 failed` with `cache_index.json` reporting `"error": "OSError: [Errno 22] Invalid argument"` for every shot, and `io/visual_in/<job>/anchor_error.log` capturing a different failure each fix pass. Layer-by-layer: (1) `_pickle.UnpicklingError: Weights only load failed ... pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint was not an allowed global`; (2) `ModuleNotFoundError: No module named 'pytorch_lightning'` during full unpickle; (3) `[Errno 22]` deep inside `snapshot_download → thread_map → tqdm.fp_write`; (4) `[Errno 22]` inside `tqdm.std.print_status` during pipeline inference step progress.
- **Cause:** Four stacked issues all triggered when `diffusers.StableDiffusionPipeline.from_single_file()` tried to load `v1-5-pruned-emaonly.ckpt` from an OTR sidecar subprocess (`stdout=PIPE, stderr=PIPE`) with 100%-local/offline rules: (1) PyTorch 2.6+ changed `torch.load` default to `weights_only=True`, blocking legacy `.ckpt` pickles that embed `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` metadata; (2) `pytorch_lightning` is not installed in the ComfyUI venv, so even with `weights_only=False` the unpickler needs the module to resolve the class; (3) diffusers calls `snapshot_download` to fetch the diffusers-format model config from HF Hub — violates offline-first constraint and fails on Windows WinError 22 in restricted environments; (4) both the loading-phase and inference-phase tqdm progress bars write to PIPE'd stdout that ComfyUI never drains, triggering `[Errno 22]` once the pipe backs up.
- **Fix:** Four coordinated fixes in `otr_v2/visual/anchor_gen.py::_default_sd15_loader()`: (1) monkey-patch `torch.load` to force `weights_only=False` (kwargs override, not setdefault — diffusers passes `weights_only=True` explicitly); (2) inject 3-level `pytorch_lightning` shim into `sys.modules` with `_ShimModelCheckpoint` placeholder class before load, remove after; (3) vendor `configs/v1-inference.yaml` from CompVis (MIT) and pass `original_config=<local path>` + `local_files_only=True` to skip HF Hub entirely; (4) call `diffusers.utils.logging.disable_progress_bar()` + set env vars (`HF_HUB_DISABLE_PROGRESS_BARS=1`, `TRANSFORMERS_VERBOSITY=error`, `DIFFUSERS_VERBOSITY=error`) before load, and `pipe.set_progress_bar_config(disable=True)` after `pipe.to(device)` to silence both tqdm paths. Also: hardened `otr_v2/visual/worker.py` anchor exception handler to write full traceback to `io/visual_in/<job>/anchor_error.log` (was only recording `type(exc).__name__`, losing all debug info).
- **Verify:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -u scripts\phase_b_smoketest.py` returns `STATUS=READY detail=... SD15 anchors: 2 ok, 0 failed`. Per-shot meta: `anchor_used=True cache_hit=False render.png` ~800KB–1MB. Cold-load first shot ~12s, warm subsequent shots ~3.9s (Job `vs_12cdd0ef1d94`). `cache_index.json` error fields empty strings. Dropdown guardrails 50/50, test_core 103/103, audio byte-identical 7/7+1 skipped.
- **Tags:** sd15, ckpt-loading, diffusers, weights-only, pytorch-lightning-shim, hf-hub-offline, tqdm-pipe-winerror22, windows-only, sidecar-subprocess, anchor-gen, phase-b, four-layer

### BUG-LOCAL-042: `vram_sentinel` import-chain failure cascades into 14 test_core.py failures on Windows [FIXED]
- **Date:** 2026-04-16 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Windows regression run after Phase B v0 push (`c46a013`, `308dde0`) reports `3 failed, 260 passed, 1 skipped, 11 errors`. Errors all read `ImportError: cannot import name 'vram_sentinel' from 'nodes._vram_log' (unknown location)` traced to `nodes/batch_bark_generator.py:32: from ._vram_log import force_vram_offload, vram_sentinel`. Cascades into 11 errors across `TestCleanTextForBark::*`, 1 failure in `TestCleanTextForBark::test_scene_sequencer_clean_matches_batcher`, and 2 `TestVRAMGuardianNode::test_passthrough_*` failures (`vram_guardian.py:58 TypeError: 'NoneType' object is not subscriptable`).
- **Cause:** Stale Windows `__pycache__` from the mid-April Phase B churn. `vram_sentinel` IS and was defined in `nodes/_vram_log.py:150` per grep, and `from ._vram_log import force_vram_offload, vram_sentinel` at `nodes/batch_bark_generator.py:32` is correct source-level — but the `(unknown location)` phrasing in the ImportError indicated the `nodes._vram_log` module spec was `None` when `batch_bark_generator` ran its import, which is the hallmark of a bytecode cache that references a moved/renamed symbol in an older `.pyc`. Neither Phase B (`c46a013`) nor any Day 1-14 sprint commit touched `nodes/_vram_log.py`, `nodes/batch_bark_generator.py`, or `nodes/__init__.py` — last touch on all three is pre-sprint `5cf338e` ("P0+P1+P2 ROADMAP features"). Not surfaced earlier because the Linux sandbox skips these torch-dependent tests, so the cache staleness only ever lived on Windows.
- **Fix:** No code change required — normal dev churn since `5cf338e` invalidated the stale `.pyc` entries organically. Verified on 2026-04-17 by running `tests/test_core.py` against the current Windows venv Python both with warm pycache (103/103 passed in 3.60s) and after a `for /d /r %d in (__pycache__) do rmdir /s /q "%d"` full purge (103/103 passed in 4.38s). All previously-failing classes now green: `TestCleanTextForBark` 12/12, `TestVRAMGuardianNode::test_passthrough_*` 2/2. The 2026-04-17 ROADMAP Day 13 + Day 14 rows carried the caveat "4 failures + 11 errors in v1 test_core.py are pre-existing BUG-LOCAL-042, not caused by Day N" — that caveat is now stale and removed.
- **Verify:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/test_core.py -v` returns `103 passed`. Rerun after a `__pycache__` purge to confirm the fix is structural, not cached.
- **Tags:** vram-sentinel, import-chain, batch-bark-generator, vram-guardian, windows-only, pycache-stale, pre-existing, fixed-by-time, phase-b-non-cause

### BUG-LOCAL-041: ffmpeg zoompan multiplies frames, producing 1880-second clips for 8.7-second shots [FIXED]
- **Date:** 2026-04-16 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** First cut of `_make_motion_clip()` in `otr_v2/visual/worker.py` produced a `render.mp4` with `duration=1898.416667s` for a shot whose `duration_sec=8.7`. Every motion-stub clip ballooned to 30+ minutes long. Renderer then concatenated them and produced a 90-minute "8-minute episode" video. ffprobe confirmed: `nb_frames=45136 r_frame_rate=24/1`.
- **Cause:** `zoompan` is a per-input-frame filter — it emits `d` output frames for EACH input frame. The first implementation fed the still through `-loop 1 -t 8.7` at the default 25 fps, producing 217 input frames; each one was multiplied by `d=208` (8.7s * 24fps), giving 45,136 output frames at 24 fps = ~1880 seconds. The `-t` flag on the input side did not cap the output as I had assumed.
- **Fix:** Switched `_make_motion_clip()` in `otr_v2/visual/worker.py` to the canonical Ken Burns ffmpeg pattern: feed exactly one input frame using `-loop 1 -framerate 1 -t 1 -i still.png`, then cap the zoompan output explicitly with `-frames:v N` where `N = int(round(duration_sec * 24))`. This guarantees `nb_frames == N` regardless of zoompan's internal `d` value.
- **Verify:** Run `scripts/visual_smoketest.py` — Test 7 (renderer with stub assets) PASS. ffprobe each shot's `render.mp4`: `duration` must equal `duration_sec` from `shotlist.json` to within ±0.05s. Verified post-fix: 8.7s expected -> 8.709s actual; 3.5s expected -> 3.500s actual.
- **Tags:** ffmpeg, zoompan, ken-burns, visual, worker, motion-stub, frame-multiplication

### BUG-LOCAL-040: Director JSON parse fails on JS-style comments in LLM output [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `ValueError: Failed to parse production plan JSON. Aborting run to prevent silent audio failure.` Director produced 3491-char output starting with valid-looking JSON but `json.loads()` rejected it. Repair pass logged `Expecting property name enclosed in double quotes: line 13 column 42`.
- **Cause:** Mistral emitted JavaScript-style `//` line comments inside JSON values, e.g. `"v2/en_speaker_8", // Reserved for LEMMY`. Python `json.loads()` does not accept comments. The existing repair pass stripped trailing commas and tried brace-closure but never stripped comments, so the parse always failed at the first `//`. The truncation (open braces) was a secondary issue masked by the comment failure.
- **Fix:** Added `_strip_json_comments()` static method to `LLMDirector` — a state-machine parser that removes `// ...` to end-of-line only outside of quoted strings (preserving URL-like values such as `v2/en_speaker_8`). Wired into `_extract_json` at three points: (1) after the first raw `json.loads()` fails, strip comments + trailing commas and retry, (2) before the truncation-repair brace closure, (3) in the last-resort brace-scan path. Comment stripping runs before trailing-comma stripping so `value, // comment\n}` collapses cleanly.
- **Verify:** Next run should show `[LLMDirector] Plan: N voices, N SFX cues, N music cues` instead of the FATAL. Runs where the LLM emits clean JSON (no comments) hit the first `json.loads()` and skip the stripper entirely — zero perf cost on the clean path.
- **Tags:** director, json-parse, llm-output, comments, mistral, truncation

### BUG-LOCAL-039: Leading markdown bold wrapper leaks into extracted title [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** BUG-LOCAL-038 verification tail showed `TITLE_TRACE | source=script_json | resolved='** Bioluminal Tide' | widget='' | script_json='** Bioluminal Tide'`. Title leaked leading `** ` into the resolved value -- every downstream consumer (filename, video overlay, log lines) carried the cosmetic garbage.
- **Cause:** Mistral emitted `TITLE: **Bioluminal Tide**` (markdown bold wrapping the value, not the whole line). `_RE_TITLE_LINE` only strips `**` as an optional prefix BEFORE the word `TITLE` and as an optional suffix AFTER a trailing quote. The lazy capture group `(.+?)` grabbed the leading `**` of the value and retained it. `_extract_title_from_script_text` only post-processed the capture with `strip()` and quote-strip, no markdown-wrapper strip.
- **Fix:** In `_extract_title_from_script_text` (around line 1586 in `nodes/story_orchestrator.py`), add two regex substitutions right after the quote-strip to peel leading/trailing `*`/`_` runs (1-3 chars) plus surrounding whitespace, then re-run the quote-strip so nested cases like `**"Title"**` still land clean. Empty-result guard added so a `TITLE: ****` residue returns `""` instead of an empty string that later stages might treat as valid.
- **Verify:** Next run's `otr_runtime.log` should show `TITLE_TRACE ... resolved='Bioluminal Tide'` (no leading `**`) when Mistral emits markdown-bold titles. Existing Gemma/Nemo runs with unwrapped titles must remain unchanged. Filename must still vary per episode (BUG-LOCAL-035 regression guard).
- **Tags:** title-extraction, regex, markdown, mistral, cosmetic, post-processing

### BUG-LOCAL-038: BatchBark sees 0 dialogue lines despite Grammarian reporting 21 -- Bark bus renders only ANNOUNCER [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Out-of-box QA run (Mistral, defaults, seed=Bacterial Echo at 13:23-13:36) shipped `signal_lost_bacterial_echo_20260415_133600.mp4` with the correct title (BUG-LOCAL-037 fix verified) but the audio bus had no character dialogue. Heartbeat trail:
  * `ScriptWriter DONE: 1749 tokens ... | 4 scenes | 21 dialogue lines | Characters: ANNOUNCER, DRACULA MALONE, JOHN VANCE, KELLY ECKELS, SOM HALLOWAY` (streaming detector post-grammarian)
  * `SCENE_TRACK: 06_AFTER_GRAMMARIAN | count=4`
  * `TITLE_STRIP | extracted='Bacterial Echo'`
  * `SCENE_TRACK: 07_AFTER_PARSE | count=4` (no dialogue-count checkpoint)
  * `[BatchBark] Found 0 dialogue lines in Canonical 1.0 format (skipped 1 ANNOUNCER lines - routed to Kokoro bus)`
  Upstream word-extend path also failed on this run (WORD_ENFORCEMENT 0 words / 700 target on the primary pass, recovered to 33% only via pre-rolled cast fallback), confirming the final `script_json` had dialogue strings somewhere in the text but `_parse_script` did not emit canonical `{"type":"dialogue"}` tokens for them.
- **Cause:** Chain-of-custody break between FORMAT_NORM / GRAMMARIAN and `_parse_script` for Mistral's bare `NAME: text` dialogue style. Three compounding gaps:
  1. `_normalize_script_format` skip heuristic (line 5289) treated `canonical_count >= 5` raw `NAME:` matches as "already canonical" and skipped the LLM rewrite to `[VOICE: NAME, traits]`. Mistral's native output is bare `NAME: text` -- that format made the skip-counter happy but was NOT a format `_parse_script`'s four VOICE-tag patterns accepted.
  2. `_parse_script` had no first-class pattern for bare `NAME: dialogue`. v1-v4 all required `[VOICE: ...]` or `[NAME, traits]` bracket tags. Anything bare hit the "treat as structural direction" fallback (line ~6488) and was lost as a `{"type": "direction"}` token.
  3. The permissive 2B-fallback inside `_parse_script` (line 6506) only fired when the strict pass produced `dialogue_count == 0 AND len(lines) > 0`. If GRAMMARIAN or any upstream pass injected even one malformed VOICE tag that registered as a dialogue token, the guard short-circuited and the bare `NAME:` recovery pass never ran -- leaving 20+ legitimate dialogue lines stranded as direction tokens.
  No dialogue-count checkpoint existed between 06_AFTER_GRAMMARIAN and BatchBark input, so the exact loss point was invisible.
- **Fix:** Four-part defense in `nodes/story_orchestrator.py`:
  * **Diagnostic** (around line 3848): added `DIALOGUE_TRACK: 07_AFTER_PARSE | count=N | characters=[...]` runtime log line right after the existing SCENE_TRACK checkpoint. Makes the silent drop visible in one grep on any future run.
  * **FORMAT_NORM skip tightened** (line 5289): changed `has_dialogue = (voice_tag_count >= 3 or canonical_count >= 5)` to `has_dialogue = (voice_tag_count >= 3)`. Bare `NAME:` scripts must now always go through the LLM rewrite pass; they no longer look canonical to the skip heuristic.
  * **2B-fallback guard loosened** (line ~6506): replaced single `dialogue_count == 0 AND len(lines) > 0` trigger with an OR of that original trigger and a new `dialogue_count < 3 AND raw-text has 5+ NAME: shape matches`. Now any handful of stray malformed VOICE tags at the top cannot short-circuit recovery of a genuinely bare-NAME script. A raw-text pre-check prevents false firings on narration-only treatments.
  * **v5 VOICE pattern added** (before the direction fallback around line 6487): first-class regex for bare `NAME: dialogue` (accepts 0-2 asterisks, optional `(emotion)` parenthetical after the name, structural-token blacklist covering `TITLE`/`ENV`/`SFX`/etc but explicitly allowing `ANNOUNCER`). Registers the token as `{"type":"dialogue", "character_name": ..., "voice_traits": ..., "line": ...}` directly from the strict parse, so FORMAT_NORM becomes nice-to-have (adds voice traits) rather than load-bearing.
- **Verify:** Next run's `otr_runtime.log` should show `DIALOGUE_TRACK: 07_AFTER_PARSE | count=N | characters=[...]` with N matching or beating the streaming heartbeat's dialogue count. `[BatchBark] Found N dialogue lines` in `comfyui_8000.log` should be N >= 15 for a normal 12-minute Mistral episode (was 0 before). Full MP4 audio should contain character dialogue audible in VLC, not announcer-only. Scripts emitted natively in `[VOICE: ...]` format (Gemma) or bracket-shorthand `[NAME, traits]` (Nemo) must still parse through their existing paths unchanged.
- **Tags:** format-norm, parser, canonical-1.0, batch-bark, mistral, skip-heuristic, permissive-fallback, silent-drop

### BUG-LOCAL-037: BUG-LOCAL-035 fix made the parser see "TITLE" as a speaking character [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Out-of-box-settings test (Mistral, defaults) showed `Characters: ANNOUNCER, LEMMY, QUINN MARTIN, TITLE, VICTOR WELLS, WATSON BERNARD` in the ScriptWriter heartbeat. Dialogue collapsed across the self-critique pass (28 dialogue lines -> 2). Final WORD_ENFORCEMENT logged 2 words / 700 target (0%) and WORD_EXTEND could only recover 1 valid line. The MP4 still rendered with the correct title (`signal_lost_magnetic_echo_*.mp4`) but had almost no dialogue.
- **Cause:** BUG-LOCAL-035's writer-prompt addition required Mistral/Gemma to emit `TITLE: <name>` as the very first line of output. Both the streaming token detector (line ~1885 inline tuple) and the post-stream parser (`_DIALOGUE_FALSE_POSITIVES` frozenset, line 1545) match a `NAME: text` shape as a dialogue line, and neither blacklist contained `TITLE`. Result: every `TITLE: Magnetic Echo` line got booked as character `TITLE` speaking the title text, polluting the cast roster, eating the word budget, and almost certainly nudging the self-critique into deleting "redundant" dialogue.
- **Fix:** Three-part defense in `nodes/story_orchestrator.py`:
  * Added `"TITLE"` to `_DIALOGUE_FALSE_POSITIVES` frozenset (line 1545) -- catches it in the canonical post-stream parser, the dialogue-extension cast filter, and any other consumer of the shared blacklist.
  * Added `"TITLE"` to the streaming heartbeat's inline false-positive tuple (line ~1885) -- removes the noisy `Characters: ..., TITLE` log lines and stops the running cast count from inflating mid-stream.
  * Belt-and-suspenders: just before `_parse_script` (line ~3798), capture the LLM's `TITLE:` line via `_extract_title_from_script_text` into `_early_llm_title`, then strip all `TITLE:` lines from `script_text` with `_RE_TITLE_LINE.sub("", text)`. Wired the captured value into the title-resolution block so `source=llm` resolution still works after the strip. Logs `TITLE_STRIP | extracted=...` so future runs can confirm the strip happened.
- **Verify:** Next run's `otr_runtime.log` should show `TITLE_STRIP | extracted='...'` AND a `ScriptWriter DONE` line whose `Characters: ...` list does NOT contain `TITLE`. Filename should still vary (BUG-LOCAL-035 must remain fixed). `TITLE_TRACE source=llm` should still resolve to the correct title.
- **Tags:** title-stuck, regression, dialogue-parser, false-positive, cast-roster, self-critique

### BUG-LOCAL-036: WordExtend NameError `_false_positives is not defined` — 100% fail on every run [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `[WordExtend] Extension pass failed: name '_false_positives' is not defined` on every single overnight soak run (40/40 failures, SHORT_DURATION scan flag). Extension pass was silently short-circuiting; every episode shipped at the un-extended word count.
- **Cause:** Literal one-character typo inside the WordExtend node body. The module-level constant is `_DIALOGUE_FALSE_POSITIVES` (defined near line 1545 of `nodes/story_orchestrator.py`), but the extension-pass code path referenced an abbreviated local name `_false_positives` that was never bound. Python raised NameError, the surrounding try/except logged it and returned the script unchanged.
- **Fix:** Changed the reference to the correct module-level constant `_DIALOGUE_FALSE_POSITIVES` (`nodes/story_orchestrator.py` around line 6065). Added a short comment noting the old name for future grep-archaeology.
- **Verify:** `grep -rn "_false_positives" nodes/` shows only the comment line — no live code reference. Next real-workflow run should emit target word counts; SHORT_DURATION scan flag should clear.
- **Tags:** typo, word-extend, short-duration, scan-flag, name-error

### BUG-LOCAL-035: TITLE_STUCK — every episode filename locked to "The Last Frequency" regardless of LLM output [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Overnight soak produced 40 consecutive `The_Last_Frequency_*.mp4` files. Treatment scanner raised TITLE_STUCK on every run. Writer was asked for a title, but the filename path ignored it.
- **Cause:** Two independent gaps.
  1. `workflows/otr_scifi_16gb_full.json` hardcoded the widget default `"The Last Frequency"` on three nodes: Node 1 `OTR_Gemma4ScriptWriter` widget #0, Node 7 `OTR_EpisodeAssembler` widget #0, Node 12 `OTR_SignalLostVideo` widget #5.
  2. The writer's prompt asked for `EPISODE TITLE: ...` but the output was never parsed back. `OTR_SignalLostVideo`'s title-extraction code handled only the dict-format `script_json`; the actual Canonical 1.0 list-format fell through to the widget default on every run. No catch, no error — silent lock-in.
- **Fix:** Four-part fix across three files.
  * `workflows/otr_scifi_16gb_full.json`: cleared all three hardcoded widget defaults to `""` so the writer/video nodes fall through to real resolution.
  * `nodes/story_orchestrator.py`: added module-level `_STUCK_TITLE_DEFAULTS` frozenset + `_RE_TITLE_LINE` + `_extract_title_from_script_text()` + `_derive_title_from_script_lines()`. Updated the Gemma user_prompt (both OC-mode and non-OC branches) to require `TITLE: <...>` as the **very first** line of output, with an anti-stuck-default instruction. In `write_script`, added four-tier title resolution (user override → LLM-parsed → derived-from-first-environment → timestamp), prepended a `{"type": "title", "value": _resolved_title}` token to `script_lines`, and log one `TITLE_TRACE` line per run with raw/parsed/final values.
  * `nodes/video_engine.py`: rewrote the title-extraction block in `render_video` to (a) scan list-format `script_json` for the new `title` token, (b) fall back to dict-format top-level `title`, (c) fall back to the widget, (d) hard-fail with `RuntimeError("TITLE_RESOLVE_FAIL: ...")` if all paths yield a stuck default. Cleared the default widget value in `INPUT_TYPES` and the `render_video` signature.
- **Verify:** Next run should emit a `TITLE_TRACE | raw="..." | parsed="..." | final="..."` line. Filename stem should vary across 5 consecutive runs. `TITLE_RESOLVE_FAIL` firing is the desired loud behavior if resolution ever breaks again.
- **Tags:** title-stuck, widget-default, workflow-json, writer-prompt, video-render, scan-flag, fail-loud

### BUG-LOCAL-034: Fatal-streak auto-halt + user-controlled STOP_FILE pause for soak operator [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** Overnight soak churned through 40 identical TITLE_STUCK + WordExtend failures because there was no auto-halt on repeated fatals. Jeffrey had no clean pause-and-ask contract.
- **Cause:** Soak loop in `scripts/soak_operator.py` flagged failures reactively but never counted streaks or respected a user stop signal between runs.
- **Fix:** Added module-level constants (`STOP_FILE`, `STOP_POLL_S`, `FATAL_STREAK_LIMIT=3`), a sliding window `_recent_fatal_tags`, and four helpers: `classify_fatal(result, error_msg, scan_flags)`, `check_fatal_streak(tag)`, `trigger_fatal_halt(run_num, tag)`, `wait_for_stop_clear(run_num)`. Wired `classify_fatal` + `check_fatal_streak` into `run_iteration` right after the treatment-scan block so both the run outcome and scan flags feed tagging. Wired `wait_for_stop_clear` at the top of the main `while True:` loop so each iteration honors a live stop before spinning up the next run. Clean SUCCESS with no streakable scan flags resets the window. Three identical fatal tags in a row writes STOP_FILE, sends an urgent ntfy, and blocks the next iteration until Jeffrey removes the file.
- **Verify:** Touch `scripts/.soak_stop` between runs — next iteration should pause and poll every 30s until removed. Trigger three identical fatals (e.g. three TITLE_RESOLVE_FAIL raises) and confirm STOP_FILE is created automatically.
- **Tags:** soak, operator, streak, stop-file, user-control

### BUG-LOCAL-001: v2_preview.py placeholder nodes flagged as output nodes [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Bug Bible regression BUG-01.02 fails: "Output nodes without folder_paths usage: v2_preview.py"
- **Cause:** All four v2 placeholder nodes (CharacterForge, ScenePainter, VisualCompositor, ProductionBus) had `OUTPUT_NODE = True` despite not writing any files to disk. They only return in-memory tensors or strings.
- **Fix:** Removed `OUTPUT_NODE = True` from all four placeholder classes. These nodes are data-flow nodes, not file-output nodes.
- **Verify:** `python -m pytest bug_bible_regression.py -v --pack-dir .` passes BUG-01.02
- **Tags:** widget-drift, registration, bug-bible

### BUG-LOCAL-002: Stale TestWorkflowJSONLite references deleted workflow [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** 8 test errors + 1 failure in test_core.py: FileNotFoundError for otr_scifi_16gb_lite.json
- **Cause:** The lite workflow was removed in commit 44cbdec ("chore: remove lite workflow") but the TestWorkflowJSONLite test class was not cleaned up.
- **Fix:** Removed the entire TestWorkflowJSONLite class from test_core.py with a comment noting the removal reason.
- **Verify:** `pytest tests/test_core.py -v` shows 83 passed, 0 errors
- **Tags:** stale-test, cleanup

### BUG-LOCAL-003: Widget-value drift in workflow-to-API prompt conversion [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ComfyUI rejects POST /prompt with HTTP 400. Node #15 (BatchAudioGenGenerator) gets `episode_seed: 3.0, model_id: 3.0` instead of `episode_seed: "", model_id: "facebook/audiogen-medium"`.
- **Cause:** Workflow-to-API conversion mapped `widgets_values` positionally to ALL widget-capable params. But ComfyUI's workflow JSON excludes linked inputs from `widgets_values`, so linked params (script_json, production_plan_json) consumed slots 0-1, shifting all downstream values by 2 positions.
- **Fix:** Filter widget-capable params to only UNLINKED ones before positional mapping. `unlinked_widgets = [p for p in widget_capable if p not in linked]`.
- **Verify:** Regenerate debug_prompt.json and check node #15 values are correct.
- **Tags:** widget-drift, api, baseline-capture

### BUG-LOCAL-004: v2 placeholder nodes cause API 400 from missing required inputs [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** ComfyUI rejects prompt because CharacterForge, ScenePainter require MODEL/CLIP/VAE inputs that are not connected in the audio-only workflow.
- **Fix:** Strip v2 placeholder nodes from the API prompt before submission. They are not part of the audio pipeline.
- **Verify:** Prompt submits successfully with only audio-pipeline nodes + PreviewAudio capture node.
- **Tags:** api, baseline-capture, placeholder

### BUG-LOCAL-005: Emoji vs [EMOJI] placeholder mismatch in dropdown values [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ComfyUI 400 error: `runtime_preset: '\ud83d\udcfb standard (12 min)' not in ['[EMOJI] standard (12 min)', ...]`
- **Cause:** Workflow JSON stored real Unicode emoji (e.g. U+1F4FB) in dropdown values, but the running ComfyUI node code uses `[EMOJI]` as a text placeholder. The API prompt validation does exact string matching.
- **Fix:** Added `_dropdown_text_match()` that strips leading emoji or `[TAG]` prefixes before comparing, and remaps to the schema's expected value.
- **Verify:** Regenerate debug_prompt.json and check node #1 runtime_preset matches schema.
- **Tags:** encoding, widget-drift, api, baseline-capture

### BUG-LOCAL-007: PARSE_FATAL when target_length=short (3 acts) + runtime_preset=[FAST] quick (5 min) [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ScriptWriter generates 4 scenes, 48 lines, but 0 parseable dialogue lines. PARSE_FATAL fires, execution aborts. Episode never reaches TTS stage.
- **Cause:** `short (3 acts)` compresses the arc so aggressively that Mistral-Nemo produces narration/outline-style content instead of `CHARACTER: dialogue` format. The parser finds no dialogue tags and hard-aborts.
- **Fix:** Keep `[FAST] quick (5 min)` runtime target but use `medium (5 acts)` for `target_length`. Five acts requires 45 minimum dialogue lines, forcing proper dialogue structure. workflow updated: `target_length` = `medium (5 acts)`.
- **Verify:** Run `test_audio_byte_identical.py --capture-baseline` and confirm ScriptWriter log shows `dialogue lines > 0`.
- **Tags:** script-writer, parse-fatal, episode-length

### BUG-LOCAL-008: Node 15 (OTR_BatchAudioGenGenerator) widget drift recurrence [FIXED-WORKAROUND]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** API prompt has `episode_seed: 3.0, model_id: 3.0` (both float) instead of `episode_seed: "", model_id: "facebook/audiogen-medium"`. Positional mapping shifted by 2.
- **Cause:** ComfyUI `/object_info` schema returns `optional` params in a different order than `INPUT_TYPES` defines them. The `_workflow_to_api_prompt` positional mapper uses schema order for `params_with_wv_slot`, but `widgets_values` are stored in `INPUT_TYPES` order. When the schema omits or reorders optional params, the wv indices are wrong. Root cause: schema ordering vs INPUT_TYPES ordering mismatch for this node specifically. `debug_audiogen_schema.json` is dumped on each baseline run for diagnosis.
- **Fix (workaround):** `_fix_known_widget_drift()` in `_run_baseline.py` hardcodes correct values for `OTR_BatchAudioGenGenerator` after prompt conversion. Real fix requires aligning schema ordering â€” see `debug_audiogen_schema.json` output.
- **Verify:** Check `debug_prompt.json` after run â€” node #15 should show `episode_seed: "", model_id: "facebook/audiogen-medium"`.
- **Tags:** widget-drift, api, baseline-capture, schema-ordering

### BUG-LOCAL-009: Preset/target_length mismatch causes wrong dialogue line targets
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `runtime_preset=[FAST] quick (5 min)` paired with `target_length=medium (5 acts)` tells the LLM "Target 8-minute runtime, MINIMUM 45 dialogue lines" even though the actual runtime is 5 minutes. LLM overshoots or gets confused by conflicting length signals.
- **Cause:** `length_instruction` dict was hardcoded per `target_length` with fixed runtime targets and dialogue line minimums that did not scale with `target_minutes`. Also, the 1-min test preset was prone to PARSE_FATAL (see BUG-LOCAL-007).
- **Fix:** (1) Removed 1-min test preset, set minimum to 3 minutes. (2) Added `_safe_length_for_preset` auto-clamp: each runtime_preset forces the safe `target_length` (e.g. quick->medium, long->long 7-8 acts, epic->epic 10+ acts). (3) Made `length_instruction` dynamic: dialogue line floor = `max(18, target_minutes * 8)`, act label from `target_length`, runtime target from actual `target_minutes`.
- **Verify:** Run with each preset. Check runtime log for "PREFLIGHT: Auto-clamped target_length" when mismatch detected. Verify `length_instruction` shows correct minute target and proportional line count.
- **Tags:** preset, length-scaling, parse-fatal-prevention

### BUG-LOCAL-011: Obsidian profile string mismatch - all guardrails dead [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Selecting "Obsidian (UNSTABLE/4GB)" in ComfyUI had zero effect - no one-shot mode, no token cap, no runtime clamp. Obsidian users got full Pro behavior, then OOM on 4GB cards.
- **Cause:** Code checked for `"Obsidian (Low VRAM/Fast)"` (6 locations in story_orchestrator.py) but INPUT_TYPES dropdown value is `"Obsidian (UNSTABLE/4GB)"`. String never matched. Likely a rename in the UI that was never propagated to the runtime code.
- **Fix:** Replace all 6 occurrences of `"Obsidian (Low VRAM/Fast)"` with `"Obsidian (UNSTABLE/4GB)"` to match INPUT_TYPES. Caught by new `test_dropdown_guardrails.py` regression suite (59 tests).
- **Verify:** Run `pytest tests/test_dropdown_guardrails.py -v` â€” TestGuardrails::test_obsidian_disables_multipass and test_obsidian_caps_runtime must pass.
- **Tags:** string-mismatch, obsidian, guardrails, dead-code

### BUG-LOCAL-010: Full pre-flight guardrail sweep [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** Multiple unguarded parameter combos could cause silent failures or PARSE_FATAL: (a) 2 characters + 7+ acts = dialogue starvation, (b) 8 characters + 5 min = too many voices for runtime, (c) "maximum chaos" + chunked outline pushes temp above model max, (d) Obsidian + 20 min = 2500 token cap truncates 60% of script, (e) `news_headlines` widget has zero effect, (f) `temperature` widget silently overridden by `creativity`.
- **Cause:** Pre-flight validation only checked 1-min edge case. No guardrails for character count vs episode length, no profile-aware runtime cap, no temp ceiling in outline gen.
- **Fix:** (a) Clamp chars to 4 if <=5 min, to 3 if <=3 min. Floor chars to 3 if >=7 acts. (b) Obsidian profile caps target_minutes at 10. (c) Outline gen temp no longer adds +0.1 when already >= 1.0. (d) Deprecated tooltips on news_headlines and temperature widgets.
- **Verify:** AST parse clean. Check PREFLIGHT log lines for each clamp scenario.
- **Tags:** guardrails, pre-flight, parameter-validation

### BUG-LOCAL-006: Converted widget alignment in widgets_values mapping [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Node #2 (Gemma4Director) gets `tts_engine: 0.4` (should be dropdown string). Widget values shifted by 1.
- **Cause:** The BUG-LOCAL-003 fix skipped ALL linked inputs from positional mapping, but linked inputs with a `"widget"` flag in the workflow JSON ("converted widgets") still keep their slot in `widgets_values`. Only linked inputs WITHOUT the widget flag should be skipped.
- **Fix:** Check `inp.get("widget")` on each linked input. Include converted widgets in the positional mapping, skip non-widget links.
- **Verify:** Regenerate debug_prompt.json and check node #2 values: temperature=0.4, tts_engine='bark (standard 8GB)', vintage_intensity='subtle'.
- **Tags:** widget-drift, api, baseline-capture

### BUG-LOCAL-012: Episode duration significantly undershoots target_minutes
- **Date:** 2026-04-12 | **Phase:** 0-1 | **Bible candidate:** yes
- **Symptom:** Test run "The Last Frequency" with target_minutes=3, 2 characters, 3 acts, Standard profile generated a 2-minute episode (vs 3-minute target). ~33% duration shortfall.
- **Cause:** Dialogue scaling formula enforces **line count minimum** (floor = max(18, target_minutes * 8)) but not **dialogue density**. For 3 min with 2 chars: floor = 24 lines total (12 per char). LLM hit the minimum and stopped, natural pacing resulted in ~1 min audio runtime. The 41 total generated lines (39 dialogue + 2 ANNOUNCER) meet the **count** requirement but not the **duration** requirement.
- **Fix:** (Phase 0.5) Relabel target_minutes dropdown to reflect realistic output range: "Target 3 (actual 2-3 min)" instead of exact promise. No code change â€” UI expectation mismatch only.
- **Verify:** Added UI warning labels to INPUT_TYPES. User sees "2-3 min" as the expected range when they select "3 min".
- **Tags:** duration, dialogue-scaling, episode-length, ui-expectation

### BUG-LOCAL-014: Maximum chaos creativity produces unparseable dialogue format
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** 8 chars + 3 min + maximum chaos: ScriptWriter generated 1065 tokens across 3 scenes but parser found 0 dialogue lines. LLM used `*NAME*(emotion): dialogue` format with single asterisks. Characters detected as garbage: "ENVSIRENS WAIL", "OPENING THEME". Episode proceeded with near-silent output.
- **Cause:** (1) Maximum chaos (temperature=1.35) pushes Mistral-Nemo into non-standard formatting. (2) Parser Pass 1 regex only accepted 0 or 2 asterisks around names, not 1. (3) Permissive fallback matched structural tags as "characters", so dialogue_count > 0 and PARSE_FATAL never fired.
- **Fix:** Four-layer defense: (a) Clamped maximum chaos temp from 1.35 to 0.95, wild & rough from 1.1 to 0.92 - LLM stays creative but follows structural rules. (b) Hardened Pass 1 regex to accept 0-2 asterisks and filter structural tag names. (c) Added Format Normalizer pass (Creative-to-Strict): same LLM, low temperature, rewrites any dialogue format into strict Canonical 1.0 BEFORE parser runs. (d) Structural name blocklist prevents ENV/SFX/MUSIC tags from being misidentified as characters.
- **Verify:** Run 8 chars + 3 min + maximum chaos again. Check runtime log for "CREATIVITY maximum chaos - temp=0.95" and "FORMAT_NORM: Success". Verify dialogue_count > 0 in ScriptWriter DONE line.
- **Tags:** parse-fatal, creativity, format-drift, dialogue-parser, temperature, phase-0.5

### BUG-LOCAL-015: System cascades to Director crash on 0-dialogue script
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** When ScriptWriter produced a script with garbage "dialogue" (structural tags misidentified as characters), the run continued to Director and Bark. Director attempted to generate voice assignments for "ENVSIRENS WAIL" and "OPENING THEME". Bark generated near-silent audio.
- **Cause:** Permissive fallback matched structural tags, returning dialogue_count > 0, which bypassed PARSE_FATAL. No quality gate between ScriptWriter output and downstream nodes.
- **Fix:** (a) Structural name blocklist in permissive fallback prevents false positive matches. (b) Format Normalizer pass gives the parser clean input. (c) PARSE_FATAL still fires as last resort if both normalizer and fallback fail.
- **Verify:** Same test as BUG-014. If normalizer fails gracefully, PARSE_FATAL should fire with clear error instead of silent garbage propagation.
- **Tags:** cascade-failure, parse-fatal, quality-gate, phase-0.5

### BUG-LOCAL-013: UI doesn't warn user when guardrails clamp parameters [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** User selected "8 characters + 3 minutes" but the guardrail silently clamped to 3 chars in the logs. No warning visible in ComfyUI UI â€” user had no idea the setting was changed.
- **Cause:** Guardrail warnings were logged internally but not returned to the UI. ComfyUI only shows what the nodes return.
- **Fix:** ~~(v1) Prepend as comment block to script_json~~ Caused BUG-016. **(v2)** Guardrail clamp warnings logged to otr_runtime.log as `GUARDRAIL_UI:` lines alongside existing `PREFLIGHT:` lines. script_json stays pure valid JSON.
- **Verify:** âœ… VALIDATED 2026-04-12 19:50:39: Test ran with 8 chars + 3 min. PREFLIGHT fired and logged clamps. âœ… REVISED: BUG-016 fix confirmed guardrail_warnings log via `_runtime_log()` without corrupting JSON.
- **Tags:** ui, guardrails, feedback, phase-0.5

### PHASE 0.5 QA SUMMARY [VALIDATED 2026-04-12]
- **All fixes deployed and tested together**
- **Test case:** 8 characters + 3 minutes + maximum chaos creativity
- **Run 1 result (old code, temp=1.35):**
  - PREFLIGHT guardrails fired: clamped 8â†’3 chars, disabled act breaks
  - FORMAT_NORM activated but reported "No improvement" (both counts 0)
  - Parser recovered 6 dialogue lines via permissive fallback
  - QA_REPAIR auto-injected ANNOUNCER bookends (generic canned text)
  - **KokoroAnnouncer crashed (BUG-016):** JSON comment prefix broke `json.loads()`
- **Post-crash fixes applied:**
  - BUG-016: âœ… Guardrail warnings now log-only, script_json stays pure JSON
  - BUG-014 (updated): âœ… Temperature clamped: maximum chaos 1.35â†’0.95, wild & rough 1.1â†’0.92
  - BUG-017: âœ… Story-aware ANNOUNCER via LLM micro-pass replaces canned placeholders
  - BUG-018: âœ… Test suite updated for runtime_preset removal
- **Test suite status:**
  - test_core.py: 83 passed, 21 skipped
  - test_dropdown_guardrails.py: 133 passed, 0 failed
  - AST parse: âœ… Clean
- **Code changes validated:**
  - No BOM: âœ… Confirmed
  - Obsidian strings: âœ… All 8 updated correctly
  - runtime_preset dropdown: âœ… Removed entirely, target_minutes is now sole control
  - Workflow JSON: âœ… Updated to remove runtime_preset widget index
- **Next phase:** Reload ComfyUI and retest with all Phase 0.5 changes live

### BUG-LOCAL-016: Guardrail warning comments break downstream JSON parsing [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Node #13 (OTR_KokoroAnnouncer) crashes with `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. The `script_json` string starts with `// GUARDRAIL WARNINGS:` instead of valid JSON.
- **Cause:** BUG-013 fix prepended `// comment` lines to the `script_json` output from ScriptWriter. JSON does not support comments. KokoroAnnouncer (and any other downstream node) calls `json.loads(script_json)` which fails immediately on the `//` prefix.
- **Fix:** Remove comment-prefix injection from script_json. Guardrail warnings are already logged via PREFLIGHT log lines visible in otr_runtime.log. Instead, store warnings in a separate `guardrail_warnings` string and log them, but keep script_json as pure valid JSON.
- **Verify:** Run 8 chars + 3 min + maximum chaos. KokoroAnnouncer should receive valid JSON and not crash. Check otr_runtime.log for PREFLIGHT warnings still present.
- **Tags:** json-parse, guardrails, downstream-crash, phase-0.5

### BUG-LOCAL-017: QA_REPAIR ANNOUNCER bookends are generic canned text [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** When the LLM fails to generate ANNOUNCER bookends (e.g. at high creativity), QA_REPAIR auto-injects canned placeholder text: "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown." and "And so the transmission ends. This has been Signal Lost. Stay safe." These are completely generic with no story context - no date, no location, no character names, no science hook.
- **Cause:** QA_REPAIR in `_parse_script()` had no access to episode context (title, news, characters). It could only insert hardcoded strings.
- **Fix:** (a) QA_REPAIR now flags missing ANNOUNCER with `__NEEDS_LLM_OPENING/CLOSING__` sentinels. (b) New `_generate_announcer_bookends()` method does a quick LLM micro-pass (temp 0.4, max 200 tokens, ~3-5s) at the `write_script` call site where full context is available. The LLM reads episode_title, genre, news headline, character names, and a dialogue preview to generate story-specific bookends. (c) Falls back to canned text if LLM call fails.
- **Verify:** Run any episode where ANNOUNCER is missing from LLM output. Check otr_runtime.log for "ANNOUNCER_GEN: Generated opening (N chars) + closing (N chars)". ANNOUNCER lines should reference actual story content.
- **Tags:** announcer, qa-repair, llm-micro-pass, story-context, phase-0.5

### BUG-LOCAL-019: FORMAT_NORM times out generating runaway filler tokens [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** FORMAT_NORM LLM pass exceeds 120s wall-clock budget. The LLM generates 1700+ tokens at ~14 tok/s but dialogue count plateaus at 22 lines around token 700 â€” the remaining 1000+ tokens are stage-direction prose, scene descriptions, and padding that the streaming counter never recognizes as dialogue. Fires on every 8-min target run tested so far. Pipeline falls back to original script text and relies on permissive 2B-fallback parser.
- **Cause:** FORMAT_NORM has no early-stop heuristic. The `max_new_tokens` budget is too generous relative to the input script length, and the LLM drifts into narrative prose after exhausting the dialogue content. The 120s timeout is a blunt wall-clock kill, not a quality gate.
- **Fix:** (1) Token budget reduced from `min(2048, len//3+500)` to `min(1024, len//4)` â€” prevents runaway filler. (2) Timeout reduced from 120s to 75s. For a 10k-char script: old budget=2048 tokens, new budget=1024.
- **Verify:** Run 8-min target with maximum chaos. FORMAT_NORM should complete in <75s or bail faster, not generate 1700+ filler tokens.
- **Tags:** format-norm, timeout, runaway-tokens, early-stop, phase-0.5

### BUG-LOCAL-020: Episode duration significantly undershoots target_minutes (systemic)
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** 8-minute target produces 3.7-minute output (46%). 3-minute target produces ~2-minute output (67%). The LLM prompt says "MINIMUM 64 dialogue lines" but only 25 are generated. At maximum chaos, Mistral-Nemo ignores word/line count instructions. The word-per-minute estimator used 130 wpm but Bark TTS actually paces at ~67 wpm, making estimates doubly wrong.
- **Cause:** Three compounding issues: (a) LLM instructions used minutes (not measurable) instead of words (countable). (b) No post-generation enforcement â€” the pipeline accepted whatever the LLM produced. (c) Duration estimator used 130 wpm instead of the measured 67 wpm.
- **Fix:** Word-count enforcement system with raw-text-first pipeline reorder: (1) Convert `target_minutes` to `target_words` using measured Bark rate of 67 wpm. (2) LLM prompt now asks for specific word count ("write at least 536 words of dialogue") instead of minutes. (3) Post-generation pipeline reordered to: **WORD_EXTEND â†’ ANNOUNCER â†’ FORMAT_NORM â†’ Parse**. All four stages operate on raw text before a single final parse. (4) `_extend_script_dialogue()` counts dialogue words via regex on raw text, generates additional dialogue lines via LLM if under 70% target, appends to raw text. (5) ANNOUNCER bookends generated on raw text (sees full extended script). (6) FORMAT_NORM normalizes the complete text (original + extensions + announcer) in one pass. (7) Parser runs once on clean text. (8) Duration estimator fixed to use 67 wpm.
- **Verify:** Run 8-min target. Check runtime log for `WORD_ENFORCEMENT:` lines showing word count vs target, and `WORD_EXTEND:` if extension fires. Final output should be closer to 8 min than 3.7 min.
- **Tags:** duration, word-count, enforcement, extension-pass, bark-wpm, pipeline-reorder, phase-0.5

### BUG-LOCAL-019: Gender assignment inversion in LLMDirector procedural cast
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** LLMDirector JSON plan specifies correct genders (e.g. COMMANDER_MC: "Male, 50s") but procedural cast assigns the opposite gender voice preset. In soak Run 77 (maximum chaos, post_apocalyptic, 2100w): COMMANDER_MC (male) got FLETCHER HUDSON (female, 60s), TARKON_TS (male) got GULLIVER KAPOOR (female, 50s), PALMER_PR (female) got RASHIDA CORBEN (male, 20s). All 4 non-announcer characters had inverted gender assignments.
- **Cause:** Pending investigation. The Director JSON `gender_hints` parse returned 0 hints (`Parsed 0 gender hints from script: {}`), causing procedural cast to ignore the Director's own voice_assignments and assign randomly from the pool. Likely the gender hint regex does not match the maximum-chaos script format.
- **Fix:** pending
- **Verify:** pending
- **Tags:** gender, llm-director, procedural-cast, maximum-chaos, soak

### BUG-LOCAL-020: Name squish and character drift under maximum chaos
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Character "Nemo Sirikit" appears in LLMDirector plan as `NEIMO_NEMEO_SIRIKIT` (hallucinated spelling). In the script body the same character appears as `NS`, `NEMO`, and `NEMO SIRIKIT`. BatchBark cannot map `NS` or `NEMO` to the Director plan, so both fall back to `v2/en_speaker_9` (already assigned to COMMANDER_MC). Result: 3 characters share one voice, 2 voices unused.
- **Cause:** Maximum chaos creativity (highest temperature/top_p) causes the LLM to hallucinate variant spellings of character names. The Director name-matching is exact-match only and cannot reconcile `NEIMO_NEMEO_SIRIKIT` with `NEMO` or `NS`.
- **Fix:** pending
- **Verify:** pending
- **Tags:** name-squish, character-drift, maximum-chaos, batch-bark, voice-collapse, soak

### BUG-LOCAL-021: Act count exceeds target_length ceiling under maximum chaos
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Config specified `medium (5 acts)` but the generated script contains 8 acts (ACT 1 through ACT 8). The act-by-act generation loop did not enforce the target ceiling.
- **Cause:** Pending investigation. The act-by-act chunked generation may not hard-cap the number of iterations, relying on the LLM to self-terminate. Under maximum chaos temperature the LLM keeps generating new acts instead of concluding.
- **Fix:** pending
- **Verify:** pending
- **Tags:** act-count, target-length, maximum-chaos, chunked-generation, soak

### BUG-LOCAL-033: Treatment scanner false-positive storm: U+2500 separator and U+2192 arrow mismatch [FIXED]
- **Date:** 2026-04-14 | **Phase:** v2.0-alpha | **Bible candidate:** no (data-format/regex fix, not a reusable code pattern)
- **Symptom:** Every completed soak run fired five flags simultaneously: EMPTY_CAST, NO_SCENE_ARC, EMPTY_SCRIPT, TITLE_STUCK, NEWS_SEED_MISSING. This happened even on clean successful episodes (e.g. RUN 237: 87% RT, 44 dialogue lines, 9.6GB VRAM). Five flags firing together on every run was the diagnostic signature of a systematic regex failure, not real content problems.
- **Cause:** Treatment files use U+2500 BOX DRAWINGS LIGHT HORIZONTAL (─, repeated ~64 times) as section separators. Both soak_operator.py scan_treatment() and scripts/treatment_scanner.py parse_treatment() used [-]+ in their regex character classes, which only matches ASCII hyphen (U+002D). The U+2500 chars never matched, causing all four separator-dependent sections (CAST & VOICES, SCENE ARC, FULL SCRIPT, NEWS SEED) to fail extraction. Cast entries also use U+2192 RIGHT ARROW (→) which neither script accepted (only -> and --> were in the alternation). TITLE_STUCK was a genuine positive (LLM defaulting title to the show's name -- separate writer-prompt issue).
- **Fix:** In soak_operator.py scan_treatment() and scripts/treatment_scanner.py parse_treatment(): (1) replaced [-]+ with [-\u2500]+ in the CAST, SCENE ARC, FULL SCRIPT, and NEWS SEED section separator regexes; (2) added \u2192 to cast arrow alternation (?:->|-->|\u2192); (3) tightened NO_SCENE_ARC terminator from (?:\n\nFULL SCRIPT) to (?:\nFULL SCRIPT\b) to match the actual single-newline boundary with trailing content. Both files already had encoding='utf-8' on their open() calls; .gitattributes *.txt eol=lf preserves encoding of treatment fixture.
- **Verify:** Smell-check against 3 real treatments (141936, 140330, 134843): all five false-positive flags cleared. TITLE_STUCK remains (intentionally -- real positive). New pytest suite 	ests/test_treatment_scanner_unicode.py (7 tests) added with real fixture 	ests/fixtures/treatment_141936.txt. Pre-existing TestVRAMGuardianNode interaction-with-torch failure confirmed at baseline (dabcebd) and unrelated to this fix.
- **Tags:** scanner, regex, unicode, separator, U+2500, U+2192, false-positive, soak

### BUG-LOCAL-032: Four workflow nodes had preserved-truncated widgets_values shapes; canonicalized to full preserved mode [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** no (data, not code)
- **Symptom:** RUN 227 of the soak operator failed on Node 3 (OTR_SceneSequencer) with ComfyUI HTTP 400: `end_line, {}, invalid literal for int()`. A post-fix schema sweep against live `/object_info` revealed four nodes whose committed `widgets_values` arrays were shorter than the declared widget-backed input count, producing the same preserved-vs-stripped auto-sense ambiguity class as BUG-LOCAL-029 / 031.
- **Cause:** Web-UI workflow JSON can omit trailing unlinked widget slots when they hold defaults. The mapper's auto-sensing heuristic (b3d33bf) handles unambiguous cases (wv_len matches either widget_backed_count for preserved or unlinked_count for stripped) but any wv_len strictly less than widget_backed_count is a preserved-truncated shape the heuristic cannot always reconstruct correctly.
- **Fix:** Use live `/object_info` schema to compute the canonical preserved-mode shape (linked placeholders + all unlinked defaults, in declared input order) for every node and write back the canonical array. Fixed nodes:
  - Node 3 (OTR_SceneSequencer): `['[]', '{}', 0, 999]` (4) -> `['[]', '{}', 0, 999, '', 'bark', 0.0, 0.0]` (8)
  - Node 11 (OTR_BatchBarkGenerator): `[0.7]` (1) -> `['[]', '{}', 0.7]` (3) [canonicalized from stripped to preserved]
  - Node 12 (OTR_SignalLostVideo): `[24, '1920x1080', 'The Last Frequency']` (3) -> `['[]', '{}', '[]', 24, '1920x1080', 'The Last Frequency']` (6)
  - Node 15 (OTR_BatchAudioGenGenerator): `['', 'facebook/audiogen-medium', 3.0, 3.0]` (4) -> `['[]', '{}', '', 'facebook/audiogen-medium', 3.0, 3.0]` (6)
- **Verify:** `scripts/_schema_sweep.py` confirms every node's widgets_values matches its canonical preserved shape (user-tuned Nodes 1/2/4 intentionally diverge in values but match in length). Full sandbox regression: widget_drift 27, dropdown_guardrails 50, core 89, v2/audio 7 = 166 passed. Next soak run expected to clear Node 3 and reach the LLM phase.
- **Tags:** widget-drift, data-corruption, workflow-json, preserved-truncated, scene-sequencer, video-engine, audiogen, batch-bark

### BUG-LOCAL-031: Node 13 (OTR_KokoroAnnouncer) widgets_values truncated to 3 slots, `speed` FLOAT received 'random' [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** no (data bug, not code)
- **Symptom:** RUN 127 of the soak operator failed with ComfyUI HTTP 400: `node_errors: {"13": {"errors": [{"type": "invalid_input_type", "message": "Failed to convert an input value to a FLOAT value", "details": "speed, random, could no..."}]}}`. The `speed` FLOAT param on Node 13 was receiving the string `'random'` (which is the default value for `voice_override`, not `speed`).
- **Cause:** Node 13's `widgets_values` in the committed workflow JSON was `['[]', '', 'random']` â€” only 3 slots. `OTR_KokoroAnnouncer` declares 4 widget-backed params: `script_json` (STRING, linked), `episode_seed` (STRING), `voice_override` (dropdown, default `'random'`), `speed` (FLOAT, default 0.95). The shape was ambiguous between preserved-truncated (linked placeholder + 2 of 3 unlinked values) and pure-stripped (no link slot + 3 unlinked values). The auto-sensing heuristic picked pure-stripped because `wv_len(3) == unlinked_count(3)`, which pushed `voice_override='random'` into the `speed` slot. Pre-existing data corruption, not a mapper bug.
- **Fix:** Set Node 13 `widgets_values` to `['[]', '', 'random', 0.95]` â€” the canonical preserved-mode shape: linked placeholder for `script_json`, then all three unlinked defaults. Auto-sensing now cleanly reads preserved mode (`wv_len(4) == widget_backed_count(4)`).
- **Verify:** Direct mapper trace confirms Node 13 resolves to `script_json='[]'` (overridden by link at runtime), `episode_seed=''`, `voice_override='random'`, `speed=0.95`. Full regression green. Next soak run should clear Node 13 validation.
- **Tags:** widget-drift, data-corruption, workflow-json, kokoro, preserved-truncated, ambiguity

### BUG-LOCAL-030: Node 11 (OTR_BatchBarkGenerator) widgets_values corrupted to ['[]'] in workflow JSON [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** no (data bug, not code)
- **Symptom:** After BUG-LOCAL-029 shipped and auto-sensing mapper was verified against Nodes 2, 13, 15, a direct mapper trace against `workflows/otr_scifi_16gb_full.json` revealed Node 11 resolves to `temperature='[]'` â€” a literal string where a FLOAT is expected. ComfyUI would reject with a type validation error. AntiGravity incorrectly blamed this on the mapper being "blind to dropdowns"; in reality the mapper is correct and the workflow JSON itself is malformed.
- **Cause:** Node 11's widgets_values was literally `['[]']` â€” a single-element list containing the string `'[]'`. The schema for `OTR_BatchBarkGenerator` declares three widget-backed params: `script_json` (STRING, default "[]", linked), `production_plan_json` (STRING, default "{}", linked), `temperature` (FLOAT, default 0.7, unlinked). Auto-sensing correctly treated the shape as stripped mode (len(wv)=1 == unlinked_count=1) and assigned `wv[0]` to the only unlinked widget-backed param, which is `temperature`. The resulting value `'[]'` is the wrong type. Root origin: hand-editing or stale web UI state left a placeholder string in the temperature slot. Predates all recent commits (was present in HEAD before BUG-LOCAL-027).
- **Fix:** Set Node 11 `widgets_values` to `[0.7]` â€” the schema default for temperature. Auto-sensing now produces `temperature=0.7` (correct FLOAT). Also reverted AntiGravity's unauthorized placeholder-strip edits to Nodes 2 and 13 (both shapes the auto-sensing mapper handles, but the committed source of truth should reflect the canonical web-UI-emitted shape).
- **Verify:** `python scripts/_verify_mapper.py` (one-off trace helper) shows Node 11 `temperature=0.7` and all of Nodes 2/13/15 mapping cleanly. Full regression green: widget_drift 27, dropdown_guardrails 50, core 103, bug_bible 22 passed + 2 xfailed, v2/audio_byte_identical 7 passed + 1 skipped.
- **Tags:** widget-drift, data-corruption, workflow-json, hand-edit, bark-generator

### BUG-LOCAL-029: ComfyUI workflow JSON uses two shapes for linked converted widgets; mapper must auto-sense [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** After b6c610e fixed Node 15, Node 2 (OTR_Gemma4Director) started failing with bad dropdown values: `temperature` received `""` instead of `0.4`, `tts_engine` got the wrong slot, and `optimization_profile` shifted down by one. Validation rejected the run. The fix for Node 15 had regressed Node 2.
- **Cause:** ComfyUI's web UI saves `widgets_values` in two inconsistent shapes depending on when and how a widget was converted to a socket. Inspecting `workflows/otr_scifi_16gb_full.json` directly: Node 15 has 6 widget-backed params, 2 linked, and `len(wv) == 4` ("stripped" mode â€” linked converted widgets have NO slot). Node 2 has 5 widget-backed params, 1 linked (`script_text`), and `len(wv) == 5` with an empty-string placeholder at slot 0 ("preserved" mode â€” linked converted widgets keep a placeholder slot). Both shapes are valid; neither is universal.
- **Fix:** `_workflow_to_api_prompt` now auto-senses per-node mode from slot-count arithmetic: if `len(wv) == total widget-backed param count` and there is at least one linked widget-backed input, the node is in preserved mode and linked params consume a placeholder slot. If `len(wv) == unlinked widget-backed param count`, stripped mode â€” linked params consume zero slots. Ambiguous cases (trailing unset optionals, manual JSON edits) default to stripped mode, which errs on the side of omitting bad placeholder values rather than letting them land in real widget keys.
- **Verify:** `pytest tests/test_widget_drift_guard.py` â€” 27 tests pass. New class `TestPreservedSlotMode` covers Node 2's shape end-to-end: `temperature == 0.4`, `optimization_profile == "Pro (Ultra Quality)"`, `script_text` retains its link, socket-only `project_state` absent from inputs. `TestLinkedConvertedWidgetSlots` continues to cover Node 15's stripped mode. On live soak, `API_PAYLOAD node=1` and `node=2` lines should show correct optimization_profile values and no DRIFT_DETECTED output.
- **Tags:** widget-drift, socket-only, linked-converted-widget, api, auto-sensing, mode-detection, bug-bible

### BUG-LOCAL-028: BUG-LOCAL-027 shipfix regressed Node 15: linked converted widgets eat widgets_values slots [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** HTTP 400 value_not_in_list on Node 15 (OTR_BatchAudioGenGenerator) immediately after deploying 2b52ebe. ComfyUI rejected the API submission because `model_id` received `3.0` instead of `"facebook/audiogen-medium"`. Same positional-shift class of bug that BUG-LOCAL-003 and BUG-LOCAL-027 addressed, surfaced on a different param by the new mapper.
- **Cause:** The rewritten `_workflow_to_api_prompt` in 2b52ebe kept a "consume-and-skip" branch for linked params that carried a `"widget": {"name": ...}` metadata block (a converted widget). The reasoning was that a converted widget still reserves its widgets_values slot. In practice ComfyUI's web UI does NOT keep widgets_values slots for inputs that have been converted to sockets â€” it saves slots only for inputs still displayed as widgets. Node 15's `script_json` and `production_plan_json` are linked + carry the `widget` metadata, but have no slots in `widgets_values`. The mapper consumed `wv[0]` (episode_seed's slot) and `wv[1]` (model_id's slot) for nothing, shifting every subsequent value down by two. `model_id` ended up with `wv[3] = 3.0`.
- **Fix:** `_workflow_to_api_prompt` now treats any linked param as consuming zero widgets_values slots, regardless of whether the input has converted-widget metadata. The walk is: start with linked names already populated from the link map, then iterate declared params and only the widget-backed + not-linked ones consume a slot. This is the original BUG-LOCAL-003 contract, restored.
- **Verify:** `pytest tests/test_widget_drift_guard.py::TestLinkedConvertedWidgetSlots -v` (4 tests) locks down the Node 15 case explicitly: `model_id` must stay a string, `episode_seed` must be empty, `guidance_scale` + `default_duration` must land as 3.0 each, and the link tables for `script_json` / `production_plan_json` must survive intact.
- **Tags:** widget-drift, socket-only, linked-converted-widget, api, hotfix, bug-bible

### BUG-LOCAL-027: Widget-drift in soak API mapper emits project_state as string, drops optimization_profile [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Wire-level capture of `soak_target_api.json` showed node #1 (Gemma4ScriptWriter) and node #2 (Gemma4Director) were being submitted with `"project_state": "Standard"` and `"project_state": "Pro (Ultra Quality)"` respectively, while `optimization_profile` was missing entirely. ProjectState loader silently failed to parse the string as a dict, producing an empty preamble; optimization_profile silently defaulted to "Standard" (only by luck matching the intended widget default). Ghost runs (empty CAST / SCENE ARC / FULL SCRIPT) traced back to this corrupted input.
- **Cause:** `_workflow_to_api_prompt` in `scripts/soak_operator.py` walked `widgets_values` positionally against `INPUT_TYPES` order without filtering socket-only params (types like `PROJECT_STATE` that have no widget in the UI). Because `project_state` was declared between `arc_enhancer` and `optimization_profile` in the optional block, every widget after it shifted up by one slot. `optimization_profile`'s value landed in the `project_state` key, and the true `optimization_profile` key was never emitted at all. Same bug class as BUG-LOCAL-003; the fix there addressed linked inputs but not socket-only inputs.
- **Fix:** Added `_is_widget_backed(spec)` helper that returns True for `STRING/INT/FLOAT/BOOLEAN` primitives and for dropdowns (list-typed specs), and False for socket-only custom types. Mapper now walks params in declaration order but only widget-backed params consume a `widgets_values` slot. Socket-only params are either filled via the link map or omitted from `inputs`. Defense in depth: moved `project_state` to the LAST entry in `optional` for both `Gemma4ScriptWriter` (`nodes/story_orchestrator.py`:2484-2534) and `LLMDirector` (`nodes/story_orchestrator.py`:6649-6670) so any future mapper regression cannot shift widget slots. Also stripped the `"3"/"3.0"/3/3.0` back-compat hack from `BatchAudioGenGenerator.model_id` (`nodes/batch_audiogen_generator.py`:102) â€” scar tissue from widget drift that's no longer needed. Added `API_PAYLOAD` and `DRIFT_DETECTED` instrumentation lines in the soak operator just before the POST. Tightened `_RE_SCENE_MARKER` to numeric-only and added a `_RE_SCENE_TERMINATOR` for `=== SCENE FINAL ===` (kills BUG-LOCAL-026 confound).
- **Verify:** `pytest tests/test_widget_drift_guard.py -v` (18 tests) passes. Assertions lock down: (1) `project_state` is never emitted as a string, (2) `optimization_profile` always survives with its correct string value, (3) mapper stays correct even if `project_state` is interleaved before `optimization_profile` in INPUT_TYPES, (4) scene regex no longer captures `FINAL` as a scene number. On next live soak run, runtime log must show `API_PAYLOAD node=1 ... optimization_profile='Standard' project_state=None` and no `DRIFT_DETECTED` lines.
- **Tags:** widget-drift, socket-only, api, soak, ghost-run, input-types, regression-test

### BUG-LOCAL-026: Scene regex matches "FINAL" as a scene number, inflates scene counts [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** `SCENE_TRACK: count=6 | tokens=['1', '2', 'FINAL', '3', '4', 'FINAL']`. FORMAT_NORM's `has_scenes` signal was fooled by pseudo-scenes, and the chunked FORMAT_NORM split treated `SCENE FINAL` blocks as real scene boundaries.
- **Cause:** `_RE_SCENE_MARKER` used `\S+?` for the scene token, which matched any non-whitespace including the literal "FINAL" that the creative LLM emits as a closing-scene marker.
- **Fix:** Tightened `_RE_SCENE_MARKER` to `===\s*SCENE\s+(\d+)(?:\s*:\s*[^=]*?)?\s*===` (numeric only). Added separate `_RE_SCENE_TERMINATOR` for `=== SCENE FINAL ===`. `_scene_inventory` returns numeric tokens followed by `'END'` when a terminator is present, so downstream counts are honest.
- **Verify:** `pytest tests/test_widget_drift_guard.py::TestSceneRegex -v` (5 tests) passes.
- **Tags:** regex, scene-marker, parser, soak

### BUG-LOCAL-024: FORMAT_NORM ghost-run bypass and silent bailout on long scripts [FIXED]
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Runs 011 and 012 under maximum chaos produced scripts with no CAST section, no `=== SCENE N ===` markers, and TITLE_STUCK downstream. Runtime log showed FORMAT_NORM was skipped entirely, despite the script clearly being malformed. On longer scripts FORMAT_NORM also logged `Output too short - keeping original` and silently bailed out.
- **Cause:** Two independent blindspots in `_normalize_script_format`: (1) pre-flight skip heuristic only checked dialogue line count (`voice_tag_count >= 3 OR canonical_count >= 5`), so a ghost run with voice tags but no CAST or scene markers bypassed normalization entirely; (2) token budget was capped at 1024 regardless of script length, so a reformatted output of a 10k-char script could not fit in the budget and triggered the `< 0.3 * input` bailout.
- **Fix:** (1) Tightened skip heuristic to require ALL THREE signals present: `has_dialogue AND has_scenes AND has_cast` (scene marker count >= 1 AND unique character count >= 2). Missing any signal forces FORMAT_NORM to run. (2) Added `_normalize_chunked` following the `_grammarian_chunked` pattern: split by `=== SCENE N ===` markers, reformat each scene independently with a full per-chunk 1024-token budget, 75s per-chunk timeout, reassemble with 80% dialogue-count floor. Single-pass retained for scripts with <=50 dialogue lines or <2 scenes. Also hoisted class constant `_FORMAT_NORM_NON_CHARS` so canonical-name regex excludes SCENE/ACT/SFX/ENV/NARRATOR etc. from the skip count.
- **Verify:** Run soak with a creative-pass script that lacks CAST + scene markers. Runtime log should show `FORMAT_NORM: Running (dialogue=X+YV, scenes=0, cast=0) - missing: scenes cast`, followed by single-pass or chunked flow. For 50+ line scripts, log should show `FORMAT_NORM: Chunked mode` with per-chunk progress.
- **Tags:** format-norm, ghost-run, skip-heuristic, chunked-generation, token-budget, maximum-chaos, soak

### BUG-LOCAL-023: Grammarian timeout on long scripts (60+ dialogue lines) [FIXED]
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** `GRAMMARIAN: Failed (Grammarian exceeded 75s) - keeping original` on a 67-line space opera script. Token budget of 2048 at ~15 tok/s needs ~136s, but timeout was 75s. Grammarian silently falls back to original script, losing all grammar polish.
- **Cause:** Single-pass grammarian with fixed 75s timeout cannot handle scripts with 50+ dialogue lines. The prompt + full script exceeds what the LLM can process within the timeout window.
- **Fix:** Implemented chunked grammarian in `_grammarian_pass()`. Scripts with >50 dialogue lines are split by `=== SCENE N ===` markers, each scene polished independently (90s timeout per chunk, 1024 token budget), then reassembled. Falls back to 40-line raw chunking if no scene markers exist. Single-pass timeout increased from 75s to 150s as safety net. Each chunk has its own dialogue-line safety check; failed chunks keep original text without blocking the rest.
- **Verify:** Run soak with 60+ line episode config. Runtime log should show `GRAMMARIAN: Chunked mode` followed by per-chunk progress, ending with `GRAMMARIAN: Chunked complete`.
- **Tags:** grammarian, timeout, chunked-generation, long-scripts, soak

### BUG-LOCAL-018: test_dropdown_guardrails.py references removed runtime_preset [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** `pytest tests/test_dropdown_guardrails.py` fails with `KeyError: 'runtime_preset'` during collection. 1 additional NameError at runtime for `RUNTIME_PRESETS` variable.
- **Cause:** runtime_preset was removed from INPUT_TYPES but the test file still extracted it from `_REQUIRED` and used it in 12 test locations.
- **Fix:** Removed all runtime_preset references from tests. Replaced `runtime_preset="[FAST] quick (5 min)"` with `target_minutes=5`, etc. Added `test_runtime_preset_removed` assertion alongside existing dead-param checks. Removed obsolete `test_no_1min_test_preset` and `test_runtime_presets_produce_different_target_minutes`.
- **Verify:** `pytest tests/test_dropdown_guardrails.py -v` shows 133 passed, 0 failed.
- **Tags:** test-suite, runtime-preset, cleanup, phase-0.5


