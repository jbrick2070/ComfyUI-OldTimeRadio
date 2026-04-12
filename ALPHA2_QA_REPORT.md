# Alpha 2.0 QA Report — Active Issues

**Project:** ComfyUI-OldTimeRadio v2.0 Visual Drama Engine
**File under active work:** `nodes/v2_preview.py`
**Date:** 2026-04-11
**Hardware target:** RTX 5080 / 16 GB VRAM / 64 GB RAM (Legion Pro 7i Gen 10)

---

## Active Bugs

### BUG-A1 — ProductionBus: keyframes mode produced 1-second video instead of full-length episode

**Priority:** P0 — breaks deliverable entirely
**Status:** Fix applied, awaiting live validation

**Root cause:** In keyframes mode, `render_fps` was clamped to `max(1, ...)`, so with 1 scene and 202s
audio the video came out as 1 frame at 1 fps = 1 second. The `-shortest` flag then cut to the shorter
stream, dropping the audio entirely.

**Fix applied (2026-04-11):** Replaced the broken `render_fps` path with FFmpeg concat demuxer. Each
scene frame is now assigned an explicit `duration = audio_duration_s / num_scenes` in a `concat.txt`
file. FFmpeg reads this as a timed slideshow, so the output is exactly `audio_duration_s` long regardless
of scene count. Even 1 scene = full 202s episode.

**Validation needed (RTX 5080):**
- T2: 720p / 1 min, keyframes — duration matches audio, MP4 plays
- T3: 720p / 10 min stress — full episode length, no OOM

---

### BUG-A2 — v2 nodes invisible to otr_runtime.log (diagnostic gap)

**Priority:** P1 — not a user-facing failure, but masks all other bugs
**Status:** Fix applied, awaiting live validation

**Root cause:** v2 nodes log to Python `logging` (ComfyUI stdout), not to `otr_runtime.log`. All
CharacterForge, ScenePainter, VisualCompositor, and ProductionBus failures were silent from the outside.

**Fix applied (2026-04-11):** Added `_runtime_log()` helper to `v2_preview.py` (mirrors v1.5 pattern).
Instrumented all four nodes with checkpoints: plan keys, scene/character counts, audio status (present /
duration / WAV save result), FFmpeg return code, output file size. Also dumps Director `production_plan_json`
to `otr_v2_debug_plan.json` at CharacterForge entry for offline inspection.

---

### BUG-A3 — Unknown: Director visual_plan scene count (1 scene observed)

**Priority:** P1 — directly limits visual quality
**Status:** Under investigation — awaiting first instrumented run

**Symptom:** Previous run produced only 1 composited frame. If the Director LLM is only generating
1 scene in `visual_plan.scenes`, the entire visual pipeline produces 1 frame regardless.

**Possible causes:**
1. `OTR_Gemma4Director` is outputting a `visual_plan` with only 1 `scenes` entry
2. `OTR_Gemma4Director` output JSON is missing `visual_plan` entirely (ScenePainter falls back to
   placeholder 1-frame dummy)
3. `OTR_SceneSequencer` (node 3) is not forwarding `production_plan_json` correctly

**Diagnostic in place:** `_runtime_log` now reports scene count and plan keys at CharacterForge and
ScenePainter entry. `otr_v2_debug_plan.json` will be written to repo root on next run.

**To resolve:** After next run, read `otr_v2_debug_plan.json` and `otr_runtime.log` to see exactly
what the Director produced.

---

### BUG-A4 — Unknown: episode_audio None or missing at ProductionBus

**Priority:** P1 — no audio in output video
**Status:** Under investigation — awaiting first instrumented run

**Symptom:** Previous run produced MP4 with no audio stream. Node 7 (EpisodeAssembler) is connected
to ProductionBus via link 47, and otr_queue.py correctly maps `episode_audio = ["7", 0]` in API format.

**Possible causes:**
1. EpisodeAssembler is failing or returning a malformed AUDIO dict
2. torchaudio.save is failing silently (exception path now fixed to clear `audio_path`)
3. The waveform tensor is zero-length or shape is unexpected

**Diagnostic in place:** `_runtime_log` reports `episode_audio=PRESENT/NONE` at ProductionBus entry,
then logs WAV save success (with shape and duration) or exception.

---

## Open Perplexities

**P1 — Why does SceneSequencer receive audio from multiple sources?**
Node 3 (OTR_SceneSequencer) receives AUDIO from nodes 4 (SFX), 13 (KokoroAnnouncer), 15
(BatchAudioGenGenerator), and 20 (KokoroAnnouncer again). But it also receives `production_plan_json`
via STRING link. The relationship between the sequencer's multi-source audio mixing and the Director's
visual plan is unclear — SceneSequencer may be the correct orchestration point or it may be a legacy
artifact from v1.5 that conflicts with v2 flow.

**P2 — Node 15 model_id receives value 3.0 instead of model name string**
BatchAudioGenGenerator (node 15) has `widgets_values = ["", "facebook/audiogen-medium", 3.0, 3.0]`
with no placeholder slots for its connected STRING inputs. The current mapper assigns `model_id = 3.0`
(the value at index 2), which happens to pass validation only because `3.0` is literally in the combo
list. The correct value should be `"facebook/audiogen-medium"`. This needs a count-based detection fix
to distinguish nodes that were created with connections already wired (no placeholder slots) vs nodes
where connections were added after creation (placeholder slots present).

---

## What Remains Before Tagging Alpha 2.0

All require RTX 5080 live run with instrumented build:

| # | Test | Pass criteria | Status |
|---|------|---------------|--------|
| T1 | Audio-only smoke | WAV written, VRAM <8 GB | Not run |
| T2 | 720p / 1 min, keyframes | Full-length MP4 with audio, VRAM <12 GB | Not run |
| T3 | 720p / 10 min stress | No OOM, preview_tmp/ cleared, VRAM <14 GB | Not run |
| T4 | 1080p dual-checkpoint | vram_power_wash=True, peak VRAM <=14.5 GB | Not run |
| T5 | Safety spot-check | 10 bad prompts blocked, run completes | Passed (sandbox) |
| T6 | Crash-safety | kill -9 mid-render, preview_tmp/ empty on restart | Not run |
| T7 | Filename fuzz | Weird output_name writes safely | Passed (sandbox) |

After T1-T7 green: run 12.02 regression checklist, tag Alpha 2.0 RC1.

---

*Active issues only. No change log. Last updated 2026-04-11.*
