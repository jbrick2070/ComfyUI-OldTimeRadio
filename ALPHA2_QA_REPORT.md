# Alpha 2.0 QA Report — Active Issues

**Project:** ComfyUI-OldTimeRadio v2.0 Visual Drama Engine
**Active file:** `nodes/v2_preview.py` (1096 lines)
**Date:** 2026-04-11
**Hardware target:** RTX 5080 / 16 GB VRAM / 64 GB RAM (Legion Pro 7i Gen 10)

---

## Critical Blocker — ComfyUI Restart Required

All v2 node fixes in `v2_preview.py` are committed to `v2.0-visual-engine` branch but are
**not yet active** because ComfyUI loaded the old module at startup. Python does not hot-reload
custom nodes. After restarting ComfyUI, all fixes below will activate.

**After restart, queue via `otr_q3.bat` on the Desktop.** The workflow JSON is pre-loaded with:
- Custom premise (real content, bypasses RSS)  
- Cache-busting seeds on CharacterForge and ScenePainter
- `feedparser` now installed in the venv

---

## Active Bugs

### BUG-A1 — ProductionBus: full-length episode video

**Priority:** P0 — core deliverable
**Status:** Fix committed, needs live validation after ComfyUI restart

**Root cause chain:**
1. `feedparser` missing → ScriptWriter RSS fallback → empty script (0 lines, 0 cast)
2. Director gets empty script → generates minimal `visual_plan` (1 default scene)
3. ScenePainter → 1 background → VisualCompositor → 1 frame
4. keyframes mode `render_fps = max(1, ...)` → 1 frame at 1fps → 1-second video instead of full episode

**Fixes applied (2026-04-11):**
- `feedparser` installed in venv — ScriptWriter will now fetch real RSS headlines
- Workflow custom_premise added — bypasses RSS entirely, guarantees real content
- Workflow seeds changed (CharacterForge: 42→1337, ScenePainter: 42→1337) — busts cache
- ProductionBus keyframes mode rewritten to use FFmpeg concat demuxer: each scene frame
  gets `duration = audio_duration_s / num_scenes`, so output is exactly `audio_duration_s` long
  regardless of how many scenes are in the plan

**Expected after restart + re-run:**
- ScriptWriter generates real 3-act script from custom_premise
- Director generates `visual_plan` with 3+ characters and 3+ scenes
- CharacterForge generates portrait per character
- ScenePainter generates background per scene
- VisualCompositor composites N frames (N = scene count)
- ProductionBus produces full-length MP4 matching audio duration

---

### BUG-A2 — v2 nodes invisible to otr_runtime.log (diagnostic gap)

**Priority:** P1
**Status:** Fix committed, activates after ComfyUI restart

`_runtime_log()` helper added to `v2_preview.py`. All four v2 nodes instrumented:
- CharacterForge: logs plan keys, character names, scene count; dumps `otr_v2_debug_plan.json`
- ScenePainter: logs scene count
- VisualCompositor: logs frame count on start/done
- ProductionBus: logs `episode_audio=PRESENT/NONE`, WAV save result, FFmpeg exit code and file size

After restart, all v2 activity will appear in `otr_runtime.log` with `[v2]` prefix.

---

### BUG-A3 — Script parser rejects critique-revised dialogue format

**Priority:** P0 — blocks full pipeline run
**Status:** Fix committed, needs live validation after ComfyUI restart

**Root cause:**
The critique/revision LLM rewrites dialogue tags from canonical `[VOICE: NAME, traits]` to
shorthand `[NAME, traits] "dialogue"` on the same line. The parser had no inline-shorthand
pattern — only a tag-only shorthand (v4) that expects dialogue on the NEXT line. Result:
`ValueError: Script parser produced 0 dialogue lines from 3157-char input.`

**Fix applied (2026-04-11):**
Added v4a `voice_shorthand_inline_pat` to `_parse_script()` in `story_orchestrator.py`.
Pattern: `^\[([A-Z][A-Z0-9_ ]{1,20}),\s*(.+?)\]\s*(.+)$` — matches `[NAME, traits] dialogue`
inline, with the same structural-tag exclusion list and malformed-name fallback as v1-v3.
Inserted before the existing v4b tag-only shorthand so inline takes priority.

**Expected after restart + re-run:**
Script parser recognizes all dialogue lines from the revised script. Pipeline proceeds past
ScriptWriter into Director, audio gen, and visual assembly.

---

### BUG-A4 — Node 15 model_id receives 3.0 instead of model name

**Priority:** P2 — low risk (3.0 is in combo list, passes validation)
**Status:** Open

`BatchAudioGenGenerator` (node 15) `widgets_values = ["", "facebook/audiogen-medium", 3.0, 3.0]`
has no placeholder slots for its connected STRING inputs (it was created with connections already
wired). The mapper assigns `model_id = 3.0` and `episode_seed = 3.0`. Correct value for model_id
should be `"facebook/audiogen-medium"`.

Fix requires count-based detection in `otr_queue.py` to distinguish nodes with/without placeholder
slots. Not yet implemented. Does not block validation but may cause unexpected behavior at runtime
if the model combo list changes.

---

## Open Perplexities

**P1 — SceneSequencer vs v2 visual pipeline relationship unclear**
Node 3 (OTR_SceneSequencer) receives AUDIO from multiple sources plus `production_plan_json`
from the Director. Its relationship to the v2 visual nodes (CharacterForge, ScenePainter, etc.)
which also read `production_plan_json` directly is unclear. SceneSequencer may be a legacy
v1.5 artifact that coexists with v2 nodes in the same workflow.

**P2 — v1.5 SignalLostVideo (node 12) produces empty video when cast is empty**
During the test run with empty script, node 12 produced a 191 MB video. This means the v1.5
pipeline is resilient to empty scripts. Understanding why will help design a similar resilience
for v2 nodes.

---

## What Remains Before Tagging Alpha 2.0

**Immediate (after ComfyUI restart):**
1. Restart ComfyUI
2. Run `otr_q3.bat` from Desktop
3. Tail `otr_runtime.log` for `[v2]` entries
4. Read `otr_v2_debug_plan.json` to confirm Director visual_plan has real scenes

**Live T-Tests (RTX 5080):**

| # | Test | Pass criteria | Status |
|---|------|---------------|--------|
| T1 | Audio-only smoke | WAV written, VRAM <8 GB | Not run |
| T2 | 720p / 3 min, keyframes | Full-length MP4 with audio, VRAM <12 GB | Not run |
| T3 | 720p / 10 min stress | No OOM, preview_tmp cleared, VRAM <14 GB | Not run |
| T4 | 1080p dual-checkpoint | vram_power_wash=True, peak VRAM <=14.5 GB | Not run |
| T5 | Safety spot-check | 10 bad prompts blocked, run completes | Passed (sandbox) |
| T6 | Crash-safety | kill -9 mid-render, preview_tmp empty on restart | Not run |
| T7 | Filename fuzz | Weird output_name writes safely | Passed (sandbox) |

After T1-T7 green: run 12.02 regression checklist, tag Alpha 2.0 RC1.

---

*Active issues only. No change log. Last updated 2026-04-11.*
