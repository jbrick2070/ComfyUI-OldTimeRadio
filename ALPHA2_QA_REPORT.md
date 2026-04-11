# Alpha 2.0 QA Report — Round 7 Final (with Addendum)

**Project:** ComfyUI-OldTimeRadio v2.0 Visual Drama Engine
**File under test:** `nodes/v2_preview.py` (989 lines, was 835)
**Date:** 2026-04-11
**QA performed by:** Claude Opus (Cowork session, sandbox)
**Hardware target:** RTX 5080 / 16 GB VRAM / 64 GB RAM (Legion Pro 7i Gen 10)
**Reviewer:** 17

---

## Executive Summary

All 9 bugs from the Round 7 Action Plan have been addressed. 6 code fixes applied, 1 dead-code deletion, 2 closed as not-reproducible. A new content safety module was created. Additionally, all 6 required edits from the V2_PREVIEW_ADDENDUM (Reviewer 17 / Gemini Deep Research) have been implemented, plus 2 nice-to-have improvements. A 57-test suite (unit + integration) passes in 0.18s. Both workflow JSONs were updated for widget compatibility. Bug Bible cross-check caught and fixed a widget-value positional mismatch that would have silently broken workflow loads.

**Live testing on the RTX 5080 has not yet been performed.** All sandbox-verifiable work is complete.

---

## Bug Disposition

| Bug | Priority | Status | Summary |
|-----|----------|--------|---------|
| BUG-002 | P0 | Fixed | `comfy.sample.sample()` call converted from fragile positional args to keyword args. Matches canonical KSampler pattern. |
| BUG-007 | P0 | Fixed | Temp frame cleanup wrapped in `try/finally`. Moved to project-local `preview_tmp/`. Added 18K frame hard cap. New `preview_mode` input (none/keyframes/full, default keyframes). |
| BUG-012 | P0 | Fixed | New `nodes/safety_filter.py` with regex denylist. `classify_prompt()` API returns allow/flag/block. Hooked into `_generate_image()` before any GPU work. 10/10 bad prompts blocked in acceptance test. |
| BUG-001 | P1 | Fixed | `_encode_prompt()` now shallow-copies the dict from `clip.encode_from_tokens()` before `.pop("cond")`. Prevents shared-dict mutation across back-to-back encodes. |
| BUG-010 | P1 | Fixed | New `vram_power_wash` BOOLEAN on CharacterForge (default True). Calls `mm.unload_all_models()` + `mm.soft_empty_cache()` at visual phase entry to reclaim audio-phase VRAM. |
| BUG-006 | P1 | Fixed | New `_safe_name()` helper strips `\/:*?"<>|` and control chars, caps at 80 chars. Applied at all `output_name` interpolation sites (temp WAV + final MP4). |
| BUG-005 | P2 | Fixed | Deleted 4-line dead scanline loop (`draw.line(fill=None, width=0)`). Real scanline effect at lines 626+ untouched. |
| BUG-003 | P2 | Closed | Grepped all node files for `error_image`/`error_frame` — not found. No dedicated error-image generator exists. Closed as not-reproducible. |
| BUG-004 | P2 | Closed | No Flux-specific stacking loop in `v2_preview.py`. Encoder is generic. Closed per action plan. |

---

## New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `nodes/safety_filter.py` | Content safety regex denylist + classify_prompt() API | 141 |
| `tests/unit/test_encode_prompt.py` | BUG-001 regression tests (mock CLIP, no GPU) | 86 |
| `tests/unit/test_safe_name.py` | BUG-006 regression tests (filename sanitization) | 76 |
| `tests/unit/test_safety_filter.py` | BUG-012 regression tests (block/flag/allow) | 131 |
| `tests/integration/test_stress_harness.py` | REQ-4: OOM/cleanup/adversarial stress tests | 170 |
| `tests/safety/bad_prompts.txt` | 10 known-bad prompts for safety acceptance | 11 |

---

## Test Results

```
57 passed in 0.18s

  unit/test_encode_prompt.py          4 tests  PASSED
  unit/test_safe_name.py             16 tests  PASSED
  unit/test_safety_filter.py         25 tests  PASSED
  integration/test_stress_harness.py 12 tests  PASSED
```

No GPU, no ComfyUI, no torch dependency (except numpy for placeholder test). Run with `python -m pytest tests/ -v`.

---

## Bug Bible Cross-Check

Cross-referenced all changes against `BUG_BIBLE.yaml` (96 entries, 12 phases). **No Bug Bible updates made** — read-only until confirmed in prod.

**Found and fixed:** Bug Bible 04.01/04.02 — both workflow JSONs had stale `widgets_values` after we added `preview_mode` (ProductionBus) and `vram_power_wash` (CharacterForge). Appended default values to correct positions in both `otr_scifi_16gb_full.json` and `otr_scifi_16gb_test.json`.

**Pre-existing flags (not regressions):**
- 01.02/01.03: **Now fixed by REQ-1.** ProductionBus uses `folder_paths.get_output_directory()` with ImportError fallback.
- 09.02: FFmpeg subprocess uses `capture_output=True`. Low risk for our use case (frame files, not piped video). Monitor during T3 stress test.

---

## V2_PREVIEW_ADDENDUM (Reviewer 17 / Gemini Deep Research)

All 6 required edits implemented. 2 nice-to-haves also completed.

| Edit | Description | Status |
|------|-------------|--------|
| REQ-1 | Output path via `folder_paths.get_output_directory()` with ImportError fallback | Done |
| REQ-2 | `output_subdir` optional STRING input on ProductionBus | Done |
| REQ-3 | `debug_vram_snapshots` BOOLEAN on CharacterForge, ScenePainter, ProductionBus (gates all `_vram_snapshot()` calls) | Done |
| REQ-4 | Stress harness: `tests/integration/test_stress_harness.py` (12 tests: orphan cleanup, crash-safety, MAX_FRAMES, adversarial filenames, QA toggles, QA_METRICS, placeholder frames) | Done |
| REQ-5 | Startup janitor `_cleanup_orphaned_preview_tmp()` at module load (kill -9 recovery) | Done |
| REQ-6 | QA toggles exposed as node inputs (preview_mode, encoding_profile, debug_vram_snapshots) | Done |
| NICE | `encoding_profile` combo: preview/balanced/quality with `_ENC_PROFILES` FFmpeg preset+crf lookup | Done |
| NICE | `QA_METRICS` JSON line in bus_log (greppable for CI) | Done |

**Not implemented (offered as future work):**
- Explicit timeline duration fields from script_json/production_plan_json (currently uses heuristic)
- RTX 5080 validation envelope note for fps/width/height limits

---

## What Remains Before Tagging Alpha 2.0

All of these require Jeffrey's RTX 5080 rig:

| # | Test | Pass criteria | Status |
|---|------|---------------|--------|
| T1 | Audio-only smoke | WAV written, VRAM <8 GB | Not run |
| T2 | 720p / 1 min, keyframes | Completes <5 min, MP4 plays, VRAM <12 GB | Not run |
| T3 | 720p / 10 min stress | No OOM, preview_tmp/ cleared, VRAM <14 GB | Not run |
| T4 | 1080p dual-checkpoint | vram_power_wash=True, peak VRAM ≤14.5 GB | Not run |
| T5 | Safety spot-check | 10 bad prompts blocked, run completes | Passed (sandbox) |
| T6 | Crash-safety | kill -9 mid-render, preview_tmp/ empty on next launch | Not run |
| T7 | Filename fuzz | Weird output_name writes safely | Passed (sandbox) |

After T1-T7 green: fill in ComfyUI commit SHA in `requirements.txt`, run the 12.02 regression checklist, tag Alpha 2.0 RC1.

---

## Rollback

One-step rollback to pre-Round-7 state:
```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
copy nodes\v2_preview.py.backup_round6 nodes\v2_preview.py
```

New files (`safety_filter.py`, `tests/`) can be deleted independently without affecting the rollback.

---

*End of QA Report. Generated 2026-04-11. All sandbox-verifiable work complete. Awaiting live validation on RTX 5080.*
