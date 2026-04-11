# Alpha 2.0 Progress Log

**Started:** 2026-04-11
**Agent:** Claude Opus (Cowork session)
**Source file:** `nodes/v2_preview.py` (835 lines, confirmed)
**Backup:** `nodes/v2_preview.py.backup_round6` (835 lines, confirmed)

---

## Log

### 2026-04-11 — Session start
- Read `alpha2actionplan.md` (Round 7, 235 lines).
- Read `v2_preview.py` — confirmed 835 lines, matches action plan expectations.
- Created `v2_preview.py.backup_round6` — verified 835 lines.
- Created this progress log.
- Beginning BUG-002 (P0).

### 2026-04-11 — BUG-002 (P0): comfy.sample positional-args fragility
- **Status:** APPLIED, awaiting Jeffrey's live test.
- **What changed:** `_generate_image()` signature now accepts `sampler_name` and `scheduler` as keyword params (defaults: `"euler"`, `"normal"`). The `comfy.sample.sample()` call at line 120 now passes `sampler_name`, `scheduler`, `positive`, `negative`, and `latent_image` as keyword args instead of positional args. This matches the canonical `common_ksampler` pattern used by ComfyUI's built-in KSampler node.
- **Lines affected:** 89-91 (function signature), 120-129 (sample call).
- **File grew:** 835 -> 840 lines (kwargs formatting).
- **AST parse:** OK.
- **No other files touched.**
- **Acceptance test needed:** 720p 20-step Euler/Normal run producing a valid latent. This requires a live ComfyUI run on the 5080 — cannot be verified in the sandbox.
- **Waiting for Jeffrey's "go" before starting BUG-007.**

### 2026-04-11 — BUG-007 (P0): Temp frame cleanup crash-safety + preview cap
- **Status:** APPLIED, awaiting Jeffrey's live test.
- **What changed (4 parts):**
  1. **Crash-safe cleanup:** The entire frame-write + FFmpeg block is now inside `try/finally`. `shutil.rmtree(temp_dir)` runs even on crash, OOM, or kill. `shutil` moved to top-level import.
  2. **Project-local temp dir:** Frames now write to `output/preview_tmp/otr_v2_*` instead of system `%TEMP%`. Leftover dirs are visible and obvious.
  3. **MAX_FRAMES hard cap:** 18,000 frames (~12 min at 24 fps). Exceeding it raises `RuntimeError` with a clear message pointing to keyframes mode.
  4. **preview_mode input:** New required combo on ProductionBus: `none` (skip video entirely), `keyframes` (1 frame/scene, fast preset), `full` (current behavior, medium preset). Default is `keyframes`.
- **Keyframes mode:** Writes 1 PNG per scene, uses FFmpeg `-preset fast`. Estimated ~95% disk write reduction on long runs.
- **Section 8 flag:** Asked Jeffrey to confirm keyframes = 1 frame/scene (vs 1 frame/second). Implemented per-scene as the action plan specified.
- **Lines affected:** 22 (shutil import), 671-674 (INPUT_TYPES), 691 (assemble sig), 753-843 (entire render block).
- **File grew:** 841 -> 868 lines.
- **AST parse:** OK.
- **Acceptance tests needed (live):**
  - Kill process mid-render -> `preview_tmp/` empty within one run cycle.
  - 15-min stress run at 1080p stays under 5 GB total disk.
- **Waiting for Jeffrey's "go" before starting BUG-012.**

### 2026-04-11 — BUG-012 (P0): Content-safety filter gap
- **Status:** APPLIED + acceptance test PASSED in sandbox.
- **What changed:**
  1. **New file `nodes/safety_filter.py`** (141 lines): keyword/regex denylist with `classify_prompt(text) -> {allow, flag, block}` interface. Pre-compiled regex patterns for performance. No model load. Ready for v2.1 drop-in replacement with CLIP-based classifier.
  2. **Hook in `v2_preview.py`** (line ~99): `_generate_image()` now calls `classify_prompt(prompt)` before any GPU work. On `block`: returns a dim gray placeholder frame (0.15 intensity), logs the reason, no sampler call. On `flag`: logs a warning and proceeds. On `allow`: unchanged behavior.
  3. **Test file `tests/safety/bad_prompts.txt`**: 10 known-bad prompts covering violence, explicit content, hate speech, self-harm, weapons, child safety.
- **Acceptance test:** All 10 bad prompts blocked, clean prompt ("A cozy 1950s radio studio with warm lighting") allowed. PASSED.
- **Lines affected:** v2_preview.py 99-108 (safety check hook). New file safety_filter.py.
- **File sizes:** v2_preview.py = 878 lines, safety_filter.py = 141 lines.
- **AST parse:** Both files OK.
- **All P0 bugs now applied. Waiting for Jeffrey's "go" to start P1 (BUG-001).**

### 2026-04-11 — BUG-001 (P1): _encode_prompt shared-dict mutation
- **Status:** APPLIED.
- **What changed:** Added `output = dict(output)` shallow copy on line 85, before `output.pop("cond")`. This prevents mutating the dict returned by `clip.encode_from_tokens()`, which ComfyUI may cache or share across calls. Without this, back-to-back encodes on the same CLIP instance could corrupt each other's conditioning data.
- **Lines affected:** Line 85 (1 line added).
- **File size:** 879 lines.
- **AST parse:** OK.
- **Acceptance:** Full acceptance requires two back-to-back encodes with different prompts on the same CLIP instance, asserting both conditionings differ. Needs live ComfyUI.
- **Waiting for Jeffrey's "go" before starting BUG-010.**

### 2026-04-11 — BUG-010 (P1): Dual-checkpoint VRAM power-wash
- **Status:** APPLIED, awaiting live VRAM measurement.
- **What changed:** Added `vram_power_wash` BOOLEAN input to CharacterForge (default `True`). When enabled, calls `mm.unload_all_models()` + `mm.soft_empty_cache()` + `_flush_vram()` at the very start of `forge()`, before any visual work. This reclaims all VRAM held by audio-phase models (Bark TTS, AudioGen, etc.) before the video checkpoint loads.
- **Section 8 Q2 note:** Proceeded with `unload_all_models()` because audio and video are distinct model objects loaded by separate ComfyUI nodes. The video model gets loaded fresh when ComfyUI evaluates CharacterForge's `model` input. Safe to unload everything.
- **Section 8 Q3 resolved:** Jeffrey confirmed 1 frame/scene for keyframes mode.
- **Lines affected:** 246-254 (INPUT_TYPES optional block), 257-259 (forge signature), 261-267 (power-wash block).
- **File size:** 892 lines.
- **AST parse:** OK.
- **Acceptance:** 1080p stress test peaks at or below 14.5 GB VRAM (measured with `nvidia-smi dmon -s u`). Needs live run.
- **Waiting for Jeffrey's "go" before starting BUG-006.**

### 2026-04-11 — BUG-006 (P1): Output filename sanitization
- **Status:** APPLIED + unit test PASSED in sandbox.
- **What changed:** Added `_safe_name(name, max_len=80)` helper that strips `\/:*?"<>|` and control chars, trims leading/trailing spaces/dots, caps at 80 chars, falls back to "untitled" on empty. Applied at both `output_name` interpolation sites: temp audio WAV (line 769) and final MP4 (line 842).
- **Added `import re`** to top-level imports.
- **Lines affected:** 23 (import re), 51-57 (new helper), 769 + 842 (interpolation sites).
- **File size:** 903 lines.
- **AST parse:** OK.
- **Acceptance test:** `_safe_name('test/../weird:name?.mp4')` -> `'test_.._weird_name_.mp4'`. Empty string -> `'untitled'`. 200-char string -> truncated to 80. All passed.
- **All P1 bugs now applied. Waiting for Jeffrey's "go" to start P2 (BUG-005).**

### 2026-04-11 — BUG-005 (P2): Dead CRT scanline loop
- **Status:** FIXED.
- **What changed:** Deleted the dead `draw.line(fill=None, width=0)` loop and its associated `ImageDraw.Draw(img)` call (was lines 625-628). The real scanline effect at lines 626+ (now using `scanline_overlay` with `alpha=25`) is untouched.
- **Lines removed:** 4 (net). File: 903 -> 899 lines.
- **AST parse:** OK.

### 2026-04-11 — BUG-003 (P2): Error image shape mismatch
- **Status:** CLOSED as not-reproducible.
- **Investigation:** `grep -rni "error_image\|error_frame\|placeholder" nodes/` found zero hits for `error_image` or `error_frame`. All "placeholder" hits are empty-state returns (no characters, no scenes) and audio silence — not a dedicated error-image generator. No fix needed per action plan rules.

### 2026-04-11 — BUG-004 (P2): Flux prompt stacking
- **Status:** CLOSED as not-reproducible.
- **Investigation:** Action plan confirmed no Flux-specific stacking loop exists in `v2_preview.py`. The encoder is generic. Closed per action plan Section 2.

---

## Summary: All bugs addressed

| Bug | Priority | Status |
|-----|----------|--------|
| BUG-002 | P0 | Applied — kwargs sampling |
| BUG-007 | P0 | Applied — crash-safe cleanup + preview_mode + MAX_FRAMES |
| BUG-012 | P0 | Applied — safety filter + 10/10 acceptance test |
| BUG-001 | P1 | Applied — shallow copy before dict mutation |
| BUG-010 | P1 | Applied — VRAM power-wash toggle on CharacterForge |
| BUG-006 | P1 | Applied — _safe_name() at all interpolation sites |
| BUG-005 | P2 | Fixed — dead scanline loop deleted |
| BUG-003 | P2 | Closed — not reproducible |
| BUG-004 | P2 | Closed — not reproducible |

**Next steps:** Live testing on 5080 (T1-T7 from action plan Section 4), then CI scaffolding (Section 5).

---

## Bug Bible Cross-Check (read-only, no updates until prod-confirmed)

Cross-referenced all Round 7 changes against `BUG_BIBLE.yaml` (96 entries, 12 phases). Findings below. **No Bug Bible updates made** — these are theoretical flags only, pending live confirmation.

### CAUGHT AND FIXED: Bug Bible 04.01/04.02 (widget positional mismatch)
- **Issue:** We added `preview_mode` (combo) to ProductionBus and `vram_power_wash` (BOOLEAN) to CharacterForge. Both workflow JSONs (`otr_scifi_16gb_full.json`, `otr_scifi_16gb_test.json`) had stale `widgets_values` arrays — 4 values where 5 were needed (ProductionBus) and 5 where 6 were needed (CharacterForge).
- **Fix applied:** Appended `"keyframes"` and `true` to the correct positions in both workflow JSONs.
- **This would have caused:** Silent widget value misalignment on workflow load — the new widget would steal a value from a neighbor, causing wrong defaults with no error message.

### NOTED (pre-existing, not introduced by us):

| Bible ID | Area | Finding | Risk | Action |
|----------|------|---------|------|--------|
| 01.02/01.03 | paths | ProductionBus builds output_dir via `os.path.dirname(__file__)` chain instead of `folder_paths.get_output_directory()`. Pre-existing, not introduced by our changes. | Medium — could break if ComfyUI changes sandbox rules. | Flag for v2.1. Not a regression. |
| 09.02 | subprocess | ProductionBus uses `subprocess.run(capture_output=True)` for FFmpeg. Bug Bible warns this can deadlock on video data piped through stdout/stderr. Our case pipes *frame files*, not raw video bytes, so the pipe only carries FFmpeg's status text. Low risk on short encodes but could stall on very long runs. | Low | Monitor during T3 (10-min stress). If it stalls, switch to `stderr=tempfile`. |
| 07.01/07.03 | VRAM | BUG-010 power-wash uses `mm.unload_all_models()` — this is the Bible-approved pattern (07.03). Confirmed safe. | None | No action needed. |
| 10.01 | safety | BUG-012 safety filter matches Bible's two-layer defense recommendation (keyword/regex blocklist as layer 1). Layer 2 (LLM system prompt) is in the audio pipeline's story_orchestrator. | None | Confirmed aligned. |
| 12.01 | testing | Unit test suite now exists (45 tests, <0.1s). Bible recommends `pytest` + `ast.parse` for fast iteration. | None | Done. |
| 12.02 | regression | Full 15-step regression checklist should run after live testing. | Medium | Run after T1-T7. |

### NO ISSUES FOUND FOR:
- 02.x (encoding) — all file writes go through Python, no PowerShell. UTF-8 no BOM.
- 03.x (registration) — all 4 v2 nodes registered in `__init__.py` via isolated loader.
- 05.x (execution model) — no new execution order dependencies introduced.
- 06.x (caching) — no IS_CHANGED or cache changes.
- 08.x (I/O) — no new output nodes; ProductionBus was already OUTPUT_NODE=True.

---

## CI Test Suite Built (Section 5)

```
tests/
  unit/
    test_encode_prompt.py    # 4 tests — BUG-001 regression
    test_safe_name.py        # 16 tests — BUG-006 regression
    test_safety_filter.py    # 25 tests — BUG-012 regression
  safety/
    bad_prompts.txt          # 10 known-bad prompts
```

**Result: 45 tests, all passed, 0.10 seconds.** No GPU needed, no ComfyUI dependency.

---

## Version Pin (Section 8 Q4)

Added platform pin comment block to `requirements.txt`. ComfyUI commit SHA left as TBD — Jeffrey fills it in after T1-T7 pass on the 5080.

---

## Workflow JSON Updates

Both workflow JSONs updated to match new INPUT_TYPES:
- `otr_scifi_16gb_full.json`: CharacterForge +`true`, ProductionBus +`"keyframes"`
- `otr_scifi_16gb_test.json`: same changes

---

## V2_PREVIEW_ADDENDUM (Reviewer 17 / Gemini Deep Research)

### 2026-04-11 — Addendum implementation

Reviewer 17 provided a V2_PREVIEW_ADDENDUM with 6 required edits (REQ-1 through REQ-6) and several nice-to-have improvements. All required edits implemented. Two nice-to-haves also completed.

#### REQ-1: Output path — `folder_paths.get_output_directory()` with fallback
- **Status:** APPLIED.
- **What changed:** Replaced repo-relative `os.path.dirname(__file__)` chain with `folder_paths.get_output_directory()`. Added `ImportError` fallback to preserve standalone testing capability.
- **Bug Bible alignment:** This was the exact same issue flagged in our cross-check (01.02/01.03). Now resolved.

#### REQ-2: `output_subdir` optional STRING input on ProductionBus
- **Status:** APPLIED.
- **What changed:** New optional STRING input `output_subdir` (default `""`). When set, creates a subdirectory under the output root and writes all files there. Uses `_safe_name()` for sanitization.

#### REQ-3: `debug_vram_snapshots` BOOLEAN on CharacterForge, ScenePainter, ProductionBus
- **Status:** APPLIED.
- **What changed:** All three visual nodes now accept `debug_vram_snapshots` (default `False`). All `_vram_snapshot()` calls are gated behind this toggle — zero overhead when disabled. Toggle is exposed as a QA knob, no source edits needed.

#### REQ-4: Stress harness — `tests/integration/test_stress_harness.py`
- **Status:** APPLIED + all tests PASSED.
- **What changed:** New integration test file with 12 tests: orphan cleanup validation, `try/finally` crash-safety, MAX_FRAMES cap, adversarial filename edge cases (null bytes, all-unsafe, unicode, long strings), QA toggle surface verification (preview_mode, encoding_profile, debug_vram_snapshots), QA_METRICS JSON parsing, placeholder frame dimension check.

#### REQ-5: Startup janitor — `_cleanup_orphaned_preview_tmp()`
- **Status:** APPLIED.
- **What changed:** Module-level function runs at import time. Scans `preview_tmp/` for any dirs matching `otr_v2_*` prefix and removes them. Handles the kill -9 scenario where `try/finally` never fires.

#### REQ-6: QA toggles as node inputs
- **Status:** APPLIED.
- **What changed:** `preview_mode`, `encoding_profile`, and `debug_vram_snapshots` are all exposed as ComfyUI node inputs. No source edits needed to toggle QA modes.

#### NICE-TO-HAVE: `encoding_profile` combo + `_ENC_PROFILES` lookup
- **Status:** APPLIED.
- **What changed:** New combo input on ProductionBus: `preview` (fast/crf23), `balanced` (medium/crf20), `quality` (slow/crf18). Lookup table `_ENC_PROFILES` maps names to FFmpeg preset+crf pairs.

#### NICE-TO-HAVE: `QA_METRICS` JSON line in bus_log
- **Status:** APPLIED.
- **What changed:** `assemble()` appends a `QA_METRICS: {...}` JSON line to the bus log at completion. Contains preview_mode, audio_duration_s, total_frames, resolution, video_path status, fps, encoding_profile. Greppable for automated CI checks.

### Test results after addendum
```
57 tests passed in 0.18s (was 45 before addendum)

  unit/test_encode_prompt.py       4 tests  PASSED
  unit/test_safe_name.py          16 tests  PASSED
  unit/test_safety_filter.py      25 tests  PASSED
  integration/test_stress_harness.py  12 tests  PASSED
```

### Workflow JSON updates for addendum
Both workflow JSONs updated with new widget defaults:
- CharacterForge: added `false` (debug_vram_snapshots)
- ScenePainter: added `false` (debug_vram_snapshots)
- ProductionBus: added `"preview"` (encoding_profile), `""` (output_subdir), `false` (debug_vram_snapshots)

### Remaining nice-to-haves (not implemented)
- Explicit timeline duration fields from `script_json`/`production_plan_json` (currently uses heuristic)
- RTX 5080 validation envelope note for fps/width/height limits

---

## Final file inventory (Round 7 + Addendum)

| File | Lines | Change |
|------|-------|--------|
| `nodes/v2_preview.py` | 989 | 6 bug fixes + 6 addendum edits |
| `nodes/v2_preview.py.backup_round6` | 835 | Rollback point |
| `nodes/safety_filter.py` | 141 | New — content safety module |
| `tests/unit/test_encode_prompt.py` | 86 | New — BUG-001 regression |
| `tests/unit/test_safe_name.py` | 76 | New — BUG-006 regression |
| `tests/unit/test_safety_filter.py` | 131 | New — BUG-012 regression |
| `tests/safety/bad_prompts.txt` | 11 | New — 10 known-bad prompts |
| `tests/integration/test_stress_harness.py` | 170 | New — REQ-4 stress/cleanup/OOM tests |
| `workflows/otr_scifi_16gb_full.json` | ~1107 | Widget values updated (Round 7 + Addendum) |
| `workflows/otr_scifi_16gb_test.json` | ~870 | Widget values updated (Round 7 + Addendum) |
| `requirements.txt` | ~27 | Platform pin added |
| `alpha2_progress.md` | this file | Progress log |
| `ALPHA2_QA_REPORT.md` | ~140 | QA report for reviewer 17 |
| `alpha2actionplan.md` | 235 | Unchanged (read-only reference) |
