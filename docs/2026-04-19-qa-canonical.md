# OTR Canonical QA Plan — `otr_scifi_16gb_TEST.json`

**Status:** CANONICAL — supersedes `2026-04-19-qa-brief-test-workflow-risks.md` and `2026-04-19-qa-plan-lean-team-response.md` as the active execution plan.
**Date:** 2026-04-19
**Branch:** v2.0-alpha
**Target:** `workflows/otr_scifi_16gb_TEST.json`
**Execution order:** Phase 1 (static) → Phase 2 (silent stub) → Phase 3 (VRAM/audio) → Phase 4 (log)

---

## Context — what we know going in

**Green:**
- AST / BOM / node contract / workflow↔registry: 0 violations (102 py files, 21 node classes).
- Regression suite: 183 passed, 2 skipped, 2 xfailed.
- Audio byte-identical: passing. Audio spine untouched by video work.
- Workflow uses 5 `OTR_Visual*` nodes, all registered in `__init__.py`.

**Known open defects:**
- **Task #72 (open):** `visual/backends/ltx_motion.py:226` calls `from_pretrained` against an incomplete HF snapshot missing `model_index.json`. Verified working loader exists in `scripts/verify_ltx_hybrid.py` but hasn't been ported to production.
- **BUG-LOCAL-046 (Task #99, open):** LTX loader failures fall back to stub mode silently — no loud log.

**Consult inputs:**
- `docs/2026-04-19-test-workflow-readiness__01_chatgpt.md` (gpt-5.4)
- `docs/2026-04-19-test-workflow-readiness__02_gemini.md` (gemini-3-pro-preview)
- `docs/2026-04-19-test-workflow-readiness__04_claude_synthesis_v1_wip.md`

---

## Risk ranking (rationale for phase order)

| # | Risk | Likelihood | Impact | Provable |
|---|------|-----------|--------|----------|
| 3 | Backend dispatch uncertainty — does TEST hit `ltx_motion` or short-circuit? | Medium | High (gates #1 and #2) | YES, static, 10 min |
| 1 | False Green — silent stub fallback masquerading as success | High | Medium | YES, one run + frame diff |
| 2 | CUDA OOM mid-video takes down unsaved audio (Rule C7 violation) | Medium | Critical | Partially, needs live VRAM + file checks |

**Why Phase 1 = Risk #3 first:** if dispatch resolves to still/FLUX, Phases 2 and 3 don't need to run.

---

## Phase 1 — Static Dispatch Check (Risk #3, ~10 min, no run)

Prove whether `ltx_motion` is even reachable before spending a run.

1. Open `workflows/otr_scifi_16gb_TEST.json`.
2. Find node `"class_type": "OTR_VisualRenderer"` (or `"type": "OTR_VisualRenderer"` in UI-format).
3. Record widget values: `backend`, `mode`, `test_mode`, `visual_backend`, `video_mode`.
4. Trace dispatch in source. Prefer `rg`; fall back to `grep -rn` if `rg` not installed:
   ```bash
   rg -n "OTR_VisualRenderer|ltx_motion|flux_anchor|still|stub|video_stack" visual/ --type py
   # fallback:
   grep -rn "OTR_VisualRenderer\|ltx_motion\|flux_anchor\|still\|stub\|video_stack" visual/ --include="*.py"
   ```

**Decision:**
- Dispatch resolves to `still` / `flux_anchor` / stub → Risks #1 and #2 are NOT live. Optional confidence run, otherwise stop.
- Dispatch resolves to `ltx_motion` / `video_stack` → proceed to Phase 2. **Record the exact widget values that caused it.**

---

## Phase 2 — Silent Stub Test (Risk #1, highest ROI)

Directly proves or disproves BUG-LOCAL-046.

**Step 2a — Run the workflow**

1. Run `otr_scifi_16gb_TEST.json` end-to-end in ComfyUI.

**Step 2b — Locate the output MP4**

Do NOT parse the workflow JSON for `filename` — ComfyUI save nodes use `filename_prefix` and append a counter + extension. Instead:
```bash
# sort ComfyUI's output directory by mtime and grab the most recent .mp4
ls -t ComfyUI/output/*.mp4 | head -1
# or watch the ComfyUI console for the saved path
```

**Step 2c — Extract frames**

For short clips or if stub may emit a looping 24-frame buffer, use fps=4 (not fps=1 — risk of sampling into loop period):
```bash
ffmpeg -i <output>.mp4 -vf fps=4 frame_%04d.png
# for strict checks on short clips, extract every frame:
# ffmpeg -i <output>.mp4 frame_%04d.png
```

**Step 2d — Calibrate the motion threshold**

Before trusting the threshold, calibrate against a known-good clip and a known-stub clip. TEST fixtures with burned-in timestamps, waveforms, or subtitles will show non-zero inter-frame delta even on still renders. Typical calibrated range is 8-12 when overlays are present; 5.0 is too loose.

```python
# check_motion.py
import sys, cv2, numpy as np

img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])
mean_diff = np.mean(cv2.absdiff(img1, img2))
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0  # default raised

print(f"Mean pixel diff: {mean_diff:.2f}  (threshold {threshold})")
print("REAL MOTION (PASS)" if mean_diff > threshold else "LIKELY STUB FALLBACK (FAIL)")
```

Run on several frame pairs (not just adjacent — also check frames 10 apart to catch looping buffers).

**Step 2e — Scan ComfyUI console**

Broaden the regex — the exact LTX success string isn't guaranteed verbatim:
```
ltx|LTXVideo|LTX.*load|from_pretrained.*ltx|from_single_file.*ltx
```

Also scan for fallback indicators:
```
stub|fallback|failed|model_index\.json|OOM|out of memory
```

**Pass:**
- Inter-frame mean pixel diff > calibrated threshold across multiple frame pairs AND at least one console line matches the LTX load regex AND no stub/fallback keywords present.

**Fail:**
- Inter-frame diff below threshold (near-identical frames) AND no LTX success log → confirms silent stub fallback. BUG-046 is firing; Task #72 is blocking meaningful video output.

---

## Phase 3 — VRAM & Audio Safety (Risk #2, conditional)

**Skip this phase if Phase 1 resolved to still/FLUX.** Only run if `ltx_motion` is reachable.

**Step 3a — Monitor VRAM during run**

```bash
nvidia-smi -l 2
# or LibreHardwareMonitor at http://localhost:8085/data.json (Jeffrey keeps LHM running 24/7)
```

Capture peak VRAM. Target ceiling: 14.5 GB. Anything over is a red flag.

**Step 3b — Verify audio persisted**

Check file mtimes AND file sizes AND actual audio duration. mtime alone only proves write order, not that the file survived a mid-run process crash:
```bash
ls -l --time-style=full-iso *.wav *.mp4
ffprobe -i output.wav 2>&1 | grep -E "Duration|bit_rate"
```

Expected audio duration should match the episode target (e.g., ~490 seconds for the current 5-min TEST episode). Short or truncated WAVs → audio died.

**Step 3c — Confirm graph execution order**

From ComfyUI console, verify audio assembler nodes execute and complete **before** heavy video renderer nodes. If video comes first or runs parallel, a video OOM can take unsaved audio down with it.

**Pass:**
- Peak VRAM ≤ 14.5 GB AND audio `.wav` exists with full expected duration AND audio completed in console before video stage.

**Fail:**
- Video OOM'd and audio WAV is missing OR truncated → **Rule C7 violation** (audio byte-identical guarantee broken at process level).

---

## Phase 4 — Logging

Log findings in `BUG_LOG.md`. Use the project's actual Bug Log schema; cite rule C7 by its exact wording from `CLAUDE.md` (not paraphrased). Capture:

- Exact widget values for `OTR_VisualRenderer` in the TEST JSON.
- Phase 1 dispatch resolution (which backend the renderer selected).
- Phase 2: mean pixel diff per frame pair, calibration threshold used, console log excerpts (LTX load hit/miss, stub/fallback keywords).
- Phase 3 (if run): peak VRAM, audio file size + ffprobe duration, graph execution order evidence.
- Bible candidate: yes/no per entry.

---

## Go / No-Go guidance

**If Phase 1 = still/FLUX:**
- Safe to run for audio+still validation. Expect ~85% smoke success.
- Do NOT declare video stack "validated" from this run.

**If Phase 1 = ltx_motion AND Phase 2 = silent stub:**
- Expected outcome given known-open #72 and #046.
- Next action: patch `visual/backends/ltx_motion.py` using the hybrid loader from `scripts/verify_ltx_hybrid.py`, **keeping `torch_dtype=torch.float8_e4m3fn`** (NOT bfloat16 — Blackwell FP8 + 14.5 GB ceiling demands it). In the same edit, flip the silent `except` to `logger.error(...)`. Rerun.

**If Phase 1 = ltx_motion AND Phase 2 = real motion:**
- Surprising but good. Proceed to Phase 3 carefully.

**If Phase 3 = OOM/audio loss:**
- Stop. Rule C7 violation. Investigate audio flush ordering before any further video runs.

---

## Key gotchas (absorbed from team fact-check)

- ComfyUI save nodes use `filename_prefix` + counter — don't parse JSON for a literal filename, use mtime sort instead.
- Motion threshold 5.0 is too loose with burned-in overlays; calibrate on known-good + known-stub, typical range 8-12.
- `fps=1` can miss looping stub buffers; prefer `fps=4` or all frames for short clips.
- `rg` may not be installed — have `grep -rn` fallback ready.
- LTX load success string isn't guaranteed — broaden regex to `ltx|LTXVideo|from_pretrained.*ltx|from_single_file.*ltx`.
- Rule C7: cite exact wording from project rules in BUG_LOG entries, not paraphrased.
- Audio survival check: mtime confirms write order, ffprobe confirms the file isn't truncated.

---

## References

- Consult inputs: `docs/2026-04-19-test-workflow-readiness__01_chatgpt.md`, `__02_gemini.md`, `__04_claude_synthesis_v1_wip.md`
- Earlier drafts (superseded): `docs/2026-04-19-qa-brief-test-workflow-risks.md`, `docs/2026-04-19-qa-plan-lean-team-response.md`
- Open tasks: #72 (ltx_motion patch), #99 (BUG-046 loud fallback), #62 (end-to-end verify)
- Rule C7: `CLAUDE.md` → v2.0 Constraints
- Verified LTX load path: `scripts/verify_ltx_hybrid.py`
