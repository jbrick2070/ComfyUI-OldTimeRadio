# Tech Brief — `otr_scifi_16gb_TEST.json` Run Risks (QA Handoff)

**Author:** Claude (via Jeffrey)
**Date:** 2026-04-19
**Branch:** `v2.0-alpha`
**Target:** QA team — identify the single most provable risk (or tackle top 3)
**Source consults:** ChatGPT gpt-5.4 + Gemini gemini-3-pro-preview (round-robin, 2026-04-19)

---

## What's known going in

- All static gates green (AST, BOM, node contract, workflow↔registry: 0 violations).
- Regression suite: 183 passed, 2 skipped, 2 xfailed.
- Audio byte-identical tests: passing. Audio spine untouched by video work.
- Workflow has 5 `OTR_Visual*` nodes — all registered in `__init__.py`.
- **Known-broken:** `visual/backends/ltx_motion.py:226` calls `from_pretrained` against an incomplete HF snapshot missing `model_index.json`. Task #72 still open.
- **Known-silent:** Failure in that loader falls back to stub-mode with no loud log. BUG-LOCAL-046 still open.

---

## Top 3 risks — ranked by combined likelihood × impact × provability

### Risk #1 — False Green (silent stub fallback masquerading as success)

**Likelihood:** High — this is already the documented behavior (BUG-LOCAL-046).
**Impact:** Medium — run "completes" but delivers still frames with procgen overlay, not real LTX motion. Audio is unaffected. Operator time wasted; false confidence is the real damage.
**Provable by QA:** **YES — this is the easiest thing to prove and the one I'd pick for a single-test QA pick.**

**How QA proves it:**
1. Run `otr_scifi_16gb_TEST.json` end-to-end.
2. Locate the produced MP4 output.
3. Step through frame-by-frame (ffmpeg: `ffmpeg -i out.mp4 -vf fps=1 frame_%03d.png`).
4. **Pass criterion:** between-frame pixel delta is non-zero in non-letterbox regions (real motion).
5. **Fail criterion:** frames are identical or differ only by overlay noise — backend silently fell back to stubs.
6. Cross-check against ComfyUI console log: absence of LTX load success message, or any `stub`/`fallback` keyword in output.

**Why this is the #1 pick:**
- Proves a known open defect (BUG-046) with a minimal test fixture.
- Does not require code changes to verify.
- Does not risk audio.
- Result is binary and unambiguous.

---

### Risk #2 — CUDA OOM in video stage takes down unsaved audio

**Likelihood:** Medium (Gemini's analysis) — even if #72 is patched, raw HF Diffusers pipeline bypasses ComfyUI's VRAM management. T5-XXL FP16 (~9.4 GB) + LTX FP8 (~2 GB) + forward-pass attention on a 14.5 GB ceiling is tight.
**Impact:** **Critical** — direct rule C7 violation. Hard CUDA OOM in a custom node crashes the ComfyUI python process. If audio nodes haven't flushed to disk yet, the whole episode dies.
**Provable by QA:** Partially — requires live VRAM monitoring + graph order inspection.

**How QA proves it:**
1. Start LibreHardwareMonitor (`http://localhost:8085/data.json`) — 2-second poll to capture peak VRAM.
2. Check file modification timestamps on the audio output (`.wav`) vs. video output (`.mp4`) **after a completed run** — audio should be written first.
3. Grep the graph execution order from ComfyUI console: which node IDs run in what order? Does the audio assembler complete before the video renderer starts?
4. **Pass criterion:** audio `.wav` exists on disk and is the expected length even if video stage errored.
5. **Fail criterion:** video stage crashed ComfyUI mid-run AND audio `.wav` is missing or truncated.

**Note:** This risk only materializes if `OTR_VisualRenderer` actually dispatches into `ltx_motion`. If TEST short-circuits to still/FLUX mode, VRAM pressure is much lower and this risk drops.

---

### Risk #3 — Backend dispatch uncertainty

**Likelihood:** Medium — Task #77 rewrote the TEST JSON to inject offline assets + hardcode telemetry, which MAY short-circuit to still mode. Not confirmed.
**Impact:** High — determines whether risks #1 and #2 are actually in play. If TEST cannot reach `ltx_motion`, both above risks drop to near zero and the run is safe.
**Provable by QA:** YES — statically, without running anything.

**How QA proves it:**
1. Open `workflows/otr_scifi_16gb_TEST.json`, find the `OTR_VisualRenderer` node, note its widget/input values (especially any `backend`, `mode`, or `test_mode` field).
2. Grep repo for renderer dispatch:
   ```
   rg -n "class OTR_VisualRenderer|ltx_motion|flux_anchor|video_stack" visual/ --type py
   ```
3. Trace the dispatch branch: given the TEST JSON's widget values, which backend name string is selected?
4. **Pass criterion:** dispatch resolves to `still` / `flux_anchor` / stub path — risks #1 and #2 do not materialize.
5. **Fail criterion:** dispatch resolves to `ltx_motion` or `video_stack` — risks #1 and #2 are live.

**Why this belongs in the top 3:** answering this statically costs ~10 minutes and gates the other two risks. Any QA doing #1 or #2 should do #3 first.

---

## My single pick if QA can only do one thing

**Risk #1 — False Green (silent stub fallback).**

Reasons:
- Cheapest to prove (one test run + one ffmpeg extraction + one visual diff).
- Proves or disproves an already-open bug without requiring any code change first.
- Zero risk to audio during the QA test.
- If they find stubs, that is immediately actionable: it tells us Task #72 is blocking the TEST from being meaningful, and BUG-046 is blocking observability.

A positive finding here (i.e., QA confirms stubs) unblocks the correct sequencing: **patch #72 keeping `float8_e4m3fn` dtype, flip #046 to loud log, rerun.**

---

## Lower-tier risks (not top 3 but worth noting)

| # | Risk | Why not top 3 |
|---|------|--------------|
| 4 | `t5xxl_fp16.safetensors` actually being a misnamed quantized file | Purely VRAM math — only matters if risks #1/#2 are live |
| 5 | FLUX anchor weights missing from expected path | Would fail loudly at load time, not silently — easier to diagnose |
| 6 | ComfyUI port confusion (8000 vs 8188) | Operational, not a pack-level defect |
| 7 | Partial-render artifacts from swallowed exceptions | Subset of BUG-046 — same fix covers it |
| 8 | Process hang during loader retry | Low probability; current code fails-fast into fallback rather than hanging |

---

## What NOT to worry about (based on the static gates)

- **Audio corruption via the code path** — audio byte-identical tests already cover this.
- **Node registration drift** — workflow↔registry audit passed.
- **Syntax / import errors** — AST audit passed 102 files.
- **BOM encoding issues** — 0 violations.

The static gates are load-bearing here. The remaining risks are all runtime behaviors that the static suite by design cannot see.

---

## Handoff checklist for QA

- [ ] Risk #3 first (10-min static check) — determines whether #1/#2 are live
- [ ] Risk #1 (one test run + frame diff) — cheapest provable defect
- [ ] Risk #2 (LHM poller + file timestamps) — only if #3 confirms `ltx_motion` is reachable
- [ ] Capture all findings in BUG_LOG.md under the `Bible candidate` field

---

## References

- Consult raw: `docs/2026-04-19-test-workflow-readiness__01_chatgpt.md`, `__02_gemini.md`
- Synthesis: `docs/2026-04-19-test-workflow-readiness__04_claude_synthesis_v1_wip.md`
- Open tasks: #72 (ltx_motion patch), #99 (BUG-046 loud fallback), #62 (end-to-end verify)
- Rule C7 (audio byte-identical): `CLAUDE.md` v2.0 Constraints
