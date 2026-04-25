# Claude Synthesis v1 (WIP) — Test Workflow Readiness

**Status:** IN PROGRESS — not yet a go/no-go decision. Pre-flight pending.
**Date:** 2026-04-19
**Branch:** v2.0-alpha
**Target workflow:** `workflows/otr_scifi_16gb_TEST.json`

## Inputs to this synthesis

- `docs/2026-04-19-test-workflow-readiness__00_question.md` — the question posted to both models
- `docs/2026-04-19-test-workflow-readiness__01_chatgpt.md` — gpt-5.4, 45.9s
- `docs/2026-04-19-test-workflow-readiness__02_gemini.md` — gemini-3-pro-preview, 31.3s

## Where they agree

- Green static/regression is necessary but not sufficient. The known-broken LTX loader is the dominant risk.
- Cheap pre-flight first — grep `OTR_VisualRenderer` dispatch + TEST JSON renderer node widgets — before firing blind.
- If TEST can hit `ltx_motion`: patch Task #72 first.
- Fix BUG-LOCAL-046 in the same edit — flip silent `except` to a loud `logger.error`. Tiny change, massive payoff.

## Where they disagree (material — Gemini wins)

ChatGPT suggested patching to `torch_dtype=bfloat16` to match the verify script. Gemini corrected this:

- `float8_e4m3fn` was chosen deliberately for Blackwell FP8 + the 14.5 GB VRAM ceiling.
- T5-XXL FP16 (~9.4 GB) + LTX BF16 (~4 GB) = 13.4 GB of weights alone, plus spatial-temporal attention during generation.
- BF16 on a 16 GB laptop will almost certainly OOM during the forward pass.

**Decision:** keep `torch.float8_e4m3fn` on the LTX model when patching #72. Do not regress to BF16 just because the verify script used it. (The verify script was a loader check, not a generation check — it never touched the forward pass, so it never exercised the VRAM spike.)

## New risk Gemini surfaced (not in ChatGPT's answer)

Raw HF Diffusers pipeline instantiated inside a custom node bypasses ComfyUI's native VRAM offload/pacing. A hard CUDA OOM mid-video will crash the ComfyUI python process. If audio hasn't finished writing to disk yet, audio dies with it — direct violation of rule C7. The audio byte-identical tests guard the code path, not the process survival.

This is the real reason not to fire blindly. Audio is king — a process-level crash during video generation takes audio down with it unless audio is already persisted.

## Pre-flight checklist (before any fire)

1. [ ] Grep `OTR_VisualRenderer` dispatch logic — does this TEST JSON's configuration route into `ltx_motion`, or short-circuit to still/FLUX?
2. [ ] Read the `OTR_VisualRenderer` node's widget values in `otr_scifi_16gb_TEST.json`.
3. [ ] Stat `models/text_encoders/t5xxl_fp16.safetensors` — confirm it is actually ~9.4 GB FP16 vs. a misnamed quantized file. VRAM math depends on this.
4. [ ] Confirm audio nodes execute (and write to disk) **before** `OTR_VisualRenderer` in the graph. If audio runs in parallel or after video, a video OOM takes down unsaved audio.

## Decision tree

**If pre-flight shows TEST does NOT hit `ltx_motion`:**
- Fire as-is. ~85% chance of clean smoke result. Audio safe.

**If pre-flight shows TEST hits `ltx_motion`:**
- Patch #72 — swap `from_pretrained(_LTX_PATH, ...)` for the hybrid `from_single_file` loader verified in `scripts/verify_ltx_hybrid.py`.
- **Keep `torch_dtype=torch.float8_e4m3fn`** on the LTX model. Do NOT use bfloat16.
- In the same edit, close BUG-LOCAL-046 minimally: flip the silent stub fallback `except` to `logger.error("LTX motion backend failed to load; falling back to stub mode: %s", exc)`.
- Confirm audio executes before video in this graph. If not, reorder or block-until-flush before video stage.
- Run regression again (Bug Bible + dropdown guardrails + core + audio byte-identical).
- Then fire.

**Don't declare "validated" based on the hybrid verify script.** That script loaded weights only. Real validation requires one actual video generation pass surviving without OOM.

## Open uncertainties

- Actual size + dtype of `t5xxl_fp16.safetensors` on disk.
- Graph execution ordering of audio vs. video in `otr_scifi_16gb_TEST.json`.
- Whether `OTR_VisualRenderer` has a test-mode short-circuit already wired (Task #77 rewrote the TEST JSON to sever LLM + inject offline assets + hardcode telemetry — this may already force still mode).

## Next action

Run the pre-flight (item 1-4 above). No code change yet. Once pre-flight answers are in, promote this doc from v1 WIP to v2 with the go/no-go decision.

## References

- CLAUDE.md "Round-Robin Consultation" rule
- Task #72 — `visual/backends/ltx_motion.py` hybrid loader patch (in_progress)
- Task #99 — BUG-LOCAL-046 surface silent stub-mode fallbacks (pending)
- Task #62 — Verify end-to-end smoke test of video_stack (in_progress)
- `scripts/verify_ltx_hybrid.py` — verified hybrid load path (loader only, no forward pass)
