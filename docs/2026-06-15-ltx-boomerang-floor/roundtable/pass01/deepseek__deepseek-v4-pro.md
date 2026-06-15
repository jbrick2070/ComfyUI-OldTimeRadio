<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no — the plan is not build-ready; missing concrete half-length computation, i2v interaction, and the decode-floor "probe" adds unspecified complexity. The core boomerang mirroring is correct, but several key decisions are unresolved.

MUST-FIX BEFORE BUILD:
1. [Section “THE FORK to decide – decode-floor handling”] The recommendation “probe 73/49 to go lower” is speculative and introduces a missing design (what probe? start-up render? fallback to (C) on failure?). This risks build-time ambiguity and fragile runtime behavior. Fix: replace with a hardcoded minimum half-length of 97 (the proven-safe value) for the boomerang path, configurable via env (e.g., `OTR_LTX_LOOP_MIN_DECODE_FRAMES` default 97). No probe, no fallback to (C). [ASSUMPTION: 97 is universally safe at 832x480; confirmed by the document’s evidence.]

2. [Section “Proposed restore”] The plan does not address i2v + boomerang interaction. If i2v is enabled (default) and the boomerang is on, the forward half uses conditioned first-frame only; mirroring produces a reversal midpoint with an unconditioned first frame of the reversed part, causing a visible seam. Fix: specify that when `boomerang` is active and `_use_i2v()` returns True, either disable the boomerang (render full-length, no loop) or force text‑only mode for the half‑render (ignore init image). This must be decided and documented before coding.

3. [Section “Original boomerang to restore” / Proposed restore] The plan states “Render HALF the target duration” but does not define how to compute that half-length given the existing frame‑length snapping and the new floor. Must provide a concrete formula: compute `half_target = (target_frame_count + 1) // 2`, then snap to `8n+1` using a separate floor equal to `max(snapped_ask, _LOOP_MIN_FRAMES_FOR_HALF)` (e.g., 97), bypassing the global `OTR_LTX_MIN_DECODE_FRAMES`. Integrate this into a helper function (e.g., `_ltx_half_length()`) called before the graph build.

4. [Section “Proposed restore”] The in‑tensor mirror `frames[-2::-1]` will fail with an index error if `len(frames) < 2`. While LTX min frames ensure `>= 9`, defensive guard: add `if len(frames) < 2: return …` or an assertion, to prevent cryptic crash if any future path yields a short decode.

5. [Section “THE FORK to decide – option (C)”] The fallback “Safe full‑render + use‑first‑half” is not needed if we adopt a hardcoded safe half‑length (97). It wastes compute (full 169 frames), reduces motion depth, and does not simplify the code. Remove option (C) from the design; rely on truncation by the composite if the boomerang slightly exceeds the beat window (same as current floor behavior).

SHOULD-FIX:
- [Section “Proposed restore”] Mention that the `raw` result dict must include `ltx_loop_via_reverse: true` (the ledger stamp). Without that, tracing and forensics break.
- [Section “THE FORK to decide”] The canvas‑aware floor (option A) is orthogonal; note that reducing the global decode floor for non‑loop renders at 832x480 should be done separately, not entangled with the boomerang patch.
- [Section “Questions for the panel”] Clarify that the boomerang doubling will often exceed the requested frame count; the composite’s truncation behavior must be verified with the actual beat window to avoid silent over‑shooting.

OPTIONAL / NICE-TO-HAVE:
- Allow per‑request override of the loop (e.g., request field `loop_via_reverse`), restoring the b005 ledger granularity. The current env‑only toggle is less flexible.

CUT THESE (over‑engineering):
- The “probe” mechanism to find a lower safe decode length (73, 49) adds unnecessary complexity and a startup validation step that can fail. Remove it. The proven‑safe 97 is sufficient for the boomerang’s motion depth gain; any further reduction yields negligible benefit and risks breaking the decode.
- Option (C) “safe full‑render + use‑first‑half” adds a fallback path that is neither simpler nor safer than a hardcoded minimum half‑length. Remove it.