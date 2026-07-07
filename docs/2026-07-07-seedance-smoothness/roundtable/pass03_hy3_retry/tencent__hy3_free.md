<!-- requested_model: tencent/hy3:free | resolved_model: tencent/hy3-20260706:free -->

VERDICT: no — conditioner corrupts its own appended clause on repeated calls because softening is not idempotent and will mutate clause text.

MUST-FIX BEFORE BUILD:
1. [Idempotence rule] Softener set is applied to the full input string regardless of whether the smooth-motion clause (or stable marker) is already present; on a second call "No whip pans" inside the clause becomes "No slowly sweeps", breaking idempotence and changing meaning. Fix: in `_condition_seedance_prompt`, detect stable marker first; if present return `(prompt, meta)` with `changed=False` and skip both softening and append.
2. [In `CloudSeedance2Engine._partner_inputs()` step 3] `duration = self._duration_seconds(...)` is written with elided arguments; grounding (Duration Behavior) shows `_duration_seconds()` reads `OTR_CLOUD_SEEDANCE_DURATION` or derives `round(timing.target_frame_count / canvas.fps)` but its signature is not shown. Verify: `self._duration_seconds` signature and call context. Fix: pass the same request/timing object the existing code uses, assign `model["duration"] = duration` exactly once, and remove any pre-existing duration assignment in that method to avoid duplicate/conflicting keys.

SHOULD-FIX:
1. [Initial softener set] "standalone handheld" is undefined; without word-boundary regex, `handheld` could match inside `handheld dolly` after the multi-word rule or corrupt other compounds. Fix: use case-insensitive `\bhandheld\b` and apply `handheld dolly` -> `stabilized dolly` before the standalone rule (or rely on identical target). Same for `whip-pan` vs `whip pan` variants.
2. [Manual QA] A "temporary local toggle" to disable conditioning is mentioned but not specified in Implementation; if added to `eng_cloud_video.py` it becomes a hidden config path. Fix: implement disable only in `tests/` via monkeypatch or an unmerged env var; do not ship in engine.
3. [Logging line] `prompt_original_sha8` / `prompt_conditioned_sha8` lack specified hash/encoding. Fix: `hashlib.sha256(s.encode('utf-8')).hexdigest()[:8]`; ensure excerpts are truncated on char boundary, not bytes.
4. [ASSUMPTION] Plan assumes `self._text_prompt_input(request)` exists (cited in step 1). Verify: method exists in `CloudSeedance2Engine` per grounding `required_inputs` mentions `text_prompt` but no method name shown. If missing, step 1 breaks.

OPTIONAL / NICE-TO-HAVE:
- Add test asserting conditioner output is stable across 3 consecutive calls (covers idempotence fix).
- Include `seedance_requested_duration_s` as