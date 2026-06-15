<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document is still a tuning question, not a build-ready bake-in spec, and the grounded code has cue-placement and release-state hazards that can change runtime behavior.

MUST-FIX BEFORE BUILD:
1. [Deliverable / Questions 1-5] The document does not actually specify the final bake-in values; it asks another reviewer to choose them. That is not buildable as-is. Concrete fix: replace the questions with one final table containing exact `STEPS`, `CFG`, `SAMPLER`, `SCHEDULER`, `CONTEXT_S`, seconds-start mapping, and final negative prompt, then update the inline defaults in `eng_stable_audio_3.py`:
   - `OTR_SA3_CONTEXT_S`
   - `OTR_SA3_STEPS`
   - `OTR_SA3_CFG`
   - `OTR_SA3_SAMPLER`
   - `OTR_SA3_SCHEDULER`
   - `_SA3_NEG_DEFAULT`

2. [Current defaults / Question 3b / `eng_stable_audio_3.py::_sa3_clip_window`] Cue placement depends on prompt substring detection for `"intro"` and `"outro"`, but the use case names cues as `opening`, `closing`, and `interstitial`. As written, a prompt containing “closing” but not “outro” will fall into the middle-slice branch, not the tail branch. Concrete fix: do not infer cue type from free text; pass explicit cue role metadata into `_sa3_clip_window`. Minimum patch if metadata is unavailable: treat `"opening"` as head and `"closing"` as tail in addition to `"intro"`/`"outro"`.

3. [Question 4 / `generate_clip`] The sampler and scheduler strings are raw env/default strings passed directly into `KSampler().sample(...)`. The document does not pin a ComfyUI version or require a preflight that verifies `dpmpp_3m_sde_gpu` and `exponential` are valid on the target install. If either name differs in the installed ComfyUI, this fails at render time. Concrete fix: before bake-in, add startup validation against the target ComfyUI sampler/scheduler registry, or pin the exact ComfyUI build where these names are known-valid. verify: exact registry/API name for sampler and scheduler enumeration in the deployed ComfyUI version.

4. [Constraints: determinism / `eng_stable_audio_3.py` module docstring vs class attributes] The docstring says SA3 is “opt-in behind `OTR_ENABLE_STABLE_AUDIO_3` until F validates render-twice determinism,” but the class is already promoted: `default_roles = ("music",)` and `requires_flag = None`. That is an explicit release-state contradiction. Concrete fix: either update the docstring to match the promoted/default behavior, or restore the flag gate if determinism has not actually been validated.

5. [Constraints: determinism / Current defaults] The document asserts deterministic single-pass generation, but the selected path uses a GPU SDE sampler string, and the grounding only shows that the integer seed is passed into KSampler. It does not show render-twice validation on the RTX 5080 / Blackwell stack. Concrete fix: add a required pre-build acceptance test: same prompt, duration, seed, defaults, and model must produce identical audio or an explicitly defined numeric tolerance. [ASSUMPTION] If bit-identical output is not required, define the allowed waveform/hash tolerance instead of using the word “determinism.”

SHOULD-FIX:
1. [Question 6 / Deliverable / `generate_clip`] Question 6 asks about denoise, but the deliverable omits it while the code hard-codes KSampler denoise to `1.0`. Concrete fix: explicitly state `DENOISE = 1.0` in the final bake-in table, or add an env-overridable `OTR_SA3_DENOISE` default if it is meant to be tunable.

2. [Constraints: env-overridable / `generate_clip`] Malformed env overrides currently raise raw `ValueError` during `float(...)` / `int(...)` parsing for `OTR_SA3_CONTEXT_S`, `OTR_SA3_STEPS`, and `OTR_SA3_CFG`. That bypasses the stated fail-closed/named-error posture used elsewhere. Concrete fix: parse these through a small helper that validates type/range and raises `EngineUnusable(..., EngineUsabilityReason.MALFORMED_CONFIG, ...)` with the offending env var name.

3. [Question 3 / `eng_stable_audio_3.py::_sa3_clip_window`] There is no validation range for `seconds_total` beyond `max(context_s, dur)`. The document discusses SA Open being trained up to ~47s, but the env override can set arbitrary large values. Concrete fix: define and enforce a sane allowed range, e.g. minimum cue duration, maximum documented SA3 context limit. verify: actual recommended maximum for Stable Audio 3 small music in the deployed ComfyUI node/model docs.

4. [Use case / Constraints] The VRAM constraint `≤14.5GB` is stated, but the document does not require measuring the final chosen defaults on the actual graph. Steps usually affect time more than peak memory, but checkpoint/text encoder/latent/node behavior still need confirmation. Concrete fix: add a pre-build smoke test on the target RTX 5080 16GB rendering 12s, 8s, and 4s cues with peak allocated/reserved VRAM recorded below the limit. [ASSUMPTION] The 14.5GB limit is intended as an enforceable release gate.

5. [Negative prompt / Use case] The negative prompt includes `out of tune`. For this specific target—theremin, eerie strings, analog tape warmth—some period-appropriate pitch instability may be desirable. Concrete fix: either validate that `out of tune` does not suppress the intended theremin/tape character, or replace it with a narrower term such as `bad intonation` / `amateur performance`. This is a listening-risk, not a code defect.

OPTIONAL / NICE-TO-HAVE:
- [Logging] The prompt hash is useful, but the final baked defaults should also be logged in one stable config line at engine load, not only per cue, to simplify release verification.
- [Deliverable] Include the exact final negative prompt as a single copy-paste string and avoid line-wrapped variants that can accidentally change punctuation or spacing.
- [Question numbering] There are two “3” sections plus “3b”; renumber before converting this into a release ticket.

CUT THESE (over-engineering):
1. [Question 3] Cut “one value, or per-cue if strongly justified” from the final build spec unless listening tests prove per-cue context is needed. A single `CONTEXT_S` is already the intended bake-in path and avoids adding another hidden tuning matrix.

2. [Current defaults / `_sa3_clip_window`] Cut free-text cue-role inference as the control mechanism. It is heavier and less reliable than passing the known cue role directly. Safe to cut because the cue type is already known upstream by the episode structure: opening, closing, interstitial.

3. [Questions 1-6] Cut the roundtable-style trade-off prose from the final implementation ticket. Keep only constants, code locations, validation tests, and rollback/env override names. Safe to cut because the prose does not execute and increases the chance of baking ambiguous defaults.