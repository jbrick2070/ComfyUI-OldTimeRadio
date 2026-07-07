<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. The Seedance-only wiring is directionally correct, but the helper contract is internally inconsistent unless the appended smooth-motion clause is protected from the softener pass.

MUST-FIX BEFORE BUILD:
1. [Implementation / Idempotence rule] The smooth-motion clause contains tokens that the softener set is required to rewrite: “whip pans”, “handheld shake”, and “rapid zooms”. On a second call, a naive implementation will mutate the already-appended clause, so `_condition_seedance_prompt()` will not be idempotent. Concrete fix: choose a stable marker sentence that contains no softened terms, e.g. `Gentle parallax only; all motion gradual and physically continuous.`, detect it before applying replacements, and either:
   - return the prompt unchanged if the marker is present; or
   - split/protect the existing smooth clause and apply softeners only to the user-authored prefix.
   Add the idempotence test against a prompt that already contains the full appended clause.

2. [Implementation / Softener set] Replacement order is a hidden dependency. `handheld dolly` must be replaced before standalone `handheld`, otherwise `handheld dolly` becomes `stabilized dolly` accidentally only if lucky, and future phrase replacements can be broken by earlier generic matches. Concrete fix: implement replacements in longest/specific-first order:
   - `dynamic dolly push`
   - `handheld dolly`
   - `whip[- ]pans?`
   - `white[- ]hot`
   - `rapid zooms?`
   - `aggressively`
   - standalone `handheld`
   Use word boundaries for standalone `handheld`.

3. [Implementation / Logging] “bounded structured line” is underspecified enough to break log parsability. Excerpts can contain newlines, tabs, quotes, or non-printing characters, so a “one line” log can become multi-line or unstructured. Concrete fix: sanitize excerpts before logging: collapse whitespace to single spaces, truncate to 160 characters after sanitization, and log as key/value fields or JSON through the module logger.

4. [Implementation / Metadata contract] `_condition_seedance_prompt(prompt) -> tuple[str, dict]` says metadata is useful for logging/tests, but the required keys are not defined. `_partner_inputs()` depends on metadata for `changed`, SHA8s, and excerpts unless it recomputes them. Concrete fix: define the exact metadata schema:
   - `changed: bool` where `changed == (conditioned_prompt != prompt)`
   - `original_sha8`
   - `conditioned_sha8`
   - `original_excerpt`
   - `conditioned_excerpt`
   - optionally `softeners_applied: list[str]`
   Or explicitly compute all logging fields inside `_partner_inputs()` and keep helper metadata test-only.

5. [Tests / Seedance unsupported fields] The test “sends the conditioned prompt and no unsupported Seedance fields” must preserve the current supported Seedance request shape from the grounding. Concrete fix: assert that `_partner_inputs()` still includes only the existing supported fields:
   - nested model fields: `model`, `prompt`, `resolution`, `ratio`, `duration`, `generate_audio`, `reference_images.image_1`, `reference_audios.audio_1`
   - top-level: `seed`, `watermark`
   Do not fail the test merely because existing supported `seed`/`watermark` are present.

SHOULD-FIX:
1. [Implementation / Regex contract] Specify that matching is case-insensitive but replacements are normalized to the exact lowercase replacement phrases. Otherwise tests may become brittle around case preservation. Concrete fix: use compiled `re.Pattern(..., re.IGNORECASE)` and assert expected normalized replacements.

2. [Implementation / Hashing] Define SHA8 calculation exactly. Concrete fix: `hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]`. Use the raw original and raw conditioned prompt, not sanitized excerpts.

3. [Implementation / Duration sequencing] The plan says compute `duration = self._duration_seconds(...)` once, but does not name the arguments. [ASSUMPTION] Existing `_duration_seconds()` likely depends on request timing/canvas. Concrete fix: preserve the current call signature exactly and only move the result into a local variable before assigning `model["duration"] = duration`; do not recompute after mutating `model["prompt"]`.

4. [Tests / Idempotence coverage] The current idempotence bullet is too broad to catch the clause/softener collision. Concrete fix: include this exact case:
   - input contains risky `music_open` text
   - first output appends the clause
   - second call returns byte-identical output
   - second metadata reports `changed == False`

5. [External system integration] [ASSUMPTION] ByteDance/Seedance may enforce a prompt length limit not shown in the grounding. The appended clause is long and could push already-long M4 prompts over a provider cap. Concrete fix: verify the installed partner/API prompt max if present; if none is exposed, add a log field for original/conditioned lengths and do not introduce silent truncation in this pass.

6. [Manual QA] The temporary local toggle for A/B risks leaking into committed code or becoming an unsupported env surface. Concrete fix: keep it uncommitted, or if committed, make it explicitly test-only and remove it before merge. The implementation section should remain unconditional Seedance-only conditioning.

OPTIONAL / NICE-TO-HAVE:
- Add `softeners_applied` to metadata so tests can verify which rules fired without parsing the full prompt.
- Add one regression test using the grounded `music_open` register text verbatim.
- Add one test that Wan/Kling/Pixverse prompts containing “handheld” are unchanged, proving the helper is not called outside `CloudSeedance2Engine._partner_inputs()`.

CUT THESE (over-engineering):
1. [Manual QA] Do not add provider-side camera/motion controls, optical flow, CFG, sampler, temperature, or motion-strength fields. Grounding confirms `ByteDance2ReferenceNode` / `Seedance2TaskCreationRequest` do not expose those fields for Seedance 2 Reference, so adding them would break request shape.
2. [Manual QA] Do not switch to `ByteDance2FirstLastFrameNode` for this pass. Grounding says it is a different partner node path and does not carry the same multimodal reference-audio shape as `ByteDance2ReferenceNode`.
3. [Duration Policy] Do not add retiming/time-compression logic. Grounding confirms downstream trimming happens in `OTR_SilentComposite` via head-frame truncation, not speed-up; the requested provider-minimum duration policy is sufficient for this pass.