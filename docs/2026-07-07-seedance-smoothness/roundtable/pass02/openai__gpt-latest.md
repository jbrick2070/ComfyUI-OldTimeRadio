<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — buildable only after choosing the insertion point and defining concrete logging/metadata interfaces; current plan leaves key code targets and data shapes ambiguous.

MUST-FIX BEFORE BUILD:
1. [Buildable Direction 1 / Open Questions] Insertion point is unresolved, and `CloudSeedance2Engine._text_prompt_input()` is not grounded in the provided code excerpts. Concrete fix: implement a verified helper in `nodes/_otr_video_engines/eng_cloud_video.py`, e.g. `_condition_seedance_prompt(prompt: str) -> tuple[str, dict]`, and call it from `CloudSeedance2Engine._partner_inputs()` immediately before assigning `model.prompt`. Do not implement in `render_driver.py` unless you also prove the engine name is available there and can be used without affecting Wan/LTX/Pixverse/Kling prompts.

2. [Buildable Direction 1] Conditioner output is underspecified for null/empty/non-string prompts. `text_prompt` is required, but the plan does not say what happens if `req["text_prompt"]` is missing or blank. Concrete fix: normalize with `prompt = str(prompt or "").strip()`, append the Seedance clause only if non-empty, or use the clause alone as fallback; log a warning when falling back.

3. [Buildable Direction 2] Verb softener is not implementable deterministically as written because replacement rules are case/punctuation/spacing-sensitive and could silently miss real prompt text, e.g. `Dynamic dolly push forward.` vs replacement key `dynamic dolly push`. Concrete fix: define exact case-insensitive regex replacements with word boundaries and preserve punctuation, e.g. `re.sub(r"\bwhip[- ]pans?\b", "slowly sweeps", prompt, flags=re.I)`.

4. [Buildable Direction 3 / Open Questions] Observability fields are listed but no sink or schema is specified. “without changing workflow JSON” is unresolved. Concrete fix: choose one existing output channel and name exact keys. If logging only, add a structured log line in `CloudSeedance2Engine._partner_inputs()` or immediately after duration calculation with keys such as:
   - `engine`
   - `shot_id` if available; otherwise omit
   - `prompt_variant`
   - `prompt_original_sha256_12`
   - `prompt_conditioned_sha256_12`
   - `prompt_original_excerpt`
   - `prompt_conditioned_excerpt`
   - `seedance_requested_duration_s`
   - `target_frame_count`
   - `fps`
   - `duration_clamped_to_provider_min`
   Do not require `timeline_quality_report` fields in this same change unless its producer/shape is verified.

5. [Buildable Direction 3] “downstream delivered frame status from `timeline_quality_report`” is not grounded. The excerpts mention `assemble_silent_timeline()` behavior but do not show a `timeline_quality_report` API, file, schema, or field names. Concrete fix: verify: existence, location, and shape of `timeline_quality_report`. If absent, cut this from Pass 01 and rely on raw/canonical/manifest frame counts already available around canonicalization/assembly.

6. [Buildable Direction 4] A/B comparison says “same still/audio/model/resolution/ratio” but Seedance seed is documented as non-deterministic. Comparing one A vs one B can misattribute random generation variance to the prompt. Concrete fix: run at least 2–3 samples per variant or explicitly label the A/B as qualitative/non-deterministic. Do not claim same seed controls the test.

7. [Buildable Direction 4] Variant C says use `Seedance 2.0` instead of `Seedance 2.0 Fast` but does not say how. Grounding shows model is controlled by `OTR_CLOUD_SEEDANCE_MODEL` and aliases are normalized. Concrete fix: document the exact override: set `OTR_CLOUD_SEEDANCE_MODEL="Seedance 2.0"` for C, restore/unset it for A/B using default `Seedance 2.0 Fast`.

SHOULD-FIX:
1. [Buildable Direction 1] Clause lacks the policy-critical wording from Open Question 3. Since under-minimum beats are head-trimmed, the useful motion must begin immediately. Concrete fix: append this exact sentence or equivalent to the Seedance-only clause: “Motion begins immediately in the first frame and remains gentle and continuous throughout.”

2. [Buildable Direction 1] The clause says “Preserve the wide 16:9 reference-image composition,” but the adapter may send `ratio` from `OTR_CLOUD_SEEDANCE_RATIO`, possibly `adaptive` per the plan. This can conflict with non-16:9 requests. Concrete fix: either only include “16:9” when `ratio == "16:9"` or change the clause to “Preserve the reference-image composition and framing.”

3. [Buildable Direction 2] Softener only covers the sample `music_open` but not other grounded risky phrases: `Slow handheld dolly forward`, `Slow orbit around the speaker`, `VU meters bounce`, etc. Concrete fix: at minimum add `handheld` -> `stabilized` and consider `orbit` -> `slow stabilized arc` for Seedance only. Keep this rule set small and explicit.

4. [Buildable Direction 3] “original prompt hash or excerpt” risks making logs insufficient or leaking full creative prompt text if implemented ad hoc. Concrete fix: log both a short hash and a bounded excerpt, e.g. first 160 chars with newlines collapsed. Avoid full prompt logging unless debug mode is enabled.

5. [Buildable Direction 3] “requested provider duration” and “requested Seedance duration” are duplicate/ambiguous. Concrete fix: use three distinct values:
   - `beat_duration_s`
   - `seedance_requested_duration_s`
   - `duration_clamped_to_provider_min: bool`

6. [Buildable Direction 4] Optional D says use explicit `16:9` instead of `adaptive`, but valid ratio values are not shown in grounding. Concrete fix: verify: accepted values for `model.ratio` in installed `ByteDance2ReferenceNode`. Only add D if `"16:9"` is accepted by that node.

7. [Buildable Direction 4] “cheap A/B” is not cheap if run across a full sequence; latest signal shows one Seedance render took about 2.5 minutes. Concrete fix: scope A/B to one representative shot first, e.g. `shot_b000_music_open`, before batch testing.

8. [Buildable Direction 5] Deferring reference-video guidance is fine, but the plan should explicitly avoid adding `reference_videos`, `auto_downscale`, or `auto_upscale` in this pass. Those fields are video-reference-only per grounding and should not be touched for still+audio prompt conditioning.

OPTIONAL / NICE-TO-HAVE:
- Add a small unit test for the Seedance prompt conditioner using the grounded `music_open` prompt and assert that forbidden phrases like `whip-pans`, `aggressively`, and `white-hot` are absent from the conditioned result.
- Add an env kill switch, e.g. `OTR_CLOUD_SEEDANCE_PROMPT_CONDITIONING=0`, if rapid rollback is needed. [ASSUMPTION] This project commonly accepts env knobs because the Seedance adapter already has several.

CUT THESE (over-engineering):
1. [Buildable Direction 3] Cut downstream `timeline_quality_report` integration from the first build unless the API already exists and is trivial to consume. It is not needed to test whether prompt conditioning improves raw Seedance motion.
2. [Buildable Direction 4] Cut variant D until aspect artifacts are actually observed and the valid `ratio` enum is verified. Prompt smoothness can be tested without changing aspect behavior.
3. [Buildable Direction 5] Cut reference-video guidance entirely for this pass. It changes input assets, cost profile, and partner request shape; prompt conditioning plus frame-count audit is the smaller change.