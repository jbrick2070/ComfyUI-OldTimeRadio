<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Proposals leave core behaviors unspecified, risk PD1 byte-identity violations on audio-affecting paths, and contain no concrete guards/tests against the shown eng_bark.py failure mode.

MUST-FIX BEFORE BUILD:
1. [BUG-276] Proposed "pre-Bark routing guard" has no location or implementation; the only shown gate is the unconditional raise at eng_bark.py:78-82 inside generate_voice, so any slip past dispatch still crashes. Fix: insert the check (and reroute for announcer/char_id cases) in the single per-line dispatch before the generate_voice call; make the check a no-op when voice_preset already starts with "v2/".
2. [BUG-264] Coercion via @model_validator is described only as "extend the coerce pattern" with open questions on trim strategy and silent drop; this touches news_interpreter.py output that feeds the announcer script and therefore audio. Fix: either (a) declare the change audio-inert on the clean path with a byte-identity test, or (b) add the validator only under an env var and keep the existing retry ladder by default.
3. [BUG-295] Retry-loop change in nodes/_otr_line_composer.py is not specified (trigger exact regex, max retries, retry vs in-place scrub); a bad choice either regresses legitimate single-name drama or silently alters spoken text. Fix: limit scope to the multi-word ALL-CAPS + *...* case only, default to scrub (not retry), and add the regression case from the mistral-nemo example.

SHOULD-FIX:
1. [BUG-276] No regression test exists that would have caught the b018 announcer line under bypass_freeze_halt=True; add one that asserts zero announcer-tagged lines ever reach BarkEngine.generate_voice without a v2/* preset.
2. [BUG-264] Downstream effect on script_brief length after coercion is unstated; add a one-line check that the post-coercion brief still satisfies the original _MAX_SCRIPT_BRIEF_CHARS before it is used for audio generation.
3. [All three] PD3 requires env-var knobs for any wiring-adjacent change; the routing guard and composer changes must be behind such a knob or proven to touch zero INPUT_TYPES/widget/socket paths.

OPTIONAL / NICE-TO-HAVE:
- Add explicit "verify: news_interpreter.py field_validator and NewsBriefs usage sites" before claiming the BUG-264 coercion is safe.
- One-line note in each fix stating whether the clean-path audio bytes are proven identical.

CUT THESE (over-engineering):
1. "prompt-the-model-shorter" alternative in BUG-264: adds a second model call and violates "smallest correct change".
2. Any new model_id widget or cast-remapping table: directly violates PD6 and PD3.