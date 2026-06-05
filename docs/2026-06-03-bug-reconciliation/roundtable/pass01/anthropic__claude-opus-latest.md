<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-4.8-opus-20260528 -->

VERDICT: no -- these are direction sketches with open questions, not a build-ready spec; the BUG-276 plan can be bypassed exactly as the 2026-05-31 soak was, and one of its proposed options violates PD1.

MUST-FIX BEFORE BUILD

1. [BUG-276] The crash happened with `bypass_freeze_halt=True`, which means the existing BUG-276 `needs_full_rerun` halt was bypassed. If the new reroute guard is added at that same halt/Gate-2 layer it will be bypassed identically and you will reproduce the soak crash. Fix: place the reroute at the per-line engine dispatch -- the chokepoint immediately before `BarkEngine.generate_voice` -- so it runs unconditionally, independent of any freeze/halt bypass flag. The fail-closed gate in `eng_bark.py` (the `if not voice_preset or not startswith("v2/")` raise, confirmed in grounding) then stays as last-resort. verify: location of the dispatch and whether `bypass_freeze_halt` short-circuits it.

2. [BUG-276] Drop the "skip-the-line" option. Skipping a narrative line shortens the full narrative audio, which is a direct PD1 violation ("never break, shorten, or degrade"). Use reroute-to-Kokoro only: a preset-less announcer/character line still produces audio via Kokoro, so length is preserved. The "assign-a-default-v2-preset" option should also be rejected (see CUT).

3. [BUG-264] The plan says "truncate `script_brief`/`news_close_brief` to their caps," but only `_MAX_SCRIPT_BRIEF_CHARS=350` is named. No cap constant is cited for `news_close_brief`. Before building, confirm/define `_MAX_NEWS_CLOSE_BRIEF_CHARS` (or that none exists and it is uncapped). A coercion that truncates to an undefined cap is unbuildable. verify: whether `news_close_brief` is a field with its own max and constant.

SHOULD-FIX

1. [BUG-264] Validator ordering and robustness: a `model_validator(mode="before")` receives the raw dict, where `key_terms` may be missing, `None`, or non-list, and brief fields may be non-str. Guard those types before slicing/truncating. Reuse the existing `_MAX_KEY_TERMS` / `_MAX_SCRIPT_BRIEF_CHARS` constants and confirm interaction with the BUG-307 `@field_validator` (before-model-validator runs first, then field validators; that ordering is fine but must be stated).

2. [BUG-264] "First N" trimming can discard the most relevant terms. Per the document's own V1 source-presence check, prefer keeping source-present `key_terms` first, then backfill to N -- otherwise the fix validates but degrades brief quality, partially defeating the point.

3. [BUG-295] The multi-word-only trigger misses single-word mid-phrase leaks. The example `safe in the ERIN SPENDER` is two words, but `safe in the ERIN` (one word, mid-phrase) would slip through and is the same defect. Document this as a known residual gap, or extend the trigger to single-word names that are NOT sentence-initial and NOT inside legitimate one-word drama -- carefully, given the stated false-positive concern.

4. [BUG-295] Prefer retry over in-place scrub. Scrubbing the leaked name leaves broken grammar (`safe in the` with the noun removed), degrading content. Use retry with a bounded budget (reuse the existing leak-filter retry budget at `_otr_line_composer.py` ~L1663-1692) and accept-on-exhaust so audio is still produced (PD1 length preserved). verify: the existing retry budget value.

5. [PD1 proof for 264 and 295] The document only asks for an inertness proof on BUG-276. BUG-264 (coercion) and BUG-295 (retry trigger) also feed TTS text. Add an explicit assertion to the regression suite that on clean inputs (strong-model output already within caps / no leak) the validators and the retry trigger DO NOT fire -- that is what keeps audio byte-identical on the clean path.

6. [BUG-276] State the interaction with the Gate-2 reviewer `apply_deterministic_cast_repairs`, which already refuses to remap a `speaker_role='character'` line onto the announcer row. The new guard does a remap in the opposite direction (announcer-tagged -> Kokoro); confirm the two cannot oscillate or contradict. verify: reviewer behavior for an announcer-tagged line with no preset.

OPTIONAL / NICE-TO-HAVE
- [BUG-276] Emit a structured warn-event when the reroute fires, so soaks surface "preset-less line rerouted" without the crash, giving you a signal that writer cast-lock is still leaking upstream.
- [BUG-295] Log the offending roster token on retry to feed prompt tuning.

CUT THESE (over-engineering)
1. [BUG-276] "assign-a-default-v2-preset" option -- cut. It silently gives a character/announcer line an arbitrary Bark voice, which is a content/voice-identity defect and contradicts the document's own statement that announcer lines belong to Kokoro. Reroute-to-Kokoro is the correct semantic and strictly simpler.
2. [BUG-295] The in-place scrub path -- cut. Retry alone meets the goal; scrub adds a second code path whose only effect is risking broken sentences (see SHOULD-FIX 4).

[ASSUMPTION] All claims about `news_interpreter.py`, `_otr_line_composer.py`, `_otr_ledger_reviewer.py`, the engine dispatch layer, and `bypass_freeze_halt` behavior are taken from the document; only `eng_bark.py` was provided as grounding. The `eng_bark.py` fail-closed gate and its `EngineUnusable(MALFORMED_CONFIG)` behavior are confirmed against grounding; the exact line number (78) is not verifiable from the excerpt.