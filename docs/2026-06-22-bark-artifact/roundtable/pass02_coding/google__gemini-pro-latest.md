<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan relies on contextual data (dialogue vs intro) that the engine interface does not receive, and proposes a risky DSP per-chunk trim that will likely clip inter-sentence speech.

MUST-FIX BEFORE BUILD:
1. [B1 / WIRING] Context-blind Engine Interface. The plan wants to toggle `speech_only` and the `[clears throat]` injection based on whether a line is "DIALOGUE vs a genuine intro". `eng_bark.py`'s `generate_voice` signature is `(self, text, voice_preset, delivery_vector, seed)`. It receives no beat type, role, or scene context. Fix: Pass the toggle via the `delivery_vector` dictionary, or drop the dynamic toggle and make `speech_only=True` a global environment/config flag.
2. [B1] Misunderstanding of first-line guard. The plan assumes `[clears throat]` is injected on "intro" lines. `eng_bark.py` actually injects it on the *first time a voice_preset is used in the session* (`is_first = voice_preset not in self._presets_started`). Fix: Gate the `is_first_line` argument in `_generate_single_line` behind a global env var (e.g., `OTR_BARK_DISABLE_THROAT_CLEAR`) rather than trying to detect intro beats.
3. [B3] Ignored `seed` parameter. `eng_bark.py`'s `generate_voice` already receives a `seed` argument, but drops it (it calls `_generate_single_line` without passing it). Fix: Pass the existing `seed` from `generate_voice` down to `_generate_single_line` and apply it to the generator/torch.

SHOULD-FIX:
1. [B1] False premise on asterisk translation. The plan says "does NOT convert `*whistles*`/`*music*` stage-directions into them". `_clean_text_for_bark`'s `_ASTERISK_TO_BARK` list does not contain whistle or music anyway (it only has laugh, chuckl, sigh, gasp, groan, sob, cough, grunt). Fix: You only need to remove `[music]`, `[whistles]`, `[sneezes]`, `[gasps]` from `_BARK_VALID_TOKENS`.
2. [B2] Arbitrary whitespace splitting. Splitting on whitespace just because a sentence exceeds `max_len` will break Bark's semantic understanding mid-clause. Fix: Fallback split on commas or semicolons first, and only use whitespace as a last resort if a single word/phrase has no punctuation.

CUT THESE (over-engineering):
1. [B2] Per-CHUNK head+tail transient trim. Trimming transients *between* chunks (which are just sentences) using RMS energy will almost certainly clip valid plosives/fricatives at the start/end of sentences. The current `_trim_trailing_silence` is explicitly designed for the final tail because Bark leaves a 1s pad at the end of generation. Safe to cut: Rely on B1 preventing the non-speech tokens that cause the squeals in the first place.
2. [B3] "RESTORE prior RNG state after". The `eng_bark.py` docstring explicitly states `generate_voice` "Runs inside the caller's deterministic_inference wrap". The orchestrator is already managing the RNG state isolation. Safe to cut: Just apply the seed for the Bark generation step.

[ASSUMPTION]: I am assuming `test_audio_byte_identical` is not a Bark path based on `eng_bark.py` stating `indextts2 is now the char_voice default`. If it is a Bark path, changing the whitelist in B1 will break the fixture and require a