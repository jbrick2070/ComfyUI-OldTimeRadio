# Bark artifact -- SMALL SPRINT PLAN (coding/wiring/QA converged; build AFTER Story-Quality R2)

Roundtable pass01 (source-side recipe) + pass02 (coding, GPT+Gemini+Grok grounded vs `_otr_bark_lib.py`
+ `eng_bark.py`). The panel made it SMALLER + corrected several seams. 3 chunks. Each: full suite + Bug
Bible -> commit + push. HARD: upstream-TTS only; audio SPINE frozen (master mix + mux-LAST +
`test_audio_byte_identical` untouched -- the byte-identical fixture is NOT a bark path: indextts2 is the
char_voice default, so B1's bark-output change does NOT break it -- VERIFY in B0); deterministic
seed-keyed; UTF-8 no BOM; SFW; 100% local.

## Converged corrections (from pass02, all code-grounded)
- The engine is CONTEXT-BLIND: `BarkEngine.generate_voice(self, text, voice_preset, delivery_vector,
  seed)` -- no beat/role/scene. So drive speech-only by the ROLE the engine already knows: it is the
  `char_voice` engine (`roles == ("char_voice",)`) -> all its lines are DIALOGUE -> default
  `speech_only=True`, plus an env override.
- `[clears throat]` is injected per-PRESET-FIRST-USE (`is_first = voice_preset not in
  self._presets_started`), NOT per-intro. Gate it behind `OTR_BARK_DISABLE_THROAT_CLEAR` (default ON
  for speech_only).
- The asterisk->token map has NO whistle/music (only laugh/chuckl/sigh/gasp/groan/sob/cough/grunt). So
  B1 is just SHRINKING the token WHITELIST `_BARK_VALID_TOKENS` -- remove `[music]`, `[whistles]`,
  `[sneezes]`, `[gasps]` (the squeal/whine sources) when speech_only; KEEP `[laughs]`/`[sighs]`.
- `seed` is ALREADY a `generate_voice` param but is DROPPED (never passed to `_generate_single_line`,
  which calls `model.generate(... do_sample=True ...)` unseeded). So Bark is NON-deterministic TODAY --
  threading the existing seed is a real bug fix, not new surface.
- CUT (panel): the per-chunk transient trim (RMS trim between sentence-chunks clips valid
  plosives/fricatives; the trailing trim exists only for Bark's ~1s end pad); the heavy
  spectral-centroid QA metric; the artifact-detect+reroll loop (B1 plummets the rate -- not needed).

## CHUNKS

### B0 -- fixture (no code)
Isolate the real "The Pencil Stays Down" ~0:24 clip: save the line text, the `_clean_text_for_bark`
OUTPUT, voice_preset, is_first(preset), temps, min_eos_p, the wav. CONFIRM the cleaned text carries a
`[music]`/`[whistles]`/`[clears throat]` (-> B1 prevents it entirely). VERIFY `test_audio_byte_identical`
is not a bark path.

### B1 -- speech-only dialogue mode (the whole prevention)
- `_clean_text_for_bark(text, *, speech_only=False)`: when True, drop `[music]`, `[whistles]`,
  `[sneezes]`, `[gasps]` from the kept-token set (keep `[laughs]`/`[sighs]`).
- `_generate_single_line(..., inject_first_line_anchor=True)`: when False, do NOT prepend
  `[clears throat]`.
- `eng_bark.generate_voice`: since `roles == ("char_voice",)` (dialogue), pass `speech_only=True` +
  `inject_first_line_anchor=False` by default; env overrides `OTR_BARK_SPEECH_ONLY` (default 1) +
  `OTR_BARK_DISABLE_THROAT_CLEAR` (default 1). Explicit kwargs threaded -- no implicit defaults.
- TESTS: `[music]`/`[whistles]` input -> stripped under speech_only, survives under False; first call to
  a preset emits NO `[clears throat]` under the gate; `[laughs]` survives.

### B2 -- deterministic seed (real bug fix; mandatory)
- Thread the EXISTING `seed` from `generate_voice` -> `_generate_single_line` -> apply before
  `model.generate` (VERIFY whether installed `BarkModel.generate` takes a `generator`; else
  `torch.manual_seed` for CPU+CUDA). The orchestrator already wraps `deterministic_inference`, so NO
  manual RNG save/restore needed (Gemini). TESTS: same seed -> identical clip; different seed ->
  different clip.

### B3 -- chunk split hardening (defensive, small)
- `_chunk_text_for_bark`: when a single sentence exceeds `max_len`, fallback-split on commas/semicolons
  FIRST, whitespace only as last resort (never mid-word). TEST: long no-punctuation input splits; very
  short final chunk handled.

## WIRING
All inside `_otr_bark_lib.py` + `eng_bark.py`. NO workflow-JSON / node / widget change. Flags are
engine-default (driven by the char_voice role) + env overrides. The byte-identical fixture is non-bark
(verify B0).

## FINAL QA
- A SIMPLE deterministic gate metric: high-band (>4 kHz) RMS / total RMS ratio in the first+last ~150 ms
  of a rendered clip, above a fixed threshold = flag (NOT spectral-centroid machinery). Add to the audio
  QA so a regression is catchable.
- Short bark-forced re-soak: a few legs, LISTEN + run the metric before/after -> artifact rate drops,
  laughs/sighs (if kept) intact, delivery unflattened.

## Build order
B0 (fixture/verify) -> B1 (speech-only -- the win) -> B2 (seed determinism) -> B3 (chunk split) -> QA.
Small; the bulk is B1 removing 4 tokens from a whitelist + gating one injection.
