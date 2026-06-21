# Bark artifact -- SMALL SPRINT build-plan draft (coding + wiring + QA pass input)

Source-side prevention (pass01_judgment): the high-pitched artifact is partly SELF-INFLICTED -- we
PRESERVE/CREATE non-speech Bark tokens + auto-inject first-line `[clears throat]`. Prevent at the
INPUT; per-chunk trim as residual. Separate sprint, runs AFTER Story-Quality R2. Each chunk: full
suite + Bug Bible -> commit + push. HARD: upstream-TTS only; audio SPINE frozen (master mix + mux-LAST
+ `test_audio_byte_identical` contract untouched -- the byte-identical fixture is NOT a bark path,
verify); deterministic / seed-keyed; UTF-8 no BOM; SFW; 100% local.

## LOCATED SEAMS (grounded vs _otr_bark_lib.py)
- `_clean_text_for_bark` (~:271-389): cleans text; WHITELISTS non-speech tokens (`[music]`/`[whistles]`/
  `[sneezes]`/`[gasps]`/`[clears throat]`/...; ~:353-355) + converts `*action*` -> those tokens (~:329-353).
- `_generate_single_line` (~:412+): per-line gen; auto-prepends first-line `[clears throat]`;
  `do_sample=True` with NO seed param / no torch.manual_seed.
- `_chunk_text_for_bark` (~:391): max_len=180 but does NOT split a long no-punctuation string.
- `_trim_trailing_silence` (~:454): trailing-only, on the CONCATENATED line.
- `_stage_temps_for_line` (~:411): semantic 0.7 / coarse 0.5 / fine 0.5 + first-line/intl caps.
- `_resolve_min_eos_p` (~:442): `OTR_BARK_MIN_EOS_P`, default 0.1 (tuned -- DO NOT raise).
- Engine adapter `_otr_audio_engines/eng_bark.py`: where the per-line path + any new flag is read.

## CHUNKS (small)

### B0 (STEP 0 -- fixture, no code): isolate the real failing clip
From "The Pencil Stays Down" ~0:24: save the line text, the `_clean_text_for_bark` OUTPUT (does it
contain `[music]`/`[whistles]`/`[clears throat]`?), voice_preset, is_first_line, temps, min_eos_p, the
wav. CONFIRM the artifact correlates with a non-speech token before tuning. This pins the fix + the QA
metric.

### B1 -- SPEECH-ONLY dialogue mode (the entire-prevention lever)
- `_clean_text_for_bark`: a `speech_only=True` path (default for character/announcer DIALOGUE) that
  STRIPS the high-risk non-speech tokens (`[music]`, `[whistles]`, `[sneezes]`, `[gasps]`) and does NOT
  convert `*whistles*`/`*music*` stage-directions into them. KEEP `[laughs]`/`[sighs]` (low-risk,
  intentional) behind a config flag. NB: the music_inter beat text is separately suppressed by R2-S1, but
  a stray inline cue still reaches a dialogue line -- this catches it.
- `_generate_single_line`: gate the first-line `[clears throat]` injection behind a flag (default OFF for
  dialogue, or keep only when the line is genuinely an intro). It sits in the artifact-prone start spot.
- TEST: a line with `[music]`/`*whistles*` -> speech_only output has NO non-speech token; `[laughs]`
  survives when kept; first-line output has no auto `[clears throat]`.

### B2 -- chunk hardening + per-chunk trim (residual)
- `_chunk_text_for_bark`: hard fallback split at safe punctuation/whitespace when a single sentence
  exceeds `max_len` (today it returns the overlong string whole). TEST: long no-punctuation input splits;
  very short final chunk handled.
- Per-CHUNK head+tail transient trim BEFORE the inter-chunk silence (the current trim is trailing-only on
  the concatenated line, so an internal-chunk squeal survives). Bounded max-trim; protect plosives/
  fricatives + a min retained-speech floor; EXCLUDE the (now-gated) first-line anchor. TESTS: leading
  squeal trimmed, trailing squeal trimmed, leading consonant kept, all-silence, short clip.

### B3 -- (OPTIONAL) deterministic reroll plumbing
- Add a `seed`/`attempt` kwarg to `_generate_single_line`; derive the seed from a stable line key + retry
  index; set torch CPU/CUDA RNG (or pass a generator if `BarkModel.generate` supports it -- VERIFY
  transformers support) and RESTORE prior RNG state after. ONLY then is a "reroll the bad clip" loop
  reproducible. TEST: same seed => identical clip; attempt N => deterministic alternate. (Defer the
  actual artifact-detect+reroll loop -- B1+B2 should plummet the rate; keep this as the hook.)

### min_eos_p / temps -- NO CHANGE (corrected)
Keep `min_eos_p` 0.1 (raising it lengthens hallucinated tails -- backwards). Optionally evaluate the
short/first-line SEMANTIC temp at 0.4 in B1 testing, but do not assume; the 0.5/0.6 caps may suffice.

## WIRING
All inside the Bark lib + `eng_bark.py` (+ how the engine knows a line is DIALOGUE vs an intro to set
`speech_only`/first-line-anchor flags). NO workflow-JSON / node / widget change. New flags are env or
engine-config, reusing the per-line path. VERIFY the `test_audio_byte_identical` fixture is NOT a bark
render (so B1's output change doesn't break it); if it is, re-baseline in the same chunk.

## FINAL QA
- A deterministic NON-SPEECH/high-band metric on a rendered clip (e.g. spectral centroid > 4-8 kHz with
  low speech-band energy, in first/last 150 ms) added to the audio QA -- so a regression is catchable.
- Short audio re-soak: a few bark-voice legs (force bark), LISTEN + run the metric before/after; confirm
  the artifact rate drops and normal delivery (laughs/sighs if kept) is intact.

## Open questions for the coding panel
- Q1: exact seam where the engine knows a line is DIALOGUE vs a genuine intro (to default speech_only +
  gate the first-line `[clears throat]`) -- `eng_bark` per-line call, or the line's speaker_role/beat?
- Q2: which non-speech tokens are HIGH-RISK (squeal/music/whistle) vs SAFE (laughs/sighs) -- confirm the
  strip list from the whitelist (~:353-355).
- Q3: `BarkModel.generate` seed/generator support in the installed transformers (B3) -- generator kwarg,
  or torch.manual_seed + RNG restore?
- Q4: is the `test_audio_byte_identical` fixture a bark path? (if yes, B1 re-baselines it.)
- Q5: per-chunk trim algorithm params (window, max-trim ms, freq/energy condition, min retained speech).
