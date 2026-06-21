# Bark artifact -- pass01 judgment (Claude = grounded judge)

Panel GPT-5.5 + Gemini-3.1-pro + Grok-4.3, grounded vs `_otr_bark_lib.py`. ~$0.10. The panel
OVERTURNED the problem statement's main hypothesis and found the artifact is PARTLY SELF-INFLICTED ->
it CAN be largely prevented at the INPUT side. That is the answer to the operator's question.

## CONFIRMED against the code (corrects the problem statement)
1. **`min_eos_p` was BACKWARDS (Gemini+GPT, code-verified).** `min_eos_p` is the threshold the EOS
   token must REACH to STOP. RAISING it makes Bark HARDER to stop -> LONGER hallucinated tails (the
   opposite of prevention). The docstring says 0.1 = "tightest + lowest variance"; a GPU sweep found
   0.2 WORSE. -> KEEP 0.1; do NOT raise. (Could test slightly lower, but 0.1 is already tuned.)
2. **We are TELLING Bark to make non-speech sounds (the real source).** `_clean_text_for_bark`
   WHITELISTS/PRESERVES non-speech tokens -- `[music]`, `[whistles]`, `[sneezes]`, `[gasps]`,
   `[clears throat]`, `[coughs]`, `[pants]`, `[sobs]`, `[grunts]`, `[groans]` (lines 353-355) -- AND
   CONVERTS asterisk stage-directions (`*whistles*`, `*music*`) INTO those tokens (lines 329-353). A
   line carrying any of these makes Bark GENERATE the squeal/whistle/music. The high-pitched artifact
   is exactly what `[music]`/`[whistles]`/`[sneezes]` produce.
3. **First-line `[clears throat]` is auto-injected** at the start-of-clip (the artifact-prone
   position) "to prevent podcast-intro hallucinations" (GPT+Gemini). It is itself audible non-speech.
4. **Deterministic reroll is NOT wired:** `_generate_single_line` uses `do_sample=True` with NO seed
   param / no torch.manual_seed -> "reroll with a derived seed" needs seed plumbing added first.
5. `_chunk_text_for_bark` does NOT enforce `max_len` on a long no-punctuation string (returns it
   whole) -> a hard punctuation/whitespace fallback split is needed.

## THE SOURCE-SIDE PREVENTION RECIPE (answer: largely YES, prevent at the input)
1. **SPEECH-ONLY mode for DIALOGUE lines (the entire-prevention lever).** For character/announcer
   dialogue, STRIP the HIGH-RISK non-speech tokens that cause the squeal -- `[music]`, `[whistles]`,
   `[sneezes]`, `[gasps]` (and the asterisk->token conversions that create them) -- so Bark is never
   TOLD to make them. KEEP the low-risk intentional emotive tokens (`[laughs]`, `[sighs]`) as a config
   choice. Make it a flag (default speech-only for dialogue).
2. **Disable / gate the first-line `[clears throat]` injection** (or make it opt-in) -- it sits in the
   exact artifact position and is itself non-speech.
3. **Keep `min_eos_p` at 0.1** (do NOT raise). Optionally test the short/first-line SEMANTIC temp at
   0.4 (the stage-temp helper isolates stages, so this won't flatten acoustics) -- evaluate, don't
   assume.
4. **Fix `_chunk_text_for_bark`** to hard-split overlong no-punctuation strings (defensive).
5. **FALLBACK for residual only:** per-CHUNK head+tail transient trim (the current trim is
   trailing-only on the concatenated line, so an internal-chunk squeal survives) -- but EXCLUDE the
   first-line anchor and intentional `[laughs]`/`[sighs]` from "artifact". The numpy-FFT high-band GATE
   is HEAVY and false-positive-prone (Gemini: CUT it; trim+speech-only will plummet the rate); if kept,
   it needs a full detector spec + intentional-non-speech exemption. For zero-tolerance broadcast,
   prefer kokoro.

## CUT (panel)
- Raising min_eos_p (backwards). The custom FFT high-band gate as a PRIMARY fix (heavy + false
  positives; speech-only + trim should suffice). Changing the frozen voice-preset selection.

## VERIFY-AT-BUILD (step 0)
Isolate the real 0:24 clip from "The Pencil Stays Down": save the original line text,
`_clean_text_for_bark` OUTPUT (does it contain `[music]`/`[whistles]`/`[clears throat]`?), the
voice_preset, is_first_line, temps, min_eos_p, and the wav -- as the bad fixture. CONFIRM the artifact
correlates with a non-speech token before tuning. (If the cleaned text shows a `[music]`/`[whistles]`
token, the speech-only recipe prevents it ENTIRELY.)

## Verdict
The artifact is largely SELF-INFLICTED (preserved non-speech tokens + the auto first-line
`[clears throat]`) -> a SPEECH-ONLY dialogue mode + dropping the first-line anchor prevents most of it
at the INPUT, with per-chunk trim as the residual fallback. min_eos_p stays 0.1. Build-ready once step
0 confirms the offending clip's cleaned text. Still a SEPARATE sprint from Story-Quality R2.
