# Bark high-pitched non-speech ARTIFACT (separate sprint -- banked 2026-06-22)

> Status: BANKED / NOT STARTED. Separate from the Story-Quality R2 coding sprint (the active step in
> GO_FORWARD_PLAN.md section 1). Pick this up as its own roundtable-then-build sprint later. Sanctioned:
> this is UPSTREAM-TTS work (the only audio work allowed while the audio SPINE stays frozen -- the master
> mix + mux-LAST contract is untouched; we only change how a Bark clip is GENERATED / post-trimmed before it
> enters the mix).

## Symptom (operator, real obs frame)
Episode "The Pencil Stays Down" (signal_lost_the_pencil_stays_down_20260621_053019), ~0:24: an awkward
HIGH-PITCHED noise / squeal that is NOT speech. Operator: "is it bark? if so can we ask bark not to do
it?"

## Call / diagnosis (Claude)
Almost certainly **Bark**. A high-pitched non-speech squeal/whine is Bark's single most common failure
mode: Bark is a GENERATIVE TTS, so alongside the words it occasionally hallucinates non-speech audio
([music], breath, squeals, a high whine). It is WORST at the START/END of a clip and on SHORT / FIRST
lines -- exactly where this artifact sits. This is DISTINCT from the earlier "whiny voice" timbre fix
(stage temperatures + recommended speakers): that fixed the VOICE being thin; this is a hallucinated
ARTIFACT.

You CANNOT reliably "ask Bark not to" -- there is no prompt token that suppresses it; it is in the
model's sampling. A prompt-only fix will not hold.

## Candidate fixes (to converge on in the sprint, ordered by leverage)
1. **Tight head+tail trim (cheap, deterministic, highest value).** Bark dumps most artifacts in the
   first/last ~100-200 ms. The existing trailing-pad trim (`cc6bacc`) only caught over-generation; add a
   head+tail silence/transient trim that clips the leading/trailing non-speech transient. Per-line,
   upstream of the mix.
2. **Lower the SEMANTIC temperature on short / first lines.** Those are the hallucination-prone ones (the
   stage-temp seam already exists: semantic 0.7 / coarse 0.5 / fine 0.5, with first-line/intl caps). Drop
   the semantic cap further for very short or first lines -> fewer hallucinations (reduces, not eliminates).
3. **Deterministic high-band artifact GATE + reroll (the "catch it every time" fix).** Scan the rendered
   Bark clip for a sustained high-frequency band spike with little/no speech energy; on a hit, regenerate
   that ONE clip with a different seed (Bark is seed-varied, so a re-roll usually comes out clean). Bounded
   retries; LOUD on exhaustion. Per-line, upstream of the mix.
4. **Broadcast lever: prefer KOKORO.** Kokoro is deterministic (not generative) and effectively never does
   this. It is already in the soak's voice rotation. For the OBS->YouTube loop, weight toward kokoro and
   treat Bark as the "character color" option behind the gate (#3).

## Constraints
- Audio SPINE frozen: the master mix + mux-LAST + `test_audio_byte_identical` contract is UNTOUCHED. We only
  change Bark clip GENERATION / per-line post-trim BEFORE the clip enters EpisodeAssembler.
- Deterministic / seed-keyed (a reroll uses a derived seed, reproducible). UTF-8 no BOM. SFW. 100% local.
- Model-agnostic-ish: the gate (#3) can also catch the rare kokoro/other glitch, but the trim/temp levers
  are Bark-specific.

## >>> ROUNDTABLE FOCUS (operator 2026-06-22): can we prevent it ENTIRELY at the INPUT / PROMPT side?
Before the post-trim/gate work, pressure-test the GENERATION INPUTS. The Bark lib (`_otr_bark_lib.py`)
already exposes the relevant surface -- the question is whether tuning these prevents the high-pitched
hallucination AT THE SOURCE:
- `_clean_text_for_bark` (text prep): strips parens/asterisks/bracket tags, transliterates. Does any
  residual token (a stray bracket tag, a trailing symbol, an empty/short string, a number, an ellipsis)
  still trigger Bark's non-speech [music]/squeal? Should it be stricter / normalize differently?
- `min_eos_p` (`OTR_BARK_MIN_EOS_P`, `_resolve_min_eos_p`): Bark continues generating until EOS; a LOW
  min_eos_p lets it run PAST the speech into a hallucinated tail (the classic high whine). Is RAISING it
  the entire-prevention lever? What value is safe (doesn't clip real speech)?
- `_stage_temps_for_line` (semantic 0.7 / coarse 0.5 / fine 0.5 + first-line/intl caps): is the SEMANTIC
  temperature the hallucination driver -- does a lower semantic temp (esp. first/short lines) prevent it
  without flattening delivery?
- `_chunk_text_for_bark` (max_len 180): do chunk boundaries / very short final chunks spawn the artifact?
- Voice PRESET (history_prompt): do certain speaker presets hallucinate more? (the audition found
  en_speaker_5/0/6 fullest -- is there an artifact-rate angle too?)
- Is there a Bark gen arg we are NOT setting that suppresses non-speech (e.g. a stricter EOS, a
  speech-only bias, a shorter max_length per the cleaned text length)?
Goal: a SOURCE-side recipe (text-prep + min_eos_p + temps) that prevents the artifact, falling back to
the post-trim/gate only for the residual. Panel = 2-3 frontier LLMs; Claude grounds vs `_otr_bark_lib.py`.

## Suggested sprint shape (when picked up)
Problem statement (this) -> roundtable (panel + Claude grounding vs the Bark engine `eng_bark` /
`_otr_bark*` + the per-line audio path) -> 1-3 green chunks (trim; high-band gate+reroll; optional temp
tweak) -> a short audio re-soak listening pass. Add a deterministic high-band-artifact metric to the
audio QA so regressions are catchable.
