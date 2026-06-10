<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

## Executive read

The plan is much stronger than a typical “turn down sadness” fix. The big wins are: **observability first**, **vector-off A/B**, **real offending lines**, **loudness-matched listening**, **ref-bank curation**, and the new **per-line seed instability** hypothesis. I agree with the five-channel framing.

My main finding: the plan still slightly underweights two things software teams routinely miss:

1. **Reference “performance” content**, not just ref acoustics. A zero-shot prompt can carry stance: question endings, politeness, low projection, narration cadence, mic distance, and room. Metrics will not fully catch this.
2. **Post/mix psychoacoustics**. A perfectly acceptable solo TTS line can read whiny if it is dry, narrow, uncompressed, low in low-mids, and competing with a denser Kokoro/music bed.

Below are specific findings.

---

# 1. Blind spots / plausible missed causes

## 1.1 Reference prompt *performance* may dominate acoustic metrics

P2a audits RMS, F0, spectral slope, breath tail, SNR, etc. Good. But the plan may still miss **what the donor was doing emotionally** inside the 8–12s prompt.

A reference can be technically “healthy” and still teach the cloner:

- uptalk / phrase-final rises,
- polite audiobook narration,
- breathy close-mic intimacy,
- apology/help/question cadence,
- smiling voice,
- low vocal effort,
- narrow dynamic range,
- childlike or young-adult formant quality,
- “reading to the mic” rather than “speaking across a room.”

This is especially relevant because the current downloader selects the “longest voiced span” / fallback first segment and peak-normalizes it (`otr_dl_indextts2_refs.py`). It does not classify read style. The plan’s audition reel partially catches this, but the audit metrics will not.

**Falsifiable 30-min test:** for one donor, manually cut three 6–8s refs from the same source:

1. declarative / assertive segment,
2. question-heavy segment,
3. soft/narration segment.

Render the same neutral text, vector omitted, same seed. If output whine follows segment style, then curation needs a “prompt performance” tag, not only acoustic metrics.

Marking IndexTTS2 mechanism as **UNSURE**: I do not know whether IndexTTS2 explicitly copies prompt prosody/style beyond timbre, but zero-shot cloners commonly leak prompt performance.

---

## 1.2 Ref content may include linguistic/prosodic leakage

The plan mentions “whether prompt CONTENT conditions output prosody” in §8, but it is not operationalized enough. The ref bank could contain donors saying apologetic, questioning, or explanatory material. Even if IndexTTS2 does not receive prompt transcript, audio prosody can encode this.

**Suggested audit addition:** for each ref, generate a quick “style card” from audio only:

- final-rise count over last 300–700 ms of voiced phrases,
- F0 range / variance,
- average phrase length,
- speaking rate,
- pause density,
- breathiness proxy,
- ratio of rising terminals to falling terminals.

This is more directly “whine-risk” than median F0 alone.

---

## 1.3 “Hottest donor ref” is not necessarily a good baseline

P0 condition 1b uses “hottest healthy donor ref” as the loud-donor baseline. Because the downloader peak-normalizes to 0.97 and never loudness-normalizes (`otr_dl_indextts2_refs.py`), high RMS could mean:

- heavy compression,
- noisy room,
- clipped/limited source,
- close-mic proximity,
- denoising artifacts,
- loud background noise.

High RMS is not the same as high vocal projection.

**Better falsifiable baseline:** choose the “best-projection donor” by a small composite:

- LUFS/RMS in normal range,
- low clipping,
- high voiced SNR,
- strong 120–350 Hz male / 180–450 Hz female voiced energy,
- low nasal-band excess,
- low final-rise rate,
- operator quick listen: “assertive.”

Then compare against the quiet originals from G18.

---

## 1.4 Sample-rate / bit-depth / resampling may affect prompt timbre more than expected

G14 notes refs are 44.1 kHz while engine output is 22.05 kHz; P0 includes one pre-resampled 22050 mono ref. Good.

But I would also test:

- 44.1k PCM16 original,
- 22.05k PCM16,
- 22.05k float WAV if supported,
- 22.05k with gentle anti-alias low-pass,
- 44.1k loudness-normalized but not peak-normalized.

The downloader currently writes `PCM_16` at `OUT_RATE = 44100` and peak-normalizes to 0.97. If IndexTTS2 internally downmixes/resamples in a mediocre path, the ref may become thinner. **UNSURE** internals.

**Falsifiable:** same ref, same text, same vector-off, same seed, only ref format changed. Measure cloned output spectral tilt and listen.

---

## 1.5 Old stamped vectors can defeat P1 unless version handling is explicit

Current render code prefers any stamped vector:

```python
_dl = ln.get("delivery")
_stamped = _dl.get("emotion_vector") if isinstance(_dl, dict) else None
if isinstance(_stamped, dict) and _stamped:
    delivery_vector = _stamped
else:
    delivery_vector = deterministic_delivery_vector(...)
```

So if an existing ledger carries v1 delivery, P1’s new “derive from prepared text” behavior will not apply unless the code checks `delivery["version"]`.

The plan says `DELIVERY_TABLE_VERSION "v2"` and derives from prepared text, but it should explicitly say:

- accept stamped vector only if version matches active table/config, or
- restamp ledger before render, or
- expose “ignore stale stamped delivery vectors” behavior.

Otherwise the fix may appear not to work on already-frozen ledgers.

---

## 1.6 Full shared normalization does not actually remove all punctuation/prosody cues

The plan says adapter should adopt “full shared normalization” to fix G12. But `_otr_script_prep.prepare_text()` still keeps much punctuation in practice:

```python
t = t.replace("…", "...")
t = _MULTIDOT.sub("...", t)
```

It removes asterisks and music notes, but it does not obviously remove:

- `!`
- `:`
- `;`
- em-dashes
- all-caps
- quotation marks
- multiple question marks
- “?!”

The doc says it keeps `. , ?`, but the code shown does not enforce that. So “use full shared normalization” is not sufficient to remove punctuation prosody.

**Concrete check:** add a corpus residual report as planned, but make it fail/flag on actual characters left after prep. Do not assume `prepare_text()` already normalizes them.

---

## 1.7 Emotion vector order / semantics should be verified against actual IndexTTS2

The code assumes this order:

```python
("happy","angry","sad","afraid","disgusted","melancholic","surprised","calm")
```

The doc calls it the canonical contract, but the grounding does not prove this matches the installed IndexTTS2 model’s internal emotion-vector order. If even two axes are swapped, “calm” or “afraid” tuning could be misleading.

**UNSURE:** I do not know IndexTTS2 internals.

**Falsifiable:** render one line with one-hot vectors for each dimension at alpha 1.0, vector values otherwise zero, same ref/seed. Label perceptual result and compare to expected emotion names.

This is cheap and should be part of P0 cell 6 or a tiny preflight.

---

## 1.8 Gender-agnostic fallback can still create bad casting surprises

Grounding `_resolve_clone_ref_path()` has a gender-agnostic last resort:

```python
role_cands = [e for e in bank if e.engine == engine and role in e.roles]
...
entry = _random.Random(_seed).choice(cands)
```

So if gender is empty/other/unspecified or no gender match exists, a male lead could get any ref. The plan’s P2c/P3b says shared filtered bank, but I would explicitly preserve a “never cross major gender class for lead characters unless configured” policy.

Otherwise a mis-stamped cast row can become a “thin male” complaint that is actually a cross-gender or ambiguous-gender ref.

---

## 1.9 Writing-level pleading is under-measured

P0 has one neutral rewrite cell. Good, but one line may understate the issue. The writer may be generating lots of:

- “please,”
- “I’m sorry,”
- “don’t,”
- “can’t,”
- “must we,”
- “what if,”
- “I don’t know,”
- “help,”
- vocatives like “sir,” “my friend,”
- repeated questions.

Those are not just emotion cues; they are direct performance instructions to the TTS through text. `_otr_delivery_vector.py` includes “help,” “sorry,” “lost,” “alone,” “what,” “who,” “how,” etc., but “please” is notably absent. The raw text itself still reaches IndexTTS2 after only `clean_spoken_text()` in `eng_indextts2.prepare_text()`.

**Cheap metric:** compute per-character “pleading lexicon density” over the episode and correlate with whiny complaints. This does not reopen the writer; it only diagnoses.

---

## 1.10 Short-line behavior may be more important than the plan suggests

P0 includes one-word vs long line. Good. I would elevate this because radio drama has many short interjections. TTS models often default short utterances to:

- rising terminal,
- exaggerated surprise,
- childlike brightness,
- breathy onset,
- “checking for confirmation” cadence.

If Hayes/Gulliver have many short question-like lines, per-line rendering prevents cross-line context from anchoring stance.

**Falsifiable:** bucket existing generated lines by token count and compare final-rise / centroid / whine ratings. If <5-token lines dominate, fixes should target short-line punctuation/text prep and maybe seed mode more than global alpha.

---

## 1.11 Dynamic range and density mismatch against Kokoro

G15 notes Kokoro is dense/compressed while IndexTTS2 is dynamic. P0a is good. The blind spot is that **listeners often perceive low-density voices as pleading even when they are not higher-pitched**. If the announcer is louder, denser, and closer, the characters sound physically weaker.

Measure not only LUFS but:

- short-term loudness around dialogue,
- low-mid energy after mix,
- crest factor,
- modulation / RMS variance,
- spectral masking against the music bed,
- intelligibility-weighted loudness, not just full-band loudness.

A line can match integrated LUFS and still sound thin if the 150–350 Hz band is masked by music.

---

## 1.12 Determinism may be weaker than assumed on the IndexTTS2 path

Current `_render_per_line()` wraps inference with:

```python
with deterministic_inference(engine_seed, warn_only=True):
```

The comments explicitly say nondeterministic CUDA ops cannot crash the opt-in render. The hard constraint says deterministic re-render = same bytes for same code/config/seed. That may not currently be guaranteed for IndexTTS2.

This is not a whine cause, but it affects how confidently P0/P4 can compare outputs byte-wise.

**Falsifiable:** render the same 3-line P0 cell twice in separate Comfy runs and compare WAV hashes. If hashes differ, regression gates need acoustic/perceptual tolerance rather than byte identity for IndexTTS2, or the engine path needs stricter determinism work.

---

# 2. Craft: what a voice director / post engineer would do

## 2.1 Cast by “authority” and “vocal objective,” not gender/F0

Voice directors would not pick “male/warm/adult” uniformly. For leads they would ask:

- Does this voice sound like it wants something?
- Does it end phrases downward?
- Does it project across the room?
- Is it chest-led or throat/nose-led?
- Does it stay grounded under stress?
- Does it sound like a lead, not a narrator?

Automatable version:

Add curation tags after blind audition:

- `lead_safe`
- `authority`
- `low_uptalk`
- `projected`
- `intimate_risk`
- `nasal_risk`
- `young_risk`
- `narrator_safe`
- `comic_risk`
- `panic_safe`

For Hayes/Gulliver-style leads, require `lead_safe` + not `intimate_risk` + not `young_risk` if available.

---

## 2.2 Build “reference reels” from assertive prompt segments

If IndexTTS2 tolerates concatenated prompt audio, create a sha’d derived reference per good donor:

- 2–3 short declarative segments,
- no questions,
- no apologies,
- no laughter,
- no extreme emotion,
- consistent mic/color,
- 100–200 ms room/silence between segments,
- loudness-normalized,
- light de-noise only if needed.

This mimics how a voice director would choose a neutral “acting sample,” not a random first 12 seconds.

**UNSURE:** whether IndexTTS2 performs better with concatenated multi-phrase refs or a single continuous phrase. Easy to test.

---

## 2.3 Add a character dialogue bus, even if default-off

Engineers tend to tune TTS when the problem is a missing dialogue chain. A minimal deterministic local post chain could be:

- high-pass at 70–90 Hz,
- gentle low-mid shelf or bell around 150–300 Hz for male body,
- small presence control around 2–4 kHz if needed,
- de-esser if sibilant,
- 2:1 compressor, slow-ish attack, medium release,
- soft saturation / tape-style harmonic thickening,
- optional short room send.

Default-off is fine. But P0a should test this early because if it fixes 70% of the complaint, the TTS-side plan should shrink.

Automatable and deterministic with `scipy`/`numpy` filters if implemented carefully.

---

## 2.4 Mix for “physical placement”

A dry generated voice pasted onto music can sound needy because it has no room/body. Kokoro may sound fine because it is expected to be announcer-like and close.

Test one whiny clip with:

- -20 to -24 LUFS dialogue normalization,
- 1–2 dB bed ducking under dialogue,
- 80–120 ms short room impulse or synthetic early reflections,
- tiny slap/ambience, mono-compatible,
- low-mid EQ lift.

If the character suddenly sounds like an actor in a room, the issue is not only IndexTTS2.

---

## 2.5 Use “stance” text transforms, not only punctuation transforms

Voice directors give playable direction: “accuse,” “command,” “deflect,” “reassure.” Software plans often only remove `?`.

Without reopening the writer, a TTS-only deterministic text doctor can convert some short non-informational questions into more grounded delivery text:

- “What?” → “What.”
- “What now?” → “What now.”
- “You saw it?” → “You saw it.”
- “Help!” might stay urgent, but reduce vector fear.

The plan has punctuation variants, but consider a broader **short-line stance normalizer** gated by P0:

- questions under N tokens,
- no interrogative semantic need,
- no quote/legal exactness requirement,
- character lines only,
- after vector derivation or before depending on P0 result.

---

## 2.6 Treat short interjections as sound design, not normal dialogue

One-word lines like “No,” “What,” “Wait,” “Hayes!” are where TTS often overacts. A director would record these as separate pickups.

Automatable option:

- detect 1–3 token interjections,
- use lower alpha,
- remove terminal question mark unless semantically needed,
- force vector omitted or low arousal,
- optional fixed “interjection seed” per character,
- mix slightly louder/drier to avoid sounding weak.

This can be versioned and deterministic.

---

# 3. Critique of plan sequencing / risk

## 3.1 P-OBS first is correct

Strong agreement. Given G2/G9/G16/G17, tuning without per-line engine/ref/seed/vector logging is guesswork.

I would add one field:

- prepared text hash or first 60 prepared characters.

This helps catch punctuation normalization effects without dumping huge scripts.

---

## 3.2 P0-zero is useful but less important after P-OBS

The doc lists P-OBS first and then P0-zero, but P0-zero is described as “zero code.” Once P-OBS exists, P0-zero is almost redundant for fresh renders. Still fine as an operator sanity check.

---

## 3.3 P0 matrix is good but may be too broad for one sitting

The P0 matrix is scientifically good, but an operator may fatigue quickly. Listening fatigue will blur “whiny” judgments.

I would force a first-pass triage order:

1. vector omitted, current bad ref,
2. vector omitted, best assertive ref,
3. current vector alpha 1.0,
4. alpha 0.45,
5. real offending lines punctuation variants,
6. per-line vs fixed-character seed.

Only expand to calm/angry/resample if the first six do not explain enough.

---

## 3.4 Do not over-trust acoustic metrics as rejection rules

The plan says “NEVER auto-reject from metrics,” which is right. Keep that. Many good character voices are technically nasal, bright, creaky, or noisy but still dramatically useful. Conversely, clean narration refs can be disastrous for drama.

Use metrics to sort “must listen,” not to decide.

---

## 3.5 P1 “derive from prepared text” needs careful version semantics

As noted above, existing stamped v1 vectors will override fallback derivation unless code checks version. This is a practical implementation trap.

Also, “prepared text” is engine-specific if adapters can override `prepare_text()`. If delivery vector derivation uses IndexTTS2-prepared text, then changing the adapter prep changes emotion derivation. That is probably desired here, but version it as both:

- delivery table version,
- prepare text version.

The doc already mentions both hooks; just make the dependency explicit.

---

## 3.6 `OTR_INDEXTTS2_EMO_ALPHA` is probably the cheapest lever

Agree with the plan: alpha env default from P0 is a cheap dominant lever. I would ship this even before the table rewrite, provided default remains current behavior until P0.

But note: in `eng_indextts2.generate_voice()`, `emo_alpha` is hardcoded:

```python
"emo_alpha": 1.0,
```

So this change is trivial and low risk.

---

## 3.7 Punctuation lever is riskier than alpha

Gemini’s warning in the doc is right. Replacing `?` with `.` can make characters flat, depressed, or passive. It can also damage intelligibility/meaning.

Make it narrower than “character lines”:

- short non-informational questions only,
- no wh-questions that need query semantics,
- no emotionally meaningful accusations,
- no line ending with multiple clauses,
- exclude lines with “why/how/where/when” unless P0 proves benefit.

The doc says “scoped to short non-informational questions,” which is good. I would keep default OFF until after full-episode listen, not just P0 cell win.

---

## 3.8 Fixed-character seed may solve instability but create sameness

G17 is real: current seed is per line:

```python
engine_seed = _seed_to_int64(engine, request.stable_line_seed)
```

But using exactly one fixed seed for every line may create repetitive phrase contours or recurring artifacts.

A more voice-director-like compromise:

- character base seed + deterministic low-amplitude line sub-seed,
- or fixed seed per scene,
- or fixed seed for short/interjection lines only.

For P0, compare:

1. current per-line seed,
2. one fixed character seed,
3. fixed per-character base plus occurrence-derived offset if easy.

If only option 2 is feasible now, still test it, but listen for “same take every line.”

---

## 3.9 P2d loudness-normalizing the four quiet originals is justified but should be additive

G18 is important: the four curated originals are 2–6x quieter by RMS. But changing existing ref files in place can make old bank IDs misleading unless ref SHA/versioning is clean.

Safer:

- create new normalized entries with new IDs and SHA,
- or bump bank ID and update `ref_sha256`,
- keep old refs available until A/B proves the normalized ones are better.

The doc says versioned asset changes, new shas. Good. I would avoid silent replacement.

---

## 3.10 P2c/P3b shared tier filtering must cover the gender-agnostic fallback

Plan correctly says `_resolve_clone_ref_path` must consume the same filtered bank. I would explicitly include the “any ref” branch in the test cases:

- male char, valid male tier-a exists,
- male char, only rejected male exists,
- unspecified gender,
- exhausted pool,
- cast has valid stamped rejected ref,
- cast has stale `voice_ref_id`.

This is where bad voices can leak.

---

## 3.11 `allow_voice_reuse=false` defer is correct

Agree with the plan’s deferral. Current render-time fallback bypasses CastLock uniqueness/tier behavior (G16), so flipping the widget early can create false confidence.

---

## 3.12 P4 done bar is subjective but acceptable

“Hayes/Gulliver no longer read as whiny” is necessarily operator-judged. Good.

But add one quantitative non-gating dashboard:

- vector total mass,
- final 500 ms F0 rise,
- centroid,
- low-mid ratio,
- output LUFS,
- crest factor,
- line length bucket.

Use it to prevent regressions, not to define “good acting.”

---

# 4. Additional cheap experiments I would add

## 4.1 One-hot emotion axis sanity test

Because emotion order/semantics are assumed, render:

- same ref,
- same neutral sentence,
- same seed,
- one-hot each of 8 dimensions at 1.0,
- alpha 1.0 and maybe 0.5.

Expected: each dimension sounds roughly like its label. If “calm” or “surprised” behaves unexpectedly, P1 strategy changes.

This is especially important because the worker only sends `emo_vector` if nonzero:

```python
if ev and any(float(x) != 0.0 for x in ev):
    kwargs["emo_vector"] = [float(x) for x in ev]
```

So vector omitted, zero vector, and calm-only are meaningfully different conditions.

---

## 4.2 Ref subsegment A/B for same speaker

Instead of comparing different refs first, compare different cuts of the same ref source. This isolates performance leakage from speaker identity.

For one good donor:

- cut assertive falling-terminal segment,
- cut soft narration segment,
- cut question/rising segment,
- normalize same LUFS,
- render vector-off.

If whine follows the cut, bank curation should prioritize reference segment selection.

---

## 4.3 Output post-chain A/B before deep TTS work

P0a already includes free offline EQ/compression/loudness. I would make the test specific:

- loudness normalize character line to same short-term LUFS as Kokoro,
- add 2:1 compression,
- +2 dB wide bell around 180–280 Hz for male,
- -1 to -2 dB around 3–5 kHz if pleading edge,
- light saturation,
- bed duck -2 dB during line.

If that fixes >50% of perceived whine in mix, do not spend days overfitting emotion vectors.

---

# 5. Wildcards: three 30-minute speculative tests

## Wildcard 1 — Speculative: “authority sandwich” reference

Create a derived ref by concatenating three short assertive donor phrases from the same speaker:

- falling terminals,
- medium projection,
- no questions,
- no apology,
- matched LUFS,
- 100 ms silence between phrases.

Compare against the original 10–12s ref.

Hypothesis: the cloner averages toward a more grounded stance than with an arbitrary prompt segment.

Risk: concatenation artifacts may confuse speaker embedding. But it is cheap and versionable as a new SHA’d ref.

---

## Wildcard 2 — Speculative: parallel body layer in post

For one rendered thin male line, create a subtle parallel low-mid reinforcement:

- duplicate voice,
- low-pass around 350–500 Hz,
- add mild saturation,
- mix back at -18 to -24 dB,
- mono-safe.

This is an old post trick for “chest” without obvious pitch-shifting.

Hypothesis: perceived pleading drops because the voice gains physical mass, even if F0 is unchanged.

Risk: mud, phase, unnatural resonance. Do not ship without mono check.

---

## Wildcard 3 — Speculative: short-line “anti-uptalk” text variant

For 10 problematic short lines only, create deterministic TTS-only variants:

- original: “What?”
- declarative: “What.”
- grounded repeat: “What now.”
- no punctuation: “What”
- comma continuation: “What,”

Render vector omitted and alpha-calibrated.

Hypothesis: the punctuation/text parser, not the emotion vector, causes most short-line pleading.

Risk: semantic distortion. This should inform a narrow `short_question_policy`, not a global rewrite.

---

# 6. Highest-priority actionable findings

If I had to pick only five changes/checks before coding deeply:

1. **Add version-aware delivery-vector handling.** Do not let old stamped v1 vectors override v2 behavior silently.
2. **Run one-hot emotion axis sanity.** Verify the 8-dim order/semantics against the installed IndexTTS2.
3. **Test same-speaker ref subsegments.** Determine whether prompt performance leaks into output.
4. **Make P0a post-chain test concrete.** Compression/EQ/loudness/ducking may dominate.
5. **Do not assume shared `prepare_text()` fixes punctuation.** It currently leaves several prosody-significant characters unless changed.

Overall: the plan is directionally right. The biggest remaining risk is treating “whiny” as mainly an emotion-vector/casting bug when it may be a **performance-prompt + short-line punctuation + mix-density** interaction.