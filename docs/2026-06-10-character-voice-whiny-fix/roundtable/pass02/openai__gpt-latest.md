<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

## Overall read

The v2 plan is much stronger than the first-pass root-cause framing. It now covers the three big obvious channels: bad/weak refs, overactive emotion vectors, and punctuation prosody. The most important remaining blind spot is that **“whiny” may be a stochastic/prosodic consistency problem, not only a ref/vector problem**: the current render path seeds every line independently from `stable_line_seed` in `_otr_voice_node_common.py`, so the same character can get a different sampled delivery contour every line even with the same reference. For a zero-shot expressive model, that can read as nervous/pleading even when the reference and emotion vector are acceptable. IndexTTS2 internals here are **UNSURE**, but it is cheap and falsifiable.

Below are findings, not a rewrite.

---

# 1. Blind spots / plausible missed causes

## A. Per-line stochastic prosody / same-character instability

Grounding: `_render_per_line` builds a `ResolvedVoiceRequest` per line, then derives:

```python
engine_seed = _seed_to_int64(engine, request.stable_line_seed)
...
audio = adapter.generate_voice(prepared, voice_ref, delivery_vector, engine_seed)
```

So two adjacent Hayes lines use different seeds. P3b fixes per-character ref resolution, but **does not fix per-character prosody seed stability**.

Why it matters: a zero-shot TTS model may sample pitch contour, intensity, pauses, breathiness, and emotional stance from its random seed. If every short line has a fresh seed, the character can sound like a different “take” each time: hesitant here, pleading there, abruptly bright on the next line. That can be perceived as whiny even if no individual line is terrible.

Falsifiable test:

- Same ref, same text, vector omitted, render 10 seeds.
- Then render a 6-line Hayes panel with:
  - current per-line seeds;
  - one fixed per-character seed;
  - fixed per-character seed plus line-index offset only if needed.
- Loudness-match and listen for “character steadiness.”
- If the fixed-char-seed version feels more grounded, add a versioned seed policy option, e.g. `OTR_INDEXTTS2_SEED_MODE=line|character`.

IndexTTS2 sampling behavior: **UNSURE**, but this is a cheap behavioral probe.

---

## B. Actual problematic lines/refs may be absent from P0

P0 uses 3 hand lines and a best/worst ref. That isolates mechanisms, but it may not reproduce the exact failure. The real failure is Hayes/Gulliver in a real episode, with real line text, real punctuation, real cast ref, real seed, and real mix context.

Grounding: G9 says resolved `voice_ref_id` is not persisted; P0-zero reads CastLock report/render_log, but not per-line resolved fallback state. Current code only logs a general line count and one bark warning if needed.

Falsifiable addition to P0:

- Include **2–3 real offending lines** from the episode, not just synthetic neutral/question/fear lines.
- Use the exact resolved ref if discoverable from CastLock report.
- Use the production line seed if reproducible.
- Compare:
  - exact current path;
  - vector omitted;
  - punctuation softened;
  - same text with fixed per-character seed;
  - same text through offline post chain.

Without this, P0 may prove a lever works on lab lines but not on the actual “whiny” perception.

---

## C. Text normalization gaps beyond `?`, `!`, `...`

The plan catches G12/G13, but the normalization problem is broader than ellipses and question marks.

Grounding: IndexTTS2 currently calls only:

```python
from .._otr_script_prep import clean_spoken_text
return clean_spoken_text(text)
```

`clean_spoken_text` strips:

- uppercase speaker prefix matching `_SPEAKER_PREFIX`;
- parentheticals only up to 80 chars;
- bracket tags only up to 40 chars;
- whitespace.

It does **not** normalize or remove:

- em dashes / double dashes;
- semicolons;
- colons;
- quoted fragments;
- all-caps words;
- numerals/acronyms;
- slashes;
- long parentheticals over 80 chars;
- long bracketed tags over 40 chars;
- mixed-case speaker labels like `Hayes:` if they occur;
- stage directions with unmatched parentheses/brackets.

Why it matters: TTS models often turn weird punctuation into uncertainty, pauses, upward inflection, or spelling behavior. A line like:

> “Hayes: Wait—what is it? I—I thought…”

can be pulled toward stammer/pleading by punctuation alone, even with emotion vector off.

Concrete audit:

- Run a corpus report over spoken character lines after `clean_spoken_text` and after full `prepare_text`.
- Count residual chars outside `[A-Za-z0-9 .,?!'-]`.
- Print top 50 lines containing em dash, colon, quote marks, slash, brackets, long parentheticals, all-caps tokens.
- Listen to a few before/after normalization.

---

## D. Micro-lines and contextless TTS

The plan mentions optional short-line dampening, but the bigger issue is **per-line contextlessness**. Radio scripts often contain one-word or two-word lines:

- “Wait!”
- “What?”
- “No.”
- “Hayes—”
- “Listen!”

Rendered in isolation, a TTS model often exaggerates the contour because it has no surrounding dramatic context. This can sound pleading or yelping even with a neutral vector.

Grounding: `_render_per_line` sends only one prepared line to `generate_voice`; no preceding/following text is available to the model.

Falsifiable tests:

1. Render the same short line alone vs embedded in a neutral carrier:
   - alone: `What?`
   - carrier: `I said, what?`
   - carrier then trim is more complex, but as a quick test it reveals whether contextlessness is the problem.
2. Detect lines under e.g. 4 words and compare:
   - current punctuation;
   - terminal `?` softened;
   - vector omitted;
   - lower alpha;
   - fixed seed.

Automatable mitigation if confirmed:

- Strongly damp delivery vectors for lines under N words.
- Convert isolated interjections to calmer text forms, e.g. `Wait.` not `Wait!`, unless line metadata marks panic.
- Optional versioned “micro-line debleat” rule.

---

## E. Ref content/read-style: the spoken words inside the prompt may matter

The plan correctly says zero-shot cloners copy read style. One additional angle: the **reference clip’s linguistic/prosodic content** may include questions, apologies, hesitations, laughter, breathy reads, or donation-script politeness. Even if IndexTTS2 nominally uses only audio prompt speaker embedding, it may condition on prosodic style in the prompt. Whether it conditions semantically on prompt content is **UNSURE**.

Grounding: `scripts/otr_dl_indextts2_refs.py` trims the longest voiced span; it does not classify the prompt text/content, because transcripts are unavailable.

Potential issue:

- A donor segment selected for “longest voiced span” could be a polite prompt, customer-service tone, list reading, question, or laugh-heavy segment.
- Concatenated refs from LJ/Rhasspy may contain multiple utterances with resets, not a single steady performance.

Falsifiable tests:

- For each candidate ref, measure end-of-utterance F0 rise, speech rate, pause density, voiced/unvoiced ratio.
- More importantly: audition through the cloner, as P2b proposes.
- Add an explicit “prompt style” operator grade: `assertive`, `neutral`, `polite`, `breathy`, `uptalk`, `narrator`, `reject`.

---

## F. Reference acoustic quality: missing SNR/reverb/room/color metrics

P2a has good metrics, but it still misses several things that strongly affect “thin/whiny” zero-shot cloning:

- SNR / noise floor during pauses.
- Hum/buzz.
- Room reverb / late reflections.
- “Enhanced” denoiser artifacts from kyutai `_enhanced.wav`.
- DC offset.
- Spectral tilt / high-frequency harshness.
- Cepstral peak prominence / harmonic-to-noise ratio as a proxy for breathiness.
- Plosive/ess energy.
- Clipping after peak normalization.

Grounding: current `classify_and_trim` only returns gender, median F0, voiced fraction, and the trimmed waveform. It peak-normalizes to 0.97.

Why it matters:

- A noisy/breathy prompt can make the clone breathy.
- A close bright mic can make the clone nasal/teen-like.
- A denoised prompt can produce metallic thinness.
- Peak normalization can make two prompts with equal peaks but radically different perceived loudness/body.

Concrete P2a additions:

- Integrated LUFS or at least gated RMS.
- Pause RMS vs voiced RMS as SNR proxy.
- Spectral tilt: low/mid/high energy ratios.
- HNR/CPP if easy through librosa/parselmouth.
- Reverb proxy: decay tail energy after utterance offsets.
- DC offset.
- Peak count near full scale and intersample-ish risk.

---

## G. “Chest-weight” metric can be fooled

P2a proposes:

> chest-weight (80–300 Hz energy / total RMS)

This is useful, but not enough.

Failure modes:

- 80–150 Hz energy can be HVAC rumble, mic handling, plosives, or proximity mud.
- For female refs, a low 80–300 ratio does not necessarily mean thin.
- A nasal male can still have strong low-mid energy.
- Peak-normalized refs make spectral ratios unstable if there is noise or silence.

Better:

- Compute spectral ratios on **voiced frames only**, not whole file.
- Use a band like 120–350 Hz for adult male body and 180–450 Hz for adult female warmth, or report both.
- Add spectral slope from 200 Hz to 4 kHz.
- Add high-mid harshness ratio, e.g. 2–5 kHz / 200–800 Hz.
- Treat the composite as “must-listen priority,” not a reject rule.

---

## H. Possible sample-rate/profile mismatch in the pack path

Grounding:

- IndexTTS2 adapter declares `sample_rate = 22050`.
- Worker emits `{"sample_rate": 22050}`.
- `_load_wav` actually reads the WAV sample rate from file and returns it.
- `_render_per_line` later sets `sr = int(profile.sample_rate or sr)` and packs with `pack_audio_batch(clips, sample_rate=sr, mono=mono)`.

If `profile.sample_rate` ever differs from the actual IndexTTS2 file sample rate, there is a risk of mislabeled sample rate or resampling omission. I cannot verify `pack_audio_batch` here, so this is a **verify-at-build** concern, not a claimed bug.

Falsifiable check:

- Log actual returned `audio["sample_rate"]` per clip for IndexTTS2.
- Assert all primary clips equal the pack `sr`.
- If not, resample primary clips deterministically like the bark fallback path already does.

A subtle rate mismatch can change perceived pitch/speed, which absolutely can read as “thin” or “childlike.”

---

## I. Perceived whine from writing, not synthesis

The plan mostly treats this as audio. But if the generated script gives Hayes/Gulliver lots of lines like:

- “Please…”
- “I don’t know!”
- “What are we going to do?”
- “I’m sorry.”
- “We can’t!”

then even a good voice will sound pleading. The current keyword table catches some of these as emotion, but the lexical content itself also matters.

Falsifiable split:

- Render a neutral/assertive rewrite of a known whiny line with the same ref/seed/vector omitted.
- If it sounds fine, the “voice” complaint is partly writer/dialogue style.
- Add a script-side report: per character, count interrogatives, apologies, fear words, negative helpless constructions.

This does not mean reopening the writer, but it prevents overfitting the TTS path.

---

# 2. Craft: what a voice director / post engineer would do

## A. Cast for role, not gender

Software instinct: “male/female pool, random choice.”  
Voice-director instinct: “Who is the lead? Who carries authority? Who can survive emotional scenes?”

For a one-person automated shop:

- Create 3–5 fixed casting archetypes:
  - `lead_male_authority`
  - `lead_male_neutral`
  - `comic_minor`
  - `fragile_minor`
  - `lead_female_authority`
- Use P2b audition reels to tag refs into these archetypes.
- For Hayes/Gulliver or recurring leads, draw only from `lead_safe` / `authority` refs.
- Let variety collapse for leads. A repeated good lead voice is better than a varied weak one.

This aligns with P2c but I would make “lead authority” more explicit than generic `quality_tier`.

---

## B. Build a dialogue bus, not just TTS clips

Post engineers do not judge raw TTS in isolation. They put dialogue on a bus:

- loudness normalization;
- high-pass cleanup;
- low-mid warmth;
- compression;
- de-essing;
- maybe mild saturation;
- bed ducking.

Automatable deterministic chain, default-off first:

1. Per character clip:
   - trim leading/trailing silence conservatively;
   - normalize to target dialogue LUFS or RMS;
   - true-peak limit.
2. Dialogue bus:
   - high-pass around 70–90 Hz;
   - gentle low-mid shelf around 150–250 Hz if thin;
   - broad dip around 1–2.5 kHz if nasal/pleading;
   - compression around 2:1, slow-ish attack, medium release;
   - limiter.
3. Music/FX:
   - deterministic sidechain ducking under dialogue, or static bed attenuation during speech.

The plan’s H2c “free post test” is excellent. I would run it early, because if a simple EQ/compressor makes the operator say “70% fixed,” then P1/P2 should be smaller.

---

## C. Direct the emotional arc, not each line independently

A voice director would not make every line maximize its local emotion. They would preserve authority and use contrast.

Automatable idea:

- Add per-character or per-scene “stance”:
  - leads default to `controlled`;
  - panic/fear requires stronger evidence;
  - questions are informational unless paired with panic cues.
- For leads, cap afraid/sad/surprised harder than for minor victims.
- Give authority characters a mild assertive prior only when P0 proves it helps:
  - not necessarily `angry=0.15`; maybe “low arousal, low surprise, zero fear.”
- Smooth emotion over adjacent lines:
  - do not let a single `?` spike surprise if previous/next lines are neutral.

This is more “direction” than keyword detection.

---

## D. Loudness-match audition reels and blind the operator

P2b audition reels are a strong idea. To make them useful:

- Randomize candidate order.
- Hide `voice_ref_id` names.
- Loudness-match all examples.
- Include at least one repeated hidden sample to measure operator consistency.
- Grade on separate axes:
  - body;
  - authority;
  - nasal/whiny risk;
  - emotional range;
  - artifact/noise;
  - identity stability across lines.

This prevents “I know this one is the suspect ref” bias.

---

## E. Measure final-F0 rise on generated lines, not only refs

A director hears “pleading” largely as pitch contour:

- high average F0 for the character;
- upward terminal contour;
- narrow, tense vibrato;
- weak low-frequency energy;
- breathy onset.

P2a measures end-of-line F0 rise on refs, but you should also measure generated lines.

Automatable metric:

- For each rendered character line:
  - median F0;
  - final 500 ms F0 slope;
  - F0 range;
  - RMS;
  - spectral centroid;
  - duration vs text length.
- Flag lines with high final rise + question punctuation + afraid/sad/surprised mass.

This gives a regression metric for “de-bleat” changes.

---

# 3. Critique of the plan

## P0-zero is right, but insufficiently diagnostic

P0-zero proves “all lines on indextts2” vs bark fallback for fresh renders. Good first action.

However, current render_log only says:

```text
char_voice: rendering N lines on 'indextts2'
...
WARNING ... rendering those on 'bark'
```

It does not map each character/line to ref. So P0-zero can close H4 but not H1. A fresh all-IndexTTS2 render may still have Hayes on a bad ref or inconsistent render-time auto assignment.

Cheaper improvement:

- Before large P1/P2 changes, add a minimal render_log line per character:
  - `char_id -> voice_ref_id/ref_path -> engine`
- This is part of P3a, but I would do the render_log-only subset earlier. It is low-risk and immediately improves all diagnosis.

---

## P0 matrix should include fixed seed / seed sweep

Current P0 includes 3 seeds. Good. But it does not explicitly test the production issue: per-line seed variation and same-character consistency.

Add one condition:

- 6-line panel, same ref, vector omitted, current line seeds vs fixed character seed.

If this wins, it may dominate alpha/table tweaks.

---

## P1 “pure-neutral text -> calm-dominant >= 0.7” may be wrong

This is the most important plan concern.

Grounding:

- G8 notes that true neutral A/B exists via `OTR_DELIVERY_VECTOR=0`, where the worker omits `emo_vector` entirely.
- Worker only passes `emo_vector` if any value is nonzero:

```python
ev = req.get("emo_vector")
if ev and any(float(x) != 0.0 for x in ev):
    kwargs["emo_vector"] = [float(x) for x in ev]
```

- Under current default, a neutral line sends `calm=1.0` at `emo_alpha=1.0`.

The doc already says calm semantics are unverified. Therefore a property test requiring neutral text to be “calm-dominant” risks baking in the very behavior P0 may disprove. If “calm” is active suppression, softness, meekness, or low-energy breathiness in IndexTTS2, then calm-dominant neutral could contribute to thin/pleading delivery.

Safer acceptance criterion:

- Neutral text should produce either:
  - all-zero vector / omitted vector; or
  - whatever P0 cell 4 proves is perceptually neutral.
- Do not assert `calm >= 0.7` until P0 confirms calm is desirable.

I would change the property from “calm-dominant” to “neutral does not send aroused dims; neutral-vector policy equals P0 winner.”

---

## Exposing `OTR_INDEXTTS2_EMO_ALPHA` is likely the cheapest high-value step

This is correctly identified. It is adapter-side only:

```python
"emo_alpha": 1.0,
```

Even before table v2, alpha gives immediate control over H2. But beware: if punctuation or bad refs dominate, alpha may appear to “fail” even though it is useful.

Recommended test interpretation:

- If vector-off fixes only 30%, alpha is not enough.
- If vector-off fixes 70%, alpha should ship immediately.
- If vector-off does nothing, prioritize punctuation/ref/mix/seed.

---

## Punctuation lever: do not default terminal `?` -> `.` too broadly

The plan keeps default OFF, which is correct.

Risk: flattening real questions can make dialogue unnatural or sarcastic. Also the vector still sees the question, so a softened `? -> .` with surprise/afraid vector may create a mismatched delivery: textual declarative with emotional question contour.

Better if promoted:

- Apply only to short, panic-prone, non-informational questions:
  - “What?”
  - “Who’s there?”
  - “How?”
- Do not apply to exposition questions:
  - “What do you mean by the signal?”
- Or implement “question mark dampening” in vector first; text rewrite second.

---

## P2a audit is useful, but do not overtrust the composite rank

Acoustic metrics should triage listening, not replace it. Especially:

- F0 threshold already caused gender artifacts per G3.
- Chest-weight can be noise/proximity.
- Nasal-band ratios are approximate.
- End-of-line F0 rise depends heavily on clip segmentation and whether the selected donor line was a question.

The strongest part of P2 is P2b: audition through the actual cloner.

---

## P2b should test vector-off and calibrated vector separately

The doc says:

> fixed 6-line panel ... vector-off + calibrated

Good. Keep both. A ref may be fine neutral but collapse under fear/surprise vectors. That matters for drama.

Add scoring columns:

- neutral authority;
- question whine;
- fear whine;
- grief whine;
- artifact under emotion;
- consistency across seeds.

---

## P2c tier prefilter is right; “lead_safe preferred” needs precise deterministic semantics

Grounding: `_pick` uniformly chooses within the winning ladder tier; scores only sort. P2c says pre-filter the pool before the ladder and “lead_safe preferred.”

Be careful: “preferred” can sneak in as weighting or nondeterministic fallback. I’d define it as a hard deterministic two-stage filter:

1. For lead chars:
   - if any tier-A `lead_safe` candidates survive gender/role constraints, use only those;
   - else tier-A;
   - else fail/degrade to tier-B only if configured.
2. Then use existing ladder/pick.

That preserves determinism and avoids “preferred but still randomly lost.”

---

## P2d `allow_voice_reuse=false` should wait for P3b

The doc says flipping `allow_voice_reuse=false` degrades gracefully because CastLock catches exhaustion and render-time resolution takes over.

Grounding: `_resolve_clone_ref_path` currently calls:

```python
assign_voice_for_slot(... allow_voice_reuse=True, bank=bank)
```

So if CastLock declines to cast a character because reuse is false/exhausted, render-time resolution can still assign with reuse allowed, and without the CastLock used-set/tier policy. Until P3b’s per-character resolve-once cache + used-set exists, flipping reuse may produce a false sense of uniqueness.

Recommendation:

- Do not flip `allow_voice_reuse=false` in the canonical graph until P3b is in.
- Or accept it only as an experiment, not as a sign-off step.

---

## P3a observability is mis-ordered; do a minimal version earlier

P3a is described as blocking P2c sign-off. I agree. That means it should be before or alongside P0/P2b, not after delivery retuning.

Minimum useful version:

- render_log always includes:
  - char_id;
  - line_id;
  - engine actually used;
  - voice_ref_id;
  - ref_path basename;
  - delivery version;
  - alpha;
  - whether vector omitted/all-zero/nonzero;
  - seed.
- Durable ledger can follow if route is annoying.

This does not violate determinism and does not affect audio bytes.

---

## P3b needs to avoid character revoice and tier bypass

The plan catches the per-line used-set hazard. Good.

Additional caution:

- Cache key should include at least `(engine, role, char_id)` and probably episode identity.
- If a cast row has `voice_ref_id`, do not override it unless invalid.
- If tiers are introduced, render-time resolution must use the same filtered bank policy as CastLock. Otherwise uncached fallback resolution can pick refs P2c would reject.

---

## P4 solo-vs-mix gate is essential, but the post/mix option may deserve earlier experiment

The doc puts the post/mix decision late. I would run the free offline post test before major bank work:

- Take an existing whiny WAV.
- Loudness normalize.
- Add mild compression/EQ/saturation.
- Put it back under the bed.
- Ask operator: “Is this 0%, 30%, 70%, or 100% fixed?”

If it is 70% fixed, P2 bank work is still useful, but the immediate production fix is a dialogue bus.

---

# 4. Additional concrete tests I would add

## Test 1: “No-vector neutral policy” test

Compare same ref/line/seed:

1. delivery off: no `emo_vector`;
2. all-zero vector explicitly sent — worker will omit because all zero;
3. `calm=1.0`;
4. `calm=0.5`;
5. P0 alpha winner with v1 vector.

Expected outcome unknown. If `calm=1.0` sounds softer/weaker than omitted vector, remove calm from neutral default.

---

## Test 2: Generated-line final-rise report

After a smoke render, compute for each character line:

- median F0;
- final F0 slope;
- line duration;
- RMS/LUFS proxy;
- spectral centroid;
- vector mass;
- terminal punctuation.

Sort by final-rise + high vector mass. If the top-ranked lines match operator “whiny” lines, you have a useful regression metric.

---

## Test 3: Current cast lottery risk report

Before curation, simulate e.g. 100 episode seeds for Hayes/Gulliver:

- which male refs get picked;
- how often suspect refs appear;
- how often same voice collision occurs;
- how much quality tiering would reduce exposure.

This validates whether uniform lottery is a major practical cause or only theoretical.

---

# 5. Wildcards: 3 thirty-minute experiments

## Wildcard 1 — Fixed per-character prosody seed

Speculation: line-level seed variation may be a hidden cause of pleading/unstable delivery.

Experiment:

- Render a 6-line Hayes panel with current per-line seeds.
- Render same panel with one deterministic character seed for all Hayes lines.
- Keep ref/text/vector identical.
- Loudness-match and blind compare.

If it helps, implement a versioned seed mode. This is local, deterministic, no new engine, and cheap.

---

## Wildcard 2 — Offline “radio dialogue bus” on existing bad WAV

Speculation: much of “whiny” is psychoacoustic thinness versus Kokoro/music.

Experiment on one already-rendered bad scene:

- Normalize character clips toward announcer loudness.
- Add gentle compression.
- Add warmth around 150–250 Hz.
- Slightly reduce 1–2.5 kHz if nasal.
- Add mild saturation.
- Duck music under dialogue by a fixed deterministic envelope.

If operator says “mostly fixed,” prioritize H2c/post path.

---

## Wildcard 3 — Generated neutral self-ref, but only as a diagnostic

Speculation/risky: render the best donor through IndexTTS2 once with a neutral paragraph/vector-off, then use that generated output as a new reference. This may “launder” donation mic/read-style into a steadier model-native prompt, or it may compound artifacts.

Test only:

- one good donor;
- one bad/thin donor;
- same 6-line reel;
- compare original ref vs generated self-ref.

If artifacts increase, discard. If authority/body improves, it may be a useful bank-entry manufacturing trick, with clear sha/versioning.