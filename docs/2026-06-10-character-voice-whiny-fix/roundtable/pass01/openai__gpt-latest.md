<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

## Review findings

Overall: the DOC correctly identifies two strong causes — weak reference lottery (G3/G4) and overdriven affect vectors (G6/G7). My main concern is that it treats “whiny” mostly as **speaker identity + emotion conditioning**, while in radio-drama perception it can also come from **reference read style, punctuation/prosody, post/mix masking, and per-line isolation**. Those can survive even after P1/P2.

---

## 1. Blind spots / plausible missed causes

### 1.1 Reference clips may be acoustically valid but directorally wrong

The DOC audits gender/F0 and some quality proxies, but not **read style**. Many donated voice clips are likely polite, careful, high-intelligibility, “sample donation” reads. A zero-shot cloner may preserve not only timbre but also **up-talk, smile, breathiness, cautious articulation, or supplicating prosody**. That can read as pleading even if F0 is not especially high.

Grounding: `otr_dl_indextts2_refs.py` trims “the longest voiced span” from donor material and stamps generic timbre tags. There is no style audit, no “authority / grounded / tense / narrator / casual” classification.

Falsifiable test: render the same neutral declarative line with 10 refs and no emotion vector. If the “whiny” quality remains on some refs even with `emo_vector` omitted, those refs are not just high-F0; they carry a bad read style for drama casting.

Automatable metric candidates:
- Median F0 and F0 range.
- Speaking rate / syllables per second proxy.
- End-of-line F0 rise: last 500 ms median F0 minus sentence median.
- Spectral tilt / harmonic-to-noise proxy.
- RMS/LUFS consistency.
- “Nasal/telephone” proxy: energy ratio around 800–1500 Hz vs 150–350 Hz.

### 1.2 Peak normalization may amplify thin/noisy refs

The ref downloader normalizes each trimmed span to peak `0.97`:

```python
peak = float(np.max(np.abs(span))) or 1.0
span = (span / peak * 0.97).astype("float32")
```

This does not loudness-normalize. A quiet/noisy/roomy clip with one transient peak can end up with inconsistent perceived level; a low-level clip can have noise and room tone raised. Peak normalization also says nothing about low-mid body.

Falsifiable test: compute integrated LUFS or RMS and noise-floor proxy for all refs. If whiny refs cluster as low-RMS/high-noise/high-centroid after peak normalization, ref conditioning may be degraded before IndexTTS2 sees it.

### 1.3 Sample-rate / resampling assumptions deserve a quick check

Refs are written at `OUT_RATE = 44100`, while the IndexTTS2 adapter declares `sample_rate = 22050` and the worker reports `22050`. It is likely IndexTTS2 internally resamples the prompt, but that is **UNSURE** from the provided code. Poor or repeated resampling can subtly thin voices.

Grounding:
- `otr_dl_indextts2_refs.py`: `OUT_RATE = 44100`
- `eng_indextts2.py`: `sample_rate = 22050`
- worker response: `"sample_rate": 22050`

Falsifiable test: prepare two versions of one strong ref: native 44.1k and a clean deterministic 22.05k/mono version. Render same text/seed/emotion-off. If 22.05k sounds fuller or more stable, standardize ref sample rate.

### 1.4 Text punctuation itself can create pleading, independent of emotion vector

The plan addresses punctuation as part of the emotion vector, but the TTS still sees the punctuation. In `IndexTTS2Engine.prepare_text`, the adapter calls only `clean_spoken_text`, not the fuller shared `prepare_text`. So `?`, `!`, `...`, unicode ellipsis, asterisks, etc. may remain unless stripped by `clean_spoken_text`.

Grounding:
```python
def prepare_text(self, text, delivery_vector=None):
    from .._otr_script_prep import clean_spoken_text
    return clean_spoken_text(text)
```

Even with `OTR_DELIVERY_VECTOR=0`, a script full of “What?”, “How?”, “Wait!”, “No...”, etc. can produce rising, uncertain, pleading line endings.

Falsifiable test: for one whiny line, render three versions with same ref/seed/emotion-off:

1. Original punctuation.
2. `?` → `.`, `?!` → `.`, `...` → `,`.
3. Original words but declarative punctuation and fewer interjections.

If version 2 is less whiny, text prep needs an IndexTTS2-specific “drama punctuation” profile, not only emotion-vector retuning.

### 1.5 Delivery vector may be derived from text that will not be spoken

The delivery derivation runs on raw `text` in `_render_per_line` before `prepared = prep(...)`. Stage directions or bracketed tags may cue emotion but then get stripped before synthesis.

Grounding:
- Delivery vector uses `text` directly.
- `clean_spoken_text` later strips parentheticals/brackets.

Example: `(afraid) What is it?` may generate fear/surprise, while the spoken text is only “What is it?” This may be intentional, but if writer text includes frequent parenthetical action cues, it can overdrive emotion.

Falsifiable test: log raw text, prepared text, and vector for 20 whiny lines. Count how many emotion cues came from stripped text. If high, derive vector from prepared text plus explicit delivery metadata only.

### 1.6 Per-line isolation can make everyone sound needy

The engine is called per line. No previous line, no scene context, no conversational momentum. Isolated TTS lines often default to “perform every line,” especially questions and short interjections. A voice director would ask actors to play the scene objective; the model gets only a sentence.

Grounding: `_render_per_line` calls `adapter.generate_voice(...)` once per line. No dialogue context is passed.

Falsifiable test: render a short exchange as:
- separate per-line renders, current path;
- one multi-sentence same-speaker chunk, if possible for adjacent same-character lines;
- same line with a neutral preceding sentence included, then manually/algorithmically trim the target sentence.  
If context reduces whine, the issue is partly per-line prosody, not just refs/vector.

IndexTTS2 behavior here is **UNSURE**; this is an experiment, not a claim.

### 1.7 Seed variance is under-tested

P0 proposes one fixed seed. Some TTS models exhibit prosody instability across seeds. A “whiny” take may be a bad draw, especially with expressive conditioning.

Falsifiable test: same ref/text/vector, 8 seeds. Measure/listen for median F0, final-rise, duration, and subjective whine. If one or two seeds are much worse, seed selection or deterministic seed offset per character/style may help.

### 1.8 Mix masking can create perceived thinness

The announcer sounds fine, but characters may be masked differently by music/SFX. A voice can be objectively okay solo and still sound thin in a drama bed if:
- music masks 100–300 Hz body;
- announcer has better compression/EQ;
- character lines are dry and narrow while beds are wide;
- character loudness is lower or more dynamic;
- no sidechain ducking exists;
- no room/ambience glues characters into the scene.

The DOC says “announcer is fine in the mix,” but does not separate **solo TTS quality** from **mix perception**.

Falsifiable test: compare each suspect line solo vs in full mix, loudness matched. If solo is acceptable but mix is whiny/thin, prioritize post chain/ducking over ref curation.

---

## 2. Craft: what a voice director / post engineer would add

### 2.1 Cast by dramatic function, not only gender/timbre

“Hayes” and “Gulliver” should not draw uniformly from all acceptable male refs. A director would maintain type buckets:

- lead male: grounded, lower final-rise, medium/low F0 range, stable;
- nervous technician: allowed thinner/brighter;
- villain/authority: lower pace, low-mid weight;
- comic/secondary: more expressive allowed.

This can stay deterministic: add optional bank tags such as `style_tags: ["grounded", "lead_safe", "tenor", "nasal_risk"]`, then filter by character role/archetype. No new engine needed.

Cheap version: after P2b, mark 3–5 refs as `lead_safe=true`; force major recurring male roles to use that subset.

### 2.2 Build a tiny “audition reel” harness

Instead of one P0 dramatic line, render a fixed panel:

1. neutral declarative,
2. urgent command,
3. question,
4. fear line,
5. grief/melancholy line,
6. technical exposition.

For each candidate ref, render emotion-off and calibrated emotion-on. Loudness-match outputs. This gives a repeatable audition reel the operator can judge quickly.

Dominates pure metadata audit because it tests the actual cloner, not just the source WAV.

### 2.3 Add a deterministic character dialogue post chain

A post engineer would often fix “thin/whiny” after recording. Automatable, local, deterministic CPU chain:

For character voices only:
- high-pass no higher than ~70–90 Hz;
- gentle low-mid shelf/body boost around 120–250 Hz;
- narrow nasal reduction around ~900–1500 Hz if energy ratio is high;
- mild compression, e.g. 2:1 or 3:1, slow-ish attack, medium release;
- de-ess if needed around 5–8 kHz;
- optional very short room/plate reverb at low wet level to glue into the scene;
- sidechain/duck music under dialogue.

This may beat ref changes if the issue is perceived thinness in mix.

Falsifiable 30-minute test: take an existing whiny rendered WAV and apply only EQ/compression/loudness matching. If operator says “70% fixed,” prioritize post chain.

### 2.4 Do “final-rise suppression” for male leads

Whiny/pleading often comes from rising line endings. For declarative dramatic lines, a director would ask for “land the sentence.” Automatable options:

- punctuation doctor: convert some `?`/`?!` to `.` when the line is rhetorical or starts with stop-word cues;
- render-time filter: detect high final F0 rise and re-render with alternate punctuation or lower alpha;
- post pitch contour correction on the last 300–700 ms, speculative but testable.

The first option is cheapest and deterministic.

### 2.5 Separate “fear” from “panic”

The current vector makes high-frequency sci-fi words produce afraid/surprised/sad. A director would rarely let a lead play every danger line as fear. Leads often play danger as **focus, command, restraint, anger, or resolve**.

So instead of mapping “danger/run/help” directly to afraid, major-role profiles should map those cues to:
- lower `afraid`;
- slight `angry` or neutral/calm;
- lower `surprised`;
- possibly no vector unless punctuation is extreme.

This can be done later with delivery profiles, but even P1 table v2 could be less literal.

---

## 3. Critique of the plan

### 3.1 P0-zero is good, but it only proves fresh-render engine identity

The H4 concern is real. In `_render_per_line`, fallback warnings only go into `render_log`, and per-line identity is not persisted. P0-zero should be first.

Caveat: it will not explain existing bad episodes unless those exact node outputs/history are still available. The plan already says “for a fresh render”; keep that limitation explicit.

### 3.2 P0 matrix is too small

One line, one seed, one “warm/baritone” ref can mislead. Minimum useful expansion:

- 3 lines: declarative, question, fear/urgent.
- 3 seeds per line.
- emotion omitted vs current vs alpha candidate.
- loudness-matched listening.

Still feasible in ~30–45 minutes if scripted.

### 3.3 Alpha exposure likely dominates table retune

P1 has two changes: table v2 and `OTR_INDEXTTS2_EMO_ALPHA=0.65`. I suspect the alpha knob is the safer first lever. The table retune is more subjective and can create new failure modes.

Recommendation as finding: expose alpha and test defaults before committing `0.65`. Values worth testing: `0.0`, `0.25`, `0.45`, `0.65`, `1.0`. IndexTTS2 internals are **UNSURE**, so do not assume 0.65 is mild.

### 3.4 “calm floor” may not mean what the plan assumes

The plan proposes “floor calm at 0.15.” But the worker only passes `emo_vector` if any value is nonzero:

```python
if ev and any(float(x) != 0.0 for x in ev):
    kwargs["emo_vector"] = [float(x) for x in ev]
```

A vector containing calm plus other emotions is not the same as reducing non-calm emotions. IndexTTS2’s interpretation of mixed calm + fear/surprise is **UNSURE**. Calm may not counteract whine linearly.

P0 cell D is therefore important, but also test “low non-calm, calm 0” vs “same non-calm, calm 0.15/0.5/1.0.”

### 3.5 P1 mass cap `<=1.2` is plausible but arbitrary

A total non-calm mass cap is right, but 1.2 may still be high if IndexTTS2 treats the vector as categorical blend or exaggeration control. Consider testing 0.5, 0.8, 1.0, 1.2. The right cap is an empirical model behavior question.

### 3.6 P2a audit should not rank “thin-risk” by F0 alone

The DOC acknowledges F0 is a heuristic, but the proposed “highest-F0 males first” could miss:
- nasal low-F0 voices;
- over-denoised voices;
- room/phone coloration;
- breathy/noisy voices;
- upbeat/up-talk reads;
- refs with clipped consonants or unstable pitch.

Add at least:
- duration;
- RMS/LUFS;
- peak/RMS crest factor;
- clipping count;
- voiced fraction;
- spectral centroid;
- low-mid energy ratio;
- final-rise estimate;
- maybe a crude “nasal band” ratio.

### 3.7 P2c filtering is better than weighting, but watch variety collapse

Filtering to tier A first is good and deterministic. But if only 2–3 male refs pass tier A, the drama may become repetitive. That is still probably better than whiny leads, but mark it as a tradeoff.

A stronger rule: major recurring leads require tier A; minor/background can use tier B. Reject never used.

### 3.8 P2d does not fully solve collision on the default path

Changing CastLock `allow_voice_reuse=false` helps only the CastLock route. The default preserve-ledger path still calls `_resolve_clone_ref_path(... allow_voice_reuse=True ...)` per character, as shown in grounding.

P3b is therefore not optional if the default route matters.

### 3.9 P3b implementation has a subtle trap: same character must stay same ref

`_resolve_clone_ref_path` is called inside the per-line loop. If a used-set is naively updated per line, the same `char_id` could get a different ref on later lines. Current behavior is stable per character because `stable_cast_seed` includes `char_id`, but adding episode uniqueness needs a per-episode cache:

- if `char_id` already resolved, reuse that ref;
- else assign a new ref excluding used refs;
- then mark it used.

This is important.

### 3.10 P3a “ledger restamp” may not persist anywhere

In `_render_per_line`, the ledger is loaded locally:

```python
led = _OTRLC.load_ledger(source)
```

The node returns only `(audio_out, render_log, done)`. Unless another path writes `led` out, mutating/restamping it inside the node will not create a saved ledger artifact. So “saved ledger forever” is not achieved by mutation alone.

Cheaper reliable observability:
- render_log per character: `char_id -> voice_ref_id -> ref_path -> engine`;
- per fallback line: `line_id`, `char_id`, fallback engine;
- include `emo_alpha` and delivery version;
- optionally write a deterministic sidecar if the architecture permits. But the kickoff says “no new files,” so render_log/history may be the only persistence.

### 3.11 The plan underweights solo-vs-mix verification

P4 says one full episode re-render and operator listen-QA. Add a gate:

- listen to suspect character lines solo, loudness-matched;
- then in mix;
- if solo fixed but mix still thin, fix post/mix;
- if solo still whiny, continue refs/vector/text.

Without this split, engineering may keep tuning TTS for a mix problem.

---

## 4. Additional cheap tests I would run

### 4.1 Emotion-off plus punctuation-normalized test

For one whiny line:

- original text, delivery off;
- punctuation softened, delivery off;
- original text, alpha 0.65;
- punctuation softened, alpha 0.65.

If punctuation-softened/off is much better, P1 table changes alone will not fix it.

### 4.2 Ref WAV “body” report

For each ref, print:

- duration;
- sample rate;
- RMS/LUFS proxy;
- crest factor;
- median F0;
- F0 std;
- voiced fraction;
- spectral centroid;
- 100–300 Hz energy / 800–1500 Hz energy;
- clipping samples;
- final-rise estimate.

Then compare known whiny refs vs acceptable refs. If clusters appear, automate tier suggestions.

### 4.3 Render-level acoustic report

For generated clips, not refs:

- median F0;
- final 500 ms F0 rise;
- duration;
- RMS;
- spectral centroid;
- low-mid ratio.

This catches cases where a good ref becomes whiny only under certain emotion/text settings.

---

## 5. Wildcards: 30-minute experiments

### Wildcard 1 — deterministic “weight” post chain

Speculation, but highly worth testing.

Take already-rendered whiny character WAVs and apply:
- +1 to +3 dB low shelf around 150–220 Hz;
- -1 to -3 dB notch around 1–1.3 kHz;
- gentle compression;
- loudness match to announcer/target dialogue;
- optional tiny room reverb.

If this improves the perception substantially, add a character-dialogue post profile before spending more time on bank curation.

### Wildcard 2 — “second-generation neutralized refs”

Speculation / risky.

For a bad-but-interesting ref, render a neutral paragraph with emotion omitted, then use that generated neutral WAV as the future reference clip. This may “launder” noisy donor acoustics and stabilize style, or it may compound model artifacts. It is local, deterministic, and uses no new TTS engine.

Test with one good and one bad donor only. Compare original-ref vs generated-ref on the audition reel.

### Wildcard 3 — render two punctuation variants and auto-select lower final-rise

Speculation, but automatable.

For lines ending in `?`, `?!`, or `...`, render:
- original punctuation;
- softened punctuation.

Analyze final F0 rise / spectral centroid / duration. Deterministically choose the version with less final-rise unless the line is explicitly tagged as a real question. This is a crude “director says don’t plead” mechanism without manual per-line work.

---

## Bottom line

The plan’s two main fixes are valid, especially P0-zero, alpha exposure, and bank curation. But I would add four missing dimensions before declaring victory:

1. **Punctuation/text prosody** independent of emotion vectors.  
2. **Reference read style** independent of F0/gender.  
3. **Solo vs mix perception** with EQ/compression/ducking tests.  
4. **Per-line context/seed instability** as a source of pleading takes.

If time is tight, the highest-leverage order I would personally run is:

1. Prove no Bark fallback.  
2. Emotion-off / alpha sweep / punctuation-softened mini audition.  
3. Ref audition reel with 5–6 lines, not just metadata.  
4. Quick EQ/compression post test in mix.  
5. Then implement durable logging, alpha config, and tier filtering.