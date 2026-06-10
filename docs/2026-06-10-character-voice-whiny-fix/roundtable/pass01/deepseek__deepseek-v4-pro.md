<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

## Independent Expert Review: Character-Voice Whiny Fix

### 1. Blind Spots (causes the DOC misses)

The plan focuses on reference-pool lottery and emotion-vector saturation, but the "whiny" / thin character sound could also stem from factors entirely outside the diagnosis matrix. These are plausible contributors absent from the discussion:

- **Reference-Clip Acoustics Beyond F0**  
  The auditor uses only median F0 as a “thin‑risk” heuristic (P2a). It ignores:
  - **Duration:** a clip shorter than ~8 s may not capture the speaker’s full timbre range and pitch variation, forcing the cloner into an under‑informed embedding that defaults toward thin, high‑frequency coloration.
  - **SNR & background noise:** low‑level hiss or room tone can become part of the cloned timbre, adding a brittle edge.
  - **Mic coloration:** many amateur donations likely exhibit bass roll‑off (typical of built‑in laptop mics). Cloning that response yields a voice lacking body.
  - **Reading style:** a male donor recorded *softly* or *nervously* may produce a spectral envelope that, when cloned, maps to “pleading” even without an emotion vector.
  - **Loudness & crest factor:** a quiet, low‑energy reference may cause the cloner to output a low‑level signal; later gain‑staging might amplify noise or thinness.
  These effects are not captured by median F0. The plan’s human listen‑pass (P2b) will catch some, but a purely algorithmic pre‑screen (spectral centroid, RMS, crest factor) could rank refs before listening.

- **IndexTTS2 Zero‑Shot Behavior on Weak Refs**  
  UNSURE: The exact internal mechanism is not publicly documented, but common TTS cloning systems derive a speaker embedding from a reference’s spectral envelope. If the envelope is thin (low energy below 300 Hz), the embedding may project a small vocal‑tract model; when overlaid with a saturated multi‑hot emotion vector, the model might exaggerate the high‑formant energy, producing a whiny resonance. The plan’s matrix (P0) will reveal this, but the DOC does not consider that the *interaction* of a thin ref with the emotion model may be non‑linear—alpha reduction alone might not fix it. A “warm” ref with strong low‑mid energy is likely the more robust countermeasure.

- **Psychoacoustics of a Lone Voice in a Drama Mix**  
  The plan addresses generation, not final mix context. Even a well‑rendered voice can sound thin when:
  - It lacks low‑frequency energy below 150 Hz (common in TTS) and is placed against a music bed with heavy bass.
  - It sits against high‑frequency‑rich SFX (hissing steam, alarms) that mask the voice’s upper harmonics.
  The mix (master‑bus EQ, voice‑centric ducking) is outside scope, but the *auditory illusion* of “whiny” might be exacerbated by this mismatch. A quick test: route a rendered voice with a high‑pass filter at 150 Hz (simulating typical TTS roll‑off) next to a music bed; the thinness becomes apparent. Without a post‑generation spectral‑shaping step, the operator may still perceive thinness even after fixing the TTS side.

- **Emotion Conditioning Order and Calm Dimension**  
  The IndexTTS2 emo‑vector order lists calm last, but the model may treat calm as an *inverse‑energy* parameter, not just a blend weight. Sending `calm=1.0` at alpha 1.0 (the current “neutral” default, G8) could force the prosody into a subdued, low‑articulation mode that, when combined with any non‑zero afraid/sad, morphs into a pleading delivery. The plan will test this (P0 cell D), but the root cause might be that the engine’s calm dimension is not a simple “rest” but an active suppression that interacts badly with arousal cues. **The delivery‑calibration proposal (P1) floors calm at 0.15 even when speech is unexcited—this may introduce mild tension where none is desired, making neutral lines sound slightly anxious.** This is a risky assumption without evidence that `calm=1.0` is harmful; P0 D should inform whether a floor is needed.

- **Text Punctuation Double‑Dip Overcorrection**  
  The plan notes that “?” adds 0.5 to surprised and “!” adds 0.3 to surprised, angry, afraid, etc. However, the fix says only “fix punctuation double‑dip” without specifying method. Simply halving the values might not be enough; the emotional meaning of punctuation is context‑dependent. A line ending in “!” after a neutral phrase does not carry the same weight as a keyword‑saturated line. Without a semantic check, reducing punctuation weight could flatten legitimate exclamations. The plan should define a precise adjustment (e.g., cap punctuation contributions at 0.5 total per line, or scale them by inverse of keyword mass already present).

- **Voice‑Bank Gender Mis‑labeling Beyond F0**  
  The audit (P2a) only flags the two suspect female tags; but the entire pool was tagged by median F0. A male donor with a permanently low larynx but a nasal reading style could land in the female pool, and vice‑versa. The plan’s human listen‑pass will correct gender, but the audit script should flag *any* voice where the F0 is within 20 Hz of the gender threshold (≈145–185 Hz) as “ambiguous,” forcing the operator to listen before fixing.

---

### 2. Craft (what a voice director would do, automatable)

A voice director aims for a consistent character sound, not just technical calibration. Concrete, automatable ideas within the existing pipeline:

- **Reference‑Clip Normalisation (“Voice‑Match”)**  
  Before cloning, apply a *spectral‑match EQ* to all reference clips so they share a common low‑end profile. Choose the best male ref (e.g., `vz_bill_boerst`) as the target. For each ref, compute its long‑term average spectrum (e.g., via `librosa.feature.melspectrogram` averaged over time) and derive a correction curve that shapes it toward the target below 1 kHz. This can be implemented as a FIR‑filter offline and saved as a processed ref. Automatable, no per‑line cost, and it preserves determinism. The bank can store a `ref_path` and an optional `ref_path_matched`; the caster picks the original if no match exists. This directly addresses timbre thinness from mic disparities.

- **Per‑Line Energy/Intensity Floor Through Compression**  
  A pleading voice often has sharp dynamic swings (quiet, high‑pitch rises). Apply a deterministic multiband compressor or upward compressor post‑render to reduce the perceived “thin” peaks. For instance, a simple `librosa.effects`‑based compressor with threshold −18 dB, ratio 3:1, and a gentle low‑shelf boost (2 dB at 250 Hz) can add weight without altering the cloner. The node can add this as a configurable post‑processing step, applied only to character voices. Deterministic if parameters are fixed.

- **Pacing and Breath Insertion**  
  Whiny delivery often includes rapid, shallow breaths. The text preparation already strips stage directions; a voice director would add *preceding pauses* to slow the pace when a line is emotionally charged. Automatable: scan the derived emotion vector; if “afraid” > 0.5, prepend a 300 ms silence and insert “…” in the text (where the TTS engine might produce a natural pause). This changes the audio but preserves the text path; the script‑prep hook can return the modified text. This is a cheap, reversible experiment.

- **Per‑Character Voice Warmup (Anchor Line)**  
  Use a fixed, neutral “anchor” line (e.g., “The weather is calm today.”) generated with the same ref and zero emotion vector before each episode. Measure its spectral centroid and peak‑to‑average ratio; if it deviates significantly from a known good baseline, that ref is likely thin and should be skipped. This is a runtime quality gate, automatable inside `_resolve_clone_ref_path`, and provides audible QA without manual listening.

---

### 3. Critique of the Plan (steps that are wrong, mis‑ordered, riskier, or dominated)

- **P1‑Delivery Calibration: Calm Floor at 0.15**  
  This is **risky and likely wrong**. The plan says “floor calm at 0.15 when any speech is unexcited.” If “unexcited” means all other emotions are zero, then a neutral line like “The door opened.” would have `calm=0.15` instead of `1.0`. This could force the TTS into a mildly tense, breathy delivery—exactly what the operator does *not* want. The open question (G8) is whether `calm=1.0` at full alpha degrades neutral lines; P0 cell D will settle it. **If P0 shows no issue with calm=1.0, remove this floor entirely.** The fix could be: keep `calm` at `max(0.15, computed)` only when arousal > 0.5, to ensure some energy in excited lines, but that’s speculative. Better to wait for P0 data.

- **P1‑Delivery Calibration: Cap “Normalize Sum to ≤1.2”**  
  A hard normalization dividing all non‑calm values by a factor may overly compress dynamics. A softer approach (e.g., apply a softmax or tanh) would preserve relative intensities while preventing saturation. The current method (`/ _CAP` with `_CAP=3.0`) already caps each dimension individually but not total mass. The proposed normalization is simple, but if the sum is 4.0, each value is divided by ~3.33, which reduces all emotions drastically, possibly losing intended nuance. A better approach: cap each dimension at 1.0 (already done) and then apply a penalty function to the total, e.g., `total_penalty = 1.0 / (1.0 + max(0, total - 1.2))`, so that beyond 1.2, further increase yields diminishing returns. This retains the shape of the vector. The plan’s method is not wrong, but **dominant alternative**: an `alpha` reduction (P1 exposes alpha) together with keyword demotion might obviate the need for a global cap, making the plan less critical.

- **P2a Audit: “clipping/SNR proxy” Not Implemented**  
  The script description says “clipping/SNR proxy” but the code in `otr_dl_indextts2_refs.py:classify_and_trim` does not compute SNR or detect clipping. It only returns F0 and voiced fraction. The plan should add a simple RMS‑based threshold and peak‑detection to flag refs with noise floor > −40 dB or sample counts at ±1.0 (clipping). Without this, badly recorded refs may be missed.

- **P2c: Filtering by Tier Inside `assign_voice_for_slot`**  
  The plan says “caster honors tiers by FILTERING (tier‑a pool first, tier‑b fallback, never ‘reject’).” However, the current `assign_voice_for_slot` takes a `bank` argument but has no concept of tiers. The plan must modify that function to accept a `quality_tiers` mapping or pre‑filter the bank before calling it. The fallback logic “tier‑a first” must still maintain the match ladder; if no tier‑a ref matches gender, it falls back to tier‑b. This requires careful implementation to avoid breaking the ladder. The plan doesn’t detail how to integrate with the seed‑based choice; it’s workable, but the risk of regression is moderate.

- **P3a: LOUD per‑LINE Persistence Might Restamp the Ledger Prematurely**  
  The ledger is frozen; restamping the `voice_ref_id` after render could cause a C7 violation if the restamp changes the ledger bytes used for subsequent mix steps. The plan says “log the swap, stamp the ledger” – if the ledger is used only downstream for metadata and not for audio byte‑identity, it’s safe, but the deterministic guarantee (same code+config+seed = same bytes) might break if the ledger text changes after freeze. Ensure the stamping is purely additive (append a new field) and does not modify existing hash‑sensitive fields. The current `stamp_delivery_vectors` is additive, so the pattern exists. Risk is low.

- **P4: `test_audio_byte_identical` Expected Bark‑Only**  
  The plan’s P4 asserts the test remains green after P1 changes. Since delivery vectors are not used in the bark path, that’s almost certain, but the delivery calibration touches `_otr_delivery_vector.py` which is imported in the per‑line path only. No risk. However, if the post‑processing compression idea from “Craft” is adopted, that new code might load a library that changes the environment, risking the deterministic test. Keep in mind.

- **Sequence:** P0‑zero (check engine) before P0 (matrix) is correct. But the plan suggests “P2a audit table is CPU and can run today” – indeed, it’s independent. However, the listen‑pass (P2b) requires the operator’s time, so that should happen after P0 confirmation to avoid auditing voices that aren’t the root cause. The operator may want to postpone P2b until after hearing P0 results.

**Cheaper Dominant Step:**  
If P0 shows that **delivery saturation is the primary cause**, exposing `OTR_INDEXTTS2_EMO_ALPHA` and setting it to 0.65 is a zero‑line‑change win (just adapter config). The plan includes it in P1, but that’s after a table rewrite. The *cheapest* fix might be to change the default env and ship; the table v2 can follow. The operator could test with a single env override and skip the table changes initially.

---

### 4. Wildcards (up to 3 unconventional 30‑minute experiments; speculation marked)

1. **“Reference‑Boost” Low‑End Enhancer**  
   *Speculation:* The whiny quality could be partly due to the human ear’s sensitivity to voices lacking fundamental harmonics. Apply a subharmonic synthesis algorithm (e.g., an octave‑generation based on pitch tracking) to the rendered mono WAV, mix it in at −12 dB. This adds perceived weight without altering the cloner. Experiment: use `pyworld` to extract F0, generate a sine wave at half frequency with a low‑pass filter at 200 Hz, and overlay. Run on a few lines from the P0 matrix; compare spectral balance.

2. **Emotion‑Vector “Context‑Drop”: Inverse Text Length Weighting**  
   *Speculation:* Short exclamations (“Help!”) concentrate all keywords in one word, leading to saturated vectors. A long sentence with the same keywords gets a diluted effect. Inverse length weighting: multiply the final vector by `min(1.0, log10(len(words)+1)/log10(5))`. Short lines are penalized, making them less hysterical. 30 min to implement and test on a few problematic lines from the script. Mark as speculation – it might flatten needed urgency.

3. **Clone from “Warm‑Target” Instead of Raw Ref**  
   *Speculation:* If a ref is thin, instead of filtering it out, create a synthetic warm version by applying a static EQ curve (bass boost +4 dB below 300 Hz, gentle treble cut above 4 kHz) before cloning. The cloner will then model a voice with more low‑end, possibly producing a fuller sound. This is an offline processing of the ref clip. Run the P0 matrix with the EQ’d ref vs original; observe if the output gains weight. Risk: the cloner might misinterpret the spectral shape and introduce artifacts. 30 minutes to try.