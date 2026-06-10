<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

Here is the independent review of the hardened plan. 

### 1. Blind Spots (What the DOC misses)
*   **The "?" Prosody Trap (Text Preparation):** The plan addresses punctuation *double-dipping* in the emotion vector, but misses the acoustic effect of the punctuation itself on the TTS engine. In many zero-shot models, a question mark `?` inherently triggers a rising, insecure, or pleading inflection. If a script is full of interrogatives ("What?!", "How?!"), IndexTTS2 will physically read them with a pleading pitch contour, *regardless* of the emotion vector. 
*   **Donor Read Style vs. Timbre:** The plan assumes "thin/nasal" acoustics (timbre) cause the whiny sound. But zero-shot cloner models clone *prosody* just as aggressively as timbre. If a CC0 volunteer read their donation clip with a flat, low-energy, or hesitant cadence, IndexTTS2 will clone that hesitant "acting" onto Hayes and Gulliver. 
*   **Loudness/Density Mismatch:** Kokoro (the announcer) is a highly optimized, heavily compressed model that outputs dense, radio-ready audio. IndexTTS2 zero-shot outputs are often highly dynamic. If IndexTTS2 is simply quieter or less compressed than Kokoro in the final mix, the psychoacoustic result is that the characters sound "weak" or "thin" compared to the Voice of God announcer.
*   **Reference Sample Rate Mismatch (UNSURE):** `otr_dl_indextts2_refs.py` explicitly resamples and saves donor WAVs at `OUT_RATE = 44100`. However, `eng_indextts2.py` and the worker declare a native `sample_rate = 22050`. If IndexTTS2's internal feature extractor expects 22.05kHz/24kHz and naively ingests 44.1kHz without proper anti-aliasing, it can cause spectral thinning or metallic artifacts that sound "weak."

### 2. Craft (The Audio Post / Voice Director View)
*   **Measure "Chest", not just Pitch:** A voice director knows that a high-pitched voice can be booming and confident (a drill sergeant), while a low-pitched voice can be incredibly whiny (Eeyore). Ranking "thin-risk" by F0 (P2a) is an acoustic fallacy. **Automatable fix:** Use `librosa.feature.spectral_centroid` alongside F0. Voices with a lower spectral centroid have more low-frequency energy relative to their highs—this is the literal acoustic measurement of "chest resonance" and vocal weight.
*   **Directing via Punctuation (The "Make it a Statement" note):** A director fixing a whiny actor will say, "Stop asking, start telling." **Automatable fix:** In `_otr_script_prep.py`, you can deterministically strip `?` and replace it with `.` *after* the delivery vector has extracted its cues. This forces IndexTTS2 to use declarative, confident prosody, while the `emo_vector` still supplies the "surprised" or "afraid" coloration.
*   **Post-Compression:** A lone voice in a mix sounds weak. A post-engineer would put a compressor on the character bus. Since you are constrained to the existing pipeline, you can't easily add a VST, but you *can* normalize the RMS (not just peak) of the IndexTTS2 clips in `eng_indextts2.py:_load_wav` to match Kokoro's RMS.

### 3. Critique of the Plan (P0–P4)
*   **P2d (Disable allow_voice_reuse) is a trap that will crash episodes.** 
    *   *Why:* `_otr_voice_bank.py` explicitly states that if `allow_voice_reuse` is false and the ladder is exhausted, it raises `VoiceCastingError`.
    *   *Risk:* If an episode script happens to have 15 male characters, and your bank only has 14 male refs, the 15th character will hard-crash the render. You *must* leave `allow_voice_reuse=True` to guarantee the "fail-closed but always render" (PD1) contract, relying instead on P3b (the per-episode used-set) to defer reuse until the pool is empty.
*   **P1.1 (Cap total mass / normalize to <= 1.2) is risky.**
    *   *Why:* If a line scores 1.0 `afraid` and 1.0 `surprised`, normalizing them down to 0.6 each might just wash out the emotion, leaving the model in an uncanny valley. IndexTTS2 might require saturated single vectors to sound convincing. 
    *   *Better approach:* Instead of normalizing the sum, use a `max()` winner-takes-all approach, or cap the *secondary* emotions while letting the primary emotion stay at 1.0.
*   **P2a (F0 ranking) is flawed.** As noted in Craft, F0 does not equal whiny. Do not waste operator time listening to a list sorted purely by F0. Sort by a combination of F0 and spectral centroid.
*   **P0-zero (Bark check) is brilliant.** Given G2/G9, the operator's ear might literally be hearing Bark. Doing this first with zero code is the mark of a mature engineering plan.

### 4. Wildcards (3 unconventional 30-minute experiments)

**Wildcard 1: The "Angry Baseline" Injection (Speculation)**
*   *Hypothesis:* In many TTS models, the "calm" or "neutral" state is trained on audiobooks/podcasts, which can sound breathy and close-mic'd. "Angry" is trained on shouting/projecting. 
*   *Experiment:* In `_otr_delivery_vector.py`, floor `angry` at `0.15` instead of flooring `calm`. Injecting a tiny amount of anger into every line forces the TTS to "project from the chest" and completely eliminates the pleading tone, without actually sounding mad.

**Wildcard 2: The "Declarative Hack"**
*   *Hypothesis:* The `?` token is the root cause of the pleading prosody.
*   *Experiment:* In `eng_indextts2.py:prepare_text`, add `text = text.replace("?", ".")`. The `_otr_delivery_vector` still sees the `?` (because it runs first and stamps the ledger), so the engine gets the `surprised` vector, but the text prompt forces a downward, confident vocal inflection. 

**Wildcard 3: Reference Clip Pitch-Shifting (Formant Preservation)**
*   *Hypothesis:* The donor pool genuinely lacks heavy male voices, and filtering (P2c) will leave you with too few refs, causing voice-reuse collisions.
*   *Experiment:* Take the 3 best-acted (most confident) male CC0 clips, regardless of their pitch. Run them through a standard audio tool (or a quick `librosa.effects.pitch_shift` script) to drop them by 2 semitones, and save them as new refs (`vz_donor_shifted_1`, etc.). Zero-shot models often clone the shifted pitch perfectly, artificially manufacturing the "chest" you need without requiring new voice actors.