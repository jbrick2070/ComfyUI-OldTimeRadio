<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

This is an exceptionally well-documented problem space. The roundtable pass caught several critical pipeline leaks (like the raw-text vector derivation and the `prepare_text` bypass). 

However, looking at this through the lens of audio engineering and DSP, the panel missed a **catastrophic bug in the reference extraction logic**, and is slightly misattributing how zero-shot cloners perceive "whininess." 

Here is my independent review.

### 1. Blind Spots
**The "Vowel-Hold" Extraction Bug (CRITICAL).** 
Look at `otr_dl_indextts2_refs.py`, specifically `classify_and_trim`. It uses `librosa.pyin` to find voiced frames, creates a `vmask`, and then extracts the **"longest contiguous voiced run"** (`if i != prev + 1: ...`). 
*   **The Math:** `pyin` uses a hop length of 256. At 44.1kHz, one hop is ~5.8ms. *Every single unvoiced consonant* (s, t, k, p, f, sh) and every breath lasts longer than 5.8ms. 
*   **The Result:** This script shatters on almost every word boundary. It is not extracting an 8-second sentence; it is extracting the longest *single uninterrupted vowel, hum, or moan* in the donor file. 
*   **The Cloner Impact:** If you feed a zero-shot cloner a 2-second clip of a donor going "aaaaaand" or "uhhhhhh" (which is what this script isolates), the cloner learns a sluggish, drawn-out, consonant-less prosody. **This is the literal acoustic definition of a whine.** This dominates H1.

**RMS vs. Peak Normalization on Refs.**
`otr_dl_indextts2_refs.py` peak-normalizes the extracted clip to 0.97 (`span / peak * 0.97`). If the extracted clip is a low-energy murmur with one stray mic pop, the actual RMS (perceived loudness) of the reference will be tiny. Zero-shot cloners fed low-RMS references often output breathy, weak, or "pleading" audio because they attempt to clone the low vocal effort.

**Short-Line Panic.**
Zero-shot cloners often struggle with 1- or 2-word lines ("What?", "No."). Without enough text to establish a rhythm, they stretch the vowels to fill their internal temporal windows. A stretched "Noooo?" sounds exactly like pleading. The DOC doesn't isolate line-length as a variable.

### 2. Craft (The Audio Post Perspective)
Software engineers treat TTS as a text-in/audio-out function. Audio engineers treat it as an acoustic instrument.

*   **Pre-Master the References:** A cloner mimics the *entire acoustic environment* of the reference, including mic proximity, EQ, and compression. If the CC0 donor was recorded on a phone across the room, the cloner will sound thin. **Action:** Run the `refs/` directory through a batch offline processor (using `pydub` or `ffmpeg`) to add broadcast compression (e.g., a 4:1 ratio, fast attack) and a +3dB EQ shelf at 150Hz. The cloner will clone the "mastered" chesty voice, instantly adding weight to the characters.
*   **Punctuation as Pacing, Not Just Pitch:** In radio drama, `?` causes an upward pitch inflection, but `.` causes a downward pitch drop (finality). If you swap `?` for `.` (as proposed in P1.3), the actor will sound depressed or robotic, not just "less whiny." To keep energy up but avoid the pleading up-talk, audio directors use em-dashes or exclamation points. `?!` often forces a TTS engine to project rather than plead.

### 3. Critique of the Plan
**P0 / P1 / P2 / P3 / P4 Breakdown:**

*   **P1.3 (Punctuation Lever `?` -> `.`):** **Risky / Bad Idea.** Flattening a question to a declarative period will destroy the dramatic acting. It will cure the whine but kill the scene. *Suggestion:* If you must alter punctuation, test `?` -> `?!` or `?` -> `...` which maintain suspense/energy without the pure upward pleading contour.
*   **P2a (Bank Audit):** **Wrong Order.** The proposed `--audit` mode measures chest-weight, nasal-band, and RMS. But because of the Vowel-Hold bug identified above, you will just be measuring the EQ of random isolated vowels. **You must fix the trimming logic before auditing the bank.**
*   **P1.1 (Delivery Table v2):** **Too complex.** The Grok "de-bleat" rule and the keyword math (`surprised += 0.5 * q + 0.3 * ex`) creates "emotion soup." If a line is "What?!", it gets surprised, angry, afraid, and happy all at once. IndexTTS2 likely blends these into a confused, whiny mess. *Suggestion:* Implement a "Winner Takes All" or "Primary + 1 Secondary" cap. Don't let a line have 5 active emotions.
*   **P0 (Audition Matrix):** **Missing Cell.** Add a cell for line length. Test a 1-word line vs a 15-word line. If the 1-word line whines and the 15-word line doesn't, your issue is temporal stretching, not the emotion vector.

### 4. Wildcards (30-Minute Experiments)

*   **Wildcard 1: The "Consonant-Tolerant" Extractor (High Confidence)**
    *   *What:* Rewrite the `classify_and_trim` loop in `otr_dl_indextts2_refs.py`. Instead of breaking the run on a single unvoiced frame, allow gaps of up to ~250ms (about 40 frames at a 256 hop).
    *   *Why:* This will extract actual sentences with consonants and natural prosody, rather than single sustained vowels. Re-download the bank using this fix. I suspect this solves 80% of H1 immediately.
*   **Wildcard 2: Offline "Radio-Ready" Ref Processing (Moderate Confidence)**
    *   *What:* Write a 20-line Python script using `pydub` or `scipy` to batch-process all existing WAVs in `C:\ComfyUI-Models\TTS\refs\indextts2\`. Apply a hard limiter, normalize to -14 LUFS (not peak), and boost 100-200Hz by 3dB.
    *   *Why:* Forces the zero-shot cloner to output dense, chesty, broadcast-ready audio, closing the "Mix Perception" (H2c) gap between IndexTTS2 and Kokoro without touching the pipeline code.
*   **Wildcard 3: The "Exclamation Override" (Speculation)**
    *   *What:* In `_otr_script_prep.py` (which you are modifying anyway for G12), add a regex that converts standalone `?` to `?!` for character lines only. 
    *   *Why:* In many TTS engines (UNSURE if IndexTTS2 specifically), `?` triggers a slow, rising "plead", while `?!` triggers a fast, sharp "bark" of disbelief. It preserves the interrogative nature but forces a higher-energy, faster read style, undercutting the whine.