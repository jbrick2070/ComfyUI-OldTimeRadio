<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

**1. Blind spots (whole-chain causes the DOC under-weights or omits)**

- **Ref acoustics beyond the P2a checklist (duration/SNR/F0/crest/chest-weight).** The refs are peak-normalized to 0.97 at 44.1 kHz (G14) with no RMS/LUFS target and no explicit check for speaking-rate or micro-pauses. A fast, clipped, or "polite-reading" donor (even if chest-weighted) will produce a zero-shot clone whose prosody is already pleading or thin before any emotion vector is applied. The cloner copies *read style* (H1) more aggressively than the DOC's current metrics capture; a 6–8 s ref that is mostly one long sentence with rising terminal intonation will bias every line.

- **Resampling path inside IndexTTS2 (UNVERIFIED).** The worker receives a 44.1 kHz prompt but the engine is native 22.05 kHz (eng_indextts2.py:22050). If the internal resampler is linear-phase or naive, it can remove low-mid body or introduce pre-echo that psychoacoustically reads as "whiny" once the clip is placed against a compressed Kokoro announcer and a music bed (G15/H2c). P0 cell 9 only tests one direction; it does not test the actual prompt the worker ships.

- **Psychoacoustic interaction of dynamic range + music bed.** Even if a character clip is loudness-matched solo, its crest factor and low-mid energy relative to the *integrated* mix can still make it sound thin or pleading. The DOC notes the mix-perception channel but does not propose measuring character clips against the *final* music+announcer bed before any post step.

- **Seed + per-line variance inside the cloner (not just episode_seed).** The deterministic_inference wrapper ( _otr_voice_node_common.py) seeds only the outer RNG; IndexTTS2's internal sampling (temperature, top-p, or diffusion steps if any) may still vary with the exact text length or punctuation count. A "?"-heavy line can therefore receive a different internal prosody contour even at the same seed.

- **Parenthetical leakage into the vector (G13 already flags this, but the fix is only "derive from prepared text").** Stage directions that survive clean_spoken_text (e.g., "(softly, almost crying)") can still trigger "sad/afraid" before the punctuation lever is applied, and the vector is computed on the raw line text in the current path.

**2. Craft moves a voice director or post engineer would make (automatable, one-person shop)**

- **"Chest proxy" reference transform (speculation).** Before adding a ref to the bank, run a cheap, deterministic 80–300 Hz shelf (+2–3 dB) + gentle 3–5 ms lookahead compressor on the donor WAV and store the processed version under a new `voice_ref_id` with `style_tags: ["chest_proxy"]`. This is a 30-line addition to `otr_dl_indextts2_refs.py --audit` and costs one extra file per donor. It directly attacks H1 without new actors.

- **Per-character "anchor line" calibration.** For every recurring character (Hayes, Gulliver), keep one fixed neutral declarative sentence in the bank metadata. At cast time, render that anchor once (vector-off, chosen alpha) and store its measured RMS and spectral centroid. Later lines are then gain-offset to match the anchor's integrated loudness before mixing. This is a tiny ledger extension and gives an objective "body" target without manual mixing.

- **Punctuation-aware prosody gate (cheaper than full P1 table).** Before any vector work, count terminal "?" and "!" per line. If the count exceeds a threshold (e.g., 2) *and* the line also contains an afraid/sad cue, automatically drop the surprised component by 0.4 before normalization. This is a one-line change in `deterministic_delivery_vector` and directly falsifies H2b without waiting for the full v2 table.

**3. Critique of the plan (mis-order, risk, cheaper alternatives)**

- **P0-zero is correctly first, but P0 cell ordering is suboptimal.** The punctuation-softened text test (cell 6) and worst-ref test (cell 7) should run *before* the alpha sweep (cell 3). If punctuation alone or a thin ref alone reproduces the whine, the alpha sweep is largely irrelevant and the cheapest dominant step is the `OTR_CHAR_DECLARATIVE_Q` flag (already noted as cheap in the plan).

- **P2d (`allow_voice_reuse=false`) is lower value than claimed.** It only prevents mid-episode voice collision; the uniform lottery (G4) and lack of tier filtering are the real sources of thin refs. Flipping reuse without P2c tier pre-filtering just forces more bark fallbacks on small pools.

- **P3a persistence is correctly prioritized but the "durable ledger route" dependency is riskier than acknowledged.** The plan already flags that the video-lane restamp path must be verified; if it is not reachable from the audio nodes, the only guaranteed persistence is the render_log STRING. That is acceptable for diagnosis but not for long-term curation claims.

- **P1 "no unconditional calm floor" is the right conservative choice, but the plan should also explicitly test a *floor on angry* (Gemini cell) as a separate P0 condition.** An angry floor of 0.10–0.15 is cheaper than any table change and directly attacks "pleading" projection.

- **Cheaper dominant step that beats several P1 items:** once P0 picks an alpha, shipping only `OTR_INDEXTTS2_EMO_ALPHA` (already accepted by the worker) plus the one-line punctuation gate above gives most of the win with zero table versioning.

**4. Wildcards (30-minute experiments, speculation marked)**

- **Speculation: "output-as-ref laundering".** Render a short neutral paragraph on the best current male ref (vector-off), write the resulting WAV back into the refs directory under a new sha'd id, and test it as a prompt. One good donor + one bad donor only. If the cloner "learns" its own chest weight, this is a free manufactured reference; if it compounds artifacts, it fails fast.

- **Speculation: F0-contour veto on the rendered clip.** After every IndexTTS2 render, run a quick `librosa.pyin` on the output and reject/re-render if terminal rise exceeds a threshold (unless the line is tagged as a genuine question). 30 min to wire into the worker response path; directly falsifies H2b at render time.

- **Speculation: "anti-whine" negative prompt via the calm dimension.** Force `calm=0.6` (instead of the derived value) on any line whose raw vector has >0.4 surprised+afraid mass. This is a 5-line patch in `emo_list` and tests whether the calm dimension is an active suppression mode (G8) rather than a rest state.