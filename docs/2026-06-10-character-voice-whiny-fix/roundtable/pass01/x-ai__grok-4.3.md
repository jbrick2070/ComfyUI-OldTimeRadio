<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

**1. Blind spots (whole-chain causes the DOC under-weights or misses)**

- **Reference acoustics beyond F0/duration/SNR proxy (P2a audit).** The bank refs are amateur donations (G3). Even if median F0 is low, mic coloration (cheap dynamic mics with rolled-off lows or 3–5 kHz peak), close-mic proximity effect absence, or read style (flat/monotone vs. naturally weighted) can make the zero-shot clone thin. The current pyin-based audit will rank high-F0 males first but will not surface “warm donor recorded on a bright mic” vs. “thin donor on a warm mic.” This is falsifiable: after P2b listen pass, measure spectral centroid or low-band energy (80–300 Hz) on the ranked list; if the top “warm” refs still produce whiny output, mic coloration is the missing variable.

- **Zero-shot cloner interaction with weak refs at full emo_alpha (G7).** IndexTTS2’s behavior on short or low-energy refs when a saturated multi-hot vector (afraid+sad+surprised) is supplied is marked UNSURE in the DOC. The worker passes the vector at alpha 1.0 unconditionally; if the model’s internal prosody model amplifies any residual high-frequency energy or breathiness in the ref, the result is pleading. P0 cell C (worst-ref + no vector) only partially isolates this; a missing cell is “worst-ref + saturated vector at alpha 0.65.”

- **Psychoacoustics of lone voice in a music-bed mix.** The DOC correctly scopes to upstream TTS (frozen master-mix invariant), but a voice that measures “correct” in isolation can still read thin once ducked under music. Low-end presence and slight dynamic-range compression are post-TTS, yet the plan never checks whether the rendered character clips already lack energy below ~250 Hz before they reach the mixer. This is testable with a 30-second spectral analysis of a P0 render vs. the Kokoro announcer on the same bed.

- **Punctuation double-dipping + interrogative cue overlap (already noted in Seat A) is worse than described.** Lines containing both “what”/“how” and “?” receive +0.5 (keyword) +0.5 (punctuation) to surprised before the /_CAP normalization. Because there is no total-mass cap, a single dramatic question line can still push surprised near 1.0 even after the proposed v2 table change if other cues are present.

**2. Craft (voice-director / post-engineer moves that are automatable in a one-person shop)**

- **Reference “weight” pre-filter.** Before any human listen pass, compute a cheap scalar “chest weight” = (RMS energy in 80–300 Hz band after 6 dB/oct tilt) / total RMS on every ref. Rank the male pool by this metric and surface only the top half for the P2b listen. This is ~20 lines of librosa code, runs on CPU, and directly attacks the timbre lottery (G4) without buying new refs.

- **Delivery-vector “de-bleat” rule.** Add a one-line heuristic in `deterministic_delivery_vector`: if the line ends with “?” and contains any afraid/sad cue, subtract 0.3 from surprised before normalization. This is a classic voice-director note (“don’t let every question sound shocked”) and is fully deterministic.

- **Per-ref seed offset for thin refs.** In `_resolve_clone_ref_path`, after tier filtering, add a small deterministic offset (+17 or similar) to the per-line seed for any ref whose quality_tier == "b". This gives IndexTTS2 a different prosody draw without breaking C7 determinism or requiring new models. Cheap and scoped to the exact refs that survive the audit.

**3. Critique of the plan (P0-zero..P4)**

- **P0-zero is correctly first and cheap; P0 2x2 matrix is under-specified.** The matrix should also include the single worst-ranked ref (by the new chest-weight metric) at alpha 0.65 so the operator can see the interaction term (H3) in <30 min. Without it, the decision gate “which of P1/P2 is primary” rests on incomplete data.

- **P1 delivery table v2 is high-value but the calm-floor change is risky.** Setting a hard floor of 0.15 on calm when arousal is low will change the rendering of every neutral or low-tension line. The property tests listed are necessary but insufficient; they must also assert that a pure-neutral line still produces calm ≥ 0.7 after the floor (otherwise the “neutral” escape hatch in G8 is broken).

- **P2d (set allow_voice_reuse=false) is low-cost and should be done immediately** (even before the listen pass). With 14 male refs it is almost certainly an oversight from the 3-ref era; leaving it true guarantees same-voice collisions (G11) on every episode.

- **P3a logging is the single highest-ROI item after P0.** Without per-line engine + voice_ref_id in the ledger, every future curation claim is untestable (G9). It should be elevated to block P2c.

- **Nothing in P1–P3 is mis-ordered once P0 data exists.** The only cheaper dominating step is exposing `OTR_INDEXTTS2_EMO_ALPHA` (already one-line) and running the P0 matrix at 0.65 before touching the delivery table at all.

**4. Wildcards (≤3 unconventional 30-min experiments)**

- **Speculation.** Run the same line+ref+seed three times with the emo_vector rotated by one dimension (afraid→sad, sad→surprised, etc.) and listen for whether the “whiny” quality tracks a specific emotion axis rather than total mass. If it does, a one-line remapping table becomes higher-leverage than the full v2 rewrite.

- **Speculation.** In `prepare_text`, replace every terminal “?” with “.” for character lines only (announcer untouched). This is a 5-line change, fully deterministic, and directly attacks the punctuation double-dip without touching the emotion table. Re-render one smoke episode; if the pleading quality drops measurably, the vector math is secondary to how IndexTTS2 interprets “?” at inference time (UNSURE).

- **Speculation.** After the P2a audit produces the ranked thin-risk list, take the single worst male ref, pitch-shift it down 4 semitones with a phase-vocoder (librosa), re-hash, and insert as a new “vz_donor_*_down” entry with quality_tier forced to “b”. Render the same line against both versions at identical seed. If the down-shifted version is subjectively weightier, the zero-shot model is sensitive to absolute pitch of the ref—an observation that would justify a permanent low-cost pitch-normalization step for all refs below a chest-weight threshold.