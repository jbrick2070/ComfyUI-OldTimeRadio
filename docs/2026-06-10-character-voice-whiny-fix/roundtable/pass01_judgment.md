# Pass 1 judgment -- live 4-model panel (2026-06-10, $0.2204)

Panel: openai/gpt-5.5-20260423, google/gemini-3.1-pro-preview-20260219, x-ai/grok-4.3-20260430, deepseek/deepseek-v4-pro-20260423. Judge/grounder: Claude (Fable). 31 distinct claims after dedupe.

## CONFIRMED (grounded against the repo; folded into plan v2)
1. Adapter prepare_text uses clean_spoken_text only -- full normalization skipped; punctuation rides into the engine (GPT 1.4; eng_indextts2.py:162-166). -> G12, P1.3.
2. Punctuation is an independent prosody channel ("?" pleading contour) (GPT/Gemini/Grok convergent; effect size UNSURE). -> H2b, P0 cell 6, OTR_CHAR_DECLARATIVE_Q experiment.
3. Delivery vector derived from RAW pre-strip text; stage directions cue then vanish (GPT 1.5; _otr_voice_node_common.py:407-414). -> G13, P1.1 derive-from-prepared.
4. Ref preprocessing peak-normalizes (0.97) with no LUFS/RMS/clipping/SNR screen; verified 3 call sites (GPT 1.2, DeepSeek). -> G14, P2a metrics.
5. Refs written 44100 Hz vs engine 22050; internal resample behavior UNSURE (GPT 1.3, Gemini). -> G14, P0 cell 9.
6. Read STYLE is cloned as aggressively as timbre; F0-only thin-ranking is flawed (ALL FOUR). -> P2a composite metrics (chest-weight 80-300 Hz/total RMS, centroid, final-rise, crest, clipping, nasal band), P2b audition reel.
7. Audition reel beats raw-WAV listening -- test the cloner, not the source (GPT 2.2). -> P2b.
8. Calm floor 0.15 is risky/unjustified before P0 cell-4 data (GPT 3.4, DeepSeek, Grok property-test). -> dropped from P1; property test pure-neutral -> calm>=0.7 added.
9. Mass-cap value AND strategy are empirical; normalize-sum may wash out emotion (GPT 3.5, Gemini winner-takes-all, DeepSeek soft-penalty). -> P0 sweep; strategy chosen from data.
10. Alpha default must be swept, not assumed 0.65 (GPT 3.3, Grok, DeepSeek); knob itself is the cheapest dominant step. -> P0 cell 3; P1.2 default = P0 winner.
11. P0 matrix one-line/one-seed too small (GPT 3.2, Grok interaction cell). -> 3 lines x 3 seeds + worst-ref x 0.65 cell.
12. Solo-vs-mix unsplit; announcer (kokoro) is dense/compressed vs dynamic indextts2 output (GPT 1.8/3.11, Gemini, Grok, DeepSeek). -> G15, H2c, P4 gate + sanctioned loudness experiment.
13. P3b naive used-set would re-voice a character mid-episode; needs per-char resolve-once cache (GPT 3.9; _resolve_clone_ref_path is inside the line loop). -> P3b design.
14. P2d (reuse=false) does NOT crash: CastLock catches VoiceCastingError per char ("NOT cast" -> continue -> render-time resolution); CastLock already keeps a used-set the widget defeats (verified cast_lock.py:347-375). -> G16; P2d re-scoped as safe canonical-graph config win; P3b still needed for preserve_ledger path.
15. assign_voice_for_slot has no tier concept; tier filtering must pre-filter the pool, preserving the ladder (DeepSeek). -> P2c implementation note.
16. De-bleat rule: terminal-? + afraid/sad cue -> demote surprised (Grok). -> P1.1 v2 table.
17. Ambiguous-gender band (F0 145-185 Hz) must-listen flag (DeepSeek). -> P2a.
18. Audit script today computes only gender/F0/voiced/trim -- claimed SNR/clipping metrics do not exist yet (DeepSeek; classify_and_trim verified). -> P2a scope is real work, not a flag flip.
19. Lead-safe casting buckets via additive style_tags; leads tier-a only (GPT 2.1, variety-collapse tradeoff accepted). -> P2b/P2c.

## MISREAD (discarded, with grounding)
- Gemini: "allow_voice_reuse=false will hard-crash episodes" -- CastLock catches the error per character and degrades to render-time resolution (cast_lock.py:371-373). Conclusion repaired, claim discarded.
- Grok: "flip P2d immediately, before the listen pass" -- safe (per above) but mis-prioritized as standalone; folded into P2d/P3b ordering.
- GPT 3.10 (partial): "ledger restamp will not persist anywhere" -- node-local mutation alone indeed does not persist (that half CONFIRMED and folded), but an additive durable-restamp route exists in the video lane; audio-side reachability is a verify-at-build, not an impossibility.

## UNVERIFIABLE -> P0 cells / verify-at-build
- IndexTTS2 calm semantics (active suppression vs rest) -- P0 cell 4.
- Alpha linearity; mixed-vector blending behavior -- P0 cell 3, sweep.
- Internal ref resampling quality 44.1k->native -- P0 cell 9.
- Per-line context/seed instability magnitude -- 3-seed matrix.
- Read-style metric correlation with perceived whine -- reel data.
- Durable-ledger restamp reachability from audio nodes -- coder verify (P3a).

## REJECTED (no mechanism / out of scope)
- Grok: per-ref seed offset for tier-b refs (a different draw is not a better draw; config noise).
- DeepSeek: runtime anchor-line QA gate inside _resolve_clone_ref_path (render cost + complexity; the offline reel covers it).
- DeepSeek: breath/pause insertion driven by the emotion vector (delivery-profiles v2.1 territory; deferred).
- Full post/mix chain as part of this fix (scope fence) -- only the free offline EQ/loudness test + the versioned default-off loudness-match experiment are sanctioned.

## Convergence status after pass 1
NOT converged -- pass 1 produced material new findings (G12-G16, H2b/H2c, P0 redesign). Plan v2 goes back to the panel as pass 2.
