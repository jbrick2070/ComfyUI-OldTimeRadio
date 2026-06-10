# Pass 2 judgment -- live 4-model panel (2026-06-10, ~$0.24)

Same panel as pass 1. Judge/grounder: Claude (Fable). Two claims settled by MEASUREMENT (scripts in this folder: measure_refs.py).

## CONFIRMED (folded into plan v3)
1. **Per-line prosody seed instability** (GPT A): engine_seed derives from stable_line_seed per line -- same character re-rolls its delivery every line. Verified in _otr_voice_node_common.py. -> G17/H2d, P0 cell 8 (fixed-char-seed panel), OTR_INDEXTTS2_SEED_MODE option (P0-gated).
2. **P0 must include real offending lines + a line-length cell** (GPT B, Gemini short-line panic, GPT D micro-lines). -> P0 line set expanded.
3. **Minimal render_log observability belongs FIRST** (GPT "P3a mis-ordered", Grok "render_log acceptable for diagnosis", DeepSeek "verify durable route early"). -> New P-OBS step ahead of everything; durable ledger stays P3a.
4. **Calm-dominant property test was premature** (GPT): would bake in behavior P0 cell 6 may disprove. -> Property re-written: neutral sends no aroused dims; calm policy = P0 winner.
5. **P2d reuse-flip gives false uniqueness until P3b** (GPT + Grok + DeepSeek convergent; G16: the render-time fallback uses the full bank with reuse=True and no tier filter). -> P2d DEFERRED until P3b; P2c/P3b spec now requires the render-time path to share CastLock's filtered-bank policy (DeepSeek's tier-reject leak).
6. **Punctuation lever direction unknown** (Gemini: `?`->`.` risks depressed/robotic; `?!` or `...` may keep energy; GPT: scope to short non-informational questions; DeepSeek: vector still pushes afraid with the lever on). -> P0 cell 3 = variant test; lever ships variant-from-data, scoped, default OFF.
7. **Emotion-soup cap candidate** (Gemini): "primary + 1 secondary max" added to the P0 cap-strategy sweep.
8. **Trimmer latent bug** (Gemini, mechanism CONFIRMED in code -- longest voiced run breaks at every unvoiced frame; <2s fallback to first-12s raw is what actually ships). -> Fix gap-tolerant merge before future downloads.
9. **MEASURED: the 4 voice-zero originals are loudness outliers** (found while checking #8): stuart_bell RMS 0.027 / peter_yearsley 0.035 / bill_boerst 0.052 / caro_davy 0.061 vs donors 0.089-0.187 (0.97 peaks). Quiet low-effort prompts can clone as weak/pleading. -> G18; P0 dual baseline (quiet original vs hot donor); P2d ref hygiene (loudness-normalize the originals, versioned).
10. **Audit metric upgrades** (GPT F/G, DeepSeek D): voiced-frames-only spectral ratios; male/female chest bands; spectral slope; pause-RMS SNR proxy; DC offset; breath-tail ratio; composite = must-listen priority, never auto-reject. -> P2a.
11. **Cloned-output spectral-tilt check** (DeepSeek 1): measure model-intrinsic thinning on matrix outputs. -> P0 spectral report layer.
12. **Generated-line acoustic report as regression metric** (GPT): final-rise + vector-mass flags on rendered lines. -> P4.
13. **Lottery exposure simulation** (GPT Test 3, DeepSeek 6): 100-seed cast simulation, CPU. -> P3c.
14. **Solo-vs-mix split EARLY on existing audio** (DeepSeek, GPT): -> P0a (was a P4-only gate).
15. **Whine-from-writing split** (GPT I): neutral-rewrite cell measures the script's contribution without reopening the writer. -> P0 cell 9.
16. **Text-normalization residual audit** (GPT C): corpus report for em-dash/colon/caps/long-parens after prep. -> P1.3.
17. **Pack-path sample-rate assert** (GPT H). -> P-OBS.

## MISREAD / REFUTED (discarded, with evidence)
- **Gemini "vowel-hold extraction shipped moan refs / dominates H1"**: REFUTED by measurement -- zero refs under 5s; donors flat 10.00s; narration 12.00s (fallback branch); originals 5.9-10.8s. Mechanism kept as latent bug (#8); catastrophe claim discarded.
- **DeepSeek "vector-omitted defaults to calm=1.0"**: the worker omits all-zero vectors; P0 cells 1 vs 6 already split omitted vs calm=1.0.
- **Gemini "P2a audit must wait for the trimmer fix"**: moot for the CURRENT bank -- the shipped refs never went through a working trim (first-12s fallback / flat-10s path); audit proceeds on the files as they are.

## REJECTED (out of scope / no mechanism)
- Render-time F0-contour veto + re-render loop (Grok): render cost, loop risk; the report metric detects, the operator re-rolls.
- Announcer-envelope style transfer (DeepSeek, self-flagged "likely too advanced").
- Full dialogue-bus post chain inside this fix (GPT B-craft): stays the scope-fenced H2c lane; only the free offline test + the versioned default-off loudness option are sanctioned.

## Convergence status after pass 2
NOT converged -- pass 2 produced material items (G17/G18, P-OBS reordering, P2d deferral, P0 restructure). Plan v3 goes to pass 3. Expect convergence: pass-2 reviews already overlap pass-1 heavily outside the items above.
