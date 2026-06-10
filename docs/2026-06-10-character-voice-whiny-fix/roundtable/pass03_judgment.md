# Pass 3 judgment -- live 4-model panel (2026-06-10) -- CONVERGED

Same panel. Judge/grounder: Claude (Fable). Verdict: **the plan structure survived unchallenged** -- all four models endorsed P-OBS-first, the channel framing (H1/H2/H2b/H2c/H2d/H4), the cell ordering, and the P2d/P2e decisions ("the single most valuable change", "masterstroke", "correctly prioritized", "extremely well-staged"). Pass 3 yielded refinements + ONE must-fix implementation trap. Per the convergence rule (no new MATERIAL must-fix beyond fold-in-able items), the loop stops here; v3.1 is the build-ready plan.

## CONFIRMED (folded as v3.1)
1. **Stamped-vector version guard (GPT 1.5 -- the must-fix):** `_render_per_line` prefers any stamped `delivery.emotion_vector`; a ledger carrying v1 stamps would silently defeat the v2 derivation. Live ledgers carry no stamps today (stamp_delivery_vectors is never called in the live path), but the trap is real for any stamped/frozen ledger. -> P1.1 version guard + property test + kickoff line.
2. **One-hot emotion-axis preflight (GPT 1.7/4.1):** the 8-dim EMOTIONS order is an assumed contract, never verified against the installed model. -> P0 preflight (8 renders).
3. **Strip-all-terminal-punctuation variant (Gemini W3 + DeepSeek cell-9 control):** isolates the engine's `?`-token bias completely. -> P0 cell 3 variant.
4. **?-count vs final-rise correlation report (Grok W3):** 10-line analysis on matrix outputs; r > 0.6 -> punctuation lever dominates. -> P0 report.
5. **Trimmer falling-final preference + terminal-question-in-ref flag (Gemini "hanging tail", DeepSeek W3):** a trim ending on a rising contour teaches the clone hesitancy. -> P2d trimmer spec + P2a flag.
6. **Seed-mode interpretation caveats (Gemini 3 + DeepSeek 3 + GPT 3.8):** fixed seed risks robotic sameness; a win can be RNG-path luck; candidates include char-base+occurrence-offset. -> P0 cell 8 caveats; option default stays `line`.
7. **Loud-baseline by projection composite, not raw RMS (GPT 1.3):** high RMS can be compression/noise. -> P0 cell 1b.
8. **Concrete P0a recipe (DeepSeek proximity-EQ + GPT 4.3).** -> P0a text.
9. **P3a cache-invalidation caution (DeepSeek):** a mid-graph ledger write must not invalidate ComfyUI node caching; render_log stays the floor. -> P3a.
10. **IndexTTS2-lane determinism caveat (GPT 1.12):** warn_only inference means same-seed re-renders may not be bit-exact on this lane; P4 regression uses acoustic tolerance there (bark/master byte contract untouched). -> verify-at-build + P4.
11. **Normalize-the-originals NOW, never silently replace (Gemini 3 + GPT 3.9):** new ids + shas. -> P2d wording.
12. **prepared-text head in P-OBS log (GPT 3.1).** -> P-OBS/P3a field.
13. **Appendix adds:** prepend-and-slice authority onset (Gemini); authority-sandwich / whisper-transcript declarative re-trim refs (GPT W1 + DeepSeek W3).

## DISCARDED / OUT OF SCOPE
- Full output-side dialogue bus / parallel body compression as part of THIS fix (GPT 2.3/W2, DeepSeek craft) -- stays the scope-fenced H2c lane; the sanctioned offline test + versioned default-off loudness option already cover the decision data.
- Per-character noise-schedule / cross-synthesis envelope morphing (DeepSeek) -- no API surface, self-flagged speculative/complex.
- "Wandering prosody is a feature, drop cell 8" (Gemini) -- overruled: cell 8 is a TEST with caveats, not a commitment; H2d evidence is worth one panel.
- Plosive-energy + room-tone audit metrics (DeepSeek) -- optional extras only; the metric list already saturates the must-listen ranking (noted, not required).

## Convergence
Pass 1: 5 new grounded findings + plan restructure. Pass 2: 2 measured findings + reordering. Pass 3: 0 structural, 1 implementation trap, rest refinement. Marginal return is below the bar -- **CONVERGED at v3.1.** Panel spend total: see run.log files (pass01 $0.2204 + pass02 + pass03).
