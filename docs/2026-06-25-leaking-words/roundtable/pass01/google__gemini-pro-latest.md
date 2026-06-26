<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The proposed strategies directly violate the stated hard invariants (transport-agnosticism, no workflow-JSON change, determinism).

MUST-FIX BEFORE BUILD:
1. **[Strategy A] Violates workflow-JSON and determinism invariants.** You cannot add a "single FINAL LLM-cleaner pass" without adding a new LLM node to the workflow JSON (violating "NO workflow-JSON change"). Furthermore, applying an LLM pass to frozen text violates the "deterministic" invariant for the freeze/TTS stage. 
   *Fix:* Cut Strategy A entirely. The final freeze stage must remain pure Python / deterministic as seen in `_otr_ledger_scrub.py`.

2. **[Strategy B] Violates model/transport-agnostic invariant.** Relying on GBNF/grammar constraints physically ties the pipeline to llama.cpp/Ollama or specific API providers that support structured outputs. You cannot implement this while remaining "model/transport-agnostic".
   *Fix:* Cut Strategy B. Do not rely on engine-specific decode constraints.

3. **[The leak classes: News-bleed] Contradicts the deterministic invariant.** The document correctly diagnoses that news-bleed "needs a meaning-level check" and is "NOT a stage direction". However, you cannot build a meaning-level check downstream without an LLM, which is forbidden by the invariants. 
   *Fix:* Move news-bleed resolution exclusively to Strategy C (Upstream prompt) and Strategy D (Accept local ceiling). Do