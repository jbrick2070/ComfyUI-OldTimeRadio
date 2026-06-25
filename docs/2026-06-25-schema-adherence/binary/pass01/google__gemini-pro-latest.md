<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The core primitive is sound, but 3 of the 4 proposed applications fundamentally misunderstand the data flow or problem space, turning the plan into a massive latency trap.

MUST-FIX BEFORE BUILD:
1. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS] Application 2 (Edit vs no-op) is chronologically impossible. The Script Doctor generates a list of edits. You cannot ask a binary "does this row need a real replacement?" *before* the model has analyzed the text and decided to emit an edit row, unless you loop over every single line of the script with a binary LLM call ($O(N)$ latency explosion). Fix: Drop Application 2. The `payload_null_repair` in `_otr_repair_prompts.py` already handles this correctly.
2. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS] Application 4 (Beat over-length) proposes a per-boundary "split here? yes/no" loop. Iterating over every sentence or token boundary with an LLM call to find a split point is an unbounded $O(N)$ latency trap that will cripple generation speed. Fix: Drop Application 4.
3. [CANDIDATE LEDGER-INTEGRITY APPLICATIONS] Application 3 (Speaker membership) solves a problem the system doesn't have. The pipeline already deterministically knows if a name is in the locked cast (this is what triggers `cast_membership_repair` in the grounding code). The LLM is needed to *resolve/remap* the phantom name to a valid one, which a binary yes/no cannot do. Fix: Drop Application 3.

SHOULD-FIX:
1. [THE PROPOSAL] The `binary_decide` signature (`*, slot_fn, question, text, ...`) lacks context. Classifying an ambiguous bare string often requires knowing the previous or next line. Fix: Add an optional `context` or `surrounding_lines` parameter to the signature.
2. [OPEN QUESTIONS FOR THE PANEL - Q1/Q5] [ASSUMPTION] The plan assumes a literal 1-token output space. Local chat models (Mistral/Gemma) forced into instruction-following will frequently output conversational filler ("Yes, this is dialogue"). Fix: The parser must explicitly scan for the first decisive token (e.g., regex `\b(A|B|yes|no)\b` ignoring case) rather than expecting the LLM to halt after one token.
3. [CONSTRAINTS] "Byte-identical for the local DEFAULT happy path". If `binary_decide` is an LLM call, it inherently breaks byte-identity unless the local default *never* hits the escalation path. Fix: Explicitly document that the local default path forces the deterministic fallback immediately when regex abstains, bypassing the binary LLM call entirely.

OPTIONAL / NICE-TO-HAVE:
- For Q1 (Output contract): "A"/"B" is generally more reliable across frontier and local models than "yes"/"no", as "yes/no" often triggers safety/alignment refusals depending on the prompt framing.

CUT THESE (scope / over-engineering):
1. Candidate Applications 2, 3, and 4. They are logically flawed, solve the wrong problem, or introduce massive latency bloat. Safe to cut because Application 1 (Dialogue vs Stage Direction) is the only use-case that actually fits the proposed "Regex abstains -> Binary escalation -> Deterministic fallback" architecture.