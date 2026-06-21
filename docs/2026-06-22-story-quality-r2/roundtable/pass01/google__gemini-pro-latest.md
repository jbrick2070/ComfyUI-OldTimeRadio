<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The document is a discovery questionnaire and a list of options, not a concrete, build-ready specification. 

MUST-FIX BEFORE BUILD:
1. [Q1 / A] The document identifies the `music_inter` placeholder leak but leaves the solution open. Fix: In `_otr_outline.py:1511`, change the hardcoded `intent` to a bracketed non-spoken cue (e.g., `"[Music: transitional bridge]"`). Bracketed text is standard for captions and typically ignored by TTS, solving both the spoken and captioning warts without breaking the timing slot.
2. [Q2 / B] The announcer close intent is currently generic. Fix: In `_otr_outline.py:1529`, update the announcer close `intent` from `"Close the episode and tag the broadcast."` to `"Close the episode with a final dramatic image. Do not summarize the theme or state a moral."` This directly alters the prompt for the line composer without requiring schema changes.
3. [Q3 / C] The document leaves the weak-model fix unresolved. Fix: Implement the cliche ban-list as a deterministic check in `_otr_line_hygiene.py` (e.g., `has_cliche(text) -> bool` checking against a frozenset of "playing with fire", "changes everything", etc.). Like `is_truncated`, this forces the existing composer to reroll the line. It is model-agnostic, zero-latency, and ledger-safe.

SHOULD-FIX:
4. [Q3 / C] To address the "meandering stage-business" without adding new LLM calls, inject `character_a_wants` and `character_b_wants` (already available in `_otr_dramatic_state.py`'s `DramaticState`) directly into the line composer's system prompt for voiced beats. This forces weak models to ground the dialogue in the structural conflict.

CUT THESE (over-engineering):
5. [Q3 / C] Cut the proposed "per-line 'make THIS line serve the opposed want' pass". Adding an extra LLM call for every single line of dialogue will severely degrade generation latency and increase failure surfaces, especially on weak/local models. Deterministic hygiene + prompt injection achieves the goal much cheaper.