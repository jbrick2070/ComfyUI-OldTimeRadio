<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no — the document is a problem analysis and options list, not an actionable architecture plan; it lacks a concrete chosen strategy and contains one materially inaccurate root-cause claim when checked against the grounding.

MUST-FIX BEFORE BUILD:
1. [missing strategy / architecture] The document ends with open questions and does not commit to a chosen architecture. A design document must propose a concrete solution. Add a “Chosen Architecture” section that selects from options A–E (or a combination), explains how it satisfies the invariants, and outlines the implementation steps (e.g., which scrubs to keep, which new component to add, what the fallback path is). Without this, no build can start.
2. [claim contradiction – Why it still leaks, paragraph “The detectors are WHITELIST/REGEX-bound…”] The statement “Every deterministic gate keys on _NARRATION_VERBS …” is false for the leading stage-direction strip. Grounding shows `scrub_leading_stage_direction` → `_leading_stage_strip`, which uses structured guards (lowercase start, copula/modal, pronoun roots, dialogue starters, terminal punctuation, boundary detection) and never references `_NARRATION_VERBS`. The “Gasping” leak was missed because the lead word was capitalised, not because the verb was absent from a whitelist. Fix: rewrite the analysis to attribute the leak correctly—the leading scrub’s lowercase-first requirement—and distinguish detectors (`is_third_person_action_clause`) that do use the verb set. Otherwise the root cause is misdiagnosed, leading to a flawed strategy selection.
3. [invariant conflict unexamined] Option D (“make the frontier writer the recommended default”) may break the invariant “model/transport-agnostic” (it effectively deprecates local lanes rather than fixing them). The document must explicitly reconcile any retained option with the stated invariants, or remove options that cannot meet them.
4. [missing feasibility against invariants] Options A (LLM-cleaner per line) and B (constrained generation) are not assessed for determinism, offline capability, or model agnosticism. A final architecture plan must either include a brief feasibility analysis for each candidate against the invariants, or state what experiments will prove feasibility before a full commit.

SHOULD-FIX:
1. [Section “What has ALREADY been tried”] The bullet “G1 (this session): measured the abstain residual … which is WHY a binary classifier lane was DROPPED” is a decision rationale, not a “tried” technique. Move it into a separate “Decision taken” or “Why we did not pursue X” list to avoid confusion.
2. [missing context on existing gated features] The grounding shows large gated additions (L1 objective-literal, L7 action split, L12 ungrounded crisis density, L4 transcript sanitizer). The document mentions none of these in “What has ALREADY been tried”. Clarify which are currently active in production vs. dormant/flags, so the panel knows the baseline when considering new strategies.

OPTIONAL / NICE-TO-HAVE:
- Add a short table mapping each leak class (stage-direction, news-bleed, caps-vocative, malformed quotes) to the specific existing defence that was supposed to catch it and the reason it failed; this would make the root cause per-leak class more precise.

CUT THESE:
- None. The document proposes no new overscoped features; it is primarily analytical.

[ASSUMPTION] The “News-bleed” class is correctly described as a semantic content problem that shape/whitelist detectors cannot catch; the grounding confirms no existing pass addresses it (there is no named-entity or fact-verification step in the provided code), so that part of the analysis is sound.