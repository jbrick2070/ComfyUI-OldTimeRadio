# BINARY ADDENDUM JUDGMENT -- CONVERGED (1 round)

All three panel + the anchor converged; nothing to discard (every panel claim
grounded true against `_otr_line_hygiene.py`).

ACCEPTED (folded):
- Prune to ONE app: dialogue vs stage-direction; CUT #2/#3/#4 (all 3 panel
  unanimous, each for a grounded reason -- O(N) loop / Levenshtein-already-resolves
  / segmentation-not-hygiene).
- Per-SPAN not per-line (GPT#1 + Gemini#1) -- whole-line A/B cannot clean a MIXED
  line; operate on deterministically-isolated spans.
- Explicit `HIT|CLEAN|ABSTAIN` tri-state, created in the PURE hygiene layer; binary
  fires only on ABSTAIN (GPT tri-state + the pure-module contract GPT#4).
- `binary_decide` in the LLM layer, thin wrapper over the pass04 core; "A"/"B" not
  yes/no (refusal-safe, Gemini); optional surrounding-line context (Gemini); strict
  single-decisive-token parse, conflict -> None (GPT#2 + Gemini#2).
- SHADOW MODE first; validate across local+remote before mutation; the "LLMs
  reliable at binary" premise is a HYPOTHESIS (all 3).
- Build GATES: G1 MEASURE the abstain residual vs the existing two-tier detectors
  (`detect_stage_business_for_reroll` Tier-2 already covers undelimited/embedded --
  DS#7) -- may shrink/kill the lane; G2 byte-identity of abstain given
  `segment_double_quotes` curly->straight folding (GPT#3).
- Determinism caveat: only the default path + fallback + parser must be
  deterministic; the remote call need not be seed-stable -> cache (all 3).

REJECTED / MISREAD: none -- the two "no" verdicts (GPT, Gemini) were "no, the
ADDENDUM is over-scoped/line-level", not "no, the lever is unsound"; both said
build Application 1 narrow. Folding their fixes turns it to yes.

CONVERGENCE: reached in one round (a focused addendum on top of an already-
converged plan). Verdict: SOUND but GATED -- build pass04 (C0-C6) first; this lane
sits on top, gated on G1 (residual exists) + G2 (byte-identity). prod/main GATED.
