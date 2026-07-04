# Claude anchor review -- r1 (arc/coherence) -- WIDGET_SURFACE_AUDIT.md
VERDICT: SOUND WITH GAPS. Self-critique (Claude authored the doc).

MUST-FIX
1. COVERAGE HONESTY (CONFIRMED): the doc says "~45 of ~60 deep-checked widgets are clean KEEP" but ~65 widgets only got the mechanical consumption grep, not the semantic pass (env-shadowing, mode-conditional). The doc should state the two-tier confidence explicitly per node, or the reader over-trusts the KEEP set.
2. VERIFICATION DEBT (CONFIRMED): several evidence line numbers came from a sonnet subagent and were not independently re-read by the judge (e.g. OTR_LedgerScriptWriter.py:1662-1682, :2098-2170, registry.py:243-365). Before any cleanup change lands, each cited line must be re-grounded. Mark them verify-at-build.
3. NODE 86 DECISION FRAMING (CONFIRMED): "drop OTR_CaptionBurn from the graph" is stated as an option without noting link re-wiring consequences (links 247 in / 266 out must be re-spliced 84->93) and that node 93's burn_captions=true was a deliberate bake (2026-06-10 capstone). The doc should carry that context so the operator call is informed.

SHOULD-FIX
4. The "HIDE" verdict is ambiguous in ComfyUI terms: there is no native hide -- it means delete-from-INPUT_TYPES (positional risk) or leave-and-tooltip. Cleanup plan batch 1 says "or leave as hidden constants" -- pick one term.
5. The doc does not check subgraph/muted nodes (mode 2/4) -- inventory showed none flagged, say so explicitly.
6. No check of tooltips already present -- some "RENAME-CLARIFY" rows may already have tooltips in INPUT_TYPES dicts; batch 2 should diff against existing tooltips.

UNVERIFIABLE (carry as hypotheses)
- Sonnet claim that GATE B profile applier does not touch stereo_policy widgets: plausible (widget_mapping.json cited) but not re-read by judge.
