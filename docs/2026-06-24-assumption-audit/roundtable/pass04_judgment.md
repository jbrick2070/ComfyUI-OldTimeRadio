# R4 judgment -- CONVERGENCE

Spend this pass: ~$0.5005. Campaign total ~$2.2958 (R1 .5947 + R2 .5939 + R3
.6067 + R4 .5005). Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro.

VERDICTS R4: GPT "yes-with-fixes, substance converged"; Gemini "yes-with-fixes,
integration sequencing finally correct"; DeepSeek "yes-with-fixes, nearly
converged". NO new assumptions surfaced -- all R4 items are build-precision
(exact token rules, role-map text, activation gate wording) + a verify-at-build
checklist. CONVERGED.

Folded from R4: seed=cast_seed after cast-lock; explicit activation gate
(story-quality flag -> byte-identical otherwise); reroll determinism (ship-valid-
else-original); split hints (ungrounded_crisis vs missing_conflict_object); exact
object-match tokenizer; per-class enrichment map + tests; K10 collision tests.

Campaign arc held: R1 found the two seams (style-never-injected, body-never-
gated) with full panel+anchor convergence; R2 made the fixes buildable + caught
in-loop placement + missing LineRequest palette; R3 nailed sequencing (contract
after news, F2 re-select removal, use_exchange bypass, K5 add-not-collapse); R4
confirmed no residual structural defects. Final deliverable: pass04_plan.md.

INVARIANTS GUARDED throughout: determinism (gate = count+token reroll, seed-
stable); 100% local; byte-identical when the story-quality flag is off (all new
fields default empty, no meta.story_contract); audio byte-identity (text-path
only; golden re-baseline already on the books from the default-flip).
