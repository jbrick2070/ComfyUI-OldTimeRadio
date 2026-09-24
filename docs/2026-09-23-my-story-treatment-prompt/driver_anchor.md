# My Story treatment field guidance -- driver anchor

VERDICT: HOLD for a narrow prompt clarification; loop causality is unproven.

CONFIRMED: `_otr_my_story._pass_treatment` resolves the pack's
`my_story_treatment_system` and supplies StoryTreatment to the native binder.
The existing logline example is declarative, but only dramatic_question has
explicit field instructions. The fields remain independent unconstrained
strings. The pack is loaded by `_otr_story_routing.resolve_story_pack`; the
canonical graph does not embed these instructions. No node, widget, link,
workflow selection or schema changes are needed.

CONFIRMED: `_full_artifact_repair` retains the original system message.
The existing retry ladder owns recovery; the prompt patch reaches its base,
lower-temperature retry and typed repair without changing the ladder.

UNVERIFIABLE: the supplied report describes Qwen3.8-27B repeating a 69-token
clause, halted at 288 tokens and recovering at 0.500. No `server_pod.log` or
matching Third Bowl output was found in this checkout. Treat those figures
as user-supplied evidence, not independently reproduced measurements. A model
having "nothing new to say" is not an established causal mechanism. Existing
PBUG-20260910-05 / Bible 12.100 already record open-string repetition; do not
mint a new PBUG or portable rule from this prompt hypothesis.

Change: define logline as one declarative premise/conflict sentence; reserve
the direct yes/no question for dramatic_question; explicitly close completed
string values and advance. Existing example already matches these roles.
No new content validator, length ceiling, text rewriting, penalty, retry or
sampling change. This is shared My Story guidance on all machines.

MUST-FIX: none in the scoped diff after delivery/regression checks.
SHOULD-FIX: none. A stochastic improvement claim requires live comparative
evidence and is deliberately absent from this receipt.

Review scope: finished-diff QA under CLAUDE.md's mechanical-change exception,
not a four-round design arc. REFUTE the claims above against the real Windows
files and pushed diff. Read only; do not edit production, tests, workflow,
plan or receipt. Report actionable defects with exact locations. Do not turn
an unverified generation hypothesis into a new runtime gate.
