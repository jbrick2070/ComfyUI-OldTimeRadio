# My Story treatment field guidance -- driver anchor

VERDICT: PASS for the scoped prompt clarification in d11c31c3;
loop causality remains unproven.

CONFIRMED: `_otr_my_story._pass_treatment` resolves the pack's
`my_story_treatment_system` and supplies StoryTreatment to the native binder.
Before d11c31c3, the logline example was declarative, but only dramatic_question
had explicit field instructions. The fields remain independent unconstrained
strings. The pack is loaded by `_otr_story_routing.resolve_story_pack`; the
canonical graph does not embed these instructions. No node, widget, link,
workflow selection or schema changes are needed.

CONFIRMED: `_full_artifact_repair` retains the original system message.
The existing retry ladder owns recovery; the prompt patch reaches its base,
lower-temperature retry and typed repair without changing the ladder.

Evidence available at review dispatch: the supplied report described
Qwen3.8-27B repeating a 69-token clause, halted at 288 tokens and recovering
at 0.500. No matching log was then available in the checkout. The original
review input and reviewer reports retain that historical limitation.

Post-review evidence update: the operator supplied
`C:/Users/jeffr/Documents/otr_verbatim_cycle_excerpt.log`; an exact copy is
archived beside this anchor. Lines 36-51 confirm the halt and retry. The
captured HEAD already repeats inside the unclosed logline; no captured
transition into dramatic_question supports the original cross-field story.
Lines 67-69 show the retry ending on EOS after 849 tokens; lines 122-126
show writer completion, 945 words and title The Extra Bowl. These are
observations of the original run, not a rerun using the changed prompt.

UNVERIFIABLE: a model having "nothing new to say" is not an established
causal mechanism, and one recovered run cannot quantify a reroll rate.
PBUG-20260910-01's pairlock_03/04 follow-ups already record repetition inside
logline and explicitly decline a title-only causal fix. This change instead
clarifies the field roles at the user's request, without claiming a cure.
PBUG-20260910-05 / Bible 12.100 retain the existing liveness contract;
no new production ID or portable rule is warranted.

Change: define logline as one declarative premise/conflict sentence; reserve
the direct yes/no question for dramatic_question; explicitly close completed
string values and advance. Existing example already matches these roles.
No new content validator, length ceiling, text rewriting, penalty, retry or
sampling change. This is shared My Story guidance on all machines.

Verification performed: load the shipped pack through resolve_story_pack and
capture the real _pass_treatment slot messages while raising one and two
GenerationDegeneracyError failures. All attempts carry the new field guidance;
observed temperatures are [0.85, 0.5] and [0.85, 0.5, 0.1], accepted objects
are unchanged, and interrupted fragments are not accepted. The 342-test scoped
suite includes runner, registry, pack, structured-call and decode-guard checks.

MUST-FIX: none in the scoped diff after delivery/regression checks.
SHOULD-FIX: none. A stochastic improvement claim requires live comparative
evidence and is deliberately absent from this receipt.

Review scope: finished-diff QA under CLAUDE.md's mechanical-change exception,
not a four-round design arc. REFUTE the claims above against the real Windows
files and pushed diff. Read only; do not edit production, tests, workflow,
plan or receipt. Report actionable defects with exact locations. Do not turn
an unverified generation hypothesis into a new runtime gate.
