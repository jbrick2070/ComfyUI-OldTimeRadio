# R4 anchor review (Claude, code-grounded) — convergence / residual defects

Focus: any remaining MUST-FIX before this is build-ready?

## VERDICT

Build-ready. No structural must-fix remains; three verify-at-build items.

## Residual checks (verify-at-build, not blockers)

1. **Line-composer request shape.** The single behavioral injection is at the
   final-character-beat line request (`_otr_line_composer`). Confirm the exact
   field/signature when building (the composer isn't in the grounding); append
   the ending_template to the system/user prompt only for that beat, empty =>
   byte-identical. This is the one place to get right.

2. **Announcer-flag reach.** `_assemble_outline` must see the flag to swap the
   close intent. Thread it consistently with the writer gate (request field or a
   single env read) — keep OFF byte-identical.

3. **The core bet.** Whether a weak local model honors a concrete pre-seeded
   ending is the unproven assumption (R1). Settle empirically with the §J A/B
   (baseline vs lever-on, ~6 episodes); if the model still drifts, the ending
   template must become more prescriptive (name the final line shape), or escalate
   to a frontier writer for the final beat only.

## Convergence note

R1 set the design (closed enum, keep the role + reframe, deterministic selector,
cut the negative ban). R2 set the coding contract (catalog data shape, OutlineRequest
vs writer, final-beat coordinate, announcer-intent gate). R3 fixed the sequencing
(resolve in the writer post-outline, inject at the line composer). R4 finds no new
material must-fix -> STOP at convergence. The whole lever is DARK / default-OFF /
deterministic; the only risk is the empirical bet, which is a measurement, not a
design hole.
