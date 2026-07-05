# Today's Coding Plan -- Handoff to Main Coder

Paste this into the main coder window. It owns sequencing.

---

You are the main coder window for today. Build ONE ordered plan covering the three work
items below and sequence them where you see fit. Before you start: state the order, give a
one-line rationale per item, and call out the dependencies. Then execute per CLAUDE.md
(workflow JSON is source of truth; one coder window; regression suite + Bug Bible after every
change; commit AND push per green chunk; fix bugs at source, never a Python band-aid; model
routing per section 9).

## The three items
1. **Big LLM prompt update across the whole workflow.** Sweeping prompt changes touching many
   nodes. Fix at source in the prompt/node/workflow; every node change lands in the workflow
   JSON in the same commit; re-validate after JSON edits.
2. **SFX ledger rip-out.** Structural removal. Same discipline as the VRAM-tier rip: do NOT
   just delete -- trace what the ledger gates first, replace gated values with explicit
   defaults (not nothing, or you get silent holes), keep a clean seam. Regression + Bug Bible
   after.
3. **Story-engine map + assertion inventory.** Read-only. Run per `docs/story-engine-map-brief.md`
   (move it in from the staging repo if it is not already in this repo's docs). Flow:
   general-purpose maps + inventories -> ONE Fable judgment pass -> Sonnet fan-out QA. Produces
   a plan, changes no code.

## Sequencing you MUST weigh (this is why order is not arbitrary)
- **Items 1 and 3 touch the SAME story engine.** If the prompt update (1) lands first, the map
  (3) documents the NEW state -- no rework. If the map runs first, it goes stale the moment (1)
  rewrites prompts. Decide deliberately: either map AFTER the prompt update, or scope the map to
  the structure (nodes / flow / assertions) that the prompt update will not move. Do not run a
  full map before a full prompt rewrite and pretend it is still accurate.
- **Item 2 is a structural rip** -- independent of the story engine, but same trace-before-delete
  rule as the VRAM tier. Sequence it wherever it is cleanest; it does not block 1 or 3.
- **Item 3 is read-only** and can interleave with 1 and 2 since it changes nothing -- BUT its
  Sonnet QA gate must be green before any coding that builds ON the map. The map itself is prep
  for the later model-combine, not today's coding, unless you decide otherwise.
- If you are genuinely torn between two orderings, run the roundtable LIVE for convergence
  (section 8) -- you are the judge -- rather than guessing.

## Deliverable
Today's ordered plan: the sequence, a one-line rationale per item, dependencies called out.
Then start executing top of the list.

---
