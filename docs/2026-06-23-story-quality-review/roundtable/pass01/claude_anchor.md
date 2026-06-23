# R1 anchor (Claude, code-grounded) -- creative/arc review of pass00_plan.md

VERDICT: yes-with-fixes. The plan's central creative bet -- move the lever upstream + deterministic, stop
re-instructing a weak model -- is correct and grounded (the soft levers it names ARE in the real code and ARE
ignored). Three creative-level fixes before it is arc-coherent.

MUST-FIX:
1. [L1] A denylist ALONE risks blandness -- if you ban "override/purge/lever" without giving the planner
   something premise-specific to reach for, a weak model will substitute the next-nearest thriller noun or go
   generic. The brief-derived PALETTE (L1b) is therefore NOT optional; denylist + palette must ship together,
   and the palette must be the primary instruction, the denylist the backstop. CONFIRMED: `allowed_things`
   already exists to seed it.
2. [L2] The "announcer fiat" ending is the biggest single CREATIVE defect (the drama never lands on-stage). The
   required on-stage climax beat + outro-references-the-choice is the highest creative payoff in the plan -- keep
   it in the core, do not defer.
3. [scope] Do not build all 6 levers as one arc. The creative core that changes the STORY is L5 (best writer) +
   L1 (premise palette/denylist) + L2 (climax + phase-function). L3/L4 are hygiene (real, but they fix
   artifacts, not story); L6 is polish. State this tiering so the build does not sprawl.

SHOULD-FIX:
- Define "better story" measurably BEFORE building: the key metric is CROSS-EPISODE sameness (does premise
  survive into the scene?), not a per-episode score. Propose a deterministic diversity check (e.g. distinct
  conflict-object n-grams across a soak) as the acceptance signal.
- Name the failure the plan must NOT reintroduce: longer-but-still-monotone (the 430w standoff).

CUT:
- L6 best-of-N from the creative core -- it cannot fix structural sameness (all N candidates share the beat),
  it only polishes lines, and it costs N generations. Re-evaluate after L1/L2.

ASSUMPTION [verify]: the news brief / `allowed_things` carries enough domain-specific objects to seed a real
palette for every premise. If a brief is thin, L1b degrades -- needs a fallback (generic-but-premise-anchored
object set derived from the logline).
