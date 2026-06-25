# OTR STORY-ENGINE ROADMAP (2026-06-24)

Operator-facing roadmap. The dense, file:line build spec is
`roundtable/pass04_plan.md`; the source of truth for the active step is
`docs/GO_FORWARD_PLAN.md`. This is the "what's next, in order" view.

The thesis: the story engine had a STRUCTURAL sameness bug (forced climax /
console standoff), not a vocabulary one. The scaffold fixes structure +
grounding; raw line-craft is a separate, later question (the local-model
ceiling).

---

## DONE -- shipped, live, ON by default (in the canonical JSON)
- **Style grammar (chunks 1-6).** Per-episode style pick + ending taxonomy ->
  the climax SHAPE varies (no more forced "irreversible_choice"); announcer
  close gate (note: that C5 gate is now moot -- superseded by KILL 5).
- **KILL 1 -- body-output gate.** Validates the SHIPPED dialogue line (not just
  beat.intent); rerolls machinery (console/lever/fuel-cell). VERIFIED live:
  0 ungrounded crisis nouns in shipped bodies; telemetry fires (body_gate_*).
- **story_scaffold UI toggle** (auto/on/off) wired into the writer node in
  `otr_scifi_16gb_full.json`; default `auto` = ON. One control for the whole
  bundle. Default flipped ON.

STATUS after the bake-off + a 3-script read: stories are GROUNDED + machinery-
free + varied endings -- a real win over the console-standoff baseline. BUT the
3 ON episodes were all the same UCLA-medical register with generic conflict
objects, because the STYLE never reaches the body and the close still narrates
the news outcome. That maps exactly to the unbuilt items below.

---

## THE BUILD QUEUE (in order of leverage)

### 1. KILL 2 -- StoryContract: make the style actually shape the story  [NEXT]
WHY: today the style is selected + logged + drives only the ending tag; its
sound_world / story_engine never reach the prompts (`render_style_grammar` has
zero callers). Result: a "psychiatric ward" style produced a transplant story; a
"satellite recovery" style produced an air-quality story. The style is cosmetic.
WHAT: build one `StoryContract` (slug/sound_world/story_engine/ending_tag),
selected BEFORE the outline, injected into the macro/phase/beat prompts AND every
body line -- not just the climax. Make conflict objects premise-specific instead
of the generic domain pool.
ACCEPTANCE: read N episodes -- the chosen style's register actually shows in the
content; two different styles on the same news produce visibly different stories.
EFFORT: medium (touches the writer, outline, line composer; ADD `meta.story_
contract`, do NOT collapse the old style fields yet).

### 2. KILL 4 + KILL 5 -- un-starve the body beats + fix the close  [AFTER 1]
WHY: setup/pressure/consequence + non-irreversible climax beats get NO
deterministic dramatic content today (only personal_stake/irreversible do). And
the announcer close still reads out the news outcome ("published in The Lancet")
because it isn't governed by the ending tag.
WHAT: role-keyed enrichment for every dramatic role; route the announcer close by
`ending_tag` (non-resolving for unresolved/revelation/quiet -- prompt AND
fallback).
ACCEPTANCE: closes match the ending class (no news-outcome narration on
non-resolving endings); body beats carry real dramatic framing.
EFFORT: small-medium.

### 3. KILL 3 -- climax POSITION (let the ending choose where it lands)  [DEFERRED]
WHY: every episode is forced to peak on the LAST beat -- a structural mono-shape.
Some endings (revelation, reversal) want to land earlier with a denouement after.
WHAT: let the ending taxonomy choose the climax position; relax the validator.
ACCEPTANCE: episodes vary in arc shape, not just ending label.
EFFORT: larger (breaks the climax-last validator + the ending-template target +
the outro's "last line = resolution" assumption -- its own build, after 1 & 2).

---

## THE STRATEGIC FORK (after the structural levers are spent)

### 4. Line-craft ceiling -> the frontier-writer decision  [NEEDS A DECISION]
Both ON and OFF still grade `arc=uneven`. The scaffold fixes sameness +
grounding, NOT raw prose quality -- that's the mistral-nemo ceiling. Once KILL
2/4/5 are in and re-soaked, the question is genuinely new and worth its own
discovery round: accept-B local, OR pay for a stronger writer (a frontier API
for the prose only, or a better local model). DO NOT run this discovery round
before the structural builds -- it'd be measuring the wrong thing.

### 5. Model-landscape scan  [CHEAP, ANYTIME]
Independent of the build queue: check whether better LOCAL writers exist for the
16GB / Blackwell box (newer qwen / gemma / mistral). Periodic, low-cost, could
move the line-craft ceiling without paying for frontier.

---

## PARKED / low priority
- DOMAIN_PALETTE scored matching + `_PERSONAL_COST` domain rows (K8/K10, small).
- Style-driven render profiles in `eng_ltx_av` (K11, visual not story).
- Model-capability gate (prefer mistral / branch weak models -- the KILL-1 body
  gate is the model-agnostic net for now).
- Audio-byte-identity baseline GPU re-capture (housekeeping from the default-flip).

---

## STANDING DISCIPLINE (every build)
- Behind the `story_scaffold` flag -> byte-identical when off.
- Full suite + Bug Bible per chunk (the 5 pre-existing 267a53e workflow-pin fails
  are not yours); commit AND push per green chunk.
- LIVE re-soak (gemma + mistral) after each STRUCTURAL change.
- The 3-test "is it baked in" check on every new feature: (1) it's CALLED, not
  just defined; (2) a real shipped output shows its telemetry/counter fired;
  (3) the delete-it test -- turn only it off, behavior reverts.
- 100% local; determinism; LOUD fallbacks; UTF-8 no BOM; SFW; prod/main GATED.
