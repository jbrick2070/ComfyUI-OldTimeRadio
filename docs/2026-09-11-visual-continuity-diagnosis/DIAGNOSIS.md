# Sprint 2 -- visual continuity: the actual cause, from the pixels

Written 2026-09-11 against canonical09 (`the_bay_area_table_20260911_074902`,
prompt `64f1c56e-5ab6-48eb-9e15-e0cb94b728f7`). GO_FORWARD Sprint 2 requires that
"regression evidence discriminates the actual cause" and that actual pixels are
checked "before claiming a face, age, presence or setting defect." This is that
evidence. **No code changed. This is a diagnosis, not a fix.**

## What the operator saw

Clothing, hair arrangement, room details and seating vary across a story that is
one continuous present-day dinner.

## What is actually in the frames

Two stills, read directly (not inferred from prompts or logs):

**`pairlock_09_still_shot_000_b1_499a71d9792f.png`** -- a man in full
Elizabethan/Jacobean dress: green brocade doublet, lace-edged white falling-band
collar, patterned waistcoat. He is standing in an unmistakably **present-day
kitchen** -- modern cabinetry, tiled backsplash, stainless pans, a modern range --
beside a mid-century radio, with candlelit tables behind him.

**`pairlock_09_still_shot_001_b9_d02ac2532f0e.png`** -- the mother in a period
brocade gown with a ruffled chemise, **and opposite her the son in a modern dark
suit jacket and white shirt**, at a present-day dining table: modern picture frame,
plain painted wall, ordinary tumblers and flatware.

So the defect is **not** beat-to-beat drift in the way it was described. Within a
SINGLE frame, one character is dressed in period costume and another is dressed
contemporary, and the room is contemporary throughout.

## The cause, and it has an owner

Three verified facts compose it:

1. **The rolled style pack carries an era.**
   `nodes/visual_styles/shakespeare_stage_realism.json` puts era and costume
   language into five separate fields:
   - `era_tail`: `"photorealistic Elizabethan and Jacobean stage realism"`
   - `positive_tail`: `"... candlelit theater realism, period costume detail ..."`
   - `portrait_look`: `"... period costume, candlelit stage light ..."`
   - `portrait_look_talking`: `"... period costume, soft stage key light ..."`
   - `scene_instruction_look`: `"... candlelit practical set, period costume ..."`

2. **The style roll is deliberately independent of the source bank.**
   `nodes/_otr_rolls.py::resolve_style_selection` states it verbatim: *"Independent
   of the bank roll in every respect: its own sentinel, its own seed env, its own
   receipt. Rolling one surface never implies the other."* So a present-day My Story
   episode can legitimately draw an era-bearing pack. **GO_FORWARD is right that the
   roll was legitimate and that no other source bank leaked** -- that reading is
   confirmed, not overturned.

3. **Both languages reach the prompts, and nothing reconciles them.**
   `pairlock_09_ledger.json` contains **47** occurrences of `period costume`. The
   story's own scene facts describe a present-day Bay Area dinner. The two arrive at
   the image model together, and the model resolves the contradiction differently
   per subject and per beat -- period on one person, modern on another, modern room
   throughout.

**The root is an unreconciled ownership boundary: the style pack's era/costume
clause dresses the PEOPLE, the story's scene facts furnish the ROOM, and no owner
decides which wins when they disagree.**

## What this rules OUT, with evidence

- **Not a source-bank leak.** The period language traces to the style pack's own
  fields, not to another bank's content.
- **Not a face/identity defect.** Faces are consistent; costume is not. Per the
  standing instruction, no face defect is inferred from clothing.
- **Not a presence/off-camera error.** Both characters are present and seated;
  off-camera does not mean absent.
- **Not (only) per-beat model variance.** Variance exists, but it does not explain
  two different eras of dress inside one frame with a period clause in the prompt.

## The design fork -- needs R1-R4 before any code

More than one defensible answer, so per CLAUDE.md this gets the full arc:

- **(a) Eligibility.** A pack that declares an era becomes ineligible when the story
  is set in the present. Cheapest to reason about; reduces the style pool for
  present-day banks, which the operator may not want -- every visual style is
  supposed to be able to craft an episode.
- **(b) Precedence.** The story's time/setting overrides the pack's costume clause,
  while the pack keeps light, texture, framing and treatment. Preserves the full
  style pool; needs a clear rule for which pack fields are "treatment" and which are
  "period assertion".
- **(c) Reword the packs.** Separate era assertion from visual treatment in the pack
  schema itself. Most honest, largest blast radius -- touches every pack.

**Do not code this today.** A style change alters every image on every machine in
tonight's four-machine wave, which destroys the baseline that wave exists to
establish. Tonight's legs each report their rolled `visual_style`, so the wave is
itself a free A/B: legs that roll an era-bearing pack against legs that do not.
Run the arc on the fork, and code it against that evidence.

## Explicitly not in scope

Per the operator's standing rules: no forbidden-word list, no new report-only
checker, no separate chunker, no added retry loop, no late subjective publication
veto, and no prose-quality work. Story quality is done; this is a correctness
defect in visual continuity, which is a different thing.
