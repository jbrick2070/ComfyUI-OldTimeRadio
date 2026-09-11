# Visual-continuity arc -- CLOSED with no code change, 2026-09-11

Three rounds ran, one reviewer each, as the operator specified: r1 codex, r2 cursor,
r3 codex. **All three returned "no."** r4 was not run, because the operator's
constraints closed the question before a fourth opinion could change it.

**Outcome: the defect is diagnosed and recorded; no code was changed, and none
should be.** This is a real result, not an abandoned task. The arc's job was to
pressure-test a design, and the design did not survive -- on engineering grounds from
three independent reviewers, and on product grounds from the operator.

## READ THIS BEFORE REOPENING -- operator framing, 2026-09-11

Operator, closing the arc: *"I realize my visual pack and story source combined is
quite complex. I'm not expecting anything exact -- this is a fun experimental app."*

**So this was never a correctness requirement, and a future session should not treat
it as one.** The pack-plus-story combination is a deliberately open-ended, generative
system; a Shakespearean staging of an LA dinner is emergent behaviour from combining
two independent surfaces, not a broken contract. Exactness is not the goal and was
never promised.

The word "defect" appears below because that is how the observation entered the
record. Read it as *an observation worth understanding*, not a bug awaiting a fix.
Everything below is kept so the mechanism is known if it is ever wanted -- not as an
open work item. **Do not reopen this as a bug hunt.**

## What was observed (this part is solid and worth keeping)

Canonical09 is a present-day **Los Angeles** dinner (the Bay Area is a spoken memory
and the title) that rolled the `shakespeare_stage_realism` visual style. In the
delivered stills the mother wears a period brocade gown while her son, at the same
table in the same shot, wears a modern suit; elsewhere a man in a full Elizabethan
doublet stands in a modern kitchen with a tiled backsplash and stainless pans.

The mechanism, found by r2 and verified: **`positive_tail` is a SHARED field.** It is
appended on scene_beat, scene_open and scene_character
(`_otr_story_brief_helpers.py::compose_still_prompt`), on portraits via
`finish_visual_prompt`, on plates via `_compose_background_plate_prompt`, and on the
ShotLock video path. One clause therefore reaches the house frame, the drama AND the
plate -- which is why a **faceless radio subject** rendered as a costumed man in a
kitchen. That single fact explains the still the driver's own diagnosis could not.

Scope, measured across all nine packs -- era/costume words:
`video_art` 0, `recur_frac` 0, `anime` 0, `cartoon` 0, `paper_origami` 0,
`storybook_engraving` 1 (and it is *"hand-tinted costume color"*, an engraving
technique, not an era claim), `sci_fi_radio` 7, `archival_documentary` 11,
`shakespeare_stage_realism` 17. **The proven packs are already clean.**

## Why no code change -- four operator constraints, each independently sufficient

1. **The pack and the story are a COMBINATION.** Neither wins. This invalidated every
   precedence design the arc had produced -- era-aware eligibility, story-overrides-
   pack, and the schema split were all attempts to pick a winner.
2. **Every still and video model treats prompts differently.** Chasing delivered
   pixels back to a style-plus-story intent is a rabbit hole. Sprint 2's own
   acceptance already conceded the limit: *"An applied prompt correction proves
   application, not correct pixels."* We can prove a word left a prompt; we cannot
   prove what a given model does with what remains.
3. **The prompts were crafted per model.** Rewriting tuned strings is the
   swashbuckling this project has been burned by before. All three reviewers
   converged on exactly that remedy, which is what disqualifies it.
4. **Prompts are CHARACTER-BUDGETED and the subtleties problem is settled.**
   Verified in code: `motion_registers` budgeted at 240 chars enforced at load
   (BUG-LOCAL-112), `_fit_motion_slot` truncating to 60 for the camera template, and
   a documented over-budget branch. Any conditional clause -- "garments follow the
   story, lighting follows the pack" -- spends budget that does not exist and
   reintroduces the per-prompt subtlety the operator has already rejected.

Constraint 4 is the one that closes it for good. Both directions are blocked: adding
nuance costs characters we do not have, and removing era words edits a tuned prompt.

## Findings kept, none of them a prompt change

- **`positive_tail` is shared across house frame, drama and plate.** The single most
  useful fact from the arc. Anyone touching visual prompts later needs it.
- **`nodes/_otr_story_brief_helpers.py:567`** assigns *"a period-dressed character,
  face clearly visible"* when appearance is empty -- unconditional and
  style-independent, so a present-day sci-fi episode with a missing appearance gets
  period dress from shared code. NOT fixed here (it is still prompt text), but it is
  the one candidate that is a placeholder rather than crafted craft, if the operator
  ever wants it revisited.
- **A second empty-appearance writer**: `compose_image_prompt_fallback` degrades to
  `_style_anchor_for_aspect` (the pack's `portrait_look`).
- **`_appearance_for_char` is a READER, not a writer.** The driver was wrong to list
  it; recorded so the next session does not repeat the error.
- **Structural, prompt-free (r3):** ShotLock (node 90) builds creative directives and
  execution plans BEFORE image prompts exist (link 255 -> node 89, link 256 ->
  dispatcher 91), so anything image-node-owned arrives too late for video planning.
  This would have bitten any fix the arc built, and it is worth knowing independently.
- **There is NO RULE about packs and eras.** `validate_pack` enforces dict-ness,
  known field names and string types -- nothing about content. An `era_tail` field
  exists but nothing governs it. Today's behaviour is not a chosen policy.

## Driver errors this arc corrected, recorded honestly

The driver's original diagnosis was directionally right and factually wrong in five
places: the setting is Los Angeles not the Bay Area; the pack has ~12 era touchpoints
not five; `plate_look` means the pack furnishes ROOMS, so "pack dresses people, story
furnishes rooms" was wrong; the two stills come from different composers and were
presented as one class of evidence; and `_appearance_for_char` is not a writer. The
driver also argued for a pack schema and then withdrew it when the operator pointed
out the proven packs are already clean -- churning working packs to solve a problem
they do not have is the wrong trade.

## What actually decides this next

Tonight's four-machine wave. Every lane reports its rolled `visual_style`, so several
era-bearing and era-free episodes land side by side. That is real evidence from real
models, which is the only thing that can settle whether the combination reads badly in
general or whether canonical09 was simply an unlucky roll. Judge it as radio drama,
with eyes on the delivered episodes -- not by reading prompts.

**Reviewers who actually ran:** codex (`gpt-6-astra`, r1 and r3) and cursor
(`cursor-grok-4.6-high`, r2). Cursor responded, so the Fable substitution the operator
authorised was not needed and is not claimed. No r4. No unanimous-clean claim -- all
three said no.
