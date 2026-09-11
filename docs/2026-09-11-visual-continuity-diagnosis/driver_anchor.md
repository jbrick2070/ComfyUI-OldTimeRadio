# Driver anchor -- visual continuity: who owns what a character is WEARING

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round, grounded
in the real Windows files at HEAD `0a1ae33e` on `v2.0-alpha`. The panel proposes; the
driver disposes and verifies every claim against these files.

**Review shape for this arc (operator directive 2026-09-11): ONE second opinion per
round.** r1 codex, r2 cursor (Fable substitutes if cursor does not respond -- operator
directive, and the standing substitute-never-block rule), r3 codex, r4 sonnet.

**UPDATED AFTER r1 AND r2.** r1 (codex) corrected the driver on six points; r2
(cursor -- it responded; no substitution needed) corrected two more and found the
mechanism behind the announcer still. All verified. Judgments:
`kibitz-runs/2026-09-11-kibitz/r1/final.md`, `.../r2/final.md`.

## OPERATOR RULINGS, 2026-09-11 -- these govern, and they kill three earlier framings

1. **`shakespeare_stage_realism` is JUST the visual style.** It is not the Shakespeare
   source bank. Source banks and visual styles are INDEPENDENT.
2. **There is NO DEFAULT pack.** Each style is its own JSON; there is only what the
   JSON says. The driver's "the default pack asserts costume" framing was wrong.
   (`sci_fi_radio` is one JSON among many, not a product-wide default policy.)
3. **THERE IS NO RULE about whether a pack may declare an era.** Verified in code:
   `validate_pack` enforces only dict-ness, known field names, and string types --
   NOTHING about content. An `era_tail` field exists in the schema, but nothing
   defines what belongs in it or what happens when it disagrees with the story. So
   today's behaviour is not a chosen policy; it is whatever prose sits in three of
   the nine JSONs. The driver's earlier "given packs may declare an era" was reading
   intent into an accident.
4. **THE VISUAL PACK AND THE STORY ARE A COMBINATION.** Neither "wins". This
   invalidates the driver's entire precedence framing -- (a) eligibility, (b) story
   overrides pack, (c) schema split were all attempts to pick a winner, and picking a
   winner is the wrong shape. The pack brings the staging; the story brings the people
   and the events; the frame is the blend of both.

**THEREFORE THE DEFECT IS RESTATED.** It is not "period costume appeared" and not
"the wrong owner won". It is that **the combination is applied INCOHERENTLY** -- the
same episode blends pack and story in different proportions per shot kind and per
character. The mother is in a period gown and her son, at the same table in the same
shot, is in a modern suit. A man is in a full doublet in a modern kitchen. A
committed Shakespearean staging of an LA dinner would be coherent; a coin-flip per
subject is not.

**AND THE PROVEN PACKS ARE ALREADY FINE.** Measured across all nine:
video_art 0, recur_frac 0, anime 0, cartoon 0, paper_origami 0 era/costume words;
storybook_engraving 1 (and it is *"hand-tinted costume color"*, an engraving
technique, not an era claim); sci_fi_radio 7; archival_documentary 11;
shakespeare_stage_realism 17. So NO schema migration is warranted -- six of nine
packs already satisfy any rule we would write, and churning proven packs to solve a
problem they do not have is the wrong trade.

---

## 1. The defect, from pixels not prompts

Canonical09, `the_bay_area_table_20260911_074902`. **The dinner is in LOS ANGELES,
not the Bay Area** (r1 correction, verified): the ledger's
`meta.source_meta.story_input.fields.setting` reads *"A comfortable Los Angeles home
dining room in the present day..."*. The Bay Area is a spoken memory and the episode
title. One continuous present-day dinner. Two stills read directly -- and note they
come from DIFFERENT composers, which the driver originally conflated:

- `pairlock_09_still_shot_000_b1_499a71d9792f.png` (**announcer_visual / scene_beat**) -- a man in full
  Elizabethan/Jacobean dress (green brocade doublet, lace falling-band collar)
  standing in an unmistakably modern kitchen: modern cabinetry, tiled backsplash,
  stainless pans, modern range.
- `pairlock_09_still_shot_001_b9_d02ac2532f0e.png` (**character_video / scene_character**) -- the mother in a period brocade
  gown with ruffled chemise, **and the son opposite her in a modern dark suit and
  white shirt**, at a present-day dining table.

So this is NOT beat-to-beat drift. Inside a SINGLE frame, two characters are dressed
in two different centuries, and the ROOM is consistently modern.

## 2. The three grounded facts that compose it

1. **The rolled pack asserts an era in SEVEN fields -- and furnishes ROOMS, not just
   people.** `nodes/visual_styles/shakespeare_stage_realism.json`:
   - `era_tail`: `"photorealistic Elizabethan and Jacobean stage realism"`
   - `positive_tail`: `"... candlelit theater realism, period costume detail ..."`
   - `portrait_look`: `"... period costume, candlelit stage light ..."`
   - `portrait_look_talking`: `"... period costume, soft stage key light ..."`
   - `scene_instruction_look`: `"... candlelit practical set, period costume ..."`
   - `portrait_instruction_look`: `"... like a period stage actor photographed during
     rehearsal"` (r1 addition)
   - `plate_look`: `"photoreal practical stage set, candlelit period interior"`
     (r1 addition -- **this is the pack describing the SET**)

2. **The style roll is independent of the bank roll BY DESIGN.**
   `nodes/_otr_rolls.py::resolve_style_selection` docstring, verbatim: *"Independent
   of the bank roll in every respect: its own sentinel, its own seed env, its own
   receipt. Rolling one surface never implies the other."* A present-day My Story
   episode drawing `shakespeare_stage_realism` is therefore LEGITIMATE, not a bug,
   and not another bank leaking. GO_FORWARD already says so; this confirms it.

3. **Both languages reach the image prompts together.**
   `pairlock_09_ledger.json` contains **47** occurrences of `period costume`, against
   a story whose own scene facts are a present-day dinner. Nothing decides which
   wins, so the image model resolves the contradiction per subject and per beat.

4. **A FOURTH costume owner exists, and it is not a style pack (r1 finding, verified).**
   `nodes/_otr_story_brief_helpers.py:567`:
   ```python
   if not subject:
       subject = "a period-dressed character, face clearly visible"
   ```
   **Unconditional and style-independent.** Any character whose `portrait_prompt` /
   `appearance` / `character_description` are all empty is described as *period-dressed*
   regardless of pack, bank or era. A present-day sci-fi episode with a missing
   appearance gets period dress out of shared code.

5. **There is no cross-bank story-time signal (r1 finding, verified).** No `meta.period`
   key exists; "present day" lives only in
   `meta.source_meta.story_input.fields.setting`, and
   `nodes/_otr_story_source.py::raw_fields_from_ledger` returns None outside My Story.
   `StoryBriefModel` has no era field.

6. **An arbitration point already exists (r1 finding).**
   `nodes/otr_meta_brief_image_prompt.py::_rewrite_char_scene_from_source` already supplies
   source corrections and style instructions TOGETHER, and `shot_001_b9` carries
   `source_rewrite.status="rewritten"` while still saying "period costume".

**Root, CORRECTED after r1:** not "no owner arbitrates" and not "pack dresses people,
story furnishes rooms" -- the driver's original split was wrong, because `plate_look`
furnishes the set and `_story_brief_helpers` dresses characters with no pack involved.
The real root is **conflicting era authority across four writers with no precedence
rule**, including inside an existing arbitration operation that already sees both.

## 3. Ruled out, with evidence

- Not a source-bank leak (period words trace to the pack's own fields).
- Not a face/identity defect (faces are consistent across beats; costume is not).
  Per standing instruction, no face defect is inferred from clothing.
- Not a presence/off-camera error (both characters present and seated).
- Not per-beat model variance ALONE -- variance is real and separately priced, but it
  does not explain two eras of dress in one frame with a period clause in the prompt.

## 4. The fork, RESHAPED after r1

**(a) Era-aware eligibility -- CUT.** Contradicts the operator's full-pool position and
cannot fix a contradiction inside a style the user explicitly selected.
`resolve_style_selection` stays independent.

**(c) Mandatory pack-schema migration -- CUT for now.** Editing existing strings
expresses the rule; `nodes/_otr_visual_styles.py::validate_pack` permits changing
existing string values, so no schema change and no hash-receipt risk. (r1 also narrowed
the hash claim correctly: `get_visual_style` verifies embedded bytes only for
`visual_storybased`; named styles resolve through the live registry.)

**(b) SURVIVES, but reshaped.** NOT "story time overrides pack" -- that presumes a
story-time input most lanes do not have (fact 5). Instead: **era-assertion vs treatment
as an editorial boundary, applied to the checked-in pack strings AND to every
deterministic fallback.** Garments, historical dates and physical surroundings are story
context; rendering medium, lighting quality and framing are treatment. Keep "candlelit
stage light"; drop the pack's unconditional "period costume"; fix the hardcoded
"period-dressed character" fallback.

**[ASSUMPTION -- OPERATOR DECISION, do not assume it away]** that
`shakespeare_stage_realism` keeps its identity as *theatrical treatment* without
mandatory Elizabethan clothing. If the operator wants that pack to always dress people
in period, the answer changes and eligibility returns.

## 5. What r2 established (do not re-derive), and r3's questions

**r2 findings, driver-verified:**
- The shakespeare pack has ~12 era touchpoints, not 7 (add `image_grade_tail`
  "crisp costume detail", `announcer_subject_object` "period tabletop radio",
  `announcer_subject_ltx_mouth`, `open_subjects.announcer` "period dials",
  `open_subjects.default` "period stage radio").
- **The mechanism behind the announcer still: `positive_tail` is SHARED.** It is
  appended on scene_beat, scene_open, scene_character
  (`_otr_story_brief_helpers.py::compose_still_prompt`), on portraits via
  `finish_visual_prompt`, on plates via `_compose_background_plate_prompt`, and on
  ShotLock video. One clause reaches house frame, drama AND plate -- which is why a
  faceless RADIO subject rendered a costumed man in a kitchen.
- A SECOND empty-appearance writer: `compose_image_prompt_fallback` degrades to
  `_style_anchor_for_aspect` (= pack `portrait_look`), so fixing
  `_otr_story_brief_helpers.py:567` alone is insufficient.
- `_appearance_for_char` is a READER, not a writer; the driver was wrong to list it.
- On non-My-Story lanes nothing writes garments at all: stills read only
  `story_brief_terms.setting[:2]`; `StoryBriefModel` has no era field;
  `raw_fields_from_ledger` returns None outside My Story; casting's
  `character_description` asks for age/role/Face/Presence/Voice and NOT clothing.

**QUESTIONS FOR r3 -- the combination model is the operator's ruling; work inside it.**

1. **WHERE DOES THE COMBINATION BECOME INCOHERENT?** Map, in execution order, every
   composer path that produces an image prompt -- portrait, scene_character,
   scene_beat, scene_open, plate, and the ShotLock video path -- and for EACH state
   exactly which pack fields and which story fields it blends, and in what order.
   The deliverable is a table showing WHY two subjects in one episode can receive
   different proportions of pack-era and story-fact. That table is the fix's spec.
2. **WHAT IS THE MINIMAL CHANGE that makes the blend uniform across those paths?**
   Not "strip era", not "pick a winner" -- both are ruled out. Something closer to:
   the same era/treatment contribution reaches every subject and every surface in an
   episode. Name the files and functions. If uniformity is impossible without a new
   field or owner, say so plainly and say why.
3. **Does uniformity alone fix the LA-kitchen frame?** If every subject and surface
   got the pack's era consistently, that still is a candlelit stage set with two
   period-dressed people -- a committed staging. Is that the correct output under the
   combination ruling? Argue it from the code and the evidence, not from taste.
4. **Is there any remaining era writer nobody has named?** r1 and r2 each found one
   the driver missed. Assume a third exists and go find it.
5. **Sequencing.** Confirm whether ANY part of this is render-inert and could ship
   before tonight's four-machine wave, or state plainly that all of it must wait.

## 6. Hard constraints -- a proposal violating any of these is rejected on sight

- **No forbidden-word lists**, no new report-only checker, no separate chunker, no
  added retry loop, no late subjective publication veto.
- **No prose/story-quality work.** Story quality is DONE by operator directive. This
  is a visual-correctness defect, which is a different thing.
- **No new content gates**, no word/duration/cast-size rejection.
- **Do not strip the author's own content on adaptation lanes.** The packs were once
  forbidding "blood, guns, knives" while adapting Macbeth; that was a fidelity
  defect, not a safety win. Do not recreate that shape for costume.
- **One canonical graph.** No second workflow JSON, no new output-path owner.
- **Nothing ships before tonight's four-machine wave.** A style change alters every
  image on every machine and would destroy the baseline that wave exists to build.
  This arc produces a DESIGN, not a diff.
