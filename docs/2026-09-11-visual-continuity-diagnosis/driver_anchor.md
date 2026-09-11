# Driver anchor -- visual continuity: who owns what a character is WEARING

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round, grounded
in the real Windows files at HEAD `0a1ae33e` on `v2.0-alpha`. The panel proposes; the
driver disposes and verifies every claim against these files.

**Review shape for this arc (operator directive 2026-09-11): ONE second opinion per
round.** r1 codex, r2 cursor (Fable substitutes if cursor does not respond -- operator
directive, and the standing substitute-never-block rule), r3 codex, r4 sonnet.

**UPDATED AFTER r1.** Codex corrected the driver on six points; all six were verified
against the real files and folded in below. Judgment:
`kibitz-runs/2026-09-11-kibitz/r1/final.md`.

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

## 5. Questions for r2 -- answer these; do not restate the above

1. **Enumerate EVERY costume/era writer** across the portrait, scene_character,
   scene_beat and plate paths, including deterministic fallbacks and
   `nodes/otr_shot_lock.py::_appearance_for_char`. A seven-field pack edit is not
   sufficient if other writers still assert era. Is the list in section 2 complete now,
   or is there a fifth writer?
2. **Per lane, name the accepted setting/source context each visual author actually
   receives** -- My Story, Original, Shakespeare, public_domain, media_archive,
   scifi_news -- including the unspecified-time case. How do adaptations stay
   period-correct WITHOUT classifying era from bank names (which is forbidden)?
3. **Does the house frame (`announcer_visual`) share the dramatic world's dress
   policy?** The two stills come from different composers; the driver conflated them.
4. **Is the editorial boundary (garments/dates/surroundings = story; medium/light/
   framing = treatment) actually applicable field by field?** Walk the seven pack fields
   and say which clause moves and which stays. Where a single string mixes both, give the
   exact rewrite.
5. **Acceptance that is provable.** "Contradictory instructions removed" is testable.
   "Identical clothing across beats" is a separate, much larger claim -- the 09 receipt
   explicitly leaves clothing continuity unqualified. Propose acceptance that does not
   promise general wardrobe continuity.

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
