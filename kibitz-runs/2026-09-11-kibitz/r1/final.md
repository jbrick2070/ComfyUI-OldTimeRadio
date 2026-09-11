# r1 judgment -- visual continuity ownership

**Round shape (operator directive 2026-09-11): ONE second opinion per round.**
r1 = codex (`gpt-6-astra`, reasoning high, logged in, quota OK). Lane health checked
rather than trusted: `codex.log` is 688 KB and `codex.md` 6.1 KB, so it genuinely
read the tree -- exit 0 alone is not evidence (a lane can rubber-stamp).

Driver = Claude Opus 5 (Cowork, 5080). `driver_anchor.md` was written BEFORE the
fan-out. Verdict below is the driver's; every codex claim was checked against the
real Windows files first.

## Codex verdict: "no -- needs a complete consumer boundary and a story-setting
authority before it can become a buildable fix." **Accepted.**

## VERIFIED AND FOLDED IN (driver checked each against the files)

**1. My evidence framing was factually wrong about the setting.** The ledger's
`meta.source_meta.story_input.fields.setting` reads *"A comfortable **Los Angeles**
home dining room in the present day..."*. The Bay Area is a spoken memory and the
episode's title; the dinner is in LA. My diagnosis said "present-day Bay Area
dinner". Corrected in the anchor. The defect is unchanged -- present-day room,
period dress -- but the evidence must be stated accurately.

**2. The pack asserts era in SEVEN fields, not five, and it furnishes ROOMS too.**
Additional to the five I listed:
- `portrait_instruction_look`: *"...like a **period stage actor** photographed
  during rehearsal"*
- `plate_look`: *"photoreal practical stage set, candlelit **period interior**"*

`plate_look` breaks my clean split. My anchor claimed "the pack dresses the PEOPLE,
the story furnishes the ROOM". That is wrong: the pack furnishes the room as well.
The real boundary is era-assertion vs treatment, on BOTH subject and set.

**3. A FOURTH costume owner exists, and it is not a style pack at all.**
`nodes/_otr_story_brief_helpers.py:567`:

```python
if not subject:
    subject = "a period-dressed character, face clearly visible"
```

This is **unconditional and style-independent**. Any character whose
`portrait_prompt` / `appearance` / `character_description` are all empty is
described as *period-dressed* regardless of the style pack, the bank, or the story's
era. A present-day sci-fi episode with a missing appearance gets period dress from
shared code. This answers anchor question 1 in the affirmative and is the single
cleanest defect in the area -- it is a hardcoded era assertion in a deterministic
fallback, owned by nobody.

**4. There is no cross-bank story-time signal.** No `meta.period` key exists;
"present day" lives only in `meta.source_meta.story_input.fields.setting`, and
`nodes/_otr_story_source.py::raw_fields_from_ledger` returns None outside My Story.
`StoryBriefModel` has no era field. So fork (b) as I wrote it presumes an input that
does not exist on most lanes. This answers anchor question 3: the signal must be
identified per lane, including an unspecified-time case -- and era must NOT be
classified from bank names.

**5. An arbitration point already exists.**
`nodes/otr_meta_brief_image_prompt.py::_rewrite_char_scene_from_source` already
supplies source corrections and style instructions TOGETHER; `shot_001_b9` shows
`source_rewrite.status="rewritten"` and still carries "period costume". So the
defect is conflicting authority INSIDE an existing operation, not absent source
access. That is a materially better framing than my "no owner arbitrates".

**6. My two stills come from DIFFERENT composers.** `shot_000_b1` is
`announcer_visual`/`scene_beat`; `shot_001_b9` is `character_video`/`scene_character`.
I presented them as one class of evidence. They are two paths, and whether the house
frame shares the dramatic world's dress policy is an open question, not an assumption.

## DRIVER RULINGS

- **Fork (a), era-aware eligibility: CUT.** Codex is right that it contradicts the
  operator's full-pool position and cannot fix contradictions inside a style the user
  explicitly selected. `resolve_style_selection` stays independent.
- **Fork (c), mandatory pack-schema migration: CUT for now.** Existing string edits
  can express the rule. `nodes/_otr_visual_styles.py::validate_pack` permits changing
  existing string values, so no schema change and no hash-receipt risk. Codex
  correctly narrowed my hash claim: `get_visual_style` verifies embedded bytes only
  for `visual_storybased`; named styles resolve through the live registry.
- **Fork (b) survives, but reshaped**: not "story time overrides pack", which needs an
  input most lanes lack, but **era-assertion vs treatment as an editorial boundary
  applied to the checked-in pack strings and to every deterministic fallback.**
  Garments, dates and physical surroundings are story context; rendering medium,
  lighting quality and framing are treatment. "candlelit stage light" stays;
  "period costume" goes.
- **[ASSUMPTION TO RECORD, operator decision]** that `shakespeare_stage_realism`
  keeps its identity as *theatrical treatment* without mandatory Elizabethan
  clothing. If the operator wants that pack to always dress people in period, the
  answer is different and eligibility comes back. Flagged, not assumed.

## CARRIED TO r2

1. Enumerate EVERY costume/era writer across portrait, scene-character, scene-beat
   and plate paths, including deterministic fallbacks -- a seven-field pack edit is
   not sufficient if `_story_brief_helpers` and `_appearance_for_char` still write.
2. Name the accepted setting/source context available to each visual author PER LANE,
   including the unspecified-time case, and how adaptations stay period-correct
   WITHOUT classifying era from bank names.
3. Whether the house frame (announcer_visual) shares the dramatic dress policy.
4. Acceptance that is provable: "contradictory instructions removed" is testable;
   "identical clothing across beats" is a separate and much larger claim and must not
   be promised.

## Unchanged constraints

Nothing from this arc ships before tonight's four-machine wave. A style change
repaints every image on every machine and destroys the baseline that wave exists to
build. This arc produces a DESIGN.
