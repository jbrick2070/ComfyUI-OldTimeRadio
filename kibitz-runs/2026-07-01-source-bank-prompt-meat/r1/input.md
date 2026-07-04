# Focused Kibitz Input: Story + Visual Sci-Fi Remnants

Status: input for a focused R1-R4 kibitz pass.

Scope: ground the existing `STORY_AND_VISUAL_SCI_FI_REMNANTS_ARTIFACT.html`
against the real Python prompt sites and produce a concrete prompt/Python update
inventory.

Do not re-litigate the broad architecture. The architecture is locked:

```text
source_bank + story_model -> LedgerWritingSpec -> existing production ledger
visual_style -> VisualStylePolicy -> still/video prompt seams
```

One production ledger remains the contract. The upstream logic changes how the
same ledger slots are filled.

## Required Deliverable

Produce concrete suggestions, not theory:

- For each new source/story model, name the prompts that must change.
- For each new source/story model, name the Python modules/functions that must
  change.
- For each visual style, name the visual prompt tails/subjects/fallbacks that
  must change.
- Name forbidden legacy phrases that must not leak into each lane.
- Keep `workflows/otr_scifi_16gb_full.json` untouched until transplant.
- No silent fallbacks. Unknown source/style/model must fail loudly.

## Source / Story Models To Cover

Science baseline:

- `science_news_default`

Media archive source bank:

- `media_restoration_adventure`
- `cinematic_humorous`
- `happy_archive_mystery`
- `gentle_thriller`
- `broadcast_history_comedy`

Public-domain source bank:

- `faithful_radio_adaptation`
- `chapter_digest_drama`
- `comic_panel_radio_adaptation`
- `stage_play_radio_adaptation`
- `storybook_puppet_show`

Custom extension lane:

- `custom_source_bank`

## Visual Styles To Cover

- `sci_fi_radio`
- `archival_documentary`
- `cinematic_35mm`
- `noir`
- `anime`
- `cartoon`
- `paper_origami`

## Grounding Files

Primary artifact:

- `docs/2026-07-01-source-bank-visual-style-code-ready/STORY_AND_VISUAL_SCI_FI_REMNANTS_ARTIFACT.html`

Prompt audits:

- `docs/2026-07-01-source-bank-visual-style-code-ready/LEDGER_PROMPT_AUDIT.md`
- `docs/2026-07-01-source-bank-visual-style-code-ready/VISUAL_PROMPT_AUDIT.md`

Locked prior convergence:

- `kibitz-runs/2026-07-01-source-bank-visual-style-transplant/r4/final.md`

Relevant story prompt/Python sites:

- `nodes/news_interpreter.py`
- `nodes/_otr_outline.py`
- `nodes/_otr_pitch_room.py`
- `nodes/_otr_story_select.py`
- `nodes/_otr_dramatic_state_llm.py`
- `nodes/_otr_line_composer.py`
- `nodes/_otr_casting.py`
- `nodes/_otr_story_quality_l12.py`
- `nodes/_otr_story_spine.py`
- `nodes/_otr_style_picker.py`
- `nodes/OTR_LedgerScriptWriter.py`

Relevant visual prompt/Python sites:

- `nodes/_otr_story_brief_helpers.py`
- `nodes/otr_meta_brief_image_prompt.py`
- `nodes/otr_shot_lock.py`
- `nodes/_otr_video_engines/render_driver.py`

## Known Story Remnants To Remove Or Profile

These should survive only under the science/default profile:

- `science-fiction audio drama`
- `sci-fi radio drama`
- `science story`
- `real science`
- `grounded in real science`
- `real news item`
- `news event`
- `NEWS PREMISE`
- `NEWS KEY TERMS`
- `real news report`
- `news facts`
- `news_close_brief` as conceptual source-note label

Compatibility note: legacy JSON keys such as `meta.news.news_close_brief` may
remain as mirrors during transplant, but the prompt meaning must become
source-neutral/profile-driven.

## Known Visual Remnants To Remove Or Profile

These should survive only when explicitly selected by style policy:

- `cinematic`
- `35mm film look`
- `film grain`
- `dramatic film lighting`
- `broadcast-distressed`
- `vintage 1940s radio`
- `1940s radio station studio`
- `radio set warming up`
- `period-accurate set`
- `1940s costume`
- `radio actor speaking into a studio`
- `chrome ribbon microphone`
- `ON AIR sign`
- `vacuum tubes`
- `tuning dial`
- `radio console`

## Concrete Review Questions

1. What exact story prompt variables are needed for each source/story model?
2. Which current prompt sites should read those variables?
3. Which current hardcoded prompt strings should remain shared radio-drama craft?
4. Which code paths must be split so `media_archive` never calls science RSS?
5. Which public-domain prompts need source-fidelity/adaptation rules?
6. Which visual style fields are needed to replace radio/cinematic fallbacks?
7. Which render-driver fallback prompts are risky enough to defer to a separate
   V3 visual project?
8. What tests should fail if any model leaks the wrong source/style language?

## Output Shape Wanted

Final document should be a direct edit map:

```text
Model: media_restoration_adventure
Prompt changes:
- file/function: replace X with Y-shaped prompt variable
Python changes:
- module/function/class: add/route/validate ...
Forbidden leakage:
- ...
Tests:
- ...
```

Repeat for each source/story model and each visual style. Keep it practical and
code-ready.
