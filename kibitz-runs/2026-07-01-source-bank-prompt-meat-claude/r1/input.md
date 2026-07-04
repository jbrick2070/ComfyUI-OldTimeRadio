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

## Architecture Preference For This Phase

Prefer cloneable story/content packs over one giant dynamic prompt string.

The phase-2 build should feel like editing separate story models:

```text
story_packs/
  media_archive/media_restoration_adventure.json
  media_archive/cinematic_humorous.json
  media_archive/happy_archive_mystery.json
  media_archive/gentle_thriller.json
  media_archive/broadcast_history_comedy.json
  public_domain/faithful_radio_adaptation.json
  public_domain/comic_panel_radio_adaptation.json
```

Those packs can carry full prompt text, examples, rubrics, forbidden patterns,
and coda rules.

First-phase assumption: the story architecture stays broadly similar to the
current many-pass ledger builder. The work is to replace sci-fi/science-news
prompt meat and source-specific Python routing wherever it is hardcoded.

Later lab option: a pack may choose a different pipeline shape:

- `legacy_many_pass`: adapts the current many-prompt ledger builder.
- `lean_5_prompt`: experiments with a smaller story-building stack.
- `simple_4_prompt_experimental`: exposed experiment:
  1. creative prompt creates the story,
  2. creative prompt uses that story to fill the ledger,
  3. technical prompt cleans/repairs the ledger until it meets the spec.
  4. final technical pass audits characters, cast, IDs, beats, continuity,
     source fields, visual directives, and downstream ledger invariants.
- `custom_pipeline`: allowed in the lab only until it proves it can fill the
  required ledger contract.

Python should select one pack, validate it, and compile it into the existing
ledger-writing contract. Avoid a fragile mega-template where every prompt is a
pile of tiny variables. For the first transplant, do not build the lean or
custom pipeline runtime; just leave the pack schema capable of growing there.

Expose planned source/story choices in the selector design. Do not hide public
domain merely because the adapter is unfinished. The product rule is: expose it
and get it working; if it does not pan out, rip it out. Do not bury it behind
silent fallbacks, hidden downgrade paths, or unpromoted code.

Still do not clone the entire downstream Comfy workflow per story model unless
explicitly chosen later for an experiment. Full workflow clones create
transplant drift: every audio/video/OBS fix would need to be copied into every
clone. The safer split is cloneable upstream story packs plus one shared
downstream workflow.

The key future flexibility requirement: a user should eventually be able to
create a new story model architecture with fewer prompts, different intermediate
passes, or a more efficient story-building flow, then test whether it produces
a valid ledger. The ledger is fixed; first transplant keeps the current broad
story pipeline while swapping the model-specific prompts and routing.

Fourth-option experiment to include in the lab and selector plan:

- `simple_4_prompt_experimental` / "Simple 4-Prompt Experimental"
- It should be exposed as experimental, not hidden.
- It must not fall back to the legacy builder if it fails.
- Expected risk: it may bump into ledger spec/detail gaps, which is fine; the
  point is to test a simplified story-building architecture against the same
  ledger contract.
- Design hope: stronger LLMs may be able to create richer stories in three
  creative/repair passes plus one final technical ledger audit than the current
  Python-heavy scaffold, with Python doing validation, schema repair, and loud
  failure instead of narrative trickery.
- Motivation: the current story generator has many Python workarounds because
  local LLMs were flaky and even frontier LLMs can drift when asked to fill a
  consistent ledger. The experiment is not anti-ledger; it tests whether modern
  models can satisfy the same ledger with fewer Python-authored story rails.

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
- If an exposed lane is not implemented yet, selecting it must raise a loud
  error naming the selected `source_bank` and `story_model`. It must not route
  to science/news, raw custom premise, default style picker, or any other hidden
  compatibility path.

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
