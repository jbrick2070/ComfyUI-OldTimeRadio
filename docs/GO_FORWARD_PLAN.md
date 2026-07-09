# OTR Go-Forward Plan

**Updated:** 2026-07-09
**Branch:** `v2.0-alpha`
**Status:** new release order set: model-slot audit -> `original_radio` ->
source-bank end-to-end sweep -> portability.

This file is for short-term coordination only. Longer runway lives in
`ROADMAP.md`; old sprint logs belong in `docs/GO_FORWARD_ARCHIVE.md`,
`docs/HANDOFF_LOG.md`, or dated docs.

## Current Status

Recent green code chunks on `v2.0-alpha`:

- Media archive seed deck is green and pushed.
- Visual-style bleed guard is green and pushed.
- `recur_frac` is now the concise `recursive fractal light field` pack; LTX
  audio-in talking prompts keep a face/mouth/lip-sync cue without cartoon
  wording. Commit: `ac919d99`.

Current runnable source banks in `nodes/story_packs/banks.json`:

- `science_news`
- `media_archive`
- `public_domain_story`
- `shakespeare`

Known non-runnable bank:

- `custom_source_bank` stays listed but intentionally fails loud.

Unrelated local file present before this plan update and left untouched:

- `docs/2026-07-08-source-banks-v2-plan.md`

## Next Action

### 1. Model-Slot Audit And End-To-End Smokes

Inventory every local model/engine exposed through the production slots:

- music
- audio/TTS
- still image
- video

For each candidate, document and test:

- required inputs and produced outputs
- supported slot/family/role
- required model files and expected VRAM class
- canonical workflow compatibility
- whether it can complete a tiny end-to-end smoke from `workflows/otr_canonical.json`

Decision rule:

- If a model fits the slot contract and finishes an end-to-end smoke, keep it
  in the tested path.
- If it cannot fit, cannot run, silently downgrades, OOMs outside its claimed
  tier, or produces the wrong artifact shape, remove it from the tested path or
  mark it non-invocable. No silent fallback.

Deliverables:

- A compact tested/retired model matrix.
- Focused tests that prove unsupported engines fail loud.
- At least one tiny canonical smoke covering the kept local paths.

### 2. `original_radio` Source Bank

Design and implement a no-source original-drama lane where the LLM creates a
random original old-time-radio premise, cast, outline, and filled ledger.

Hard requirements:

- No news/source attribution.
- No franchise or modern IP wording.
- No guns, knives, smoking, or source-seed leakage.
- Small cast, clear radio conflict, coherent ending.
- Fail loud after bounded repair attempts if the LLM cannot produce a valid
  brief/ledger.

Architecture direction:

- Add `original_radio` as its own source bank, not as a fake RSS/source lane.
- Add an original multi-pass runner only when its tests exist.
- Let the original branch generate compatibility `meta["news"]` fields for
  current downstream code, while stamping honest provenance under sidecars.
- Deep-think the creative route before coding; if the shape is still ambiguous,
  use the repo roundtable/Fable rules for a grounded design pass.

Dynamic visual-style direction:

- Explore an improvising visual pack for original episodes.
- It must validate through the same visual-style schema, never write ad-hoc
  disk packs during a render, and must not leak the style name into story
  premise, title, narration, or dialogue.

### 3. Source-Bank End-To-End Sweep

After the model-slot audit and `original_radio` work, prove every runnable bank
works end to end or fix it before moving on:

- `science_news`
- `media_archive`
- `public_domain_story`
- `shakespeare`
- `original_radio` once runnable

Use 30-word/random smokes first. Look specifically for:

- source-bank drift
- story/source leakage
- title or premise pollution from visual style
- weak cast separation
- stale sci-fi wording in non-sci-fi banks
- bad coda/source-note behavior
- forbidden content
- broken still/video/audio routing

Stop when each lane is good enough to proceed; do not polish forever.

### 4. Portability

Only after the above is green, continue portability work:

- no-GPU / procedural
- all-cloud
- RTX 8 GB, 12-16 GB, and 24 GB+
- Mac
- AMD where practical
- RunPod/cloud GPU

Canonical workflow remains `workflows/otr_canonical.json`; exported workflows
must be generated from canonical, not hand-maintained.

## Last Validation

Latest code chunk (`ac919d99`):

```text
pytest -q -p no:cacheprovider

7038 passed, 31 skipped, 1 xfailed
```

Bug Bible:

```text
cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
pytest -q -p no:cacheprovider tests\bug_bible_regression.py

16 passed, 7 skipped, 3 xfailed
```

This plan update is docs-only; no workflow JSON, node schema, or code surface
changed.

## Standing Rules

- `workflows/otr_canonical.json` is the canonical workflow.
- Any node/widget/wiring change must update that workflow in the same change.
- Every headless/API smoke must load the canonical workflow.
- Reset selectively before headless runs; never blanket-kill Python.
- Render assets go straight to `otr\episodes\<ep>\`, final to `otr\obs\`.
- Do not revert unrelated/user changes.
- Fix root causes, not shims.
- No silent fallback.
- JSON owns content/config.
- Python owns validation/routing/execution.
- Commit and push every green chunk to `origin/v2.0-alpha`.

## Pointers

- `ROADMAP.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/BUG_LOG.md`
- `docs/GO_FORWARD_ARCHIVE.md`
- `docs/2026-07-08-source-banks-v2-plan.md`
- `docs/google_tts_ideas.md`
- `docs/multimodal-story-schema/MEDIA_ARCHIVE_QA_HANDOFF.md`
- `workflows/otr_canonical.json`
