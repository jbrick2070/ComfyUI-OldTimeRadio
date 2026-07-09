# OTR Go-Forward Plan

**Updated:** 2026-07-08
**Branch:** `v2.0-alpha`
**Status:** media archive seed deck green; next chunk starts from
`ROADMAP.md` source-pack order.

This file is for short-term coordination only. Longer runway lives in
`ROADMAP.md`; old sprint logs belong in `docs/GO_FORWARD_ARCHIVE.md`,
`docs/HANDOFF_LOG.md`, or dated docs.

## Current Status

### Completed In This Green Chunk

Media archive seed deck:

- Added `nodes/story_packs/media_archive/drama_seeds.json`.
- Media archive interpreter now loads, validates, deterministically selects,
  and injects exactly one dramatic seed from the source payload hash.
- Prompt now states that RSS/source material remains the source of truth and
  the seed is only a fictional lens, not source fact.
- Removed the loose "National Treasure / Nancy Drew" prompt phrasing.
- Story-pack routing now admits the media archive seed deck as a sidecar JSON
  without treating it as a routable story model.

Google BYO/API state:

- `v2.0-alpha` was already at origin parity before this chunk.
- Existing Google TTS/direct-API code was present and committed before this
  work began.

### Changed Files

- `nodes/_otr_media_archive_interpreter.py`
- `nodes/_otr_story_routing.py`
- `nodes/story_packs/media_archive/drama_seeds.json`
- `tests/test_media_archive_interpreter.py`
- `tests/test_story_routing_stage2.py`

Unrelated local file present before this work and left untouched:

- `docs/2026-07-08-source-banks-v2-plan.md`

## Validation

Focused media/source bundle:

```text
pytest -q -p no:cacheprovider tests/test_media_archive_sources.py tests/test_media_archive_interpreter.py tests/test_source_payload_chunk3.py tests/test_story_routing_stage2.py tests/test_story_rules_4a.py tests/test_outline_seams_lane1.py tests/test_exchange_seam_lane2.py tests/test_style_catalog.py

230 passed
```

Full repo suite:

```text
pytest -q -p no:cacheprovider

6955 passed, 32 skipped, 2 xfailed, 5 warnings
```

Bug Bible:

```text
cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
pytest -q -p no:cacheprovider tests\bug_bible_regression.py

16 passed, 7 skipped, 3 xfailed
```

Additional hygiene:

- `git diff --check` clean.
- `py_compile` clean for touched Python/test files.
- Workflow JSON was not edited; no node/widget/wiring surface changed.

## Next Action

Continue `ROADMAP.md` source-pack order.

Current runnable source banks in `nodes/story_packs/banks.json`:

- `science_news`
- `media_archive`
- `public_domain_story`
- `shakespeare`

Next concrete build chunk:

- Add the `original_radio` source-bank lane, including its pack/rules/runner
  surface and focused tests, or if that lane is explicitly deferred, begin the
  30-word smoke sweep across the four runnable banks.

## Standing Rules

- `workflows/otr_canonical.json` is the canonical workflow.
- Any node/widget/wiring change must update that workflow in the same change.
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
- `docs/google_tts_ideas.md`
- `docs/multimodal-story-schema/MEDIA_ARCHIVE_QA_HANDOFF.md`
- `workflows/otr_canonical.json`
