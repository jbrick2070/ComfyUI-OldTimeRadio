# OTR Go-Forward Plan

**Updated:** 2026-07-08
**Branch:** `v2.0-alpha`
**Status:** current sprint + next sprint handoff. Later sprints live in
`ROADMAP.md`.

This file is for short-term coordination only. It should not become a change
log. Move historical detail to `docs/GO_FORWARD_ARCHIVE.md`,
`docs/HANDOFF_LOG.md`, or dated docs.

## Current Sprint

### Google BYO API Build

Dedicated build thread:

- `019f3fbd-77e4-7700-aa5b-89a8f769c431`
- title: `OTR Google BYO API Build`

Authorized scope:

- Start with `google_tts`.
- Use `docs/google_tts_ideas.md`.
- Use `kibitz-runs/2026-07-07-google-tts-gemini-pro-r3/r3/final.md`.

Hard constraints:

- direct Google/Gemini BYO API, not Comfy Cloud / Partner nodes
- no local fallback
- no cross-provider fallback
- same-provider Google model retry only if explicit, bounded, logged, and no Pro
  by default
- `runtime: direct_api` is explicit-selection-only and excluded from automatic
  rank/default/fallback chains
- fail loud before invoke where possible
- redact API keys from errors/logs
- official Interactions REST shape remains source of truth unless a live probe
  proves it changed
- voice-quality gate: British-leaning announcer style, deterministic male/female
  announcer mix, gender-aware character casting, announcer voice not reused for
  characters by default

Required closeout:

- focused tests
- full repo suite
- Bug Bible
- commit and push green chunk to `origin/v2.0-alpha`
- verify `HEAD == origin/v2.0-alpha`

## Next Sprint

### Media Archive Seed Deck

Goal:

- Add a compact media-archive mystery/adventure seed deck.
- RSS remains the source of truth, parallel to the science lane.
- The seed is only the dramatic lens that turns the archive item into radio
  drama.
- Replace loose "National Treasure / Nancy Drew" prompt phrasing with a
  deterministic deck-selected lens.

Seed deck:

1. The Lost Reel
2. The China Girl Mystery
3. The Hidden Cut
4. The Locked Archive
5. The Vanishing Witness
6. The Unseen Frame
7. The Forgotten Broadcast
8. The Secret Location
9. The Returned Tape
10. The Missing Expedition
11. The Buried Interview
12. The False Ending
13. The Last Screening
14. The Vanished Landmark
15. The Clue in the Film

Implementation shape:

- Add `nodes/story_packs/media_archive/drama_seeds.json`.
- JSON owns the seed content.
- Python only loads, validates, selects, and injects the seed.
- Select deterministically from RSS payload/source hash.
- Inject prompt line like:
  `Dramatic seed lens: The China Girl Mystery`
- Prompt must state the RSS/source material remains source fact; the seed is not
  source fact.

Tests:

- JSON schema loads.
- Exactly 15 unique seeds.
- Includes `The China Girl Mystery`.
- Forbidden drift terms absent.
- Same payload selects the same seed.
- Different payloads can select different seeds.
- Built prompt includes exactly one selected seed.
- Prompt explicitly preserves RSS/source material as the source of truth.
- Existing media archive focused tests stay green.

## Near-Term Order

1. Finish and push the Google BYO API green chunk.
2. Add and test the media archive seed deck.
3. Move to `ROADMAP.md` sprint order.

## Current Known Good Checks

Recent fixture-level media archive check:

```text
tests/test_media_archive_sources.py
tests/test_media_archive_interpreter.py
tests/test_source_payload_chunk3.py
tests/test_story_routing_stage2.py
tests/test_story_rules_4a.py
tests/test_outline_seams_lane1.py
tests/test_exchange_seam_lane2.py
tests/test_style_catalog.py

210 passed
```

Doc ownership:

- `GO_FORWARD_PLAN.md` owns current sprint + next sprint only.
- `ROADMAP.md` owns all later sprints.
- The old `GO_FORWARD_PLAN.md` was stale and replaced with this short handoff.
- `docs/GO_FORWARD_ARCHIVE.md` remains the deep-history bucket.
- `docs/GO_FORWARD_NEXT/` contains older scoped specs; treat them as reference
  only unless a current task explicitly reopens one.

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
