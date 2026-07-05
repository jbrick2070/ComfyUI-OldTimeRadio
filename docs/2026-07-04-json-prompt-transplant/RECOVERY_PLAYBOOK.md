# RECOVERY PLAYBOOK -- Phase A pivot: sibling transplant -> in-repo (Phase A-R)

2026-07-04. Verdict: R (in-repo), unanimous HIGH (`kibitz/arch_decision_JUDGMENT.md`).
Sibling `main@7df7c80` + uncommitted Phase A (55 green). OTR `v2.0-alpha@a7bdc42d`, clean.

## Disposition of the uncommitted sibling work

| Sibling artifact | Verdict | Destination |
|---|---|---|
| `contracts.py` 4 seam keys (`outline_macro/phase/beat_system`, `line_composer_system`) | SALVAGE | seam vocabulary tuple in new `nodes/_otr_prompt_packs.py` |
| `extractor.py` `get_pack_prompt_or_none` (None=passthrough, fail-loud unknowns) | SALVAGE | same file -- port semantics verbatim, backed by direct `json.load` |
| Byte-identity harness (`tests/snapshots/`, `test_phase_a_byte_identity.py`) | SALVAGE | OTR `tests/test_phase_a_byte_identity.py` -- AST-extract the 5 constants from OTR's OWN `nodes/` files (mirror indirection deleted) |
| Extractor coverage tests | SALVAGE | OTR `tests/test_prompt_pack_coverage.py` |
| `profiles.py` 4 `str\|None` fields | PARK | lab machinery; in-repo shape has no StoryPromptProfile |
| `production_mirror/` refresh (Chunk 0) + bridge artifact | DISCARD | the drift-tax the judgment killed |
| Anchor doc edits | PARK | sibling stays scratch reference |

**Tree handling: commit-then-abandon.** One commit on sibling branch `phase-a-scratch` ("Phase A scratch -- superseded by OTR in-repo A-R"); never merge to `main`. Not stash (invisible, rots, violates never-lose-work); not dirty (one careless checkout erases the port's reference diff). The commit IS the port's diff-source.

## In-repo Phase A-R shape

- **Packs live at `nodes/story_packs/<bank>__<model>__<pipeline>.json`** -- `prompt_stages` keyed by seam.
- **Loader:** `nodes/_otr_prompt_packs.py` -- direct `json.load()`, fail-loud (`PromptPackError`) on unknown pack id or seam key; absent/empty seam returns None = production keeps its Python literal. Consumed by the existing resolver `nodes/_otr_creative_prompt_router.py` (pack override first, else the module constant). No new node, no new widget.
- **Byte-identity guarantee:** ship ZERO overriding packs, so the router's audio-C7 contract holds -- default path returns the SAME object references as today; the harness pins the 5 constants' exact bytes; `test_audio_byte_identical` stays green.
- **Chunks (each = one commit + push + full suite + Bug Bible, `v2.0-alpha`):**
  - **A-R1:** loader + seam vocabulary + fail-loud tests.
  - **A-R2:** byte-identity harness; snapshots committed IN the same commit (no skip-on-first-run drift).
  - **A-R3:** coverage test over all packs x seams.
  - **A-R4:** wire loader into the router (None-passthrough); docs; full regression + soak, audio byte-identical vs a7bdc42d.

## GO_FORWARD_PLAN.md update (paste into section 1, top)

```
**JSON PROMPT EXTRACTION -- PIVOTED IN-REPO (Phase A-R), 2026-07-04.** Kibitz verdict R
(in-repo), unanimous HIGH: sibling transplant STOPPED; production_mirror + bridge DISCARDED;
sibling = scratch only (uncommitted Phase A parked on sibling branch phase-a-scratch).
Source of truth: docs/2026-07-04-json-prompt-transplant/RECOVERY_PLAYBOOK.md
(PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md is SUPERSEDED -- port reference only).

CURRENT STEP = Phase A-R chunk A-R1: create nodes/_otr_prompt_packs.py (direct json.load
loader, fail-loud unknown id/seam, None-passthrough) + nodes/story_packs/ + tests.
Then A-R2 byte-identity harness -> A-R3 coverage -> A-R4 router wire-in. Each chunk =
commit+push+suite+Bug Bible on v2.0-alpha. No workflow JSON change in Phase A-R (no widgets).
Baseline: a7bdc42d. Invariant: audio byte-identical; default path = zero overriding packs.
```

## Do-not-lose

1. **Router object-identity IS the byte-identity contract** (audio C7): default path must return the same string objects; override only when a pack explicitly carries the seam. Snapshots UTF-8 no BOM, exact bytes.
2. **`line_grounding` stays deferred** (conditional f-string) and `OTR_PERIOD_SYSTEM_PROMPT`/`test_period_prompts.py` stay Python literals -- do not "helpfully" extract either.
3. **No workflow-JSON edit in A-R** -- and if a later phase adds a widget, append at END of `widgets_values` in the SAME change (BUG-LOCAL-097 / rule 0).

## Confidence

HIGH -- every salvage target has a green sibling reference diff and a named OTR seam already in the tree.
