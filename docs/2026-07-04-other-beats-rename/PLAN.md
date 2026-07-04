# ISOLATED CODING PLAN -- rename `other_beats_image` -> `character_image` (naming consistency)

Date: 2026-07-04
Branch: v2.0-alpha
Owner: the separate coding window (this is a scoped plan; grounded against the real files)

## 0. WHY (the confusion)

The role model is announcer / music / **character**. The VIDEO side already reflects this:
the rip-sfx-broll consolidation renamed `other_beats_video_model` -> **`character_video_model`**
(role_slots.py: "the legacy other_beats_video_model slot is GONE"). But the **IMAGE** side was
left half-migrated -- it still uses **`other_beats_image`** (slot_matrix.py:38
`IMAGE_KEYS = ("announcer_image", "music_image", "other_beats_image")`).

Result: `character_video_model` sits next to `other_beats_image` for the SAME role. Anyone who
tinkers (profiles, force-maps, the ImageDirector widgets) hits this drift and gets confused --
and it already bit a live run (a role_override on `character_image` errored "no widget-mapping
entry" because the key is still `other_beats_image`).

**Goal: rename the IMAGE role key + its widget from `other_beats*` to `character*`, so image
matches video.** Pure rename, no behavior change.

## 1. SCOPE (what changes)

- role KEY:    `other_beats_image`        -> `character_image`
- widget NAME: `other_beats_image_model`  -> `character_image_model`  (mirrors the video widget)
- any `OTHER_BEATS` image role token / `other_beats_visual` IMAGE references -> `CHARACTER` / `character_visual`
  ONLY where they denote the image role. LEAVE the historical/legacy-guard comments that document
  the already-removed `other_beats_video_model` slot (those are accurate history, not live keys).

## 2. GROUNDED SITE MAP (code -- change together in ONE commit)

- `nodes/_otr_shared/slot_matrix.py` -- `IMAGE_KEYS` (:38) + the `other_beats_visual` refs (:24,49,63)
  where they mean the IMAGE slot; `ROLE_TO_PROFILE_KEY` / `ALL_ROLES` if the image role appears.
- `nodes/_otr_shared/role_slots.py` -- the role->slot map entry for the image/character role (3 refs;
  keep the "legacy other_beats_video_model is GONE" history note).
- `nodes/otr_image_director.py` (4 refs) -- the `other_beats_image_model` INPUT_TYPES widget +
  the emitted policy key. This is the user-facing widget rename.
- `nodes/otr_image_gen_dispatcher.py` (3 refs) -- the role/slot lookups (incl. the `_radio_face_169`
  neighbourhood).
- `nodes/otr_meta_brief_image_prompt.py` -- `derive_scene_still_targets(... other_beats=...)` param.
- the **widget mapping** used by `nodes/_otr_workflow_apply.apply_profile` (the applier that failed
  on `character_image`): add/rename the `other_beats_image` -> node/widget entry to `character_image`.
- `nodes/_otr_engine_profiles.py` / `nodes/_otr_shared/capability_profiles.py` -- `_IMAGE_DIRECTOR_WIDGETS`
  admissible set (`otr_workflow_apply.py:144` lists `announcer_image_model, music_image_model,
  character_image_model`) -- confirm it already expects `character_image_model` (it does), so the
  widget rename ALIGNS the workflow with the code (this is why the drift existed).

## 3. WORKFLOW JSON (hard -- §0, POSITIONAL)

- `workflows/otr_scifi_16gb_full.json`: the ImageDirector node (`OTR_ImageGenDispatcher` /
  `OTR_ImageDirector`) has an `other_beats_image_model` widget. RENAME THE WIDGET NAME IN PLACE to
  `character_image_model` -- do NOT reorder or move it; `widgets_values` is POSITIONAL (BUG-LOCAL-097),
  so keep its slot index + value identical. Same change lands in the SAME commit as the code.
- Re-validate: `OTR_WorkflowValidator` + JSON round-trip + link/widget audit (widget count vs live
  `INPUT_TYPES`, every wired input-name in `INPUT_TYPES`).

## 4. COMPAT / SAFETY

- Profiles (`16gb_full` etc.) are CODE-defined + regenerated, so a clean rename of the role key is
  safe (no on-disk profile persists `other_beats_image`). Grep confirms no ledger schema field named
  `other_beats_image` (the key lives in profiles + policy JSON, both regenerated per run).
- Keep it a PURE rename: after it, `grep other_beats` should return ONLY the legacy history comments
  about the removed `other_beats_video_model` slot (intentional), not live keys/widgets.
- OPTIONAL (if you want zero-friction for any saved custom profile a tinkerer wrote): accept BOTH
  keys for one release -- map `other_beats_image` -> `character_image` at the applier with a LOUD
  deprecation log -- then drop it next release. Recommend the clean break unless you know external
  profiles exist.

## 5. TESTS (invert in the same commit)

~20 test files reference `other_beats*` (test_image_platform_c1, test_route_a_14b_promotion,
test_slot_matrix_soak, test_still_spine_helpers, test_video_platform_aseam, test_rip_sfx_broll_guard,
test_workflow_live_passes_validator, ...). Update the IMAGE-role references to `character_image`;
leave assertions that document the GONE video slot as-is. The soak SCRIPTS under `scripts/` that pass
`other_beats_image` role_overrides must update too (they're how you'll re-run the matrix).

## 6. VERIFY

Full suite + Bug Bible + strict-types after the change; workflow-JSON edited in the SAME commit;
`grep -r other_beats` shows only intentional legacy history; commit + push per green chunk to
v2.0-alpha. A mechanical rename this size is a good candidate for a codex/kibitz grep-audit pass to
confirm no live `other_beats` key/widget survives.
