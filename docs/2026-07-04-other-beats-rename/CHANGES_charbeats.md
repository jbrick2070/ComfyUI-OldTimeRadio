# REVIEW TARGET -- image role rename `other_beats_*` -> `char_beats_*` (uncommitted working tree)

Operator directive (2026-07-04): the third image role token must be **`char_beats`**, NOT `character`
and NOT `other_beats`. The image MODEL widget `character_image_model` is a DIFFERENT, already-migrated
name (2026-07-03) and is CORRECT as-is -- it is not an other_beats token, leave it.

## What changed (verify each is correct + complete, grounded against the real files)
- `nodes/_otr_shared/slot_matrix.py:38` IMAGE_KEYS -> ("announcer_image","music_image","char_beats_image").
- `config/profiles/widget_mapping.json:50` key -> role_overrides.char_beats_image (target widget stays
  character_image_model).
- `config/profiles/16gb_full.json`, `8gb_lite.json` -> char_beats_image.
- `nodes/otr_image_director.py` INPUT_TYPES key + direct() signature param + consumer var ->
  char_beats_granularity (dict key character_image_model unchanged).
- `workflows/otr_scifi_16gb_full.json` node 88 input[4]: localized_name + name + widget.name ->
  char_beats_granularity; widgets_values[2] == "per_object" (positional, unchanged).
- ~15 gitignored soak scripts: role key other_beats_image -> char_beats_image; stale widget
  other_beats_image_model -> character_image_model (the real widget).
- Tests: test_image_platform_c1 kwargs, test_rip_sfx_broll_guard name+comment, test_still_spine_helpers
  method+comments; run_otr_30word_smoke slot -> character_image_model. New regressions in
  test_slot_matrix_soak.py (IMAGE_KEYS==char_beats_image; node-88 widget audit).

## Deliberately NOT changed (flag if you disagree)
- All `other_beats_visual` / `other_beats_video_model` mentions: these belong to the SEPARATE
  2026-07-03 VIDEO-slot consolidation (that slot is now `character_visual`, NOT char_beats). They appear
  as (a) protective regression GUARDS that name the retired slot on purpose to prove it's gone
  (test_rip_sfx_broll_guard, test_route_a_14b_promotion, test_video_platform_aseam, test_slot_matrix_soak:66,
  test_still_spine_helpers:351), (b) already-broken live keys in some soak scripts that should be
  character_visual, (c) the dead-key scrubber literal slot_matrix.py:49, (d) historical result-data JSON.

## Questions for the reviewer
1. Is the char_beats image-role rename COMPLETE and CORRECT (no live other_beats_image /
   other_beats_granularity / other_beats_image_model producer left; no coupling half-applied)?
2. Any place where char_beats_image (key) -> character_image_model (widget) coexistence actually breaks?
3. Did I miss any live char_beats/other_beats collision (e.g. a substring mangle from the scripts pass)?
