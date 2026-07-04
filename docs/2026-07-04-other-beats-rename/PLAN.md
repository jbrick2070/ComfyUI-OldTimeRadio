# CODING PLAN -- retire the last live `other_beats*` IMAGE surfaces -> `character*`

> SUPERSEDED TOKEN NOTE (2026-07-04): the third image role token is **character**, not `character`. This doc has been updated to `character_image` / `character_granularity`. The image MODEL widget stays `character_image_model` (separate already-done migration).


Date: 2026-07-04  Branch: v2.0-alpha
Status: PLAN ONLY -- not yet coding. Hardened via kibitz r1-r4 (codex gpt-5.5 + antigravity
gemini-3.5-pro) + a Fable final structural gate. Every file:line below was grounded against the
real Windows files this session.

## 0. WHY
Role model = announcer / music / **character**. Video already migrated (`character_video_model` /
`character_visual`); the image MODEL widget was renamed 2026-07-03 -> `character_image_model` (code +
workflow JSON node 87 both carry it). What was left behind: the role KEY `other_beats_image`, one
granularity WIDGET `other_beats_granularity`, and a scatter of soak/smoke scripts. Result: a
`character_image` role_override errors "no widget-mapping entry", and several scripts query a widget
that no longer exists. Finish the migration. **Pure rename, no behavior change.**

## 1. ALREADY DONE -- DO NOT TOUCH
- `character_image_model` model widget: DONE in `otr_video_director.py`, `otr_image_gen_dispatcher.py`,
  and workflow node 87 (OTR_VideoDirector, input[6] name+widget both `character_image_model`).
- `otr_meta_brief_image_prompt.py`: no `other_beats` refs.
- `_otr_shared/role_slots.py`: `other_beats` refs = HISTORY comments (removed VIDEO slot
  `other_beats_video_model`). LEAVE.
- `slot_matrix.py:48-51` `_DEAD_ROLE_OVERRIDE_KEYS` (`other_beats_visual`) = dead-key scrubber. LEAVE.
  Do NOT add `other_beats_image` to it -- fail-loud on a stale key is the correct clean-break behavior.
- `16gb_full.json`/`8gb_lite.json` are HAND-MAINTAINED (no generator writes them) -- edit directly.
- OUT OF SCOPE (video-migration debt, not this image rename; the §5 image gate does NOT match them):
  `other_beats_visual` in `otr_run_leg.ps1`, `otr_3d_quick_tests.ps1`, `otr_coverage_sweep.py:89`
  comment, and `tests/test_coverage_sweep_acceptance.py` (`sweep_other_beats_visual_*` fixtures).
  Track separately if desired.

## 2. SCOPE -- two live renames, driven by the §5 ignore-blind grep (not this hand-list).
`apply_profile` (`nodes/_otr_workflow_apply.py:468-473`) fails LOUD on any profile key with no
widget-mapping entry, so every producer of the KEY must move together.

### A. Role/profile KEY: `other_beats_image` -> `character_image`
- `nodes/_otr_shared/slot_matrix.py:38` `IMAGE_KEYS` (+ :36-37 comment).
- `config/profiles/widget_mapping.json:50` KEY `role_overrides.other_beats_image` ->
  `role_overrides.character_image`. TARGET widget stays `character_image_model` (unchanged).
- Checked-in profiles: `16gb_full.json:15`, `8gb_lite.json:14`.
- Scripts carrying the key: `_otr_yoga_soak:38`, `_otr_visual_soak_6leg:48`, `_otr_talking_radio_night:198`,
  `_otr_overnight_story_soak:198`, `_otr_overnight_420_soak:8,146` (incl. the line-8 comment),
  `_otr_ordered_soak:50`, `_otr_night_soak:140`, `_otr_night_matrix_soak:44`, `_otr_nightly_anthology_soak:261`,
  `_otr_cov_runner:49`, `_otr_cloud_audio_babysit:89`, `_otr_cloud_matrix_soak:41`, `_otr_cloud_video_soak:42`.
- **SILENT-DROP TRAP -- `scripts/_otr_combo_soak.py:83`** passes `image_engines={"other_beats_image": ...}`
  into `slot_matrix.build_all_role_profile()`. If not renamed with `IMAGE_KEYS`, the character override
  is SILENTLY dropped to `DEFAULT_IMAGE_BASELINE` (no crash). Rename the dict key to `character_image`.

### B. Granularity WIDGET: `other_beats_granularity` -> `character_granularity`
All FIVE move in ONE commit or ComfyUI raises `TypeError` (INPUT_TYPES names are passed to `direct()`
as kwargs by INPUT name):
1. `nodes/otr_image_director.py:253` INPUT_TYPES key.
2. `nodes/otr_image_director.py:314` `direct()` signature param.
3. `nodes/otr_image_director.py:351` consumer var (dict KEY `character_image_model` already correct).
4. Workflow JSON node 88 (see §3).
5. `tests/test_image_platform_c1.py:176,241,274` (`other_beats_granularity=` kwargs).

### C. Stale DEAD-widget references (name the retired MODEL widget `other_beats_image_model`)
Already broken vs the live JSON; rename to `character_image_model`:
`scripts/run_otr_30word_smoke.py:77,445`, `scripts/_otr_nightly_anthology_soak.py:216`,
`scripts/_otr_cloud_desktop_probe.py:41`.

## 3. WORKFLOW JSON (hard -- §0)  node 88 = OTR_ImageDirector
- Grounded shape: node 88 `input[4]` = `{"localized_name":"other_beats_granularity",
  "name":"other_beats_granularity","type":"COMBO","widget":{"name":"other_beats_granularity"},"link":null}`.
  Rename ALL THREE name fields -> `character_granularity` (`localized_name`, `name`, `widget.name`).
- `widgets_values` = `["per_object","per_object","per_object",15,"request_hash",42,"{}"]`; granularity
  value is index 2 and is POSITIONAL -- the rename touches only `inputs[]` name strings, so the value
  list is UNTOUCHED (BUG-LOCAL-097 safe). `link` is null -- no link-integrity impact.
- Node 87 model widget already `character_image_model` -- NO change. `OTR_ImageDirector` class name /
  `NODE_CLASS_MAPPINGS` in `__init__.py` are unchanged (only a widget renames).
- Same commit as the code. Re-validate: `OTR_WorkflowValidator` + JSON round-trip + link/widget audit.

## 4. COMMIT REALITY (Fable gate -- the non-obvious one)
13 of the ~16 in-scope script files are **git-ignored** (a `.gitignore` `scripts/_*.py`-class rule;
verify the exact rule with `git check-ignore -v <file>`. `git check-ignore` confirms the `_otr_*`
scripts are ignored while `run_otr_30word_smoke.py` is tracked). Consequence:
- Those 13 files CANNOT be part of the "one commit" -- git refuses them. Edit them in the same working
  SESSION (so nothing crashes on next run), but only the TRACKED files (`slot_matrix.py`,
  `widget_mapping.json`, `16gb_full.json`, `8gb_lite.json`, `otr_image_director.py`, the workflow JSON,
  `run_otr_30word_smoke.py`, tracked tests) land in the commit.
- The §5 grep gate MUST run ignore-blind (`rg --no-ignore` / `rg -uu`) or it FALSE-GREENS: a git-aware
  grep only sees `run_otr_30word_smoke.py` and misses all 13 ignored scripts still on the old key.

## 5. COMPAT + TESTS + VERIFY
- **Clean break, NO runtime shim (LOCKED).** codex + agy + Fable each confirmed the only on-disk
  persisters of `other_beats_image` are the three checked-in profiles (all in scope); no external
  profile corpus exists. `apply_profile` fails loud on any stray stale key -- correct behavior.
- **New regression:** assert `slot_matrix.IMAGE_KEYS == ("announcer_image","music_image","character_image")`
  and `"other_beats_image"` absent (covers the silent-drop; `test_slot_matrix_soak` checks only video keys).
- **New workflow audit:** assert node 88 input[4] `localized_name` == `name` == `widget.name` ==
  `otr_image_director` INPUT_TYPES key == `direct()` kwarg == `character_granularity`, AND
  `widgets_values[2] == "per_object"` unchanged (complements `test_workflow_json_guardrails.py`).
- **Test refs to update:** `test_image_platform_c1.py:176,241,274`; guard
  `test_rip_sfx_broll_guard.py:143-146` (name + "KEPT" comment); `test_still_spine_helpers.py:582,584,593`
  (method name `test_slot_absent_uses_other_beats_default` + comments). Do NOT touch already-clean
  `test_still_aspect_and_labels` / `test_credits_s2_durable_stamps`, and do NOT touch the exact
  membership checks `test_route_a_14b_promotion.py:144` and `test_still_spine_helpers.py:351`
  (`"other_beats" not in ...` -- they stay green and would break if edited). Leave GONE-VIDEO-slot asserts.
- **Grep gate (SCOPED + IGNORE-BLIND):** `rg -uu` over `nodes/`, `scripts/`, `tests/`, `config/`,
  `workflows/otr_scifi_16gb_full.json`; exclude `__pycache__`/`*.pyc`, `*.log`, `*.out`, `docs/`,
  `kibitz-runs/`. PASS = no live `other_beats_image`, `other_beats_image_model`, or
  `other_beats_granularity`. ALLOWED to remain: the `other_beats_video_model` history comments,
  the `other_beats_visual` scrubber list (`slot_matrix.py:48-50`), and the one migration-history
  comment `otr_image_gen_dispatcher.py:151-153` (contains the old literal by design; reword or allow).
  Save the exact pre-edit ignore-blind hit list to the work log so the inventory can't drift.
- Full suite + Bug Bible + strict-types. JSON edited in the SAME commit as the code; validator +
  round-trip green. Commit + push per green chunk to v2.0-alpha.
