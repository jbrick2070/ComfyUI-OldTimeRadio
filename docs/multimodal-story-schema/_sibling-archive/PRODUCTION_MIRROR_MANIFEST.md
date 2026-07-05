# Production Mirror Manifest

Created: 2026-07-02. This lab was cleared and rebuilt as a transplant
workspace. The v1 standalone lab (contracts/catalogs/fixtures/preview/nodes)
is preserved in git history at commit `41c6512` (snapshot) / `cf14138`
(merge); nothing was lost.

## Baseline

Mirrored from `ComfyUI-OldTimeRadio` at:

```text
commit a7bdc42de2a7dde32c4ea5350141e770aa8ae03a
date   2026-07-04 18:07:59 -0700
title  docs: July-4 sprint queue results -- Sprints 1+2 + Sprint 3 item 2 DONE/green/pushed; item 1 (big LLM prompt update) BLOCKED as underspecified (no converged code-ready diff; USL transplant is gated behind an explicit chunk) -- operator unblock options recorded
```

Phase A refresh (2026-07-04): re-mirrored from OTR `v2.0-alpha` at
`a7bdc42d` under the 7-chunk plan at
`docs/2026-07-04-json-prompt-transplant/PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md`
in the sibling OTR repo. The 5 docs under
`production_mirror/docs/2026-07-01-source-bank-visual-style-code-ready/`
never lived in OTR (they were snapshotted into the sibling at v1 rebuild
commit `ccde304`); their hashes/sizes are unchanged and they are excluded
from the Phase A refresh.

The prior `d48a9d76` baseline was AFTER rip-sfx-broll (6bad6e5b,
2026-07-01): the workflow and all mirrored code are the SFX-free
surface. `a7bdc42d` remains on the SFX-free surface.

## Layout And Rules

- `production_mirror/` - pristine read-only reference copies. Never edit.
  Diff transplant work against these.
- `workflows/otr_scifi_16gb_full.json` - the editable working copy of the
  SFX-free canonical workflow. Transplant edits happen HERE first, validated
  here, and only later applied to production in one explicit chunk.
- `production_mirror/workflows/otr_scifi_16gb_full.json` - the untouched
  baseline of the same file (hash below) for diffing.
- Production repo stays untouched until the transplant chunk. Nothing in this
  lab is imported by production.
- Before applying any transplant chunk to production, re-run the drift check:
  compare `production_mirror` hashes against the live files; if production
  moved past `a7bdc42d`, re-mirror and re-validate first.
- Gates for touching the real workflow JSON:
  `docs/FABLE_FINAL_REVIEW_2026-07-02.md` (TEST / VALIDATION GATES section).

## Copied Files (SHA256 first 16 hex, size bytes)

```text
5F6028E0A3F552B3  workflows\otr_scifi_16gb_full.json  35090
169BAEABF1B76124  nodes\OTR_LedgerScriptWriter.py  307390
89DB11644A8EACE3  nodes\OTR_LedgerFreezeCascade.py  22498
8D22076839E1196A  nodes\news_interpreter.py  37801
59242F82C2827555  nodes\_otr_outline.py  114108
8012F63D052ED54C  nodes\_otr_pitch_room.py  22793
432FF00F164A2375  nodes\_otr_story_select.py  40874
BCAAD9811B4369A2  nodes\_otr_dramatic_state_llm.py  24841
FFADD3B181BBCA93  nodes\_otr_line_composer.py  174252
0E6F2AE9FFB09C28  nodes\_otr_casting.py  86480
837ABE6C24B22E8C  nodes\_otr_style_picker.py  35301
24A6E12A888CE4C6  nodes\_otr_story_quality_l12.py  38405
349029FA70668436  nodes\_otr_story_spine.py  45973
6321913E9C3E3919  nodes\_otr_story_brief_helpers.py  27120
CBC2C481021EB5E5  nodes\otr_meta_brief_image_prompt.py  102112
73192667D9FC0E34  nodes\otr_shot_lock.py  43486
65057BC40CCB8D6E  nodes\_otr_video_engines\render_driver.py  145334
041F04E9893A3464  nodes\_otr_ledger_freeze.py  32789
134629DAAFD9F30E  nodes\_otr_legacy_to_stage1_adapter.py  25581
53E857A9354E87AA  nodes\_otr_speaker_role.py  9828
FFA55BEAFFC8B997  nodes\_otr_workflow_apply.py  22501
C4B5C7463704D5E9  nodes\_workflow_validation.py  17366
5845C8C56EF71E8C  nodes\_otr_workflow_validator.py  19123
19241445772459C6  nodes\_otr_shared\role_slots.py  4236
78E75C883176924B  nodes\_otr_shared\role_compat.py  6292
DAED2580842493CC  scripts\otr_api.py  38916
611BE254E0663DF8  docs\...\LEDGER_PROMPT_AUDIT.md  8492
D2FB4F7ED0643A0D  docs\...\PHASE2_PROMPT_PY_UPDATE_MAP.md  24623
96CC7B6320601833  docs\...\VISUAL_PROMPT_AUDIT.md  8706
769F8EC1143C9FDA  docs\...\STORY_AND_VISUAL_SCI_FI_REMNANTS_ARTIFACT.html  18958
FAC600B554CAED6C  docs\...\VISUAL_SCI_FI_REMNANTS_ARTIFACT.html  14549
```

Selection rationale: every file named as a transplant edit site or gate
reference in `docs/FABLE_FINAL_REVIEW_2026-07-02.md`, `PROMPT_SURGERY_CHECKLIST.md`,
and the production Phase 2 map - prompt sites (story + visual), the meta.news /
news_seed shape owners, role vocabulary, workflow validators, and the
API/apply whitelists.
