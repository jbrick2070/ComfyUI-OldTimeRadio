VERDICT: yes-with-fixes. The plumbing is sound, but the seam wording change ignores the JSON pack routing and will fail to ship for real banks.

MUST-FIX BEFORE BUILD:
1. [Row 6] defect: The `announcer_intro_safe_system` prompt is pack-routed (documented in `_otr_creative_prompt_router.py`). Changing the Python constant `_ANNOUNCER_INTRO_SYSTEM_SAFE` only updates the science extraction fixture. The new Fable seam wording will not reach production episodes without updating the JSON banks.
   fix: Update the `announcer_intro_safe_system` string in all JSON `source_banks` files, and update the byte-identity assertions in `tests/test_story_pack_stage1.py`.
2. [Row 3] defect: The plan mandates "no dialogue parsing to recover a value we already hold" but doesn't specify how the I.4.9 rewrite gets the title. The `_rw_brief` (which is a `ProducedOpenBriefModel`) does not contain `work_title`.
   fix: At `OTR_LedgerScriptWriter.py` line 6084, explicitly thread `work_title=_identity.work_title` (or `_OTRSID.identity_from_meta(meta).work_title`) directly into the `SafeOpenBrief` constructor, exactly how `era` is threaded from `meta`.

SHOULD-FIX:
1. [Row 10] defect: The plan suggests either rendering "WORK: a scene from <title>" or letting the row-6 sentence carry it. Mixing phrasing logic into the Fable seam sentence ties wording to the JSON packs, making future updates harder.
   fix: Render `WORK: a scene from <title>` in Python (`_build_macro_user_prompt` and `compose_announcer_intro`) when `work_title` is non-empty. This keeps the data payload clean and keeps phrasing logic in code.

OPTIONAL / NICE-TO-HAVE:
- [Row 8] For the cross-play leak check test, use a static list of known plays/locations from the Shakespeare manifest fixture to assert the absence of other rows' signature names.

CUT THESE:
- None.

[ASSUMPTION] I am assuming the test suite has access to the Shakespeare manifest to implement the Row 8 cross-play leak check.
