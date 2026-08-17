VERDICT: no. The plan's core assumption that `identity_from_meta` yields `""` for `media_archive` is empirically false, which breaks the interface contract for all media archive episodes by treating the feed publisher as a play.

MUST-FIX BEFORE BUILD:
1. [WHAT NEITHER LANE HAD] Interface Contract Flaw. The driver asserts: "On original and media_archive, identity_from_meta yields \"\", the context tuple's if value filter omits WORK entirely". This is false. `nodes/_otr_source_identity.py` (lines 217-233) explicitly maps `media_archive`'s `source_label` (publisher) to `work_title`. Consequently, `media_archive` episodes will receive a populated `_work_title` (e.g. "Now See Hear!"). This breaks the build by injecting `WORK: a scene from {publisher}` into the announcer's prompt and `Adapted work: {publisher} -- the setting must belong to this work...` into the outline request, hallucinating a play where none exists. Concrete fix: In `nodes/OTR_LedgerScriptWriter.py`, conditionally clear the title for non-adaptation lanes: `_work_title = _identity.work_title if _identity.source_kind in ("shakespeare", "public_domain") else ""`.

SHOULD-FIX:
1. [ADOPTED r2 PLAN / 9] Test Suite Masking. The driver plans to "Update the byte-identity assertions in `tests/test_story_pack_stage1.py`". If the `media_archive` bug above is committed as-is, the test suite will fail. Updating the assertions *without* fixing the `_work_title` extraction logic would cement the bug into the test suite. Concrete fix: Apply the conditional fix for `_work_title` above *before* re-baselining the test assertions, ensuring `media_archive` stays byte-identical.

OPTIONAL / NICE-TO-HAVE:
1. [THE ONE FORK] The `_work_line` helper in `_otr_line_composer.py` correctly encapsulates the "a scene from" logic, but `fallback_safe_open` re-implements it inline (`tonight = f" Tonight, a scene from {work}." if work else ""`). Reusing `_work_line` or a shared constant prefix would prevent future drift between the prompt and the structural fallback.
2. [DOMAIN PROFILE: IMPORT ISOLATION] The module-scope import of `_otr_source_identity` in `nodes/OTR_LedgerScriptWriter.py` is safe because `_otr_source_identity.py` only imports `dataclass` and `typing` (stdlib). It passes the ComfyUI heavy-import prohibition.

CUT THESE (over-engineering):
None.
