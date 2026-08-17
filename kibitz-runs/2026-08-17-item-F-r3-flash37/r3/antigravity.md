VERDICT: yes-with-fixes. The core wiring and sequencing (unconditional `_work_title` resolution, `SafeOpenBrief` field addition, composer-side `WORK:` label rendering, and viability separation) are structurally sound and preserve node contracts and lazy-import invariants, but the manifest-derived cross-play detector test fails in execution due to token/possessive mismatch, and `fallback_safe_open` retains legacy `getattr` calls contrary to direct-access policy.

MUST-FIX BEFORE BUILD:
1. [ADOPTED r2 PLAN row 10 / DEFECTS IN THE PANEL'S OWN PROPOSED CODE item 3 / `tests/test_cross_play_frame_leak.py`]
   Defect: `test_cross_play_frame_leak.py::test_the_detector_would_have_caught_the_shipped_defect` fails (`AssertionError: the detector does not flag the frame that actually shipped`). The detector extracts capitalized tokens from `curated_scenes.sample.json` (`["Capulet's", 'Juliet', 'Romeo', ...]`), but the manifest synopses contain neither "Verona" nor "Montagues", and the possessive token `"Capulet's"` fails the regex word-boundary match against `"Capulets"` in the shipped frame (`"Tonight, from Verona, where the Capulets and the Montagues keep their long quarrel."`). The detector produces 0 hits on the exact shipped defect it was written to pin.
   Fix: In `tests/test_cross_play_frame_leak.py`, normalize `_signature_terms` to strip possessives (`.rstrip("'s")` to extract root `Capulet`) and index base/plural forms (`Capulet`, `Capulets`, `Capulet's`), and ensure `curated_scenes.sample.json` rows or signature dictionaries include canonical setting and house tokens for Shakespeare plays (e.g. Verona, Mantua, Venice, Illyria, Montagues, Capulets).

2. [ADOPTED r2 PLAN row 6 / `nodes/_otr_line_composer.py:fallback_safe_open`]
   Defect: In `nodes/_otr_line_composer.py:1285-1286`, `fallback_safe_open` accesses `getattr(safe_open_brief, "setting", "")` and `getattr(safe_open_brief, "time_of_day", "")` while line 1292 accesses `safe_open_brief.work_title` directly. This violates the direct attribute access contract mandated in Section "DEFECTS IN THE PANEL'S OWN PROPOSED CODE" Item 1 ("DIRECT attribute access, never getattr-with-default").
   Fix: In `nodes/_otr_line_composer.py:1285-1286`, replace `getattr` calls with direct attribute access: `clean_one_line(safe_open_brief.setting)` and `clean_one_line(safe_open_brief.time_of_day)`.

SHOULD-FIX:
1. [ADOPTED r2 PLAN row 1 and row 5 / `nodes/OTR_LedgerScriptWriter.py:4080 and 4139`]
   Defect: `_OTRSID.identity_from_meta(meta)` is evaluated twice in `nodes/OTR_LedgerScriptWriter.py` -- once at line 4080 to bind `_work_title = _OTRSID.identity_from_meta(meta).work_title`, and a second time at line 4139 (`_identity = _OTRSID.identity_from_meta(meta)`) inside the `provenance_normalize` block.
   Fix: Bind `_identity = _OTRSID.identity_from_meta(meta)` once at line 4080; assign `_work_title = _identity.work_title`, and reuse `_identity` directly at line 4139 without re-parsing `meta`.

2. [ADOPTED r2 PLAN row 7 / `nodes/_otr_outline.py:609 and 1029`]
   Defect: In `nodes/_otr_outline.py`, `_build_macro_user_prompt` (lines 1029-1033) and `_build_user_prompt` (lines 609-613) use raw `req.work_title.strip()` inside the appended instruction `Adapted work: {req.work_title.strip()} -- the setting must belong to this work and to no other.`. If `req.work_title` contains embedded newlines or trailing punctuation from an uncleaned source payload, it is not sanitized before prompt formatting.
   Fix: Wrap `req.work_title` with `clean_one_line(req.work_title)` matching the normalization applied in `nodes/_otr_line_composer.py:_work_line`.

OPTIONAL / NICE-TO-HAVE:
1. [tests/test_closing_seams_bank_routing.py:167]
   Audit remaining tests across the suite to ensure all test fixtures pass concrete `SafeOpenBrief` dataclass instances rather than legacy `types.SimpleNamespace` objects.
2. [workflows/otr_canonical.json]
   Confirmed: No changes to `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, or widget positional indices in `nodes/OTR_LedgerScriptWriter.py`. `workflows/otr_canonical.json` requires no migration.

CUT THESE (over-engineering):
1. [Pro SF1 Hand-Listed Blocklist]
   Hand-curating a static list of play names and locations in `tests/test_cross_play_frame_leak.py` is safely cut in favor of manifest-derived signature terms, provided token stemming/possessive stripping is implemented (MUST-FIX 1).
2. [Adding work_title to ProducedOpenBriefModel]
   `nodes/_otr_story_brief.py:ProducedOpenBriefModel` does not need `work_title`. The announcer rewrite step in `nodes/OTR_LedgerScriptWriter.py:6102-6116` correctly injects `_work_title` from top-level `meta`, which avoids non-deterministic LLM extraction from scene 1 dialogue.
