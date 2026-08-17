VERDICT: yes-with-fixes. The data-flow architecture (Rows 1-5, 7, 8) is correctly grounded in `SourceIdentity` and `SafeOpenBrief`, but contains 3 implementability defects (scoping/unbound local assumption in `OTR_LedgerScriptWriter.py`, missing `public_domain` pack seam update in Row 6, and unspecified fallback string template in Row 5) that will cause runtime crashes or test regressions if built as written.

MUST-FIX BEFORE BUILD:
1. [Row 2 & Row 3 / `nodes/OTR_LedgerScriptWriter.py:4090-4124, 4861, 6084`] Unbound local / missing import at `SafeOpenBrief` instantiation sites.
   - DEFECT: `_otr_source_identity` is imported only conditionally inside `if bool((_source_bank_row.defaults or {}).get("provenance_normalize", False)):` at lines 4090-4122, and `_identity` is a local variable inside that branch. Calling `_identity.work_title` at line 4816 (`OutlineRequest`), line 4861 (`SafeOpenBrief` in-loop), or line 6084 (`SafeOpenBrief` rewrite) will raise `UnboundLocalError` on any execution path where `provenance_normalize` is false or bypassed. Furthermore, `_OTRSID` is not in module scope.
   - FIX: Import `_otr_source_identity as _OTRSID` at module level in `OTR_LedgerScriptWriter.py` (it is a pure-Python dataclass module with no heavy dependencies). In `run()`, compute `_work_title = _OTRSID.identity_from_meta(meta).work_title` unconditionally before section F, and pass `work_title=_work_title` to `_OTRO.OutlineRequest` (line 4816), `_OTRLC.SafeOpenBrief` in-loop (line 4861), and `_OTRLC.SafeOpenBrief` at the I.4.9 rewrite (line 6084).

2. [Row 6 / `nodes/story_packs/public_domain/faithful_radio_adaptation.json:16`] Pack seam contradiction on public domain adaptation lane.
   - DEFECT: Row 6 amends `announcer_intro_safe_system` with Fable's lines ("Sentence 1 names tonight's work from the WORK line..." and "Use ONLY the WORK title and the proper names in the cast list below; invent none."), but the plan only targets Shakespeare. `public_domain` is also an adaptation lane where `identity_from_meta(meta)` extracts `work_title` (`Nonsense Novels`, `Dracula`) and `SafeOpenBrief` will now render `WORK:`. Leaving `public_domain`'s seam as "Use ONLY the proper names in the cast list below; invent none" creates an explicit prompt contradiction where the user message supplies `WORK:` but the system prompt forbids using it.
   - FIX: Update `announcer_intro_safe_system` in BOTH `nodes/story_packs/shakespeare/folger_scene_adaptation.json` and `nodes/story_packs/public_domain/faithful_radio_adaptation.json` to include the `WORK` line allowance.

3. [Row 5 & Row 10 / `nodes/_otr_line_composer.py:1248-1255`] Unspecified deterministic fallback string template.
   - DEFECT: Row 5 specifies that "`fallback_safe_open` carries the work when non-empty", and Row 10 requires presenting "a scene from" the work, but neither provides the concrete string template. Leaving this undefined leads to inconsistent implementations and breaks fallback contract assertions (`tests/test_announcer_safe_open_contract.py:302`).
   - FIX: Implement the exact fallback signature and template in `nodes/_otr_line_composer.py`:
     ```python
     def fallback_safe_open(safe_open_brief) -> str:
         work = clean_one_line(getattr(safe_open_brief, "work_title", ""))
         setting = clean_one_line(getattr(safe_open_brief, "setting", ""))
         time_of_day = clean_one_line(getattr(safe_open_brief, "time_of_day", ""))
         where = ", ".join(p for p in (time_of_day, setting) if p)
         prefix = f"Tonight, a scene from {work}. " if work else ""
         if where:
             return f"Good evening. This is SIGNAL LOST. {prefix}We open on {where}."
         if work:
             return f"Good evening. This is SIGNAL LOST. Tonight, a scene from {work}."
         return "Good evening. This is SIGNAL LOST."
     ```

SHOULD-FIX:
1. [Row 4 / `nodes/_otr_outline.py:551-605, 982-1044`] Prompt builder parity between `_build_macro_user_prompt` and legacy `_build_user_prompt`.
   - DEFECT: Row 4 adds `work_title` rendering to `_build_macro_user_prompt`, but leaves `_build_user_prompt(req)` unmodified. While production uses the Stage 1 macro prompt, unit test harnesses (`tests/test_line_composer_arc.py`, `tests/test_otr_casting.py`) continue to exercise `_build_user_prompt`.
   - FIX: Add `if req.work_title.strip(): parts.append(f"Work: {req.work_title.strip()}")` to both `_build_macro_user_prompt(req)` and `_build_user_prompt(req)`.

2. [Row 1 & Row 10 / `nodes/_otr_line_composer.py:1386-1397`] Settle Row 10 prompt formatting on the clean title symbol.
   - DEFECT: Row 10 leaves open whether to format the context line as `WORK: a scene from <title>` or `WORK: <title>`. Formatting with "a scene from" pollutes the raw bibliographic field and complicates exact string tests (Row 7).
   - FIX: Settle Row 10 explicitly on `WORK: {clean_one_line(safe_open_brief.work_title)}`. Let the system prompt seam (Row 6) provide the framing instruction ("Sentence 1 names tonight's work from the WORK line... as a scene from...").

3. [Row 8 / `tests/test_announcer_safe_open_contract.py` or `tests/test_closing_seams_bank_routing.py`] Concrete fixture for cross-play leak check.
   - DEFECT: Row 8 defines the cross-play leak test conceptually, but omits the manifest source and token extraction mechanism.
   - FIX: Ground the test on `config/source_banks/shakespeare/curated_scenes.sample.json`. Extract all 14 `play_title` entries and maintain a test fixture mapping each play to signature tokens (e.g. `{"Twelfth Night": ["Illyria", "Orsino"], "Romeo and Juliet": ["Verona", "Capulet", "Montague"], "Hamlet": ["Elsinore", "Denmark"]}`). For a generated Twelfth Night run, assert that no signature token from any disjoint play row appears in the output.

OPTIONAL / NICE-TO-HAVE:
1. [Row 3 / `nodes/_otr_story_brief.py:883-895`] Update docstring on `ProducedOpenBriefModel` to explicitly document that `work_title` is intentionally omitted from the scene-1 derive model (preserving input starvation) and passed by the caller directly from `meta`.
2. [Row 1 / `tests/test_announcer_safe_open_contract.py`] Add a test case verifying that `safe_open_viability` still returns `"missing_scene_context"` if `work_title` is populated but `setting` and `opening_status_quo` are empty, guaranteeing that a bare title without scene context never bypasses the starvation check.

CUT THESE (over-engineering):
1. [Section 6 / Pro's SF2 Runtime Proper-Name Gate]: The driver and Fable already replaced Pro's runtime assertion with a test-only check. Reaffirm cutting any runtime validation/filtering of proper names in `setting` or `compose_announcer_intro`. Safe to cut because runtime rejection violates THE LAW (no episode rejection) and breaks 5 of 14 valid Shakespeare manifest rows (`Elsinore`, `Capulet's garden`, `Ephesus`).

[ASSUMPTION] `config/source_banks/shakespeare/curated_scenes.sample.json` is the authoritative 14-scene manifest used for test grounding.
[ASSUMPTION] `public_domain` bank's story pack is intended to share the `WORK:` safe-open mechanism since `identity_from_meta` extracts `work_title` for public domain as well.
