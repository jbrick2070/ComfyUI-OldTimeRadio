R1 JUDGMENT (Claude, sole judge) -- STAGE1_SUBPLAN.md

Panel: codex + antigravity (per operator kibitz rule; claude CLI dropped). Both
returned VERDICT=no, code-grounded, and CONVERGED on the same structural set. I
grounded every load-bearing claim against the real files.

ACCEPTED (folded into STAGE1_SUBPLAN.md v2):
1. [CONFIRMED] Object-identity coupling. `resolve_creative_system_prompt(repo_id,
   phase)` returns `_MODERN_BY_PHASE[phase]` by object identity; grounded:
   `_otr_creative_prompt_router.py:67,100`, test asserts `out is expected`
   (`test_creative_prompt_router.py:62`), and `_otr_outline.py:1847`
   `if resolved is _SYSTEM_PROMPT`. A JSON-loaded string breaks these `is` checks.
   FOLD: Stage 1 ships the DORMANT foundation only (loader + pack + tests, NO
   consumer). Consumer wiring moves to a separately-gated "Stage 1b" whose
   precondition is the deliberate `is`->`==` migration (+ test migration).
2. [CONFIRMED] Router signature has no pack identity; callers pass only
   creative_repo_id+phase (`_otr_outline.py:1839-1840`, `_otr_line_composer.py:
   2063-2066`). FOLD: Stage 1b introduces a repo_id->pack-coordinates map in the
   router (not a hardcode scattered across callers); documented, not built in S1.
3. [CONFIRMED] Composite runtime prompts. coda = `_NEWS_CODA_SYSTEM +
   _NEWS_CODA_SYSTEM_V2_EXAMPLES` (`_otr_line_composer.py:3407`); outro tail
   (:3517); inventor/chooser are system+user pairs (`_otr_style_picker.py:296/301/
   329/334`); announcer intro has `_SAFE` twin (:2905/2926). FOLD: seam keys are
   defined at RUNTIME-MESSAGE granularity (split keys), and the byte-identity test
   targets the ASSEMBLED runtime string, not a single constant.
4. [CONFIRMED] pydantic undeclared; `news_interpreter.py:66-70` has a v1 fallback;
   requirements/pyproject pin nothing. FOLD: hand-rolled stdlib validator (no dep,
   works v1/v2, "quietest+secure").
5. [CONFIRMED] `pack_value or PY_CONST` is a hidden fallback for a MIGRATED seam.
   FOLD: migrated seam missing/empty -> RAISE (get_pack_prompt); None only for a
   not-yet-migrated seam (get_pack_prompt_or_none). Python constant = the
   byte-identity ORACLE (test-time), never a runtime fallback. Unknown triple = raise.
6. [CONFIRMED] Duplicate JSON keys survive `json.load`. FOLD: object_pairs_hook
   dup-key rejection.
7. [CONFIRMED] "Canonical workflow untouched" must be a GATE. FOLD: sha256/no-diff
   assertion on otr_scifi_16gb_full.json in the Stage 1 test suite.
8. [ACCEPTED] Exact PRODUCTION_SEAM_ALLOWLIST literal in the doc; status validation
   deferred; unused future fields kept as inert/tolerated (known-field set so
   typos still reject) but NOT validated/used in Stage 1.
9. [ACCEPTED wording] "byte-identical" = character-exact str==str (AST-extracted
   str), never compared to `bytes`.

NOTED / DEFERRED (not Stage 1):
- Validator/critic system prompts (_CONTINUITY/_QA/EDITOR_CONSTRAINTS/_AUDITOR/
  _CRITIC) stay Python = acknowledged deferred debt; out of the creative-seam scope.
- Centralizing all seam resolution through one entrypoint (antigravity SHOULD#2) =
  a Stage 2 principle; premature in S1.

REJECTED: none material -- both reviews grounded cleanly.

Net: Stage 1 shrinks to a provably-safe DORMANT foundation; the risk (identity
coupling + live wiring) is quarantined into Stage 1b with its own gate.
