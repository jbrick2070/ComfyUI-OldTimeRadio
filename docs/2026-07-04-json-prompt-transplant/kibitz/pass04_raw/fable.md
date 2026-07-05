# r4 Review -- Fable (convergence)

**VERDICT: GO-WITH-ONE-FIX.** No drift, no collisions, imports and alias are clean -- but the Chunk 5 sibling harness *as specified* cannot execute. One genuinely new must-fix cluster, mechanically resolvable in-round; everything else is code-ready.

## MUST-FIX

**MF-R4-1 -- Chunk 5 sibling snippet is triple-broken; respec to extraction-based:**
- `mirror_nodes` fixture returns a **Path** (`tests/conftest.py:23-25`), not a module -- `mirror_nodes._otr_outline._SYSTEM_PROMPT` is an AttributeError.
- Importing the mirror instead is barred and impossible: `test_compat_drift.py:3-5` ("Never imports mirrored modules -- dependency-incomplete by design"); mirror `_otr_outline.py:37` imports `._otr_episode_budget`, which is **absent** from `production_mirror/nodes/` (no `__init__.py` either). Snapshot the 5 constants via a `compat.py`-style AST/text extractor over the mirror files -- that IS the "established mechanism" r3 cited, and the snippet contradicts it.
- `Registry(FIXTURES)` is wrong: Registry appends `fixtures` itself (`registry.py:72`); pass repo ROOT like `conftest.py:20` does, or it raises "fixtures folder missing".
- `reg.list_packs(...)` does not exist (`registry.py:197-223` -- only `bank/pack/pack_path/style`); iterate `reg.packs` keys filtered on `key[0]`.

## SHOULD-FIX

- SF1: Chunk 7 bug-bible block omits `$env:PYTHONUTF8=1` and `-p no:cacheprovider` (CLAUDE.md mandates UTF-8; OTR/sibling blocks have it).
- SF2: extractor snippet hand-rolls `.packs.get(...)`; prefer `registry.pack(...)` (raises `UnknownIdError`, a `RegistryError` subclass -- docstring stays true, error text stays canonical).
- SF3: first Chunk 5 test takes `mirror_nodes` but never uses it; drop the param.

## Answers to the 7 checks

1. **Drift: none.** Every pass03 line ref verified at HEAD (table below).
2. **Collisions: none.** Sibling `tests/` has no `test_phase_a_byte_identity.py`/`test_extractor_coverage.py`; OTR's 427 test files contain no `*identity*` or `*extractor*` module; no `test_identity_check_outline.py`.
3. **No extra setup needed.** OTR `tests/conftest.py:31,:38` sets `CUDA_VISIBLE_DEVICES=''`/`OTR_TEST_MODE=1` via `setdefault` at collection; plain pytest from OTR root inherits it. Bonus: the identity assertion is sound -- a non-curated repo_id falls through to `_MODERN_BY_PHASE["outline"]` (`_otr_creative_prompt_router.py:96-100`), same object as `_otr_outline._SYSTEM_PROMPT`.
4. **Runnable as-is** (modulo SF1); commands match CLAUDE.md's canonical venv/cd-relative prescriptions. Venv/bug-bible paths sit outside my connected folders -- existence UNVERIFIABLE here, but doc-canonical.
5. **Residual invariant risk = MF-R4-1** -- worst failure mode is a coder "fixing" the AttributeError by importing the mirror, violating the never-import invariant.
6. **No circular import.** `contracts` imports only pydantic; `registry` imports `contracts`; `extractor` is a leaf importing both -- strict DAG; package `__init__` already imports both parents.
7. **Alias safe.** Sole `TEMPLATE_SEAMS` consumers are contracts-internal (`:185,:189,:232,:236,:351`); `__init__.py:9-41` does not export it; repo-wide grep finds no other importer.

## Grounding table

| Claim | Evidence | Status |
|---|---|---|
| contracts.py :25/:185/:232/:351, profile fields :270-279 | read/grep | CONFIRMED |
| profiles.py :60-65 line_grounding, stage() :86-95 | read | CONFIRMED |
| registry.py :70-72 root, :169-170 `.get(seam,"").strip()` | read | CONFIRMED |
| OTR `_otr_outline.py` :532/:1102/:1115/:1130; `_make_system` :1854-1857; `_otr_line_composer.py:1174` | grep/read | CONFIRMED |
| `mirror_nodes` = Path; mirror unimportable (`_otr_episode_budget` missing) | conftest:23-25; mirror glob | CONFIRMED |
| No `list_packs`; `Registry(FIXTURES)` double-fixtures bug | registry.py:66-79,197-223 | CONFIRMED |
| No test-name collisions either repo | globs | CONFIRMED |
| venv / bug-bible paths exist | outside connected folders | UNVERIFIABLE |
