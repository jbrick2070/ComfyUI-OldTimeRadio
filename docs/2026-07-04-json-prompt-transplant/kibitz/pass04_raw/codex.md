VERDICT: yes-with-fixes — Chunk 5 contains runnable-test defects, and Chunk 0/Chunk 1 still leave incompatible implementation choices.

MUST-FIX BEFORE BUILD:
1. [Chunk 5] [CONFIRMED] The sibling byte-identity pytest is not runnable as written. `tests/conftest.py:23-25` in `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab` defines `mirror_nodes` as a `Path`, but the plan uses `mirror_nodes._otr_outline._SYSTEM_PROMPT` like an imported module. Also `Registry(FIXTURES)` is wrong because `registry.py:70-74` appends `/fixtures` internally, and `Registry` has no `list_packs()` method (`registry.py:207-223` only exposes `pack`/`pack_path`). Concrete fix: use `ROOT = Path(__file__).resolve().parents[1]`; construct `Registry(ROOT)`; iterate `for pack_key in reg.packs if pack_key[0] == "science_news"`; extract mirror constants by AST from `mirror_nodes / "_otr_outline.py"` and `mirror_nodes / "_otr_line_composer.py"` instead of importing the mirror.
2. [Chunk 0] [CONFIRMED] Mirror refresh is still under-specified. The plan says “re-running whatever tool built the mirror originally”; no committed refresh tool exists under `scripts/` or `tools/`, and `.gitignore` mentions ignored `_rebuild_mirror.ps1`. Concrete fix: replace this with a deterministic manual procedure: copy every file listed in `PRODUCTION_MIRROR_MANIFEST.md` from OTR `a7bdc42d`/current code baseline into matching `production_mirror/` paths, update commit/date/title plus SHA256/size entries, then run `tests/test_compat_drift.py` and the new prompt-constant snapshot test.
3. [Chunk 1] [CONFIRMED] `SEAM_RUNTIME_VARIABLES` instruction is ambiguous and can create scope creep. Current `contracts.py:25-42` has no `style_pick_inventor_user_template`, while `contracts.py:59-68` keeps inventor runtime vars on `style_pick_inventor`. The plan says move them “if that seam is being added; else leave,” but Phase A only adds four outline/composer seams. Concrete fix: delete this instruction from Phase A or explicitly state “leave `SEAM_RUNTIME_VARIABLES` unchanged; do not add `style_pick_inventor_user_template` in Phase A.”

SHOULD-FIX:
1. [Chunk 5] [CONFIRMED] The OTR identity test conflicts with its own fixture note. The snippet calls `resolve_creative_system_prompt("some_default_repo_id", "outline")`, while the text says pin `creative_repo_id=None`. Existing OTR tests already cover modern object identity with the real default Mistral-Nemo at `tests/test_creative_prompt_router.py:51-66`. Fix: either cut the new OTR-side test or make it explicitly reuse the existing default repo id and assertion shape.
2. [Chunk 7] [CONFIRMED] Bug Bible command omits `$env:PYTHONUTF8 = "1"` while OTR/CLAUDE rules require it for the test runner. Fix: set PYTHONUTF8 in the Bug Bible block too, so each command block is independently runnable.

OPTIONAL / NICE-TO-HAVE:
- Add `get_pack_prompt_or_none` to `src/upstream_story_lab/__init__.py` only if callers need package-level import. Current `__init__.py` does not export helper modules; leaving it module-local avoids extra API surface.

CUT THESE:
1. [Chunk 5] Cut `tests/test_identity_check_outline.py` if it only duplicates `tests/test_creative_prompt_router.py:51-66` and `tests/test_creative_prompt_router.py:88-107`. Existing tests already pin modern resolver object identity and remote/unknown fallback.
2. [Chunk 1] Cut all `style_pick_inventor_user_template` work from Phase A. It is not one of the four target seams and is not needed for passthrough extraction.

VERIFY-AT-BUILD checklist:
- [CONFIRMED] OTR code line drift: `a7bdc42d` is an ancestor of current OTR HEAD `6f1c7ce2`, and diff from `a7bdc42d..HEAD` is docs-only; still verify no changed files under `nodes/`, `tests/`, `scripts/`, `workflows/` before coding.
- [CONFIRMED] Test-name collision: sibling currently has no `test_phase_a_byte_identity.py` or `test_extractor_coverage.py`; OTR currently has no `test_identity_check_outline.py`.
- [CONFIRMED] OTR plain pytest loads test-mode env via `tests/conftest.py:31-38` (`CUDA_VISIBLE_DEVICES`, `OTR_TEST_MODE`).
- [CONFIRMED] New four seams are absent from all current sibling story packs; keep extractor tests asserting `None` for science packs.
- [CONFIRMED] Circular import risk is low if `extractor.py` imports only `.contracts` and `.registry`; `registry.py` imports `contracts.py`, and no existing module imports `extractor.py`.
- [CONFIRMED] Top-level `TEMPLATE_SEAMS` imports in sibling source are not present outside `contracts.py`; aliasing `TEMPLATE_SEAMS = ALL_TEMPLATE_SEAMS` is safe for current source consumers.
- verify: after implementation, run sibling full pytest, OTR full pytest, Bug Bible, AST parse touched `.py`, BOM/0-byte checks, and `HEAD == origin` for both repos.