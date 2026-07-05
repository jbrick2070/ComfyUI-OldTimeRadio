# r2 Anchor Review (Claude driver, pre-fan-out)

Target: kibitz-runs/2026-07-02-upstream-transplant-v2/r1/final.md
Focus: coding plan / implementability.

VERDICT: implementable tonight. Self-identified coding risks:

MUST-FIX:
1. Package import shape. v1 used src-layout + sys.path insertion from
   nodes.py (git 41c6512:nodes.py `_ensure_lab_importable`). Tests, CLI, and
   ComfyUI import must share ONE path-bootstrap helper or the three entry
   points drift. Plan should name it (src/upstream_story_lab/_bootstrap
   pattern or conftest.py + nodes.py both inserting src/).
2. Pydantic pin. Machine Python is 3.10.11; contracts use `X | None` and
   pydantic v2 ConfigDict - fine on 3.10 - but requirements.txt must pin
   `pydantic>=2,<3` and tests must not import pydantic v1 idioms. v1
   requirements.txt already did this; keep it.
3. JSON writing discipline: UTF-8 no BOM, LF, trailing newline, ASCII-safe
   (ensure_ascii=False but content authored ASCII); canonical-hash rule from
   r1 must live in ONE function used by both provenance and tests.
4. AST extraction fragility budget: the four extractors must fail with a
   pointed message ("re-pin against production_mirror") rather than a stack
   trace, or drift day-1 becomes a debugging session. Each extractor gets
   its own unit test against the CURRENT mirror bytes.
5. StoryPack schema must carry schema_version + status enum from v1
   (ready_fixture/experimental/not_implemented) so v1 recovered packs
   migrate by ADDING fields only - no silent semantic change.

SHOULD-FIX:
- registry cache: v1 nodes.py used a state-digest cache for choice lists;
  v2 registry should reuse the digest approach for IS_CHANGED but tests
  must be able to construct an uncached registry (no module-global-only).
- Windows/POSIX path handling in fixtures: PD manifests use forward-slash
  relative paths; keep validating with PurePosixPath semantics + the
  absolute/.. rejection from v1 nodes.py:155-166.
- simple_4 FakeLLM runner: define pass I/O as plain dataclasses, not
  pydantic, to keep the runner dependency-light and obviously test-only.

UNVERIFIABLE: none material.
