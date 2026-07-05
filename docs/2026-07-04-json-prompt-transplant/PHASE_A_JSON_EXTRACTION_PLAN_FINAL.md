# PHASE A -- JSON PROMPT EXTRACTION -- FINAL CODE-READY PLAN

**Sprint:** 2026-07-04 Sprint 3 Item 1 -- the "big LLM prompt update"
narrowed to Phase A: py-to-JSON assertion / prompt extraction.
**Kibitz arc:** r1..r4 with Codex + Fable + Sonnet triple-panel; all 3
say GO at r4.
**OTR baseline:** ``v2.0-alpha`` @ ``a7bdc42d`` (production code state
Phase A operates against; docs-only commits above this SHA do not
affect Phase A).
**Lab baseline:** ``main`` @ ``7df7c80`` (sibling
``ComfyUI-OTR-UpstreamStoryLab``).

## Phase B gate (top-of-doc reminder)

**Phase A must ship green before Phase B planning starts.** Green
means:

- All 7 chunks below committed + pushed.
- All 8 invariants preserved (byte-identical audio, ROW_KEYED merge,
  ledger schema, ``test_period_prompts.py`` asserts, critic/reroll
  seam, IS_CHANGED, env-flag gating, `l3-2026-05-14` ledger schema).
- At least one full-episode soak run with audio byte-identical against
  pre-Phase-A baseline.

Then and only then, Phase B planning begins in a new sprint with its
own kibitz + roundtable arc. Phase B = the full architectural
transplant described in the sibling anchor docs (4-axis
source_bank/story_model/story_pipeline/visual_style routing, bridge
artifact wiring in production, visual policy transplant, workflow JSON
widget adds). See ``PHASE_B_STUB.md``.

Phase A carves out ONLY the prompt-string-extraction subset. Nothing
in Phase A modifies production code or the workflow JSON.

## Scope statement

**Phase A does:**

- Add 4 new seam keys to the lab's ``PRODUCTION_TEMPLATE_SEAMS``
  (``outline_macro_system``, ``outline_phase_system``,
  ``outline_beat_system``, ``line_composer_system``) matching real OTR
  production constants at ``_MACRO_SYSTEM_PROMPT`` /
  ``_PHASE_SYSTEM_PROMPT`` / ``_BEAT_SYSTEM_PROMPT`` /
  ``_otr_line_composer._SYSTEM_PROMPT``.
- Add 4 corresponding ``str | None`` fields to
  ``StoryPromptProfile``.
- Add ``get_pack_prompt_or_none(registry, source_bank_id,
  story_model_id, story_pipeline_id, seam_key) -> str | None``
  extractor helper.
- Ship a byte-identity snapshot harness that pins the 5 production
  constants against the sibling's ``production_mirror/``.
- Refresh sibling ``production_mirror/`` to OTR ``a7bdc42d``.
- Rewrite sibling anchor doc sections that pull Phase B machinery
  in.

**Phase A does NOT:**

- Touch any OTR production ``.py`` file.
- Touch ``workflows/otr_scifi_16gb_full.json``.
- Add new widgets to any ComfyUI node.
- Add any new bank/model/pipeline/style content.
- Modify any story pack JSON (packs already at ``7df7c80`` state stand
  as-is).
- Extract ``line_grounding`` (deferred to Phase B; production rider is
  a conditional f-string incompatible with mechanical extraction).
- Extract the story-critic seam (out of the transplant scope; Fable
  step 6 doesn't include it).
- Modify ``profiles.py:60-65`` line_grounding hard-require check
  (unnecessary now that line_grounding stays as-is).

## Phase A invariants (must all hold at every intermediate commit)

- **I1: audio byte-identical.** ``test_audio_byte_identical`` and the
  fixed-seed regression episode both green pre- and post-Phase-A.
- **I2: prompt-string bytes identical.** For every (bank, seam) tuple
  used by production, the string production sees is the exact bytes it
  saw pre-Phase-A. Chunk 5 harness pins this.
- **I3: ROW_KEYED merge invariants** in
  ``OTR_LedgerScriptWriter.py`` unchanged (grep asserts).
- **I4: ledger schema `l3-2026-05-14`** unchanged.
- **I5: critic/reroll seam** untouched
  (``nodes/_otr_story_critic.py``, ``_otr_ledger_reviewer.py``).
- **I6: ``test_period_prompts.py`` assertions** unchanged. Not
  touched by Phase A because it asserts on
  ``OTR_PERIOD_SYSTEM_PROMPT`` at ``_otr_period_prompts.py:37``
  which stays a Python literal.
- **I7: ``IS_CHANGED`` and VRAM** unchanged. Extractor is lab-side; no
  ComfyUI node module in the sibling.
- **I8: env flags unchanged.** ``OTR_ENABLE_PITCH_ROOM``,
  ``OTR_GROUNDING_LEVER``, ``OTR_TEST_MODE``, ``CUDA_VISIBLE_DEVICES``
  gating stays intact.

## 7-chunk build plan

Each chunk is ONE commit + push. Regression at end of every chunk.
Chunks 0, 1, 3, 4, 5, 6 touch sibling repo only. Chunk 7 runs full
regression across both.

### Chunk 0: Refresh sibling ``production_mirror/`` to ``a7bdc42d``

**Repo:** sibling (``ComfyUI-OTR-UpstreamStoryLab``).
**Branch:** ``main``.

**Steps (manual deterministic procedure):**

1. Read the file list under
   ``PRODUCTION_MIRROR_MANIFEST.md`` (currently pins ``d48a9d76``).
2. For each mirrored file, copy from
   ``C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio``
   at HEAD ``a7bdc42d`` into ``production_mirror/`` at the same
   relative path.
3. Recompute SHA256 and size entries; update the manifest table.
4. Update the baseline block at ``PRODUCTION_MIRROR_MANIFEST.md:10-16``:

   ```text
   commit a7bdc42dab...
   date   2026-07-04 <hh:mm:ss> -0700
   title  docs: July-4 sprint queue results ...
   ```

**Test:** ``pytest -q -p no:cacheprovider tests/test_compat_drift.py``
green.

**Regression command block:**

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab
$env:PYTHONUTF8 = "1"
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

**Commit + push:**

```powershell
git add production_mirror/ PRODUCTION_MIRROR_MANIFEST.md
git commit -m "chore(mirror): refresh production_mirror/ to OTR a7bdc42d"
git push origin main
$local = git rev-parse HEAD ; $remote = git rev-parse origin/main
if ($local -ne $remote) { throw "HEAD != origin" }
```

### Chunk 1: ``contracts.py`` + ``profiles.py`` plumbing

**Repo:** sibling. **Branch:** ``main``.

**File: ``src/upstream_story_lab/contracts.py``**

Before (:25-42):

```python
TEMPLATE_SEAMS = (
    "outline_system", "pitch_room_system", "story_select_system",
    "dramatic_state_system", "line_grounding", "coda_system",
    "title_system", "style_pick_inventor", "style_pick_chooser",
    "style_pick_chooser_user_template",
    "pass_1_creative_story", "pass_2_creative_ledger_fill",
    "pass_3_technical_schema_cleanup", "pass_4_technical_ledger_audit",
)
```

After:

```python
PRODUCTION_TEMPLATE_SEAMS = (
    "outline_system",
    "pitch_room_system",
    "story_select_system",
    "dramatic_state_system",
    "line_grounding",
    "coda_system",
    "title_system",
    "style_pick_inventor",
    "style_pick_chooser",
    "style_pick_chooser_user_template",
    # Phase A additions (2026-07-04):
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "line_composer_system",
)
EXPERIMENTAL_PIPELINE_SEAMS = (
    "pass_1_creative_story",
    "pass_2_creative_ledger_fill",
    "pass_3_technical_schema_cleanup",
    "pass_4_technical_ledger_audit",
)
ALL_TEMPLATE_SEAMS = PRODUCTION_TEMPLATE_SEAMS + EXPERIMENTAL_PIPELINE_SEAMS
# Back-compat alias -- validators at :185, :232, :351 keep using this name.
TEMPLATE_SEAMS = ALL_TEMPLATE_SEAMS
```

Add 4 fields to ``StoryPromptProfile`` at :270-279:

```python
outline_macro_system_prompt: str | None = None
outline_phase_system_prompt: str | None = None
outline_beat_system_prompt: str | None = None
line_composer_system_prompt: str | None = None
```

**Leave ``SEAM_RUNTIME_VARIABLES`` UNCHANGED.** Do NOT add
``style_pick_inventor_user_template`` seam. That is a Phase B concern.

**File: ``src/upstream_story_lab/profiles.py``**

Add 4 corresponding ``stage(...)`` calls at :86-95:

```python
outline_macro_system_prompt=stage("outline_macro_system"),
outline_phase_system_prompt=stage("outline_phase_system"),
outline_beat_system_prompt=stage("outline_beat_system"),
line_composer_system_prompt=stage("line_composer_system"),
```

**Test:** full sibling suite green. Existing pack tests unchanged. New
packs (if any test tries to construct them) accept the new seam keys.

**Commit + push per push discipline block above.**

### Chunk 3: Extractor helper ``get_pack_prompt_or_none``

**Repo:** sibling. **Branch:** ``main``.

**File: ``src/upstream_story_lab/extractor.py`` (new)**

```python
"""Phase A prompt extractor -- returns pack overrides or None (passthrough).

None means "no override -- production caller uses its Python literal".
Reserved solely for intentional empty override. All structural failures
(unknown bank/model/pipeline, unknown seam) raise RegistryError.
"""
from __future__ import annotations

from .contracts import PRODUCTION_TEMPLATE_SEAMS
from .registry import Registry, RegistryError, UnknownIdError


def get_pack_prompt_or_none(
    registry: Registry,
    source_bank_id: str,
    story_model_id: str,
    story_pipeline_id: str,
    seam_key: str,
) -> str | None:
    """Return pack.prompt_stages[seam_key] if present and non-empty; else None."""
    if seam_key not in PRODUCTION_TEMPLATE_SEAMS:
        raise RegistryError(
            f"unknown Phase A production seam: {seam_key!r}"
        )
    # registry.pack raises UnknownIdError (a RegistryError subclass) on
    # unknown triple -- canonical error text; don't hand-roll .packs.get().
    pack = registry.pack(source_bank_id, story_model_id, story_pipeline_id)
    value = pack.prompt_stages.get(seam_key, "").strip()
    return value or None
```

**Test file: ``tests/test_extractor_helper.py``**

```python
from pathlib import Path

from upstream_story_lab.contracts import PRODUCTION_TEMPLATE_SEAMS
from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry, RegistryError

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_seam_raises():
    reg = Registry(ROOT)
    packs = list(reg.packs)
    bank, model, pipeline = packs[0]
    try:
        get_pack_prompt_or_none(reg, bank, model, pipeline, "not_a_seam")
    except RegistryError:
        return
    raise AssertionError("expected RegistryError for unknown seam")


def test_unknown_bank_raises():
    reg = Registry(ROOT)
    try:
        get_pack_prompt_or_none(reg, "not_a_bank", "not_a_model",
                                "not_a_pipeline",
                                PRODUCTION_TEMPLATE_SEAMS[0])
    except RegistryError:
        return
    raise AssertionError("expected RegistryError for unknown triple")
```

### Chunk 5: Byte-identity snapshot harness (BEFORE Chunk 4)

**Repo:** sibling. **Branch:** ``main``.

**Key mechanics (per r4 fix):**

- Use AST/text extraction over mirror files. **NEVER import the
  mirror** (``test_compat_drift.py:3-5``:
  "dependency-incomplete by design").
- ``Registry`` takes ROOT (repo root), NOT ``ROOT / "fixtures"`` --
  registry appends ``/fixtures`` itself at :70-74.
- ``Registry.list_packs`` does not exist -- iterate ``reg.packs``
  filtered by bank id.

**File: ``tests/test_phase_a_byte_identity.py`` (new)**

```python
"""Phase A byte-identity harness -- pins the 5 production constants and
the extractor's None-return semantics against the sibling mirror."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
SNAP_ROOT = ROOT / "tests" / "snapshots" / "phase_a"

NEW_SEAMS = (
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "line_composer_system",
)

MIRROR_CONSTANTS = (
    ("_otr_outline.py", "_SYSTEM_PROMPT", "outline_system"),
    ("_otr_outline.py", "_MACRO_SYSTEM_PROMPT", "outline_macro_system"),
    ("_otr_outline.py", "_PHASE_SYSTEM_PROMPT", "outline_phase_system"),
    ("_otr_outline.py", "_BEAT_SYSTEM_PROMPT", "outline_beat_system"),
    ("_otr_line_composer.py", "_SYSTEM_PROMPT", "line_composer_system"),
)


def _extract_constant(mirror_root: Path, filename: str, name: str) -> str:
    """AST-extract a module-level string constant from a mirror file.
    Never imports the module (mirror is dependency-incomplete)."""
    src = (mirror_root / "nodes" / filename).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        return node.value.value
    raise AssertionError(f"{name} not found as module-level str in {filename}")


def test_phase_a_new_seams_absent_from_science():
    """Phase A new seams are absent from every science_news pack ->
    extractor returns None -> production keeps its Python literal."""
    reg = Registry(ROOT)
    science_keys = [k for k in reg.packs if k[0] == "science_news"]
    assert science_keys, "expected at least one science_news pack"
    for pack_key in science_keys:
        for seam in NEW_SEAMS:
            assert get_pack_prompt_or_none(reg, *pack_key, seam) is None, (
                f"unexpected override for {pack_key} seam {seam}"
            )


def test_mirror_constants_match_snapshot(mirror_nodes: Path):
    """The 5 Python constants Phase A defers to are byte-stable against
    the production_mirror at a7bdc42d."""
    SNAP_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, const_name, seam_key in MIRROR_CONSTANTS:
        current = _extract_constant(mirror_nodes, filename, const_name)
        snap = SNAP_ROOT / f"{seam_key}.txt"
        if not snap.exists():
            # First run: commit the snapshot alongside the test.
            snap.write_text(current, encoding="utf-8")
            pytest.skip(f"snapshot created: {snap.name}; re-run to assert")
        assert snap.read_text(encoding="utf-8") == current, (
            f"drift: {const_name} in mirror != snapshot {snap.name}"
        )
```

**No OTR-side identity test.** Existing
``tests/test_creative_prompt_router.py:51-66`` and ``:88-107`` already
pin modern resolver object identity and remote/unknown fallback. Adding a
duplicate at OTR-side is redundant per Codex CUT + Fable.

### Chunk 4: Extractor coverage tests (AFTER Chunk 5)

**Repo:** sibling. **Branch:** ``main``.

**File: ``tests/test_extractor_coverage.py`` (new)**

```python
"""Table-driven coverage: for every (bank, model, pipeline, seam) tuple in
the loaded registry, extractor returns str for populated seams, None for
absent/empty."""
from pathlib import Path

from upstream_story_lab.contracts import PRODUCTION_TEMPLATE_SEAMS
from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def test_extractor_coverage_all_packs():
    reg = Registry(ROOT)
    for pack_key in reg.packs:
        pack, _path = reg.packs[pack_key]
        for seam in PRODUCTION_TEMPLATE_SEAMS:
            got = get_pack_prompt_or_none(reg, *pack_key, seam)
            raw = pack.prompt_stages.get(seam, "").strip()
            expected = raw or None
            assert got == expected, (
                f"{pack_key} seam {seam}: got={got!r} expected={expected!r}"
            )
```

### Chunk 6: Anchor doc rewrites (docs-only)

**Repo:** sibling. **Branch:** ``main``.

**File: ``docs/R1_ARCHITECTURE_AND_CODING_PLAN_V2.md``**

Add a top-of-doc callout:

> **Phase A subset (2026-07-04):** the Phase A extraction hardened via
> the OTR sibling's kibitz arc r1..r4 covers sections 3-4 (axes +
> seams) + section 5 pins the 4 additional seams (macro/phase/beat +
> line_composer) as production constants. Sections 5 (compat mirrors),
> 6 (visual policy), 7b upgrades 2-5, 8 (bridge artifact emit), 10
> (workflow JSON edit) are Phase B ONLY.

**File: ``docs/JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md``**

At the R2 Coding Plan section: replace ``catalogs.py`` (deleted at
7df7c80) with ``registry.py``, ``profiles.py``, ``bridge.py``.

**Doc-hygiene follow-up (NOT a Phase A code change):**
``_otr_creative_prompt_router.py:15-19`` docstring claims 4 phases;
only 2 wired. Coder window should trim aspirational text in a
separate follow-up.

### Chunk 7: Full regression + Bug Bible + rollback

**Repos:** both.

**Regression commands (canonical, PYTHONUTF8 everywhere):**

Sibling:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab
$env:PYTHONUTF8 = "1"
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

OTR:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
$env:PYTHONUTF8 = "1"
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Bug Bible (separate repo per CLAUDE.md; PYTHONUTF8 explicit):

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
$env:PYTHONUTF8 = "1"
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\bug_bible_regression.py -q -p no:cacheprovider
```

Post-push verification:

```powershell
# Both repos:
$local = git rev-parse HEAD
$remote = git rev-parse "origin/$branch"  # v2.0-alpha for OTR, main for sibling
if ($local -ne $remote) { throw "HEAD != origin" }
# 0-byte / BOM / AST checks on touched files (see CLAUDE.md).
```

**Rollback discipline (any post-push RED):**

```powershell
git revert --no-edit <chunk_sha>
git push origin <branch>
$local = git rev-parse HEAD; $remote = git rev-parse "origin/$branch"
if ($local -ne $remote) { throw "revert push failed" }
# Then re-open the chunk with the fix.
```

Never force-push. Never reset ``main`` or ``v2.0-alpha``. Never
``--no-verify``. If a chunk fails 3 revert attempts, halt and escalate
via the operator's autonomy directive.

## Verify-at-build checklist (Codex r4 confirmed)

Coder window MUST verify each before starting:

- ``git log a7bdc42d..HEAD -- nodes/ tests/ scripts/ workflows/`` in
  OTR returns nothing (Phase A operates against pure code state at
  ``a7bdc42d``).
- Sibling has no ``test_phase_a_byte_identity.py`` or
  ``test_extractor_coverage.py`` at start of Chunk 3.
- Extractor's DAG is leaf-safe (``.contracts`` + ``.registry`` only).
- ``TEMPLATE_SEAMS`` alias survives round-trip (no external importer).
- OTR ``tests/conftest.py:31-38`` handles test env autouse (no
  per-test setup needed).
- Bug Bible file exists at
  ``C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\tests\bug_bible_regression.py``.

## Chunk ordering (locked)

```
Chunk 0 -> Chunk 1 -> Chunk 3 -> Chunk 5 -> Chunk 4 -> Chunk 6 -> Chunk 7
```

Rationale:

- Chunk 0 first: mirror refresh gives Chunk 5's AST extractor a truth
  source.
- Chunk 1 second: contracts.py + profiles.py plumbing must exist before
  Chunk 3 references ``PRODUCTION_TEMPLATE_SEAMS``.
- Chunk 3 third: extractor exists before Chunk 5 tests it.
- Chunk 5 fourth (BEFORE Chunk 4): byte-identity harness pins the
  invariant before extractor coverage tests run against arbitrary
  packs.
- Chunk 4 fifth: coverage tests validate extractor across all packs.
- Chunk 6 sixth: docs-only cleanup.
- Chunk 7 last: full regression proves everything green.

Chunk 2 was dropped in r3 -- line_grounding relaxation is unnecessary
once line_grounding extraction is deferred to Phase B.

## Kibitz arc audit trail

- **r1** (arc): ``kibitz/pass01_plan.md`` + ``pass01_judgment.md`` +
  ``pass01_raw/{codex,fable,sonnet,claude_anchor}.md``. 8 MUST-FIX
  identified.
- **r2** (coding): ``kibitz/pass02_plan.md`` + ``pass02_judgment.md``
  + ``pass02_raw/*.md``. r1 MF-C1/C3/C6 corrected; 7-chunk plan
  drafted.
- **r3** (wiring): ``kibitz/pass03_plan.md`` + ``pass03_judgment.md``
  + ``pass03_raw/*.md``. StoryPromptProfile plumbing added;
  Chunk 5 moved before Chunk 4; old Chunk 2 + old Chunk 4 dropped.
- **r4** (convergence): ``kibitz/pass04_plan.md`` +
  ``pass04_judgment.md`` + ``pass04_raw/*.md``. Chunk 5 execution
  defects fixed; Chunk 0 procedure spelled out; Chunk 1
  SEAM_RUNTIME_VARIABLES ambiguity resolved; OTR-side identity test
  cut as redundant.

All artifacts committed + pushed to ``v2.0-alpha`` under
``docs/2026-07-04-json-prompt-transplant/kibitz/``.

## Panel signatures

Both agents at r4 said GO:

- **Codex r4:** "yes-with-fixes" -- all fixes mechanical, folded above.
- **Fable r4:** "GO-WITH-ONE-FIX" -- Chunk 5 triple-broken snippet
  fixed above.
- **Sonnet r4:** "GO (with one MUST-FIX, mechanical,
  same-round-resolvable)" -- ``list_packs`` fixed above.

Zero rejections, zero unresolved concerns, zero architectural rework.

## Final note

**This document is the code-ready plan.** A coder window can execute
Chunk 0 -> 7 sequentially, running the regression command block at end
of each chunk, committing + pushing per green chunk. No further
kibitz needed for Phase A.

Phase B kicks off in a new sprint after Phase A soaks green. See
``PHASE_B_STUB.md``.
