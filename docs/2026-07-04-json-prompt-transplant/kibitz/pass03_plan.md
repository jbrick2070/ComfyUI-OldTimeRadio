# pass03_plan.md -- r3 (WIRING) synthesis

**Round:** r3 -- wiring / integration / sequencing.
**Panel:** Codex + Fable (Agent) + Sonnet (Agent) + Claude anchor.
**Grounding:** OTR ``a7bdc42d`` (production code baseline; docs tip
``7655ead0``); lab ``7df7c80``.

## Convergent verdict

Codex: NO. Fable: NOT WIRE-READY / GO-WITH-FIXES. Sonnet: NO-GO.
**Consensus: pass02's 7-chunk plan is under-wired for the lab side --
`resolve_profile()`, `StoryPromptProfile`, and `TEMPLATE_SEAMS`
consumers all need code the plan never named. Chunk order also lets a
byte-identity-breaking commit ship before the snapshot harness exists.**

r3 rewrites the chunk plan with concrete lab-side hunks + snapshot-
first ordering.

## Corrections to r2 items

### R2-Chunk-1 (was under-scoped): TEMPLATE_SEAMS split breaks validators

Codex MF5 grounded: three consumers use ``TEMPLATE_SEAMS`` for schema
validation at ``contracts.py:185`` (SourceBankSpec),
``:232`` (StoryPack.prompt_stages), and ``:351`` (PassDecl).

**Fix:** define THREE constants:

- ``PRODUCTION_TEMPLATE_SEAMS = (...)`` -- 10 existing + 4 new = 14
- ``EXPERIMENTAL_PIPELINE_SEAMS = (...)`` -- 4 ``pass_*`` keys
- ``ALL_TEMPLATE_SEAMS = PRODUCTION_TEMPLATE_SEAMS + EXPERIMENTAL_PIPELINE_SEAMS`` (union tuple)

All three schema validators keep using ``ALL_TEMPLATE_SEAMS`` (name
unchanged in body semantically -- just a variable rename in the three
validator sites). Only the new extractor uses
``PRODUCTION_TEMPLATE_SEAMS``.

Add 4 NEW seam keys to ``PRODUCTION_TEMPLATE_SEAMS``:
``outline_macro_system``, ``outline_phase_system``,
``outline_beat_system``, ``line_composer_system``.

### R2-Chunk-1 + R2-Chunk-3 (missing plumbing): 4 new fields on StoryPromptProfile

Sonnet MF1 + Codex MF4 grounded:

- ``profiles.py:67-95`` calls ``stage(name)`` for the 10 existing keys
  only; no plumbing for the 4 new keys.
- ``StoryPromptProfile`` at ``contracts.py:270-279`` has fields for the
  10 existing keys only.

**Fix (r3 revised Chunk 1):**

Add 4 fields to ``StoryPromptProfile`` in ``contracts.py``:

```python
outline_macro_system_prompt: str | None = None
outline_phase_system_prompt: str | None = None
outline_beat_system_prompt: str | None = None
line_composer_system_prompt: str | None = None
```

And 4 corresponding ``stage(...)`` calls in ``profiles.py:86-95``.

**Design decision:** extractor uses ``registry.pack(...).prompt_stages[seam_key]``
directly to read pack contents (Codex MF4 option A), not
``StoryPromptProfile`` fields. Rationale: profile fields are for a
future consumer; the extractor's caller wants the RAW pack override,
None-if-absent, so it can decide passthrough. Profile fields still get
populated so tests + future consumers work. This is a two-path design:

- **Lab test path:** ``resolve_profile()`` -> ``StoryPromptProfile``
  with all fields.
- **Phase A extractor path:** ``get_pack_prompt_or_none(...)`` reads
  ``registry.pack(...).prompt_stages[seam_key]`` -- returns None if
  absent (skip empty-string).

Two consumers of the same pack JSON; two different failure semantics.
Documented in the extractor docstring.

### R2-Chunk-2 (correct-but-too-narrow): banks.json required_seams stays the way it is

r2 said Chunk 2 relaxes ``profiles.py:60-65`` line_grounding check.
Codex MF2 + Sonnet MF3 grounded:

- ``line_grounding`` cannot be omitted for science: profiles.py + contracts.py
  + `_otr_story_prompt_profile.py:31-48` all hard-require it as a str.
- ``line_grounding`` extraction was DEFERRED to Phase B in r2. So Phase A
  never touches line_grounding.
- Therefore: banks.json ``required_seams`` for science stays as-is (6
  items). No relaxation.

**Fix (r3 revised Chunk 2):** simplifies to a doc-only note. No code
change to profiles.py or banks.json. line_grounding stays populated in
the science pack (as it already is).

Actually r3 collapses this to: **there is no Chunk 2**. The r3 chunk
sequence starts Chunk 0 (mirror refresh) -> Chunk 1 (contracts.py +
profiles.py plumbing) -> Chunk 3 (extractor) -> Chunk 5 (harness) ->
Chunk 4 (extractor tests) -> Chunk 6 (docs) -> Chunk 7 (full
regression).

### R2-Chunk-3 (missing registry input): explicit signature

r2 signature was implicit-CWD. Codex MF3 grounded:

- ``Registry`` requires explicit root at ``registry.py:70-72``.
- Depending on CWD/import side effects is fragile.

**Corrected signature:**

```python
def get_pack_prompt_or_none(
    registry: Registry,
    source_bank_id: str,
    story_model_id: str,
    story_pipeline_id: str,
    seam_key: str,
) -> str | None:
    """Return pack.prompt_stages[seam_key] if present and non-empty; else None.

    None means "no override -- production caller uses its Python literal"
    (Phase A byte-identity passthrough).

    Raises:
      RegistryError on unknown (bank, model, pipeline) triple.
      RegistryError on unknown seam_key not in PRODUCTION_TEMPLATE_SEAMS.
    """
```

Tests construct ``Registry(ROOT)`` explicitly; no import-order or
CWD dependency.

### R2-Chunk-4 + R2-Chunk-5 (order swap): harness-first

Codex MF6 grounded: r2 Chunk 4 modifies packs BEFORE Chunk 5's
byte-identity harness exists -> a red push is possible without a guard.

**Fix (r3 revised sequence):** Chunk 5 (harness) lands BEFORE Chunk 4
(extractor tests). And realistically, Chunk 4 as originally scoped
(pack rewrite) is UNNECESSARY:

- Phase A does not modify production code -> production consumes its
  Python literals.
- The science pack already has 7 keys and doesn't need to change to
  prove passthrough.
- The 4 new seam keys are ABSENT from every pack at ``7df7c80`` --
  extractor returns None for them, production literal wins.
- Passthrough is proven by the extractor's None-return, not by a pack
  rewrite.

**Corrected Chunk 4:** extractor coverage tests. Table-driven test that
for each (bank, model, pipeline, seam_key) tuple:

- science_news + 4 new seams -> extractor returns None (absent from
  pack -> passthrough).
- media_archive + populated seam -> extractor returns the string.
- unknown seam_key -> RegistryError.

No pack file rewrites in Phase A. Sonnet MF3 corollary: production is
already byte-identical, no Chunk 4 pack changes needed to prove it.

### R2-Chunk-5 (mechanism specified): AST/mirror snapshot

Fable MF-W3 + Sonnet MF4 grounded: ``_make_system`` at
``_otr_outline.py:1854-1857`` is a CLOSURE inside ``generate_outline``;
capturing "live writer" output requires a stubbed
``structured_call`` / ``generate_fn`` with no fixture spec.

**Fix (r3 revised Chunk 5):** use the established mechanism at
``tests/test_compat_drift.py:27-52`` with the ``mirror_nodes`` fixture.

Snapshot the CONSTANTS (not the assembled output):

- ``mirror.nodes._otr_outline._SYSTEM_PROMPT`` (:532)
- ``mirror.nodes._otr_outline._MACRO_SYSTEM_PROMPT`` (:1102)
- ``mirror.nodes._otr_outline._PHASE_SYSTEM_PROMPT`` (:1115)
- ``mirror.nodes._otr_outline._BEAT_SYSTEM_PROMPT`` (:1130)
- ``mirror.nodes._otr_line_composer._SYSTEM_PROMPT`` (:1174)

Assertion: ``get_pack_prompt_or_none(registry, "science_news",
default_model, default_pipeline, seam_key) is None`` for each of the 4
new seams, plus the snapshot files match the constants.

**Separate identity pytest for MF-C1** goes in the **OTR-side** suite
(not sibling): ``resolve_creative_system_prompt(default, "outline") is
nodes._otr_outline._SYSTEM_PROMPT``. Sibling suite doesn't import OTR's
``nodes`` package; keeping the identity test in OTR-side avoids
cross-repo heavy imports (Fable MF-W3).

Snapshot fixture pins ``creative_repo_id=None`` explicitly (Sonnet MF4)
so ``_make_system`` is the identity function -- snapshots capture bare
constants, not overlay-augmented text.

### R2-Chunk-6 (same): anchor rewrites

No changes from r2. Chunk 6 rewrites sibling anchor sections that
carry Phase B machinery.

### R2-Chunk-7 (add rollback): explicit revert command

All 3 panelists (Fable + Sonnet + Codex) flagged rollback story
absent.

**Fix (added to Chunk 7):**

- On any post-push chunk RED: ``git revert <chunk_sha>``, push revert,
  rerun suite. Never force-push. Never reset ``main`` or
  ``v2.0-alpha``.
- Sibling branch discipline: verify ``HEAD == origin/main`` after each
  sibling push. OTR: ``HEAD == origin/v2.0-alpha``.
- Cross-repo chunk 5 (spans both repos): sibling push first, then OTR
  push (OTR test imports sibling nothing -- no cross-repo dep at test
  time, but pushing sibling first ensures a rollback of sibling
  precedes any OTR-side commit that would grep the mirrored file).

## Revised r3 chunk sequence

| # | Name | Repo | Byte-identity risk |
|---|---|---|---|
| 0 | Mirror refresh to ``a7bdc42d`` | sibling | none (no code change) |
| 1 | contracts.py + profiles.py plumbing + TEMPLATE_SEAMS split | sibling | none (add-only) |
| 2 | (dropped -- was line_grounding relaxation, no longer needed) | -- | -- |
| 3 | Extractor helper ``get_pack_prompt_or_none`` | sibling | none (new module) |
| 5 | Byte-identity snapshot harness | sibling + OTR pytest | pins byte-identity |
| 4 | Extractor coverage tests (table-driven) | sibling | none (test-only) |
| 6 | Anchor doc rewrites (docs-only) | sibling | none |
| 7 | Full regression + rollback discipline + push verification | OTR + sibling | none |

Note the reorder: Chunk 5 (harness) NOW runs before Chunk 4 (extractor
coverage tests), so byte-identity is provable before the extractor is
tested.

Old Chunk 2 (line_grounding relaxation) is dropped. line_grounding
stays as-is; extractor never touches it in Phase A.

## Per-chunk hunks (r3 detail)

### Chunk 0 -- mirror refresh

File: ``ComfyUI-OTR-UpstreamStoryLab\PRODUCTION_MIRROR_MANIFEST.md``.

Update baseline block at :10-16 to:

```text
commit a7bdc42dab...
date   2026-07-04 <hh:mm:ss> -0700
title  docs: July-4 sprint queue results -- Sprints 1+2 + Sprint 3 item 2 DONE...
```

Refresh files under ``production_mirror/`` to match OTR ``a7bdc42d``
by re-running whatever tool built the mirror originally (search sibling
for a mirror-refresh script; if none exists, document as a manual
copy-with-hash step in Chunk 0).

Test: ``test_compat_drift.py`` re-runs and passes (already exists per
Fable grounding).

### Chunk 1 -- contracts.py + profiles.py plumbing

File: ``src/upstream_story_lab/contracts.py``.

Before (at :25-42):

```python
TEMPLATE_SEAMS = (
    "outline_system", "pitch_room_system", ..., "pass_4_technical_ledger_audit",
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
# Back-compat alias (still-used name):
TEMPLATE_SEAMS = ALL_TEMPLATE_SEAMS
```

The `TEMPLATE_SEAMS` alias means SourceBankSpec (:185), StoryPack (:232),
PassDecl (:351) don't need to change -- they still validate against
``ALL_TEMPLATE_SEAMS`` under the old name.

Add 4 fields to StoryPromptProfile at :270-279:

```python
outline_macro_system_prompt: str | None = None
outline_phase_system_prompt: str | None = None
outline_beat_system_prompt: str | None = None
line_composer_system_prompt: str | None = None
```

File: ``src/upstream_story_lab/profiles.py``.

At :86-95 add:

```python
outline_macro_system_prompt=stage("outline_macro_system"),
outline_phase_system_prompt=stage("outline_phase_system"),
outline_beat_system_prompt=stage("outline_beat_system"),
line_composer_system_prompt=stage("line_composer_system"),
```

Fix ``SEAM_RUNTIME_VARIABLES`` at :59-68 per Sonnet SF-C1 / Codex SF2:
move ``n_required``, ``seed_sample_block``, ``article_excerpt`` from
``style_pick_inventor`` to a new key ``style_pick_inventor_user_template``
(if that seam is being added; else leave attached and note that it's
production consumer's problem to match the user-template).

Regression: full sibling suite pass; existing pack tests unchanged.

### Chunk 3 -- extractor

File: ``src/upstream_story_lab/extractor.py`` (new).

```python
from pathlib import Path
from .contracts import PRODUCTION_TEMPLATE_SEAMS
from .registry import Registry, RegistryError

def get_pack_prompt_or_none(
    registry: Registry,
    source_bank_id: str,
    story_model_id: str,
    story_pipeline_id: str,
    seam_key: str,
) -> str | None:
    """Return pack.prompt_stages[seam_key] if present and non-empty; else None."""
    if seam_key not in PRODUCTION_TEMPLATE_SEAMS:
        raise RegistryError(f"unknown production seam: {seam_key!r}")
    pack, _path = registry.packs.get(
        (source_bank_id, story_model_id, story_pipeline_id),
        (None, None),
    )
    if pack is None:
        raise RegistryError(
            f"no pack for {source_bank_id!r}/{story_model_id!r}/{story_pipeline_id!r}"
        )
    value = pack.prompt_stages.get(seam_key, "").strip()
    return value or None
```

Regression: unit tests explicit registry construction; no CWD
dependency.

### Chunk 5 -- byte-identity harness (BEFORE Chunk 4)

File: ``tests/test_phase_a_byte_identity.py`` (new, sibling).

```python
from pathlib import Path
from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry

FIXTURES = Path(__file__).parent.parent / "fixtures"

NEW_SEAMS = (
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "line_composer_system",
)

def test_phase_a_new_seams_absent_from_science(mirror_nodes):
    """New Phase A seams are absent from every pack under science_news -- 
    extractor returns None so production keeps its Python literal."""
    reg = Registry(FIXTURES)
    for pack_key in reg.list_packs("science_news"):
        for seam in NEW_SEAMS:
            assert get_pack_prompt_or_none(reg, *pack_key, seam) is None, (
                f"unexpected override for science pack {pack_key} seam {seam}"
            )

def test_snapshot_matches_mirror_constants(mirror_nodes):
    """The Python constants Phase A defers to are byte-stable against
    the production_mirror at a7bdc42d."""
    expected_outline = mirror_nodes._otr_outline._SYSTEM_PROMPT
    expected_macro = mirror_nodes._otr_outline._MACRO_SYSTEM_PROMPT
    # ... etc for phase, beat, line_composer
    # Compare against committed snapshot files under tests/snapshots/.
    snap_root = Path(__file__).parent / "snapshots" / "phase_a"
    for name, expected in (
        ("outline_system", expected_outline),
        ("outline_macro_system", expected_macro),
        # ...
    ):
        assert (snap_root / f"{name}.txt").read_text(encoding="utf-8") == expected
```

Separate OTR-side pytest at ``tests/test_identity_check_outline.py``:

```python
def test_outline_identity_check_preserved():
    from nodes import _otr_outline
    from nodes._otr_creative_prompt_router import resolve_creative_system_prompt
    result = resolve_creative_system_prompt("some_default_repo_id", "outline")
    # Modern default -> identity == module constant
    assert result is _otr_outline._SYSTEM_PROMPT
```

Fixture: ``creative_repo_id=None`` explicitly to pin default config.

Regression: both suites green. Snapshot files committed.

### Chunk 4 -- extractor coverage tests

File: ``tests/test_extractor_coverage.py`` (new, sibling).

Table-driven test over every (bank, model, pipeline, seam) tuple in the
loaded registry. For each:

- If seam is in the pack's prompt_stages with non-empty value: extractor
  returns that string.
- If seam is absent OR empty in the pack: extractor returns None.
- Unknown seam_key: RegistryError.

Regression: green.

### Chunk 6 -- anchor doc rewrites (docs-only)

File: ``ComfyUI-OTR-UpstreamStoryLab/docs/R1_ARCHITECTURE_AND_CODING_PLAN_V2.md``.

- Section 5 (compat mirrors): mark as PHASE B ONLY.
- Section 6 (visual policy): mark as PHASE B ONLY.
- Section 7b upgrades 2-5: mark as PHASE B ONLY.
- Section 9 adaptive cleanup: keep, note "already docs-only".

File: ``ComfyUI-OTR-UpstreamStoryLab/docs/JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md``.

- R2 file list: replace ``catalogs.py`` with ``registry.py`` /
  ``profiles.py`` / ``bridge.py``.

Doc-hygiene note (not a Phase A code change):
``_otr_creative_prompt_router.py:15-19`` docstring claims 4 phases; only
2 wired. Coder window should trim aspirational text in a follow-up.

### Chunk 7 -- full regression + rollback discipline

**Regression commands:**

OTR:
```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
$env:PYTHONUTF8 = "1"
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Bug Bible (separate repo per CLAUDE.md):
```powershell
cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\bug_bible_regression.py -q
```

Sibling:
```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab
$env:PYTHONUTF8 = "1"
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

**Push verification (per chunk):**

```powershell
$local = git rev-parse HEAD
$remote = git rev-parse "origin/$branch"  # v2.0-alpha for OTR, main for sibling
if ($local -ne $remote) { throw "HEAD != origin" }
# 0-byte check
Get-ChildItem -Recurse -File | Where-Object Length -eq 0 | ForEach-Object { throw "0-byte: $_" }
# BOM check on touched .md .py .json
# AST parse on touched .py
```

**Rollback discipline:**

If a chunk goes RED post-push:

```powershell
git revert --no-edit <chunk_sha>
git push origin <branch>
# Verify HEAD == origin; then re-open the chunk.
```

Never force-push. Never reset. Never `--no-verify`.

## Grounding table (r3 pass)

| claim | source | status |
|---|---|---|
| TEMPLATE_SEAMS 3 validator consumers at :185, :232, :351 | ``contracts.py`` (Codex) | CONFIRMED |
| ``StoryPromptProfile`` has no fields for 4 new seams | ``contracts.py:270-279`` (Sonnet + Codex) | CONFIRMED |
| ``profiles.py:67-95`` has no stage() calls for 4 new seams | Sonnet grounding | CONFIRMED |
| Registry requires explicit root at :70-72 | Codex | CONFIRMED |
| ``registry.py:169-170`` `.get(seam, "").strip()` -> omit == empty | Fable + Codex | CONFIRMED |
| ``profiles.py:60-65`` hard-error empty line_grounding | Fable + Codex | CONFIRMED |
| ``StoryPromptProfile.line_grounding_instruction`` required str at :266 | Fable + Codex | CONFIRMED |
| ``_otr_story_prompt_profile.py:31-48`` rejects empty profile | Codex | CONFIRMED |
| ``test_science_profile_leaves_style_picker_constants`` tests _otr_story_prompt_profile, not extractor | Sonnet | CONFIRMED |
| ``_make_system`` closure at :1854-1857 | Fable + Sonnet | CONFIRMED |
| ``_make_system`` identity when creative_repo_id is None | Sonnet | CONFIRMED |
| ``test_compat_drift.py:27-52`` established mirror snapshot pattern | Fable | CONFIRMED |
| ``a7bdc42d`` ancestor of OTR HEAD ``7655ead0`` | Sonnet git merge-base | CONFIRMED |
| Chunks 0-6 touch zero OTR production .py | Fable | CONFIRMED |
| IS_CHANGED moot in Phase A (lab has no ComfyUI node module) | Fable | CONFIRMED |

## Judgment log

**Accepted from all 3:** the four r3 convergent findings
(StoryPromptProfile plumbing missing; TEMPLATE_SEAMS validator break;
Chunk 4-before-5 order; Chunk 3 signature under-specified).

**Refined:** Chunk 4 completely redefined -- from "pack rewrite" to
"extractor coverage tests" (since production isn't consuming packs in
Phase A, pack rewrite is unnecessary). This drops MF-C6 as a Phase A
concern entirely.

**Rejected / deferred:**

- Chunk 2 (line_grounding relaxation): dropped from Phase A.
- ``style_pick_inventor_user_template`` new seam (Codex SF1): defer to
  Phase B unless the user-template case actually blocks Chunk 3
  (r4 decides).
- ``_otr_creative_prompt_router.py:15-19`` docstring cleanup (Sonnet
  + Codex SF): doc-hygiene follow-up, not Phase A code.

Zero panel claims outright rejected. Every panel-flagged concern
either folded, refined, or explicitly deferred.

## Delta to feed into r4

r4 input = ``pass03_plan.md`` (this file). r4 focus =
CONVERGENCE / residual defects. Every panelist confirms code-ready or
raises a NEW must-fix that MUST be resolvable in the same round.
Panel unchanged.

r4 explicit questions:

- Any file:line drift between pass03 and OTR HEAD (still ``a7bdc42d``)?
- Any test-name collision with existing sibling tests?
- Does the Chunk 5 identity pytest depend on OTR test-mode setup that
  needs a sibling equivalent?
- Are the regression commands runnable as-is?
- Any Phase A invariant still at risk?
