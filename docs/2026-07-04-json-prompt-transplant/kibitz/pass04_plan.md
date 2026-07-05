# pass04_plan.md -- r4 (CONVERGENCE) synthesis

**Round:** r4 -- convergence / residual defects.
**Panel:** Codex + Fable (Agent) + Sonnet (Agent) + Claude anchor.
**Grounding:** OTR ``a7bdc42d`` (production code baseline; docs tip
``6f1c7ce2``); lab ``7df7c80``.

## Convergent verdict

**Codex: yes-with-fixes. Fable: GO-WITH-ONE-FIX. Sonnet: GO
(with one MUST-FIX).**

**Panel converges on GO.** All r4 MUST-FIX items are mechanical
one-line-scale corrections to chunk-level code snippets. No
architectural rework needed. Plan is code-ready modulo the fold-in
below.

## r4 MUST-FIX (all mechanical)

### MF-R4-A. Chunk 5 sibling pytest is not runnable as written

All 3 panelists caught this. Combined findings:

- **`mirror_nodes` fixture returns a `pathlib.Path`, not a module.**
  ``tests/conftest.py:23-25`` in the sibling. Accessing
  ``mirror_nodes._otr_outline._SYSTEM_PROMPT`` is AttributeError.
- **Importing the mirror is BARRED.** ``tests/test_compat_drift.py:3-5``
  header: "Never imports mirrored modules -- dependency-incomplete by
  design". The mirror's ``_otr_outline.py:37`` imports
  ``._otr_episode_budget`` which is absent from ``production_mirror/nodes/``
  (no ``__init__.py`` either).
- **``Registry(FIXTURES)`` is wrong.** ``registry.py:70-74`` appends
  ``/fixtures`` internally. Pass ROOT.
- **``Registry.list_packs`` DOES NOT EXIST.** Real accessors:
  ``bank()``, ``pack()``, ``pack_path()``, ``style()``, ``pipeline()``,
  ``resolve()`` (``registry.py:207-223``).

**Fix (folded into FINAL plan):** rewrite Chunk 5 sibling test to use
AST/text extraction over mirror files (that IS the ``compat.py``
established mechanism); pass ROOT to Registry; iterate
``[k for k in reg.packs if k[0] == "science_news"]``. See FINAL Chunk 5
below.

### MF-R4-B. Chunk 0 mirror refresh mechanism under-specified (Codex)

Codex MF2 grounded: no committed refresh tool exists under
``scripts/`` or ``tools/``; ``.gitignore`` mentions an ignored
``_rebuild_mirror.ps1``.

**Fix:** replace the vague "re-run whatever tool" phrasing with a
deterministic manual procedure:

1. For each file listed in ``PRODUCTION_MIRROR_MANIFEST.md``:
2. Copy from OTR ``a7bdc42d`` into ``production_mirror/`` at the same
   relative path.
3. Update the SHA256 + size entries in ``PRODUCTION_MIRROR_MANIFEST.md``.
4. Update baseline block (commit / date / title) to ``a7bdc42d``.
5. Run ``tests/test_compat_drift.py`` -- must pass.

### MF-R4-C. Chunk 1 SEAM_RUNTIME_VARIABLES instruction ambiguous (Codex)

pass03's Chunk 1 said "move variables to a new key
``style_pick_inventor_user_template`` if that seam is being added; else
leave attached". Codex + Fable: ambiguous. Codex CUT recommends
dropping the style_pick_inventor_user_template add from Phase A.

**Fix:** explicit instruction in FINAL: **leave ``SEAM_RUNTIME_VARIABLES``
UNCHANGED in Phase A. Do NOT add ``style_pick_inventor_user_template``
seam. That's Phase B**.

## r4 SHOULD-FIX (fold into FINAL)

- **SF-R4-a (Fable + Codex):** Chunk 7 Bug Bible command block omits
  ``$env:PYTHONUTF8="1"``. Add.
- **SF-R4-b (Codex):** OTR-side identity test may duplicate existing
  ``tests/test_creative_prompt_router.py:51-66,88-107``. **Cut**
  ``tests/test_identity_check_outline.py`` from Phase A -- existing test
  already pins modern resolver object identity + remote/unknown
  fallback. Codex CUT confirmed.
- **SF-R4-c (Fable):** Chunk 5 first test takes ``mirror_nodes`` param
  but doesn't use it; drop the parameter.
- **SF-R4-d (Fable):** Chunk 3 extractor snippet uses ``.packs.get()``;
  prefer ``registry.pack(...)`` which raises ``UnknownIdError`` (a
  ``RegistryError`` subclass). Docstring stays true; error text stays
  canonical.

## Verify-at-build checklist (Codex r4)

Per Codex r4 confirmed:

- OTR ``a7bdc42d`` is ancestor of current OTR HEAD ``6f1c7ce2``;
  diff is docs-only. Verify no ``nodes/*.py``, ``tests/*.py``,
  ``scripts/*.py``, ``workflows/*.json`` changes before coding starts.
- No test-name collisions.
- OTR plain pytest loads test-mode env via ``tests/conftest.py:31-38``.
- 4 new seams absent from all current sibling story packs -> extractor
  returns None for science packs.
- ``extractor.py`` importing ``.contracts`` + ``.registry`` is a strict
  DAG leaf. No cycle risk.
- ``TEMPLATE_SEAMS = ALL_TEMPLATE_SEAMS`` alias safe -- no external
  importer.

## Judgment log

**Accepted from all 3 (convergent):**

- Chunk 5 execution defects (Registry.list_packs missing +
  mirror_nodes/Path + Registry(FIXTURES) wrong) -- FOLDED into
  FINAL Chunk 5 rewrite.

**Accepted from Codex additionally:**

- Chunk 0 refresh procedure -- FOLDED.
- Chunk 1 SEAM_RUNTIME_VARIABLES ambiguity -- FOLDED, no seam add.
- Bug Bible PYTHONUTF8 -- FOLDED.
- OTR-side identity test is redundant -- CUT.

**Accepted from Fable additionally:**

- ``mirror_nodes`` fixture semantics ("never import the mirror") --
  explicitly documented in FINAL Chunk 5.
- Unused ``mirror_nodes`` param on first test -- CUT.
- ``registry.pack(...)`` preferred over ``.packs.get()`` -- FOLDED
  into FINAL Chunk 3.

**Accepted from Sonnet additionally:**

- All 7 verify-at-build items CONFIRMED via own grounding.

**No panel claim rejected.** No architectural rework required.

## Delta into FINAL

r4 is CONVERGED. Next step is to fold every MUST-FIX and SHOULD-FIX
into ``PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md`` -- the code-ready
deliverable. FINAL structure:

1. Phase A scope + operator gate ("Phase A ships before Phase B
   planning starts").
2. 7-chunk build plan with corrected Chunk 5 mechanics + Chunk 0
   procedure + Chunk 1 no-seam-add + Chunk 7 rollback discipline.
3. Verify-at-build checklist.
4. Kibitz arc audit trail (r1..r4 with pass0N artifacts + panel
   grounding).
5. Phase B stub reference.

## Reader's guide

- pass01_plan.md -- r1 arc convergence (identified 8 MUST-FIX).
- pass02_plan.md -- r2 coding plan (corrected r1 MF-C1/C3/C6; 7 chunks).
- pass03_plan.md -- r3 wiring (added missing plumbing; reordered
  chunks; dropped old Chunk 2 + Chunk 4 pack rewrite).
- pass04_plan.md -- r4 convergence (this file; 3 mechanical MUST-FIX
  + 4 SHOULD-FIX to fold).
- PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md -- next; code-ready.

## Grounding table (r4 pass)

| claim | source | status |
|---|---|---|
| Panel-convergent Chunk 5 defects | Codex + Fable + Sonnet r4 | CONFIRMED |
| ``mirror_nodes`` = Path fixture | ``tests/conftest.py:23-25`` sibling | CONFIRMED (Fable, verified independently) |
| ``test_compat_drift.py:3-5`` "never imports mirrored modules" | Fable | CONFIRMED |
| ``_otr_episode_budget`` absent from mirror | Fable glob | CONFIRMED |
| ``registry.py:70-74`` appends /fixtures internally | Fable + Codex | CONFIRMED |
| No ``list_packs`` on Registry | Codex + Sonnet | CONFIRMED |
| No committed mirror-refresh tool at 7df7c80 | Codex ``.gitignore`` ref | CONFIRMED |
| OTR ``a7bdc42d`` ancestor of ``6f1c7ce2``; diff docs-only | Codex + Sonnet | CONFIRMED |
| No test-name collisions | Fable + Sonnet | CONFIRMED |
| OTR conftest handles test env autouse | Sonnet + Fable | CONFIRMED |
| Extractor is strict DAG leaf | Codex + Fable | CONFIRMED |
| TEMPLATE_SEAMS alias safe (no external importer) | Fable + Codex + Sonnet grep | CONFIRMED |

## Final verdict

**r4 CONVERGED. Ready to produce PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md.**
