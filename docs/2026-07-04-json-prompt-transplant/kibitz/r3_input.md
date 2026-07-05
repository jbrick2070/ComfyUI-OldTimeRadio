# KIBITZ R3 INPUT -- PHASE A JSON PROMPT EXTRACTION (wiring)

## Round focus: r3 -- WIRING / INTEGRATION / SEQUENCING

r2 delivered a 7-chunk build plan (Chunk 0 mirror refresh -> Chunk 7
full regression). r3 must judge whether the WIRING is safe:

- Does each chunk order preserve byte-identical audio at every
  intermediate commit? An audio regression can only happen at a chunk
  boundary if the wiring is off.
- Does sibling-repo branch discipline work? Sibling `ComfyUI-OTR-UpstreamStoryLab`
  is on `main` @ `7df7c80`; OTR is on `v2.0-alpha` @ `a7bdc42d`
  (docs tip `7655ead0`). Coder window will bounce between repos --
  what's the branch/PR discipline?
- Does the extractor's esolve_profile() wrap survive ComfyUI
  `IS_CHANGED` / import order / test-time mocking?
- Does Chunk 1's schema change break any existing lab test?
- Does Chunk 4's science-pack rewrite break the sibling's own byte-
  identity test (`test_science_profile_leaves_style_picker_constants`)?
- Does Chunk 5's snapshot capture happen against
  a live writer OR against constants? If against a live writer, what's
  the fixture/env setup?
- What's the rollback story if a chunk goes RED post-push?

Panel: Codex + Fable + Sonnet. Antigravity dropped (stalled).

For each of the 7 chunks (0..6, with 7 = full-suite), r3 must judge:

- Are the file paths + specific hunks named?
- Are the tests specified with real assertion bodies (not "add a
  test")?
- Is the commit/push discipline byte-identical-safe per-chunk?

## Repos to grep

- Production OTR: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio` at `a7bdc42d` (docs tip `7655ead0`).
- Lab: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab` at `main` `7df7c80`.

Return your review in VERDICT / MUST-FIX / SHOULD-FIX format with a
grounding table. Every claim CONFIRMED / MISREAD / UNVERIFIABLE.

---

# CURRENT PLAN STATE (input to r3 review)

Source: `docs\2026-07-04-json-prompt-transplant\kibitz\pass02_plan.md`

# pass02_plan.md -- r2 (CODING PLAN) synthesis

**Round:** r2 -- coding plan / implementability.
**Panel:** Codex (`codex exec`) + Fable (Agent) + Sonnet (Agent) +
Claude anchor synthesis.
**Grounding:** OTR ``a7bdc42d`` (production code baseline; docs tip
``c98a67ab``); lab ``7df7c80``.

## Convergent verdict

Codex: NO. Fable: NOT code-ready / GO-WITH-FIXES. Sonnet: NO-GO.
**Consensus: pass01 had factual errors in MF-C1 / MF-C3 / MF-C6 that r2
MUST correct before elaborating chunks.**

r2 rewrites those three items from the ground up, adds a 7-chunk build
plan, and pins the extractor helper signature to a form the lab's data
model actually supports.

## Corrections to r1 items

### MF-C1 (CORRECTED): identity check is OUTLINE-only

r1 called this "the load-bearing find". Grounded correction from Sonnet
+ Fable:

- Repo-wide grep for ``is _SYSTEM_PROMPT|is _MODERN`` returns **exactly
  one hit**: ``nodes/_otr_outline.py:1847``. Sonnet confirmed.
- ``nodes/_otr_line_composer.py:2060-2061`` does ``system =
  _SYSTEM_PROMPT`` (direct assign gated on ``creative_repo_id is
  None``) -- NOT an identity check. Codex + Sonnet + Fable all confirm.
- The router (``_otr_creative_prompt_router.py:43-46,61-64``) imports
  BOTH constants as module-level singletons; ``_MODERN_BY_PHASE`` maps
  both phases -- but only the outline consumer checks identity.

**Rule (corrected):** MF-C1 applies to `outline_system` alone. Any
Phase A action on ``_otr_outline.py:532`` must keep ``_SYSTEM_PROMPT``
as a module-level singleton bound to the string the router expects.
``line_composer_system`` extraction (MF-C2) is separable -- its consumer
does direct-assign, no identity guarantee to preserve.

### MF-C3 (CORRECTED): real Python constant names + seam count

r1 invented seam names ``outline_macro_system`` /
``outline_phase_system`` / ``outline_beat_system``. Grounded correction:

- Real Python constants at ``_otr_outline.py`` are ``_MACRO_SYSTEM_PROMPT``
  (:1102), ``_PHASE_SYSTEM_PROMPT`` (:1115), ``_BEAT_SYSTEM_PROMPT``
  (:1130). Sonnet + Fable + Codex all cite the real names.
- Consumer sites use these constants at :1868 / :1996 / :2101 (Fable).
- Real seam count in lab: ``TEMPLATE_SEAMS`` has 14 entries (10
  production + 4 experimental). Codex: split into
  ``PRODUCTION_TEMPLATE_SEAMS`` (10) + ``EXPERIMENTAL_PIPELINE_SEAMS``
  (4).
- Phase A adds 4 new keys to ``PRODUCTION_TEMPLATE_SEAMS``:
  ``outline_macro_system``, ``outline_phase_system``,
  ``outline_beat_system``, ``line_composer_system``. Total: 14
  production seam keys + 4 experimental (unchanged).

Seam-name-to-constant map for the 4 new keys (Phase A production
scope):

| new lab seam key | production constant | file:line |
|---|---|---|
| ``outline_macro_system`` | ``_MACRO_SYSTEM_PROMPT`` | ``nodes/_otr_outline.py:1102`` |
| ``outline_phase_system`` | ``_PHASE_SYSTEM_PROMPT`` | ``nodes/_otr_outline.py:1115`` |
| ``outline_beat_system`` | ``_BEAT_SYSTEM_PROMPT`` | ``nodes/_otr_outline.py:1130`` |
| ``line_composer_system`` | ``_otr_line_composer._SYSTEM_PROMPT`` | ``nodes/_otr_line_composer.py:1174`` |

Consumer-side (macro/phase/beat) uses NO identity check -- extraction-
safe (Fable SF-R2-1). Only ``outline_system`` (the legacy :532 seam
per Fable SF-R2-2) carries the MF-C1 identity guarantee.

### MF-C4 (CORRECTED): signature is 4-tuple keyed

r1 signature was ``get_pack_prompt_or_none(bank_id, seam_key)``.
Codex MF3 grounded correction: packs are keyed by ``(source_bank_id,
story_model_id, story_pipeline_id)`` at ``registry.py:117-124``;
``resolve_profile()`` requires all three ids at ``profiles.py:31-35``;
multiple packs per bank (e.g. ``media_archive/broadcast_history_comedy.json``,
``cinematic_humorous.json``).

**Corrected signature (Phase A extractor helper):**

```python
def get_pack_prompt_or_none(
    source_bank_id: str,
    story_model_id: str,
    story_pipeline_id: str,
    seam_key: str,
) -> str | None:
    """Return the pack's seam string if present and non-empty; else None.

    None -> production caller uses its own Python literal (byte-identical
    passthrough). This is the empty-science-override path.

    Raises RegistryError on:
      - unknown bank / model / pipeline triple
      - unknown seam_key not in PRODUCTION_TEMPLATE_SEAMS
      - malformed JSON on load
      - required seam empty (per banks.json.required_seams)

    None is reserved SOLELY for intentional empty override.
    """
```

Wraps ``resolve_profile()``; does not duplicate resolution logic.
(Codex SF3 + Sonnet SF-C2.)

### MF-C6 (CORRECTED): empty-string is ONLY safe outside required_seams

Grounded correction from Sonnet + Fable + Codex:

- ``fixtures/banks.json:24-31`` shows ``science_news.required_seams``
  is 6 items: ``outline_system, pitch_room_system,
  dramatic_state_system, line_grounding, coda_system, title_system``.
- ``registry.py:167-176`` raises ``RegistryError`` if any required seam
  resolves falsy (`.strip()` on `""` is falsy).
- ``profiles.py:60-65`` separately raises on empty ``line_grounding``.
- ``science_news_default.json`` has 7 keys total, **omits**
  ``style_pick_*`` keys (absence, not empty-string).

**Corrected rule:** for each pack (bank_id, model_id, pipeline_id):

- Seams in ``banks[bank_id].required_seams`` MUST be either populated
  with non-empty content OR OMITTED from the pack JSON. An
  empty-string override on a required seam fails LOUD.
- Seams NOT in required_seams MAY be empty-string overridden (production
  Python literal wins).

**Additional r2 finding (Fable MF-R2-2):** ``line_grounding`` in
production is a **conditional two-variant f-string** at
``_otr_line_composer.py:1621-1636`` with ``{req.conflict_object}``
interpolation. Neither empty-override nor literal-move works cleanly
for it in Phase A. r2 decision: **``line_grounding`` extraction is
DEFERRED to Phase B**. Phase A leaves it in Python untouched.

Corollary: ``line_grounding`` is dropped from Phase A production seam
adds; Phase A adds 3 new seams to ``PRODUCTION_TEMPLATE_SEAMS``, not 4
(``line_composer_system`` still added; ``line_grounding`` remains lab-
only for now).

**Revised total (final for Phase A):**
- ``PRODUCTION_TEMPLATE_SEAMS`` before r2: 10 real prod seams.
- Phase A adds: 3 new keys (macro/phase/beat) + ``line_composer_system``
  = 4.
- ``PRODUCTION_TEMPLATE_SEAMS`` after Phase A: 14 keys.
- ``EXPERIMENTAL_PIPELINE_SEAMS``: 4 keys unchanged, docs-only.
- ``line_grounding`` stays in TEMPLATE_SEAMS but production consumer
  keeps its f-string literal. Chunk 5's byte-identity test explicitly
  proves ``get_pack_prompt_or_none(..., "line_grounding") is None`` for
  the science lane.

### MF-C5 (CORRECTED): manifest exists, pins d48a9d76 (2 days behind)

Codex found ``PRODUCTION_MIRROR_MANIFEST.md`` at
``ComfyUI-OTR-UpstreamStoryLab\PRODUCTION_MIRROR_MANIFEST.md:10-16``
pinning:

```text
commit d48a9d76f39db6db16c758d9b2c1c22a9af38d3f
date   2026-07-02 00:22:46 -0700
title  talking-radio B: LTX-only mouth-forward radio-face still...
```

OTR is now at ``a7bdc42d`` (2 days later; sprints 1/2/3-item-2 landed
in between). Drift is real.

**r2 decision:** Chunk 0 refreshes mirror to ``a7bdc42d``. Details in
chunk plan below.

## Anchor doc rewrites (MF-C7 applied for real)

Codex MF7 flagged that MF-C7 CUTs were listed but not applied in the
anchor sections. r2 explicit action: rewrite anchor 1 sections
delivered in the sibling repo to CUT (not just mark) the following for
Phase A:

- Section 5 (compat mirrors) -> Phase B.
- Section 6 (visual policy) -> Phase B.
- Section 7b upgrades 2-5 -> Phase B.
- Section 9 adaptive cleanup -> already docs-only in
  ``fixtures/pipelines.json:39-42``; explicit "not Phase A" in the doc.
- Anchor 2 R2 file list still cites ``catalogs.py`` (deleted at 7df7c80);
  replace with ``registry.py``, ``profiles.py``, ``bridge.py``.
- Router docstring at ``_otr_creative_prompt_router.py:15-19`` claims
  4 phases; only 2 wired. Doc-hygiene fix; NOT Phase A code.

These rewrites happen in Chunk 6 (docs) alongside the pass02 output.
They do NOT touch production code.

## 7-chunk coding plan (Phase A)

Each chunk is one commit + push to ``v2.0-alpha``. Regression
harness runs at end of every chunk. Byte-identical audio green
non-negotiable.

### Chunk 0: Refresh production mirror to ``a7bdc42d``

- Repo: sibling ``ComfyUI-OTR-UpstreamStoryLab``.
- Action: refresh ``production_mirror/nodes/*.py`` etc. from
  OTR ``a7bdc42d``; update ``PRODUCTION_MIRROR_MANIFEST.md`` baseline +
  file hashes.
- No production code change.
- Test: manifest checksum table matches actual mirror files.
- Regression: ``pytest -q -p no:cacheprovider tests`` in sibling repo.

### Chunk 1: ``contracts.py`` schema hygiene + 4 new seams

- Repo: sibling.
- Actions in ``src/upstream_story_lab/contracts.py``:
  - Split ``TEMPLATE_SEAMS`` into ``PRODUCTION_TEMPLATE_SEAMS`` (10 +
    the 4 new keys = 14) + ``EXPERIMENTAL_PIPELINE_SEAMS`` (4 pass_*
    keys, docs-only).
  - Add 4 new seam keys: ``outline_macro_system``,
    ``outline_phase_system``, ``outline_beat_system``,
    ``line_composer_system``.
  - Fix ``SEAM_RUNTIME_VARIABLES["style_pick_inventor"]`` to bind
    variables to the correct user-template seam
    (``style_pick_inventor_user_template`` if added) not the system
    seam (Sonnet SF-C1 + Codex SF2). Grounded at
    ``_otr_style_picker.py:296,301,329,334``.
- Update ``StoryPack.prompt_stages`` allowed keys accordingly.
- Test: schema round-trip; ``PRODUCTION_TEMPLATE_SEAMS`` count == 14;
  ``EXPERIMENTAL_PIPELINE_SEAMS`` count == 4; existing packs still load.

### Chunk 2: ``profiles.py`` required-seam relaxation

- Repo: sibling.
- Action: relax hard-error at ``profiles.py:60-65`` to fire only if the
  bank declares ``line_grounding`` required. Route through banks.json
  ``required_seams`` per Fable MF-R2-2.
- Alternatively: leave the check but adjust ``banks.json`` so science's
  ``required_seams`` list matches the Phase A empty-overrides intent
  (drop line_grounding from science required_seams if the operator
  agrees; else keep + defer line_grounding to Phase B per r2 decision).
- Test: existing pack tests remain green; new test loads a pack with
  ``line_grounding`` omitted for a bank that does not require it.

### Chunk 3: Extractor helper ``get_pack_prompt_or_none``

- Repo: sibling.
- Add ``src/upstream_story_lab/extractor.py`` with signature per MF-C4
  corrected form (see above).
- Wraps ``resolve_profile()``; enforces None-vs-string contract; raises
  ``RegistryError`` on structural failures (Codex SF3).
- Test: table-driven; every (bank, model, pipeline, seam) tuple returns
  either a str (populated) or None (absent/empty-override).

### Chunk 4: Empty-overrides in ``science_news_default.json``

- Repo: sibling.
- Action: rewrite ``fixtures/story_packs/science_news/science_news_default.json``
  to remove PARAPHRASE strings that currently populate seams like
  ``outline_system``, ``pitch_room_system``, ``dramatic_state_system``,
  ``coda_system``, ``title_system``. Replace with either omission (per
  MF-C6 rule for required seams -> must be omitted OR retained
  populated) or empty-string (non-required seams).
- Concrete decision for science_news: for each of its 6 required
  seams, keep populated ONLY if we want the pack to override production
  (we don't in Phase A -- science stays Python-authoritative); else
  OMIT the key. For non-required seams: use empty-string.
- Requires operator sign-off on: does science pack want ANY overrides
  at Phase A, or is it purely a passthrough? r2 recommendation:
  passthrough (omit all keys), then Phase B populates when overrides
  are wanted.
- Test: ``test_science_profile_leaves_style_picker_constants`` pattern
  extended to all 14 seams; ``get_pack_prompt_or_none`` returns None
  for every science pack seam.

### Chunk 5: Byte-identity harness

- Repo: sibling AND OTR (harness tests in sibling; snapshots include OTR
  outputs).
- Action:
  - Add ``tests/test_byte_identity_snapshot.py`` in sibling.
  - Capture pre-Phase-A ASSEMBLED string for every (bank, seam) tuple
    used by production. For outline: capture the assembled
    ``_MACRO/_PHASE/_BEAT`` STAGE system produced by the current writer
    (not just router return, per Codex MF6).
  - Assertion: post-Phase-A, ``get_pack_prompt_or_none(...) or
    PRODUCTION_LITERAL == snapshot`` for every tuple.
  - For science_news: ``get_pack_prompt_or_none == None`` for every
    seam (proves passthrough).
  - For outline: ADDITIONAL identity-preserving pytest:
    ``resolve_creative_system_prompt(default, "outline") is
    module._SYSTEM_PROMPT`` -- pins MF-C1 audio C7 contract.
- Test: this IS the test. Snapshot committed under
  ``tests/snapshots/byte_identity/``.

### Chunk 6: Anchor doc rewrites (docs-only)

- Repo: sibling (docs).
- Rewrite ``docs/R1_ARCHITECTURE_AND_CODING_PLAN_V2.md`` sections 5, 6,
  7b (upgrades 2-5), 9 -- CUT from Phase A per MF-C7.
- Rewrite ``docs/JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md`` R2
  file list -- replace ``catalogs.py`` with ``registry.py`` /
  ``profiles.py`` / ``bridge.py``.
- Doc-hygiene fix: ``_otr_creative_prompt_router.py:15-19`` docstring
  -- flag as stale (claims 4 phases; only 2 wired). NOT a Phase A
  code change; noted here for the coder window's awareness.

### Chunk 7 (final): full regression + Bug Bible

- Repo: OTR + sibling.
- Actions:
  - OTR: ``cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio; $env:PYTHONUTF8=1; & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider``
  - Bug Bible: ``cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide; & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\bug_bible_regression.py -q``
  - Sibling: ``cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab; & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider``
- Verify: HEAD == origin on both repos, no 0-byte files, no BOM, AST
  parse on touched .py files.
- Commit + push per chunk to ``v2.0-alpha`` (OTR) and ``main``
  (sibling; the sibling has different branch discipline per its own
  CLAUDE.md).

## What r3 must do (wiring)

r3 elaborates each chunk with:

- Exact file:line hunks (before/after code blocks) per chunk.
- Per-chunk test name + assertion body.
- Confirm the sibling-repo branch discipline (main vs a feature
  branch for these chunks).
- Verify the operator's "no production code touched" holds under
  chunks 4-6 (they all touch SIBLING code, not OTR production).
- Confirm chunk order does not break byte-identity at any intermediate
  commit -- audio must stay green at end of chunk 0, chunk 1, chunk 2,
  etc.

Panel structure unchanged: Codex + Fable + Sonnet, independent
reviews, Claude anchor synthesis.

## Grounding table (this pass)

| claim | source | status |
|---|---|---|
| ``PRODUCTION_MIRROR_MANIFEST.md`` exists, pins ``d48a9d76`` | ``ComfyUI-OTR-UpstreamStoryLab\PRODUCTION_MIRROR_MANIFEST.md:10-16`` | CONFIRMED (Codex found; my re-check verified) |
| OTR production baseline is ``a7bdc42d`` (docs tip ``c98a67ab``) | git rev-parse | CONFIRMED |
| Only 1 ``is _SYSTEM_PROMPT`` site in nodes/ | Sonnet repo-wide grep | CONFIRMED |
| line_composer at :2060-2061 is direct assign, no identity | ``_otr_line_composer.py:2060-2061`` (via Sonnet, Fable, Codex) | CONFIRMED |
| Real constant names ``_MACRO/_PHASE/_BEAT_SYSTEM_PROMPT`` at :1102/:1115/:1130 | ``_otr_outline.py`` grep | CONFIRMED |
| Consumer sites for macro/phase/beat at :1868/:1996/:2101 (no identity) | Fable grep | CONFIRMED |
| Packs are 3-tuple keyed | ``registry.py:117-124`` (via Codex) | CONFIRMED |
| ``profiles.py:60-65`` hard-errors on empty line_grounding | Fable + Codex | CONFIRMED |
| ``registry.py:167-176`` raises RegistryError on falsy required seam | Sonnet | CONFIRMED |
| ``science_news`` required_seams is 6 items excluding style_pick_* and story_select_system | ``banks.json:24-31`` (Sonnet) | CONFIRMED |
| science_news_default.json has 7 keys, omits style_pick_* | Sonnet | CONFIRMED |
| line_grounding rider is conditional f-string with ``{req.conflict_object}`` | ``_otr_line_composer.py:1621-1636`` (Fable) | CONFIRMED |
| ``_INVENTOR_SYSTEM`` no placeholders; ``_INVENTOR_USER_TEMPLATE`` has ``.format()`` placeholders | ``_otr_style_picker.py:296,301,329,334`` (Sonnet, Codex) | CONFIRMED |
| Router docstring claims 4 phases, only 2 wired | ``_otr_creative_prompt_router.py:15-19`` (Sonnet, Fable) | CONFIRMED |
| Anchor 1 still lists compat mirrors, visual policy, etc. that MF-C7 CUT | ``input.md:420-428,501-545,577-592,602-628`` (Codex) | CONFIRMED |
| Anchor 2 still names deleted ``catalogs.py`` | Codex | CONFIRMED |

## Judgment log

**Accepted from Codex r2:** All 7 MUST-FIX + 4 SHOULD-FIX + 3 CUTs.
Zero rejections; codex delivered the strongest ground-truthing on
concrete signatures.

**Accepted from Fable r2:** All 4 MUST-FIX + 3 SHOULD-FIX. Fable's
MF-R2-2 (line_grounding f-string) drives the r2 decision to defer
line_grounding to Phase B.

**Accepted from Sonnet r2:** All 4 MUST-FIX + 2 SHOULD-FIX. Sonnet's
factual correction of MF-C1 (only outline has identity check) is the
single most important r2 correction; nothing else was factually right.

**Refined:**
- MF-C5 elevated from "recommend Option A" to Chunk 0 explicit action
  after codex found the manifest and pin.
- MF-C1 downgraded from "load-bearing" to "outline-only, one pytest to
  pin".

**Rejected / deferred:**
- None. Every panel claim ground-truthed.
- ``line_grounding`` extraction moved to Phase B (Fable MF-R2-2 forces
  the deferral).

## Delta to feed into r3

r3 input = ``pass02_plan.md`` (this file) + operator scope + Chunk
0-7 spec. r3 focus = wiring: exact file:line hunks, test bodies,
sibling-repo branch discipline, intermediate-commit byte-identity
proof. Panel unchanged.
