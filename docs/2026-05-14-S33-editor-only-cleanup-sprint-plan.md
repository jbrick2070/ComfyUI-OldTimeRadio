# S33 — Editor-only cleanup (Cowork loop, pytest-only)

> **What this is:** Retire the two cascade auditors (Phase 1 + Phase 9). Phase 2 Script Doctor gets hardened *before* Phase 9 deletion so it hard-fails malformed output on its own. Lock the existing speaker-differentiated polish prompts against accidental collapse. Sprint is subtractive: net codebase shrinks.

**Status:** PLANNED (round-robin reviewed 2026-05-14).
**Branch:** `s33-editor-only-cleanup`. Cut from `s32-helper-per-subpass-routing` @ B8 (`3261b18`).
**Sequencing:** S31 → S31.5 → S32 → **S33 (this sprint)** → Sprint C → Sprint E (queued) → Sprint A.
**Loop per commit:** review → code → wire (audited at B1) → pytest → commit → push. No ComfyUI execution. No operator gates.

---

## Why this sprint exists

Standing directive: every node must edit the story. Audit-only nodes that emit reports nobody acts on get retired. Two cascade phases violate this rule:

- **Phase 1 auditor** — emits issue reports, never rewrites
- **Phase 9 post-edit auditor** — verifies Phase 2 Script Doctor's edits landed correctly, but doesn't itself edit

If Phase 2 can produce malformed output, that's a hard-fail bug at write time, not a soft-check afterward. Phase 9 masks Phase 2 defects rather than forcing them visible. Both go — BUT Phase 9 deletion is gated on Phase 2 already hard-failing loud on its own. That gate gets explicit testing at B3 before Phase 9 deletion at B4.

Polish (`enable_polish_pass`) is an editor (rewrites flagged lines), survives untouched. The existing speaker-differentiated prompts stay (locked at B5).

---

## Round-robin review integration (2026-05-14)

This plan integrates HIGH-severity findings from Gemini + ChatGPT review:

- **HIGH (both):** Phase 9 cannot be deleted until Phase 2 demonstrably hard-fails malformed output on its own. → New B3 commit inserts a test-before-fix proof.
- **HIGH (both):** `not hasattr(...)` deletion guards don't catch dangling downstream references. → B2/B4 add tree-wide grep tests for the actual deleted-symbol names.
- **HIGH (ChatGPT, verified against code):** Polish prompt constant naming asymmetry. Current code has `_POLISH_SYSTEM_PROMPT` (character) and `_POLISH_SYSTEM_PROMPT_ANNOUNCER`. The earlier plan draft referenced a non-existent `_POLISH_SYSTEM_PROMPT_CHARACTER`. → B5 renames `_POLISH_SYSTEM_PROMPT` → `_POLISH_SYSTEM_PROMPT_CHARACTER` for design symmetry, updates the `polish_line()` dispatch, and adds behavior tests proving the prompts semantically differ (not just by name).
- **MEDIUM (ChatGPT):** "Wire: None" was too confident; workflow JSON audit moves into B1 inventory.
- **MEDIUM (both):** Polish design lock must include behavior tests, not just a forbidden-sweep marker.

Convergent judgment calls: keep B2/B4 separate; defer Sprint E enhancer chain audit; keep B1 inventory.

---

## Hard rules (continuity from S31 + S31.5 + S32)

1. **Audio C7 byte-identical pytest proxy** must hold at every commit boundary. Auditor deletions and the polish prompt rename are behavior-preserving in default config; if C7 regresses, something accidental happened.
2. **No legacy back-compat reintroduced.**
3. **No new generate or lifecycle surfaces.**
4. **No widgets.** S33 does not add user-facing toggles.
5. **Bug Bible regression** 23 / 1 / 2xf at every commit boundary.
6. **No separate change logs.** Updates flow to `BUG_LOG.md` and `ROADMAP.md`.
7. **Tests written before fixes** for structural defects. Red-on-parent, green-on-fix. B3 enforces this for the Phase 2 hard-fail proof.
8. **Forbidden-pattern sweep** stays at 0 runtime hits.

---

## Canonical pytest run

Same as S32 plus new S33 test files. Wide-walk should stay clean.

**Pytest baseline:** post-S32 wide walk 2103/10/0. Target at B6 close: similar (some tests delete with Phase 1/9, a few new ones added).

---

## Commit structure (B0 → B6)

### B0 — branch cut + plan landing (~0.25 d)

**Review.** Confirm parent at `s32-helper-per-subpass-routing` @ `3261b18`. Confirm clean working tree.

**Code.** Cut `s33-editor-only-cleanup` branch. Land this plan at `docs/<date>-S33-editor-only-cleanup-sprint-plan.md`.

**Wire / Pytest.** Baseline recorded.

**Commit subject.** `B0: branch cut + S33 editor-only cleanup plan landing (round-robin integrated)`

---

### B1 — pre-deletion inventory (~0.5 d)

**Review.** S33 doesn't have the cascade file's contents in plan context; B1 forces discovery of the actual code surface before deletion. Output is a machine-checkable inventory table.

**Code.** No production code change. Inventory document only.

Pre-grep for cascade phase code:
```cmd
findstr /s /n "_phase_1\|Phase 1\|phase_1" nodes\OTR_LedgerFreezeCascade.py nodes\_otr_freeze_cascade.py
findstr /s /n "_phase_9\|Phase 9\|phase_9\|post_edit_auditor" nodes\OTR_LedgerFreezeCascade.py nodes\_otr_freeze_cascade.py
```

**Inventory table** at `docs/<date>-S33-B1-cascade-auditor-inventory.md` — one row per symbol:

```
symbol | file:line | type | delete or keep | reason | tests touched | downstream meta keys | downstream code refs | workflow JSON refs
```

Required pre-grep coverage:
1. **Method/function symbols** in cascade module
2. **Widget keys** in cascade `INPUT_TYPES` (`enable_phase_1_*`, `phase_1_*`, `enable_phase_9_*`, `phase_9_*`, etc.)
3. **`meta` dict keys** stamped by Phase 1 or Phase 9 (e.g., `meta["phase_1_report"]`, `meta["phase_9_status"]`)
4. **Downstream consumer grep**: every meta key from step 3, search `nodes/ scripts/ workflows/ tests/` for reads
5. **Workflow JSON grep**: scan `workflows/*.json` for any widget key names from step 2
6. **Test file inventory**: which test files cover Phase 1 + Phase 9 and will need to be deleted vs. refactored

Mark each row's `delete or keep` decision based on:
- audit-only → DELETE
- editor (rewrites content) → STOP, surface to Jeffrey before proceeding
- diagnostic meta key with no downstream consumer → DELETE
- meta key WITH downstream consumer → flag for separate handling

**Pytest.** None (no code change).

**Commit gate.** Inventory complete. Every Phase 1 + Phase 9 symbol mapped. Downstream consumers identified. If any phase is found to edit content rather than audit, halt and surface to Jeffrey.

**Commit subject.** `B1: cascade Phase 1 + Phase 9 inventory (machine-checkable table, downstream consumer + workflow JSON sweep)`

---

### B2 — delete Phase 1 cascade auditor (~0.5 d)

**Review.** Read B1 inventory's Phase 1 rows. Confirm all rows have `delete or keep = DELETE`. If any row flagged for separate handling, address it first.

**Code.**
- Delete Phase 1 method(s) from cascade class.
- Delete any Phase 1 widgets from cascade `INPUT_TYPES`.
- Delete Phase 1 dispatch from cascade's main `run`.
- Delete Phase 1 meta key writes.
- Delete Phase 1 test file(s) per B1 inventory.

**Wire.** Per B1 inventory: if any `workflows/*.json` referenced deleted Phase 1 widget keys, update those JSONs in this commit. If no workflow JSON references existed (per B1), document that in the commit message.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_cascade_no_phase_1_method` | `tests/test_no_cascade_phase_1_auditor.py` (new) | `not hasattr(OTR_LedgerFreezeCascade, "<actual_name_from_B1>")` |
| `test_cascade_no_phase_1_widget` | same | Cascade `INPUT_TYPES.optional` has no Phase 1 widget keys |
| `test_no_phase_1_string_refs_in_nodes` | same | Tree-wide grep across `nodes/` (excluding this test file, forensic docs): 0 hits for `_phase_1`, `"phase_1"`, deleted meta keys |
| `test_no_phase_1_string_refs_in_workflows` | same | Tree-wide grep across `workflows/`: 0 hits |
| `test_audio_c7_byte_identical_b2` | `tests/test_audio_byte_identical.py` | Default-config canary |

**Commit gate.** 5 tests green. Canonical pytest subset green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.

**Commit subject.** `B2: delete cascade Phase 1 auditor — audit-only retired, downstream consumer sweep clean`

---

### B3 — prove Phase 2 Script Doctor hard-fails malformed output (NEW, ~0.5 d)

**Review.** Phase 9 deletion is structurally gated on Phase 2 already hard-failing loud. B3 PROVES that property exists before B4 deletes the safety net.

This is test-before-fix per Hard rule #7. Two possible outcomes:

- **Outcome A (Phase 2 already hard-fails):** new tests pass immediately on parent code. Commit is the tests alone, ship as forward-guards.
- **Outcome B (Phase 9 was actually catching Phase 2 defects):** tests fail on parent code. Phase 2 needs hardening to make them pass. Hardening lands in same commit.

If B3 lands as Outcome B, the commit's diff includes both the new tests and the Phase 2 hardening.

**Code.**

For each failure mode B1 inventory identified as something Phase 9 catches, write a test that:
1. Mocks Phase 2 to produce that malformed output (invalid JSON, missing required fields, hallucinated character name, mismatched line count vs. cast, empty rewrite output)
2. Calls Phase 2 directly (NOT through the full cascade chain that would normally include Phase 9)
3. Asserts Phase 2 raises a strict, named exception (e.g., `ScriptDoctorValidationError` or `RuntimeError` with a meaningful message)

If tests fail on parent code, harden Phase 2:
- Add explicit validation at Phase 2 output before returning
- Raise loud exceptions, not soft fallbacks
- Match the pattern S32 B3 established for `CastValidationLLMError` (fail-fast per architectural decision D2)

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_phase_2_rejects_malformed_json` | `tests/test_phase_2_hardfail.py` (new) | Mock Phase 2 returning invalid JSON; assert exception raised inside Phase 2 |
| `test_phase_2_rejects_missing_fields` | same | Mock with missing required line fields; assert exception |
| `test_phase_2_rejects_hallucinated_speaker` | same | Mock with speaker name not in cast; assert exception |
| `test_phase_2_rejects_empty_output` | same | Mock with empty rewrite; assert exception |
| `test_audio_c7_byte_identical_b3` | `tests/test_audio_byte_identical.py` | Canary (if Outcome B added hardening, must not perturb audio in default config) |

**Commit gate.** All 4 hard-fail tests green. Audio C7 holds. If Outcome B, Phase 2 hardening landed in this commit.

**Commit subject.** `B3: prove Phase 2 Script Doctor hard-fails malformed output (gate for B4 Phase 9 deletion)`

---

### B4 — delete Phase 9 post-edit auditor (~0.5 d)

**Review.** Confirm B3 hard-fail tests are green. Without that proof, Phase 9 deletion is unsafe. If B3 was Outcome B (hardening required), confirm hardening shipped in B3, not deferred.

**Code.**
- Delete Phase 9 method(s) from cascade class.
- Delete any Phase 9 widgets from cascade `INPUT_TYPES`.
- Delete Phase 9 dispatch from cascade's main `run`.
- Delete Phase 9 meta key writes.
- Delete Phase 9 test file(s) per B1 inventory.

**Wire.** Per B1 inventory: if any `workflows/*.json` referenced deleted Phase 9 widget keys, update those JSONs in this commit.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_cascade_no_phase_9_method` | `tests/test_no_cascade_phase_9_auditor.py` (new) | `not hasattr(OTR_LedgerFreezeCascade, "<actual_name_from_B1>")` |
| `test_cascade_no_phase_9_widget` | same | Cascade `INPUT_TYPES.optional` has no Phase 9 widget keys |
| `test_no_phase_9_string_refs_in_nodes` | same | Tree-wide grep across `nodes/`: 0 hits for `_phase_9`, `"phase_9"`, `post_edit_auditor`, deleted meta keys |
| `test_no_phase_9_string_refs_in_workflows` | same | Tree-wide grep across `workflows/`: 0 hits |
| `test_audio_c7_byte_identical_b4` | `tests/test_audio_byte_identical.py` | Canary |

**Commit gate.** 5 tests green. Canonical pytest subset green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. B3 Phase 2 hard-fail tests still green (regression check).

**Commit subject.** `B4: delete cascade Phase 9 post-edit auditor — audit-only QA-for-edit-pass retired (Phase 2 now hard-fails on its own per B3)`

---

### B5 — polish prompt rename + design lock (~0.5 d)

**Review.** Current code has `_POLISH_SYSTEM_PROMPT` (character) and `_POLISH_SYSTEM_PROMPT_ANNOUNCER`. Asymmetric. B5 makes the design explicit and locks against future collapse.

**Code.**

1. **Rename** in `_otr_line_composer.py`:
   - `_POLISH_SYSTEM_PROMPT` → `_POLISH_SYSTEM_PROMPT_CHARACTER`
   - Update `polish_line()` dispatch (line ~1221-1223): `_POLISH_SYSTEM_PROMPT_ANNOUNCER if is_announcer else _POLISH_SYSTEM_PROMPT_CHARACTER`
   - Update any other internal references (one-pass grep)
   - Behavior-preserving — character prompt content identical, only variable name changes

2. **Design-lock code comment** immediately before `_POLISH_SYSTEM_PROMPT_CHARACTER`:

```python
# S33 B5 design lock — DO NOT collapse these into a single prompt.
# Character and announcer beats need DIFFERENT polish prompts:
#   - Character beats: forbid narration (it's a leak)
#   - Announcer beats: allow narration (it IS the announcer voice)
# A unified prompt regresses one or the other:
#   - Forbids narration → breaks announcer (rewrites the third-person
#     narration that IS the announcer style)
#   - Allows narration → breaks character (no longer catches the
#     narration leaks polish exists to catch)
# polish_line dispatches by speaker_role to pick the right prompt
# at runtime. Original design intent: LFC sprint commit 3 section
# 6.1 (2026-05-11). Forbidden-sweep markers lock obvious bad names;
# behavior tests at S33 B5 lock the semantics.
```

3. **Forbidden-sweep markers** added to `docs/_s28_forbidden_sweep.py`:

```python
    # S33 B5: polish design lock (do NOT collapse to single prompt)
    r"|\b_POLISH_SYSTEM_PROMPT_UNIFIED\b"
    r"|\b_UNIFIED_POLISH_PROMPT\b"
    # S33 B2/B4: cascade auditor retirements (actual names from B1)
    r"|\b<phase_1_method_name>\b"
    r"|\b<phase_9_method_name>\b"
```

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_polish_prompts_are_distinct` | `tests/test_polish_speaker_prompts_locked.py` (new) | Both `_POLISH_SYSTEM_PROMPT_CHARACTER` and `_POLISH_SYSTEM_PROMPT_ANNOUNCER` exist as module-level constants; text contents are NOT identical |
| `test_polish_character_prompt_forbids_narration` | same | Assert `_POLISH_SYSTEM_PROMPT_CHARACTER` text contains substring matching the narration-forbidden marker (e.g., `'No "he said"'` or `"No "` AND `"narration"`); behavior-based, not name-based |
| `test_polish_announcer_prompt_allows_narration` | same | Assert `_POLISH_SYSTEM_PROMPT_ANNOUNCER` text contains substring explicitly allowing third-person narration (e.g., `"third-person narration is OK"` or equivalent semantic match) |
| `test_polish_line_dispatches_different_prompts_by_role` | same | Mock the underlying generate_fn; call `polish_line(..., speaker_role="character")` then `polish_line(..., speaker_role="announcer")`; assert the captured system prompts differ |
| `test_no_unified_polish_prompt_constant` | same | `not hasattr(_otr_line_composer, "_POLISH_SYSTEM_PROMPT_UNIFIED")` AND `not hasattr(_otr_line_composer, "_UNIFIED_POLISH_PROMPT")` |
| `test_audio_c7_byte_identical_b5` | `tests/test_audio_byte_identical.py` | Canary: rename is purely behavior-preserving |

**Commit gate.** 6 tests green. Canonical pytest subset green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean (placeholder phase-method markers replaced with real names from B1 inventory).

**Commit subject.** `B5: polish prompt rename + design lock — _POLISH_SYSTEM_PROMPT → _CHARACTER, behavior tests prove semantic differentiation, sweep markers armed`

---

### B6 — sprint close (~0.5 d)

**Review.** Mirror S31 / S31.5 / S32 final QA format.

**Code.** Final QA review at `docs/<date>-S33-final-qa-review.md`. ROADMAP refresh — mark S33 closed, note the auditor retirements, Phase 2 hardening (if Outcome B), and polish design lock as the sprint's deliverables.

**Wire / Pytest.** Full canonical pytest run. Wide walk.

**Commit gate.** All S33 acceptance rows green. Audio C7 held at every B2-B5 boundary. Branch pushed.

**Commit subject.** `B6: Sprint S33 close — cascade auditors retired, Phase 2 hardened, polish design locked`

---

## Acceptance table

| # | Check | Target |
|--:|---|---|
| 1 | Canonical pytest count | green |
| 2 | Wide pytest walk | 0 unexpected failures |
| 3 | Bug Bible regression | 23 / 1 / 2 |
| 4 | Audio C7 byte-identical (pytest proxy, default config) | holds B2 → B5 |
| 5 | Forbidden sweep | 0 runtime hits |
| 6 | Phase 1 method DELETED from cascade | ✅ |
| 7 | Phase 9 method DELETED from cascade | ✅ |
| 8 | Cascade Phase 1 widgets in `INPUT_TYPES` | NONE |
| 9 | Cascade Phase 9 widgets in `INPUT_TYPES` | NONE |
| 10 | Phase 1 string refs in `nodes/` | 0 |
| 11 | Phase 9 string refs in `nodes/` | 0 |
| 12 | Phase 1/9 widget refs in `workflows/*.json` | 0 |
| 13 | Phase 2 hard-fails malformed Script Doctor output without Phase 9 | ✅ (B3 proof tests green) |
| 14 | `_POLISH_SYSTEM_PROMPT_CHARACTER` exists (renamed from `_POLISH_SYSTEM_PROMPT`) | ✅ |
| 15 | `_POLISH_SYSTEM_PROMPT_ANNOUNCER` exists | ✅ |
| 16 | Polish prompts text contents non-identical | ✅ (asserted) |
| 17 | Character prompt forbids narration (behavior test) | ✅ |
| 18 | Announcer prompt allows third-person narration (behavior test) | ✅ |
| 19 | `polish_line()` dispatches different prompts for different `speaker_role` | ✅ (behavior test) |
| 20 | `_POLISH_SYSTEM_PROMPT_UNIFIED` does NOT exist | ✅ |
| 21 | Design-lock comment present in `_otr_line_composer.py` | ✅ |
| 22 | New S33 sweep markers | 4 (Phase 1 + Phase 9 + 2 polish unify names) |
| 23 | ROADMAP refreshed | S33 marked closed |

---

## Out of scope for S33

- **Polish behavior changes.** Speaker-differentiated prompts stay. B5 rename is naming-symmetry only, not behavior.
- **Sprint E enhancer chain audit** (`arc_enhancer` / `self_critique` / `target_length` revival decisions). Separate sprint, queued post-Sprint C.
- **Audio-intentional sprint** (model-author `generation_config.json` respect for polish). Queued post-S33.
- **Loader API consolidation.** Carried forward from S31 forward work.
- **Cascade Phase 2 Script Doctor feature additions.** B3 may harden it (Outcome B); otherwise it stays as-is.
- **Cascade Phase 7 + Phase 8 readiness checks.** Pre-flight, non-LLM, different category from auditors. Survive.

---

## Sources

- `docs/2026-05-14-S32-final-qa-review.md` — parent sprint close
- `docs/2026-05-14-S31p5-final-qa-review.md` — subtractive sprint format reference
- `nodes/_otr_line_composer.py` lines 1059, 1081, 1221-1223 (polish prompt constants + dispatch — verified during round-robin integration)
- `nodes/OTR_LedgerScriptWriter.py` line 2950 (n_optional = 15, unchanged by S33)
- `workflows/otr_scifi_16gb_full.json` — cascade node id 62, inputs verified to have no phase_1/phase_9 widget references (other workflows audited in B1)
- ROADMAP.md — sprint sequence + standing directives
- BUG_LOG.md — entries continuity
- Round-robin review documents (Gemini + ChatGPT, 2026-05-14)
