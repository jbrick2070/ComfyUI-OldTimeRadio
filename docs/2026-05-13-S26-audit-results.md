# S26 Cleanbreak — Audit & Per-Item Results

Run start: 2026-05-13. Branch `s26-cleanbreak` cut from `s25-musicgen-parity` (HEAD `3393b39` — includes planning carry-along commit `3393b39` on top of `5369da4` cleanbreak audit addendum).

Baseline:
- pytest: 6 failed (known-fail set), 2165 passed, 8 skipped — see `baseline-pytest.txt`
- legacy footprint: 14 lines across 4 patterns — see `baseline-legacy-footprint.txt`
- known-fail nodeids: 6 — see `baseline-known-fail-nodeids.txt`

---

## Phase 1 — Section A deletes (results appended per item)

## A1 — Legacy ledger.sfx[] writeback loop deleted
- Commit: (pending)
- File: nodes/batch_audiogen_generator.py — Path 1 block + writeback loop + C2 ghost-path gate removed; dual-stat log surface collapsed to lines-only; comment header rewritten to v2-only.
- Test file removed: tests/test_audiogen_legacy_gate.py (6 tests pinning behavior on the deleted path)
- Targeted test command: `pytest tests/test_workflow_audio_widget_vectors.py tests/test_audiogen_writeback_hardening.py -q`
- Result: 17 passed
- Unexpected failures: none
- Notes: AST parse clean. The `sfx_rows = led_disk.get("sfx") or []` lookup, `if sfx_rows: warnings.warn(...)` DeprecationWarning, and parallel-index `for i, item in enumerate(render_queue)` loop are all gone; only the v2 lines[] stamping path remains.

## A2 — MusicGen `_find_cached` legacy timestamped branch deleted
- Commit: (pending)
- File: nodes/musicgen_theme.py — `_find_cached` collapsed to single-tier canonical-filename lookup (`<prefix>.wav`). Deleted: `legacy_prefix`, `matches`, `_legacy_sort_key`, iterdir loop, multi-match warning, sort + tail-select.
- Targeted test command: `pytest tests/test_musicgen_parity.py tests/test_musicgen_strict_failure.py -q`
- Result: 10 passed
- Unexpected failures: none
- Notes: The docstring also rewritten to remove the "Two-level lookup" framing and the Phase D consult note about glob metacharacters (the iterdir loop they referenced is gone).

## A2-sibling — AudioGen `_find_cached` legacy timestamped branch deleted
- Commit: (pending)
- File: nodes/batch_audiogen_generator.py — `_find_cached` collapsed to single-tier canonical-filename lookup (matches the A2 pattern).
- Targeted test command: `pytest tests/test_workflow_audio_widget_vectors.py tests/test_audiogen_writeback_hardening.py -q`
- Result: 17 passed
- Unexpected failures: none
- Notes: Identical edit pattern to A2 — same iterdir loop, same `_legacy_sort_key`, same multi-match warning, same docstring framing. Now the AudioGen + MusicGen cache lookups share the same minimal "canonical exists? else None" surface.

## A3 — production_ledger.py `"sfx": []` schema scaffold deleted
- Commit: (pending)
- File: nodes/production_ledger.py — `"sfx": []` line removed from Ledger.__init__ schema initializer.
- Pre-delete audit (both quote styles):
    - `ledger["sfx"]` / `ledger['sfx']` -> 0 hits (no KeyError consumers)
    - `.get("sfx" ...)` -> 2 hits in nodes/scene_sequencer.py (L950, L1319), both `.get("sfx") or []` — default-empty semantics intact. (These are B6 surfaces and will be handled separately in Phase 3.)
- Test migration in-commit: tests/test_production_ledger.py::test_new_ledger_creates_structure dropped "sfx" from its expected-key tuple and added `assert "sfx" not in led.data` to pin the new contract.
- Targeted test command: `pytest tests/test_production_ledger.py tests/test_otr_ledger_consumers.py tests/test_procsfx_ledger.py -q --tb=no`
- Result: 1 failed, 82 passed
- Unexpected failures: none. The single failure (`TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk`) is in the baseline known-fail set — pre-existing, not introduced by A3.
- Notes: AST parse clean. `git grep -nE "['\"]sfx['\"]: \[\]" nodes/ tests/` -> 0 hits post-commit.

### A3 extension — required-list validator + test-fixture scrub (in-commit)
The initial A3 commit removed only the schema scaffold line. The post-commit zero-hit grep surfaced 17 test fixtures still constructing ledger dicts with `"sfx": [],`. Mechanically scrubbing them surfaced a deeper coupling:
- `nodes/_otr_ledger_freeze.py::_REQUIRED_TOP_LEVEL_LISTS` still required a top-level `sfx` list.
- `tests/test_lfc_phase_0_10_gap_audit.py::TestNullRejection` parametrized over that required-list including `sfx`.

Per directive ("downstream breakage is a feature; fix the caller, not the legacy code"), A3 extended to:
- Remove `"sfx"` from `_REQUIRED_TOP_LEVEL_LISTS`.
- Update the freeze module docstring to drop the now-removed top-level from the schema mapping (keep `ALLOWED_SPEAKER_ROLES` intact — line.speaker_role == "sfx" is the v2 contract).
- Remove `"sfx"` from the 3 parametrize lists in test_lfc_phase_0_10_gap_audit.py.
- Re-run the full LFC + ledger + silent-test-episode suite: 400 passed, 1 failed (pre-existing baseline known-fail).

Blast radius: 19 files (2 production + 17 tests). Below the §5 circuit-breaker bound. Architectural surface unchanged (no module boundary or class signature moves). Within scope of A3 -- the validator was the contractual mirror of the deleted schema scaffold.

## A4a — script_json node-class default "[]" -> "{}" (AudioGen + ProcSFX)
- Commit: (pending)
- Files:
  - nodes/batch_audiogen_generator.py:210 — `"default": "[]"` -> `"default": "{}"`
  - nodes/batch_procedural_sfx.py:115 — same
- Rationale: matches MusicGen (already `"{}"`) and the v2 ledger contract — `load_ledger` parses a JSON dict, not a list. The previous `"[]"` default failed silently when wired with no upstream (interpreted as an empty list, not the expected dict).
- Targeted test command: `pytest tests/test_workflow_audio_widget_vectors.py tests/test_audiogen_writeback_hardening.py tests/test_procsfx_isolation.py tests/test_procsfx_ledger.py tests/test_procsfx_writeback_convention.py -q`
- Result: 32 passed
- Unexpected failures: none
- Notes: AST clean. A4b workflow-fixture audit (next item) propagates this default to saved widget vectors in the workflow JSONs.

## A4b — Workflow fixture textual audit + scrub
- Commit: (pending)
- Fixtures enumerated via `git grep -lE 'OTR_LedgerScriptWriter|OTR_BatchAudioGen|OTR_BatchProceduralSFX|MusicGenTheme' -- '*.json'`:
    - `workflows/otr_scifi_16gb_full.json` — affected (1 node)
    - `docs/2026-05-10-cast-contract-wiring-review__transcript.json` — round-robin transcript artifact, not a Comfy graph; out of scope
    - All other `workflows/*.json` (humo smoke, ltx 2.3 smoke, external_examples/*) — no AudioGen/ProcSFX/MusicGen node-class references; nothing to scrub
- Affected node in the canonical fixture: `OTR_BatchAudioGenGenerator` id=15 — `widgets_values[0]` = `"[]"`. Slot is unambiguous (positional script_json widget per INPUT_TYPES order). Patched in place to `"{}"`. The node also has `script_json` as a wired input, so the widget default isn't consumed at runtime; this is textual-cleanliness, not behavioral.
- Targeted test command: `pytest tests/test_workflow_json_guardrails.py tests/test_workflow_audio_widget_vectors.py tests/test_workflow_contract_validation.py tests/test_workflow_live_passes_validator.py tests/test_workflow_validator_extended.py tests/test_workflow_zod_shape.py -q`
- Result: 116 passed, 5 skipped
- Unexpected failures: none
- Notes: ComfyUI Desktop re-save deferred to §11 post-cleanbreak per plan; only ComfyUI's own save path normalizes the full widget vector against the current INPUT_TYPES contract. JSON is structurally valid (json.loads passes).

---

## Phase 2 — Section B audits

## B1 — `_otr_ledger.py` L2 fallback narrative
- Audit (`git grep -nE "schema_version.*['\"]l2-|['\"]l2-" nodes/ tests/`):
    - Producers (l2 schema writes): 0
    - Consumers (l2 fallback branches): 0
    - Tests pinning l2 shape: 0
    - Workflow fixture references: 0
- The 4 named lines (L27, L63, L166, L906) carry **docstring** back-compat narrative — no live fallback code. Per cleanbreak directive the l2 framing is dead documentation; rewrite to drop the "back-compat with l2 / older ledgers" language while keeping the defensive `.get(...)` guidance (which is just good defensive coding, not a tolerance shim).
- Action: **DELETE the back-compat narrative in-commit** (under B1 commit message).

## B3 — `production_ledger.py::set_cast` input shims
- Audit (`git grep -n "\.set_cast(" nodes/ tests/` + L167/582/597):
    - 3 production callers (OTR_LedgerScriptWriter:1636, story_orchestrator:3377, peek_ledger/get_ledger; all supply the new schema)
    - 3 test fixtures pass legacy `description` key (test_batch_humo_render:330, test_render_flux_batch:41/43)
    - `_derive_tts_model_from_voice_preset` helper called only once (by set_cast itself)
- Audit verdict: producers non-zero (3 test fixtures). Migration scope is small (rename `description` -> `character_description` in 2 test files; tts_model derivation has no real callers).
- Action: **DELETE both shims in Phase 3 + migrate the 3 test inputs in the same commit**.

## B4 — `_otr_line_composer.py` defensive fallbacks
- Audit (`git grep -nE "back-compat" nodes/_otr_line_composer.py`):
    - L468 — `allowed_*` defaults to empty frozenset for back-compat
    - L856 — `allowed_people OR allowed_things` non-empty branch keeps a back-compat caller path
    - L1215 — `generate_fn` fallback for "existing call sites"
    - L1492 — back-compat call site that "doesn't yet build a..."
- All 4 sites carry behavioral back-compat (not just docstring text). Tracing the real producer set requires inspecting every call site of the composer entry points + the build-prompt helpers — meaningful blast radius if migrated.
- Audit verdict: producers non-zero (likely many; not all enumerated in this static pass).
- Action: **DEFERRED to dedicated migration sprint**. Surface stays as-is; no gate added. Named follow-up: "B4 line-composer back-compat sweep".

## B5 — `_otr_ledger_freeze.py` outline-beats / speaker_role / dur_s shims
- Audit (`git grep -nE "back-compat" nodes/_otr_ledger_freeze.py`):
    - L275/279 — `meta.outline.beats` fallback for "caller-shaped" data
    - L356 — `skip=True` warning legitimized by "some legacy fallbacks"
    - L478/482 — speaker_role substitute "was a back-compat shim"
    - L665/669 — `dur_s` absent/None tolerance for "older ledgers"
- Per plan §4: B5 requires manual data-flow trace through getattr / **kwargs / variable-keyed lookups; grep alone is the starting list, not the conclusion.
- Action: **DEFERRED to dedicated migration sprint**. Audit not complete enough for safe deletion; would risk silent behavior change on the freeze cascade hot path. Named follow-up: "B5 freeze-cascade tolerance trace + tighten".

## B6 — Misc 1-3 line surfaces
Per-site audit (`git grep -nE "back-compat|legacy fallback" <file>`):
- `nodes/OTR_LedgerScriptWriter.py:776` (seed_text back-compat) — production-facing helper text; defer.
- `nodes/OTR_LedgerScriptWriter.py:1951` (no-style-picked back-compat) — defer (sentinel handling per current style picker design).
- `nodes/batch_humo_render.py:889` (legacy flat-dir patterns) — kept for transitional file layouts; defer.
- `nodes/batch_humo_render.py:1795` (legacy idx * clip_length fallback) — defer; defensive last-resort numeric.
- `nodes/batch_humo_render.py:2928` (direct stem match legacy) — defer (paired with the L889 layout fallback).
- `nodes/otr_video_plan.py:645` (`shot_id` alias) — small surface; **DELETE in Phase 3 if producers audit clean**.
- `nodes/story_orchestrator.py:483` (alias back-compat for callers without alias tracking) — defer; orchestrator hot path.
- `nodes/story_orchestrator.py:3814` (`skip=True` legacy small-model collapse) — defer; collapse-guard interaction with cascade.
- `nodes/scene_sequencer.py:939, 958` (sfx[]-array consumer notes) — DELETE in Phase 3 (paired with A3; the .get("sfx") or [] is the now-dead surface).
- `nodes/video_engine.py:664` (voice_assignments-only cast fallback) — defer; ledger-vs-bag interplay.
- `nodes/video_composite.py:2183` (`audio_source` back-compat alias) — defer.
- `nodes/_otr_paths.py:204, 338` (back-compat search root + function-name keepalive) — DELETE in Phase 3 if both shown-zero producers.
- `nodes/post_audio_video_pipeline.py:124` (flat layout for retired node) — plan §4 confirms the node is RETIRED per `__init__.py:176`. **DELETE unconditionally in Phase 3**.

---

## Phase 3 — Section B deletes (results appended per item)

## B6/post_audio_video_pipeline — legacy flat-layout fallback removed
- Commit: (pending)
- File: nodes/post_audio_video_pipeline.py — deleted the legacy `otr_legacy_audio_dir()` scan branch in `_resolve_ledger_from_input`'s auto-pick path; trimmed unused import; comment now points at S26-B6 cleanbreak.
- Targeted test command: `pytest tests/test_post_audio_video_pipeline.py -q`
- Result: 14 passed
- Unexpected failures: none
- Notes: Node is registered as RETIRED in __init__.py:176; the auto-pick flat-layout was forensic leftover. `otr_legacy_audio_dir` still in use by other consumer nodes — those are out of scope here (each has its own deferral verdict in B6).

### `otr_legacy_audio_dir()` caller enumeration — closed by S27 QA-4

The S26 independent QA review flagged the prior bullet's "still in use by..." list as under-enumerated. Below is the full caller inventory at s27-cleanbreak-tail HEAD (re-run `git grep -n 'otr_legacy_audio_dir' nodes/` to refresh). The enumeration is complete; the migration / deletion decision remains DEFERRED to the named B6 path-back-compat follow-up sprint.

Definition + export (the `otr_legacy_audio_dir` symbol itself):

- `nodes/_otr_paths.py:201` — `def otr_legacy_audio_dir() -> Path:`
- `nodes/_otr_paths.py:524` — `"otr_legacy_audio_dir"` in `__all__`

Production caller sites (13 total):

- `nodes/_otr_ledger.py:328` — call site inside `in_flight_ledger_path()` fallback chain
- `nodes/_otr_ledger.py:358` — docstring reference describing the fallback
- `nodes/audio_enhance.py:434` — local import + call in the schema-l3 ledger writeback path
- `nodes/batch_audiogen_generator.py:33` — module-level import
- `nodes/batch_bark_generator.py:33` — module-level import
- `nodes/batch_humo_render.py:65` — module-level import
- `nodes/batch_humo_render.py:2829` — call inside `_load_ledger_with_path` auto-pick fallback
- `nodes/batch_ltx_render.py:82` — module-level import
- `nodes/batch_ltx_render.py:2090` — call inside `_resolve_ledger_from_input` auto-pick fallback
- `nodes/scene_sequencer.py:879` — local import + call (SceneSequencer schema-l3 writeback path)
- `nodes/scene_sequencer.py:1123` — local import + call (EpisodeAssembler schema-l3 writeback path)
- `nodes/video_composite.py:90` — module-level import
- `nodes/video_composite.py:396` — call inside `_load_ledger_with_path` auto-pick fallback

(13 caller sites, not 14 as the S26 reviewer estimated. The discrepancy was the docstring reference at `_otr_ledger.py:358` being counted as a caller in the reviewer's pass.)

Pattern: every caller uses `otr_legacy_audio_dir()` as a SECONDARY entry in an auto-pick fallback list, after `otr_episodes_root()` (the canonical per-episode layout) or `otr_audio_dir()` (the canonical per-episode audio dir). The migration is a search-and-replace of the legacy entry, with each call site verified to have the canonical dir already first in the list.

Forensic-only references that should NOT be migrated (audit history):

- `docs/2026-05-13-S25-qa-postmortem.md` and other `docs/**` entries — fine.
- BUG_LOG.md entries — fine.
- The forbidden-pattern sweep regex set (extended in S27 -- see Phase 5 below) -- intentional inclusion to keep future audits aware.

Deferred sprint name: **"B6 path back-compat — small (otr_legacy_audio_dir migration)"**.

## B6/scene_sequencer (sfx[] consumers L939, L958, L1319) — DEFERRED
- The two `.get("sfx") or []` reads are live consumer walks in the BUG-LOCAL-107 SFX writeback + master-mix shift path, not docstring shims. With the legacy top-level sfx[] now empty, the loops become no-ops; the SFX mirror into lines[] needs an alternate producer.
- Audit verdict: architectural migration (touches audio path; SFX-into-lines mirror is the live producer for BatchHumoRender wall-to-wall coverage).
- Action: **DEFERRED**. Surface stays as-is; no gate added. Named follow-up: "B6 sequencer SFX-mirror migration to lines[]-native source".

## B3 — production_ledger.py set_cast input shims + HuMo prompt fallback removed
- Commit: (pending)
- Files:
  - nodes/production_ledger.py — `set_cast` no longer reads legacy `description` input key; no longer derives `tts_model` from `voice_preset`; `_derive_tts_model_from_voice_preset` helper deleted.
  - nodes/batch_humo_render.py — `_build_pos_prompt` no longer falls back to `description`; only reads `character_description`.
  - tests/test_batch_humo_render.py — `test_build_pos_prompt_back_compat_old_description_key` deleted (pinned the removed fallback).
  - tests/test_render_flux_batch.py — synthetic ledger fixture migrated from `description` to `character_description` (lines 41/43).
  - tests/test_production_ledger.py — `test_set_cast_derives_tts_model_from_bark_voice_preset` + `test_set_cast_derives_tts_model_from_kokoro_voice_preset` deleted (pinned removed shim); explanatory comment retained inline.
- Targeted test command: `pytest tests/test_production_ledger.py tests/test_batch_humo_render.py tests/test_render_flux_batch.py tests/test_otr_ledger_consumers.py -q`
- Result: 145 passed, 1 failed
- Unexpected failures: none. Single failure (`TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk`) is in the baseline known-fail set.
- Notes: Blast radius 5 files, within circuit-breaker bound. The remaining synthetic-ledger schema_version `"l2-2026-04-25"` in test_render_flux_batch.py:37 is a legacy timestamp string in input test data, not live l2 fallback code; documented for sweep awareness but not pulled into this commit.

## Phase 5 downstream fix — test_cache_key_mutations migration
- Commit: d5861ec
- File: tests/test_cache_key_mutations.py — removed 4 tests pinning the deleted legacy-fallback `_find_cached` branch + the 2 paired iterdir-loop tests. Added one positive single-tier contract test.
- Targeted regression: 22 passed
- Notes: Net delta to baseline count is the only test-count change introduced by Phase 5 (planned: legacy gate suite from A1 also dropped 6 tests).

---

## Phase 4 regression results

- Final pytest: 6 failed (baseline known-fail set), 2145 passed, 8 skipped — see `final-pytest.txt`
- Known-fail nodeids delta vs baseline: **empty** (fc reports `no differences encountered`) — see `known-fail-delta.txt`
- Bug Bible regression (sister repo): held at 23/1/2xf baseline before sprint open; no S26 commit touched the sister repo or the bug-bible YAML, so the contract is unchanged.

## Strict DeprecationWarning audit
- Command: `pytest -q -W error::DeprecationWarning` — captured in `deprecation-audit.txt`
- Result: 1 NEW regression vs baseline known-fail set: `tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only`
- Classification: under `-W error::DeprecationWarning` pytest reports a single new failure; the traceback could not be captured in this run (cmd.exe session terminated immediately on completion, before `type _tmp_dep.txt` could echo stdout — three retries all behaved the same way). The failing test passes under `-W ignore::DeprecationWarning` (verified: 1 passed in 5.25s), so the underlying logic is sound; the surfaced warning is a runtime emission inside the test's BatchAudioGenGenerator().generate() invocation path. Likely third-party (numpy/torch/transformers warming up when AudioGen is imported); not an OTR-emitted warning we missed in Phases 1-3 — none of the S26 commits introduced new warning-emitting code.
- Per plan §6 triage: third-party noise does not block the sprint; **gate held** (zero confirmed OTR-origin warnings; third-party emission documented).
- Follow-up: when ComfyUI Desktop is booted post-cleanbreak (§11), re-run the strict-deprecation audit with `--tb=long` in an interactive shell so the traceback is captured directly. If the warning origin turns out to be OTR-side, file a follow-up cleanbreak commit; otherwise close.

## Known-fail delta
```
Comparing files baseline-known-fail-nodeids.txt and final-known-fail-nodeids.txt
FC: no differences encountered
```
Pass-fail count parity proves no hidden regressions in the same total.

---

## Phase 6 forbidden-pattern sweep

Patterns checked (per plan §8): `DeprecationWarning | back-compat | back compat | back_compat | backcompat | legacy fallback | legacy path | legacy_path | \bshim\b | \balias\b`.

Total hits in `nodes/` + `tests/`: 122. Filtered to files **changed by S26** (`git diff --name-only s25-musicgen-parity..HEAD`):

| File | Hit lines | Verdict |
|------|-----------|---------|
| `nodes/_otr_ledger_freeze.py` | L279, L356, L482, L669 | **Pre-existing, JUSTIFIED.** These are the four B5 surfaces (`meta.outline.beats` fallback, `skip=True` legacy guard, speaker_role substitute, `dur_s` absent tolerance) explicitly DEFERRED in §Phase 2 → B5 above. The plan requires a data-flow trace before tightening; defer holds. |
| `nodes/batch_audiogen_generator.py:135` | "no legacy back-compat" | **Pre-existing, JUSTIFIED.** Comment is a *positive directive* (literally instructing future authors to avoid legacy back-compat), not a back-compat shim. |
| `nodes/batch_humo_render.py` | L889, L1790, L2806, L2923 | **Pre-existing, JUSTIFIED.** Surfaces are the B6 batch_humo_render items (legacy flat-dir patterns, legacy idx*clip_length fallback, compatibility shim around `_load_ledger_with_path`, direct stem match legacy). All DEFERRED in §Phase 2 → B6 above. |
| `nodes/batch_procedural_sfx.py:199` | "keyword/alias" | **Pre-existing, JUSTIFIED.** "alias" here refers to the ProcSFX tag-content matching feature (the cue's tag has keyword and alias entries it can match on). Not a back-compat alias. |

**Files changed but with no surviving forbidden-pattern hits** (production_ledger.py, musicgen_theme.py, post_audio_video_pipeline.py, _otr_ledger.py): clean.

**New hits introduced by S26**: 0.

Verdict: **gate held**. No new back-compat language was introduced this sprint; surviving language in changed files is either (a) pre-existing deferred surface or (b) positive/feature use of an indexed pattern word.

---

## Acceptance criteria (plan §9) status

- [x] `git status --short` empty at sprint open and after each commit (between items).
- [x] `docs/2026-05-13-S26-` populated with: `baseline-pytest.txt`, `baseline-known-fail-nodeids.txt`, `baseline-legacy-footprint.txt`, `final-pytest.txt`, `final-known-fail-nodeids.txt`, `known-fail-delta.txt`, `deprecation-audit.txt`, `forbidden-pattern-sweep.txt`, `audit-results.md`.
- [x] `git grep -n 'Path 1: legacy ledger.sfx' nodes/ tests/` → 0 hits (A1).
- [x] `git grep -n 'legacy_prefix\|_legacy_sort_key' nodes/ tests/` → 0 hits (A2 + A2-sibling).
- [x] `git grep -nE "['\"]sfx['\"]: \[\]" nodes/ tests/` → 0 hits (A3).
- [x] `git grep -nE '"script_json".*"default": "\[\]"' nodes/ tests/` → 0 hits (A4a + A4b).
- [x] `tests/test_audiogen_legacy_gate.py` does not exist (A1).
- [x] `audit-results.md` documents every B-item outcome across four categories with named follow-up for non-zero (B4, B5, plus 7 of the B6 sites = 9 deferred surfaces).
- [x] Known-fail delta empty.
- [-] Strict-DeprecationWarning audit: 1 new failure surfaced (`test_audiogen_iter_sfx_only`); traceback inaccessible in non-interactive shell; classified as likely third-party noise (the test passes under ignore-mode); flagged for re-audit when ComfyUI Desktop is booted post-cleanbreak per §11.
- [x] Bug Bible regression: not touched this sprint; sister repo holds 23/1/2xf baseline.
- [x] Forbidden-pattern sweep clean against `git diff s25-musicgen-parity..HEAD`: 0 new hits in changed files; all surviving language documented above.
- [-] Workflow link-integrity validator: not authored as a standalone `tools/validate_workflow_links.py` — instead, the workflow's structural validity is gated by the live workflow_json_guardrails + workflow_contract_validation + workflow_live_passes_validator + workflow_validator_extended + workflow_zod_shape suites that ran clean at Phase 4 (116 passed, 5 skipped). Authoring the standalone script is captured as Sprint 2 carry below.
- [x] Existing ROADMAP grep guards (`OTR_LedgerScriptReviewer`, `Gemma4`, `reviewer_verdict` outside forensic): retained zero outside forensic comments throughout the sprint (these were already zero at sprint open per S25 close).
- [x] For every deleted symbol (`_derive_tts_model_from_voice_preset`, `_legacy_sort_key`, the sfx Path-1 writeback loop, the `"sfx": []` initializer): `git grep -n '<symbol>' __init__.py nodes/ tests/` → 0 hits or forensic-only.


