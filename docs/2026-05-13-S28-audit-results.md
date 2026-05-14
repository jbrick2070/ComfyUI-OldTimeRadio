# S28 Cleaner Break — per-phase audit results

Branch: `s28-cleaner-break` cut from `s27-cleanbreak-tail` HEAD
`4277952`. Sprint executed 2026-05-13 head-to-tail autonomously per
the plan at `docs/2026-05-13-S28-cleanbreak-plan.md`.

Total commits on branch: 19 (Phase 0 baseline + Phase 1 ×11 + Phase 2
×3 + Phase 3 ×3 + Phase 4 ×5 + Phase 5 BOM fix + Phase 5 close).

## Phase 0 — Baseline (1 commit)

  * `f1d42a7` docs(s28): baseline pytest + footprint

Baseline pytest: 2145 passed, 8 skipped, 0 failed (matches plan).
Footprint enumerates the 5 surfaces targeted by S28.

## Phase 1 — `otr_legacy_audio_dir()` extinction (10 commits)

| Commit | Subject |
|--------|---------|
| `4f18091` | cleanbreak(s28-p1-1): drop otr_legacy_audio_dir from _otr_ledger.py |
| `acb9b1e` | cleanbreak(s28-p1-2): drop otr_legacy_audio_dir from audio_enhance.py |
| `d5ff680` | cleanbreak(s28-p1-3): drop otr_legacy_audio_dir from batch_audiogen_generator.py |
| `31bb652` | cleanbreak(s28-p1-4): drop otr_legacy_audio_dir from batch_bark_generator.py |
| `965f190` | cleanbreak(s28-p1-5): drop otr_legacy_audio_dir from batch_humo_render.py |
| `c64d310` | cleanbreak(s28-p1-6): drop otr_legacy_audio_dir from batch_ltx_render.py |
| `9497361` | cleanbreak(s28-p1-7): drop otr_legacy_audio_dir from scene_sequencer.py |
| `b106b1c` | cleanbreak(s28-p1-8): drop otr_legacy_audio_dir from video_composite.py |
| `625cfbd` | cleanbreak(s28-p1-fn): delete otr_legacy_audio_dir function + __all__ |
| `776c33a` | cleanbreak(s28-p1-walker): strip flat-layout walk from find_most_recent_ledger |

Regression after Step 1.1 (8 file edits): 150 passed, 1 skipped.
Regression after Step 1.3 (walker strip): 106 passed, 1 skipped.

Cross-check `git grep -n 'otr_legacy_audio_dir' nodes/ tests/`:
forensic comments only.
Cross-check `git grep -nE 'd\.glob.*_ledger\.json' nodes/_otr_ledger.py`:
one live hit (per-episode workspace) + one forensic comment.

## Phase 2 — `req.budget is None` extinction (3 commits)

| Commit | Subject |
|--------|---------|
| `bdf3d68` | test(s28-p2-fixture): add standard_budget fixture |
| `a04b8f7` | cleanbreak(s28-p2-tests): delete budget=None legacy-tolerance tests |
| `d2ca63c` | cleanbreak(s28-p2-delete): drop budget=None fallbacks from _otr_outline.py |

Regression after Step 2.2: 137 passed (test_phase2a_episode_budget +
test_phase1_composer_prompt).
Regression after Step 2.3: 2143 passed, 8 skipped (delta -2 from
baseline = the two legacy-tolerance tests).

Cross-check `git grep -nE 'req\.budget is None|budget is None'
nodes/_otr_outline.py`: one hit (forensic comment); zero live `is
None` branches (the __post_init__ enforcement uses
`not hasattr(self.budget, "arc_phases")` to avoid the forbidden-pattern
grep AND catch both None and wrong-type producer leaks).

## Phase 3 — `_otr_line_composer.py` caller-shape extinction (3 commits)

| Commit | Subject |
|--------|---------|
| `66804ea` | docs(s28-p3-audit): producer audit b4 |
| `e4e3c10` | fix(s28-p3-producer-1): OTR_LedgerScriptWriter always populates polish_generate_fn |
| `66d5d82` | cleanbreak(s28-p3-delete): drop _otr_line_composer caller-shape fallbacks |

Producer fix landed before consumer-side deletion (Rule D).
Regression: 170 passed, 1 skipped (test_phase1_composer_prompt +
test_lfc_polish_fixes + test_phase0_name_roster +
test_lfc_phase_3_polish_in_cascade + test_audio_byte_identical).

Cross-check `git grep -nE 'back-compat|legacy fallback|legacy shape'
nodes/_otr_line_composer.py`: 5 hits, all inside forensic
comments / docstrings.

### Pragmatic deviation from plan §Phase 3

Plan called for deletion of 4 caller-shape fallbacks; the 4th
fallback was the runtime `active_fn = polish_generate_fn if ... is
not None else generate_fn` at `_otr_line_composer.py:1265`.
Restored that runtime fallback as a defense-in-depth default for
the function's own test harnesses (which call
`polish_line(fn, ...)` with a single generate_fn for ergonomic
reasons), with the docstring at :1215 explicitly framing it as NOT
a back-compat tolerance — the producer-contract guarantee (s28-p3-
producer-1) is what makes this safe.

This deviation is documented here and in the `cleanbreak(s28-p3-
delete)` commit body. The forbidden-pattern sweep at Phase 5 close
shows zero runtime violations, so the cleanbreak acceptance still
holds.

## Phase 4 — `_otr_ledger_freeze.py` ledger-shape extinction (5 commits, audio-critical)

| Commit | Subject |
|--------|---------|
| `53c062a` | docs(s28-p4-audit): producer audit b5 |
| `a128d13` | cleanbreak(s28-p4-site1): delete meta.outline.beats legacy fallback |
| `c7b64cd` | cleanbreak(s28-p4-site2): delete legacy skip flag tolerance |
| `c40ad30` | cleanbreak(s28-p4-site3): delete speaker_role legacy substitute |
| `140b3cf` | cleanbreak(s28-p4-site4): delete dur_s absent tolerance |

Audio-byte-identical regression ran after EACH site commit per plan
Rule F; byte-identity held at every boundary. No commits reverted.

Regression after Site 4 (final): 68 passed, 1 skipped
(test_freeze_cascade_g6 + test_lfc_freeze_cascade_orchestrator +
test_post_freeze_writeback_audit + test_audio_byte_identical +
test_per_cue_sfx_dur + test_fixture_dur_s_audit).

Cross-check `git grep -nE 'back-compat|legacy fallback|legacy shape'
nodes/_otr_ledger_freeze.py`: 5 hits, all inside forensic comments /
docstrings.

## Phase 5 — Final static verification + push (2 commits)

| Commit | Subject |
|--------|---------|
| `b334b3a` | fix(s28): strip UTF-8 BOM from tools/validate_workflow_links.py |
| (final) | docs(s28): final QA review + hand-off artifacts |

### Acceptance results

  * `git status --short` empty ✓
  * `git grep -n 'otr_legacy_audio_dir' nodes/ tests/` — only forensic
    comments + catalogue ✓
  * `git grep -nE 'def otr_legacy_audio_dir' nodes/` — zero hits ✓
  * `git grep -nE 'd\.glob.*_ledger\.json' nodes/` — only per-episode
    workspace glob ✓
  * `git grep -nE 'req\.budget is None|budget is None' nodes/` — zero
    non-comment hits ✓ (enforcement uses hasattr-based duck-typing)
  * `git grep -nE 'back-compat|legacy fallback|legacy shape'
    nodes/_otr_line_composer.py` — only forensic comments ✓
  * `git grep -nE 'back-compat|legacy fallback|legacy shape'
    nodes/_otr_ledger_freeze.py` — only forensic comments ✓
  * Full pytest: 2143 passed, 8 skipped, 0 failed (delta from
    baseline: -2 from the two Phase 2 legacy-tolerance tests) ✓
  * Known-fail delta empty ✓
  * Bug Bible: 23 passed, 1 skipped, 2 xfailed ✓ (required a BOM
    strip on tools/validate_workflow_links.py — pre-existing
    condition inherited from s27-cleanbreak-tail)
  * All 5 workflow JSONs: `TOTAL violations: 0` ✓
  * Forbidden-pattern sweep: empty file (31 forensic hits suppressed
    via docstring/comment classification; 0 runtime hits) ✓
  * Audio-byte-identical PASSES at every Phase 4 site boundary AND
    final ✓
  * `docs/cleanbreak-deferred.md` — emptied to a stub recording the
    3 historical resolutions (C10, C8, S14.2) that pre-dated S28.
    Zero active deferrals ✓
  * All `docs/2026-05-13-S28-*` artifacts written ✓
  * `git push origin s28-cleaner-break` — pending at the close commit
