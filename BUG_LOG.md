# OTR Bug Log

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**Owner:** Jeffrey A. Brick
**Last entry:** BUG-LOCAL-223 (2026-05-13) -- general lesson surfaced by the s26-downstream missed-regression sweep: a sprint that deletes a contract MUST run pytest before declaring its verification phase complete; an EXPECTED_FAILED_NODEIDS byte-identical delta is not the same as an actual-suite-green delta.
**Stack head when last updated:** S26 cleanbreak final QA commit (s26-cleanbreak branch)
**Promotion target:** `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`
**Bible candidates pending promotion:** 19 entries (BUG-LOCAL-201, 202, 204, 205, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 222, 223) -- see "Bible candidates pending promotion" section below. Batch-promote after v2.0 ships per `feedback_roadmap_buglog_live_docs`. 221 remains deferred pending interactive shell re-audit.

---

## What this file is

Live, append-only record of every bug found in OTR development.
Per CLAUDE.md project rule:

> Maintain `BUG_LOG.md` actively. Every bug logged the moment it's
> found -- no batching, no waiting. Live document tracking the
> build history.

Bugs are numbered `BUG-LOCAL-NNN` with monotonic per-era ranges:

- `001-129` -- pre-voice-path-cleanbreak era (LFC + earlier)
- `200+`    -- voice-path-cleanbreak era (P1-P3, S1-S15)

Numbering reset is intentional -- it gives a clean visual cut
between sprint epochs and lets a reader skim "find me a v2.0-alpha
bug" without scrolling through legacy entries.

---

## Entry schema (per CLAUDE.md)

```markdown
### BUG-LOCAL-NNN: Title
- **Date:** YYYY-MM-DD | **Phase:** 0-6 | **Bible candidate:** yes/no
- **Symptom:** exact error / console output
- **Cause:** root cause (or "pending -- awaiting investigation")
- **Fix:** what resolved it (or "pending")
- **Verify:** how to confirm
- **Tags:** vram, widget-drift, ffmpeg, subprocess, parse-fatal, dialogue-scaling, json-wiring, etc.
```

Mark `[FIXED commit-sha YYYY-MM-DD]` after the title when resolved
-- do not delete entries. When `Bible candidate: yes` and the fix
is verified, promote to the survival guide repo per CLAUDE.md
"Bug Log Pipeline" section.

---

## Active known failures (S15 quarantine)

The 6 entries in `tests/conftest.py::EXPECTED_FAILED_NODEIDS` (and
mirrored in `docs/known-failures.md`) are NOT bugs in this file's
sense -- they are failing tests under quarantine, not unfixed
production runtime bugs. The `pytest_sessionfinish` hook
(S15.1+S15.2 / commit `f813b37`) enforces that the failure SET
stays exactly that 6.

Entries below are bugs found in production logic during S6-S15
sprints, regardless of fix-status.

---

## Voice-path-cleanbreak era (BUG-LOCAL-200+)

### BUG-LOCAL-200: G7 contract drift in consumer widgets [FIXED 3090007 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** AudioGen widget `default_duration.min` was the literal `0.5`; ProcSFX widget `default_duration.min` was the literal `0.1`. The freeze cascade enforced G7 SFX `dur_s` bounds at `[SFX_DUR_MIN_S=0.5, SFX_DUR_MAX_S=10.0]` (post-S6.4 tightening). ProcSFX accepted writer values down to 0.1 -- silently disagreeing with the freeze contract -- and clamped them post-hoc. Internal per-cue clamps in BOTH consumers also used magic-number literals (`max(0.5, min(10.0, ...))`, `max(0.1, min(10.0, ...))`).
- **Cause:** Magic-number literal at the consumer surface AND in the consumer clamp; no import of the freeze cascade's authoritative constants. A future bound shift in `_otr_ledger_freeze.py` would have left consumer surfaces silently disagreeing.
- **Fix:** S10.1 -- export `SFX_DUR_MIN_S` and `SFX_DUR_MAX_S` from `_otr_ledger_freeze.py::__all__`; both consumers import them for widget min/max AND internal clamp. Plus drift guard `tests/test_g7_consumer_constants.py` (5 tests) including object-identity assertion catching the local-shadow refactor case.
- **Verify:** `findstr /SI "0.1\|0.5\|10.0\|12.0" nodes\batch_procedural_sfx.py` filtered to widget/clamp sites returns zero hits. Same gate on AudioGen returns zero clamp/widget hits.
- **Tags:** widget-drift, magic-number, contract-honesty, g7

### BUG-LOCAL-201: AudioGen cache key was model-id-blind and guidance-scale-blind [FIXED 574038e 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Switching AudioGen models (`facebook/audiogen-medium` -> `facebook/audiogen-large`) or guidance scales (CFG 3.0 -> 5.0) between runs silently returned the prior model's wav -- the cache key didn't include either input. The user got a "cached" render that was the wrong model's output.
- **Cause:** `_cache_prefix(prompt, duration_sec, episode_seed)` payload was `f"{duration_sec}|{prompt}|{episode_seed}"`. Output-determining inputs `model_id` and `guidance_scale` were never hashed.
- **Fix:** S12.3 -- keyword-only signature `_cache_prefix(*, prompt, duration_sec, episode_seed, model_id, guidance_scale)`. JSON-canonical payload via `json.dumps(..., sort_keys=True, separators=(",", ":"))`. Truncation extended `[:8] -> [:12]` for collision-resistance. Three new dimension tests + drift guards.
- **Verify:** `pytest tests/test_audiogen_cache_keys.py::test_audiogen_cache_prefix_changes_when_model_id_changes -v` (and the guidance_scale + float-canonical siblings).
- **Tags:** cache-key, ledger-derived, audiogen, content-addressed
- **Bible candidate rationale:** General lesson -- cache keys must include every output-determining input. Standing-directive #9 in OTR codifies this; the survival guide should publish it as a pattern.

### BUG-LOCAL-202: ProcSFX silently overwrote on dur_s iteration [FIXED c4ab258 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** When the writer iterated on scene timing -- emitting the same `line_id` at a new `dur_s` between runs -- the second render OVERWROTE the first wav on disk. The user lost A/B history; the cache identity (the on-disk filename) didn't reflect the changed input. Found via the F-6 finding in the S6-S8 round-robin: "active iteration on scene timing is a real workflow."
- **Cause:** Filename was `proc_<sfx_type>_<line_id>.wav`. Identity surface keyed only by line, not by duration. ProcSFX has no formal cache layer, so the on-disk filename IS the de-facto identity.
- **Fix:** S12.1 -- filename extended to `proc_<sfx_type>_<line_id>_<perm>.wav` where `<perm>` is `hashlib.sha256(f"{cue_duration:.3f}|{chosen_type}|{line_id}").hexdigest()[:8]`. Disk usage grows with iteration count; procedural wavs are kB-scale so the trade-off is favorable.
- **Verify:** `pytest tests/test_audiogen_cache_keys.py::test_procsfx_filename_perm_hash_varies_with_dur_s -v`. Also the source-level guard `test_procsfx_perm_hash_in_module_source` catches a future refactor that strips the perm segment.
- **Tags:** cache-key, on-disk-identity, procsfx, content-addressed
- **Bible candidate rationale:** Same general lesson as BUG-LOCAL-201 in a no-cache-layer variant -- the on-disk filename IS the identity surface and must include every output-determining input.

### BUG-LOCAL-203: cast contract accepted structural tokens as character names [FIXED badcae5 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** A cast row like `{"name": "TITLE", "voice_preset": "v2/en_speaker_0", ...}` passed `_assert_voice_preset_invariant` and `_assert_unique_bark_voices` cleanly. The two existing assertions had no opinion on the *name* shape. An LLM hallucination that emitted any of TITLE / NOTE / TARGET / STYLE / NARRATOR as a character name rendered as a Bark voice line in production with no contract pushback.
- **Cause:** Cast contract was preset-shape-aware but not name-shape-aware. The deleted `story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS` had screenplay-meta-direction patterns (KEVIN VOICEOVER, JOHN V.O., etc.) but never anchored the structural-token names from `_BRACKET_STRUCTURAL_TOKENS`.
- **Fix:** S13.1 -- ported the pre-S7.1 `_SFX_CAST_BLOCKLIST_PATTERNS` into `_otr_casting._NON_CHARACTER_CAST_PATTERNS` and EXTENDED with five exact-match patterns (`r"^TITLE$"`, etc). Anchored as exact-match on upcased names to minimize false positives ("Anna Title-Holder" passes). New `_assert_no_structural_tokens_in_cast(cast)` wired into `lock_cast()`. 10-test parametrized + sanity pin in `tests/test_cast_contract_rejects_structural_tokens.py`.
- **Verify:** `pytest tests/test_cast_contract_rejects_structural_tokens.py -v` (10 tests, all green). Audit doc `docs/audit-S13.1.md`.
- **Tags:** cast-contract, structural-tokens, llm-hallucination, defense-in-depth

### BUG-LOCAL-204: no enforcement of line_id uniqueness across ledger.lines[] [FIXED 02ca26c 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Two lines with the same `line_id` in `ledger.lines[]` silently overwrote each other in BOTH places ProcSFX's filename scheme keys by line_id (post-S12.1) AND every ledger write-back path (`patch_line_fields`, `apply_line_timings`) keys by line_id. The user noticed only when an episode rendered with the wrong audio in a slot -- too late to abort cleanly.
- **Cause:** No invariant in the FreezeCascade enforced `line_id` uniqueness. The writer was *expected* to emit unique ids, and historically did, but the contract was implicit.
- **Fix:** S13.2 -- new G8 invariant `_check_g8_line_id_uniqueness` in `_otr_ledger_freeze.py`, wired into `run_gap_audit` alongside G1-G7. Phase 0 collects (warn-mode), Phase 10 raises FreezeAssertionError. Diagnostic caps displayed duplicates at 5 + `(+N more)` suffix.
- **Verify:** `pytest tests/test_g8_line_id_uniqueness.py -v` (7 tests, all green). Production fixtures all pass G8 cleanly -- the writer was already emitting unique ids; G8 makes the invariant load-bearing.
- **Tags:** invariant, freeze-cascade, g8, line-id, write-back
- **Bible candidate rationale:** General lesson -- any system with paths that key by an ID needs structural enforcement that the ID is unique. Structural invariant complements the implicit producer contract.

### BUG-LOCAL-205: regex `\bV\.O\.\b` and `\bO\.S\.\b` patterns never matched [FIXED badcae5 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Pre-S7.1 `story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS` had `r"\bV\.O\.\b"` and `r"\bO\.S\.\b"` intended to catch screenplay meta-direction artefacts like `JOHN V.O.` and `JANE O.S.`. The patterns were DEAD -- they never matched. The trailing `\b` after the final `\.` never fires because Python regex word-boundary doesn't trigger between a non-word char and end-of-string. So the legacy heuristic silently allowed the artefacts through. Discovered during the S13.1 port when the new test `test_legacy_sfx_cue_artefacts_still_caught` failed on `JOHN V.O.`.
- **Cause:** Misuse of `\b` after a non-word char. Regex word-boundary semantics: `\b` matches between a word char and a non-word char. After `.` (non-word) at end-of-string, there is no word char on the right, so `\b` doesn't fire. The pattern silently rejected every input it was supposed to match.
- **Fix:** S13.1 (during port) -- dropped the trailing `\b`. New patterns: `r"\bV\.O\."` and `r"\bO\.S\."`. Verified the post-fix patterns match `JOHN V.O.` via reproduction script before commit.
- **Verify:** `python3 -c "import re; print(bool(re.search(r'\bV\.O\.', 'JOHN V.O.')))"` -> True. Regression test `test_legacy_sfx_cue_artefacts_still_caught` covers it.
- **Tags:** regex, word-boundary, port-found, dead-pattern, legacy-bug
- **Bible candidate rationale:** General lesson -- `\b` after `.` (or any non-word char) at end-of-string is a no-op. Audit any regex of the shape `\b<word>\.<word>\.\b` for the same bug. IMP-15 in the S10-S15 QA doc proposes a codebase sweep.

### BUG-LOCAL-206: `_resolve_genre("")` returned `" audio drama"` with leading space [FIXED 47eb644 2026-05-12]
- **Date:** 2026-05-12 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** During S6-A initial implementation, `_resolve_genre("")` returned `" audio drama"` (leading space) due to f-string concatenation `f"{words} audio drama"` where `words` was empty after the `.replace("_", " ").strip()` chain. Caught during pre-commit dev iteration, never shipped.
- **Cause:** Naive f-string concatenation without checking the substituted value's emptiness. `f"{''} audio drama"` = `" audio drama"`, not `"audio drama"`.
- **Fix:** S6-A pre-commit -- conditional `f"{words} audio drama" if words else "audio drama"`. Then S10.2 retired the silent fallback entirely; `_resolve_genre` now raises ValueError on empty input. The mechanical fallback survives only in `_preview_genre` (UI helper, isolated from writer / freeze paths by AST-walk test).
- **Verify:** `pytest tests/test_musicgen_style_palette.py::test_resolve_genre_empty_raises -v`.
- **Tags:** f-string, empty-input, dev-iteration, fallback-isolation
- **Bible candidate rationale:** Cosmetic in isolation; the broader S10.2 lesson (never silently degrade on production surfaces) is the standing directive #1, already canonical.

### BUG-LOCAL-207: `production_plan_or_empty` was an orphan Director-derived fallback [FIXED b443f46 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S15.5.1 pre-flight legacy audit surfaced `nodes/_otr_ledger_consumers.production_plan_or_empty(plan_json)` -- a helper that parses an optional `production_plan_json` Director-shape string and returns `{}` for empty / None / invalid input. The docstring framed it as "the graceful fallback for the optional Director input under v2." Repo-wide grep showed ZERO production callers outside the helper's own module + its own test file (`tests/test_otr_ledger_consumers.py::TestProductionPlanOrEmpty`). It was a dead Director-derived fallback that violated standing directive #11 ("no Director-derived fallbacks").
- **Cause:** The L3 consumer rewrite sprint (2026-05-09/10) preserved this helper as a "Pattern 5 demotion" path so old consumers could degrade gracefully when the Director was unwired. Subsequent voice-path-cleanbreak P2/P3 deleted the Director class + the production_plan_json sockets, but this helper was overlooked when the sprint scope tightened. The "no production callers" status was never re-checked.
- **Fix:** S23.6 -- deleted the function from `nodes/_otr_ledger_consumers.py`, removed the `__all__` entry, dropped the helper-list mention from the module docstring. Deleted `TestProductionPlanOrEmpty` (9 tests) from `tests/test_otr_ledger_consumers.py` in lockstep. Forensic comment preserved at the deletion site citing S23.6 + directive 11.
- **Verify:** `git grep -n "production_plan_or_empty" -- '*.py' '*.json'` returns zero hits across nodes/ scripts/ visual/ tests/ (excluding docs/ which carries the migration history).
- **Tags:** legacy-fallback, directive-11, audit-found, orphan-helper, voice-path-cleanbreak
- **Bible candidate rationale:** General lesson -- a "graceful fallback" surface introduced for a now-deleted upstream consumer is dead weight that lulls future contributors into thinking the upstream is still alive. Audit fallbacks tied to deleted upstreams in the same commit that deletes the upstream; or run a periodic "no production callers" sweep on helpers whose docstring mentions a known-deleted class.

### BUG-LOCAL-210: AudioGen widget vector carried a stale `{}` shifting every subsequent slot [FIXED f7a5ca0 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S24 C3 dependency audit found `OTR_BatchAudioGenGenerator.widgets_values` in the production workflow JSON was `['[]', '{}', '', 'facebook/audiogen-medium', 3.0, 3.0]` -- 6 values. INPUT_TYPES declared 6 slots in order `[script_json, episode_seed, model_id, guidance_scale, default_duration, allow_silence_fallback]`. The `{}` at position 1 was the default value of the legacy `production_plan_json` REQUIRED input that voice-path-cleanbreak P2 deleted; the widget value survived the input deletion. Net effect: positions 2-5 all map to the wrong INPUT_TYPES slot. ComfyUI's permissive type coercion masked the misalignment in soak (the runtime didn't crash) but the values getting bound to `episode_seed` (got '{}'), `model_id` (got ''), and `guidance_scale` (got the model_id string) were nonsense.
- **Cause:** When P2 deleted `production_plan_json` from INPUT_TYPES.required, the widget vector in the workflow JSON wasn't trimmed in lockstep. ComfyUI doesn't validate widget-vector length against INPUT_TYPES on load; it accepts any length and silently positionally maps what's there.
- **Fix:** C3 -- widget vector realigned to `['[]', '', 'facebook/audiogen-medium', 3.0, 3.0, False]` with the stale `{}` removed and `allow_silence_fallback=False` appended. C6 added a parametrized test (`test_no_stale_dict_residue_in_widget_vector`) that reflects each class's INPUT_TYPES against the workflow JSON and asserts shape match. Future drift fires here.
- **Verify:** `pytest tests/test_workflow_audio_widget_vectors.py::test_no_stale_dict_residue_in_widget_vector -v`. The runtime values that flow through the graph are now correct.
- **Tags:** widget-vector, position-pinned, cleanbreak-debris, audiogen, voice-path-cleanbreak
- **Bible candidate rationale:** General lesson -- when a cleanbreak deletes a REQUIRED INPUT_TYPES entry, the workflow JSON's widgets_values vector MUST be trimmed in the same commit. ComfyUI's permissive load means the misalignment ships silently. Future plans should add a step to the cleanbreak playbook: "delete input X" -> "shrink every saved-workflow widget vector by 1 at X's index". This batch's C6 widget-vector test catches the next instance.

### BUG-LOCAL-223: Sprint Phase 4 must run pytest, not just delta `EXPECTED_FAILED_NODEIDS` [FIXED — s26-downstream sweep general lesson 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S26's final QA review claimed `regression delta byte-identical` (baseline and final `EXPECTED_FAILED_NODEIDS` sets matched). The next-morning s26-downstream sweep showed the actual full-suite pytest run had **7 failures** not in any documented quarantine. The byte-identical-delta gate had passed cleanly while real test failures shipped under it. The gap: the byte-identical claim referred to the expected-fail SET, not to the actual run output. A sprint can satisfy "no new entries in EXPECTED_FAILED_NODEIDS" without ever running pytest end-to-end.
- **Cause:** S26's Phase 4 verification used the `[KNOWN-FAIL-GUARD]` summary captured during earlier phases as the regression artifact. That summary is only emitted when the conftest hook actually fires; if the cleanbreak commits' direct test runs all targeted subsets (each subset trips the 80%-collected guard and the diff returns early), the full-suite hook never runs against the cleanbreak HEAD. The S26 reviewer accepted the targeted-subset coverage as "regression delta" evidence — a category error.
- **Fix:** Phase B downstream sweep ran the full suite from `5bf9d3a`, surfaced 7 real failures (6 in the targeted-baseline file list + 1 missed by the file list -- legacy-token scan vs interior docstring), classified each per the directive's table, and shipped 4 fixes (commits `ba8a02e`, `a70aeb8`, `8181950`, `39b1670`). End-state: 2159 passed, 8 skipped, 0 failed, empty `EXPECTED_FAILED_NODEIDS`, zero `[KNOWN-FAIL-GUARD]` lines on full-suite re-run. The 7 failures had been latent on `s26-cleanbreak` HEAD; nothing in S27 caused them.
- **Verify:** Future Phase 4 verification must satisfy BOTH of: (1) `EXPECTED_FAILED_NODEIDS` delta empty AND (2) the audit-results doc contains the actual `============ N passed, M skipped, K failed ============` summary line from a full-suite run at the cleanbreak HEAD. (1) alone is not a gate.
- **Tags:** sprint-verification, expected-fail-vs-actual-fail, full-suite-gate, regression-delta, s26-downstream, general-lesson
- **Bible candidate rationale:** General lesson -- a quality gate's PASS evidence must be the artifact the gate actually defends against, not a proxy for it. `EXPECTED_FAILED_NODEIDS` defends against silent shifts in the known-fail SET; it does NOT defend against the suite acquiring new failures that aren't yet in the set. The full-suite pytest summary line is the artifact that defends against the latter. Both must appear in audit-results.md for the regression-delta gate to be satisfied. Add to the cleanbreak playbook: "Phase 4 pass gate requires actual `N passed / M failed` summary, not just delta-vs-expected-fails." Pairs naturally with BUG-LOCAL-221's bible lesson ("any quality gate that surfaces regressions must surface the classification evidence in the same artifact") -- gates demand both the result and the evidence in their final artifact.

### BUG-LOCAL-222: Audit-completeness signal: zero-hit grep on changed surfaces is the gate, not the audit [FIXED — S26 cleanbreak general lesson 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S26 cleanbreak A3 (`production_ledger.py` sfx schema scaffold deletion) shipped a single-line code change with a clean per-commit grep audit. The follow-up zero-hit acceptance check (`git grep -nE "['\"]sfx['\"]: \[\]"`) surfaced 17 test fixtures still constructing ledger dicts with `"sfx": [],` baked in -- and one of those fixtures triggered failures via a *separate* validator at `nodes/_otr_ledger_freeze.py::_REQUIRED_TOP_LEVEL_LISTS` that ALSO required the now-deleted top-level key. The audit completed correctly on the first surface and missed two coupled surfaces because the initial grep was narrow.
- **Cause:** The audit checklist focused on the named deletion target; the broader pattern audit (every place the soon-to-be-deleted shape appears) was deferred to "after the commit." When the broader grep ran, it surfaced a contractual validator that mirrored the deleted shape -- a hidden coupling that would have caused runtime errors in any pipeline run that walked the freeze-cascade audit on a v2 ledger missing the legacy field.
- **Fix:** S26-A3 extended in-commit to: (a) drop `"sfx"` from `_REQUIRED_TOP_LEVEL_LISTS`, (b) update the freeze docstring schema mapping, (c) drop `"sfx"` from 3 parametrize lists in the gap-audit test, (d) mechanically scrub all 17 test fixtures. Blast radius 19 files; below the §5 circuit-breaker bound; architectural surface unchanged.
- **Verify:** Pre-amend: `pytest tests/test_lfc_phase_0_10_gap_audit.py` 20 failed. Post-amend: 0 new failures. `git grep -nE "['\"]sfx['\"]: \[\]" nodes/ tests/` → 0 hits.
- **Tags:** audit-completeness, zero-hit-grep, validator-mirror, schema-cleanbreak, ledger
- **Bible candidate rationale:** General lesson per S25 post-mortem pattern #1 -- whenever a deletion is about a shape (not just a symbol), the audit must enumerate every code path that produces, consumes, or validates that shape. The fastest single-pass discipline is: BEFORE deletion, run a zero-hit grep across the broadest pattern (single quotes + double quotes + variable-keyed access + validator constants). If hits are non-zero, the deletion's blast radius is the actual blast radius, not the optimistic one. The validator-mirror surface (`_REQUIRED_TOP_LEVEL_LISTS`) is the most easily missed because it's a tuple of strings, not a code path that *uses* the shape; pure-data declarations don't show up in call-graph traces. Code-shape audits need string-table awareness.

### BUG-LOCAL-221: Strict-deprecation audit cannot be classified in a non-interactive cmd.exe shell [DEFERRED — S26 cleanbreak instrumentation gap 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S26 Phase 4 strict-deprecation audit (`pytest -W error::DeprecationWarning`) surfaced 1 NEW regression vs baseline known-fail set: `tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only`. The test passes under `-W ignore::DeprecationWarning` (5.25s; 1 passed). The strict-mode traceback could not be captured because the parent cmd.exe shell terminated on every retry attempt before stdout redirection flushed -- three attempts, all behaved the same way. The audit ran successfully (148 lines of test progress + summary captured to `docs/s26-cleanbreak/deprecation-audit.txt`) but the per-failure traceback that would let us classify origin (OTR vs third-party) was never readable.
- **Cause:** Likely a `-W error::DeprecationWarning` interaction with one of: (1) torch / numpy / transformers warming up inside `BatchAudioGenGenerator().generate()`; (2) a Windows-specific stdout buffering edge case under Desktop Commander's cmd.exe → child process redirect; (3) a pytest plugin (e.g. asyncio, anyio) emitting a Deprecation during teardown that the strict flag escalates to an unhandled exception inside fixture cleanup. Insufficient evidence to isolate without an interactive shell + a verbose `--tb=long --showlocals` run.
- **Fix:** Pending. Documented under `audit-results.md::"Strict DeprecationWarning audit"` with the triage path forward.
- **Verify:** When ComfyUI Desktop is booted post-cleanbreak (§11), open an interactive PowerShell window and run: `& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only -W error::DeprecationWarning --tb=long --showlocals -v` directly. The full traceback will tell us whether the warning origin is in `nodes/_otr_*` (in scope -- file a follow-up cleanbreak commit) or in `torch/numpy/transformers/pytest-asyncio/...` (out of scope -- document under `third_party_deprecations` and close).
- **Tags:** deprecation-audit, instrumentation-gap, cmd-exe-shell, strict-warning, audiogen
- **Bible candidate rationale:** General lesson -- the strict-deprecation audit's *result* line (the regression node-id) is captured even when its *traceback* is not. Plan for both. If the audit harness can write the summary but not the per-failure traceback, the harness is missing a `--tb=long --showlocals -v` mode that escalates the same audit to a known-readable form. Add to CLAUDE.md project rules: "any quality gate that surfaces regressions must surface the classification evidence in the same artifact." Otherwise the gate's pass/fail is a hand-wave; only the pass/fail and the evidence together make the gate trustworthy. Bible-pattern lesson: instrumentation completeness is part of the gate's contract, not a separate concern.

### BUG-LOCAL-220: `_fallback/` directory had no garbage collection [FIXED f4403e6+d289e29 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S24/C2's `_fallback/` redirect (correct fix for cache-poisoning on short-output renders) shipped without a cleanup hook. Across re-runs of the same episode, the `<cache_dir>/_fallback/` directory accumulated orphan silence wavs that no consumer ever read. Bounded per cue but unbounded across iterations -- effectively an unowned cache that grew until manual cleanup.
- **Cause:** The C2 fix was scoped to "stop poisoning the canonical cache_path on the short-output path." The `_fallback/` dir was framed as ephemeral but the cleanup was never wired -- a classic "we'll get to that later" oversight that lasted from S24 close to S25 open. The "soft-rollout deadlock" sibling (BUG-LOCAL-219) shares the same anti-pattern: ship the alarm wiring, defer the implementation, never wire the implementation.
- **Fix:** S25/AG-1+MG-7 -- per-episode `_fallback/` cleanup hook added immediately after `_cache_dir()` resolves in BOTH `batch_audiogen_generator.py::BatchAudioGenGenerator.generate()` AND `musicgen_theme.py::MusicGenTheme.render()`. Wipes stale `.wav` entries and logs `_fallback/ cleanup: removed N stale wav(s)` to batch_log / render_log when N > 0.
- **Verify:** Manual: drop a file in `output/otr/episodes/<ep>/audio/_fallback/foo.wav`, run AudioGen, confirm the file is gone and the log line fires.
- **Tags:** cache-cleanup, ephemeral-dir, c2-followup, audiogen, musicgen, soft-rollout-debt
- **Bible candidate rationale:** General lesson -- whenever a fix introduces a "this is ephemeral" surface (cache, scratch dir, temp file), the cleanup hook lands in the same commit. "We'll get to that later" cleanup hooks accumulate across sprints and create unowned cache surfaces that no one notices until disk fills up.

### BUG-LOCAL-219: "Soft-rollout never flipped" deadlock [FIXED 9afa54a+f592d71 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `audit_post_freeze_writeback` shipped at S18.2 with the docstring "use strict=False for the soft-rollout phase -- consumers log violations to batch_log." No consumer ever called it. `ProcSFX.strict_writeback` defaulted to False at S18.3 with the flip-criterion "once the audit walker has stayed clean for one full pipeline run" -- but the walker was never running, so the criterion was unreachable. Net: a Phase-0 alarm shipped as "off but ready to flip on once it proves itself clean", the path to "prove itself clean" required the consumer to call it, the consumer never did, and the audit slept for five sprints.
- **Cause:** Two safety surfaces shipped with flip-criteria that referenced each other in an unreachable cycle. Neither shipped with an inline owner; neither sprint after S18 audited whether the criteria had been met. The "soft rollout" framing made each individual half look like work-in-progress rather than the deadlock it was in aggregate.
- **Fix:** S25/AG-5..9 -- wired `audit_post_freeze_writeback` in soft mode at all three line-writing consumers (AudioGen, MusicGen, ProcSFX; VideoComposite writeback doesn't touch any audited line field so it's documented N/A). Flipped `ProcSFX.strict_writeback` default to True in the same sprint -- with the walker actually running, the criterion is now satisfiable and the strict default is honest about what the production contract is.
- **Verify:** `pytest tests/test_procsfx_writeback_convention.py -v` (10 passed; the two strict-default pins now lock True). Grep audit: `grep -rn 'audit_post_freeze_writeback' nodes/ --include='*.py' | grep -v _otr_ledger_consumers.py` returns 3 active call sites.
- **Tags:** soft-rollout, deadlock, audit-walker, flip-criterion, ownerless-defer
- **Bible candidate rationale:** General lesson -- any feature shipped behind a "soft rollout" flag MUST include (a) an inline flip-criterion that is *checkable* from the current commit's state, and (b) a named owner with a sprint deadline. Without both, "soft rollout" deterministically becomes "permanent off" because each sprint's planning pass treats it as "already shipped, not my problem." If the criterion references "the audit walker stays clean for one run", the same commit MUST wire the walker -- otherwise the criterion is unreachable and the flag is dead.

### BUG-LOCAL-218: Silent `model_id` repair contradicted loud-fail comment [FIXED d289e29 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `batch_audiogen_generator.py:293-294` silently mapped `str(model_id) in {"3", "3.0"}` to `"facebook/audiogen-medium"` while the INPUT_TYPES comment at `:255-259` explicitly stated "Fail loudly on bad input." The widget vector that originally triggered the drift (BUG-LOCAL-027) was already cleaned at S24/C3 -- the runtime repair had no production case left to defend against AND was masking misconfiguration AND contradicted the documented loud-fail behavior.
- **Cause:** When BUG-LOCAL-027 was fixed at the root by the C3 widget vector realignment, the downstream defender wasn't audited and deleted. It accumulated as silent-repair debris -- a code block whose triggering condition was fixed upstream months earlier.
- **Fix:** S25/AG-4 -- deleted the active `if str(model_id) in ["3", "3.0"]: model_id = "facebook/audiogen-medium"` lines; forensic comment preserved citing the deletion sprint + tying it to the original BUG-LOCAL-027 root-cause fix at S24/C3. Updated INPUT_TYPES.optional.model_id comment to remove the contradiction: loud-fail is now the literal behavior (combo-list enforces).
- **Verify:** `pytest tests/test_audiogen_legacy_gate.py::test_model_id_silent_repair_removed tests/test_audiogen_legacy_gate.py::test_model_id_input_combo_list_intact -v`.
- **Tags:** silent-repair, defender-debris, comment-code-drift, audiogen, c3-followup
- **Bible candidate rationale:** General lesson -- when a defensive code block's triggering condition is fixed at the root, audit and delete the downstream defenders. Otherwise they accumulate as silent-repair landmines that mask the next class of misconfiguration AND contradict the code's documented "loud fail" contract. Pattern: every "fix root cause" commit should grep for downstream defenders against the original bug's symptom and prune them in lockstep.

### BUG-LOCAL-217: AudioGen legacy `ledger.sfx[]` skipped C2 gate [FIXED d289e29 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** S24/C2's ghost-path fix landed on the new v2 `ledger.lines[]` writeback path at `batch_audiogen_generator.py:765-775` but not on the parallel legacy `ledger.sfx[]` loop at `:696-724`. Same field (`wav_path`), same failure mode (ledger row points at a path that was never confirmed on disk). The legacy path is dead code for current v2 producers but is the contract for any external producer still emitting the legacy shape -- and an "unused" path that ships with a bug is still a bug.
- **Cause:** C2's audit framing was "fix the new v2 writeback path." The legacy parallel loop wasn't in scope. A `git grep wav_path` against `batch_audiogen_generator.py` would have caught both paths in seconds; the audit was narrower than the field surface.
- **Fix:** S25/AG-2 -- mirrored the C2 gate `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)` onto the legacy loop. Failure branch stamps `row["wav_path"] = ""` per §6.16. Added `sfx_render_status` stamping on the legacy loop too so the audit walker (S25/AG-5) sees a consistent enum surface across both paths. CD-3 audit (S25 phase 7) confirmed zero current producers populate the legacy `sfx[]` -- the gate is conservative belt-and-suspenders until the S26.X deletion lands.
- **Verify:** `pytest tests/test_audiogen_legacy_gate.py -v`. Grep audit: 2 `os.path.isfile(cache_path)` sites in `batch_audiogen_generator.py` (v2 lines[] + legacy sfx[]).
- **Tags:** parallel-path, ghost-path, c2-followup, legacy-loop, sibling-audit
- **Bible candidate rationale:** General lesson -- when a safety fix lands on path A, audit every parallel path that handles the same ledger field. The audit should be a `git grep <field>` across the entire module (or repo, depending on field scope), not a manual walk of the changed file's neighbors. The "but it's the legacy path / dead code" framing is exactly when the bug ships unnoticed because the fix author and the reviewer both skip it.

### BUG-LOCAL-216: Style slug drift surface (writer pool vs palette) [FIXED 9679217 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `musicgen_theme._STYLE_PALETTE` (10-key dict mapping slug -> cue prompts) and `OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL` (10-tuple of slugs) were maintained as two parallel lists in two files. Drift between them caused MusicGen to halt mid-pipeline (after the script writer + freeze cascade had already spent minutes of model time + thousands of tokens) -- the writer emitted a slug that the palette didn't cover, MusicGen raised at lookup, the operator lost the run.
- **Cause:** Two surfaces, one contract, no enforcement. When a new style slug was added, the contributor had to remember to update both -- and even when they did, a future contributor renaming one entry would silently break the other.
- **Fix:** S25/MG-6 -- hoisted both sources of truth to `nodes/_otr_style_palette.py` with `STYLE_PALETTE` + `KNOWN_STYLE_SLUGS`. `musicgen_theme.py` re-imports as `_STYLE_PALETTE`; the writer pool stays its own surface but `tests/test_style_palette_drift.py` pins set-equality with `KNOWN_STYLE_SLUGS`. Freeze cascade gained an additional check in `_check_meta_invariants` that validates `meta.gen_params_initial.style ∈ KNOWN_STYLE_SLUGS` -- writer drift now surfaces at freeze time, before MusicGen even tries to look the slug up.
- **Verify:** `pytest tests/test_style_palette_drift.py -v` (5 tests: palette == known, writer pool == known, every entry has 3 cues, freeze rejects unknown, freeze accepts known).
- **Tags:** drift, source-of-truth, parallel-list, freeze-cascade, style-palette
- **Bible candidate rationale:** General lesson -- any data contract maintained as two parallel lists in two files becomes drift-prone in O(weeks). Hoist to a shared module on first drift detection (or pre-emptively if the parallel structure is visible at design time). Pin set-equality with a unit test that imports both surfaces; any future drift fires at unit-test time, not soak time.

### BUG-LOCAL-215: MusicGen NODE_CLASS_MAPPINGS prefix drift [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** no
- **Symptom:** `musicgen_theme.py` registered the node as `NODE_CLASS_MAPPINGS = {"MusicGenTheme": MusicGenTheme}` while `__init__.py:_NODE_MODULES` registers the same class under the canonical `OTR_MusicGenTheme` key. The in-module dict was dead code (the top-level `__init__.py` re-registers from the class object directly, not from the in-module mapping dict) -- but ANY test or external consumer that imported `NODE_CLASS_MAPPINGS` from the module directly got the bare name. Display name also carried a literal `"[EMOJI]"` placeholder string.
- **Cause:** The OTR_ prefix migration touched the registration site in `__init__.py` but not the leftover in-module declarations on each node file. No test pinned the in-module dict's key to match the top-level registration.
- **Fix:** S25/MG-5 -- aligned the in-module mapping to `{"OTR_MusicGenTheme": MusicGenTheme}`, dropped the `"[EMOJI]"` placeholder string from the display name. New regression test `test_musicgen_parity.py::test_node_registered_under_otr_prefix` pins the prefix; `test_node_display_name_has_no_placeholder` pins the no-placeholder rule.
- **Verify:** `pytest tests/test_musicgen_parity.py::test_node_registered_under_otr_prefix tests/test_musicgen_parity.py::test_node_display_name_has_no_placeholder -v`.
- **Tags:** node-registration, prefix-drift, dead-code, display-name
- **Bible candidate rationale:** Cosmetic in isolation; the broader lesson (any registration surface stamped in multiple files needs a test pin) is covered by the general drift-pattern entries (BUG-LOCAL-216, IMP-43).

### BUG-LOCAL-214: Silence fallback ignored cue duration [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `musicgen_theme._silent_audio_dict(sample_rate)` emitted a fixed `int(sample_rate * 0.1)`-sample clip regardless of cue. On the ImportError + `allow_silence_fallback=True` path, the 12-second opening cue and the 8-second closing cue both got 100 ms of silence -- which then propagated into EpisodeAssembler and broke the timeline (opening theme slot was 100 ms of silence then a 11.9s gap to the dialogue, closing theme was 100 ms then nothing). The bug shipped silently because the fallback path is rarely exercised (transformers is installed in production).
- **Cause:** Helper was a one-liner written under the assumption "we only need a brief placeholder." The signature didn't take a duration; every caller passed nothing; nobody noticed the EpisodeAssembler downstream needed real per-cue durations.
- **Fix:** S25/MG-4 -- `_silent_audio_dict(duration_sec, sample_rate=MUSICGEN_SAMPLE_RATE)` -- duration is now required. The ImportError fallback loop passes `CUE_DURATIONS[cue_id]`. Test `tests/test_musicgen_parity.py::test_silent_audio_dict_honors_duration` pins the contract.
- **Verify:** `pytest tests/test_musicgen_parity.py::test_silent_audio_dict_honors_duration -v`.
- **Tags:** silence-fallback, duration, musicgen, timeline, rarely-exercised
- **Bible candidate rationale:** General lesson -- a rarely-exercised code path (transformers ImportError on a box where transformers IS installed) is exactly the kind of fallback that ships with sloppy semantics for years because nobody hits it in soak. Audit fallback paths for "does this honor every contract the success path honors?" -- in this case, the cue duration is part of the EpisodeAssembler timeline contract, and the fallback emitted nonsense.

### BUG-LOCAL-213: MusicGen `music_render_status` documented but never written [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `musicgen_theme.MusicGenTheme.INPUT_TYPES` docstring and the ImportError fallback comment both promised stamping `music_render_status="fallback_silence"` on each affected ledger row. No code path actually wrote the field. Compounded by an early `return` in the ImportError fallback that bypassed the writeback block entirely -- so even if the field had been stamped in `cues[cue_id]`, the writeback never ran on the fallback path.
- **Cause:** Two bugs in one surface: (1) the writeback block didn't include `row["music_render_status"]` (it stamped wav_path, dur_s, tts_engine, etc., but not the status enum), and (2) the ImportError branch returned early before the writeback block could fire. Both bugs are easy to introduce when "soft path returns early" looks like a defensive shortcut.
- **Fix:** S25/MG-3 -- writeback block now always stamps `row["music_render_status"] = str(cue.get("_render_status") or "ok")`. ImportError branch refactored to fall through to the writeback block (added `else:` clause on the try/except so the model-loading code only runs when the import succeeded, then the writeback fires on whatever shape the cues are in). Audit walker (`audit_post_freeze_writeback`) gained `ALLOWED_MUSIC_RENDER_STATUS` enum check so typos surface.
- **Verify:** Source-level pin in `tests/test_post_freeze_writeback_audit.py` via `ALLOWED_MUSIC_RENDER_STATUS`. End-to-end pin via the wired-walker calls in Phase 5.
- **Tags:** comment-code-drift, early-return, render-status, enum, musicgen
- **Bible candidate rationale:** General lesson -- comments promising ledger behavior must be exercised by an acceptance test in the same commit. Otherwise the documentation drifts from the code and becomes a silent contract drift that future contributors believe is honored. Pattern: when a docstring says "stamps X", the same commit should add a test `assert "X" in ledger_row_dict`. Belt-and-suspenders: include the field in `audit_post_freeze_writeback`'s field list so the soft-mode walker fires on any consumer that drops the stamp.

### BUG-LOCAL-212: MusicGen writeback ghost-path [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Sibling of S24/C2's AudioGen ghost-path. `musicgen_theme.py:771` stamped `row["wav_path"] = str(cache_path)` unconditionally whenever `cache_path` was a truthy string. Cache_path was set from the canonical cache filename builder regardless of whether the save eventually succeeded -- so on `_save_wav` failure (BUG-LOCAL-211), short-output fallback, or ImportError silence-fallback, the ledger row pointed at a path that was never confirmed on disk.
- **Cause:** Same root cause as BUG-LOCAL-209 (AudioGen): writeback gated on the string variable instead of the save outcome. The implicit assumption was "if we computed a cache_path, the file is there." That's only true on the happy path; the ImportError + short-output + disk-failure paths all violate it.
- **Fix:** S25/MG-2 -- writeback now reads `save_ok = bool(cue_dict.get("_save_ok"))` and `had_cache_hit = bool(cue_dict.get("_had_cache_hit"))` and gates `row["wav_path"] = str(cache_path)` on `cache_path AND (save_ok OR had_cache_hit) AND os.path.isfile(cache_path)`. Failure paths stamp `row["wav_path"] = ""` per §6.16. The cache-hit branch in the resolve loop also stamps `cue["_had_cache_hit"] = True` so the gate distinguishes a fresh-save from a load-from-disk hit.
- **Verify:** Source-level pin via `tests/test_workflow_audio_widget_vectors.py` (BUG-LOCAL-210 sibling test) + `tests/test_musicgen_parity.py::test_save_wav_returns_bool_on_success/failure` (which gates the upstream save outcome). End-to-end: any production run with the ImportError fallback now produces `row["wav_path"] = ""` instead of a ghost path.
- **Tags:** ghost-path, writeback, c2-sibling, musicgen, save-proof
- **Bible candidate rationale:** General lesson -- already covered by BUG-LOCAL-209's promotion. The Bible entry that lands from #209 should explicitly enumerate "audit every sibling consumer with the same shape" so #212 doesn't ship in S25 the way it did. The pattern audit needs to run BEFORE the bible promotion, not after.

### BUG-LOCAL-211: MusicGen `_save_wav -> None` [FIXED f4403e6 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Direct sibling of BUG-LOCAL-209 (AudioGen). `_save_wav` in `nodes/musicgen_theme.py:281` declared `-> None`. Success path fell off after `os.replace(tmp, path)`; the except path fell off after `log.warning(...)`. Both returned implicit None. Callers couldn't distinguish a confirmed write from a swallowed exception. The writeback path keyed on `cache_path` (the string variable, always truthy after the filename builder) instead of the save's outcome -- so any save failure left a ledger row pointing at a path that was never written. BUG-LOCAL-212 is the immediate downstream consequence.
- **Cause:** Same implicit-None bug as BUG-LOCAL-209 -- function signature declared `-> None`, both code paths just ran off the end and returned None. The function was originally written as fire-and-forget (only the log.warning mattered on failure) but the writeback path quietly became a consumer of its outcome.
- **Fix:** S25/MG-1 -- signature changed to `-> bool` with explicit `return True` after `os.replace` and `return False` from the except branch. The render path captures `save_ok = _save_wav(...)` and stores it in `cue["_save_ok"]`. Writeback gates `wav_path` stamping on `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)` per §6.16.
- **Verify:** `pytest tests/test_musicgen_parity.py::test_save_wav_returns_bool_on_success tests/test_musicgen_parity.py::test_save_wav_returns_bool_on_failure -v`.
- **Tags:** implicit-none, save-proof, audiogen-sibling, musicgen, bug-209-mirror
- **Bible candidate rationale:** General lesson -- the BUG-LOCAL-209 Bible entry (when it promotes after v2.0 ships) should explicitly include an audit step "grep for `-> None` on every save-style function whose callers check truthiness" -- across the whole repo, not just the consumer that triggered the original entry. The sibling-audit gap that let #211 ship five sprints after #209 is the real lesson here; the per-function fix is mechanical once the audit fires.

### BUG-LOCAL-209: AudioGen `_save_wav` returned None on both success and failure paths [FIXED 2002958 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** C2 audit of nodes/batch_audiogen_generator.py:200 found `_save_wav` declared `-> None`. The success path fell off after `os.replace(tmp, path)`; the except path fell off after the warning log. Both returned `None`. The writeback block at L688-720 unconditionally stamped `sfx_wav_path = item["cache_path"]` whenever `cache_path` was truthy, with no proof the file actually existed on disk. Net: the ledger could carry a sfx_wav_path pointing at a path that was never written.
- **Cause:** Implicit-None on a function whose return value WAS being consumed. The writeback path didn't check the return; it checked the cache_path string variable, which was set regardless of save outcome.
- **Fix:** C2 -- _save_wav signature changed to `-> bool` with explicit `return True` after os.replace and `return False` from the except branch. The render-path now captures `save_ok = _save_wav(...)` and stores it in `item["_save_ok"]`. The writeback gates `sfx_wav_path` stamping on `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)`. Failure paths stamp `sfx_wav_path=""` per §6.16. 3 source-level pins in tests/test_audiogen_writeback_hardening.py.
- **Verify:** `pytest tests/test_audiogen_writeback_hardening.py::test_save_wav_returns_bool tests/test_audiogen_writeback_hardening.py::test_writeback_gates_sfx_wav_path_on_save_proof -v`.
- **Tags:** silent-failure, return-value, save-proof, sfx_wav_path, audiogen
- **Bible candidate rationale:** General lesson -- when a function's return value is consumed by a contract (in this case, "did the write succeed?"), the function must return an explicit bool, not implicit None. Implicit-None on a function whose callers branch on the return is silent-failure scaffolding. Audit `-> None` declarations on functions whose callers check truthiness.

### BUG-LOCAL-208: `visual/bridge.py` carried a live `production_plan_json` socket [FIXED b443f46 2026-05-13]
- **Date:** 2026-05-13 | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** S15.5.1 audit surfaced `visual/bridge.py:270` -- the OTR_VisualBridge node declared `production_plan_json` as an optional `STRING` input in INPUT_TYPES, the execute() signature accepted it as `production_plan_json: str = "{}"`, and the body wrote the value to `<job_dir>/production_plan.json` via `atomic_write_text`. Grep across the sidecar / visual worker confirmed NO downstream consumer read the file -- the bridge wrote it for an audience that no longer existed.
- **Cause:** When the legacy LLMDirector was deleted in voice-path-cleanbreak S2 (commit 249bc06) the Director's outputs were no longer being produced anywhere upstream of the bridge. The bridge's optional socket survived because S2 scoped to the audio path, and the visual bridge is sidecar-isolated -- the deletion wave didn't reach this side of the repo until the S15.5.1 audit.
- **Fix:** S23.7 -- deleted the INPUT_TYPES entry, the kwarg from execute()'s signature, and the atomic_write_text(production_plan.json) call. Module + class docstrings rewritten to reflect "script_json + scene_manifest_json" as the actual input contract. Forensic comment at the deletion site cites S23.7 + directive 11.
- **Verify:** `git grep -n "production_plan_json" visual/` returns zero hits.
- **Tags:** legacy-socket, directive-11, audit-found, sidecar-isolation, voice-path-cleanbreak
- **Bible candidate rationale:** Bookend to BUG-LOCAL-207 -- when a deletion wave is scoped to one subsystem, sidecar-isolated subsystems can carry the deletion's debris forward for sprints. A repo-wide audit grep at the END of every cleanbreak (not just inside the affected subsystem) catches this class of survival.

---

## Promotion to Bug Bible

Promotion target: `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`.
Per CLAUDE.md "Bug Log Pipeline" section, when `Bible candidate: yes`
and the fix is verified:

1. Add entry to `BUG_BIBLE.yaml` (schema: `id`, `phase`, `area`,
   `symptom`, `cause`, `fix`, `verify`, `tags`, `legacy_id`).
2. Add regression test to `tests/bug_bible_regression.py` in the
   survival guide repo.
3. Update `README.md` entry count.
4. Run the three-file contract test to confirm sync.

**Bible candidates pending promotion:**

- BUG-LOCAL-201 (cache key includes every output-determining input)
- BUG-LOCAL-202 (on-disk filename IS the identity surface for no-cache renderers)
- BUG-LOCAL-204 (structural ID-uniqueness enforcement complements implicit producer contracts)
- BUG-LOCAL-205 (`\b` after non-word char at end-of-string is a no-op; audit similar regex shapes)
- BUG-LOCAL-207 (graceful-fallback helpers tied to deleted upstreams are dead weight; audit at deletion time)
- BUG-LOCAL-208 (subsystem-scoped deletion waves leave debris in sidecar-isolated subsystems; run a repo-wide audit at the END of every cleanbreak)
- BUG-LOCAL-209 (functions whose return is consumed must declare an explicit bool, not implicit None; audit `-> None` on save/write helpers)
- BUG-LOCAL-210 (cleanbreak deleting a REQUIRED INPUT_TYPES entry MUST trim every saved-workflow widget vector at the same index in lockstep)
- BUG-LOCAL-211 (sibling-audit on every Bible-pattern landing -- the BUG-209 `-> None` audit should have run on every save-style helper repo-wide at S24 close, not just AudioGen)
- BUG-LOCAL-212 (ghost-path siblings -- a writeback safety fix on path A audits every parallel path that handles the same ledger field; covered by the sibling-audit lesson from #211)
- BUG-LOCAL-213 (comments promising ledger behavior must be exercised by an acceptance test in the same commit; otherwise documentation drift becomes silent contract drift)
- BUG-LOCAL-214 (rarely-exercised fallback paths must honor every contract the success path honors -- in particular timeline-relevant outputs like duration)
- BUG-LOCAL-216 (any data contract maintained as parallel lists in two files is drift-prone; hoist to a shared module + pin set-equality with a unit test)
- BUG-LOCAL-217 (parallel-path safety drift -- when a safety fix lands on path A, audit every parallel path via `git grep <field>`)
- BUG-LOCAL-218 (when a defensive code block's triggering condition is fixed at the root, audit and delete the downstream defenders in lockstep)
- BUG-LOCAL-219 (any "soft rollout" flag MUST include an inline flip-criterion AND an owner; the criterion must be reachable from the same commit's state, not require future wiring)
- BUG-LOCAL-220 (introducing an "ephemeral" surface -- cache, scratch dir, temp file -- requires the cleanup hook to land in the same commit; "we'll get to that later" cleanup never lands)

Per memory note ("Keep ROADMAP + BUG_LOG live; Bible promotion
waits until v2.0 ships"), batch-promote after v2.0 lands.
