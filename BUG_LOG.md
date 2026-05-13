# OTR Bug Log

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**Owner:** Jeffrey A. Brick
**Last entry:** BUG-LOCAL-210 (2026-05-13)
**Promotion target:** `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`

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

Per memory note ("Keep ROADMAP + BUG_LOG live; Bible promotion
waits until v2.0 ships"), batch-promote after v2.0 lands.
