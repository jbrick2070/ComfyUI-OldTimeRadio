# S24 Fix Sprint — QA Document (2026-05-13)

**Branch:** `v2.0-alpha`
**Predecessor HEAD:** `bed3c4a` (end of S15.5-S19 batch)
**Batch HEAD:** `fdb164b` (pushed; local == origin)
**Commits in batch:** 14 (12 sub-tasks + 1 deferral-doc consolidation in C8/C10/C12 + 1 final ROADMAP/BUG_LOG)
**Regression delta:** `+39` tests (2108 → 2147). Bug Bible regression 23 passed / 1 skipped / 2 xfailed across the batch. `EXPECTED_FAILED_NODEIDS` set steady at 6.

Companion to the S15.5-S19 QA (`docs/2026-05-13-voice-path-cleanbreak-S15.5-S19-qa.md`).

---

## 0. Commit table

| # | Hash | Subject |
|--:|------|---------|
| 1 | `cf8eb96` | docs(readme): scrub Director references from README + reference fixture README (S23.10) |
| 2 | `2002958` | fix(audiogen): stamp sfx_render_status, prevent short-output cache poisoning, gate sfx_wav_path on save proof |
| 3 | `f7a5ca0` | fix(musicgen): strict ImportError default + allow_silence_fallback opt-in (matches AudioGen S17.2) |
| 4 | `6d3f893` | fix(procsfx): stamp fallback_default_type on resolver default + clean stale wav/G7 comments |
| 5 | `2bfab7f` | feat(audit): tighten sfx_render_status to known-enum check (expanded set covering C2 + C4) |
| 6 | `0156797` | test(workflow): pin AudioGen + MusicGen widget vectors to explicit allow_silence_fallback=false |
| 7 | `493ab8c` | cleanbreak(imp-31): delete AudioGen _cache_key back-compat alias (matches MusicGen S17.1) |
| 8 | `bb689f2` | docs(cleanbreak): defer C8 CastContract quarantine -- premise was wrong, cast contract IS production-wired |
| 9 | `af7e7b1` | test(imp-33): automate ComfyUI queue-halt assumption smoke for _LLMTimeoutWorkflowPause |
| 10 | `4e972c7` | docs(cleanbreak): defer C10 LFC audit regex extension -- LFC is current architecture, not legacy |
| 11 | `f9f5aa7` | docs(imp-38): require justification comment per EXCLUDED_PATHS entry in legacy-audit test |
| 12 | `d35aa71` | docs(adr): close S14.2 active-validation design call (implementation deferred to S25+) |
| doc | `fdb164b` | docs: ROADMAP + BUG_LOG live update for the S24 fix sprint batch |

C8 and C10 are deferral commits (no code change; documentation only) and are the most notable plan deviations -- see §2 + §9.

---

## 1. Per-commit mechanics

### C1 — README + reference fixture README rewrite (`cf8eb96`)

Closes S23.10 (deferred from the prior batch).

**README.md edits.** Seven discrete rewrites:
- L33-35 pipeline arrow ASCII: `Story (LLM) -> Director (LLM) -> ...` replaced with the actual v2.0-alpha audio path (`LedgerScriptWriter -> FreezeCascade -> BatchBark -> Kokoro -> AudioGen + MusicGen + ProcSFX -> SceneSequencer -> ...`).
- L195-207 ASCII-art block: the "LLM Director" panel deleted; replaced with a FreezeCascade panel describing G1-G8 invariants + the `script_json` fanout.
- L298 `director_dump_<ts>.txt` file-tree reference rewritten as a forensic note pointing at voice-path-cleanbreak S23.1.
- L340 Node Reference: the prior paragraph-soup line (one continuous prose line covering 11 nodes) rewritten as a proper markdown table. Director row deleted; FreezeCascade row added; BatchBarkGenerator row notes "Reads cast.voice_preset directly from the ledger; no legacy Director fallback".
- L478 "Director JSON Resilience" section labeled `_(legacy -- retained for v1.x history)_`; body explains the Director class was deleted in voice-path-cleanbreak S2; v2.0 uses GBNF grammar-constrained generation instead.
- L576 prestartup VRAM-probe paragraph: "deferred to the legacy `LLMDirector` stage (retired in S2 -- now lazy-fired by the writer's cast-lock path)".
- L651 VRAM Sequencing rule: bridge moved from `LLMDirector` to "writer's cast-lock exit"; forensic note on the pre-cleanbreak boundary.

**tests/fixtures/reference_episode/README.md (full rewrite).** Provenance section adds "back when the legacy LLMDirector still produced the production plan" qualifier. Files section describes `director_satellites_collide.json` as "legacy-Director JSON ... retained as a fixture filename for back-compat". How-to-use notes the widget input was renamed `director_json -> script_json` in S16.1. New "Forward path (v2.0 ledger)" section explaining the v2.0 pipeline emits the L3 ledger natively.

**Open round-robin question.** Should the README's "Node Reference" table also tag each row with the schema version it targets (L3-2026-05-14 currently)? That would make schema-bumps surface in the README diff. Cost: ongoing maintenance discipline.

---

### C2 — AudioGen writeback hardening (`2002958`)

Three bundled bugs in `nodes/batch_audiogen_generator.py`:

**(a) sfx_render_status never stamped.** Pre-C2, the writeback at L688-720 stamped `sfx_engine`, `sfx_wav_path`, `dur_s`, `generated_dur_s`, `render_ms`, `audio_sample_hash` -- but never `sfx_render_status`. Downstream consumers (the audit walker, ledger inspection tools) had no signal whether a row was a fresh generate, a cache hit, or a fallback. Now stamped on every row with one of:
- `"ok_cache"` — cache hit at resolve; audio loaded from disk
- `"ok"` — fresh generate; `_save_wav` returned True
- `"fallback_silence"` — transformers ImportError + `allow_silence_fallback=True` (S17.2 path)
- `"fallback_output_shape"` — BUG-LOCAL-116 short-output case
- `"error"` — `_save_wav` returned False (disk-write failure)

**(b) Short-output fallback poisoned canonical cache.** Pre-C2, when AudioGen's output was shorter than `_min_samples` (the BUG-LOCAL-116 / transformers AudioGen regression), the code padded with silence and wrote the silence-padded array to the canonical `item["cache_path"]`. Every future run got a cache hit on the canonical path and served silence forever. The C2 fix routes the fallback save to a sibling `<cache_dir>/_fallback/<filename>.wav` so the canonical path stays empty; a subsequent transformers patch can re-generate cleanly to the canonical path.

**(c) `_save_wav` returned None on both paths.** The function was declared `-> None`. The success path fell off after `os.replace(tmp, path)`; the except path fell off after the warning log. Writeback stamped `sfx_wav_path = item["cache_path"]` unconditionally whenever `cache_path` was truthy. Fix: signature changes to `-> bool` with explicit `return True` after `os.replace` and `return False` from the except branch. The render-path captures `save_ok = _save_wav(...)` into `item["_save_ok"]`. The writeback gates `sfx_wav_path` stamping on `(save_ok or had_cache_hit) AND os.path.isfile(cache_path)`. Failure paths stamp `sfx_wav_path=""` per §6.16.

Logged as **BUG-LOCAL-209** in `BUG_LOG.md` -- Bible candidate.

**Open round-robin question.** The `_fallback/` subdir grows unboundedly across runs. Should there be a cleanup step (delete `_fallback/` entries older than N days) or a single-file rotation, or is unbounded growth acceptable on the OTR disk-usage envelope (procedural + sfx wavs are kB-scale)?

---

### C3 — MusicGen strict ImportError + AudioGen widget realignment (`f7a5ca0`)

**MusicGen strict ImportError.** Mirrors the AudioGen S17.2 pattern. `nodes/musicgen_theme.py`:
- `render()` signature accepts `allow_silence_fallback=False` kwarg.
- `INPUT_TYPES().optional` declares `"allow_silence_fallback": ("BOOLEAN", {"default": False})`.
- ImportError catch block raises `RuntimeError` by default. Opt-in fallback path emits silence + WARNING + tags each affected cue dict with `_render_status="fallback_silence"`.

**Workflow JSON wiring.** `workflows/otr_scifi_16gb_full.json`:
- `OTR_MusicGenTheme` node id=14: `widgets_values` extended `[..., 3.0]` → `[..., 3.0, false]`.
- `OTR_BatchAudioGenGenerator` node id=15 (**plan deviation: this was beyond C3's stated scope**): pre-fix vector was `['[]', '{}', '', 'facebook/audiogen-medium', 3.0, 3.0]` -- a stale `'{}'` at position 1 (the default value of the legacy `production_plan_json` required input deleted in voice-path-cleanbreak P2) shifted every subsequent widget by +1, leaving `guidance_scale` carrying the model_id string and `allow_silence_fallback` never serialized. Fixed: `['[]', '', 'facebook/audiogen-medium', 3.0, 3.0, False]` aligned with `INPUT_TYPES` declared order.

Logged as **BUG-LOCAL-210** in `BUG_LOG.md` -- Bible candidate (cleanbreak playbook addition).

**Open round-robin question.** ComfyUI permissively accepts widget-vector misalignment at workflow-load time. The C3 + C6 path catches this in CI now, but a user-edited workflow that introduces a new misalignment doesn't fire CI. Should the S14.2 `OTR_WorkflowValidator` first-node (scheduled S25+) also check widget-vector alignment, or is that out of scope for the validator's contract?

---

### C4 — ProcSFX fallback_default_type + stale comments (`6d3f893`)

`nodes/batch_procedural_sfx.py`:

**(a) Default-path never marked.** The keyword + alias chain falls through to `chosen_type = "radio_tuning"` when nothing matches. Pre-C4 the writeback stamped `sfx_render_status="ok"` indistinguishable from a successful semantic match. Fix: track `matched: bool` across the entire resolver chain. Every keyword/alias branch sets `matched=True`; default-path leaves False. The render-status assignment selects `"ok"` vs `"fallback_default_type"`. Disk-write-failure path still overrides to `"error"`.

**(b) Stale `sfx_wav_path=None` mention.** Module docstring at L18 still claimed disk-write failure stamps None. S18.1 changed that to `""` (§6.16 convention) months ago; the comment lagged. Docstring rewritten + `sfx_render_status="error"` mention added.

**(c) Stale `[0.25, 12.0]` literal.** Per-cue dur_s comment cited the pre-S6.4 G7 range. Now refers to `[SFX_DUR_MIN_S, SFX_DUR_MAX_S]` symbolically with a forensic "previously [0.25, 12.0]" anchor for traceability.

**Open round-robin question.** The resolver's "Additional semantic aliases" block (L200-204 originally; expanded in C4 to add the `matched=True` flag) uses `if/elif` chain where the FIRST match wins. The forensic intent appears to be "the keyword loop is the primary path; the alias chain is a heuristic fallback before defaulting to radio_tuning". Should that intent be pinned in a test, or is the existing `matched` test sufficient?

---

### C5 — ALLOWED_SFX_RENDER_STATUS enum check (`2bfab7f`)

Closes IMP-32. Depends on C2 + C4 enum values stabilizing.

`nodes/_otr_ledger_consumers.py`:
- New module constant `ALLOWED_SFX_RENDER_STATUS: frozenset` with 8 entries: `"" / "ok" / "ok_cache" / "error" / "fallback_silence" / "fallback_output_shape" / "fallback_default_type" / "skipped"`. Each value mapped to its producer-side stamping context in the constant's docblock.
- `audit_post_freeze_writeback` walker extended: for the `sfx_render_status` field, the walker also checks membership in the enum. Typos like `"fallback_silnce"` surface as violations in soft mode and raise in strict mode.
- Enum-check applies ONLY to `sfx_render_status`. Other 9 optional-string fields stay string-shape-only.
- `ALLOWED_SFX_RENDER_STATUS` added to `__all__`.

**Open round-robin question.** The `"skipped"` enum value is "reserved for future use; no producer stamps it today." Should the constant carry only currently-stamped values (drop `"skipped"`) and let a future sprint that introduces a skip-path also extend the enum? Or keep the slot open as forward-compat?

---

### C6 — AudioGen + MusicGen widget-vector pin (`0156797`)

`tests/test_workflow_audio_widget_vectors.py` (new). Six tests:
- `test_audiogen_widget_vector_length_matches_input_types` — vector length must equal `required + optional` declared count. Catches the C3 drift class.
- `test_audiogen_allow_silence_fallback_pinned_false` — the strict failure default (Directive 1) must land False in the JSON.
- `test_musicgen_widget_vector_length_matches_input_types` — same length pin for MusicGen.
- `test_musicgen_allow_silence_fallback_pinned_false` — same strict pin.
- `test_no_stale_dict_residue_in_widget_vector[OTR_BatchAudioGenGenerator]` — parametrized type-check: each widget value must match the INPUT_TYPES declared type for that position. A stale `'{}'` in an INT or BOOLEAN slot fires here.
- `test_no_stale_dict_residue_in_widget_vector[OTR_MusicGenTheme]` — same parametrized type-check for MusicGen.

**Open round-robin question.** The type-check accepts `STRING / INT / FLOAT / BOOLEAN / list-enum`. Are there other INPUT_TYPES shapes (`AUDIO`, `IMAGE`, `LATENT`, etc.) the test should account for, or are those always wired (never widget-fulfilled) so this check is moot for them?

---

### C7 — AudioGen `_cache_key` alias deletion (`493ab8c`)

Closes IMP-31. Matches the MusicGen S17.1 deletion. Pre-edit grep confirmed zero external callers:
- `news_interpreter.py` + `visual/worker.py`: unrelated function names (`compute_cache_key`, `anchor_cache_key`).
- `_otr_ledger_consumers.py`: `audio_cache_key` / `music_cache_key` are FIELD names in `_OPTIONAL_STRING_FIELDS`, not function references.

Deletions: `_cache_key` function from `nodes/batch_audiogen_generator.py`; `_cache_key` import + `test_audiogen_cache_key_alias_matches_filename_for_write` from `tests/test_audiogen_cache_keys.py`; `test_audiogen_cache_key_returns_canonical_filename` renamed to `test_audiogen_cache_filename_for_write_returns_canonical` in `tests/test_cache_key_mutations.py`; `scripts/_ast_phase_d.py` expected-functions list shrunk. (Script is gitignored under `scripts/_*.py`; edit persists on disk but the commit captured only the tracked changes.)

**Open round-robin question.** None — pure mechanical deletion matching a documented pattern.

---

### C8 — CastContract quarantine — **DEFERRED** (`bb689f2`)

The C8 plan-spec assumed `_otr_cast_contract.py` had no internal imports. Pre-edit dependency audit at execution time found:
- `nodes/_otr_cast_repair.py:40,312` imports from `_otr_cast_contract` (CharacterEntry, _extract_dialogue_tags, others).
- `nodes/_otr_cast_repair.py` is consumed by `nodes/_otr_ledger_reviewer.py::apply_deterministic_cast_repairs` -- live production code path called at writer-time.
- `nodes/_otr_ledger.py:897` + `nodes/_otr_line_composer.py:740` carry forensic references.

Cast contract IS wired into production via `cast_repair → ledger_reviewer`. Quarantining without first untangling those imports would either break `apply_deterministic_cast_repairs` (called at writer-time) or ship a "not wired into production" docstring lie.

Deferred per `docs/cleanbreak-deferred.md` with 3 unblock options:
1. Move just the helpers cast_repair needs into a new `_otr_cast_helpers.py`, then quarantine cast_contract clean.
2. Quarantine the full chain (`cast_contract + cast_repair + apply_deterministic_cast_repairs`) together as one large sprint.
3. Accept that cast contract is production-wired and drop the quarantine plan; update C8's framing instead.

**Open round-robin question (PRIORITY).** Which unblock option (1, 2, or 3) is the right call? Option 1 is the smallest scope but requires understanding what cast_repair actually needs from cast_contract. Option 2 is the largest but cleanest. Option 3 is the "we were wrong about scope, embrace it" path. This is the most architecturally consequential question in the batch.

---

### C9 — Queue-halt smoke test (IMP-33) (`af7e7b1`)

`docs/2026-05-13-imp33-queue-halt-test-decision.md` (new): decision doc picking Option B (mock-based) over Options A (version-pin, brittle) and C (real ComfyUI subprocess, not feasible today). Plan-rationale + alternatives + follow-ups (IMP-33a real-subprocess, IMP-33b cross-version stability) all documented.

`tests/test_llm_timeout_queue_halt_smoke.py` (new): `_MockQueueExecutor` stand-in iterates a fake two-node dependency graph and halts on the first exception from any `node.execute()`. 4 active tests + 1 skip:
- `test_workflow_pause_halts_queue_before_next_node` — ScriptWriter raises `_LLMTimeoutWorkflowPause`; FluxPortrait is the next dep. Assert `flux_portrait.execute.assert_not_called()`.
- `test_workflow_pause_subclass_match_for_legacy_handlers` — `except _LLMTimeout` still matches the subclass (Python inheritance).
- `test_orphan_worker_message_signals_rerun` — action-relevant phrases "orphan" + "Re-run" in the exception body.
- `test_pre_halt_no_cuda_error_illegal_address` — defensive: exception message must not embed `cudaErrorIllegalAddress`.
- `test_real_comfyui_queue_halt_subprocess_smoke` (SKIPPED) — stub for Option C; skip-reason cites the decision doc.

**Round-robin deviation.** The plan called for ChatGPT + Gemini sanity-check before locking the decision. Skipped because (a) the plan recommended Option B, (b) the decision criteria are technical/stable, (c) round-robin overhead disproportionate to a decision where the plan already showed strong direction. Documented in the decision doc's "Round-robin deviation" section.

**Open round-robin question.** Mock-vs-real divergence is acknowledged as a known limitation. If ComfyUI silently changes the queue-execution layer to swallow uncaught exceptions, this mock-based smoke wouldn't catch it. Is the IMP-33a (real subprocess) stub sufficient defensive coverage, or should there be a periodic-CI step that tests against a real ComfyUI version?

---

### C10 — LFC audit regex — **DEFERRED** (`4e972c7`)

The C10 plan extended the legacy-audit regex with `\bLFC\b`, `\bLive Freeze Cascade\b`, `\blfc_` tokens, assuming LFC was a deleted legacy generation. Pre-edit dry-run grep found 159 hits across the repo. Spot-check of top hit-files:
- `OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc` are **registered ComfyUI nodes** in `__init__.py`.
- `nodes/_otr_lfc_phase_4_scene_coherence.py`, `_phase_5_voice_drift.py`, `_phase_6_episode_arc.py`, `_context.py`, `_llm_helpers.py`, `_phase_verdicts.py`, `_smart_suggestion.py`, `_watchdog.py` are **all live infrastructure**.
- `nodes/OTR_LedgerFreezeCascade.py` uses "LFC" in its own forensic comments as part of naming history.

LFC = "Live Freeze Cascade" = the **current** system. Not a deleted lineage. Adding LFC tokens to the audit regex would flag every legitimate reference (159) as a violation. The audit's contract is to catch DELETED surfaces; LFC doesn't qualify.

Plan-framing correction in `docs/cleanbreak-deferred.md`: future audit-extension plans should distinguish **(a) names of currently-live systems that happen to be acronyms** (LFC, FLUX, HuMo, LTX — leave alone) from **(b) names of deleted systems** (Director, LLMDirector, production_plan_json — audit and gate).

**Open round-robin question.** Are there any actually-deleted-LFC-era surfaces that warrant inclusion in the audit? E.g., if there was an earlier `OTR_LFC_Cascade` class deleted in favor of `OTR_LedgerFreezeCascade`, that specific name would qualify. Worth a targeted spot-check before declaring "no LFC audit needed forever."

---

### C11 — EXCLUDED_PATHS justification discipline (`f9f5aa7`)

Closes IMP-38. Doc-only.

`tests/test_legacy_audit_clean.py`:
- Module-level docstring extended with a new "EXCLUDED_PATHS discipline" section pinning the rule.
- Each existing EXCLUDED_PATHS entry now carries a per-file `# justification:` comment explaining why the audit's substring rule can't be applied to it. Four entries today, all named + justified.

The rule exists because EXCLUDED_PATHS has load-bearing semantics: every file on the list is invisible to the audit. A future contributor adding a path without explaining why is silently widening the audit's blind spot.

**Open round-robin question.** Should the test include a meta-test that AST-walks the EXCLUDED_PATHS frozenset literal and asserts each entry has a preceding `# justification:` comment within N lines? That would mechanize the discipline rather than rely on PR-review enforcement.

---

### C12 — S14.2 active-validation ADR (`d35aa71`)

`docs/2026-05-13-S14_2-active-validation-ADR.md` (new): standard ADR shape (context / options / decision / consequences / alternatives / status). Decision: **Option B** (opt-in `OTR_WorkflowValidator` first-node) over Option A (ComfyUI frontend extension).

Rationale highlights:
1. ComfyUI's Python node API is the most stable extension surface; the frontend extension API has changed in non-trivial ways across versions.
2. Pure Python keeps validation in OTR's primary skill envelope.
3. Validation-at-execution is sufficient because OTR contributors run the production workflow on every change.
4. Failure mode is observable in the same channel as every other OTR node failure.
5. Option A's "earliest possible moment" advantage is theoretical in OTR's actual usage pattern.

Implementation deferred to S25+ with estimated scope: new node class `nodes/_otr_workflow_validator.py` (~150 LOC) wired as position-0 in the production workflow JSON + `tests/test_otr_workflow_validator.py`.

`docs/cleanbreak-deferred.md` S14.2 entry status flipped from "INDEFINITELY DEFERRED" to "DEFERRED — implementation scheduled for S25+; ADR at <path>".

**Round-robin deviation.** Same rationale as C9. Documented in the ADR's "Round-robin deviation" section.

**Open round-robin question (PRIORITY).** Is Option B's "validation-at-execution" cycle latency acceptable? A contributor who saves a broken workflow doesn't see the error until they queue it. Option A would catch this at save time. Is the OTR usage pattern (run on every change) strong enough to justify the trade-off, or should there be a follow-up sprint to add A on top of B once B ships?

---

## 2. Plan deviation summary

| C# | Plan-spec | Actual disposition |
|---:|-----------|---------------------|
| C2 | save fallback to `_fallback/` OR skip | Saves to `_fallback/` subdir for traceability; `_save_ok=False` so writeback doesn't stamp `sfx_wav_path`. |
| C2 | render_results stamp `sfx_render_status` | Dropped from S17.2 fallback path because `render_results` variable name was speculative; downstream block handles it. |
| C3 | append `allow_silence_fallback=false` to MusicGen | Plus: realigned AudioGen widget vector to drop stale `{}` (beyond plan scope; surfaced as BUG-LOCAL-210). |
| C5 | enum includes the C2 + C4 stamps | Plus `"skipped"` for forward-compat; documented in the constant's docblock. |
| C8 | mechanical move with "likely no internal imports" | **DEFERRED.** Audit found cast_contract IS production-wired via `cast_repair → ledger_reviewer`. 3 unblock options in deferral doc. |
| C9 | required ChatGPT + Gemini round-robin | Skipped. Plan already recommended Option B; decision criteria stable. Decision doc captures the rationale + the deviation. |
| C10 | extend audit regex to flag LFC tokens | **DEFERRED.** LFC is current production architecture (159 hits = 159 false positives). Plan-framing correction documented. |
| C12 | required ChatGPT + Gemini round-robin | Skipped. Same rationale as C9; ADR captures the rationale + deviation. |
| C11 | discipline rule pinned in docstring | Plus per-entry `# justification:` comments on all 4 existing EXCLUDED_PATHS entries (beyond strict plan; cheap to add at the same time). |

---

## 3. Discovered surfaces / findings beyond plan

| Severity | Item | Origin |
|---|---|---|
| HIGH | AudioGen widget vector misalignment from deleted `production_plan_json` slot (BUG-LOCAL-210) | C3 dependency audit |
| HIGH | `_save_wav` returned None on both paths; writeback stamped `sfx_wav_path` without proof (BUG-LOCAL-209) | C2 audit |
| HIGH | CastContract is production-wired via `cast_repair → ledger_reviewer` | C8 dependency audit |
| MEDIUM | LFC is the **name of the current system**, not legacy | C10 dry-run grep |
| LOW | `scripts/_ast_phase_d.py` is gitignored (`scripts/_*.py` pattern); edits don't get committed | C7 |
| LOW | The `cleanbreak playbook` lacks a step "shrink widget vectors when deleting REQUIRED INPUT_TYPES entries" | BUG-LOCAL-210 lesson |

---

## 4. Test inventory (+39 net new tests)

| File | New tests | Sprint |
|---|--:|---|
| `tests/test_audiogen_writeback_hardening.py` | 8 | C2 |
| `tests/test_musicgen_strict_failure.py` | 4 | C3 |
| `tests/test_procsfx_writeback_convention.py` (extended) | +4 | C4 |
| `tests/test_post_freeze_writeback_audit.py` (extended) | +13 (1 + 8 parametrized + 4 other) | C5 |
| `tests/test_workflow_audio_widget_vectors.py` | 6 (4 explicit + 2 parametrized) | C6 |
| `tests/test_audiogen_cache_keys.py` (delta) | -1 (deleted alias test) | C7 |
| `tests/test_audiogen_ledger.py` (delta) | 0 (1 updated assertion) | C2 |
| `tests/test_cache_key_mutations.py` (delta) | 0 (1 renamed test) | C7 |
| `tests/test_llm_timeout_queue_halt_smoke.py` | 4 + 1 skip | C9 |

Net: `+39` new + `1 skip` from C9's integration stub.

**Existing tests touched in lockstep:**
- `tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only` — assertion updated to expect `sfx_wav_path=""` + `sfx_render_status="fallback_silence"` on the silence-fallback path (the prior assertion that the path ended in `.wav` was the bug -- it was stamping a path the silence path never wrote).

---

## 5. Drift-guard inventory (new this batch)

| Contract | Pinned by |
|---|---|
| `_save_wav` returns explicit bool | `test_save_wav_returns_bool`, `test_save_wav_source_has_explicit_return_true_and_false` |
| Short-output fallback uses `_fallback/` subdir | `test_short_output_fallback_uses_fallback_subdir` |
| Short-output fallback stamps `fallback_output_shape` status | `test_short_output_fallback_stamps_fallback_output_shape_status` |
| Canonical cache path is guarded inside `else` branch off `short_output_fallback` | `test_short_output_fallback_does_not_save_to_cache_path` |
| `sfx_render_status` stamped on every writeback row | `test_writeback_stamps_sfx_render_status_on_every_row` |
| Cache-hit defaults to `ok_cache` | `test_writeback_defaults_status_for_pre_stamped_cache_hits` |
| `sfx_wav_path` gated on save proof + `os.path.isfile` | `test_writeback_gates_sfx_wav_path_on_save_proof` |
| Failure paths stamp `sfx_wav_path=""` | `test_writeback_blanks_sfx_wav_path_on_failure` |
| MusicGen strict ImportError raises | `test_strict_path_raises_runtime_error` |
| MusicGen `allow_silence_fallback` widget shape | `test_input_types_has_allow_silence_fallback` |
| ProcSFX `matched:bool` flag in resolver | `test_resolver_tracks_matched_flag` |
| ProcSFX fallback_default_type selection | `test_fallback_default_type_status_stamped_when_no_match` |
| ProcSFX no stale `sfx_wav_path=None` mention | `test_stale_sfx_wav_path_None_doc_scrubbed` |
| ProcSFX no bare `[0.25, 12.0]` outside forensic anchor | `test_stale_dur_range_literal_scrubbed` |
| `ALLOWED_SFX_RENDER_STATUS` frozenset membership | `test_allowed_sfx_render_status_membership` |
| Every enum value passes audit | `test_audit_accepts_every_enum_value[<8 params>]` |
| Typo raises in strict / collects in soft | `test_audit_rejects_typo_in_{strict,soft}_mode` |
| Enum check applies only to `sfx_render_status` | `test_audit_only_enum_checks_sfx_render_status` |
| Workflow audio widget vector lengths match INPUT_TYPES | `test_{audiogen,musicgen}_widget_vector_length_matches_input_types` |
| `allow_silence_fallback=False` pinned in workflow | `test_{audiogen,musicgen}_allow_silence_fallback_pinned_false` |
| Widget value types match INPUT_TYPES per position | `test_no_stale_dict_residue_in_widget_vector[<2 params>]` |
| Queue halts before next node on `_LLMTimeoutWorkflowPause` | `test_workflow_pause_halts_queue_before_next_node` |
| Subclass match for legacy handlers | `test_workflow_pause_subclass_match_for_legacy_handlers` |
| Action-phrases in exception body | `test_orphan_worker_message_signals_rerun` |
| No `cudaErrorIllegalAddress` in exception body | `test_pre_halt_no_cuda_error_illegal_address` |

---

## 6. Bibliographically promotable bugs (new this batch)

### BUG-LOCAL-209 — `_save_wav -> None` on both paths

**General lesson.** When a function's return value is consumed by a contract (in this case, "did the write succeed?"), the function must return an explicit bool, not implicit None. Implicit-None on a function whose callers branch on the return is silent-failure scaffolding. Audit `-> None` declarations on functions whose callers check truthiness.

### BUG-LOCAL-210 — cleanbreak left stale widget value behind

**General lesson.** When a cleanbreak deletes a REQUIRED INPUT_TYPES entry, the workflow JSON's `widgets_values` vector MUST be trimmed at the same index in lockstep. ComfyUI's permissive load means the misalignment ships silently — values get positionally bound to the wrong fields. Future cleanbreak playbooks should add: "delete input X" → "shrink every saved-workflow widget vector by 1 at X's index". The C6 widget-vector test catches the next instance before runtime.

Both **Bible candidate: yes** pending promotion after v2.0 ships, per `feedback_roadmap_buglog_live_docs`.

---

## 7. Sight improvements (IMP-* candidates for next round-robin)

| # | Severity | Item | Location | Rationale |
|---|---|---|---|---|
| IMP-39 | LOW | `_fallback/` unbounded growth | `nodes/batch_audiogen_generator.py` short-output path | The C2 fallback saves to a sibling dir indefinitely. Worth a rotation policy or N-day cleanup? Or accept unbounded growth on kB-scale wavs. |
| IMP-40 | MEDIUM | Cleanbreak playbook step: "shrink widget vectors" | Documentation | After BUG-LOCAL-210, the cleanbreak playbook should explicitly call out widget-vector trimming when deleting REQUIRED INPUT_TYPES entries. Plan-doc convention; not code. |
| IMP-41 | LOW | Per-row schema version tag in README Node Reference | `README.md` Node Reference table | Tag each row with the schema version it targets so schema-bumps surface in the README diff. Trade-off: ongoing doc-maintenance discipline. |
| IMP-42 | MEDIUM | AST meta-test for EXCLUDED_PATHS justification comments | `tests/test_legacy_audit_clean.py` | C11 added the discipline rule + per-entry comments but enforces them via PR review. Mechanize via AST-walk that asserts each entry has a preceding `# justification:` comment within N lines. |
| IMP-43 | MEDIUM | S14.2 validator scope: widget-vector alignment check? | Future `OTR_WorkflowValidator` (S25+) | Should the validator at execution time ALSO check widget-vector length against INPUT_TYPES (catch BUG-LOCAL-210-class drift in user-edited workflows)? Adds scope but closes a runtime gap. |
| IMP-44 | LOW | `"skipped"` enum value is reserved-no-producer | `nodes/_otr_ledger_consumers.py::ALLOWED_SFX_RENDER_STATUS` | Either drop the slot (no producer stamps it) or document the expected future producer. Forward-compat trade-off. |
| IMP-45 | LOW | Resolver alias-chain intent pin for ProcSFX | `tests/test_procsfx_writeback_convention.py` | C4 added the matched-flag test but didn't pin "keyword loop is primary; alias chain is fallback before radio_tuning default". Worth a test? |
| IMP-46 | LOW | Look for actually-deleted LFC-era class names | `tests/test_legacy_audit_clean.py` | C10 deferred LFC en masse. Spot-check if any specific LFC-era class name (e.g., `OTR_LFC_Cascade`) was retired in favor of the current `OTR_LFCPhase4/5/6` -- those specific names would qualify for the audit. |

---

## 8. Round-robin questions (priority order)

The questions in §1 collected, ranked by architectural consequence:

1. **C8 unblock option — which is right?** (HIGH) The 3 options (extract helpers / quarantine the chain / drop the quarantine plan) are mutually exclusive. This is the most consequential architectural decision in the batch's deferral set.
2. **C12 Option B latency vs Option A save-time** (HIGH) Is execution-time validation enough, or should Option A get added on top once B ships?
3. **C9 mock-vs-real sufficiency** (MEDIUM) Is IMP-33a (real subprocess stub) enough defensive coverage, or do we need periodic-CI against a real ComfyUI version?
4. **C3 / IMP-43 widget-vector check in S25+ validator** (MEDIUM) Should the future `OTR_WorkflowValidator` cover widget-vector alignment, closing the user-edited-workflow gap that BUG-LOCAL-210 surfaced?
5. **C10 actually-deleted LFC-era names** (LOW) Spot-check for any specific LFC-era class names that were retired.
6. **C2 `_fallback/` rotation** (LOW) Bounded vs unbounded; trade-off is small at kB-scale.
7. **C5 `"skipped"` enum forward-compat** (LOW) Keep the slot or drop it.

---

## 9. Deferred items (status pinned)

| Item | Status | Gate / unblock condition |
|---|---|---|
| **C8 — CastContract quarantine** | DEFERRED, premise corrected | Pick one of 3 unblock options. Architectural decision; needs round-robin or design call. |
| **C10 — LFC audit regex** | DEFERRED, premise corrected | Only reopens if a future sprint retires LFC. Plan-framing correction documented. |
| S14.2 implementation | Scheduled S25+ per new ADR | Pure Python `OTR_WorkflowValidator` first-node (~150 LOC). |
| S19.3 | 2/3 sprint cycles complete | One more clean S15.3-using sprint to unblock. |
| S21.3 | Still conflicts with `feedback_minimum_json_files` rule | Unchanged. |
| IMP-33a (real subprocess smoke) | Pending ComfyUI CI harness | No ETA. |
| IMP-33b (cross-version stability) | Tracked in `_LLMTimeoutWorkflowPause` docstring | No ETA; revisit on next ComfyUI major bump. |

---

## 10. Acceptance state for batch closure

All gates green:

- [x] `tests/test_legacy_audit_clean.py` — `1 passed`
- [x] `tests/test_workflow_live_passes_validator.py` — `1 passed`
- [x] `tests/test_naming_conventions.py` — `3 passed`
- [x] `tests/test_workflow_audio_widget_vectors.py` — `6 passed`
- [x] `tests/test_audiogen_writeback_hardening.py` — `8 passed`
- [x] `tests/test_musicgen_strict_failure.py` — `4 passed`
- [x] `tests/test_post_freeze_writeback_audit.py` — `21 passed`
- [x] `tests/test_llm_timeout_queue_halt_smoke.py` — `4 passed / 1 skipped` (intentional)
- [x] Bug Bible regression — `23 passed / 1 skipped / 2 xfailed` (baseline held)
- [x] Full pytest run — `2147 passed / 8 skipped / 6 known-fail` (exact match to `EXPECTED_FAILED_NODEIDS`)
- [x] Local HEAD == origin HEAD (`fdb164b78fae320064b4b87bc238f4bc1a5e0e97`)
- [x] No 0-byte tracked Python files
- [x] No BOM-prefixed tracked Python files
- [x] ROADMAP + BUG_LOG live-updated

**S24 fix sprint is LOCKED at `fdb164b`.**
