# OTR v2.0 Bug Log

Active bug log for the v2.0 build. Every bug gets logged the moment it is found.
Entries are never deleted.

---

### NON-BUG-2026-05-11: script-writing-architecture Phases 0-3 shipped — phantom-name gate, prompt enrichment, episode budget, progressive ledger, three-pass cast-gated reviewer

- **Date:** 2026-05-11 | **Phase:** v2.0-alpha post-news_interpreter | **Bible candidate:** no (sprint milestone)
- **Symptom:** CLOSES the script-writing-architecture round-robin design (`docs/2026-05-10-script-writing-architecture__00_question.md` + uploaded `04_synthesis (1).md`). Composer no longer ships phantom names silently into the ledger; reviewer catches anything that slips through. Outline structure now driven by an explicit `EpisodeBudget` rather than the prior `target_length` combo.
- **Phase 0 (name-roster gate, detect + flag, no reroll):**
  - `_otr_line_composer.py`: new `LineResult(text, compose_flags)` return type. `build_allowed_roster(cast_rows, key_terms)` builds the UPPERCASE roster (cast + ANNOUNCER + key_terms only — §6.A Option 1, strict). `detect_phantom_names(text, speaker, roster)` runs three heuristics (ALL-CAPS, titled "Dr. Patel" form, Title-Case bigrams mid-sentence — skip sentence-start). Phantoms flagged via `LineResult.compose_flags = ("phantom_name:<token>", ...)`. Composer NEVER rerolls on a name violation (cast is locked; LLM cannot invent a different correct name). `aggregate_compose_flags(led.data)` rolls up by kind for `meta.compose_flag_summary`.
  - `OTR_LedgerScriptWriter.py`: roster built once after cast lock + news_interpreter; passed to every `LineRequest`; per-beat flags stamped via `patch_line_fields`. End-of-run aggregate.
  - Tests: `tests/test_phase0_name_roster.py` (roster build + phantom detect + composer flag-stamp + aggregate).
- **Phase 1 (composer prompt enrichment + sliding window):**
  - `LineRequest` gets `style_descriptor / outline_spine / character_voice_card`. Prompt restructured static-first (style → canon → spine → roster → CHARACTER → recent dialogue → WRITE LINE) for future KV-cache reuse. `render_outline_spine` + `build_voice_card` helpers. `LAST_LINES_WINDOW` 3 → 5.
  - Writer wires all three fields per beat.
  - Tests: `tests/test_phase1_composer_prompt.py`.
- **Phase 2A (widget delta + episode budget + outline validators + arc_phase):**
  - NEW `_otr_episode_budget.py`: `default_act_count(target_words)` (1/2/3 default tiers), `max_act_count(target_words)` (cap = `target_words // 50`, ceiling 7). `ACT_COUNT_CONFIG[1..7]` with arc_phases / per-act word fractions / voiced-beats-per-act / per-beat word range. `ARC_PHASE_GUIDANCE` for composer. `EpisodeBudget` dataclass. `compute_episode_budget(target_words, act_count, include_act_breaks, num_characters)` validates the combo (rejects below-default, above-max, below-30-words) and derives all per-phase numbers.
  - `_otr_outline.py`: `Beat.arc_phase: Optional[str]` (None on back-compat outlines; required-non-None when `req.budget` is set). Beats cap raised 24 → 32 for 6-/7-act room. `OutlineRequest.budget: object` (lazy duck-type to avoid module-load coupling). New `validate_outline_against_budget`: 8 validators (per-phase word totals ±20%, per-phase beat counts ±1/+2, per-beat word-range, arc_phase ordering monotonic, music_inter count match, announcer count match, every speaker in cast). §6.E word-drift validator is WARN-only at ±25%. `generate_outline` retry loop runs validators and rerolls on any hard violation.
  - `_otr_line_composer.py`: `LineRequest.arc_phase` field. Prompt block `ARC PHASE: <phase>\n  <guidance>` rendered when set.
  - `OTR_LedgerScriptWriter.py`: new `act_count` INT widget appended at END of optional INPUT_TYPES (preserves legacy `widgets_values` positional mapping). `_resolve_inputs` resolves 0 → `default_act_count(target_words)`. Run-time `compute_episode_budget` call; budget threaded through OutlineRequest; `beat.arc_phase` plumbed to every LineRequest.
  - NEW `web/js/otr_act_count_widget.js` + `WEB_DIRECTORY = "./web"` in `__init__.py`. Live UI clamp of `act_count` choices to `[default..max]` band as `target_words` changes. Python is authoritative; JS is UI feedback only.
  - Tests: `tests/test_phase2a_episode_budget.py` (default/max tables, compute_episode_budget happy + reject branches, ACT_COUNT_CONFIG shape, schema arc_phase + 32-cap, prompt block render, all 8 validator branches, composer arc_phase prompt).
- **Phase 2B (progressive ledger writes):**
  - `production_ledger.py`: `init_lines_from_outline(outline, char_id_by_name)` pre-stamps skeleton rows (voiced beats `text=""` pending compose; non-voiced beats stamped from sfx_cue/intent at init time). `update_line_text(beat_id, text)` in-place text update + char/word count recompute. Existing `set_lines` retained for non-writer callers.
  - Writer: drops the in-memory `line_rows` accumulator. Per-beat loop now calls `update_line_text` + `patch_line_fields` (compose_flags / char_id / traits) + `led.save()` after every line. News-wiring overlay rewritten to operate on `led.data["lines"]`.
  - Tests: `tests/test_phase2b_progressive_ledger.py` (skeleton init, in-place update, synthetic crash recovery, atomic-save semantics, set_lines back-compat).
- **Phase 3 (two-pass cast-gated reviewer):**
  - NEW `_otr_ledger_reviewer.py`: `audit_cast_contract(ledger, label)` (single function called twice — Pass 1 pre / Pass 3 post — per synthesis G4). `apply_deterministic_cast_repairs` handles bad_casing / wrong_char_id / role_mismatch / alias_used / invented_name (auto-remap via Levenshtein per §6.A + G8 test table) / speaker_unknown (stop, `cast_unrecoverable`). `compute_edit_cap(voiced_beats) = min(8, max(3, voiced_beats // 3))` (G1). `run_script_doctor` (Pass 2) reads post-repair candidate, proposes ≤ edit_cap rewrite/skip/annotate edits, fails soft on any LLM error. `apply_phantom_skip_fallback` (Step 2.5, M2) — deterministic titled-phantom mute between Pass 2 and Pass 3. `review_ledger(generate_fn, led)` orchestrates all three passes, snapshots original ledger, restores on any failure path, stamps `meta.reviewer_verdict` (six pinned literals per G7).
  - NEW `OTR_LedgerScriptReviewer.py` node — thin wrapper around `review_ledger`. Reads ledger via `get_ledger()`; `script_text` socket is passthrough for graph wiring. Registered in `__init__.py` between writer and director.
  - Programmatic bypass via `meta.skip_reviewer=True` for unit tests (G9). Production runs always run the reviewer.
  - Tests: `tests/test_phase3_ledger_reviewer.py` (Levenshtein G8 table, edit_cap scaling, audit_cast_contract clean/dirty/LLM-fail/malformed-JSON, each deterministic repair kind, phantom-skip fallback, end-to-end disposition for ALL SIX verdicts: clean_no_edits / improved / cast_unrecoverable / too_many_edits / needs_full_rerun / post_audit_failed / skip_reviewer bypass).
- **Cast data model resolution (synthesis §10):** Read pass concluded acceptance criterion #1 — single canonical source at `led.data["cast"]` (structured rows from `_OTRCAST.lock_cast`). `character_cast` is a trivial tuple view. `_otr_cast_contract.CastContract` is greenfield Phase-0+ infra not wired into the v2 writer. No unification commit needed.
- **KV cache verdict (synthesis §8 pre-flight):** `_otr_model_loader.make_generate_fn` does NOT pass `past_key_values=` between `model.generate()` calls. Decision: pay the full prefill cost per composer call (path b). The ~800-tok composer prompt is acceptable on RTX 5080 + Mistral-Nemo 12B. The §6.D plot-leak mitigation (trim spine to current arc_phase) doubles as the no-KV fallback if soak shows wall-clock degradation; not active today.
- **Workflow JSON wiring:** `OTR_LedgerScriptReviewer` registered in `__init__.py` and discoverable in ComfyUI's node menu. Manual wiring into `workflows/otr_scifi_16gb_full.json` left to Jeffrey (drag writer.script_text → reviewer → director.script_text) — bash sandbox mount went stale around the workflow file, safer to wire in the GUI than hand-edit 47kB JSON with link-id math from a stale view. `act_count` widget appended at END of writer's INPUT_TYPES so the existing graph's 17-entry `widgets_values` still maps positionally; ComfyUI fills the 18th slot with default 0 (auto-derive).
- **Invariants held:** local-only; VRAM ≤14.5 GB unchanged; schema additivity only (every new field on `lines[]`/`meta` is additive); SFW; one commit per logical phase. ADR at `docs/script-writing-architecture-adr.md`.
- **Verify (Jeffrey post-handoff):** PowerShell block at the end of the handoff message. AST parse on every changed module; `pytest tests/test_phase0_name_roster.py tests/test_phase1_composer_prompt.py tests/test_phase2a_episode_budget.py tests/test_phase2b_progressive_ledger.py tests/test_phase3_ledger_reviewer.py -v`; Bug Bible regression; in-module self-test for `_otr_line_composer.py` + `_otr_episode_budget.py`. Workflow JSON load in ComfyUI Desktop (act_count widget should appear at the bottom of the writer node; reviewer should be available in the OldTimeRadio category).
- **Tags:** phase-0, phase-1, phase-2a, phase-2b, phase-3, name-roster, episode-budget, progressive-ledger, reviewer, json-wiring

---

### NON-BUG-2026-05-10: news_interpreter sprint commit 5 — era literals stripped + 5 text-scan canaries flipped — SPRINT COMPLETE (commit 4f45c7c, amended from 92e58e5)

- **Date:** 2026-05-10 | **Phase:** 4 | **Bible candidate:** no (sprint milestone)
- **Symptom:** CLOSES the news_interpreter sprint. Era flavor now flows ONLY through the two North Star variables (news_story + style). No hardcoded period literals remain in the narrative plane.
- **Diagnosis:** Per ADR section 7 surgical fixes. Three files edited: `script_critic.py` (3 literal strips at lines 330, 339-340, 556), `story_orchestrator.py:_LTX_STYLE_BRIEF_PROMPT` (full rewrite per ADR section 7.4 Option A, three style-spanning examples replacing the three baked vacuum-tube examples), `test_downstream_prompt_contract.py` (5 xfail-strict markers removed in lockstep with the literal removal per the canary mechanic).
- **Fix:**
  - `script_critic.py:330`: "Period-inappropriate vocabulary in 1940s setting" → "Vocabulary that contradicts the established style/setting" (preserves rubric intent; drops era anchor).
  - `script_critic.py:339-340`: "script doctor for SIGNAL LOST, a 1940s-style {genre_human} radio drama" → "script doctor for a {genre_human} audio drama episode" (drops both brand-priming and era anchor).
  - `script_critic.py:556`: "revising a 1940s {genre_human} radio drama script" → "revising a {genre_human} audio drama script".
  - `story_orchestrator.py:3394-3411`: full `_LTX_STYLE_BRIEF_PROMPT` rewrite. Opens with "during an audio drama" (era-neutral). New bullet: "Use equipment design language that fits the setting AND style — do not default to any specific era's hardware unless the story explicitly implies it." Three new examples spanning near-future newsroom, deep-space vessel, rust-belt industrial decay. No vacuum tubes, no brass speaker grilles, no radio bays.
  - `test_downstream_prompt_contract.py`: removed `xfail(strict=True)` markers on 5 text-scan canaries. Bodies unchanged. Test docstrings + error messages updated to "LANDED in commit 5; marker removed in lockstep; literal must not regress."
- **History note:** Originally shipped at `92e58e5` with the wrong subject line ("docs: news_interpreter sprint commit 4 logged...") due to the CLAUDE.md commit-message anti-pattern — a chained `git commit -F .git\COMMIT_EDITMSG` consumed a stale message file when a parallel `Write` to update it raced and was rejected. The code diff was correct from the start; only the subject was wrong. Force-amended to `4f45c7c` with Jeffrey's explicit OK (force-push on the working v2.0-alpha branch).
- **Verify:** AST + no-BOM clean on all 3 edited files. `tests/test_news_interpreter.py`: 12 passed. `tests/test_news_interpreter_wiring.py`: 13 passed. `tests/test_downstream_prompt_contract.py`: 6 passed (5 freshly-flipped + case 12) + 2 xfailed (RADIO portrait + MusicGen, still armed as future-ADR canaries per ADR section 1 out-of-scope). `tests/test_otr_casting.py`: 47 passed. Bug Bible regression: 15p/2x/1s (baseline held across all 5 sprint commits). Push verified: local HEAD == origin HEAD == `4f45c7c77c1401a03c9a14cdca211182adf83330`.
- **Sprint summary (commits 1-5, all on v2.0-alpha):**
  - `6f3218d` commit 1 — ADR + xfail-strict canary tests
  - `70d25eb` commit 2 — agnostic news_interpreter module + GBNF grammar
  - `f518fb3` commit 3 — wire briefs into writer + cast + outline + schema bump + canary case 12 flip
  - `9f82685` commit 4 — announcer closing-line override + post-assembly key_terms audit + 13 new wiring tests
  - `4f45c7c` commit 5 — strip era literals + flip 5 text-scan canaries
- **Tags:** news-interpreter, sprint-complete, era-literals-stripped, canary-mechanic-end-to-end, north-star-honored, force-amend-recovery

---

### NON-BUG-2026-05-10: news_interpreter sprint commit 4 — announcer close + post-assembly key_terms (commit 9f82685)

- **Date:** 2026-05-10 | **Phase:** 4 | **Bible candidate:** no (sprint milestone)
- **Symptom:** Commit 3 (f518fb3) stamped `meta.news = briefs.dict()` and piped briefs into cast + outline. Commit 4 closes the loop: the announcer's closing line gets overridden with `news_close_brief` so the listener hears the journalistic content, and a post-assembly check audits whether `key_terms` actually landed in dialogue.
- **Diagnosis:** Per ADR section 5 commit 4 + section 4.4. Two pure operations run on the in-flight `line_rows` between the per-beat composition loop and `set_lines`: (1) override the LAST announcer line's text with `news_close_brief` when present; (2) word-boundary-check each `key_term` across all voiced lines (character + announcer), stamp `meta["post_assembly_key_terms"] = {landed, missing, min_required, passed, repair_pass}` for downstream nodes.
- **Fix:**
  - `nodes/_otr_news_wiring.py` (new): pure helpers `override_announcer_close` and `post_assembly_keyterm_check`. Pulled into their own tiny module so tests don't have to import the heavy OTR_LedgerScriptWriter.
  - `nodes/OTR_LedgerScriptWriter.py`: new I.5 section after the per-beat loop and before `set_lines`. Reads `meta["news"]`; calls both helpers; stamps diagnostic results back onto `meta`. No-ops when `meta["news"]` is None (graceful degrade carries through from commit 3).
  - `tests/test_news_interpreter_wiring.py` (new, 13 cases): override-announcer-close behavior (stamps last announcer line, no-op on empty / whitespace / no announcer line, idempotent); key_terms check (all landed, some missing, word-boundary precision so "AI" does NOT match "paid"/"afraid", non-voiced beats ignored, case-insensitive, both `_speaker_role` and `speaker_role` keys accepted, empty-term defensive skip).
  - `tests/test_downstream_prompt_contract.py`: reason text updated on the 2 remaining xfail-strict integration canaries (RADIO portrait + MusicGen). Per ADR section 1 those sites are explicitly OUT OF SCOPE of the news_interpreter sprint and land in future ADRs. Canaries stay armed — body is unchanged, only the `reason=` documentation was updated to point at "future ADR (TBD)" rather than commit 4.
- **ADR deviation:** Section 4.4 canonical policy at zero `key_terms` landed is hard-fail + repair pass on the line whose intent is closest to the missing term's topic. Commit 4 ships **warn-only** and DEFERS the repair pass. `meta["post_assembly_key_terms"]["repair_pass"]` is stamped `"deferred"` so future code knows this is a planned follow-up. Rationale: alpha-branch pragmatism — the episode ships and the diagnostic field surfaces the issue clearly, rather than blocking the writer on a missing repair-pass implementation.
- **Verify:** AST + no-BOM clean. `tests/test_news_interpreter_wiring.py`: 13 passed (new). `tests/test_news_interpreter.py`: 12 passed (unchanged). `tests/test_downstream_prompt_contract.py`: 7 xfailed + 1 passed (case 12 stays flipped from commit 3; cases 1 + 2 still xfailed as future-ADR canaries). `tests/test_otr_casting.py`: 47 passed. Bug Bible regression unchanged: 15p/2x/1s. Push verified: local HEAD == origin HEAD == `9f82685a7a643ba5660e191ce4040b683dc7e1b8`.
- **Tags:** news-interpreter, announcer-close-override, post-assembly-keyterm-audit, lockstep, sprint-milestone, adr-deviation-tracked

---

### NON-BUG-2026-05-10: news_interpreter sprint commit 3 — briefs wired into writer + cast + outline (commit f518fb3)

- **Date:** 2026-05-10 | **Phase:** 4 | **Bible candidate:** no (sprint milestone)
- **Symptom:** Commit 2 (70d25eb) shipped the agnostic module standalone. Commit 3 wires it into the production pipeline per Jeffrey's rule "always wire lockstep" — work is not done until it's called.
- **Diagnosis:** Per ADR section 5. The news_interpreter LLM call inserts between style-resolve (D.2) and cast-lock (D.3) in OTR_LedgerScriptWriter. Briefs land at `ledger.meta.news`; downstream cast LLM reads `casting_brief` (replacing the mechanical 500-char slice of `news_seed`); outline LLM reads `script_brief` + `key_terms` (replacing raw news_seed in the prompt + injecting a "Required terms" line). Schema migration: old ledgers without `meta.news` access cleanly via `.get('news')` returning None; consumers fall back to raw `news_seed` for cast/outline and to a synthesized line for the announcer (commit 4 wires the announcer fallback).
- **Fix:**
  - `nodes/_otr_ledger.py:48`: `CURRENT_SCHEMA_VERSION` bumped `"l3-2026-05-08"` → `"l3-2026-05-14"`. SCHEMA_VERSION participates in news_interpreter cache key, so schema bumps force brief regeneration.
  - `nodes/production_ledger.py:254`: hardcoded fallback bumped to match.
  - `nodes/OTR_LedgerScriptWriter.py`: `_fetch_rss_seed_or_die` returns `dict` (was `str`) with headline / summary / full_text / source / date / link / seed_text. `_resolve_inputs` plumbs `news_article` dict into resolved (custom_premise path synthesizes the same shape). New D.2.5 calls `build_news_briefs()` and stamps `meta["news"] = briefs.model_dump()`; graceful degrade (warn + fall back) when build_news_briefs raises. `lock_cast` call gains `casting_brief`; OutlineRequest gains `script_brief` + `key_terms`.
  - `nodes/_otr_casting.py`: `_build_user_prompt` + `cast_one_character` + `lock_cast` gain optional `casting_brief: str = ""` kwarg. When non-empty, replaces the 500-char `news_seed` slice on the prompt's `Story:` line. Empty default preserves every existing test fixture.
  - `nodes/_otr_outline.py`: `OutlineRequest` gains `script_brief: str = ""` and `key_terms: tuple[str, ...] = ()` defaults. `_build_user_prompt` substitutes `script_brief` for `news_seed` when non-empty + injects a "Required terms" line when key_terms is non-empty.
  - `tests/test_downstream_prompt_contract.py`: case 12 (`test_old_ledger_without_meta_news_loads_with_warning`) — xfail-strict marker REMOVED in lockstep with the fix. Test body rewritten to actually exercise the graceful-degrade contract (meta.get('news') returns None on pre-commit-3 ledgers; meta.news=None sentinel survives JSON round-trip). The canary mechanic from commit 1 enforces this lockstep automatically: if the marker stayed, the suite would have XPASSed and failed under strict.
- **Verify:** AST + no-BOM clean on all 6 edited files. `tests/test_news_interpreter.py`: 12 passed. `tests/test_downstream_prompt_contract.py`: 7 xfailed + 1 passed (case 12 flipped). `tests/test_otr_casting.py`: 47 passed (additive kwarg preserved every fixture). Bug Bible regression unchanged: 15 passed / 2 xfailed / 1 skipped. Push verified: local HEAD == origin HEAD == `f518fb3c534684f1c306fdb3724fb86fc804a5e6`.
- **Tags:** news-interpreter, writer-wiring, schema-bump-l3-2026-05-14, graceful-degrade, canary-lockstep, sprint-milestone

---

### NON-BUG-2026-05-10: news_interpreter sprint commit 2 — agnostic module + GBNF grammar shipped (commit 70d25eb)

- **Date:** 2026-05-10 | **Phase:** 4 | **Bible candidate:** no (sprint milestone)
- **Symptom:** Commit 1 (6f3218d) locked the API surface via `tests/test_news_interpreter.py` (12 cases skipped via `pytest.importorskip`). Commit 2 lands the module those tests describe.
- **Diagnosis:** Per ADR section 5 commit order. Module must be strictly LLM-agnostic: `generate_fn(messages, *, temperature, max_new_tokens) -> str` only. No model branches, no chat-template assumptions, no grammar-file kwarg passing (loader-side concern). Gemma 4 + MTP, llama.cpp + GBNF, vLLM, HF Transformers all sit behind generate_fn opaquely.
- **Fix:**
  - `nodes/news_interpreter.py` (~700 LOC): NewsBriefs pydantic v2 model + NewsInterpreterError + FORBIDDEN_ERA_TERMS + PROMPT_VERSION/SCHEMA_VERSION/DEFAULT_DECODER_PROFILE + v1/v2/v3 validators (with source-context allowance) + build_source_wrapper + compute_cache_key + extract_json_block + build_news_briefs with 3-attempt T=0.7/0.8/repair@0.3 ladder. Production 2-6 key_terms bound enforced at orchestration layer (V0 check in build_news_briefs); schema field accepts 1-6 so unit tests can isolate V1/V2/V3 with single-term fixtures.
  - `grammars/news_interpreter.gbnf` (~30 lines): GBNF grammar for future llama.cpp loaders. Loader-side, not passed by the module. Picked up by convention (`<repo>/grammars/<module>.gbnf`) when a llama.cpp loader integrates.
- **Verify:** AST + no-BOM clean. `tests/test_news_interpreter.py`: 12 passed (was 12 skipped pre-commit). `tests/test_downstream_prompt_contract.py`: 8 xfailed (canaries still armed for commits 3-5). `tests/test_otr_casting.py`: 47 passed (cast contract unaffected). Bug Bible regression unchanged: 15 passed, 2 xfailed, 1 skipped (baseline held). Push verified: local HEAD == origin HEAD == `70d25ebe78e1a2c772fee45e4584c1d950668a5a`.
- **Schema bump deferred:** SCHEMA_VERSION is set to "l3-2026-05-14" but `production_ledger.py` schema bump + `meta.news` field stamp lands in commit 3 alongside the writer wiring.
- **Tags:** news-interpreter, agnostic-control-plane, gbnf, pydantic-v2, sprint-milestone

---

### NON-BUG-2026-05-10: news_interpreter sprint commit 1 — ADR + xfail-strict canary tests armed (commit 6f3218d)

- **Date:** 2026-05-10 | **Phase:** 4 | **Bible candidate:** no (sprint milestone, not a defect)
- **Symptom:** Downstream prompt audit (Cowork-driven, full sweep across 25 prompt-building sites) surfaced 5 hardcoded era-literal violations: `script_critic.py:330,339-340,556` ("1940s setting" / "1940s-style" / "You are revising a 1940s ...") and `story_orchestrator.py:3401,3407-3409` (`_LTX_STYLE_BRIEF_PROMPT` "vintage-radio elements ... skinned for the setting" + three vacuum-tube example anchors). Cast LLM `Story:` line was getting a mechanical 500-char slice of `headline + summary`; the per-line composer + announcer closing read never saw the article body at all. News content degraded silently before any prompt fired.
- **Diagnosis:** Round-robin (ChatGPT gpt-5.5 + Gemini 3.1 Pro + NVIDIA) on the question staged at `outputs/news_interpreter_question.md` converged on a 4-output news_interpreter LLM stage (casting_brief / script_brief / news_close_brief / key_terms) inserted between style-resolve (D.2) and cast-lock (D.3) in `OTR_LedgerScriptWriter`. ADR captured at `docs/news_interpreter_adr.md` (771 lines). Q1-Q4 verdicts locked: one unified call, headline + summary + first 1500 chars (+ last 500 on long bodies), word-boundary regex `key_terms`, news_close_brief on the same upfront call.
- **Fix:** Commit 1 of 5 lands the safety net first per ADR section 5:
  - `docs/news_interpreter_adr.md` (canonical ADR)
  - `tests/test_news_interpreter.py` (12 unit tests, `pytest.importorskip` dormant until commit 2 lands the module; locks the API surface NewsBriefs / v1_validate / v2_validate / v3_validate / build_source_wrapper / compute_cache_key / extract_json_block / build_news_briefs)
  - `tests/test_downstream_prompt_contract.py` (8 xfail-strict canaries; flips to XPASS and fails the suite the moment commit 5 strips a literal, forcing the marker to be removed in lockstep with the fix)
- **Verify:** AST parse + no-BOM clean on both test files. `pytest --collect-only`: 8 contract tests collected, news_interpreter file skipped cleanly. Verbose run: all 8 contract tests xfailed as expected. Bug Bible regression unchanged: 15 passed, 2 xfailed, 1 skipped (baseline held). Push verified: local HEAD == origin HEAD == `6f3218d48c6952e452194f03ad9a7192de2bb5a1`.
- **Commits 2-5 queued:** module + GBNF + validators (commit 2), wire into writer/cast/outline + meta.news schema + graceful degrade (commit 3), wire announcer + line composer + post-assembly key_terms check (commit 4), strip era literals from script_critic + story_orchestrator (commit 5, flips the 5 text-scan xfails).
- **Tags:** news-interpreter, canary, xfail-strict, adr, downstream-prompt-audit, period-literals, sprint-milestone

---

### NON-BUG-2026-05-10: downstream visual chain audited clean — no L3 rewrites needed (sprint follow-up to ledger-consumer-rewrite eec4718)

- **Date:** 2026-05-10 | **Phase:** 3 | **Bible candidate:** no (recon-verdict, not code change)
- **Symptom:** Sprint plan called for "full-rigor downstream review and fix" of every visual / post-process / cast / utility node downstream of the L3 ledger writer. Concern: any node still parsing the legacy parser-list `script_json` shape (`[{"type":"environment", ...}, {"type":"dialogue", ...}]`) or reading legacy field names (`character_name`, `voice_traits`, list-index access on `lines[]`, regex on `[VOICE: NAME]`) would crash on L3 input from the new `OTR_LedgerScriptWriter`.
- **Diagnosis:** Recon pass over every active downstream node:
  - `visual/batch_flux_render.py` (radio bookend + dead env stills)
  - `visual/batch_flux_portrait_render.py` (per-cast portraits)
  - `nodes/batch_humo_render.py` (character lip-sync)
  - `nodes/batch_ltx_render.py` (non-character motion)
  - `nodes/video_composite.py` (1080p mux + gap fill)
  - `nodes/rtx_upscale.py` (post-upscale)
  - `nodes/otr_post_upscale_procgen_blend.py` (procgen overlay)
  - `nodes/otr_save_to_episode_workspace.py`, `nodes/otr_save_copy.py`, `nodes/otr_video_concat.py`, `nodes/otr_video_plan.py`, `nodes/otr_shot_duration_calculator.py`
  - `nodes/_otr_cast_repair.py`, `nodes/_otr_voice_resolver.py`, `nodes/_voice_backends/*.py`, `nodes/voice_render.py`, `nodes/_otr_period_prompts.py`
  - `nodes/post_audio_video_pipeline.py` (RETIRED — backward-compat registration only)
  - All LLM prompt construction sites: `_otr_outline.py`, `_otr_line_composer.py`, `_otr_period_prompts.py`, `script_critic.py`, `story_orchestrator.py`, `_otr_legacy_writer.py`
- **Method:** Grep for danger patterns (`payload.get("tokens")`, `for x in payload`, `[VOICE: NAME]` regex, `item.get("type") == "dialogue"`, list-index access on `ledger.lines[]`, legacy field names). Direct read of every prompt construction site. Cross-check against L3 schema in `_otr_ledger.py` (CURRENT_SCHEMA_VERSION = "l3-2026-05-08").
- **Finding:** Every active downstream node already uses L3-native field names exclusively (`line_id`, `char_id`, `speaker_role`, `start_s`, `dur_s`, `text`, `word_count`, `shot_id`, `cast[].char_id`, `cast[].name`, `cast[].voice_preset`, `cast[].portrait_path`, `meta.gen_params_initial.style`, `meta.radio_bookend_path`, `episode_id`), reads ledger from disk via `_OTRL.in_flight_ledger_path()` / `production_ledger.get_ledger()` singleton + `load_ledger_safe`, and degrades gracefully with `.get(...)` defaults on missing fields. The only legacy-parser-list parsing site (`visual/batch_flux_render.py:_parse_env_prompts`) is dead code: bypassed by default widget `skip_env_stills=True`, and even if enabled would degrade to fallback prompts (no crash) on L3 dict input. Confirmed ROADMAP line 67 prediction.
- **Fix:** None needed. Verdict: **AUDITED CLEAN, no rewrites needed.** Documented in ROADMAP.md "Visual chain recon — AUDITED CLEAN, 2026-05-10" subsection.
- **Verify:** Bug Bible regression 23/1/2/0 baseline held throughout. New helper API tests `tests/test_otr_ledger_consumers.py` 48/48 PASS. Dry-run gate suite (5 gates: AST parse, node types registered, widget count vs INPUT_TYPES, link socket bounds, link types non-empty) ALL PASS.
- **Tags:** l3-ledger, recon, audit-clean, sprint-followup

---

### NON-BUG-2026-05-08: "HuMo hangs on 2nd/3rd clip in full episode runs" was the soak cap firing as designed; cap=3 was too tight for production scripts

- **Date:** 2026-05-08 evening | **Phase:** 5 | **Bible candidate:** yes
  (cap-tuning advisory rather than code bug, but worth a Bible entry
  so the diagnosis isn't redone every time someone sees HuMo halt
  mid-script)
- **Symptom:** Multiple full-episode runs reported HuMo "hanging" or
  "failing on the 2nd/3rd HuMo clip." Smoke runs of the same code
  worked. The full-run logs ended with no traceback, no fatal abort,
  no OOM -- the workflow simply stopped advancing past HuMo even
  though the script declared 6+ character lines.
- **Cause:** Workflow widget `humo_max_lines_per_process` was set to
  `3` (BUG-LOCAL-126 defensive backstop after the 2026-05-07
  overnight allocator-drift fatal abort). When a script generated
  more than 3 character lines, BatchHumoRender hit the cap and
  exited via the `HumoSoakCapReached` raise (default
  `stop_workflow_on_soak_cap=True`). The workflow halted with no
  error trace because the raise is the designed soft-stop signal,
  not a crash. From the operator's perspective, this looked
  identical to "HuMo broke."
- **Verification:** 2026-05-08 4-HuMo smoke (commit `9c6353d`,
  `workflows/otr_humo_4x_smoke.json`) ran four character-line clones
  of `l002` (synthetic ledger
  `output/otr/episodes/signal_lost_lunar_dawn_20260508_133930/audio/synthetic_4humo_ledger.json`).
  Result: clip 1 done in 377 s, clip 2 in 374 s, clip 3 in 369 s --
  three consecutive HuMo renders back-to-back with **zero OOM, zero
  allocator drift, zero hang**. The 4th clip didn't run because the
  cap fired exactly as designed (`3 fresh of cap 3; SOFT_CAP mode`).
  The "hang" was the cap, not HuMo. Allocator drift only manifests
  AFTER a caught OOM cycle; healthy successive renders don't drift.
- **Fix:** Bump `humo_max_lines_per_process` from `3` to `0`
  (unlimited) in all four workflow JSONs (commit `02a5749`):
    - `otr_scifi_16gb_full.json` (production)
    - `otr_humo_smoke.json` (E2E smoke)
    - `otr_humo_only_smoke.json` (strict HuMo isolation)
    - `otr_humo_4x_smoke.json` (4-HuMo reproducer)
  `cuda_hard_reset_on_oom` stays `True` so the BUG-126 cleanup chain
  still fires if a real OOM happens; we just don't artificially halt
  after N successful clips.
- **Future watch:** If a real allocator drift returns (the original
  2026-05-07 9-clip overnight fatal abort), the cap can be re-set to
  6-8 (the empirically safe per-process budget per the BUG-126
  tooltip). The 4-HuMo smoke is the regression detector.
- **Tags:** humo, soak-cap, BUG-126, widget-config, false-positive,
  allocator-drift

---

### BUG-LOCAL-135 [FIXED]: Music / SFX / gap segments freeze the radio set with 47.8 s of static-radio fill instead of looping a motion clip (caught by 2026-05-08 13:39 clean run, design call by Jeffrey)

- **Date:** 2026-05-08 late afternoon | **Phase:** 6 | **Bible candidate:** yes
- **Symptom:** The `signal_lost_lunar_dawn_20260508_133930` clean run
  reported `BUG-084 gap-fill: inserted 3 static-radio segment(s),
  total 47.794 s of coverage` — meaning the announcer LTX clip
  (5.6 s) and one HuMo character clip (4.9 s) were the only moving
  imagery in a 58.3 s episode; the remaining 47.8 s (music_open,
  music_close, interstitial slot, inter-clip gaps) was painted as
  the radio bookend PNG held perfectly still. Pre-2026-05-04
  runs (e.g. `signal_lost_boosting_one_protein_helps_the_brain_fig_20260502_170555`)
  rendered LTX motion clips for music_opening_001/002 and
  music_closing_001/002 via a `lines[]` mirror; that mirror's
  appended rows are not landing in the post-rename ledger snapshot
  this build, so `BatchLTXRender`'s plan loop sees only the
  `announcer` line and the per-music-cue LTX path silently
  evaporates.
- **Cause:** Two stacked issues. (a) The music mirror's `lines[]`
  appends are getting clobbered before `BatchLTXRender` reads the
  ledger (likely a save-race between `EpisodeAssembler.lines[]
  +=2` and `SignalLostVideo`'s post-rename re-save with a stale
  in-memory ledger). (b) Even if (a) were fixed, the static-radio
  helper (`_render_static_radio_segment`, BUG-LOCAL-129a) is the
  only fallback wired into `VideoComposite` for gap_fill segments,
  so any non-LTX-covered span lands as a frozen frame instead of
  inheriting the broadcast-set motion that the announcer LTX
  already produced for free.
- **Fix:** Sidesteps (a) entirely by attacking (b) at the
  `VideoComposite` consumer side. New helper
  `_render_loop_motion_segment` ffmpeg-`-stream_loop`'s an existing
  motion clip to fit any requested duration, with the same
  invariants as the static helper (frame-exact via `-frames:v`,
  locked timebase 12800, libx264 yuv420p crf=18 preset=fast,
  `-an` so master mix attaches once in the final mux). New
  resolver `_resolve_loop_motion_for_fill` picks a loop source
  in this order: pre-baked
  `<episode_dir>/loop/loop_radio_motion.mp4` (reserved for a
  future Phase 1 producer), then the first ledger
  `clips[]` entry with `source_kind='ltx'` whose mp4 lives on
  disk (typically the announcer clip — zero extra render cost
  since it already exists), then any `clips_dir/l*.mp4` as a
  catch-all. All three `VideoComposite` fill sites — missing-clip
  fallback, BUG-084 inter-clip gap-fill, BUG-084 trailing
  tail-fill — now try motion-loop first and fall through to
  BUG-LOCAL-129a static-radio on any failure (resolver miss,
  ffmpeg subprocess error, etc.). Reporting upgraded to log the
  loop-vs-static split per run so the regression is observable
  from the next clean-run audit.
- **Verify:** AST clean,
  `python -m pytest tests/test_batch_ltx_render.py -q` 33/33
  passing, Bug Bible regression 23/23 passing. Next live run
  should report `BUG-084 gap-fill: inserted N segment(s) total
  X.XX s (loop=N static=0)` where today's run reported
  `(loop=0 static=3)`. Visual check: the radio bookend should
  show subtle motion across the full episode instead of three
  freeze segments.
- **Tags:** ltx, video-composite, gap-fill, ffmpeg, static-fill,
  motion-loop, BUG-084, BUG-129a

---

### BUG-LOCAL-134 [FIXED]: BAD_IMAGE_SAVE gate false-fires on BatchFluxRender skip_env_stills sentinel placeholder (caught by 2026-05-08 13:07 clean run)

- **Date:** 2026-05-08 mid-day | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** The 13:07 clean run
  (`signal_lost_stellar_whispers_20260508_130733`) hit the
  BUG-LOCAL-133 image-save-size gate with this signature:
  ```
  [SaveToEpisodeWorkspace] save failed for image 1:
    BAD_IMAGE_SAVE: full_env_00001_.png wrote 74 bytes
    (under 4096-byte gate); input tensor shape=(16, 16, 3)
    dtype=torch.float32 min=0.0000 max=0.0000.
  ```
  The shape `(16, 16, 3)` and `min == max == 0.0` is the
  intentional placeholder tensor that
  `OTR_BatchFluxRender` emits on its IMAGE output when
  `skip_env_stills=True` (BUG-LOCAL-078 follow-up). The gate
  caught a legitimate "nothing to save" signal as an upstream
  fault. Run continued past the raise (existing per-image
  try/except in `OTR_SaveToEpisodeWorkspace` logged + continued)
  but the noise made the audit harder to read.
- **Cause:** BUG-LOCAL-133's image-save-size gate was correct in
  shape but didn't account for the deliberate empty-marker
  pattern that one upstream branch uses. The 4096-byte threshold
  catches both real upstream failures AND benign sentinels --
  ambiguous signal.
- **Fix:** Added a sentinel-recognition predicate BEFORE the
  size gate. If the input tensor has its spatial dims <=64 AND
  is all zeros, treat it as a known skip-stub:
  log info, advance the index, do NOT write a PNG, do NOT
  raise. The 64-pixel ceiling is generous (real FLUX/LTX
  renders are >=512px) so a legitimately-black large render
  still hits the size gate as designed.
- **Verify:** AST clean, LTX regression unchanged. Next run
  with `skip_env_stills=True` should produce no
  `BAD_IMAGE_SAVE` log line for `full_env_*.png`; the
  SaveToEpisodeWorkspace step logs an
  `[SaveToEpisodeWorkspace] skipping all-zero marker tensor
  (shape=(16, 16, 3))` info line instead.
- **Tags:** image-save-validation, sentinel-pattern, skip-stub,
  BUG-133-followup, false-positive
- **Bible candidacy:** yes -- the lesson is *size gates over
  raw bytes are coarse-grained; pair them with a shape-aware
  sentinel predicate when an upstream branch deliberately emits
  empty markers*. Two predicates compose cleanly: shape-based
  recognition catches known stubs; size-based catches actual
  upstream failures.

### BUG-LOCAL-133 [FIXED]: Stage 1 post-run patches -- BUG-031 silent-skip wired, image-save-size gate added, soft-cap mode added (caught by 2026-05-08 Stage 1 post-run synthesis)

- **Date:** 2026-05-08 mid-day | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Stage 1 smoke run
  (`signal_lost_space_station_control_room_20260508_120907`)
  validated the BUG-LOCAL-128 cap counter split, but the run also
  exposed three real upstream/downstream gaps that prevented
  end-to-end validation:
  1. Bark generated 0.1s of post_bark audio TOTAL across 11
     character lines (verified via `meta.audio_gates[0].dur_s=0.1`,
     `sample_count=2400`, `sample_rate=24000`). Per-line ledger
     entries showed `dur_s=0.004166s` (1 frame at 240fps),
     `bark_wav_path=''`, `render_ms=None`. Bark was producing
     near-empty placeholder WAVs every line.
  2. HuMo dutifully lipsynced 6 fresh clips against the silent
     character slots (`humo_render_ms=~75000ms` per clip,
     `mp4_dur_s=1.12s`, files 1.6KB on disk -- garbage outputs).
     The existing BUG-LOCAL-031 silent-skip gate helper
     (`_audio_slice_rms_db`) was DEFINED but never wired into the
     per-line loop ("helper present but loop wiring deferred"
     comment at execute() line 1473).
  3. `stills/full_env_00001_.png` saved as 74 BYTES (PNG header
     only, no IDAT chunk) because `OTR_BatchFluxRender` yielded
     an empty IMAGE tensor for that branch. The Director's
     `scenes`/`shots` arrays were populated (scenes=1, shots=5)
     so this is NOT an empty-Director issue -- some other FLUX
     execution path produced empty output. `OTR_SaveToEpisodeWorkspace`
     wrote the empty PNG silently with no validation.
  4. The cap firing on `HumoSoakCapReached` raised an exception,
     which terminated the workflow and prevented LTX / Composite /
     Upscale / PostProcgenBlend from running on the 6 partial
     clips. End-to-end DAG validation impossible while cap is in
     effect.
- **Cause:** Three independent gaps that compounded:
  1. BUG-LOCAL-031 design existed but the runtime check was never
     wired into the HuMo per-line loop -- graceful-degradation to
     static-radio-fill (BUG-LOCAL-129a) couldn't fire because
     nothing was checking RMS against the threshold.
  2. `OTR_SaveToEpisodeWorkspace` had no post-write validation,
     so a 74-byte PNG was as valid as a 700KB one from the
     node's perspective.
  3. The Step 4 cap-as-exception design was correct for the
     production overnight pattern (where the watcher re-queues)
     but invalid for end-to-end smoke validation (where the
     downstream DAG must run on partial output).
- **Fix:** Three patches in one commit per the Stage 1 post-run
  synthesis section 3 (Steps 2, 3, 4):
  - **Step 2 / `nodes/batch_humo_render.py`:** wired the existing
    `_audio_slice_rms_db` helper into the per-line loop right
    after the existing missing-prompt / missing-chunks skips. A
    line whose audio slice RMS falls below `min_speech_rms_db`
    (default `-28.0` dBFS) gets:
      * `report_lines.append(...SILENT_SKIP...)`
      * `lines[].render_method = 'static_radio_fill'` stamped via
        `_otr_ledger.stamp_line_render_method`
      * `continue` -- HuMo is skipped; downstream
        `OTR_VideoComposite` paints the static radio bookend for
        that time slot per BUG-LOCAL-129a.
    Set `min_speech_rms_db = -90` to disable the gate entirely.
  - **Step 3 / `nodes/otr_save_to_episode_workspace.py`:** added
    a 4096-byte post-save validation gate. Any saved PNG under
    that threshold gets unlinked and a `BAD_IMAGE_SAVE`
    `RuntimeError` raised with diagnostic info: filename, byte
    count, input tensor `shape / dtype / min / max`. One log
    line tells you whether the upstream produced empty data or
    the save logic mishandled it.
  - **Step 4 / `nodes/batch_humo_render.py`:** added
    `stop_workflow_on_soak_cap` BOOLEAN widget (default `True`).
    When `True`, cap fires raise `HumoSoakCapReached` (existing
    behavior). When `False`, cap fires save the ledger + soak_cap
    stamp, log `SOAK_CAP_REACHED`, and `break` out of the per-
    line loop so downstream DAG (LTX, Composite, Upscale,
    PostProcgenBlend) still executes on partial output. Used by
    end-to-end smoke validation; not safe in production resume
    mode until LTX grows its own resume-from-ledger scan.
  - **Workflow JSON** (`otr_scifi_16gb_full.json` Node 51):
    extended `widgets_values` to 18 entries (added `False` for
    `stop_workflow_on_soak_cap`). Set Stage 1 cap to `3`. After
    Stage 1 validates end-to-end, flip the boolean back to `True`
    and bump cap to `6` for Stage 2 / production.
- **Verify:** AST clean, LTX regression unchanged. Stage 1 re-run
  with `cap=3, stop_workflow_on_soak_cap=False, min_speech_rms_db
  =-28.0` will tell us:
  * Bark fix path: silent-skip catches dead audio -> static-radio
    fallback fires -> Composite still produces a usable frame
    sequence
  * FLUX env still fix path: 74-byte PNG either no longer happens
    OR `BAD_IMAGE_SAVE` raises with shape/dtype/min/max so we know
    where to look upstream
  * Cap fix path: cap=3 fires, ledger persisted, downstream DAG
    runs on 3 fresh clips, full pipeline completes
- **Tags:** silent-skip, BUG-031, image-save-validation, soft-cap,
  end-to-end-validation, stage-1-followup, multiple-fix-classes
- **Bible candidacy:** yes -- three lessons:
  * "helper present but loop wiring deferred" is a real cost; if
    the wiring is small AND the helper has a graceful-degradation
    target, ship the wiring on first attempt
  * any save-to-disk node MUST validate the post-write file size
    against a threshold informed by what real output looks like
  * exception-as-control-flow for "soft stop" cap signals breaks
    the rest of the DAG; structured exit + flag is the production
    pattern

### BUG-LOCAL-132 [FIXED]: Node 55 widget array missing the control_after_generate slot for the seed widget (BUG-131 reverted by 2026-05-08 CAG-correction round-robin)

- **Date:** 2026-05-08 morning | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** After the BUG-LOCAL-131 trim shipped (commit
  `052724d`), Node 55's widgets_values length dropped to 5,
  which (correctly counted against backend `INPUT_TYPES`) was
  one short of what ComfyUI's frontend expects. The frontend
  auto-injects a `control_after_generate` (CAG) widget into
  `widgets_values` immediately after every `seed: INT` input,
  but that widget does NOT appear in the Python class's
  `INPUT_TYPES`. Backend has 5 widget fields; frontend renders
  6 widgets; widgets_values must therefore be length 6.
  After BUG-131's trim, the 5-entry array
  `['', 1, 'ffmpeg', '', 22.0]` position-mapped as:
  ```
  slot  widget                  value         consequence
  [0]   ledger_json             ''            link 90 overrides; harmless
  [1]   seed                    1             correct
  [2]   control_after_generate  'ffmpeg'      invalid CAG token; frontend
                                              probably defaults to 'fixed'
  [3]   ffmpeg                  ''            EMPTY ffmpeg path -> encode
                                              crash on first invoke
  [4]   humo_clips_dir          22.0          wrong type; link 91 overrides
  [5]   clip_length             (missing)     defaults to 7.0; we wanted 22.0
  ```
  Two real consequences in the post-trim state: (a) ffmpeg path
  is empty so LTX render would crash at first encode; (b)
  clip_length silently falls to FLOAT default 7.0, producing
  7-second LTX clips instead of the intended 22-second clips.
- **Cause:** BUG-LOCAL-131's diagnosis was incomplete. The
  reviewer correctly verified that the backend `INPUT_TYPES` has
  5 widget-renderable fields (no `seed_mode`), but missed that
  ComfyUI's frontend auto-inserts the CAG widget for the seed
  input. The original 6-entry array was correct; trimming was the
  fault.

  Equivalent verification done by post-mortem analysis: Nodes 51
  and 23 (also have `seed: INT` inputs) similarly carry an extra
  entry past `INPUT_TYPES` count for their CAG widget. The 2026-
  05-07 overnight soak that rendered 9 HuMo clips before fatal
  abort ran the same 13-entry Node 51 widgets_values with
  `'randomize'` at slot 5 -- if the reviewer's "validation crash
  on bad cast" theory were correct, that soak would have failed
  at queue time. Empirical evidence: those 9 clips proved the
  array IS correctly position-mapped with CAG accounted for.
- **Fix:** JSON-only revert. Restore Node 55's widgets_values to
  6 entries by inserting `'fixed'` (CAG default) at slot 2:
  `['', 1, 'fixed', 'ffmpeg', '', 22.0]`. This is the literal
  pre-BUG-131 state.
- **Verify:** Workflow re-parses; Node 55 widget count = 6
  matches backend `INPUT_TYPES` (5) + CAG (1). Manual canvas
  inspection in ComfyUI shows: seed=1, CAG dropdown reads
  "fixed", ffmpeg="ffmpeg", humo_clips_dir="" (link-only),
  clip_length=22.0.
- **Tags:** workflow-json, widget-drift-CORRECTION,
  control_after_generate, frontend-vs-backend-schema,
  reversal-of-BUG-131
- **Bible candidacy:** yes -- the lesson is *ComfyUI's
  `widgets_values` is NOT 1:1 with backend `INPUT_TYPES`. Any
  node with a `seed: INT` input gets a frontend-injected
  `control_after_generate` widget that must be accounted for in
  position-mapped audits.* The .pyc disassembly verifying
  `INPUT_TYPES` is necessary but insufficient -- it tests the
  Python class, not the frontend serialization contract. The
  diagnostic for "widget count off by exactly one with a seed
  widget" is "CAG slot present? if missing, add it; if extra,
  the extra IS the CAG and is correct."

### BUG-LOCAL-131 [REVERTED]: Node 55 (OTR_BatchLTXRender) widget drift -- BUG-131 trim was incorrect, see BUG-132

- **Date:** 2026-05-08 morning (reverted same day) | **Phase:** 5
- **Status:** Trim shipped in commit `052724d` was REVERTED by
  BUG-LOCAL-132's restore patch. The original 6-entry
  widgets_values WAS correct -- the `'fixed'` value at slot 2 is
  the legitimate CAG widget for the seed input, not stale
  `seed_mode` debris.
- **Lesson:** Position-mapping audits against backend
  `INPUT_TYPES` alone miss the frontend's auto-injected CAG
  widget. See BUG-LOCAL-132 for the full corrected analysis.
- **Tags:** widget-drift, false-positive, CAG-widget,
  reverted-by-BUG-132

- **Date:** 2026-05-08 morning | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `workflows/otr_scifi_16gb_full.json` Node 55
  (`OTR_BatchLTXRender`) had 6 `widgets_values` entries against an
  `INPUT_TYPES` schema that expected 5 widget-renderable inputs.
  Saved layout: `["", 1, "fixed", "ffmpeg", "", 22.0]`. Effective
  position-mapped binding was:
  ```
  slot  expects             gets     consequence
  [0]   ledger_json         ""       link 90 overrides; harmless
  [1]   seed                1        correct
  [2]   ffmpeg              "fixed"  invokes binary "fixed" -> runtime crash
  [3]   humo_clips_dir      "ffmpeg" link 91 overrides; harmless
  [4]   clip_length         ""       empty STRING in FLOAT field
                                     -> ComfyUI validation crash
  [5]   (overflow)          22.0     ignored
  ```
  Even if validation passed (e.g. ComfyUI defaulted `clip_length`
  to 7.0 on empty), the production runs would have produced
  7-second LTX chunks instead of the intended 22-second chunks.
- **Cause:** Same fault class as BUG-LOCAL-097 and BUG-LOCAL-118:
  `INPUT_TYPES` had a `seed_mode` widget at position [2] in a
  prior version, and after that widget was removed every saved
  value shifted left by one slot. ComfyUI position-maps
  `widgets_values[]` -> `INPUT_TYPES`, no name lookup, so
  removing a widget without sweeping every committed workflow
  JSON rotates every saved value past that point.
- **Fix:** JSON-only patch in `workflows/otr_scifi_16gb_full.json`,
  Node 55 `widgets_values` rewritten as 5 entries:
  `["", 1, "ffmpeg", "", 22.0]` (drop the leftover `"fixed"`).
- **Verify:** Workflow re-parses; Node 55 widget count = 5 matches
  the live `INPUT_TYPES` (3 link-only inputs + 5 widget inputs:
  ledger_json STRING, seed INT, ffmpeg STRING, humo_clips_dir
  STRING, clip_length FLOAT). Manual smoke: ComfyUI's `validate`
  on the workflow no longer crashes at the FLOAT-empty-string
  conversion site.
- **Tags:** workflow-json, widget-drift, position-mapping,
  schema-vs-savefile, BUG-097-relative, BUG-118-relative,
  validation-crash
- **Bible candidacy:** yes -- THIRD instance of this fault class
  in the same repo (BUG-097, BUG-118, BUG-131). The class promotes
  immediately and the next pre-soak checklist should explicitly
  include "for every committed workflow JSON, sweep
  `widgets_values[]` length against the current
  `INPUT_TYPES` of the referenced node class."

### BUG-LOCAL-130 [FIXED]: Node 25 (OTR_SaveToEpisodeWorkspace) wired to UnloadAll passthrough instead of BatchFluxRender env stills (caught by 2026-05-08 JSON QA round-robin)

- **Date:** 2026-05-08 morning | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Node 25 (`OTR_SaveToEpisodeWorkspace`,
  `widgets_values=["stills","full_env"]`) was supposed to write
  the BatchFluxRender environment-stills batch to disk under
  `stills/full_env_*.png`. Instead, it received its IMAGE input
  from Node 24 (`OTR_UnloadAll`), which is a passthrough whose
  IMAGE was the portrait_batch from Node 59 (`OTR_PortraitRender`).
  Net effect: every "environment still" file written to disk was
  actually a portrait tensor, and the real env stills produced by
  Node 23 had no on-disk save path.
- **Cause:** Link routing in `workflows/otr_scifi_16gb_full.json`:
  - Node 23 (BatchFluxRender) outputs[0].links = [101]
    (only fed Node 59 PortraitRender, never Node 25)
  - Node 24 (UnloadAll) outputs[0].links = [46, 83]
    (link 46 fed Node 25; link 83 correctly fed Node 51 HuMo)
  - Node 25 inputs[0].link = 46 (the wrong source)
- **Fix:** JSON-only link patch (no Python change):
  - Add new link 104: Node 23 → Node 25 (IMAGE)
  - Remove link 46: Node 24 → Node 25
  - Update Node 23 outputs[0].links: [101] -> [101, 104]
  - Update Node 24 outputs[0].links: [46, 83] -> [83]
  - Update Node 25 inputs[0].link: 46 -> 104
  - Bump last_link_id: 103 -> 104
- **Verify:** ComfyUI loads the workflow without dangling-link
  warnings. Manual node-25 inspection in canvas shows
  `images <- Node 23` after reload. Run 1 of the staged validation
  produces `stills/full_env_*.png` files on disk that match the
  Node 23 environment renders, NOT portraits.
- **Tags:** workflow-json, link-routing, save-to-disk, portrait-vs-env
- **Bible candidacy:** yes -- the lesson is *passthrough nodes
  carry their input forward, not their type*. `OTR_UnloadAll`'s
  IMAGE output is whatever IMAGE its caller fed in, NOT a
  topologically-correct branch of the graph. Whenever a save-to-
  disk node sits downstream of a passthrough, verify the IMAGE
  source by walking the link graph back to its true producer.

### BUG-LOCAL-129 [FIXED]: BatchLTXRender silently routes to v0_9 sampler chain when production JSON loads v2.3 checkpoint (caught by 2026-05-08 JSON QA round-robin)

- **Date:** 2026-05-08 morning | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `OTR_LTX_ENGINE_DEFAULT = "v0_9"` in
  `nodes/batch_ltx_render.py`, but the production workflow loads
  `ltx-2.3-22b-dev.safetensors` (v2.3 family) + the Gemma 3 12B
  encoder (v2.3-path-only). With the env var unset, the node
  routes to the v0_9 sampler chain (CFGGuider + euler_cfg_pp +
  VAEDecodeTiled) and hands packed v2.3 latents to a v0_9-shape
  decoder. Result: tensor shape mismatch crash deep in the sampler
  call OR garbage output that looks like noise.
- **Cause:** Two separate failure modes overlapping:
  1. The default engine ('v0_9') is the smoke-test fast path; the
     production JSON expects 'v2_3'. The mismatch is silent because
     the env var is the only signal connecting them.
  2. The existing fail-fast dep check at line 1009 is asymmetric --
     it validates RES4LYF/LTXVideo deps when engine='v2_3' but does
     NOT validate the loaded checkpoint family when engine='v0_9'.
- **Fix:** Two-part defense in `nodes/batch_ltx_render.py`:
  1. **Loud env-var-unset warning at engine resolve.** When
     `OTR_LTX_ENGINE` is not set in the environment, log a
     prominent WARNING explaining the production workflow is rigged
     for v2.3 and pointing the user at the env var fix. Doesn't
     change behavior; makes the silent failure mode visible.
  2. **Symmetric defensive guard.** Check `_otr_ckpt_name`
     (a forward-compat attribute that future loaders can stamp)
     against the resolved engine. Raise RuntimeError on mismatch
     in EITHER direction (v0_9 with 22B checkpoint OR v2_3 with
     2B v0.9 checkpoint). Today this is forward-compat -- stock
     CheckpointLoaderSimple doesn't stamp the name, so the guard
     is a no-op and falls back to the warning. When a future
     LowVRAMCheckpointLoader stamps it, the guard activates
     automatically.
  3. **Operational fix (NOT in this commit):** Jeffrey sets
     `OTR_LTX_ENGINE=v2_3` in his launch environment (HKCU\\Environment
     for Windows User scope, or in a launcher .cmd if he uses one).
     Required before the next 6-line validation soak.
- **Verify:** AST parse clean. Existing LTX regression 33/33 still
  passes (the new guard only raises when `_otr_ckpt_name` is
  actually stamped; in the test environment it isn't, so the
  guard's `try` block exits via the broad-except fall-through).
  Live verification deferred to the validation soak with the
  env var set.
- **Tags:** ltx, engine-selector, env-var, checkpoint-mismatch,
  defensive-guard
- **Bible candidacy:** yes -- the lesson is *engine selectors
  controlled by env vars are silent by default; production
  workflows that depend on a non-default value need a loud
  reminder + a defensive guard, OR the default needs to flip*.
  The "fail-fast on missing dep" pattern in the same file
  (line 1009) was already half the answer; this commit completes
  the symmetry.

### BUG-LOCAL-128 [FIXED]: HuMo soak cap counter conflates resumed + fresh; cap fires after 1 fresh render on resume (caught by 2026-05-08 external round-robin synthesis Item A)

- **Date:** 2026-05-08 morning | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Single `rendered` counter in `batch_humo_render.py`
  was incremented on BOTH the resume branch and the fresh-render
  branch. The cap check (`rendered >= humo_max_lines_per_process`)
  fired off the combined value. A resume run that found 6
  existing clips on disk would start the loop with `rendered == 6`
  and trigger the cap after **one** fresh render -- the opposite
  of the intended "render six fresh clips per process" behavior.
- **Cause:** Counter accounting ambiguity. The `rendered` name
  was load-bearing for both the cap check (which wanted "fresh
  work this process") and the report/return value (which wanted
  "total clips on disk including resumed").
- **Fix:** Split into two counters per round-robin synthesis Item A:
  - `total_clips_output` -- resumed + fresh; used for report
    line, completion log, and the node's INT return value
  - `fresh_rendered_this_process` -- fresh HuMo work only; used
    for the cap check, `HumoSoakCapReached.lines_completed`, and
    `meta.soak_cap.lines_completed_when_hit`
  Resume branch now bumps only `total_clips_output`. Fresh-render
  branch bumps both.
- **Verify:** All existing tests still pass; cap behavior is
  empirically validated by Run 1 (fresh start, expect 6 fresh
  clips + cap raise) + Run 2 (immediate resume, expect 6 resumed
  + 6 more fresh + cap raise -- diagnostic for whether the fix
  landed correctly).
- **Tags:** counter-bug, soak-cap, resume-from-ledger, off-by-N,
  external-round-robin
- **Bible candidacy:** yes -- the lesson is *one counter cannot
  serve two semantics*. Whenever a helper has both "total work
  done" and "incremental work this iteration" meanings layered on
  the same name, the bookkeeping is fragile to the resume / retry
  case. The two-counter pattern + fresh-only-for-cap pairing is
  generic.

### BUG-LOCAL-127 [FIXED]: save_ledger_safe non-atomic Path.write_text bricks ledger on hard crash mid-write (caught by 2026-05-08 ledger round-robin)

- **Date:** 2026-05-08 morning | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** `save_ledger_safe` in `nodes/_otr_ledger.py` used
  `Path(path).write_text(...)` -- NON-ATOMIC. On a system designed
  to soak through OOMs (BUG-LOCAL-126), a hard CUDA crash during
  an in-progress ledger write would leave a 0-byte / truncated
  JSON, destroying the production state for that episode.
- **Cause:** Identical pattern to BUG-LOCAL-124's `lock_to_episode`
  fault. The ledger is the single source of truth for every
  downstream node + the multi-batch resume path; non-atomic write
  on a recovery-hot path is load-bearing for both BUG-122
  (lock_to_episode) AND BUG-126 (HuMo soak survival). Identifying
  one fault made the other inevitable.
- **Fix:** `save_ledger_safe` now uses `tempfile.mkstemp` +
  `os.fsync` + `os.replace` -- same atomic-rename pattern as
  BUG-LOCAL-124. Same-directory tempfile prefixed
  `.ledger.save.*.tmp.json` so cross-fs replace can't trip EXDEV.
  Cleanup path unlinks the temp on any write/replace failure so
  partial debris doesn't accumulate.
- **Verify:**
  - `python -m pytest tests/test_ledger_l3_2026_05_08.py -v` ->
    27/27 PASSED including
    `test_save_ledger_safe_writes_atomically` (asserts only the
    final file exists post-save, no temp debris) and
    `test_save_ledger_safe_overwrites_existing` (idempotent
    overwrite).
- **Tags:** ledger, atomic-write, os-replace, BUG-124-relative,
  recovery-hot-path
- **Bible candidacy:** yes -- second instance in 24 hours of the
  same fault class (non-atomic JSON write to a single source of
  truth) on a system that can crash mid-write. The class lesson
  promotes immediately.

### BUG-LOCAL-126 [FIXED]: HuMo soak terminates mid-episode with `Fatal Python error: Aborted` after recovered OOM cycles fragment the CUDA pool

- **Date:** 2026-05-08 morning (post-mortem of the
  `signal_lost_signal_from_the_red_dust_20260507_221546` run)
  | **Phase:** 5 | **Bible candidate:** yes (when fix lands)
- **Symptom:** Overnight FULL acceptance soak rendered 9 of 56
  planned lines (l002-l009) over 2h 14min wall time, then ComfyUI
  process aborted at 00:29:11 with `Fatal Python error: Aborted`
  five seconds after the previous line (l009) finished cleanly. No
  `[BatchHumoRender] line lXXX failed` message before the crash --
  this was a C-level abort (SIGABRT-equivalent on Windows), not a
  caught Python exception. Lines l010-l056 never rendered. Final
  composite + RTXUpscale + procgen blend never ran.
- **Cause (preliminary):** Cumulative CUDA pool fragmentation. The
  log shows the pipeline survived TWO caught HuMo OOMs earlier in
  the run:
  - `[2026-05-07 22:45:20] [BatchHumoRender] line l002 failed:
    Allocation on device 0 would exceed allowed memory. Currently
    allocated: 24.50 GiB, Device limit: 15.92 GiB`
  - `[2026-05-07 23:52:56] [BatchHumoRender] line l004 failed:
    Allocation on device 0 would exceed allowed memory. Currently
    allocated: 24.50 GiB, Device limit: 15.92 GiB`
  Both caught OOMs left HuMo with chunk01 saved but chunk02 lost.
  The pipeline kept going (l003, l005, l007, l008, l009 all
  succeeded -- l006 likely fell through to static-radio fill per
  BUG-LOCAL-129a). After ~2 hours of repeated HuMo
  load/unload/sample/recover cycles, the next sample call
  (presumably starting l010) triggered a fatal-level CUDA abort
  instead of a clean OOM exception. The "24.50 GiB allocated on a
  15.92 GiB device" line in the OOM messages indicates PyTorch's
  pool drifted way above the real device limit -- consistent with
  pool fragmentation rather than legitimate working-set growth.
- **Stack trace:** Final crash at
  `comfy/samplers.py:325 _calc_cond_batch ->
  uni_pc.py:868 sample_unipc -> nodes.py:1556 common_ksampler ->
  batch_humo_render.py:1888 execute`. Same KSampler/UniPC code
  path that handled the earlier caught OOMs. The abort came from
  inside the C extensions (PyTorch's CUDA allocator), so no Python
  try/except could intercept it.
- **Fix (commit pending; live-verify on next soak):** Implemented the
  alarm-plumbing carve-out per Jeffrey's directive ("if things are
  crashing the workflow we do kinda have to chase it" -- 2026-05-08
  morning). Two parts in `nodes/batch_humo_render.py`:

  1. **Hard CUDA reset after caught OOM.** Module-top helpers
     `_is_oom_exception(exc)` and `_hard_reset_cuda_context()` plus
     a new `cuda_hard_reset_on_oom` BOOLEAN INPUT_TYPES widget
     (default ON). On a caught OOM in the per-line render loop, the
     except path calls the helper which runs:
     `mm.unload_all_models() -> gc.collect() ->
     mm.soft_empty_cache(force=True) -> torch.cuda.synchronize() ->
     torch.cuda.empty_cache() -> torch.cuda.ipc_collect()`. Every
     step is best-effort; missing API on an older torch / Comfy
     combo is logged as a partial (NOT a re-raise -- recovery code
     can't escalate the fault).

  2. **Per-process HuMo soak cap.** New
     `humo_max_lines_per_process` INT INPUT_TYPES widget
     (default 0 = disabled). When > 0, after each successful HuMo
     line the loop checks `rendered >= cap`; if reached, persists
     the ledger via the existing per-clip incremental save path,
     then raises `HumoSoakCapReached(lines_completed, cap)` -- a
     structured `RuntimeError` subclass that distinguishes
     soft-stop-by-design from random fault. Pairs with
     `resume_from_ledger=True` so a follow-up ComfyUI run picks
     up where the cap fired. Empirical recommendation in the
     widget tooltip: 6-8 on the RTX 5080 16 GB while the underlying
     allocator drift is investigated; the overnight 2026-05-07 soak
     survived 9 lines before fatal-aborting.

  Plus alarm visibility: added `"Fatal Python error: Aborted"` to
  `scripts/audit_otr_full_run.py` `FAIL_PATTERNS` so the watcher /
  auditor surfaces this hard-abort signal instead of reporting
  PASS on a quiet-but-aborted dir.

- **Verify (unit):**
  - `python -m pytest tests/test_humo_oom_recovery.py -v` -> 9/9
    PASSED. Tests cover: `_is_oom_exception` matches both
    `torch.OutOfMemoryError` and the legacy `RuntimeError("...out
    of memory...")` form (case-insensitive); `HumoSoakCapReached`
    carries `lines_completed` and `cap` attributes and is a
    `RuntimeError` subclass; `_hard_reset_cuda_context()` does
    not raise even when torch / mm aren't available; the audit
    script's FAIL_PATTERNS includes the
    `"Fatal Python error: Aborted"` literal.
  - LTX regression: 33/33 unchanged.
  - Full Phase 0+ suite: 113/113 unchanged.

- **Verify (live, pending):** Next FULL acceptance soak with
  `humo_max_lines_per_process=6` and `cuda_hard_reset_on_oom=True`.
  Expected: either (a) soak runs 6 lines clean and raises
  HumoSoakCapReached, ledger persisted; or (b) a caught OOM
  triggers the hard-reset and the next line renders. Either path
  is success. Failure mode to watch: a third
  `Fatal Python error: Aborted` despite the reset being run --
  that escalates the fix to BUG-LOCAL-050-style chained-teardown
  investigation (Wan2.1-14B + Whisper + umt5_xxl model patches not
  releasing PCIe state on unload).
- **Tags:** vram, humo, cuda-abort, soak-termination, pool-
  fragmentation, BUG-050-relative, fatal-python-error
- **Related:** BUG-LOCAL-050 (chained backend teardown -- same
  family of cumulative-state fault); BUG-LOCAL-086 (HuMo chunked
  + concat -- the chunking that saved chunk01 from each caught
  OOM); BUG-LOCAL-129a (static-radio fill -- the path that probably
  filled l006).
- **Bible candidacy:** yes (when fixed) -- the lesson is that
  *recovered CUDA OOMs leave the allocator pool in a degraded
  state that catches up downstream*. A defensible diffusion-
  pipeline soak design must either prevent OOM entirely or hard-
  reset the CUDA context between recovered OOMs.

### BUG-LOCAL-125 [FIXED]: _voice_backends registry empty on first lookup if drivers not separately imported (caught by 2026-05-08 morning round-robin Element 4)

- **Date:** 2026-05-08 morning | **Phase:** 0+ | **Bible candidate:** yes
- **Symptom:** First call to `get_factory("bark")` from a process that
  imported only `from nodes._voice_backends import get_factory`
  (without separately importing `nodes._voice_backends.bark` etc.)
  raises `KeyError: voice backend 'bark' not registered (currently
  registered: [])`. `OTR_VoiceRender.render()` would have hit this on
  every fresh ComfyUI startup once registered.
- **Cause:** Bundled drivers self-register at module import time via
  `register("bark", BarkBackend)` at module scope. The package
  `__init__.py` did NOT auto-trigger that import; it provided a
  `_register_default_drivers()` helper but never called it. So the
  registry stayed empty until a caller separately imported each
  driver module.
- **Fix:** Added `_ensure_defaults_registered()` -- once-per-process
  guard that fires `_register_default_drivers()` on first call.
  `get_factory()` and `available_engines()` both call it before
  returning. Catches ImportError silently so a missing optional driver
  doesn't crash the registry-shape introspection path; surfaces the
  empty-registry message via the standard KeyError instead.
- **Verify:**
  - `python -m pytest tests/test_voice_backends.py -v` -> 19/19 PASSED
    (added `test_get_factory_lazy_initializes_default_drivers` and
    `test_available_engines_lazy_initializes_default_drivers`,
    each using a fresh-process simulator that clears `sys.modules`
    AND the package's cached submodule attributes).
- **Tags:** phase-0+, voice-backends, registry, lazy-init,
  round-robin-catch
- **Bible candidacy:** yes -- the lesson is that *self-registering
  driver modules need an explicit trigger from the registry's own
  getters*. A `_register_default_drivers()` function that exists but
  is never called is decorative.

### BUG-LOCAL-124 [FIXED]: lock_to_episode non-atomic write_text bricks episode dir on crash (caught by 2026-05-08 morning round-robin Element 1)

- **Date:** 2026-05-08 morning | **Phase:** 0+ | **Bible candidate:** yes
- **Symptom:** ComfyUI process killed (manual kill, OOM, power loss)
  during `locked_path.write_text(...)` would leave a 0-byte or
  truncated `cast_contract.locked.json` on disk. The next run's
  read-and-compare-version path (BUG-LOCAL-122) hits the
  `json.JSONDecodeError` branch and raises a fatal `RuntimeError`,
  permanently bricking that episode dir until manually deleted.
- **Cause:** `Path.write_text()` is NOT atomic. Python writes incrementally
  to the target inode; an interrupt mid-write leaves whatever was
  flushed so far on disk. The BUG-LOCAL-122 read-and-compare-version
  fix correctly refused to overwrite a corrupt locked file (data-
  loss prevention), which made this fragility load-bearing.
- **Fix:** Switched to `tempfile.mkstemp()` in the same dir + write
  + `os.fsync()` + `os.replace()`. Atomic rename guarantees the
  final lockfile is either fully present or absent. Cleanup path
  unlinks the temp file if anything in the write/replace sequence
  raises so we don't leave debris like
  `.cast_contract.lock.abc123.tmp`.
- **Verify:**
  - `python -m pytest tests/test_cast_contract.py -v` -> 13/13 PASSED
    (added `test_lock_to_episode_writes_atomically_no_temp_debris`
    and `test_lock_to_episode_truncated_existing_raises_unreadable`).
- **Tags:** phase-0+, cast-contract, lock-to-episode, atomic-write,
  os-replace, round-robin-catch
- **Related:** BUG-LOCAL-122 (the read-and-compare-version path that
  this atomic write protects).
- **Bible candidacy:** yes -- the lesson is *any helper that writes
  state another node will read back must use atomic file replacement*,
  not `Path.write_text()`. The pairing of "refuse to overwrite corrupt"
  + "non-atomic write" is the actual brick: either alone is survivable.

### BUG-LOCAL-123 [FIXED]: repair_orphans plateau crashes when classifier returns DISCARD/NARRATIVE_LEAK/GENUINELY_NEW (caught by 2026-05-08 morning round-robin Element 3)

- **Date:** 2026-05-08 morning | **Phase:** 0+ | **Bible candidate:** yes
- **Symptom:** A classifier that correctly buckets an orphan as
  DISCARD (stage-direction noise like `FOOTSTEPS:`),
  NARRATIVE_LEAK (description leaked into a tag like
  `THE LIGHTHOUSE:`), or GENUINELY_NEW (a new character the cast
  doesn't have) would crash the repair loop with
  `CastContractUnreparable` on iteration 2. The 5-bucket Enum was
  designed for exactly these dispositions but the loop's plateau
  detector treated them as failure.
- **Cause:** `apply_classifications` only mutates the contract for
  TYPO_OF_EXISTING / ALIAS_OF_EXISTING. The other 3 buckets are no-
  ops by design (caller handles reroll / demote / drop outside the
  loop). But the loop recomputed `residual_orphans` purely from
  contract lookup -- so the un-mutated orphans stayed in the
  residual, equaled the previous iteration's residual, and triggered
  the plateau raise. The classifier's correct decision was being
  treated as no-progress.
- **Fix:** Track a `decided_residuals: set[str]` of orphans the
  classifier explicitly bucketed as DISCARD / NARRATIVE_LEAK /
  GENUINELY_NEW, and subtract that from the residual on every
  iteration. Now a classifier that decides "this orphan is not an
  alias" correctly resolves the orphan instead of stalling the loop.
- **Verify:**
  - `python -m pytest tests/test_cast_repair.py -v` -> 22/22 PASSED
    (replaced
    `test_repair_orphans_invokes_classifier_for_residual` to assert
    the classifier-decided GENUINELY_NEW path returns RepairOutcome
    cleanly, added 3 new tests:
    `test_repair_orphans_discard_bucket_does_not_plateau`,
    `test_repair_orphans_narrative_leak_bucket_does_not_plateau`,
    `test_repair_orphans_mixed_buckets_in_one_pass`).
  - Plateau path tested via
    `test_repair_orphans_plateau_raises_when_classifier_returns_alias_with_unknown_id`
    -- only mutating-bucket-but-can't-apply triggers genuine plateau.
- **Tags:** phase-0+, cast-contract, repair-loop, plateau-detection,
  round-robin-catch
- **Bible candidacy:** yes -- the lesson is *plateau detection on
  raw residuals conflates "can't progress" with "explicitly
  decided"*. Any iterative resolver with a multi-bucket disposition
  must distinguish "still unknown" from "decided non-mutating".

### BUG-LOCAL-122 [FIXED]: lock_to_episode blind-refusal breaks ComfyUI rerun / crash recovery (caught by 2026-05-08 round-robin Element 2)

- **Date:** 2026-05-08 early-AM | **Phase:** 0+ | **Bible candidate:** yes
- **Symptom:** Original `lock_to_episode` raised `RuntimeError` whenever
  `cast_contract.locked.json` already existed in the episode dir,
  regardless of whether the in-memory contract matched it. ComfyUI
  re-queues the same prompt on rerun, partial regenerations point at
  the same `episode_dir`, and crash-recovery flows all hit this path
  -- in every case the run hard-failed before any production work
  resumed. ChatGPT (Element 2: 40% red) and Gemini both flagged this
  as the load-bearing weak spot in the Element 2 review.
- **Cause:** Refusal logic was content-blind -- a pure `.exists()`
  check rather than a contract-version compare. The whole point of the
  sha-8 version stamp from §1 was to *enable* this kind of
  content-addressed comparison; the lock helper just wasn't using it.
- **Fix:** Read-and-compare-version path. New `CastContractMismatch`
  exception class (subclass of `RuntimeError` so existing callers that
  catch RuntimeError still work). Logic:
  1. Always stamp version on the in-memory contract first.
  2. If locked file does not exist -> write it, return path.
  3. If locked file exists and parses to a contract whose
     `version` field equals the in-memory contract's version ->
     pass through (return existing path, no rewrite, no raise).
  4. If locked file exists and version differs -> raise
     `CastContractMismatch` with both versions in the message.
  5. If locked file exists and is unreadable / corrupt JSON ->
     raise `RuntimeError` with the underlying exception (do NOT
     silently overwrite a corrupt file -- that's a real fault to
     surface).
- **Verify:**
  - `python -m pytest tests/test_cast_contract.py -v` -> 11/11 PASSED
    (was 9; replaced
    `test_lock_to_episode_writes_file_and_refuses_overwrite` with
    `test_lock_to_episode_writes_file_and_idempotent_on_same_version`,
    added
    `test_lock_to_episode_raises_on_version_mismatch` and
    `test_lock_to_episode_raises_on_corrupt_existing`).
  - Full cast contract suite: 52/52 PASSED.
  - LTX regression: 33/33 PASSED unchanged.
- **Tags:** phase-0+, cast-contract, lock-to-episode, comfyui-rerun,
  content-addressed, round-robin-catch
- **Related:** BUG-LOCAL-120 parent Phase 0+ work; BUG-LOCAL-121
  sibling round-robin catch in the same code-review pass; round-robin
  transcripts at
  `docs/2026-05-08-cast-contract-shipped-code-review__01_chatgpt.md`
  and `__02_gemini.md`.
- **Bible candidacy:** yes -- the lesson is *content-blind refusal
  defeats the value of content-addressed versioning*. Whenever a
  helper has a sha / hash / version stamp on the data structure AND a
  "refuse to overwrite" guard, the guard MUST compare the stamps, not
  just file presence. Otherwise the version field is decorative.

### BUG-LOCAL-121 [FIXED]: build_contract_from_director_plan KeyError on padded voice_assignments keys (caught by 2026-05-08 round-robin)

- **Date:** 2026-05-08 early-AM | **Phase:** 0+ | **Bible candidate:** yes
- **Symptom:** `build_contract_from_director_plan()` would raise
  `KeyError` if any key in `director_plan["voice_assignments"]` had
  surrounding whitespace. Example failing input:
  `{"voice_assignments": {"  MONTY  ": "v2/en_speaker_3"}}` -- the
  generator expression stripped the key to `"MONTY"`, then
  `assignments["MONTY"]` raised because the original dict still keyed
  by `"  MONTY  "`. Pure stdlib helper -- no torch, no VRAM.
- **Cause:** The clean-key derivation and the value-lookup were
  decoupled. `sorted_names` came from a stripped-key generator;
  `assignments[name]` indexed the ORIGINAL dict with the stripped
  name. ChatGPT's gpt-5.5 round of the round-robin caught this on the
  first pass; Gemini's gemini-3.1-pro-preview-customtools confirmed
  the trap and recommended the same `clean_assignments` rebuild fix.
- **Fix:** Rebuild `clean_assignments: dict[str, object]` once -- key
  is the stripped name, value is the original raw value. Use
  `setdefault` so a collision (`"MONTY"` + `"  MONTY  "` after
  stripping) deterministically keeps the FIRST occurrence rather than
  silently overwriting. Then iterate `sorted(clean_assignments.keys(),
  key=str.upper)` and look up via `clean_assignments[name]`.
- **Verify (commit pending):**
  - `python -m pytest tests/test_cast_contract_helpers.py -v` -> 22/22
    PASSED including two new regression tests:
    `test_build_contract_handles_padded_keys_without_keyerror` (mixed
    whitespace + tab/newline padded keys), and
    `test_build_contract_padded_collision_first_wins` (exercises the
    setdefault collision rule).
  - Full cast contract suite: 50/50 PASSED (unchanged behavior on the
    other 48 tests).
  - LTX regression: 33/33 PASSED unchanged.
- **Tags:** phase-0+, cast-contract, helpers, keyerror, padded-keys,
  round-robin-catch
- **Related:** BUG-LOCAL-120 (the parent Phase 0+ Cast Contract work --
  this fix lands on the same `dfe26e6` Phase B helper that just
  shipped).
- **Round-robin transcript:**
  `docs/2026-05-08-cast-contract-shipped-code-review__01_chatgpt.md`
  (Element 4, fix probability 55%, RED -- the highest non-Element-2
  risk score in the review). Gemini transcript at
  `__02_gemini.md`. Synthesis at `__04_synthesis.md`.
- **Bible candidacy:** yes -- the lesson is the *clean-derivation /
  raw-lookup decoupling antipattern*. Whenever a helper normalizes a
  collection of keys (strip / lower / case-fold) and then iterates the
  normalized set, every downstream lookup MUST go through the
  normalized lookup map -- not the original collection. Generic enough
  to land in the Bible as a recurring class.

---

## Workflow tip — ask Claude for a live risk artifact after a fix

When Claude has shipped a non-trivial fix and you want a quick gut-check on residual risk WITHOUT triggering more code changes, you can ask: "round-robin the shipped code and give me a live artifact with your % chance of fix needed." Claude will:

1. Write a code-review-shaped consult question (not a prevention-plan question — the framing matters).
2. Run `scripts/_consult_round_robin.py` against the shipped code (ChatGPT + Gemini + NVIDIA Nemotron).
3. Read the transcripts under `docs/<date>-<topic>__01_chatgpt.md` etc.
4. Render an inline artifact card-grid with one card per fix element. Each card shows: one-line description + per-element follow-up-fix probability % + ChatGPT verdict badge + Gemini verdict badge + one-line reasoning. Color-coded by risk tier (green <15%, amber 15-30%, red 30%+).
5. Below the cards: "what to watch in next soak" callout + "where they disagreed" section + sources footer.

**Proven pattern (2026-05-03 EVENING for BUG-027 + BUG-028):** transcripts at `docs/2026-05-03-bug-027-028-shipped-code-review__*.md`, commit `832d134`. ChatGPT (108.6s, 21 KB) + Gemini (34.2s, 4.5 KB) converged; NVIDIA round failed silently but 2-of-3 was sufficient signal. The artifact made residual risk landscape immediately legible — load-bearing weak spot ("BUG-028 Site 3: humo wildcard glob, 40%") jumped out at first glance instead of being buried in 25 KB of model prose.

**Skip when:** fix is trivial (one-line typo, mechanical edit), or you've already moved on. **Use when:** tough decision area (LLM prompt templates, audio C7, VRAM determinism, save paths, anything with multi-site write+read alignment) and you want peace of mind without round-tripping through more text walls. Round-robin alone is text-heavy; the artifact is the part that makes it visually decision-ready.

---

### BUG-LOCAL-120 [IN PROGRESS]: Phase 0+ Cast Contract Extensions §1+§2+§3 — skeletons + helpers landed; orchestrator integration deferred

- **Date:** 2026-05-07 LATE | **Phase:** 0+ | **Bible candidate:** yes (post-soak)
- **Symptom:** Same root cause as the character drift observed in the 2026-05-07 PM
  runs (Run 2 + Run 3): `MONTY` / `MONTGOMERY` name leak in dialogue tags +
  silent `AEGEUS` -> `MONTY` voice preset pooling at BatchBark group time +
  no canonical roster the ScriptWriter prompt could be held to.
- **Cause:** Cast roster lives implicitly in the Director plan's
  `voice_assignments` dict; no first-class data contract; no version stamping;
  no per-episode lock; no character canon (identity layer separate from
  routing). Documented in the ROADMAP "Phase 0+ candidates" -> "Cast Contract
  Extensions" section as a 5-part design (§1 versioning + §2 lock + §3 canon
  + §4 adversarial classification + §5 plateau-bounded repair).
- **Fix (commit `b8c26f4` -- predecessor baseline, 2026-05-07 LATE):**
  Three skeleton modules + 28 unit tests, scope §1+§2+§3 only:
  - `nodes/_otr_cast_contract.py` — `CharacterEntry`, `CastContract`,
    content-addressed sha-8 versioning (`stamp_version`), alias-aware
    `lookup`, `lock_to_episode` (immutable per §2),
    `load_locked`. Module is stdlib-only (no torch/VRAM coupling).
  - `nodes/_otr_voice_resolver.py` — `VoiceSpec` (frozen dataclass),
    `parse_voice_spec("engine:preset")` with forward-compat unknown-engine
    pass-through. `KNOWN_ENGINES = {bark, kokoro, cosyvoice, xtts, piper}`.
  - `nodes/_otr_canon.py` — `CharacterCanonEntry`, `render_canon_markdown`
    (omits empty fields for terser prompt), `write_canon` /
    `load_canon` round-trip via `character_canon.md`.
  - `tests/test_cast_contract.py` (9 tests), `tests/test_voice_resolver.py`
    (12 tests), `tests/test_canon.py` (7 tests) -- 28 total green.
- **Fix (commit `dfe26e6` -- Phase B helpers, 2026-05-08 early-AM autonomous
  session):** Two pure-stdlib §1 helpers added to
  `nodes/_otr_cast_contract.py`, plus a 20-test suite at
  `tests/test_cast_contract_helpers.py`:
  - `build_contract_from_director_plan(director_plan) -> CastContract` --
    reads `voice_assignments`, sorts canonical names alphabetically,
    assigns stable `c01`/`c02`/... ids in that order, defaults voice
    spec engine prefix to `bark:` when omitted, accepts both
    `"engine:preset"` string form and `{"engine": ..., "preset": ...}`
    dict form, stamps the sha-8 version before return. Empty/None plan
    returns an empty versioned contract (caller decides whether that's
    a hard failure).
  - `detect_aliases(script, contract) -> dict[str, str]` -- pure
    heuristic alias detector. Scans uppercase dialogue tags via
    `_extract_dialogue_tags` (filters structural headers SCENE / ACT /
    FADE / INT / EXT / NARRATOR), skips tags already in canonical or
    alias list, then checks for >=4-character shared prefix in either
    direction (truncation MONTGOMERY -> MONTY *and* expansion MONT ->
    MONTY). First-match-wins on prefix collisions; §4 is the canonical
    disambiguator. Returns first-seen-order dict so merge logs are
    deterministic. No mutation of contract; no LLM call.
- **Verify (commit `dfe26e6`):**
  - `python -m pytest tests/test_cast_contract.py tests/test_voice_resolver.py tests/test_canon.py tests/test_cast_contract_helpers.py -q`
    -> 48/48 PASSED (28 baseline + 20 helpers).
  - `python -m pytest tests/test_batch_ltx_render.py -q` -> 33/33 PASSED
    (handoff brief stated 23/23; the suite has grown to 33 -- still
    unchanged by Phase 0+ work, no regression).
  - AST clean across `nodes/_otr_cast_contract.py`,
    `nodes/_otr_voice_resolver.py`, `nodes/_otr_canon.py`,
    `tests/test_cast_contract_helpers.py`.
- **Tags:** phase-0+, cast-contract, character-drift, autonovel-pattern,
  skeleton, helpers, deferred-integration
- **Related:** ROADMAP.md "Phase 0+ candidates" sections; the autonovel
  reference (`https://github.com/NousResearch/autonovel`) for source patterns
  (state.json versioning, characters/canon split). BUG-LOCAL-118 (workflow
  widget order — sibling Phase 0 fix that proves the pipeline plumbing).
- **Next steps (deferred to follow-up session -- story_orchestrator.py
  integration was *not* attempted this session because the FULL acceptance
  soak `signal_lost_signal_from_the_red_dust_20260507_221546` was still
  in flight at session start, mid-LTX render with newest file written
  ~14 min before phase-check; per the standing rule we do not edit
  story_orchestrator.py / production_ledger.py / scene_sequencer.py /
  batch_bark_generator.py / batch_humo_render.py / video_composite.py
  while a soak is rendering):**
  - `story_orchestrator.py` integration at lines 6423 (§1 hook: build
    `CastContract` from Director plan post-`_parse_script`, stamp version
    into ledger), ~640 (§2 hook: `lock_to_episode(contract, episode_dir)`
    after `_bark_health_check_for_cast`), 920 (§3 hook: feed canon into
    `_check_voice_consistency` as rubric input).
  - `production_ledger.py` merge guard requiring matching
    `cast_contract_version`.
  - §4 (adversarial classification of orphan tags into 5 buckets:
    TYPO_OF_EXISTING / ALIAS_OF_EXISTING / GENUINELY_NEW /
    NARRATIVE_LEAK / DISCARD) -- this is where the helper
    `detect_aliases` becomes the cheap fast-path; the adversarial
    LLM classifier handles only the residual (genuinely ambiguous)
    cases.
  - §5 (plateau-bounded repair loop: 3 attempts max, raise
    `CastContractUnreparable` if no progress between iterations).
  - Voice Backend Abstraction is a separate Phase 0+ candidate; sequenced
    AFTER Cast Contract proves out end-to-end.
- **Bible candidacy:** yes -- the lesson is that *implicit cast rosters embedded
  in Director plan dicts cannot survive multi-LLM-pass scripts*. A versioned,
  episode-locked, canonical contract is the substrate every downstream node
  (ScriptCritic, BatchBark, KokoroAnnouncer, BatchHumoRender) should read from
  rather than guessing from dialogue tag strings. The Phase B helpers
  (`build_contract_from_director_plan` + `detect_aliases`) close the
  "translate the implicit Director-plan dict into a versioned contract,
  then reconcile the LLM's drifted dialogue tags against it" loop without
  any LLM calls -- that's what makes §4 viable as a *fallback* rather
  than the primary classifier. Promote when story_orchestrator integration
  is verified end-to-end on a real soak.

### BUG-LOCAL-119 [FIXED]: audit_otr_full_run.py FAIL_PATTERNS includes two strings that false-positive on every healthy run

- **Date:** 2026-05-07 PM | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Run 2 of Jeffrey's 2026-05-07 PM soak
  (`signal_lost_silent_countdown_20260507_193951`) completed end-to-end with
  the final composite + RTXUpscale + procgen blend all written to disk and
  audio C7 byte-identity preserved -- but two log strings in
  `scripts/audit_otr_full_run.py` `FAIL_PATTERNS` would have flagged the run
  as FAILED:
  1. `"strict_c7=True"` -- printed in the VideoComposite banner of EVERY
     healthy run (it's a config-flag indicator, not a failure signal):
     `[VideoComposite] audio_source=master_mix_per_clip_mux ... strict_c7=True`.
     The actual strict_c7 raise path produces a Python exception, not this
     string; including the string-form was a category error.
  2. `"duration contract VIOLATED"` -- `nodes/video_composite.py` line 1383
     warns this string in BOTH directions of the contract check. The
     benign direction (video LONGER than audio, e.g. Run 2's
     `delta -0.539s`) tags the same line with `audio C7 preserved` and
     only causes -shortest to clip trailing video; nothing breaks. The
     breaking direction (audio overruns video) attempts a tail-pad
     fallback and on tail-pad failure logs `audio may be truncated` --
     which is already a separate (correct) hard-fail pattern. So matching
     `"duration contract VIOLATED"` as a flat string conflates benign with
     breaking.
- **Cause:** Audit-script pattern-list authored without checking that the
  matched strings only appear in failure paths. `strict_c7=True` is a
  literal banner string from the production happy path; `duration contract
  VIOLATED` is a bidirectional check log line whose breaking direction has
  its own dedicated signal.
- **Fix:** Edited `scripts/audit_otr_full_run.py` `FAIL_PATTERNS` tuple:
  - Removed `"strict_c7=True"` (banner literal, not a failure).
  - Replaced `"duration contract VIOLATED"` with the regex
    `re.compile(r"duration contract VIOLATED(?!.*audio C7 preserved)")` --
    negative-lookahead on the same line excludes the benign-direction
    instance while still catching breaking-direction instances. Python
    regex `.` doesn't match newlines by default so the lookahead is
    correctly per-line.
  - Kept `"audio may be truncated"` as the canonical breaking signal for
    the tail-pad-failed path.
  - Added a triage block of comments above `FAIL_PATTERNS` documenting
    each surviving entry's intent.
- **Verify:** With the audit script under `python -B exec(open(...).read())`
  to bypass importlib caching, four-case smoke confirms:
  - benign duration contract VIOLATED line -> PASS (audit clean)
  - breaking direction (no `audio C7 preserved` on same line) -> FAIL caught
  - healthy banner with `strict_c7=True` -> PASS (audit clean)
  - synthetic `audio may be truncated by 1.500s` -> FAIL caught
- **Tags:** audit-script, fail-patterns, duration-contract, false-positive,
  c7-byte-identity, regex-lookahead
- **Related:** BUG-LOCAL-084 (the duration-contract check itself, working
  as designed). BUG-LOCAL-128 (tail-pad +0.500s, occasionally pushes
  benign-direction delta over the 0.040s tolerance).
- **Bible candidacy:** yes -- the lesson is *audit/health-check pattern
  lists must be authored against actual production happy-path log corpus*,
  not against intuited failure messages. Including a string that prints in
  the success banner would have caused every FULL run to FAIL the audit
  pre-fix, masking real failures.

### BUG-LOCAL-118 [FIXED]: otr_scifi_16gb_full.json widgets_values stale after BUG-113 humo_pillar_width move (live-run validation failure)

- **Date:** 2026-05-07 PM | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Mid-run (Jeffrey's manual FULL acceptance soak), `comfyui_8000.log`
  showed three validation errors at `got prompt` time, repeating on every queue:
  ```
  Failed to validate prompt for output 58:
  * OTR_VideoComposite 52:
    - Value not in list: audio_source: 'False' not in ['master_mix_per_clip_mux', 'humo_concat', 'master_mix']
    - Value 512.0 bigger than max of 9.0: fallback_clip_length
    - Value 1 smaller than min of 128: humo_pillar_width
  Output will be ignored
  ```
  VideoComposite (and downstream output nodes 56, 58) were excluded from
  the execution plan; the rest of the pipeline (story -> Bark -> procgen
  -> FLUX -> LTX -> HuMo -> upscale) ran but no final composite was
  produced.
- **Cause:** BUG-LOCAL-113 (2026-05-06) MOVED `humo_pillar_width` from
  between `humo_target_height` and `fallback_clip_length` to the END of
  the optional dict in `OTR_VideoComposite.INPUT_TYPES` (so that saved
  workflows would backfill the new slot with the default cleanly). The
  fix on the node side was correct, but `workflows/otr_scifi_16gb_full.json`
  still had the OLD positional order in `widgets_values[]`, so on this
  workflow's first use after the schema change, ComfyUI mapped saved
  values to the new slots and produced three off-by-one errors: 512
  (old humo_pillar_width default) landed in the fallback_clip_length
  slot (capped at 9.0); the cleanup BOOL (false) landed in the
  audio_source slot (string enum); the strict_c7 BOOL (true) landed in
  the humo_pillar_width slot (INT min 128, true converts to 1).
- **Fix:** Edited `workflows/otr_scifi_16gb_full.json` node 52
  `widgets_values[]` in place. Old order
  `[..., 832, 512, 7.0, "ffmpeg", false, "master_mix_per_clip_mux", true]`
  rotated to the new order
  `[..., 832, 7.0, "ffmpeg", false, "master_mix_per_clip_mux", true, 512]`.
  No code change required -- BUG-113's INPUT_TYPES move is correct;
  only the saved workflow needed to catch up.
- **Verify:** `python -c "import json; n = [x for x in json.load(open('workflows/otr_scifi_16gb_full.json'))['nodes'] if x['type']=='OTR_VideoComposite'][0]; print(n['widgets_values'])"`
  prints `['', '', '', 'lighten', 0.0, 1472, 832, 25, 832, 7.0, 'ffmpeg', False, 'master_mix_per_clip_mux', True, 512]`.
- **Confirmed live (2026-05-07 19:42 run 2):** After the JSON edit was
  saved and Jeffrey re-queued via ComfyUI Desktop, `comfyui_8000.log`
  line 1910 shows
  `[VideoComposite] audio_source=master_mix_per_clip_mux -- C7 byte-perfect
  path ... strict_c7=True` -- VideoComposite validated and executed
  cleanly. Run 2 episode `signal_lost_silent_countdown_20260507_193951`
  completed end-to-end with `composited/<ep>.mp4`, RTXUpscale to
  1920x1080, and PostUpscaleProcgenBlend final all written. No further
  validation errors logged for OTR_VideoComposite.
- **Tags:** workflow-json, widget-drift, video-composite, schema-vs-savefile,
  bug-113-followup, validation-fail
- **Related:** BUG-LOCAL-113 (the schema move that needed this saved-file
  catch-up); BUG-LOCAL-097 (earlier widget-drift class); BUG-LOCAL-129a
  (static-radio fill that depends on VideoComposite executing).
- **Bible candidacy:** yes -- recurring class. The lesson is that any
  INPUT_TYPES re-order requires a parallel sweep of every committed
  workflow JSON's positional widgets_values[]. ComfyUI does not name-map
  saved widgets; it position-maps them. A regression test that loads
  every committed workflow JSON and runs ComfyUI's validate() against
  the current node registry would have caught this at commit time.

### BUG-LOCAL-117e [FIXED]: Allow up to 25s of continuous radio per beat (mega-duration cap raise)

- **Date:** 2026-05-07 PM | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Music cues (opening/closing themes) were getting chunked into 5s/7s
  segments at the synthesis step (`scene_sequencer.py` mirror block, local
  `_HUMO_MAX_CLIP_DUR_S = 7.0`), then those chunks fed BatchLTXRender's
  per-line render with the same 7.0s `clip_length` widget default. Result:
  a 10s music_opening cue was rendered as two 5s LTX clips ffmpeg-concat'd
  into one mp4. Visible chunk-boundary snap because the radio scene
  reset between halves.
- **Cause:** Two different constants were both pinned at 7s for
  historical/HuMo-symmetry reasons. Post-BUG-129b music routes to LTX
  (not HuMo), and the 2026-05-07 PM mega-duration empirical test proved
  LTX 2.3 22B-dev BF16 + distilled LoRA renders a 25s @ 832x480 clip
  cleanly on RTX 5080 16 GB + 64 GB RAM with no temporal collapse and no
  identity drift (i2v anchor + radio mechanical-scene content). The 7s
  cap was a pre-117e safety value, not a load-bearing limit.
- **Fix:**
  - `nodes/scene_sequencer.py`: rename `_HUMO_MAX_CLIP_DUR_S` -> `_MUSIC_MAX_CHUNK_DUR_S`,
    bump from 7.0 to 22.0. Music cues up to 22s now flow as a single
    continuous radio scene instead of being chopped.
  - `nodes/batch_ltx_render.py`: `clip_length` widget default 7.0 -> 22.0;
    `LTX_MAX_FRAMES` already at 705 (BUG-117c earlier today).
    `execute()` signature default also bumped.
  - `workflows/otr_scifi_16gb_full.json`: explicit `widgets_values` entry
    for clip_length=22.0 (was falling through to default).
  - `workflows/ltx_2_3_downstream_smoke.json`: clip_length 7.0 -> 22.0.
  - `tests/test_batch_ltx_render.py`: pinned new constants
    (`LTX_MAX_FRAMES==705`, `clip_length default==22.0`, `max==28.16`).
- **Verify:** `pytest tests/test_batch_ltx_render.py -v` passes the new
  constant pins. End-to-end: queue `otr_scifi_16gb_full.json`, watch the
  banner log line `clip_length=22.0`, confirm cargo_hold opening_theme
  runs as a single LTX render rather than two concatenated chunks.
- **Tags:** ltx, chunking, audio-timing, bug-117 family
- **Related:** BUG-LOCAL-091 (chunking dispatch -- still fires, just at
  the new 22s cap), BUG-LOCAL-117d (boomerang -- pairs with this so the
  actual sample window is 11s).

---

### BUG-LOCAL-117d [FIXED]: ffmpeg boomerang post-process for seamless radio loops (default ON)

- **Date:** 2026-05-07 PM | **Phase:** 5 | **Bible candidate:** yes
- **Symptom:** Multi-chunk LTX renders showed a visible snap at chunk
  boundaries because each chunk started at the radio_bookend (i2v anchor
  at frame 0) but ended wherever the model wandered. concat-demuxer
  stitched them with a hard cut at the transition. BUG-LOCAL-117c tried
  end-anchor pinning via `LTXVImgToVideoInplaceKJ` at strength 0.4-0.7 to
  force the loop closed; result at 0.7 was glitching middle frames
  (model couldn't reconcile two strong constraints in 8 distilled
  steps, same architectural ceiling as BUG-LOCAL-032 on the older 2B
  model).
- **Cause:** Stacked anchors inside the diffusion graph hit a
  capacity ceiling. Bypass the in-graph constraint entirely by doing
  the loop closure as a post-process.
- **Fix:** Render HALF the chunk's audio-target duration, then ffmpeg
  reverse-and-concat to double back to full duration. Output starts at
  radio_bookend (frame 0), reaches peak motion at the midpoint, then
  plays the same frames backwards to land back at radio_bookend.
  end-of-clip-N == start-of-clip-N+1 == radio_bookend -> seamless concat
  in VideoComposite. No GPU cost (ffmpeg is CPU-bound, dwarfed by sample
  wall time). Free side benefit: sample wall time HALVES because we
  render half-duration.
  - New helper `_make_boomerang_via_ffmpeg(mp4_path, ffmpeg='ffmpeg')`
    using filter graph `[0:v]split[a][b];[b]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];[a][r]concat=n=2:v=1:a=0[out]`.
    The `trim=start_frame=1` step drops the duplicate midpoint frame so
    the loop reads as smooth oscillation rather than one-frame freeze.
  - Env var `OTR_LTX_LOOP_VIA_REVERSE` defaults to `"on"`. Set to `off`
    for legacy full-duration single-pass renders (e.g. for A/B compare).
  - Cache key extended with `"boomerang"|"linear"` flag so toggling the
    env var doesn't serve stale cached renders.
  - Per-clip ledger record stamps `ltx_loop_via_reverse: bool` for audit.
- **Verify:** banner log shows `boomerang: ON ...`. After execute, each
  rendered chunk's `ltx_render_ms` is roughly half of the pre-117d
  baseline at the same audio target. Visual: chunk boundaries should
  be invisible because both ends are radio_bookend.
- **Tests:** `tests/test_batch_ltx_render.py` adds 5 boomerang tests:
  default-on pin, truthy-set pin, helper-exists, missing-input-raises,
  filter-graph source pin.
- **Tags:** ltx, ffmpeg, post-process, loop-closure, bug-117 family
- **Related:** BUG-LOCAL-117c (in-graph end-anchor approach, abandoned --
  this is the replacement), BUG-LOCAL-032 (architectural ceiling for
  stacked anchors on the older 2B model), BUG-LOCAL-117e (raised cap
  pairs with boomerang so render is 11s for a 22s audio target).

---

### BUG-LOCAL-117a [FIXED]: LTX 2.3 + RES4LYF integration into BatchLTXRender + workflow JSON cutover (default = v0_9 + euler_cfg_pp per A/B verdict)

- **Date:** 2026-05-06 LATE NIGHT | **Phase:** post-BUG-117 production cutover | **Bible candidate:** YES (dual-engine env-var pattern, MultimodalGuider DiT requirement)
- **Symptom:** post-BUG-117 the production episode pipeline (`otr_scifi_16gb_full.json` + `nodes/batch_ltx_render.py`) was still wired to LTX 2B v0.9 + euler + CFGGuider chain that produced static frames. Smoke proved LTX 2.3 + ClownSampler_Beta + MultimodalGuider produces "perfect subtle zoom in" smooth motion; need to plumb that into OTR production while preserving v0.9 as rollback.
- **Cause / decisions (round-robin transcripts at `docs/2026-05-06-bug-117-ltx23-res4lyf-migration__*`):**
  - Round-robin verdict: ChatGPT + Gemini + NVIDIA all agreed on env-var engine selector + explicit float32 sigma cast + Gemma encoder requirement + per-line GC.
  - Critical correction (Gemini + NVIDIA against ChatGPT): `MultimodalGuider` is structurally required for LTX 2.3 DiT, not a video-only optional. Substituting `CFGGuider` would crash with tensor shape mismatch. Original Claude lean was wrong here.
  - Workflow JSON count: external models recommended two parallel JSONs for rollback; Jeffrey's standing rule (`feedback_minimum_json_files.md`) wins -- single JSON, git tag pre-cutover commit, rollback is `git checkout <tag>`.
- **Fix:**
  - **`workflows/otr_scifi_16gb_full.json` cutover:**
    - Node #54 widget: `ltx-video-2b-v0.9.safetensors` -> `ltx-2.3-22b-dev.safetensors`.
    - Inserted node #60 (`LoraLoaderModelOnly`, ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors @ strength 0.5) and node #61 (same LoRA @ 0.2). Re-routed link 87 (was #54.MODEL -> #55.model) through #60 then #61, with new links 102 (#60->#61) and 103 (#61->#55).
    - Node #57 type swap: `CLIPLoader` -> `LTXAVTextEncoderLoader`. Widgets `[t5xxl_fp16.safetensors, ltxv, default]` -> `[gemma_3_12B_it_fp4_mixed.safetensors, ltx-2.3-22b-dev.safetensors, default]`. Output type stays `CLIP` so node #55 input link 94 unchanged.
  - **`nodes/batch_ltx_render.py` dual-engine refactor (~194 LOC added):**
    - `OTR_LTX_ENGINE` env var (default `v2_3`, valid `v0_9` for emergency rollback). Loud startup banner logs engine + model + encoder + sampler + guider + decode + sigma constants.
    - Fail-fast dep check on engine=v2_3 against `["ClownSampler_Beta", "MultimodalGuider", "GuiderParameters", "LTXVTiledVAEDecode"]`. Refuses to start if any missing; clear error message points at RES4LYF install.
    - Engine-branch inside the per-chunk render loop:
      - **v0_9 path:** legacy `CFGGuider(cfg=1.0)` + `KSamplerSelect("euler")` (pre-baked, shared) + `SamplerCustomAdvanced` + `VAEDecodeTiled`. Verbatim from pre-cutover state for byte-identity.
      - **v2_3 path:** `GuiderParameters(modality="VIDEO", cfg=3.0, stg=1.0, perturb_attn=True, rescale=0.9, modality_scale=3.0, skip_step=0, cross_attn=True)` (mirrors stock workflow VIDEO modality widgets exactly) + `MultimodalGuider(model, positive, negative, parameters, skip_blocks="28")` + `ClownSampler_Beta(eta=0.25, sampler_name="exponential/res_2s", seed=shot_seed, bongmath=True)` + `SamplerCustomAdvanced` + `LTXVTiledVAEDecode` (LTX-specific decoder, same OTR-Goofer-proven tile params).
    - Per-chunk GC: `del frames; del samples_out; del _denoised; del latent_chunk; del empty_latent; del noise; del guider; (v2_3-extras del); gc.collect(); torch.cuda.empty_cache()`. Round-robin (Gemini) flagged Python lazy GC would OOM at chunk 3+ on a 14.5 GB peak / 16 GB ceiling without aggressive cleanup.
    - Sigmas: existing `LTX_DISTILLED_SIGMAS` constant unchanged. Confirmed via stock workflow inspection that the 9-value distilled schedule is identical between v0.9 and 2.3 distilled. `torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)` cast already explicit (Gemini caught: ComfyUI overrides default dtype, mandatory not optional).
  - **`tests/test_core.py`:** added `LTXAVTextEncoderLoader` to the workflow node-type whitelist (test_node_types_otr_or_known). RES4LYF nodes called via `_call()` from Python don't appear in JSON node list so they don't need whitelist entries.
- **Verify (this commit):**
  - AST parse of `nodes/batch_ltx_render.py` clean (1469 LOC, +194 from pre-cutover 1275).
  - JSON parse of `workflows/otr_scifi_16gb_full.json` clean (34 nodes, 59 links, last_node_id=61, last_link_id=103).
  - Test suite: 185 passed, 2 skipped, 2 xfailed across `tests/test_core.py` + `tests/test_dropdown_guardrails.py` + `tests/test_audio_byte_identical.py` + `bug_bible_regression.py`.
  - **Real-world regression: PENDING.** See `docs/2026-05-06-handoff-bug-117a-regression.md` for the morning-after sirens_print test plan, expected wall time, what to watch in console + Task Manager, and rollback procedure. NOT marked [FIXED] until that runs green.
- **Tags:** ltx-2.3, res4lyf, multimodal-guider-required, env-var-engine-selector, dual-engine, dit-conditioning, round-robin-verified, bug-117-followup, c7-audio-untouched
- **A/B verdict 2026-05-07 LATE MORNING (cargo_hold smoke, 5 chunks, identical anchor):**
  - v2_3 (RES4LYF res_2s + MultimodalGuider + AV-stub): 36.85 min total, 6-8 min/clip, 64 GB RAM at 100% (swap thrashing)
  - v0_9 (euler_cfg_pp + CFGGuider): 8.3 min total, 1.5-2.5 min/clip, comfortable on 32 GB RAM
  - Jeffrey eyes-on after watching both: "no difference in the latest vids"
  - Verdict: v0_9 + euler_cfg_pp is visually equivalent to v2_3 + res_2s on OTR's mechanical-radio content. The community's res_2s warnings (over-cooked detail, slow-motion) were on portrait/skin content at 1080p+ which doesn't apply.
  - DEFAULT FLIPPED: `OTR_LTX_ENGINE_DEFAULT = "v0_9"` (was "v2_3"). v2_3 retained as opt-in for users with 64 GB+ RAM who want the marginal extra detail.
- **Render cache (BUG-LOCAL-117b):** identical-input chunks (e.g. music_open_001 + music_open_002 from same beat split into 2 chunks) now copy-from-canonical instead of re-rendering. Cargo_hold smoke: 5 chunks -> 3 unique renders, ~40% wall-time saving stacks on top of the engine choice.
- **Provenance:** Jeffrey: "code it all im sleeping" 2026-05-06 ~22:30. Round-robin transcripts under `docs/2026-05-06-bug-117-ltx23-res4lyf-migration__*`. Synthesis writeup at `__04_synthesis.md`. Pre-cutover git tag: `pre-bug-117a-cutover`. A/B verdict 2026-05-07 11:35 with eyes-on smoke comparison.
- **Related:**
  - BUG-LOCAL-117 (the model class diagnosis -- this is the integration that ships the fix into OTR production).
  - BUG-LOCAL-097 (widget drift -- env-var engine selector specifically chosen to avoid widget-array reordering).
  - BUG-LOCAL-008 (CFG=1.0 erases negative -- v2_3 path uses cfg=3.0 in GuiderParameters so negative IS active again, but only at the multimodal guider level not via stock CFGGuider).

---

### BUG-LOCAL-117 [FIXED]: LTX 2B v0.9 cannot produce motion regardless of prompt rewrites -- BUG-LOCAL-112 root cause was model-class limitation, not prompt design
- **Date:** 2026-05-06 EVENING | **Phase:** BUG-112 followup / LTX 2.3 migration | **Bible candidate:** YES (when prompt fixes hit a wall, suspect the model)
- **Symptom:** After BUG-LOCAL-112's ~110 LOC prompt rewrite shipped (motion-centric short prompts, brief/scene_env/style suppressed, role templates dropped to ~180 chars), strength sweeps at i2v_strength [0.75, 0.50, 0.25, 0.05] all still produced effectively static clips on `ltx-video-2b-v0.9.safetensors`. The 0.25 sweep showed slight motion; everything else froze. Quantitative MAD analysis confirmed.
- **Cause:** LTX 2B v0.9 (the 8.73 GB checkpoint we had) has insufficient model capacity for motion synthesis when forced to follow an i2v anchor at production strength. The static-image failure mode is the model's training-distribution-out-of-bounds collapse, not a prompt or sampler problem. Fixing it required upgrading the model class entirely.
- **Fix:** Migrated to `Lightricks/LTX-2.3` 22B-dev BF16 fused checkpoint (42.98 GB) + `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7.08 GB) at `loras/ltxv/ltx2/`. Stock workflow: `custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`. Required RES4LYF custom node pack from `https://github.com/ClownsharkBatwing/RES4LYF` for `ClownSampler_Beta` + `MultimodalGuider` + `GuiderParameters` + `ManualSigmas`. Sampler config `rk_type: res_2s` (Runge-Kutta 2nd-order Singlestep, exponential integrator).
- **Verify (2026-05-06 EVENING):**
  - T2V smoke: stock workflow with default tea-ceremony prompt produced visible motion at 768x432x41f x 15 steps. Wall time ~12.5 min on RTX 5080 16 GB.
  - I2V smoke: same workflow with `bypass_i2v=false` + `radio_bookend.png` loaded into `LoadImage` produced "perfect subtle zoom in" on the radio still. First frame matched the anchor; subsequent 40 frames showed coherent camera motion. Jeffrey verbatim confirmation.
- **Disk economics:**
  - LTX 2.3 dev fused 46 GB BF16 (Path B chosen over Kijai split FP8 23 GB which required workflow surgery to use UNETLoader + separate VAE/text-encoder loaders).
  - Schnell + SD 3.5 Large + HunyuanWorld-Mirror cache + Kijai partial XET blob recycled simultaneously: 42 GB recovered, net cost of upgrade ~4 GB.
- **Tags:** ltx-2.3, model-capacity-not-prompt-design, res4lyf, clown-sampler, runge-kutta-2s, i2v-anchor-works, bug-112-resolution
- **Provenance:** Path B disk-pull approved by Jeffrey after Path A (Kijai split FP8) revealed it would require restructuring the official workflow to swap CheckpointLoaderSimple to UNETLoader + add separate LTXVAudioVAELoader/VAELoader chains. Path B traded 46 GB download for zero workflow surgery.
- **Related:**
  - BUG-LOCAL-112 (the motion problem the prompt rewrite tried to fix; root cause was model capacity, not prompt; rewrite still useful as defensive coding for short prompts but not load-bearing).
  - BUG-LOCAL-008 (CFG=1.0 erases negative -- still applies on 2.3, prompts must remain positive-only).
  - BUG-LOCAL-095 (LTXVImgToVideoConditionOnly i2v anchor pattern -- same node, now actually produces motion against it).
  - Future: cut over `nodes/batch_ltx_render.py` + `workflows/otr_scifi_16gb_full.json` from 2B v0.9 to 2.3 dev so production episodes get motion. Until then keep 2B v0.9 file on disk.

---

### BUG-LOCAL-114 [FIXED]: ltx_motion_batch workflow JSON rejected by ComfyUI Zod validator -- "id" field was a slug, not a UUID
- **Date:** 2026-05-06 LATE MORNING | **Phase:** BUG-112 verification harness | **Bible candidate:** YES (workflow-JSON authoring rule)
- **Symptom:** Loading `workflows/ltx_motion_batch.json` in ComfyUI's UI rendered the graph (60% loaded) but blocked submission with: `Invalid workflow against zod schema: Validation error: Invalid uuid at "id"`.
- **Cause:** `scripts/build_ltx_motion_workflow.py::WorkflowBuilder.to_json` set `"id": "ltx-motion-batch-2026-05-06"` -- a human-readable slug, not a UUID. ComfyUI's frontend uses Zod schema validation that requires the workflow `id` to match the standard UUIDv4 format (8-4-4-4-12 hex). The existing `workflows/otr_ltx_smoke.json` uses a valid UUID (`a4d1c7e2-5b6f-4f3a-9e0c-7b3a5d8e2f10`); I missed that constraint when authoring the new workflow.
- **Fix (`scripts/build_ltx_motion_workflow.py`, ~5 LOC):** pinned a valid UUIDv4 at the builder's `id` field: `1e7ec912-1b40-4c0d-9a5b-6c0d5e7a9b3e`. Diff-friendly across re-builds. Comment cites BUG-LOCAL-114 so the next person who copies this builder for a new workflow doesn't re-introduce the slug mistake.
- **Verify:**
  - Regex match against UUIDv4 pattern: PASS.
  - Workflow JSON file size + node/link counts unchanged.
  - Workflow loads in ComfyUI without the Zod error.
- **Tags:** comfyui-frontend, zod-validation, uuid, workflow-authoring, bug-112-followup
- **Provenance:** discovered when Jeffrey loaded the freshly-built `ltx_motion_batch.json` and the UI surfaced the Zod error in an Alert. Graph rendered fine; submit button was the choke point.
- **Related:**
  - BUG-LOCAL-012 (FIXED): "ComfyUI frontend Zod validation rejected workflows/otr_ltx_smoke.json at load time" -- same Zod constraint family, different schema field.
- **Authoring rule for future workflow JSONs:** any `"id"` field that ComfyUI's UI parses MUST be a valid UUIDv4. If hand-authoring, copy a UUID from an existing-working workflow JSON. If generating, use `uuid.uuid4()` or pin a known-good UUID literal in the builder.

---

### BUG-LOCAL-113 [FIXED]: OTR_VideoComposite saved workflows broke at queue time -- humo_pillar_width inserted in middle of optional dict shifted all subsequent positional values
- **Date:** 2026-05-06 LATE MORNING | **Phase:** post-BUG-112 ship, attempting to load canonical workflow | **Bible candidate:** YES (re-application of BUG-097 lesson)
- **Symptom:** Loading any OTR workflow JSON saved before 2026-05-03 EVENING failed validation at queue time with the following stack:
  ```
  Failed to convert an input value to a FLOAT value
    fallback_clip_length, ffmpeg, could not convert string to float: 'ffmpeg'
    Value 7 smaller than min of 128
    humo_pillar_width, Value not in list
    audio_source: 'True' not in ['master_mix_per_clip_mux', 'humo_concat', 'master_mix']
  ```
- **Cause:** `nodes/video_composite.py::OTR_VideoComposite.INPUT_TYPES` had `humo_pillar_width` (INT, min=128) INSERTED at slot 7, between `humo_target_height` (slot 6) and `fallback_clip_length` (slot 7-old) on 2026-05-03 EVENING per BUG-LOCAL-030 layered-composite spec. Saved workflow JSONs store widget values POSITIONALLY in `widgets_values[]`. After the insertion, every saved positional value at slot >=7 shifted down by one slot:
  - saved `fallback_clip_length=7.0` (was slot 7) -> now slot 7 = `humo_pillar_width` (INT min 128) -> "Value 7 smaller than min of 128"
  - saved `ffmpeg="ffmpeg"` (was slot 8) -> now slot 8 = `fallback_clip_length` (FLOAT) -> "could not convert 'ffmpeg' to FLOAT"
  - saved `cleanup_clips_after_assembly=True` (was slot 10) -> now slot 10 = `audio_source` (enum) -> "'True' not in [...]"
  This is the SAME class of bug as BUG-LOCAL-097 (LTX widget order). The standing rule is: NEVER insert new optional widgets in the middle of an INPUT_TYPES dict; always APPEND to the end so saved workflows backfill defaults cleanly.
- **Fix (`nodes/video_composite.py`, ~30 LOC of comment + relocation):**
  - Moved `humo_pillar_width` from its mid-dict position (between `humo_target_height` and `fallback_clip_length`) to the END of the optional dict (after `strict_c7`).
  - Kept the docstring + behavior unchanged. Default still 512, min 128, max 1920.
  - Saved workflows that pre-date the original 2026-05-03 EVENING insertion now load cleanly: positions 0-10 match the pre-2026-05-03 layout exactly, and `humo_pillar_width` (now slot 11) backfills to its default 512.
- **Verify:**
  - AST parse clean.
  - INPUT_TYPES check: `optional` keys end with `humo_pillar_width` at position 11.
  - Function signature `_execute_body` already accepts `humo_pillar_width: int = 512` -- positional change has no effect on the kwarg API.
  - Manual reproduction of the failure mode: loading the canonical workflow that produced the four-error stack should now load cleanly.
- **Tags:** widget-position-drift, comfyui-workflow-validation, append-not-insert, bug-097-followup, video-composite, jeffrey-canonical-workflow
- **Provenance:** discovered when Jeffrey loaded the canonical workflow after restarting ComfyUI to pick up BUG-LOCAL-112 prompt rewrite, ran into the four validation errors at queue time. Errors confirmed widget-shift pattern by computing offsets between saved positional layout and current INPUT_TYPES order.
- **Related:**
  - BUG-LOCAL-097 (LTX widget order -- clip_length moved to last; same lesson, same shape).
  - BUG-LOCAL-104 (green_only_overlay BOOLEAN widget inserted mid-dict -- same trap).
  - BUG-LOCAL-002 (soak_operator widget indices stale -- positional drift made downstream test harness write to wrong slots).
- **Followup not in this commit:** audit every OTR node's INPUT_TYPES in this repo for widgets inserted mid-dict between Jan 2026 and now. Move any mid-dict additions to the end. Standing rule -- "always append, never insert" -- is in CLAUDE.md but multiple drift incidents in 2026 suggest the rule is being broken silently. A unit test that locks the optional dict order would catch this at CI time instead of at user-load time.

---

### BUG-LOCAL-112 [FIXED]: LTX clips rendering mostly-static -- prompt dilution + OOD clip length + i2v anchor produced "beautiful still images" instead of moving radio shots
- **Date:** 2026-05-06 MORNING | **Phase:** post-BUG-110 visual QA | **Bible candidate:** YES (LTX prompt design / motion-vs-aesthetic separation)
- **Symptom:** quantitative motion analysis (mean absolute pixel difference between consecutive frames within a single LTX clip) on the 2026-05-06 morning soak runs showed effectively static output:
  ```
  stellar_echoes l001 (5.16s):  35.95 -> 3.32 -> 3.21 -> 3.38   (one scene-cut spike, then static)
  stellar_divide l001 (6.76s):   2.36 -> 15.53 -> 9.80 -> 32.78 (only "good" one - noise-seed luck)
  deserted_space_habitat_spinning l001 (2.28s): 2.04 -> 4.31 -> 5.86 -> 5.92 (subtle/static)
  cramped_spaceship_cockpit_humming l001 (5.16s): 1.86 -> 2.05 -> 7.01 -> 32.29 (STATIC + scene cut)
  ```
  Real motion (a dolly forward, a dial sweep) produces sustained 15-30 MAD. Three of four clips were essentially still images. Jeffrey: "ltx is not showing any radio movement or action."
- **Cause:** Round-robin (ChatGPT gpt-5.5 + Gemini gemini-3.1-pro-preview-customtools + NVIDIA llama-3.3-nemotron-super-49b-v1.5; transcripts at `docs/2026-05-06-bug-112-ltx-static-motion__*.md`) all converged on H1 (prompt dilution) being dominant.
  - The prior `_PROMPT_BY_ROLE` templates were ~450 chars of static set-dressing language ("obsidian console", "purple lighting", "vintage 1940s broadcast set", "35mm film grain") with motion verbs buried in the middle and "no people / unattended equipment" negation language at the end.
  - BUG-LOCAL-110's `_build_ltx_role_prompt` further prepended the per-episode `ltx_style_brief` (200-300 chars) plus appended `scene_env` and `style` -- pushing the final LTX prompt to 600-800 chars.
  - T5-XXL (LTX's text encoder) handles 512 tokens, ~2000 chars -- so truncation is NOT the issue. ATTENTION DILUTION is. The model interpreted the prompt as "render a beautiful static set" rather than "make a continuous moving shot."
  - Gemini's load-bearing insight: LTX-Video v0.9 was natively trained on **121 frames (4.84s @ 25fps)**. Our typical announcer beat is 5-7s = 125-169 frames -- OOD on length. The classic OOD failure mode for diffusion video is to FREEZE INTO A STATIC IMAGE to prevent temporal collapse. Diluted prompt + OOD length = perfect recipe for static output.
  - "no people in frame" / "unattended equipment" in POSITIVE prompt was a separate hazard: T5 doesn't reliably negate; the token "people" still activates people concepts. CFG=1.0 makes the negative branch mathematically inert (BUG-LOCAL-008) so positive negation was the only suppression strategy in play, and it was actively harmful.
- **Fix (`nodes/batch_ltx_render.py`, ~110 LOC of rewrite + comment):**
  - **`_PROMPT_BY_ROLE` rewritten:** all 5 role templates dropped from ~450 chars to 161-188 chars each. Lead with "Continuous shot, same console throughout." (suppresses scene-cut spikes that show as MAD spikes >30). Front-load LOCAL motion verbs (dial sweeps, tubes pulse, grille trembles, dust drifts). End with CAMERA motion (slow dolly forward / pull back / orbit). NO negation language. NO static set-dressing nouns -- those flow through the FLUX bookend init image via i2v, not the LTX prompt.
  - **`_build_ltx_role_prompt` simplified:** returns ONLY the role template verbatim. No more brief prepending, scene_env append, or style tone append. The brief is still stamped to `ledger.meta.ltx_style_brief` by OTR_LLMScriptWriter (BUG-LOCAL-110 Layer 2) -- it's just deferred to BUG-LOCAL-111 (FLUX bookend integration, future commit) where it'll drive visual identity at the i2v anchor stage instead of competing with motion at the LTX stage.
  - `line` and `ledger` arguments retained for API stability (callers pass them) but documented as intentionally unused.
- **Round-robin disagreements (Gemini + NVIDIA right, ChatGPT wrong):**
  - ChatGPT bluffed about T5-XXL token limits ("700 chars approaches the encoder's effective limit"). Gemini corrected: 512-token window, 700 chars is ~150-180 tokens, well under. Issue is attention dilution, not truncation.
  - ChatGPT suggested "81 frames for 4n+1 compatibility." Gemini corrected: LTX uses **8n+1** temporal compression. OTR's existing `ltx_length_for_dur` already produces 8n+1 valid integers (57, 121, 129, 169 -- all 8n+1). Math was already right.
  - ChatGPT recommended dropping `LTX_I2V_STRENGTH` 0.75 -> 0.60. Gemini and NVIDIA both objected: DMM ships 0.75 and gets motion; lowering it would cause visual drift without guaranteed motion gain. Kept 0.75.
- **What we explicitly did NOT change (round-robin scope discipline):**
  - `LTX_CFG = 1.0` -- distilled-sigma path requires it
  - `LTX_I2V_STRENGTH = 0.75` -- DMM uses this and gets motion
  - `LTX_DISTILLED_SIGMAS` -- same as DMM, proven schedule
  - `KSamplerSelect("euler")` -- not chasing euler_ancestral or other samplers
  - `LTX_FPS = 25` -- changing fps changes duration math everywhere
  - `ltx_length_for_dur(dur_s)` -- not capping clip length tonight (would break Rule C7 audio sync without a chunking architecture)
- **Verify:**
  - AST parse clean.
  - `_PROMPT_BY_ROLE` lengths: announcer=179, music_open=188, music_close=161, music_inter=174, sfx=180 (all under 200, all motion-centric).
  - `_build_ltx_role_prompt` returns the role template verbatim; brief / scene_env / style correctly suppressed in unit-spot-check.
  - Test suite: 56 passed, 1 skipped, 2 xfailed (post_upscale_procgen_blend + filename_pattern_audit + meta_paths + bug bible regression) -- no regressions.
  - **Real-world verification pending:** next LTX render needs to show MAD between consecutive frames sustained at 15-30 instead of clustered at 2-6. Sample a fresh announcer clip after the next run.
- **Tags:** ltx-prompt-design, motion-vs-aesthetic, attention-dilution, ood-clip-length, t5-encoder-negation-trap, round-robin-verified, bug-008-followup, bug-095-followup, bug-110-followup
- **Provenance:** discovered when Jeffrey reviewed `cramped_spaceship_cockpit_humming` and reported "ltx is not showing any radio movement or action." Quantitative MAD analysis on 4 episodes confirmed. Round-robin transcripts: `docs/2026-05-06-bug-112-ltx-static-motion__*.md`.
- **Related:**
  - BUG-LOCAL-008 (CFG=1.0 erases negative -- this fix accepts that and works around with positive-only motion language).
  - BUG-LOCAL-095 (LTXVAddGuide -> LTXVImgToVideoConditionOnly -- the i2v anchor is what we're now relying on for visual identity).
  - BUG-LOCAL-032 (drop LTX end-frame anchor -- complementary to the prompt simplification).
  - BUG-LOCAL-110 (the brief generator that's now waiting for BUG-LOCAL-111 to consume it on the FLUX side).
  - BUG-LOCAL-111 (planned: integrate brief into FLUX radio_bookend prompt so the still itself reflects per-episode setting; LTX inherits via i2v).

---

### BUG-LOCAL-110 [FIXED]: episode title not propagating end-to-end -- ledger had no title field, on-disk filename used news-headline slug instead of LLM-resolved title, doubled-underscore slug cosmetic
- **Date:** 2026-05-05 LATE EVENING / 2026-05-06 MIDNIGHT | **Phase:** post-LTX-style-brief sprint | **Bible candidate:** YES (multi-layer canonical-id propagation pattern)
- **Symptom:** Recent ledgers all showed `title: None` despite the script writer correctly computing a `_resolved_title` via the BUG-LOCAL-035 fallback chain (user widget -> LLM "TITLE:" line -> derived from environment -> timestamp). Final on-disk filenames looked like `signal_lost_scientists_connect_time_crystal_to_real__20260505_222015_procgen_blended.mp4` -- built from the news-headline slug rather than the resolved title, with a cosmetic doubled underscore between `_real` and `_20260505`.
- **Cause:** Three-layer breakage.
  1. **Layer 1 -- slug doubled underscore:** `nodes/video_engine.py:1482` strips punctuation (kept: alnum + "_" + " ") then `replace(" ", "_")`. A title like `"Signal Lost - The Crystal!"` strips `-` and `!` to produce `"Signal Lost  The Crystal"` (note the DOUBLE SPACE between "Lost" and "The") which then becomes `"signal_lost__the_crystal"`. Round-robin Gemini caught this; ChatGPT initially attributed it to trailing-underscore source text (incorrect).
  2. **Layer 2 -- ledger schema gap:** `_resolved_title` was prepended to `script_lines[0]` as a `{"type": "title", "value": ...}` token but was NEVER written to `ledger.title` or `ledger.meta.title`. So the ledger appeared title-less to all downstream readers.
  3. **Layer 3 -- video_engine read site:** `nodes/video_engine.py` constructed the on-disk filename from the inbound `episode_title` parameter, which was the user's empty widget value (workflow link carried "" through). It never looked at the ledger for the resolved title, so the filename fell back to whatever slug the workflow produced upstream (the news headline).
- **Fix (round-robin verified -- transcripts at `docs/2026-05-05-bug-110-title-flow__*.md`, ChatGPT gpt-5.5 + Gemini gemini-3.1-pro-preview-customtools + NVIDIA llama-3.3-nemotron-super-49b-v1.5 all signed off on shipping Layer 1+2+3 with hardenings):**
  - **Layer 1 (`nodes/video_engine.py`):** added `import re as _re` at the top. Slug pipeline now: drop punctuation -> collapse whitespace runs FIRST (Gemini's correction) -> underscore-ize -> truncate -> collapse underscore runs and strip ends (catches truncation landing on `_`) -> empty-string fallback to `"untitled"`.
  - **Layer 2 (`nodes/story_orchestrator.py`):** stamps `led.data["title"] = _resolved_title` at the top level of the ledger and `led.data.setdefault("meta", {})["title_source"] = _title_source` for forensics ("user", "llm", "derived", "timestamp_fallback"). Stamped right before the final `led.save()` in the same try-block where `meta.ltx_style_brief` lives. Round-robin all three converged on top-level `title` (not `meta.title`) being correct because title is a first-class identity field alongside `episode_id`, `commit`, `total_episode_dur_s`.
  - **Layer 3 (`nodes/video_engine.py`):** before slug construction, reads `_early_led.data.get("title")` (the ledger singleton already imported earlier in the function for `out_dir` resolution) and prefers it over the inbound `episode_title` parameter when non-empty. Logs a forensic line when the substitution happens. Falls back gracefully on any exception. Comment marks this as a transitional v2.0-alpha bridge; v2.1 cleanup is to wire an explicit title socket (ROADMAP.md L374).
- **Round-robin disagreements (Gemini + NVIDIA right, ChatGPT wrong):**
  - Gemini caught the space-collapse-first slug bug ChatGPT initially attributed to trailing-underscore source text. Adopted Gemini's order: `re.sub(r"\s+", " ", ...)` BEFORE `replace(" ", "_")`.
  - Gemini + NVIDIA flagged ComfyUI graph-caching risk on Layer 3 (singleton might be stale if orchestrator gets cached and skipped). Honest assessment: orchestrator inputs change every run (timestamps, news_seed) so caching almost never hits in practice; failure mode if it does is "filename uses stale title" -- debuggable, non-catastrophic. Shipped simple read with try/except fallback to inbound; no episode_id guard. If we ever see leakage in the wild, promotes to guarded read.
- **Verify:**
  - AST parse clean on both files.
  - Bug Bible regression: 103 passed, 1 skipped, 2 xfailed, 0 failed.
  - Slug logic unit-spot-check (manual REPL):
    - `"The Signal__From   Beyond!!!"` -> `"the_signal_from_beyond"` (collapsed runs, ended cleanly)
    - `"!!!"` -> `"untitled"` (empty-string fallback)
    - `"Signal Lost - The Crystal!"` -> `"signal_lost_the_crystal"` (no double underscore from the double space)
- **What this DOES NOT fix (deferred per round-robin scope discipline):**
  - `episode_id` / per-episode folder naming (still uses news-headline slug for the dir name; only the final mp4 filename uses the resolved title). Folder rename happens via `Ledger.rename_episode(ep_id)` AFTER video_engine writes the mp4, and `ep_id` is derived from the mp4 basename so the folder NOW matches the resolved title too -- partial side-effect win.
  - `"Signal Lost"` hardcoded prefix in the filename (separate v2.1 `show_name`-configurable cleanup; ROADMAP.md L644).
  - LLMDirector hardcoded `"episode_title": "Signal Lost"` in its JSON output (separate prompt-template work).
- **Tags:** title-propagation, slug-cleanup, ledger-schema, transitional-bridge, comfyui-singleton, round-robin-verified, bug-035-followup, bug-038-followup
- **Provenance:** discovered during the in-flight `signal_lost_scientists_connect_time_crystal_to_real__20260505_222015` run; ledger had `title: None` on inspection. Round-robin transcripts and synthesis: `docs/2026-05-05-bug-110-title-flow__*.md`.
- **Related:**
  - BUG-LOCAL-035 (TITLE_STUCK fix that introduced `_resolved_title` fallback chain).
  - BUG-LOCAL-022 (BatchHumoRender stem-swap broken when `safe_title[:40]` truncates).
  - BUG-LOCAL-097 (widget position drift; pattern of "fix the widget but not the consumer").
  - ROADMAP.md L374 (v2.1 socket cleanup), L636-644 (show_name configurable cleanup).

---

### BUG-LOCAL-109 [FIXED]: ScriptCritic silently bypassed every run with cleanup_model_id="auto (use story model)"
- **Date:** 2026-05-05 LATE EVENING | **Phase:** post-LTX-style-brief soak (signal_lost_scientists_connect_time_crystal_to_real_20260505_222015) | **Bible candidate:** YES (sentinel-resolution pattern)
- **Symptom:** Live tail of a Pro-quality run showed:
  ```
  [ScriptCritic] inherited from ledger.cleanup_model_id: model=auto (use story model) ...
  [ScriptCritic] running critic model=auto (use story model) len=3800 timeout=90s
  ...
  Failed to load Tokenizer 'auto'. Is it downloaded? Hub error: auto is not a local folder
  and is not a valid model identifier listed on 'https://huggingface.co/models'
  ...
  [ScriptCritic] critic call failed (...) - SKIPPED
  [ScriptCritic] stamped script_gates[] entry: verdict=PASS score=None
  ```
  Run continued (non-fatal) but the critique gate was silently bypassed. Every "auto (use story model)" run -- which is the default widget value -- was running WITHOUT the structural critique pass that catches missing ANNOUNCER closings, BUG-027 dialogue-erased revisions, and similar gate failures. The verdict=PASS score=None ledger entry made it look like the gate ran cleanly when in fact it was skipped entirely.
- **Cause:** Two-layer bug.
  1. `nodes/script_critic.py` line 765 checked `cm.lower() not in ("auto", "(auto)", "")` -- only matches the BARE strings "auto" / "(auto)" / "". The widget's canonical sentinel is "auto (use story model)" (with parenthesized hint, matching the dropdown label). That's not in the tuple, so the check thought it was a real model ID. Then line 815 stripped on space -> "auto" -> `AutoTokenizer.from_pretrained("auto")` -> HF 404 -> exception caught -> silent skip.
  2. `nodes/story_orchestrator.py::_load_llm` had no defensive guard against receiving a literal "auto*" model_id. If anything else ever bypassed the resolver (stale workflow JSON, broken caller, future regression), the same buried-404 trace would fire with no actionable error.
- **Fix:**
  - **`nodes/script_critic.py` (~10 LOC):** replaced the equality-against-tuple check with `str(cm).strip().lower().startswith("auto")` -- mirrors the resolver in `story_orchestrator.py` line 4839. Now ANY value starting with "auto" (case-insensitive, after strip) is treated as the sentinel and falls back to ledger.model_id (the writer's main model). Includes the empty-string + None case in one expression.
  - **`nodes/story_orchestrator.py::_load_llm` (~12 LOC defensive guard):** added an explicit check at the top of the function body that raises `RuntimeError` if `model_id` is empty OR starts with "auto". Error message names BUG-LOCAL-109 and tells the caller exactly how to fix it (resolve the sentinel before calling `_load_llm`). Belt-and-suspenders against future drift.
- **Verify:**
  - AST parse clean on both files.
  - Bug Bible regression: 103 passed, 1 skipped, 2 xfailed, 0 failed.
  - On the next run with `cleanup_model_id="auto (use story model)"`, ScriptCritic should log `inherited from ledger.model_id (cleanup was auto): model=mistralai/Mistral-Nemo-Instruct-2407` instead of `model=auto (use story model)`, then proceed to a real critic call.
- **Tags:** sentinel-resolution, scriptcritic, silent-skip, gate-bypass, defensive-guard, bug-009-adjacent
- **Provenance:** discovered via Jeffrey-pasted live tail 2026-05-05 LATE EVENING. The "auto" sentinel widget label was added in the BUG-068 follow-up two-LLM split (2026-04-26) and the script_critic resolver was updated to look for it but only for the BARE "auto" form -- the widget label drifted to include "(use story model)" hint without updating the resolver tuple.
- **Related:**
  - BUG-LOCAL-068 (two-LLM split that introduced the cleanup_model_id widget).
  - BUG-LOCAL-027 (critique pass dialogue wipe -- this gate was supposed to catch that family).
  - BUG-LOCAL-097 (widget position drift caused by inserting widgets without updating downstream consumers).

---

### BUG-LOCAL-108 [FIXED]: obs/ broadcast folder held two mp4s per episode (pre-blend upscale + post-blend final) -- enforce "one final mp4 per episode" contract
- **Date:** 2026-05-05 MORNING | **Phase:** post-BUG-106 dark_transponder QA | **Bible candidate:** YES (architectural rule for output paths)
- **Symptom:** after the dark_transponder soak run completed cleanly, `output/otr/obs/` contained both `signal_lost_dark_transponder_20260505_084250.mp4` (9 MB pre-blend RTXUpscale output) and `signal_lost_dark_transponder_20260505_084250_procgen_blended.mp4` (18.7 MB post-blend final). Same shape in earlier runs (crystalline_whispers had 2 files, echo_in_stasis had 1 source + 8 BLEND_TEST variants). Jeffrey: "obs should only have one final output file its my broadcast folder so any pre upscaled files should be in the v episode specific video folder." Path module's `otr_obs_dir()` docstring already documented the "exactly one mp4 per episode" contract; the code was breaking it because both `OTR_RTXUpscale` and `OTR_PostUpscaleProcgenBlend` wrote into obs/.
- **Cause:** `nodes/rtx_upscale.py:598` did `out_dir = otr_obs_dir()` and wrote `<ep>.mp4` there. `nodes/otr_post_upscale_procgen_blend.py:482` did `output_path = src.with_name(...)` -- "next to source" -- which because the source was in obs/ landed the blended output in obs/ too. Pre-procgen-blend (BUG-099 era and earlier) the chain ended at RTXUpscale, so writing the upscaled to obs/ was correct then. After PostUpscaleProcgenBlend joined the chain the contract drifted without anyone noticing.
- **Fix:**
  - `nodes/_otr_paths.py`: added `otr_upscaled_dir(episode_id)` helper -> `<output>/otr/episodes/<ep>/upscaled/`. Updated `otr_obs_dir` and `otr_composited_dir` docstrings to reflect the new chain. Added `otr_composited_dir`, `otr_upscaled_dir`, `otr_obs_dir`, `otr_state_dir` to `__all__` (they were missing).
  - `nodes/rtx_upscale.py`: `out_dir = otr_upscaled_dir(src.stem)` instead of `otr_obs_dir()`. Updated bypass-branch comment + report-line text + module docstring.
  - `nodes/otr_post_upscale_procgen_blend.py`: added `from _otr_paths import otr_obs_dir`; `output_path = otr_obs_dir() / f"{src.stem}{out_suffix}{src.suffix}"` with explicit `mkdir(parents=True, exist_ok=True)` so the blended file always lands in obs/ regardless of where the source lived.
  - `nodes/rtx_upscale.py` spacesaver guard: updated obs-existence path from `obs/<ep>.mp4` to `obs/<ep>_procgen_blended.mp4` to match the new final filename. Spacesaver call site is still in RTXUpscale so the guard now always fail-closes (procgen blend is downstream); spacesaver becomes a silent no-op until the call site is moved to PostUpscaleProcgenBlend (planned followup, not in this commit). Added `upscaled` to the wipe-list of subdirs for when the move happens. Spacesaver default is OFF in story_orchestrator so this regression has no production impact.
- **Verify:**
  - AST parse clean on all 5 touched files.
  - test_post_upscale_procgen_blend.py: 17 passed (3.16s). Added `_isolate_obs_dir` autouse fixture that pins `OTR_OUTPUT_DIR=tmp_path` so blend tests don't write into Jeffrey's real output tree.
  - test_filename_pattern_audit.py: 3 passed. Updated ALLOWLIST entry from `obs" / f"{ep_id}.mp4"` to `obs" / f"{ep_id}_procgen_blended.mp4"` to match the new spacesaver guard string and refreshed the comment.
  - test_meta_paths.py: 13 passed (no changes; meta.paths.obs_final is owned by `_build_meta_paths` and was not touched in this patch).
  - Bug Bible regression: 69 passed, 1 skipped, 2 xfailed, 1 failed (pre-existing BUG-01.02 for otr_save_copy.py + batch_flux_portrait_render.py without folder_paths -- explicitly listed as known-failure in HANDOFF_NIGHTLY_SOAK_REVIEW.md, not load-bearing for this patch).
- **Net result:** new run lands `output/otr/episodes/<ep>/upscaled/<ep>.mp4` (intermediate) + `output/otr/obs/<ep>_procgen_blended.mp4` (sole broadcast cut). Existing pre-blend mp4s in obs/ are now strays under the new contract -- one-time cleanup script can move them to their per-episode upscaled/ dirs separately.
- **Tags:** path-cleanup, broadcast-folder, output-contract, otr-paths-doctrine, jeffrey-directive
- **Related:**
  - Original "exactly one mp4 per episode in obs/" contract: 2026-05-02 EVENING path reorg (see `docs/2026-05-02-path-reorg-spacesaver-qa__04_synthesis.md`).
  - BUG-LOCAL-099 (procgen blend introduced; first contract drift).
  - BUG-LOCAL-106 (`screen` + `green_only=True` production default; the run that surfaced the contract violation).
- **Followup not in this commit:** move the spacesaver call site from `OTR_RTXUpscale.execute` to `OTR_PostUpscaleProcgenBlend.blend` so the existence guard fires at the correct stage. Also a one-time migration script for existing pre-blend mp4s currently in obs/.

---

### BUG-LOCAL-107: nightly soak `alien_mind` stalled in video pipeline after audio mux -- only 1 of 42 video clips landed in 8 hours, ComfyUI server then died
- **Date:** 2026-05-05 MORNING | **Phase:** BUG-106 nightly soak verification | **Bible candidate:** TBD (need root cause first)
- **Symptom:** Episode `signal_lost_alien_mind_20260504_221326` (commit `a054b80`, 42 lines, 6.5min dialogue, news_seed picked 21:49 May 4) ran the audio pipeline cleanly: audio master mp4 muxed at 22:16 May 4 (329 MB, 48 kHz mono AAC), ledger renamed to `<ep>_ledger.json` at 22:18, all 3 portraits rendered by 22:18. Video pipeline then started -- and over the next 8 hours produced exactly ONE per-line video chunk: `videos/l003__chunk01.mp4` (663,332 bytes, mtime 06:19 May 5). No other clips, no OBS scene mp4 in `output/otr/obs/`, `meta.post_upscale_blend = None` in the ledger. As of 06:30 May 5: ComfyUI HTTP 8000 refuses connections (`connect refused` to both ::1 and 127.0.0.1), `tasklist` shows the prior ComfyUI PID 37920 (~23.5 GB RAM) gone. Final blended mp4 with BUG-106 defaults was therefore never produced for this run.
- **Pre-death tells:** at 06:25 May 5 (just before the server died) `/queue` returned `running: 0, pending: 0` and LibreHardwareMonitor showed RTX 5080 GPU Memory Used = 3.75 GB BUT D3D Shared Memory Used = 17.7 GB -- i.e. ~17 GB of pageable system-RAM-shared spillover from VRAM, classic PCIe-thrash symptom. That's the BUG-LOCAL-050 chained-backend teardown smell, but on a much larger scale than expected.
- **Cause:** unknown pending diagnosis. Three candidates ordered by likelihood:
  1. **HuMo / LTX sidecar didn't release weights between clips** -- per-clip wall time on RTX 5080 is ~10-12 min for HuMo character lines and shorter for LTX, so 42 lines should be 4-7 hours, not 8+ hours with 1 clip. If the sidecar fell into shared-mem spillover after clip 1, throughput would crater by 10-100x.
  2. **Mistral-Nemo (or another LLM) was not unloaded before the video pass** -- 23.5 GB RAM on the dead ComfyUI process suggests model weights stayed resident. Should have been flushed via `_flush_vram_keep_llm()` or full unload before the visual pipeline subprocess launched.
  3. **OOM kill** -- ComfyUI just died; could have been the CUDA driver killing it, or Windows page-fault death after shared-mem ballooned past system RAM.
- **Fix:** pending diagnosis. Don't attempt code changes until the ComfyUI console tail covering 22:18 May 4 -> 06:30 May 5 is read; Jeffrey paste required. The l003__chunk01 mtime of 06:19 May 5 means SOMETHING was producing output as late as ten minutes before death -- so the death itself is recent, not at the start of the video pass.
- **Verify:** pending. Need: (1) ComfyUI console tail; (2) confirm sidecar process logs in `logs/` if they exist; (3) re-run alien_mind workflow from a clean ComfyUI restart and watch VRAM + shared-mem timeline at the visual-pipeline start; (4) if shared-mem balloon repeats, audit the chained-backend `finally:` block teardown for HuMo/LTX (BUG-LOCAL-050 pattern).
- **Tags:** soak-stall, video-pipeline, vram-shared-spillover, comfy-died, BUG-050-related, BUG-106-soak-verification, blend-never-ran
- **Provenance:** discovered during morning triage 2026-05-05 06:30 PT. The companion smoke-render `signal_lost_crystalline_whispers_20260504_214504_procgen_blended.mp4` (commit a054b80, 2 lines, 27.85s dialogue, queued ~21:45 May 4 right after BUG-106 commit at 21:28) DID complete cleanly with `blend_mode='screen'`, `blend_opacity=1.0`, visible green-CRT phosphor text in corners (sampled at 14s: green-dominant pixels = 0.063% / 1,314 px on 1920x1080 frame). So BUG-106 itself is verified working in isolation; this stall is a separate video-pipeline issue.

---

### BUG-LOCAL-106 [FIXED]: ship `screen` + `green_only=True` as production default after BUG-105 A/B confirmed `screen_GREEN_crush18` was visibly correct
- **Date:** 2026-05-04 LATE EVENING | **Phase:** BUG-105 A/B verification | **Bible candidate:** YES (default visual signature for procgen overlay)
- **Symptom:** post BUG-105 fix, A/B test workflow rendered 8 outputs (4 with green_only_overlay=True, 4 without). Jeffrey reviewed and signed off on `signal_lost_echo_in_stasis_20260504_170903_BLEND_TEST_screen_GREEN_crush18.mp4` as "perfect" -- that combo is now the canonical procgen blend signature for v2.0. Default needs to flip from BUG-099's `lighten` + green_only_off to `screen` + green_only_on.
- **Why screen, not lighten or addition:**
  - `lighten = max(A, B)` per channel collapses to white over fully-saturated source pixels. With green_only_overlay zeroing procgen R+B, max((255,0,255), (0,255,0)) = (255,255,255). The wireframe goes white-on-magenta -- looks like glare, not phosphor.
  - `addition = A + B` (clamped) is too aggressive in mid-tones; pushes green to clipping over moderately bright source content, washes out wireframe edges.
  - `screen = A + B - A*B` is multiplicative-additive: preserves source R and B exactly when procgen R and B are 0 (since the `- A*B` term goes to 0 too), and lifts source G by `A_g + B_g - A_g*B_g` -- visible green CRT phosphor that scales with source brightness rather than collapsing or clipping.
- **Fix (`nodes/otr_post_upscale_procgen_blend.py`, ~10 LOC):**
  - `_DEFAULT_BLEND_MODE` flipped from `"lighten"` to `"screen"`.
  - New `_DEFAULT_GREEN_ONLY = True` constant added.
  - `green_only_overlay` widget default flipped from `False` to `_DEFAULT_GREEN_ONLY`.
  - `blend()` method signature: `green_only_overlay: bool = _DEFAULT_GREEN_ONLY`.
- **Companion -- canonical workflow `workflows/otr_scifi_16gb_full.json`:**
  - OTR_PostUpscaleProcgenBlend node id=58 widgets_values padded from 7 entries to 9, with the new defaults baked in:
    ```
    [0] ''                        # source_mp4_path (wired via link)
    [1] ''                        # procgen_mp4_path (wired via link)
    [2] 'screen'                  # blend_mode (BUG-106)
    [3] 1.0                       # blend_opacity
    [4] 'ffmpeg'
    [5] False                     # bypass
    [6] '_procgen_blended'        # out_suffix
    [7] 18                        # shadow_crush_threshold (BUG-103)
    [8] True                      # green_only_overlay (BUG-106)
    ```
  - This means: production runs from this point forward emit the v1.7 SIGNAL LOST phosphor-green CRT signature on top of HuMo+LTX scene renders by default. No widget-tuning needed in ComfyUI; just queue the canonical workflow.
- **Updated test:** `test_default_blend_mode_is_lighten` -> `test_default_blend_mode_is_screen_with_green_only`. Asserts both new defaults so a silent revert is caught at unit-test time.
- **Verify:**
  - AST parse clean.
  - test_post_upscale_procgen_blend -> 17 passed in 3.16s (test_default_blend_mode renamed and now asserts both screen + green_only).
- **Tags:** procgen-blend, default-flip, screen-mode, green-only, v17-signal-lost-restoration, BUG-105-followup, jeffrey-signoff
- **Provenance:** Jeffrey approved the visual on `signal_lost_echo_in_stasis_20260504_170903_BLEND_TEST_screen_GREEN_crush18.mp4` direct quote: "this was one was perfect". Visual sign-off captured here so the design history shows WHICH combo was chosen and on which source.
- **Related:**
  - BUG-LOCAL-099 (previous default `lighten` -- now superseded for the green_only=True path).
  - BUG-LOCAL-105 (the triple-trap that made BUG-104 invisible -- without that fix, BUG-106 couldn't have shipped).
  - BUG-LOCAL-103 (shadow crush at threshold 18 still applies in the chain before colorchannelmixer).
- **Followup:** the canonical workflow change is for `otr_scifi_16gb_full.json`. If other saved workflows exist that wire OTR_PostUpscaleProcgenBlend (e.g. older BUG-099 era variants), they'll continue to work -- the widget defaults backfill on load -- but won't get the new defaults until they're re-saved or the user manually toggles the widgets. Document this in the next workflow audit pass.

---

### BUG-LOCAL-105 [FIXED]: green_only_overlay was firing but invisible -- triple-trap: widget position drift + YUV blend swallow + lighten-collapse-to-white
- **Date:** 2026-05-04 LATE EVENING | **Phase:** BUG-104 verification | **Bible candidate:** YES (each of the three traps is a generalizable lesson)
- **Symptom:** BUG-104 shipped a `green_only_overlay` BOOLEAN widget. Test workflow regenerated with `green_only=True` baked in, queued, 8 outputs landed -- Jeffrey reported "they all look the same." External round-robin consult identified three compounding bugs that together made the fix invisible.
- **Trap 1 -- widget position drift (re-broke BUG-097 rule):**
  - I inserted `green_only_overlay` (BOOLEAN) BETWEEN `out_suffix` (STRING) and `shadow_crush_threshold` (INT) in the optional dict.
  - ComfyUI parses `widgets_values` positionally. A workflow saved BEFORE BUG-104 had position-8 = `shadow_crush_threshold` (INT, default 18). After BUG-104, position-8 became `green_only_overlay` (BOOLEAN). The saved INT 18 was read as `bool(18) = True`.
  - Net: green-only WAS firing on saved workflows -- but `shadow_crush_threshold` then defaulted to its INPUT_TYPES default (also 18) so the visible filter chain was actually correct. The bug here is robustness: had the new widget been inserted ahead of a STRING widget, validation would have crashed. **Fix: append new optional widgets at the END of the dict, never insert in the middle.** Same lesson as BUG-097. Restated explicitly in a comment at the widget definition site.
- **Trap 2 -- ffmpeg blend ran in YUV, not RGB (the actual visible cause):**
  - libx264 wants `yuv420p`. When `ffmpeg -filter_complex` produces output for libx264, ffmpeg auto-converts the filter graph internals to YUV unless explicitly pinned otherwise.
  - The `blend=all_mode=lighten` filter then ran `lighten` per-plane on Y, U, V. Lighten on Y is "pick brighter luma"; lighten on U/V is "pick the higher chroma offset", which is meaningless mathematically. Result: the green wireframe's RGB intent got mangled into Y/U/V splatter that compressed away to nothing visible.
  - **Fix:** explicitly pin both inputs to `format=gbrp` (planar RGB) before the blend, then `format=yuv420p` after. ffmpeg now runs the per-RGB-channel math the colorchannelmixer was designed for. Filter chain when `green_only_overlay=True`:
    ```
    [1:v] scale -> crop -> setpts -> lutrgb (crush) ->
          colorchannelmixer (zero R+B) -> format=gbrp [pgn]
    [0:v] format=gbrp [main]
    [main][pgn] blend=mode:opacity:shortest -> format=yuv420p [v]
    ```
  - Legacy path (`green_only_overlay=False`) is untouched -- no format pin, blend reads `[0:v]` direct -- to preserve byte-identity for any saved workflow that was relying on the old behavior.
- **Trap 3 -- lighten on saturated magenta + green = white (math correction):**
  - I had been recommending `blend_mode=lighten` for the green-only case. That math collapses incorrectly:
    - Source pixel (saturated magenta): `(255, 0, 255)`
    - Procgen-after-mixer (pure green):  `(0, 255, 0)`
    - lighten = max-per-channel:         `(255, 255, 255)` -- pure white, not green!
  - Wherever the source has fully-saturated magenta or warm content, the wireframe goes white. Looks like glare, not phosphor.
  - **Fix:** widget tooltip and BUG_LOG explicitly recommend pairing `green_only_overlay=True` with `blend_mode='screen'` or `'addition'`:
    - screen formula `out = A + B - A*B` produces e.g. `(0.7, 1.0, 0.3)` for warm-source + green = visible green-shifted highlight.
    - addition `out = A + B` (clamped) produces a more aggressive green-bias.
    - Both preserve phosphor hue better than lighten in the saturated-source case. Lighten remains AVAILABLE in the dropdown, just not recommended for this combo.
- **Companion: rebuilt `workflows/blend_test.json` as an A/B grid** -- top 4 nodes (blue-tinted) have `green_only=True` with modes screen/addition/lighten/overlay; bottom 4 (red-tinted) have `green_only=False` with the same 4 modes. Queue once, get 8 outputs with the BUG-105 fix vs without, side-by-side.
- **Verify:**
  - AST parse clean.
  - INPUT_TYPES check: optional keys are `[blend_mode, blend_opacity, ffmpeg, bypass, out_suffix, shadow_crush_threshold, green_only_overlay]` -- green_only_overlay LAST.
  - `_build_blend_cmd(green_only=True, mode=screen)` filter contains `format=gbrp[pgn]`, `[0:v]format=gbrp[main]`, `[main][pgn]blend=...:shortest=1,format=yuv420p[v]`. `colorchannelmixer` zeros R+B. `lutrgb` crush still present.
  - `_build_blend_cmd(green_only=False, mode=lighten)` filter contains NO `format=gbrp`, NO `format=yuv420p`, blend reads `[0:v]` direct -- legacy path preserved.
  - test_post_upscale_procgen_blend -> 17 passed in 3.07s.
- **Tags:** procgen-blend, ffmpeg-yuv-vs-rgb, format-pinning, widget-position-drift, blend-math-correction, BUG-104-verification, round-robin-payoff
- **Round-robin credit:** the three traps were caught by an external second-AI consult (problem statement provided in chat 2026-05-04 LATE EVENING). The YUV trap (#2) was the ACTUAL visible cause; the widget drift (#1) was a robustness issue that happened to land on a non-fatal value in the test case; the lighten/white collapse (#3) was a documentation / recommendation correction. Without the consult, would have shipped the green-only flag with all three traps unfixed.
- **Related:**
  - BUG-LOCAL-097 (widget position drift first lesson; broke saved workflow validation).
  - BUG-LOCAL-104 (green_only_overlay introduction -- was correct code-wise but invisible due to YUV trap).
  - BUG-LOCAL-103 (shadow crush, runs first in the procgen filter chain).
- **Followup not blocking:** if x264 compression eats the 1%-sparse green wireframe even with the format pin, add an optional `green_halation` widget (FLOAT 0-3, default 0) that inserts `boxblur=N:1` after `colorchannelmixer` to thicken the wireframe before encode. Flagged by the round-robin AI as a likely next iteration.

---

### BUG-LOCAL-104 [FIXED]: green_only_overlay -- isolate procgen G channel before blend so v1.7 phosphor-green CRT is visible regardless of source scene color
- **Date:** 2026-05-04 LATE EVENING | **Phase:** BUG-103 followup | **Bible candidate:** YES (channel-isolated overlay pattern)
- **Symptom:** even with BUG-103 shadow crush at 18, the 8-mode test workflow against echo_in_stasis produced thumbnails that all looked magenta/pink. The procgen file was confirmed to be the correct v1.7 green-CRT render (file hash 20DA132B... 50.0 MB, dark max RGB (57, 95, 51) green-dominant, 99% pure black, sparse green wireframe content). The SOURCE file (file hash 50F5E10E... 15.4 MB, OBS upscale of HuMo+LTX master mix) had its own scene content with magenta porthole + warm room. lighten/screen/addition blend modes preserve source where source > procgen, and the sparse green wireframe lost the brightness contest against bright source pixels. Result: green CRT was technically in the output but visually drowned out -- the v1.7 SIGNAL LOST signature was effectively invisible.
- **Cause:** RGB blend modes (lighten, screen, etc.) operate per-channel on RAW R/G/B values from each input. When procgen has bright green wireframe pixels at, say, (5, 200, 5), and source has magenta room lighting at (180, 80, 180), `lighten` produces max-per-channel = (180, 200, 180) -- the green CRT shows as a slight green-tinted highlight in an otherwise still-magenta scene. The G channel won, but R and B from source dominate the visible color. To make green CRT pop, only the G channel should ever contribute from procgen.
- **Fix (`nodes/otr_post_upscale_procgen_blend.py`, ~25 LOC):**
  - New widget: `green_only_overlay` (BOOLEAN, default False).
  - When True, `_build_blend_cmd` inserts `colorchannelmixer=rr=0:rg=0:rb=0:gr=0:gg=1:gb=0:br=0:bg=0:bb=0` AFTER the BUG-103 shadow crush and BEFORE the blend. This zeros procgen R and B (any procgen R or B contribution becomes 0); preserves procgen G as-is. Blend math then becomes:
    - lighten: out_R = max(src_R, 0) = src_R; out_G = max(src_G, pgn_G); out_B = max(src_B, 0) = src_B
    - screen: out_R = src_R + 0 - src_R*0 = src_R; out_G = src_G + pgn_G - src_G*pgn_G; out_B = src_B
    - addition: out_R = src_R; out_G = src_G + pgn_G; out_B = src_B
  - Net effect: source R and B pass through untouched (no color shift on the scene); source G gets boosted exactly where the green CRT wireframe lives. Visible result is pure phosphor-green CRT lines/text/EQ overlaid on whatever scene the source contains, regardless of source color.
  - Companion: rebuilt `workflows/blend_test.json` with `green_only_overlay=True` baked into all 8 modes (suffix: `_BLEND_TEST_<mode>_greenonly_crush18`).
- **Verify:**
  - AST parse clean.
  - INPUT_TYPES check: `optional` keys now include `green_only_overlay` and `shadow_crush_threshold`.
  - `_build_blend_cmd(green_only=True)` -> filter contains `colorchannelmixer=rr=0:rg=0:rb=0:gr=0:gg=1:gb=0:br=0:bg=0:bb=0`.
  - `_build_blend_cmd(green_only=False)` -> filter has NO colorchannelmixer (clean disable path).
  - test_post_upscale_procgen_blend -> 17 passed in 3.49s.
- **Tags:** procgen-blend, channel-isolation, ffmpeg-colorchannelmixer, BUG-103-followup, v17-signal-lost-restoration
- **Related:**
  - BUG-LOCAL-099 (lighten 1.0 default for procgen blend -- still the right default for the BLEND mode itself; BUG-104 is the channel filter applied BEFORE the blend).
  - BUG-LOCAL-103 (shadow crush -- runs first in the procgen filter chain; green_only_overlay runs AFTER the crush).
  - BUG-LOCAL-102 (dropdown expansion -- now any of the 16 modes can be tested with green_only_overlay=True).
- **Followup:** the long-term goal Jeffrey stated is "all the LTX and HuMo will have the green bits overlay" -- green_only_overlay should likely become the v2.0 default once a green-CRT visual is confirmed against the upcoming HuMo+LTX scene-rendered episodes. Default is currently False to avoid silently changing behavior on any saved workflow.

---

### BUG-LOCAL-103 [FIXED]: pre-blend shadow crush -- procgen "black" was actually (5,5,10) blue-tinted near-black, lifting source darks toward magenta/pink in highlights
- **Date:** 2026-05-04 LATE EVENING | **Phase:** BUG-099 root cause | **Bible candidate:** YES (any procgen-overlay pipeline)
- **Symptom:** even after BUG-099 swapped `screen 1.0` -> `lighten 1.0`, residual color cast persisted in highlights of procgen-blended episodes. Jeffrey suspected the procgen "black" background wasn't true black but had alpha or color tint leaking through any brighter-than-source blend mode.
- **Diagnostic measurement (PowerShell + Pillow on `signal_lost_echo_in_stasis_20260504_170903` procgen mp4 @ 0:22):**
  - Total pixels: 2,073,600 (1080p)
  - Dark pixels (luminance < 32): 2,051,840 (99.0% of frame)
  - **Dark mean RGB: (4.86, 4.54, 10.06)** -- B channel is 2x R/G, clear blue cast
  - Dark min RGB: (0, 0, 0)
  - Dark max RGB: (57, 95, 51) -- legit motion content (greens) reaches up to 95
  - **True #000 fraction: 0.014%** -- only 1 in 7000 pixels is actual black
- **Cause:** procgen video models (the OTR_SignalLostVideo generator's source diffusion model) cannot produce true `#000000` because of decoder noise floor. The "near-black" output averages RGB(5,5,10) with a blue cast. Blend formula consequences:
  - `screen`: `out = A + B - A*B` -- B contribution lifts proportionally, B's 2x bias produces magenta/pink in source highlights (the BUG-096 symptom).
  - `lighten`: `out = max(A, B)` per channel -- in source dark regions where source channel < procgen channel, procgen's blue-tinted "black" overrides source. Result: blue-violet bloom in shadows, not the clean radio-CRT signature intended.
  - `addition`/`dodge`: same lift problem, just stronger.
- **Fix (`nodes/otr_post_upscale_procgen_blend.py`, ~25 LOC):**
  - New widget: `shadow_crush_threshold` (INT, default=18, range 0-50). 0 disables.
  - `_build_blend_cmd` inserts a `lutrgb` filter step between procgen scale/crop and the blend itself when `crush > 0`:
    - `lutrgb=r=val*gte(val\,T):g=val*gte(val\,T):b=val*gte(val\,T)`
    - Any RGB channel value below threshold is clamped to 0; values >= threshold pass unchanged.
    - Comma in `gte(val,T)` escaped per ffmpeg filter-arg quoting rules.
  - Default `T=18` covers the (5-10) noise floor with margin while preserving the (95-max) green motion-content flecks. Tunable per-workflow without code edits.
  - Companion: rebuilt `workflows/blend_test.json` with the new widget value baked in (`crush=18`) and 8-mode side-by-side via 2 PrimitiveNode string sources -> 8 OTR_PostUpscaleProcgenBlend (16 wired links). Edit the source/procgen paths once on the two PrimitiveNodes; the 8 mode comparisons inherit them.
- **Verify:**
  - AST parse clean.
  - INPUT_TYPES check: `optional` keys now `[blend_mode, blend_opacity, ffmpeg, bypass, out_suffix, shadow_crush_threshold]`.
  - `_build_blend_cmd(crush=18)` -> filter_complex contains `lutrgb=r=val*gte(val\,18):g=...:b=...`.
  - `_build_blend_cmd(crush=0)` -> filter_complex has NO lutrgb (clean disable path).
  - test_post_upscale_procgen_blend -> 17 passed in 3.42s. (No new tests yet for crush; the existing tests don't cover the lutrgb step but pass on the API surface change.)
  - bug_bible_regression -> 22 passed, 1 unrelated pre-existing failure on save_copy / portrait nodes (BUG-01.02), 1 skipped, 2 xfailed.
- **Tags:** procgen-blend, color-cast, ffmpeg-lutrgb, shadow-crush, BUG-099-followup, video-model-noise-floor
- **Related:**
  - BUG-LOCAL-096 (initial pink at screen 1.0 -- root cause was already this noise floor; BUG-099 worked around it by switching mode, BUG-103 fixes it at the source).
  - BUG-LOCAL-099 (lighten 1.0 default -- still the right default, but now with crush=18 the blue-bloom-in-shadows side effect is gone too).
  - BUG-LOCAL-102 (dropdown expansion -- the test workflow that exposed how visible the residual cast was across modes).
- **Followup (not blocking):** harden the procgen prompt to explicitly request "pure black background, RGB(0,0,0), no haze, no color cast" so the noise floor narrows even before crush. Belt-and-suspenders.

---

### BUG-LOCAL-102 [FIXED]: blend_mode dropdown only had 5 modes -- expanded to 16 ffmpeg blend filter modes
- **Date:** 2026-05-04 LATE EVENING | **Phase:** BUG-099 tuning workflow | **Bible candidate:** YES (dropdown coverage)
- **Symptom:** standalone test workflow `workflows/blend_test.json` loaded into ComfyUI Desktop with 8 OTR_PostUpscaleProcgenBlend nodes, each pre-set to a different blend mode for A/B comparison. Five of the 8 nodes showed the mode value in RED text on canvas (`hardlight`, `softlight`, `dodge`, `multiply`, `darken`) -- ComfyUI's frontend marker for "this dropdown value is not in the allowed choices". Validation would reject the queue.
- **Cause:** `_BLEND_MODE_CHOICES` in `nodes/otr_post_upscale_procgen_blend.py` only listed 5 modes: `["lighten", "screen", "addition", "overlay", "normal"]`. The ffmpeg blend filter supports many more (hardlight, softlight, multiply, darken, dodge, burn, vividlight, linearlight, pinlight, difference, exclusion, etc.) but the OTR node never exposed them. The 5 original choices were a conservative initial set; nothing in the implementation prevented the wider set from working.
- **Fix (`nodes/otr_post_upscale_procgen_blend.py`, ~10 LOC):** expanded `_BLEND_MODE_CHOICES` from 5 to 16 modes:
  - Brightening tier: `lighten`, `screen`, `addition`, `overlay`, `hardlight`, `softlight`, `dodge`, `vividlight`, `linearlight`, `pinlight`
  - Darkening tier: `multiply`, `darken`, `burn`
  - Contrast tier: `difference`, `exclusion`
  - Trivial: `normal`
  - All 16 are valid ffmpeg `blend=all_mode=` values per ffmpeg filter docs. Sorted by usefulness for OTR's bright-overlay aesthetic.
- **Companion: rebuilt `workflows/blend_test.json` (8 nodes pre-set to lighten / screen / addition / overlay / hardlight / softlight / dodge / vividlight at 1.0 opacity).** Loads cleanly into ComfyUI now; queue once, get 8 mp4s in `output/otr/obs/` for side-by-side comparison.
- **Verify:**
  - AST parse clean (21240 bytes, 1143 nodes).
  - test_post_upscale_procgen_blend -> 17 passed in 3.38s.
  - Live: load `workflows/blend_test.json` into ComfyUI Desktop -- all 8 mode dropdowns should display in normal text (not red).
- **Tags:** procgen-blend, dropdown-coverage, ffmpeg-blend-filter, BUG-099-tuning
- **Related:** BUG-LOCAL-099 (the procgen overlay tuning that BUG-102 enables faster iteration on). The default mode + opacity (BUG-099 lighten 1.0) is unchanged; this entry just widens the dropdown so users can A/B test other modes without code edits.

---

### BUG-LOCAL-101 [FIXED]: SDPA prefill OOM on 310-word run -- Mistral-Nemo context_cap dropped 16384 -> 8192
- **Date:** 2026-05-04 LATE EVENING | **Phase:** post BUG-098 tripwire validation | **Bible candidate:** YES (context-window VRAM tradeoff)
- **Symptom:** 310-word run at 18:02:24 OOM'd during main script generation prefill. `torch.OutOfMemoryError: Currently allocated 25.41 GiB / Device limit 15.92 GiB. Requested 4.15 GiB`. The single 4.15 GiB allocation request was the SDPA attention buffer for one layer's QKV scaled-dot-product on the long prompt.
- **What this is NOT (BUG-098 was a misdiagnosis):**
  - The BUG-098 NF4 tripwire FIRED on BOTH loads (first load AND the OpenClose-triggered second load) and PASSED both times: `linear4bit_count=280 is_loaded_in_4bit=True vram_delta=7.74GiB (ceiling=11.00GiB)`. NF4 quantization IS being applied correctly on every load. The "fast 33 it/s vs slow 22 it/s" weight-loading delta I attributed to "bitsandbytes silent fp16 fallback" was wrong -- it's just Windows filesystem cache warm-up after the first load. Both ChatGPT and Gemini round-robin had warned that the speed delta was suggestive, not conclusive; the tripwire's module-class inspection is the ground truth, and it confirms NF4 every time.
  - The `LLM cache mismatch [budget_profile: 'Standard'->'Pro (Ultra Quality)']` unload+reload at 18:02:05 IS a separate cache-hygiene issue (some earlier inference passed Standard profile when the orchestrator should have been using Pro Ultra Quality). It triggers an unnecessary reload, but NF4 still applies on the reload, so it's not the OOM cause.
- **Cause:** the actual OOM is inference-time prefill attention. With Mistral-Nemo 12B + bnb-NF4 + bf16 compute + max_position_embeddings=16384 + long prompt (~6-8k tokens of winning spine + cast roster + news body + format spec + announcer bookend prompt), the SDPA prefill needs to allocate per-layer attention buffers that scale as O(seq_len^2 * n_heads). At 16384 cap, one layer's attention buffer can hit 4+ GiB. Combined with the 7.74 GiB NF4 model + KV cache + activation buffers, total allocation reaches 25 GiB on a 16 GiB device. v1.7's known-good cap was 6144; the cap was raised to 16384 in BUG-LOCAL-065 (2026-04-29) to preserve long Gemma 4 E2B body-pass prompts but the change wasn't VRAM-budgeted for Mistral-Nemo's larger architecture.
- **Fix (`nodes/story_orchestrator.py`, 1-line change):**
  - **`_MODEL_CONTEXT_CAPS["mistralai/Mistral-Nemo-Instruct-2407"]: 16384 -> 8192`**. Halving the cap reduces:
    - Per-layer attention buffer worst case by ~4x (the N^2 component of SDPA scales quadratically; halving N reduces the buffer by 4x).
    - KV cache reservation by 2x.
    - Net: ~6-10 GiB headroom recovered for the prefill spike.
  - Companion caps for other 12B-class models (Qwen 14B, Captain-Eris, Mag-Mell) also dropped from 12288 to 8192 by the same logic. Gemma 2/4 caps preserved at 16384/8192 because those are smaller models with smaller per-layer attention budgets.
  - 8192 is a middle ground -- v1.7 used 6144; 16384 OOMs on long prompts. 8192 preserves enough prompt headroom for a 310-word OpenClose run while keeping inference VRAM under the 14.5 GiB ceiling. If 8192 still OOMs on a 700-word run, drop to 6144 next.
- **Verify:**
  - AST parse clean (573613 bytes, 39863 nodes).
  - test_core + test_critique_dialogue_preservation + test_news_history_ttl -> 139 passed in 5.22s.
  - Live: next 310-word run should complete the main script generation phase without OOM. The runtime log will show `[StoryOrchestrator] Hardening: Capping 128k context to 8192` instead of the prior `to 16384`.
- **What this changes for the BUG-098 entry:** moving BUG-098 from `[PARTIAL FIX SHIPPED]` to `[FIXED -- TRIPWIRE PROVED NF4 NEVER WAS THE PROBLEM]`. The tripwire is still valuable as a regression guard; BUG-098 just turns out to have been the wrong diagnosis for the OOM symptom. The OOM is BUG-101.
- **Tags:** llm-context, sdpa-prefill, attention-buffer, vram-ceiling, mistral-nemo-12b, BUG-098-misdiagnosis-followup
- **Related:** BUG-LOCAL-098 (the NF4 tripwire that proved this isn't a quantization issue). BUG-LOCAL-065 (the 2026-04-29 cap raise that this entry partially reverts for the 12B-class models). BUG-LOCAL-004 (earlier ScriptWriter OOM history; same family of "main gen OOMs at long prompt sizes"). The `budget_profile: 'Standard'->'Pro (Ultra Quality)'` cache drift is a separate hygiene issue that deserves its own follow-up but is not blocking. Tomorrow: build the isolated NF4 reload test harness anyway -- now we know the SECOND-load NF4 path is fine, so the harness pivots to validating the cap reduction holds across rerun cycles + investigating where Standard sneaks into the budget_profile chain.

---

### BUG-LOCAL-100 [DIAGNOSED, FIX TOMORROW]: Bark over-pads short character lines with hallucinated noise tail
- **Date:** 2026-05-04 LATE EVENING | **Phase:** post BUG-099 hotfix observation | **Bible candidate:** YES (Bark TTS lifecycle / silence-trim / dur-cap)
- **Symptom (live `signal_lost_echo_in_stasis_20260504_170903`):** at 0:34 in the final composite the audio is "harsh noise / garbage" -- not dialogue. Visual is RUFE's HuMo lipsync (correct routing), but the underlying audio is incoherent noise.
- **Diagnosis from ledger:**
  - `lines[2]` (l003): `text="Golly, it! THE STOWAGE TUBE COLLAPSED."`, `word_count=6`, `dur_s=14.4`, `bark_wav_dur_s=14.4`, `bark_render_ms=28070`. The text is a ~6-word phrase that should take ~2.4s at normal speech rate (2-3 words/sec). Bark generated 14.4s of audio anyway, then OTR placed the whole 14.4s wav into the master mix at start_s=21.08s. The first ~3s is real speech; the trailing ~11s is Bark hallucinating tail noise to fill the requested duration.
  - Episode-wide degenerate output: `total_episode_dur_s=43.08`, `total_word_count=23` for a 100-word target. The script writer collapsed the script into 3 short lines, then each line's audio was over-padded the same way.
- **Likely cause (one of three; needs investigation):**
  1. **`max_length` param on Bark generation is set per-LINE based on line `dur_s` instead of word_count.** When the LLM stamps `dur_s=14.4` on a 6-word line, Bark gets asked for 14.4s and obediently produces noise to fill.
  2. **No VAD / silence-trim post-Bark.** Even if Bark over-generates, a trim pass at the speech-end boundary would cut the noise tail before placement.
  3. **Speaker preset `v2/en_speaker_1` has known long-tail hallucination on short prompts.** Some Bark voices end better than others; en_speaker_1 historically tends to ramble. Switching default character voice to a cleaner preset (en_speaker_3, en_speaker_9) could mitigate.
- **What this is NOT:**
  - Not a pipeline placement bug -- the HuMo + LTX + composite ledger is correct; the audio overlap with RUFE's lipsync IS the right routing.
  - Not the same as BUG-LOCAL-027 (critique pass wiping dialogue) -- the LLM produced dialogue, just degenerate-short.
  - Not a Bark loader / VRAM issue -- Bark ran for 28s and produced a wav, just one with hallucinated tail.
- **Fix candidates (pick after harness tomorrow):**
  - **A) Word-count-derived duration cap before Bark.** `target_dur_s = max(min_dur, word_count / 2.2)`. Pass this as Bark's max_length. Cuts the over-pad at the source.
  - **B) Post-Bark silence/VAD trim.** Run silero-vad or a simple energy threshold trim on the Bark wav before placement. Cuts the hallucinated tail.
  - **C) Drop trailing low-energy region.** Simpler than VAD: scan the wav from the end and drop any segment below -40dBFS until real speech is found. Adds ~50ms post-process per line.
  - **D) Cap LLM-allocated `dur_s` on the line.** Make ScriptWriter clamp `dur_s = min(stated_dur, words / 2.0)` so the contract Bark sees is sane.
- **Recommendation:** A + B together. A prevents over-generation; B cleans up edge cases where Bark still pads. C is a fallback if VAD library fails to install on the Blackwell stack.
- **Workaround for tonight:** none. Bark's output is what it is for this run. Restart + re-queue picks a different headline + script and may produce healthier dur_s allocations.
- **Tags:** bark-tts, vad, silence-trim, dur-mismatch, audio-quality, BUG-027-adjacent
- **Related:** BUG-LOCAL-027 (critique pass wiping dialogue -- different cause, similar surface symptom of "not enough actual dialogue"). Mistral-Nemo + creativity=maximum chaos + 100-word target combo may amplify this; might also help to clamp creativity for ultra-short presets.

---

### BUG-LOCAL-099 [FIXED]: procgen overlay produced global magenta tint -- "screen" -> "lighten" at full strength
- **Date:** 2026-05-04 LATE EVENING | **Phase:** post BUG-096 hotfix | **Bible candidate:** YES (defaults tuning)
- **Symptom:** screenshot from live composite at 0:00:03 (`Echo in Stasis` episode) showed the entire frame magenta/pink. Radio room walls, porthole, TV screen, control panel -- everything tinted. The post-BUG-096 default of `screen` blend at 1.0 opacity was adding procgen's color values to every pixel, producing a global magenta cast in regions where procgen had uniform mid-tone color (the SIGNAL LOST scene background).
- **Cause:** `screen` mode formula = `1 - (1 - A) * (1 - B)`. When B (procgen) has a uniform color value > 0 across the frame, the result is uniformly lifted toward that color across the entire frame. The 1940s SIGNAL LOST procgen aesthetic includes pink/magenta scanline color in mid-tone regions, which screen mode then propagates everywhere it adds. BUG-096 traded "weak overlay" for "tinted overlay".
- **Fix (`nodes/otr_post_upscale_procgen_blend.py`, ~6 LOC):**
  - **`_DEFAULT_BLEND_MODE = "lighten"`** (was `"screen"`). `lighten` mode = pixel-wise `max(upscale, procgen)`. Bright procgen elements (white SIGNAL LOST text, scanlines, waveform graphics) show through at full intensity because they're brighter than the underlying frame. Mid-tone procgen regions (the magenta ambient cast) defer to the upscale wherever the upscale is brighter, eliminating the global tint. Keeps the brightness intent of BUG-096 without the color-cast side effect.
  - **`_DEFAULT_BLEND_OPACITY = 1.0`** (unchanged from BUG-096) -- still full strength.
  - **Workflow JSON `workflows/otr_scifi_16gb_full.json`** -- saved widget values updated from `["screen", 1.0]` to `["lighten", 1.0]` so existing workflow loads pick up the new defaults without canvas adjustment.
- **Test update (`tests/test_post_upscale_procgen_blend.py`):**
  - Renamed `test_default_blend_mode_is_screen` -> `test_default_blend_mode_is_lighten`. Assertion follows the constant.
  - `test_default_blend_opacity_is_full_strength` unchanged (still 1.0).
- **Verify:**
  - `tests/test_post_upscale_procgen_blend.py` -> 17 passed in 4.74s.
  - Live: next composite should show procgen's bright graphics overlaid on the upscale's HuMo/LTX content without a global color cast. Radio scene retains its native sepia/grey/orange palette except where procgen scanlines/text are explicitly bright.
- **Per-episode override:** the widget remains user-tunable. If a future episode wants the magenta-soaked screen effect intentionally (e.g. a "transmission breaking up" beat), set `blend_mode = "screen"` and `blend_opacity = 1.0` on that run's canvas. Default stays `lighten 1.0` for the SIGNAL LOST baseline aesthetic.
- **Tags:** procgen-blend, post-upscale, defaults, ffmpeg-blend-filter, color-cast, signal-lost-aesthetic, BUG-096-followup
- **Related:** BUG-LOCAL-096 (the brightness-bump fix that BUG-099 retunes; together they converge on "bright procgen elements visible at full strength, no global color tint"). BUG-LOCAL-030 Phase B (the original blend pipeline). The C7 audio passthrough remains unchanged across BUG-096 + BUG-099.

---

### BUG-LOCAL-098 [PARTIAL FIX SHIPPED -- tripwire + accelerate clear; rehydrate path deferred to test harness]: NF4 silently fails on second `_load_llm` after `_unload_llm`
- **Round-robin transcripts:** `docs/2026-05-04-bug-098-nf4-second-load__01_chatgpt.md` (gpt-5.5, 128.6s), `docs/2026-05-04-bug-098-nf4-second-load__02_gemini.md` (gemini-3.1-pro-preview-customtools, 60.7s), synthesis at `docs/2026-05-04-bug-098-nf4-second-load__04_synthesis.md`.
- **Convergent recommendations both LLMs accepted (shipped tonight):**
  1. **Path 3 -- post-load NF4 tripwire**: after `from_pretrained()`, count `bitsandbytes.Linear4bit` modules, check `model.is_loaded_in_4bit`, measure CUDA allocation delta against an 11.0 GiB ceiling. If quantization was requested but did not materialize, log diagnostics + clean up the broken model + raise a clear `RuntimeError` referencing BUG-LOCAL-098 with a "restart ComfyUI Desktop" workaround. This converts the silent OOM cascade into a single loud failure with actionable next-step guidance.
  2. **`accelerate.clear_device_cache()` in `_unload_llm`** (Gemini specific suggestion): clear accelerate's device-dispatch cache between loads. Defensive; no-op when accelerate is absent or the API has shifted.
  3. **Fresh-config comment pin**: explicit comment block above the existing `BitsAndBytesConfig(...)` instantiation noting that the construction MUST stay in-function (not module-cached) -- transformers mutates the config during `from_pretrained` and a reused instance can silently skip quantization. Verified the existing code already constructs it fresh; the comment prevents a future "optimization" from regressing this.
- **Divergent recommendations deferred (test harness needed first):**
  - **Path 2 -- `model.cuda()` rehydrate of the cached CPU-parked model**: ChatGPT recommends with strong guards; Gemini rejects citing `Linear4bit` `quant_state`/`absmax` corruption risk on naive device move. Need an isolated 3-iteration `load NF4 + 1-token inference + unload` test harness to validate `.cuda()` works on the specific stack (torch 2.10 nightly + Blackwell sm_120 + bitsandbytes for CUDA 13.0) before shipping. Tomorrow.
  - **Path 4 -- LLM subprocess isolation**: ChatGPT's fallback if Path 2 fails. Heavier change; defer until Path 2 is ruled out.
- **Cannot ship Gemini's other suggestion**: dropping `model.cpu()` from `_unload_llm`. That code is BUG-LOCAL-073 hardening for abandoned `_run_with_timeout` worker threads -- without it the second load sees 31+ GiB. Removing it re-breaks BUG-073.
- **Workaround for live runs (unchanged from initial diagnosis):** restart ComfyUI Desktop between full runs. The first load per process is NF4-correct. Post-098 the run will now FAIL FAST on the broken second-load path with a clear error message instead of cascading into multiple 24 GiB loads.
- **Verify (tonight's partial fix):**
  - AST parse clean (572579 bytes, 39863 nodes).
  - test_core + test_critique_dialogue_preservation + test_news_history_ttl -> 139 passed in 4.42s. No existing tests broken by the tripwire / accelerate-clear additions.
  - Live: next run with the second-load NF4 failure should now log `[BUG-098 tripwire] post-load: linear4bit_count=0 is_loaded_in_4bit=False vram_delta=24.XX GiB (ceiling=11.00GiB)` followed by `RuntimeError: BUG-LOCAL-098: NF4 quantized load did not materialize for ...`. Clean failure, no OOM cascade.
- **Tags:** llm-cache, bitsandbytes, nf4, vram, oom, round-robin-shipped, BUG-085-related, tripwire
- **Related:** BUG-LOCAL-085 (the HF_HOME fix that resolved FIRST-load NF4 silent-failure; this is the SECOND-load equivalent). BUG-LOCAL-073 (`_unload_llm()` synchronize-before-cpu hardening that we cannot remove). Round-robin process: ChatGPT vs Gemini disagreement was material on Path 2 (rehydrate); convergence on Path 3 (tripwire) drove tonight's ship. Tomorrow's harness work tracked as BUG-LOCAL-098a.

---

### BUG-LOCAL-098 [DIAGNOSED, FIX PENDING]: NF4 silently fails to apply on second `_load_llm` call after `_unload_llm`
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-097 hotfix soak | **Bible candidate:** YES (bitsandbytes/transformers state hazard)
- **Symptom (live queue 16:31:57 PT):** Mistral-Nemo loaded successfully on first attempt, ran 3 inferences (news ranking + body re-rank + 800-token story plan) at NF4 (~7-8 GiB allocated, 22 it/s weight loading speed which is the cost of bnb quantizing each weight). Then `_unload_llm()` fired clean (`VRAM allocated=0.02 GiB reserved=0.10 GiB`). The NEXT `_load_llm()` for OpenClose's first spine claimed `Enabling 4-bit quantization (NF4)` in the log, but weight loading hit **33 it/s** (faster = no quantization) and the resulting model occupied **24.54 GiB of VRAM** -- fp16 size for 12B Mistral-Nemo. First inference OOM'd at SDPA prefill. Fallback `_load_llm()` calls accumulated more 24 GiB models until the run died at 25.77 GiB allocated / 4.24 GiB requested / 15.92 GiB device limit.
- **Smoking gun:** the **22 it/s vs 33 it/s** delta on weight loading. NF4 quantization is computational; bitsandbytes inserts a quantize step per weight load. 22 it/s = quantizing. 33 it/s = raw fp16 load, no quantization. The config `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")` is being passed but NOT applied. This is a SILENT failure mode -- transformers neither raises nor warns; it just hands back an unquantized model.
- **Cause hypothesis (needs round-robin confirm):** bitsandbytes carries module-level state (CUDA context references, CUDA-stream handles, the `Linear8bitLt`/`Linear4bit` factory cache) that survives `_unload_llm()`'s `model.cpu()` + `del` + `gc.collect()` + `torch.cuda.empty_cache()` + `comfy.model_management.soft_empty_cache()`. The first quantization succeeds because state is fresh from import. After unload, bitsandbytes' internal CUDA references are stale (point at evicted memory), but the module thinks it's still initialized, so on the second `from_pretrained` call it short-circuits the quantization path -- presumably checking some `_is_initialized` flag that says yes -- and the model loads at fp16 dtype with the BitsAndBytesConfig silently ignored.
- **Compounding factor (BUG-085 history):** the BUG-LOCAL-085 fix for HF_HOME / canonical-snapshot resolution worked perfectly on the FIRST load (verified live earlier today). It also fires on the SECOND load (`OTR_HF_ENV] snapshot resolved...` log present). So the cache resolution path is correct. The failure is post-resolution, INSIDE bitsandbytes' quantization step, and is independent of BUG-085's fix domain.
- **Why the unload happens at all:** `_load_llm()` line 2043 calls `_unload_llm()` when `cache_deltas` is non-empty. The OpenClose path triggers this -- something between the third inference and OpenClose's first spine call drifts a cache field (most likely `model_evicted_to_cpu` since accelerate may park weights on CPU under memory pressure). Pre-093 the dispatch hid this with stopgaps; post-093 it's visible.
- **Workaround for tonight (no code change):** restart ComfyUI Desktop between full runs. The FIRST load per process is NF4-correct. Subsequent reloads are broken. As long as the workflow completes within a single load lifecycle, NF4 holds.
- **Proper fix (PENDING; round-robin tomorrow):** three candidate paths, none small enough to ship without consultation:
  1. **Module-level reset of bitsandbytes** during `_unload_llm()`: delete `bitsandbytes` and `transformers.integrations.bitsandbytes` from `sys.modules` so the next load re-imports them fresh. Risky if torch/accelerate hold strong references.
  2. **Skip `_unload_llm()` for same-model_id reloads**: when the cache_delta is only `model_evicted_to_cpu` AND `quantized=True` AND `model_id` matches the cached one, do `model.cuda()` to bring weights back instead of unload+reload. Sidesteps the bitsandbytes state issue entirely. Simplest semantically.
  3. **Post-load NF4 assertion**: after `from_pretrained`, check `torch.cuda.memory_allocated()` against an expected-NF4 ceiling (e.g. <10 GiB for Mistral-Nemo). If higher, raise a loud `RuntimeError` referencing BUG-098. This is a TRIPWIRE not a fix -- the run still fails, but with a clear error instead of a silent OOM.
- **Recommendation:** Path 2 (skip the unload) is the safest semantic fix. Path 3 (tripwire) lands alongside as the safety net. Path 1 (module reset) is the nuclear option if Paths 2+3 don't hold.
- **Acceptance criteria:**
  - Second `_load_llm()` after `_unload_llm()` produces an NF4 model (allocated < 10 GiB for Mistral-Nemo).
  - Three full episodes generated back-to-back in one ComfyUI session without OOM.
  - Bug Bible regression unchanged.
- **Tags:** llm-cache, bitsandbytes, nf4, vram, oom, round-robin-needed, BUG-085-related
- **Related:** BUG-LOCAL-085 (the HF_HOME / canonical-snapshot fix that resolved the FIRST-load NF4 silent-failure; BUG-098 is the SECOND-load equivalent). BUG-LOCAL-005 + BUG-LOCAL-004 (earlier ScriptWriter OOM history). BUG-LOCAL-073 (`_unload_llm()` synchronize-before-cpu hardening, which got us to clean-unload but did not address the bitsandbytes-state-on-reload issue).

---

### BUG-LOCAL-097 [FIXED]: BatchLTXRender widget order broke existing workflows -- clip_length moved to last
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-091 hotfix | **Bible candidate:** YES (widget-position positional-parse hazard)
- **Symptom (live queue):** `Failed to validate prompt for output 56: OTR_BatchLTXRender 55: Failed to convert an input value to a FLOAT value: clip_length, ffmpeg, could not convert string to float: 'ffmpeg'`. Workflow refused to run; LTX node + downstream (VideoComposite, RTXUpscale, procgen blend) all skipped before any inference started.
- **Cause:** BUG-LOCAL-091 added `clip_length` (FLOAT) as the FIRST entry in `INPUT_TYPES["optional"]`. ComfyUI parses `widgets_values` positionally -- the saved workflow JSON had 5 values `["", 1, "fixed", "ffmpeg", ""]` matching pre-091 order (ledger_json, seed, seed_mode, ffmpeg, humo_clips_dir). After BUG-091 the optional dict order became (clip_length, ffmpeg, humo_clips_dir), so position [3] in the saved values was now expected to be FLOAT clip_length but contained the literal string `"ffmpeg"`. Validation failed before any node executed.
- **Fix (`nodes/batch_ltx_render.py`, ~50 LOC reorder):**
  - Moved `clip_length` to the LAST entry of `INPUT_TYPES["optional"]` (after `ffmpeg` and `humo_clips_dir`). Saved workflow values now line up with their original slots; `clip_length` is a NEW position [5] that old workflow JSONs leave unset, falling through to the FLOAT default of 7.0.
  - Reordered the `execute()` signature kwargs to match the new optional dict order (`ffmpeg, humo_clips_dir, clip_length=7.0`).
  - Added a comment block at the new clip_length position explaining the BUG-097 backward-compat reason so a future refactor doesn't innocently move it back to the front.
- **NEW test (`tests/test_batch_ltx_render.py::test_clip_length_widget_appears_after_existing_optional_widgets`):** asserts `clip_length` is the LAST key in `INPUT_TYPES["optional"]`. A future refactor that re-inserts it earlier in the dict fails this test before workflow validation surfaces the same error in production.
- **Verify:**
  - AST parse clean (56043 bytes, 3979 nodes).
  - test_batch_ltx_render -> 22 passed in 1.64s (was 21; +1 BUG-097 guard).
  - Live: queue the existing workflow JSON, validation should now pass for the LTX node and the rest of the graph runs.
- **Tags:** widget-order, comfyui-positional-parse, backward-compat, BUG-091-followup
- **Related:** BUG-LOCAL-091 (the original chunking change that added the widget); BUG-LOCAL-086 (HuMo equivalent that did NOT have this issue because clip_length was already in INPUT_TYPES pre-086 -- only the max value changed). General lesson: NEW widgets in INPUT_TYPES["optional"] must always go at the END to preserve workflow-JSON backward compat. Bible candidate.

---

### BUG-LOCAL-096 [FIXED]: procgen overlay blend too weak -- bumped to screen mode at full strength
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-095 LTX fix | **Bible candidate:** YES (default value, not a code defect)
- **Symptom:** Jeffrey: "the procgen video mix is to weak ... bring out the colors and full lighting of the procgen video in the final concat. Now it looks weak. I want it as bright as the original just overlayed". The blended final mp4 looked washed out -- procgen colors at half intensity, the SIGNAL LOST CRT signature barely visible over the upscale base.
- **Cause (defaults, not a code defect):**
  - `_DEFAULT_BLEND_MODE = "lighten"` -- per-pixel `max(A, B)`. Wherever procgen pixel is darker than upscale, you see upscale. Sharp-edged but dim-region procgen content disappears.
  - `_DEFAULT_BLEND_OPACITY = 0.5` (default tooltip even called it "moderate sheen"). At 0.5 the filter mixes 50% blended result with 50% original source, so even bright procgen pixels get faded to half intensity.
  - Combined effect: procgen layer visible only where it's BOTH brighter than upscale AND has its strength halved. Net: weak, washed-out look that doesn't honor the procgen's intended brightness.
- **Fix (`nodes/otr_post_upscale_procgen_blend.py`, ~10 LOC):**
  - **`_DEFAULT_BLEND_MODE = "screen"`** (was `"lighten"`) -- canonical bright-additive overlay. Result = `1 - (1-A)(1-B)`, always brighter than either input layer. Two black inputs = black; two white inputs = white; mixed = lifted. Classic film-projector / double-exposure aesthetic that preserves the upscale visible underneath while bringing procgen colors at full strength.
  - **NEW constant `_DEFAULT_BLEND_OPACITY = 1.0`** (was inline `0.5`). At 1.0 the filter emits the full blended result without mixing back to the original.
  - Function signature default `blend_opacity: float = _DEFAULT_BLEND_OPACITY` (was hardcoded `0.5`) so the in-process default and the widget default stay in lockstep automatically.
  - Tooltip rewritten to explain that 1.0 = full blend strength, drop to 0.5 for "moderate sheen" if the user wants the old behaviour.
  - **Workflow JSON `workflows/otr_scifi_16gb_full.json`** -- saved widget values updated from `["lighten", 0.5]` to `["screen", 1.0]` so existing workflow loads pick up the new defaults without manual canvas adjustment.
- **NEW + UPDATED tests (`tests/test_post_upscale_procgen_blend.py`):**
  - Renamed `test_default_blend_opacity_is_05` to `test_default_blend_opacity_is_full_strength`; assertion changed to `1.0`.
  - NEW `test_default_blend_mode_is_screen` -- pin the new mode.
- **Verify:**
  - AST parse clean (19627 bytes, 1132 nodes).
  - `tests/test_post_upscale_procgen_blend.py` -> 17 passed in 3.23s.
  - 5-suite combined sweep -> 95 passed in 3.27s.
  - Live: next final mp4 should show procgen colors at full intensity, classic bright-overlay look. Upscale (HuMo + LTX) still visible underneath because `screen` is additive, not replacement.
- **Tags:** procgen-blend, post-upscale, defaults, ffmpeg-blend-filter, brightness, signal-lost-aesthetic
- **Related:** BUG-LOCAL-030 Phase B (the original blend pipeline that BUG-096 retunes the defaults of). The C7 audio passthrough (`-c:a copy`, no `-shortest`) is unchanged -- audio path stays byte-identical to v1.5 baseline regardless of visual blend params. Per-user override: drop `blend_opacity` to 0.5-0.7 if the new screen-1.0 default is too bright for a specific episode.

---

### BUG-LOCAL-095 [FIXED]: LTX clips visibly static -- LTXVAddGuide is keyframe pinning, not i2v init
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-091 LTX chunking soak | **Bible candidate:** YES (LTX i2v dispatch)
- **Symptom:** Jeffrey reported "the LTX radio being honest its not animating at all... we thought removing the end frame would do it but i guess not". Even after BUG-LOCAL-032 removed the end-frame guide, LTX clips still rendered visibly static -- the radio scene held a near-frozen pose for the entire clip duration. CFG, sampler, sigmas, dimensions, strength, frame_rate all matched the historically-working ComfyUI-Goofer / comfyui-data-media-machine paths, so the parameters weren't the issue.
- **Diagnostic process:** side-by-side comparison of OTR's `nodes/batch_ltx_render.py` vs comfyui-data-media-machine's `nodes/dmm_batch_video.py::_apply_i2v_conditioning` (called by `DMMBatchVideoGenerator`). DMM's i2v path produces animated output every run; OTR's doesn't. Sigmas (9-element distilled schedule), CFG=1.0, sampler=euler, fps=25, strength=0.75 were all identical. The ONE difference: the i2v ComfyUI node call.
- **Cause:** OTR called `LTXVAddGuide` for i2v conditioning. `LTXVAddGuide` is a KEYFRAME PINNING node -- it attaches the image to the positive/negative conditioning as a hard anchor at `frame_idx=0`, clamping the latent at that frame and constraining motion away from it. Even at strength=0.75, frame 0 stays rigidly locked; the model resists evolving away from the start image because the conditioning is telling it "this is what frame 0 must look like". Pre-BUG-032 there were TWO guides (start strength 0.75 + end strength 0.6), explaining the original ping-pong static behaviour. BUG-032 removed the end guide, but the start guide was still pinning. DMM uses the canonical `LTXVImgToVideoConditionOnly` node instead -- it encodes the image into the FIRST FRAMES of the latent and adds a noise mask for strength control. Same starting frame, but the model sees "evolve from this" rather than "stay locked to this".
- **Fix (`nodes/batch_ltx_render.py`, ~30 LOC):**
  - **Replaced `_call("LTXVAddGuide", positive, negative, vae, latent, image, frame_idx=0, strength)` with `_call("LTXVImgToVideoConditionOnly", vae, image, latent, strength)`**. Returns a single conditioned LATENT instead of (modified positive, modified negative, latent).
  - **`CFGGuider` now receives `cond_pos` / `cond_neg` straight from `LTXVConditioning`** (the original conditioning, unchanged). Pre-095 it received the LTXVAddGuide-modified pair, which carried the keyframe anchor.
  - **Updated required-nodes error message** to reference `LTXVImgToVideoConditionOnly` so fresh-install users know what to look for.
  - **Updated stale BUG-032 commentary** that referenced the removed `LTXVAddGuide` path.
- **NEW tests (`tests/test_batch_ltx_render.py`, +2):**
  - `test_i2v_dispatch_uses_img_to_video_condition_only` -- regex-based source guard that asserts `_call("LTXVImgToVideoConditionOnly", ...)` is present AND `_call("LTXVAddGuide", ...)` is NOT present in the file. A future refactor that re-introduces LTXVAddGuide for i2v fails before reaching a live render.
  - `test_required_nodes_list_mentions_img_to_video_condition_only` -- pin the error message stays accurate.
- **Verify:**
  - AST parse clean (55053 bytes, 3979 nodes).
  - test_batch_ltx_render + test_batch_humo_render + test_news_history_ttl + test_portrait_render_skip_announcer -> 78 passed in 3.08s (was 76; +2 BUG-095 guards).
  - Live: next LTX render against the radio bookend should produce visibly animated output -- camera drift, light flicker, atmospheric movement -- matching the DMM repo's behaviour. Static-frame artefact gone.
- **Tags:** ltx, i2v-dispatch, dmm-comparison, BUG-032-followup, animation, keyframe-vs-i2v
- **Related:** BUG-LOCAL-032 (removed the end-frame guide; necessary but insufficient because the start guide was still pinning); BUG-LOCAL-091 (LTX chunking; the per-chunk render now uses the corrected i2v path so each chunk gets free motion). LTXVAddGuide is preserved in ComfyUI for genuine keyframe pinning use cases (e.g., a 60-frame clip that must hit a specific pose at frame 30); OTR just doesn't have that need.

---

### BUG-LOCAL-094 [FIXED]: skip_announcer guard never fired -- wasted ~30s FLUX per announcer cast member
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-093 cleanup | **Bible candidate:** YES (cast filter + skip dispatch)
- **Symptom:** every recent run rendered a portrait for the ANNOUNCER cast member even though announcer beats route to LTX (BUG-129b) and the portrait is never used. ~30s of FLUX wall time wasted per portrait. Visible in the runtime log as e.g. `[OTR_BatchFluxPortraitRender] c01 -> c01_portrait.png (29.7s)` for `c01: ANNOUNCER`.
- **Cause:** `OTR_BatchFluxPortraitRender` had a `skip_announcer` widget (default True) and an in-loop check `role = (c.get("speaker_role") or c.get("role") or "").lower(); if skip_announcer and role == "announcer": continue`. But the ledger cast block doesn't carry `speaker_role` -- it lives per-line in `ledger.lines[]`. The cast dict shape is `{char_id, name, description, gender, voice_preset, line_count, word_count, portrait_path}` -- no role field. So `c.get("speaker_role") or c.get("role")` always returned an empty string, the `if` never fired, and every cast member got a portrait regardless of role.
- **Fix (`visual/batch_flux_portrait_render.py`, ~80 LOC):**
  - **NEW: walk `ledger.lines[]` to build `char_id_has_character_line: dict[str, bool]`**. For each line, call `resolve_speaker_role(ln)` (canonical helper from `_otr_speaker_role`). If any line for a char_id has `speaker_role == "character"`, mark True. Cast members with all-non-character lines (announcer, music_*, sfx) end up False.
  - **Two-tier skip dispatch in the cast loop**:
    1. **Tier 1 (line-driven, canonical)**: if `skip_announcer` is on AND `lines[]` was non-empty AND this `char_id` has zero character lines -> skip with log `cast[i] NAME (cid) all lines non-character; skip per skip_announcer=True (BUG-LOCAL-094)`.
    2. **Tier 2 (legacy name-match fallback)**: if `skip_announcer` is on AND `lines[]` was empty/missing AND `cast.name == "ANNOUNCER"` (case-insensitive) -> skip with log `cast[i] NAME (cid) name=ANNOUNCER (legacy fallback, no lines block); skip per skip_announcer=True`. Covers degraded ledgers from earlier schema versions.
  - **Diagnostic log line**: `BUG-094 cast filter: M/N cast member(s) have >=1 character-role line` so the post-mortem can verify the filter ran.
  - **Lazy import of `_otr_speaker_role`**: tries `from nodes._otr_speaker_role` first, falls through to `from _otr_speaker_role` (sibling-on-sys.path pattern), then to a string fallback `SPEAKER_ROLE_CHARACTER = "character"` if the helper is unavailable. Defensive at module load.
  - **Falls back to "render anyway"** when `lines[]` is missing/empty AND name doesn't match "ANNOUNCER" -- safer to render an unused portrait than to skip silently.
- **NEW tests (`tests/test_portrait_render_skip_announcer.py`, 5 tests):**
  - `test_skip_announcer_widget_default_true` -- pin the widget default.
  - `test_dispatch_uses_lines_block_not_cast_speaker_role` -- assert source contains `char_id_has_character_line` and `resolve_speaker_role`.
  - `test_dispatch_has_legacy_name_match_fallback` -- assert legacy `speaker.upper().strip() == "ANNOUNCER"` fallback is present.
  - `test_no_pre094_speaker_role_check_remaining` -- regex regression guard against re-introducing the broken `c.get("speaker_role") or c.get("role")` predicate.
  - `test_skip_logs_bug_094_reference` -- pin `BUG-LOCAL-094` is referenced in skip-path logs for audit traceability.
- **Verify:**
  - AST parse clean (21063 bytes, 1815 nodes).
  - test_portrait_render_skip_announcer + test_batch_humo_render + test_batch_ltx_render + test_news_history_ttl -> 76 passed in 3.07s (was 71; +5 BUG-094 guards).
  - Bug Bible OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed (same baseline; the BUG-01.02 failure on `batch_flux_portrait_render.py` for missing `folder_paths` predates 094 and is outside this fix's scope).
  - Live: next soak should log `cast filter: 2/3 cast member(s) have >=1 character-role line` followed by `cast[1] ANNOUNCER (c02) all lines non-character; skip per skip_announcer=True (BUG-LOCAL-094)`. ANNOUNCER's portrait is NOT rendered. Total FLUX portrait time drops by ~30s (one slot saved).
- **Tags:** flux-portraits, cast-filter, skip-dispatch, BUG-129b-followup, wall-time-savings
- **Related:** BUG-LOCAL-078 (per-cast portrait pass that BUG-094 filters); BUG-LOCAL-129b (announcer routing to LTX which is why the portrait is unused); BUG-LOCAL-088 (cast-still binding which BUG-093 removed -- this same line-driven walk pattern could have replaced 088 too if we wanted to keep the binding alive for non-character coverage). One portrait per character is preserved (each c0X writes a single c0X_portrait.png; the skip_announcer filter just removes the unused entries entirely).

---

### BUG-LOCAL-093 [FIXED]: HuMo portrait stopgaps removed -- wrong-face is worse than no-face
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-092 hardening | **Bible candidate:** YES (failure-mode policy)
- **Symptom:** even after BUG-092 inverted the dispatch order, two stopgaps remained that could still produce wrong faces:
  1. `cast_still_map` -- defense-in-depth fallback that bound `char_id` to `full_env_*.png` FLUX environment stills.
  2. `_find_portrait` tiers 4-5 -- internal fallback to `full_env_*.png` when no character portrait was found.
  Both fired only when BUG-LOCAL-078's per-cast portrait pass had failed AND the dispatch couldn't find a real portrait, but the failure mode was visibly wrong: HuMo would lipsync against an environment scene that happened to contain an unrelated person. Better to skip the line and let VideoComposite cover it with static-radio fill (BUG-129a, same handling as music/sfx) than render a wrong actor.
- **Cause (policy / not a code defect):** these were pragmatic stopgaps from before BUG-LOCAL-078 added the per-cast portrait pass. Now that BUG-078 reliably writes `c0X_portrait.png` and stamps `cast[].portrait_path`, the stopgaps mask real upstream bugs (a portrait-pass failure goes silent and produces wrong-face output instead of a loud SKIP that gets fixed).
- **Fix (`nodes/batch_humo_render.py`, ~80 LOC removed):**
  - **`_find_portrait` simplified to 3 tiers** (was 5): keeps cast.portrait_path, indexed pass1 portraits, any pass1 portrait. Removed tier 4 (`full_env_*` indexed by cast position) and tier 5 (any `full_env_*`). When no real portrait is found, returns None.
  - **`cast_still_map` dispatch removed from `execute()`**: the `_resolve_cast_stills_from_ledger()` call + log-line block (~40 LOC) gone. `cast_still_map` reduced to an empty dict so the downstream dispatch reads cleanly without restructuring (cheap defense-in-depth pin in case a future refactor re-adds binding logic).
  - **Dispatch priority** is now strictly `_find_portrait` -> `_find_composite` -> None. The third branch (`if not ref_png and char_id and char_id in cast_still_map`) was deleted entirely.
  - When `ref_png` is None, the existing SKIP path fires: log `WARNING line lXXX speaker=... role=...: no portrait AND no radio still`, append `SKIP no portrait` to the report, and `continue` to next line. VideoComposite's BUG-129a static-fill covers the time slot with the radio bookend image -- visible "missing-portrait" gap that's loud enough to surface upstream bugs without ruining the episode.
- **NEW tests (`tests/test_batch_humo_render.py`, +2):**
  - `test_find_portrait_returns_none_when_only_env_stills_exist` -- writes `full_env_00001_.png` + `full_env_00002_.png` to a tmp dir, calls `_find_portrait("EDNA", cast, tmp_path)`, asserts None. Pre-093 this would have returned the env still.
  - `test_ref_dispatch_no_env_still_fallback` -- source-code regression guard; asserts no live assignment of `ref_source = "ledger-cast-fresh"` exists in `batch_humo_render.py`. Comments / docstrings can mention the term for history; the actual code path is gone.
  - Renamed `test_ref_dispatch_prefers_find_portrait_over_cast_still_map` to `test_ref_dispatch_runs_find_portrait_before_find_composite` and updated assertions to reflect the simplified two-tier dispatch.
- **Verify:**
  - AST parse clean (108611 bytes, 8508 nodes -- ~3KB / 337 nodes smaller than pre-093).
  - test_batch_humo_render + test_batch_ltx_render + test_news_history_ttl -> 71 passed in 3.12s (was 69; +2 BUG-093 guards).
  - Bug Bible OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed (same baseline).
  - Live: a future episode where BUG-078 fails to render a portrait will now log `[BatchHumoRender] line lXXX speaker=... role=...: no portrait AND no radio still` and SKIP that line. VideoComposite covers it with static radio. The bug becomes visible (a character beat plays as static radio) instead of silent (HuMo renders a wrong face). Same handling as music/sfx.
- **Tags:** humo, ref-image-dispatch, portrait-priority, stopgap-removal, failure-mode-policy
- **Related:** BUG-LOCAL-078 (per-cast portrait pass that becomes a hard requirement post-093); BUG-LOCAL-088 (cast-still binding which is now fully removed); BUG-LOCAL-092 (priority inversion that this entry hardens further by removing the fallbacks entirely); BUG-LOCAL-129a (VideoComposite static-radio fill that covers skipped lines).

---

### BUG-LOCAL-092 [FIXED]: HuMo lipsync against FLUX env stills instead of character portraits
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-091 soak | **Bible candidate:** YES (ref-image dispatch priority)
- **Symptom (live composite from `signal_lost_scientists_map_how_down_syndrome_reshape_20260504_142107`):** Three artefacts visible in viewer screenshots:
  1. At 0:42 / 0:47 -- different male actors visible during what should be character lipsync, with weird "lip sync to environment" effect (HuMo trying to articulate an environment image into facial motion).
  2. At 0:32 vs 1:35 -- two clips that should be the SAME character (both AFFIRMATIVE SIR per the ledger lines block, char_id=c01) rendered with totally different faces.
  3. Inconsistent identity across multi-line same-character runs.
- **Cause:** ledger.clips[] entries showed `ref_png_name: "full_env_00001_.png"` and `ref_source: "ledger-cast-fresh"` for every character clip. HuMo was lipsyncing against the FLUX environment still (the radio scene), not the character portrait. Tracking the dispatch in `execute()`:
  ```python
  # batch_humo_render.py lines 1607-1621 (pre-fix)
  if char_id and char_id in cast_still_map:        # <- runs FIRST
      ref_png = cast_still_map[char_id]
      ref_source = "ledger-cast-fresh"
  if not ref_png:
      ref_png = _find_composite(...)
  if not ref_png:
      ref_png = _find_portrait(...)                # <- runs LAST
  ```
  The cast_still_map is populated by `_resolve_cast_stills_from_ledger()` which globs `full_env_*.png` (FLUX environment stills, not portraits). It assigns those to char_ids by mtime-descending cast index. Pre-BUG-078 this was a stopgap when no character portraits existed. Post-BUG-078 the per-cast portrait pass writes `<episode>/portraits/c0X_portrait.png` and stamps `cast[i].portrait_path` -- but the dispatch was checking cast_still_map FIRST, so the FLUX env still always won and the proper portrait was never reached.
  Why two clips of the same character look different despite the binding being deterministic: each char_id IS bound to a single env still for the whole run, but env stills don't carry character identity. APPRENTICE (male, dry, 50s) bound to a `full_env` that happens to contain a woman = HuMo articulates the woman's face. Different env stills for different char_ids = different "actors" on screen. And HuMo's per-chunk seed stride drifts the rendered identity further across chunks of the same line.
- **Fix (`nodes/batch_humo_render.py` lines 1607-1640, ~30 LOC):** invert the dispatch order so `_find_portrait` runs FIRST. `_find_portrait` tier 1 is `cast[i].portrait_path` (the BUG-078 portrait), with tier 4-5 falling through to `full_env_*` only when no portrait file exists. cast_still_map remains as defense-in-depth for episodes where the portrait pass didn't run.
  ```python
  # New order:
  ref_png = _find_portrait(speaker, cast, portraits_dir_path)   # tier 1: cast.portrait_path
  if ref_png: ref_source = "find_portrait"
  if not ref_png:
      ref_png = _find_composite(shot_id, speaker, portraits_dir_path)
      if ref_png: ref_source = "find_composite"
  if not ref_png and char_id and char_id in cast_still_map:
      ref_png = cast_still_map[char_id]
      ref_source = "ledger-cast-fresh"
  ```
- **NEW test (`tests/test_batch_humo_render.py::test_ref_dispatch_prefers_find_portrait_over_cast_still_map`):** source-code regression guard that locks the dispatch order. If a future refactor re-inverts the priority, this test fails before any live render surfaces the wrong-face artefact again. Asserts that the `'ref_source = "find_portrait"'`, `'ref_source = "find_composite"'`, and `'ref_source = "ledger-cast-fresh"'` string literals appear in source code in that order.
- **Verify:**
  - AST parse clean (111636 bytes, 8845 nodes).
  - test_batch_humo_render + test_batch_ltx_render + test_news_history_ttl -> 69 passed in 2.89s.
  - Live: next soak ledger.clips[] should show `ref_png_name: "c0X_portrait.png"` (NOT `full_env_NNNNN_.png`) and `ref_source: "find_portrait"` (NOT `"ledger-cast-fresh"`) for every character clip. Same character across multiple lines should render with the SAME face.
- **Tags:** humo, ref-image-dispatch, portrait-priority, bug-078-followup, audio-video-sync
- **Related:** BUG-LOCAL-078 (per-cast portrait pass that BUG-092 lets win); BUG-LOCAL-088 (cast-still binding which is now defense-in-depth instead of primary); BUG-LOCAL-086 (chunking; same per-chunk seed stride means within-line identity drift may still be visible after BUG-092 -- if so, BUG-LOCAL-092a would carry latent state across chunks).

---

### BUG-LOCAL-091 [FIXED]: LTX clips frozen on last frame -- chunking + 353-frame cap parity with HuMo
- **Date:** 2026-05-04 EVENING | **Phase:** post BUG-086 LTX parity | **Bible candidate:** YES (audio/video timeline alignment, BUG-086 sister fix)
- **Symptom:** BatchLTXRender used a hardcoded ``LTX_MAX_FRAMES = 177`` (~7.08 s @ 25 fps), and ``ltx_length_for_dur`` silently capped at that value. Any non-character audio line longer than 7.08 s (typical announcer monologue: 10-15 s, music intro/outro: 8-12 s) had its tail truncated; the LTX render produced a 7-second clip while the audio kept playing, leaving the radio scene frozen on its last frame for the back half of the line. Same root cause as BUG-LOCAL-086 but for the non-character render path. Docstring at line 270 even documented it: *"For lines longer than 10.28 s, VideoComposite downstream can ping-pong-loop or freeze-frame extend"* -- the workaround was the bug.
- **Cause (multi-layer):**
  1. ``LTX_MAX_FRAMES = 177`` constant matched HuMo's pre-086 cap (intentionally, for "timing contract simplicity" per a 2026-05-01 Jeffrey directive). When BUG-086 raised HUMO_MAX_FRAMES to 353, LTX was left behind.
  2. ``ltx_length_for_dur`` clamped at the cap, silently dropping the tail frames. No widget existed to override.
  3. Comment claimed cap should be 257 (10.28 s native, proven in ComfyUI-Goofer with VAEDecodeTiled) but the constant was 177 -- comment/value drift since the original LTX setup.
- **Fix (`nodes/batch_ltx_render.py`, ~150 LOC across 6 sites):**
  - **Constant bumps** -- ``LTX_MAX_FRAMES = 353`` (8*44+1 = 14.12 s @ 25 fps, matches the post-086 HuMo cap). New ``LTX_CHUNK_FRAMES = 177`` for the chunking fallback when a line still exceeds the user-configurable cap.
  - **NEW `ltx_length_for_dur_uncapped(dur_s)`** -- 8n+1 frame snap without the LTX_MAX_FRAMES ceiling. Used by the chunking dispatch.
  - **NEW `_concat_clips_via_ffmpeg(chunk_paths, out_path, ffmpeg)`** -- ffmpeg concat-demuxer wrapper with `-c copy`. Mirrors the BUG-086 helper in batch_humo_render.py; duplicated rather than imported to keep the LTX render path self-contained.
  - **NEW `clip_length` widget** in `INPUT_TYPES.optional` -- FLOAT, default 7.0, max 14.12, step 0.04. Same UX as BatchHumoRender's BUG-086 widget. Tooltip points to BUG-LOCAL-091.
  - **`execute()` signature** now accepts `clip_length=7.0` keyword arg.
  - **Plan-build refactor** -- per-line entry now carries `chunks: list[{dur_s}]`. Lines whose `dur_s <= clip_length` get a single-chunk plan (current behaviour, unchanged). Lines exceeding `clip_length` get an N-way even split where N = `ceil(dur_s / clip_length)`. New log line: `BUG-LOCAL-091: line lXXX dur_s=YY > clip_length=ZZ -- splitting into N chunks of W.WWs each`.
  - **Per-line render loop refactor** -- iterates `entry["chunks"]`, dispatches one LTX render per chunk against the same prompt + radio bookend ref. Per-chunk `shot_seed = seed + idx*1009 + chunk_idx*7919` so chunks 1+2 of the same line don't render with identical seed (would produce visible "stutter back to start" at the join). Single-chunk lines write directly to `<line_id>.mp4`. Multi-chunk lines write to `<line_id>__chunk{NN}.mp4` part files then `_concat_clips_via_ffmpeg` stitches them and the part files are unlinked.
  - **Ledger record** -- single entry per line (not per chunk) with new `n_chunks` field for traceability. Downstream (VideoComposite) sees one mp4 per line at `<line_id>.mp4`, regardless of chunk count.
  - **`import folder_paths` made try/except** so headless pytest collection works (folder_paths is provided by the ComfyUI runtime; pytest doesn't have it). Runtime still uses it via the `_otr_paths` helpers which already have their own folder_paths fallback chain. Comment kept so Bug Bible BUG-01.02 string-content check still finds the reference.
- **NEW tests (`tests/test_batch_ltx_render.py`, 19 tests):**
  - `test_ltx_constants` -- pin LTX_FPS=25, LTX_MIN_FRAMES=9, LTX_MAX_FRAMES=353, LTX_CHUNK_FRAMES=177
  - `test_ltx_length_for_dur` -- parametrize 8 cases including 14.12s -> 353 (cap exactly), 16s+ -> 353 (capped)
  - `test_ltx_length_for_dur_always_returns_8n_plus_1` -- 9 dur values
  - `test_ltx_length_for_dur_uncapped_skips_cap` -- 30s -> 753 uncapped, 353 capped
  - `test_clip_length_widget_present` / `test_clip_length_default_is_seven` / `test_clip_length_max_respects_humo_ceiling`
  - `test_execute_signature_accepts_clip_length` -- inspect `execute()` signature
  - `test_concat_helper_*` -- 4 tests for the ffmpeg concat wrapper (empty list rejected, single-chunk copies, single-chunk no-op when path matches)
- **Known caveats / what we're not pretending:**
  - Per-chunk seed stride (7919) is a guess at preventing same-frame regression at chunk joins. If the seam is visible in test, future BUG-LOCAL-091a should carry over the last frame's latent as a continuity hint instead of relying on stride randomness.
  - LTX_MAX_FRAMES=353 at 16 GiB Blackwell is **untested in a live run**. ComfyUI-Goofer proved 257 fine; 353 is extrapolated. If the next LTX render OOMs on a 14s clip, drop the constant back to 257 (10.28s) -- the chunking dispatch handles whatever the cap is.
  - LTX chunks share the same start frame (radio bookend) so multi-chunk renders have a "snap back" at the boundary. For OTR's stylized 1940s radio scene this is acceptable; if needed, future upgrade carries last-frame latent across chunks (BUG-LOCAL-091a).
- **Verify:**
  - AST parse clean on `nodes/batch_ltx_render.py` (53704 bytes, 3982 nodes).
  - `tests/test_batch_ltx_render.py` -> 19 passed in 1.77s.
  - `tests/test_batch_ltx_render.py + test_batch_humo_render.py + test_news_history_ttl.py` -> 68 passed in 2.99s.
  - Bug Bible regression OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed.
  - Live: next LTX render with a >7s announcer line should log `BUG-LOCAL-091: line lXXX dur_s=YY > clip_length=7.0s -- splitting into N chunks` and finish with `(N chunks, BUG-LOCAL-091 chunked + concat)`.
- **Tags:** ltx, vram-ceiling, audio-video-sync, chunking, ffmpeg-concat, BUG-086-parity
- **Related:** BUG-LOCAL-086 (HuMo equivalent that this fix mirrors). BUG-LOCAL-105 (silent-clamp predecessor; chunking dispatch replaces the LTX side of that pattern but the capped `ltx_length_for_dur` still defends each individual chunk per chunk).

---

### BUG-LOCAL-090 [FIXED]: news_history.json grows unbounded -- 5-day TTL + state-dir relocation
**Update 2026-05-04 (commit follow-up):** the file was also moved out of the source repo into the per-machine state tier.

#### Part 2 — relocation to ``<output>/otr/state/`` (2026-05-04 follow-up)

The TTL fix kept the file at ``<repo>/config/news_history.json`` -- repo-local, which is wrong tier. Per-machine runtime state should live under the ComfyUI output tree where every other persistent OTR artifact (episodes/, obs/) lives. Hand-rolled paths under ``__file__/../../config/`` also tripped Bug Bible BUG-01.02 (output nodes should use ``folder_paths``).

- **NEW `nodes/_otr_paths.py::otr_state_dir()`** -- returns ``<output>/otr/state/``. Per-machine state tier; per-episode state continues to live at ``otr/episodes/<ep_id>/``.
- **`_NEWS_HISTORY_PATH`** now resolves to ``<output>/otr/state/news_history.json`` via ``otr_state_dir()``. Falls through to ``~/.otr_state/news_history.json`` defensively if ``otr_state_dir()`` is unavailable at import time (e.g. tests that monkey-patch).
- **`_NEWS_HISTORY_LEGACY_PATH`** retained pointing at the old ``<repo>/config/news_history.json`` for migration carry-forward.
- **`_load_news_history()`** -- reads new path first; if empty/missing, falls back to legacy path so the user's existing dedup window carries forward on the first post-fix run.
- **`_record_news_usage()`** -- writes only to the new path. On the first save, if the new path is empty, seeds from legacy entries so they're not silently lost. After that single save, legacy is dead-but-harmless.
- **NEW helper `_read_news_history_file(path)`** -- shared JSON-parse-with-fallback; used by both load and record so the migration semantics stay in lockstep.
- **`.gitignore`** -- added ``config/news_history.json`` so the legacy file never accidentally enters git history while it's still on disk for migration purposes.
- **NEW tests (3 added; total 10):**
  - `test_legacy_path_fallback_when_new_missing` -- legacy entries surface on first run after migration
  - `test_new_path_takes_precedence_over_legacy` -- when both files exist, new wins
  - `test_record_seeds_new_path_from_legacy_on_first_save` -- first save preserves legacy entries
  - `test_file_missing_returns_empty` + `test_corrupted_json_returns_empty` updated to monkey-patch BOTH paths so the real on-disk legacy file doesn't bleed into the test.
- **Verify (Part 2):** AST clean (565407 bytes, 39563 nodes). News history TTL suite -> 10 passed in 1.78s. Bug Bible OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed.

#### Part 1 — original 5-day TTL fix (2026-05-04 EVENING)
- **Date:** 2026-05-04 EVENING | **Phase:** soak hygiene (NewsFetcher) | **Bible candidate:** YES (best-effort dedup with TTL)
- **Symptom (live runs 2026-05-04 14:10 + earlier):** `[NewsFetcher] Filtered 43 previously-used candidate(s) via news_history (0 remaining of 43)` followed by `[NewsFetcher] All 43 candidate(s) filtered out by history -- restoring unfiltered pool so the writer still gets a real article`. Every fresh run hit 100% prior-use rate. Fallback restored the unfiltered pool so generation continued, but the dedup intent (avoid back-to-back same-headline runs) was effectively dead because the history was set-membership-only with no expiration -- a once-used URL was blocked forever. Rolling cap was 200 entries, but with 8 RSS feeds returning ~5-6 stories each (~43 unique URLs/day) the entire daily pool gets blocked within ~5 days of normal use.
- **Cause:** `nodes/story_orchestrator.py::_load_news_history()` returned `{entry["url"] for entry in data if entry.get("url")}` -- a flat set of every URL ever recorded. `_record_news_usage()` writes timestamps but `_load_news_history()` ignored them. No TTL filter, so a headline used 30 days ago still blocked the candidate pool today.
- **Fix (`nodes/story_orchestrator.py`):**
  - **NEW constant `_NEWS_HISTORY_FILTER_DAYS = 5`** -- only URLs used within the last 5 days are kept in the active filter set. Older entries stay on disk for audit (so the file remains a usage log) but no longer block the pool.
  - **`_load_news_history()` refactored** to parse `entry["timestamp"]` via `datetime.fromisoformat()` and only include entries whose timestamp is `>= now - timedelta(days=5)`. Entries with missing or malformed timestamps default to fresh (safer to filter once than to surface a same-day repeat). File-not-found, JSON-parse errors, and any other I/O failure still return an empty set (best-effort, never blocks generation).
  - **`from datetime import datetime, timedelta`** -- existing import extended for the cutoff math.
- **NEW tests (`tests/test_news_history_ttl.py`, 7 tests):**
  - `test_ttl_constant_is_five_days` -- pins the window so a silent revert is caught.
  - `test_load_filters_old_entries` -- 5-entry fixture spanning today/2d/4d/6d/10d ago; asserts only the first three survive.
  - `test_missing_timestamp_treated_as_fresh` -- missing field, empty string, malformed string all fail-to-fresh.
  - `test_missing_url_skipped` -- URL-less entries dropped silently.
  - `test_file_missing_returns_empty` -- first-run case.
  - `test_corrupted_json_returns_empty` -- invalid JSON file is not an error.
  - `test_entries_at_exactly_ttl_boundary` -- entry +1s inside the window is fresh, -1s outside is stale.
- **Verify:**
  - AST parse clean on `nodes/story_orchestrator.py` (563374 bytes, 39512 nodes).
  - `tests/test_news_history_ttl.py` -> 7 passed in 1.72s.
  - `tests/test_core.py + test_critique_dialogue_preservation.py + test_dropdown_guardrails.py + test_news_history_ttl.py` -> 183 passed in 101.89s (full coverage including modules that import story_orchestrator).
  - Bug Bible regression OTR-scoped -> 22 passed / 1 pre-existing baseline failure / 1 skipped / 2 xfailed.
  - Live: next NewsFetcher run with a 5+ day-old `news_history.json` should report a non-zero remaining pool count (e.g. `Filtered 23 previously-used candidate(s) via news_history (20 remaining of 43)` instead of the prior `0 remaining`).
- **Tags:** news-history, ttl, dedup, NewsFetcher, story-orchestrator
- **Related:** Same `signal_lost_for_nasas_tess_..._110640` soak that surfaced BUG-086. Independent issue but observed via the same console paste. The `config/news_history.json` file on disk is preserved as-is -- the TTL is read-time, not write-time, so existing entries past the window simply stop participating in the filter.

---

### BUG-LOCAL-086 [FIXED]: HuMo per-line clips frozen on last frame for half the audio (177-frame hard cap + silent clamp)
- **Date:** 2026-05-04 EVENING | **Phase:** acceptance soak (BUG-085 follow-up) | **Bible candidate:** YES (HuMo render budget + audio-video sync)
- **Symptom (live full re-render `signal_lost_for_nasas_tess_stellar_eclipses_..._110640`):** First episode-length run since BUG-085 fix completed cleanly through ScriptWriter / Bark / FLUX portraits / HuMo. ScriptWriter shipped a 177-word, 9-line script (3 cast: ANNOUNCER, JAKE, MAYA + a hallucinated CORE VOICE). HuMo rendered 7 character clips at ~10:20 each. Final composite played correctly through the announcer LTX scenes and the 7-second JAKE lines, but on every MAYA line + every JAKE line >7s the character video froze on the last rendered frame while audio kept playing. Three Media Player screenshots from Jeffrey at 0:33 / 1:01 / 1:05 all show portraits with motionless lips while the corresponding Bark audio is mid-sentence. 5 of 7 character clips affected (l003, l005, l006, l007, l008 = MAYA 13.99s, MAYA 11.31s, JAKE 12.99s, MAYA 13.99s, CORE VOICE 10.42s) — all clips with `dur_s > 7s`.
- **Cause (multi-layer):**
  1. `nodes/batch_humo_render.py::HUMO_MAX_FRAMES = 177` was the "last empirically verified value on RTX 5080 Laptop 16GB" — meaning untested above. 177 frames @ 25fps = 7.08s ceiling on per-clip render duration.
  2. `humo_length_for_dur(dur_s)` capped its return at HUMO_MAX_FRAMES, then BUG-LOCAL-105 (deep_earth_echoes 2026-04-28) added an explicit `dur_s` clamp so audio_dur_fed_to_humo never exceeded the cap. Together these meant: any character line longer than 7.08s had its tail audio silently clamped off, then HuMo only received the first 7.08s of audio, decoded a 6.88s mp4 (after warmup pad trim), and stopped. The remaining seconds of audio played against a frozen portrait in the composite.
  3. The `clip_length` workflow widget had `max=7.08`, hard-locking the user out of the higher ceiling even if VRAM allowed.
- **Fix (`nodes/batch_humo_render.py`, ~250 LOC across 7 sites):**
  - **Constant bumps** — `HUMO_MAX_FRAMES = 353` (4·88+1 = 353 frames @ 25fps = 14.12s, covers 14s Bark dialogue in a single pass). New `HUMO_CHUNK_FRAMES = 177` for the chunking fallback.
  - **NEW `humo_length_for_dur_uncapped(dur_s)`** — same 4n+1 snap as the capped helper but without the HUMO_MAX_FRAMES ceiling. Used by the chunking dispatch to decide whether to split a line.
  - **NEW `_concat_clips_via_ffmpeg(chunk_paths, out_path)`** — ffmpeg concat-demuxer wrapper with `-c copy` for stitching per-chunk mp4s into the canonical `<line_id>.mp4`. Safe because every chunk goes through `_save_clip_via_ffmpeg` with identical fps + sample rate.
  - **Plan-build refactor (BUG-LOCAL-086 chunking dispatch)** — replaces the BUG-LOCAL-105 silent-clamp at lines 1490-1512 with: if `(dur_s + pad_s) <= clip_length` → single-chunk path (current behaviour); else → split into `n_chunks = ceil(dur_s / chunk_max_dur_s)` evenly. Each chunk gets its own audio slice + warmup pad. Plan entries now carry `chunks: list[{audio, start_offset_s, dur_s}]` instead of a single `audio` dict.
  - **Phase B refactor** — Whisper audio encoding now iterates `entry["chunks"]`, encoding one `audio_emb` per chunk. Single-chunk lines unchanged in behaviour; multi-chunk lines pay N × Whisper cost (cheap; <1s per chunk).
  - **Phase C render-loop refactor** — per-line render now iterates chunks, dispatches HuMo once per chunk with that chunk's `audio_emb` and the same portrait `ref_image`, saves each chunk to either `<line_id>.mp4` (single-chunk) or `<line_id>__chunk{NN}.mp4` (multi-chunk), then ffmpeg-concats multi-chunk parts into `<line_id>.mp4` and deletes the part files. Per-chunk shot_seed = `seed + idx*1009 + chunk_idx*7919` so chunks 1+2 of the same line don't render with identical seed (would produce visible "stutter back to start" at the chunk boundary). Single ledger record per line regardless of chunk count; `mp4_frames` / `mp4_dur_s` / `humo_render_ms` are sums across chunks; `audio_fed_to_humo_dur_s` accounts for N pads (one per chunk). New `n_chunks` field added to ledger clip records for traceability.
  - **Widget bump** — `clip_length` max raised from 7.08 to 14.12 (default unchanged at 7.0). Power users can opt into single-pass for typical Bark dialogue. Lines longer than `clip_length` still chunk regardless.
- **Test updates (`tests/test_batch_humo_render.py`):**
  - Updated `test_humo_length_for_dur` parametrize cases for the new cap (8s → 201, 9s → 225, 14.12s → 353, 16s+ → 353 capped).
  - Updated `test_humo_constants` (HUMO_MAX_FRAMES = 353; new HUMO_CHUNK_FRAMES = 177 assertion).
  - Updated `test_clip_length_max_respects_humo_ceiling` (max = 14.12).
  - NEW `test_humo_length_for_dur_uncapped_skips_cap` — pins that the chunking-dispatch helper bypasses the cap (30s → 753, capped helper would return 353).
  - Existing `test_humo_length_for_dur_always_returns_4n_plus_1` extended to include 14s in the parametrize set.
- **Known caveats / what we're not pretending:**
  - `HUMO_MAX_FRAMES = 353` at 16 GiB Blackwell is **untested in a live run**. If the next soak OOMs at single-pass for a 14s clip, drop the constant back to 257 (10.28s) or 177 (7.08s) — the chunking dispatch handles whatever the cap is. Tracked as BUG-LOCAL-086a if it surfaces.
  - Per-chunk shot_seed stride (7919) is a guess at preventing same-frame regression at chunk joins. If the seam is visible in test, bump the stride or carry over the last chunk's final-frame latent as a continuity hint (BUG-LOCAL-086b future).
  - Whisper feeding silence into a chunk could still produce no-lip-motion (Jeffrey's bottom-left screenshot showed this at the l002→l003 boundary). Not in BUG-086 scope; logged as BUG-LOCAL-090 candidate (Bark line lead-in silence handling).
- **Verify:**
  - AST parse clean on `nodes/batch_humo_render.py` (110206 bytes, 8846 nodes).
  - `tests/test_batch_humo_render.py` + `tests/test_humo_warmup_pad.py` + `tests/test_dropdown_guardrails.py` → 107 passed in 108.91s.
  - `tests/test_core.py` → 108 passed in 4.46s.
  - Bug Bible regression scoped to OTR pack → 22 passed / 1 pre-existing failure (otr_save_copy.py, batch_flux_portrait_render.py missing folder_paths; unrelated to BUG-086) / 1 skipped / 2 xfailed.
  - Live: next soak should show `[BatchHumoRender] BUG-LOCAL-086: line lXXX dur_s=YY > clip_length=7.0s -- splitting into N chunks of Z.ZZs each` for any line >7s, and `[BatchHumoRender] lXXX done in M ms (N chunks, BUG-LOCAL-086 chunked + concat)`. Composite mp4 should show lipsync continuing through the FULL audio duration of every character line; no frozen-tail artefacts.
- **Tags:** humo, vram-ceiling, audio-video-sync, chunking, ffmpeg-concat, wan-2.1, blackwell, BUG-LOCAL-105-supersession
- **Related:** BUG-LOCAL-105 (silent-clamp predecessor; chunking dispatch replaces the clamp but the capped `humo_length_for_dur` still defends each individual chunk, preserving 105's safety property per chunk). BUG-LOCAL-102 (warmup pad — applied per chunk in 086, not just first chunk). BUG-LOCAL-094 (per-line timing estimate; unchanged). Pending: BUG-LOCAL-087 (title lost between ScriptWriter and SignalLostVideo), BUG-LOCAL-088 (CORE VOICE hallucinated cast member), BUG-LOCAL-089 (Director phase produces unparseable output on Gemma-4-E2B). All three observed in the same `signal_lost_for_nasas_tess_..._110640` run that surfaced 086.

---

### BUG-LOCAL-085 [FIXED]: NF4 silently failing because HF_HOME not in ComfyUI process env
- **Date:** 2026-05-04 MORNING | **Phase:** acceptance (BUG-LOCAL-084 follow-up) | **Bible candidate:** YES (Windows/Electron env-inheritance footgun)
- **Symptom (live full re-render attempt 2026-05-03 23:47, post BUG-084):** ComfyUI restarted clean, BUG-084 fixes loaded, full workflow queued. ScriptWriter started Mistral-Nemo load. Crashed at SDPA prefill with `torch.OutOfMemoryError: Currently allocated 24.00 GiB / Device limit 15.92 GiB`. 24 GiB matches Mistral-Nemo 12B at fp16 exactly — meaning NF4 quantization did not actually apply despite the runtime log printing `[StoryOrchestrator] Enabling 4-bit quantization (NF4)`.
- **Cause (multi-layer):**
  1. ComfyUI Desktop's Electron parent process did not inherit `HF_HOME` from `HKCU\Environment`. PowerShell confirmed: `HF_HOME (User) = C:\ComfyUI-Models\huggingface`, `HF_HOME (Process) = (empty)`. Per-user env vars are inherited by processes started from Explorer but not always by Electron-spawned children.
  2. `nodes/story_orchestrator.py::_load_llm` resolved cache via `os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))` → fell through to `~/.cache/huggingface` because env var was missing.
  3. With `cache_dir` wrong AND `local_files_only=True` AND Mistral-Nemo's sharded-safetensors layout on Windows, transformers' Hub-resolution layer misresolved the model location. Instead of erroring out, it silently fell back to a partial-fp16 load path. `quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...)` was passed but never applied.
  4. fp16 12B Mistral-Nemo = 24 GiB → device_map={"": 0} forced 100% GPU → OOM when KV cache built during prefill of the first generate() call.
- **Verified in isolation (`scripts/check_nf4_load.py`):** When the snapshot directory path (`C:\ComfyUI-Models\huggingface\hub\models--mistralai--Mistral-Nemo-Instruct-2407\snapshots\<sha>\`) is passed directly to the loader, the model loads at **7.79 GiB allocated**, **280/281 Linear modules quantized to 4-bit NF4**, generation works ("Once upon a time, there was a little girl named Lily"). Confirms the bug is the Hub-resolution path, not the quantization config.
- **Fix:**
  - **NEW `nodes/_otr_hf_env.py`** — `ensure_hf_home()` reads `HF_HOME` from `HKCU\Environment` via `winreg`, exports it to `os.environ['HF_HOME']` and `os.environ['HF_HUB_CACHE']` so downstream HF tooling picks it up automatically. `resolve_snapshot_dir(model_id, hf_home=None)` returns the absolute snapshot directory path under the canonical cache (`<hf_home>/hub/models--<org>--<name>/snapshots/<sha>/`). Both functions idempotent + cache-safe.
  - **`nodes/story_orchestrator.py::_load_llm`** — calls `ensure_hf_home()` at start; calls `resolve_snapshot_dir(model_id)` to get the canonical snapshot path; passes the snapshot path (not the `model_id`) to `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM` loaders. Bypasses transformers' Hub-resolution layer entirely. Falls through to the legacy `model_id` + `cache_dir` path only if snapshot resolution returns None (model not cached).
- **Disk cleanup landed alongside (not committed; outside repo):**
  - Deleted 14.23 GB of `.incomplete` partial Mistral-Nemo blobs at `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub\` (stray duplicate cache from before HF_HOME migration; never read by ComfyUI).
  - Aligned `C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\extra_model_paths.yaml` (install-dir copy) with the Roaming canonical so any process reading either YAML resolves to `C:\ComfyUI-Models` paths consistently.
- **Verify:**
  - Standalone load check: `python scripts/check_nf4_load.py` reports `PASS: NF4 working (7.79 GiB; expected ~6 GiB)` with all 280 Linear modules quantized.
  - In ComfyUI: next full run should log `[StoryOrchestrator] HF_HOME resolved -> C:\ComfyUI-Models\huggingface` followed by `[OTR_HF_ENV] snapshot resolved mistralai/Mistral-Nemo-Instruct-2407 -> <abs_snapshot_path>` followed by `LLM model loaded from canonical snapshot (no HTTP checks)`. ScriptWriter VRAM should peak at ~7-8 GB instead of crashing at 24 GB.
  - AST parse clean on both files; Bug Bible regression 22 passed / 1 pre-existing failure / 1 skipped / 2 xfailed.
- **Tags:** hf-cache, electron-env-inheritance, nf4, bitsandbytes, hub-resolution, sharded-safetensors, windows-symlinks, vram-ceiling, ScriptWriter
- **Related:** BUG-LOCAL-084 (composite gap-fill + duration contract — shipped 7f2d03f, not yet verified in a full live run because BUG-085 blocked it). BUG-085 fix is required before BUG-084 can be exercised end-to-end via the full workflow. Smoke harness path is unaffected (no LLM load). Commit `56cf493` on `v2.0-alpha`.

---

### BUG-LOCAL-084 [FIXED]: composite missed gap-fill + LTX ledger stamp incomplete
- **Date:** 2026-05-04 LATE NIGHT | **Phase:** acceptance (BUG-LOCAL-031 Track 1 follow-up) | **Bible candidate:** YES (audio/video timeline alignment)
- **Symptom (live composite output `signal_lost_skindeep_microneedle_..._222516.mp4`):** Visual sync broken end-to-end. At video time 0:12 viewer sees apprentice face with no lip movement (audio is l001 LTX announcer, should be radio scene). At video time 0:27 viewer sees foreman face with lips moving (audio is l002 apprentice voice). Cumulative drift ~10s by end of episode; final mp4 was ~35s short of master audio (`-shortest` truncated trailing audio).
- **Cause:** VideoComposite per_clip_mux concatenated the 6 per-line clips back-to-back at t=0 with no gap-fill. Audio (master mix from procgen.mp4) starts l001 at 9.5s, has 0.6s gaps between adjacent lines, ends at 87.2s vs procgen 94.75s. Without gap-fill segments the video is structurally shorter than the audio, audio leads video by the cumulative gap duration, and `-shortest` chops the tail.
- **Fix (4 sites in 2 files):**
  - **Fix 1 (`batch_ltx_render.py`):** stamp `start_s` on each `ledger.clips[]` entry from matching `ledger.lines[]` by line_id; ffprobe rendered file for real `dur_s` instead of audio target (BUG-LOCAL-033 lie); preserve audio target as `audio_target_dur_s` for audit.
  - **Fix 2 (`video_composite.py`):** confirmed already wired from BUG-031 Track 1 — per-clip ffprobe + duration_gap_s + extend_tail_s/truncate_to_s pass-through into `_layered_per_clip_silent`.
  - **Fix 3 (`video_composite.py`):** NEW gap-fill pass after `timeline.sort()`. Walks sorted timeline, for any gap > 0.1s inserts static-radio segment of exact gap length using existing `_render_static_radio_segment` helper. Trailing tail-fill from last clip end to ffprobe(procgen) episode_dur. Logs `BUG-084 gap-fill: inserted N segments, total Xs coverage`.
  - **Fix 4 (`video_composite.py`):** NEW duration-contract assertion before final mux. ffprobe `silent_combined` and procgen, compare within 40ms tol. If audio overruns video, tail-pad silent_combined with `tpad clone-frame` so `-shortest` truncates inaudible video, not audio. Belt-and-suspenders.
- **Verify:** AST parse clean both files; Bug Bible regression 22 passed (1 pre-existing failure unchanged). End-to-end live verification BLOCKED on BUG-LOCAL-085 (NF4 OOM); smoke workflow exercises BUG-084 but was also blocked overnight by BUG-082 + BUG-083 fixes that landed earlier.
- **Tags:** composite, gap-fill, duration-contract, c7-audio-byte-identity, bug-031-followup, ledger-clips-stamp
- **Related:** BUG-LOCAL-031 (RTXUpscale range + PostBlend duration). Commit `7f2d03f` on `v2.0-alpha`.

---

### BUG-LOCAL-083 [FIXED]: probe_duration_s kwarg mismatch (ffmpeg vs ffprobe)
- **Date:** 2026-05-03 LATE NIGHT | **Phase:** smoke harness | **Bible candidate:** YES (kwarg signature drift)
- **Symptom:** Smoke workflow crashed at VideoComposite per_clip_mux with `RuntimeError: strict_c7=True and master_mix_per_clip_mux failed. Reason: probe_duration_s() got an unexpected keyword argument 'ffmpeg'`. Caught by smoke harness on first run after BUG-082 landed.
- **Cause:** Two call sites in `video_composite.py` (BUG-031 Track 1 per-clip duration matching, lines 1033 and 1135) passed `ffmpeg=ffprobe` to `_otr_probe.probe_duration_s()`. The function signature is `def probe_duration_s(path, *, ffprobe="ffprobe")` — the kwarg is named `ffprobe`, not `ffmpeg`. TypeError caught by the strict_c7 master_mix_per_clip_mux exception handler, which then refused to fall back to humo_concat (correctly — 3x AAC re-encodes break C7).
- **Fix:** rename kwarg `ffmpeg` → `ffprobe` to match the actual function signature at both call sites.
- **Verify:** AST parse clean; smoke harness composite stage now completes in 4.4s with `tail-pad 0.500s (BUG-128) + 3.040s sync (BUG-031) on surviving last clip l006`.
- **Tags:** kwarg-signature, smoke-harness, bug-031-followup
- **Related:** Commit `e601ee8` on `v2.0-alpha`.

---

### BUG-LOCAL-082 [FIXED]: VideoComposite missing BUG-118 underscore-mismatch fallback
- **Date:** 2026-05-03 LATE NIGHT | **Phase:** acceptance | **Bible candidate:** YES (writer/reader filename convention drift)
- **Symptom:** Live full run died at 23:10:25 with `RuntimeError: VideoComposite: derived ledger from .mp4 not found: ...drug__20260503_222516_ledger.json` (note double underscore). LTX completed cleanly just before this; composite was the only stage that crashed.
- **Cause:** SignalLostVideo writes the procgen .mp4 with a double underscore before the timestamp (`signal_lost_..._drug__20260503_222516.mp4`); the ledger writer uses single underscore (`signal_lost_..._drug_20260503_222516_ledger.json`). VideoComposite's `_load_ledger_with_path` derived the ledger filename via naive `replace('.mp4', '_ledger.json')` → got `...drug__20260503_222516_ledger.json` (double underscore) which doesn't exist on disk. BatchLTXRender already had this fallback in place; VideoComposite was the orphan.
- **Fix:** ported the BUG-LOCAL-118 underscore-collapse fallback from BatchLTXRender to VideoComposite. When the primary derivation misses AND `__` appears in the stem, also try the single-underscore variant before raising.
- **Verify:** AST parse clean; smoke harness with broken-cache episode loads ledger correctly via fallback path with log `BUG-LOCAL-118 underscore-mismatch fallback`.
- **Tags:** writer-reader-drift, filename-convention, mp4-stem, bug-118-port
- **Related:** Commit `b34d272` on `v2.0-alpha`.

---

### BUG-LOCAL-081 [FIXED]: portrait node wired to wrong source — Node 59 never produced portraits
- **Date:** 2026-05-03 LATE EVENING | **Phase:** acceptance (BUG-LOCAL-078 follow-up) | **Bible candidate:** YES (workflow-wiring footgun, silent failure)
- **Symptom (live run `signal_lost_the_creepy_feeling_in_old_buildings_migh_20260503_215919`):** Episode workspace had `audio/`, `stills/`, `videos/` but no `portraits/` subdirectory. Ledger had 3 cast members (c01=ANNOUNCER, c02=JAX, c03=KAI), all with `portrait_path` empty. `otr_runtime.log` had ZERO log lines mentioning `BatchFluxPortraitRender` across the whole 49,154-line / 3.7 MB run. Module import + `INPUT_TYPES` registration both verified clean.
- **Cause (two distinct workflow-JSON bugs in `workflows/otr_scifi_16gb_full.json`):**
  1. **Bogus `ledger_json` source.** Link 100 wired Node 12 (`OTR_SignalLostVideo`) `video_path` output (a `.mp4` filesystem path) into Node 59's `ledger_json` input. The portrait node's `_load_ledger` tried `json.loads()` on the `.mp4` path, hit `JSONDecodeError`, fell into `except Exception: return (None, None)`, then `execute()` raised `RuntimeError("cannot load ledger from <path.mp4>")`. The error went to the ComfyUI executor (not OTR logger), so `otr_runtime.log` stayed silent.
  2. **Wrong execution position in DAG.** Because Node 12 was an upstream dependency of Node 59, ComfyUI scheduled Node 59 to run AT THE END of the workflow — long after HuMo (Node 51) had already executed without portraits and fallen through to tier-4 env-still stopgap. Even if (1) were fixed, (2) alone made the portrait pass useless: HuMo would never see the portraits that hadn't been rendered yet.
- **Fix (workflow JSON only — no code changes):**
  - Drop link 100 entirely (`Node 12.video_path → Node 59.ledger_json`); set Node 59's `ledger_json` widget to empty string so `_load_ledger` falls through to `_OTRL.in_flight_ledger_path()` auto-pickup.
  - Re-route link 45 from `(Node 23 → Node 24)` to `(Node 59 → Node 24)`. New chain: `BatchFluxRender (env stills, 23) → BatchFluxPortraitRender (59) → UnloadAll (24) → BatchHumoRender (51)`. Portraits now render BEFORE HuMo while FLUX is still loaded in VRAM, then UnloadAll dumps FLUX, then HuMo picks up the portraits via the in-flight ledger.
- **Verify:**
  - JSON validates: `nodes=32 links=57`, no orphan/dangling refs.
  - Node 59 inputs: `ledger_json link=null` (auto-pickup), `flux_done_gate link=101` (waits on Node 23). Outputs: `portrait_batch links=[45]` (gates UnloadAll → HuMo).
  - Widget count went from 10 → 11 (prepended `""` for `ledger_json`).
  - Portrait module import + `INPUT_TYPES` clean.
  - Real-run acceptance (pending): next queue should produce `<ep>/portraits/c02_portrait.png` + `c03_portrait.png` (announcer skipped per `skip_announcer=True`), and HuMo's per-line `_find_portrait` should hit tier 1 instead of tier 4 — visible in HuMo log lines as `portrait_path: <path>` instead of `falling back to env still`.
- **Tags:** workflow-wiring, link-100, link-45, silent-failure, bug-078-followup, portraits, comfyui-dag-ordering, c7-untouched
- **Related:** BUG-LOCAL-078 (the portrait node itself, shipped EVENING with correct internal logic). The wiring slip happened during workflow JSON edit when Node 59 was added to `otr_scifi_16gb_full.json` — the `ledger_json` socket was wired to the only nearby STRING source on the canvas (Node 12's `video_path`) rather than left empty for auto-pickup, and the portrait node was inserted BELOW Node 12 in the DAG instead of between Node 23 and Node 24. Bible candidate because the silent-failure mode (RuntimeError invisible in OTR log + dependency-chain inversion) is the kind of trap any future graph edit could fall into. Commit `413ef3a` on `v2.0-alpha`.

---

### BUG-LOCAL-031 [FIXED]: HuMo + LTX visual content destroyed by RTXUpscale (range normalization bug) + duration overrun in PostUpscaleProcgenBlend
- **Date:** 2026-05-03 EVENING (post-soak diagnosis) | **Phase:** acceptance (BUG-LOCAL-030 wave) | **Bible candidate:** YES (severe, wave-blocker)
- **Symptom (live soak run `signal_lost_what_a_decade_of_gene_therapy_research_f_20260503_173957`):**
  - User saw TWO outputs in OBS folder (only ONE expected): `<ep>.mp4` (1.63 MB, "audio + all black") and `<ep>_procgen_blended.mp4` (14.48 MB, "procgen scanlines visible, NO HuMo/LTX content visible underneath").
  - Per-stage ffprobe nailed the failure to RTXUpscale: composite output 1472x832 / 50.36s / **1544 kbps** / 672 KB sample frame (real content). RTXUpscale output 1920x1080 / 50.36s / **96 kbps** / 56 KB sample frame (solid black). Same dims, same frame count -- only the visual content disappeared. Post-blend overran to 113.92s vs source 50.36s with audio at 50.34s (50s of audio over 113s of video).
- **Cause (two distinct bugs):**
  1. **Bug 1 -- range normalization mismatch in `nodes/rtx_upscale.py::_chunked_upscale`:** NVIDIA's `nvvfx.VideoSuperRes` expects input in **0.0-1.0 float** range (matching ComfyUI IMAGE convention) and produces output in **0.0-1.0 float** range. The OTR node read raw RGB24 bytes from ffmpeg (uint8 0-255), did `.float()` which keeps the values numerically 0-255, and fed nvvfx those out-of-distribution values. nvvfx internally clamped/saturated them, producing garbage near 1.0. The output was then `clamp(0.0, 255.0).byte()`'d -- which is a no-op for 0-1 values, then `.byte()` truncated 0.95 -> 0 and 1.0 -> 1, producing essentially solid black (every pixel value 0 or 1 out of 255). H.264 compresses solid color to nothing (96 kbps).
  2. **Bug 2 -- PostUpscaleProcgenBlend duration overrun:** the previous round-robin (Gemini, 2026-05-03 EVENING) said don't use `-shortest`. The advice was about the muxer-level `-shortest` flag (which IS unsafe -- truncates audio if procgen ends first, breaking C7). Today's round-robin (Gemini, again) self-corrected: filter-level `shortest=1` INSIDE the blend filter is C7-safe because audio is mapped separately via `-c:a copy`. Without it, the blend filter outputs the LONGER input duration -- procgen 113.92s wins over source 50.36s.
- **Fix:**
  - **`nodes/rtx_upscale.py::_chunked_upscale`** -- normalize input AND denormalize output:
    - Input: `gpu_in = ... .float().contiguous() / 255.0` (uint8 0..255 -> float32 0..1, what nvvfx actually expects).
    - Output: detect range (`gpu_out.max() <= 1.5`) and multiply by 255 before the `.byte()` cast (forward-compat: future nvvfx versions that change output convention won't break).
  - **`nodes/otr_post_upscale_procgen_blend.py::_build_blend_cmd`** -- append `:shortest=1` to the blend filter expression: `blend=all_mode={blend_mode}:all_opacity={blend_opacity:.3f}:shortest=1[v]`. Filter-level flag only clamps video output; audio mapped via `-c:a copy` from source is untouched (C7 holds).
- **NEW DIAGNOSTIC TOOLING (this commit):**
  - **`scripts/smoke_downstream_from_assets.py`** -- skip the 90-min upstream pipeline and exercise ONLY the downstream chain (Composite -> RTXUpscale -> PostUpscaleProcgenBlend) on pre-rendered assets from a completed run. Iteration loop drops from ~90 min to ~70 sec. Saves per-stage ffprobe + sample frame for visual inspection. Optional `--diagnostic-dump` flag propagates a dump dir into RTXUpscale.
  - **`nodes/rtx_upscale.py`** -- `diagnostic_dump_dir` optional kwarg on `RTXUpscale.execute()` and `_chunked_upscale()`. When set, dumps three PNGs per chunk (input_uint8, post_nvvfx_float_xN, post_clamp_byte) plus a `chunk_stats.txt` with per-chunk min/max/mean for input + nvvfx output + post-clamp byte. The stats file alone localizes any future range-mismatch / silent-zero / dimension-error bug in seconds. No-op when disabled (production paths unchanged).
- **Verify:**
  - AST parse on touched files: green.
  - **Smoke harness end-to-end:** RTXUpscale output 1.63 MB / 96 kbps / 56 KB frame (BLACK) -> 17.47 MB / 2734 kbps / 1056 KB frame (REAL CONTENT). Post-blend output 14.48 MB / 113.92s (overrun) -> 19.70 MB / 50.36s (clamped). Visual inspection of `frame_post_blend_out.png` confirms metallic corridor + CRT panel + procgen scanlines overlay = SIGNAL LOST visual signature working as designed.
  - Diagnostic dump confirmed root cause: every chunk reported `nvvfx(min=0.0000, max=1.0000)` after the broken (no-input-divide) variant; with the input-divide fix, ratios look right and visual inspection is clean.
  - **Real-run acceptance (pending):** queue a fresh episode after restart. Expect `obs/<ep>.mp4` to be ~10+ MB (was 1.63 MB), `obs/<ep>_procgen_blended.mp4` to be ~20+ MB at exactly the source duration (was 14.48 MB at 113s overrun), and the visible video to show HuMo character clips + LTX broadcast units + procgen scanlines composited per the Phase A + B design.
- **Tags:** rtx-vsr, nvvfx, range-normalization, blend-shortest, c7-safe, video-pipeline, post-bug-030, smoke-harness
- **Related:** BUG-LOCAL-030 wave (parent — Phase A composite + Phase B procgen blend); the previous "drop -shortest" fix (commit `a486fd1`) was correct in intent but missed that filter-level `shortest=1` is C7-safe while muxer-level `-shortest` is not. This commit corrects the over-correction. Round-robin: ChatGPT (gpt-5.5) + Gemini (gemini-3.1-pro-preview) both converged on the diagnosis from the bitrate-collapse signal alone. The diagnostic dump tool is the larger contribution -- it makes future RTXUpscale-style bugs localizable in one smoke cycle.

---

### BUG-LOCAL-030-LONGFORM-HARDENING [FIXED]: Composite + blend chain not soaked at >5 min episode length — DRAM canary, gc/empty_cache, ffmpeg thread caps, intermediate cleanup
- **Date:** 2026-05-03 EVENING (post audit-completion) | **Phase:** preventative hardening | **Bible candidate:** YES (long-form scaling)
- **Symptom:** preventative — no soak failure. Round-robin risk-#10 review (`docs/2026-05-03-soak-risk-10-dram-ceiling-longform__*.md`) flagged that the BUG-LOCAL-030 Phase A + B composite chain has zero soak data above ~3 min audio. Real risk surface for a >5 min episode: ~100 per-line layered intermediates on disk simultaneously (1.5-4.5 GB transient bloat), ffmpeg `blend` filter at 1920x1080 buffering frames from BOTH inputs, PyTorch holding RAM/VRAM across phase boundaries, "Too many packets buffered" error on long-form blends.
- **Cause:** organic growth — Phase A + B were designed for the typical short-act soak length, not stress-tested for the longer episodes Jeffrey is now ready to queue. Gemini caught this via correct ComfyUI float32-tensor math (12 bytes/pixel, not RGB24's 3) which would be catastrophic IF RTXUpscale loaded the full upscale into memory; verification of `nodes/rtx_upscale.py` confirmed it pipes raw RGB24 in/out of ffmpeg via `subprocess.Popen` chunks (RETURN_TYPES = STRING, not IMAGE) — so the 110 GB OOM scenario does NOT apply. The five remaining cheap-win hardening recommendations DO apply.
- **Fix (single commit, 5 hardening sites + new helper module + tests):**
  - **NEW `nodes/_otr_memory.py`** — shared DRAM/VRAM hygiene helpers. `phase_gc(label)` runs `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` (best-effort, never raises, idempotent). `dram_canary(min_free_gb=6.0, label)` runs `psutil.virtual_memory().available` check; degrades OPEN (returns `(True, reason)`) when psutil missing or syscall fails so the canary never blocks a render that would otherwise complete. Returns `(False, reason)` ONLY when psutil successfully reports below threshold. Default 6.0 GB per Gemini round-robin.
  - **`nodes/video_composite.py::_render_master_mix_per_clip_mux_mode`** — at function entry, calls `phase_gc("VideoComposite/per_clip_mux entry")` + `dram_canary` and appends warning to report if canary trips. After concat-demuxer succeeds (`silent_combined.mp4` written), immediately `unlink(missing_ok=True)` every `pillarboxed[]` intermediate — saves 0.5-1.5 GB transient disk on a 100-clip episode. Best-effort; cleanup failures logged but never raised.
  - **`nodes/otr_post_upscale_procgen_blend.py::_build_blend_cmd`** — appended `-filter_complex_threads 2 -filter_threads 2 -threads 4 -max_muxing_queue_size 1024` to the ffmpeg cmd. Caps thread fanout (prevents thread×framebuffer DRAM multiplication on long-form blends) + raises mux queue (guards against "Too many packets buffered for output stream" failure mode).
  - **`nodes/otr_post_upscale_procgen_blend.py::PostUpscaleProcgenBlend.blend`** — same `phase_gc` + `dram_canary` at entry as VideoComposite. Phase barrier handoff from RTXUpscale releases any RAM/VRAM PyTorch may still be holding before the blend pass starts buffering 1920x1080 frames from both inputs.
  - **`tests/test_otr_memory.py`** (NEW, 7 tests) — `phase_gc` never-raises (including with torch missing); `dram_canary` default threshold = 6.0 GB; degrades open when psutil missing OR syscall fails; returns False below threshold; returns True above threshold.
  - **`tests/test_post_upscale_procgen_blend.py`** — added `test_blend_cmd_includes_longform_hardening_flags` verifying all four new flags (`-max_muxing_queue_size 1024`, `-filter_complex_threads 2`, `-filter_threads 2`, `-threads 4`) appear in the generated cmd with correct values.
- **Verify:**
  - AST parse on all 5 touched files: green.
  - Targeted regression: **76 passed in 3.44s** (`test_otr_memory + test_post_upscale_procgen_blend + test_video_composite_layered + test_video_composite_per_clip_mux + test_per_line_audio_meta`). Bug Bible regression: 24 passed / 1 skipped / 1 xfailed in 1.39s.
  - **Real-run acceptance (pending — long-form soak):** queue an episode at >5 min audio length. Expect (a) `per_clip_mux: reclaimed N pillarbox intermediate(s)` line in VideoComposite report (where N matches the per-line clip count), (b) `PostUpscaleProcgenBlend: DRAM canary WARNING -- ...` line in report ONLY if available DRAM dropped below 6 GB pre-blend, (c) ffmpeg blend pass completes WITHOUT "Too many packets buffered" error even with ~100+ clips on the timeline, (d) C7 audio byte-identity preserved end-to-end (audio path is `-c:a copy` everywhere; new ffmpeg flags only affect video processing).
- **Tags:** dram-ceiling, long-form, ffmpeg-flags, gc-empty_cache, psutil-canary, post-bug-030, round-robin-risk-10
- **Related:** BUG-LOCAL-030 (parent); BUG-LOCAL-030-AUDIT-COMPLETION (sibling — also a defensive-hardening pass on the same chain). Round-robin synthesis: ChatGPT correctly identified RTXUpscale as the primary risk surface but mis-calculated tensor sizes (used RGB24 3 bytes/pixel instead of ComfyUI float32 12 bytes/pixel) and recommended `-shortest` (would violate C7); Gemini caught both errors and provided the four ffmpeg flags + psutil canary + intermediate cleanup recommendations adopted here. Verification of RTXUpscale source confirmed it's a CLI wrapper (chunked ffmpeg pipes), not an IMAGE-tensor consumer, so Gemini's 110 GB OOM hypothetical does NOT apply.

---

### BUG-LOCAL-030-AUDIT-COMPLETION [FIXED]: Per-line audio render metadata not stamped to ledger across 4 audio engines (forensic gap)
- **Date:** 2026-05-03 EVENING (post-final_video_path stamp) | **Phase:** acceptance hardening | **Bible candidate:** YES (forensic provenance)
- **Symptom (from artifacts-grid audit, no soak failure — preventative gap closure):** ledger only knew which engine produced what audio for a single field (Bark’s `bark_render_ms`). Other engines: KokoroAnnouncer wrote ZERO ledger fields; MusicGenTheme stamped `wav_path + dur_s` only; BatchAudioGen stamped `wav_path + dur_s` only. Cannot answer “which engine + voice + render time + sample hash produced this row?” without re-reading the wav from disk + cross-referencing logs.
- **Cause:** historical organic growth — Bark got the BUG-LOCAL-101 forensic block, the other three audio engines never got the same treatment. No common helper existed to stamp the canonical `tts_engine / voice_preset / render_ms / generated_dur_s / audio_sample_hash` bundle.
- **Fix (single commit, 4 nodes + 2 helpers + 1 test file):**
  - `nodes/_otr_ledger.py`: two new public helpers — `compute_audio_sample_hash(arr_or_bytes, n_bytes=1024) -> str` (8-char SHA256 hex of leading bytes; tripwires sample-rate / channel / pad drift; best-effort ""-on-failure), and `stamp_per_line_audio_meta(ledger, line_id, *, tts_engine, voice_preset="", render_ms=0, generated_dur_s=0.0, audio_sample_hash="") -> bool` (wraps `patch_line_fields`, skips empty/zero values so partial bundles don’t clobber pre-existing fields).
  - `nodes/batch_bark_generator.py`: extends existing per-line stash + write-back. New row fields `tts_engine="bark"`, `voice_preset` (from `preset` in scope), `render_ms` (mirrors `bark_render_ms`), `generated_dur_s` (mirrors `bark_wav_dur_s`), `audio_sample_hash` (computed from per-line `audio_np`).
  - `nodes/musicgen_theme.py`: tracks per-cue `_render_ms` around the `model.generate()` call + per-cue `_audio_sample_hash`. Existing `wav_path + dur_s` write-back extended with `tts_engine="musicgen"`, `render_ms`, `generated_dur_s`, `audio_sample_hash` on each `ledger.music[]` row. Generation prompt was already populated by LLMDirector, so this closes the loop on the render-result side.
  - `nodes/batch_audiogen_generator.py`: same pattern — per-sfx `_render_ms` + `_audio_sample_hash` stash, write-back extended with `tts_engine="audiogen"`, `render_ms`, `generated_dur_s`, `audio_sample_hash` on each `ledger.sfx[]` row.
  - `nodes/kokoro_announcer.py` (was ZERO ledger touches before this commit): per-line `_render_ms` + `_audio_sample_hash` tracked in a `per_line_meta[]` parallel list. New post-loop write-back uses `_OTRL.in_flight_ledger_path()` singleton (same Phase G discovery as BatchBark) + text-match against `ledger.lines[]` (same first-unmatched-wins strategy as BUG-LOCAL-096). Stamps `tts_engine="kokoro"`, `voice_preset=<chosen voice_id>`, `render_ms`, `generated_dur_s`, `audio_sample_hash` on every matching row. Best-effort: any I/O failure is logged, never raised.
- **Verify:**
  - AST parse on all 5 touched node files: green.
  - **New `tests/test_per_line_audio_meta.py` (10 tests, all passing):** `compute_audio_sample_hash` determinism / divergence / numpy-array support / empty-on-unhashable / leading-bytes-only contract; `stamp_per_line_audio_meta` full-bundle stamp / skip-empty / unknown-line returns False / partial bundle (engine + render_ms + hash only) / never-raises on bad ledger.
  - Cumulative regression: **200 passed, 1 skipped in 116.85s** (`test_dropdown_guardrails + test_core + test_audio_byte_identical + test_per_line_audio_meta + test_meta_paths + test_post_upscale_procgen_blend`). Bug Bible regression: 24 passed / 1 skipped / 1 xfailed in 1.38s.
  - **Real-run acceptance (pending):** queue an episode after restart. Expect the per-episode `<ep>_ledger.json` to show on every dialogue row `"tts_engine": "bark"|"kokoro"`, `"voice_preset": "<preset>"`, `"render_ms": <int>`, `"generated_dur_s": <float>`, `"audio_sample_hash": "<8hex>"`; on every `music[]` row `"tts_engine": "musicgen"` + render fields; on every `sfx[]` row `"tts_engine": "audiogen"` + render fields. C7 byte-identity unaffected (audio bytes themselves are NOT modified — this is metadata-only).
- **Tags:** ledger-schema, forensic-metadata, bark, kokoro, musicgen, audiogen, audit-completion, post-bug-030
- **Related:** BUG-LOCAL-030 (parent — final_video_path stamp closed the video forensic gap; this closes the audio forensic gap with the same audit-completion theme); BUG-LOCAL-018 (l3-2026-05-02 schema bump that established `meta.paths` precedent for additive ledger fields). Helpers added in this fix are usable by any future audio engine.

---

### BUG-LOCAL-030 [FIXED]: All-black final video — HuMo portrait squeezed into landscape canvas + per-clip-mux mode bypasses procgen visual layer
- **Date:** 2026-05-03 EVENING | **Phase:** acceptance (post-029 follow-up surfaced by 14:08 soak) | **Bible candidate:** YES (architecture pivot)
- **Symptom (from soak `signal_lost_static_echo_20260503_140824`, ffprobe of per-line clips):**
  - HuMo per-line clips render at native **480x832 PORTRAIT** (l003.mp4 is 480x832; l001.mp4 LTX is 832x480 landscape — confirmed via ffprobe).
  - Pillarbox formula `scale=-2:480:force_original_aspect_ratio=decrease, pad=832:480:...:color=black` applied to 480x832 portrait input scales width to ~276px and pads with **278 pixels of pure black on each side** of the 832-wide canvas. Result: HuMo content occupies only ~33% of the canvas width (a thin portrait strip in the center).
  - PLUS the per-clip-mux mode (`audio_source: master_mix_per_clip_mux`) at `nodes/video_composite.py:_render_master_mix_per_clip_mux_mode` builds a video filter chain that ONLY pillarboxes HuMo clips — it never overlays the procgen visual layer (`procgen` parameter is consumed but only used for audio extraction). So even where HuMo content wasn't rendered, the canvas was solid black.
  - Net result Jeffrey reported: "all black short videos." HuMo + LTX per-line clips look great in isolation; composite renders them as a thin strip surrounded by black bars.
  - LTX clips were OK (832x480 landscape filled the 832x480 canvas correctly), so non-character lines briefly showed visible content.
- **Cause:** Two coupled bugs:
  1. Canvas/HuMo orientation mismatch (the 2026-05-01 default of 832x480 was based on the incorrect assumption that HuMo renders landscape — it renders 480x832 portrait, the only fully-trained dim per Wan2.1-HuMo-14B + ROADMAP "Stable shape: length=97 (3.88 s @ 25 fps), 480x832, batch=1").
  2. Per-clip-mux pillarbox-each-clip-individually approach has no concept of layered composition — each clip renders alone on a black canvas.
- **Fix (Phase A — simple-pillarbox composite per Jeffrey's REVISED spec, 2026-05-03 EVENING):**
  - **HuMo render dims:** `nodes/batch_humo_render.py` widget defaults stay at the canonical Wan2.1-HuMo-14B trained dim **480x832 PORTRAIT @ 25 FPS** (per ROADMAP "Stable shape: length=97, 480x832, batch=1"). An earlier draft of this fix attempted 1280x720 landscape for face detail; Jeffrey reverted to native ("humo native portrait render then native scaled to 1472x832 with black pillaboxes") to avoid OOD lipsync drift + ~2.3x VRAM overhead.
  - **LTX render dims:** `nodes/batch_ltx_render.py` module constants stay at native **LTX_WIDTH=832, LTX_HEIGHT=480 landscape**. An earlier draft attempted 1216x704 for higher pre-upscale detail; Jeffrey reverted to native ("ltx native landscape render downscaled to 1472x832") for the same trained-distribution-safety reason.
  - **Composite canvas:** `nodes/video_composite.py` INPUT_TYPES defaults canvas_width 832→1472, canvas_height 480→832, humo_target_height 480→832. Workflow JSON node 52 widget values updated to match. New widget `humo_pillar_width` default 512 reserved for Phase B layered-mode use; unused by the active Phase A simple-pillarbox flow.
  - **`_layered_per_clip_silent(...)` simple-pillarbox branch:** all clips (HuMo + LTX) scale-FIT into 1472x832 with `force_original_aspect_ratio=decrease` then pad-with-black to canvas dims. NO `crop=` (preserves source aspect, no content lost). HuMo 480x832 stays at native (height=832 already matches canvas), padded with ~496px BLACK BARS per side. LTX 832x480 scaled to height=832 = 1442x832, padded with ~15px black per side (effectively full canvas). Both paths drop audio (`-an`); master mix attaches at the final mux step so C7 byte-identity holds.
  - **No env-still backdrop in Phase A:** `_render_master_mix_per_clip_mux_mode` always passes `background_png=None` to `_layered_per_clip_silent` so the helper takes its simple-pillarbox branch. The `_resolve_episode_background` helper + `_layered_per_clip_silent` layered-overlay branch are kept in the codebase (with their own tests) for future use cases where a static env-still backdrop IS desired -- but the current Phase A renderer never invokes them.
  - **Procgen visual fill -- PHASE B SHIPPED in same session (separate commit):** the visible HuMo black pillarbox bars are intentional. Procgen renders at native 1920x1080 (was 832x480) via the updated `OTR_SignalLostVideo` resolution default. New node `OTR_PostUpscaleProcgenBlend` (`nodes/otr_post_upscale_procgen_blend.py`, registered as `OTR_PostUpscaleProcgenBlend`) takes the RTXUpscale 1920x1080 output + the 1920x1080 procgen, builds an ffmpeg `-filter_complex` blend chain (`[0:v][procgen]blend=all_mode=lighten:all_opacity=0.5[v]`), maps source audio with `-c:a copy` (zero re-encodes -- C7 byte-identity preserved end-to-end). Output filename: `<source_stem>_procgen_blended.mp4` in the same dir as source. Per Jeffrey: "proc gen 1920x1080 ... then a final ffmpeg w/ the proc gen mix for final 1080p." Post-upscale blend turns the visible black surround into audio-reactive CRT scanlines -- the SIGNAL LOST visual signature -- without going through the AI upscaler (which would smear synthetic patterns) and without touching the per-clip-mux mode (which would risk C7 byte-identity). Three failure-mode fallbacks (bypass widget, missing procgen, ffmpeg failure) all degrade to source-copy so the pipeline always produces a deliverable. **Workflow JSON wiring SHIPPED in same commit** per Jeffrey directive ("plwsae dont dfeer anyting we need to test veryting"): new node id 58 added at `pos=[4900, 1100]`, link 95 wires `RTXUpscale.upscaled_mp4_path -> source_mp4_path`, link 96 wires `SignalLostVideo.video_path -> procgen_mp4_path`. `last_node_id` bumped 57->58, `last_link_id` bumped 94->96. Verified via `_verify_wiring.py`: 31 nodes, 53 links, all slot indices + types correct, JSON parses cleanly.
  - **`_render_master_mix_per_clip_mux_mode` rewired:** both call sites (in-loop + post-loop tail-pad re-pillarbox) switched from `_pillarbox_humo_silent` to `_layered_per_clip_silent`. The legacy helper is kept for back-compat with non-layered standalone use cases.
  - **Math sanity:** HuMo 480x832 → scale to height=832 = 480x832 (no scale, native quality preserved) → pad to 1472x832 = 480x832 centered + 496px black per side. LTX 832x480 → scale to height=832 = 1442x832 → pad to 1472x832 = 1442x832 centered + 15px black per side. Final RTXUpscale 1472x832 → 1920x1080 is clean (16:9 source → 16:9 delivery, 1.30x scale).
- **Phase B (queued — not in this commit):** procgen render at native 1920x1080 (currently 832x480) + new `OTR_PostUpscaleProcgenBlend` node that overlays procgen on the RTXUpscale output AT delivery res. Architecture: keep procgen visual OUT of the per-clip-mux composite (so C7 audio-identity protected mode stays untouched), blend it in post-RTXUpscale where it doesn't get smeared by the AI upscaler. Per Jeffrey: "proc gen at 0 [composite stage], native 1920x1080, and concat happens after upscale." Blend is `-c:a copy` so audio passes through with zero re-encodes.
- **Verify:**
  - AST parse on `nodes/video_composite.py`, `nodes/batch_humo_render.py`, `nodes/batch_ltx_render.py`: green.
  - Workflow JSON node 51 (HuMo) widgets_values width=1280, height=720; node 52 (VideoComposite) widgets_values canvas_width=1472, canvas_height=832, humo_target_height=832, humo_pillar_width=512: confirmed.
  - **New `tests/test_video_composite_layered.py` — 12 tests passing:** `_resolve_episode_background` priority order (env still > radio bookend > None) + the 4 ffmpeg cmd shape branches of `_layered_per_clip_silent` (character+bg → layered, character+no-bg → simple, non-character → simple, tail-pad in both paths) + INPUT_TYPES default sanity + LTX module constants.
  - Existing `tests/test_video_composite_per_clip_mux.py` updated: 2 tests that patched `_pillarbox_humo_silent` re-patched to `_layered_per_clip_silent` (the renderer's new call target). All 29 tests in the file green.
  - Cumulative regression: **203 passed in 3.88s** (`tests/test_video_composite_layered.py + test_video_composite_per_clip_mux.py + test_critique_dialogue_preservation.py + test_save_to_episode_workspace.py + test_prompt_format_safety.py + test_production_ledger.py + test_radio_still_resolver.py + test_filename_pattern_audit.py + test_cache_key_mutations.py + test_meta_paths.py + test_ledger_rename.py`).
  - Bug Bible regression: 24 passed / 1 skipped / 1 xfailed in 1.32s.
  - **Real-run acceptance (pending):** queue an episode after restarting ComfyUI. Expect (a) HuMo per-line clips at 1280x720 not 480x832 (ffprobe each `videos/lNNN.mp4`), (b) per_clip_mux report includes log line `BUG-030 layered composite: HuMo character backdrop = full_env_NNNNN_.png` (or radio_bookend if no env still), (c) final composite mp4 dims = 1472x832 (not 832x480), (d) RTXUpscale OBS final at 1920x1080 with **visible HuMo pillar in center over scene backdrop on character lines + LTX broadcast unit filling canvas on non-character lines**.
- **Tags:** video-composite, simple-pillarbox, landscape-canvas, humo-portrait, ffmpeg-pad, procgen-phase-b, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-027 (dialogue wipe — orthogonal); BUG-LOCAL-028 (per-episode FLUX save paths — provides the per-episode workspace structure used here, even though Phase A no longer reads env stills as backdrop); BUG-LOCAL-029 (ULTRA_SMOKE format normalize — orthogonal). Round-robin consult was SKIPPED per direct user override; AST + invariant audits + targeted regression + Bug Bible regression all green pre-commit. HuMo + LTX stay at native trained dims (480x832 + 832x480) so no OOD risk; an earlier draft of this fix attempted 1280x720 + 1216x704 for higher pre-upscale detail, but Jeffrey reverted to native ("humo native portrait render then native scaled to 1472x832 with black pillaboxes ... ltx native landscape render downscaled to 1472x832 ... then a final ffmpeg w/ the proc gen mix for final 1080p"). The visible black surround on character lines is intentional and gets filled with audio-reactive CRT scanlines by the Phase B post-RTXUpscale procgen blend — the SIGNAL LOST visual signature.

---

### BUG-LOCAL-029 [FIXED]: ULTRA_SMOKE preset bypasses BUG-027 dialogue-preservation gate (parser format mismatch)
- **Date:** 2026-05-03 EVENING | **Phase:** acceptance (post-027 follow-up surfaced by headless soak) | **Bible candidate:** YES
- **Symptom (from headless soak `signal_lost_static_echo_20260503_140824` per otr_runtime.log):**
  - L47819 `[14:00:27] ScriptWriter: PARSE_OK attempt=1 has_scene=True voice_hits=4 bare_hits=0 smoke=ultra:True/tiny:False` — writer's PARSE_OK validator counted 4 `[VOICE: ...]` markers in the draft.
  - L47868 `[14:01:22] CRITIQUE: Character line counts - draft={} revised={}` — `_count_character_lines` regex (BUG-027-extended for `[N] CHARNAME:`) DOES NOT match the ULTRA_SMOKE-specific `[VOICE: NAME, attrs, ...]: text` line format. Counter returned `{}` for both draft and revised.
  - L47869 `[14:01:22] CRITIQUE: Revised script accepted (sim=40.4%, len=244%)` — gate accepted a revision with 244% length expansion + 40% similarity (radical rewrite). Gate skipped per `if draft_total >= 3` short-draft skip — but draft_total was 0 only because the parser couldn't see the dialogue.
  - L47879-80 `BUG-109b: cast members with 0 lines: PETER ECKELS, REN KANE` / `1/1 scene(s) have 0 dialogue lines` — same downstream failure mode as the original BUG-027.
- **Cause:** BUG-LOCAL-005 (Sprint 1) added a `[VOICE: NAME, attrs, ...]: text` strict-VOICE format for the ULTRA_SMOKE preset, with a separate PARSE_OK validator (`voice_hits` counter). BUG-LOCAL-027 fixed `_count_character_lines` to accept `CHARNAME:` and `[N] CHARNAME:` formats but NOT the ULTRA_SMOKE `[VOICE: ...]` format. Result: ULTRA_SMOKE drafts parsed as `{}` in the critique pipeline, the BUG-027 total-collapse gate had no signal to enforce, and ULTRA_SMOKE silently bypassed the dialogue-preservation guarantee.
- **Fix (per Jeffrey directive 2026-05-03 EVENING — "ULTRA_SMOKE need to abide by all the rules"):** new helper `LLMScriptWriter._normalize_voice_format_to_standard(text)` in `nodes/story_orchestrator.py` — staticmethod that converts `[VOICE: NAME, attrs, ...]: text` → `NAME: text` AND strips inline `[VOICE: ...]` blocks from dialogue content. Wired into `_critique_and_revise` at TWO points: (a) at function entry on `draft_text` BEFORE the critique pass runs, so the critique LLM, the per-character preservation gate, and the total-collapse hard gate all see the canonical format; (b) on `revised_text` after the revision pass, so the gate counter compares apples-to-apples even if the revision LLM slipped back into `[VOICE: ...]` shape under high-temp creativity. Idempotent on already-standard text. C7-safe (deterministic regex transformation; same input always produces same normalized output, so byte-identity holds).
- **Architectural choice:** Jeffrey explicitly chose conversion-to-standard over extending the parser to handle multiple formats. Rationale: ONE source of truth for dialogue preservation (the standard `CHARNAME:` format) and one set of rules (the BUG-027 gate machinery). ULTRA_SMOKE keeps its strict-VOICE writer prompt + PARSE_OK validator (BUG-005 contract intact), but its output gets normalized before any downstream pipeline stage. Alternative was to remove ULTRA_SMOKE entirely, deferred.
- **Verify:**
  - AST parse on `nodes/story_orchestrator.py`: green.
  - **New tests in `tests/test_critique_dialogue_preservation.py` (7 added, 21 total in file, all green):** `test_normalize_standalone_voice_prefix_to_charname`, `test_normalize_voice_with_no_attrs`, `test_normalize_strips_inline_voice_block_from_dialogue`, `test_normalize_idempotent_on_standard_format`, `test_normalize_handles_empty_and_none`, `test_normalize_then_count_recovers_dialogue_for_ultra_smoke` (end-to-end: normalize + count yields correct character counts on a realistic ULTRA_SMOKE draft mirroring the actual L47868 failure shape), `test_critique_calls_normalize_at_function_entry` (static check that `_critique_and_revise` body actually calls the normalizer on BOTH `draft_text` AND `revised_text`).
  - Cumulative regression: **162 passed in 4.01s** (targeted set: production_ledger + radio_still_resolver + filename_pattern_audit + cache_key_mutations + meta_paths + ledger_rename + critique_dialogue_preservation + save_to_episode_workspace + prompt_format_safety) PLUS Bug Bible regression **24 passed / 1 skipped / 1 xfailed** in 1.49s.
  - **Real-run acceptance (pending):** queue an ULTRA_SMOKE episode (target_length="30 words (smoke, 1 act)"); expect (a) `CRITIQUE: ULTRA_SMOKE format normalized (N1 -> N2 chars)` log line if `[VOICE: ...]` lines were present, (b) `CRITIQUE: Character line counts - draft={'CHARNAME': N, ...}` with NON-EMPTY draft dict (was `{}` before this fix), (c) BUG-109b should NOT fire if the gate correctly preserves dialogue.
- **Tags:** ultra-smoke, voice-format, normalization, bug-027-extension, critique-pipeline, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-005 (created the strict `[VOICE: ...]` format for ULTRA_SMOKE); BUG-LOCAL-027 (fixed the standard-path counter, missed ULTRA_SMOKE). Round-robin consult was SKIPPED per direct user override; AST + targeted regression + Bug Bible regression all green pre-commit.

---

### BUG-LOCAL-028 [FIXED]: FLUX env stills + radio bookend save to legacy flat dirs — VideoComposite finds no scenery; final video black
- **Date:** 2026-05-03 | **Phase:** acceptance (post-026 hotfix soak) | **Bible candidate:** YES (directly causes "black video" failure mode)
- **Symptom (from `dir /s C:\...\output\otr\` for episode `signal_lost_astronomers_finally_solve_the_gammacas_x_20260503_002536`):**
  - Episode workspace `output/otr/episodes/<ep>/` has `audio/` `videos/` `composited/` subdirs (Phase B writes correctly).
  - **`output/otr/episodes/<ep>/stills/` does NOT exist.** **`output/otr/episodes/<ep>/portraits/` does NOT exist.**
  - FLUX outputs landed in legacy flat dirs:
    - Radio bookend → `output/otr/_legacy_stills/radio_bookend_<ep>.png` (filename has ep_id baked in but DIR is legacy).
    - Env stills → `output/otr/stills/full_env_NNNNN_.png` with a global counter shared across all episodes since 4/26 (213 PNGs accumulated, none stamped to a specific episode by name).
  - VideoComposite reads from per-episode `videos/` (correct) but cannot find env stills or radio bookend in the per-episode `stills/` (because they're not there). Result: no scenery layer in composite → mostly-black canvas.
  - Final composited mp4 = 1.72 MB; obs/ final = 1.18 MB; Jeffrey reports "black video, 15s of audio, no announcer in final."
- **Recurring (NOT a one-off):** affects every episode since the per-episode workspace reorg (Phase B, 2026-05-02 EVENING). Audio/video paths got reorged; FLUX outputs were not.
- **Cause (two separate sites):**
  - **Site 1: `visual/batch_flux_render.py:833`** — `stills_dir = _OTRP.otr_stills_dir()` called with no `episode_id` argument. Per `nodes/_otr_paths.py:208-218`, `otr_stills_dir()` without an episode_id falls back to `output/otr/_legacy_stills/`. The `episode_id` variable is in scope from line 768/772 (resolved from the in-flight ledger singleton via the same Phase G singleton-discovery path used by BUG-LOCAL-021). One-line fix.
  - **Site 2: `workflows/otr_scifi_16gb_full.json` node id 25** — stock ComfyUI `SaveImage` with hardcoded `filename_prefix: "otr/stills/full_env"` widget value. ComfyUI writes to `output/<filename_prefix>_<auto_counter>_.png`. The path doesn't change per-episode because the widget is static. Listed in ROADMAP.md "Known remaining suspects" (lines 47-55) under the same Phase G blast-radius pattern, but the visual-layer impact wasn't appreciated until this run. Architectural fix: replace stock SaveImage with a custom OTR node that reads the in-flight ledger singleton and routes to `otr_stills_dir(<ep_id>)`.
- **Fix (four sites — write + read alignment):**
  - **Site 1 (writer, radio bookend):** `visual/batch_flux_render.py:845` — changed `_OTRP.otr_stills_dir()` → `_OTRP.otr_stills_dir(episode_id)`. The `episode_id` variable was already in scope at line 768/772 (resolved via the in-flight ledger singleton, same Phase G discovery path used by BUG-LOCAL-021). Radio bookend now lands at `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` per the canonical Phase B layout.
  - **Site 2 (writer, env stills):** new node `OTR_SaveToEpisodeWorkspace` in `nodes/otr_save_to_episode_workspace.py`. Reads `_otr_ledger.in_flight_ledger_path()` to derive episode_id at runtime; routes to `otr_stills_dir(ep_id)` or `otr_portraits_dir(ep_id)` based on `role_kind` widget ("stills" | "portraits"). Falls back to legacy dirs (preserving existing behavior) if no singleton is available — never raises in headless/test contexts. Registered in `__init__.py`. Workflow JSON `workflows/otr_scifi_16gb_full.json` node 25 retyped from `SaveImage` to `OTR_SaveToEpisodeWorkspace` with `role_kind="stills"`, `filename_pattern="full_env"`.
  - **Site 3 (reader, BatchHumoRender env-still binding):** `nodes/batch_humo_render.py:_resolve_cast_stills_from_ledger` and `_find_portrait` — added per-episode glob pattern `otr/episodes/*/stills/full_env_*.png` alongside the existing legacy `otr/stills/full_env_*.png` and `otr_stills/full_env_*.png` patterns. Without this, after Site 2 starts writing to per-episode dirs, HuMo's cast→still binding would find ZERO fresh stills and fall back to stale prior-episode stills (or unmapped, then fall through to portrait/composite tiers). The mtime-based freshness filter (`fresh_floor = ledger_mtime - 60s`) in the same function still enforces episode-correctness, so cross-episode pollution is mathematically impossible.
  - **Site 4 (reader, BatchLTXRender radio bookend):** `nodes/batch_ltx_render.py:374` — changed `otr_stills_dir() / f"radio_bookend_{eid}.png"` → `otr_stills_dir(eid) / f"radio_bookend_{eid}.png"`. Without this, after Site 1 starts writing to per-episode dirs, LTX would look in `_legacy_stills/` and find nothing, falling back to a generic motion clip with no scene continuity. (`nodes/video_composite.py:163` was already correct — passes `eid` — verified.)
- **Verify:**
  - AST parse on all 6 touched files (story_orchestrator, batch_flux_render, batch_humo_render, batch_ltx_render, otr_save_to_episode_workspace, __init__) green.
  - JSON parse + node-type audit on `workflows/otr_scifi_16gb_full.json`: 30 nodes, 0 stock SaveImage remaining, 1 OTR_SaveToEpisodeWorkspace registered.
  - **New `tests/test_save_to_episode_workspace.py` — 8 passed in <1s.** Covers: with active singleton → resolves to per-episode dir; no singleton → falls back to legacy dir; role_kind="stills" → otr_stills_dir; role_kind="portraits" → otr_portraits_dir; filename_pattern preserved; per-episode counter starts at 1 (independent of any global counter); never raises on mkdir failure; node registered in `NODE_CLASS_MAPPINGS`.
  - Cumulative regression: **155 passed in 3.27s** across `tests/test_production_ledger.py + test_radio_still_resolver.py + test_filename_pattern_audit.py + test_cache_key_mutations.py + test_meta_paths.py + test_ledger_rename.py + test_critique_dialogue_preservation.py + test_save_to_episode_workspace.py + test_prompt_format_safety.py`. PLUS Bug Bible regression **24 passed / 1 skipped / 1 xfailed** in 1.24s.
  - **Real-run acceptance (pending):** queue any episode; expect `output/otr/episodes/<ep>/stills/full_env_NNNNN_.png` with COUNTER STARTING AT 1 (per-episode counter, not global), AND `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` co-located. BatchHumoRender log line `[BatchHumoRender] cast-still binding: N/M cast members matched to fresh stills` should report N>0. BatchLTXRender should find the radio bookend at the per-episode path. Final video should have visible scenery (not 100% black) when `blend_opacity > 0` is also set.
- **Tags:** flux, save-paths, per-episode-workspace, phase-g-blast-radius, video-composite-empty, write-read-alignment, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-021 (Phase G singleton sweep — fixed audio side, missed visual side); BUG-LOCAL-027 (dialogue wipe — orthogonal failure that compounds with this bug; fixed in same commit). Round-robin consult was SKIPPED per direct user override ("yes ofrget rop8u7hnd robins just fix fix fix") — extra verification in lieu: AST + format-safety + targeted regression + Bug Bible regression all green pre-commit.
- **Headless soak status (2026-05-03 EVENING — UPDATED post-launch):** ATTEMPTED + IN PROGRESS at session end. ComfyUI was relaunched headless via the venv python (`main.py --listen 127.0.0.1 --port 8000 --highvram`) after the autonomous-pass cleanup taskkill cleared its previous session. Boot succeeded (36 OTR nodes loaded, including the new `OTR_SaveToEpisodeWorkspace`; verified via `/object_info`). Soak queued via `scripts/soak_bug027_028.py`, prompt_id `d29e3d8f-edce-48e2-a960-7245ff989543`. **Widget patches FAILED** with `widgets_values length mismatch on node 1 (OTR_LLMScriptWriter): len(wv)=15 vs len(widget_names)=16` — the saved workflow JSON has 15 widget values but the live schema reports 16, indicating widget drift on `OTR_LLMScriptWriter` since the workflow was last saved. Soak still queued with the workflow's saved values: `target_words=350, target_length="short (3 acts)", num_characters=2, style="tense claustrophobic", creativity="balanced"`. The "balanced" creativity (temp=0.85) is LESS aggressive on the BUG-027 trigger conditions than the original "maximum chaos" (temp=0.95) repro shape — soak validates the pipeline + new node registration but may not exercise the total-collapse gate fire. Last status check (~12 min into run): `queue_running: 1`, prompt_id still in queue, otr_runtime.log frozen at last warmup line `[01:11:40] WARMUP: CUDA kernels compiled in 0.6s`, ComfyUI python.exe at 18.3 GB working set (model loaded + active inference). Logs are buffered between LLM call boundaries; Jeffrey will see acceptance signatures populate in the morning when the run completes. **Followup BUG candidate:** widget drift on `OTR_LLMScriptWriter` between saved workflow + live schema — log as BUG-LOCAL-029 if confirmed (15 vs 16 widget mismatch). One-shot soak script `scripts/soak_bug027_028.py` is committed and ready for re-run when widget drift is resolved.

---

### BUG-LOCAL-027 [FIXED]: Critique/revision pass returns SCENE/ENV/SFX-only script, dropping all CHARACTER dialogue — Bark gets 0 lines
- **Date:** 2026-05-03 | **Phase:** acceptance (post-026 hotfix soak) | **Bible candidate:** TBD (likely YES — recurring across multiple runs)
- **Widget config (from screenshot):** target_words=110, num_characters=2, target_length="short (3 acts)", style="noir mystery", creativity="maximum chaos", arc_enhancer=ON, self_critique=ON, open_close=ON, optimization_profile="Pro (Ultra Quality)", model=google/gemma-4-E2B-it. Standard short(3) preset — NOT ultra_smoke / tiny_smoke (so BUG-LOCAL-005's CHARACTER:/SCENE: enforcement does not apply to this code path).
- **Symptom — current run "Cold Circuit" (otr_runtime.log line numbers):**
  - L47167 `[00:14:27] ScriptWriter: PARSE_OK attempt=1 has_scene=True voice_hits=18 bare_hits=0` — initial draft healthy: 3 scenes, 18 dialogue lines, characters ANNOUNCER + FLETCHER WELLS + KENJI BERNARD.
  - L47256 `[00:16:30] CRITIQUE: Character line counts - draft={} revised={}` — character-line counter returns empty dicts for both draft and revised. Parser disagreement: draft visibly had 18 `[N] CHARNAME:` lines, but the critique-pipeline counter sees zero. Revised pass also legitimately produced zero (only `=== SCENE N ===`, `ENV:`, `SFX #N:` lines emitted across the entire 100s revision generation — no `[N] CHARNAME:` lines at all).
  - L47257 `[00:16:30] CRITIQUE: Revised script accepted (sim=83.0%, len=118%)` — acceptance gate let the dialogue-stripped revision through. Similarity stayed high because SCENE/ENV/SFX scaffolding overlaps; length grew because model padded with extra atmosphere.
  - L47265 `[00:16:30] WORD_ENFORCEMENT: 0 words vs 110 target (0%) | @140wpm -> ~0.0 min [0 lines detected]`
  - L47267 `[00:16:30] BUG-109b: cast members with 0 lines: FLETCHER WELLS, KENJI BERNARD`
  - L47268 `[00:16:30] BUG-109b: 3/3 scene(s) have 0 dialogue lines`
  - FORMAT_NORM single-pass attempted recovery; ANNOUNCER bookends were generated separately (Kokoro-bound, not Bark-bound); CHARACTER dialogue was never restored.
  - Final downstream effect: `[BatchBark] Found 0 dialogue lines in Canonical 1.0 format (skipped 2 ANNOUNCER lines - routed to Kokoro bus)`. SceneSequencer pre-rendered 1 TTS + 8 SFX + 2 ANNOUNCER, zero Bark clips.
- **Recurring across runs (NOT a one-off):**
  - L44646 `[22:00:57] CRITIQUE: Character line counts - draft={} revised={}` (sim=56.6%, len=176% accepted) — earlier run, same shape.
  - L46713 `[23:43:26] CRITIQUE: Character line counts - draft={} revised={'ANNOUNCER': 2}` (sim=90.4%, len=94% accepted) — preserved ANNOUNCER only, dropped character dialogue.
  - L47256 `[00:16:30] CRITIQUE: Character line counts - draft={} revised={}` — current run.
  - Pattern: critique pipeline character counter consistently returns `{}` for draft regardless of writer output; acceptance gate (similarity + length only) cannot detect dialogue loss; multiple runs have shipped dialogue-stripped scripts to the audio cascade.
- **Cause (CONFIRMED via source dive 2026-05-03 EVENING):**
  - **Two coupled gaps in `nodes/story_orchestrator.py`.**
  - **Gap 1 (parser blindness):** `_count_character_lines` (line 6890) regex was `r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'` — required line to START with optional whitespace + uppercase name. The writer's actual output format is `[12] FLETCHER WELLS: text` (numbered-bracket prefix), so the regex never matched and returned `{}` for both draft and revised. Acceptance gate at line 7174 iterated `draft_char_counts` (empty dict) → no-op → revision accepted regardless.
  - **Gap 2 (gate too narrow):** even with parser working, the per-character preservation check (line 7174-7184) only catches "FLETCHER dropped from 8 to 1." It does NOT catch "every character wiped at once" because the loop iterates the draft dict per-char; if the revision wipes ALL characters, no individual character drops below the floor (they all dropped from N to 0, but the loop doesn't compare totals). Surface metrics — similarity ratio (0.83) + length ratio (1.18) — both pass on a SCENE/ENV/SFX-only revision because the scaffolding overlaps.
  - **Secondary contributor:** revision pass uses `temperature` (passed in from caller — for "maximum chaos" creativity = 0.95). High temp + critique demanding "fix every flagged problem" can push the model into pure-prose rewriting mode where it drops dialogue in favor of atmospheric SFX/ENV. The structural floor (`structural_temp=0.6` for similarity/length checks) doesn't gate this — it's a separate variable.
- **Fix (three-part):**
  - **Part 1 (parser regex, line 6916):** added optional non-capturing group `(?:\[\d+\]\s+)?` so both `CHARNAME:` and `[N] CHARNAME:` formats parse. Also tightened the structural-token exclude check at line 6924 to do BOTH exact-match AND first-word-match (`first_word in _struct_exclude`), so multi-word headers like `ACT 2:` or `SCENE 3:` no longer slip through as character names.
  - **Part 2 (total-collapse hard gate, after line 7184):** belt-and-suspenders for the per-character check. Computes `draft_total = sum(draft_char_counts.values())` and `revised_total = sum(revised_char_counts.values())`; if `draft_total >= 3` (threshold to apply ratio) and `revised_total < max(1, ceil(draft_total * 0.5))`, logs `CRITIQUE_REJECTED - total character lines collapsed from N to M (min=K, threshold=50%%)` and returns the draft unchanged. Below 3 lines the draft is too short for a meaningful ratio — the per-character check (with `min_line_count_per_character=2` floor) handles those cases.
  - **Part 3 (revision prompt hardening, line 7034 area):** added explicit `ABSOLUTE REQUIREMENT — DIALOGUE MUST SURVIVE THE REVISION` clause to the revision LLM prompt. Tells the model EXPLICITLY that producing a SCENE/ENV/SFX-only output is a "TOTAL FAILURE" and that every CHARACTER speaker present in the draft MUST appear in the revision. Also documented that the optional `[N]` prefix from the draft may be kept or omitted (both parse). Format-safety smoke (`tests/test_prompt_format_safety.py`) confirms no unescaped `{}` braces in the new prose (BUG-026 lesson).
- **Verify:**
  - AST parse on `nodes/story_orchestrator.py`: green.
  - **New `tests/test_critique_dialogue_preservation.py` — 14 passed in <1s.** Covers: parser handles bare `CHARNAME:`; parser handles `[N] CHARNAME:`; parser handles mixed format in same text; structural tokens (SCENE/ACT/MUSIC/SFX/ENV) excluded by exact-match AND first-word-match; empty/None text returns empty dict; ANNOUNCER counted as character; gate REJECTS total dialogue wipe (the actual L47256 case); gate REJECTS announcer-only revision (the L46713 case); gate ACCEPTS minor dialogue trim (83% retention); gate ACCEPTS at exactly 50% threshold; gate SKIPS short drafts (`< 3` lines); gate handles empty dicts safely; revision prompt has no unescaped braces (BUG-026 footgun gate); ABSOLUTE REQUIREMENT clause is present in source.
  - `tests/test_prompt_format_safety.py` — 1 passed (BUG-026 regression test passes against the new prompt prose).
  - Cumulative regression: 155 passed in 3.27s + Bug Bible 24 passed / 1 skipped / 1 xfailed in 1.24s.
  - **Real-run acceptance (pending):** queue the same widget config (110 words / 2 chars / short(3) / noir mystery / maximum chaos); expect (a) `CRITIQUE: Character line counts - draft={'ANNOUNCER': N, 'CHAR1': M, ...}` with NON-EMPTY draft dict (parser fix), (b) if revision wipes dialogue, log `CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed from N to M` and the pipeline uses the original draft, (c) `[BatchBark] Found >=1 dialogue lines in Canonical 1.0 format` in the final pre-render summary.
- **Tags:** critique, revision, character-dialogue, parser-mismatch, acceptance-gate, bark-empty, qa-soak-2026-05-03, recurring, fixed
- **Related:** BUG-LOCAL-005 (CHARACTER:/SCENE: enforcement — only applies to ULTRA_SMOKE path; this is short(3)). BUG-109b detector (pre-existing observability — fired correctly here, no auto-recovery existed before this fix). Round-robin consult was SKIPPED per direct user override — extra verification in lieu: AST + format-safety + targeted regression + Bug Bible regression all green pre-commit.
- **Headless soak status (2026-05-03 EVENING):** DEFERRED. Same reason + same handoff as BUG-LOCAL-028 above: ComfyUI was terminated mid-session and the one-shot soak script at `scripts/soak_bug027_028.py` queues both 027 + 028 acceptance signatures in a single run. Jeffrey runs the soak in the morning after restarting ComfyUI Desktop.

---

### BUG-LOCAL-026 [FIXED]: Phase H regression — unescaped curly braces in DIRECTOR_PROMPT crashed `.format()` mid-pipeline
- **Date:** 2026-05-03 | **Phase:** G/H hotfix | **Bible candidate:** YES (classic `.format()` footgun)
- **Symptom:** Live soak crash on 2026-05-02 23:46. Episode "Exponential Tremor Echoes" (style="a claude cowork test session", target_words=80) ran cleanly through ScriptWriter (3-outline OpenClose evaluator, critique pass, revision pass, ScriptCritic verdict REVISE with 7 issues, revision applied). Crashed at LLMDirector.direct (`nodes/story_orchestrator.py:9951`):
  ```
  IndexError: Replacement index 0 out of range for positional args tuple
  ```
  ~10 minutes of LLM compute lost.
- **Cause:** Phase H BUG-LOCAL-023 added an EXCLUDE-ANNOUNCER clause to `DIRECTOR_PROMPT`. The added prose contained literal `visual_plan.characters{}` and `voice_assignments{}` — two unescaped `{}` empty-brace pairs. The surrounding template uses `str.format()` with kwargs (`script_text`, `voice_mapping_rules`); Python's `.format()` interpreted `{}` as a positional arg slot reference, looked up `args[0]`, found nothing, raised `IndexError`. **Cardinal mistake — adding prose with literal braces to a `.format()` template without escaping.**
- **Fix:** Removed the literal `{}` symbols from the EXCLUDE-ANNOUNCER prose. Kept the semantic content ("EXCLUDE narrator/announcer roles from the visual plan characters object", etc.) — readable to the LLM, no longer breaks `.format()`. Either `{{ ... }}` escaping OR removing the braces from prose is valid; chose removal for prose-readability.
- **Verify:**
  - Standalone `_director_prompt_test.py` smoke: `DIRECTOR_PROMPT.format(script_text=..., voice_mapping_rules=...)` returns 5501 chars, no exception.
  - **Permanent regression test** `tests/test_prompt_format_safety.py` — extracts `DIRECTOR_PROMPT` constant via regex, calls `.format()` with the production kwargs, asserts no `IndexError`/`KeyError`/`ValueError`. **Passed in 1.74s.** Future Phase-N additions to the prompt that re-introduce unescaped braces will fail this test before they reach a live run.
- **Tags:** phase-h-regression, str-format, prompt-template, hotfix, bible-candidate
- **Lesson learned for future autonomous mode:** when editing any constant that's later passed to `.format()`, run `_director_prompt_test.py`-style format smoke as part of the AST guard pass. Don't ship template edits without confirming `.format()` survives.

---

### BUG-LOCAL-025 [FIXED]: LTX role prompts ignore story style + scene context (every episode looked the same)
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** `nodes/batch_ltx_render.py::_PROMPT_BY_ROLE` is a hardcoded dict mapping `{announcer, music_open, music_close, music_inter, sfx}` → fixed motion prompts ("Vintage 1940s radio broadcast set, glowing tuning dial pulses gently, copper vacuum tubes warm amber glow..."). Every episode renders the SAME LTX motion regardless of the story's style or scene atmosphere. Jeffrey: *"be sure story arc or better shot/scene arc is being fed into FLUX and LTX as well to match the short."*
- **Cause:** Original `_PROMPT_BY_ROLE` design treated LTX as a generic radio-animator with no story awareness. Acceptable when the radio bookend (the i2v reference image) carries all visual identity — but downstream review confirmed the motion prompt itself influences mood (dial sweep speed, tube glow rhythm, dolly direction).
- **Fix:** New `_build_ltx_role_prompt(role, line, ledger)` helper enriches each role base prompt with two ledger-derived layers:
  1. **Per-line scene context.** Lookup chain: `line.shot_id` → `ledger.shots[*].scene_id` → `ledger.scenes[*].env / .description`. Truncated to 60 chars, appended as `, scene context: <env>`. Each LTX clip now matches the SCENE it accompanies (early scenes get tense env, late scenes get resolved env).
  2. **Episode style suffix.** Read from `ledger.meta.gen_params_initial.style` (or `.gen_params.style`) — same singleton-fed source Phase G fixed for radio bookend. Appended as `, <style> broadcast tone`.
  Bounded so the role's motion intent isn't drowned. Per-line lookup means one episode's announcer LTX clips can vary across scenes if those scenes have different `env` text.
- **Verify:** AST + full pytest (1150 / 8 / 1 in 131.62s) green. **Real-run acceptance (pending):** the `[BatchLTXRender]` log lines should now show enriched prompts; two episodes with different styles should produce visibly different LTX motion intent.
- **Tags:** ltx, story-arc, scene-context, style-aware, qa-pass-2026-05-03

---

### BUG-LOCAL-024 [FIXED]: Radio bookend FLUX prompt fell back to generic when style missing OR ledger stale
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes
- **Symptom:** Soak run on 2026-05-02 logged `[BatchFluxRender] radio still prompt source=fallback (no style)` — radio rendered as generic "sci-fi retrofuturistic radio broadcast unit" despite user setting style="space opera epic" in the widget. Compounded with BUG-LOCAL-021 (FLUX read a stale April 26 ledger via the broken `find_most_recent_ledger` walker), the radio NEVER reflected the actual episode story.
- **Cause:** `_build_dynamic_radio_prompt` in `visual/batch_flux_render.py` only looked at two fields (`gen_params_initial.style` + `gen_params.style`) before falling to a single hardcoded fallback. No fallback chain through `style_custom`, scene environment, or episode title.
- **Fix:** Six-tier resolution with per-tier branch logging:
  1. `gen_params_initial.style` (primary widget value)
  2. `gen_params.style` (back-compat)
  3. `gen_params_initial.style_custom` (free-text override)
  4. First scene's `env` / `description` (scene-driven mood)
  5. `episode_id` slug (strip "signal_lost_" prefix + trailing timestamp, replace underscores)
  6. Hardcoded `_RADIO_FALLBACK_PROMPT` (true last resort)
  Plus: scene-context hint (`set in <first_scene_env>`) appended whenever distinct from descriptor, so style + scene combine. New log line `[BatchFluxRender] radio prompt: branch=<which> -> <preview>` tells the runtime tail which tier fired. Bounded length: descriptor capped at 80 chars, scene_hint at 60 chars.
- **Verify:** AST + full pytest green. **Real-run acceptance (pending):** with Phase G singleton lookup feeding the CURRENT ledger, the radio prompt should now log `branch=gen_params_initial.style` and the radio should render as "space opera epic radio broadcast unit, set in derelict orbital lab, ..."
- **Tags:** flux, radio-bookend, story-arc, fallback-chain, qa-pass-2026-05-03

---

### BUG-LOCAL-023 [FIXED]: ANNOUNCER portrait wasted FLUX context + skewed scene composition
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes
- **Symptom:** Jeffrey caught mid-soak: `LLMDirector` generates a `portrait_prompt` for ANNOUNCER under `visual_plan.characters`, then `OTR_VideoPlan.compose_shot_prompt` concatenates ALL character portraits into every scene's PASS3 visual prompt. The announcer is never on screen as a person (BUG-LOCAL-129b: routed to Kokoro voice + radio bookend visual; HuMo skips them). Including their portrait wastes FLUX prompt budget AND skews scene composition by forcing every shot to fit an extra character (50yo silver-haired woman in flight gear).
- **Cause:** Visual_plan.characters was generated for every speaker without a "appears on screen?" filter. PASS3 compose treats the dict as canonical.
- **Fix (belt-and-suspenders, two layers):**
  1. **LLMDirector prompt rule** in `nodes/story_orchestrator.py` (VISUAL PLAN RULES section): explicit instruction "EXCLUDE narrator/announcer roles from visual_plan.characters. The ANNOUNCER (and any voice that only narrates without appearing on screen) must NOT be included under visual_plan.characters{}. Their voice mapping still belongs in voice_assignments{}; only visual_plan.characters skips them." Catches it at the source.
  2. **`OTR_VideoPlan` filter** in `nodes/otr_video_plan.py:438`: new `NON_VISUAL_ROLES = {"ANNOUNCER", "NARRATOR"}` set; before composing portraits, partition `chars_dict.keys()` into `all_char_names` (visible roles) and `_skipped_non_visual` (logged as info). Catches future LLM regressions where layer 1 fails. Honors explicit `focus_character` requests for non-visual roles (lets a debugging workflow request the announcer portrait specifically).
  Audio is unaffected: `voice_assignments.notes` is a SEPARATE field that audio nodes (Bark/Kokoro) consume; `portrait_prompt` doesn't feed audio at all.
- **Verify:** AST + full pytest (1150 / 8 / 1 in 131.62s) green. **Real-run acceptance (pending):** scene visual prompts in `[BatchFluxRender] shot N/M:` log lines should NOT lead with "Female, 50s, gravelly voice..." when there's an ANNOUNCER in the cast; should see `OTR_VideoPlan: skipped non-visual role(s) from portrait composition: ANNOUNCER` log line.
- **Tags:** flux, announcer, visual-plan, scene-composition, qa-pass-2026-05-03

---

### BUG-LOCAL-022 [FIXED]: BatchHumoRender stem-swap is mathematically broken when safe_title[:40] truncates the title
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** `BatchHumoRender._load_ledger_with_path` (line 1791-1865 pre-Phase-G) takes a `.mp4` path input from `SignalLostVideoRenderer` and derives the ledger via stem swap (`<file>.mp4` → `<file>_ledger.json`). When `video_engine.py:1450` truncates the procgen mp4 filename via `safe_title = "...".replace(...)[:40]`, the resulting mp4 stem may NOT equal the canonical `episode_id`. Stem swap looks for a ledger that doesn't exist. Combined with BUG-LOCAL-020 (mp4 in legacy dir), the failure mode is "derived ledger from .mp4 not found". Even after BUG-LOCAL-020 fix puts the mp4 in the per-episode dir, stem swap can still fail if title truncation drops characters.
- **Cause:** Discovery coupled to mp4 filename instead of the on-disk per-episode workspace structure (`output/otr/episodes/<ep>/audio/<ep>_ledger.json`).
- **Fix:** Add Tier 0 layout-aware lookup BEFORE the legacy stem-swap tiers in `_load_ledger_with_path`. Detection: `audio_dir.name == "audio"` AND `audio_dir.parent.parent.name == "episodes"`. When detected, the parent dir name IS the `episode_id` by construction. Try canonical `<ep_dir_name>_ledger.json` first; fall back to globbing `*_ledger.json` (non-pending) in the same audio_dir if the slug rule doesn't match. Decoupled from mp4 stem entirely. Legacy stem-swap tiers preserved as fallback for old artifacts in the legacy flat layout.
- **Verify:** AST + full pytest suite (1149 / 8 / 2 in 112s) green. **Real-run acceptance (pending):** re-queue the same workflow JSON; BatchHumoRender should log `Phase G layout-aware ledger lookup: <ep>_ledger.json` and proceed past the prior crash point.
- **Tags:** humo, ledger-discovery, layout-aware, stem-swap, qa-pass-2026-05-03

---

### BUG-LOCAL-021 [FIXED]: Audio-side nodes used global mtime walker for write-back (latent BUG-LOCAL-014 wrong-episode shape)
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Seven sites in the audio-side write-back chain used `_otr_ledger.find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])` to locate the in-flight ledger for write-back: `musicgen_theme.py:98+494`, `batch_audiogen_generator.py:58+511`, `batch_bark_generator.py:703`, `scene_sequencer.py:920+1257`, `audio_enhance.py:436`. Plus `visual/batch_flux_render.py:641` used the same walker (with the wrong dirs — `otr_audio_dir()` returns `_legacy_audio` when called with no episode_id, so it never even scanned the per-episode tree). On the 2026-05-02 soak, FLUX radio bookend stamped to `signal_lost_signal_abyss_20260426_161737` (a 6-day-old episode) instead of the in-flight `signal_lost_cramped_cargo_bay_vibrating_20260502_220824`. Same wrong-episode shape as BUG-LOCAL-014 — Phase A fixed it for `rtx_upscale.py` only; the rest of the codebase had it latent.
- **Cause:** Mtime-based discovery is fundamentally racy across queue boundaries and across runs. The `_CURRENT` Ledger singleton (set by `LLMScriptWriter` via `new_ledger()`) tracks the in-flight episode by construction; ComfyUI sequential queue + LLMScriptWriter's `IS_CHANGED = time.time()` guarantee the singleton is fresh on every queue invocation.
- **Fix:** Add `_otr_ledger.in_flight_ledger_path()` helper that reads the singleton's `path` (which advances correctly through `Ledger.rename_episode` per Phase B) and falls back to the legacy mtime walker only if the singleton is somehow unavailable. Sweep all 7 audio-side sites + the FLUX radio bookend site to use the helper. Late-import via try/except inside the helper avoids circular import with `production_ledger.py`. The walker is preserved (for `post_audio_video_pipeline.py:126` empty-input fallback and for the helper's own last-resort path).
- **Consult sources:** `docs/2026-05-03-phase-g-path-reorg-blast-radius__01_chatgpt.md` (gpt-5.5, 117.8s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 46.1s — caught critical "ComfyUI cache trap" risk for singleton; verified-already-mitigated by LLMScriptWriter's IS_CHANGED), `__03_nvidia.md` (mistral-nemotron, 190.7s).
- **Verify:** Phase G AST + full pytest (1149 / 8 / 2 in 112s) green. **Real-run acceptance (pending):** re-queue; FLUX radio bookend should stamp to the CURRENT episode_id, not a stale leftover. Two-episode soak: B's audio nodes should write to B's ledger, not A's.
- **Tags:** ledger-discovery, singleton, find_most_recent_ledger, wrong-episode, defensive-sweep, qa-pass-2026-05-03

---

### BUG-LOCAL-020 [FIXED]: video_engine.py procgen mp4 written to legacy `output/otr/audio/` instead of per-episode workspace (SOAK BLOCKER)
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Live soak crash on 2026-05-02. After 12 minutes of pipeline progress (LLM ladder + audio cascade + procgen video), `BatchHumoRender` crashed with `RuntimeError: BatchHumoRender: derived ledger from .mp4 not found: C:\...\output\otr\audio\signal_lost_cramped_cargo_bay_vibrating_20260502_220824_ledger.json (also tried collapsed-underscore variant + directory scan in C:\...\output\otr\audio)`. The procgen mp4 was at `output/otr/audio/<file>.mp4` (legacy flat) but the ledger had been moved to `output/otr/episodes/<ep>/audio/<ep>_ledger.json` by Phase B's `Ledger.rename_episode`. Stem swap looked in the wrong dir and found nothing.
- **Cause:** `video_engine.py:1443-1446` (pre-Phase-G) hardcoded `out_dir = .../output/otr/audio` — the legacy flat layout. The path reorg (Phases A/B/E) moved every per-episode asset under `output/otr/episodes/<ep>/audio/`, but `video_engine.py` was missed. The mp4 ended up OUTSIDE the per-episode tree, so `Ledger.rename_episode` (which renames the parent dir) couldn't move it along with the rest of the workspace.
- **Fix:** Read `out_dir` from `get_ledger().out_dir` (the in-flight Ledger singleton's audio path = `episodes/pending_<ts>/audio/` at this point). Write the procgen mp4 there. After `Ledger.rename_episode(<ep_id>)` moves the parent dir, recompute `final_out_path = Path(led.out_dir) / pending_out_path.basename` and verify it exists. Update the `subfolder` hint in the ComfyUI UI return value to the post-rename relative path. Defensive try/except wraps the ledger lookup so test/headless paths fall back to the legacy `output/otr/audio/` location.
- **Consult sources:** Same as BUG-LOCAL-021. ChatGPT specifically caught the post-rename stale-path risk; Gemini caught the `safe_title[:40]` truncation issue (addressed in BUG-LOCAL-022).
- **Verify:** Phase G AST + full pytest green. **Real-run acceptance (pending):** re-queue the same 30-word smoke; expect `[Video] Saved: <abs path under episodes/pending_<ts>/audio/>` followed by `[Ledger] per-episode dir moved pending_<ts> -> <ep_id>` followed by `[Video] post-rename mp4 path: <pending> -> <final>` (if recompute fired) and BatchHumoRender progress past the prior crash point.
- **Tags:** path-reorg, video-engine, soak-blocker, post-rename, qa-pass-2026-05-03

---

### BUG-LOCAL-019 [FIXED]: Sprint 1 full-suite acceptance — pre-existing test rot (Phase B fallout + per-episode reorg fallout)
- **Date:** 2026-05-02 | **Phase:** Sprint 1 acceptance | **Bible candidate:** no (test-only fixes, no behavior change)
- **Symptom:** `python -m pytest tests/` (Sprint 1 acceptance line item) ran to completion in 113s but with **5 failures** in two distinct clusters. The original BUG-LOCAL-006 hang at `TestDropdownsHaveEffect::test_creativity_produces_different_temps` was already resolved by intervening conftest work — that test now passes in 10s standalone — but other latent failures had been masked because the three explicit suites used in Phase A→E regression (Bug Bible, dropdown_guardrails, core) didn't include the ones that broke.
- **Cluster 1 (2 failures, `tests/test_production_ledger.py`):** `TestLedgerBeats::test_rename_updates_path_and_data` and `TestDualLedgerFix::test_rename_episode_moves_file_on_disk` both raised the Phase B (BUG-LOCAL-015) hard-fail RuntimeError "both source and destination episode directories exist". Root cause: the tests passed `tmp_path` directly as the audio dir to `Ledger(...)`. Phase B's `rename_episode` walks `os.path.dirname` up two levels to find the per-episode root — from `tmp_path = pytest-of-jeffr/pytest-NNN/test_K/`, that walked up to `pytest-of-jeffr/`, then constructed `new_ep_dir = pytest-of-jeffr/signal_lost_black_sphere_20260424_142006`. That destination accumulated across pytest sessions (siblings from prior runs of the same test pollute the user's TEMP root), so on any run after the first the conflict guard fired correctly. The TESTS were buggy — they assumed the pre-Phase-B silent split-state recovery and depended on global TEMP pollution that no longer works under the hard-fail invariant.
- **Cluster 2 (3 failures, `tests/test_radio_still_resolver.py`):** `TestFilesystemFallback::test_filesystem_fallback_finds_by_episode_id`, `TestFilesystemFallback::test_filesystem_fallback_when_ledger_path_stale`, `TestBug121Hardening::test_zero_byte_file_falls_through_to_layer3` all failed with `TestX.<locals>.<lambda>() takes 0 positional arguments but 1 was given`. Root cause: `monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)` mocks the helper with a 0-arg lambda, but the per-episode workspace reorg (2026-05-02 EVENING, BUG-LOCAL-033) gave `otr_stills_dir` an `episode_id` parameter. Production code calls `otr_stills_dir(episode_id)` with 1 arg. The 7 fallback tests that DON'T trigger this path passed silently; the 3 that do reach it failed.
- **Cause summary:** Cluster 1 = direct fallout from Phase B's hard-fail invariant correctly rejecting test setups that depended on the buggy old behavior. Cluster 2 = stale test mock signatures from the per-episode workspace reorg (BUG-LOCAL-033 era). Both pre-existing, surfaced because no one had run `pytest tests/` to completion since they were introduced.
- **Fix:** Cluster 1 — both failing tests now build a proper `tmp_out/episodes/<ep>/audio/` per-episode dir before instantiating the `Ledger`. The rename invariant has clean room to operate; no global TEMP pollution. Cluster 2 — `monkeypatch.setattr(..., lambda *a, **kw: tmp_path)` (10 sites updated via `replace_all`). Variadic tolerates the new `(episode_id)` arg without changing test semantics.
- **Verify:** Targeted suite (`pytest tests/test_production_ledger.py tests/test_radio_still_resolver.py -v`) — 76 passed in 1.87s. Full suite (`pytest tests/ -q --ignore=tests/v2`) — **1126 passed / 7 skipped / 0 failed in 113.28s**. The 107 errors in an earlier run were transient pytest tmp_path session race (`pytest-264` got reaped while a parallel pytest invocation was still using it); not reproducible on clean runs.
- **Promotes BUG-LOCAL-006 from [PARTIAL] to [FIXED]:** the conftest CUDA mask works, AND the originally-blamed `test_creativity_produces_different_temps` now passes (cause was either incidentally fixed by Phase B/C/D/E work or transient under a specific environment that no longer reproduces). Sprint 1 acceptance line "python -m pytest tests/ runs to completion green" is now satisfied — net cumulative count: 1126 / 7 / 0 across the full directory.
- **Tags:** test-rot, phase-b-fallout, per-episode-reorg-fallout, sprint-1-acceptance, no-active-bug

---

### BUG-LOCAL-006 [FIXED, was PARTIAL]: pytest hang at session-start when ComfyUI on same GPU
- **Date:** 2026-05-02 PM EVENING (re-verified) | **Phase:** 0 (test infra) | **Bible candidate:** yes
- **Update on the prior PARTIAL status:** `tests/conftest.py` (committed earlier this session) sets `CUDA_VISIBLE_DEVICES=""` + `OTR_TEST_MODE=1` at module import, registers the `requires_cuda` marker, auto-skips marked tests when CUDA is masked. The original PARTIAL note flagged `TestDropdownsHaveEffect::test_creativity_produces_different_temps` as still-hanging. Re-verified 2026-05-02 PM: that test now passes in 10s standalone, and the full directory `pytest tests/` runs to completion in 113s (1126 / 7 / 0). Either the hang was incidentally fixed by Phase A→E work (path reorg + Phase B's atomic rename + cache key cleanup may have removed a fixture that touched a heavy import), or it was transient under a specific environment. No further bisect needed; the acceptance gate is satisfied.
- **Verify:** `python -m pytest tests/ -q --ignore=tests/v2` → 1126 passed / 7 skipped / 0 failed in ~113s. With ComfyUI Desktop up on `:8000`, same result.
- **Tags:** test-infra, cuda-context, comfyui-cohabit, bible-candidate, was-partial

---

### BUG-LOCAL-018 [FIXED]: Ledger schema bump l3-2026-05-02 + meta.paths block
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** No active bug — additive schema enhancement. The QA pass (`docs/2026-05-02-rtx-upscale-qa-pass.md` Phase E) prescribed adding a `meta.paths` block to the production ledger so downstream nodes can look up canonical episode dirs without reconstructing them from `episode_id`. Slug-reconstruction was the root cause of the BUG-LOCAL-014/015/017 cluster — Phases A/B/C/D fixed the live instances; Phase E removes the temptation entirely by stamping the absolute, on-disk-truth paths into the ledger at every save.
- **Cause:** N/A — preventive change.
- **Fix:**
  - `nodes/_otr_ledger.py`: bump `CURRENT_SCHEMA_VERSION` from `l3-2026-04-28` to `l3-2026-05-02`. Add `_build_meta_paths(ledger_path, episode_id)` helper that detects layout (per-episode-workspace under `output/otr/episodes/<ep>/audio/<ep>_ledger.json` vs legacy flat under `output/audio/<ep>_ledger.json`) and stamps an appropriate `meta.paths` block. `save_ledger_safe` now calls it on every write. The block is **resolved fresh on every save** from the actual on-disk path, so it self-corrects after `Ledger.rename_episode` (Phase B) — no caller has to update it.
  - `nodes/production_ledger.py`: `Ledger.save()` also stamps `meta.paths` (via the same `_otr_ledger._build_meta_paths` helper) so the path data is consistent regardless of which write path produced the ledger. Hardcoded fallback `SCHEMA_VERSION` updated to match. Best-effort try/except wraps the meta stamp — a stamping failure must NEVER break the actual ledger write.
  - `docs/ledger_schema.md`: created. Documents the schema (top-level fields + meta block + meta.paths block + per-episode vs legacy-flat layout shapes), the lineage table, the reader contract (`dict.get(...)` not direct subscript), and the rules for downstream nodes.
- **Verify:**
  - AST + Phase E invariant guards (schema string `l3-2026-05-02`, `_build_meta_paths` present, both layouts detected, dual-write stamping in both files) — green.
  - **New `tests/test_meta_paths.py` — 13 passed.** Covers: per-episode layout detection + all 6 dirs stamped + obs_final stamped when obs/ exists + ledger_path absolute; legacy flat layout detection + no fabricated subdirs + minimal paths only; `save_ledger_safe` stamps meta.paths AND preserves pre-existing meta keys; old ledger without meta.paths loads cleanly via `dict.get(...)` (back-compat regression); `Ledger.save()` stamps meta.paths too; **after `rename_episode`, the next save's meta.paths self-corrects to the new dir** (the killer property — proves stale references can't accumulate).
  - Three CLAUDE.md regression suites + all phase-A-through-D tests: **234 passed / 1 skipped / 2 xfailed in 106.37s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3 + cache_key_mutations 30 + meta_paths 13).
  - **Real-run acceptance (pending):** end-of-stack soak. New ledgers should carry `meta.paths`; old ledgers (if any survive in `output/audio/`) still load via `dict.get` defaults.
- **Tags:** schema, additive, meta-paths, back-compat, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase E section). No round-robin needed — additive only, no behavior change for existing readers (all already use `meta.get(...)`).
- **Reader contract enforced:** see `docs/ledger_schema.md` "Reader contract" section. `meta.paths` MUST be accessed via `led.get("meta", {}).get("paths", {}).get(field)`, never `led["meta"]["paths"][field]`.

---

### BUG-LOCAL-017 [FIXED]: MusicGen + AudioGen cache miss every run — `_cache_key` returned a fresh timestamped path
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Two files (`nodes/musicgen_theme.py`, `nodes/batch_audiogen_generator.py`) had identical structural bugs in their cache logic. `_cache_key()` returned a filename with the *current* millisecond timestamp baked in (`<role>_<sha8>_<ts_ms>.wav`). The call site immediately checked `os.path.exists(cache_path)` against that exact path — which never existed because the timestamp was "now". Result: cache miss every single run, ~22s of wasted MusicGen rendering per episode + N seconds of wasted AudioGen rendering per SFX cue, AND a Rule C7 violation because each run wrote a different timestamped filename, and FFmpeg embeds input WAV filenames in MP4 metadata streams → final mp4 bytes drifted between identical-input runs.
- **Cause:** Single function (`_cache_key`) tried to do two incompatible jobs: produce a deterministic identity for cache lookup AND a unique filename for write. The docstring explicitly described the timestamp as "guaranteed unique across episodes" but that defeated the entire cache.
- **Fix:** Split lookup identity from write filename in both files:
  - **`_cache_prefix(...)`** — deterministic identity prefix (`<role>_<sha8>` for MusicGen, `sfx_<safe_name>_<sha8>` for AudioGen). No timestamp.
  - **`_cache_filename_for_write(...)`** — canonical write filename (`<prefix>.wav`). No timestamp. Same inputs always land at the same filename → byte-identical mp4 metadata across runs (Rule C7 holds even on clean-cache runs).
  - **`_cache_key(...)`** — back-compat alias, returns the canonical write filename.
  - **`_find_cached(cache_dir, prefix)`** — two-level lookup: canonical `<prefix>.wav` first, fallback to legacy `<prefix>_<ts>.wav` files for back-compat with existing on-disk caches. Uses `iterdir() + startswith()` (per Phase D Gemini consult — `Path.glob()` chokes on `[` in filenames). Sorts legacy matches by parsed filename timestamp (not mtime; mtime is unstable across copy/restore).
  - **`_save_wav` made atomic** — writes through sibling `.tmp` then `os.replace()` (Phase D Gemini consult: prevents corrupted cache hits if process is killed mid-write). Explicit `format="WAV"` because soundfile can't infer format from `.tmp` extension. Cleanup of orphan `.tmp` on failure.
- **Verify:**
  - AST + 7 invariant guards per file (function presence, atomic write, iterdir-not-glob, etc.) — green.
  - **New mutation suite `tests/test_cache_key_mutations.py` — 30 passed in 2.87s.** 5 MusicGen mutations + 5 AudioGen mutations + 12 lookup tests (canonical-wins, legacy-fallback, newest-timestamp-wins, no-cross-prefix-match, glob-metachar-tolerance) + 2 atomic-write tests + 2 cache_key back-compat tests + 4 atomic-failure tests. Confirms: every identity dimension produces fresh sha; the cosmetic AudioGen `safe_name` is NOT used as identity (full-prompt change beyond first 20 chars still produces fresh sha); `Path.glob` would have failed on `[`-containing prefix but `iterdir+startswith` works.
  - Three CLAUDE.md regression suites: **221 passed / 1 skipped / 2 xfailed in 110.36s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3 + cache_key_mutations 30).
  - Existing on-disk timestamped cache files transparently start hitting after deploy (legacy fallback path).
  - **Real-run acceptance (pending):** two consecutive identical-input runs should produce one file per `(role, sha)` pair; second run should log `CACHE HIT` with the canonical `.wav` name. End-of-stack soak covers this together with Phases A, B, C.
- **Tags:** cache, c7-byte-identity, ffmpeg-metadata, atomic-write, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-phase-d-cache-key-consult__01_chatgpt.md` (gpt-5.5, 108.1s — proposed strong-form deterministic write), `docs/2026-05-02-phase-d-cache-key-consult__02_gemini.md` (gemini-3.1-pro-preview-customtools, 32.0s — caught atomic-write requirement and `Path.glob` bracket bug), `docs/2026-05-02-phase-d-cache-key-consult__03_nvidia.md` (llama-3.3-nemotron-49b, 127.0s — confirmed all decisions). All three converged unanimously: drop timestamp on writes, two-level lookup, iterdir+startswith, atomic write, defer model_name digest expansion.
- **Deferred to v2 cache-key migration (separate scope):** add `model_name`, `sample_rate`, `decode_mode`, `guidance_scale` to the digest payload. Today these are effectively constants per-file but if the user starts varying them at runtime, cache identity will be wrong until v2 lands.

---

### BUG-LOCAL-016 [FIXED]: Filename pattern audit — slug-reconstruction regression guard
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** No active bug — this is a regression guard. The QA pass (`docs/2026-05-02-rtx-upscale-qa-pass.md` Phase C) prescribed an audit of all `nodes/` files for the dangerous anti-pattern: code constructing `f"{ep_id}_..."` to *find* or *delete* a file on disk. The actual on-disk filenames for cache files (musicgen, audiogen wavs) follow the format `<role>_<sha>_<ts>.wav` — produced by the writer, indexed by sha. Slug-reconstruction-for-discovery breaks every time the producer's naming convention diverges from the slug.
- **Cause:** Phase A and Phase B already absorbed the live instances of this anti-pattern (rtx_upscale spacesaver and production_ledger sidecar rename now use `audio_dir.glob(...)`). What remained was the risk of *future* drift — someone reintroducing slug reconstruction in a discovery path without realizing the cache filenames don't match the slug.
- **Fix:** Audit complete (0 substantive code changes). The remaining `f"{ep_id}.mp4"` and similar usages in the codebase are all canonical writer/reader pairs sharing a contract by construction (RTXUpscale OBS-existence guard ↔ VideoComposite mp4 writer; Ledger class authoring `<ep>_ledger.json`). New regression test `tests/test_filename_pattern_audit.py` codifies the rule:
  - **`test_no_audio_cache_slug_reconstruction`** — static-analyzes all `nodes/*.py` for banned patterns: `audio_dir / f"opening_{ep_id}.wav"`, `audio_dir / f"sfx_{ep_id}_..."`, etc. Will fail loudly on any future drift.
  - **`test_destructive_paths_use_glob_not_reconstruction`** — positive assertion that the rtx_upscale spacesaver (Phase A) and ledger sidecar rename (Phase B) still use glob discovery; if a refactor accidentally replaces `audio_dir.glob("*_treatment.txt")` with slug reconstruction, this test catches it.
  - **`test_allowlist_entries_still_present`** — every entry in the test's ALLOWLIST (legit canonical writer/reader pairs) must still resolve to a real source line. Stale entries surface for pruning instead of silently shielding future drift.
- **Verify:**
  - 3/3 audit tests pass in 1.60s.
  - Combined regression: **191 passed / 1 skipped / 2 xfailed in 98.00s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3).
- **Tags:** audit, regression-guard, slug-reconstruction, qa-pass-2026-05-02, no-active-bug
- **Consult sources:** `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase C section, table of canonical writers/lookups). No round-robin needed — mechanical audit, no determinism implications.

---

### BUG-LOCAL-015 [FIXED]: production_ledger treatment rename gap + os.replace silent split state
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after real two-episode soak run with Phase A)
- **Symptom:** Two adjacent bugs in `Ledger.rename_episode` (`nodes/production_ledger.py`):
  1. **Treatment file rename gap (Finding 2 from QA pass).** The function moved the per-episode dir and renamed `pending_<ts>_ledger.json` → `<new_id>_ledger.json` but did NOT rename `pending_<ts>_treatment.txt` → `<new_id>_treatment.txt`. The treatment file (written early by `OTR_LLMScriptWriter` before the title is finalized) sat in the new dir with the old prefix. Phase A's spacesaver kept it via a defensive `glob("*_treatment.txt")`, but that defensive measure was a workaround.
  2. **`os.replace` fallback silent split state (Finding 3 from QA pass).** When the dir-move `os.replace(old_ep_dir, new_ep_dir)` failed (Windows Defender lock, indexer holding a handle, partial dir from a prior crash, **or destination dir already existing — `os.replace` ALWAYS fails on Windows when destination dir exists, even empty**), the code logged a warning and continued. It updated `self.episode_id` and `self.data["episode_id"]` to the new id but left `self.out_dir` pointing at the old path. The next `self.save()` wrote a finalized-id ledger into the OLD dir while every downstream node (BatchHumoRender, VideoComposite, RTXUpscale) built paths from the new id. Net effect: confusing "file not found" cascades far away from the rename failure.
- **Cause:** Single function trying to advance in-memory state regardless of on-disk success. Missing state-matrix handling for: both dirs exist; both missing; old missing + new exists. No retry on transient Windows locks. Treatment files outside the rename loop. Filename-construction slug not consistent between ledger and treatment paths.
- **Fix:** Rewrite `Ledger.rename_episode` around a strict invariant: **either complete with canonical episode dir + canonical ledger + canonical treatment, OR raise BEFORE mutating in-memory episode state.** Specifics:
  - Case-insensitive `os.path.normcase` same-path early-return (no-op for case-only changes on Windows).
  - State matrix: `(old_exists, new_exists)` resolved into one of {happy retry path, conflict raise, idempotent recovery, both-missing raise} BEFORE any mutation.
  - 3 × 0.5s inline retry on `os.replace(old_ep_dir, new_ep_dir)` with attempt-aware logging. After the third failure: `RuntimeError` with message that explicitly tells the user to check for files open in Notepad / VLC / Explorer preview / editors (per Gemini consult — system locks clear in ms but human-held locks need user intervention).
  - In-memory state (`episode_id`, `data["episode_id"]`, `out_dir`) only advances AFTER dir is in final on-disk position.
  - Ledger file rename (best-effort warn-only, dir invariant already satisfied).
  - Treatment + sidecar rename: glob `<old_slug>_*.txt` (NOT `pending_*` — narrower, no risk of catching unrelated files), rename each to `<new_slug>_*.txt`. Uses the same `_slugify(..., limit=120)` as the ledger path. Per-file warn-only on failure.
- **Verify:**
  - AST + 9 invariant guards (hard-fail message, retry sleep, conflict check, both-missing check, sidecar glob, slug consistency, normcase, etc.) — green
  - **New targeted suite `tests/test_ledger_rename.py` — 10 passed in 1.78s.** Covers: happy path renames dir+ledger+all sidecars; same-id no-op; conflict raises; both-missing raises; idempotent recovery (old missing + new exists); dir-move retries 2/3 then succeeds; dir-move fails 3/3 → RuntimeError + state unchanged; error message mentions human-held locks; treatment failure does not raise; sidecar glob uses old-id prefix not pending wildcard.
  - Three CLAUDE.md regression suites: Bug Bible (23 passed / 1 skipped / 2 xfailed), `tests/test_dropdown_guardrails.py` + `tests/test_core.py` (155 passed). Total **178 passed / 1 skipped / 2 xfailed in 101.62s**.
  - **Real-run acceptance (pending):** kill mid-rename (Ctrl-C between treatment write and rename), restart, confirm clean recovery and no orphan `<old>_*.txt` after the next successful run. Two-episode-in-flight soak covers Phase A + B together.
- **Tags:** ledger, rename, atomicity, windows-replace, retry, slug-consistency, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-phase-b-rename-consult__01_chatgpt.md` (gpt-5.5, 150.8s), `docs/2026-05-02-phase-b-rename-consult__02_gemini.md` (gemini-3.1-pro-preview-customtools, 31.9s — caught the critical "Windows os.replace always fails on existing dest dir, even empty"), `docs/2026-05-02-phase-b-rename-consult__03_nvidia.md` (llama-3.3-nemotron-super-49b-v1.5, 66.7s)

---

### BUG-LOCAL-014 [FIXED]: Spacesaver wrong-episode wipe via global mtime ledger scan
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after real two-episode run)
- **Symptom:** `_spacesaver_cleanup_if_flagged` in `nodes/rtx_upscale.py` discovered the ledger to read the `meta.perfect_run_spacesaver` flag from by calling `_otr_ledger.find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])`. That walker returns the newest `*_ledger.json` by mtime across the **entire** `otr/episodes/` tree. If Episode A is mid-RTXUpscale when Episode B is queued and writes its pending ledger, A's spacesaver pass would discover B's ledger, derive `ep_dir = ledger.parent.parent` (B's tree), and wipe B's `stills/`, `portraits/`, `videos/`, `composited/` while B was still rendering.
- **Cause:** Use of a global mtime-based discovery in a destructive code path. The existing substring sanity guard (`"episodes" in parts and "otr" in parts`) only verified the wiped tree was *somewhere* under `otr/episodes/`, not that it was the **right** episode for the current RTXUpscale call.
- **Fix:** Derive `ep_dir` directly from the `src` argument the upstream node already passes in. `src` is always `otr/episodes/<ep>/composited/<ep>.mp4` (the VideoComposite output), so `src.resolve().parent.parent` is the episode root by construction. Replace substring guard with `ep_dir.relative_to(otr_episodes_root().resolve())` plus a `len(rel.parts) == 1` depth-1 invariant. Load the ledger from THIS episode's `audio/*_ledger.json` glob, prefer non-pending. Add an OBS-existence precondition (`otr/obs/<ep>.mp4` must exist on disk) so spacesaver refuses to fire if the run order ever flips and the final deliverable hasn't landed yet. Build the keep-list from real on-disk filenames (`audio_dir.glob("*_treatment.txt")` plus the discovered ledger path) so a slug mismatch between `ep_id` and the on-disk filename can't accidentally delete the ledger or treatment.
- **Verify:**
  - AST + Bug Bible regression (23 passed / 1 skipped / 2 xfailed) + `tests/test_dropdown_guardrails.py` + `tests/test_core.py` (155 passed in 107.84s) all green post-fix.
  - Source no longer references `find_most_recent_ledger` from the spacesaver path (verified by grep + AST sanity script).
  - **Real-run acceptance (pending):** queue Episode A, queue Episode B before A's RTXUpscale fires; inspect `[OTR_RTXUpscale] spacesaver:` log lines and confirm `ep_dir` resolves to A's path, never B's. Bypass-safety run with `src` outside `otr/episodes/` should log `refusing destructive cleanup` with no deletion. Delete `otr/obs/<ep>.mp4` before cleanup fires and confirm the new precondition aborts.
- **Tags:** spacesaver, ledger, two-episode, destructive-cleanup, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-path-reorg-spacesaver-qa__01_chatgpt.md`, `docs/2026-05-02-path-reorg-spacesaver-qa__02_gemini.md`, `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase A)
- **Follow-up phases queued:** B (production_ledger.py treatment rename + os.replace fallback), C (slug-reconstruction sweep), D (cache-key timestamp drop), E (schema bump + meta.paths block)

---

### BUG-LOCAL-001: Pre-existing test infrastructure rot blocks `pytest tests/` regression baseline
- **Date:** 2026-05-02 | **Phase:** 0 (regression infra) | **Bible candidate:** yes
- **Symptom:** Running the canonical `python -m pytest tests/` cannot reach a clean green pass. Three distinct failure modes observed in one run:
  1. **8 collection ImportErrors** (`No module named 'otr_v2.visual'`) on:
     `tests/test_anchor_gen.py`, `tests/test_camera_path_determinism.py`,
     `tests/test_character_regression.py`, `tests/test_cold_open_canary.py`,
     `tests/test_episode_dry_run.py`, `tests/test_lhm_monitor.py`,
     `tests/test_three_minute_continuous.py`, `tests/test_visual_phase_a.py`.
     Without `--ignore=` flags, pytest aborts the run after collection (`Interrupted: 8 errors during collection`).
  2. **`tests/test_backend_dispatch.py` 14 failures** (`FFFFFFFFFFFFFF` in -q output). Not investigated yet.
  3. **`tests/test_dropdown_guardrails.py` deterministic hang** after the first 12 tests pass (`............` then no further progress for >2 min, until externally killed).
- **Cause:**
  1. `otr_v2/visual/` package was deleted in commit `7706660` ("Fix BUG-LOCAL-047: FLUX anchor dtype ladder"). Test modules that imported it were not updated or removed in the same commit.
  2. test_backend_dispatch failure mode unknown — needs investigation.
  3. test_dropdown_guardrails.py hang likely waiting on a network/model/subprocess fixture that no longer resolves; not yet bisected.
- **Fix:** **Pending — do NOT fix mid-test per ground rules.** Captured here as v2.0-beta era opening entry. Likely fix sequence (next session): (a) delete or rewrite the 8 stale visual test modules; (b) bisect dropdown_guardrails hang to identify the wedged test; (c) investigate backend_dispatch failures separately. Also note: CLAUDE.md references `tests/v2/test_audio_byte_identical.py` which doesn't exist (path is `tests/test_audio_byte_identical.py`); CLAUDE.md test-command block is stale.
- **Verify:** After fix, `python -m pytest tests/` runs to completion, no collection errors, all hangs resolved, backend_dispatch failures triaged (fixed or marked xfail with reason).
- **Tags:** test-infra, pre-existing, otr_v2-orphans, claude-md-staleness

### BUG-LOCAL-002 [FIXED]: `scripts/soak_operator.py` widget indices stale (drift since episode_title + num_characters added)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke harness) | **Bible candidate:** yes
- **Symptom:** `scripts/soak_operator.py` declares `WV_GENRE=1`, `WV_TARGET_WORDS=2`, `WV_CREATIVITY=11`, `WV_OPT_PROFILE=13`. Reading `nodes/story_orchestrator.py::OTR_LLMScriptWriter.INPUT_TYPES` shows the actual widget order is now: `[0]episode_title, [1]target_words, [2]num_characters, [3]model_id, [4]cleanup_model_id, [5]custom_premise, [6]include_act_breaks, [7]self_critique, [8]open_close, [9]target_length, [10]style, [11]style_custom, [12]creativity, [13]arc_enhancer, [14]optimization_profile`.
- **Cause:** `episode_title` and `num_characters` widgets were added to the script-writer node, plus `style_custom` and `arc_enhancer` were inserted. soak_operator constants were never updated. Anything calling `supersoaker.py::patch_workflow` writes to the wrong slots: `creativity` write lands on `style_custom` (string field, broken), `optimization_profile` write lands on `arc_enhancer` (boolean field, broken), `target_words` write lands on `num_characters`, and `WV_GENRE=1` writes target_words.
- **Fix (shipped 2026-05-02):** the entire WV_*-positional approach was retired. `scripts/soak_operator.py` is now a 100-line legacy shim that only retains the read-only `scan_treatment` helper kept for the test import. `scripts/supersoaker.py` was deleted. Canonical surface for talking to the ComfyUI HTTP API is now `scripts/otr_api.py`, which patches widgets BY NAME using the live `/object_info` schemas — eliminating the entire class of widget-position-drift bugs by construction. Confirmed in code 2026-05-05; BUG_LOG entry tag updated accordingly.
- **Verify:** `head -25 scripts/soak_operator.py` reads "LEGACY SHIM (BUG-LOCAL-002 fix, 2026-05-02)"; the WV_* constants are gone; `scripts/otr_api.py` exists as the name-based patcher.
- **Followup (separate, low priority — Jeffrey 2026-05-05):** build ONE good stable randomizer-soaker on top of `scripts/otr_api.py` for the current canonical workflow. Touches only validated widget options, never the dangerous ones. After higher-priority bug fixes ship.
- **Tags:** widget-drift, soak-harness, supersoaker, bible-candidate, fixed-via-rewrite

### BUG-LOCAL-003: ComfyUI Desktop launch does not inherit user-scope `HF_HOME`
- **Date:** 2026-05-02 | **Phase:** 0 (smoke harness) | **Bible candidate:** yes
- **Symptom:** First smoke queue (prompt_id `a455fc20-...`) failed in NewsCuration / NewsCurationDeep / NewsSummary phases with repeated `local_files_only=True failed for model (mistralai/Mistral-Nemo-Instruct-2407 does not appear to have files named ('model-00001-of-00005.safetensors', ...))` and `huggingface.co` connection-failed fallback. Each phase took the timeout (65s NewsCuration, 40s NewsCurationDeep) and never recovered. ComfyUI then created a fresh empty `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub\models--mistralai--Mistral-Nemo-Instruct-2407\` skeleton (timestamp 2026-05-02 00:40), proving it was looking at the wrong cache root.
- **Cause:** `HKCU\Environment` has `HF_HOME=C:\ComfyUI-Models\huggingface` (canonical, populated with all 5 Mistral-Nemo shards under `hub\models--mistralai--Mistral-Nemo-Instruct-2407\snapshots\04d8a905...\`). When ComfyUI Desktop is launched via `Start-Process` from a parent process that does NOT have `HF_HOME` already in its env (e.g. the Cowork sandbox), the Electron renderer + bundled Python backend inherit only the parent's env, NOT user-scope env vars. So huggingface_hub falls back to its default `~/.cache/huggingface/hub` (which on this machine is junctioned/aliased to `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub` — empty).
- **Fix (verified 2026-05-02):** Before launching `ComfyUI.exe`, explicitly set `HF_HOME=C:\ComfyUI-Models\huggingface` in the parent shell. Re-queue confirmed: `LLM tokenizer loaded from cache (no HTTP checks)` log line appeared, NewsSummary generated 2572 chars, ScriptWriter ran end-to-end @ 18.1 tok/s. **Permanent fix needed:** ComfyUI Desktop launcher should read user-scope HF_HOME via `winreg.OpenKey(HKEY_CURRENT_USER, "Environment")` at startup and inject into the Python child process's env. Until then, document the launch-via-elevated-shell-only pattern in README.
- **Verify:** `[00:48:59] LLM tokenizer loaded from cache (no HTTP checks)` in runtime log + ScriptWriter generation completes without `local_files_only=True failed`.
- **Tags:** comfyui-desktop, hf_home, env-inheritance, launch-pattern, bible-candidate

### BUG-LOCAL-004: OOM in OTR_LLMScriptWriter on 30-word ultra-smoke after parse-retry loop (peak 29.5 GB on 16 GB device)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke verification) | **Bible candidate:** yes
- **Symptom:** Smoke prompt_id `e6b87239-...` ran with `target_length="30 words (smoke, 1 act)"`, target_words=30, num_characters=2. Pipeline order: NewsSummary OK → ScriptWriter generated 571 tokens (parsed 0 scenes/0 lines/Characters: none) → OpenClose 3-outline evaluator (CHARACTER-DRIVEN, SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) all returned 0 chars and were DISCARDED → "OPENCLOSE: All outlines failed" → next `_generate_with_llm` call OOMed: `Allocation on device 0 would exceed allowed memory. Currently allocated: 26.53 GiB / Device limit: 15.92 GiB / Free (according to CUDA): 0 bytes`. Exception type `torch.OutOfMemoryError`, raised at `nodes/story_orchestrator.py:3137 model.generate()` from `nodes/story_orchestrator.py:5188 write_script._generate_with_llm`. Final VRAM_SNAPSHOT before OOM: `current_gb=8.275 peak_gb=29.498` — peak indicates cumulative KV cache + activations from successive calls were never released.
- **Cause:** Hypothesis (needs code dive): after each LLM `model.generate()`, the KV cache + intermediate activations are not torch.cuda.empty_cache()'d before the next call. With 4-bit NF4 weights ~7.5 GB resident, plus context_cap=16384 prompt tokens of KV cache (40 layers × 2 K/V × 16384 × hidden_dim × 2 bytes ≈ 6.4 GB) per call, four cumulative calls overflow. Compounded by the 30-word preset's parse-fail retry path (since 0 lines parsed, system retries) — each retry is another full forward pass without inter-call eviction.
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) audit `_generate_with_llm` and `_critique_and_revise` for an explicit `torch.cuda.empty_cache()` + KV-cache delete after every generate call; (b) add a hard parse-retry cap (e.g. ≤2) to prevent runaway retry on the ultra-smoke preset; (c) log the prompt-token count alongside `llm_generate_entry` snapshot so future OOMs can be bisected. CLAUDE.md says "use `_flush_vram_keep_llm()` between LLM phases" — verify that is actually being called between OpenClose synth and write_script retry.
- **Verify:** Re-queue 30-word ultra-smoke; expect peak_gb < 14.5 GB across full LLM ladder; expect `MAX_RETRIES_EXCEEDED` (graceful) instead of OOM if parse keeps failing.
- **Tags:** vram, oom, llm-cache, retry-loop, ultra-smoke, bible-candidate

### BUG-LOCAL-005: 30-word ultra-smoke ScriptWriter output unparseable (0 scenes / 0 dialogue lines / 0 characters from 571 tokens)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke verification) | **Bible candidate:** yes
- **Symptom:** With patched inputs `target_length="30 words (smoke, 1 act)"`, `target_words=30`, `num_characters=2`, the LLM (Mistral-Nemo-Instruct-2407 4-bit NF4) generated 571 tokens at 18.1 tok/s but the post-generation parser counted: `0 scenes | 0 dialogue lines | Characters: none`. Confirmed 30-word ULTRA-SMOKE preset was applied (history.current_inputs shows the patched values). OpenClose 3-outline evaluator (CHARACTER-DRIVEN, SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) also returned 0-char outputs — all 3 DISCARDED ("too short: 0 chars"), "OPENCLOSE: All outlines failed".
- **Cause:** Hypothesis: the new "30 words (smoke, 1 act)" preset's prompt does not enforce `CHARACTER:` / `SCENE:` markers (CLAUDE.md note: "BUG-007 root cause: Short (3 acts) prompt now explicitly enforces CHARACTER: dialogue format" — the same fix may not have been carried forward to the 30-word preset). The model generates prose that satisfies token count but lacks the structural markers the parser greps for. Also possible: the SPINE_MODE upper bound (commit `6454d91`, BUG-LOCAL-132) collides with `max_new_tokens=150` in OPENCLOSE such that nothing generated reaches the parser. Needs prompt-trace logging.
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) instrument `write_script` to dump the actual prompt + raw model output at TRACE level when parse yields 0 lines, so the format gap is visible; (b) port the BUG-007 CHARACTER:/SCENE: enforcement clause from the "short (3 acts)" prompt into the 30-word ultra-smoke prompt; (c) add a unit test that asserts the 30-word preset's compiled prompt contains the literal substrings `CHARACTER:` and `SCENE:`.
- **Verify:** Re-run 30-word ultra-smoke, expect `1 scene | 3 dialogue lines | 2 characters` parse and a non-empty ledger.
- **Tags:** prompt-format, ultra-smoke, parse-fail, bug-007-regression, bible-candidate

### BUG-LOCAL-006: `pytest tests/` hangs at session-start when ComfyUI is running on the same GPU
- **Date:** 2026-05-02 | **Phase:** 0 (test infra) | **Bible candidate:** yes
- **Symptom:** Running `python -m pytest tests/test_core.py tests/test_arc_check.py ... -q` while ComfyUI Desktop is up on `:8000` produces only the standard pytest banner ("test session starts", "platform win32...", "plugins: anyio...") and then hangs with the python.exe at ~2.7 GB RSS, no further output, no test names, for 90+ seconds. Killing the python.exe is the only way to recover. Same hang shape was observed during the baseline `pytest tests/` (which paused at `tests/test_dropdown_guardrails.py ............` mid-suite). Both behave identically: stable RSS, no I/O progress.
- **Cause:** Hypothesis: pytest collection imports OTR's `__init__.py` which transitively imports torch + transformers + bitsandbytes. ComfyUI already owns the CUDA primary context. Either bitsandbytes' `cuda_setup` or transformers' device-probe is stalling on CUDA-context-create while ComfyUI holds the device. Not yet bisected — could also be a network call (HF model resolver) or filesystem walk over `C:\ComfyUI-Models\huggingface` (1.85 TB free, deeply nested cache).
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) add an autouse conftest fixture that sets `CUDA_VISIBLE_DEVICES=""` for unit tests so collection never tries to bind to GPU; (b) move the OTR node imports out of the package's `__init__.py` top level into lazy load (already partially done for some modules, audit completeness); (c) document in CLAUDE.md that local pytest runs require ComfyUI to be killed first. Until then, regression baseline is unverifiable when ComfyUI is up.
- **Verify:** With ComfyUI killed, `pytest tests/test_core.py -q` runs to completion in <30 s. With ComfyUI up + the conftest CUDA-mask fixture in place, same.
- **Tags:** test-infra, cuda-context, comfyui-cohabit, bible-candidate

---

## 2026-05-02 fix landings

The five entries above (BUG-LOCAL-001 through 006, minus 002 which was logged separately) received fixes in this session's mega-commit. Status update per fix-lands-here pattern: `[FIXED]` with verify recipe.

- **BUG-LOCAL-003 [FIXED]** — `scripts/run_comfyui.cmd` reads `HF_HOME` + `HUGGINGFACE_HUB_CACHE` from `HKCU\Environment` via PowerShell, exports them, and launches `ComfyUI.exe`. README.md "Launching ComfyUI Desktop on Windows" section documents the pattern. Verify: kill ComfyUI, run `scripts\run_comfyui.cmd`, queue any episode that touches an HF model — expect `LLM tokenizer loaded from cache (no HTTP checks)` in `otr_runtime.log`.
- **BUG-LOCAL-004 [FIXED]** — `nodes/story_orchestrator.py` `write_script` short-episode branch (target_words ≤ 700) now (a) calls `_flush_vram_keep_llm()` before the main `_generate_with_llm` call so KV cache + activation peaks from prior LLM phases (NewsSummary, OpenClose 3-outline + evaluator, synthesizer) don't ride along into the main forward pass, and (b) wraps the call in a `MAX_PARSE_RETRIES = 2` loop with a cheap `[VOICE: ...]` / `CHARACTER:` marker count as the parseability check; on exhaustion logs `MAX_PARSE_RETRIES_EXCEEDED, accepting last output` and lets the parse-fail observability stamp it in the ledger instead of OOMing on a fourth forward pass. Verify: re-queue 30-word smoke; expect peak_gb < 14.5 GB across LLM ladder; if parse keeps failing, expect `MAX_PARSE_RETRIES_EXCEEDED` in `otr_runtime.log`, not `torch.OutOfMemoryError`.
- **BUG-LOCAL-005 [FIXED v2]** — `nodes/story_orchestrator.py` `write_script` now (a) detects `is_ultra_smoke` / `is_tiny_smoke` BEFORE `_open_close_expansion` is called and short-circuits the 3-outline evaluator entirely (round-robin verdict 2026-05-02: ChatGPT 5.5 + Gemini 3.1 + NVIDIA Nemotron 49B all flagged this as the actual root cause of the 29.5 GB-on-16-GB OOM since the evaluator holds 3 parallel KV caches at once); (b) clamps `max_new_tokens` to 256 for ultra-smoke and 384 for tiny smoke so a degenerate model output cannot run away to 571+ tokens; (c) swaps in the streamlined ULTRA_SMOKE prompt with explicit `[VOICE: ...]` enforcement; (d) replaces the original permissive `_bare_hits` regex with a negative-lookahead variant that excludes `TITLE:` / `SCENE:` / `GENRE:` / `ENV:` / `SFX:` / `MUSIC:` / `VOICE:` / `CAST:` / `AUTHOR:` so a "TITLE only" output cannot falsely PARSE_OK; (e) under ultra-smoke mode requires `=== SCENE N ===` AND `>=2 [VOICE: ...]` lines for PARSE_OK (strict-VOICE contract). The standard path keeps the looser scene-plus-any-marker check. `[VOICE: ...]` regex held strict per Gemini + NVIDIA: relaxing it would desync from the downstream parser and risk dropping audio lines, violating C7. Verify: queue 30-word smoke; expect peak_gb < 14.5 across LLM ladder, ledger has `1 scene | >=2 [VOICE: ...] dialogue lines | 2 named characters`.
- **BUG-LOCAL-006 [PARTIAL]** — `tests/conftest.py` created. Sets `CUDA_VISIBLE_DEVICES=""` + `OTR_TEST_MODE=1` at module import (before any `tests/test_*.py` collection), registers `requires_cuda` marker, auto-skips marked tests when CUDA is masked. The fix lets pytest progress further than baseline (24 dots in `test_dropdown_guardrails.py` vs 12 before) but the suite still hangs at `TestDropdownsHaveEffect::test_creativity_produces_different_temps`. The hang is INSIDE that test's call to `_run_preflight` -- not a CUDA-init issue, since CUDA is masked here. Likely root cause: a fixture or _generate_with_llm mock interaction that still touches a heavy import. Reproduce: `python -m pytest tests/test_dropdown_guardrails.py::TestDropdownsHaveEffect::test_creativity_produces_different_temps -q -s` (run alone). Next-session work: bisect the hang inside that test class -- this is a separate bug from the CUDA-context-create hang the conftest fixed, deserves its own follow-up entry.
- **BUG-LOCAL-001 [PARTIAL]** — 8 stale `otr_v2.visual` test collectors deleted (`test_anchor_gen.py`, `test_camera_path_determinism.py`, `test_character_regression.py`, `test_cold_open_canary.py`, `test_episode_dry_run.py`, `test_lhm_monitor.py`, `test_three_minute_continuous.py`, `test_visual_phase_a.py`) AND 10 sidecar-era tests in the same family (`test_backend_dispatch.py`, `test_wan21_loop.py`, `test_wall_clock_estimator.py`, `test_vhs_postproc.py`, `test_pulid_portrait.py`, `test_planner.py`, `test_ltx_motion.py`, `test_flux_keyframe.py`, `test_flux_anchor.py`, `test_florence2_sdxl_comp.py`). 38 test_*.py files remain (was 48 + 8 = 56; 18 deleted). This subsumes the original "14 `test_backend_dispatch` failures" entry — that file is gone. Verify: `python -m pytest tests/ --collect-only -q` reports zero `otr_v2.visual` collection errors.
- **BUG-LOCAL-002 [FIXED]** — `scripts/supersoaker.py` deleted. `scripts/soak_operator.py` slimmed from a 1500-line soak runner to a ~270-line legacy shim retaining only `scan_treatment` (used by `tests/test_treatment_scanner_unicode.py`). New canonical surface: `scripts/otr_api.py` exposes `load_workflow`, `fetch_schemas`, `patch_widget_by_name` (uses live `/object_info` schemas — robust against future widget reorders), `workflow_to_api_prompt` (port of soak_operator's BUG-LOCAL-027/029-fixed converter), `submit_prompt`, `poll_history`, `queue_snapshot`, `cancel_queue`. `scripts/queue_smoke.py` + `smoke_watcher.py` rebuilt on `otr_api`. `tests/test_widget_drift_guard.py` rerouted via private alias `mod._workflow_to_api_prompt = mod.workflow_to_api_prompt`. Verify: `python scripts/queue_smoke.py` produces a `/history` entry with `current_inputs.target_words=[30]`, `num_characters=[2]`, `target_length=["30 words (smoke, 1 act)"]`.

### Round-robin QA — 2026-05-02

`docs/2026-05-02-v2.0-beta-sprint-qa__01_chatgpt.md` (gpt-5.5), `__02_gemini.md` (gemini-3.1-pro-preview-customtools), `__03_nvidia.md` (mistral-nemotron-super-49b-v1.5), `__04_synthesis.md`, `__transcript.json`. All three external models converged on **BLOCK** for the initial Sprint 1 commit. Three must-fix items unanimously prescribed:

1. Move ultra-smoke / tiny-smoke detection BEFORE `_open_close_expansion` so the 3-outline evaluator's parallel KV caches don't run (Gemini calculated ~6 GB of cache from 3 simultaneous outlines = the actual source of the 29.5 GB peak; ChatGPT and NVIDIA both endorsed). **Applied.**
2. Replace the permissive `_bare_hits` regex with a structural-marker negative-lookahead so `TITLE:` / `GENRE:` etc. cannot falsely PARSE_OK; require `=== SCENE N ===` PLUS `>=2 [VOICE: ...]` lines for ultra-smoke. **Applied.**
3. Clamp `max_new_tokens` to 256 for ultra-smoke (384 for tiny smoke) so a runaway 571-token degenerate output cannot recur. **Applied.**

Disagreement caught: ChatGPT recommended relaxing the `[VOICE: ...]` regex to be more permissive. Gemini and NVIDIA both rejected this — relaxing the validation regex desyncs from the downstream parser and could silently drop dialogue lines, violating C7. Held the regex strict.

Non-blocking follow-ups for next session: investigate `live_ledger=True` for retained GPU tensors, audit `_generate_with_llm`'s explicit `del` of intermediates, write a unit test that mocks `_generate_with_llm` to return bad-then-good output and asserts exactly 2 attempts max, port any still-valid contracts (frame count `4n+1`, etc.) from the deleted sidecar tests into in-graph test coverage.

Post-fix regression: `python -m pytest tests/ --ignore=tests/test_dropdown_guardrails.py -q` → **932 passed, 6 skipped, 0 failed in 10.8 s**. AST-clean.

---

## 2026-05-02 — Sprint 3 mega-sprint (LTX wiring + RTX VSR upscale)

### BUG-LOCAL-007 [DEVIATION-LOGGED]: LTX 2B v0.9 bundled checkpoint forces CheckpointLoaderSimple-family loader

- **Date:** 2026-05-02 | **Phase:** S3.1 (LTX wiring) | **Bible candidate:** yes
- **Symptom:** ROADMAP Architecture Truth (locked 2026-05-02) specified `UNETLoader + CLIPLoader (T5) + VAELoader` for LTX 2B fp16, "NOT CheckpointLoaderSimple". Reason given: split-load lets ComfyUI offload T5/VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.
- **Cause:** Lightricks ships LTX 2B v0.9 ONLY as a bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB, all components in one safetensors file). No standalone LTX UNet / LTX VAE artifacts exist on the upstream HF repo for the 2B v0.9 line. The Architecture Truth assumed split files exist; they don't.
- **Fix:** Use ComfyUI-LTXVideo's `LowVRAMCheckpointLoader` for LTX 2B v0.9 (it IS a `CheckpointLoaderSimple` subclass, but adds a `dependencies` input that ComfyUI uses to force sequential load). The C2 sequencing intent (HuMo unloads before LTX claims VRAM) is satisfied via the dependency edge + the existing strict teardown in `batch_humo_render.py` (`unload_all_models + gc + empty_cache + cuda.synchronize` in finally). The "no carve-out for CheckpointLoaderSimple" rule was about preventing OOM from parallel-load on a hot cache; sequencing eliminates that risk.
- **Verify:** Log line `[OTR_HUMO] teardown complete` precedes `[OTR_LTX_LOADER] loading ltx-video-2b-v0.9.safetensors`. No `Allocation on device 0 would exceed allowed memory` between HuMo teardown and LTX render.
- **Promote-to-Bible-only-if:** a future LTX line (3B, 5B, 13B) ships with split UNet/T5/VAE artifacts AND we re-validate that LowVRAMCheckpointLoader's dependency edge keeps the C2 sequencing guarantee under 14.5 GB. Until then this stays as a documented OTR-local deviation.
- **Tags:** ltx, loader, c2-deviation, sequencing, deps-edge

### BUG-LOCAL-008 [FIXED]: LTX CFG=1.0 mathematically erases the negative prompt

- **Date:** 2026-05-02 (logged) / 2026-05-05 (fix verified + reinforced) | **Phase:** S3.1 (LTX wiring) | **Bible candidate:** yes
- **Symptom:** Round-robin Gemini caught: standard CFG math is `output = uncond + CFG * (cond - uncond)`. At CFG=1.0 this simplifies to `output = cond` -- the negative prompt is 100% unused. ROADMAP locks `LTX_CFG = 1.0` for the distilled sigma schedule. So the negative prompt in `_LTX_NEGATIVE` ("person, human, face, woman, man, hands, fingers, body, ...") is mathematically discarded by the sampler. Faces / people may still appear in LTX clips because the prompt suppression we *thought* was active isn't.
- **Cause:** Distilled LTX (`LTX_DISTILLED_SIGMAS` from Goofer) is tuned for CFG=1.0 because higher CFG with low-step distillation produces overcooked / artifacted output. The negative prompt was carried over from non-distilled LTX patterns where CFG≥1.5 made the negative effective.
- **Fix (shipped 2026-05-05):**
  - **Already-shipped baseline (commit `d57535a`):** `_PROMPT_BY_ROLE` was created with `"no people in frame"` as an explicit positive cue in every role (announcer / music_open / music_close / music_inter / sfx). No human-implying nouns ("announcer at microphone", "radio host", "broadcaster") in the body of any role. So the positive-only suppression strategy was effectively in place since the file's first commit; this BUG_LOG entry's "Fix (deferred)" plan to "tighten POSITIVE prompts" was already complete in code when the bug was logged 2026-05-02.
  - **Reinforcement (this commit):** added `"unattended equipment, empty studio"` to every role's positive prompt as a belt-and-suspenders bias, matching the positive-only suppression strategy CFG=1.0 mathematics requires. CFG/sigma schedule itself untouched (locked architecture).
- **Verify:**
  - sirens_print 2026-05-05 LTX announcer clip @ 0:03 and TV broadcast wide @ 0:15 -- no unwanted faces / people in either; only the radio set + TV broadcast equipment animating as intended.
  - dark_transponder 2026-05-05 LTX announcer beats -- same, clean equipment-only renders.
- **Future option (deferred):** if a future LTX line ships with non-distilled sigma support, raising CFG to 1.3-2.0 and re-enabling the negative prompt would let us suppress humans via the negative branch the way most other diffusion models do. Until then, positive-only suppression is the canonical pattern.
- **Tags:** ltx, cfg, prompt-policy, distilled-sigma, positive-only-suppression

### BUG-LOCAL-009 [DEFERRED]: Per-stage VRAM logging across HuMo→LTX→VC→RTX boundary

- **Date:** 2026-05-02 | **Phase:** S3 observability | **Bible candidate:** no (observability, not behavior)
- **Symptom:** No production VRAM snapshot at HuMo teardown / LTX loader entry / LTX teardown / RTX upscale entry. If a 16 GB OOM appears at any boundary, we have no per-stage signal to bisect from.
- **Fix (deferred to next sprint):** Add `[OTR_VRAM] free=X.XX allocated=Y.YY reserved=Z.ZZ` lines at:
  - `batch_humo_render.py` end of teardown
  - `batch_ltx_render.py` after `mm.load_models_gpu([model])` and after teardown
  - `rtx_upscale.py` after each chunk and at upscale exit
  Pattern matches existing `vram_snapshot()` helper in `nodes/_vram_log.py`.
- **Verify:** smoke run produces a clean VRAM ladder log; `peak_gb` per stage extractable via grep.
- **Tags:** vram, observability, deferred

### Round-robin QA — 2026-05-02 mega-sprint pre-smoke

`docs/2026-05-02-mega-sprint-consult__01_chatgpt.md` (gpt-5.5, 140s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 40s). NVIDIA round did not complete in window; two-of-three is sufficient per CLAUDE.md round-robin rule.

**Must-fix items applied before smoke:**
1. **Anti-clobber in `batch_ltx_render.py`** (ChatGPT + Gemini converged): `if out_mp4.exists(): skip` defends against role-filter drift overwriting HuMo character clips.
2. **Windows ffmpeg pipe deadlock** (Gemini): `rtx_upscale.py` was using `stderr=subprocess.PIPE` which blocks at 64 KB on Windows. Routed to `subprocess.DEVNULL`.
3. **ComfyUI cache desync** (Gemini): rewired link 86 from `BatchHumoRender.clips_dir` (slot 0, stable per episode_id) to `BatchHumoRender.report` (slot 2, varies per run from per-clip elapsed_ms). Forces ComfyUI to re-evaluate `LowVRAMCheckpointLoader` on every queue, keeping the loader's state machine in sync with the actual mm-unload call HuMo makes.
4. **Drop `-shortest`** (Gemini): silent video and audio source share frame count by construction; flag was at best dead code, at worst a footgun.

**Disagreement caught:**
- ChatGPT framed CFG=1.0 negative prompt influence as "weak"; Gemini corrected with the math: `output = uncond + CFG * (cond - uncond)` reduces to `output = cond` at CFG=1.0, so the negative prompt is 100% ignored, not weak. Logged as BUG-LOCAL-008 (deferred to next sprint, since changing CFG mid-sprint deviates from locked architecture).

**Documented for next sprint (not blocking smoke):**
- BUG-LOCAL-008 — CFG=1.0 + negative prompt mathematically inert.
- BUG-LOCAL-009 — per-stage VRAM logging missing.
- Gemini false alarm: `temporal_size=4096` in tiled VAE decode is fine because LTX_MAX_FRAMES=177 (the temporal window only matters above its value; 4096 means "decode whole sequence in one temporal pass", spatial tiling handles VRAM). Goofer-proven on RTX 5080 Blackwell.

### BUG-LOCAL-010 [BLOCKER on Sprint 3 acceptance]: LLM OOM regression at write_script main call (BUG-LOCAL-004 returns)

- **Date:** 2026-05-02 | **Phase:** S3 acceptance smoke | **Bible candidate:** yes
- **Symptom:** Sprint 3 smoke prompt_id `bc7136bb-50ab-471f-8caf-83e9cfefa481` (`target_words=30`, `num_characters=2`, `target_length="30 words (smoke, 1 act)"`, `optimization_profile=Standard`) OOM'd at `OTR_LLMScriptWriter` -> `write_script` (line 5432) -> `_generate_with_llm` (line 3211) -> `model.generate(...)`. Exact error: `Allocation on device 0 would exceed allowed memory. Currently allocated: 24.54 GiB / Device limit: 15.92 GiB / Free: 0 bytes / Requested: 3.94 GiB`. Peak allocated `29005 MiB` per torch's allocator print.
- **Cause (hypothesis):** BUG-LOCAL-004 fix (Sprint 1) added `_flush_vram_keep_llm()` before the main write_script `_generate_with_llm` call. BUG-LOCAL-005 fix (Sprint 1) added max_new_tokens clamp 256 for ultra_smoke and short-circuited the OpenClose 3-outline evaluator. Both are confirmed active on this run (runtime log shows `Loading LLM model: mistralai/Mistral-Nemo-Instruct-2407 (quantized=True)` plus the three sequential `[StoryOrchestrator] Starting inference` lines at max_new_tokens 64 / 800 / 256). Despite both fixes, peak allocated still hits ~29 GB. Likely roots: (a) `_flush_vram_keep_llm()` is not actually clearing the prior phases' KV cache + activations -- a Python reference is keeping intermediates alive; (b) NewsSummary's `max_new_tokens=800` build-up is the actual culprit, not write_script's call (the OOM fires DURING write_script's prefill but the 24 GB was already accumulated before that call started); (c) the Mistral-Nemo `_prefill` path in transformers 4.x has a regression where `past_key_values` is not freed between `model.generate` invocations even with explicit `torch.cuda.empty_cache()`.
- **Fix:** **Pending -- needs its own bisect window.** Plan: (a) instrument `_generate_with_llm` to log `torch.cuda.memory_allocated()` at entry / after generate / after the explicit `del` / after empty_cache call, so the actual leak source surfaces; (b) check that `_flush_vram_keep_llm()` survived the recent refactors and is in fact called between NewsSummary and write_script; (c) audit whether NewsSummary leaves a `transformers.cache_utils.DynamicCache` instance on the model object; (d) if (c) confirmed, monkey-patch `model._cache_implementation` to `None` between phases.
- **Verify:** re-queue 30-word smoke with `optimization_profile=Standard`, expect `peak_gb < 14.5 GB` across the LLM ladder (instrumented snapshot lines), expect to reach `OTR_SignalLostVideo` (the audio gate) without OOM.
- **Tags:** vram, oom, llm-cache, regression, blocks-s3-acceptance

### Sprint 3 mega-sprint: shipped code, live acceptance BLOCKED

The Sprint 3 mega-sprint code (LTX wiring + RTX VSR upscale + consult fixes) is committed on `v2.0-alpha`. The wiring is:
- AST-clean (3 modified .py files all parse).
- Regression-clean (Bug Bible, dropdown_guardrails, core, parse_retry, otr_api_type all green; 23 + 46 + 108 + 48 = 225 tests pass).
- Workflow JSON valid (`json.loads` round-trips, `last_link_id=93`, all 51 links intact, no orphan inputs).
- ComfyUI registers all three new nodes (`OTR_BatchLTXRender`, `OTR_RTXUpscale`, `LowVRAMCheckpointLoader`).
- ComfyUI accepts the patched workflow at `/prompt` and runs to OTR_LLMScriptWriter, where it hits BUG-LOCAL-010 (LLM OOM, pre-existing).

**Sprint 3 acceptance is BLOCKED on BUG-LOCAL-010**, NOT on a Sprint 3 wiring failure. The video-wiring code never executed because the smoke can't get past the LLM phase. Once BUG-LOCAL-010 is fixed in a follow-up bisect, re-queue the same workflow JSON and the S3.x acceptance bullets (ledger source_kind=ltx rows, ffprobe 832x480 pre-upscale + 1920x1080 post-upscale, audio byte-identity via stream MD5, peak VRAM < 14.5/15.5 GB) become directly observable.

### BUG-LOCAL-011 [FIXED]: BatchLTXRender raised on first live run -- _load_ledger missing the .mp4 -> _ledger.json stem-fallback that sister nodes have

- **Date:** 2026-05-02 EVENING | **Phase:** S3 live test | **Bible candidate:** yes
- **Symptom:** Live run on Jeffrey's ComfyUI Desktop with Gemma-4 E2B (which dodges BUG-LOCAL-010) progressed cleanly through the LLM ladder, audio cascade, FLUX bookend, and all 4 HuMo character clips. At HuMo teardown the dependency edge correctly fired LowVRAMCheckpointLoader -> BatchLTXRender, but BatchLTXRender raised: `RuntimeError: BatchLTXRender: ledger could not be loaded from inline JSON or path` at `batch_ltx_render.py:446`. Wallclock to failure: 00:58:53 (LLM ~10 min, audio ~3 min, FLUX ~3 min, HuMo ~40 min, then LTX failed immediately).
- **Cause:** `OTR_SignalLostVideo.0` (the STRING input feeding `BatchLTXRender.ledger_json` via link 90) emits the **mp4 path**, not the `_ledger.json` path. `BatchHumoRender._load_ledger_with_path` and `OTR_VideoComposite._load_ledger_with_path` both have a multi-tier stem-fallback that swaps `.mp4` -> `_ledger.json` with collapsed-underscore + fuzzy-match tiers (BUG-LOCAL-118 hardening). My BatchLTXRender's `_load_ledger` skipped that fallback -- it called `load_ledger_safe(.mp4)` directly, got `None`, returned `(None, None)`, raised. Round-robin consult flagged "ledger / clips_dir union" but missed this inner discrepancy because the node *interface* matches HuMo (both take a STRING called `ledger_json`); only the *internal resolver* differs.
- **Fix:** Replaced `BatchLTXRender._load_ledger` with a port of `BatchHumoRender._load_ledger_with_path`. Same multi-tier behaviour: (1) empty input -> auto-pick newest non-pending under audio dirs; (2) inline JSON -> parse; (3) `.mp4` path -> direct stem swap, then collapsed-underscore variant, then fuzzy directory-scan with <1h freshness gate; (4) `.json` path -> direct load. Same `(dict_or_None, Path_or_None)` return contract so the existing call site at `:425` is unchanged.
- **Verify:** Re-queue the same workflow JSON. Expect log lines `[BatchLTXRender] episode=signal_lost_..._...` and `radio_bookend: radio_bookend_<ep>.png` (the loader resolved the .mp4 -> _ledger.json swap and read radio_bookend_path from `ledger.meta`). Pre-fix repro: queue a workflow that wires SignalLostVideo.0 directly into BatchLTXRender.ledger_json with no manual ledger path; expect the fix to make this path-shape work end-to-end.
- **Tags:** ltx, ledger, stem-fallback, signallost-mp4, sister-node-divergence, bible-candidate

### Sprint 3 live-run progress observed on workflow JSON 7c4dfd4 (Gemma-4 E2B path)

- LLM phase (Gemma-4 E2B + E4B): clean. ~10 min. Peak VRAM ~14 GB. Output: parseable script with TITLE + SCENE + 6 [VOICE: ...] lines + 1 SFX + MUSIC closing.
- Audio cascade (Bark + Kokoro + MusicGen + AudioGen + AudioEnhance + EpisodeAssembler): clean. ~3 min. Episode duration 113s = 1.88 min.
- SignalLostVideo procgen: clean. mp4 saved 52.2 MB / 113s / 2712 frames in ~14s.
- BatchFluxRender (5 cast portraits + radio bookend): clean. **S3.2 acceptance VERIFIED**: radio bookend rendered at 1248x720 then Lanczos-downscaled to 832x480.
- BatchHumoRender Phase A (text encoding 4+1) + Phase B (Whisper) + Phase C (4 lines): clean. Peak VRAM 14.2 GB GPU dedicated. Per-clip ~10:00-10:20 wallclock at 6 sampler steps × ~97s/step. 4 character lines correctly routed to HuMo; 2 announcer lines correctly skipped (BUG-129b).
- LowVRAMCheckpointLoader -> BatchLTXRender: dependency edge fired correctly (sequencing intent SHIPPED), then BUG-LOCAL-011 raised inside `_load_ledger`. Fix landed in this commit; re-queue to verify the rest of the S3.x acceptance bullets.

### Round-robin consult on BUG-LOCAL-011 fix -- 2026-05-02 EVENING

`docs/2026-05-02-bug-local-011-fix-review__01_chatgpt.md` (gpt-5.5, 97s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 35s), `__03_nvidia.md` (nvidia/llama-3.3-nemotron-super-49b-v1.5, 93s).

**Three-way convergence (verdict: tighten before next live run):**

- Tier 1 (.mp4 -> .json exact stem swap) and Tier 2 (collapsed-underscore variant) are both correct and necessary.
- **Tier 3 fuzzy directory scan must be killed for LTX.** Non-deterministic (depends on directory contents + mtimes + wall-clock); could plausibly bind to a wrong neighbour ledger if exact match fails to load. Burning ~1 hour rendering against bad metadata is the failure mode 2/3 consultants flagged as the real risk.
- **Restore `_OTRL.load_ledger_safe()` for path loads** (consistent `[OTR_Ledger]` log prefix; future-proof against any hardening added there).
- **Fail loud on file-load errors.** If exact (Tier 1) or collapsed (Tier 2) ledger candidate file EXISTS but fails to parse / read (PermissionError from Windows file-locking, JSONDecodeError from a partial write), raise instead of falling through. Silent fall-through to a wrong neighbour was Gemini's strongest framing.
- **Document `humo_clips_dir` widget as a sequencing-only DAG edge.** Add tooltip + inline comment + `del humo_clips_dir` so a future maintainer doesn't remove it as dead code (which would let the LTX checkpoint load race HuMo's 16.5 GB MODEL teardown and OOM on 16 GB).

**Disagreement caught (consultants vs reality):**

- Gemini + NVIDIA flagged `log` and `time` as missing imports -> false alarms. `log = logging.getLogger("OTR.batch_ltx_render")` is at line 81; `import time` at line 51.
- Gemini + NVIDIA assumed `_OTRL.load_ledger_safe` does schema migration -> false alarm. It just wraps `json.loads` with three exception handlers (FileNotFound / JSONDecodeError / generic Exception), all logging WARNING and returning None. So restoring it gains consistent log-prefix and centralised error handling, NOT schema migration.

**Hardening pass applied in commit (next):**

1. Replaced raw `json.load()` calls with a local `_read(p)` helper that delegates to `_OTRL.load_ledger_safe(p)` when importable; raises RuntimeError when the loader returns None on an existing file.
2. Deleted Tier 3 fuzzy scan code path. Resolver now returns `(None, None)` after Tier 1 + Tier 2 miss, with a WARNING log line that explicitly notes Tier 3 was removed by the 2026-05-02 round-robin.
3. `humo_clips_dir` INPUT_TYPES tooltip rewritten to flag the widget as a DAG sequencing edge (NOT data); execute() body explicitly `del humo_clips_dir` with a comment explaining the "remove this and LTX OOMs" failure mode.

Resolver test (offline, against the live-run cached ledger) confirms all 4 branches still resolve correctly: .mp4 stem-swap, explicit .json path, inline JSON, empty input auto-pick.

**Companion artifact: `workflows/otr_ltx_smoke.json`** -- a 5-node fast-smoke harness (LowVRAMCheckpointLoader -> OTR_BatchLTXRender -> OTR_VideoComposite -> OTR_RTXUpscale + Note) that consumes the cached ledger.json + procgen mp4 + 4 HuMo character clips from the live run. ledger_json widgets on both BatchLTXRender + VideoComposite are the .mp4 PATH so the smoke truly repros the BUG-LOCAL-011 crash surface (i.e. exercises the .mp4 -> _ledger.json stem-swap chain). Wallclock target ~10 min vs ~60 min for full pipeline. Re-aim at a different episode by swapping the PROCGEN_MP4 + HUMO_VIDEOS_DIR widget values; both must come from the same episode_id so LTX writes into the dir HuMo wrote into.

### Path consolidation (Jeffrey directive, 2026-05-02 EVENING after first end-to-end smoke landed)

- **Date:** 2026-05-02 EVENING | **Phase:** post-smoke cleanup | **Bible candidate:** no (project-specific layout)
- **Change:** Final episode mp4 deliverables moved from `<output>/episodes_for_obs/<ep>/` (sibling of otr/) to `<output>/otr/episodes/<ep>/` (nested INSIDE otr/). Every project output now lives under one tidy `otr/` umbrella:
  - `otr/audio/<ep>.mp4` -- procgen mp4 + ledger.json
  - `otr/stills/` -- FLUX bookends + cast environments
  - `otr/portraits/` -- PASS1 character portraits
  - `otr/videos/<ep>/<line_id>.mp4` -- per-line HuMo + LTX clip pieces
  - **NEW: `otr/episodes/<ep>/<ep>.mp4` and `<ep>_1080p.mp4` -- final user-facing deliverables ONLY**
- **Why:** OBS's directory_sorter still gets a clean root with only finished episodes (now `output/otr/episodes/`), but the entire project workspace has one nested root instead of two siblings (`otr/` + `episodes_for_obs/`). Easier mental model + easier to back up + easier to scrub.
- **Files touched:**
  - `nodes/_otr_paths.py::episodes_for_obs_dir` -- function name kept for back-compat with existing imports; return value changed to `comfy_output_dir() / "otr" / "episodes" / episode_id`.
  - `nodes/video_composite.py` -- comment block updated.
  - `scripts/render_episode_concat.py` -- comment + default `out_dir` expression updated.
  - `tests/test_render_episode_concat_discovery.py` -- pinned source-string assertion updated to require `"otr" / "episodes" / episode_id`.
- **OBS pointer change:** if you have OBS / external tooling configured to watch `output/episodes_for_obs/`, repoint it to `output/otr/episodes/`.

### Cleanup: torch.from_numpy non-writable warning in OTR_RTXUpscale (2026-05-02 EVENING)

- **Date:** 2026-05-02 EVENING | **Phase:** post-smoke cleanup | **Bible candidate:** no (cosmetic)
- **Symptom:** First successful smoke surfaced `UserWarning: The given NumPy array is not writable, and PyTorch...` at `rtx_upscale.py:216`.
- **Cause:** `np.frombuffer(chunk_bytes, dtype=np.uint8)` returns a view over an immutable bytes buffer; `torch.from_numpy` warns when handed a non-writable ndarray.
- **Fix:** Added `.copy()` after `np.frombuffer()` so torch gets a writable buffer. Also confirms a clean ownership boundary between the ffmpeg-stdout-bytes and the cuda transfer.

### BUG-LOCAL-013 [FIXED]: LTX 2B v0.9 bundled checkpoint has no CLIP/T5 -- LowVRAMCheckpointLoader returns CLIP=None -> NoneType.tokenize crash

- **Date:** 2026-05-02 EVENING (T+~30 min after smoke load) | **Phase:** S3 fast-smoke first execution | **Bible candidate:** yes
- **Symptom:** Smoke loaded cleanly (after the BUG-LOCAL-012 UUID fix), Queue Prompt accepted, LowVRAMCheckpointLoader fired, then BatchLTXRender raised at line 554: `AttributeError: 'NoneType' object has no attribute 'tokenize'`. ComfyUI runtime log printed `no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.` immediately above the crash.
- **Cause:** The bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB on disk at `C:\ComfyUI-Models\checkpoints\`) ships only UNet + VAE; it does NOT carry the T5 text encoder. `LowVRAMCheckpointLoader` (a `CheckpointLoaderSimple` subclass) then returns `(MODEL, None, VAE)` for the (model, clip, vae) tuple. My BatchLTXRender wired CLIP straight from LowVRAM, so `clip.tokenize(...)` immediately NoneType-crashed. Note: this means BUG-LOCAL-007's deviation from the locked Architecture Truth was wrong on the premise -- the original Architecture Truth (UNETLoader + CLIPLoader + VAELoader for LTX 2B) was actually correct, because `t5xxl_fp16.safetensors` has to be loaded separately. The LowVRAMCheckpointLoader is still useful for the UNet+VAE side (sequential-load via `dependencies` input survives), but the CLIP comes from a sibling CLIPLoader, not from the bundled file.
- **Fix:**
  1. Add a `CLIPLoader` node loading `t5xxl_fp16.safetensors` (already on disk at `C:\ComfyUI-Models\text_encoders\`) with `type='ltxv'`, `device='default'`. Verified `'ltxv'` is in `/object_info/CLIPLoader` allowed types on the live ComfyUI alongside sd3 / wan / mochi / flux2 / etc.
  2. Rewire `OTR_BatchLTXRender.clip` from the new CLIPLoader, NOT from `LowVRAMCheckpointLoader.CLIP`.
  3. Apply same fix to BOTH `workflows/otr_ltx_smoke.json` AND `workflows/otr_scifi_16gb_full.json` so the next live full-pipeline run also doesn't hit this crash. Production workflow now has new node 57 (CLIPLoader, T5, ltxv) wired via new link 94 to BatchLTXRender (55).1. Old link 88 (54.1 -> 55.1, the dead CLIP edge from LowVRAM) deleted.
- **Verify:** Re-run smoke; expect log line `[BatchLTXRender] episode=signal_lost_..._170555` followed by per-clip render lines (no `NoneType.tokenize` AttributeError). Final mp4 should write to `output/episodes_for_obs/<ep>/<ep>.mp4` and a separate `<ep>_1080p.mp4` from the upscaler.
- **Why the smoke harness paid off here:** the original failed live run (BUG-LOCAL-011 on the .mp4-as-ledger problem) hid this CLIP=None bug behind it. If I had only fixed BUG-LOCAL-011 and re-queued the full pipeline, we would have burned ~50 more min of HuMo wallclock just to crash here. The smoke surfaced this in <30s of LTX cold-load -- exactly what fast iteration loops are for.
- **Architecture Truth retroactively re-validated:** the locked Architecture Truth from 2026-05-02 (UNETLoader + CLIPLoader + VAELoader for LTX 2B) was correct in spirit; only the bundled-vs-split question was open. We're now using a hybrid: LowVRAMCheckpointLoader for the bundled UNet+VAE (with the dependencies input for sequential-load), separate CLIPLoader for T5. Same end result as the locked plan; cleaner sequencing edge.
- **Tags:** ltx, cliploader, t5, bundled-checkpoint, hidden-by-prior-bug, smoke-paid-off, bible-candidate

### BUG-LOCAL-012 [FIXED]: ComfyUI frontend Zod validation rejected `workflows/otr_ltx_smoke.json` at load time

- **Date:** 2026-05-02 EVENING | **Phase:** S3 fast smoke harness | **Bible candidate:** yes (broadly applicable to anyone hand-building ComfyUI workflow JSONs)
- **Symptom:** Jeffrey loaded `workflows/otr_ltx_smoke.json` (commit `f60d2e4` / `4df4e72`) into ComfyUI Desktop and the frontend rejected the workflow with two Zod validation alerts: `Invalid workflow against zod schema: Validation error: Invalid uuid at "id"`. Raw `json.loads` round-trips cleanly; this is a frontend-side schema validation failure, not a JSON syntax error.
- **Root cause confirmed:** workflow root `id` field MUST be a valid UUID (8-4-4-4-12 hex format). My hand-built smoke had `id: "otr-ltx-smoke"` -- a freeform slug. ComfyUI's Vue 3 frontend Zod schema enforces uuid format on this field. Production `otr_scifi_16gb_full.json` has a valid UUID so it loads cleanly.
- **Cause (hypothesised pending error-text capture):** The smoke JSON was hand-built by `outputs/_build_ltx_smoke.py` rather than exported by the ComfyUI UI. Three candidate divergences from the canonical shape, identified by structural diff against `workflows/otr_scifi_16gb_full.json` + `Nvidia_RTX_Nodes_ComfyUI/example_workflows/rtx_video_upscale.json` (both load cleanly):
  1. **`_meta` field on every node.** Mine has `_meta: {title: ...}`; canonical workflows use a top-level `title` field on the node. Some Zod schemas reject unknown fields strictly.
  2. **`shape: 7` on optional inputs.** Mine puts `shape: 7` on `dependencies` + `humo_clips_dir`; neither known-good workflow uses this. LiteGraph shape values 1-7 are valid in the LiteGraph runtime, but the Vue frontend's Zod schema may not list `shape` as an allowed input field.
  3. **Missing `slot_index` on outputs.** Production has `{name, type, links, slot_index}` on every output; mine omits `slot_index`. Vue frontend may require it to track slot-position semantics.
- **Fix:** apply all 3 candidate corrections in `outputs/_build_ltx_smoke.py` and re-emit `workflows/otr_ltx_smoke.json`. Drop `_meta` -> rename to `title`. Drop `shape` from inputs entirely. Add `slot_index: <i>` to every output. Land an offline regression check (`tests/test_workflow_zod_shape.py`) that asserts these shape invariants on every JSON under `workflows/` so this class of bug is caught by the test suite before it reaches the UI.
- **Verify:** Jeffrey reloads `workflows/otr_ltx_smoke.json` in ComfyUI Desktop; no Zod error; nodes appear on canvas; Queue Prompt accepts the workflow.
- **Why prior consults missed it:** all three rounds reviewed the ComfyUI execution semantics (object_info, link types, DAG ordering, .mp4 path stem-swap, audio passthrough). None reviewed the frontend's Zod schema, which is a separate validation layer between drag-and-drop UI load and the backend's `/prompt` endpoint. The CLI `submit_prompt` path bypasses Zod entirely (it goes straight to `/prompt` with the API-converted workflow), which is why my offline JSON tests + the `queue_smoke.py` script + the `_test_ledger_resolver.py` all passed -- they exercise different code paths than the UI loader.
- **Tags:** zod, comfyui-frontend, workflow-json-shape, hand-built-workflow, bible-candidate



