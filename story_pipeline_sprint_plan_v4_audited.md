# Story Pipeline -- Ledger Writer Hardening Plan v4 (CODE-AUDITED)

- **Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha`
- **Current HEAD:** `0c76ee7` (v3 plan was written against `2183397` -- 3 commits stale; the audited files below are current at `0c76ee7`).
- **Audit date:** 2026-05-24. Every code claim in v3 was checked against the real source. This is a **separate working file** -- ROADMAP is intentionally untouched until this batch lands.
- **What changed from v3:** corrections applied in place; each correction carries a `> AUDIT:` callout with file:line evidence. Nothing in v3's *strategy* was wrong -- the decomposition philosophy holds. What was wrong: ~8 config key names, the HuMo tier names, one "new" claim, and two items already done in the code.

---

## 0. AUDIT CORRECTIONS SUMMARY (read this first)

| Plan item | v3 said | Verdict | Correction |
|---|---|---|---|
| Sprint 0 `json_str` bug | "appears to reference json_str out of scope" | **CONFIRMED -- real bug** | `_otr_story_brief.py:648`, schema-repair arm. `NameError`; correct variable is `raw`. Caught by a broad `except` so it never crashes -- it silently turns every schema-repair into an instant failure sentinel. One-word fix. |
| Sprint 0 pick_style comment | stale | CONFIRMED | Writer comment says pick_style "routes both passes to creative"; Pass 2 (chooser) actually routes technical. |
| Sprint 0 `technical_fn` dead param | dead weight | **CONFIRMED BUT INCOMPLETE** | `_otr_line_composer.py:1505` `generate_fn = creative_fn`; `technical_fn` never read *internally*. 2026-05-24 follow-up: it is NOT free dead weight -- `tests/test_helper_paired_signatures.py::test_compose_line_accepts_paired_generators` mandates `technical_fn` as a keyword-only param, and the paired `creative_fn`+`technical_fn` contract is deliberate uniformity across 4 sibling helpers (`pick_style`, `lock_cast`, `compose_line`, `build_news_briefs`), kept for the planned B2/B3/B4 per-sub-pass dispatch. Dropping it is a decision, not a mechanical fix. See Open Decision 6. |
| Sprint 0 helper_context gaps | 3 sites unwrapped | CONFIRMED | outline / title regen / story-brief reflection bypass `helper_context` -> `<unattributed>` bucket. |
| Sprint 0 "CI check: every call site tagged" | implies none exists | **PARTIAL** | Per-file count tests already exist (`tests/test_writer_slot_routing.py` asserts >=8 creative + >=1 technical; `tests/test_story_brief_c5a1.py`). No AST-level per-call-site sweep. The item is real but scoped down to "add the global sweep." |
| Sprint 1 HuMo tier names | `humo_low_vram_default`, `humo_high_quality_unsafe_on_16gb` | **NAMES WRONG** | Actual tiers: `low_vram_default`, `high_quality`, `experimental_gguf` -- **three**, no `humo_` prefix (`_otr_humo_tier_loader.py:101-125`). Rename target is `high_quality` -> `high_quality_unsafe_on_16gb`. v3 missed the third tier entirely. |
| Sprint 1 `resume_from_ledger` default-on | "turn on" | **ALREADY DONE** | `batch_humo_render.py:1443` default `True`. |
| Sprint 1 `cuda_cleanup_on_oom` default-on | "turn on" | **NAME WRONG + DONE** | Real widget `cuda_hard_reset_on_oom`, `batch_humo_render.py:1479`, default `True`. |
| Sprint 1 `stop_on_soak_cap` default-on | "turn on" | **NAME WRONG + DONE** | Real widget `stop_workflow_on_soak_cap`, `batch_humo_render.py:1495`, default `True`. |
| Sprint 1 `humo_max_lines_per_process = 6` | set to 6 | **CONFLICTS WITH A LOGGED DECISION** | Real widget exists (`batch_humo_render.py:1460`) but default is **0** (disabled) -- deliberately bumped 3 -> 0 per `BUG_LOG.md:210`. Setting 6 reverts that. **Needs your decision before any edit.** |
| Sprint 1 `clip_length_seconds` lock 7.0 | "lock" | **NAME WRONG** | Real widget `clip_length`, `batch_humo_render.py:1332`, default already `7.0` but freely editable (range 1.32-14.12). "Locking" needs a real code change (fixed/hidden widget), not a default flip. |
| Sprint 2B +0.15 repair temp bump | exists | CONFIRMED | `_otr_story_brief.py`: `_REPAIR_TEMPERATURE_BUMP = 0.15`, ceiling `0.55`, base `0.3` -> repair runs at `0.45`. |
| Sprint 2D single-shot cleanup passes | no retry | CONFIRMED | `audit_cast_contract` + `run_script_doctor` each one call, failure -> sentinel. |
| Sprint 2E GBNF | built, never wired | CONFIRMED | `grammars/news_interpreter.gbnf` + `grammars/style_picker.gbnf` exist; `GRAMMAR_PATH` constant defined; loader never consumes it. |
| Sprint 3A `apply_deterministic_cast_repairs` | "extends today's [function]" | **FILE WRONG** | Function lives in `_otr_ledger_reviewer.py:476`, **not** `_otr_cast_repair.py`. Levenshtein logic is **already there** -- `auto_remap_phantom` + `_levenshtein` (threshold 3). |
| Sprint 3A/3B `LAST_LINES_WINDOW` | "pattern already stabilizing the composer" | **DOES NOT EXIST** | No such constant. The composer never slices `last_lines` -- the caller passes a pre-trimmed list. Use a real reference or define the constant as new work. |
| Sprint 3A `_otr_line_composer.py` | shown as a design surface | **ALREADY EXISTS (~1660 lines)** | 3A is a *rewrite* of a large existing file, not a greenfield module. Effort estimate should reflect that. |
| Sprint 3B "drop `target_words` from LLM schema" | framed as a functional fix | **ALREADY HANDLED** | `_BeatFleshout` does carry `target_words`, but `_allocate_phase_target_words` already overrides it (`_otr_outline.py:1683-1707`) -- Python is *already* authoritative. Removing the field is token-budget cleanup only, not a behavior change. |
| Sprint 3C Doctor combined ask + thin rows | both true | CONFIRMED | `_DOCTOR_SYSTEM_PROMPT` asks pacing+voice+arc+JSON in one call; Doctor rows carry only `line_id, speaker_role, char_id, text`. |
| Sprint 3D casting picks `voice_preset` | LLM-decided | CONFIRMED | `CastingResponse` = `character_description, gender, voice_preset`. No Python global gender/timbre balance -- only a static prompt line "~40/40/20". |
| Sprint 3F cast auditor `confidence >= 0.8` | flat 0.8 | **IMPRECISE** | `confidence` float is real (`CastViolation`). Threshold is **0.8 for `bad_casing` only**; `wrong_char_id` and `role_mismatch` gate at **0.9**; `alias_used`/`invented_name`/`speaker_unknown` need no confidence. |
| Sprint 3E title post-hoc substitution | misses paraphrases | CONFIRMED | `OTR_LedgerScriptWriter.py:2687` substitutes the new title into line text that quoted the old one -- a verbatim string match, so paraphrases do slip through. |
| Sprint 4 "behavioral HuMo VRAM gate" | new work | **ALREADY EXISTS** | `vram_safety_threshold_gb` (default `10.0`) + `auto_downgrade` (default `True`) + gate logic in `_otr_humo_tier_loader.py:_resolve_tier`. Reframe as "verify / extend the existing gate," not "build." |
| Config block `steps: 20`, `cfg: 5.0` | render defaults | **WRONG SURFACE** | `BatchHumoRender` widget defaults are `steps=6`, `cfg=1.0`. `20`/`5.0` only appear in the `low_vram_default` tier table, which overrides the widgets at runtime via output sockets. |

**Net read:** v3's architecture is sound and worth executing. Sprint 1 is roughly half-done already (three flags on, gate built) and half-misnamed -- it shrinks to "rename the high tier + decide the two open config values." Sprints 0, 2, 3 are accurate where it matters. The one item that must not be auto-applied is `humo_max_lines_per_process = 6`.

---

## 1. BUILD TRACKING PROTOCOL (how progress + bugs are tracked)

**Two documents, zero overlap. They are linked by a pointer, never a copy.**

- **This file is the single source of truth for build PROGRESS.** Sprint status, checkboxes, and the dated progress log all live here and nowhere else. ROADMAP is NOT touched until the entire build lands.
- **`BUG_LOG.md` stays the single source of truth for BUGS.** Per CLAUDE.md, every bug found during the build is logged there immediately as `BUG-LOCAL-NNN` -- no batching, no waiting. Bug detail (symptom / cause / fix / verify) lives ONLY in BUG_LOG.md.
- **The link is a pointer.** When a sprint item surfaces a bug, the item gets a `-> BUG-LOCAL-NNN` tag here. The plan never duplicates bug detail; BUG_LOG never tracks sprint progress. Nothing is "married" -- each fact has exactly one home.

### Rules

1. **Sprint Status Board** (below) -- update the status cell the moment a sprint changes state: `NOT STARTED` / `IN PROGRESS` / `COMPLETE` / `BLOCKED`.
2. **Checkboxes** -- a `- [ ]` item flips to `- [x]` only when: the item is done AND its regression passed AND (if it touched a node) the workflow JSON is re-wired (CLAUDE.md Prime Directive 3).
3. **Build Progress Log** (bottom of this file) -- append one dated entry per work session: what landed, commit hash, regression result, any new `BUG-LOCAL-NNN` ids opened. Append-only -- never rewrite past entries; git holds the rest.
4. **Bug found mid-build** -> log it in `BUG_LOG.md` first (`BUG-LOCAL-NNN`, with the standard schema), THEN add the `-> BUG-LOCAL-NNN` pointer to the relevant sprint item here. Never the reverse.
5. **Completion gate** -- the build is "done" only when every non-deferred sprint item is `[x]`, all regression suites (Bug Bible + core + audio-byte-identical) are green, and every `BUG-LOCAL` opened during the build is `[FIXED]` or explicitly parked. ONLY THEN: fold the finished plan into ROADMAP, mark this file superseded, and move it to `docs/`.

### Sprint Status Board

| Sprint | Status | Bug pointers | Notes |
|---|---|---|---|
| 0 -- telemetry + cosmetic + hot bug | COMPLETE | BUG-LOCAL-268 | all items landed: `e7a8eb6` json_str fix + test, helper_context wraps, pick_style comments; `51f7226` pick_style routing-test refresh; unused paired params DROPPED in `6940209` (Decision 6 reversed); CI AST `# LLM slot:` sweep + 4-test suite `c99fdfb`; sweep parse-failure hardening + 2 tests `b3d6355` |
| 1 -- render seatbelts | COMPLETE | -- | tier rename landed `df7f9b1`; decisions 1-2 resolved to no-change; render-flag defaults already on |
| 2A -- structured_call helper | COMPLETE | -- | helper `61f8cfa`; extensions `fed6327`; all call sites converted -- ledger `7f3b65f`, story_brief `3f41fc8`, news `b4c6e83`, casting `6e2950d`, outline `476eabc` |
| 2B -- repair temp fix | COMPLETE | -- | baked into `structured_call` (structural retry < base, asserted at entry) `61f8cfa` |
| 2C -- typed repair prompts | COMPLETE | -- | `1fa6b40` -- six typed factories + dispatcher in `_otr_repair_prompts.py`; `structured_call` Attempt 3 accepts a factory-returned schema instance (cast_membership deterministic no-LLM path via `auto_remap_phantom`); all 8 call sites wired |
| 2D -- cleanup pass retries | COMPLETE | -- | `audit_cast_contract` + `run_script_doctor` -> `structured_call(max_attempts=4)` `7f3b65f` |
| 2E -- GBNF | COMPLETE | -- | decision 4 re-resolved 2026-05-25: DELETE -- loader (HF Transformers 5.5.0, no llama.cpp / transformers-cfg / outlines) has no GBNF support; dead scaffolding removed |
| 3A -- split compose_line | COMPLETE | -- | `e24b327` -- compose_line split into compose_line_draft (creative job) + thin orchestrator; new cast_strip step wraps auto_remap_phantom (threshold=1); _word_bands / _strip_named_prefix helpers. Behaviour-preserving except cast_strip. Operator live-run pending (Prime Directive 1) |
| 3B -- outline Stage 3 | COMPLETE | -- | `3992607` -- beat prompt gains adjacency context (prev intent + next speaker + phase summary); `target_words` dropped from `_BeatFleshout` (Python allocation already authoritative). `next_beat_intent` -> `next_beat_speaker` -- a later beat's intent does not exist at sequential generation time |
| 3C -- split Script Doctor | COMPLETE | -- | `74438ff` -- `run_script_doctor` split into `_diagnosis` + `_edits` passes; +1 LLM call (technical, tagged; sweep 21/21). Doctor rows now 7/7 -- `beat_intent`/`target_words` stamped on the ledger `14818bb` (2026-05-25 follow-up) |
| 3D -- split Casting | COMPLETE | -- | `5fe9931` -- `precompute_ensemble_slots` / `llm_write_description` / `python_assign_voice_preset`; Python owns gender + voice, LLM writes description only |
| 3E -- title scratchpad | COMPLETE | -- | `d230cd6` -- forced scratchpad (DETAILS->CANDIDATES->TITLE); `EPISODE_TITLE: TBD` late binding; post-hoc title substitution removed |
| 3F -- cast auditor confidence | COMPLETE | BUG-LOCAL-271 [FIXED] | `e14d364` -- `confidence` field removed from `CastViolation`; `apply_deterministic_cast_repairs` resolves via exact case-fold / Levenshtein (`_resolve_cast_member` reuses `auto_remap_phantom`), ambiguous ties escalate. BUG-LOCAL-271 (`wrong_char_id` repair contract mismatch) FIXED `3e120df` |
| 3G -- reflection sanitize | COMPLETE | -- | `088aba1` -- `_build_reflection_input` pre-sanitizes cast names + proper nouns to neutral tokens; `_REFLECTION_PROMPT` suppression list trimmed; output reject-list safety net untouched |
| 4 -- VRAM hardening | COMPLETE | BUG-LOCAL-272 [FIXED] | `6b9300e` -- code-side verify: Zero-Prime Wash / Sovereignty 2.5 GB / 2B-12B caps / bf16+tf32 confirmed present; attn-selector dead code fixed (BUG-272); gate fires for renamed tier. Close-out 2026-05-25: 14B cap resolved no-change (keep Sovereignty branch -- Jeffrey); prompt-cache bullet resolved N/A (llama.cpp param, no HF equivalent). Operator live-RTX-5080 confirm pending (Prime Directive 1) -- non-blocking, same gate as 3A |
| 5 -- continuity + critic + reroll | IN PROGRESS | -- | 5A DONE `8fef3c5` (continuity ledger); 5B DONE `4b7db99` (whole-script critic); 5C NOT STARTED -- blocked on an architecture decision (see Build Progress Log 2026-05-25 + session_handoff.md "Sprint 5C open fork") |
| 6 -- critic->render coupling | NOT STARTED | -- | ships with Sprint 5; needs the 5C decision first (both touch the cascade node surface + workflow JSON) |

**Round-robin consultation WAIVED 2026-05-24 (Jeffrey):** these changes were round-robined in earlier sessions, so no sprint carries a consultation gate -- the build runs sprint-after-sprint. No sprint is `BLOCKED`; that status is now reserved for a genuine hard dependency only.

---

## Operating Philosophy (unchanged from v3 -- still correct)

1. One job per call. A model dropping a constraint = the prompt is overloaded.
2. Deterministic strips run before LLM output enters rolling context.
3. Python owns what Python can compute -- word budgets, gender balance, voice selection, confidence thresholds, name sanitization.
4. Repair prompts are typed by failure class. No generic "fix your JSON" passes.
5. One shared structured-call helper everywhere.
6. Air-gap the LLM from ledger state. The LLM proposes; deterministic wrappers commit.
7. Models are interchangeable. Quality lives in the multi-pass scaffold, not the slot.

---

## Locked Decisions (corrected)

| Decision | Value |
|---|---|
| Current creative + technical slot models | Stay. No swap. |
| New models | Appendix only. After Sprints 0-5 land. |
| Context cap | 8192 tokens. RoPE scaling off. |
| Slot residency | One model resident at a time. Zero-Prime Wash between swaps. |
| Sovereignty Buffer | 2.5 GB reserved. |
| HuMo default tier | `low_vram_default` (existing name -- no `humo_` prefix). |
| HuMo high tier | Rename `high_quality` -> `high_quality_unsafe_on_16gb`. |
| HuMo third tier | `experimental_gguf` exists -- leave as-is unless you want it renamed too. |
| GBNF | Sprint 2E RESOLVED 2026-05-25: DELETE. Loader has no GBNF support; scaffolding removed. |

---

## Sprint 0 -- Telemetry, Cosmetic, Hot Bug (Hours, Not Gated)

- [x] **FIX the `json_str` `NameError` -- confirmed bug, do this first.** **[DONE e7a8eb6 2026-05-24 -- BUG-LOCAL-268; +4-test regression `tests/test_story_brief_repair_pass.py`]** `_otr_story_brief.py:648`, inside `run_story_brief_reflection`'s schema-validation repair arm: `_repair_pass(failed_output=json_str, ...)` references `json_str`, which is never bound. The raw LLM output is held in `raw` (bound at L604). The `NameError` is swallowed by the broad `except (Exception, ValidationError) as exc2` at L657, so it never crashes -- instead **every schema-repair attempt becomes an instant `_failure_sentinel(REJECT_SCHEMA)`**. Repair is dead on that arm. Fix: `json_str` -> `raw`.
  > AUDIT: The content-validation repair arm at L679 already passes `failed_output=brief_model.story_brief` correctly -- the bug is isolated to the schema arm. One-word fix; add a regression test that forces a `ValidationError` and asserts `_repair_pass` actually runs.
- [x] Wrap `generate_outline`, `_generate_title_from_script`, `run_story_brief_reflection` in `slot_scheduler.helper_context(...)`. `<unattributed>` bucket -> 0. **[DONE e7a8eb6 2026-05-24 -- wrapped at the 3 `OTR_LedgerScriptWriter.py` call sites; helper names `generate_outline` / `generate_title` / `story_brief_reflection`]**
- [x] Fix stale `pick_style` comment in `OTR_LedgerScriptWriter.py` (Pass 2 / chooser routes to technical, not creative). **[DONE e7a8eb6 2026-05-24 -- two stale comments corrected (S30 routing-table block + the pick_style call-site block); verified against `_otr_style_picker.py:605-624`]**
> **PULLED (2026-05-24, Open Decision 6 resolved -- keep).** Sprint 0 originally listed a `technical_fn` drop from `compose_line` as dead weight; pulled after verification -- the param is contractually load-bearing per the S32 B1 paired-contract, retained pending B3/B4 per-sub-pass dispatch. Dropping it now would reverse S32 B1 and force the same uniformity to be re-added when B3/B4 land.
- [x] **CI check (scoped down) -- DONE 2026-05-25 (`c99fdfb` sweep + tests; `b3d6355` parse-failure hardening).** Per-file `# LLM slot:` count tests already existed (`tests/test_writer_slot_routing.py`, `tests/test_story_brief_c5a1.py`). The new AST-level sweep `docs/_s28_llm_slot_sweep.py` walks every `*.py` under `nodes/` (exempt: internal plumbing + the writer scheduler + `vram_context_test.py`), AST-parses each, finds every `structured_call` / `generate_fn` / `creative_fn` / `technical_fn` / `polish_generate_fn` / `request_slot` call site, and asserts a `# LLM slot:` tag within ±8 lines. 20 sites found, all 20 tagged. Regression `tests/test_llm_slot_sweep.py` -- 6 tests (zero-untagged, floor count ≥12, synthetic missing-tag catch, exempt-file guard, plus the `b3d6355` hardening pair: no node file fails to AST-parse + synthetic parse-failure catch). `b3d6355` hardening: a node file that fails to parse was silently swallowed by the call-site walk, making its call sites invisible -- `find_parse_failures` now surfaces that loud and `main()` exits 1 on any unparseable file.
- [x] **(Added 2026-05-24 -- 5th Sprint 0 item, replaces the pulled `technical_fn` drop.)** **[DONE 51f7226 2026-05-24]** Refresh the stale `test_pick_style_internally_uses_creative_fn_default` test (`tests/test_helper_paired_signatures.py`). It asserts `technical_fn` is never called by `pick_style` -- false since S32 B2 routed pass 2 (chooser) to `technical_fn`. The test passes today only by accident: its inventor mock data trips the mode-collapse reject, so `_run_chooser` is never reached. Same root cause as the stale `pick_style` comments -- both predate S32 B2. Rewrite it to assert the B2 routing (pass 1 inventor -> creative, pass 2 chooser -> technical).

---

## Sprint 1 -- Render-Side Seatbelts (Config Only, Not Gated)

- [x] **[DONE df7f9b1 2026-05-24]** Rename HuMo high tier `high_quality` -> `high_quality_unsafe_on_16gb` **everywhere**: `_TIER_TABLE` + `_TIER_CHOICES` in `_otr_humo_tier_loader.py:101-125`, and every workflow JSON that wires the tier widget (`workflows/otr_scifi_16gb_full.json` and any siblings). Per CLAUDE.md Prime Directive 3, the JSON re-wire is part of "done."
  > AUDIT: `experimental_gguf` is a real third tier v3 never mentioned. **Decision 3 RESOLVED 2026-05-24: stays -- not renamed.**
- [x] ~~Defaults on: `resume_from_ledger`, `cuda_cleanup_on_oom`, `stop_on_soak_cap`.~~ **ALREADY DONE.** All three exist and default `True` -- real names are `resume_from_ledger`, `cuda_hard_reset_on_oom`, `stop_workflow_on_soak_cap`. No edit needed; just verify the workflow JSON doesn't override them to `False`.
- [x] **DECISION 1 RESOLVED 2026-05-24 -- `humo_max_lines_per_process` stays `0`.** No change. The logged `BUG_LOG.md:210` decision (bumped 3 -> 0) stands; v3's request for `6` was declined, so no reversal entry is needed.
- [x] **DECISION 2 RESOLVED 2026-05-24 -- `clip_length` stays editable, default `7.0`.** No lock, no code change; the operator keeps the editable widget.

---

## Sprint 2 -- Retry Discipline + GBNF

### 2A. One shared structured-call helper (not gated) -- **[STEP 1 LANDED 61f8cfa: `nodes/_otr_structured_call.py` + 11 tests. EXTENSIONS LANDED fed6327: `post_validator` content-check hook + `max_new_tokens` per-caller budget + 7 tests -- the 4 content-validating call sites and the 160-3500 token spread could not convert without them. CALL SITES: `_otr_ledger_reviewer` (audit + doctor) converted 7f3b65f; `_otr_story_brief` / `news_interpreter` / `_otr_casting` / `_otr_outline` pending.]**

New module `nodes/_otr_structured_call.py`. Single retry ladder for every structured JSON pass. Signature per v3 (a `structured_call(*, prompt, schema, slot_fn, base_temperature, structural_retry_temperature, repair_prompt_factory, grammar_path, max_attempts, helper_name) -> T`).

Ladder: Attempt 1 base temp; Attempt 2 same prompt at `structural_retry_temperature` (LOWER than base -- see 2B); Attempt 3 typed repair prompt; Attempt 4 grammar-enforced (if available) else fail loud.

Call sites to convert (all verified to exist):
- `news_interpreter.build_news_briefs` -- currently 3 attempts incl. repair.
- `_otr_casting.cast_one_character` -- currently 3 attempts; attempt 3 repair routes to the technical slot.
- `_otr_outline` macro / phase / beat stages -- currently 3 attempts each via `_run_call_with_retry`.
  > AUDIT: the three outline stages are **inline blocks inside `generate_outline()`**, not standalone functions, and they already share `_run_call_with_retry` / `_BeatFleshout` etc. Stage 2 (phase) is the odd one -- falling temp schedule `(0.35, 0.25, 0.15)`, a deterministic fallback (`_deterministic_phase_skeleton`), and it is skipped entirely for a singleton cast. Don't flatten Stage 2's fallback away when converting.
- `_otr_story_brief.run_story_brief_reflection` -- currently 1 fresh + 1 repair per arm (see 2B/Sprint 0).
- `_otr_ledger_reviewer.audit_cast_contract` -- currently single-shot.
- `_otr_ledger_reviewer.run_script_doctor` -- currently single-shot (post-split, see 3C).

### 2B. Repair temperature inversion -- fix (CONFIRMED)

`_otr_story_brief.py` raises repair temperature: `_REPAIR_TEMPERATURE_BUMP = 0.15`, base `_REFLECTION_TEMPERATURE = 0.3`, clamped at `_REPAIR_TEMPERATURE_CEILING = 0.55` -> repair fires at `0.45`. Raising entropy during JSON-schema repair encourages further structural hallucination. Keep temperature static or lower it; change the **payload** instead (inject the exact `ValidationError` trace, rotate the system instruction). Codified in 2A: `structural_retry_temperature` is *below* `base_temperature`; Attempt 3 changes prompt content, not heat.

### 2C. Typed repair prompts by failure class (not gated)  [COMPLETE 2026-05-25 -- `1fa6b40`]

Repair factory dispatches by class: `json_syntax_repair`, `schema_field_repair`, `cast_membership_repair`, `too_many_words_repair`, `narration_leak_repair`, `forbidden_name_repair`. Cast-membership repair never calls the LLM if Levenshtein resolves the typo deterministically.
> AUDIT: the Levenshtein matcher already exists -- `_levenshtein` + `auto_remap_phantom` in `_otr_ledger_reviewer.py` (threshold 3). Reuse it; do not write a second one.

- [x] **DONE `1fa6b40`.** Six typed builders + `make_dispatching_repair_factory` live in the new pure module `nodes/_otr_repair_prompts.py`. The dispatcher routes by error: `json.JSONDecodeError` -> json_syntax, `pydantic.ValidationError` -> schema_field, `PostValidationError` -> classified by message substring (`locked cast` -> cast_membership, `named_character` -> forbidden_name, `dialogue_verb`/`plot_verb` -> narration_leak, `too_long` -> too_many_words), unrecognised content failure -> `default_repair_prompt_factory`.
- [x] **Deterministic no-LLM path.** `structured_call` Attempt 3 now accepts a factory-returned `schema` instance and returns it after a `post_validator` re-check, with no LLM repair call. The outline phase stage supplies a `_phase_cast_phantom_repair` callback that reuses the existing `auto_remap_phantom` (threshold 3) -- no second Levenshtein.
- [x] **Factory -> call-site mapping.** All 8 `structured_call` sites pass `make_dispatching_repair_factory()`: `news_interpreter` (build_news_briefs), `_otr_casting` (cast_one_character), `_otr_ledger_reviewer` x2 (audit_cast_contract + run_script_doctor), `_otr_outline` x3 (macro/phase/beat), `_otr_story_brief` (run_story_brief_reflection). Only the outline phase site -- the one site with a locked cast -- passes a `deterministic_repair` callback. forbidden_name / narration_leak / too_many_words fire at the story_brief site; cast_membership at the outline phase site; json_syntax / schema_field everywhere.

### 2D. Cleanup pass retries (not gated)

- [x] **[DONE 7f3b65f 2026-05-24]** `audit_cast_contract` and `run_script_doctor`: single-shot -> `structured_call` with `max_attempts=4`. Both kept their never-raises contract (`StructuredCallFailedError` + a broad `except Exception` -> the existing `_audit_failed_sentinel` / `needs_full_rerun`). `audit_cast_contract` base 0.2 / structural retry 0.1 / 2000 tok; `run_script_doctor` base 0.5 / structural retry 0.3 / 3500 tok. No node surface touched -- no workflow JSON re-wire.

### 2E. GBNF -- wire or delete  [COMPLETE 2026-05-25 -- DELETED]
Confirmed dead scaffolding: `grammars/news_interpreter.gbnf` + `grammars/style_picker.gbnf` ship, `news_interpreter.py` defines `GRAMMAR_PATH`, the loader never enforces it. Wiring it into `structured_call` Attempt 4 was the v4 preferred path -- but the audit's escape clause ("delete only if loader work proves blocking") triggered at implementation time. The LLM backend is HF Transformers 5.5.0 (`make_generate_fn` -> `model.generate()`), with NO grammar-constrained decoding. GBNF is a llama.cpp format; no `llama-cpp-python`, `transformers-cfg`, or `outlines` is installed. A real wire would need a new dependency (a GBNF `LogitsProcessor`) -- rejected: it breaks offline-first and is a compatibility gamble on the bleeding-edge transformers 5.5.0 stack. Separately, `_otr_style_picker.py` never used `structured_call` at all (it hand-rolls its own two-pass picker), so "wire `style_picker.gbnf` into `structured_call`" was never possible as written. Structural safety already holds without GBNF: the `structured_call` 3-rung ladder + pydantic schema validation + `post_validator`, and the picker's `DESCRIPTOR_RE` regex + distinctness checks.
- [x] **GBNF: deleted** -- `grammars/news_interpreter.gbnf` + `grammars/style_picker.gbnf` removed; `GRAMMAR_PATH` constant + `Path` import dropped from `news_interpreter.py`; the dead `grammar_path` plumbing removed from `structured_call` (param, `_GRAMMAR_TEMPERATURE`, the never-reachable Attempt 4 block, `_invoke_slot`'s TypeError fallback). The ladder is now a clean 3-rung ladder (`_DEFAULT_MAX_ATTEMPTS` 4 -> 3); the three `max_attempts=4` call sites (ledger reviewer x2, story brief) dropped to 3 -- behaviour-identical, Attempt 4 never fired without a `grammar_path`. Stale GBNF docstrings in `_otr_style_picker.py` + a false "uses GBNF grammar-constrained generation" claim in `README.md` corrected.

---

## Sprint 3 -- Task Decomposition (Headline)

### 3A. Split `compose_line`
Confirmed overloaded: `_build_user_prompt` (`_otr_line_composer.py:933-1105`) assembles all 17 v3-listed context blocks (style, theme, canon header, named entities split into people/things, cast/character voice cards, outline spine, current beat, position-or-arc-phase, SFX, last-spoken window, role induction, mood, beat intent, word target). That is not one job.
> AUDIT: `_otr_line_composer.py` already exists (~1660 lines) -- 3A is a **rewrite of an existing file**, effort ~2-3 days, not 1-2. `technical_fn` is a dead param (drop in Sprint 0). The combined `allowed_roster` is *not* rendered into the prompt today -- it feeds the downstream phantom gate only; keep that separation.

New design (`compose_line_draft` does the one creative job; `compose_line` orchestrates draft -> strip -> polish). Critical ordering: deterministic strips run **before** the stripped line is appended to the rolling window, so the next line never inherits a hallucinated name.
> AUDIT: there is **no `LAST_LINES_WINDOW` constant** -- the composer never trims; the caller hands it a pre-trimmed `last_lines` list. If you want a named window size, that is *new* work -- define the constant in the composer and move trimming in, or keep trimming caller-side and drop the v3 reference. `cast_strip` should wrap the existing `auto_remap_phantom` Levenshtein path from `_otr_ledger_reviewer.py`; `vocative_strip` is regex-only.

### 3B. Outline Stage 3 -- adjacency + Python-owned budgets
1. **Inject adjacency context** -- CONFIRMED useful. `_build_beat_user_prompt` (`_otr_outline.py:1032-1065`) today gives the beat *no* neighbour context (docstring: "Beat-localized... NO other beat context"). Add `previous_beat_intent` + `next_beat_intent` + phase summary.
2. ~~**Drop `target_words` from the LLM schema** as a functional fix~~ -- **already non-functional.** `_BeatFleshout` carries `target_words`, but `_allocate_phase_target_words` (`_otr_outline.py:1683-1707`) rebuilds the object with Python's allocation and discards the LLM number. Removing the field is still worth doing as **token-budget cleanup**, but it changes no behaviour -- reclassify it from "fix" to "cleanup."

`BeatFleshOut` becomes `intent / mood / dramatic_function` (drop `target_words`).

### 3C. Split Script Doctor -- Diagnosis then Edit
CONFIRMED: `_DOCTOR_SYSTEM_PROMPT` (`_otr_ledger_reviewer.py:626-655`) asks pacing + voice consistency + arc adherence + a strict JSON `edits` array in one call. CONFIRMED: Doctor input rows (`_render_lines_for_audit:311-323`) carry only `line_id, speaker_role, char_id, text` (text truncated to 200 chars) -- it is asked to judge pacing with no `beat_intent`, `arc_phase`, `mood`, or word counts per row.

Two fixes, both required: (1) feed enriched rows (`beat_id, arc_phase, beat_intent, mood, target_words, actual_words, text`); (2) split into `run_script_doctor_diagnosis` (free/structured-prose, names the failure per line, no edits) then `run_script_doctor_edits` (strict JSON edit array, takes diagnostics as input, cannot rewrite a line whose diagnostics named no failure). `run_script_doctor` becomes the orchestrator. +1 LLM call per episode.

### 3D. Split Casting -- Python owns the ensemble
CONFIRMED: `cast_one_character` -> `CastingResponse` = `character_description + gender + voice_preset`; the LLM picks `voice_preset` from a Python-narrowed pool. CONFIRMED: no Python global gender/timbre balance -- only a static prompt line "~40% male, ~40% female, ~20% other" plus a "cast so far" block.

New design: `precompute_ensemble_slots` (Python decides gender/timbre/role balance up front), `llm_write_description` (LLM writes the description for one slot only), `python_assign_voice_preset` (Python picks from the pool by timbre). Net LLM call count lower -- voice selection leaves the LLM.
> AUDIT: Python already enforces voice *uniqueness* (`_assert_unique_bark_voices`, pool pre-filter). The new code owns *distribution* too; don't duplicate the uniqueness check.

### 3E. Title -- scratchpad + late binding
CONFIRMED: `_generate_title_from_script` is single-shot; after regen, `OTR_LedgerScriptWriter.py:2687` substitutes the new title into any line text quoting the old outline title -- a **verbatim string match**, so paraphrased/conceptual references to the old title slip through.

Two fixes: (1) force a scratchpad before the final title (extract 3 physical details -> draft 3 candidates -> `TITLE:` line, Python parses the last line); (2) use `EPISODE_TITLE: TBD` in the canon header during composition so no provisional title is ever spoken, and generate the final title from a richer excerpt set (`opening_lines`, `middle_lines`, `ending_lines`, `premise`, `arc_verdict`). Late binding removes the fragile post-hoc substitution entirely.

### 3F. Cast Auditor -- strip floating-point confidence
CONFIRMED: `CastViolation` carries `confidence: float`. CONFIRMED but **imprecise in v3**: the gate is **not** a flat `>= 0.8`. `apply_deterministic_cast_repairs` uses `>= 0.8` for `bad_casing`, `>= 0.9` for `wrong_char_id` and `role_mismatch`, and **no** confidence gate for `alias_used` / `invented_name` / `speaker_unknown`. Small models cannot reliably distinguish 0.7 from 0.8 *or* 0.8 from 0.9 -- the argument holds for both thresholds. Remove confidence scoring from the auditor's job; auditor emits pure anomaly extraction (`found / expected_in_cast / violation_type`); Python decides replacement via exact case-fold match or Levenshtein <= 3, escalating ambiguous ties.

### 3G. Reflection Brief -- sanitize input, drop suppression instructions
CONFIRMED: `_build_reflection_input` (`_otr_story_brief.py:270-280`) emits a `CAST:` block with every character name + description, then `_REFLECTION_PROMPT` (L196-204) tells the model: no cast names, no proper nouns, no dialogue/plot verbs, no invented dates/places. The model is staring at names it is told to suppress.

Pre-sanitize: replace cast names + known proper nouns with neutral tokens (`character_a`, `source_entity`) before the LLM sees them. Prompt collapses to "write a visual atmosphere brief, use no names." Schema-side reject lists stay as a safety net.

---

## Sprint 4 -- VRAM Hardening Protocol (Not Gated)

- [x] **VERIFIED 2026-05-25 (`6b9300e`).** Zero-Prime Wash between every slot swap: `torch.cuda.synchronize()` -> `comfy.model_management.unload_all_models()` -> `torch.cuda.empty_cache()` -> `torch.cuda.ipc_collect()` -> `gc.collect()`. Confirmed present in `load_llm`, `unload_llm`, and `free_otr_pipeline_residue`.
- [x] **VERIFIED 2026-05-25 (`6b9300e`).** Sovereignty Buffer enforced at loader: 2.5 GB free, always. `_otr_model_loader.py` reserves `total_vram - 2.5` GiB when `total_vram >= 12.0`.
- [x] Per-class VRAM caps in loader: 2B->3.2 GiB, 8-9B->6.8 GiB, 12B->6.8 GiB, 14B->10.1 GiB. **RESOLVED 2026-05-25:** 2B / 8-9B / 12B caps confirmed present (`6b9300e`). The 14B cap is resolved as **no-change (Jeffrey 2026-05-25): keep the Sovereignty branch governing 14B** -- a 14B model falls into the `total_vram >= 12.0` branch (13.5 GiB budget on a 16 GB card); no explicit 10.1 GiB cap is added, so the live load path is untouched and no RTX 5080 load-path verification is required for this item.
- [x] **VERIFIED 2026-05-25 (`6b9300e`).** `torch_dtype=bfloat16`, `tf32_matmul=True` globally -- both already present in `_otr_model_loader.py`.
- [x] **VERIFIED + FIXED 2026-05-25 (`6b9300e`).** Attention selector: Flash Attention 2 -> SDPA -> SageAttention, log the choice on every load. **SDPA fallback mandatory on Blackwell sm_120 / Windows / torch 2.10.** The selector was computed + logged but `common_kwargs` hardcoded `sdpa` -- dead code (BUG-LOCAL-272). Fixed: `common_kwargs` consumes `attn_impl`; explicit resolved-value log line on every load.
- [x] Prompt cache on: `cache_prompt=True`, `n_cache_reuse` tuned. **RESOLVED 2026-05-25 as N/A -- llama.cpp parameters, no HF Transformers equivalent.** `cache_prompt` / `n_cache_reuse` are `llama.cpp` server parameters; the OTR backend is HF Transformers 5.5.0 (`make_generate_fn` / `make_polish_generate_fn` -> `model.generate()` in `_otr_model_loader.py`). HF `generate()` already runs the within-call KV cache by default (`use_cache=True`); cross-call prefix-cache reuse has no HF API and OTR's slot calls do not share prompt prefixes. Same class of finding as Sprint 2E's GBNF resolution -- a plan bullet written against a llama.cpp backend. No code change.
- [x] ~~Behavioral HuMo VRAM gate.~~ **ALREADY EXISTS.** `_otr_humo_tier_loader.py`: `vram_safety_threshold_gb` (default `10.0`) + `auto_downgrade` (default `True`); `_resolve_tier` (L285-348) downgrades a high tier to `low_vram_default` or raises when post-cleanup free VRAM is below threshold. Remaining work: confirm it fires for the renamed `high_quality_unsafe_on_16gb` tier and that the threshold default is what you want.
  > AUDIT: the four bullets above (Zero-Prime Wash, Sovereignty Buffer, per-class caps, attention selector) were NOT individually verified against code in this pass -- treat them as design intent to confirm against `_otr_model_loader.py` / `_otr_vram_levers.py` before implementation.

---

## Sprint 5 -- Continuity Ledger + Story Critic + Targeted Reroll
Justification confirmed by the 2026-05-24 LLM-call audit: **no LLM pass anywhere judges story quality** -- every gate is deterministic or structural. This is the direct fix for the `ozempics_glitch`-class failure.

### 5A. Continuity ledger -- BEFORE line composition

New module `nodes/_otr_continuity.py` (confirmed: does not exist). One LLM call after the outline lands, populating `ContinuityState` (known/unknown facts per character, active props, location, elapsed beats). `compose_line_draft` receives a per-speaker `ContinuitySlice` rendered as hard constraints.

### 5B. Whole-script critic -- dimension-walk rubric

One pass post-Script-Doctor. Structured rubric, one dimension at a time: §1 continuity, §2 voice drift, §3 flat lines, §4 arc verdict, §5 reroll targets, §6 render priority. Pydantic schema enforces all six sections. Slot: technical. Retry via `structured_call` (`max_attempts=4`).

### 5C. Targeted reroll

- [ ] Hook `reroll_targets[]` into `compose_line_draft`.
- [ ] Cap at 2 critic->reroll cycles. Cycle 3 -> `needs_full_rerun`.
- [ ] Add `cycle_count` to ledger.

---

## Sprint 6 -- Critic -> Render Coupling
- [ ] `render_selection: dramatic_peaks_only` reads `render_priority[]` from 5B.
- [ ] `flat_lines[]` excluded from render unless rerolled.
- [ ] `arc_verdict in [mid_collapse, flat]` blocks render until critic cycle 2 clears.
- [ ] `render_max_n` default 6. `protagonist_only` + `manual_line_ids` override policies.

---

## Implementation Order Summary (corrected effort)

| Sprint | Scope | Gated | Effort |
|---|---|---|---|
| 0 | `json_str` bug fix + telemetry + 2 cosmetic + scoped CI sweep | No | Hours |
| 1 | HuMo high-tier rename + JSON re-wire; 2 config decisions | No | Hours (3 flags already done) |
| 2A | Shared `structured_call` helper | No | 1 day |
| 2B | Repair temperature fix | No | Folded into 2A |
| 2C | Typed repair prompts (reuse existing Levenshtein) | No | 1 day |
| 2D | Cleanup pass retries via 2A | No | Hours |
| 2E | GBNF -- deleted (loader has no grammar support) | No | Done |
| 3A | Rewrite `compose_line` (~1660-line file) + strips-before-window | No | 2-3 days |
| 3B | Outline Stage 3 adjacency (+ `target_words` token cleanup) | No | 1 day |
| 3C | Split Script Doctor (Diagnosis + Edit) + enrich rows | No | 1-2 days |
| 3D | Split Casting (Python owns ensemble + voice) | No | 1 day |
| 3E | Title scratchpad + late binding | No | 1 day |
| 3F | Cast Auditor: strip confidence (note dual 0.8/0.9 thresholds) | No | Hours |
| 3G | Reflection: sanitize input, drop suppression | No | Hours |
| 4 | Zero-Prime Wash + Sovereignty + attention fallback (VRAM gate already exists -- verify only) | No | 2-3 days |
| 5 | Continuity ledger + critic + targeted reroll | No | Week |
| 6 | Critic->render coupling | No (ships with 5) | Days |
| Appendix A | Optional model additions | No | As needed |

Round-robin consultation: WAIVED 2026-05-24 (Jeffrey) -- already done in earlier sessions. No sprint is gated; the build proceeds sprint-after-sprint.

---

## Open Decisions for Jeffrey -- ALL RESOLVED (2026-05-24)

1. **`humo_max_lines_per_process`** -- **RESOLVED 2026-05-24: stays `0`** (logged `BUG_LOG.md:210` decision not reverted; v3's `6` declined).
2. **`clip_length`** -- **RESOLVED 2026-05-24: leave editable, default `7.0`** (no lock, no code change).
3. **`experimental_gguf` tier** -- **RESOLVED 2026-05-24: leave untouched** (not renamed).
4. **GBNF (2E)** -- **RE-RESOLVED 2026-05-25: DELETE.** The 2026-05-24 "wire" decision was overturned at implementation time: the loader (HF Transformers 5.5.0, no llama.cpp / transformers-cfg / outlines) cannot enforce a GBNF grammar, and `_otr_style_picker.py` never used `structured_call`. Per the audit's own escape clause ("delete only if loader work proves blocking"), the scaffolding was removed rather than adding a dependency to a bleeding-edge stack.
5. **Story-quality critic track (5B)** -- confirm it gets its own roadmap track once this batch lands.
6. **`technical_fn` on `compose_line` (Sprint 0)** -- **RESOLVED 2026-05-24: option (a) -- keep `technical_fn`; item pulled from Sprint 0.** Background: the v4 audit called it dead weight, but `tests/test_helper_paired_signatures.py::test_compose_line_accepts_paired_generators` asserts `technical_fn` MUST exist as a keyword-only param, and the paired `creative_fn`+`technical_fn` contract is deliberate uniformity across 4 sibling helpers, kept for the planned B2/B3/B4 per-sub-pass dispatch. Options: **(a)** keep `technical_fn` -- skip this Sprint 0 item, accept that `compose_line` carries an unused-but-contractual param (recommended -- it stays consistent with its 3 sibling helpers and B2/B3/B4 will need the slot again); **(b)** drop it AND update/retire `test_compose_line_accepts_paired_generators` plus ~21 test call sites across 5 test files, reversing the S32 B1 decision.

---

## Config Block -- Production Default (corrected key names)

```text
# HuMo tiers (real names)
humo_tier_default: low_vram_default
humo_high_tier_alias: high_quality_unsafe_on_16gb     # rename of `high_quality`
humo_third_tier: experimental_gguf                    # exists; left as-is (Decision 3 -- not renamed)

# Render flags (real widget names; all three already default True)
resume_from_ledger: true
cuda_hard_reset_on_oom: true                          # NOT cuda_cleanup_on_oom
stop_workflow_on_soak_cap: true                       # NOT stop_on_soak_cap
humo_max_lines_per_process: 0                         # Decision 1 RESOLVED: stays 0
clip_length: 7.0                                      # Decision 2 RESOLVED: stays editable, 7.0

# HuMo render widget defaults (BatchHumoRender) -- distinct from tier-table values
resolution: 480x832                                   # confirmed
steps: 6                                              # widget default (tier table emits 20 for low_vram)
cfg: 1.0                                              # widget default (tier table emits 5.0 for low_vram)

# Existing VRAM gate (do not rebuild)
vram_safety_threshold_gb: 10.0                        # NOT vram_threshold_gb
auto_downgrade: true

# Structured-call helper (Sprint 2 -- new)
structured_call_max_attempts: 4
structural_retry_temperature: 0.2                     # lower than base, NOT higher
repair_payload_rotation: on

# Task decomposition (Sprint 3 -- new)
compose_line_split: on
strips_run_before_rolling_window: on
outline_stage3_adjacency_window: 1
script_doctor_split: on
casting_python_owns_ensemble: on
title_scratchpad: on
title_late_binding: on
cast_auditor_confidence_scoring: off
reflection_input_sanitization: on
critic_rubric_mode: dimension_walk

# GBNF (Sprint 2E)
gbnf_enforcement: removed                             # 2026-05-25: deleted -- loader has no GBNF support
```

---

## Build Progress Log

*Append-only. One entry per work session. Update the Sprint Status Board above in the same edit. Bug detail goes to `BUG_LOG.md` -- only the `BUG-LOCAL-NNN` pointer appears here.*

### 2026-05-24 -- v4 plan created, no sprint work yet
- v3 plan (`story_pipeline_sprint_plan (1).md`, uploaded) audited against real code at HEAD `0c76ee7`. Four parallel agents read the story-brief, HuMo, outline, casting, reviewer, line-composer, cast-repair, and title-path source. v4 produced with a per-claim corrections table.
- No sprint started. No `BUG-LOCAL` ids opened yet.
- Carry-forward: the `json_str` `NameError` (Sprint 0 item a, `_otr_story_brief.py:648`) is confirmed but NOT yet in `BUG_LOG.md`. The session that fixes it opens the `BUG-LOCAL-NNN` entry first, then tags Sprint 0 here.
- Regression: not run this session (no code touched).

### 2026-05-24 -- Sprint 0 partial: 3 of 5 items landed
- Commit `e7a8eb6` on `v2.0-alpha` (4 files, +308/-27). Predecessor HEAD `0c76ee7`.
- **Landed (3):** (1) `json_str` -> `raw` at `_otr_story_brief.py:648` -- the dead schema-repair arm -> `BUG-LOCAL-268`, logged + fixed + flipped `[FIXED]` same session. (2) Three `helper_context(...)` wraps at the `OTR_LedgerScriptWriter.py` call sites for `generate_outline` / title regen / `run_story_brief_reflection`. (3) Two stale `pick_style` routing comments corrected (S30 routing-table block + the call-site block) -- verified against `_otr_style_picker.py:605-624` (pass 1 inventor -> creative, pass 2 chooser -> technical).
- **New regression:** `tests/test_story_brief_repair_pass.py` -- 4 tests; forces a `StoryBriefModel` `ValidationError` and proves `_repair_pass` executes (the `_repair_pass` spy records zero calls on the pre-fix code).
- **Not landed (2):** `technical_fn` drop from `compose_line` -- BLOCKED; the audit's "dead weight" claim is incomplete -- `technical_fn` is test-enforced paired-contract surface (`tests/test_helper_paired_signatures.py`). Raised as Open Decision 6. CI AST sweep -- deferred, out of this session's scope.
- **Regression:** green 2026-05-24 -- new test 4 passed; `test_core` + `test_audio_byte_identical` + `test_meta_slot_transitions` + `test_writer_slot_routing` + `test_helper_paired_signatures` 89 passed / 2 skipped; story-brief suite 107 passed / 4 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed. 0 failed across every suite.
- **New bug ids:** `BUG-LOCAL-268` (fixed, verified).
- Build run with two parallel subagents, one per file (`_otr_story_brief.py` + new test; `OTR_LedgerScriptWriter.py`) -- zero file overlap.

### 2026-05-24 -- Sprint 0 cont.: technical_fn pulled + stale routing test refreshed
- Commit `51f7226` on `v2.0-alpha` (1 file, +51/-35).
- **Open Decision 6 resolved (Jeffrey's call):** keep `technical_fn` on `compose_line`. The Sprint 0 `technical_fn`-drop item is PULLED -- the param is test-enforced paired-contract surface, not dead weight. Plan amended: the Sprint 0 checklist item is replaced with a PULLED note; Decision 6 marked RESOLVED; corrections-table row already reclassified.
- **5th Sprint 0 item landed:** refreshed the stale `test_pick_style_internally_uses_creative_fn_default` (`tests/test_helper_paired_signatures.py`). It asserted `pick_style` never calls `technical_fn` -- false since S32 B2; it passed only because its inventor mock data tripped the mode-collapse reject so `_run_chooser` was never reached. Renamed to `test_pick_style_routes_inventor_creative_and_chooser_technical`, rewritten with valid inventor data so the chooser pass is genuinely exercised, now asserts pass 1 -> creative + pass 2 -> technical. Module docstring corrected.
- **Regression:** green 2026-05-24 -- `test_helper_paired_signatures` + `test_pick_style_routing` + `test_otr_style_picker` + `test_core` + `test_audio_byte_identical` 125 passed / 2 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed. 0 failed.
- **New bug ids:** none.
- **Sprint 0 state:** 4/5 items landed; `technical_fn` drop pulled (not a defect); only the deferred CI AST sweep remains before Sprint 0 can formally close.
- Built with one subagent on `tests/test_helper_paired_signatures.py`.

### 2026-05-24 -- Wave 1: Sprint 1 (HuMo tier rename) + Sprint 2A step 1 (structured_call helper)
- Commits `df7f9b1` (Sprint 1 -- 3 files, +40/-29) and `61f8cfa` (Sprint 2A -- 2 new files, +869) on `v2.0-alpha`.
- **Open Decisions 1-3 resolved (Jeffrey):** (1) `humo_max_lines_per_process` stays `0` -- no change; (2) `clip_length` stays an editable widget, default `7.0` -- no lock, no code change; (3) `experimental_gguf` tier NOT renamed -- left as-is.
- **Sprint 1 COMPLETE:** HuMo tier `high_quality` -> `high_quality_unsafe_on_16gb` across `_otr_humo_tier_loader.py`, `tests/test_humo_tier_loader.py`, and one `__init__.py` comment. No workflow JSON wires the tier value (verified -- zero occurrences in `workflows/*.json`), so no JSON re-wire was needed. The other Sprint 1 items resolved to no-change per decisions 1-2; the three render-flag defaults were already on.
- **Sprint 2A step 1 LANDED:** new `nodes/_otr_structured_call.py` -- the shared 4-attempt structured-call retry ladder (`structured_call(...)`, `StructuredCallFailedError`, `RepairPromptFactory` Protocol, `default_repair_prompt_factory`). The 2B temperature principle is baked in: the structural retry is LOWER than base, asserted at entry (fails loud). `tests/test_structured_call.py` -- 11 tests over every ladder rung. Still pending in 2A: converting the 6 call sites, the typed repair factories (2C), and GBNF wiring (2E).
- **Regression:** green 2026-05-24 -- `test_humo_tier_loader` 24 + `test_structured_call` 11 + `test_core` 59 + `test_audio_byte_identical` 9 (1 skipped) = 103 passed / 1 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed. 0 failed.
- **New bug ids:** none.
- Built with two parallel subagents on disjoint file sets (humo loader + its test + `__init__.py`; new `_otr_structured_call.py` + new test).

### 2026-05-24 -- round-robin gating removed from the plan
- Jeffrey waived the round-robin consultation requirement -- these changes were round-robined in earlier sessions. Every "round-robin gated" / round-robin-`BLOCKED` marker is removed from this plan: Status Board (2E + 3A-3G + 5 + 6 -> NOT STARTED), the sprint section headers, the Implementation Order "Gated" column (all No), Open Decision 4 (GBNF -> RESOLVED: wire), and the config block. The build now runs sprint-after-sprint with no consultation gate.
- Plan-only edit; no code touched.
- Note: `CLAUDE.md` still carries a "## Round-Robin Consultation" section as a general project rule. Not edited here -- the instruction scoped to the sprint plan. Flag for Jeffrey if he wants CLAUDE.md amended to match.

### 2026-05-24 -- Wave 2 prep: structured_call extended + _otr_ledger_reviewer converted
- Commits `fed6327` (structured_call extensions -- 2 files, +271/-16) and `7f3b65f` (`_otr_ledger_reviewer` conversion -- 1 file, +73/-48) on `v2.0-alpha`. Predecessor HEAD `cc0e85a`.
- **Wave 2 blocker found + fixed.** Auditing the six call sites against the shipped `structured_call` (`61f8cfa`) surfaced two ways the helper could not host them faithfully: (1) four of the five target files run CONTENT validation beyond the pydantic schema that drives a retry -- casting voice-pool membership, news `v1/v2/v3` validators, story-brief `_validate_brief`, outline `_run_call_with_retry` `extra_check`; (2) the helper hardcoded `max_new_tokens=512` but the real passes need 160 (story brief) to 2000 (cast audit) to 3500 (script doctor) -- 512 would truncate the auditor's violations array and the doctor's edits array. `structured_call` gained two keyword-only params: `post_validator(instance) -> str|None` (a content check; a non-None return raises the new `PostValidationError(ValueError)` and advances the ladder exactly like a schema failure, feeding the typed-repair factory) and `max_new_tokens` (per-caller budget, default 512). 7 new tests (11-17 in `tests/test_structured_call.py`).
- **Sprint 2D -- `_otr_ledger_reviewer` converted.** `audit_cast_contract` (base 0.2 / structural retry 0.1 / 2000 tok) and `run_script_doctor` (base 0.5 / structural retry 0.3 / 3500 tok) each replaced their single-shot call + 3 hand-rolled failure arms with `structured_call(max_attempts=4)`. Both keep their never-raises contract: `StructuredCallFailedError` -> the existing `_audit_failed_sentinel` / `needs_full_rerun` report, plus a broad `except Exception` so a raising slot fn (LLM loader error) maps to the same verdict the prior `except Exception` on the generate call produced. No node `INPUT_TYPES` / widget / socket changed -- both are internal functions -- so no workflow JSON re-wire (Prime Directive 3 N/A).
- **Regression:** green 2026-05-24 -- `test_structured_call` 18 passed; ledger suites (`test_phase3_ledger_reviewer`, `test_script_doctor_hardfail`, `test_cast_repair`, `test_cast_contract`, `test_legacy_audit_clean`, `test_no_phase_9_call_b3`) 94 passed; full OTR suite green (exit 0, 2643 passed / 21 skipped); Bug Bible 16 passed / 7 skipped / 3 xfailed. 0 failed.
- **New bug ids:** none.
- **Wave 2 state:** 1 of 5 files converted (`_otr_ledger_reviewer.py` -- 2 of 6 call sites). Remaining: `_otr_story_brief.py`, `news_interpreter.py`, `_otr_casting.py`, `_otr_outline.py`. Per-file conversion notes + the two open wrinkles (casting's `validation_fn` repair-slot routing, which `structured_call`'s single `slot_fn` cannot express; news's `NewsBriefs` subset-key construction) are recorded in `session_handoff.md`.
- Built lead-only (no subagents): the `structured_call` extension is a shared-module change (not file-disjoint, must be serial), and the ledger conversion establishes the conversion pattern for the remaining four files.

### 2026-05-24 -- Wave 2: four structured_call call-site conversions

- Commits on `v2.0-alpha`: `3f41fc8` (`_otr_story_brief` -- 5 files, +234/-474), `b4c6e83` (`news_interpreter` -- 1 file, +129/-146), `6e2950d` (`_otr_casting` -- 3 files, +144/-151), `476eabc` (`_otr_outline` -- 2 files, +139/-271). Predecessor HEAD `f996544`.
- **All four remaining structured-JSON call sites converted onto the shared `structured_call` retry ladder.** Each helper's hand-rolled call -> parse -> validate -> repair loop is replaced by one `structured_call`; the content validation that drove each loop's retry moves onto `post_validator`; `StructuredCallFailedError` plus a broad `except` (structured_call does not catch slot-fn exceptions) map to each function's existing failure contract. Every converted pass's structural retry now LOWERS temperature (Sprint 2B) -- story_brief / news / casting / outline Stages 1+3 previously RAISED it.
- `run_story_brief_reflection`: never-raises / 8-key meta-delta contract preserved. `build_news_briefs`: full-dict `model_validate` (`NewsBriefs` extra="ignore"), a slot-call counter preserves the `attempts` telemetry the writer logs. `cast_one_character`: voice-pool check on `post_validator`; the `attempts`-list length is rebuilt so `lock_cast`'s `CastValidationLLMError` promotion still fires on a full exhaustion. `generate_outline`: 3 stages converted, `_run_call_with_retry` + `_REPAIR_PROMPT_TEMPLATE` deleted, the BUG-LOCAL-259 deterministic Stage 2 fallback + singleton-cast skip preserved.
- **S32 B3 reversed.** casting's `validation_fn` (technical-slot repair routing) removed -- `structured_call`'s single `slot_fn` cannot switch slots per attempt. Resolves the casting wrinkle from the Wave 2 handoff (Jeffrey 2026-05-24: full convert).
- Dead code removed: story_brief `_repair_pass` / `_build_repair_messages` / repair-temp constants; news + casting `_REPAIR_RAW_CAP_CHARS`; outline `_run_call_with_retry` / `_REPAIR_PROMPT_TEMPLATE`. Obsolete tests pruned / rewritten to the ladder contract; `tests/test_story_brief_clamp_logging.py` removed (its SA-101 repair-pass clamp log no longer exists).
- No node `INPUT_TYPES` / widget / socket changed -- all five are internal helpers -- so no workflow JSON re-wire (Prime Directive 3 N/A).
- **Regression after each file:** full OTR suite green (2637-2639 passed / 21 skipped across the four runs; collected count drops 2 as obsolete outline tests were pruned); Bug Bible 16 passed / 7 skipped / 3 xfailed; audio-byte-identical green. 0 failed.
- **New bug ids:** none.
- **Follow-up queued:** Jeffrey 2026-05-24 -- remove the S32 B1 unused paired-signature params (`lock_cast.technical_fn`, `build_news_briefs.creative_fn`, `compose_line.technical_fn`) as a dedicated commit. Reverses Sprint 0 Decision 6 (keep `technical_fn`). [DONE -- see entry below.]

### 2026-05-25 -- S32 B1 unused paired-signature params removed

- Commit `6940209` on `v2.0-alpha` -- 13 files, +168/-341.
- The S32 B1 "paired contract" handed all four writer helpers both `creative_fn` + `technical_fn` for call-site uniformity, even where a helper never used one slot. With Wave 2 done and S32 B3 reversed, three params were dead weight and are removed: `lock_cast.technical_fn`, `build_news_briefs.creative_fn`, `compose_line.technical_fn`. `pick_style` keeps both -- inventor runs creative, chooser runs technical.
- **Reverses Sprint 0 Decision 6** (keep `lock_cast.technical_fn`). The decision is superseded: the slot it fed -- `cast_one_character`'s `validation_fn` -- was removed with the casting structured_call conversion (`6e2950d`), so the param fed nothing.
- Writer: the four helper call sites in `OTR_LedgerScriptWriter.py` now pass only the slot kwargs each helper accepts. `test_writer_paired_wiring.py`'s AST tripwire was updated to check per-helper expected kwargs.
- No node `INPUT_TYPES` / widget / socket changed -- no workflow JSON re-wire (Prime Directive 3 N/A).
- **Regression:** full OTR suite 2636 passed / 21 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed; audio-byte-identical green. 0 failed. (The first sweep run surfaced 21 failures in three `compose_line` caller test files -- `test_phase0_name_roster`, `test_phase1_composer_prompt`, `test_vocative_drift` -- all fixed in the same commit.)
- **New bug ids:** none.

### 2026-05-25 -- Sprint 2E: GBNF scaffolding deleted

- Commit `b2b3f7d` on `v2.0-alpha` (code -- 9 files, +52/-227). Predecessor HEAD `65f586e`.
- **Sprint 2E COMPLETE -- resolved as DELETE.** Open Decision 4 (2026-05-24: "wire") was overturned at implementation time. The v4 audit's escape clause -- "delete only if loader work proves blocking" -- triggered: the LLM backend is HF Transformers 5.5.0 (`make_generate_fn` -> `model.generate()`) with no grammar-constrained decoding; GBNF is a llama.cpp format and no `llama-cpp-python` / `transformers-cfg` / `outlines` is installed. A genuine wire would need a new dependency (a GBNF `LogitsProcessor`), rejected as offline-first-breaking and a compatibility gamble on the bleeding-edge transformers 5.5.0 stack. Separately, `_otr_style_picker.py` never used `structured_call` -- the plan's "wire `style_picker.gbnf` into `structured_call`" was not possible as written.
- **Removed:** `grammars/news_interpreter.gbnf` + `grammars/style_picker.gbnf`; `GRAMMAR_PATH` constant + `Path` import from `news_interpreter.py`; from `_otr_structured_call.py` the `grammar_path` param, `_GRAMMAR_TEMPERATURE`, the never-reachable Attempt 4 block, and `_invoke_slot`'s TypeError fallback. The ladder is now a clean 3-rung ladder (`_DEFAULT_MAX_ATTEMPTS` 4 -> 3); the three `max_attempts=4` call sites (`_otr_ledger_reviewer` x2, `_otr_story_brief`) dropped to 3 -- behaviour-identical, Attempt 4 never fired without a `grammar_path`. Stale GBNF docstrings in `_otr_style_picker.py` and a false "uses GBNF grammar-constrained generation" claim in `README.md` corrected.
- **Structural safety unaffected:** the `structured_call` 3-rung ladder (base -> structural retry -> typed repair) + pydantic schema validation + `post_validator` content checks, plus the style picker's `DESCRIPTOR_RE` regex + distinctness rule, already cover what GBNF would have enforced.
- No node `INPUT_TYPES` / widget / socket changed -- no workflow JSON re-wire (Prime Directive 3 N/A).
- **Regression:** full OTR suite 2635 passed / 21 skipped (baseline 2636; -1 = the removed Attempt-4 test `test_no_grammar_path_ends_ladder_at_attempt_three`); Bug Bible 16 passed / 7 skipped / 3 xfailed; audio-byte-identical green. 0 failed.
- **New bug ids:** none.
- **Sprint state:** 2E COMPLETE. Remaining: Sprint 2C (typed repair prompts), Sprint 0's deferred CI AST `# LLM slot:` sweep, Sprints 3A-3G, 4, 5, 6.
- Built lead-only (no subagents): a single shared-module change (`_otr_structured_call.py`) plus tightly coupled call-site + docstring edits across five files.

### 2026-05-25 -- Sprint 2C: typed repair prompts by failure class

- Commit `1fa6b40` on `v2.0-alpha` (code -- 9 files, +950/-14). Predecessor HEAD `5f1fca7`.
- **Sprint 2C COMPLETE.** New pure module `nodes/_otr_repair_prompts.py`: six typed `RepairPromptFactory` builders (`json_syntax_repair`, `schema_field_repair`, `cast_membership_repair`, `too_many_words_repair`, `narration_leak_repair`, `forbidden_name_repair`) plus `make_dispatching_repair_factory`. The dispatcher routes a `structured_call` Attempt 3 failure by inspecting the error: `json.JSONDecodeError` -> json_syntax, `pydantic.ValidationError` -> schema_field, `PostValidationError` -> classified by message substring (`locked cast` -> cast_membership, `named_character` -> forbidden_name, `dialogue_verb`/`plot_verb` -> narration_leak, `too_long` -> too_many_words), any unrecognised content failure -> the generic `default_repair_prompt_factory`.
- **`structured_call` extension.** A repair factory may now return a finished `schema` instance instead of a repair prompt. The Attempt 3 block detects it, runs it through `post_validator`, and returns it with NO LLM repair call. This is the deterministic cast-membership path the v4 plan requires -- when `auto_remap_phantom` (threshold 3) resolves a phantom speaker unambiguously, no LLM call fires. A deterministic "fix" that is still content-invalid is still caught: it fails `post_validator` and the ladder exhausts loudly.
- **No second Levenshtein.** The outline phase stage supplies a deterministic callback (`_phase_cast_phantom_repair`) that reuses the project's existing `auto_remap_phantom` from `_otr_ledger_reviewer.py` via a lazy import (keeps `_otr_outline` off the reviewer module-load import graph).
- **All 8 `structured_call` sites wired** to `make_dispatching_repair_factory()`: `news_interpreter`, `_otr_casting`, `_otr_ledger_reviewer` x2 (audit + doctor), `_otr_outline` x3 (macro/phase/beat), `_otr_story_brief`. Only the outline phase site -- the one call site with a locked cast -- passes a `deterministic_repair` callback.
- No node `INPUT_TYPES` / widget / socket changed -- all changes are internal helpers -- so no workflow JSON re-wire (Prime Directive 3 N/A). No new LLM call introduced: the typed factories only reshape the prompt of the existing Attempt 3 repair call, and the deterministic path removes a call (Prime Directive 6 `# LLM slot:` tag N/A).
- **Tests:** new `tests/test_repair_prompts.py` (18 tests -- six builders + dispatch routing + deterministic short-circuit); `tests/test_structured_call.py` +3 (factory-returns-instance path, coverage map renumbered to 19).
- **Regression:** full OTR suite 2656 passed / 21 skipped (baseline 2635; +21 = the new tests); Bug Bible 16 passed / 7 skipped / 3 xfailed; audio-byte-identical 9 passed / 1 skipped. 0 failed.
- **New bug ids:** none.
- **Sprint state:** 2C COMPLETE. Remaining: Sprint 0's deferred CI AST `# LLM slot:` sweep, Sprints 3A-3G, 4, 5, 6.

### 2026-05-25 -- Sprint 0 CLOSED: CI AST `# LLM slot:` sweep + parse-failure hardening

- Commits `c99fdfb` (CI sweep -- 6 files, +290) and `b3d6355` (sweep hardening -- 2 files, +89/-5) on `v2.0-alpha`. Predecessor HEAD `4742d8c` / `6349301`.
- **Sprint 0 COMPLETE.** The last open Sprint 0 item -- the deferred CI AST `# LLM slot:` sweep -- landed in `c99fdfb`, plus a QA hardening pass in `b3d6355`. All five Sprint 0 items are now `[x]`; the Status Board flips `0` to COMPLETE.
- **`c99fdfb` -- CI AST sweep.** New `docs/_s28_llm_slot_sweep.py`: AST-walks every `*.py` under `nodes/`, finds all `structured_call` / `generate_fn` / `creative_fn` / `technical_fn` / `polish_generate_fn` / `request_slot` call sites, and verifies a `# LLM slot:` tag within ±8 lines. Exempt files: internal plumbing (`_otr_structured_call`, `_otr_repair_prompts`, `_otr_model_loader`, `_otr_loader_backends`, `_otr_creative_prompt_router`, `_otr_json`), the writer scheduler (`OTR_LedgerScriptWriter.py` -- `request_slot` is pass-through plumbing there), and `vram_context_test.py`. 20 call sites found, all 20 tagged. Same commit added 10 logical `# LLM slot:` tags (16 comment lines) across `_otr_style_picker.py`, `_otr_outline.py`, `_otr_ledger_reviewer.py`, `_otr_line_composer.py`, and the 4-test regression `tests/test_llm_slot_sweep.py`.
- **`b3d6355` -- QA hardening.** QA of `c99fdfb` found one real gap: `find_llm_call_sites` does `except (SyntaxError, ValueError): continue`, so a node file that fails to AST-parse is silently dropped -- its call sites become invisible and the sweep reports a clean pass for it. The floor-count test (≥12 of ~20) would not reliably catch one file dropping out. Added `find_parse_failures(nodes_dir)` -- the loud counterpart; `main()` runs it first and exits 1 on any unparseable file. `find_llm_call_sites` still swallows so the audit never crashes. +2 tests (no node file fails to parse; synthetic parse-failure catch) -- `tests/test_llm_slot_sweep.py` now 6 tests. Not logged as a `BUG-LOCAL` -- defensive hardening of a new CI script, no shipped defect (0 parse failures in `nodes/` today).
- No node `INPUT_TYPES` / widget / socket changed -- CI script + tests only -- so no workflow JSON re-wire (Prime Directive 3 N/A). No new LLM call (Prime Directive 6 N/A).
- **Regression:** sweep CLI 20/20 tagged, 0 parse failures, exit 0; `test_llm_slot_sweep` 6 passed; full OTR suite 2662 passed / 21 skipped (baseline 2660; +2 = the new hardening tests); Bug Bible 16 passed / 7 skipped / 3 xfailed. 0 failed across every suite.
- **New bug ids:** none.
- **Sprint state:** Sprints 0, 1, 2A-2E all COMPLETE. Remaining: Sprints 3A-3G, 4, 5, 6.

### 2026-05-25 -- Sprints 3F + 3G (parallel subagents)

- Commits `e14d364` (3F -- 2 files, +446/-55) and `088aba1` (3G -- 2 files, +481/-19) on `v2.0-alpha`. Predecessor HEAD `bdb2333`.
- Built with **two parallel subagents on disjoint file sets** -- 3F on `nodes/_otr_ledger_reviewer.py` + `tests/test_phase3_ledger_reviewer.py`; 3G on `nodes/_otr_story_brief.py` + `tests/test_story_brief_c5a1.py`. Zero file overlap. Neither subagent committed; the lead ran the authoritative combined-tree regression and committed each sprint separately.
- **Sprint 3F COMPLETE.** Removed the `confidence: float` field from `CastViolation` -- small models cannot reliably distinguish 0.7/0.8 or 0.8/0.9, so the auditor's self-score was noise driving a real gate. The auditor now does pure anomaly extraction (`found` / `expected` / `kind`); `_AUDITOR_SYSTEM_PROMPT` dropped all confidence instructions. New `_resolve_cast_member` resolves the auditor's `expected` deterministically -- exact case-fold match first (returns the canonical roster spelling), then the EXISTING `auto_remap_phantom` / `_levenshtein` (threshold 3); ambiguous ties return `None`. No second Levenshtein. `apply_deterministic_cast_repairs` -- `bad_casing` / `wrong_char_id` / `alias_used` no longer gate on confidence; each applies the repair only on a unique resolution and escalates an unresolved or ambiguous-tie row to the Script Doctor. `role_mismatch` keeps its allowed-role enum check. The never-raises contract + `# LLM slot: technical` tag preserved.
- **Sprint 3G COMPLETE.** `_build_reflection_input` now runs a deterministic strip BEFORE the LLM sees the text -- every cast-name surface form maps to a neutral `character_*` token (all forms of one character collapse onto one token); proper nouns map to `source_entity_*` tokens; the mapping is stable within one call. The proper-noun sweep is conservative (multi-word Title Case always; lone capitalized words only mid-sentence) so descriptive prose keeps its visual signal. `_REFLECTION_PROMPT` -- the now-redundant "No cast names. No proper nouns." line collapses to a concise positive "use no names" instruction; the dialogue/plot-verb + invented-period guidance stays. The OUTPUT safety net (`_validate_brief` reject lists) is untouched and proven still live by new tests.
- No node `INPUT_TYPES` / widget / socket changed in either sprint -- both touch internal helpers / a pure module -- so no workflow JSON re-wire (Prime Directive 3 N/A). No LLM call added or removed (Prime Directive 6 N/A).
- **Regression (authoritative combined-tree run by the lead):** full OTR suite 2692 passed / 21 skipped (2713 collected; +30 vs the 2662 baseline = the new 3F+3G tests); Bug Bible 16 passed / 7 skipped / 3 xfailed; audio-byte-identical green; LLM-slot sweep 6 passed. 0 failed.
- **New bug ids:** none.
- **Minor follow-up nit (not blocking):** 3G's `_PROPER_NOUN_STOPWORDS` frozenset lists `"interior"` twice -- harmless (set dedupes), cosmetic cleanup only.
- **Sprint state:** Sprints 0, 1, 2A-2E, 3F, 3G COMPLETE. Remaining: 3A (compose_line rewrite -- 2-3 days), 3B/3C/3D/3E (node-surface refactors, operator-gated -- need a live ComfyUI episode run to validate), 4 (VRAM -- live-hardware-gated), 5, 6.

### 2026-05-25 -- Wave 1: Sprints 3B + 3C + 3D + 3E (four parallel subagents)

- Commits on `v2.0-alpha`: `3992607` (3B), `74438ff` (3C), `5fe9931` (3D), `d230cd6` (3E). Predecessor HEAD `2374079`.
- Built with **four parallel subagents on disjoint file sets** -- 3B on `nodes/_otr_outline.py`, 3C on `nodes/_otr_ledger_reviewer.py`, 3D on `nodes/_otr_casting.py`, 3E on `nodes/OTR_LedgerScriptWriter.py`, plus each lane's own test files. Zero file overlap. No subagent committed or pushed; the lead ran the authoritative combined-tree regression and committed each sprint separately.
- **Sprint 3B COMPLETE.** `_build_beat_user_prompt` injects adjacency context -- the previous beat's real generated intent, the next beat's speaker, and a phase summary (new `_phase_summary()` reusing `ARC_PHASE_GUIDANCE`). `_BeatFleshout` reduced to `intent / mood`; the LLM `target_words` field dropped (Python's `_allocate_phase_target_words` was already authoritative -- behaviour-neutral token cleanup; allocation now flows to the combiner via a `beat_allocations` channel). NOTE: the plan asked for `next_beat_intent`, but Stage 3 fleshes beats sequentially so a later beat's intent does not exist at generation time -- `next_beat_speaker` (from the phase skeleton) is the available forward signal. A true `next_beat_intent` would need a two-pass-per-phase restructure.
- **Sprint 3C COMPLETE (headline split done; row enrichment partial).** `run_script_doctor` split into `run_script_doctor_diagnosis` (names the per-line failure, no edits) and `run_script_doctor_edits` (strict JSON edit array; `_drop_undiagnosed_edits` deterministically drops any edit on a line the diagnosis did not flag); `run_script_doctor` is now the orchestrator and keeps its never-raises contract. New `_render_lines_for_doctor` enriches rows to `line_id, beat_id, arc_phase, mood, actual_words, text` (up from 4 fields). The plan's `beat_intent` + `target_words` are NOT delivered: they are not persisted on the production ledger (transient outline only) -- not fabricated, no socket added; they render conditionally if a future stamping change adds them. **Follow-up:** stamping `beat_intent` / `target_words` onto the ledger is a separate change in `OTR_LedgerScriptWriter.py` / `production_ledger.py`.
- **Sprint 3D COMPLETE.** `cast_one_character` restructured into three single-job stages: `precompute_ensemble_slots` (Python owns gender / timbre / role distribution, largest-remainder ~40/40/20), `llm_write_description` (the lone LLM call -- prose description only), `python_assign_voice_preset` (Python picks the voice by gender + timbre, reusing the existing `_assert_unique_bark_voices` guard). `CastingResponse` and `lock_cast` signatures unchanged -- the writer call site is untouched.
- **Sprint 3E COMPLETE.** `_generate_title_from_script` rewritten to a forced scratchpad (DETAILS -> CANDIDATES -> a parsed `TITLE:` line; token budget 24 -> 160; new `_build_title_excerpt_set` for whole-arc grounding). Late binding: the canon header carries the literal `EPISODE_TITLE: TBD` during composition, so no provisional title is ever spoken; the post-hoc verbatim title substitution (`_substitute_title_in_text`, Section J.6, the `meta.title_substitution` stamp) is deleted.
- **Prime Directive 6.** 3C adds +1 LLM call/episode (the diagnosis pass). Both `run_script_doctor_diagnosis` and `run_script_doctor_edits` carry a `# LLM slot: technical` tag; the model id is reused from the in-scope technical slot (no new widget). The CI AST sweep now finds 21 call sites, all 21 tagged (was 20). 3E's single title call carries `# LLM slot: creative` (the writer is sweep-exempt -- tagged and verified by eye).
- **Prime Directive 3.** No lane changed a node `INPUT_TYPES` / widget / output socket -- all four touch internal helpers or internal title flow -- so no workflow JSON re-wire was needed.
- `test_no_phase_9_call_b3.py`'s LLM-call-count assertion was updated 2 -> 3 to match 3C's planned +1 call; the Phase 9 retirement guard it protects is unchanged. Not a bug -- a planned consequence.
- **Regression (authoritative combined-tree run by the lead):** full OTR suite 2781 passed / 21 skipped (2802 collected); Bug Bible 16 passed / 7 skipped / 3 xfailed; LLM-slot sweep 21/21 tagged, 0 parse failures. 0 failed across every suite. No BOM, no 0-byte files.
- **New bug ids:** none. Pre-existing nit (not introduced by this wave): `OTR_LedgerScriptWriter.py`'s `__main__` self-test asserts 14 optional widgets but the live `INPUT_TYPES` has 15 (`lemmy_cameo`, BUG-LOCAL-260, added without updating the count). The `__main__` block is not collected by pytest, so no suite is affected -- flagged for a follow-up.
- **Sprint state:** Sprints 0, 1, 2A-2E, 3B, 3C, 3D, 3E, 3F, 3G COMPLETE. Remaining: 3A (compose_line rewrite -- 2-3 days, lead-driven), 4 (VRAM -- live-hardware-gated), 5, 6. Next: ONE live ComfyUI episode run validates the 3B-3E batch (audio-is-king reversion gate, operator-gated).

### 2026-05-25 -- LLM-audit punch list: BUG-LOCAL-271 + 3C enrichment + WORD_BUDGET_DRIFT (three parallel subagents)

- Commits on `v2.0-alpha`: `14818bb` (3C ledger enrichment), `3e120df` (BUG-LOCAL-271), `dfd63ee` (WORD_BUDGET_DRIFT fix). Predecessor HEAD `a294297` (this work sits on the seed-decouple / seed-widget-removal commits `61dda9c` / `906a57f` / `d0ea595`).
- Built with **three parallel subagents on disjoint file sets** -- Lane A on `nodes/_otr_ledger_reviewer.py` (BUG-271), Lane B on `nodes/production_ledger.py` + `nodes/OTR_LedgerScriptWriter.py` (ledger stamping), Lane C on `nodes/_otr_outline.py` (word-budget diagnosis). Zero file overlap. No subagent committed or pushed; the lead integrated, ran the authoritative combined-tree regression, and committed each item separately. Closes the LLM-audit punch-list bug + follow-up items.
- **BUG-LOCAL-271 FIXED (`3e120df`).** The cast auditor emits `wrong_char_id` violations with `expected` = a char_id, but `apply_deterministic_cast_repairs` resolved `expected` as a NAME -- every `wrong_char_id` violation missed and escalated unrepaired to a Script Doctor that flagged none. Fix (approach b): the repair branch builds a case-fold `valid_char_ids` map from `cast_rows` (+ `announcer` fallback) and validates `expected` against it, falling back to `_resolve_cast_member` name resolution only for a name-shaped value; a line already carrying the resolved char_id counts as `repaired` (no escalation -- fixes the small-model over-flagging). `_AUDITOR_SYSTEM_PROMPT` updated so `wrong_char_id.expected` is documented as a char_id and the auditor is told not to flag already-correct char_ids.
- **3C Doctor-row enrichment 5/7 -> 7/7 (`14818bb`).** `Ledger.init_lines_from_outline` + `Ledger.set_lines` now stamp `beat_intent` (from `Beat.intent`) and `target_words` (from `Beat.target_words`) onto every per-line ledger record. `_render_lines_for_doctor` was already pre-wired to render both when present, so it needed only a comment refresh (in `3e120df`). `OTR_LedgerScriptWriter.py` needed no change -- the writer already hands the full outline to the ledger.
- **WORD_BUDGET_DRIFT FIXED (`dfd63ee`).** Root cause: the writer's word-budget check summed `target_words` over ALL beats but compared against a voiced-dialogue-only target; announcer beats' fixed ~15-word overhead tripped a false `ratio=2.00` on small targets. The check now sums voiced (`speaker_role == "character"`) beats only, mirroring `validate_outline_against_budget` validator #1. The outline allocator in `_otr_outline.py` was correct -- Lane C concluded no `_otr_outline.py` edit was warranted.
- No node `INPUT_TYPES` / widget / output socket changed -- all edits are internal logic, new per-line JSON fields, or prompt text -- so no workflow JSON re-wire (Prime Directive 3 N/A). No LLM call added or removed (Prime Directive 6 N/A).
- **Regression (authoritative combined-tree run by the lead):** full OTR suite 2787 passed / 21 skipped (2808 collected); Bug Bible 16 passed / 7 skipped / 3 xfailed. 0 failed.
- **New bug ids:** none.
- **Sprint state:** unchanged -- Sprints 0, 1, 2A-2E, 3B-3G COMPLETE; 3A, 4, 5, 6 remain. Remaining punch-list items: live-validation of the cast/style/seed batch (operator-gated) and Sprints 3A/4/5/6.

### 2026-05-25 -- Sprint 3A (compose_line split, lead-driven) + Sprint 4 (VRAM verify, subagent lane)

- Commits on `v2.0-alpha`: `e24b327` (Sprint 3A), `6b9300e` (Sprint 4 + BUG-LOCAL-272). Predecessor HEAD `263d9cd`.
- Built as two concurrent lanes on disjoint files -- Sprint 3A lead-driven on `nodes/_otr_line_composer.py`; Sprint 4 a parallel subagent on `nodes/_otr_model_loader.py`. The lead ran the authoritative combined-tree regression and committed each sprint separately.
- **Sprint 3A COMPLETE.** `compose_line` split into single-job stages: `compose_line_draft` (the creative job -- generate / format-strip / named-prefix strip / size-band / retry ladder; returns the draft string, raises `LineCompositionFailedError` on exhaustion) and a thin `compose_line` orchestrator (draft -> optional polish -> deterministic strip pipeline -> `LineResult`). New `cast_strip` strip step wraps the reviewer's `auto_remap_phantom` Levenshtein matcher to remap a near-miss phantom name to its cast spelling at compose time, before the line enters the rolling `last_lines` window. `cast_strip` uses `threshold=1` -- deliberately tighter than the reviewer's default 3: the regression guard caught a distance-3 false match ("CARLA" -> the news term "CERN"); compose-time mutation with no story context must fire on slam-dunk typos only, and the reviewer keeps the full threshold-3 with-context pass. `_word_bands` + `_strip_named_prefix` extracted as shared helpers. Behaviour-preserving for the no-near-miss path; `cast_strip` is the one intended behaviour change -- the operator-gated live run validates Prime Directive 1, same pattern as the 3B-3E wave. NOTE: `_build_user_prompt` was left intact -- the audited "New design" only specified the `compose_line` split, and splitting the 17-block prompt builder adds risk for no behaviour gain. The module `__main__` self-test still calls `compose_line` positionally (stale since the signature became keyword-only); not pytest-collected -- flagged as a follow-up nit.
- **Sprint 4 IN PROGRESS** (code-side verify done; operator + decision gates remain). Verify-and-extend audit: Zero-Prime Wash, Sovereignty Buffer (2.5 GB), 2B/8-9B/12B VRAM caps, `bfloat16` + `tf32` -- all confirmed already present. The VRAM gate fires correctly for the renamed `high_quality_unsafe_on_16gb` tier. BUG-LOCAL-272 fixed (dead attention selector -- `common_kwargs` hardcoded `sdpa` instead of consuming the computed `attn_impl`). Open: the 14B 10.1 GiB cap (14B currently falls into the Sovereignty branch -- decision for Jeffrey), the `cache_prompt` bullet (unverified), and the operator live-RTX-5080 confirmation.
- No node `INPUT_TYPES` / widget / output socket changed in either sprint -- both touch internal helpers -- so no workflow JSON re-wire (Prime Directive 3 N/A). No LLM call added or removed (Prime Directive 6 N/A).
- **Regression (authoritative combined-tree run by the lead):** full OTR suite 2787 passed / 21 skipped (2808 collected); Bug Bible 16 passed / 7 skipped / 3 xfailed; audio-byte-identical + LLM-slot sweep green. 0 failed. The regression guard caught one intended-behaviour collision (`test_phase0_name_roster::test_phantom_does_not_trigger_reroll`) mid-build -- resolved by the `cast_strip` threshold=1 decision above; suite green with no test modified.
- **New bug ids:** BUG-LOCAL-272 (attention-selector dead code, FIXED).
- **Sprint state:** Sprints 0, 1, 2A-2E, 3A, 3B-3G COMPLETE; Sprint 4 IN PROGRESS (code-side done). Remaining: Sprint 4 close-out (14B cap decision + prompt-cache + live hardware), Sprints 5 + 6. Next: ONE live ComfyUI episode run validates the 3A batch (audio-is-king reversion gate, operator-gated) -- this run also still covers the carried cast/style/seed + BUG-271 batch.

### 2026-05-25 -- Sprint 4 CLOSED (docs-only close-out)

- Docs-only commit on `v2.0-alpha`. No code touched -- the two open Sprint 4 `[ ]` items both resolve without a code change.
- **14B VRAM cap -- RESOLVED no-change (Jeffrey 2026-05-25): keep the Sovereignty branch.** The 14B class falls into the `total_vram >= 12.0` Sovereignty branch (13.5 GiB budget on a 16 GB card); no explicit 10.1 GiB cap is added. The live load path is untouched, so this item needs no RTX 5080 load-path verification.
- **Prompt-cache bullet -- RESOLVED N/A.** `cache_prompt` / `n_cache_reuse` are `llama.cpp` server parameters with no HF Transformers equivalent. The OTR backend is HF Transformers 5.5.0 (`_otr_model_loader.py` `make_generate_fn` / `make_polish_generate_fn` -> `model.generate()`); HF `generate()` runs the within-call KV cache by default (`use_cache=True`), and cross-call prefix reuse has no HF API. Same class of finding as Sprint 2E's GBNF resolution -- a plan bullet written against a llama.cpp backend.
- **Status Board: Sprint 4 -> COMPLETE.** The only carried item is the operator live-RTX-5080 VRAM confirmation (Prime Directive 1) -- non-blocking and tracked the same way as the 3A operator live-run, so it does not hold the sprint open.
- No node `INPUT_TYPES` / widget / output socket touched -- no workflow JSON re-wire (Prime Directive 3 N/A). No LLM call added or removed (Prime Directive 6 N/A).
- **Regression:** not run -- docs-only change, no code touched. The combined-tree baseline from the `e24b327` / `6b9300e` session stands: full OTR suite 2787 passed / 21 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed.
- **New bug ids:** none.
- **Sprint state:** Sprints 0, 1, 2A-2E, 3A-3G, 4 COMPLETE. Remaining: Sprint 5 (continuity ledger + story critic + targeted reroll) and Sprint 6 (critic -> render coupling).

### 2026-05-25 -- Sprint 5A (continuity ledger) + 5B (whole-script critic)

- Commits on `v2.0-alpha`: `8fef3c5` (5A -- 4 files, +1151), `4b7db99` (5B -- 3 files, +933). Predecessor HEAD `056fca1`.
- Built with **two parallel subagents on disjoint NEW files** (`_otr_continuity.py` + its test; `_otr_story_critic.py` + its test); the lead did all shared-file integration serially and ran the authoritative combined-tree regression.
- **Sprint 5A COMPLETE (`8fef3c5`).** New module `nodes/_otr_continuity.py`: `ContinuityFact` / `ContinuityState` pydantic models; `build_continuity_ledger` -- ONE structured LLM call (technical slot) made right after the outline lands, extracting the episode's narrative facts each tagged `known_by` / `hidden_from` / `established_beat`; the pure `render_continuity_slice` projector. NEVER raises -- degrades to `ContinuityState.neutral()` on any failure (Prime Directive 1). Wiring: `OTR_LedgerScriptWriter.py` section H.5 calls `build_continuity_ledger` on `technical_generate_fn`, stamps `meta["continuity"]`, builds a `beat_index_by_id` map; the per-beat closure `_build_line_request_for_beat` threads `render_continuity_slice` into a new `LineRequest.continuity_slice` field; `_otr_line_composer._build_user_prompt` renders a CONTINUITY CONSTRAINTS block in the per-beat tail (above POSITION / WRITE LINE), gated on a non-empty value. `# LLM slot: technical` tagged by eye (the writer is CI-sweep-exempt). No node surface change -- no workflow JSON re-wire.
- **Sprint 5B COMPLETE (`4b7db99`).** New module `nodes/_otr_story_critic.py`: the 6-section `StoryCriticReport` (continuity_issues, voice_drift, flat_lines, arc_verdict, reroll_targets, render_priority) + sub-models; `run_story_critic` -- ONE LLM call (technical slot) through the shared `structured_call` 3-rung ladder, with a lenient `post_validator` (rejects only reports citing line_ids absent from the ledger). Reuses the Script Doctor's `_render_lines_for_doctor` + `_render_doctor_episode_context`. NEVER raises -- returns `StoryCriticReport.clean()` on any failure. Wiring: `_otr_freeze_cascade.run_freeze_cascade` calls `run_story_critic` on the non-terminal path (after the terminal-verdict short-circuit, before Phase 7) and stamps `meta["story_critic_report"]`. For 5B the report is ADVISORY -- it changes no line text. No node surface change -- no workflow JSON re-wire. `# LLM slot: technical` tag is inside `run_story_critic` (the CI sweep now finds 23 call sites, all tagged -- verified green).
- **Prime Directive 6:** both new LLM call sites are technical-slot structured passes; no node exposes a `model_id` widget; the model id is the slot `generate_fn` threaded in by the writer / cascade. **Prime Directive 1:** both passes never raise -- a continuity or critic failure degrades to a neutral/clean result, never an aborted run.
- **Regression (authoritative combined-tree run by the lead):** after 5A and again after 5B -- full OTR suite 2812 passed / 21 skipped (2833 collected; +25 vs the 2787 baseline = the new continuity + critic tests); Bug Bible 16 passed / 7 skipped / 3 xfailed; LLM-slot sweep green (23/23 tagged). 0 failed. Both new files UTF-8 no BOM, AST-clean.
- **New bug ids:** none.
- **Sprint 5C NOT STARTED -- blocked on an architecture decision.** The reroll must re-compose critic-flagged lines, but the Script Doctor + critic run in the `OTR_LedgerFreezeCascade` node, which keeps only the TECHNICAL model resident and does not persist the outline context (`outline_spine`, `canon_header`, `theme`) that `compose_line_draft` needs. The fork: **(option A)** reroll re-composes on the technical model already resident in the cascade (like the Doctor's existing post-cleanup edit pass) -- no node-surface change, no workflow JSON re-wire, no VRAM swap, tagged `# LLM slot: technical`; deviates from the plan's literal "creative slot" wording; **(option B)** add a `creative_writing_model` input socket to the cascade node + re-wire the workflow JSON + stamp outline context onto `meta` in the writer + reroll via `compose_line_draft` on the creative model -- plan-faithful but adds a technical->creative->technical VRAM swap inside the cascade and a node-surface change needing an RTX 5080 live-run. Either option also needs the writer to stamp `outline_spine` / `canon_header` / `theme` onto `meta`. Five downstream reroll-loop design questions are recorded in `session_handoff.md` ("Sprint 5C open fork"). **Sprint 6 is gated behind the same decision** -- it also adds widgets to the cascade node surface and re-wires the workflow JSON.
- **Sprint state:** Sprints 0, 1, 2A-2E, 3A-3G, 4 COMPLETE; 5A + 5B COMPLETE; 5C + 6 NOT STARTED (blocked on the cascade-reroll architecture decision above).
