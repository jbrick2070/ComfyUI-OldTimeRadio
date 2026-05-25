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
| 0 -- telemetry + cosmetic + hot bug | IN PROGRESS | BUG-LOCAL-268 | 3/5 landed `e7a8eb6` (json_str fix + test, helper_context wraps, pick_style comments); `technical_fn` drop BLOCKED -- Open Decision 6; CI AST sweep deferred |
| 1 -- render seatbelts | NOT STARTED | -- | ~half already done; blocked on decisions 1-3 |
| 2A -- structured_call helper | NOT STARTED | -- | |
| 2B -- repair temp fix | NOT STARTED | -- | folds into 2A |
| 2C -- typed repair prompts | NOT STARTED | -- | |
| 2D -- cleanup pass retries | NOT STARTED | -- | |
| 2E -- GBNF wire/delete | BLOCKED | -- | round-robin + decision 4 |
| 3A -- split compose_line | BLOCKED | -- | round-robin gated |
| 3B -- outline Stage 3 | BLOCKED | -- | round-robin gated |
| 3C -- split Script Doctor | BLOCKED | -- | round-robin gated |
| 3D -- split Casting | BLOCKED | -- | round-robin gated |
| 3E -- title scratchpad | BLOCKED | -- | round-robin gated |
| 3F -- cast auditor confidence | BLOCKED | -- | round-robin gated |
| 3G -- reflection sanitize | BLOCKED | -- | round-robin gated |
| 4 -- VRAM hardening | NOT STARTED | -- | VRAM gate already exists -- verify only |
| 5 -- continuity + critic + reroll | BLOCKED | -- | round-robin gated |
| 6 -- critic->render coupling | BLOCKED | -- | ships with Sprint 5 |

`BLOCKED` here means "gated on a round-robin consultation or an operator decision," not "broken." Sprint 0 and the ungated Sprint 2 items can start now.

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
| GBNF | Decision required Sprint 2E. Wire (preferred) or delete. |

---

## Sprint 0 -- Telemetry, Cosmetic, Hot Bug (Hours, Not Gated)

- [x] **FIX the `json_str` `NameError` -- confirmed bug, do this first.** **[DONE e7a8eb6 2026-05-24 -- BUG-LOCAL-268; +4-test regression `tests/test_story_brief_repair_pass.py`]** `_otr_story_brief.py:648`, inside `run_story_brief_reflection`'s schema-validation repair arm: `_repair_pass(failed_output=json_str, ...)` references `json_str`, which is never bound. The raw LLM output is held in `raw` (bound at L604). The `NameError` is swallowed by the broad `except (Exception, ValidationError) as exc2` at L657, so it never crashes -- instead **every schema-repair attempt becomes an instant `_failure_sentinel(REJECT_SCHEMA)`**. Repair is dead on that arm. Fix: `json_str` -> `raw`.
  > AUDIT: The content-validation repair arm at L679 already passes `failed_output=brief_model.story_brief` correctly -- the bug is isolated to the schema arm. One-word fix; add a regression test that forces a `ValidationError` and asserts `_repair_pass` actually runs.
- [x] Wrap `generate_outline`, `_generate_title_from_script`, `run_story_brief_reflection` in `slot_scheduler.helper_context(...)`. `<unattributed>` bucket -> 0. **[DONE e7a8eb6 2026-05-24 -- wrapped at the 3 `OTR_LedgerScriptWriter.py` call sites; helper names `generate_outline` / `generate_title` / `story_brief_reflection`]**
- [x] Fix stale `pick_style` comment in `OTR_LedgerScriptWriter.py` (Pass 2 / chooser routes to technical, not creative). **[DONE e7a8eb6 2026-05-24 -- two stale comments corrected (S30 routing-table block + the pick_style call-site block); verified against `_otr_style_picker.py:605-624`]**
- [ ] Drop dead `technical_fn` parameter from `compose_line` signature (`_otr_line_composer.py:1449-1461`). **[BLOCKED 2026-05-24 -- audit miss: `technical_fn` is unused internally but is test-enforced paired-contract surface (`tests/test_helper_paired_signatures.py::test_compose_line_accepts_paired_generators` mandates the keyword-only param). Dropping it reverses the S32 B1 decision and turns that test red. See Open Decision 6.]**
- [ ] **CI check (scoped down) -- NOT DONE 2026-05-24 (deferred, out of this session's scope; the existing per-file count tests still hold):** per-file `# LLM slot:` count tests already exist (`tests/test_writer_slot_routing.py`, `tests/test_story_brief_c5a1.py`). Remaining work = a single AST-level sweep that walks every `generate_fn`/`technical_fn`/`creative_fn` call site repo-wide and asserts a `# LLM slot:` tag within N lines. Park next to `docs/_s28_forbidden_sweep.py`.

---

## Sprint 1 -- Render-Side Seatbelts (Config Only, Not Gated)

- [ ] Rename HuMo high tier `high_quality` -> `high_quality_unsafe_on_16gb` **everywhere**: `_TIER_TABLE` + `_TIER_CHOICES` in `_otr_humo_tier_loader.py:101-125`, and every workflow JSON that wires the tier widget (`workflows/otr_scifi_16gb_full.json` and any siblings). Per CLAUDE.md Prime Directive 3, the JSON re-wire is part of "done."
  > AUDIT: `experimental_gguf` is a real third tier v3 never mentioned. Decide whether it gets a parallel rename or stays.
- [x] ~~Defaults on: `resume_from_ledger`, `cuda_cleanup_on_oom`, `stop_on_soak_cap`.~~ **ALREADY DONE.** All three exist and default `True` -- real names are `resume_from_ledger`, `cuda_hard_reset_on_oom`, `stop_workflow_on_soak_cap`. No edit needed; just verify the workflow JSON doesn't override them to `False`.
- [ ] **DECISION NEEDED -- `humo_max_lines_per_process`.** Currently `0` (disabled) by a deliberate change logged at `BUG_LOG.md:210` (bumped 3 -> 0). v3 wants `6`. This reverts a logged decision -- do not apply without confirming. If 6 is wanted, log the reversal in `BUG_LOG.md` with the reason.
- [ ] `clip_length` (real widget name; not `clip_length_seconds`) already defaults `7.0`. To "lock" it you must change the widget to a fixed value or remove operator editability -- a code change, not a default flip. Decide: lock, or leave editable with 7.0 default.

---

## Sprint 2 -- Retry Discipline + GBNF

### 2A. One shared structured-call helper (not gated)

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

### 2C. Typed repair prompts by failure class (not gated)

Repair factory dispatches by class: `json_syntax_repair`, `schema_field_repair`, `cast_membership_repair`, `too_many_words_repair`, `narration_leak_repair`, `forbidden_name_repair`. Cast-membership repair never calls the LLM if Levenshtein resolves the typo deterministically.
> AUDIT: the Levenshtein matcher already exists -- `_levenshtein` + `auto_remap_phantom` in `_otr_ledger_reviewer.py` (threshold 3). Reuse it; do not write a second one.

### 2D. Cleanup pass retries (not gated)

- [ ] `audit_cast_contract` and `run_script_doctor` (post-split): single-shot -> `structured_call` with `max_attempts=4`. Confirmed both are currently one call with failure-to-sentinel.

### 2E. GBNF -- wire or delete (round-robin gated)

Confirmed dead scaffolding: `grammars/news_interpreter.gbnf` + `grammars/style_picker.gbnf` ship, `news_interpreter.py` defines `GRAMMAR_PATH`, the loader never enforces it. Wiring it into `structured_call` Attempt 4 makes every structured pass near-invulnerable to JSON-format failure.
- [ ] Round-robin decision: **wire** (preferred). Delete only if loader work is blocking.

---

## Sprint 3 -- Task Decomposition (Headline)

### 3A. Split `compose_line` (round-robin gated)

Confirmed overloaded: `_build_user_prompt` (`_otr_line_composer.py:933-1105`) assembles all 17 v3-listed context blocks (style, theme, canon header, named entities split into people/things, cast/character voice cards, outline spine, current beat, position-or-arc-phase, SFX, last-spoken window, role induction, mood, beat intent, word target). That is not one job.
> AUDIT: `_otr_line_composer.py` already exists (~1660 lines) -- 3A is a **rewrite of an existing file**, effort ~2-3 days, not 1-2. `technical_fn` is a dead param (drop in Sprint 0). The combined `allowed_roster` is *not* rendered into the prompt today -- it feeds the downstream phantom gate only; keep that separation.

New design (`compose_line_draft` does the one creative job; `compose_line` orchestrates draft -> strip -> polish). Critical ordering: deterministic strips run **before** the stripped line is appended to the rolling window, so the next line never inherits a hallucinated name.
> AUDIT: there is **no `LAST_LINES_WINDOW` constant** -- the composer never trims; the caller hands it a pre-trimmed `last_lines` list. If you want a named window size, that is *new* work -- define the constant in the composer and move trimming in, or keep trimming caller-side and drop the v3 reference. `cast_strip` should wrap the existing `auto_remap_phantom` Levenshtein path from `_otr_ledger_reviewer.py`; `vocative_strip` is regex-only.

### 3B. Outline Stage 3 -- adjacency + Python-owned budgets (round-robin gated)

1. **Inject adjacency context** -- CONFIRMED useful. `_build_beat_user_prompt` (`_otr_outline.py:1032-1065`) today gives the beat *no* neighbour context (docstring: "Beat-localized... NO other beat context"). Add `previous_beat_intent` + `next_beat_intent` + phase summary.
2. ~~**Drop `target_words` from the LLM schema** as a functional fix~~ -- **already non-functional.** `_BeatFleshout` carries `target_words`, but `_allocate_phase_target_words` (`_otr_outline.py:1683-1707`) rebuilds the object with Python's allocation and discards the LLM number. Removing the field is still worth doing as **token-budget cleanup**, but it changes no behaviour -- reclassify it from "fix" to "cleanup."

`BeatFleshOut` becomes `intent / mood / dramatic_function` (drop `target_words`).

### 3C. Split Script Doctor -- Diagnosis then Edit (round-robin gated)

CONFIRMED: `_DOCTOR_SYSTEM_PROMPT` (`_otr_ledger_reviewer.py:626-655`) asks pacing + voice consistency + arc adherence + a strict JSON `edits` array in one call. CONFIRMED: Doctor input rows (`_render_lines_for_audit:311-323`) carry only `line_id, speaker_role, char_id, text` (text truncated to 200 chars) -- it is asked to judge pacing with no `beat_intent`, `arc_phase`, `mood`, or word counts per row.

Two fixes, both required: (1) feed enriched rows (`beat_id, arc_phase, beat_intent, mood, target_words, actual_words, text`); (2) split into `run_script_doctor_diagnosis` (free/structured-prose, names the failure per line, no edits) then `run_script_doctor_edits` (strict JSON edit array, takes diagnostics as input, cannot rewrite a line whose diagnostics named no failure). `run_script_doctor` becomes the orchestrator. +1 LLM call per episode.

### 3D. Split Casting -- Python owns the ensemble (round-robin gated)

CONFIRMED: `cast_one_character` -> `CastingResponse` = `character_description + gender + voice_preset`; the LLM picks `voice_preset` from a Python-narrowed pool. CONFIRMED: no Python global gender/timbre balance -- only a static prompt line "~40% male, ~40% female, ~20% other" plus a "cast so far" block.

New design: `precompute_ensemble_slots` (Python decides gender/timbre/role balance up front), `llm_write_description` (LLM writes the description for one slot only), `python_assign_voice_preset` (Python picks from the pool by timbre). Net LLM call count lower -- voice selection leaves the LLM.
> AUDIT: Python already enforces voice *uniqueness* (`_assert_unique_bark_voices`, pool pre-filter). The new code owns *distribution* too; don't duplicate the uniqueness check.

### 3E. Title -- scratchpad + late binding (round-robin gated)

CONFIRMED: `_generate_title_from_script` is single-shot; after regen, `OTR_LedgerScriptWriter.py:2687` substitutes the new title into any line text quoting the old outline title -- a **verbatim string match**, so paraphrased/conceptual references to the old title slip through.

Two fixes: (1) force a scratchpad before the final title (extract 3 physical details -> draft 3 candidates -> `TITLE:` line, Python parses the last line); (2) use `EPISODE_TITLE: TBD` in the canon header during composition so no provisional title is ever spoken, and generate the final title from a richer excerpt set (`opening_lines`, `middle_lines`, `ending_lines`, `premise`, `arc_verdict`). Late binding removes the fragile post-hoc substitution entirely.

### 3F. Cast Auditor -- strip floating-point confidence (round-robin gated)

CONFIRMED: `CastViolation` carries `confidence: float`. CONFIRMED but **imprecise in v3**: the gate is **not** a flat `>= 0.8`. `apply_deterministic_cast_repairs` uses `>= 0.8` for `bad_casing`, `>= 0.9` for `wrong_char_id` and `role_mismatch`, and **no** confidence gate for `alias_used` / `invented_name` / `speaker_unknown`. Small models cannot reliably distinguish 0.7 from 0.8 *or* 0.8 from 0.9 -- the argument holds for both thresholds. Remove confidence scoring from the auditor's job; auditor emits pure anomaly extraction (`found / expected_in_cast / violation_type`); Python decides replacement via exact case-fold match or Levenshtein <= 3, escalating ambiguous ties.

### 3G. Reflection Brief -- sanitize input, drop suppression instructions (round-robin gated)

CONFIRMED: `_build_reflection_input` (`_otr_story_brief.py:270-280`) emits a `CAST:` block with every character name + description, then `_REFLECTION_PROMPT` (L196-204) tells the model: no cast names, no proper nouns, no dialogue/plot verbs, no invented dates/places. The model is staring at names it is told to suppress.

Pre-sanitize: replace cast names + known proper nouns with neutral tokens (`character_a`, `source_entity`) before the LLM sees them. Prompt collapses to "write a visual atmosphere brief, use no names." Schema-side reject lists stay as a safety net.

---

## Sprint 4 -- VRAM Hardening Protocol (Not Gated)

- [ ] Zero-Prime Wash between every slot swap: `torch.cuda.synchronize()` -> `comfy.model_management.unload_all_models()` -> `torch.cuda.empty_cache()` -> `torch.cuda.ipc_collect()` -> `gc.collect()`.
- [ ] Sovereignty Buffer enforced at loader: 2.5 GB free, always.
- [ ] Per-class VRAM caps in loader: 2B->3.2 GiB, 8-9B->6.8 GiB, 12B->6.8 GiB, 14B->10.1 GiB.
- [ ] `torch_dtype=bfloat16`, `tf32_matmul=True` globally.
- [ ] Attention selector: Flash Attention 2 -> SDPA -> SageAttention, log the choice on every load. **SDPA fallback mandatory on Blackwell sm_120 / Windows / torch 2.10.**
- [ ] Prompt cache on: `cache_prompt=True`, `n_cache_reuse` tuned.
- [x] ~~Behavioral HuMo VRAM gate.~~ **ALREADY EXISTS.** `_otr_humo_tier_loader.py`: `vram_safety_threshold_gb` (default `10.0`) + `auto_downgrade` (default `True`); `_resolve_tier` (L285-348) downgrades a high tier to `low_vram_default` or raises when post-cleanup free VRAM is below threshold. Remaining work: confirm it fires for the renamed `high_quality_unsafe_on_16gb` tier and that the threshold default is what you want.
  > AUDIT: the four bullets above (Zero-Prime Wash, Sovereignty Buffer, per-class caps, attention selector) were NOT individually verified against code in this pass -- treat them as design intent to confirm against `_otr_model_loader.py` / `_otr_vram_levers.py` before implementation.

---

## Sprint 5 -- Continuity Ledger + Story Critic + Targeted Reroll (Round-Robin Gated)

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

## Sprint 6 -- Critic -> Render Coupling (Round-Robin Gated)

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
| 2E | GBNF wire | **Yes** | Days |
| 3A | Rewrite `compose_line` (~1660-line file) + strips-before-window | **Yes** | 2-3 days |
| 3B | Outline Stage 3 adjacency (+ `target_words` token cleanup) | **Yes** | 1 day |
| 3C | Split Script Doctor (Diagnosis + Edit) + enrich rows | **Yes** | 1-2 days |
| 3D | Split Casting (Python owns ensemble + voice) | **Yes** | 1 day |
| 3E | Title scratchpad + late binding | **Yes** | 1 day |
| 3F | Cast Auditor: strip confidence (note dual 0.8/0.9 thresholds) | **Yes** | Hours |
| 3G | Reflection: sanitize input, drop suppression | **Yes** | Hours |
| 4 | Zero-Prime Wash + Sovereignty + attention fallback (VRAM gate already exists -- verify only) | No | 2-3 days |
| 5 | Continuity ledger + critic + targeted reroll | **Yes** | Week |
| 6 | Critic->render coupling | **Yes (with 5)** | Days |
| Appendix A | Optional model additions | No | As needed |

Round-robin gating: 2E, 3A-G, 5A-C, 6.

---

## Open Decisions for Jeffrey (must answer before the relevant sprint)

1. **`humo_max_lines_per_process`** -- leave at `0` (current, logged decision) or set to `6` (v3 request, reverts `BUG_LOG.md:210`)?
2. **`clip_length`** -- truly lock at 7.0 (code change, removes operator editability) or leave editable with the existing 7.0 default?
3. **`experimental_gguf` tier** -- rename in parallel with the high-tier rename, or leave untouched?
4. **GBNF (2E)** -- wire or delete? (round-robin gated)
5. **Story-quality critic track (5B)** -- confirm it gets its own roadmap track once this batch lands.
6. **`technical_fn` on `compose_line` (Sprint 0)** -- the v4 audit called it dead weight, but `tests/test_helper_paired_signatures.py::test_compose_line_accepts_paired_generators` asserts `technical_fn` MUST exist as a keyword-only param, and the paired `creative_fn`+`technical_fn` contract is deliberate uniformity across 4 sibling helpers, kept for the planned B2/B3/B4 per-sub-pass dispatch. Options: **(a)** keep `technical_fn` -- skip this Sprint 0 item, accept that `compose_line` carries an unused-but-contractual param (recommended -- it stays consistent with its 3 sibling helpers and B2/B3/B4 will need the slot again); **(b)** drop it AND update/retire `test_compose_line_accepts_paired_generators` plus ~21 test call sites across 5 test files, reversing the S32 B1 decision.

---

## Config Block -- Production Default (corrected key names)

```text
# HuMo tiers (real names)
humo_tier_default: low_vram_default
humo_high_tier_alias: high_quality_unsafe_on_16gb     # rename of `high_quality`
humo_third_tier: experimental_gguf                    # exists; rename TBD

# Render flags (real widget names; all three already default True)
resume_from_ledger: true
cuda_hard_reset_on_oom: true                          # NOT cuda_cleanup_on_oom
stop_workflow_on_soak_cap: true                       # NOT stop_on_soak_cap
humo_max_lines_per_process: 0                         # DECISION PENDING (v3 wanted 6)
clip_length: 7.0                                      # widget default; "lock" = code change

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

# GBNF (Sprint 2E -- gated)
gbnf_enforcement: wired                               # pending round-robin
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
