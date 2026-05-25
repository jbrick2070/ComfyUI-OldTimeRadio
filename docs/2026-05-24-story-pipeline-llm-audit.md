# Story-Writing + Cleanup Pipeline -- LLM-Call Audit

- **Date:** 2026-05-24 (updated 2026-05-25)
- **Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha` (HEAD `3d6ad91`)
- **Build status (2026-05-25):** Sprints 0, 1, 2A-2E, 3B-3G COMPLETE and pushed. The Sprint 3B-3E wave (commits `3992607` / `74438ff` / `5fe9931` / `d230cd6`) is **live-validated** -- the 2026-05-25 `signal_lost_vances_promise` run completed end-to-end (see the "Live-run validation" section below). Post-wave, the cast + style-picker RNGs were decoupled from the `seed` widget (BUG-LOCAL-269/270) and the `seed` widget was removed entirely (HEAD `d0ea595`). Remaining: 3A, 4, 5, 6. BUG-LOCAL-271 + the 3C Doctor-row enrichment + WORD_BUDGET_DRIFT were fixed 2026-05-25 (commits `3e120df` / `14818bb` / `dfd63ee`) -- see the punch list. See the "Multi-agent execution plan" section for the lane map.
- **Scope:** every LLM call from `OTR_LedgerScriptWriter` (LPL v2.0) through `OTR_LedgerFreezeCascade` -- the path that produces and cleans the episode script. Visual-side LLM calls are out of scope.
- **Method:** systematic call-site inventory across the writer/outline/casting/composer/reviewer/cascade modules, plus verification of the load-bearing findings (GBNF wiring, slot tags).
- **Why:** downstream audio + video quality is gated entirely by script quality. The 2026-05-24 `signal_lost_ozempics_glitch` run is the motivating evidence -- a structurally valid, cast-clean, budget-correct script that was dramatically empty (~12 words of character dialogue, one hallucinated line, `freeze_verdict=needs_full_rerun`), and nothing in the pipeline caught it.

---

## Next-sprint punch list (added 2026-05-25 -- READ FIRST)

**Status 2026-05-25:** punch-list items 1-3 (BUG-LOCAL-271, the 3C
Doctor-row enrichment, WORD_BUDGET_DRIFT) are **RESOLVED and pushed**
(commits `3e120df` / `14818bb` / `dfd63ee`). **Open work to move this
audit to completion: item 4 (live-validation) + items 5-8 (Sprints
3A / 4 / 5 / 6).** Items below stay in priority order; resolved items
keep their detail inline for the build record.

**Bugs to fix**

1. **BUG-LOCAL-271 -- the cast auditor `wrong_char_id` repair is dead.**
   The auditor flags `wrong_char_id` with `expected` set to a char_id;
   the repair (`apply_deterministic_cast_repairs`, `_otr_ledger_reviewer.py`
   ~L815-838) resolves `expected` as a cast NAME, so every `wrong_char_id`
   violation goes unrepaired and escalates to a Script Doctor that flags
   none. Fix: make the auditor prompt and the repair branch agree on
   what `expected` carries (auditor emits a NAME, or the repair accepts
   a char_id), and tighten the auditor prompt so it stops over-flagging
   lines that already carry the correct char_id. Full detail in
   `BUG_LOG.md` -> `### BUG-LOCAL-271`.

   **RESOLVED 2026-05-25 (commit `3e120df`).** Approach (b): the
   `wrong_char_id` repair branch now accepts a char_id in `expected`
   (validated against a case-fold `valid_char_ids` map built from
   `cast_rows`), and the auditor prompt no longer over-flags lines that
   already carry the correct char_id. A line already on the right
   char_id counts as `repaired`, not escalated. Live-run verification
   still pending (operator-gated).

**Follow-ups surfaced by the wave**

2. **3C Doctor-row enrichment is 5/7.** `_render_lines_for_doctor` feeds
   the Script Doctor `beat_id, arc_phase, mood, actual_words, text`;
   `beat_intent` + `target_words` were deferred -- they are not on the
   production ledger. Stamping them needs a change in
   `OTR_LedgerScriptWriter.py` / `production_ledger.py`.

   **RESOLVED 2026-05-25 (commit `14818bb`).** `Ledger.init_lines_from_outline`
   + `Ledger.set_lines` (`production_ledger.py`) now stamp `beat_intent`
   (from `Beat.intent`) and `target_words` (from `Beat.target_words`)
   onto every per-line record. `_render_lines_for_doctor` was already
   pre-wired to render both when present, so the Doctor rows are now
   7/7. `OTR_LedgerScriptWriter.py` needed no change -- the writer
   already hands the full outline to the ledger.
3. **WORD_BUDGET_DRIFT.** The validation run's outline allocated 60
   words against a 30-word target (ratio 2.00). Non-fatal warn -- check
   the Sprint 3B word-budget path.

   **RESOLVED 2026-05-25 (commit `dfd63ee`).** Not an allocator bug --
   the Sprint 3B path in `_otr_outline.py` is correct. Root cause: the
   word-budget check in `OTR_LedgerScriptWriter.py` summed `target_words`
   over ALL beats but compared against a voiced-dialogue-only target;
   announcer beats' fixed ~15-word overhead each forced `ratio=2.00` on
   the 30-word smoke target. The check now sums voiced
   (`speaker_role == "character"`) beats only, mirroring
   `validate_outline_against_budget` validator #1. No `_otr_outline.py`
   change was needed.
4. **Live-validate the open batch (now also covers BUG-271).** One
   ComfyUI episode on current HEAD `3d6ad91` confirms, in a single run:
   BUG-LOCAL-269/270 + the `seed`-widget removal (the cast now varies,
   the writer node has no `seed` widget); BUG-LOCAL-271
   (`audit_cast_contract:pre` repairs `wrong_char_id` violations instead
   of escalating all to the Script Doctor); WORD_BUDGET_DRIFT no longer
   false-fires; the episode completes + freezes clean. **This is the
   immediate next action.**

**Remaining build -- to "complete" this audit**

5. **Sprint 3A** -- DONE 2026-05-25 (`e24b327`). `compose_line` split
   into `compose_line_draft` (the creative job) + a thin orchestrator;
   new `cast_strip` step wraps `auto_remap_phantom` at `threshold=1`.
   Operator live-run still pending (Prime Directive 1). See the plan's
   Build Progress Log.
6. **Sprint 4** -- IN PROGRESS 2026-05-25 (`6b9300e`). Code-side verify
   done: Zero-Prime Wash / Sovereignty Buffer / 2B-12B caps / bf16+tf32
   confirmed present; BUG-LOCAL-272 fixed (dead attention selector).
   Open: 14B 10.1 GiB cap decision, prompt-cache bullet, live-RTX-5080
   confirm.
7. **Sprint 5** -- continuity ledger + story-quality critic + targeted
   reroll. This is the direct fix for Finding C below -- the validation
   run is structurally clean but thin (21 words of character dialogue).
8. **Sprint 6** -- critic -> render coupling (ships with Sprint 5).

Per-sprint detail is in the "Multi-agent execution plan" section below
and in `story_pipeline_sprint_plan_v4_audited.md`. Other carried bugs
(news_interpreter schema, HuMo VRAM probe) stay tracked in `BUG_LOG.md`.

---

## Architecture summary

All generation routes through `_SlotScheduler` (`OTR_LedgerScriptWriter.py`). Two slots -- `creative` and `technical` -- each a `generate_fn(messages, *, temperature, max_new_tokens, stop=None) -> str` closure. The model id is resolved once from the two writer widgets (`creative_writing_model` / `technical_model`) and threaded to every consumer; no other node exposes a `model_id` widget, and the cleanup cascade reads the technical model from a broadcast socket. **Prime Directive 6's wiring rule is satisfied structurally** -- every call site carries a `# LLM slot:` tag, now enforced by a CI AST sweep (`docs/_s28_llm_slot_sweep.py`, commit `c99fdfb`).

**Cross-cutting finding -- GBNF is dead scaffolding.** ~~`grammars/news_interpreter.gbnf` and `grammars/style_picker.gbnf` exist on disk~~ **UPDATE 2026-05-25: Sprint 2E (commit `b8ebdc8`) deleted all `.gbnf` files and scaffolding. This finding is resolved.** The loader never consumed grammar files; all structured passes use plain-text generation + Python-side JSON parsing + Pydantic + reroll via the 3-rung `structured_call` ladder (Sprint 2A/2B/2C).

---

## Per-call-site inventory (20 sites per AST sweep)

> **UPDATE 2026-05-25:** Sprint 0 CI sweep (`docs/_s28_llm_slot_sweep.py`)
> finds 20 AST call sites across `nodes/`. All 20 carry a `# LLM slot:` tag.
> The original manual inventory below identified 15 logical passes (some
> of which expand to multiple AST call sites, e.g. compose_line has a
> primary call + a TypeError-fallback call).

| # | Call site | Job | Slot | Output handling | Retry |
|---|-----------|-----|------|-----------------|-------|
| 1 | `_otr_style_picker._run_inventor` | Invent 5 distinct snake_case style descriptors | creative | free text -> regex grammar (`DESCRIPTOR_RE`) + distinctness | 3 attempts |
| 2 | `_otr_style_picker._run_chooser` | Pick the single best descriptor | technical | free text -> must exactly match a candidate | **single shot, fail-loud** |
| 3 | `news_interpreter.build_news_briefs` | 4 news briefs + key_terms from the article | technical | JSON + Pydantic `NewsBriefs` + V0-V3 validators | 3 attempts (incl. repair) |
| 4 | `_otr_casting.cast_one_character` (per character) | Description + gender + voice_preset | creative (gen) / technical (repair) | JSON + Pydantic `CastingResponse` + voice-pool check | 3 attempts; attempt 3 repair -> technical |
| 5-7 | `_otr_outline` macro / phase / beat stages | Macro shape; per-phase speaker routing; per-beat intent/mood/words | creative | JSON + Pydantic + extra_check | 3 attempts each |
| 8 | `_otr_line_composer.compose_line` (per beat) | One dialogue line | creative | free text -> strip + word-band + phantom/vocative flags | 2 attempts |
| 9 | `_otr_line_composer.polish_line` | Narration-leak cleanup of a composed line | creative | free text -> re-strip + needs_polish recheck | gated by `needs_polish()`; single pass; reverts on failure |
| 10 | `_otr_line_composer.compose_announcer_intro` | Opening announcer line | creative | free text -> `validate_announcer_line` | **single shot**, deterministic fallback |
| 11 | `_otr_line_composer.compose_announcer_outro` | Closing announcer line | creative | free text -> `validate_announcer_line` | **single shot**, deterministic fallback |
| 12 | `OTR_LedgerScriptWriter._generate_title_from_script` | 2-5 word episode title | creative | free text -> first line, wrapper strip, cliche reject | **single shot**, falls back to outline title |
| 13 | `_otr_story_brief.run_story_brief_reflection` | meta-only story-brief reflection | technical | JSON + Pydantic `StoryBriefModel` | 1 fresh + 1 repair |
| 14 | `_otr_ledger_reviewer.audit_cast_contract` | Cleanup Pass 1: cast-contract auditor | technical | JSON + Pydantic `PreAuditReport` | **single shot, no retry** |
| 15 | `_otr_ledger_reviewer.run_script_doctor` | Cleanup Pass 2: Script Doctor structural edits | technical | JSON + Pydantic `ScriptDoctorReport` | **single shot, no retry** |

Between Pass 1 and Pass 2, `apply_deterministic_cast_repairs` runs -- a non-LLM editor that fixes phantom names / casing / char_id where the answer is mechanically known. (`story_orchestrator.py` also carries two `# LLM slot: technical` RSS-rerank tags, but that node is not on the writer->cascade path; excluded.)

> **UPDATE 2026-05-25 -- the inventory above is the 2026-05-24 audit-as-taken; several rows have since shipped:**
> - Rows 14 + 15 (`audit_cast_contract`, `run_script_doctor`) are no longer single-shot -- Sprint 2D routed both through the `structured_call` retry ladder (`max_attempts=3`).
> - Row 13 (`run_story_brief_reflection`) now runs on the `structured_call` ladder (Sprint 2A); Sprint 3G additionally pre-sanitizes its input (cast names + proper nouns -> neutral tokens before the LLM sees them).
> - Row 14 (`audit_cast_contract`): Sprint 3F removed the per-violation `confidence` field -- the auditor now does pure anomaly extraction; Python resolves repairs deterministically (case-fold / Levenshtein).
> - Row 4 (`cast_one_character`): the technical-slot repair routing was removed when casting moved onto `structured_call` (its single `slot_fn` cannot switch slots per attempt).
>
> **UPDATE 2026-05-25 -- the Sprint 3B-3E wave (commits `3992607` / `74438ff` / `5fe9931` / `d230cd6`) further changed the call inventory:**
> - **Sprint 3C (`74438ff`):** Row 15 (`run_script_doctor`) is now an orchestrator over two passes. `run_script_doctor_diagnosis` is a **NEW LLM call site** (`# LLM slot: technical` -- names the per-line failure, emits no edits); `run_script_doctor_edits` (`# LLM slot: technical` -- strict JSON edit array) is the second. The AST sweep now finds **21** call sites (was 20), all 21 tagged. The model id for the new call is reused from the in-scope technical slot -- no new widget. Doctor input rows enriched via the new `_render_lines_for_doctor` (`beat_id, arc_phase, mood, actual_words` added; `beat_intent` / `target_words` deferred -- not persisted on the ledger).
> - **Sprint 3D (`5fe9931`):** Row 4 (`cast_one_character`) -- the LLM call is now `llm_write_description`, writing only the prose description; gender + voice selection moved to Python (`precompute_ensemble_slots` + `python_assign_voice_preset`). Slot unchanged (creative).
> - **Sprint 3B (`3992607`):** Rows 5-7 (`_otr_outline` outline stages) -- the beat-stage prompt gains adjacency context; `target_words` dropped from `_BeatFleshout`. Slots unchanged (creative).
> - **Sprint 3E (`d230cd6`):** Row 12 (`_generate_title_from_script`) -- now a forced-scratchpad pass, still one LLM call (`# LLM slot: creative`); late title binding removes the post-hoc substitution.

---

## Findings

### A. Tagging is complete and CI-enforced

Every call site carries the `# LLM slot:` tag -- the rule itself is met. **As of commit `c99fdfb`, this is enforced by a CI AST sweep** (`docs/_s28_llm_slot_sweep.py`) with a regression suite (`tests/test_llm_slot_sweep.py`). The sweep AST-parses every `*.py` under `nodes/`, finds all `structured_call` / `generate_fn` / `creative_fn` / `technical_fn` / `polish_generate_fn` / `request_slot` invocations, and verifies a `# LLM slot:` tag exists within ±8 lines. Internal plumbing files are exempt. Any new call site without a tag will fail the sweep test. **UPDATE 2026-05-25: commit `b3d6355` hardened the sweep** -- a node file that fails to AST-parse used to be silently swallowed (its call sites invisible, the sweep passing vacuously for it); `find_parse_failures` now surfaces that loud and the suite (6 tests) asserts every `nodes/` file parses. NOTE: `OTR_LedgerScriptWriter.py` is in the sweep's exempt list -- a new untagged LLM call added to the writer will NOT be auto-caught.

~~`slot_calls_by_helper` telemetry attribution is still incomplete -- three call sites bypass `helper_context()`.~~ **RESOLVED 2026-05-25 (Sprint 0, commit `e7a8eb6`):** `generate_outline`, `_generate_title_from_script`, and `run_story_brief_reflection` are now wrapped in `slot_scheduler.helper_context(...)`; the `<unattributed>` bucket is empty.

### B. Slot assignment -- mostly correct, two latent issues

Assignments are sound against the creative/technical definition, with caveats:

- **Outline Stage 2 (per-phase speaker routing) runs on the creative slot but is structured routing** -- it picks a name from a fixed locked cast. The code even acknowledges this ("structured routing, not creative prose") and gave it a falling temperature schedule, but left it wired to the creative generate_fn. Harmless in the default config (creative and technical are the same model) but a latent mis-slot if an operator ever picks a cheaper technical model.
- ~~**`compose_line` accepts a `technical_fn` parameter but hard-overrides to `creative_fn`.**~~ **RESOLVED 2026-05-25 (commit `6940209`):** the unused paired-signature params (`compose_line.technical_fn`, `lock_cast.technical_fn`, `build_news_briefs.creative_fn`) were dropped once Wave 2 + the S32 B3 reversal made them dead weight.
- ~~**Stale comment:** the writer's pick_style call-site comment says it "routes both passes to creative".~~ **RESOLVED 2026-05-25 (Sprint 0, commit `e7a8eb6`):** both stale pick_style routing comments corrected -- Pass 1 inventor -> creative, Pass 2 chooser -> technical.

### C. Missing / underused passes -- the headline finding

**There is no LLM pass anywhere that judges story quality.** Every quality gate in the pipeline is either deterministic (word budgets, regex strips, phantom-name detection, vocative-strip) or structural (the cast-contract audit). What exists is a generator-with-mechanical-guards; what is absent:

1. **No scene/beat continuity check.** Each `compose_line` call sees only a sliding `last_lines` window + the current beat + the outline spine. Nothing reviews the *assembled* dialogue for continuity contradictions -- a character knowing something they shouldn't, a prop appearing then vanishing, time/tense drift. Structural beat order is guaranteed; narrative coherence is not.
2. **No in-character / voice-consistency check.** A voice card is fed into each line prompt, but no pass reads all of a character's lines together to verify a consistent register. Voice drift across a long episode is unguarded.
3. **No dramatic-arc / pacing reviewer.** Word budgets are enforced deterministically and the outline plans intent/mood per beat -- but no pass evaluates whether the finished script actually *delivers* rising tension and payoff. The arc is planned, never reviewed.
4. **No dialogue-quality gate.** `compose_line` checks only: non-empty, word-band, phantom names, vocative drift. Nothing asks whether a line is *good* dialogue -- on-intent, dramatically alive, not flat exposition. A bland but correctly-sized line ships unchallenged.
5. **The cleanup stage is structural-only.** `review_ledger` (auditor + Script Doctor) polices the cast contract exclusively -- phantom names, casing, char_id, role mismatch. The Doctor is explicitly prevented from seeing or improving story content. Cleanup never improves prose, pacing, or coherence.

Directly answering the framing question -- *"are there story-specific LLM calls we are not taking advantage of?"* -- yes, in two senses: (a) an entire category is missing: a story/scene/line **quality critic**; and (b) GBNF grammar enforcement is **built but never wired**. The pipeline can emit a structurally valid, budget-correct, cast-clean script that is dramatically empty, and nothing catches it. The `ozempics_glitch` run is exactly that failure made visible.

### D. Cleanup side -- thin and under-resilient

The LLM cleanup is well-engineered but deliberately narrow (cast hygiene only) and **has no retry**. Pass 1 (`audit_cast_contract`) and Pass 2 (`run_script_doctor`) are each single-shot: any LLM/JSON/schema failure discards the whole cleanup and forces `needs_full_rerun`. This is inconsistent with the rest of the pipeline -- news_interpreter, casting, and the outline all invest in 3-attempt + repair ladders. Without GBNF and without a reroll, a single malformed-JSON response from a 7B-14B model nukes a multi-minute run rather than re-rolling one call. (The `moon_base_countdown` run in the BUG-266 console showed exactly this: the Script Doctor's schema validation failed and the cascade returned `needs_full_rerun`.) The deterministic cast repairs are the genuinely robust part of the cleanup.

---

## Recommendations (priority order)

1. **Add a story-quality LLM pass.** The single highest-value gap. A pass that reads the assembled script and judges continuity / voice consistency / dramatic delivery -- at minimum a per-line or per-scene dialogue-quality critic that can flag or trigger a reroll. New LLM call -> Prime Directive 6 applies; an architecture-level addition -> round-robin gated. This is the direct fix for the `ozempics_glitch`-class failure.
2. **Give the two cleanup passes a 1-repair retry.** Small, consistency fix -- bring the auditor and Script Doctor in line with the 3-attempt discipline used everywhere else, so a transient JSON glitch re-rolls one call instead of forcing a full episode rerun. **UPDATE: Sprint 2A/2D already did this** -- both `audit_cast_contract` and `run_script_doctor` now route through the 3-rung `structured_call` ladder. This recommendation is resolved.
3. ~~**Decide GBNF: wire it or delete it.**~~ **RESOLVED: Sprint 2E (commit `b8ebdc8`) deleted all `.gbnf` files and scaffolding.** The 3-rung `structured_call` ladder (Sprint 2A/2B/2C) provides robust JSON/schema failure handling without grammar enforcement.
4. ~~**Wrap the outline, title regen, and story-brief reflection in `helper_context`.**~~ **RESOLVED 2026-05-25 (Sprint 0, commit `e7a8eb6`):** all three call sites wrapped; `<unattributed>` bucket empty.
5. ~~**Clean the two cosmetic issues.**~~ **RESOLVED 2026-05-25:** stale pick_style slot comment fixed (`e7a8eb6`); the dead `technical_fn` parameter dropped from `compose_line` (`6940209`).

Recommendation 1 is the story-quality critic -- it is **Sprint 5** in `story_pipeline_sprint_plan_v4_audited.md` (round-robin consultation was waived 2026-05-24 for the whole plan). Recommendations 2, 3, 4, and 5 are all resolved. The remaining audited work -- Sprints 3A-3E + 4 + 5 + 6 -- is laid out as parallel-subagent lanes in the next section.

---

## Live-run validation -- 2026-05-25 (3B-3E batch)

Episode `signal_lost_vances_promise_20260525_125401` (episode id `pending_20260525_125025`, commit `1514e11`, 30-word smoke target, `num_characters=1`, `gemma-2-2b-it` on the creative slot) ran end-to-end and is the operator-gated validation of the Sprint 3B-3E wave. `Prompt executed in 00:45:23`; froze `frozen_with_warns` (a PASS verdict, not `needs_full_rerun`); full audio + HuMo/LTX video + final mp4 produced. Prime Directive 1 (audio is king) held.

**3B-3E all confirmed working live:**

- **3B** -- `OTR_Outline` ran the macro / phase / beat stages; the singleton-cast Stage 2 bypass fired (`assigned 'HAYES VANCE' to all 3 beats`); the outline succeeded with 5 beats.
- **3C** -- the split Script Doctor ran both passes (`run_script_doctor_diagnosis` then `run_script_doctor_edits`). The deterministic guard fired exactly as designed: `edits pass proposed an edit on line_id=b003 which the diagnosis did NOT flag with a failure -- dropping it deterministically`.
- **3D** -- casting logged the Python-owned ensemble slot: `cast HAYES VANCE -> voice=v2/en_speaker_5 gender=male (timbre=warm role=lead)`.
- **3E** -- the canon header carried the literal `EPISODE_TITLE: TBD` during composition (late binding); the scratchpad title pass then produced `"Vance's Promise"` from the assembled script.

**Findings from the run:**

1. **BUG-LOCAL-271 -- cast auditor `wrong_char_id` violations all go unrepaired.** `audit_cast_contract:pre` flagged all 5 lines `wrong_char_id` with `expected='c01'/'c02'` (char_ids); `apply_deterministic_cast_repairs` treats `expected` as a NAME and resolved none ("no cast member resolves"); all 5 escalated to the Script Doctor, which flagged 0. Benign on this episode but the `wrong_char_id` auto-repair is dead. See `BUG_LOG.md`.
2. **`WORD_BUDGET_DRIFT ratio=2.00`** -- the outline allocated 60 words against a 30-word target; the episode landed at 45 words total (character=21, announcer=24). Non-fatal warn; worth tracking against the Sprint 3B word-budget path.
3. **Finding C confirmed, not yet fixed.** The episode is structurally valid, froze clean, and is still *thin* -- 21 words of character dialogue across 3 lines, one of which carries a phantom name (`Pandora`). This is exactly the Finding-C failure class. 3B-3E hardened the pipeline's *structure* -- it now runs clean and does not crash -- but *quality* (catching a thin or hollow episode) is **Sprint 5**, which is not yet built. The pipeline "works"; making the output consistently good is the remaining 3A / 5 / 6 work.
4. `post-assembly key_terms ZERO landed` -- none of the 5 news key_terms reached the script. Known + deferred (ADR section 4.4, warn-only); not a regression.

---

## Multi-agent execution plan (remaining sprints)

As of 2026-05-25, Sprints 0, 1, 2A-2E, 3B-3G are COMPLETE and pushed -- the 3B/3C/3D/3E wave (commits `3992607` / `74438ff` / `5fe9931` / `d230cd6`) shipped via four parallel subagents on disjoint files, validating this method again. The remaining audited work is Sprints 3A, 4, 5, 6. This section's lane map is retained for Sprint 3A and as the build-pattern record.

### The method

**Parallel subagents on disjoint file sets.** One subagent per primary file, with zero file overlap, so concurrent edits never clobber each other. The lead orchestrates and integrates serially:

- The lead launches all lanes of a wave in one batch and lets them run concurrently.
- Each subagent reads `CLAUDE.md` + the relevant sprint section of the plan + its target source file, implements the sprint, and runs the regression. It **does not** commit, push, or edit `BUG_LOG.md` / the sprint plan -- those are shared files and would conflict. It reports its files, changes, test results, and any bugs back to the lead.
- The lead runs the **authoritative** full OTR suite + Bug Bible on the combined tree (subagent test numbers interleave when runs overlap and cannot be trusted), then commits one sprint per commit, updates the plan Status Board + Build Progress Log, makes the `docs:` commit, and pushes.
- Tooling: tests + git via Desktop Commander (`shell: "cmd"`, venv python). The Cowork Linux `Bash` mount is stale for this repo -- do not test through it.

This is the method that shipped Sprint 0 hardening and Sprints 3F + 3G; it is the project's validated build pattern (see the Build Progress Log entries that read "built with N parallel subagents on disjoint file sets").

### Lane map -- the Sprint 3 decomposition

Each lane is one parallel subagent. The primary files are mutually disjoint, so lanes B/C/D/E run as a single concurrent wave.

| Lane | Sprint | Primary file (disjoint) | Effort | New LLM call? | Lane-specific notes |
|------|--------|-------------------------|--------|---------------|---------------------|
| A | 3A -- split `compose_line` | `nodes/_otr_line_composer.py` | 2-3 days | no | A rewrite of a ~1660-line file. Too large for a reliable one-shot subagent -- run it lead-driven or as its own dedicated session, NOT inside the parallel wave. |
| B | 3B -- outline Stage 3 | `nodes/_otr_outline.py` | ~1 day | no | Inject `previous_beat_intent` / `next_beat_intent` + phase summary into the beat prompt; drop `target_words` from `_BeatFleshout` (token cleanup -- `_allocate_phase_target_words` already overrides it, so behaviour is unchanged). |
| C | 3C -- split Script Doctor | `nodes/_otr_ledger_reviewer.py` | 1-2 days | **YES (+1/episode)** | Split into `run_script_doctor_diagnosis` (prose, names the per-line failure) then `run_script_doctor_edits` (strict JSON edits, cannot rewrite a line no diagnostic named); feed enriched rows. New LLM call -> **Prime Directive 6**: add a `# LLM slot:` tag at the new call site, wire the model id from the writer's broadcast socket (no new widget), update the routing table. The Sprint 0 CI sweep will fail until the tag is present. |
| D | 3D -- split Casting | `nodes/_otr_casting.py` | ~1 day | net **fewer** | `precompute_ensemble_slots` (Python owns gender/timbre/role balance) -> `llm_write_description` (LLM writes one slot's description) -> `python_assign_voice_preset` (Python picks the voice). Voice selection leaves the LLM. Do not duplicate the existing `_assert_unique_bark_voices` uniqueness check. |
| E | 3E -- title scratchpad | `OTR_LedgerScriptWriter.py` + title path | ~1 day | no | Scratchpad before the final title; `EPISODE_TITLE: TBD` in the canon header during composition; late binding removes the fragile post-hoc string substitution. **WARNING:** `OTR_LedgerScriptWriter.py` is EXEMPT from the CI sweep -- a new untagged LLM call here is NOT auto-caught; tag manually. |

**Wave 1 (parallel):** Lanes B + C + D + E -- four disjoint files, four subagents at once. **[COMPLETE 2026-05-25 -- `3992607` / `74438ff` / `5fe9931` / `d230cd6`.]**
**Lane A (3A):** stands alone -- dedicated effort, not in the parallel wave. **[NOT STARTED -- the 3B-3E live-run validation is done (2026-05-25); 3A is the next build lane.]**

### Dependencies and gating

- Lanes A-E are independent decompositions of separate modules -- no lane depends on another, so B/C/D/E parallelize cleanly. (The plan's "in plan order" sequencing is superseded by the parallel-wave method.)
- **Every lane is regression-gated:** full OTR suite (`pytest tests -q`) + Bug Bible, 0 failures, on the combined tree before any commit.
- **3B-3E change script-pipeline behaviour.** After the wave lands, ONE live ComfyUI episode run validates the batch -- the audio-is-king reversion gate (Prime Directive 1). This is operator-gated: Jeffrey starts the run and pastes the console output; the AI has no real-time ComfyUI log access. Plan the wave so a single live run covers all four sprints.
- Any lane that changes a node `INPUT_TYPES` / widget / output socket must re-wire the workflow JSON in the same change set (Prime Directive 3). 3F + 3G did not need this (internal helpers only); 3C/3E are the likeliest to touch a node surface -- verify per lane.

### Sprints 4 / 5 / 6 (later waves)

- **Sprint 4 -- VRAM hardening.** The behavioural HuMo VRAM gate already exists (`_otr_humo_tier_loader.py`); Sprint 4 is verify-and-extend, not build. Single lane, operator-gated (must be confirmed against live RTX 5080 hardware).
- **Sprint 5 -- continuity ledger + story critic + targeted reroll.** New module `nodes/_otr_continuity.py` + a whole-script critic + a targeted-reroll loop. **Depends on Sprint 3A** -- the reroll hooks into `compose_line_draft`, which 3A creates. New LLM calls -> Prime Directive 6 each. This is the direct fix for the `ozempics_glitch`-class failure (Finding C). Not part of the 3B-3E wave.
- **Sprint 6 -- critic -> render coupling.** Ships with Sprint 5.
