# Story-Writing + Cleanup Pipeline -- LLM-Call Audit

- **Date:** 2026-05-24 (updated 2026-05-25)
- **Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha` (HEAD `bb53870`)
- **Build status (2026-05-25):** Sprints 0, 1, 2A-2E, 3F, 3G COMPLETE and pushed. Remaining: 3A, 3B, 3C, 3D, 3E, 4, 5, 6. See the "Multi-agent execution plan" section at the foot of this file for the parallel-subagent lane map for the remaining work.
- **Scope:** every LLM call from `OTR_LedgerScriptWriter` (LPL v2.0) through `OTR_LedgerFreezeCascade` -- the path that produces and cleans the episode script. Visual-side LLM calls are out of scope.
- **Method:** systematic call-site inventory across the writer/outline/casting/composer/reviewer/cascade modules, plus verification of the load-bearing findings (GBNF wiring, slot tags).
- **Why:** downstream audio + video quality is gated entirely by script quality. The 2026-05-24 `signal_lost_ozempics_glitch` run is the motivating evidence -- a structurally valid, cast-clean, budget-correct script that was dramatically empty (~12 words of character dialogue, one hallucinated line, `freeze_verdict=needs_full_rerun`), and nothing in the pipeline caught it.

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

## Multi-agent execution plan (remaining sprints)

As of HEAD `bb53870` (2026-05-25), Sprints 0, 1, 2A-2E, 3F, 3G are COMPLETE and pushed. The remaining audited work is Sprints 3A-3E, 4, 5, 6. This section is the lane map for running them as parallel subagents.

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

**Wave 1 (parallel):** Lanes B + C + D + E -- four disjoint files, four subagents at once.
**Lane A (3A):** stands alone -- dedicated effort, not in the parallel wave.

### Dependencies and gating

- Lanes A-E are independent decompositions of separate modules -- no lane depends on another, so B/C/D/E parallelize cleanly. (The plan's "in plan order" sequencing is superseded by the parallel-wave method.)
- **Every lane is regression-gated:** full OTR suite (`pytest tests -q`) + Bug Bible, 0 failures, on the combined tree before any commit.
- **3B-3E change script-pipeline behaviour.** After the wave lands, ONE live ComfyUI episode run validates the batch -- the audio-is-king reversion gate (Prime Directive 1). This is operator-gated: Jeffrey starts the run and pastes the console output; the AI has no real-time ComfyUI log access. Plan the wave so a single live run covers all four sprints.
- Any lane that changes a node `INPUT_TYPES` / widget / output socket must re-wire the workflow JSON in the same change set (Prime Directive 3). 3F + 3G did not need this (internal helpers only); 3C/3E are the likeliest to touch a node surface -- verify per lane.

### Sprints 4 / 5 / 6 (later waves)

- **Sprint 4 -- VRAM hardening.** The behavioural HuMo VRAM gate already exists (`_otr_humo_tier_loader.py`); Sprint 4 is verify-and-extend, not build. Single lane, operator-gated (must be confirmed against live RTX 5080 hardware).
- **Sprint 5 -- continuity ledger + story critic + targeted reroll.** New module `nodes/_otr_continuity.py` + a whole-script critic + a targeted-reroll loop. **Depends on Sprint 3A** -- the reroll hooks into `compose_line_draft`, which 3A creates. New LLM calls -> Prime Directive 6 each. This is the direct fix for the `ozempics_glitch`-class failure (Finding C). Not part of the 3B-3E wave.
- **Sprint 6 -- critic -> render coupling.** Ships with Sprint 5.
