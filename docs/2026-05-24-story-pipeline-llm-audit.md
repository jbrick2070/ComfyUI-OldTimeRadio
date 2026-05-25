# Story-Writing + Cleanup Pipeline -- LLM-Call Audit

- **Date:** 2026-05-24 (updated 2026-05-25)
- **Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha` (HEAD `c99fdfb`)
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

---

## Findings

### A. Tagging is complete and CI-enforced

Every call site carries the `# LLM slot:` tag -- the rule itself is met. **As of commit `c99fdfb`, this is enforced by a CI AST sweep** (`docs/_s28_llm_slot_sweep.py`) with a 4-test regression suite (`tests/test_llm_slot_sweep.py`). The sweep AST-parses every `*.py` under `nodes/`, finds all `structured_call` / `generate_fn` / `creative_fn` / `technical_fn` / `polish_generate_fn` / `request_slot` invocations, and verifies a `# LLM slot:` tag exists within ±8 lines. Internal plumbing files are exempt. Any new call site without a tag will fail the sweep test.

`slot_calls_by_helper` telemetry attribution is still incomplete -- three call sites bypass `helper_context()`: the outline, title regen, and story-brief reflection. Their calls fall to the `<unattributed>` bucket. Low severity -- forensic hygiene, not a runtime defect.

### B. Slot assignment -- mostly correct, two latent issues

Assignments are sound against the creative/technical definition, with caveats:

- **Outline Stage 2 (per-phase speaker routing) runs on the creative slot but is structured routing** -- it picks a name from a fixed locked cast. The code even acknowledges this ("structured routing, not creative prose") and gave it a falling temperature schedule, but left it wired to the creative generate_fn. Harmless in the default config (creative and technical are the same model) but a latent mis-slot if an operator ever picks a cheaper technical model.
- **`compose_line` accepts a `technical_fn` parameter but hard-overrides to `creative_fn`** -- the originally-planned technical-side critic was dropped. The parameter is now dead weight on the signature.
- **Stale comment:** the writer's pick_style call-site comment says it "routes both passes to creative"; the actual code routes Pass 2 (chooser) through the technical slot. The comment misleads anyone auditing from the writer alone.

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
4. **Wrap the outline, title regen, and story-brief reflection in `helper_context`** so `slot_calls_by_helper` telemetry is accurate. Forensic hygiene; small. Still open.
5. **Clean the two cosmetic issues:** fix the stale pick_style slot comment; drop the dead `technical_fn` parameter from `compose_line`'s signature. Still open.

Recommendation 1 is an architecture / new-LLM-call decision and is round-robin gated per CLAUDE.md. Recommendations 2 and 3 are resolved. Recommendations 4 and 5 are small and not gated.
