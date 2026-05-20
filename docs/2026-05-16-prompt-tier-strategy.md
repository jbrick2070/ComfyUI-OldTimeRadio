# Round-robin brief -- prompt tier strategy

**Branch:** sprint-d-period-llm @ 5b0d0ba
**Author:** Jeffrey Brick
**Date:** 2026-05-16
**Reviewers:** ChatGPT (gpt-4.1), Gemini (gemini-2.5-pro)
**Format:** lean brief; goal is a converged recommendation, not a 1500-line audit

---

## The question, in one paragraph

OTR has 12 LLM call sites (writer + reflection + cascade + news_interpreter). Every prompt today was tuned on Mistral-Nemo 12B at 8192 context. The curated catalog now spans models from `talkie-1930-13b-it` (GPTQ int4, ~7.5 GB, 4096 context, period-trained) up through Qwen2.5-14B-Instruct (~28 GB unquant). Small models break on long prompts; big models waste budget on short ones. **Operator leading hypothesis: small LLMs actually prefer lean prompts, so normalize every prompt to the smallest-model-friendly size while keeping the story_brief variable scaffolding intact. One prompt body per call site, deliberately authored short.** This brief lays out that path (Option F), a "two workflow JSONs" fallback (Option E), and four heavier alternatives (A-D) for the round-robin to argue against.

---

## Current state (factual)

### The 12 call sites

| # | Phase            | Slot      | Period-routed at D2b | File / symbol                                  |
|---|------------------|-----------|----------------------|------------------------------------------------|
| 1 | style_inventor   | creative  | no                   | `_otr_style_picker._INVENTOR_SYSTEM`           |
| 2 | style_chooser    | technical | no                   | `_otr_style_picker._CHOOSER_SYSTEM`            |
| 3 | outline          | creative  | yes                  | `_otr_outline._SYSTEM_PROMPT`                  |
| 4 | cast_lock        | creative  | no (not in router)   | `_otr_casting.cast_one_character` attempt 1+2  |
| 5 | cast validator   | technical | no                   | `_otr_casting.cast_one_character` repair pass  |
| 6 | line_composer    | creative  | yes                  | `_otr_line_composer._SYSTEM_PROMPT`            |
| 7 | polish_character | creative  | yes                  | `_otr_line_composer._POLISH_..._CHARACTER`     |
| 8 | polish_announcer | creative  | yes                  | `_otr_line_composer._POLISH_..._ANNOUNCER`     |
| 9 | reflection       | technical | no                   | `_otr_story_brief._REFLECTION_PROMPT`          |
| 10| cast_audit       | technical | no                   | `_otr_ledger_reviewer._AUDITOR_SYSTEM_PROMPT`  |
| 11| script_doctor    | technical | no                   | `_otr_ledger_reviewer._DOCTOR_SYSTEM_PROMPT`   |
| 12| news_interpreter | technical | no (D3 unrouted)     | `news_interpreter.build_news_briefs`           |

**11 unique system-prompt constants + 3 repair variants = 14 prompt bodies total.**

### The curated catalog (sprint-d-period-llm `_otr_model_catalog.py`)

| repo_id                                | ctx  | approx GB | profile       | tier   |
|----------------------------------------|------|-----------|---------------|--------|
| `mistralai/Mistral-Nemo-Instruct-2407` | 8192 | ~22       | modern        | PASS   |
| `google/gemma-4-E2B-it`                | 8192 | (compact) | modern        | PASS   |
| `google/gemma-4-E4B-it`                | 8192 | (compact) | modern        | PASS   |
| `Qwen/Qwen2.5-14B-Instruct`            | 8192 | ~28       | modern        | WARN   |
| `Nitral-AI/Captain-Eris_Violet-12B`    | 8192 | ~24       | modern        | WARN   |
| `inflatebot/MN-12B-Mag-Mell-R1`        | 8192 | ~24       | modern        | WARN   |
| `talkie-lm/talkie-1930-13b-it`         | **4096** | ~7.5  | otr_1940s_v1  | UNKNOWN|

**The floor is talkie at 4096 ctx.** Every other curated model is 8192.

### What's already wired (D2b)

`_otr_creative_prompt_router.resolve_creative_system_prompt(repo_id, phase)` already swaps a modern prompt for `OTR_PERIOD_SYSTEM_PROMPT` when the catalog row says `prompt_profile = "otr_1940s_v1"`, at exactly 4 phases: `outline`, `line_composer_system`, `polish_character`, `polish_announcer`. **That is the only profile-aware prompt selection in the code today.** No phase routes on VRAM, model size, or context window.

---

## The six options

### Option F (NEW, leading candidate) -- normalize every prompt to the smallest-model-friendly size, keep variable scaffolding intact.
One prompt body per call site, deliberately authored short for the smallest model on the catalog. Variable interpolation slots (`{lighting_terms}`, `{cast_table}`, `{outline_spine}`, etc.) stay intact -- only the human-written instructional prose shrinks. Schema-strict outputs (GBNF, pydantic) unchanged because the small-model variant is what tunes the validator anyway.

**Why this is probably the right answer:**
- Empirical: small LLMs often outperform on shorter, more direct prompts. Long instructional prose hurts coherence past ~3 nested conditions and trades model attention away from the actual variables. This is a free win in many cases, not a quality compromise.
- Maintenance: 11 prompt bodies. ONE per call site. Same number we have today.
- Schema strictness: unchanged. story_brief 8-key, news_interpreter GBNF, cast_lock pydantic all keep their current contracts.
- C7 audio baseline: if rewritten carefully, byte-identity might survive on Mistral-Nemo with the leaner prompt; if not, the new prompt becomes the new C7 baseline (one-time reset, not a recurring cost).
- Onboarding: dropdown stays simple. Any model "just works" because the prompt was designed for the floor and the ceiling models tolerate it.

**Cost:** 1 sprint. Author lean variants, validate output schema on Mistral-Nemo (no regression), measure JSON pass rate on Gemma-4-E2B and talkie. Update story_brief reflection prompt to confirm variable interpolation still feeds correctly.
**Risk:** longest-prompt sites (line_composer, cast_lock) may lose creative steering room. Mitigated by keeping the high-leverage few-shot exemplar slots if the soak shows quality drop. The variables themselves stay -- it's the surrounding prose that shrinks.
**Quality ceiling:** very high if "small models like lean prompts" generalizes to OTR's specific use; medium if creative-slot calls need the verbose steering. Soak-testable per phase.

### Option E (NEW) -- two workflow JSONs, one per VRAM tier, with the right prompts baked in.
Ship `otr_scifi_16gb_full.json` (current) and `otr_scifi_8gb_compact.json` (new). Each workflow points at a different prompt module via a node default or env switch. No runtime router; the choice is "which workflow did you open?"

**Cost:** medium. Author the second prompt set, second workflow JSON, second test suite. But the dispatch is dead simple -- no router logic, no size-class detection, no surprises at runtime.
**Risk:** user-facing fork. "Which workflow do I open?" becomes a support question. Drift between the two JSONs becomes a forever maintenance debit. Two prompt files to keep aligned -- worse than Option F's one, better than Option D's two-per-site.
**Quality ceiling:** good. Each workflow tuned independently.
**Sprint estimate:** 1.5 sprints. Less than Option D because the test matrix is per-workflow not per-(phase, size_class).

### Option A -- one prompt body per call site, period.
Status quo modulo current D2b 4-phase routing. Tune for the median (Mistral-Nemo 12B) and accept that talkie clips at 4096 and tiny models lose coherence on the longest prompts (line_composer + cast_lock).

**Cost:** zero engineering.
**Risk:** quality cliff at the small-model end. Talkie at 4096 ctx may not survive line_composer's full system prompt + voice card + outline spine + last-N context window. Tiny models may fail validator-grade JSON at cast_lock.
**Quality ceiling:** good on Mistral-Nemo, unknown on others.

### Option B -- VRAM-tier bucket per call site (4 / 6 / 8 / 12 GB rows).
Author 3-4 versions of each of the 11 system prompts. Router picks bucket from catalog row's `approx_safetensors_gb`. Variable inventory must stay aligned across buckets so the user prompt template is shared.

**Cost:** big. 11 prompts * 4 tiers = 44 prompt bodies to author + maintain. Schema-strict outputs (GBNF, pydantic) must still satisfy the SAME validator across all tiers, which means the small-tier prompts have to coax the same shape out of a less capable model. Every workflow JSON wiring change has to be re-tested per tier.
**Risk:** prompt drift. Tier-3 fix gets forgotten on tier-1 next sprint. Three months in you have 44 prompts that disagree.
**Quality ceiling:** highest if you commit to tier maintenance discipline. Lowest in practice if you don't.
**Sprint estimate:** 2-3 sprints (author + validate + soak + wire router + tests).

### Option C -- intelligent runtime selection from a single rich prompt.
One system prompt per call site, written rich. At call time, the router measures `(prompt_tokens, model.context_window, max_new_tokens, model.size_class)` and elides optional sections (few-shot exemplars, repeated rules, verbose framing) to fit. Output schema unchanged.

**Cost:** medium. One new helper per call site (`compose_system_prompt(model, phase) -> str`) that owns the elision logic. Tests per (phase x size_class) cell. No prompt fork.
**Risk:** elision logic IS the prompt -- bugs hide in what got cut. The "compress" path needs its own validator: did the small-model variant still pass the cast schema? did the talkie variant still output the GBNF shape?
**Quality ceiling:** matches Option B at the median, slightly lower at the extremes (you give up some prompt-specific tuning).
**Sprint estimate:** 1.5-2 sprints. Builds on D2b router pattern.

### Option D -- dual-prompt switch tied to model size class.
Two prompt bodies per call site: "big" (>=10B unquant) and "small" (<10B or GPTQ). Router picks one based on the catalog row's `approx_safetensors_gb`. No mid-prompt elision; clean fork.

**Cost:** small-medium. 11 prompts * 2 = 22 prompt bodies. Half the maintenance of Option B, three-fourths of the quality ceiling. Test matrix is half. Workflow JSON is unchanged (router invisible to the JSON).
**Risk:** still-real prompt drift across the two bodies. Choosing the cut at 10B is arbitrary -- talkie at 7.5 GB GPTQ acts like a 7B in some ways and a 13B in others.
**Quality ceiling:** good on both ends, with a thin band of ambiguity in the middle.
**Sprint estimate:** 1 sprint (author + wire + soak).

---

## Decision criteria (rank them for the consultants)

1. **Quality at the small-model end.** Talkie is 4096 ctx. Several prompts (line_composer especially) are already close to that. Do we accept clipping, or is fitting talkie a real product requirement?
2. **Maintenance load.** Jeffrey is a one-person creative + engineering team with chronic pain limiting weekly hours. Every additional prompt variant is a permanent maintenance debit on every future sprint.
3. **Schema strictness.** 5 of the 12 calls use GBNF or pydantic-validated output. Tier-shrinking those prompts cannot break the schema -- the validator can't be relaxed per tier.
4. **C7 audio baseline.** Default config (both slots on Mistral-Nemo, prompt_profile = "modern" everywhere) MUST remain byte-identical. Any router refactor has to preserve object-identity stable returns under default config -- the existing D2b router does this; new options must too.
5. **Onboarding.** New users picking a model from the dropdown should get good output without reading docs. Whatever the option, the small-model experience must "just work" or be clearly gated.

---

## Two cross-cutting questions the consultants should answer

### Q1: Is the talkie 4096 ctx floor a real constraint or a research-lane curiosity?

Talkie is `license_audit_status = research_lane`, NOT eligible for default workflow JSON until license flips to `mit_equivalent`. If talkie never ships in default config, designing for 4096 ctx as the floor is over-engineering. **If talkie is research-only and the production floor is Gemma-4-E2B at 8192 ctx, Option A may actually hold up.** Consultants: argue for or against.

### Q2: Is the prompt-size-vs-model-size correlation strong enough to justify the engineering?

Small models DO lose coherence past ~3 nested conditions and they DO break on >400-token system prompts. But "small" and "big" in OTR's catalog are ~7.5 GB vs ~22 GB -- not 1B vs 70B. A 12B at 4-bit quant and a 12B unquant should follow nearly the same prompt. Consultants: at the OTR catalog's actual size spread, is tiered prompting worth the maintenance debit, or is the real lever **model selection** (don't expose talkie unless the user opts in)?

---

## Sprint shape if the answer is Option F (lean normalize)

For Jeffrey's planning, not for the consultants:

```
Sprint H (lean prompt normalize)
  H1 -- audit each of 11 prompt bodies for cuttable prose; keep
        every {variable} slot; keep schema-anchoring lines verbatim
  H2 -- rewrite line_composer + cast_lock + outline (the long three)
        lean; preserve story_brief variable interpolation
  H3 -- rewrite remaining 8 prompts lean
  H4 -- Bug Bible regression after each rewrite (no batching)
  H5 -- C7 byte-identity check on Mistral-Nemo
        - if it survives: ship, no baseline reset
        - if it breaks: stamp the new prompt as new C7 baseline,
          document in BUG_LOG, single-pass reset
  H6 -- soak: Gemma-4-E2B + talkie (gated by license) for quality
        + JSON pass rate
  H7 -- workflow JSON: no changes; same call sites, same variables
```

Estimated 2-3 sessions of 2-3 hour blocks. Much smaller than Option D.

## Sprint shape if the answer is Option E (two workflows)

```
Sprint H' (dual workflow JSON)
  H1 -- new prompt module for compact tier (11 prompts, lean)
  H2 -- second workflow JSON otr_scifi_8gb_compact.json
        pointing at the compact prompt module
  H3 -- wire prompt-module selection via node default or env switch
  H4 -- per-workflow soak + Bug Bible regression
  H5 -- README + dropdown labels say which workflow for which rig
```

Estimated 3-4 sessions. More work than F, simpler dispatch than C/D.

---

## What I want from the round-robin

1. Pick a winning option (A, B, C, D, E, F) and defend it. Operator hypothesis: **Option F is leading.** Argue with or against.
2. Validate or kill the talkie-as-research-lane reframe.
3. For Option F specifically: is "small LLMs prefer lean prompts" actually true at OTR's catalog spread (12B unquant down to 13B GPTQ int4), or only at the 1B-vs-70B extreme? Cite anything specific you know.
4. Flag any 7th option I haven't considered.
5. If you disagree on the winning option, name the specific datum you'd want to break the tie (e.g., "rewrite line_composer system prompt lean and measure C7 byte-identity on Mistral-Nemo -- if it survives, Option F; if it breaks, Option E").

Halt with a single recommendation + the riskiest assumption you're making to justify it.

---

## Constraints

- ASCII only in any inline code
- No em-dashes (`--` only)
- No "dummy" word
- Keep it lean -- 250 lines or under in final synthesis
- The OTR audio C7 baseline is non-negotiable: default config bit-identical
