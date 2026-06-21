# Roundtable pass 01 -- judgment log (Claude as judge)

Panel: `openai/gpt-5.5-20260423`, `google/gemini-3.1-pro-preview-20260219`, `x-ai/grok-4.3-20260430`, `deepseek/deepseek-v4-pro-20260423`. Spend $0.17. Plus Claude's own grounded code review (read the real source, not just the doc).

The panel ran on the skill's generic "is this plan build-ready" review prompt, so it critiqued the problem statement AS A DOCUMENT and proposed engine fixes along the way. Both are useful. Every claim below was graded against the actual source.

## Grading of the load-bearing claims

| # | Claim (who) | Grade | Grounding |
|---|---|---|---|
| A | Length undershoot is NOT a mechanical (20,35)x14 cap; Appendix A is 700 words, proving the band isn't a hard output cap (GPT, Grok) | **CONFIRMED -- corrects our doc** | `compute_episode_budget` widens per-beat ceiling: `eff_hi=min(80,max(base_hi,required_hi))`; at 864 -> ~64, so 14x64=896>=864 (`_otr_episode_budget.py:286-296`). The doc's "~280-490 no matter what" is false at 864. |
| B | The real undershoot driver is the unconditional "about 20-30 words" line tail (Gemini, Grok) | **CONFIRMED** | `_otr_line_composer.py:1287-1292` emits "...one breath, about 20-30 words..." on EVERY voiced beat regardless of target_words; plus 2-attempt ladder + `max_new_tokens=min(200,...)`. |
| C | `episode_valid=False 76%` is because the Stage 5 JSON omits `costly_choice_beat` -> add it to the prompt schema (Gemini) | **MISREAD -- rejected** | `costly_choice_beat` is set deterministically by `pick_costly_choice_slot` and re-stamped (`_otr_dramatic_state.py:184-198`, `_otr_dramatic_state_llm.py:300`). The LLM never emits it. Adding it to the prompt would fight the deterministic stamp. |
| D | The TRUE costly-choice cause: slot-id list desync (Claude code review) | **CONFIRMED** | Costly slot picked from ledger-lines list incl. announcer slots (`OTR_LedgerScriptWriter.py:2785-2790`) but `must_turn` checked against outline-beats list (`:2934-2936`); `ids[-2]/[-1]` can land on the announcer outro slot. Fix: pick+check from character-only slots. |
| E | dramatic_state (Stage 5) runs AFTER outline (Stage 4) and never reads it, so it can't shape beats -> swap order (Gemini) | **CONFIRMED (structural) -- accepted as Tier 3** | Pipeline order matches; costly slot is assigned post-hoc from finished lines. Reorder helps variety+placement but is larger than the desync fix (D). Compounds with arc-variety, not a prerequisite. |
| F | Outro is not conditioned on the resolved ending -> hedge contradicts success (Grok + all) | **CONFIRMED** | `compose_announcer_outro` docstring: context is "script_brief + news_close_brief + intro_text only -- never the full script" (`_otr_line_composer.py:2369-2371`). |
| G | lock_cast never puts gender/pronouns in the description; Stage 7 sees only the description -> gender hallucination (Gemini) | **CONFIRMED** | Cast contract is portrait-only (`_otr_casting.py:373-414`); composer threads `character_description`, not raw `{gender}`. Explains the SOM CORBEN "Mister"/"a man" bug. |
| H | Anti-decorative rider is conditional -> lands on ~1 line (Grok, Claude) | **CONFIRMED** | `_otr_line_composer.py:1264-1274` gated on dramatic fields; `must_turn` true on at most one slot. (Grok's link of THIS to the 76% audit is imprecise -- that's the desync D; keep them separate.) |
| I | Self-naming + stage-direction-as-dialogue partly handled already (Claude) | **CONFIRMED** | `_otr_line_hygiene.py` scrubs self-vocative + parentheticals (`:69-103`, wired `_otr_story_spine.py:505-516`) but misses 3rd-person narration ("He paces...", "Donna questions...") and differently-spelled self-address ("Ayisha"). |
| J | The story-QA critic would rubber-stamp the monotony (implied by doc) | **PARTIAL -- clarified** | `_otr_creative_qa.py` is a per-beat defect ROUTER (PASS/MICRO_REPAIR/REJECT), gated OFF (`OTR_ENABLE_STORY_QA` default 0). The "arc_verdict: strong 44/51" comes from a DIFFERENT module (`_otr_story_critic.py`). Neither sees cross-episode monotony. Reinforces C3: a single-episode QA can't catch sameness. |
| K | `ledger_scrub_status FAIL`, `qa SKIPPED`, reviewer 404 are infra telemetry, not story defects -> tell panel to ignore (GPT, DeepSeek) | **CONFIRMED -- doc should label these** | All three are post-script/infra, gated-off, or a dead model endpoint. Not story-quality blockers. |
| L | Premise dedup needs persistent history, which C4's "no DB" seems to forbid (GPT) -> a small local forbid-list is allowed (Gemini, Grok) | **ACCEPTED (scoped)** | A tiny local JSON "recently-used" list is a small logic constant, not a vector DB/RAG. The deeper lever (same RSS item seeding 4 eps) is upstream of the shown pipeline -> verify-at-build. |

## Doc errors the panel caught (to patch in the problem statement)
- Section 6 says "three separate one-shot announcer calls (intro, outro)" -- there are only **two**. (GPT, Gemini, DeepSeek) -- FIX.
- Section 3 "the undershoot is mechanical / caps ~280-490 no matter what target_words" -- false at 864 (claim A) -- FIX.
- "ZERO errors" vs "scrub FAIL 51/51 / length-pass ERROR 33/51" reads as a contradiction; clarify "zero generation crashes before frozen ledger" vs post-pass/infra failures. (GPT)
- Label the infra telemetry (K) as "do not solve here." (GPT, DeepSeek)

## Accepted into the plan (see STORY_ENGINE_IMPROVEMENT_PLAN.md)
Tier 1: F1 length tail (A,B), F2 costly-choice desync (C-rejected, D-accepted), F3 ending-aware outro + DEFER act-bridges (F + unanimous cut).
Tier 2: F4 gender in cast (G), F5 speech-register (Grok), F6 unconditional rider (H), F7 narration hygiene (I).
Tier 3: F8 arc-shape variety (Grok/DeepSeek), F9 condition outline on dramatic_state (E), F10 premise forbid-list (L).

## Rejected
- Gemini "add costly_choice_beat to Stage 5 JSON schema" (C: already deterministic).
- The unified intro+bridge+outro announcer pass for v1 -- all 4 models + judge: over-engineering; changes beat count/seams/timing/tests/workflow JSON; doesn't fix undershoot. Revisit as a separate scoped experiment after Tier 1.
- Grok "announcer-over-music" bridges -- render/timing ambiguity, out of the stated scope.

## Convergence verdict
**Converged for a v1 plan.** Four independent models named the same top fixes (length tail, ending-aware outro, costly-choice binding, arc variety, voice differentiation), and the judge's code grounding corrected two doc claims (A) and one panel misread (C). A pass 2 would mostly re-litigate; recommend stopping unless the user wants the rewritten IMPROVEMENT_PLAN itself hardened. Open verify-at-build items: F9 reorder, F10 RSS-source dedup (code not in scope).
