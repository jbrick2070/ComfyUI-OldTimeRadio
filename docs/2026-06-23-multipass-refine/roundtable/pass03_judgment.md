# R3 judgment (wiring / integration) -- the decisive round

Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro ALL substantive (token bump fixed the cutoff).
Spend $0.3803. Cumulative $0.8818.

## ACCEPTED -- all three models independently CONFIRMED against my own code
1. **build_sq_data mutation defeats the metric (GPT MF2/MF5, DeepSeek MF1, Gemini MF1).** CONFIRMED:
   `ground_crisis_nouns` in `_otr_story_quality_l12.py` SUBSTITUTES the generic crisis nouns in
   `beat.intent`, so `ungrounded_crisis_density` measured AFTER `build_sq_data` is ~0. FIX: the scorer is
   PURE and computes on RAW intents BEFORE any grounding (call `count_ungrounded_crisis` directly);
   `build_sq_data` runs EXACTLY ONCE, on the winner (the existing F2 block) -- never during scoring.
2. **No `episode_seed` exists (GPT MF1/MF8, DeepSeek MF4, Gemini MF2).** CONFIRMED: the `seed` widget was
   removed (BUG-LOCAL-269/270). FIX: derive the per-candidate seed from `cast_seed` (in scope):
   `sha256(f"{cast_seed}:outline:{n}")`.
3. **The RNG hash was never wired into the LLM (Gemini MF3, GPT MF1).** CONFIRMED: `generate_outline` takes
   no seed. FIX: `torch.manual_seed(int(h,16)%2**64)` + `random.seed(...)` immediately BEFORE each
   `generate_outline` call inside the N-loop.
4. **No diversity hook on OutlineRequest (DeepSeek MF3, GPT MF1).** CONFIRMED frozen dataclass. FIX: add
   optional `diversity_hint: str = ""` to `OutlineRequest`, thread it into `_build_user_prompt` (empty =>
   byte-identical), set per candidate.
5. **Local gate must include Comfy Credits (all three).** CONFIRMED two paid lanes. FIX:
   `if resolved["creative_writing_model"].startswith(("openrouter:","comfy:")): N=1`.
6. **build_sq_data double-run / non-idempotent _enrich_intent (GPT MF2/3, DeepSeek MF2, Gemini MF1).**
   CONFIRMED `_enrich_intent` can append twice. FIX: build_sq_data once on the winner only.
7. **Telemetry merge, not replace (GPT MF9).** ACCEPTED: `meta.setdefault("story_quality",{})["best_of_n"]=...`.
8. **Flag parsing undefined (GPT MF7, DeepSeek MF6).** FIX: `OTR_STORY_BEST_OF_N` int; unset/0/1 => disabled,
   >=2 => N clamped to a max (6).
9. **Wrap each candidate in try/except OutlineFailedError (Gemini MF5).** CONFIRMED can raise. FIX: continue
   on a failed candidate; if ALL fail, fall through to one normal call (LOUD) -- never-fail.

## CUT (panel unanimous)
- **v1 post-compose "B+" loop CUT from the initial build** (GPT CUT2, DeepSeek CUT, my R2 split) -- separate
  integration project (recompose/regrade/freeze orchestration); validate outline-level best-of-N first.
- **`character_want_clarity` CUT from v0** (GPT CUT1) -- no dramatic_state/wants at the D.5 insertion point.
- **Do NOT re-validate candidates (Gemini CUT, GPT SF1)** -- `generate_outline` already calls
  `validate_outline_against_budget` + `stamp_dialogue_slot_ids` internally; just assert, don't re-run.

## CONVERGENCE
Unanimous, code-grounded convergence on a materially revised, now build-ready v0. Carrying to R4 only to
confirm no NEW must-fix. The design is in pass04_plan_FINAL.md.
