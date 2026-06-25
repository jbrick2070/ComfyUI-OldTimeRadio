# R3 JUDGMENT -- wiring / sequencing (Claude, grounded)

Panel: GPT-5.5, Gemini-3.1-pro (verbose, 10.3k tok), DeepSeek-v4-pro. Spend
~$0.1828. Convergence: HIGH on the call-site handoffs + one real sequence bug.

## ACCEPTED (CONFIRMED vs code) -> folded into pass03
- **In-place-mutation sequence bug (Gemini#1, GPT#4, DeepSeek#5):** build_sq_data
  mutates beat.intent (l12:803) and KILL-4 now enriches SETUP -> capture
  `opening_status_quo` from the ORIGINAL setup-beat intent right after
  generate_outline, BEFORE build_sq_data. The sharpest catch of the campaign.
  pass03 STEP D.
- **OutlineRequest wiring (DeepSeek#1/#2):** the contract output must actually be
  injected into the `OutlineRequest(...)` call (:3032), not just defined. STEP C.
- **Phase/beat threading -- decide now (GPT#3/DeepSeek#3/Gemini#4):** `macro` is
  the LLM output; request fields don't pass through -> add explicit params to
  `_build_phase/beat_user_prompt`. STEP C (story_engine only at phase/beat;
  sound_world stays at macro to avoid the dialogue-adjacent leak).
- **select_style: replace source, don't delete (Gemini#5/GPT#2):** set
  `_style_slug = contract.slug` under flag; keep the late call OFF. STEP G.
- **Outro wiring (all 3):** climax-line lookup by beat_id (LAST matching line,
  Gemini SHOULD#1); `climax_character_line or final_character_line` (GPT#8);
  suppress the resolved-fiction branch under an explicit gate (DeepSeek#8) +
  inject ending_change as forbidden (DeepSeek#9); early-out + empty-brief
  fallbacks route to `fallback_news_coda_outro`, never `_resolved_outro_fallback`
  (Gemini#3 leak); validate the RAW body before prefixing the lead-in
  (GPT#7/Gemini#2). STEP F.
- **Flag hoist (GPT#1):** compute `_style_grammar_on` at run() top so it gates the
  pre-outline contract; pass `story_scaffold=_style_grammar_on` to composers
  (GPT SHOULD#1). STEP A. **Contract built every pass, outside the refine-skip
  guard (GPT#10).** STEP B.

## JUDGE-LEVEL SCOPE CUTS (panel-justified; remove the riskiest wiring)
- **CUT the per-line register tag from the first build.** It was undefined
  (DeepSeek#4 said ending_tag, Gemini#2 said label) AND needed the reroll-rebuild
  thread (R3 reroll-loss). The contract reaches the body via outline injection +
  conflict_object already. Dissolves DeepSeek#6 + Gemini#2. Add later (via meta +
  build_reroll_line_request) only if re-soak shows flat dialogue.
- **DEFER the spoiler belt.** Import-cycle risk (GPT SHOULD#3/DeepSeek#1) +
  ending_change-availability at open time (GPT#5/Gemini#3). Input starvation is the
  guarantee; the belt is explicitly deferrable (GPT CUT#1, DeepSeek CUT#1). If
  added, extract tokens WRITER-side, pass a frozenset (no l12 import in composer).

## CONSUME render_style_grammar (kills the literal KILL-2 "zero callers")
Carry ONE `OutlineRequest.style_grammar = render_style_grammar(slug)` rendered in
the macro prompt -- this gives `render_style_grammar` a real caller (the KILL-2
evidence was "zero callers") and bundles story_engine+sound_world+ending_mode at
the structural level, one field not two. (GPT SHOULD#5/CUT#3 -- don't compute dead
grammar; here it is consumed.)

## VERIFIED THIS PASS
- `_style_grammar_on` is the single flag (`style_grammar_enabled()` reader);
  `_apply_story_scaffold_env` :1551->:2402 is the one plumb.
- `_refine_loop` shares one cast_seed (:2190); contract is deterministic per pass.
- build_sq_data runs :3245 (after generate_outline :3158, before line loop) ->
  capture-before-mutation is feasible.

## CONVERGENCE CALL
R3 closed the call-site handoffs + the sequence bug; the two cuts removed the
remaining high-risk wiring. The plan is now a linear, grounded build. Proceed to
R4 (residual defects / convergence confirmation), NOT exit early.
