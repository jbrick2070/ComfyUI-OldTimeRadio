# Signal Lost -- Story-Engine Improvement Plan (roundtable pass 01, grounded)

**Source:** the 4-model roundtable (`GPT-5.5`, `Gemini-3.1-pro`, `Grok-4.3`, `DeepSeek-v4-pro`) on `STORY_ENGINE_PROBLEM_STATEMENT.md`, plus Claude's own code-grounded review of the real source. Full critiques in `roundtable/pass01/`; accept/reject reasoning in `roundtable/pass01_judgment.md`.
**Constraints honored:** news seed is a permanent staple (C1); ledger schema unchanged, additive `meta` only (C2); NO new QA/scoring/reject pass (C3); no architecture change (C4); one box (C5).
**Line numbers** are from the read at commit `f99af26` and are approximate -- confirm before editing. Per the build rules, any beat-count/wiring change goes into `workflows\otr_scifi_16gb_full.json` in the SAME commit, and the regression suite + Bug Bible run after each change.

This is a coder-ready punch list, ranked by leverage. Each item: tag, exact location, the change, why, confidence, and landmines.

---

## TIER 1 -- ship first (high confidence, minimal, grounded)

### F1. `[PROMPT]` Make the per-line length instruction track `target_words`
**Where:** `nodes\_otr_line_composer.py:1287-1292` -- the tail emitted on EVERY voiced beat:
```
"Ground this line in the news facts and this scene's premise; do not invent people, places, or objects the news does not imply. Keep it spoken-length -- one breath, about 20-30 words, concrete, no nested clauses."
```
**Change:** derive the word figure from the beat's own band instead of hardcoding it. Before -> after:
- before: `...one breath, about 20-30 words, concrete...`
- after:  `...one breath, about {beat_lo}-{beat_hi} words, concrete...` (interpolate the per-beat band the budget already computes), OR drop the number entirely: `...one breath, concrete, no nested clauses.`
**Why (corrected from the problem statement):** the budget is NOT the cap. `compute_episode_budget` widens the per-beat ceiling to ~64 words at target 864 (`_otr_episode_budget.py:286-296`), so 14x64 >= 864 is reachable; Appendix A is already 700 words. The ONLY thing pinning lines near 28 words is this universal "about 20-30 words" tail. This is the single highest-leverage undershoot fix.
**Also adjust (same change):** the per-line token cap `max_new_tokens = min(200, max(40, target_words*4))` and `_MAX_NEW_TOKENS_PER_LINE=200` (`:75`), and consider raising the 2-attempt ladder -- otherwise longer targets get truncated.
**Confidence:** HIGH -- all 4 models + code review.
**Landmine:** test `_build_user_prompt` Test 3 asserts the literal target string ("15") and block presence; update the test. Don't touch the frozen legacy prompt path.

### F2. `[LOGIC]` Bind the costly-choice beat to a real character line (fixes `episode_valid=False` on 76%)
**Where:** `_otr_dramatic_state.py:184-198` (`pick_costly_choice_slot`) + `OTR_LedgerScriptWriter.py:2785-2790` (builds `voice_slot_ids`) vs the contract loop at `:2934-2936`.
**Root cause (grounded -- this corrects the problem statement and rejects one panel fix):** `costly_choice_beat` is NOT missing and is NOT an LLM field -- it is set deterministically and re-stamped. The audit fails because the costly slot is **picked from the ledger-lines slot list (which includes the two announcer slots)** but `must_turn` is **checked against the outline-beats slot list**; `pick_costly_choice_slot` returning `ids[-2]`/`ids[-1]` can land on the **announcer outro slot**, which never gets a `must_turn` character contract -> `validate_episode_contracts` reports "no slot carries the costly-choice turn" (`_otr_slot_drama_contract.py:716`).
**Change:** compute `voice_slot_ids` from **character-role voiced beats only** (exclude announcer/music slots), and have `pick_costly_choice_slot` choose among character slots, so the picked slot always has a contract. Keep the `^d\d{3}$` id shape and the `"d001"` empty-fallback.
**Why:** directly fixes the most-failing audit in the corpus; it is a binding bug, not a missing plan.
**Confidence:** HIGH (code-grounded). **REJECT** the panel suggestion to "add `costly_choice_beat` to the Stage 5 JSON schema" -- it is already deterministic and the LLM emitting it would conflict.
**Landmine:** the contract loop `continue`s on a slot with no speaker (`:2950-2953`) -- ensure the picked slot is one that will get a contract.

### F3. `[PROMPT]` Make the outro ending-aware -- and DEFER the act-bridge pass
**Where:** `compose_announcer_outro`, `nodes\_otr_line_composer.py:2360-2436` (user prompt assembled `:2417-2425`).
**Change:** thread `meta.dramatic_state.ending_change` and the final character line into the outro user prompt, and add to the outro system prompt: "If the story resolved its dramatic question, state the outcome plainly; do NOT hedge with 'remains to be seen / open question' when the script already answered it." Before the existing "Closing brief..." block, add:
```
How the story actually ended:
{ending_change}
The last thing a character said:
{final_character_line}
```
**Why:** the outro currently sees only `script_brief + news_close_brief + intro_text`, never the resolved ending (docstring confirms, `:2369-2371`) -- which is why it hedges over episodes that visibly succeed (`four_solid_green_lights`, `names_on_the_board`). Additive `meta` read only; ledger-safe.
**Confidence:** HIGH -- all 4 models + code review.

**ACT-BRIDGE RULING (answers Section 6 of the problem statement):** the panel was **unanimous: do NOT build the unified intro+bridge+outro announcer pass for v1.** It changes the beat count, act seams, `start_s` timing, the announcer-count validator, the byte-identity tests, and the workflow JSON -- and it does not fix the core undershoot. "Announcer-over-music" was also rejected (render/timing ambiguity, outside scope). The endorsed minimal move is F3 (ending-aware outro). **Revisit act bridges later as a separate, explicitly-scoped experiment** -- and if approved, as plain `speaker_role="announcer"` lines mapped to pre-allocated slot ids (the renderer orders `lines[]` by slot/`start_s`, so appended-at-end bridges would otherwise play last), never as mixed music behavior. So: your instinct is good, but sequence it after length + binding + ending-aware outro land.

---

## TIER 2 -- strong, slightly more involved

### F4. `[PROMPT]` Put gender + pronouns in the cast description (fixes the "Mister Corben"/"a man" bug)
**Where:** the CHARACTER VISUAL CONTRACT block, `nodes\_otr_casting.py:373-414`.
**Change:** add a rule: "State the character's gender and use consistent pronouns in the description so downstream stages address them correctly." (Or pass the raw `{gender}` into the compose_line CHARACTER block.) **Why:** Stage 7 sees only `character_description`, not the raw `{gender}` field, so it guesses -- that is the SOM CORBEN clash. **Confidence:** HIGH (grounded).

### F5. `[PROMPT]` Add a speech-register cue to casting for voice differentiation
**Where:** same casting contract block; surfaces to the composer via `all_voice_cards` (`_otr_line_composer.py:1060-1064`) with no schema change.
**Change:** append one line to the contract: "Also emit a short speech signature (cadence, vocabulary, a verbal habit) distinct from the rest of the cast; the character uses it throughout." **Why:** the contract is portrait-only today, so every skeptic talks alike. **Confidence:** MED-HIGH.

### F6. `[PROMPT]` Make the anti-decorative rider unconditional
**Where:** `nodes\_otr_line_composer.py:1264-1274` -- currently gated on `dramatic_question/objective/turn/next_turn`.
**Change:** emit "Do not summarize the objective. Do not explain the turn. Perform the objective indirectly. The situation must be different after this line." on **every** voiced beat (or also trigger on `arc_phase`). **Why:** today it lands on at most one slot, so most lines are free to over-explain; this attacks the verification-ritual / over-tidy feel. **Confidence:** MED-HIGH. (Note: this is a craft fix; it is NOT the cause of the 76% audit failure -- that is F2.)

### F7. `[LOGIC]` Add a 3rd-person-narration-as-dialogue detector to line hygiene (+ "never speak your own name")
**Where:** `nodes\_otr_line_hygiene.py` (new deterministic check) wired into the existing spine hygiene loop (`_otr_story_spine.py:505-516`); plus a negative line in the compose_line OUTPUT FORMAT (`:980-1000`): "Never speak your own name."
**Change:** flag a line that opens with `He/She/They/<speaker-name> + narration-verb` (reuse the verb set in `_NARRATION_LEAK_REGEXES`, `_otr_line_composer.py:1310-1330`) and route it to the existing `_hy_recompose` seam. **Why:** the current scrub catches self-vocative + parentheticals but misses "Donna questions the risks..." / "He paces..." and differently-spelled self-address ("Ayisha"). **Confidence:** HIGH. **C3 note:** this is a deterministic scrub/recompose, NOT a scoring/reject pass -- explicitly allowed.

---

## TIER 3 -- variety + structure (medium confidence, more design)

### F8. `[PROMPT/LOGIC]` Add arc-shape variety (the highest-value, least-mechanical fix)
**Where:** `_otr_dramatic_state_llm.py:_TEMPLATES` (`:145-166`, only 4 templates, all "opposed wants / one test") and the macro prompt `_MACRO_SYSTEM_PROMPT`.
**Change:** add an `arc_shape` choice in the macro stage (e.g. `setup_complication_resolution | investigation_without_answer | slow_dread | heist | betrayal`) and pass it into dramatic_state; add structurally-distinct templates for each. **Why:** 6/9 spine episodes are the same advocate-vs-skeptic-then-sign arc; this is the root of the monotony. **Confidence:** MED -- needs care to be real structure, not a renamed label. **Landmine:** keep the macro schema `{title,premise,setting,time_of_day,central_tension}` and `_make_post_validator`'s key-term + opposed-wants requirements intact (C2).

### F9. `[ORDERING]` Condition the outline on the dramatic_state (verify-at-build)
**Where:** pipeline order -- dramatic_state (Stage 5) currently runs AFTER the outline (Stage 4) and never reads it.
**Change:** derive (or pre-derive) the dramatic_state before/alongside the outline and feed A-wants / B-wants / arc_shape into phase+beat planning, so the costly choice is placed structurally rather than retrofitted. **Why:** makes F8's variety and F2's binding land in the beats instead of after them. **Confidence:** MED / verify-at-build -- this is a larger reorder; F2 already fixes the binding cheaply, so treat F9 as a compounding structural improvement, not a prerequisite.

### F10. `[LOGIC]` Premise/style anti-repeat via a small local forbid-list
**Where:** the style chooser user template (`_otr_style_picker.py` Pass 2) and/or the news-seed selection upstream.
**Change:** maintain a tiny local JSON of recently-used style descriptors / news-item ids (e.g. last 30 days) and forbid reuse in the chooser prompt. **Why:** ~17/51 episodes are orbital-rescue variants and one RSS item seeded 4 near-duplicate episodes. **Confidence:** MED. **C4 note:** a small local list is a logic constant, not a vector DB/RAG -- in bounds. **Verify-at-build:** the deeper lever is the RSS/article selection (code not in the reviewed set); the forbid-list is a floor, not the full fix.

---

## Problem-statement corrections (apply to STORY_ENGINE_PROBLEM_STATEMENT.md)
1. Section 6: "three separate one-shot announcer calls (intro, outro)" -> **two** calls.
2. Section 3 / Appendix: the "undershoot is mechanical / caps ~280-490 no matter what `target_words`" claim is **false at 864** -- the budget widens the per-beat ceiling to ~64; the real driver is F1 (the unconditional line tail + token cap + 2-attempt ladder).
3. Reframe the costly-choice item as a **slot-id desync** (F2), not "planned then not placed."
4. Label `ledger_scrub_status FAIL`, `story_qa SKIPPED`, reviewer-404, and length-pass ERROR as **non-story infra telemetry -- do not solve here**; note "arc_verdict: strong" comes from `_otr_story_critic.py`, a different module from the gated-off `_otr_creative_qa.py`.

## Suggested first commit
F1 + F2 + F3 together: they are the three highest-leverage, lowest-risk fixes (length, the 76% audit, the contradicting outro), all single-file or near-single-file, all ledger-safe. Run the regression suite + Bug Bible, update the workflow JSON only if F-items touch beat wiring (F1-F3 do not), commit + push to `v2.0-alpha`.

## Convergence
Pass 01 converged: four independent models named the same top fixes and the code grounding corrected two doc claims and one panel misread. A pass 02 is optional (low marginal value); run it only to harden THIS plan if desired.
