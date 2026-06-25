<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan for the News Coda actively fights the existing system prompts, and the Open's fallback guarantees the exact spoilers the plan forbids.

MUST-FIX BEFORE BUILD:

1. [Section 2 - JOB 3 / NEWS CODA] **Outro System Prompt Contradicts the Coda.** The plan claims the coda is "largely WIRED" because `news_close_brief` is passed in. But `_ANNOUNCER_OUTRO_SYSTEM` (`_otr_line_composer.py:2549`) explicitly commands: `Do NOT state a moral, lesson, or news-summary`. Furthermore, `compose_announcer_outro` (`:2854`) explicitly injects: `The dramatic question RESOLVED... State this outcome plainly in the close.` 
   *Fix:* You must rewrite `_ANNOUNCER_OUTRO_SYSTEM` to explicitly demand the pivot to reality, and DELETE the `resolved and ending` branch in `compose_announcer_outro` that forces the fictional outcome into the announcer's mouth. 

2. [Section 2 - JOB 1 / THE OPEN] **Fallback Guarantees Spoilers.** The plan demands "no spoilers" for the open. But if the LLM fails, `fallback_announcer_intro` (`_otr_line_composer.py:2614`) blindly injects `script_brief` into the text (`"Tonight: {brief}"`). If the brief contains the outcome, the fallback hard-fails the no-spoiler constraint.
   *Fix:* Sever `script_brief` from the intro entirely. The fallback must be built from a new, strictly bounded `premise_setup` or `opening_situation` string that structurally cannot contain the ending.

3. [Section 4 / Ask #4 - KILL 3] **Outro Coupling to Climax Position.** You asked if there is hidden coupling preventing the climax from moving away from the last beat. Yes: `compose_announcer_outro` (`:2849`) takes `final_character_line` and uses it to bridge the outro. If the climax is at `n-2`, the `final_character_line` is just falling action/denouement. The announcer will bridge off the wrong dramatic beat.
   *Fix:* Change the parameter from `final_character_line` to `climax_character_line` and pass the text of the actual climax beat, regardless of its index.

SHOULD-FIX:

4. [Section 5 / Ask #2 - OPEN DETERMINISM] **You cannot post-gate a spoiler.** A regex or LLM judge cannot reliably detect if a twist was spoiled without high false positives. 
   *Fix:* Enforce determinism via *input starvation*. Do not pass `script_brief` to the intro composer. Pass only `era`, `setting`, `cast_names`, and a newly extracted `inciting_incident` from the outline. If the prompt doesn't have the ending, it can't spoil it.

5. [Section 5 / Ask #1 - NEWS CODA PHRASING] **Teachability requires a fixed anchor.** Small models (mistral/gemma) will blend the fictional outcome and the real news if asked to transition "naturally". 
   *Fix:* Use a strict, fixed lead-in tag injected *deterministically* before the LLM generation or forced as a prefix. E.g., `"The reality behind the signal: "` or `"In the real world: "`. Do not leave the pivot to the LLM's stylistic discretion.

6. [Section 3 - KILL 4] **Truncation Order.** The plan notes fixing the 200-char truncation order (`_otr_story_quality_l12.py:801`). 
   *Fix:* Explicitly slice the *original* intent if `len(original) + len(enrichment) > 200`, keeping the enrichment intact at the end. Do not just truncate the final concatenated string, which destroys the injected role constraints.

CUT THESE (scope / over-engineering):

1. [Section 1 / Ask #3 - KILL 2 INJECTION] **Cut `sound_world` and `story_engine` from the character `LineRequest`.** Injecting macro-level worldbuilding into every single micro-level dialogue prompt is the single-prior trap. It will cause the LLM to hallucinate sound cues (`[wind howling]`) or narrate the story engine instead of speaking character dialogue. 
   *Why it's safe to cut:* Style should shape the *outline* and the *beat intents*. The line composer's only job is to execute the immediate dramatic intent in the character's voice. Pass `ending_tag` and `register/tone`, but drop the heavy worldbuilding fields from the line-level prompts.

[ASSUMPTION] The plan assumes the `OutlineRequest` can cleanly separate the `opening_situation` from the `script_brief` without modifying the upstream news interpreter. You will likely need to add an explicit `opening_status_quo` field to the outline schema to feed the new Announcer Open.