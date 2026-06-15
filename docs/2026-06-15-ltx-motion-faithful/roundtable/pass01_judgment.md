# Pass01 Judgment (Claude, grounded vs render_driver.py)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4. Spend ~$0.10. Converged hard: pass00 was STALE.

## ACCEPTED (all grounded against render_driver.py)
- **Prompts already exist; pass00's "restore them" is wrong.** CONFIRMED `_LTX_MOTION_PROMPT_BY_ROLE` L472,
  `finish_visual_prompt`/`get_story_brief_ltx` L942-964. Pivoted the plan to verify+unsuppress.
- **Deliberate smear-suppression (Gemini #1, GPT #3).** CONFIRMED L505-511: aggressive open verbs SMEAR on
  2B LTX -> opening retargeted to calm `music_inter`; `OTR_LTX_OPEN_MOTION_KEY` default `music_inter`. This
  is the dominant cause, and it was the OPERATOR's own 6/12 decision.
- **`text_prompt` precedence (GPT #2).** CONFIRMED L886 `and not text_prompt` gates the motion block ->
  M4/ShotLock creative prompt bypasses the motion-role prompt. Real second mechanism.
- **Prompt budget is 240 not 188 (GPT #3, DeepSeek #3).** CONFIRMED `_LTX_MOTION_PROMPT_MAX = 240` L493.
  pass00 corrected.
- **My earlier smoke used a GENERIC calm prompt, not the real template (DeepSeek #5, GPT #5).** TRUE -> the
  "prompt is the lever" claim is unproven; added the controlled music_inter-vs-music_open smoke.
- **Current loop source is 97, not 81 (GPT #4).** CONFIRMED (my own `_ltx_loop_source_length` default 97).
  Length lever reworded; 81 is an env-test, not current behavior.
- **Negative-prompt lever (DeepSeek #4)** + **keep ksampler OUT of the prompt change for attribution
  (GPT-cut #3)** -- accepted as secondary/sequencing.

## CLAUDE'S ADDED INSIGHT (not from the panel)
The smear that justified the 6/12 suppression occurred at 1472x832 -- the SAME over-resolution mush fixed
TODAY (4fc4268 -> 832x480). So `music_open` may now render clean. This makes "test music_open at 832x480"
the highest-value first step (connects the two threads).

## REJECTED / DEFERRED
- **The LLM motion-prompt pass (operator-favored)** -- all 3 panelists CUT it as REDUNDANT (brief-grounded
  per-beat prompts already exist for non-open beats via finish_visual_prompt). Judge: DEFER, not kill --
  honor the grounding (it's mostly built), but leave the door open for a SEPARATE deterministic open-beat
  variety pass IF static music_open isn't enough. Be honest with the operator that their idea is ~80%
  already implemented.
- **Engine-local prompt fallback (GPT-cut #2)** -- CUT; prompt composition stays in render_driver (engine is
  adapter-only).

## CONVERGENCE
One pass. The reframe (suppressed-not-lost + the smear/canvas connection) is decisive and testable. STOP;
the next action is the controlled music_open-at-832x480 smoke, not more panel rounds.
