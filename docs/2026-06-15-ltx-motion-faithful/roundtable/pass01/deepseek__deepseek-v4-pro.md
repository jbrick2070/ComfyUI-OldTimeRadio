<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan's primary restoration is already in the grounded code, the LLM pass is undecided, the prompt budget is wrong, and the plan lacks verification of the current state.

MUST-FIX BEFORE BUILD:
1. [Proposed restoration] The plan assumes the dynamic motion-prompt language is missing and needs restoring. The grounded `render_driver.py` already contains `_LTX_MOTION_PROMPT_BY_ROLE` and uses it for ltx_video open roles. The plan must include a verification step to confirm the actual per-beat LTX prompt is calm; if it is already dynamic, the plan is redundant. Fix: add a verification step that logs the prompt for an announcer beat and compares with the 5/30 templates. If they match, the plan should pivot to other levers or refine the templates.
2. [Questions / Proposed restoration] The optional LLM motion-prompt pass is not decided. A build-ready plan must decide. Fix: remove the LLM pass from the plan, or commit to it with concrete hook location (engine-local vs brief-level), model, and the correct 240-char budget.
3. [Proposed restoration] The plan quotes a prompt-length budget of "~188 chars" (BUG-112). The grounded code uses `_LTX_MOTION_PROMPT_MAX = 240` for motion-centric prompts. Using 188 chars would truncate the dynamic templates. Fix: use the correct budget of 240 chars for motion prompts, and cite the constant.
4. [Candidate levers] The plan does not consider the negative prompt. The 5/30 recipe might have used a different negative that allows more motion. Fix: add a step to compare the 5/30 negative prompt with the current `_LTX_DEFAULT_NEGATIVE` and test if adjusting it closes the gap.
5. [Measured gap] The plan's smoke test used a calm template, not the actual per-beat prompt. The conclusion that the prompt is the dominant lever is not based on the current real prompt. Fix: run a controlled smoke using the current `_LTX_MOTION_PROMPT_BY_ROLE` templates (e.g., announcer) to measure motion and compare with the 5/30 target.
6. [Proposed restoration] The plan limits dynamic motion restoration to announcer/music roles. Other LTX-capable roles (scene_broll, background_abstract) may also need dynamic motion. Fix: extend the motion templates to those roles or justify the exclusion.

SHOULD-FIX:
1. [Candidate levers] The plan identifies source length as a secondary lever but does not propose a change. The boomerang source length might still be longer than the 5/30's 81 frames. Add a recommendation to test a shorter source length (e.g., set `OTR_LTX_LOOP_MIN_DECODE_FRAMES=81` or adjust the target).
2. [Proposed restoration] The plan should verify that the boomerang source length for the announcer bookend is close to 81 frames (the 5/30 value). The current `_ltx_loop_source_length` logic may yield a different number. Add a check.

OPTIONAL / NICE-TO-HAVE: none.

CUT THESE (over-engineering):
1. The LLM motion-prompt pass. The static `_PROMPT_BY_ROLE` templates already provide dynamic, varied motion prompts and are deterministic. The LLM pass adds complexity, latency, and a dependency on an LLM without evidence that it would improve motion beyond the templates. The plan's own question suggests it may not be worth it. Cut it to keep the plan focused.

[ASSUMPTION] We assume the current per-beat LTX prompt for open roles is the dynamic motion template from `_LTX_MOTION_PROMPT_BY_ROLE`. The grounding shows this is the case.