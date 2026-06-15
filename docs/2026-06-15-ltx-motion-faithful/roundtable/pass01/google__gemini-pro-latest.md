<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan's primary proposal directly reverts an operator quality fix, and its secondary proposal is already implemented.

MUST-FIX BEFORE BUILD:
1. [Primary: restore the dynamic motion-prompt language] The plan assumes the aggressive 5/30 motion prompts ("whip-pans", "vibrates aggressively") were accidentally lost in the cleanbreak. They were not. `render_driver.py` (`_ltx_motion_role_key`) explicitly suppresses them: it retargets the opening music beat to the calmer `music_inter` prompt because the aggressive verbs "SMEAR on the 2B LTX model -> the 'first radio, not sharp' open" (Operator 2026-06-12). Reinstating them will re-break the look. Fix: Acknowledge the smearing trade-off; if you want the motion back, you must test if the smearing is now acceptable or use `OTR_LTX_OPEN_MOTION_KEY=music_open` to bypass the safety.

CUT THESE (over-engineering):
1. [OPTION: a NEW LLM motion-prompt pass] Cut this entirely. `render_driver.py` already implements a brief-grounded LLM prompt composition for non-open LTX beats (calling `get_story_brief_ltx` and `finish_visual_prompt` around line 990). Building a *new* LLM pass duplicates existing code that already fulfills the requirement.