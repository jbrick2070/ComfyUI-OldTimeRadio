# LTX Motion Restoration -- HARDENED (pass01, roundtable-converged + code-grounded)

3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4, $0.10) + Claude grounding vs the real
`render_driver.py`. The panel unanimously found pass00 STALE -- the prompts were never lost.

## THE REFRAME (grounded in render_driver.py)
The 5/30 dynamic motion prompts were NOT lost in the cleanbreak. They live in
`_LTX_MOTION_PROMPT_BY_ROLE` (L472) with the full 5/30 variety (whip-pans / "vibrates aggressively" /
dynamic dolly push). A brief-grounded LLM prompt path also already exists (`finish_visual_prompt` +
`get_story_brief_ltx`, L942-964) for non-open beats. So the motion is SUPPRESSED, not missing.

## WHY THE MOTION IS CALM (two grounded mechanisms)
1. **Deliberate smear-avoidance (dominant).** `_ltx_motion_role_key` (L505-511): the aggressive open-music
   verbs "SMEAR on the 2B LTX model -> the 'first radio, not sharp' open" (operator 2026-06-12), so the
   opening beat is retargeted to the CALM `music_inter` template. `OTR_LTX_OPEN_MOTION_KEY` defaults to
   `"music_inter"`; the aggressive `music_open` is bypassed by default.
2. **`text_prompt` precedence.** `build_request_from_shot` L886 gates the LTX motion block on
   `and not text_prompt`: if ShotLock/M4 supplies a creative prompt, the motion-role prompt is SKIPPED and
   LTX gets the (calmer) creative prompt instead of the motion language.

## KEY INSIGHT -- the smear was almost certainly TODAY's blur bug
The smear that drove the 6/12 suppression occurred when LTX rendered at **1472x832** (the over-resolution
mush = BUG-412). **That canvas bug was FIXED today** (`4fc4268`, native 832x480). So the aggressive motion
prompts may now render CLEANLY -- the original reason for suppressing them may no longer hold. PRIME
HYPOTHESIS to test before any code change.

## PLAN (test-first, then minimal fix)
1. **Controlled attribution smoke.** Hold still/seed/canvas(832x480)/sampler/strength/source-length fixed;
   vary ONLY the prompt: real `music_inter` (current) vs real `music_open` (aggressive) from
   `_LTX_MOTION_PROMPT_BY_ROLE`. Measure framediff/flow + EYEBALL smear. (Fixes the panel's catch that my
   earlier smoke used a GENERIC calm prompt, so it never tested the real templates -- the "prompt is the
   lever" claim is unproven until this runs.)
2. **If `music_open` is clean at 832x480:** flip `OTR_LTX_OPEN_MOTION_KEY` default to `music_open`, OR make
   it canvas-aware (aggressive <=480p, calm above). One-line default + update the 6/12 comment. If it still
   smears, operator picks the motion/smear trade.
3. **M4 precedence fix (GPT #2).** For `engine_id=="ltx_video"` open roles, the motion-role prompt should WIN
   over (or append its motion verbs to) the M4 `text_prompt`, so the motion language actually reaches LTX.
   Stamp `prompt_subsource` for audit. (Do NOT touch the engine -- prompt composition stays in render_driver,
   panel consensus.)
4. **Secondary levers (only if 1-3 insufficient).** Shorter loop source (`OTR_LTX_LOOP_MIN_DECODE_FRAMES` ~81
   to match 5/30 vs the current 97) -> more motion/frame; compare `_LTX_DEFAULT_NEGATIVE` to the 5/30
   negative; ksampler default (env-selectable, do NOT bundle with the prompt change -- destroys attribution).
   Prompt budget is **240** (`_LTX_MOTION_PROMPT_MAX`), NOT 188.

## THE LLM MOTION PASS (operator-favored) -- grounded verdict
LARGELY REDUNDANT for this build: `finish_visual_prompt` + `get_story_brief_ltx` already generate per-beat
brief-grounded prompts for NON-open LTX beats (the operator's "unique motion per shot, in line with the Meta
brief" is already served there). The OPEN bookends (announcer/music) use the static templates BY DESIGN. A
small LLM pass COULD add per-episode variety to the OPEN beats, but it adds nondeterminism / latency / VRAM
residency risk (all 3 panelists CUT it for now). RECOMMENDATION: prove the smear/canvas hypothesis first; if
static `music_open` variety is enough, skip the LLM pass; if the operator still wants per-episode open-beat
variety, scope it as a SEPARATE deterministic + fail-closed pass with a defined hook/cache/timeout.

## ACCEPTANCE / INVARIANTS
Determinism (template/key selection by shot id, not RNG); audio byte-identical; no workflow-JSON change;
<=14.5GB; the boomerang stays. Mechanical acceptance: trace rows for the target LTX beats show the intended
`prompt_source` + a non-calm `prompt_sha8` + `init_source=="scene_still"`. Build-time config-assert smoke
logs the EFFECTIVE ckpt/sampler-mode/sampler-name/cfg/strength/canvas/source-length/seed/prompt-source for
the beat (the panel's "stop claiming identical, prove it" fix).
