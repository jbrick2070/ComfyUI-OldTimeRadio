# Maximal motion per lane: what the PROMPT can reach, and what only a KNOB can

**Written because the driver is genuinely unsure** (operator: *"if you are
unsure of prompting/input knobs for maximal effect and motion, give me a
detailed problem statement with exact recipe for each and model info"*). The
prompt half is now baked in per lane. This is the half I cannot answer from the
repo alone.

## THE HARD CONSTRAINT THAT SHAPES THE WHOLE QUESTION

**The recipes are not on the table** (standing ruling,
`docs/OTR_STANDING_RULINGS.md:1440`): *"We spent a lot of time perfecting the
recipes to look good and we can't lose that."* No VRAM, speed or motion finding
justifies a recipe change; measurement runs the SHIPPED recipe unchanged.

So this document does **not** propose retuning. It asks a narrower question:
**for each lane, is motion reachable by PROMPT alone, or is it gated by a knob
we are forbidden (or unable) to move?** If the latter, that lane's motion has a
hard ceiling and we should stop paying render time expecting more.

## WHAT IS ALREADY DECIDED AND NEEDS NO RESEARCH

* **Audio-in lanes**: answered by the lab's human-reviewed envelope --
  `LIPSYNC_MOTION_ENVELOPE.md`. One main motion plus one minor motion; no
  locomotion. Baked in. Do not re-open.
* **Prompt wording on every lane**: baked in per-lane dialect, 2026-08-27
  (`65538f41`). Zero damping words remain in any live engine default.

## THE LANES, THEIR EXACT RECIPES, AND THE OPEN QUESTION

### A. LTX 2.5 family -- `ltx25_video`, `ltx25_foley_plus`, `ltx25_mime`

Recipe is CONSTANTS in `nodes/_otr_video_engines/ltx25_recipe.py`, **no env
knobs at all**:

| constant | value |
|---|---|
| DiT | `LTX-2.5-Distilled-Q3_K_M.gguf` |
| text encoder | `gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf` |
| sampler | `euler_ancestral_cfg_pp` |
| steps | **8** |
| all three CFGs | **1.0** |
| frames / fps | **97 / 25** (3.88 s) |
| canvas | 832x480, rendered 2x = 1664x960 |

**THE CFG IS A VRAM CONTRACT, NOT A TASTE SETTING** (`ltx25_recipe.py:98-101`):
raising any of the three is lab-measured to push past 16 GiB -- instant OOM
against the 14.5 GiB clamp. The file says in as many words: *"turn the CFG up a
little" is not a small change here. Leave them.*

**QUESTION 1:** on a distilled 8-step CFG-1.0 model, how much of motion
amplitude is actually prompt-reachable? Distilled low-step models are widely
reported to be motion-damped by the distillation itself. If LTX 2.5's motion
ceiling is set by 8 steps and CFG 1.0, then prompt work has a low ceiling here
and we should know the number rather than keep rewording.

### B. WAN family -- `wan_ti2v` (`wan22_high_video`), `fastwan_8gb` (`wan22_high_fast`)

**`wan_i2v` was RETIRED 2026-08-26; `wan_ti2v` is the live successor.**

Env knobs EXIST on this family (unlike LTX 2.5):
`OTR_WAN_TI2V_STEPS`, `_CFG`, `_SHIFT`, `_SAMPLER`, `_SCHEDULER`, `_MAX_FRAMES`;
and `OTR_FASTWAN_8GB_STEPS`, `_CFG`, `_SHIFT`, `_SAMPLER`, `_SCHEDULER`,
`_MAX_FRAMES`, `_LORA_STRENGTH`.

Shipped values are in `config/profiles/otr_8gb_wan.env.json`
(`OTR_WAN_TI2V_STEPS=30`, `CFG=5.0`, `SAMPLER=euler`, `SCHEDULER=simple`,
`MAX_FRAMES=81`).

**QUESTION 2:** `SHIFT` is the classic WAN motion-amount control and `CFG=5.0`
is a real (non-distilled) guidance value -- so this family plausibly HAS
prompt-reachable motion headroom that LTX 2.5 does not. Which of shift / cfg /
steps actually moves motion amplitude on **Wan 2.2 TI2V**, and does the shipped
30-step CFG-5.0 recipe already sit in the responsive region? For `fastwan_8gb`,
does its speed LoRA damp motion the way distillation does?

### C. MiniMax H3 silent -- `minimax_h3_video` (`h3_low_video`)

**NOT audio-in.** FL2VA unet `minimax_h3_fl2va_pruned_int8_convrot.safetensors`,
CLIP `qwen3vl_32b_minimax_h3_nvfp4_awq`, video VAE fp16. Takes **no negative
prompt at all**. Frame grid `n % 17 == 5`; model runs 24 fps against OTR's 25.
**No motion env knobs found in the adapter.**

**QUESTION 3:** with no negative prompt and no exposed knobs, is prompt the ONLY
motion control on this lane? Its own directive says *"name the subject, then one
action and its speed"* -- does naming SPEED measurably change amplitude, and is
`pruned int8 convrot` quantisation itself motion-damping versus the reference
model?

### D. LTX 2.3 family -- `ltx_video`, `ltx_8gb`

Knobs exist: `OTR_LTX_STEPS`, `_CFG`, `_SAMPLER`, `_I2V_STRENGTH`,
`_DISTILLED_LORA_STRENGTH`, `_MAX_FRAMES`; and the 8 GB tier adds
`OTR_LTX_8GB_BASE_SHIFT` / `_MAX_SHIFT`.

**QUESTION 4:** `I2V_STRENGTH` and the shift pair are the likely motion levers
here. What are the shipped values, and does lowering init-image strength buy
motion at an acceptable identity cost? (Identity drift is a known OTR defect
class -- do not trade a stable face for movement without saying so.)

### E. Excluded, settled

`mesh_stage` -- a Blender camera orbit around a GLB; operator ruled it OUT
(*"there's no action"*). Only `OTR_HY3D_CFG` / `_STEPS`, which govern mesh
generation, not performance.
`animatediff15_v3_haunted_video` -- Ghost lane, owns its whole prompt via
`ghost_signal_prompt.py`; only `OTR_GHOST_HAUNTED_LORA_STRENGTH`.

## WHAT COUNTS AS AN ANSWER

**Admissible:** the model card or paper for the exact checkpoint; the upstream
ComfyUI node's source; the `vram-recipe-lab` if it already probed the knob; a
live A/B on this box **running the SHIPPED recipe** and varying only the prompt.

**NOT admissible:** analogy from a different quantisation or a different base
model; "more steps means more motion" without a citation; a single clip.

**And the rule that governs any live test:** a knob probe that changes the
recipe is a RECIPE CHANGE, which is forbidden. If the honest answer is "motion
on this lane is knob-gated and the knob is locked", that is a *finding* -- write
it down and stop spending render cycles on prompt rewrites that cannot move it.

## WHY THIS MATTERS COMMERCIALLY

Operator: *"let's get people the motion their render time deserves."* An LTX 2.5
beat costs ~4-5 minutes of GPU. If that lane's motion is capped by an 8-step
CFG-1.0 distillation, the money is better spent routing motion-heavy beats to a
lane with headroom (WAN) than re-wording prompts for a lane that cannot obey
them. **The deliverable is a per-lane verdict: prompt-reachable, knob-gated, or
capped.**
