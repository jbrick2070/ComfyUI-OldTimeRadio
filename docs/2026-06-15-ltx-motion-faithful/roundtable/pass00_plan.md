# Faithfully Reproduce the 5/30 LTX Motion in the New Engine -- Plan to Harden

## Goal
The 5/30 LTX radio beats had GREAT movement; the current engine's first/last LTX beats barely move.
Operator wants the **5/30 motion recipe reproduced faithfully in the new architecture** (recipe-identical,
not literally byte-identical output -- the code was rewritten in the cleanbreak). Find the dominant lever(s)
and the faithful-restoration path.

## Measured gap (this session, all 832x480, same metric)
- **5/30 b005 (commit `9ac437b`, `batch_ltx_render.py`): framediff 9.43, optical-flow 0.344** -- the TARGET.
- 6/3 b005 (`59d9179`): framediff 2.34, flow 0.061.
- current engine, isolated smoke, GENERIC calm prompt:
  - distilled 8-step cfg1 (current default): framediff 0.72, flow 0.014.
  - ksampler 30-step cfg3 (BUG-113b): framediff 2.25, flow 0.039.
  - ksampler 15-step cfg3: framediff 1.07, flow 0.035.
- So ksampler buys ~3x over distilled but is still ~4x short of 5/30. The constants alone don't explain it.

## What is ALREADY identical (grounded -- NOT the cause)
5/30 (`9ac437b` `batch_ltx_render.py`) vs current (`eng_ltx_video.py`): same model `ltx-video-2b-v0.9`,
sampler `euler_cfg_pp`, 8-step distilled `LTX_DISTILLED_SIGMAS`, CFG 1.0, i2v strength 0.75, end-anchor 0.0
(retired BUG-032), `frame_rate=25` in LTXVConditioning, 832x480, loop_via_reverse (now restored). The
recipe CONSTANTS match -- so the 13x gap is elsewhere.

## Candidate levers (grounded; rank in the roundtable)
1. **MOTION PROMPT (prime suspect).** 5/30 `batch_ltx_render.py` had a built-in `_PROMPT_BY_ROLE` dict with
   VARIED, DYNAMIC per-role motion templates -- e.g. "Dial **whip-pans** across ... Speaker grille **vibrates
   aggressively**. **Dynamic dolly push** forward", "**Snap zoom** on the ...", plus calmer pull-back/settle
   variants. The NEW `eng_ltx_video.py` has `required_inputs=("text_prompt",)` and takes the prompt from
   UPSTREAM (render_driver / story-brief / `otr_meta_brief_image_prompt` et al.). The cleanbreak moved prompt
   composition out of the engine; the dynamic motion language may be diluted/calmer now. My isolated smoke
   used ONE calm template -> 0.72 (vs 9.43). Strongly implicates the prompt. (BUG-LOCAL-112 + `db14f9e`
   "restore 6/5 motion-centric prompts" are the prior fixes in this area -- verify they survived into the
   live per-beat LTX prompt.)
2. **SAMPLER default.** BUG-113b (`e3edce9`, 6/12) found the new distilled chain = subtle, ksampler-30-cfg3 =
   dynamic, and made ksampler the default; BUG-412 (`21bfe7a`, 6/14) reverted to distilled for SPEED
   ("30-step too slow"). The boomerang's half-render now offsets that speed cost. Secondary lever (~3x in the
   smoke), not sufficient alone.
3. **SOURCE LENGTH.** 5/30 b005 rendered ~81 source frames (loop, 6.4s beat) -> more motion PER FRAME; the
   smoke rendered 169 no-loop. Shorter source compresses the motion arc -> higher framediff. Secondary.
4. **SEED / INIT STILL.** Smoke used a fixed test still + seed 42; 5/30 used the episode radio_bookend FLUX
   still + OS-entropy seed. Could contribute; least controllable.

## Proposed faithful-restoration (to be hardened)
- **Primary: restore the dynamic motion-prompt language** into whatever now composes the per-beat LTX prompt
  for announcer/music (upstream of the engine). Reinstate the 5/30 `_PROMPT_BY_ROLE` dynamic variety
  (whip-pans / snap-zoom / aggressive-vibrate / dynamic-dolly), seeded per-beat for variety + determinism.
  Locate the CURRENT prompt-composition site first (render_driver `build_request_from_shot` / the brief
  helpers) -- VERIFY-AT-BUILD where the LTX text_prompt is actually built.
- **OPTION (OPERATOR-FAVORED): a NEW LLM motion-prompt pass for LTX stills.** Instead of (or layered over)
  static templates, add a small LLM pass that takes the Meta story-brief for the beat and generates a
  UNIQUE, dynamic LTX motion prompt per still ("take the Meta brief and say: we need a unique motion prompt
  for this beat"). Reuses the existing writer-LLM lane (local gemma/mistral-nemo, or an OpenRouter slot --
  same infra as the script/image-prompt passes); fail-CLOSED to the 5/30 dynamic `_PROMPT_BY_ROLE` template
  if the LLM is unavailable; seed-keyed for determinism + per-beat variety. FIRST verify whether such a pass
  already exists -- `get_story_brief_ltx` (C5e), `otr_meta_brief_image_prompt.py`, `_otr_music_prompt.py`
  and the brief helpers already route the Meta brief into prompts; the question is whether the LTX motion
  prompt is LLM-generated + dynamic today or a calm static line.
- **Secondary: reconsider the sampler default** (ksampler vs distilled) now that the boomerang halves render
  cost -- operator speed/motion call.
- Keep the boomerang; no workflow-JSON change preferred; determinism (seed-keyed); audio byte-identical;
  <=14.5GB; the LLM pass must be local-capable + fail-closed (no hard cloud dep).

## Questions for the panel
1. Given identical sampler CONSTANTS but a 13x motion gap, is the PROMPT the dominant lever (vs sampler vs
   length)? What's the cleanest way to confirm attribution (a controlled smoke with the 5/30 dynamic prompt
   vs the current prompt)?
2. How to faithfully reinstate the 5/30 dynamic `_PROMPT_BY_ROLE` motion language in the NEW upstream
   prompt pipeline without breaking the story-brief integration -- engine-local fallback vs brief-level?
3. Does source-frame count (81 vs 169) materially change LTX per-frame motion -- should the loop render even
   fewer source frames for the announcer bookend?
4. Anything we're missing in the cleanbreak rewrite that silently dampens motion (conditioning, negative
   prompt, sigma schedule, guidance).
5. **LLM motion-prompt pass vs static dynamic templates.** The operator favors a NEW small LLM pass that
   turns the Meta story-brief into a UNIQUE per-beat LTX motion prompt. Does an LLM-driven LTX motion prompt
   already exist (`get_story_brief_ltx` / brief helpers), or is the live LTX prompt a calm static line? If
   new: best design -- where it hooks (engine-local vs brief-level), which model (local gemma/mistral-nemo
   vs OpenRouter slot), how to keep it deterministic + fail-closed to the 5/30 dynamic template, prompt-length
   budget (LTX <=~188 chars, BUG-112), and how NOT to regress the story-brief or audio byte-identity. Is the
   LLM pass worth it over simply reinstating the 5/30 dynamic `_PROMPT_BY_ROLE` variety?
