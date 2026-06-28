# HuMo quality bakeoff -- OPEN-ENDED optimization review (for a PLANNED bakeoff)

GOAL: design an ISOLATED HuMo quality bakeoff (mirror scripts/run_ltx_av_q_bakeoff.py) that
(a) confirms HuMo's audio-driven FACE / lip-sync quality, (b) finds the best-quality settings
within the 14.5 GB VRAM ceiling, and (c) proves the 14B is NOT silently downgrading to 1.7B.
Engine = nodes/_otr_video_engines/eng_humo.py. This is an OPEN-ENDED review: propose + critique
the optimization levers the bakeoff should test. Read the real engine; ground every claim.

## Current state (grounded in eng_humo.py)
- 14B keystone (DEFAULT): Kijai fp8 UNET + lightx2v 480p distill LoRA @ 6 steps / cfg 1.0 /
  ModelSamplingSD3 shift 8. Portrait 480x832 @ 25 fps (humo_1.7B_169 variant = wide 832x480).
  Wan 2.1 VAE, 4n+1 length; _HUMO_MIN_FRAMES 33 / _MAX 177.
- 1.7B tier: no-LoRA, ~20 steps, slower, lower quality. Auto-downgrade chain
  humo(14B) -> humo_1.7B -> still_kenburns on VRAM/OOM (LOUD restamp). The de-blue fix dropped the
  1.7B cfg 5.0 -> 1.0 (killed a blue cast).
- Env knobs: OTR_HUMO_STEPS (6), OTR_HUMO_CFG (1.0), OTR_HUMO_UNET_NAME, OTR_HUMO_LORA_NAME
  (set =none to drop the distill LoRA), OTR_HUMO_CKPT, OTR_HUMO_NEGATIVE.
- Operator's worry: HuMo "lost quality" -- most likely the 14B silently downgrading to 1.7B under
  the AV stack's VRAM pressure (the stack cycles ~24 GB through the 16 GB card).

## Constraints / invariants (ComfyUI + OTR)
ISOLATED bakeoff: fixed still + driving-audio + seed, vary ONE lever per leg, side-by-side clips to
otr/episodes/_bakeoff_humo/<leg>.mp4 for the OPERATOR'S EYEBALL (quality is his call). Single
resident heavy engine <= 14.5 GB (HARD, host NVML); reset the box SELECTIVELY before each leg
(CIM by CommandLine, never a blanket python kill); 100% local; LOUD fallbacks; UTF-8 no BOM; SFW.
ComfyUI: respect VRAM/model-management (fp8 14B + LoRA residency), the node-class contract,
tensor layouts (Wan 2.1 VAE 4n+1), import isolation. Do NOT touch the frozen audio spine.

## OPEN QUESTION (the review)
What optimizations should the bakeoff TEST to maximize face/lip-sync QUALITY within 14.5 GB --
and to stop the 14B downgrading? Propose the bakeoff LEGS (lever x values), the METRICS, and
code-grounded gotchas. Starter levers to critique + extend (add/cut freely):
- TIER: 14B FORCED (no-downgrade) vs 1.7B -- the headline quality + confirm which actually runs.
- STEPS: 6 (lightx2v distill) vs 8/10 -- does 14B benefit from more steps, or does the distill cap it?
- CFG: 1.0 vs higher -- quality vs blue/over-saturation.
- LoRA: lightx2v ON (6-step) vs OFF (more steps, slower, maybe sharper / less distill artifact).
- SHIFT (ModelSamplingSD3): 8 vs alternatives.
- RESOLUTION: 480x832 native vs others (what is HuMo's trained sweet spot?).
- FRAMES: within 33-177; does length affect identity/lip-sync drift?
- DOWNGRADE PROTECTION: can the 14B be reserved/evicted so it never silently falls to 1.7B under
  VRAM pressure (akin to the LTX-AV reserve)?
- NEGATIVE prompt + the de-blue cfg interaction.

Deliver: the recommended bakeoff leg set (what to vary, what to hold fixed), the metrics (VRAM peak,
s/it, + any objective face/lip-sync proxy), the single fixed still+audio choice, and the gotchas
(e.g. the auto-downgrade masking results -- the LTX bakeoff's #1 risk was measuring the wrong graph).
