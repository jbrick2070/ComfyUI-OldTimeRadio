# Claude anchor -- r1 (HuMo improvements, arc)

Grounded vs eng_humo.py.

## CONFIRMED (levers are real + VRAM-neutral)
- Env-exposed, no VRAM cost: `OTR_HUMO_STEPS` (14B=6), `OTR_HUMO_CFG` (1.0), `OTR_HUMO_LORA_NAME`
  (lightx2v distill), `OTR_HUMO_UNET_NAME`/`OTR_HUMO_CKPT`, ModelSamplingSD3 shift 8. These move quality
  without raising VRAM -> the operator's "same 1.7B / same VRAM" constraint is satisfiable via
  steps/cfg/shift/LoRA-weight + the INIT PORTRAIT (the flux/z_image face HuMo conditions on) -- the init is
  likely the biggest quality lever and costs no HuMo VRAM.
- Underrun mush root CONFIRMED: `_HUMO_MAX_FRAMES = 177` (the empirical 480x832 ceiling). A 434-frame beat
  gets 177 -> composite holds the last frame = the murk. `safe_render_frames` trims/mirror-extends to the
  audio target. So the fix = extend-to-target (loop/ping-pong/mirror) instead of hold-last-frame.

## MUST-FIX (arc)
1. **Clip-fill = the #1 quality win** (ties GO_FORWARD S-A). Underrunning HuMo must LOOP/ping-pong/mirror to
   the beat's frame target, never hold a static last frame. Must stay no-fallback-compliant (it's a
   frame-fill within the SAME delivered engine, NOT an engine swap) + LOUD-stamped
   (attempted/delivered frames in the ledger). Decide the seam: composite vs render_driver.
2. **Host = radio, not a face** (operator hard pref). Announcer + music bookends route to a radio visual,
   NOT HuMo. Grounding TODO for the panel: locate `_NEVER_HUMO_ROLES` (the plan cites _otr_speaker_role.py
   but that path 404s -- find the real guard) and confirm it already keeps announcer/music off HuMo; the
   NEW work is routing them to the animated-radio visual (shared with the viz + mesh radio-bookend theme).

## SHOULD
- Portrait quality: an A/B of init-portrait framing/prompt + steps/cfg/shift is only meaningful with the
  HuMo-ISOLATION SMOKE (bake one episode's audio+ledger+portraits, re-render ONLY the HuMo beat, swap one
  knob). Build that fixture FIRST or the tuning is ~40-min/episode guesswork.
- Dropdown labels: auto-derived only (the `_engine_id_from_pick` " (" round-trip); no custom label string.

## SCOPE
- The "animated talking radio" is a candidate small new engine/motif -- scope it as its OWN thing, shared
  by HuMo-bookend + mesh-bookend + viz. Don't balloon eng_humo with it.
- HuMo phrase-chunking (long dialogue vs the frame cap) is the upstream root fix but is S-C territory --
  keep it POINTED-TO, not folded into this quality pass.
