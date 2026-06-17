# OTR v2.1 roadmap (captured 2026-06-16)

Queued from the motion/character-consistency work. Each is a spec -> roundtable -> build
-> E2E item (the path that worked for motion_clause).

## 1. Wardrobe-change intelligence (LLM, stateful)
The smart half of outfit handling (the simple LOCK ships in v2.0 now). An LLM walks the
ordered beats and decides, per character, when a costume change is NARRATIVELY justified
(time skip, "next morning", new location/day, explicit script costume change) vs. keep the
locked outfit. Produces a per-character WARDROBE TIMELINE (start outfit + justified
change-points), persisted in the ledger, feeding the FLUX `{appearance}`.
- Default bias = continuity; only insert a change when the script justifies it.
- Stateful (carries outfit state across beats), not a per-beat coin-flip.
- Reuses make_writer_generate_fn + per-episode pass + ledger persistence (motion_clause
  family). Builds ON TOP of the v2.0 outfit-lock.

## 2. Second image engine -- Lumina (more than just FLUX)
Operator wants >1 image generator; FLUX is engine #1, **Lumina is the #2 pick** (confirm
exact model/lane at spec time -- local vs cloud, Blackwell/torch compat). Trace the Lumina
image path and confirm it carries the SAME goodies as FLUX (appearance/wardrobe injection,
still output contract) so it can feed LTX + HuMo.
- OTR already has a pluggable image-engine selector: `role_overrides.*_image` (today
  `flux_gen1`). A 2nd engine = a new image-engine ADAPTER registered in that selector.
- **HARD requirement (operator): the 2nd engine MUST play nice with the video engines.**
  Its stills/portraits are the INIT IMAGE for LTX i2v AND HuMo (audio-driven face) AND
  Wan etc. So the still->video handoff contract must hold: matching canvas/resolution
  (e.g. 1472x832 landscape / portrait sizes the video engines expect), file format,
  color, and character fidelity. Spec must include a still->LTX + still->HuMo
  compatibility gate (render a still from engine #2, confirm LTX i2v + HuMo consume it).
- Selector UX: per-role image engine pick (announcer/music/character), same as the video
  A/B/C selector pattern.

## 3. Pixel-true character consistency -- IP-adapter / reference
Prompt-level outfit-lock (v2.0) stops wholesale swaps but text-to-image still wiggles. For
pixel-true same-character/same-outfit across scenes: IP-adapter / character reference image
/ per-character LoRA. Heavier (VRAM + a reference pass) -- spec + roundtable. Pairs with #1.

## Shipping now (v2.0): simple outfit-LOCK
Pin a per-character wardrobe clause in the locked appearance so every beat's FLUX prompt
carries the same outfit (scene/pose vary). Low-risk, gated, the demo-critical "characters
stay themselves" win. (This file = the v2.1 queue; the lock is a v2.0 commit.)
