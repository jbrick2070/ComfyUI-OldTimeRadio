CLAUDE ANCHOR -- r1 (high-level arc / creative coherence) -- grounded vs the real code

VERDICT: yes-with-fixes. The arc is coherent and matches the goal (fix the bakeoff-isolated
defects via the two levers the bakeoff measured), but it does not close all THREE original
complaints, and the "recipe-agnostic" claim needs one explicit guard.

CONFIRMED (verified against the files this session):
- Decode temporal_size/overlap is a HARDCODED dict in eng_ltx_av.py:556-559, set AFTER the
  recipe branch -> the change is genuinely recipe-agnostic (applies to sharp_lora /
  distilled_native / m0_base alike). CONFIRMED.
- Scaler is otr_silent_composite._seg_vf (~319-325), a Python ffmpeg string, NOT a node widget.
  CONFIRMED -> no canonical-workflow-JSON edit needed (CLAUDE.md S0 governs node/widget changes).
- Ceiling = 14500 (wrapper_bridge.py:37). CONFIRMED. Whole-clip peaked 14338-14473 (27-162 MB
  headroom); 128/32 peaked 14272 (~228 MB). The plan's choice of 128/32 for headroom is coherent.
- Whole-clip 4096/8 == eng_ltx_video.py:766-767. CONFIRMED (the byte-for-byte sibling).

MUST-FIX (arc coherence):
1. The plan fixes SEAM (decode) + SOFTNESS (scaler) but NOT the third original complaint, the
   init-hold STUTTER. The plan honestly notes freezedetect read 0 at baseline so the bakeoff
   never reproduced it -- but "the LTX upgrade" framing implies all three are closed. Make it
   explicit: this upgrade ships 2 of 3; stutter (i2v 0.62) stays an OPEN eyeball item, not fixed.

SHOULD-FIX:
1. State the seam metric's residual honestly: 128/32 is "imperceptible" (jump 0.57x the local
   median), NOT zero like whole-clip. The arc trades a faint residual seam for VRAM safety -- a
   defensible call, but name it as a trade, not a clean win.
2. The recipe-agnostic decode change benefits sharp_lora too (same VAE-tiling mechanism), but the
   bakeoff only measured distilled_native -- note that sharp_lora's seam behavior is INFERRED
   (same node, same VAE), a verify-at-build, not a measured result.

[ASSUMPTION] none beyond the above; all decode/scaler/ceiling claims are code-grounded.
