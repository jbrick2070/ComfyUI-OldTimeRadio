CLAUDE ANCHOR -- HuMo optimization review (r1) -- grounded vs eng_humo.py

VERDICT: the brief is sound; the headline lever is the distill-LoRA tradeoff, and the
load-bearing safety rail is forcing/recording the TIER that actually ran.

CONFIRMED (eng_humo.py):
- 14B default = Kijai fp8 + lightx2v 480p distill LoRA @ 6 steps / cfg 1.0 / shift 8. The distill
  LoRA is what ENABLES 6 steps -- and distillation trades QUALITY for speed. So the single most
  important quality lever is NOT "more steps on the distill" (the LoRA caps it) but
  **14B WITHOUT the distill LoRA (OTR_HUMO_LORA_NAME=none) at ~20-30 steps + cfg ~4-6**. That is
  the "max-quality 14B" config the bakeoff must compare against the fast 6-step default.
- The auto-downgrade chain 14B->1.7B->still means a VRAM-tight leg SILENTLY measures 1.7B. This is
  the LTX bakeoff's #1 lesson (measuring the wrong graph). The bakeoff MUST force-pin the 14B (no
  downgrade) AND record the tier/unet that actually ran in a fail-loud per-leg manifest.

MUST-FIX (anchor):
1. Per-leg manifest records the RESOLVED tier/unet/lora/steps/cfg + the tier that ACTUALLY ran
   (abort the leg if it downgraded) -- else a hot leg measures 1.7B and the result is garbage.

SHOULD-FIX:
1. The headline 3-way: (a) 14B + lightx2v distill @ 6 steps cfg 1.0 (the fast default), (b) 14B
   NO-LoRA @ ~25 steps cfg ~5 (max quality), (c) 1.7B (the fallback). This tells the operator
   whether the 6-step distill is the perceived quality loss and what the speed cost of the
   no-LoRA path is at the 14.5 GB ceiling.
2. cfg only matters WITHOUT the distill LoRA (distill models want cfg~1); pair cfg with the
   no-LoRA legs, not the distill legs.
3. Hold RESOLUTION at native 480x832 (HuMo's trained sweet spot); off-res risks the LTX-style
   softness/quality loss -- canvas is a fixed, not a swept, lever.
4. DOWNGRADE PROTECTION leg: force 14B with an EXTRA_RESERVED_VRAM bracket (the LTX-AV
   _ltx_av_vram_reserve pattern) so the AV stack's ~24 GB cycling can't push 14B into the 1.7B
   downgrade -- if that holds <=14.5 GB, it is the real production fix, not just a bakeoff knob.

[ASSUMPTION] HuMo's training res is 480x832 portrait (the code's native canvas); verify the
trained ckpt res before sweeping resolution.
