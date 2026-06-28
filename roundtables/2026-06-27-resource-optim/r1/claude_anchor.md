CLAUDE ANCHOR -- OTR resource-optim + portability (r1) -- HuMo-reliability lens

Operator's real goal: a STABLE, HQ, RELIABLE HuMo that runs smoothly on the 5080 with headroom and
degrades for modest boxes. Since render_shot has NO fallbacks (raises loud), HEADROOM = RELIABILITY:
if HuMo's 14B peak exceeds the ceiling, the episode FAILS. So this review is the reliability question.

CONFIRMED (grounded):
- Ceiling = motion_common.VRAM_CEILING_MB 14500 (of 16 GB), but dynamic_vram_ceiling_mb() is
  env-overridable per profile (GATE B S1) -- so the ceiling is tier-aware, not a hard constant.
- Portability infra SHIPPED: _otr_shared/capability_profiles.py + OTR_WorkflowValidator host-detect
  (_otr_workflow_validator.py:275-310) ABORTS with a suggestion table (no-cuda->cpu_floor;
  VRAM<10GB->8gb_lite; mac->cpu_floor). _otr_workflow_apply imports it. So host-detect + tiers are
  declared and engine registries carry cpu_floor lane filters.

MUST-VERIFY (the load-bearing portability claim -- for the agents + me to ground):
1. Do 8gb_lite / cpu_floor actually RUN end-to-end, or does the validator just ABORT-with-suggestion
   on a sub-spec box (detect != run)? A 32 GB-RAM / small-GPU user needs a tier that RENDERS, not one
   that tells them to go away. If only the 16gb full tier truly runs, OTR is effectively 5080-only.

CONCERNS (anchor):
1. HEADROOM is TIGHT. 14.5 GB of 16 = ~1.5 GB margin. A laptop daily-driver (driver + desktop +
   browser) routinely eats 1-2 GB, so a "smooth, headroom" target may need the ceiling LOWER (or the
   heavy engines lighter). The 2026-06-26 LTX bakeoff measured the picked path at ~15.1-15.5 GB
   ISOLATED -- at/above the 14.5 ceiling. If HuMo 14B fp8 peaks similarly, it rides the edge -> the
   exact "HuMo sometimes fails / looks off" the operator feels. [verify HuMo 14B real peak.]
2. The HQ-vs-reliable tension: HuMo's HQ path (14B no-LoRA ~25 steps, per the bakeoff) costs MORE
   VRAM + time than the fast 6-step distill -> may not fit the headroom budget. The reliable default
   might have to stay the distill; HQ becomes an opt-in higher tier. The bakeoff must report BOTH
   peaks so we know if no-LoRA even fits.
3. Quant headroom lever: if LTX-AV/HuMo can drop one quant step (e.g. Q3_K_M->a tuned Q3/Q2 mix) and
   hold quality, that buys the 1-2 GB of real headroom that makes the 5080 STABLE under daily load --
   directly the operator's ask. The LTX bakeoff already has the quant ladder; HuMo's fp8 is the peer.

[ASSUMPTION] HuMo 14B fp8 resident peak is in the 13-15 GB band (Kijai fp8 + LoRA + Wan VAE);
the JOB 3 Phase-B sentinel is what actually measures it under the AV stack -- verify there.
