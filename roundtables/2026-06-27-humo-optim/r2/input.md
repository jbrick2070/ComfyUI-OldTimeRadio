# HuMo optimization settings -- r2 (coding/implementability) + operator decision gate

Advancing the r1 HuMo plan. r1 established (VERIFIED): there is NO silent downgrade (render_shot raises
loud); the shipping `config/profiles/16gb_full.json` PINS `video_render_engine: humo_1.7B`; the 14B
keystone was demoted by BUG-265 (2026-05-24) because the 5/21 recipe staged ~16.5GB and OOM-thrashed.
The 5/21 QUALITY TARGET = HuMo-14B fp8 + lightx2v 480p distill LoRA + ModelSamplingSD3 shift 8.

## OPERATOR DECISION GATE (this is the whole point of r2)
IF 14B can be made to fit SAFELY with REAL headroom -- peak <= ~13.5 GB so it is NOT riding 100% of the
16 GB card -- PROMOTE it (flip 16gb_full role+slot 1.7B->14B, same-change profile edit + re-validate).
IF it CANNOT fit safely, KEEP 1.7B (reliability wins) and harden the 1.7B settings instead.

## r2 questions -- ground every claim in the real code, give an IMPLEMENTABLE answer
1. HOW to fit 14B under ~13.5 GB. The 5/21 blocker was the 14B fp8 + LoRA + the umt5 text-encoder
   (~5.2 GB resident, CS-4) co-residency. Candidate levers -- assess feasibility + exact code site:
   - lazy umt5-TE DETACH/evict after conditioning, before the HuMo forward (CS-4-open names this; where
     in eng_humo.py / wrapper_bridge would it hook? does reclaim_idle_models already cover the TE?).
   - a smaller or quantized text-encoder (GGUF umt5?) -- does the loader support it?
   - a lighter 14B quant (the unet is fp8 today; is a GGUF Q-quant of the 14B available/loadable like LTX?).
   - the BUG-291 detach + the in-process render path (the 5/21 88x-slowdown batch arch is GONE) -- how
     much did real peak already drop vs the 5/21 16.5 GB stage?
   For EACH: feasible on this stack (torch 2.10 / cu13 / sm_120 / Windows / Wan-2.1 VAE 4n+1)? code site?
2. OPTIMAL settings per tier for reliability + quality: steps / cfg / shift / quant / negative, for BOTH
   the 14B (distill 6-step vs no-LoRA ~25-step) and the 1.7B fallback (the de-blue cfg 1.0 is applied).
   Which knobs actually move quality vs just cost VRAM/time?
3. NEWER MODEL CHECK (operator asked): is there a NEWER, MORE RELIABLE audio-driven talking-face / lip-sync
   model than HuMo-14B that would fit 16 GB with headroom AND run on this stack (Blackwell sm_120, torch
   2.10/cu13, ComfyUI custom-node, local-only)? Name candidates with the honest caveat (dep risk, license,
   maturity) -- we will NOT adopt one unproven, but flag if HuMo is no longer the best 16 GB choice.

## Constraints / invariants
Single resident <= 14.5 GB hard (target the heavy engine <= ~13.5 GB for real headroom); 100% local;
don't touch the frozen audio spine; UTF-8 no BOM; SFW; any node/widget change goes IN
workflows/otr_scifi_16gb_full.json SAME change (CLAUDE.md S0). Quality stays the operator's eyeball.

## Deliver
A ranked, code-grounded implementation answer to the decision gate (the ONE best way to fit 14B safely,
with its code site + expected peak, OR the verdict "keep 1.7B because X"), the optimal per-tier settings,
and an honest newer-model verdict (stick with HuMo, or a named candidate worth a probe).
