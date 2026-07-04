# HuMo talking-head quality + VRAM-fit -- PROBLEM STATEMENT for the panel

Harden the SOLUTION SPACE (ideas, ranked, with implementability + wiring), not a
finished plan -- this starts as an open problem driven by fresh operator eyeball
feedback. Diagnostic stance: nothing in production gets changed without an
operator-gated green light. Constraints below are HARD.

## Context (grounded; see the files)
OTR is a ComfyUI custom-node pack (`nodes/_otr_video_engines/eng_humo.py`) whose
audio-driven talking-head engine is HuMo. Tiers (eng_humo.py): `humo` /
`humo_14B_169` (14B fp8 + lightx2v 6-step distill LoRA + ModelSamplingSD3 shift 8,
cfg 1.0), `humo_1.7B` / `humo_1.7B_169` (no LoRA, 20 steps, cfg 2.5). The shipping
profile `config/profiles/16gb_full.json` pins `humo_1.7B`. In-process render via
`nodes/_otr_video_engines/wrapper_bridge.py` (run_graph + encode_frames_to_silent_mp4).

A just-completed STANDALONE diagnostic bakeoff (this folder's RESULTS.md;
`scripts/run_humo_bakeoff.py` + `scripts/build_humo_bakeoff_workflow.py` + sibling
`custom_nodes/otr_bakeoff_helper` OTR_BakeoffReclaim encoder-only evict) rendered 4
legs on an RTX 5080 16GB (fixed still c02_466a19906ccb.png + audio c02_b002_line.wav,
seed 0, 49f @832x480). External nvidia-smi peaks: 14B single 15996 MB, 14B two-stage
(umt5+whisper evicted pre-sampler) 15779 MB, 1.7B control 15089 MB, sentinel (LTX-AV
resident then 14B two-stage) 15974 MB. The two-stage encoder evict shaved only ~217 MB.

## Operator eyeball verdict (the NEW signal driving this)
On the 14B-vs-1.7B side-by-side: **14B = usable/keeper; 1.7B = REJECT for final.**
The 1.7B has foreground shoulder/neck clutter, an accidental over-the-shoulder crop,
weak expression, a less clear face, and "AI mush" on the coat edges. BOTH tiers render
a **weak, unrealistic mouth/teeth interior** ("broken mouth, no teeth, not realistic")
-- worse on the 1.7B; softened further by the 6-step distill.

## The two problems to solve
1. **VRAM FIT of the preferred look.** The operator-preferred 14B does NOT fit safely
   on 16GB (rides ~15.8-16 GB; the single-resident invariant is <=14.5 GB,
   `wrapper_bridge.VRAM_CEILING_MB`). The two-stage encoder evict is insufficient. We
   need the 14B-quality talking head to peak with REAL headroom (target <= ~13.5 GB) so
   it does not OOM under production cross-engine pressure / longer beats.
2. **MOUTH/TEETH REALISM.** The audio-driven mouth interior is unconvincing on HuMo at
   the fast distill settings. Need ideas to materially improve lip/teeth realism.

## Candidate levers (seed list -- critique, expand, rank; do NOT treat as decided)
- Quantized HuMo-14B GGUF (lower the ~14 GB fp8 UNET weight itself) -- r2_plan flagged
  this as "the only lever that lowers the UNET weight" if two-stage can't hit 13.5.
  HuMo today uses `UNETLoader`/`CLIPLoader` (NOT `UnetLoaderGGUF`); a GGUF HuMo would
  need a HuMo-specific GGUF loader + audio-cross-attn mapping + /object_info verify.
- 14B no-distill-LoRA at higher steps (~20-25, cfg up) for mouth quality -- but raises
  VRAM + reintroduces blue cast; r2 judged it does not fix the core VRAM defect.
- Smaller / quantized text encoder (umt5_xxl_fp8 is ~5 GB resident) or a true two-stage
  graph split that frees the TE block before the heavy forward (BUG-265 says naive
  inter-node eviction fragmented the allocator into OOM -- so HOW matters).
- Lower native resolution / shorter clip / tiled VAE decode to cut activation memory.
- A different/newer audio-driven-face model (LatentSync / MuseTalk / Sonic / Hallo /
  newest open lip-sync) -- prior note: LatentSync/MuseTalk were declined earlier on
  Blackwell (sm_120 / torch 2.10 / CUDA 13) dependency risk; re-evaluate.
- Mouth-only refinement pass (composite a dedicated lip-sync model over the HuMo face)
  vs full-frame regeneration.
- Better/higher-res input portrait (the still feeding HuMo) to reduce face mush.

## HARD constraints (reject any idea that breaks one)
- Single resident heavy engine <= 14.5 GB (target <= 13.5 for headroom); RTX 5080 16GB,
  Blackwell sm_120, torch 2.10 / CUDA 13, Windows. 100% local, open-source, offline-first.
- In-process ComfyUI wrapper-node path (no HTTP, no GraphBuilder); cold-import clean
  (V-12: no torch at module scope). UTF-8 no BOM, SFW. Output stays ALWAYS-SILENT
  (only OTR_MasterAudioMux adds audio); frozen master audio is read-only.
- Any node/widget/wiring change goes IN `workflows/otr_scifi_16gb_full.json` in the SAME
  change as the code (CLAUDE.md S0). Suite + Bug Bible + B7 green before each commit;
  commit+push per green chunk to v2.0-alpha; prod/main + tags GATED.
- The bakeoff harness (the 3 new files) is diagnostic and must NOT edit eng_humo.py /
  the workflow JSON / the OTR pack __init__ for measurement.

## What we want OUT of the panel
A ranked set of ideas to (1) make the 14B-quality look fit <= ~13.5 GB and (2) improve
mouth/teeth realism -- each with a concrete first implementation + wiring step, the main
risk, and how to measure it on this exact box. Prefer levers that reuse the existing
in-process wrapper_bridge graph + the just-built bakeoff harness for measurement.
