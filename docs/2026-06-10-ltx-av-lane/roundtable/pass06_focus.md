# PASS 06 REVIEW FOCUS: HARDWARE

You are one panelist in an adversarial review of the plan below. THIS pass
is the HARDWARE pass. Pass01-05 LOCKED -- one-line flags only.

Box: RTX 5080 Laptop 16GB VRAM (sm_120, Blackwell), torch 2.10/cu130 main
venv, machine-NVML single-resident ceiling 14.5GB, Windows, Comfy Desktop
for the operator + a headless launcher for renders. The existing 2B
ltx_video does 1472x832 opens at ~6 min/clip. System RAM size UNKNOWN to
the plan (the M0 sheet must record it).

JUDGE-VERIFIED file sizes (HF API, 2026-06-10 -- treat as ground truth;
do not parrot community VRAM folklore over these):
- Kijai 22B distilled-1.1 fp8_scaled transformer: 23.5 GiB file
- QuantStack LTX-2.3-distilled GGUF: Q2_K 11.6 / Q3_K_S 13.0 /
  Q3_K_M 13.7 / Q4_K_S 15.6 / Q4_K_M 16.5 / Q5_K_S 17.3 / Q5_K_M 18.1 /
  Q6_K 19.6 / Q8_0 23.7 GiB
- gemma_3_12B_it_fp8_scaled text encoder: 13.2 GB (placement
  models/text_encoders; CPU/RAM-offloaded text encoding is a known
  community lever; a GGUF Q3 encoder variant exists "comfortably in
  12GB"; Lightricks issue #303 = no official smaller encoder yet)
- LTX23 audio VAE 365MB, video VAE 1.45GB, taeltx2_3 23.5MB,
  text_projection 2.3GB, distilled-1.1 dynamic LoRA 2.7GB
- NVFP4: ltx-2.3-22b-DEV-nvfp4 21.7GB, needs cu130 (present), native on
  Blackwell, open loading-failure report (Comfy issue #11864), DEV not
  distilled (more steps).

Pressure-test exactly these:

1. LANE DECISION GATE: define the M0 decision TABLE -- for each lane
   (L1 fp8_scaled 23.5GiB w/ ComfyUI block-swap/weight-streaming; L2
   GGUF per-quant; L3 NVFP4), the measured columns (NVML peak,
   wall-time/clip at 1472x832 x ~6s, quality eyeball vs 2B baseline)
   and the PASS criteria (NVML <= 14.5 sustained; wall time budget --
   propose one vs the ~6 min 2B opens; quality >= 2B). Given Q4_K_M =
   16.5GiB FILE, is full residency arithmetic already dead for Q4 on a
   14.5 ceiling, leaving Q3_K_S/Q3_K_M as the realistic full-resident
   GGUF picks unless ComfyUI-GGUF per-layer offload measures under the
   ceiling? Say so explicitly.
2. ENCODER PHASING: the 13.2GB fp8 gemma encoder + a 12-16GiB
   transformer CANNOT co-reside under 14.5. Specify the phase
   discipline the adapter must rely on (encode -> free encoder ->
   load transformer; ComfyUI model management vs explicit
   reclaim_idle_models), whether text-encode-on-CPU is the v1 default
   for this lane, and what the AS-3 lease wraps (the whole render or
   per-phase?). Check eng_humo/wrapper_bridge/gpu_residency in the
   grounding for the existing lease + BUG-291 reclaim mechanics and
   mirror them.
3. SYSTEM RAM: block-swap/offload eats system RAM (23.5GiB fp8 file
   streamed). The M0 sheet must record RAM size + peak commit; name the
   failure mode if RAM is short (paging -> wall-time blowup) and the
   gate (e.g. wall-time/clip ceiling catches it).
4. EPISODE TIME BUDGET: with ~2 talk beats + 1 music open per 30-word
   episode, propose the per-clip wall-time PASS bar and the episode
   -level budget delta vs today (the 2B opens already cost ~6 min);
   when does the lane become operationally unusable even if it renders
   (e.g. >15 min/clip)?
5. TWO-STAGE: base+latent-upscale doubles cost; v1 = base-only at
   1472x832 was locked in pass02. Confirm no hardware reason to revisit
   (or flag one).
6. FLUX CO-RESIDENCY: portraits render upstream (FLUX) before video;
   confirm sequential phases (image batch THEN video batch) mean no
   co-residency requirement, per the existing pipeline order; flag if
   the grounding suggests otherwise.
7. L3 NVFP4: given DEV-only weights (not distilled -> more steps),
   issue #11864, and 21.7GB file, judge whether L3 stays a stretch
   -goal column in the M0 sheet or gets cut from M0 to save probe time
   (the operator can add it later).

Rules: cite grounding or the judge-verified numbers above; arithmetic
must be explicit (GiB vs GB consistent); the 14.5 NVML ceiling and V-1
are non-negotiable; no new pip into cu130 (STOP rule). Output: numbered
MUST-FIX (file/section + what), SHOULD-CONSIDER, OPEN-QUESTIONS. Terse.
