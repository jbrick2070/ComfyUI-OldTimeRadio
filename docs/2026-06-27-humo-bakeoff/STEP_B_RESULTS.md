# Step B -- 17B-GGUF feasibility (RESULTS, 2026-06-27)

Per the roundtable decision (passB_plan.md): no HuMo-14B GGUF exists, so test the on-disk
17B GGUF as the fits-the-ceiling, 14B-class candidate. Built a `humo_17b_gguf_q3` bakeoff
leg: 14B topology with the unet node swapped to `UnetLoaderGGUF{unet_name:HuMo-17b-Q3_K_M.gguf}`,
LoRA-free (the 14B-shaped lightx2v won't apply), 20 steps, cfg 1.0, ModelSamplingSD3 shift 8,
two-stage encoder evict. Loader-agnostic checkpoint/manifest. 33-frame min-smoke first.

## Q3 smoke result (the make-or-break tests)
- **Audio cross-attn on a GGUF-loaded UNET: WORKS.** `WanHuMoImageToVideo` accepted the
  `UnetLoaderGGUF` model object + the whisper audio conditioning and sampled cleanly (20/20),
  12.69 s/it (faster than the 14B fp8's 18.7), 33/33 frames, silent clip produced. The #1
  feasibility risk both panel agents flagged is CLEARED.
- **VRAM:** true-alloc (torch max_allocated, post-reset) = **11528 MB** (<= 13.5 GB target);
  NVML peak = **15662 MB** (> 14.5 GB hard). For the GGUF path the weights ARE torch-tracked
  (8.4 GB Q3 + ~3 GB activations = 11.5 GB), UNLIKE the fp8 14B (untracked, true-alloc read a
  bogus 3.4 GB). So the ~4.1 GB gap to NVML is allocator RESERVE/cache.
- **Colour:** B-R frame = **-7.25** (RED-shifted) vs 14B +9.9 (balanced) / 1.7B +21.9 (blue).
  Distinctly different look -> operator eyeball required.

## Reading
- The 17B Q3 GGUF's TRUE demand (11.5 GB) fits under 13.5 with headroom -- a real shift from
  the fp8 14B, whose ~14 GB weights are unavoidable. The blocker is now the ~4 GB NVML
  RESERVE, not the weights.
- Because the GGUF weights live in torch's caching allocator (the fp8 path did not), an
  allocator-config A/B (expandable_segments / max_split_size, or dropping --cuda-malloc's
  reserve) has a real chance of pulling the NVML peak under 14.5 here -- worth testing
  (it could NOT help the fp8 14B). KILL-GATE if it can't: Q3 stays > 14.5 NVML -> not
  resident-safe as-is.
- Q5 (11.86 GB) would ride higher; test only if Q3's allocator-trimmed NVML fits AND the
  operator wants more quality than Q3.

## Allocator A/B on Q3 (tested)
`expandable_segments:True` -> NVML 15678 MB vs 15662 MB baseline = NO CHANGE. The
`--cuda-malloc` (cudaMallocAsync) backend ignores expandable_segments, so the ~4 GB pool is
NOT reclaimable via that knob. The pool is non-essential (true live tensors = 11.5 GB);
cudaMallocAsync grows it because the 16 GB card has room and nothing else competes. Open
levers to constrain it: drop `--cuda-malloc` (native allocator + expandable_segments /
max_split_size) -- but that edits the shared launcher, a deeper experiment; OR rely on
cudaMallocAsync yielding the pool under real cross-engine pressure (the sentinel-style test).

## Bottom line
Q3 17B-GGUF renders the 14B-class look, FASTER than the 14B (12.7 vs 18.7 s/it), with TRUE
demand 11.5 GB (fits 13.5) -- a real candidate. The only blocker is the cudaMallocAsync pool
inflating NVML to ~15.7 GB; that is an allocator-config question, not a model-weight wall
(unlike the fp8 14B). Colour (B-R -7.25, red) is the quality question for the operator.

## Next
1. Operator eyeball: 3-way montage (14B vs 1.7B vs 17B-Q3-GGUF) -- colour + mouth.
2. Allocator A/B on the Q3 GGUF leg (try to bring NVML <= 14.5 since true-demand is 11.5 GB).
3. If Q3 fits + looks acceptable -> it is the promotable "14B-class look that fits"; else keep
   1.7B + de-blue. Nothing promoted; production untouched.
