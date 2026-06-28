# Step A -- honest VRAM meter + allocator A/B (RESULTS, 2026-06-27)

Built + ran the final.md Step A: two new sibling nodes (`OTR_BakeoffVramReset` latent
passthrough = `torch.cuda.reset_peak_memory_stats()` right before the sampler;
`OTR_BakeoffVramProbe` image passthrough = log `max_memory_allocated`/`max_memory_reserved`
after decode), spliced into every bakeoff leg; `--alloc-conf` boot knob for the A/B.
Unit tests 8/8; build + runner `--dry-validate` PASS (new nodes registered + spliced).

## A/B on leg ii (14B two-stage, 49f @832x480, fixed still+audio+seed)

| run | PYTORCH_CUDA_ALLOC_CONF | torch max_allocated | NVML peak | s/it | promotable |
|---|---|---|---|---|---|
| A baseline | (default) | **3444 MB** | **15863 MB** | 18.75 | no |
| B expandable | expandable_segments:True | **3444 MB** | **15874 MB** | 19.0 | no |

## Conclusion -- the "it's just allocator cache" hypothesis is REJECTED
- NVML is **stable at ~15.86 GB** across baseline and the allocator change (delta 11 MB).
  The allocator config does NOT reclaim it, so the ~15.86 GB is **real card occupancy
  (the 14B fp8 weights), not fragmentation/cache** an allocator knob could recover.
- `torch.cuda.max_memory_allocated` = 3444 MB **under-reports** here: it misses the ~14 GB
  fp8 UNET weights because ComfyUI loads them outside torch's caching-allocator stats (the
  `--cuda-malloc` / model_management path). So torch-allocated is NOT a valid fit meter for
  this stack -- **NVML is the truth.** (This also refines Gemini's r2 NVML caveat: the trap
  was the inverse of expected -- torch under-counts, NVML is right.)

## Verdict (unchanged, now stronger)
The cheap probe did its job: it RULED OUT the free win. The 14B genuinely needs ~15.86 GB
regardless of allocator config -> it does NOT fit <= 13.5 GB (or even <= 14.5 GB), and
encoder-evict + allocator tuning cannot recover enough. **The only remaining lever to get
the operator-preferred 14B LOOK under the ceiling is a smaller-WEIGHT 14B = the quantized
HuMo-14B GGUF (Step B)**, which lowers the actual ~14 GB weight floor. Mouth/teeth realism
remains a separate track (Step C / model swap).

Next per final.md: Step B GGUF feasibility gate (UnetLoaderGGUF / unet_name, mirror
eng_wan_i2v; /object_info; 33f min-smoke for audio cross-attn + LoRA-merge), only if a
HuMo-14B GGUF exists. Nothing promoted; production untouched.
