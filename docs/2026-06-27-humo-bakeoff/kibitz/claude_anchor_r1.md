# Claude anchor review -- r1 (code-grounded, written BEFORE reading the panel)

VERDICT: the problem statement is sound; the seed lever list is mostly right but
mis-weighted. The single highest-leverage UNKNOWN is whether the 14B's ~15.8 GB
nvidia-smi peak is REAL demand or allocator cache -- that is cheap to settle and
gates everything else.

## Grounded facts (CONFIRMED in the files)
- `eng_humo.py` `_node_candidates` lists `"unet": ("UNETLoader",)` ONLY -- there is NO
  GGUF load path for HuMo today. A quantized-14B idea is a real code change (add an
  `UnetLoaderGGUF` candidate + verify `WanHuMoImageToVideo` accepts a GGUF-loaded model
  with audio cross-attn), NOT a config flip. CONFIRMED.
- The 14B fp8 UNET (`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`) is ~14 GB
  resident; that is the weight FLOOR. The two-stage `OTR_BakeoffReclaim` evicts only the
  umt5 (~5 GB) + whisper encoders and only shaved 217 MB (RESULTS.md) -- because
  `wrapper_bridge._soft_free`/comfy already offloads the TE under pressure and the
  allocator cache refills the freed space. CONFIRMED (RESULTS + wrapper_bridge.py).
- distill cfg is 1.0 (NO CFG doubling) for the 14B; the 1.7B control used cfg 2.5
  (doubled batch) yet peaked LOWER (15089) than the 14B (15779-15996) -- so the 14B peak
  is WEIGHT-dominated, not activation-dominated. CONFIRMED.
- Output is always-silent; in-process wrapper graph; cold-import clean. Any idea must
  hold these. CONFIRMED.

## My ranked ideas
1. **Allocator-cache probe (CHEAPEST, do FIRST).** Re-run the two-stage leg with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and/or a `max_split_size_mb`) set
   in the boot env (`scripts/_otr_soak_server_launch.cmd` lane / the runner boot env).
   nvidia-smi "used" includes the cached reserve pool; if true demand is < 13.5 GB the
   "fit" problem partly evaporates and the fp8 14B may be promotable as-is. MEASURE:
   `run_humo_bakeoff.py` leg ii peak with the env on vs off. Risk: none (env only);
   it may not move the peak (then idea 2 is mandatory).
2. **Quantized HuMo-14B GGUF (the real weight-floor fix).** A Q4_K/Q5 HuMo-14B UNET
   (~7-9 GB) is the only lever that lowers the 14 GB floor. FIRST STEP: add an
   `UnetLoaderGGUF` candidate in `eng_humo._node_candidates` behind an env, source a
   HuMo-14B GGUF, and add a bakeoff leg. RISK: a HuMo-specific GGUF may not exist / the
   audio-cross-attn path may not survive GGUF; verify on `/object_info` + a 1-leg render.
   Highest impact on FIT; medium-high effort; research lane.
3. **Mouth realism -- input-still quality (cheap) + a no-LoRA quality-ceiling probe.**
   HuMo animates the ref portrait; a higher-res, face-forward still gives the mouth more
   to work with. Cheap, reuses the image pipeline. Separately, a bakeoff leg with
   `OTR_HUMO_LORA_NAME=none` + ~20-25 steps measures whether more compute fixes the
   mouth (expect higher VRAM/blue; it is a CEILING probe, not a ship config).
4. **Newer / dedicated lip-sync model (deeper mouth fix).** A mouth-region second pass
   (LatentSync/MuseTalk) or a full swap (Sonic/Hallo2/EchoMimic). Separate dep project;
   Blackwell sm_120 / torch 2.10 risk (LatentSync/MuseTalk were declined before on that).
   Highest mouth impact; highest effort/risk.

REJECT: raising cfg on the 14B distill to "sharpen" the mouth -- distill is trained for
cfg 1.0; higher cfg gives blue + artifacts, not detail (eng_humo de-blue history).
