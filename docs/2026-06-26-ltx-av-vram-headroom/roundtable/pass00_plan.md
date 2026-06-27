# LTX-AV VRAM Headroom Fix -- Plan (pass00)

Focused technical roundtable: why does the unified `ltx_audio_in` engine render
at 6.84 s/it one beat and 223 s/it the next on identical reported VRAM, and what
is the correct ROOT-CAUSE fix (no shim) on a 16 GB RTX 5080?

## Problem

16 GB RTX 5080 laptop, torch 2.10.0+cu130, Windows, ComfyUI Desktop. The unified
`ltx_audio_in` engine (LTX-2.3 22B, GGUF Q3_K_M) renders character/bookend beats
at wildly inconsistent, often catastrophic per-step speed. Same episode,
consecutive beats, all 512x288 8-step:

- b001 (`ltx_video`, announcer, NO audio): loaded PARTIALLY (5455 MB loaded +
  5081 MB offloaded) -> steady **9.58 s/it**
- b002 (`ltx_audio_in`): full load, 11196 MB usable -> **25.0 s/it**
- b003 (`ltx_audio_in`): full load, 12297 MB usable -> **6.84 s/it** (FAST)
- b004 (`ltx_audio_in`): full load, 12297 MB usable -> **223 s/it** (33x slower
  than b003 at byte-identical reported load)

One 512x288 8-step clip thus ranges 54 s to ~30 min. Goal: steady, predictable
speed with no quality loss and no new download if avoidable.

## Grounded evidence (verified against the live box + source)

- nvidia-smi DURING the slow b004 step: **15793 MiB used / 16303 total -> 211
  MiB free**. The card is maxed.
- The AV stack is ~24 GB cycled through a 16 GB card every beat: LTX-AV text
  encoder (`LTXAVTEModel_`) 11201 MB + GGUF unet (`LTXAV`, Q3_K_M) 10537 MB +
  `VideoVAE` 1384 MB + `AudioVAE` 693 MB. ComfyUI loads them sequentially
  (encode -> unet denoise -> tiled VAE decode).
- VRAM reclaim is ALREADY comprehensive and firing every beat:
  `free_otr_pipeline_residue()` runs pre-render + inter-beat; every image engine
  (z_image/flux/qwen/lumina) + HuMo + ltx_audio_in + the 3D mesh stage call
  `reclaim_idle_models` (detach only, NO `unload_all_models`). Log shows
  `free=14785 MB` / `free_gb_after=12.28` before each beat. **b004 got the SAME
  freed ~12.3 GB as the fast b003 and still crawled** -> more freeing is NOT the
  lever.
- `wrapper_bridge.reclaim_idle_models` carries the comment: "an aggressive
  free_memory here only fragments the 16 GB" -- the team already treats
  fragmentation as the enemy and deliberately avoids aggressive frees.
- `eng_ltx_av.py` sets NO load budget. The heavy load happens inside
  `wrapper_bridge.run_graph(graph, classes, free_after_use=True, keep={"unet",
  TERMINAL, "lora"/"modelsampling"})` via ComfyUI's own model_management loader
  (hence the `loaded completely; ... full load: True` lines). `free_after_use`
  already evicts the Gemma/umt5 encoder before the unet+decode peak; the unet is
  KEPT.
- Quants on disk: `ltx-2.3-22b-dev-Q3_K_M.gguf` (10.03 GB),
  `ltx-2.3-22b-dev-Q4_K_S.gguf` (12.22 GB). **No Q2_K present.**
- `ltx_audio_in` render canvas is clamped to 512x288 for ALL roles
  (render_driver ~L1115); the slowness is NOT a resolution mistake.

## Diagnosis

When ComfyUI sees ~12.3 GB usable it FULLY loads the 10537 MB unet, leaving only
~1.76 GB for activations. The audio-conditioned activation peak (audio
cross-attention adds footprint over plain `ltx_video`) tips past the remaining
VRAM and the NVIDIA driver silently spills to system RAM (sysmem fallback) ->
~33x slowdown. It is non-deterministic because it sits on the knife's edge (b003
fit, b004 did not). The plain `ltx_video` beat avoids this precisely because it
loaded PARTIALLY (offloaded ~5 GB) and kept activation headroom -> steady 9.58.

## Proposed fix (primary)

Force the AV unet to load with a fixed activation HEADROOM (partial load) instead
of to the brim, so the audio activation peak always fits in VRAM -- trading a
steady ~10-12 s/it for never spiking to 223. Concretely: before `run_graph` in
`eng_ltx_av.render_clip`, raise ComfyUI's reserved-inference VRAM (e.g.
`model_management` reserved-vram global / a `minimum_memory_required` hint) by
~2-3 GB for the duration of the AV render, then restore it in a finally block.
Scope strictly to `ltx_audio_in` beats so other engines are untouched.

## Secondary / complementary

- B. CUDA allocator config to cut fragmentation: `PYTORCH_CUDA_ALLOC_CONF` with
  Windows-supported knobs (`garbage_collection_threshold`, `max_split_size_mb`;
  `expandable_segments` ONLY if supported on torch 2.10 Windows), set in OTR
  prestartup BEFORE torch CUDA init. Complements A; unlikely to suffice alone
  since at 211 MB free the unet genuinely fills the card.
- D. Set NVIDIA "CUDA - Sysmem Fallback Policy" to prefer-no-fallback so a true
  overflow OOMs LOUD instead of silently crawling (surfaces regressions; not a
  speed fix by itself).

## Fallback (only if A+B insufficient)

- C. Q2_K unet: ~6.5 GB download (not on disk), frees ~3.5 GB headroom. Lossy Q2
  on a 22B audio-conditioned model -- the audio cross-attention is the most
  quant-sensitive part. Last resort.

## Invariants to guard (a "fix" that breaks one is rejected)

- No fallbacks: `render_shot` fails LOUD; `ltx_audio_in.fallback_engine=None`.
- Never `unload_all_models` (V-4/V-5); reclaim = detach only.
- The fix WIRED + ON in `workflows/otr_scifi_16gb_full.json` in the same change
  (no waiting to turn on) -- a pure runtime VRAM reserve may need no widget.
- Stay within the 14.5 GB measured ceiling guard
  (`_MC.assert_vram_within_ceiling`).
- UTF-8 no BOM, ASCII, SFW. Regression suite + Bug Bible green; commit AND push
  to `v2.0-alpha`.
- 100% local / no new paid deps.

## Open questions for the panel

1. Is forcing partial load via a ComfyUI reserved-VRAM bump the right
   root-cause lever, or is there a cleaner ComfyUI-native way to guarantee
   activation headroom for ONE engine's load?
2. On torch 2.10.0+cu130 Windows, which `PYTORCH_CUDA_ALLOC_CONF` knobs actually
   take effect (is `expandable_segments` supported there yet?), and would they
   ALONE create ~2 GB of headroom, or is partial-load mandatory?
3. Risk of the reserve approach: does forcing partial load reintroduce per-step
   layer-streaming overhead that is itself slow, and what is the headroom sweet
   spot (1.5 / 2 / 3 GB)?
4. Is the `AudioVAE` (693 MB) or the staged `VideoVAE` (1384 MB) co-residing
   during the denoise loop and stealing headroom -- should they be
   evicted/deferred until decode?
5. Anything else that makes b003 fast but b004 slow at byte-identical reported
   load besides activation sysmem-spill (fragmentation, a leaked ModelPatcher,
   allocator caching across beats)?
