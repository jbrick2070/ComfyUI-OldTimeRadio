# HuMo / LTX-2.3 local-video limits: evidence review

**Research date:** 2026-08-02  
**Target stack:** RTX 5080 Laptop, 16 GB VRAM, Windows, PyTorch 2.10 / CUDA 13, SageAttention or SDPA, no Flash Attention 2  
**Inputs reviewed:** `2026-08-02-DEEP-RESEARCH-BRIEF.md`, `2026-08-02-DEEP-RESEARCH-PROBLEM-STATEMENT.md`, and `ENGINE_MATRIX.md`

## Executive answer

1. **The 49-frame landscape cap is not supported by HuMo/Wan's architecture or by public measurements.** At equal frame count, 480×832 and 832×480 produce the same latent and attention tensor sizes. Wan uses square spatial patching, global attention, and equal-size spatial RoPE components. Swapping height and width changes positional values, not allocation sizes. Axis-asymmetric VAE tiling or backend workspaces can create a small practical difference, but no source or benchmark supports a landscape-only change from 177 to 49 frames.

2. **That does not prove even 97—let alone 177—fits this exact 14.5 GB ceiling.** No one has published paired peak-VRAM measurements for the exact Kijai scaled fp8 checkpoint on a 16 GB card at both orientations. The nearest 16 GB result is 832×480×125 with a Q4_K_M GGUF and 28 blocks swapped—not the requested fp8 checkpoint. The defensible change is to remove the *orientation-specific* rule, then establish one cap for both orientations with a short local A/B ladder. HuMo was trained at 97 frames and its authors warn that longer generation may degrade, so **97 is the first quality-supported target to test, not a memory-safe cap**. The production cap is the highest common value that passes the exact-stack memory test; public evidence cannot set it above the locally proven value.

3. **No fixed 100–200 ms HuMo lip-sync correction has been published.** Neither the paper, maintainers, Kijai issues, nor comparable-hardware reports endorse a pad length. The implementation does give the first latent an unusually zero-heavy audio window—five zero feature positions followed by audio frames 0–2—which plausibly explains a cold/static opening. It does not establish a five-frame or 200 ms correction.

4. **First determine whether the defect is onset-only or constant.** Leading-silence pre-roll followed by an equal trim can move an onset warm-up out of the visible clip, but it cannot repair a constant lag. For a constant lip delay, advance the conditioning features, or generate extra video and discard the delayed opening while retaining the original audio. A 3–6-frame sweep (120–240 ms at 25 fps) is justified by the observed defect; no individual value in that range is externally established.

5. **A bounded temporal VAE tile bounds that decoder's working set, not the complete pipeline peak.** Sampling still sees the full latent sequence unless it too is context-windowed. Moreover, Kijai's HuMo `tiled_decode` is spatially tiled and still hands all temporal latents to each tile; a separate, genuine temporal wrapper is required. Flat peaks through 81 frames do not license flat peaks through 177.

6. **HuMo's reference is native subject/identity conditioning, not a start-frame lock.** It is deliberately appended to the end of the video sequence to prevent first-frame continuation. Independent calls start from fresh noise and carry neither prior motion state nor phase. There is no supported continuous HuMo chaining method in the public implementation.

7. **No exact public 449-frame LTX-2.3 dev + external-audio/A2Vid peak exists.** Users have completed long joint text-to-audio-video LTX-2.3 clips on 16 GB, including a reported 30-second 832×480 run, but that is a different conditioning path and its precision and peak were not disclosed. The exact audio-in graph must be measured. The official video lattice is `F = 1 + 8k`; 449 is valid and the audio-conditioned variant uses the same video-frame rule.

## Evidence scale used below

| Grade | Meaning |
|---|---|
| **A** | Official paper, model repository, or implementation; architectural fact |
| **B** | Exact or near-exact firsthand measurement with disclosed hardware and settings |
| **C** | Adjacent firsthand result with a material mismatch or missing peak data |
| **D** | Source-derived inference; useful, but not a measurement |
| **Not published** | A targeted search found no public number or validated procedure |

## Decision table

| Decision | Recommendation | Evidence strength |
|---|---|---|
| Landscape HuMo cap | Delete the orientation-specific 49 rule; use the same tested cap for both orientations | **A** for equal allocation shapes; exact safe cap **not published** |
| Next common HuMo targets | Test 65/81/97; make the production cap the highest exact-stack memory pass. Treat 97 as a quality-supported checkpoint, not a memory guarantee | **A** for 97-frame training warning; memory limit **not published** |
| HuMo onset correction | Measure early/middle/late offset; test 3–6 frames; do not canonize 200 ms | Public magnitude **not published** |
| Onset-only defect | Add real-audio or silence pre-roll, round generation up to legal 4n+1, then discard the pre-roll and lattice surplus | **D**, supported by exact boundary code |
| Constant lip delay | Advance 25 Hz features by measured `d`, or generate the next legal length and retain decoded frames `[d:d+N]` | **D** timing algebra; measure `d` locally |
| HuMo continuity | Treat every split as JUMP; do not pass the last frame as if it were a lock | **A** |
| Tiled-VAE extrapolation | Do not infer 177 from flat 17/49/81 totals | **A/D** |
| LTX 449 | Benchmark exact graph; 449 is lattice-valid, but 832×480 feasibility under 14.5 GB is unproved | **A** lattice; peak **not published** |
| Identity | Reuse HuMo's exact native reference batch and prompts; do not bolt on SDXL/FLUX adapters | **A** compatibility; cross-cut gain unmeasured |
| Cut placement | Use transcript alignment plus VAD/energy and cut at silence midpoints | **A** tools; thresholds corpus-specific |

---

## 1. Published HuMo-14B fp8 VRAM measurements

### Finding

**No public benchmark reports peak VRAM for the exact** `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` **on a 16 GB card at either 480×832 or 832×480, and no public paired-orientation benchmark exists.** The missing internal “~15.9 GB” result therefore cannot be independently reproduced from public evidence.

The exact Kijai scaled checkpoint is [17,892,294,098 bytes](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/commit/8289e3743850e32cc475bfe38b1bafe019675a1b), about 16.66 GiB on disk. It necessarily needs offload/block swapping on a 16 GB card, but file size is not a VRAM peak.

### Closest public observations

| Result | Hardware/settings | What it proves | Comparability |
|---|---|---|---|
| [Q4_K_M GGUF at 832×480×125, 28 blocks swapped](https://www.reddit.com/r/comfyui/comments/1nikvwc/humo_lipsync_available_on_the_wan_video_wrapper/) | 16 GB VRAM; 64 GB RAM recommended | HuMo landscape above 49 frames can complete on a 16 GB consumer configuration | Same VRAM/canvas; **different quantization**, no GPU model or peak |
| [Q4_K_S workflow on a 4060 Ti 16 GB](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1250#issuecomment-3309454212) | 480×832, 35-block swap, 100-frame context windows | Long output can be assembled on 16 GB | Different quantization and context-window path; not full-sequence memory |
| [Native Comfy HuMo fp8 report](https://marcus-story.tistory.com/293) | RTX 4060 Ti 16 GB; roughly 5 seconds; reported around 12.5 GB | A different fp8 HuMo workflow reportedly ran within a 16 GB card | Different checkpoint; frame count, FPS, resolution, and measurement method insufficiently specified |
| [Kijai's official example workflow](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/6d05cc5cf99ff5140d1d036396db71e0ee3d1c3f/example_workflows/wanvideo_HuMo_example_01.json) | Exact Kijai checkpoint; landscape 1280×720×65; 20 blocks swapped | The wrapper/checkpoint accepts landscape lengths above 49 | Hardware and peak unstated; larger canvas |
| [Kijai maintainer reply](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1508) | Says the 832×480 example fits 24 GB easily and fits **within** 10 GB VRAM with full block swap | Residency is highly configurable and 832×480 is not structurally a 15.9 GB floor | Capacity claim, not a peak trace; frame count and hardware path are incomplete |
| [Official HuMo repository](https://github.com/Phantom-video/HuMo) | Wrapper integration reported runnable on RTX 3090 24 GB | General consumer-GPU feasibility | 24 GB, not 16 GB |

**Trust:** High that the exact number is not publicly available; medium for transferring the adjacent 16 GB successes.  
**Comparable hardware:** The Q4 and native-fp8 reports use 16 GB consumer cards, but none matches the exact checkpoint, graph, and peak definition.

### A provenance clue, not evidence

The unexplained “~15.9 GB” happens to match the [native Comfy HuMo fp8 checkpoint's](https://huggingface.co/Comfy-Org/HuMo_ComfyUI/blob/main/split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors) roughly 17.1 GB decimal file size expressed as GiB. The exact Kijai scaled file is larger. It is possible that a disk-weight figure was copied into a VRAM table, but there is no surviving document with which to prove that. Treat this only as a bookkeeping lead.

---

## 2. Does equal-pixel orientation change memory?

### Finding

**For these two canvases, the architecture-prescribed tensor element counts are orientation-invariant.** The working assumption is correct at the model-shape level; exact runtime peak equality still requires measurement because backend workspaces and allocator state are not architecture-prescribed.

HuMo constructs a latent target with temporal length `(F-1)//4 + 1 + refs` and spatial shape `H//8 × W//8`, then applies the Wan patch size `[1,2,2]`. The effective DiT grid is therefore:

| Canvas (W×H) | VAE grid | DiT spatial grid | Tokens per latent time |
|---|---:|---:|---:|
| 480×832 | 60×104 | 30×52 | 1,560 |
| 832×480 | 104×60 | 52×30 | 1,560 |

The [official HuMo tensor construction](https://github.com/Phantom-video/HuMo/blob/845f44736e21be93aa5d8cf406b6eb01af9bff67/humo/generate.py#L531-L548), [14B configuration](https://github.com/Phantom-video/HuMo/blob/845f44736e21be93aa5d8cf406b6eb01af9bff67/humo/configs/models/Wan_14B_I2V.yaml#L6-L17), and [Kijai conditioning code](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/HuMo/nodes.py#L201-L247) all depend on the product, not an orientation premium.

Wan's 3D RoPE does assign separate frequency values to time, height, and width, but the final allocation is reshaped to `f*h*w`; swapping `h` and `w` preserves its size. See the [official HuMo/Wan RoPE implementation](https://github.com/Phantom-video/HuMo/blob/845f44736e21be93aa5d8cf406b6eb01af9bff67/humo/models/wan_modules/model_humo.py#L38-L66) and [Kijai's implementation](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/wanvideo/modules/model.py#L215-L260). The configured attention window is `[-1,-1]`, meaning global rather than a non-square local window.

With one reference image:

| Pixel frames | Video latent times | + one ref | DiT tokens |
|---:|---:|---:|---:|
| 49 | 13 | 14 | 21,840 |
| 97 | 25 | 26 | 40,560 |
| 125 | 32 | 33 | 51,480 |
| 177 | 45 | 46 | 71,760 |

These counts are identical at both orientations. Moving from 49 to 177 is a real 3.29× token-length increase. Full attention compute grows approximately with the square of token length, while memory-efficient SDPA/SageAttention avoids storing the full quadratic attention matrix; Q/K/V, feed-forward activations, latents, and some workspaces still grow roughly linearly with sequence length.

### Mechanisms that can create a small practical delta

- **Asymmetric VAE tiling.** Kijai's default spatial tile is square, but its x/y strides differ. At these dimensions, rotation changes the number of edge tiles. That can alter runtime and a small workspace or fragmentation term; it does not change the maximum tile shape or justify a 3.6× frame-cap ratio. See the [tile controls](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/nodes.py#L2081-L2097).
- **Backend shape selection.** cuDNN, Triton/Inductor, or a fused kernel can pick different algorithms or workspace sizes for transposed shapes.
- **Compile caches and fragmentation.** Kijai warns that on Windows the first run of a new input size can use drastically more VRAM, particularly with compiled kernels; LoRA and block-swap residency also matter. See the [wrapper README warning](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/readme.md#L4-L23).
- **Padding granularity.** Equal pixels are not enough in the general case if one dimension crosses a patch/tile multiple. Here both 480 and 832 are divisible by the total 16× spatial factor, so that counterexample does not apply.

No published counterexample was found for this HuMo/Wan configuration.

**Trust:** Very high architectural confidence.  
**Comparable hardware:** No paired 16 GB benchmark; the conclusion is hardware-independent tensor arithmetic.

### Operational conclusion

The 49 cap should not be retained **because the output is landscape**. Replace it with a common, measured cap. Do not infer that 177 is safe merely because portrait was configured for 177; the portrait value was also not backed by a surviving peak trace.

---

## 3. HuMo's frame lattice and the 33-frame floor

### Finding

The real lattice is:

\[
F = 1 + 4k,\qquad k\ge 0.
\]

Wan-VAE compresses time by 4 after a causal first frame. Its encoder consumes temporal chunks `1,4,4,...`, giving `1 + (F-1)//4` latent times. See the [Wan-VAE source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L542) and [Wan technical report](https://arxiv.org/abs/2503.20314).

Both [official HuMo](https://github.com/Phantom-video/HuMo/blob/845f44736e21be93aa5d8cf406b6eb01af9bff67/humo/generate.py#L531-L548) and [Kijai's node](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/HuMo/nodes.py#L119-L125) round a request down with `4*((F-1)//4)+1`. Kijai does not impose a 33-frame architecture minimum.

**33 is a local policy or quality floor, not a model law.** Positive tensor-valid values begin 1, 5, 9, 13, … . Extremely short clips may be poor or operationally wasteful, but that is separate from validity.

**Trust:** A.  
**Comparable hardware:** Hardware-independent.

---

## 4. Does bounded temporal VAE decode make peak flat in frame count?

### Finding

**It can bound the VAE decoder's transient working set, but it does not make total pipeline peak independent of total frames.** Flat 17/49/81 measurements are real evidence for that tested interval and execution path only.

ComfyUI's [VAEDecodeTiled documentation](https://docs.comfy.org/built-in-nodes/VAEDecodeTiled) exposes `temporal_size` and `temporal_overlap` for video VAEs. A true streaming implementation processes at most a bounded temporal tile plus overlap, so per-tile convolution activations can remain roughly constant while runtime and host/output storage increase.

There are three important qualifications:

1. **Kijai HuMo's own tiled decoder is spatial, not automatically temporal.** It passes all `T` latents into each spatial tile; the decoder loops over time and concatenates results before moving the decoded tile to CPU. See [`tiled_decode`](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/wanvideo/wan_video_vae.py#L1238-L1268) and the [decode loop](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/wanvideo/wan_video_vae.py#L1131-L1167). A `vae_temporal=16` setting only proves bounded decoding if the production path actually wraps latent time into independent overlapping chunks.
2. **The sampler still receives the full sequence.** HuMo creates full-length latent, mask, zero-frame, RoPE, video-attention, and audio-conditioning tensors. Unless sampling also uses a context-window scheme, its growing allocations remain.
3. **The pipeline peak is the maximum phase peak.** A flat VAE phase can cease to be the maximum when a longer sampler sequence overtakes it. Output tensors, allocator reserves, and full decoded tiles may also scale depending on placement.

Therefore, flat-to-81 does **not** license flat-to-177. It licenses a hypothesis worth testing at 97, 125, 149, and 177.

**Trust:** A for implementation behavior; D for extrapolation.  
**Comparable hardware:** The user's own figures are the best hardware-comparable evidence, but only through 81 frames.

---

## 5. Which allocation dominates?

### Finding

There is no universal dominant phase; it depends on block swapping and whether temporal decode is genuine. No public source gives a phase-by-phase peak decomposition for this exact HuMo workflow.

| Phase | Principal allocations | Scaling with frame count | Likely role here |
|---|---|---|---|
| DiT sampling | Resident block weights, Q/K/V and feed-forward activations, latent/mask/RoPE tensors, attention workspace | Video activations grow with latent `T×H×W`; global-attention compute grows faster | Most likely growing peak at long `F` when decode is genuinely bounded |
| VAE decode | Decoder weights and convolution activations; decoded tile/output accumulation | Untiled grows strongly with `F`; true temporal tiling bounds the tile working set | Can dominate short/untiled paths; may look flat when bounded |
| Whisper/audio | Whisper weights plus 25 Hz embeddings and audio cross-attention K/V | Feature tensor grows linearly with time; only 16 audio tokens per latent time in HuMo | Usually smaller than 1,560 video tokens per latent time at 480p; weight residency still matters if not unloaded |
| Model/LoRA residency | fp8 model blocks, dequantization buffers, LoRA weights | Mostly independent of `F`; controlled by block swap/offload | The exact checkpoint is 16.66 GiB on disk, so residency policy is decisive |

At 480p, each latent time has 1,560 video tokens versus 16 HuMo audio tokens. That makes the growing audio embedding an implausible explanation for a large 49→177 peak increase. The sampler's video path or an incompletely temporal VAE is the more likely source.

**Trust:** D, derived from exact implementations; peak decomposition **not published**.  
**Comparable hardware:** None.

---

## 6. `ref_image` semantics and continuous chaining

### Finding

**The reference is a subject/appearance condition and explicitly not a first-frame lock.** “Identity hint” is directionally correct, although the reference can also influence pose, clothing, composition, and background.

HuMo concatenates VAE reference latents to the **end** of the noisy video sequence specifically so the model will not interpret them as a starting frame. Self-attention extracts and propagates subject information instead. This is stated in [§3.3 of the paper](https://arxiv.org/html/2509.08519v1#S3.SS3) and implemented in [official generation code](https://github.com/Phantom-video/HuMo/blob/845f44736e21be93aa5d8cf406b6eb01af9bff67/humo/generate.py#L508-L614) and [Kijai's reference node](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/HuMo/nodes.py#L210-L245).

Every call samples a new video from fresh noise. There is no input for the prior terminal latent, motion state, or locked first frame. Passing segment N's last frame as segment N+1's `ref_image` merely provides another soft reference; it does not make frame 0 equal that image.

Official maintainers confirm that current HuMo lacks full-size first-frame I2V and warn that borrowed long-generation/context methods can create temporal inconsistency: [issue #31](https://github.com/Phantom-video/HuMo/issues/31#issuecomment-3344709568) and [issue #32](https://github.com/Phantom-video/HuMo/issues/32#issuecomment-3347323981). Kijai described an experimental HuMo/InfiniteTalk continuation attempt as poor rather than supported: [issue #1214](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1214#issuecomment-3293310786). The official longer-generation checkpoint and movie-level guide remain unreleased/TODO in the public repository.

**There is no supported continuous HuMo chaining method.** JUMP is the correct continuity contract.

**Trust:** A.  
**Comparable hardware:** Hardware-independent.

---

## 7. HuMo lip-sync onset error

### What is documented

HuMo resamples Whisper features from 50 Hz to a 25 Hz conditioning clock. Audio at 16 kHz is counted in 640-sample, 40 ms units. The first video latent receives an eight-position conditioning block containing **five zero feature positions followed by audio positions 0, 1, and 2**; later latents receive ordinary local windows. See the [official audio processor](https://github.com/Phantom-video/HuMo/blob/845f44736e21be93aa5d8cf406b6eb01af9bff67/humo/utils/audio_processor_whisper.py#L87-L157).

Kijai reproduces that function in [HuMo/nodes.py](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/HuMo/nodes.py#L19-L51) and invokes it with `frame0_idx=0` in the [sampler](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/nodes_sampler.py#L424-L438). There is no public timeline-offset control.

This boundary condition is a plausible explanation for weak/static startup motion. The five zero slots are part of a receptive-field window, however; **they do not prove a fixed five-frame or 200 ms output lag**.

The [HuMo paper](https://arxiv.org/html/2509.08519v1#S4) reports aggregate Sync-C and Sync-D scores. It does not report onset offset. No HuMo model card, official/Kijai issue, maintainer reply, or 16 GB benchmark found in this search documents a stable 100–200 ms defect, an accepted silence pad, or a correction magnitude.

The Lightx2v distillation LoRA is also a possible confounder: Kijai reported that it was not fully compatible with HuMo and involved tradeoffs in [HuMo issue #6](https://github.com/Phantom-video/HuMo/issues/6#issuecomment-3285746946) and a [later comment](https://github.com/Phantom-video/HuMo/issues/6#issuecomment-3288608809). The observed error cannot be safely attributed to base HuMo without an A/B against that LoRA.

### The accepted magnitude

**Not published.** A universal value for Whisper-conditioned face models does not exist; the model's feature clock, temporal compression, audio window convention, and mux path all matter.

### Correct diagnosis and fix

Measure AV offset in speech-active early, middle, and late windows. The original local [SyncNet implementation](https://github.com/joonson/syncnet_python) reports AV offset in frames and was designed to remove temporal lags, but it expects a muxed 25 fps / 16 kHz video. Create a diagnostic CFR-25 mux with zero-based, verified stream timestamps; record SyncNet confidence, reject low-confidence windows, and first validate the reported sign with an artificially shifted control clip. A literal one-second onset crop may be too short if it lacks several clear mouth-motion events.

#### Case A: onset-only cold start

If the middle and late windows are aligned, move speech away from the special first latent:

1. Prepend `p` frames of conditioning history. For internal segment boundaries, use the real preceding audio rather than silence; at the beginning of an episode, use silence.
2. For `N` desired visible frames and leading trim `r=p`, request the next legal HuMo length

   \[
   G=1+4\left\lceil\frac{N+r-1}{4}\right\rceil.
   \]

   Pad the conditioning tail through `G`.
3. Decode and discard the first `r` generated frames, then discard the final `G-r-N` surplus frames.
4. Mux the untouched `N`-frame audio.

This creates no frozen or repeated content. The discarded frames are extra pre-roll; every retained audio instant still receives newly generated video.

#### Case B: constant lips-late offset

If lips lag by `d` through the clip, leading silence plus an equal trim does not fix it. Algebraically, prepadding by `p` and trimming `p` leaves the same `d` delay.

Use one of these instead:

- **Conditioning offset:** for video time `t`, feed the audio feature at `t+d`: advance/left-shift the 25 Hz feature tensor by `d`, zero-pad its tail, and keep `frame0_idx=0`. This preserves HuMo's trained special first-window pattern while moving the source timeline. Setting positive `frame0_idx` is a different experiment: it also replaces normally missing left-boundary context with real features, so it tests both offset and boundary warm-up at once.
- **Video advance:** condition normally, use the legal-length formula above with `r=d`, tail-pad audio through `G`, discard the first `d` generated video frames and the final surplus, then mux the original `N` audio frames.
- **Audio delay:** prepend `d` to final playback audio and generate the required extra tail video. This changes episode timing and is less attractive here.

For combined startup pre-roll `p` and constant delay `d` without a feature offset, set `r=p+d`, generate legal `G`, discard the leading `r` frames, and discard the final surplus. Arbitrary 3–6-frame trims are valid after decode even though the generated request itself must remain on HuMo's 4n+1 lattice.

At 25 fps:

| Frames | Time |
|---:|---:|
| 3 | 120 ms |
| 4 | 160 ms |
| 5 | 200 ms |
| 6 | 240 ms |

The observed 100–200 ms defect justifies a **3, 4, 5, 6-frame sweep**. The five zero positions only motivate testing the startup boundary: the audio projection collapses that receptive-field block, so they do not map one-to-one to five output frames. No candidate is a published answer.

A true temporal offset should remain constant in seconds and should not depend on clip length. Keep correction arithmetic and trimming on HuMo's 25 Hz source clock, before any downstream frame-rate conversion; do not recompute the model-side offset from the eventual display FPS. If offset grows from beginning to end, investigate resampling, container timestamps, or an actual frame-rate mismatch rather than adding a fixed pad.

**Trust:** A for the boundary code; high confidence that no public magnitude exists; D for the proposed correction until locally measured.  
**Comparable hardware:** No comparable public measurement. Sync itself should not depend on VRAM capacity.

---

## 8. Restart pose and motion phase

### Finding

HuMo does **not** guarantee that a new call starts at the literal reference pose—the design intentionally avoids start-frame continuation. It also cannot preserve motion phase across calls.

Every segment:

- starts from fresh noise;
- reuses soft reference, text, and audio conditions only;
- resets the audio window to its zero-padded left boundary;
- carries no previous head/body velocity, pose trajectory, or latent state.

Thus “always returns to the exact reference pixels” is unsupported, but a repeated reference/prompt/seed can make independently regenerated opening layouts similar enough to produce the observed snapback. A fixed seed offers repeatability, not continuity, and may make the reset more visibly repetitive. Audio conditioning influences motion inside the new call; it cannot convey the prior clip's physical phase.

No public study quantifies HuMo seam displacement, reset-pose frequency, or the benefit of any seed policy across edited clips.

**Trust:** A for absence of state; D/observational for visual severity.  
**Comparable hardware:** Hardware-independent.

---

## 9. LTX-2.3 dev + audio VAE at about 449 frames

### Finding

**No public source gives peak VRAM for the exact combination** `ltx-2.3-22b-dev` + audio-conditioned/A2Vid + about 449 frames + a 16 GB consumer GPU. Any numeric answer would be invented.

The full bf16 [dev checkpoint is 46.1 GB](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors), while the official [fp8 file is 29.1 GB](https://huggingface.co/Lightricks/LTX-2.3-fp8/blob/main/ltx-2.3-22b-dev-fp8.safetensors). Both require substantial offload on 16 GB before activations and audio/video VAEs are considered.

### Closest evidence

| Public result | What is comparable | Missing or different |
|---|---|---|
| [30-second prompted dialogue at 832×480 on RTX 4070 Ti Super 16 GB / 64 GB RAM](https://www.reddit.com/r/StableDiffusion/comments/1rm4hk3/ltx_23_can_do_30_second_spongebob_clips_on_4070/) | Same VRAM class/canvas; author says full 22B model | Joint T2AV, not supplied-audio A2Vid; exact frame count, precision, dev/distilled variant, and peak absent |
| [RTX 4080 16 GB, 10 s at 720p](https://www.reddit.com/r/StableDiffusion/comments/1rmhp7i/ltx_23_workflows_working_on_my_4080_16gb_vram/) | 16 GB consumer completion | Q4_K-S distilled and much shorter |
| [16 GB Q4 GGUF joint audio/video run](https://www.localainews.co/tutorials/how-to-run/run-ltx-2-3-gguf-under-16gb/) | 4070 Ti Super 16 GB; synchronized generated A/V | T2AV rather than supplied-audio A2Vid; different quantization/canvas/length and no peak |
| [Official discussion: dev-fp8, 481 frames, 1280×720](https://huggingface.co/Lightricks/LTX-2.3/discussions/16) | Near length and fp8 dev model | RTX 5090 32 GB; no peak |
| [LTX-2 paper](https://arxiv.org/html/2601.03233v1) | Official architecture and benchmark | Benchmarks 121 frames/720p on H100; no VRAM table for this case |

These establish that offloaded 16 GB LTX-2.3 paths exist, not that the external-audio A2Vid graph stays under 14.5 GB. A2Vid encodes/freezes supplied audio and can have a different peak from joint T2AV generation.

### Source-derived risk comparison

LTX's video VAE uses temporal/spatial compression 8×32×32. For a valid 449-frame clip, latent time is `1+(449-1)/8 = 57`.

| Clip | Latent grid | Video tokens |
|---|---:|---:|
| 449 at 512×288 | 57×16×9 | 8,208 |
| 169 at 832×480 | 22×26×15 | 8,580 |
| 449 at 832×480 | 57×26×15 | 22,230 |

Therefore 449 at 512×288 has slightly fewer video tokens than 169 at 832×480, while 449 at 832×480 has 2.59× as many. This is a risk-ranking inference, **not a VRAM prediction**: LTX-2.3 also has a 5B audio stream, audio latents, cross-attention, VAE phases, and implementation-dependent offload.

The official one-stage path requires dimensions divisible by 32. The official two-stage A2Vid helper calls `assert_resolution(..., is_two_stage=True)` and requires divisibility by 64: see [the helper](https://github.com/Lightricks/LTX-2/blob/9377758131b1ffde4b7f766804590a6617bf2ab9/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py#L321-L331) and [A2Vid call](https://github.com/Lightricks/LTX-2/blob/9377758131b1ffde4b7f766804590a6617bf2ab9/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py#L132-L136). A ComfyUI graph may use a different one-stage or normalized route; 832×480 and 512×288 are not valid direct inputs to that official **two-stage** helper.

**Recommendation:** Probe the exact 449-frame graph first at its lowest supported one-stage canvas. If using the official two-stage path, choose a /64 canvas such as 576×320 or 768×448. Only then attempt 832×480. Record stage peaks; do not extrapolate a number from the public 16 GB successes.

**Trust:** High that the exact peak is not published; C for neighboring 16 GB feasibility; D for token-risk comparison.  
**Comparable hardware:** Same-capacity consumer results exist, but no exact graph/peak.

---

## 10. LTX-2.3 length lattice

### Finding

The exact video contract is:

\[
F=1+8k.
\]

The official core maps pixel frames to latent frames as `F' = 1+(F-1)/8` and requires `(F-1)%8==0`: [LTX core README](https://github.com/Lightricks/LTX-2/blob/9377758131b1ffde4b7f766804590a6617bf2ab9/packages/ltx-core/README.md#L330-L335). The trainer documentation explicitly lists 1, 9, 17, …: [troubleshooting guide](https://github.com/Lightricks/LTX-2/blob/9377758131b1ffde4b7f766804590a6617bf2ab9/packages/ltx-trainer/docs/troubleshooting.md#L147-L153).

“9 + 8k” describes the same moving-video set if a local UI excludes the one-frame case; it is not the mathematical model rule. `449 = 1 + 8×56`, so 449 is valid.

The audio-conditioned implementation uses the same `num_frames` to construct its video shapes; audio latent duration is derived from `num_frames/frame_rate`. See the official [A2Vid stage-one construction](https://github.com/Lightricks/LTX-2/blob/9377758131b1ffde4b7f766804590a6617bf2ab9/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py#L150-L166) and [stage two](https://github.com/Lightricks/LTX-2/blob/9377758131b1ffde4b7f766804590a6617bf2ab9/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py#L214-L218).

**Trust:** A.  
**Comparable hardware:** Hardware-independent.

---

## 11. Identity preservation across independent cuts

### Finding

The current premise needs one correction: if the shared portrait is connected to HuMo's `ref_image` / `reference_images`, the pipeline **already has native identity conditioning**. The paper reports strong within-clip subject-consistency metrics and supports one or multiple references, but it does not benchmark identity drift across independently sampled clips joined in an editor.

### Best supported practice for HuMo

1. Use the same high-quality canonical face reference on every segment.
2. Keep crop, orientation, color treatment, identity/appearance/costume clauses, negative prompt, reference batch, and HuMo conditioning scales stable. Allow action, camera, and shot wording—and usually the seed—to vary so the identity controls do not force the same opening layout.
3. If memory permits, experimentally test a small batch of complementary clean references (for example, frontal face plus a consistent three-quarter view or costume reference). HuMo and Kijai support a reference batch, but no public study proves that multiple face views improve cross-cut identity over one strong face.
4. QC face similarity across segment samples and rerender outliers. A local ArcFace-style embedding can rank identity consistency, although the paper's published metrics are within-clip, not seam-specific.
5. Do not treat fixed seed as identity conditioning. It controls randomness and may repeat the same opening pose, worsening the reset signature.

The native mechanism is documented in [HuMo §3.3](https://arxiv.org/html/2509.08519v1#S3.SS3); Kijai encodes the complete reference image batch and appends its latents in [HuMo/nodes.py](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/088128b224242e110d3906c6750e9a3a348a659b/HuMo/nodes.py#L210-L245).

### Why the named adapters are not established fixes

- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) publishes adapters for SD1.5/SDXL-family backbones.
- [InstantID](https://github.com/InstantID/InstantID) is an SDXL identity pipeline.
- [PuLID](https://github.com/ToTheBeginning/PuLID) publishes SDXL/FLUX variants.

They are not checkpoint-agnostic modules. HuMo would need a Wan/HuMo-specific trained projection and injection implementation. An unofficial [IPAdapterWAN](https://github.com/kaaskoek232/IPAdapterWAN) exists, but no HuMo compatibility result or cross-segment identity benchmark was found. It should be treated as research, not current best practice.

A subject LoRA trained for the actual Wan/HuMo backbone could improve identity if native reference conditioning remains insufficient, but no public HuMo cross-cut comparison establishes the gain. It would still not transfer motion phase.

**Trust:** A for native conditioning and adapter incompatibility; cross-cut best-method comparison **not published**.  
**Comparable hardware:** Identity mechanism is hardware-independent; extra references/LoRA residency must be measured locally.

---

## 12. Silence-aware local cut placement

### Finding

Yes. This is an established, fully local scheduling problem; it does not need a model feature or service.

Available local signals:

- [FFmpeg `silencedetect`](https://ffmpeg.org/ffmpeg-filters.html#silencedetect) emits silence start/end/duration from configurable noise and duration thresholds. Its defaults (-60 dB, 2 s) are too strict/long for ordinary dialogue and should be tuned.
- [Silero VAD](https://github.com/snakers4/silero-vad) is small, MIT-licensed, offline, supports 8/16 kHz, and returns speech timestamps quickly on CPU.
- Because the transcript is known, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/alignment_example.html) can supply word and phone boundaries. This is safer than energy alone, which can mistake unvoiced consonants for silence.

### Recommended scheduler

1. Prefer TTS-native word/phoneme timings if the engine emits them; otherwise force-align transcript and waveform.
2. Detect non-speech/low-energy intervals with Silero or FFmpeg.
3. For each segment, search backward within roughly the final 0.5–1.0 seconds before the model cap.
4. Choose the 25 fps frame boundary nearest the midpoint of the widest non-speech interval that also lies between aligned words; split audio at that same timestamp. This keeps constant-frame-rate video and audio aligned while moving at most half a frame (20 ms) from the silence midpoint.
5. If none exists, use the last word boundary before the cap. A phone boundary inside a word is only a degraded emergency fallback, not the normal policy.
6. Render up to the next legal model length and trim the decoded surplus to the selected visible frame boundary. The visible cut need not lie on the model's 4n+1 or 8n+1 generation grid.
7. For a whole beat, dynamic programming can minimize a cost such as: number of cuts + speech energy at seam + distance from cap, subject to a minimum useful segment duration.
8. Give each HuMo successor real preceding-audio pre-roll for conditioning and discard the corresponding generated pre-roll frames. This improves acoustic boundary context without duplicating visible video or audio.

No HuMo-specific VAD threshold, search window, or seam-cost benchmark has been published. Calibrate thresholds to this TTS/noise floor.

**Trust:** A for the local tools; D for scheduler weights; HuMo-specific tuning **not published**.  
**Comparable hardware:** CPU/local and independent of GPU capacity.

---

## Exact local tests that settle the remaining unknowns

### A. Paired HuMo orientation/cap test

Use the production checkpoint, Whisper, Lightx2v LoRA, reference, audio, seed, sampler, steps, CFG schedule, attention backend, block swap, and VAE settings unchanged.

Test this ladder in both orientations:

`49 → 65 → 81 → 97 → 125 → 149 → 177`

For each `(orientation, frames)` pair:

1. Start from a clean process or deliberately record both cold and warm behavior. Treat cold and warm ceilings as separate results: a warm run inherits compile caches and allocator reservations unless its baseline is subtracted and recorded.
2. Run the shape twice. Kijai's Windows compile-cache warning makes first and second runs separate facts. Decide separately whether production pre-warms every supported shape; if not, the operational cap must survive cold execution.
3. Before each measured phase, call `torch.cuda.synchronize()`, record current allocated/reserved baselines, then call `torch.cuda.reset_peak_memory_stats()`. Synchronize again before reading `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
4. Poll per-process GPU memory continuously with high-frequency NVML sampling. A point-in-time `nvidia-smi` reading can miss a brief peak.
5. Record separate peaks after Whisper, DiT sampling, and VAE decode. A single end-of-run figure cannot identify the scaling phase.
6. Alternate orientation order or restart between orientations to avoid allocator-order bias.
7. Define the 14.5 ceiling in exact bytes or MiB/GiB before testing; do not mix decimal GB with binary GiB. Stop when either the required cold ceiling or production-representative warm ceiling is breached.
8. Inspect quality separately. The official 97-frame training warning means a memory pass at 125/149/177 is not a quality pass.

Expected falsification criterion: if landscape consistently exceeds portrait by a meaningful amount at identical `F`, inspect VAE tile count, kernel selection, and compile/allocator traces. The model shapes themselves predict equality.

### B. HuMo sync calibration

Use clear frontal-face lines with several sharp bilabial/plosive onsets. Generate at least three seeds on the exact fp8 + Lightx2v production stack under these conditions:

1. uncorrected production stack;
2. 3/4/5/6-frame real-audio or silence pre-roll with equal head trim and legal-length tail trim;
3. 3/4/5/6-frame feature-tensor advances with tail padding while retaining `frame0_idx=0`;
4. optional positive-`frame0_idx` sweeps as a separate boundary-context experiment.

Use matched no-LoRA renders only as an attribution control; do not substitute them for the production correction sweep. Create a diagnostic zero-based CFR-25/16-kHz mux, then estimate SyncNet offset and confidence in speech-active early, middle, and late windows. Validate SyncNet's sign on a deliberately shifted control first.

- Early-only error that disappears after pre-roll supports the boundary-warm-up hypothesis.
- Equal early/middle/late error supports a constant conditioning or mux offset.
- A growing error supports rate/timestamp mismatch.
- An error present only in the matched LoRA A/B identifies the distillation path rather than HuMo base behavior.

Choose the median offset across lines/seeds, not a single attractive clip.

### C. LTX-2.3 449-frame allocation test

1. Verify whether the production graph is one-stage or official two-stage; enforce its /32 or /64 canvas rule.
2. Run 449 frames at the lowest supported canvas first.
3. Record peaks for model loading, audio VAE/conditioning, sampling, video VAE, and final decode separately.
4. Repeat at the intended talking canvas only if the lower-canvas run leaves enough margin.
5. Keep a host-RAM and pagefile trace as well: offloading a 29.1–46.1 GB checkpoint can fail outside VRAM.

## Publicly unanswered questions, stated plainly

- Exact peak VRAM for Kijai's scaled HuMo fp8 on a 16 GB GPU at either 480×832 or 832×480: **not published**.
- Paired orientation peak delta for that checkpoint: **not published**.
- Exact safe HuMo frame cap under a 14.5 GB RTX 5080 Laptop ceiling: **not published**.
- A documented HuMo 100–200 ms onset defect and accepted pad/offset value: **not published**.
- HuMo restart-pose frequency or a supported stateful chaining method: **not published**; public architecture provides no stateful chaining.
- Cross-cut identity benchmark comparing one/multiple references, seed policy, LoRA, or adapters for HuMo: **not published**.
- Exact peak for LTX-2.3 dev + audio VAE + 449 frames on 16 GB, at any stated canvas: **not published**.

Those are the GPU hours worth spending. The orientation question itself is no longer an architectural unknown; only the exact operational ceiling is.
