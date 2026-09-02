VERDICT: build-ready as-is? no. The plan prioritizes complex image-injection, audio-scaling, and temporal experimentation over fixing the root cause of lost visual style, while simultaneously breaking the non-negotiable 8 GB floor by gating AnimateDiff on 11 GB Klein stills.

MUST-FIX BEFORE BUILD:
1. [§7 Prerequisite, §2, §6, §8] Breaking the 8 GB floor by gating AnimateDiff on Klein stills.
   Defect: Declaring `accepts_still = True` removes the lane from `_NO_STILL_VIDEO_ENGINES` in `scripts/otr_provision.py:1546`, forcing an 11 GB `flux2_klein` download and execution on 8 GB cards (RTX 4060). On 8 GB, Klein takes 20-40s per still (or 42 minutes if memory thrashes), breaking the 8 GB floor invariant and violating §8's rule that 4060 behavior must not regress.
   Concrete fix: Do not make `accepts_still = True` global for the lane. Either gate still-conditioning behind a 16 GB profile flag so 8 GB remains strictly text-to-video, or create a lightweight native SD1.5 still generation path for 8 GB before enabling stills-in.

2. [§7 E9/E14, §2, §3] Prompt and seed loss on disk ranked as Arm 9/14 instead of Day 0 blocker.
   Defect: `Ledger.save()` and `_merge_with_disk` (`nodes/production_ledger.py:1592-1598`) drop `video.shots[]` on disk re-saves because `TOP_PRESERVE` lacks `"video"`. Only a truncated 100-character prefix in ephemeral logs survives. The document ranks fixing this at Arm 9/14 while admitting in §7:373 that prompt persistence is a strict precondition for evaluating any visual A/B test.
   Concrete fix: Promote E9 to Day 0 prerequisite. Add `"video"` to `TOP_PRESERVE` and `Ledger.data` in `nodes/production_ledger.py`, and ensure `OTR_VideoRenderBatch` stamps `prompt_sha8`, `request_seed`, and the full composed positive/negative prompt into each clip receipt.

3. [§5B, §7 E0/E10/E11] Inversion of the root cause of lost visual style.
   Defect: Base SD1.5 (`v1-5-pruned-emaonly-fp16.safetensors`) cannot draw anime, cartoon, engraving, or origami from a 2-word prompt prefix, especially when `v3_sd15_adapter.ckpt` at strength 1.0 re-injects photographic video training grime (`nodes/_otr_video_engines/eng_ghost_signal_official.py:103-113`). The plan buries per-style checkpoints, style LoRAs, and adapter adjustment as Arm 11, while naively hoping img2img stills (E1/E2) will transfer style.
   Concrete fix: Promote E11 and adapter tuning (E0) to Arm 1. Allow the domain adapter strength to scale down to 0.0 on non-photographic styles (`anime`, `storybook_engraving`, `paper_origami`), and wire optional style LoRAs or SD1.5 checkpoints defined in the visual style pack schema directly into `CheckpointLoaderSimple` and `LoraLoaderModelOnly`.

4. [§4, §7 E1/E2] Destructive 2.875x downscaling of stills obliterating latent style fidelity.
   Defect: Scene stills are minted at 1472x832 (`nodes/otr_meta_brief_image_prompt.py:480`). Downscaling 2.875x down to 512x288 turns sharp cel-shaded lines and fine engraving details into muddy blur, which SD1.5 VAE encodes into a 64x36 latent tensor. At denoise 0.75-0.9, SD1.5 hallucinations overwrite the style; at denoise 0.6, identical repeated latents create static "breathing stills". [ASSUMPTION: High-frequency style detail will be lost during latent downsampling, mirroring historical findings in `BUG_LOG_2026-06.md` line 109].
   Concrete fix: Configure AnimateDiff's `still_plan` to request native aspect/resolution stills (e.g. 512x288 or 832x480 via `still_dims_for_aspect("wide")`), bypassing the destructive 1472x832 downsampling step entirely.

5. [§2, §5B, §7 E3/E12] Unbudgeted prompt expansion overflowing the SD1.5 77-token CLIP window.
   Defect: `compose_ghost_prompt_v2` (`nodes/_otr_video_engines/ghost_signal_prompt.py:874`) already consumes 65-75 CLIP tokens. E3 and E12 propose appending setting descriptions, lighting, palette, and scene instructions. In standard SD1.5 `CLIPTextEncode` (`nodes/_otr_video_engines/eng_ghost_signal.py:764`), tokens past 77 are silently truncated or diluted across attention chunks, causing either the mode law, motif, or style to be completely ignored.
   Concrete fix: Establish a hard token budget across prompt components: max 20 tokens for style/era, 20 tokens for motif/identity, 20 tokens for drawable beat, and 15 tokens for framing/mode law. Forbid unparsed ledger string dumps into the prompt.

SHOULD-FIX:
1. [§7 E0] Adapter sweep must include 0.0 and low values.
   Defect: Sweeping only 0.5, 0.75, and 1.0 leaves the photographic video grime active on every run.
   Concrete fix: Sweep 0.0, 0.25, 0.5, and 1.0 against non-photographic styles (`anime`, `storybook_engraving`) to verify if 0.0 immediately restores style rendering on base SD1.5.

2. [§7 Ranking] Reorder the 14 experiment arms by actual dependency and impact.
   Defect: The 14 arms are ranked haphazardly, placing temporal filters and audio scaling ahead of prompt persistence and style restoration.
   Concrete fix: Move Arm 9 (Prompt persistence) to Step 0; move Arm 11/E0 (Adapter sweep + style LoRA/checkpoint) to Step 1; move Arm 12 (Style pack prompt restoration within 77 tokens) to Step 2; demote Arms 1/2 (Still-seeded latent / injection) to Step 3/4.

3. [§5B, §7 E13] Activate style-aware roll immediately.
   Defect: Rolling unsupported styles on a lane that cannot physically render them guarantees failure on screen.
   Concrete fix: Update the style roll logic for `animatediff15_v3_haunted_video` to weight or exclude styles that SD1.5 cannot draw without dedicated checkpoints. Zero code cost, zero download.

4. [§3, §7 E4] Schema violation in proposed speech-energy conditioning.
   Defect: `VideoRequest` enforces `extra="forbid"` (`schemas.py:136`). Adding external energy curves violates schema invariants unless handled engine-locally.
   Concrete fix: If speech-energy scaling is ever tested, calculate RMS strictly inside `eng_ghost_signal.py` from `audio_ref.path` without modifying `VideoRequest`.

OPTIONAL / NICE-TO-HAVE:
- [§7 E5] ContextRef long-beat anchoring: Test `ADE_ContextExtras_ContextRef` (mode First) solely on beats exceeding 64 frames (4+ context windows), verifying that VRAM remains below the 14.5 GiB gate on the RTX 5080.

CUT THESE (scope / over-engineering):
1. [§7 E4] Speech-energy motion scaling: Cut. Audio volume/RMS does not correlate with meaningful dramatic camera or character movement in radio drama. Dynamic multival scaling on ADE causes severe visual distortion, and it solves neither style nor identity.
2. [§7 E6] FreeInit for temporal coherence: Cut. Multiplies render time by 2x to 3x (pushing an 11-clip run from 29 min to ~75 min and overnight runs past 4 hours) for marginal smoothing on a v3 motion module that already eliminated v2 flicker.
3. [§7 E7] SparseCtrl RGB keyframes: Cut. Requires installing `ComfyUI-Advanced-ControlNet` (violating OTR's self-contained packaging architecture), downloading unvetted external weights, and introducing heavy ControlNet VRAM overhead that risks the 14.5 GiB gate and breaks 8 GB entirely.
4. [§7 E8] CameraCtrl Gen2 loader swap: Cut. Requires abandoning the proven Gen1 loader (`ADE_AnimateDiffLoaderGen1`) for Gen2 nodes, downloading external camera pose models, and rewriting the core graph topology in violation of §8.

UNANSWERED BUILDER QUESTIONS:
1. How will the 8 GB profile generate stills without gating on the 11 GB Klein bundle or stalling execution?
2. Why generate 1472x832 stills for a 512x288 video engine rather than minting at native canvas resolution?
3. What exact token budget allocation will prevent SD1.5 CLIP silent truncation past 77 tokens when style and setting text are added?
4. Does setting the v3 domain adapter to 0.0 immediately restore visual style under base SD1.5 without any graph alterations?
