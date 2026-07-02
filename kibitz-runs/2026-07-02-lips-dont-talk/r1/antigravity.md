VERDICT: yes-with-fixes. The plan identifies the right set of potential deltas, but needs critical semantic prompt fixes for bookends and resolution/VRAM calibration to succeed.

MUST-FIX BEFORE BUILD:
1. [## KNOWN DELTAS between working and failing] - Text prompt (Delta 3) lacks talking instruction for announcer/music bookends.
   - Defect: The positive motion prompts for announcer/music open/close/inter in `_LTX_MOTION_PROMPT_BY_ROLE` (defined in [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L546-L561)) do not mention talking, mouth movement, or lips opening/closing. Because the video model only animates what is explicitly prompted, the lips remain static even under audio conditioning.
   - Concrete Fix: Modify the announcer prompt in `_LTX_MOTION_PROMPT_BY_ROLE` to explicitly include talking description (e.g., "A vintage radio with its grille-cloth lips opening and closing in sync with the speech...").
2. [## KNOWN DELTAS between working and failing] - Base-pass resolution (Delta 1) is too low.
   - Defect: The default canvas size `OTR_LTX_AV_RENDER_CANVAS` of `512x288` (defined in [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L1347)) forces the base-pass resolution down to `256x144` (8x4.5 latent patches). The mouth region is sub-patch at this size, preventing the motion pass from detecting and articulating syllables [ASSUMPTION].
   - Concrete Fix: Increase default `OTR_LTX_AV_RENDER_CANVAS` to at least `832x480` (base pass `416x240`) or `1280x720` (base pass `640x360`) to provide sufficient spatial latent density for mouth articulation.
3. [## What the kibitz must deliver] - VRAM ceiling vs resolution conflict.
   - Defect: The 14.5GB VRAM ceiling on a 16GB GPU conflicts with high-resolution rendering. When running inside the full production pipeline (where other models may hold residual VRAM), decoding a `1280x720` canvas can cause out-of-memory errors.
   - Concrete Fix: Ensure that `_ltx_av_vram_reserve` (defined in [eng_ltx_av.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py#L143-L170)) is active during the A/B tests to evict models and force partial unet loading for the target resolution.
4. [## KNOWN DELTAS between working and failing] - Spatial dimension mismatch in latent mask concatenation (Delta 4).
   - Defect: In `_build_graph_ia2v` (defined in [eng_ltx_av.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py#L805-L819)), the `SolidMask` node is created using the full target `width` and `height` (e.g. 512x288) but applied to the audio latent which is concatenated with a halved base video latent, causing spatial mismatch.
   - Concrete Fix: Update `solidmask` dimensions in Stage A of the graph to match `base_w` and `base_h`.

SHOULD-FIX:
1. [## KNOWN DELTAS between working and failing] - Character beat framing lacks close-up face focus (Delta 7).
   - Defect: Character visual beats run on the wide landscape scene still `scene_character` (see [test_ltx_audio_in_routing.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_ltx_audio_in_routing.py#L99-L105)), rather than a close-up portrait. The character's face is small in a wide frame, so the mouth is too small to articulate.
   - Concrete Fix: Implement a framing override to crop/focus on the character face before calling `ltx_audio_in` [ASSUMPTION].
2. [## KNOWN DELTAS between working and failing] - Default negative prompt (Delta 6) contains motion-suppressing tokens.
   - Defect: `OTR_LTX_AV_NEGATIVE` contains `"static, frozen pose, still image"` (defined in [eng_ltx_av.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py#L66-L68)), which can conflict with audio-latent conditioning.
   - Concrete Fix: Add an A/B test case with the shorter canonical negative prompt.
3. [## KNOWN DELTAS between working and failing] - Frame rate discrepancy (Delta 2).
   - Defect: Production runs at 25fps while canonical runs at 24fps. Alignment of audio/video latents in LTX-2.3 depends on exact frame-rate scaling factors.
   - Concrete Fix: Run an A/B test parameter setting the target frame rate to 24fps.

OPTIONAL / NICE-TO-HAVE:
- Verify audio normalizations or RMS thresholds to ensure audio conditioning triggers sufficient onset response in the VAE.

CUT THESE (scope / over-engineering):
1. Mutating the frame count `8n+1` alignment logic (Delta 8) - Safe to cut because LTX-2.3 requires `8n+1` alignment to prevent downsampling crashes, and both paths already satisfy it.
