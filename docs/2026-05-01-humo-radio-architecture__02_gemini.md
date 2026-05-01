# Round B -- Gemini (gemini-3-pro-preview) elapsed=32.5s

**1. State whether you AGREE, PARTIALLY AGREE, or DISAGREE**
I AGREE with the core recommendation (Option 1), as it correctly aligns the architecture with the model's actual capabilities while strictly protecting the 14.5 GB VRAM ceiling and C7 audio determinism.

**2. FACTUAL ERRORS in the ChatGPT answer**
*   **ComfyUI Internals Misunderstanding:** ChatGPT claims that because `ref_image` is VAE-encoded and appended to positive conditioning via `reference_latents`, this proves it is a "reference-guided talking-head model, not a generic I2V model." This is factually incorrect. In ComfyUI's `comfy_extras/nodes_wan.py`, appending `reference_latents` to the conditioning dictionary is the *standard* implementation for generic Wan2.1 Image-to-Video. The reason your model hallucinates faces is strictly due to the *model weights* (HuMo being finetuned on talking heads), not the conditioning injection mechanism. Your reading of the code was correct, but ChatGPT drew the wrong technical conclusion from it.
*   **Audio Concat Risk:** ChatGPT suggests testing if "ffmpeg concat introduces a resample." If you are using `-c:a copy`, ffmpeg physically cannot resample. If you aren't using `-c:a copy`, you are already violating C7. 

**3. IMPORTANT THAT WAS OMITTED**
*   **C7 Muxing Architecture:** ChatGPT completely missed *how* to guarantee C7 when mixing HuMo clips and static radio clips. You cannot concatenate individual Audio/Video chunks, as container padding will alter the audio bytes. You must concatenate a **video-only** stream (`-an`), and then do a single, final mux of the pristine master audio track over the concatenated video track using `-c:v copy -c:a copy`.
*   **Frame Math for Static Clips:** For `music_*` and `sfx` ffmpeg clips, audio durations rarely land on perfect 24fps frame boundaries (e.g., 1.05 seconds = 25.2 frames). You need a strict frame-rounding policy (e.g., `round(audio_duration * 24)`) for the static video generation to prevent cumulative A/V desync across a 30-minute episode.

**4. My own short recommendation**
*   **Commit to Option 1:** Add ANNOUNCER to the LLM cast schema, generating a host portrait via `batch_flux_render.py` and routing it through the existing `BUG-088` portrait resolver.
*   **Bypass HuMo for Non-Dialogue:** In `nodes/_otr_speaker_role.py:12-22`, explicitly route `music_*` and `sfx` to a new `StaticVideoRender` path. 
*   **Generate the Radio Still Once:** Render the radio still via FLUX at the start of the pipeline. Use an `ffmpeg -loop 1 -i radio.png -t <duration> -c:v libx264 -pix_fmt yuv420p` subprocess call to generate the non-dialogue video clips. This costs 0 VRAM.
*   **Enforce Video-Only Assembly:** Assemble the final video track from the HuMo and ffmpeg clips without audio, then mux the untouched v1.5 master audio file onto it at the very end.

**5. Items where I am uncertain and would want to verify**
*   **HuMo Resolution Constraints:** I am uncertain if your HuMo finetune expects a specific aspect ratio or resolution (e.g., 512x512 or 768x768) compared to the FLUX radio still. You may need to ensure `batch_flux_render.py` outputs the ANNOUNCER portrait at the exact resolution HuMo was trained on to avoid distortion.
*   **SFX Fallback Logic:** I am uncertain how your ledger handles overlapping audio (e.g., SFX playing *under* dialogue). If your architecture flattens everything into sequential clips, a 0.5s SFX interrupting dialogue will cause jarring video cuts. You need to verify if short `sfx` roles should just inherit/extend the video of the currently speaking character rather than cutting to the radio.
