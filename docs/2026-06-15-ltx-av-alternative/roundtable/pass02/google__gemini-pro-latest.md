<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core isolation and fallback strategies are sound, but the plan ignores a known, documented VAE crash and introduces a fundamental audio-sync contradiction.

MUST-FIX BEFORE BUILD:
1. [ARCH / TICKETS M0] VAE Decode Floor Crash. The plan targets "4-6 s clips" (100-150 frames at 25fps) but ignores the `_LTX_DECODE_FLOOR_DEFAULT = 169` documented in `eng_ltx_video.py`. The installed `VAEDecode` wrapper raises a tensor shape mismatch (256 vs 128) on smaller latents. Fix: M0 must explicitly probe the VAE decode floor at the new 384x216 / 512x288 resolutions. If the floor exists, `eng_ltx_av.py` must implement a `_ltx_frame_length` floor/raise logic identical to `eng_ltx_video.py`.
2. [BUGS/RISKS (C) / I/O contracts] Audio-Sync Contradiction. The document claims "clip length == beat audio" to prevent sync drift, but LTX enforces an `8n+1` frame rule. The Sprint Plan states `canonicalize` will "TRIM... or PAD-BY-LAST-FRAME" to match the integer target `T`. If the model generates lip-sync for an `8n+1` duration, and you arbitrarily trim or pad the tail to match a beat window, the sync will drift or freeze. Fix: The audio slice fed to the model for conditioning must be exactly padded/trimmed to the `8n+1` duration *before* generation, not just post-processed in `canonicalize`. 

SHOULD-FIX:
1. [Dims validator] Divergent Frame Math. The Sprint Plan explicitly dictates `next_8n1(n)` must "snap UP" and says "never copy" the legacy `eng_ltx_video` formula. However, `eng_ltx_video.py` *is* the current production code and it snaps DOWN (`((length - 1) // 8) * 8 + 1`). Diverging the fundamental frame math between Lane A and Lane B risks composite/window fill bugs. Fix: Reconcile `av_dims.py` to use the same `8n+1` snapping logic as `eng_ltx_video.py` or update both.
2. [CLAUDE'S PANELIST CRITIQUE] Audio Latent Separation. The critique asserts "the decode must take the VIDEO branch only (LTXVSeparateAVLatent...)". [ASSUMPTION] If LTX-2.3 A2V uses audio purely as *conditioning* (as stated elsewhere in the doc), it likely emits a standard video latent, not a joint AV latent. Fix: Verify the actual output of the LTX-2.3 A2V wrapper node in M0 before mandating a `LTXVSeparateAVLatent` step.

OPTIONAL / NICE-TO-HAVE:
- [WIRING (B)] The fallback chain `humo->humo_1.7B->latentsync->still_kenburns` is long. Consider short-circuiting directly to `latentsync` if `humo` is known to fail under the same VRAM pressure that would kill `ltx_av_talk`.

CUT THESE (over-engineering):
1. [Hardware] "NVFP4 dev 21.7 GB" and "L3 NVFP4 CUT from M0". Safe to cut entirely from the sprint plan. NVFP4 is a Blackwell-specific dev feature that exceeds the 16GB VRAM target anyway. Stick to the Distilled/GGUF paths.