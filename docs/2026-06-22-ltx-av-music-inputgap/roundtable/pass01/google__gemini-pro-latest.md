<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Option 1 is catastrophic for audio-conditioned models, and the existing shot-timing fallback is reading fields that do not exist in the `ShotRow` schema.

MUST-FIX BEFORE BUILD:
1. [Candidate fixes / Option 1] Risk: Slicing the WHOLE master audio for a missing-timing beat will feed a massive (e.g., 30-minute) WAV file into `ltx_av_music`'s audio encoder to condition a ~2-second clip. This will