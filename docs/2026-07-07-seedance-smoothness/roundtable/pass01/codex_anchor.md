VERDICT: yes-with-fixes. The plan has the right shape, but it must make the Seedance prompt conditioner concrete and must verify any provider knob before touching code.

MUST-FIX BEFORE BUILD:

1. [S4.1] The conditioner is currently a prose idea, not an implementable contract. Define a pure helper that only runs when `engine_id == "cloud_seedance_2"` after the final prompt is chosen in `nodes/_otr_video_engines/render_driver.py`. The helper should preserve the existing prompt source observability and add Seedance-specific observability, rather than replacing the normal prompt path.

2. [S4.1] The word substitutions are too ad hoc as written. They need a bounded mapping or a leading stabilizer clause, not broad free-text rewriting that could damage story content. Prefer appending/leading a compact stabilization clause and only soften the known sci-fi radio motion-register phrases from `nodes/visual_styles/sci_fi_radio.json`.

3. [S4.2] Duration changes are not build-ready. `CloudSeedance2Engine._duration_seconds()` already derives `round(target_frame_count / fps)` and clamps to `4..15` in `nodes/_otr_video_engines/eng_cloud_video.py`; global `OTR_CLOUD_SEEDANCE_DURATION` would flatten per-beat timing. Add trace logging before changing duration behavior.

4. [S5] Do not implement `temperature`. The grounded Seedance adapter sends only `model`, `prompt`, `resolution`, `ratio`, `duration`, `generate_audio`, `reference_images`, `reference_audios`, `seed`, and `watermark`; no checked code exposes a temperature field.

SHOULD-FIX:

1. [S4.3] Make the model A/B a manual or harness-level experiment first. `cloud_model_ids.py` defaults `cloud_seedance_2` to `Seedance 2.0 Fast`, and the adapter accepts `Seedance 2.0`; changing the default before an eyeball A/B would be premature.

2. [S4.5] The A/B harness should operate on an already-minted still/audio pair and must not rerun the writer, image phase, or audio phase. This keeps the comparison about Seedance prompt/model conditioning only.

3. [S4.4] Audio smoothing is plausible but lower priority. The current live logs show valid per-beat audio slices; altering audio should wait until prompt A/B proves insufficient.

OPTIONAL / NICE-TO-HAVE:

1. Add manifest fields for requested Seedance model, requested duration, actual duration, target frames, and prompt sha to make future eyeball reports less squishy.

CUT THESE:

1. [S4.4] Cut audio preprocessing from the first patch. It is a separate media transformation with more regression surface than prompt conditioning.

2. [S4.2] Cut duration snapping from the first patch. Instrument first; only snap after observed mismatch.
