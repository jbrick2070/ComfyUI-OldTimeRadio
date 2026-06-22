# GROUNDING EXCERPTS -- real code from nodes/scene_sequencer.py (verbatim).
# Verify claims against THESE. Do not invent APIs you cannot see here.

# ----- nodes/scene_sequencer.py:93 -- PER-CLIP normalization (the seam to change) -----
def _normalize_clip(clip_np, target_peak=0.85):
    """Normalize a 1-D float32 clip to a target peak amplitude.

    Bark outputs vary wildly in volume between characters and takes.
    This brings every dialogue clip to a consistent level so the Commander
    doesn't whisper while the Pilot screams.

    Uses peak normalization (not RMS) to preserve dynamics within each clip
    while matching overall loudness across clips.
    """
    peak = np.abs(clip_np).max()
    if peak < 1e-6:
        return clip_np  # silence - don't amplify noise floor
    return (clip_np * (target_peak / peak)).astype(np.float32)


# ----- nodes/scene_sequencer.py:109 -- EPISODE-level master (applied ONCE in assemble) -----
def _master_loudness(waveform, ceiling_dbfs: float = -1.0, makeup_db=None):
    """Final episode loudness master: makeup gain + tanh soft limiter + peak.

    The legacy master stage peak-normalized to -1.0 dBFS only. Speech has a
    high crest factor, so peak-only leaves the episode perceptually quiet.
    This lifts perceived loudness a touch above the streaming norm and is
    peak-SAFE: a tanh soft-knee limiter rounds the gained peaks, then the true
    peak is trimmed back to ``ceiling_dbfs``. Fully deterministic (no RNG).

    ``makeup_db`` (default env OTR_MASTER_MAKEUP_DB, else 4.0; clamped 0..12)
    sets the boost. 0 disables the limiter -> pure peak-normalize to the
    ceiling (legacy behavior). Returns ``(waveform, makeup_db_used)``.
    """
    import os
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = waveform.abs().max()
    if float(peak) < 1e-8:
        return waveform, 0.0
    # Normalize to the ceiling first so the limiter sees a known full scale.
    waveform = waveform * (ceiling / peak)
    if makeup_db is None:
        makeup_db = float(os.environ.get("OTR_MASTER_MAKEUP_DB", "4.0"))
    makeup_db = max(0.0, min(12.0, float(makeup_db)))
    if makeup_db > 0.0:
        g = 10.0 ** (makeup_db / 20.0)
        denom = float(torch.tanh(torch.tensor(g)))
        if denom > 1e-8:
            waveform = torch.tanh(waveform * g) / denom
        peak2 = waveform.abs().max()
        if float(peak2) > 1e-8:
            waveform = waveform * (ceiling / peak2)
    return waveform, makeup_db


# ----- nodes/scene_sequencer.py:1116 -- where the master is applied in assemble() -----
# (segments = opening theme + main scene mix + closing theme, equal-power sqrt
#  crossfades, THEN one master pass:)
#     episode_waveform, _makeup_db = _master_loudness(episode_waveform)
#
# NOTE: the per-CLIP _normalize_clip runs upstream during scene/dialogue assembly;
# _master_loudness runs ONCE here at the episode level. The proposed change targets
# the per-segment stage (perceived-loudness target) and must retune the master makeup
# so the two gains do not stack.
#
# GATE: tests/ has test_audio_byte_identical -- any change to the per-segment gain
# changes output bytes and is a DELIBERATE, operator-gated golden re-baseline.
# Invariant I-11: post-engine audio DSP is CPU-only (no CUDA), for determinism.
