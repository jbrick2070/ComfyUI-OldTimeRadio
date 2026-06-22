# GROUNDING -- wiring facts (verify claims against these)

## No audio-normalization widget in the workflow (verified)
`grep -i 'makeup|loudness|normaliz|LOUDNORM|dBFS|MAKEUP'` on
`workflows/otr_scifi_16gb_full.json` -> 0 matches. `OTR_MASTER_MAKEUP_DB` is read from `os.environ`
inside `_master_loudness`, NOT a workflow widget. Precedent for env-only audio knobs.

## tests/test_audio_byte_identical.py (key mechanics, verbatim-derived)
- Fixtures: `tests/fixtures/baseline_v1.5.wav` + `tests/fixtures/baseline_v1.5.sha256`.
- `_HAS_BASELINE = isfile(wav) and isfile(sha)`. Structural tests skip if absent, else check
  fixture integrity + that the workflow has the v2 audio nodes + `OTR_EpisodeAssembler`. These are
  CONTENT-INDEPENDENT (no audio is generated to run them).
- The byte-compare test:
    @pytest.mark.skipif(not _HAS_BASELINE, ...)
    @pytest.mark.skipif(not os.environ.get("OTR_REGRESSION_RUNTIME"), reason="...Requires ComfyUI + GPU.")
    def test_audio_byte_identical_to_baseline(self):
        from tests._run_baseline import run_episode_and_get_audio_bytes
        audio_bytes = run_episode_and_get_audio_bytes(FIXED_SEEDS)
        assert sha256_bytes(audio_bytes) == _load_expected_hash()
  -> SKIPS unless OTR_REGRESSION_RUNTIME is set AND fixtures exist. Default peak mode => identical bytes.
- Capture / RE-BASELINE entry point:
    python tests/test_audio_byte_identical.py --capture-baseline
  -> `_capture_baseline()` calls `tests/_run_baseline.run_episode_and_save_wav(FIXED_SEEDS, _BASELINE_WAV)`,
     then writes the SHA file. Comment: for a reproducible C7 run, start ComfyUI with
     OTR_CAST_SEED / OTR_STYLE_SEED (e.g. =42) so cast + style are deterministic.

## Call-site wiring
The 3 dialogue normalize calls (:747 announcer / :753 pre-rendered TTS / :775 inline-Bark) are INSIDE
`OTR_EpisodeAssembler` (node present + wired per the workflow). Editing them changes live production
behavior immediately (no dormant code). SFX call (:726) stays peak.
