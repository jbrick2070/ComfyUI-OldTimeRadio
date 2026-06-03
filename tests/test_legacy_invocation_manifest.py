"""R0a (e): the committed legacy invocation manifest must match the live
legacy-node widget surfaces. Drift = an audio-baseline-reset trigger
(legacy_manifest_sha), caught here at pytest time."""
from nodes._otr_legacy_manifest import (
    LEGACY_AUDIO_NODES,
    build_manifest,
    extract_widget_vector,
    load_committed_manifest,
    widgets_sha256,
)


def test_committed_manifest_matches_live_surface():
    assert build_manifest() == load_committed_manifest()


def test_all_legacy_audio_nodes_present():
    # Bark (1a) and Kokoro (1b) retired from the manifest in the audio
    # clean-break -- they are per_line registry engines now, not batch-delegated
    # legacy nodes. The remaining legacy audio nodes (MusicGen / AudioGen) stay
    # frozen here.
    m = load_committed_manifest()
    assert set(m["nodes"]) == set(LEGACY_AUDIO_NODES)


def test_each_node_sha_is_64_hex_and_consistent():
    m = load_committed_manifest()
    for entry in m["nodes"].values():
        assert len(entry["widgets_sha256"]) == 64
        int(entry["widgets_sha256"], 16)  # valid hex
        assert entry["widgets_sha256"] == widgets_sha256(entry["widgets"])


def test_extractor_skips_forceinput_keeps_widgets():
    it = {
        "required": {"wired": ("STRING", {"forceInput": True})},
        "optional": {"temp": ("FLOAT", {"default": 0.7}),
                     "mode": (["a", "b"], {})},
    }
    assert extract_widget_vector(it) == [
        {"name": "temp", "default": 0.7},
        {"name": "mode", "default": "a"},
    ]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
