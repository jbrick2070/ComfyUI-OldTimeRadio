"""Clean-break 1c guard: the batch-delegation layer + legacy manifest are gone.

After 1a/1b/1c every audio engine is self-contained (per_line / clip). The shared
batch-delegation dispatch (_delegate_batch / frozen_batch_widgets / _manifest_path),
the theme node's batch path, the legacy musicgen + audiogen nodes, and the legacy
invocation manifest were all retired. This guard fails if any reappears.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("rel", [
    "nodes/musicgen_theme.py",
    "nodes/batch_audiogen_generator.py",
    "nodes/_otr_legacy_manifest.py",
    "config/legacy_invocation_manifest.json",
])
def test_retired_files_are_gone(rel):
    assert not (REPO / rel).exists(), f"{rel} must stay deleted (clean-break 1c)"


@pytest.mark.parametrize("mod", [
    "nodes.musicgen_theme",
    "nodes.batch_audiogen_generator",
    "nodes._otr_legacy_manifest",
])
def test_retired_modules_not_importable(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


def test_batch_dispatch_symbols_gone_from_voice_common():
    src = (REPO / "nodes" / "_otr_voice_node_common.py").read_text(encoding="utf-8")
    for sym in ("_delegate_batch", "frozen_batch_widgets", "_manifest_path",
                "make_batch_node", 'interface == "batch"'):
        assert sym not in src, f"{sym!r} must be gone from the shared dispatch"


def test_theme_node_has_no_batch_path():
    src = (REPO / "nodes" / "stable_audio_theme.py").read_text(encoding="utf-8")
    assert "_delegate_batch" not in src
    assert 'interface == "batch"' not in src
    assert "frozen_batch_widgets" not in src


def test_musicgen_is_clip_engine():
    from nodes._otr_audio_engines import get_engine

    mg = get_engine("musicgen")
    assert mg.interface == "clip"
    assert not hasattr(mg, "make_batch_node")


def test_no_registered_audio_engine_is_batch():
    from nodes._otr_audio_engines import get_engine

    for name in ("bark", "chatterbox", "indextts2", "kokoro", "musicgen",
                 "stable_audio_music"):
        assert get_engine(name).interface in ("per_line", "clip"), name


def test_legacy_nodes_unregistered_in_init():
    text = (REPO / "__init__.py").read_text(encoding="utf-8")
    for needle in ("musicgen_theme", "OTR_MusicGenTheme",
                   "batch_audiogen_generator", "OTR_BatchAudioGenGenerator"):
        assert needle not in text, f"{needle!r} must be gone from __init__.py"


def test_musicgen_guidance_matches_profile_ssot():
    """eng_musicgen.guidance_scale must equal the music_musicgen_v1 profile
    default so the hardcoded constant cannot drift from the curated SSOT (D5)."""
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_engine_profiles import load_resolver

    mg = get_engine("musicgen")
    prof = load_resolver().profile_for("music", "musicgen")
    assert prof is not None
    assert float(mg.guidance_scale) == float(prof.default_params["guidance_scale"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
