"""Clean-break 1b guard: the legacy Kokoro announcer node must stay retired.

Kokoro is now a per_line registry engine (nodes/_otr_audio_engines/eng_kokoro.py).
The OTR_KokoroAnnouncer batch node + module were deleted in lockstep with the
per_line flip. This guard FAILS if either reappears (module file, import,
registration, or frozen-manifest entry). Behavioral pins only -- not a repo grep
-- so the workflow-JSON denylists that name the type as a string stay valid.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_kokoro_announcer_module_file_is_gone():
    assert not (REPO / "nodes" / "kokoro_announcer.py").exists(), (
        "nodes/kokoro_announcer.py must stay deleted (clean-break 1b); the kokoro "
        "announcer engine lives in nodes/_otr_audio_engines/eng_kokoro.py"
    )


def test_kokoro_announcer_module_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nodes.kokoro_announcer")


def test_kokoro_announcer_not_in_legacy_audio_nodes():
    from nodes._otr_legacy_manifest import LEGACY_AUDIO_NODES

    assert "OTR_KokoroAnnouncer" not in LEGACY_AUDIO_NODES


def test_kokoro_announcer_not_in_committed_manifest():
    from nodes._otr_legacy_manifest import load_committed_manifest

    assert "OTR_KokoroAnnouncer" not in load_committed_manifest()["nodes"]


def test_init_has_no_kokoro_announcer_registration():
    text = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "kokoro_announcer" not in text, (
        "__init__.py must not register or reference the deleted kokoro module"
    )
    assert "OTR_KokoroAnnouncer" not in text


def test_kokoro_is_per_line_registry_engine():
    from nodes._otr_audio_engines import get_engine

    kokoro = get_engine("kokoro")
    assert getattr(kokoro, "interface", None) == "per_line"
    assert not hasattr(kokoro, "make_batch_node"), (
        "kokoro must not expose make_batch_node after the per_line flip"
    )
    assert getattr(kokoro, "voice_ref_field", None) == "voice_ref_id"


def test_kokoro_speed_matches_profile_ssot():
    """The hardcoded eng_kokoro.speed must equal the announcer_kokoro_v1 profile
    default so the constant cannot silently drift from the curated SSOT (D5)."""
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_engine_profiles import load_resolver

    kokoro = get_engine("kokoro")
    prof = load_resolver().profile_for("announcer_voice", "kokoro")
    assert prof is not None
    assert float(kokoro.speed) == float(prof.default_params["speed"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
