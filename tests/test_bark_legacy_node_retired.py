"""Clean-break 1a guard: the legacy batch Bark node must stay retired.

Bark is now a per_line registry engine (nodes/_otr_audio_engines/eng_bark.py)
with its inference in nodes/_otr_bark_lib.py. The batch node module
(nodes/batch_bark_generator.py / class OTR_BatchBarkGenerator) was deleted in
lockstep with the per_line flip. This guard FAILS if either reappears -- as a
module file, an import target, a registration, or a frozen-manifest entry -- so
a future change cannot silently re-introduce the delegating batch path.

Behavioral pins only (file / import / registration / manifest) -- NOT a repo
grep, so the legitimate "no legacy INSTANCE on the active path" denylists in
the workflow guards (which name the type as a string on purpose) are untouched.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_batch_bark_module_file_is_gone():
    assert not (REPO / "nodes" / "batch_bark_generator.py").exists(), (
        "nodes/batch_bark_generator.py must stay deleted (clean-break 1a); "
        "Bark inference lives in nodes/_otr_bark_lib.py + eng_bark.py"
    )


def test_batch_bark_module_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nodes.batch_bark_generator")


def test_batch_bark_not_in_legacy_audio_nodes():
    from nodes._otr_legacy_manifest import LEGACY_AUDIO_NODES

    assert "OTR_BatchBarkGenerator" not in LEGACY_AUDIO_NODES


def test_batch_bark_not_in_committed_manifest():
    from nodes._otr_legacy_manifest import load_committed_manifest

    assert "OTR_BatchBarkGenerator" not in load_committed_manifest()["nodes"]


def test_init_has_no_batch_bark_registration():
    text = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "batch_bark_generator" not in text, (
        "__init__.py must not register or reference the deleted bark batch module"
    )
    assert "OTR_BatchBarkGenerator" not in text


def test_bark_is_per_line_registry_engine():
    """Positive guard: bark is the per_line char_voice engine, with no batch
    surface (no make_batch_node) and the preset-routing attr appendix A reads."""
    from nodes._otr_audio_engines import get_engine

    bark = get_engine("bark")
    assert getattr(bark, "interface", None) == "per_line"
    assert not hasattr(bark, "make_batch_node"), (
        "bark must not expose make_batch_node after the per_line flip"
    )
    assert getattr(bark, "voice_ref_field", None) == "voice_preset"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
