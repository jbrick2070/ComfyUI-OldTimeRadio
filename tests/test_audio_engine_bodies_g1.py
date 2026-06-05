"""G1 engine-body headless contract (Wave 3 prep).

The three model engines now carry inference bodies implemented to their
documented assumed_call. Headless (engine libs absent) every body must fail
closed with a clear "not installed" RuntimeError -- never a silent pass, and
never a heavy import at module top (C-5). ``supports_external_generator`` stays
False until the F GPU pilot verifies the external-generator binding, so the
engines remain disqualified from bit_exact mode until then.
"""
import importlib
import importlib.util

import pytest


def _engine(modname, clsname):
    mod = importlib.import_module("nodes._otr_audio_engines." + modname)
    return getattr(mod, clsname)()


def _absent(libmod):
    return importlib.util.find_spec(libmod) is None


def test_supported_kwargs_drops_unknown_and_keeps_known():
    from nodes._otr_audio_engines.base import supported_kwargs

    def fn(a, b=1, c=2):
        return (a, b, c)

    assert supported_kwargs(fn, b=10, zzz=99) == {"b": 10}


def test_supported_kwargs_passes_through_var_keyword():
    from nodes._otr_audio_engines.base import supported_kwargs

    def fn(**kw):
        return kw

    assert supported_kwargs(fn, anything=1, two=2) == {"anything": 1, "two": 2}


@pytest.mark.parametrize(
    "modname,clsname,libmod",
    [
        ("eng_chatterbox", "ChatterboxEngine", "chatterbox"),
    ],
)
def test_voice_engine_fails_closed_without_lib(modname, clsname, libmod):
    if not _absent(libmod):
        pytest.skip(libmod + " present -- real inference needs the GPU pilot")
    eng = _engine(modname, clsname)
    with pytest.raises(RuntimeError, match="not installed"):
        eng.generate_voice("hello there", None, None, 7)


def test_indextts2_pathb_fails_closed_without_worker(monkeypatch, tmp_path):
    # IndexTTS2 is Path B (isolated subprocess worker), so "fail closed" means a
    # missing isolated venv / worker / weights -- NOT an in-process import (it
    # never imports the conflicting library in ComfyUI's process). Point the venv
    # at a nonexistent path so load() fails fast with a named error and never
    # spawns a real worker.
    monkeypatch.setenv("OTR_INDEXTTS2_VENV", str(tmp_path / "nope" / "python.exe"))
    eng = _engine("eng_indextts2", "IndexTTS2Engine")
    with pytest.raises(RuntimeError, match="not installed"):
        eng.generate_voice("hello there", None, None, 7)


def test_music_engine_fails_closed_without_lib():
    if not _absent("stable_audio_tools"):
        pytest.skip("stable_audio_tools present -- real inference needs the GPU pilot")
    eng = _engine("eng_stable_audio", "StableAudioMusicEngine")
    with pytest.raises(RuntimeError, match="not installed"):
        eng.generate_clip("warm ambient pad", 4.0, 7)


@pytest.mark.parametrize(
    "modname,clsname",
    [
        ("eng_chatterbox", "ChatterboxEngine"),
        ("eng_indextts2", "IndexTTS2Engine"),
        ("eng_stable_audio", "StableAudioMusicEngine"),
    ],
)
def test_engine_not_external_generator_ready_until_pilot(modname, clsname):
    from nodes._otr_audio_engines.base import engine_supports_external_generator

    assert engine_supports_external_generator(_engine(modname, clsname)) is False
