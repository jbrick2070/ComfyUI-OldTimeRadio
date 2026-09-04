"""ffmpeg resolution: explicit -> OTR_FFMPEG -> PATH, with PATH LEFT AVAILABLE.

WHY THIS FILE EXISTS (2026-09-04, kibitz runpod-found-fixes r1-r2). Six
resolver copies did ``if ffmpeg and (which(ffmpeg) or isfile(ffmpeg)): return
ffmpeg`` first -- so the caller's own signature default, the bare string
``"ffmpeg"``, won on any box with ffmpeg on PATH and ``OTR_FFMPEG`` was never
consulted. Measured live on the 5080 before the fix: every one of them handed
back the PATH binary with the pin set to a different file. The existing tests
(``test_wave1_boot_and_ffmpeg_resolution.py``) stub ``shutil.which`` to
``None`` and therefore run under the ONE condition where the defect cannot
show. Every test here keeps a PATH ffmpeg in play.

The rule under test is ``ffprobe``'s: a bare name with no directory is not a
choice; anything carrying a directory, or a non-default name, is.
"""
from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest

from nodes._otr_shared import ffmpeg as ffm


@pytest.fixture()
def box(tmp_path, monkeypatch):
    """A box with ffmpeg on PATH, a DIFFERENT existing file for the pin, and
    a third file an operator might type into a widget. The pin is NOT set
    here -- each test decides."""
    path_bin = tmp_path / "path" / "ffmpeg.exe"
    env_bin = tmp_path / "env" / "ffmpeg.exe"
    explicit = tmp_path / "explicit" / "ffmpeg-7.1.exe"
    for p in (path_bin, env_bin, explicit):
        p.parent.mkdir()
        p.write_bytes(b"")

    def which_with_one_ffmpeg(name):
        # PATH knows exactly one ffmpeg and nothing else.
        return str(path_bin) if name in ("ffmpeg", "ffmpeg.exe") else None

    monkeypatch.setattr(shutil, "which", which_with_one_ffmpeg)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(ffm, "_WINDOWS_INSTALL_CANDIDATES", ())
    return SimpleNamespace(path=str(path_bin), env=str(env_bin),
                           explicit=str(explicit), tmp=tmp_path)


# --------------------------------------------------------------------------- #
# The owner
# --------------------------------------------------------------------------- #
def test_an_explicit_existing_path_wins_over_env_and_path(box, monkeypatch):
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    assert ffm.resolve_ffmpeg(box.explicit) == box.explicit


def test_the_bug_a_bare_default_does_not_beat_the_pin(box, monkeypatch):
    """THE defect: PATH has an ffmpeg, the operator pinned a different one,
    the caller passed its own default literal. The pin wins."""
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    assert ffm.resolve_ffmpeg("ffmpeg") == box.env


def test_a_bare_dot_exe_is_also_not_a_choice(box, monkeypatch):
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    assert ffm.resolve_ffmpeg("ffmpeg.exe") == box.env


def test_a_non_default_bare_name_is_a_choice(box, monkeypatch):
    """``ffmpeg-7.1`` carries no directory but is not the default literal, so
    it IS a preference -- and it resolves through PATH like any name."""
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    seen = []

    def which(name):
        seen.append(name)
        return box.explicit if name == "ffmpeg-7.1" else None

    monkeypatch.setattr(shutil, "which", which)
    assert ffm.resolve_ffmpeg("ffmpeg-7.1") == box.explicit
    assert seen == ["ffmpeg-7.1"]


def test_an_explicit_path_that_does_not_exist_falls_to_the_pin(box, monkeypatch):
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    assert ffm.resolve_ffmpeg(str(box.tmp / "gone" / "ffmpeg.exe")) == box.env


def test_an_explicit_path_that_does_not_exist_and_no_pin_falls_to_path(box):
    assert ffm.resolve_ffmpeg(str(box.tmp / "gone" / "ffmpeg.exe")) == box.path


def test_a_bare_default_with_no_pin_is_path(box):
    assert ffm.resolve_ffmpeg("ffmpeg") == box.path
    assert ffm.resolve_ffmpeg(None) == box.path
    assert ffm.resolve_ffmpeg("") == box.path


def test_a_pin_spelled_as_the_bare_name_means_use_path(box, monkeypatch):
    """An operator may pin ``OTR_FFMPEG=ffmpeg`` to say "the PATH one". That
    is a pin that resolves, not a pin to ignore."""
    monkeypatch.setenv("OTR_FFMPEG", "ffmpeg")
    assert ffm.resolve_ffmpeg("ffmpeg") == box.path


def test_a_pin_that_does_not_resolve_is_skipped_not_returned(box, monkeypatch):
    monkeypatch.setenv("OTR_FFMPEG", str(box.tmp / "nowhere" / "ffmpeg.exe"))
    assert ffm.resolve_ffmpeg("ffmpeg") == box.path


def test_the_windows_install_dirs_are_the_last_resort(box, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    winget = box.tmp / "winget" / "ffmpeg.exe"
    winget.parent.mkdir()
    winget.write_bytes(b"")
    monkeypatch.setattr(ffm, "_WINDOWS_INSTALL_CANDIDATES",
                        (str(box.tmp / "absent" / "ffmpeg.exe"), str(winget)))
    assert ffm.resolve_ffmpeg("ffmpeg") == str(winget)


def test_nothing_anywhere_is_none_and_never_raises(box, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert ffm.resolve_ffmpeg("ffmpeg") is None
    assert ffm.resolve_ffmpeg(None) is None
    assert ffm.resolve_ffmpeg(str(box.tmp / "gone.exe")) is None


# --------------------------------------------------------------------------- #
# The adapters -- every runtime owner hands back the SAME answer, and keeps
# its own contract on "none".
# --------------------------------------------------------------------------- #
def _adapters():
    from nodes import otr_caption_burn as cb
    from nodes import otr_credits_roll as cr
    from nodes import otr_master_audio_mux as mux
    from nodes import otr_silent_composite as sc
    from nodes import video_engine as ve
    from nodes._otr_shared import content_oracle as co
    from nodes._otr_shared import encode_sink as es
    from nodes._otr_shared import scope_draw as sd
    from nodes._otr_video_engines import render_driver as rd
    from nodes._otr_video_engines import wrapper_bridge as wb
    # The three argv sites r3 found that no resolver copy ever touched -- they
    # ran the literal, so the AST guard could not see them.
    from nodes import otr_post_upscale_procgen_blend as pu
    from nodes._otr_audio_engines import eng_google_lyria as ly
    from nodes._otr_video_engines import foley_stems as fs
    return {
        "post_upscale._ffmpeg_bin": lambda: pu._ffmpeg_bin("ffmpeg"),
        "lyria._ffmpeg_bin": lambda: ly._ffmpeg_bin(),
        "foley_stems._ffmpeg_bin": lambda: fs._ffmpeg_bin(),
        "mux._ffmpeg_bin": lambda: mux._ffmpeg_bin("ffmpeg"),
        "caption._ffmpeg_bin": lambda: cb._ffmpeg_bin("ffmpeg"),
        "silent._ffmpeg_bin": lambda: sc._ffmpeg_bin("ffmpeg"),
        "credits._ffmpeg_bin": lambda: cr._ffmpeg_bin(),
        "video_engine._find_ffmpeg": lambda: ve._find_ffmpeg(),
        "content_oracle._ffmpeg": lambda: co._ffmpeg("ffmpeg"),
        "encode_sink.find_ffmpeg": lambda: es.find_ffmpeg("ffmpeg"),
        "scope_draw.find_ffmpeg": lambda: sd.find_ffmpeg("ffmpeg"),
        "wrapper_bridge.resolve_ffmpeg": lambda: wb.resolve_ffmpeg("ffmpeg"),
        "render_driver._slicer_ffmpeg_bin": lambda: rd._slicer_ffmpeg_bin(),
    }


@pytest.mark.parametrize("name", sorted(_adapters()))
def test_every_adapter_honours_the_pin_with_path_available(name, box,
                                                           monkeypatch):
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    assert _adapters()[name]() == box.env, name


@pytest.mark.parametrize("name", sorted(_adapters()))
def test_every_adapter_uses_path_when_nothing_is_pinned(name, box):
    assert _adapters()[name]() == box.path, name


def test_each_adapter_keeps_its_own_answer_on_none(box, monkeypatch):
    """The owner says None; each caller had already decided what that costs."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    from nodes import otr_caption_burn as cb
    from nodes import otr_credits_roll as cr
    from nodes import otr_master_audio_mux as mux
    from nodes import otr_silent_composite as sc
    from nodes import video_engine as ve
    from nodes._otr_shared import content_oracle as co
    from nodes._otr_shared import encode_sink as es
    from nodes._otr_shared import scope_draw as sd
    from nodes._otr_video_engines import render_driver as rd
    from nodes._otr_video_engines import wrapper_bridge as wb

    assert mux._ffmpeg_bin("ffmpeg") == ""
    assert cb._ffmpeg_bin("ffmpeg") == ""
    assert sc._ffmpeg_bin("ffmpeg") == ""
    assert es.find_ffmpeg("ffmpeg") is None
    assert sd.find_ffmpeg("ffmpeg") is None
    assert ve._find_ffmpeg() is None
    # The bridge and the slicer keep a STRING so argv[0] is never None and
    # subprocess's FileNotFoundError still reaches their named refusals.
    assert wb.resolve_ffmpeg("ffmpeg") == "ffmpeg"
    assert wb._with_resolved_ffmpeg(["ffmpeg", "-y"])[0] == "ffmpeg"
    assert rd._slicer_ffmpeg_bin() == "ffmpeg"
    assert co._ffmpeg(None) == "ffmpeg"
    with pytest.raises(cr.CreditsDataError, match="ffmpeg not found"):
        cr._ffmpeg_bin()
    from nodes import otr_post_upscale_procgen_blend as pu
    from nodes._otr_audio_engines import eng_google_lyria as ly
    from nodes._otr_video_engines import foley_stems as fs
    assert pu._ffmpeg_bin("ffmpeg") == "ffmpeg"
    assert pu._ffmpeg_bin("") == "ffmpeg"
    assert ly._ffmpeg_bin() == "ffmpeg"
    assert fs._ffmpeg_bin() == "ffmpeg"


def test_the_sites_that_call_the_owner_by_name_are_bound_to_THE_owner():
    """The canonicalizers, the encode sink, the slicer and Lyria import
    ``resolve_ffmpeg`` at module scope and call it with no seam of their own,
    so the matrix above cannot drive them with media. What CAN be proven is
    that the name they call is the owner's function object -- not a second
    module instance of it (the flat-import hazard post_upscale had)."""
    from nodes._otr_audio_engines import eng_google_lyria as ly
    from nodes._otr_shared import cloud_media_canonical as cmc
    from nodes._otr_shared import encode_sink as es
    from nodes._otr_video_engines import render_driver as rd
    assert cmc.resolve_ffmpeg is ffm.resolve_ffmpeg
    assert es.resolve_ffmpeg is ffm.resolve_ffmpeg
    assert rd._resolve_ffmpeg is ffm.resolve_ffmpeg
    assert ly.resolve_ffmpeg is ffm.resolve_ffmpeg
    # post_upscale inserts nodes/ into sys.path and used to import BOTH tool
    # owners flat-first, which made a second instance of each (r3, r4).
    from nodes import otr_post_upscale_procgen_blend as pu
    from nodes._otr_shared import ffprobe as ffp
    assert pu.resolve_ffmpeg is ffm.resolve_ffmpeg
    assert pu._ffp is ffp


def test_the_raw_video_sink_refuses_by_name_on_none(box, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    from nodes._otr_shared import encode_sink as es
    sink = es.RawVideoSink.__new__(es.RawVideoSink)
    sink.mode = "file"
    sink.ffmpeg = "ffmpeg"
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        sink.__enter__()


def test_the_probe_sibling_steps_go_through_the_owner(box, monkeypatch):
    """resolve_ffprobe's last resort is "the ffprobe beside the ffmpeg this
    box runs" -- which must be the OWNER's answer, pin included."""
    from nodes._otr_shared import ffprobe as fp
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    probe = box.tmp / "env" / "ffprobe.exe"
    probe.write_bytes(b"")
    monkeypatch.delenv("OTR_FFPROBE", raising=False)
    monkeypatch.setattr(shutil, "which",
                        lambda name: box.path if name == "ffmpeg" else None)
    assert fp.resolve_ffprobe() == str(probe)


def test_a_pass_through_env_reader_is_not_needed_for_the_pin(box, monkeypatch):
    """The viz lanes used to hand ``os.environ.get("OTR_FFMPEG", "ffmpeg")``
    into ``find_ffmpeg`` themselves. The owner reads the pin; None is enough."""
    from nodes._otr_shared import scope_draw as sd
    monkeypatch.setenv("OTR_FFMPEG", box.env)
    assert sd.find_ffmpeg(None) == box.env
    assert os.path.isfile(sd.find_ffmpeg(None))
