"""Linked paths cannot carry a read or a write out of the approved folders.

Registry policy review, 2026-10-22 (Codex, against the shipped diff since
2.3.35): four new sinks checked a path's SPELLING but not where it RESOLVES,
so a junction or symlink already on disk could redirect them:

* the caption burn's shipped-font copy (a junction at ``otr_fonts``);
* the Kokoro voice load (a junctioned ``voices`` folder);
* the portrait-cache fallback a co-reference reads;
* the reference staging copy into ComfyUI's input dir (a link at the
  staged name).

Each is now proven inside an approved root, resolved, before it is touched.
A junction needs no privilege on Windows; a box that can make neither a
symlink nor a junction skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nodes import otr_caption_burn as CB
from nodes import otr_image_gen_dispatcher as DISP
from nodes import _otr_script_fonts as SF
from nodes._otr_audio_engines import eng_kokoro as EK
from nodes._otr_video_engines import wrapper_bridge as WB


def _make_dir_link(link: Path, target: Path) -> bool:
    """A directory symlink, else a Windows junction (no privilege needed)."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return True
        except (OSError, AttributeError):
            pass
    return False


@pytest.fixture
def allowed(tmp_path, monkeypatch):
    """Narrow the operator's extra roots to ``tmp_path/allowed`` so a sibling
    ``outside`` folder is genuinely outside every approved root."""
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", str(root))
    outside = tmp_path / "outside"
    outside.mkdir()
    return root, outside


def test_the_caption_font_copy_refuses_a_junctioned_fonts_folder(allowed):
    root, outside = allowed
    policy, (_file, family) = next(iter(SF.BUNDLED.items()))
    ass = root / "episode.ass"
    ass.write_text("Style: Default," + family + ",48\n", encoding="utf-8")
    if not _make_dir_link(root / CB.SHIPPED_FONTS_DIRNAME, outside):
        pytest.skip("this box can create neither a symlink nor a junction")
    with pytest.raises(Exception):
        CB._shipped_fonts_option(str(ass))
    assert not any(outside.iterdir()), "a font was written outside the tree"


def test_the_caption_font_copy_still_works_inside_the_tree(allowed):
    root, _outside = allowed
    policy, (_file, family) = next(iter(SF.BUNDLED.items()))
    ass = root / "episode.ass"
    ass.write_text("Style: Default," + family + ",48\n", encoding="utf-8")
    assert CB._shipped_fonts_option(str(ass)) == ":fontsdir=%s" % CB.SHIPPED_FONTS_DIRNAME
    assert any((root / CB.SHIPPED_FONTS_DIRNAME).iterdir())


def test_a_junctioned_kokoro_voices_folder_is_refused(tmp_path, monkeypatch, allowed):
    _root, outside = allowed
    (outside / "af_heart.pt").write_bytes(b"not a voice")
    kokoro = tmp_path / "models" / "TTS" / "KokoroTTS"
    kokoro.mkdir(parents=True)
    if not _make_dir_link(kokoro / "voices", outside):
        pytest.skip("this box can create neither a symlink nor a junction")
    monkeypatch.setattr(EK, "_kokoro_model_dir", lambda: str(kokoro))
    path = EK._kokoro_voice_path("af_heart")
    assert os.path.basename(path) == "_refused_voice_id_.pt"
    assert not os.path.exists(path)


def test_a_real_kokoro_voice_still_resolves(tmp_path, monkeypatch):
    kokoro = tmp_path / "models" / "TTS" / "KokoroTTS"
    (kokoro / "voices").mkdir(parents=True)
    (kokoro / "voices" / "af_heart.pt").write_bytes(b"voice")
    monkeypatch.setattr(EK, "_kokoro_model_dir", lambda: str(kokoro))
    assert EK._kokoro_voice_path("af_heart") == str(kokoro / "voices" / "af_heart.pt")


def test_the_portrait_cache_fallback_reads_nothing_through_a_link(
        tmp_path, monkeypatch, allowed):
    root, outside = allowed
    (outside / "abc.png").write_bytes(b"\x89PNG\r\n\x1a\nsecret")
    if not _make_dir_link(root / "pool", outside):
        pytest.skip("this box can create neither a symlink nor a junction")
    monkeypatch.setattr(DISP._pl, "portrait_path_for_hash",
                        lambda _h, _o: str(root / "pool" / "abc.png"))
    assert DISP._portrait_png({}, "abc", str(root)) == ""


def test_staging_refuses_a_link_at_the_staged_name(tmp_path, monkeypatch, allowed):
    _root, outside = allowed
    inp = tmp_path / "input"
    inp.mkdir()
    src = tmp_path / "photo.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\nphoto")
    if not _make_dir_link(inp / "otr_ref_x.png", outside):
        pytest.skip("this box can create neither a symlink nor a junction")
    monkeypatch.setattr(WB, "comfy_input_dir", lambda: str(inp))
    with pytest.raises(WB.GraphExecutionError):
        WB.stage_into_comfy_input(str(src), "otr_ref_x.png")
    assert not any(outside.iterdir())


def test_staging_refuses_a_name_with_a_folder_in_it(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    inp.mkdir()
    src = tmp_path / "photo.png"
    src.write_bytes(b"photo")
    monkeypatch.setattr(WB, "comfy_input_dir", lambda: str(inp))
    with pytest.raises(WB.GraphExecutionError):
        WB.stage_into_comfy_input(str(src), "..\\escape.png")
    assert WB.stage_into_comfy_input(str(src), "ok.png") == "ok.png"


def test_a_bundled_my_story_photo_still_stages_as_a_reference(tmp_path, monkeypatch):
    """Sonnet QA 2026-10-22: the reference read's confinement first refused the
    pack's own bundled photos, which the built-in My Story reads in place --
    every built-in My Story still would have failed."""
    from nodes._otr_image_engines import flux3_klein as F3
    from nodes._otr_shared.character_photos import BUNDLED_PHOTO_DIR
    photo = os.path.join(BUNDLED_PHOTO_DIR, "jeffrey_1.jpg")
    assert os.path.isfile(photo)
    staged = {}

    class _Bridge:
        @staticmethod
        def stage_into_comfy_input(src, name):
            staged["src"], staged["name"] = src, name
            return name

    name = F3._stage_reference(_Bridge, photo)
    assert name.startswith("otr_ref_") and name.endswith(".jpg")
    assert os.path.samefile(staged["src"], photo)


def test_a_reference_outside_every_root_is_refused(allowed):
    from nodes._otr_image_engines import flux3_klein as F3
    _root, outside = allowed
    stray = outside / "stray.png"
    stray.write_bytes(b"\x89PNG\r\n\x1a\n")

    class _Bridge:
        @staticmethod
        def stage_into_comfy_input(src, name):
            raise AssertionError("must not stage")

    with pytest.raises(Exception):
        F3._stage_reference(_Bridge, str(stray))


def test_a_photo_picked_from_the_output_folder_still_stages(tmp_path, monkeypatch, allowed):
    """Sonnet QA 2026-10-22: "x.png [output]" resolves into ComfyUI's output
    folder, which is not a read root; the picker allows it, so staging must."""
    from nodes._otr_image_engines import flux3_klein as F3
    from nodes._otr_shared import character_photos as CP
    out = tmp_path / "comfy_output"
    out.mkdir()
    photo = out / "x.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\nphoto")
    real_root = CP._root_dir
    monkeypatch.setattr(CP, "_root_dir",
                        lambda kind: str(out) if kind == "output" else real_root(kind))

    class _Bridge:
        @staticmethod
        def stage_into_comfy_input(src, name):
            return name

    assert F3._stage_reference(_Bridge, str(photo)).startswith("otr_ref_")
