"""The script fonts the pack ships (2026-10-21).

A Linux RunPod pod has no CJK or Devanagari face, and a Mandarin episode burned
every caption, title card and credit as empty boxes. The pack now ships Noto
Sans SC and Noto Sans Devanagari, reached only after the OS's own faces miss.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_captions as CAP  # noqa: E402
from nodes import _otr_script_fonts as SF  # noqa: E402
from nodes import otr_caption_burn as BURN  # noqa: E402
from nodes import otr_credits_roll as CR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SAMPLES = {"cjk": "一路顺风こんにちは",
           "devanagari": "नमस्ते"}


@pytest.mark.parametrize("policy", sorted(SF.BUNDLED))
def test_each_shipped_face_is_there_and_draws_its_script(policy):
    tt = pytest.importorskip("fontTools.ttLib")
    path = SF.bundled_path(policy)
    assert path and Path(path).is_file(), policy
    font = tt.TTFont(path)
    assert font["name"].getDebugName(1) == SF.bundled_family(policy)
    cmap = font.getBestCmap()
    assert all(ord(ch) in cmap for ch in SAMPLES[policy]), policy
    licence = Path(path).parent / ("OFL-" + Path(path).stem.split("-")[0] + ".txt")
    assert licence.is_file(), licence
    assert "SIL Open Font License" in licence.read_text(encoding="utf-8")


def test_the_registry_bundle_keeps_the_fonts():
    ignore = (REPO / ".comfyignore").read_text(encoding="utf-8")
    shipped = {"nodes", "nodes/fonts", "nodes/fonts/*", "*.otf", "*.ttf"}
    assert not any(line.strip().rstrip("/") in shipped for line in ignore.splitlines())
    assert all(Path(SF.bundled_path(p)).parent.name == "fonts" and
               Path(SF.bundled_path(p)).parent.parent.name == "nodes" for p in SF.BUNDLED)


def test_the_shipped_cjk_face_is_the_last_candidate(tmp_path):
    paths = CR._credits_script_font_paths(str(tmp_path), "cjk")
    assert paths[-1] == SF.bundled_path("cjk")
    assert paths[:-1] == CR._os_script_font_paths(str(tmp_path), "cjk")


def test_the_shipped_devanagari_face_comes_before_linuxs_latin_less_ones(tmp_path):
    """Noto Sans Devanagari has no Latin letters, so the merge goes ahead of
    the Linux faces -- and still behind Nirmala UI and Sangam."""
    paths = CR._credits_script_font_paths(str(tmp_path), "devanagari")
    shipped = paths.index(SF.bundled_path("devanagari"))
    assert all(not p.startswith("/usr/") for p in paths[:shipped])
    assert all(p.startswith("/usr/") for p in paths[shipped + 1:])
    assert any("Nirmala" in p for p in paths[:shipped])
    assert sorted(paths[:shipped] + paths[shipped + 1:]) == sorted(
        CR._os_script_font_paths(str(tmp_path), "devanagari"))


def test_the_shipped_devanagari_face_carries_latin():
    tt = pytest.importorskip("fontTools.ttLib")
    cmap = tt.TTFont(SF.bundled_path("devanagari")).getBestCmap()
    assert all(ord(ch) in cmap for ch in "AZaz09/:-")


def test_latin_gets_no_shipped_face(tmp_path):
    assert CR._credits_script_font_paths(str(tmp_path), "latin_arial") == []


def test_captions_name_the_shipped_family_only_without_an_os_face(monkeypatch, tmp_path):
    monkeypatch.setenv("WINDIR", str(tmp_path))          # no Fonts folder there
    monkeypatch.setattr(CR, "_os_script_font_paths", lambda fd, policy: [str(tmp_path / "x.ttc")])
    assert CAP._shipped_family_if_no_os_face("cjk") == "Noto Sans SC"
    assert CAP._shipped_family_if_no_os_face("devanagari") == "OTR Noto Devanagari Latin"
    assert CAP._shipped_family_if_no_os_face("latin_arial") is None
    own = tmp_path / "msyh.ttc"
    own.write_bytes(b"face")
    monkeypatch.setattr(CR, "_os_script_font_paths", lambda fd, policy: [str(own)])
    assert CAP._shipped_family_if_no_os_face("cjk") is None


def test_the_caption_builder_uses_the_seam():
    import inspect
    source = inspect.getsource(CAP)
    assert "shipped = _shipped_family_if_no_os_face(font_policy)" in source
    assert "st[\"font\"] = title_font = shipped" in source


def test_the_burn_hands_libass_the_shipped_face(tmp_path):
    ass = tmp_path / "episode_captions.ass"
    ass.write_text("Style: Default,Noto Sans SC,40,&H00FFFFFF\n", encoding="utf-8")
    assert BURN._shipped_fonts_option(str(ass)) == ":fontsdir=otr_fonts"
    copied = tmp_path / "otr_fonts" / Path(SF.bundled_path("cjk")).name
    assert copied.is_file() and copied.stat().st_size == Path(SF.bundled_path("cjk")).stat().st_size
    plain = tmp_path / "plain.ass"
    plain.write_text("Style: Default,Microsoft YaHei,40,&H00FFFFFF\n", encoding="utf-8")
    assert BURN._shipped_fonts_option(str(plain)) == ""


def test_the_burn_command_carries_the_option():
    import inspect
    assert 'f"ass={ass_name}{fonts_opt},fps=' in inspect.getsource(BURN)
