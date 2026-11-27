"""Edges of the path redaction that a second reader found by trying to break it.

Each case here was reproduced against the first version of the fix it tests
(Sonnet QA of e919d17a4..d53469267, 2026-11-27):

* an install folder with a space UNDER the home folder still leaked its
  layout, because the shorter home prefix was cut first;
* a home of ``/root`` was matched inside URLs and inside longer paths;
* a path inside JSON text lost the backslash that escapes its closing quote,
  so the text no longer parsed;
* members of an ``ExceptionGroup`` and a namedtuple in ``args`` were skipped;
* two new filesystem roots also matched bare HTTP routes.

Backslashes are built with ``chr(92)`` so no line here depends on how an
escape is read. CPU only. UTF-8 no BOM.
"""
from __future__ import annotations

import collections
import json
import sys
import types

import pytest

import nodes._otr_error_text as mod
from nodes._otr_error_text import redact_exception, redact_paths

SEP = chr(92)


def _win(*parts: str) -> str:
    return SEP.join(parts)


def _pairs(*folders: str) -> tuple:
    found = set()
    for folder in folders:
        bare = folder.replace(SEP, "/").rstrip("/").split("/")[-1]
        for spelling in mod._spellings(folder):
            found.add((spelling, bare))
    return tuple(sorted(found, key=lambda pair: len(pair[0]), reverse=True))


def test_an_install_with_a_space_under_the_home_folder_is_cut_whole(monkeypatch):
    home = _win("C:", "Users", "someone")
    install = _win(home, "AI Tools", "ComfyUI")
    monkeypatch.setattr(mod, "_HOME_PREFIXES", tuple(sorted(mod._spellings(home), key=len,
                                                            reverse=True)))
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", _pairs(install, _win(install, "output")))
    shown = redact_paths("wrote " + _win(install, "output", "otr", "obs", "ep.mp4") + " ok")
    assert shown == "wrote ep.mp4 ok"
    assert redact_paths("docs " + _win(home, "Documents", "x.wav")) == "docs x.wav"


@pytest.mark.parametrize("text, expected", [
    ("fetch https://example.com/root/x failed", "fetch https://example.com/root/x failed"),
    ("see https://huggingface.co/root/model/resolve/main/a.bin",
     "see https://huggingface.co/root/model/resolve/main/a.bin"),
    ("read /var/root/x.txt", "read x.txt"),
    ("HF_HOME=/root/.cache/huggingface", "HF_HOME=huggingface"),
    ("opened file:///root/out/ep.mp4", "opened file://ep.mp4"),
    ("home is /root", "home is ~"),
    ("cd /root/ComfyUI/output", "cd output"),
])
def test_a_short_home_is_not_matched_inside_other_text(monkeypatch, text, expected):
    """A container or a pod runs as root: the home folder is ``/root``."""
    monkeypatch.setattr(mod, "_HOME_PREFIXES", ("/root",))
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", ())
    assert redact_paths(text) == expected


def test_a_home_segment_inside_a_url_is_left_alone(monkeypatch):
    monkeypatch.setattr(mod, "_HOME_PREFIXES", ("/home/jab",))
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", ())
    text = "GET https://huggingface.co/home/jab/x returned 404"
    assert redact_paths(text) == text
    assert redact_paths("saved /home/jab/out/ep.mp4") == "saved ep.mp4"


@pytest.mark.parametrize("inner_path, expected", [
    ("/home/someone/episodes/x.wav", "x.wav"),
    (_win("C:", "Users", "someone", "episodes", "a.wav"), "a.wav"),
    (_win("D:", "out", "folder") + SEP, "folder"),
])
def test_json_text_still_parses_after_redaction(monkeypatch, inner_path, expected):
    """A report shown on the canvas is JSON, sometimes JSON inside JSON. The
    backslash before a closing quote escapes the quote; it is not the path's."""
    monkeypatch.setattr(mod, "_HOME_PREFIXES", ())
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", ())
    once = json.dumps({"path": inner_path, "n": 3})
    twice = json.dumps({"report": once, "ok": True})
    for text in (once, twice):
        shown = redact_paths(text)
        assert "someone" not in shown
        parsed = json.loads(shown)
        if "report" in parsed:
            parsed = json.loads(parsed["report"])
        assert parsed == {"path": expected, "n": 3}


def test_an_escaped_newline_after_a_path_is_kept(monkeypatch):
    monkeypatch.setattr(mod, "_HOME_PREFIXES", ())
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", ())
    text = "wrote /tmp/work/a.wav" + SEP + "nERR: next line"
    assert redact_paths(text) == "wrote a.wav" + SEP + "nERR: next line"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup is 3.11+")
def test_the_members_of_an_exception_group_are_redacted():
    group = ExceptionGroup("two failed", [  # noqa: F821 -- builtin on 3.11+
        ValueError("missing /home/someone/a.wav"),
        OSError("cannot read /home/someone/b.wav"),
    ])
    redact_exception(group)
    assert [str(member) for member in group.exceptions] == [
        "missing a.wav", "cannot read b.wav"]


def test_a_namedtuple_in_args_does_not_stop_the_redaction():
    Where = collections.namedtuple("Where", "path label")
    err = ValueError("failed at /home/someone/a.wav", Where("/home/someone/a.wav", "voice"))
    redact_exception(err)
    assert err.args[0] == "failed at a.wav"
    assert err.args[1] == Where("a.wav", "voice")


@pytest.mark.parametrize("text", [
    "GET /storage/v1/object/public/x failed",
    "font from /Library/Fonts/Arial.ttf",
    "POST /api/prompt returned 400",
])
def test_bare_routes_and_shared_system_folders_are_left_alone(monkeypatch, text):
    monkeypatch.setattr(mod, "_HOME_PREFIXES", ())
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", ())
    assert redact_paths(text) == text


def test_the_install_folders_are_never_frozen(monkeypatch):
    """Not the partial answer of a call before ComfyUI is loaded, and not a
    stubbed ``folder_paths`` an earlier caller left behind."""
    monkeypatch.setattr(mod, "_INSTALL_PREFIXES", None)
    monkeypatch.setattr(mod, "_INSTALL_CACHE", ((), ()))
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    before = {prefix for prefix, _bare in mod._install_prefixes()}
    assert "/opt/First Install/output" not in before

    comfy = types.ModuleType("folder_paths")
    comfy.get_output_directory = lambda: "/opt/First Install/output"
    monkeypatch.setitem(sys.modules, "folder_paths", comfy)
    first = {prefix for prefix, _bare in mod._install_prefixes()}
    assert "/opt/First Install/output" in first

    comfy.get_output_directory = lambda: "/opt/Second Install/output"
    second = {prefix for prefix, _bare in mod._install_prefixes()}
    assert "/opt/Second Install/output" in second
    assert "/opt/First Install/output" not in second


@pytest.mark.parametrize("shown, expected", [
    (["saved /home/someone/out/ep.mp4"], ["saved ep.mp4"]),
    (("one /home/someone/a.wav", "two"), ["one a.wav", "two"]),
    ("only /home/someone/b.wav", ["only b.wav"]),
])
def test_canvas_text_is_redacted_whatever_container_holds_it(shown, expected):
    """ComfyUI's contract is a list of strings; a node that hands back a bare
    string or a tuple is normalized to that list rather than skipped."""
    result = {"ui": {"text": shown}, "result": ("/home/someone/out/ep.mp4",)}
    out = mod._redact_ui_text(result)
    assert out["ui"]["text"] == expected
    assert out["result"] == ("/home/someone/out/ep.mp4",)
    if isinstance(shown, list):
        assert out["ui"]["text"] is shown          # redacted in place
