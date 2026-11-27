"""Node errors carry no absolute paths (nodes/_otr_error_text.py).

ComfyUI copies str(exception) into /history and the websocket error event, so
the pack cuts absolute paths to their last component at the node boundary.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
from pathlib import Path

import pytest

from nodes._otr_error_text import (
    guard_node_class,
    guard_node_classes,
    redact_exception,
    redact_paths,
)

PACK_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"missing C:\Users\someone\ComfyUI\models\tts\voice.pt", "missing voice.pt"),
        ("missing C:/Users/someone/models/x.safetensors now", "missing x.safetensors now"),
        ("could not read '/home/someone/ComfyUI/output/ep1/final.mp4'", "could not read 'final.mp4'"),
        (r"share \\nas\media\clips\a.wav gone", "share a.wav gone"),
        ("folder /home/someone/out/ is empty", "folder out is empty"),
    ],
)
def test_absolute_paths_are_cut_to_the_last_component(text, expected):
    assert redact_paths(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "fetch https://huggingface.co/org/model/resolve/main/a.bin failed",
        "relative otr/episodes/ep1/audio.wav is fine",
        "ratio 3/4 and 16/9",
        "no paths here",
        "~/notes/today.txt stays relative to home",
        "POST /v1/chat/completions returned 500",
        "route /api/prompt failed",
    ],
)
def test_urls_relative_paths_and_plain_text_are_unchanged(text):
    assert redact_paths(text) == text


@pytest.mark.parametrize(
    "home, text, expected",
    [
        (r"C:\Users\John Smith", r"missing C:\Users\John Smith\Documents\x.wav", "missing x.wav"),
        (r"C:\Users\John Smith", "missing C:/Users/John Smith/models/y.pt here", "missing y.pt here"),
        (r"C:\Users\John Smith", r"repr 'C:\\Users\\John Smith\\a\\b.json'", "repr 'b.json'"),
        ("/home/mary jane", "no /home/mary jane/models/z.bin", "no z.bin"),
        ("/home/mary jane", "home is /home/mary jane", "home is ~"),
    ],
)
def test_a_home_folder_with_spaces_is_cut_whole(monkeypatch, home, text, expected):
    """The path rule stops at whitespace; the home prefix must not leave the
    account name behind (Sonnet QA on 9ed01d70d)."""
    import nodes._otr_error_text as mod

    monkeypatch.setattr(mod, "_HOME_PREFIXES", (
        home.replace("\\", "\\\\"), home, home.replace("\\", "/"),
    ))
    out = redact_paths(text)
    assert out == expected
    assert "Smith" not in out and "mary" not in out


@pytest.mark.parametrize(
    "text, expected",
    [
        # A colon right before the path is a key/value separator, not a URL.
        ("cache dir:/home/someone/.cache/hf", "cache dir:hf"),
        # A file:// URL's third slash is the root of a local path.
        ("opened file:///home/someone/out/ep.mp4", "opened file://ep.mp4"),
    ],
)
def test_a_path_after_a_colon_or_in_a_file_url_is_cut(text, expected):
    out = redact_paths(text)
    assert out == expected
    assert "someone" not in out


@pytest.mark.parametrize(
    "text, expected",
    [
        # Another folder whose name merely STARTS with the account name.
        ("other /Users/jabber/secret/x.wav", "other x.wav"),
        ("dotted /Users/jab.smith/notes/y.txt", "dotted y.txt"),
        # The home folder itself, ending a sentence or a path.
        ("home is /Users/jab.", "home is ~."),
        ("in /Users/jab/ComfyUI/output/ep.mp4", "in ep.mp4"),
    ],
)
def test_the_home_cut_never_splits_a_longer_folder_name(monkeypatch, text, expected):
    import nodes._otr_error_text as mod

    monkeypatch.setattr(mod, "_HOME_PREFIXES", ("/Users/jab",))
    out = redact_paths(text)
    assert out == expected
    assert "ber" not in out and ".smith" not in out


def test_os_error_filename_attribute_is_redacted():
    try:
        open(r"C:\Users\someone\missing_folder\clip.wav", "rb")  # noqa: SIM115
    except OSError as exc:
        redact_exception(exc)
        assert "someone" not in str(exc)
        assert "clip.wav" in str(exc)
    else:  # pragma: no cover
        pytest.fail("the file should not exist")


def test_called_process_error_command_is_redacted():
    exc = subprocess.CalledProcessError(1, ["/opt/someone/bin/ffmpeg", "-i", "/home/someone/a.mp4"])
    redact_exception(exc)
    assert "someone" not in str(exc)
    assert "ffmpeg" in str(exc) and "a.mp4" in str(exc)


class _Node:
    FUNCTION = "run"

    def run(self, where):
        raise FileNotFoundError(f"no clip at {where}")


class _ClassMethodNode:
    FUNCTION = "go"

    @classmethod
    def go(cls, where):
        raise ValueError(f"bad {where}")


class _AsyncNode:
    FUNCTION = "work"

    async def work(self, where):
        raise RuntimeError(f"failed on {where}")


def test_guarded_node_keeps_the_exception_object_and_type():
    guard_node_class(_Node)
    with pytest.raises(FileNotFoundError) as info:
        _Node().run(where="/home/someone/ComfyUI/output/x.mp4")
    assert str(info.value) == "no clip at x.mp4"


def test_guarded_classmethod_stays_a_classmethod():
    guard_node_class(_ClassMethodNode)
    assert isinstance(vars(_ClassMethodNode)["go"], classmethod)
    with pytest.raises(ValueError, match="^bad y.json$"):
        _ClassMethodNode.go(where=r"D:\data\someone\y.json")


def test_guarded_async_node_stays_a_coroutine_function():
    import inspect

    guard_node_class(_AsyncNode)
    assert inspect.iscoroutinefunction(_AsyncNode().work)
    with pytest.raises(RuntimeError, match="^failed on z.wav$"):
        asyncio.run(_AsyncNode().work(where="/srv/someone/z.wav"))


class _CanvasTextNode:
    FUNCTION = "show"

    def show(self, where):
        return {"ui": {"text": ["saved to " + where, "second " + where],
                       "video": [{"filename": "ep.mp4", "subfolder": "otr/obs",
                                  "type": "output"}]},
                "result": (where, "report for " + where)}


class _AsyncCanvasTextNode:
    FUNCTION = "show"

    async def show(self, where):
        return {"ui": {"text": ["saved to " + where]}, "result": (where,)}


def test_guarded_node_redacts_its_canvas_text_and_nothing_else():
    """A node's `ui` text is stored in /history and sent on the websocket with
    no login, exactly like exception text, so the same wrapper cuts paths in
    it. The file references beside it are already relative, and `result` is
    what the next node receives: both pass through whole."""
    guard_node_class(_CanvasTextNode)
    where = "/home/someone/ComfyUI/output/otr/obs/ep.mp4"
    out = _CanvasTextNode().show(where)
    assert out["ui"]["text"] == ["saved to ep.mp4", "second ep.mp4"]
    assert out["ui"]["video"] == [{"filename": "ep.mp4", "subfolder": "otr/obs",
                                   "type": "output"}]
    assert out["result"] == (where, "report for " + where)


def test_guarded_async_node_redacts_its_canvas_text():
    guard_node_class(_AsyncCanvasTextNode)
    out = asyncio.run(_AsyncCanvasTextNode().show(r"C:\Users\someone\out\ep.mp4"))
    assert out["ui"]["text"] == ["saved to ep.mp4"]
    assert out["result"] == (r"C:\Users\someone\out\ep.mp4",)


def test_a_plain_tuple_result_passes_through_the_guard_untouched():
    class _Tuple:
        FUNCTION = "run"

        def run(self):
            return ("/home/someone/x.wav",)

    guard_node_class(_Tuple)
    assert _Tuple().run() == ("/home/someone/x.wav",)


def test_guarding_twice_wraps_once():
    guard_node_class(_Node)
    first = vars(_Node)["run"]
    assert guard_node_classes([_Node, _Node]) == 2
    assert vars(_Node)["run"] is first


def test_pack_init_guards_every_registered_node():
    """The helper is wired: __init__.py calls it on NODE_CLASS_MAPPINGS."""
    tree = ast.parse((PACK_ROOT / "__init__.py").read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "guard_node_classes"
    ]
    assert calls, "__init__.py must call guard_node_classes"
    assert "NODE_CLASS_MAPPINGS" in ast.unparse(calls[0])


def test_the_cause_and_context_of_an_exception_are_redacted_too():
    """THE DEFECT THIS PINS (shipped through 2.3.64): only the top exception
    was redacted. ComfyUI's traceback, in its log and in /history, prints
    every link of the chain with its own message, so the path survived on the
    cause."""
    # Built from parts: a traceback also prints each frame's SOURCE LINE, and
    # a literal path on those lines would be this test's own text, not the
    # exception's.
    where = "/".join(("", "home", "someone", "models", "tts", "missing_voice.pt"))

    def _work():
        try:
            open(where)
        except OSError as err:
            raise ValueError("voice load failed for " + where) from err

    class _Chained:
        FUNCTION = "run"

        def run(self):
            _work()

    guard_node_class(_Chained)
    with pytest.raises(ValueError) as caught:
        _Chained().run()
    top = caught.value
    assert "someone" not in str(top)
    assert top.__cause__ is not None and "someone" not in str(top.__cause__)
    assert "missing_voice.pt" in str(top.__cause__)
    import traceback
    text = "".join(traceback.format_exception(type(top), top, top.__traceback__))
    assert "/home/someone" not in text


def test_an_implicit_context_is_redacted_and_a_cycle_ends():
    try:
        try:
            raise KeyError("/home/someone/a.json")
        except KeyError:
            raise RuntimeError("while reading /home/someone/b.json")
    except RuntimeError as err:
        err.__context__.__context__ = err          # a cycle must not loop
        redact_exception(err)
        assert "someone" not in str(err)
        assert "someone" not in str(err.__context__)
