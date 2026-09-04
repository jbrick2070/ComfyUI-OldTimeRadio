"""One nvenc decision, and it must be a real encode probe.

WHY THIS FILE EXISTS. The "is h264_nvenc usable" question has now been answered
wrongly three separate times, in three modules, by the same one-line mistake:

    "h264_nvenc" in (ffmpeg -codecs)      # compiled with it != can encode with it

`encode_sink.has_nvenc` was rewritten on 2026-08-30 to run a real one-frame
encode after that string test cost a whole episode, and its docstring then
claimed to be the only such decision in the pack. It was not: a third copy
survived in `_otr_shared/scope_draw.py`, and because the four viz_* engines
encode through THAT module rather than through `RawVideoSink`, they kept
choosing a dead encoder. A rented RTX 4090 found it on 2026-09-03 -- ffmpeg
lists the encoder and cannot open a session, which is the ordinary case in a GPU
container with no NVENC passthrough:

    [h264_nvenc] OpenEncodeSessionEx failed: unsupported device (2)
    [h264_nvenc] No capable devices found          (ffmpeg exits 187)

The failure surfaced as `ffmpeg closed the pipe after 3 frame(s)` eight minutes
into a leg, blaming ffmpeg rather than the encoder choice.

A comment saying "delegate to the probe" is not enforcement. This is.
"""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
NODES = REPO / "nodes"

#: The module that owns the decision. Everyone else delegates to it.
OWNER = NODES / "_otr_shared" / "encode_sink.py"

#: The string test, in the spellings it has actually appeared in.
STRING_TEST = re.compile(
    r"""["']h264_nvenc["']\s+in\s+\(?\s*(out|res|proc|result)\b""")


def _python_sources():
    for path in NODES.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_only_encode_sink_probes_nvenc_by_encoding():
    """The owner runs a real encode; nobody else may answer the question."""
    text = OWNER.read_text(encoding="utf-8")
    assert "-c:v" in text and "h264_nvenc" in text, (
        "encode_sink must still contain the real probe")
    assert "-frames:v" in text, (
        "encode_sink.has_nvenc must probe by ENCODING a frame, not by listing "
        "codecs -- that is the whole point of this file")


@pytest.mark.parametrize("path", sorted(_python_sources()), ids=lambda p: p.name)
def test_no_module_reimplements_the_nvenc_string_test(path):
    """`"h264_nvenc" in (ffmpeg -codecs)` is banned outside the owner.

    It reports that ffmpeg was COMPILED with nvenc, which is true on machines
    where encoding is impossible. Delegate to
    `_otr_shared.encode_sink.has_nvenc` instead.
    """
    if path.resolve() == OWNER.resolve():
        return
    src = path.read_text(encoding="utf-8", errors="replace")
    if "h264_nvenc" not in src:
        return
    hit = STRING_TEST.search(src)
    assert not hit, (
        "%s re-implements the nvenc STRING test (%r).\n"
        "That answers 'was ffmpeg built with nvenc', not 'can this machine "
        "encode'. Call _otr_shared.encode_sink.has_nvenc instead -- it probes "
        "by encoding a frame. Three copies of this bug have shipped already."
        % (path.relative_to(REPO), hit.group(0)))


def test_scope_draw_delegates_rather_than_deciding():
    """The specific regression: scope_draw is where copy three lived, and the
    viz_* engines reach ffmpeg through it, not through RawVideoSink."""
    import ast

    path = NODES / "_otr_shared" / "scope_draw.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_has_nvenc"), None)
    assert fn is not None, "scope_draw._has_nvenc disappeared; update this test"
    # CODE ONLY, NOT THE DOCSTRING. The docstring deliberately QUOTES the
    # banned pattern so the next reader knows what was wrong; checking the raw
    # source therefore flags the very comment that explains the fix.
    stripped = ast.FunctionDef(
        name=fn.name, args=fn.args, decorator_list=[], returns=None,
        body=[n for n in fn.body
              if not (isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Constant)
                      and isinstance(n.value.value, str))],
        type_params=[])
    ast.fix_missing_locations(stripped)
    code = ast.unparse(stripped)
    assert "encode_sink" in code, (
        "scope_draw._has_nvenc must DELEGATE to encode_sink.has_nvenc")
    assert "-codecs" not in code, (
        "scope_draw._has_nvenc must not shell out to `ffmpeg -codecs`; that is "
        "the string test that shipped three times")
