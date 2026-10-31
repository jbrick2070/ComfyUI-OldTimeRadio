"""Node 12 reports no file: its silent cut is deleted by the asset cleanup.

PBUG-20261031-01. A Comfy Cloud deployment fetches every file a node reports in
``ui`` when the job ends. ``OTR_SignalLostVideo`` reported its silent CRT cut,
which lives in the episode folder that the mux's asset cleanup deletes after it
publishes, so two finished episodes came back ``failed``. The mux reports the
published obs copy; nothing else may report a file the run removes.

Source inspection is the right tool here: the property is about every ``return``
in ``render_video``, and driving the real render needs a GPU and a ledger.
"""
import ast
import pathlib

VIDEO_ENGINE = pathlib.Path(__file__).resolve().parents[1] / "nodes" / "video_engine.py"
FILE_KEYS = {"gifs", "images", "video", "audio", "animated"}


def _render_video_returns():
    tree = ast.parse(VIDEO_ENGINE.read_text(encoding="utf-8"))
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "SignalLostVideoRenderer":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "render_video":
                    return [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    raise AssertionError("SignalLostVideoRenderer.render_video not found")


def test_render_video_reports_no_file_in_ui():
    returns = _render_video_returns()
    assert returns, "render_video has no return statement"
    for ret in returns:
        if not isinstance(ret.value, ast.Dict):
            continue
        for key, value in zip(ret.value.keys, ret.value.values):
            if isinstance(key, ast.Constant) and key.value == "ui":
                assert isinstance(value, ast.Dict), "ui must be a literal dict"
                reported = {k.value for k in value.keys if isinstance(k, ast.Constant)}
                assert not reported & FILE_KEYS, (
                    "node 12 reports %s in ui (line %d); that file is deleted by "
                    "the asset cleanup" % (sorted(reported & FILE_KEYS), ret.lineno))


def test_render_video_still_returns_its_result():
    """The two sockets downstream nodes wire to are unchanged."""
    found = False
    for ret in _render_video_returns():
        if isinstance(ret.value, ast.Dict):
            keys = [k.value for k in ret.value.keys if isinstance(k, ast.Constant)]
            found = found or "result" in keys
    assert found, "render_video lost its result return"
