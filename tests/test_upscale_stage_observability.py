"""A finished render must SAY whether the upscale stage engaged.

WHY THIS FILE EXISTS. On 2026-08-09 a live 45-word leg finished green with
`upscale_engine='spandrel_esrgan'` on node 84, assembled 8 beats, published a
30 MB deliverable -- and left NO evidence whatsoever about the upscale stage.
Grepping the whole server log found:

  * no `spandrel_esrgan` / `ModelLoader` / `ESRGAN` line (the adapter logs
    nothing on a successful load),
  * no `model skip` line (the other branch of _encode_segment),
  * and node 84 produces no /history entry, so its status string -- the only
    carrier of "upscale=<engine>@<device>" -- is unreachable.

Both branches were silent, so the render could not distinguish "ran fine" from
"never engaged". Item 8 chip 4 had been owed a proof leg since the ship and
could not have been closed by ANY leg, because no leg could produce evidence.
These tests keep the evidence in place.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "nodes" / "otr_silent_composite.py")


def _tree():
    return ast.parse(SRC.read_text(encoding="utf-8"))


def _log_message_constants() -> list[str]:
    """Every string literal handed to a log.<level>(...) call in the module."""
    out: list[str] = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name) and fn.value.id == "log"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            out.append(node.args[0].value)
    return out


def test_both_upscale_branches_are_observable():
    """The defect was SYMMETRIC silence -- fixing only one branch still leaves
    a render that cannot be read."""
    msgs = " || ".join(_log_message_constants())
    assert "upscale MODEL PATH" in msgs, (
        "no positive receipt: a render that DID upscale still looks identical "
        "to one that did not"
    )
    assert "upscale FAST PATH" in msgs, (
        "no negative receipt: when an engine is selected but a segment skips "
        "the model, the render must say so -- that silence is what made the "
        "2026-08-09 leg unreadable"
    )
    assert "upscale engine LOADED" in msgs, (
        "no load receipt: the adapter logs nothing on success, so without this "
        "'loaded fine' and 'never engaged' are indistinguishable"
    )


def test_the_off_default_stays_silent():
    """Byte-identity insurance. The `off` sentinel must add no log lines, or
    every existing render's output changes for a feature nobody enabled.

    The fast-path receipt is guarded by `engine is not None and engine.name !=
    "off"`; this pins that the guard exists rather than trusting the comment.
    """
    src = SRC.read_text(encoding="utf-8")
    idx = src.find("upscale FAST PATH")
    assert idx > 0, "fast-path receipt missing entirely"
    window = src[max(0, idx - 400):idx]
    assert 'engine.name != "off"' in window, (
        "the fast-path receipt is not gated on a non-off engine, so an `off` "
        "render would start emitting new lines"
    )


def test_load_receipt_reports_the_resolved_checkpoint():
    """The path matters, not just the engine name.

    On the headless topology folder_paths maps upscale_models at a directory
    holding no .pth, so the checkpoint is reachable ONLY through the
    repo-relative fallback -- the precise divergence 088dabc8 fixed. A receipt
    naming the engine but not the file would not show that.
    """
    src = SRC.read_text(encoding="utf-8")
    idx = src.find("upscale engine LOADED")
    assert idx > 0
    window = src[idx:idx + 400]
    assert "checkpoint=" in window, "load receipt omits the resolved model path"
    assert "_resolve_model" in src, (
        "the receipt should source the path from the engine's own resolver, "
        "not re-derive it -- re-deriving is how the two answers diverged "
        "in the first place"
    )


def test_diagnostics_cannot_fail_a_render():
    """A receipt that can raise is worse than no receipt.

    The resolver call is best-effort and must stay wrapped, so a broken or
    third-party engine cannot turn a diagnostic into a failed render.
    """
    src = SRC.read_text(encoding="utf-8")
    idx = src.find("_resolver = getattr(engine")
    assert idx > 0, "resolver lookup missing"
    window = src[max(0, idx - 200):idx + 400]
    assert "try:" in window and "except Exception" in window, (
        "the diagnostic resolver call is not guarded"
    )
