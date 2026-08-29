"""The GGUF VRAM preflight ADVISES; it must never refuse.

Operator directive 2026-08-29: "prove me wrong with an OOM but don't put an
artificial gate", restating the standing rule "I don't want guards to kill
anything, an OOM is the only killer".

The check used to raise GGUFNativeConfigError when its estimate exceeded free
VRAM. That estimate is `model_path.stat().st_size` -- the WHOLE file -- and it
ignores `n_gpu_layers`, which is exactly the lever that makes a large model fit
a small card. A 12B Q4_K_M split across GPU and CPU was refused on the
arithmetic of a configuration nobody requested.

What must NOT come back: a silent context downgrade. The old 4096->2048
fallback truncated generations, which is why the raise existed at all.
Warning-and-attempting keeps the requested configuration exactly as asked.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re

from nodes import _otr_gguf_backend as ggf


def _preflight_window() -> str:
    src = inspect.getsource(ggf)
    i = src.find("VRAM Preflight")
    assert i > 0, "preflight log line not found -- did the block move?"
    return src[max(0, i - 3000): i + 3000]


def test_the_preflight_does_not_raise_on_an_estimate():
    """Checked against the AST, not a text window -- the branch body exactly."""
    src = pathlib.Path(ggf.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test)
        if "free_gb" in test_src and "estimated_needed_gb" in test_src:
            guarded.append((test_src, node))

    assert guarded, "the free-vs-needed comparison vanished entirely"
    for test_src, node in guarded:
        raises = [n for stmt in node.body for n in ast.walk(stmt)
                  if isinstance(n, ast.Raise)]
        assert not raises, (
            "the VRAM preflight raises again on `%s` -- that is the artificial "
            "gate the operator removed. It must warn and attempt, and let an "
            "actual OOM be the authority." % test_src)


def test_it_still_warns_loudly_with_the_actionable_advice():
    """Removing the gate must not remove what the error told the operator."""
    block = _preflight_window()
    for phrase in ("PROCEEDING ANYWAY", "gguf_n_ctx", "OTR_GGUF_N_GPU_LAYERS"):
        assert phrase in block, (
            "the warning dropped %r -- the advice the old error carried must "
            "survive the gate's removal" % phrase)


def test_no_silent_context_downgrade_was_reintroduced():
    """The reason the raise existed in the first place."""
    block = _preflight_window()
    assert not re.search(r"n_ctx\s*=\s*(2048|1024)\b", block), (
        "something assigns a smaller n_ctx near the preflight -- a silent "
        "downgrade truncates generations, and avoiding it is why the raise "
        "was added. Warn and attempt; never quietly shrink the request.")


def test_partial_offload_is_called_out_as_an_upper_bound():
    """With n_gpu_layers >= 0 the whole-file estimate is simply wrong high."""
    block = _preflight_window()
    assert "upper bound" in block and "eff_n_gpu_layers" in block, (
        "the warning does not distinguish a partial offload, so it over-reports "
        "need for exactly the configuration that makes a big model fit a small "
        "card -- which was the defect, not just the gate")
