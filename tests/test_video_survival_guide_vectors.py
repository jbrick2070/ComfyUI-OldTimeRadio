"""A-S7 CPU tests -- the comfyui-custom-node-survival-guide regression vectors.

The A-S7 EXIT gate requires the package to pass the four canonical survival-guide
failure-mode vectors. These run them against the model-agnostic video platform +
the A-S7 retry / fallback / QC surface:

* ghost-node (BUG 12.23, registration import gap) -- no engine's declared
  ``fallback_engine`` is a ghost (every hop names a REGISTERED engine and the
  chain terminates at a registered always-available radio floor); the A-S7
  helpers add no ghost node mapping.
* VRAM-leak (V-4 / BUG-291) -- the entire NEW video-platform surface never CALLS
  ``unload_all_models()`` (AST-checked, so docstring mentions do not count); the
  OOM block_class tears down + walks the fallback with zero retries.
* widget-serialization (BUG-258) -- the A-S7 helper modules add no ComfyUI node
  or widget surface, so there is nothing new to mis-serialize.
* pipe-deadlock (BUG 09.02) -- the retry taxonomy BOUNDS every retry (finite
  attempts for every kind) and a timeout escalates with zero retries, so the
  retry/fallback layer can never spin forever.

All CPU; the live subprocess / ffmpeg behavior is the operator smoke, NOT here.
"""
from __future__ import annotations

import ast
import pathlib

from nodes._otr_shared import nsfw_frame_qc as qc
from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_shared.fallback import resolve_fallback_chain
from nodes._otr_video_engines import registry as vreg
# Register every engine so the real fallback graph can be walked.
from nodes._otr_video_engines import eng_humo            # noqa: F401
from nodes._otr_video_engines import eng_ltx_video       # noqa: F401
from nodes._otr_video_engines import eng_wan_i2v         # noqa: F401
from nodes._otr_video_engines import cheap_families      # noqa: F401

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODES = REPO_ROOT / "nodes"
PLATFORM_DIRS = (NODES / "_otr_video_engines", NODES / "_otr_shared")


def _platform_py_files():
    for d in PLATFORM_DIRS:
        for p in sorted(d.glob("*.py")):
            yield p


def _calls_function(source: str, fn_name: str) -> bool:
    """True iff ``source`` CALLS ``fn_name`` (AST -- ignores docstrings / comments)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == fn_name:
                return True
            if isinstance(f, ast.Attribute) and f.attr == fn_name:
                return True
    return False


# --------------------------------------------------------------------------- #
# ghost-node (BUG 12.23 -- registration import gap)
# --------------------------------------------------------------------------- #
def test_no_ghost_fallback_engine_references():
    for name in vreg.all_engine_names():
        nxt = getattr(vreg.get_engine(name), "fallback_engine", None)
        if nxt is not None:
            assert vreg.is_registered(nxt), (
                "engine %r declares fallback_engine %r which is not registered "
                "(ghost reference, BUG 12.23)" % (name, nxt))


def test_fallback_chains_resolve_to_a_registered_radio_floor():
    def fallback_of(n):
        return getattr(vreg.get_engine(n), "fallback_engine", None)
    for start in ("humo", "humo_1.7B"):
        chain = resolve_fallback_chain(start, fallback_of)
        floor = chain[-1]
        assert vreg.is_registered(floor)              # terminus is real
        assert fallback_of(floor) is None             # and terminal
        # an always-available radio floor has no opt-in flag.
        assert getattr(vreg.get_engine(floor), "requires_flag", "x") is None


def test_a_s7_modules_register_no_ghost_node():
    for mod in (rt, qc):
        assert not hasattr(mod, "NODE_CLASS_MAPPINGS")


# --------------------------------------------------------------------------- #
# VRAM-leak (V-4 / BUG-291 -- never unload_all_models)
# --------------------------------------------------------------------------- #
def test_new_platform_never_calls_unload_all_models():
    offenders = [p.name for p in _platform_py_files()
                 if _calls_function(p.read_text(encoding="utf-8"),
                                    "unload_all_models")]
    assert offenders == [], (
        "V-4 / BUG-291: the new video platform must never CALL "
        "unload_all_models(); offenders: %s" % offenders)


def test_oom_block_class_tears_down_and_walks_fallback():
    d = rt.classify(rt.FailureKind.OOM)
    assert d.is_hard and d.escalate_to_fallback is True
    assert d.same_seed_retries == 0 and d.reseed_retries == 0   # 0 retries
    assert d.touches_audio is False and d.aborts_episode is False


# --------------------------------------------------------------------------- #
# widget-serialization (BUG-258 -- forceInput / widget integrity)
# --------------------------------------------------------------------------- #
def test_a_s7_surface_adds_no_widget_to_misserialize():
    for mod in (rt, qc):
        for obj in vars(mod).values():
            if isinstance(obj, type) and obj.__module__ == mod.__name__:
                assert not hasattr(obj, "INPUT_TYPES"), (
                    "%s.%s exposes INPUT_TYPES; A-S7 must add no widget surface "
                    "(BUG-258)" % (mod.__name__, obj.__name__))


# --------------------------------------------------------------------------- #
# pipe-deadlock (BUG 09.02 -- nothing may spin forever)
# --------------------------------------------------------------------------- #
def test_every_retry_is_bounded_no_infinite_loop():
    for k in rt.FailureKind:
        d = rt.classify(k)
        assert 1 <= d.max_attempts < 100              # finite, bounded
        if d.is_hard:
            assert d.escalate_to_fallback is True     # always an escape hatch


def test_timeout_escalates_with_zero_retries():
    d = rt.classify(rt.FailureKind.TIMEOUT)
    assert d.same_seed_retries == 0 and d.reseed_retries == 0
    assert d.escalate_to_fallback is True             # hung render -> fallback


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
