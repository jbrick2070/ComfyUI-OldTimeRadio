"""Modules that `scripts/` load BY PATH must still import in a fresh interpreter.

THE DEFECT THIS CLOSES, and it shipped for a few hours on 2026-09-04. The env/proc
single-owner migration added a two-rung import ladder to ~100 modules:

    try:     from ._otr_shared import env as otr_env      # packaged
    except:  from _otr_shared  import env as otr_env      # flat, needs nodes/ on sys.path

Several `scripts/` helpers deliberately load a node module BY PATH under a
non-package name -- `otr_provision.py` says it outright, "load the pure catalog
without importing the ComfyUI node package". In that mode NEITHER rung resolves:
there is no parent package for the relative form, and nothing has put `nodes/` on
`sys.path` for the flat one. Four modules that loaded before the migration stopped
loading after it, including the pod provisioner's writer catalog and the portable
voice-bank authority.

WHY THE 13,500-TEST SUITE STAYED GREEN, which is the part worth keeping: pytest
imports these modules INSIDE a process where `nodes/` is already importable, so
the flat rung always resolves and the missing third rung is invisible. Only a
FRESH interpreter can see it. That is why every case below runs in a subprocess
with a clean `sys.modules` and no inherited path -- an in-process version of this
file would pass while the provisioner was broken.

THE FIX each module carries is a `_NODES_DIR` insert INSIDE the `except` arm --
not before the try. Putting it before would mutate `sys.path[0]` on the ordinary
packaged import too, making the flat spelling resolvable everywhere and inviting a
SECOND module instance of the owner; inside the arm it runs only when the relative
rung has already failed, i.e. only when there is no package instance to duplicate.
(Placement corrected by a cursor cross-check, 2026-09-04.)
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Node modules a `scripts/` helper loads by PATH, with the caller that does it.
#: A module joins this list when a script starts loading it standalone.
STANDALONE_LOADED = {
    "nodes/_otr_model_catalog.py":
        "scripts/otr_provision.py::_load_writer_catalog (the pod provisioner)",
    "nodes/_otr_voice_bank.py":
        "scripts/otr_provision.py + scripts/otr_make_portable_voice_bank.py",
    "nodes/_otr_kokoro_voice_prefetch.py":
        # CORRECTED by a cursor cross-check, 2026-09-04. My first attribution
        # said scripts/otr_fetch_lane_weights.py, which only NAMES this module
        # in a comment explaining that kokoro voices are deliberately absent --
        # it never loads it. The real standalone loader is prestartup_script.py,
        # which ALREADY inserts nodes/ before its flat import, so BOOT was never
        # broken; only this stricter probe saw it. The probe stays, because a
        # module that needs an ambient sys.path insert to import is one edit
        # away from breaking.
        "prestartup_script.py (which inserts nodes/ itself, so boot was safe)",
    "nodes/otr_post_upscale_procgen_blend.py":
        "its own documented FLAT load by ComfyUI's custom-node loader",
}

_PROBE = textwrap.dedent(
    """
    import importlib.util, os, sys
    path = sys.argv[1]
    name = "otr_standalone_probe_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    print("LOADED")
    """
)


def _scrubbed_env():
    """The probe must not inherit a PYTHONPATH that already contains `nodes/`.

    If it did, the flat rung would resolve for the wrong reason: a genuinely
    broken module would print LOADED and this guard would fail OPEN, which is
    worse than not having it. Keeps only what a Windows interpreter needs to
    start (cursor cross-check, 2026-09-04)."""
    import os
    keep = ("SYSTEMROOT", "PATH", "COMSPEC", "TEMP", "TMP", "PATHEXT",
            "WINDIR", "PYTHONUTF8", "HOME", "USERPROFILE", "LANG")
    return {k: v for k, v in os.environ.items() if k.upper() in keep}


def _load_in_fresh_interpreter(rel: str):
    """Exactly what a `scripts/` helper does: spec_from_file_location on a path,
    under a NON-package name, in a process that has not imported the pack."""
    return subprocess.run(
        [sys.executable, "-c", _PROBE, str(REPO / rel)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO.parent),        # NOT the repo root -- no accidental sys.path[0]
        env=_scrubbed_env(),         # and no inherited PYTHONPATH
        timeout=120,
    )


@pytest.mark.parametrize("rel", sorted(STANDALONE_LOADED))
def test_the_module_loads_standalone_in_a_fresh_interpreter(rel):
    done = _load_in_fresh_interpreter(rel)
    assert done.returncode == 0 and "LOADED" in done.stdout, (
        "%s no longer imports when loaded BY PATH in a fresh interpreter.\n"
        "Loaded standalone by: %s\n"
        "This is what the in-process suite cannot see -- pytest already has "
        "nodes/ importable, so a missing sys.path bootstrap stays invisible "
        "there.\n"
        "FIX: insert _NODES_DIR on sys.path BEFORE the owner-import ladder, the "
        "way otr_post_upscale_procgen_blend.py does.\n\n"
        "--- stderr ---\n%s" % (rel, STANDALONE_LOADED[rel], done.stderr[-1500:]))


def test_the_probe_would_actually_fail_on_a_broken_module(tmp_path):
    """Not vacuous: a module with the two-rung ladder and NO bootstrap must be
    seen to fail. Without this, a probe that silently always passed would look
    exactly like the guard working."""
    broken = tmp_path / "broken_ladder.py"
    broken.write_text(
        "try:\n"
        "    from ._otr_shared import env as otr_env\n"
        "except ImportError:\n"
        "    from _otr_shared import env as otr_env  # type: ignore\n",
        encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, str(broken)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO.parent), env=_scrubbed_env(), timeout=120)
    assert done.returncode != 0
    assert "ModuleNotFoundError" in done.stderr


def test_the_list_only_names_files_that_exist():
    """A renamed or retired module must leave this list, or it is decoration."""
    for rel, why in STANDALONE_LOADED.items():
        assert (REPO / rel).is_file(), rel
        assert why.strip(), rel
