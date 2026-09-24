# -*- coding: utf-8 -*-
r"""``_models_root()`` has ONE owner, and moving it changed nothing.

WHY THIS FILE EXISTS. The models root -- the answer to "where are the weights"
that audio engines, video engines, the provisioner, the lane weight fetcher and
the asset index all need -- used to live inside one optional writer backend. That
made a pack-wide question depend on a component that is being retired, so the
backend could not be removed without taking the shared answer with it.

It now lives in ``nodes/_otr_models_root.py``. These tests pin the two things
that make that move safe rather than merely tidy: the PRECEDENCE is unchanged
under every branch, and the file sits at the DEPTH its ``__file__`` arithmetic
assumes.

THE DEPTH IS THE SUBTLE ONE. Step 4 walks three ``dirname()`` calls up from
``__file__`` to find ComfyUI's ``models/`` beside ``custom_nodes/``. Move the
module one directory in or out and that walk silently returns a different
directory -- no error, no failed import, just a wrong root, which on a fresh
Linux pod is how 3.7 GB of weights once landed in a directory literally named
``C:\ComfyUI-Models`` inside the repo.
"""
import os
import pathlib
import sys

import pytest

from nodes import _otr_models_root as mr

_ENV = ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestThePrecedenceSurvivedTheMove:
    """Each step in order, because the order is the whole contract."""

    def test_1_the_first_env_var_wins_outright(self, clean_env, tmp_path):
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "pinned"))
        assert mr._models_root() == pathlib.Path(str(tmp_path / "pinned"))

    def test_1b_the_second_env_var_is_the_fallback(self, clean_env, tmp_path):
        clean_env.setenv("COMFYUI_MODELS_ROOT", str(tmp_path / "second"))
        assert mr._models_root() == pathlib.Path(str(tmp_path / "second"))

    def test_1c_the_first_env_var_beats_the_second(self, clean_env, tmp_path):
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "first"))
        clean_env.setenv("COMFYUI_MODELS_ROOT", str(tmp_path / "second"))
        assert mr._models_root() == pathlib.Path(str(tmp_path / "first"))

    def test_1d_an_env_pin_is_user_expanded(self, clean_env):
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", "~/otr-weights")
        got = mr._models_root()
        assert "~" not in str(got), got

    def test_2_an_env_pin_beats_the_legacy_tree(self, clean_env, tmp_path):
        """The reference machine HAS the legacy tree; a pin must still win.

        This is the step that makes a pod run reproducible: every pod pins the
        env var, and if the legacy literal could win on a box that happens to
        have it, the pin would be advisory rather than authoritative.
        """
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "pinned"))
        assert mr._models_root() == pathlib.Path(str(tmp_path / "pinned"))

    def test_5_off_windows_with_nothing_resolvable_it_REFUSES(self, clean_env):
        r"""It must raise rather than return a relative garbage path.

        ``C:\ComfyUI-Models`` is not an absolute path on POSIX -- it is a legal
        relative FILENAME, colon and backslashes included -- so returning it
        creates a directory of that name wherever the process happens to be
        standing. That is exactly what happened on a Linux pod.
        """
        clean_env.setattr(os, "name", "posix")
        clean_env.setattr(pathlib.Path, "is_dir", lambda self: False)
        clean_env.setattr(os.path, "isdir", lambda p: False)
        clean_env.setitem(sys.modules, "folder_paths", None)
        with pytest.raises(mr.ModelsRootUnresolved) as exc:
            mr._models_root()
        assert "OTR_COMFYUI_MODELS_ROOT" in str(exc.value)


def test_the_module_sits_where_its_file_arithmetic_assumes():
    """Step 4 counts three dirname() calls from THIS file. Pin the depth.

    Asserted structurally rather than behaviourally: on a developer box an
    earlier step usually wins, so a behavioural test would pass with the module
    at the wrong depth and prove nothing.
    """
    here = os.path.abspath(mr.__file__)
    assert os.path.basename(os.path.dirname(here)) == "nodes", (
        "_otr_models_root.py must live in nodes/. Step 4 walks three dirname() "
        "calls up from __file__ to reach ComfyUI's models/ beside "
        "custom_nodes/; moving this file changes that answer silently. Got %r"
        % here)
    walked = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert walked == os.path.dirname(repo), (
        "the three-dirname walk should land on the directory holding this "
        "pack; got %r for a repo at %r" % (walked, repo))


def test_step_4_finds_comfyuis_models_dir_BESIDE_custom_nodes():
    r"""The off-by-one that caused the 2026-09-21 incident, pinned.

    Step 4 walked THREE dirname() calls until 2026-09-23, landing on
    ``<comfy>/custom_nodes/models`` -- a directory INSIDE custom_nodes that a
    normal install does not have. So on a Linux pod with no env var and no
    running ComfyUI it found nothing, fell through, and returned the Windows
    literal, which put 3.7 GB of weights into a directory literally named
    ``C:\ComfyUI-Models`` inside the repo.

    THE OLD TEST COULD NOT CATCH THIS. It monkeypatched ``isdir`` to accept any
    path ending in "models", so it passed with three dirname() calls and would
    have passed with five. It pinned the ORDERING (sibling before literal) and
    said nothing about WHICH directory. This one names the path.
    """
    import re

    src = inspect_source_of_models_root()
    # The sibling expression, however it is formatted across lines.
    joined = re.sub(r"\s+", "", src)
    assert "os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(" in joined, (
        "step 4 must walk FOUR dirname() calls from __file__ to reach the "
        "models/ directory beside custom_nodes/; three lands inside it")

    here = os.path.abspath(mr.__file__)
    walked = here
    for _ in range(4):
        walked = os.path.dirname(walked)
    assert os.path.basename(walked) != "custom_nodes", (
        "the walk still ends inside custom_nodes/: %r" % walked)


def inspect_source_of_models_root():
    import inspect
    return inspect.getsource(mr._models_root)


def test_the_retiring_backend_no_longer_owns_an_implementation():
    """One owner, not two that agree today and drift tomorrow.

    The pack carried two models-root implementations once before. Their
    docstrings both claimed "the same override chain" for weeks while one
    existence-gated the legacy literal and the other returned it
    unconditionally. A second definition is the defect, not the disagreement.
    """
    import inspect
    from nodes import _otr_gguf_backend as backend

    src = inspect.getsource(backend)
    assert "def _models_root" not in src, (
        "the retiring backend defines _models_root again; it must re-export "
        "the one owner, never carry a copy")
    assert backend._models_root is mr._models_root, (
        "the backend's _models_root must BE the owner's function object")
    assert backend.ModelsRootUnresolved is mr.ModelsRootUnresolved


def test_every_caller_imports_the_owner_and_not_the_retiring_backend():
    """The consumers import LAZILY, so this reads their source.

    Both of these deliberately import inside a function to keep their own cold
    imports clean -- each has a test asserting exactly that -- so there is no
    module attribute to compare. Source inspection is the right tool for "is
    the import at its real site naming the right module", which is the one
    question here.
    """
    import inspect

    from nodes._otr_audio_engines import base as audio_base
    from nodes._otr_video_engines import wan_shared

    for module in (audio_base, wan_shared):
        src = inspect.getsource(module)
        assert "_otr_models_root import _models_root" in src, (
            "%s should import the models root from its neutral owner"
            % module.__name__)
        assert "_otr_gguf_backend import _models_root" not in src, (
            "%s still reaches the models root through the retiring backend"
            % module.__name__)


def test_wan_shareds_configured_root_still_matches(monkeypatch, tmp_path):
    """The equality that an earlier drift broke, re-asserted after the move."""
    from nodes._otr_video_engines import wan_shared

    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "w"))
    assert wan_shared.configured_models_root() == str(mr._models_root())


def test_the_owner_stays_cold_import_clean():
    """No torch, no transformers, no model library at module scope.

    A caller must be able to ask where the weights are without loading one, and
    the provisioner asks this before anything heavy exists. ``folder_paths`` is
    the single runtime-only import and it is made lazily inside the function.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mr))

    # WALK EVERYTHING OUTSIDE A FUNCTION, not just tree.body. The first version
    # of this test inspected only top-level Import nodes, so a heavy import
    # hidden inside a module-level `try:` or `if:` was invisible -- and this
    # module's own env-shim import uses exactly that try/except shape, so the
    # blind spot covered its most likely future edit rather than a hypothetical
    # one. A QA pass proved it by hiding `import torch` in a try block and
    # watching this test pass.
    def module_scope_imports(node, inside_function=False):
        found = []
        for child in ast.iter_child_nodes(node):
            entering = isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if not (inside_function or entering):
                if isinstance(child, ast.Import):
                    found += [a.name.split(".")[0] for a in child.names]
                elif isinstance(child, ast.ImportFrom) and child.module:
                    found.append(child.module.split(".")[0])
            found += module_scope_imports(
                child, inside_function or entering)
        return found

    heavy = {"torch", "transformers", "llama_cpp", "numpy", "folder_paths"}
    got = set(module_scope_imports(tree)) & heavy
    assert not got, "module-scope heavy import(s): %r" % (sorted(got),)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
