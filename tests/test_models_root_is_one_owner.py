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

THE DEPTH IS THE SUBTLE ONE. Step 4 walks four ``dirname()`` calls up from
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


def _live_comfy(checkpoints_dir):
    """A folder_paths that answers like a running ComfyUI whose config lists
    ``checkpoints_dir`` first."""
    import types
    fake = types.ModuleType("folder_paths")
    fake.get_folder_paths = lambda name: [str(checkpoints_dir)]
    fake.models_dir = r"C:\somewhere\ComfyUI\models"
    return fake


class TestAConfiguredTreeBeatsAFolderThatMerelyExists:
    r"""The 4060 measured it on 2026-09-25: a leftover, EMPTY
    ``C:\ComfyUI-Models`` won on bare existence and sent the writer's LLM
    folder out of ComfyUI Desktop's ``ComfyUI-Shared\models``, while every
    other category resolved there. A directory existing is not a
    configuration."""

    def test_inside_comfy_the_configured_tree_wins_over_the_legacy_dir(
            self, clean_env, tmp_path):
        shared = tmp_path / "ComfyUI-Shared" / "models"
        clean_env.setitem(sys.modules, "folder_paths",
                          _live_comfy(shared / "checkpoints"))
        clean_env.setattr(pathlib.Path, "is_dir", lambda self: True)
        assert mr._models_root() == shared

    def test_the_reference_machine_resolves_the_same_tree(self, clean_env):
        r"""The 5080's live /internal/folder_paths lists
        ``C:\ComfyUI-Models\checkpoints`` first, so the new step lands on the
        tree the legacy literal used to find."""
        clean_env.setitem(sys.modules, "folder_paths",
                          _live_comfy(r"C:\ComfyUI-Models\checkpoints"))
        assert mr._models_root() == pathlib.Path(r"C:\ComfyUI-Models")

    def test_an_env_pin_still_beats_the_configured_tree(self, clean_env, tmp_path):
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "pinned"))
        clean_env.setitem(sys.modules, "folder_paths",
                          _live_comfy(tmp_path / "shared" / "checkpoints"))
        assert mr._models_root() == pathlib.Path(str(tmp_path / "pinned"))

    def test_outside_comfy_an_empty_legacy_dir_is_a_leftover(self, clean_env):
        legacy = pathlib.Path(r"C:\ComfyUI-Models")
        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setattr(pathlib.Path, "is_dir",
                          lambda self: self == legacy)
        clean_env.setattr(pathlib.Path, "iterdir", lambda self: iter(()))
        sibling = [None]

        def only_sibling(p):
            sibling[0] = os.path.normpath(p)
            return True
        clean_env.setattr(os.path, "isdir", only_sibling)
        assert str(mr._models_root()) == sibling[0]

    def test_outside_comfy_a_populated_legacy_dir_still_wins(self, clean_env):
        legacy = pathlib.Path(r"C:\ComfyUI-Models")
        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setattr(pathlib.Path, "is_dir",
                          lambda self: self == legacy)
        clean_env.setattr(pathlib.Path, "iterdir",
                          lambda self: iter([self / "checkpoints"]))
        assert mr._models_root() == legacy


def test_the_module_sits_where_its_file_arithmetic_assumes():
    """This module must live in nodes/, where step 4's walk is counted from.

    Asserted structurally rather than behaviourally: on a developer box an
    earlier step usually wins, so a behavioural test would pass with the module
    at the wrong depth and prove nothing.
    """
    here = os.path.abspath(mr.__file__)
    assert os.path.basename(os.path.dirname(here)) == "nodes", (
        "_otr_models_root.py must live in nodes/. Step 4 walks four dirname() "
        "calls up from __file__ to reach ComfyUI's models/ beside "
        "custom_nodes/; moving this file changes that answer silently. Got %r"
        % here)
    # THREE HERE IS NOT STEP 4'S FOUR, and the collision of those two numbers
    # is how the off-by-one survived. Three levels up from the module reaches
    # the directory HOLDING this pack -- custom_nodes/ -- which is the only
    # thing this test is about. Step 4 walks one further, to the ComfyUI root;
    # that level is pinned by
    # test_step_4_resolves_the_comfy_models_dir_not_one_inside_custom_nodes.
    holding = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert holding == os.path.dirname(repo), (
        "this module should sit two levels below the directory holding the "
        "pack; got %r for a repo at %r" % (holding, repo))


def test_step_4_resolves_the_comfy_models_dir_not_one_inside_custom_nodes(
        monkeypatch):
    r"""The off-by-one, pinned BY BEHAVIOUR rather than by source text.

    Step 4 walked THREE dirname() calls until 2026-09-23, landing on
    ``<comfy>/custom_nodes/models`` -- a directory inside custom_nodes that a
    normal install does not have -- instead of the ``models/`` beside it. So the
    step that exists to make a Linux or Docker box work with no environment
    variable never resolved on one.

    TWO EARLIER TESTS COULD NOT CATCH IT, each for its own reason, and both are
    worth remembering:
      * the sibling test in the neighbouring file monkeypatches ``isdir`` to
        accept ANY path ending in "models", so it passes with three dirname()
        calls and would pass with five -- it pins the ORDERING, not the path;
      * the first version of THIS test matched the literal source text for four
        nested ``os.path.dirname(`` calls. That killed the revert-to-three
        mutation, but a review proved it also fails a loop, a small helper, and
        ``pathlib.Path(__file__).resolve().parents[3]`` -- three refactors that
        compute the identical directory. A test that forbids correct code is a
        defect of its own kind.

    So this drives the real function and asks which directory it returns, with
    only one of the two candidates made to exist at a time.
    """
    here = os.path.abspath(mr.__file__)
    up = here
    for _ in range(4):
        up = os.path.dirname(up)
    right = os.path.join(up, "models")            # <comfy>/models
    wrong = os.path.join(os.path.dirname(here), "..", "..", "models")
    wrong = os.path.normpath(wrong)               # <comfy>/custom_nodes/models

    # A CHECKOUT SHALLOW ENOUGH THAT dirname() HITS A DRIVE ROOT collapses the
    # two candidates into one path, and this test would then FAIL on correct
    # code -- a false CI failure, reproduced by a QA pass in a worktree two
    # levels below a drive root. No real deployment is that shallow: the 5080,
    # the 4060 and the pod layouts are all far deeper. Skipping is the honest
    # answer, because the test genuinely cannot discriminate there and should
    # say so rather than report a defect that is not present.
    if os.path.normpath(right) == os.path.normpath(wrong):
        pytest.skip(
            "checkout too shallow to tell the two candidates apart (%r); this "
            "test cannot discriminate at a drive root" % right)

    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self: False)
    monkeypatch.setitem(sys.modules, "folder_paths", None)

    # Only the CORRECT directory exists -> it must be found.
    monkeypatch.setattr(os.path, "isdir", lambda p: os.path.normpath(p) == right)
    assert str(mr._models_root()) == right, (
        "step 4 did not resolve the models dir beside custom_nodes")

    # Only the WRONG directory exists -> it must NOT be returned. On Windows the
    # function falls through to the literal; off Windows it refuses. Either is
    # acceptable; returning `wrong` is not.
    monkeypatch.setattr(os.path, "isdir", lambda p: os.path.normpath(p) == wrong)
    try:
        got = str(mr._models_root())
    except mr.ModelsRootUnresolved:
        got = None
    assert got != wrong, (
        "step 4 returned the directory INSIDE custom_nodes (%r); that is the "
        "off-by-one this test exists for" % wrong)


def test_no_other_module_defines_a_second_models_root():
    """One owner, not two that agree today and drift tomorrow.

    The pack carried two models-root implementations once before. Their
    docstrings both claimed "the same override chain" for weeks while one
    existence-gated the legacy literal and the other returned it
    unconditionally. A second definition is the defect, not the disagreement.
    The module that held the duplicate has since been deleted outright, so
    this now watches the whole package rather than that one file -- which is
    the check that should have existed first.
    """
    import pathlib as _pl

    owner = _pl.Path(mr.__file__).resolve()
    offenders = []
    for path in sorted((owner.parent).rglob("*.py")):
        if path.resolve() == owner:
            continue
        if "def _models_root" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == [], (
        "a second _models_root implementation is back, in %r; every consumer "
        "must import the one owner instead" % (offenders,))


def test_every_caller_imports_the_owner_at_its_real_site():
    """The consumers import LAZILY, so this reads their source.

    Each deliberately imports inside a function to keep its own cold import
    clean, so there is no module attribute to compare. Source inspection is the
    right tool for "is the import at its real site naming the right function",
    which is the one question here. Per-type readers and writers ask
    ``model_type_dir``; the audio base wants the tree itself and asks
    ``_models_root``. The HF cache default does not: on Windows it is
    ``C:\\ComfyUI-Models\\huggingface`` only when that directory exists,
    duplicated in prestartup, which must not import this package.
    """
    import importlib
    import inspect

    expected = {
        "nodes._otr_audio_engines.base": "_otr_models_root import _models_root",
        "nodes._otr_video_engines.wan_shared": "_otr_models_root import model_type_dir",
        "nodes._otr_video_engines.eng_humo": "_otr_models_root import model_type_dir",
        "nodes._otr_video_engines.eng_mesh_stage": "_otr_models_root import model_type_dir",
        "nodes._otr_visual_assets": "_otr_models_root import model_type_dir",
    }
    for name, needle in expected.items():
        src = inspect.getsource(importlib.import_module(name))
        assert needle in src, "%s should import %r" % (name, needle)


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
