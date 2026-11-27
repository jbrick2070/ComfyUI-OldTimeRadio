"""The startup script changes nothing that belongs to another package.

Through 2.3.64 ``prestartup_script.py`` did two things to the whole ComfyUI
process, on every start, for every other pack:

* it replaced ``transformers.safetensors_conversion`` in ``sys.modules`` with a
  module of no-ops, switching a transformers feature off process-wide to keep
  one noisy thread out of Bark's first download; and
* it put this pack's ``nodes`` folder at the FRONT of ``sys.path`` and left it
  there, with about three hundred module names (bare ones such as
  ``video_engine`` among them) ahead of every other pack's.

The registry's standards ask a pack not to interfere with other custom nodes,
and three reviewers of the published zip named both. The first is now a guard
scoped to Bark's one Hub load (``_otr_bark_lib._no_background_conversion``);
the second is taken back out as soon as the boot fetch has been imported.

Each process-level case runs the real script in a FRESH interpreter. CPU only.
UTF-8 no BOM.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "prestartup_script.py"
NODES = REPO / "nodes"

_ENV = dict(os.environ, PYTHONUTF8="1", OTR_SKIP_KOKORO_PREFETCH="1",
            HF_HUB_OFFLINE="1", OTR_TEST_MODE="1")
_ENV.pop("HF_TOKEN", None)
_ENV.pop("HUGGING_FACE_HUB_TOKEN", None)


def _run(code: str) -> list:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, encoding="utf-8", env=_ENV, cwd=str(REPO),
                         timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()


def test_the_nodes_folder_is_not_left_on_the_import_path():
    code = ("import os, runpy, sys\n"
            "before = list(sys.path)\n"
            "runpy.run_path(%r, run_name='prestartup_script')\n"
            "nodes = os.path.normcase(%r)\n"
            "print([p for p in sys.path if os.path.normcase(p) == nodes])\n"
            "print(sys.path == before)\n" % (str(SCRIPT), str(NODES)))
    lines = _run(code)
    assert lines[-2] == "[]"
    assert lines[-1] == "True"


def test_a_path_entry_that_was_already_there_is_left_alone():
    """Someone else's entry is not ours to remove."""
    code = ("import os, runpy, sys\n"
            "sys.path.append(%r)\n"
            "runpy.run_path(%r, run_name='prestartup_script')\n"
            "print(%r in sys.path)\n" % (str(NODES), str(SCRIPT), str(NODES)))
    assert _run(code)[-1] == "True"


def test_no_transformers_module_is_replaced():
    code = ("import runpy, sys\n"
            "runpy.run_path(%r, run_name='prestartup_script')\n"
            "print(sorted(n for n in sys.modules if n.startswith('transformers')))\n"
            % str(SCRIPT))
    assert _run(code)[-1] == "[]"


def test_no_credential_is_put_into_the_environment():
    # The delta, not the whole environment: the machine running the suite may
    # carry token variables of its own, which are none of this script's doing.
    # TOKEN as a whole word of the name (HF_TOKEN, HUGGING_FACE_HUB_TOKEN):
    # TOKENIZERS_PARALLELISM is a library switch, not a credential.
    code = ("import os, runpy\n"
            "before = set(os.environ)\n"
            "runpy.run_path(%r, run_name='prestartup_script')\n"
            "added = sorted(set(os.environ) - before)\n"
            "print(sorted(k for k in added if 'TOKEN' in k.upper().split('_')))\n"
            "print(added)\n"
            % str(SCRIPT))
    lines = _run(code)
    assert lines[-2] == "[]"
    # And the whole of what it adds is three library defaults (plus HF_HOME
    # when the machine had none).
    assert set(ast.literal_eval(lines[-1])) <= {
        "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS", "HF_HOME"}


def test_the_only_sys_modules_entry_the_script_writes_is_its_own_guard():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    written = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "modules"):
                written.append(ast.unparse(target.slice))
    assert written == ["_OTR_PRESTARTUP_GUARD"], written


def test_every_flat_import_in_the_pack_is_only_a_fallback():
    """Why the path entry can go: no module under nodes/ imports a pack-local
    module by its flat name EXCEPT in the except-arm of a relative import. A
    flat import outside one would work only while ``nodes`` sits on
    ``sys.path``, and would make that entry load-bearing again."""
    skip = {"__pycache__", "_misaki", "_spacy_model"}
    local = set()
    files = []
    for base, dirs, names in os.walk(NODES):
        dirs[:] = [d for d in dirs if d not in skip]
        local.update(dirs)
        for name in names:
            if name.endswith(".py"):
                files.append(Path(base) / name)
                if name != "__init__.py":
                    local.add(name[:-3])

    unguarded = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, rel):
            self.rel = rel
            self.depth = 0

        def visit_Try(self, node):
            for child in node.body + node.orelse + node.finalbody:
                self.visit(child)
            for handler in node.handlers:
                kind = handler.type
                names = ([kind.id] if isinstance(kind, ast.Name) else
                         [e.id for e in kind.elts if isinstance(e, ast.Name)]
                         if isinstance(kind, ast.Tuple) else [])
                fallback = kind is None or bool(
                    {"ImportError", "ModuleNotFoundError", "Exception"} & set(names))
                self.depth += fallback
                for child in handler.body:
                    self.visit(child)
                self.depth -= fallback

        def _check(self, top, line):
            if top in local and not self.depth:
                unguarded.append("%s:%d imports %s" % (self.rel, line, top))

        def visit_Import(self, node):
            for alias in node.names:
                self._check(alias.name.split(".")[0], node.lineno)

        def visit_ImportFrom(self, node):
            if node.level == 0 and node.module:
                self._check(node.module.split(".")[0], node.lineno)

    for path in files:
        Visitor(path.relative_to(REPO).as_posix()).visit(
            ast.parse(path.read_text(encoding="utf-8")))
    assert len(local) > 250 and len(files) > 250
    assert unguarded == []


def _fake_transformers(monkeypatch):
    real = object()
    fake_utils = types.ModuleType("transformers.modeling_utils")
    fake_utils.auto_conversion = real
    fake_pkg = types.ModuleType("transformers")
    fake_pkg.modeling_utils = fake_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_pkg)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_utils)
    return fake_utils, real


def test_barks_guard_is_scoped_to_one_load_and_always_restored(monkeypatch):
    from nodes import _otr_bark_lib as bark

    fake_utils, real = _fake_transformers(monkeypatch)
    monkeypatch.delenv(bark._CONVERSION_SWITCH, raising=False)

    with bark._no_background_conversion():
        assert os.environ[bark._CONVERSION_SWITCH] == "1"      # transformers' own switch
        assert fake_utils.auto_conversion is not real
        assert fake_utils.auto_conversion("repo", ignore_errors_during_conversion=True) is None
    assert fake_utils.auto_conversion is real
    assert bark._CONVERSION_SWITCH not in os.environ           # gone again, not left set

    try:
        with bark._no_background_conversion():
            raise RuntimeError("the load failed")
    except RuntimeError:
        pass
    assert fake_utils.auto_conversion is real                  # restored on failure too
    assert bark._CONVERSION_SWITCH not in os.environ

    del fake_utils.auto_conversion                             # a build without the name
    with bark._no_background_conversion():
        assert not hasattr(fake_utils, "auto_conversion")
        assert os.environ[bark._CONVERSION_SWITCH] == "1"
    assert not hasattr(fake_utils, "auto_conversion")


def test_barks_guard_puts_back_a_switch_the_user_had_set(monkeypatch):
    from nodes import _otr_bark_lib as bark

    _fake_transformers(monkeypatch)
    monkeypatch.setenv(bark._CONVERSION_SWITCH, "0")
    with bark._no_background_conversion():
        assert os.environ[bark._CONVERSION_SWITCH] == "1"
    assert os.environ[bark._CONVERSION_SWITCH] == "0"


def test_two_threads_in_the_guard_leave_nothing_behind(monkeypatch):
    """Sonnet review, 2026-11-27: the guard changes two process-wide things
    and restores what it found. With two guards overlapping on different
    threads, the second found the first's change, and on leaving last it put
    THAT back: the switch stayed "1" and the no-op stayed installed for the
    rest of the process. The second now waits for the first."""
    import threading

    from nodes import _otr_bark_lib as bark

    fake_utils, real = _fake_transformers(monkeypatch)
    monkeypatch.delenv(bark._CONVERSION_SWITCH, raising=False)
    first_inside, first_may_leave = threading.Event(), threading.Event()
    second_inside, second_may_leave = threading.Event(), threading.Event()

    def first():
        with bark._no_background_conversion():
            first_inside.set()
            first_may_leave.wait(10)

    def second():
        with bark._no_background_conversion():
            second_inside.set()
            second_may_leave.wait(10)

    one, two = threading.Thread(target=first), threading.Thread(target=second)
    one.start()
    assert first_inside.wait(10)
    two.start()
    # The overlap: without the lock the second thread is inside at once.
    overlapped = second_inside.wait(0.5)
    first_may_leave.set()
    one.join(10)
    assert second_inside.wait(10)
    second_may_leave.set()
    two.join(10)

    assert not overlapped, "the second guard entered while the first was open"
    assert fake_utils.auto_conversion is real
    assert bark._CONVERSION_SWITCH not in os.environ


def test_the_guard_wraps_both_of_barks_model_loads():
    """The wiring. Sonnet's review of the first version reproduced that the
    CACHED load (local_files_only=True) starts the conversion check as well,
    so guarding only the Hub load left the common case uncovered."""
    import inspect
    import textwrap

    from nodes import _otr_bark_lib as bark

    tree = ast.parse(textwrap.dedent(inspect.getsource(bark._load_bark)))
    guarded, unguarded = [], []

    def walk(node, inside):
        if isinstance(node, ast.With):
            names = [ast.unparse(item.context_expr) for item in node.items]
            inside = inside or "_no_background_conversion()" in names
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "BarkModel.from_pretrained":
            (guarded if inside else unguarded).append(node.lineno)
        for child in ast.iter_child_nodes(node):
            walk(child, inside)

    walk(tree, False)
    assert len(guarded) == 2 and unguarded == []


def test_the_installed_transformers_still_knows_the_switch_the_guard_sets():
    """The guard leans on transformers' own switch. If a future build drops
    it, this says so: the thread swap is then the only cover."""
    import inspect

    import pytest

    modeling_utils = pytest.importorskip("transformers.modeling_utils")
    from nodes import _otr_bark_lib as bark

    assert bark._CONVERSION_SWITCH in inspect.getsource(modeling_utils)
