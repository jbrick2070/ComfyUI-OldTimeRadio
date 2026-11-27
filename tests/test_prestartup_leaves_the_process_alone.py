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


def test_barks_guard_is_scoped_to_one_load_and_always_restored(monkeypatch):
    from nodes import _otr_bark_lib as bark

    real = object()
    fake_utils = types.ModuleType("transformers.modeling_utils")
    fake_utils.auto_conversion = real
    fake_pkg = types.ModuleType("transformers")
    fake_pkg.modeling_utils = fake_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_pkg)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_utils)

    with bark._no_background_conversion():
        assert fake_utils.auto_conversion is not real
        assert fake_utils.auto_conversion("repo", ignore_errors_during_conversion=True) is None
    assert fake_utils.auto_conversion is real

    try:
        with bark._no_background_conversion():
            raise RuntimeError("the load failed")
    except RuntimeError:
        pass
    assert fake_utils.auto_conversion is real          # restored on failure too

    del fake_utils.auto_conversion                     # a build without the name
    with bark._no_background_conversion():
        assert not hasattr(fake_utils, "auto_conversion")
    assert not hasattr(fake_utils, "auto_conversion")


def test_the_guard_wraps_barks_hub_load_and_nothing_else():
    """The wiring: the guard is used at the one load that reaches the Hub."""
    import inspect

    from nodes import _otr_bark_lib as bark

    source = inspect.getsource(bark._load_bark)
    assert source.count("with _no_background_conversion():") == 1
    guarded = source.split("with _no_background_conversion():", 1)[1].split("log.info", 1)[0]
    assert "BarkModel.from_pretrained(" in guarded
    assert "local_files_only" not in guarded           # the Hub load, not the cache load
