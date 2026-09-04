"""ONE owner answers where the output tree is: ``nodes/_otr_paths.py``.

The ratchet for the output root (kibitz runpod-found-fixes, 2026-09-04). The
tree had FOUR owners -- ``_otr_paths``, the mux, the ledger and the package
``__init__`` -- and two more readers (the silent composite's and the caption
burn's fallbacks, plus the portrait ledger's) that each asked ``folder_paths``
directly. On a server launched with ``--output-directory`` they disagreed, and
the disagreement was invisible on the box where all the answers coincide.

Rules, by AST walk over ``nodes/``:

* ``folder_paths.get_output_directory`` is called only by the owner, plus a
  NAMED list of readers that have a reason the owner cannot serve. Each entry
  says why; a new reader fails; retiring one shrinks the list.
* ``OTR_OBS_DIR`` is read only by the owner and by the ledger's publication
  VALIDATOR (which checks a written path against the authorized roots).
* ``OTR_OUTPUT_DIR`` is read only by the owner (the package ``__init__``,
  which WRITES it, is outside ``nodes/``).
"""
from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
NODES = REPO / "nodes"
OWNER = NODES / "_otr_paths.py"

#: Readers of folder_paths.get_output_directory that the owner cannot serve.
_GET_OUTPUT_DIRECTORY_ALLOWED = {
    # ComfyUI's SaveGLB refuses any path outside folder_paths' OWN output dir,
    # so the mesh stage must ask folder_paths first and the pin second. Its
    # own comment says so; it is the one legitimate exception.
    NODES / "_otr_video_engines" / "eng_mesh_stage.py",
    # A diagnostic node writing otr/vram_tests -- outside the episodes/obs
    # contract by design, never part of a render. GO_FORWARD_PLAN 1.4a.
    NODES / "vram_context_test.py",
}
_OBS_DIR_READERS = {OWNER, NODES / "_otr_ledger.py"}
_OUTPUT_DIR_READERS = {OWNER, NODES / "_otr_video_engines" / "eng_mesh_stage.py"}


def _offenders(path: pathlib.Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get_output_directory" \
                and path != OWNER and path not in _GET_OUTPUT_DIRECTORY_ALLOWED:
            out.append(f"{rel}:{node.lineno} calls folder_paths.get_output_directory()")
        elif isinstance(node, ast.Constant) and node.value == "OTR_OBS_DIR" \
                and path not in _OBS_DIR_READERS:
            out.append(f"{rel}:{node.lineno} reads OTR_OBS_DIR")
        elif isinstance(node, ast.Constant) and node.value == "OTR_OUTPUT_DIR" \
                and path not in _OUTPUT_DIR_READERS:
            out.append(f"{rel}:{node.lineno} reads OTR_OUTPUT_DIR")
    return out


def test_the_owner_exists():
    assert OWNER.is_file()


def test_the_allowlist_only_names_files_that_exist_and_still_need_it():
    """A retired reader must leave its list, or the list is decoration --
    for EVERY exception set, not only the folder_paths one."""
    sets = (
        (_GET_OUTPUT_DIRECTORY_ALLOWED, "get_output_directory"),
        (_OBS_DIR_READERS - {OWNER}, "OTR_OBS_DIR"),
        (_OUTPUT_DIR_READERS - {OWNER}, "OTR_OUTPUT_DIR"),
    )
    for members, needle in sets:
        for path in members:
            assert path.is_file(), path
            src = path.read_text(encoding="utf-8")
            assert needle in src, (
                f"{path.relative_to(REPO)} no longer reads {needle} -- "
                f"remove it from its exception set")


def test_no_second_owner_of_the_output_root_under_nodes():
    offenders = []
    for py in sorted(NODES.rglob("*.py")):
        offenders.extend(_offenders(py))
    assert offenders == [], (
        "a second answer to 'where is the output tree' exists outside "
        "nodes/_otr_paths.py -- delegate to it:\n  " + "\n  ".join(offenders)
    )
