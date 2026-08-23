"""tests/test_no_duplicate_top_level_defs.py -- the shadowed-definition guard.

LEAN_MEAN_CLEANUP order 1 ("truth and prevention"): add this guard BEFORE the
order-2 deletions, so the pattern cannot come back after the dead copies go.

THE PATTERN, and it is silent by construction. Python rebinds a module-level
name on re-definition, so a file carrying two ``def foo`` at module scope keeps
ONLY the last one. Nothing warns. The earlier body still reads like live code --
it has a docstring, it parses, coverage tools will happily report it -- but it
can never execute, and an editor who "fixes a bug" in the first copy changes
nothing at all. That is strictly worse than dead code that looks dead.

``nodes/story_orchestrator.py`` carried FOUR such definitions (two names, twice
each, ~2200 lines apart). All four were unreachable: the first pair by rebinding,
the second pair by having no caller at all.

WHY THIS SCANS AST AND NOT TEXT. A regex over ``^def `` cannot tell a module-level
definition from one nested in a class, a function, or a ``try``/``except
ImportError`` compatibility shim -- and this repo uses that shim shape
deliberately and everywhere (``try: from ._x import y / except ImportError:``),
including re-defining names under ``if TYPE_CHECKING`` and stub-class fallbacks.
Those are INTENTIONAL conditional rebinds, not shadowing, so only definitions in
``tree.body`` -- the unconditional top level -- are counted.

RATCHET, matching ``test_node_temp_hygiene``: the allowlist below is empty and is
meant to stay that way. A genuine conditional re-definition belongs inside an
``if`` / ``try`` (where this scanner already ignores it), never in the allowlist.
"""
from __future__ import annotations

import ast
import collections
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCAN_ROOTS = ("nodes", "scripts")

#: Directories whose contents are not this repo's live source.
_SKIP_PARTS = {".claude", "__pycache__", "worktrees", "tmp", ".git",
               "kibitz", "kibitz-runs", ".venv"}

#: DELIBERATELY EMPTY. See the module docstring: a legitimate conditional
#: re-definition is already invisible to this scanner because it is not in
#: ``tree.body``. An entry here would be an admission that a module ships a
#: definition it cannot reach, which is the exact defect this guards.
_ALLOWLIST: dict = {}


def _skip(path: Path) -> bool:
    return any(part in _SKIP_PARTS for part in path.parts)


def _duplicate_top_level_defs(path: Path) -> dict:
    """``{name: [lineno, ...]}`` for module-level names defined more than once.

    Counts ``def`` / ``async def`` / ``class`` only. Module-level ASSIGNMENTS are
    deliberately NOT counted: re-assigning a module constant is ordinary Python
    (accumulators, ``x = _build(x)``, platform switches) and flagging it would
    bury the real signal in noise.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return {}
    seen: dict = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[node.name].append(node.lineno)
    return {name: lines for name, lines in seen.items() if len(lines) > 1}


def _scanned_files() -> list:
    """Every file the ratchet actually walks.

    Split out from ``_scan`` so the ratchet's own DISCOVERY can be asserted.
    Without this the guard is one bad path away from a green test that checks
    nothing: a wrong ``_REPO``, a renamed directory, or both ``is_dir()`` checks
    failing would make ``_scan()`` return ``{}`` and the ratchet would pass
    having read zero files.
    """
    out: list = []
    for root in _SCAN_ROOTS:
        base = _REPO / root
        if not base.is_dir():
            continue
        out.extend(p for p in sorted(base.rglob("*.py")) if not _skip(p))
    return out


def _scan() -> dict:
    out: dict = {}
    for path in _scanned_files():
        dupes = _duplicate_top_level_defs(path)
        if dupes:
            out[path.relative_to(_REPO).as_posix()] = dupes
    return out


def test_the_ratchet_actually_reaches_the_repo():
    """Discovery proof -- without this the ratchet below can pass vacuously.

    QA finding, 2026-08-22: the ratchet's other self-tests exercise the AST
    detector against synthetic files, which proves the detector but says nothing
    about whether ``_scan()`` ever found the real tree. A guard that can report
    "clean" because it read nothing is the defect it was written to prevent.
    """
    files = _scanned_files()
    assert len(files) > 50, (
        f"the shadowed-definition ratchet only discovered {len(files)} file(s) "
        f"under {_SCAN_ROOTS} -- it is not reaching the repo, so its 'clean' "
        f"result would be meaningless. Check _REPO ({_REPO}) and _SKIP_PARTS.")
    names = {p.name for p in files}
    # Two files that must always exist; a rename here should fail LOUD and be
    # updated deliberately, not silently reduce the ratchet's coverage.
    for anchor in ("story_orchestrator.py", "video_engine.py"):
        assert anchor in names, (
            f"{anchor} was not among the {len(files)} scanned files -- the "
            f"ratchet is walking the wrong tree or over-skipping.")


def test_no_module_level_definition_is_shadowed_by_a_later_one():
    """A module-level def/class defined twice keeps only the last one."""
    found = _scan()
    offenders = {rel: dupes for rel, dupes in found.items()
                 if rel not in _ALLOWLIST}
    assert not offenders, (
        "Module-level definition(s) shadowed by a later definition of the same "
        "name -- Python keeps only the LAST one, so every earlier body is "
        "unreachable and edits to it do nothing:\n" + "\n".join(
            f"  {rel}: " + ", ".join(
                f"{name} at lines {lines}" for name, lines in sorted(dupes.items()))
            for rel, dupes in sorted(offenders.items())))


def test_the_scanner_actually_detects_the_pattern(tmp_path):
    """The guard must fail on a known-bad file, or it proves nothing.

    Without this, an over-eager ``_skip`` or a broken parse would make
    ``_scan()`` return ``{}`` and the ratchet above would pass vacuously
    forever -- a green test asserting nothing.
    """
    bad = tmp_path / "shadowed.py"
    bad.write_text(
        "def repeated():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def repeated():\n"
        "    return 2\n",
        encoding="utf-8")
    assert _duplicate_top_level_defs(bad) == {"repeated": [1, 5]}


def test_conditional_redefinition_is_not_flagged(tmp_path):
    """The import-shim shape this repo uses everywhere must stay legal."""
    ok = tmp_path / "shimmed.py"
    ok.write_text(
        "try:\n"
        "    from ._real import Streamer\n"
        "except ImportError:\n"
        "    class Streamer:\n"
        "        pass\n"
        "\n"
        "\n"
        "class Streamer2:\n"
        "    pass\n",
        encoding="utf-8")
    assert _duplicate_top_level_defs(ok) == {}
