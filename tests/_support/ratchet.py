"""The shrinking-set ratchet both migration guards share.

A migration that touches a hundred files cannot ship its guard on the last
day: the two boxes both push to ``v2.0-alpha``, and a guard that arrives at
the end protects nothing for the days in between. So the guard ships FIRST,
carrying an explicit set of files not yet migrated, and asserts BOTH
directions every run:

* no file OUTSIDE the pending set offends -- the migration cannot regress;
* every file INSIDE the pending set still offends -- a migrated file left in
  the list fails, so the set cannot rot into decoration.

A shrinking COUNT would allow offender swapping (fix one file, add a read in
another, the number is unchanged). A named set cannot: the new offender is
outside the set and fails immediately. Same shape as the allowlists in
``tests/test_output_root_single_owner.py`` and ``tests/test_node_temp_hygiene.py``,
with the second direction those already assert.

Paths are POSIX-relative to the repo root so the 5080, the 4060 and the pod
compare equal.
"""
from __future__ import annotations

import ast
import pathlib
from typing import Callable, Iterable

REPO = pathlib.Path(__file__).resolve().parents[2]

#: An offender finder: (tree, relative posix path) -> list of "path:line reason".
Finder = Callable[[ast.AST, str], list]


def scan(roots: Iterable[pathlib.Path], finder: Finder) -> dict:
    """Every offending file under ``roots``, mapped to its reasons.

    ``roots`` may name directories (walked for ``*.py``) or single files, so a
    guard can cover ``nodes/`` plus the two root modules by explicit path.
    """
    found: dict = {}
    for root in roots:
        paths = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for path in paths:
            if not path.is_file():
                continue
            rel = path.relative_to(REPO).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:  # a broken file is a finding, never a skip
                found[rel] = [f"{rel}: does not parse ({exc.msg} at line {exc.lineno})"]
                continue
            reasons = finder(tree, rel)
            if reasons:
                found[rel] = reasons
    return found


def assert_ratchet(found: dict, pending: set, *, owner_hint: str) -> None:
    """Both directions. ``found`` from :func:`scan`; ``pending`` POSIX-relative."""
    new = sorted(set(found) - set(pending))
    assert not new, (
        "a site outside the pending set still decides this for itself -- "
        f"{owner_hint}\n  " + "\n  ".join(
            r for f in new for r in found[f]))

    stale = sorted(set(pending) - set(found))
    assert not stale, (
        "these files are migrated and must leave the pending set, or the set "
        "is decoration:\n  " + "\n  ".join(stale))

    missing = sorted(p for p in pending if not (REPO / p).is_file())
    assert not missing, (
        "the pending set names files that do not exist:\n  " + "\n  ".join(missing))
