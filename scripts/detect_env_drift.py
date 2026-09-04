"""A-0 of the registry collapse: the default-drift RECEIPT, read-only.

Lists every environment knob that is read at more than one site under
``nodes/`` (plus the two root files) with a DIFFERENT default or a different
cast, so the spelling-only migration cannot paper over a decision that is
already made twice. Prints; changes nothing. Resolves ``import os as X``
aliases and ``from os import environ / getenv``.

Usage: python scripts/detect_env_drift.py [--all]   (--all lists every knob)
"""
from __future__ import annotations

import ast
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "nodes").rglob("*.py")) + [ROOT / "__init__.py", ROOT / "prestartup_script.py"]
CASTS = {"int", "float", "bool", "str", "Path"}


def _reads(tree: ast.AST):
    """Yield (name, default_repr, cast, lineno) for every constant-name env read."""
    os_names, environ_names, getenv_names = {"os"}, set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "os":
                    os_names.add(a.asname or "os")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for a in node.names:
                if a.name == "environ":
                    environ_names.add(a.asname or "environ")
                elif a.name == "getenv":
                    getenv_names.add(a.asname or "getenv")
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def is_environ(n):
        return (isinstance(n, ast.Attribute) and n.attr == "environ" and isinstance(n.value, ast.Name)
                and n.value.id in os_names) or (isinstance(n, ast.Name) and n.id in environ_names)

    def cast_of(n):
        p = parents.get(n)
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id in CASTS and p.args and p.args[0] is n:
            return p.func.id
        return ""

    for node in ast.walk(tree):
        name = default = None
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "get" and is_environ(f.value):
                pass
            elif (isinstance(f, ast.Attribute) and f.attr == "getenv" and isinstance(f.value, ast.Name)
                  and f.value.id in os_names) or (isinstance(f, ast.Name) and f.id in getenv_names):
                pass
            else:
                continue
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                name = node.args[0].value
                default = ast.unparse(node.args[1]) if len(node.args) > 1 else "<none>"
        elif isinstance(node, ast.Subscript) and is_environ(node.value) and isinstance(node.slice, ast.Constant):
            name, default = str(node.slice.value), "<KeyError>"
        if name:
            yield name, default, cast_of(node), node.lineno


def main(show_all: bool) -> int:
    sites = collections.defaultdict(list)
    for f in FILES:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"PARSE FAIL {f}: {exc}")
            continue
        rel = f.relative_to(ROOT).as_posix()
        for name, default, cast, ln in _reads(tree):
            sites[name].append((rel, ln, default, cast))
    drift = {}
    for name, rows in sorted(sites.items()):
        defaults = {d for _, _, d, _ in rows}
        casts = {c for _, _, _, c in rows}
        if len(rows) > 1 and (len(defaults) > 1 or len(casts) > 1):
            drift[name] = rows
    print(f"{len(sites)} distinct knobs read at {sum(len(r) for r in sites.values())} sites in {len(FILES)} files")
    print(f"{len(drift)} knobs read with more than one default or cast:\n")
    for name, rows in drift.items():
        print(name)
        for rel, ln, default, cast in sorted(rows):
            print(f"    {rel}:{ln}  default={default}  cast={cast or '-'}")
    if show_all:
        print("\nevery knob:")
        for name, rows in sorted(sites.items()):
            print(f"  {name}: {len(rows)} site(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main("--all" in sys.argv[1:]))
