"""Transitive dead-code sweep: follow the dependency chain to a fixpoint.

A one-pass audit finds the leaves. It cannot find what those leaves were
holding up: rip A, and the helper B that only A called is dead too, and the
constant C that only B read dies with it. On 2026-09-04 that happened on the
first cut -- deleting ``corpus_ledgers`` orphaned ``_derive_beats`` in the same
file, and nothing said so (operator: "even though there may be dependency B,
what does dependency B lead to, dependency C -- follow the chain to find truly
rippable code").

So this walks the graph instead of the list:

  1. Every top-level ``def`` / ``class`` / CONSTANT in the scanned tree is a
     node. Its EDGES are every name its own body mentions.
  2. Module-level code, ``__all__``, the node-class contract and everything
     under ``tests/`` are ROOTS -- they run, or they are the contract.
  3. A node reachable from no root is dead. Remove it from the model and
     re-run: its edges disappear, which can strand the next node. Repeat until
     nothing changes. That fixpoint is the chain.

READ-ONLY. It prints candidates; a human deletes. It is deliberately
conservative and says where it is blind: a name reached by ``getattr``, a
registry string, a workflow JSON key or an ``importlib`` call looks dead here
and is not. Verify every candidate before cutting -- the printed ``git grep``
line is that check.

Usage:
    python scripts/dead_code_closure.py                  # candidates + chains
    python scripts/dead_code_closure.py --include-tests  # count tests as code
                                                         # rather than as roots
"""
from __future__ import annotations

import ast
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIRS = ["nodes", "scripts"]
ROOT_FILES = ["__init__.py", "prestartup_script.py"]

#: Docs that can PROTECT a symbol the code no longer references. A name that
#: appears in one of these is reported separately and never as a clean
#: candidate, because the repo has already ruled on it: a guard nobody calls
#: that must stay ("deleting it would silently accept a loss nobody chose"), a
#: declared contract shape, or -- the expensive one -- an unwired FIX for an
#: OPEN production bug. On 2026-09-04 a 73-agent verification pass cleared
#: fifteen such symbols as dead, because every agent searched the CODE for
#: reachability and none of them read the rulings; a QA pass caught it before
#: the commit. Reachability is not the only question. Permission is the other.
PROTECTIVE_DOCS = [
    "docs/OTR_STANDING_RULINGS.md",
    "docs/2026-08-22-dead-symbol-inventory.md",
    "docs/PROD_BUG_LOG.md",
    "CLAUDE.md",
]

#: Whole modules a ruling protects wholesale, which no per-name scan can see.
#: The ruling for the first one is literal: "every symbol in
#: _otr_scifi_p0_contract stays untouched because that module is the subject of
#: the finding above -- deleting p0_contract_instruction would destroy the
#: evidence for it", and PROD_BUG_LOG ties p0_source_char_budget to OPEN bug
#: PBUG-20260729-03 as the diagnosed-but-unwired fix.
PROTECTED_MODULES = {
    "nodes/_otr_scifi_p0_contract.py":
        "OTR_STANDING_RULINGS: every symbol stays -- it is the evidence for "
        "OPEN PBUG-20260729-03, whose fix is unwired in this module",
}

#: Names ComfyUI or a launcher reaches without a Python reference to them.
DYNAMIC_HINTS = {
    "INPUT_TYPES", "RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY",
    "OUTPUT_NODE", "IS_CHANGED", "DESCRIPTION", "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", "main",
}

#: Decorators that REGISTER their symbol somewhere at import time. A class
#: decorated with one of these is reached by the side effect, and its own name
#: is then never mentioned again -- which is exactly what a naive sweep reads
#: as dead. ``@register`` on the engine adapters (47 of them) is the live
#: example: it instantiates the class into the engine registry, so
#: ``KokoroEngine`` is production code that no line of production code names.
#: Anything decorated is treated as a ROOT unless the decorator is in
#: :data:`INERT_DECORATORS` below, because a decorator is a call site the
#: reference graph cannot see through.
INERT_DECORATORS = {
    "dataclass", "dataclasses.dataclass", "runtime_checkable", "property",
    "staticmethod", "classmethod", "abstractmethod", "lru_cache",
    "functools.lru_cache", "cache", "functools.cache",
    "contextmanager", "contextlib.contextmanager",
}


def _files():
    """Everything scanned. ``tests/`` is ALWAYS read -- the flag decides whether
    it counts as a root or as code, never whether it is opened. (The first cut
    skipped the directory entirely when tests were roots, so a symbol used only
    by a test read as unreferenced: ``expected_category`` has five test callers
    and was proposed for deletion.)"""
    out = []
    for name in CODE_DIRS:
        out += sorted((ROOT / name).rglob("*.py"))
    out += [ROOT / name for name in ROOT_FILES]
    out += sorted((ROOT / "tests").rglob("*.py"))
    return [p for p in out if p.is_file()]


def _names_in(node: ast.AST) -> set:
    """Every identifier a node mentions -- attributes and string literals too.

    Strings count because this pack reaches symbols by name through registries
    and ``getattr``. Counting them makes the sweep MISS real dead code rather
    than propose ripping something live, which is the right way to be wrong.
    """
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            text = child.value.strip()
            found.add(text)
            # A string can BE a reference: a forward-referenced annotation
            # (`characters: "list[CharacterAliases]"` in a pydantic model) or a
            # registry key built from a name. Add every identifier inside it,
            # not just the whole literal -- the fan-out rescued
            # `CharacterAliases` from this exact blind spot on 2026-09-04.
            if len(text) <= 200:
                found.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text))
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for imported in child.names:
                found.add(imported.asname or imported.name.split(".")[0])
                found.add(imported.name.split(".")[-1])
    return found


def build(include_tests: bool):
    definitions: dict = {}
    edges: dict = {}
    root_names: set = set()
    by_name = collections.defaultdict(set)

    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print("PARSE FAIL " + rel + ": " + str(exc), file=sys.stderr)
            continue
        is_test = rel.startswith("tests/") and not include_tests
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if is_test:
                    root_names |= _names_in(node)
                    continue
                decorators = {ast.unparse(d).split("(")[0]
                              for d in node.decorator_list}
                if decorators - INERT_DECORATORS:
                    # Registered at import by a decorator's side effect: live,
                    # and its edges are live with it.
                    root_names |= _names_in(node)
                    continue
                key = rel + ":" + node.name
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                definitions[key] = (rel, node.name, node.lineno, kind)
                edges[key] = _names_in(node) - {node.name}
                by_name[node.name].add(key)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if is_test:
                    root_names |= _names_in(node)
                    continue
                # A module-level assignment RUNS: its value is live code.
                if node.value is not None:
                    root_names |= _names_in(node.value)
                targets = ([node.target] if isinstance(node, ast.AnnAssign)
                           else list(node.targets))
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if name.isupper() or name.startswith("_"):
                        key = rel + ":" + name
                        definitions[key] = (rel, name, node.lineno, "const")
                        edges.setdefault(key, set())
                        by_name[name].add(key)
            else:
                root_names |= _names_in(node)
    return definitions, edges, root_names, by_name


def sweep(include_tests: bool):
    definitions, edges, root_names, _by_name = build(include_tests)
    dead: set = set()
    rounds: list = []
    while True:
        mentioned = set(root_names)
        for key, names in edges.items():
            if key not in dead:
                mentioned |= names
        newly = set()
        for key, (_rel, name, _lineno, _kind) in definitions.items():
            if key in dead or name in DYNAMIC_HINTS or name.startswith("__"):
                continue
            if name not in mentioned:
                newly.add(key)
        if not newly:
            break
        rounds.append(sorted(newly))
        dead |= newly
    return definitions, rounds


def protected_names() -> dict:
    """Symbol -> the doc that speaks for it. See :data:`PROTECTIVE_DOCS`."""
    out: dict = {}
    for rel in PROTECTIVE_DOCS:
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", text):
            out.setdefault(word, rel)
    return out


def main() -> int:
    include_tests = "--include-tests" in sys.argv[1:]
    definitions, rounds = sweep(include_tests)
    total = sum(len(batch) for batch in rounds)
    scope = "nodes/ + scripts/ + roots"
    scope += "; tests/ scanned as code" if include_tests else "; tests/ are ROOTS"
    print("transitive dead-code candidates: %d over %d round(s) [%s]\n"
          % (total, len(rounds), scope))
    guarded = protected_names()
    flagged = []
    for batch in rounds:
        for key in list(batch):
            rel, name = definitions[key][0], definitions[key][1]
            doc = PROTECTED_MODULES.get(rel) or guarded.get(name)
            if doc:
                flagged.append((key, doc))
                batch.remove(key)
    for index, batch in enumerate(rounds, 1):
        if index == 1:
            label = "directly unreferenced"
        else:
            label = "orphaned BY round %d's removals -- the dependency chain" % (index - 1)
        print("--- round %d: %d (%s) ---" % (index, len(batch), label))
        for key in batch:
            rel, name, lineno, kind = definitions[key]
            print("  %s:%d  %s %s" % (rel, lineno, kind, name))
            print("      verify: git grep -n -w -F %s -- '*.py'" % name)
        print()
    if flagged:
        print("NAMED IN A PROTECTIVE DOC -- unreferenced, but the repo has")
        print("already ruled on these. READ THE DOC before proposing a cut:")
        for key, doc in sorted(flagged):
            rel, name, lineno, kind = definitions[key]
            print("  %s:%d  %s %s" % (rel, lineno, kind, name))
            print("      ruled on in: %s" % doc)
        print()
    print("BLIND SPOTS -- verify before cutting: getattr(module, 'name'),")
    print("registry strings, workflow JSON keys, importlib, and anything")
    print("reached only from a doc or a launcher are NOT seen as references.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
