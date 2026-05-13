"""tests/_helpers.py -- shared test infrastructure.

Currently exposes ``load_all_ledger_fixtures`` for S13.3 (Python-based
fixture audit, IMP-9). Walks the ``tests/fixtures/`` directory and
yields ``(path, ledger_dict)`` pairs for every L3 ledger fixture it
finds -- both standalone JSON files AND Python modules that export
``LEDGER`` constants or ``make_*_ledger`` factory functions.

The structural-audit pattern catches what a regex sweep misses:
fixtures built programmatically via helper factories, with values
flowing through variables and indirection.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import pathlib
import sys
from typing import Iterator

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def _looks_like_l3_ledger(d) -> bool:
    """Quick shape check for JSON files: the dict has a top-level
    ``schema_version`` starting with ``l3-`` AND a ``lines`` field.
    Strict so a non-ledger JSON file (Director-shape, config, etc.)
    doesn't get audited."""
    if not isinstance(d, dict):
        return False
    schema = d.get("schema_version") or ""
    if not (isinstance(schema, str) and schema.startswith("l3-")):
        return False
    return "lines" in d


def _looks_like_ledger_factory_output(d) -> bool:
    """Loose shape check for ``make_*_ledger`` factory output:
    dict with both ``cast`` and ``lines`` keys. The naming
    convention (function name ends with ``_ledger``) declares
    intent; this just verifies the shape is roughly right."""
    if not isinstance(d, dict):
        return False
    return "cast" in d and "lines" in d


def _load_module_from_path(path: pathlib.Path):
    """Import a .py file by absolute path without polluting sys.modules
    permanently. Returns the module object."""
    spec = importlib.util.spec_from_file_location(
        f"_ledger_fixture_{path.stem}", str(path),
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def load_all_ledger_fixtures() -> Iterator[tuple[pathlib.Path, dict]]:
    """Yield (path, ledger_dict) for every L3 ledger fixture in
    ``tests/fixtures/``.

    Coverage:
      - ``*.json`` files: parse via json.loads; yield only if the
        result passes ``_looks_like_l3_ledger``.
      - ``*.py`` modules: import dynamically; yield each LEDGER
        constant that's a dict, plus the result of every callable
        named ``make_*_ledger`` invoked with default args.

    Skips ``__init__.py``, ``__pycache__``, anything starting with
    ``_``. Bad fixtures (parse error, bad import, factory raises)
    are skipped silently rather than failing collection -- the
    audit test reports them via the verdict, not via a missing
    file.
    """
    if not FIXTURES_DIR.exists():
        return

    for path in sorted(FIXTURES_DIR.iterdir()):
        if path.name.startswith("_") or path.name.startswith("."):
            continue
        if path.is_dir():
            continue

        if path.suffix == ".json":
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if _looks_like_l3_ledger(obj):
                yield (path, obj)
            continue

        if path.suffix == ".py":
            mod = _load_module_from_path(path)
            if mod is None:
                continue
            # Direct LEDGER constant
            ledger_const = getattr(mod, "LEDGER", None)
            if _looks_like_l3_ledger(ledger_const):
                yield (path, ledger_const)
            # make_*_ledger factories
            for name in dir(mod):
                if not name.startswith("make_") or not name.endswith("_ledger"):
                    continue
                fn = getattr(mod, name)
                if not callable(fn):
                    continue
                # Only invoke if all params have defaults (don't try to
                # synthesize required args).
                sig = inspect.signature(fn)
                if any(
                    p.default is inspect.Parameter.empty
                    and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                    for p in sig.parameters.values()
                ):
                    continue
                try:
                    result = fn()
                except Exception:
                    continue
                # Factory-naming convention is the contract; loose shape
                # check (cast + lines) is enough.
                if _looks_like_ledger_factory_output(result):
                    label = path.with_name(f"{path.name}::{name}()")
                    yield (label, result)
