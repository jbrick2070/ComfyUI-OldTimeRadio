"""Windows-side tree verification: AST-parse all lab .py, JSON-parse all
fixtures, no BOM, no 0-byte files. Run with the system Python; no deps."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "kibitz-runs"}

errors: list[str] = []
py_count = json_count = 0
for path in ROOT.rglob("*"):
    if not path.is_file() or SKIP_DIRS.intersection(path.parts):
        continue
    data = path.read_bytes()
    if len(data) == 0:
        errors.append(f"0-byte: {path}")
        continue
    if data[:3] == b"\xef\xbb\xbf":
        errors.append(f"BOM: {path}")
    if path.suffix == ".py":
        py_count += 1
        try:
            ast.parse(data.decode("utf-8"))
        except SyntaxError as exc:
            errors.append(f"AST {path}: {exc}")
    elif path.suffix == ".json":
        json_count += 1
        try:
            json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"JSON {path}: {exc}")

print(f"py={py_count} json={json_count} errors={len(errors)}")
for e in errors:
    print(" -", e)
sys.exit(1 if errors else 0)
