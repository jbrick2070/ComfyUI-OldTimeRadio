# S28 LLM-slot tag sweep test helper.
from __future__ import annotations

import ast
import os
import re
from pathlib import Path


SEARCH_WINDOW: int = 8

EXEMPT_FILES: frozenset[str] = frozenset({
    "_otr_structured_call.py",
    "_otr_repair_prompts.py",
    "_otr_model_loader.py",
    "_otr_loader_backends.py",
    "_otr_creative_prompt_router.py",
    "_otr_json.py",
    "OTR_LedgerScriptWriter.py",
    "vram_context_test.py",
})

CALL_SITE_NAMES: frozenset[str] = frozenset({
    "structured_call",
    "generate_fn",
    "creative_fn",
    "technical_fn",
    "polish_generate_fn",
    "request_slot",
})

TAG_RE = re.compile(r"#\s*LLM\s+slot:\s*(creative|technical|per-sub-pass)")
_SKIP_DIRS: frozenset[str] = frozenset({"__pycache__", "tests", "visual"})


def _iter_py_files(nodes_dir: str):
    """Yield non-exempt Python files under *nodes_dir*."""
    for dirpath, dirnames, filenames in os.walk(nodes_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if fname.endswith(".py") and fname not in EXEMPT_FILES:
                yield Path(dirpath) / fname


def _extract_call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) and func.id in CALL_SITE_NAMES:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in CALL_SITE_NAMES:
        return func.attr
    return None


def find_llm_call_sites(nodes_dir: str) -> list[tuple[str, int, str]]:
    """Return all LLM call sites as (filepath, lineno, call_name)."""
    results: list[tuple[str, int, str]] = []
    for py_file in _iter_py_files(nodes_dir):
        try:
            src = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(py_file))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _extract_call_name(node)
                if name is not None:
                    results.append((str(py_file), node.lineno, name))
    return results


def find_parse_failures(nodes_dir: str) -> list[tuple[str, str]]:
    """Return (filepath, error) for every non-exempt file that fails to parse."""
    failures: list[tuple[str, str]] = []
    for py_file in _iter_py_files(nodes_dir):
        try:
            src = py_file.read_text(encoding="utf-8", errors="replace")
            ast.parse(src, filename=str(py_file))
        except (SyntaxError, ValueError) as exc:
            failures.append((str(py_file), f"{type(exc).__name__}: {exc}"))
    return failures


def find_untagged_call_sites(nodes_dir: str) -> list[tuple[str, int, str]]:
    """Return only call sites missing a nearby # LLM slot: tag."""
    untagged: list[tuple[str, int, str]] = []
    lines_cache: dict[str, list[str]] = {}
    for filepath, lineno, call_name in find_llm_call_sites(nodes_dir):
        if filepath not in lines_cache:
            lines_cache[filepath] = (
                Path(filepath)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        lines = lines_cache[filepath]
        lo = max(0, lineno - 1 - SEARCH_WINDOW)
        hi = min(len(lines), lineno - 1 + SEARCH_WINDOW + 1)
        if not any(TAG_RE.search(lines[i]) for i in range(lo, hi)):
            untagged.append((filepath, lineno, call_name))
    return untagged
