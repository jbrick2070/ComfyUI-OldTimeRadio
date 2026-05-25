# S28 LLM-slot tag sweep (CI audit script).
#
# Walks every *.py file under nodes/ and AST-parses each one to find
# LLM call sites (structured_call, generate_fn, creative_fn,
# technical_fn, polish_generate_fn, request_slot).  For each call
# site at line L, scans the source text in a ±SEARCH_WINDOW window
# for a ``# LLM slot: (creative|technical|per-sub-pass)`` tag.
# Any call site missing the tag is reported as an untagged gap.
#
# Exit code 0 = all call sites tagged.  Exit code 1 = gaps found.
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
NODES_DIR = ROOT / "nodes"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_WINDOW: int = 8

EXEMPT_FILES: frozenset[str] = frozenset({
    "_otr_structured_call.py",
    "_otr_repair_prompts.py",
    "_otr_model_loader.py",
    "_otr_loader_backends.py",
    "_otr_creative_prompt_router.py",
    "_otr_json.py",
    # Internal plumbing: the writer's _SlotScheduler._account_and_get_entry
    # calls request_slot as a pass-through; per-sub-pass tag sits at the
    # writer's dispatch level. The title-regen generate_fn call is tagged
    # inline.
    "OTR_LedgerScriptWriter.py",
    # Test utility, not a consumer call site.
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

TAG_RE = re.compile(r'#\s*LLM\s+slot:\s*(creative|technical|per-sub-pass)')

# Directories to skip when walking nodes/.
_SKIP_DIRS: frozenset[str] = frozenset({"__pycache__", "tests", "visual"})


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def _iter_py_files(nodes_dir: str):
    """Yield Path objects for every non-exempt *.py file under *nodes_dir*."""
    for dirpath, dirnames, filenames in os.walk(nodes_dir):
        # Prune skipped directories in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname in EXEMPT_FILES:
                continue
            yield Path(dirpath) / fname


def _extract_call_name(node: ast.Call) -> str | None:
    """Return the function name if *node* matches a CALL_SITE_NAMES pattern."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in CALL_SITE_NAMES:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in CALL_SITE_NAMES:
        return func.attr
    return None


def find_llm_call_sites(nodes_dir: str) -> list[tuple[str, int, str]]:
    """Return all LLM call sites as (filepath, lineno, call_name) tuples."""
    results: list[tuple[str, int, str]] = []
    for py_file in _iter_py_files(nodes_dir):
        try:
            src = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(py_file))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _extract_call_name(node)
            if name is not None:
                results.append((str(py_file), node.lineno, name))
    return results


def find_untagged_call_sites(nodes_dir: str) -> list[tuple[str, int, str]]:
    """Return only the call sites missing a ``# LLM slot:`` tag."""
    all_sites = find_llm_call_sites(nodes_dir)
    untagged: list[tuple[str, int, str]] = []
    # Cache file lines per path to avoid re-reading.
    lines_cache: dict[str, list[str]] = {}
    for filepath, lineno, call_name in all_sites:
        if filepath not in lines_cache:
            lines_cache[filepath] = (
                Path(filepath)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        lines = lines_cache[filepath]
        # lineno is 1-indexed; convert to 0-indexed for the list.
        lo = max(0, lineno - 1 - SEARCH_WINDOW)
        hi = min(len(lines), lineno - 1 + SEARCH_WINDOW + 1)
        found = False
        for i in range(lo, hi):
            if TAG_RE.search(lines[i]):
                found = True
                break
        if not found:
            untagged.append((filepath, lineno, call_name))
    return untagged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    nodes_dir = str(NODES_DIR)
    all_sites = find_llm_call_sites(nodes_dir)
    untagged = find_untagged_call_sites(nodes_dir)
    print(
        f"LLM call sites: {len(all_sites)}  "
        f"tagged: {len(all_sites) - len(untagged)}  "
        f"untagged: {len(untagged)}"
    )
    if untagged:
        print("--- untagged call sites ---")
        for filepath, lineno, call_name in untagged:
            rel = os.path.relpath(filepath, str(ROOT))
            print(f"  {rel}:{lineno} {call_name}")
        return 1
    print("All LLM call sites have a # LLM slot: tag.  OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
