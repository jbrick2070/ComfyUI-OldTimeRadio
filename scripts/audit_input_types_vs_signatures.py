r"""audit_input_types_vs_signatures.py

Walk every OTR node class, extract its INPUT_TYPES widget keys, find
the matching `FUNCTION` method, and flag any widget the function will
NOT accept as a kwarg.

This is the exact bug shape that broke VideoComposite today: widget
added to INPUT_TYPES, execute() signature never updated, ComfyUI calls
fn(**inputs) and Python raises TypeError.

Output: human-readable report. Exit code 0 = clean, 1 = mismatches found.

Usage:
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe \
        scripts\audit_input_types_vs_signatures.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def _walk_py_files(root: Path) -> list[Path]:
    """Find every .py file under nodes/ and visual/."""
    files: list[Path] = []
    for sub in ("nodes", "visual"):
        d = root / sub
        if not d.exists():
            continue
        files.extend(d.rglob("*.py"))
    return sorted(files)


def _get_string_value(node: ast.AST) -> Optional[str]:
    """Extract a string literal from an ast.Constant or ast.Str node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_widget_keys_from_input_types(
    method: ast.FunctionDef,
) -> set[str]:
    """Find the dict returned by INPUT_TYPES and pull its widget names.

    INPUT_TYPES returns a dict like:
        {"required": {"widget_a": (..., {...})}, "optional": {"widget_b": ...}}

    We pull every key from the inner dicts under "required" / "optional" /
    "hidden". Everything else gets ignored.
    """
    keys: set[str] = set()
    for child in ast.walk(method):
        if not isinstance(child, ast.Return):
            continue
        if not isinstance(child.value, ast.Dict):
            continue
        for top_key, top_val in zip(child.value.keys, child.value.values):
            top_name = _get_string_value(top_key)
            if top_name not in ("required", "optional", "hidden"):
                continue
            if not isinstance(top_val, ast.Dict):
                continue
            for inner_key in top_val.keys:
                name = _get_string_value(inner_key)
                if name is None:
                    continue
                # Hidden widgets like "PROMPT", "EXTRA_PNGINFO" are
                # ComfyUI-supplied -- skip from the audit since they
                # come in as kwargs without being declared in the
                # signature explicitly (ComfyUI handles them).
                if name.isupper() and name in {
                    "PROMPT", "EXTRA_PNGINFO", "UNIQUE_ID", "DYNPROMPT",
                }:
                    continue
                keys.add(name)
    return keys


def _find_function_params(
    cls: ast.ClassDef,
    method_name: str,
) -> Optional[list[str]]:
    """Return the kwargs accepted by `cls.method_name`, or None if missing."""
    for item in cls.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if item.name != method_name:
            continue
        # If method takes **kwargs, all widgets pass through -- skip the audit.
        if item.args.kwarg is not None:
            return ["**kwargs"]
        params: list[str] = []
        for arg in item.args.args:
            if arg.arg == "self":
                continue
            params.append(arg.arg)
        # kwonly args (after *) also count
        for arg in item.args.kwonlyargs:
            params.append(arg.arg)
        return params
    return None


def _get_class_attr_string(
    cls: ast.ClassDef,
    attr_name: str,
) -> Optional[str]:
    """Pull a string class-level attribute value (e.g. FUNCTION = 'execute')."""
    for item in cls.body:
        if not isinstance(item, ast.Assign):
            continue
        if len(item.targets) != 1:
            continue
        tgt = item.targets[0]
        if not isinstance(tgt, ast.Name) or tgt.id != attr_name:
            continue
        val = _get_string_value(item.value)
        if val is not None:
            return val
    return None


def _has_input_types_method(cls: ast.ClassDef) -> bool:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
            return True
    return False


def _audit_class(
    cls: ast.ClassDef,
    file_path: Path,
) -> list[str]:
    """Return a list of issue strings for this class. Empty = clean."""
    issues: list[str] = []
    if not _has_input_types_method(cls):
        return issues  # not a node class
    fn_name = _get_class_attr_string(cls, "FUNCTION")
    if not fn_name:
        return issues  # not a real ComfyUI node class
    # Find the INPUT_TYPES method body and pull widget keys.
    input_types_method = None
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
            input_types_method = item
            break
    if input_types_method is None:
        return issues
    widget_keys = _extract_widget_keys_from_input_types(input_types_method)
    fn_params = _find_function_params(cls, fn_name)
    if fn_params is None:
        issues.append(
            f"  MISSING_FUNCTION: class {cls.name} declares "
            f"FUNCTION='{fn_name}' but no method by that name was found"
        )
        return issues
    if fn_params == ["**kwargs"]:
        return issues  # accepts anything
    fn_param_set = set(fn_params)
    # Widget present but function won't accept it = the bug shape
    missing_in_signature = sorted(widget_keys - fn_param_set)
    # Function param that isn't a widget = potentially dead code OR a
    # required positional that ComfyUI would never supply
    extra_in_signature = sorted(fn_param_set - widget_keys)
    if missing_in_signature:
        issues.append(
            f"  WIDGET_NOT_IN_SIGNATURE ({cls.name}.{fn_name}): "
            f"{missing_in_signature}"
        )
    # Filter out *args / common false-positives. Skip 'self' implicit and
    # any param with a default the dev intentionally added beyond
    # INPUT_TYPES (e.g. helper-only params). For now report them under
    # a softer label so we can scan but not panic.
    if extra_in_signature:
        # Be quiet about params that LOOK like internal helpers
        noisy = [
            p for p in extra_in_signature
            if not p.startswith("_") and p not in {"prompt_id"}
        ]
        if noisy:
            issues.append(
                f"  SIGNATURE_HAS_NON_WIDGET_PARAM ({cls.name}.{fn_name}): "
                f"{noisy}  (low priority -- function accepts kwargs not "
                f"declared in INPUT_TYPES; usually fine if they have defaults)"
            )
    return issues


def main() -> int:
    files = _walk_py_files(REPO_ROOT)
    total_classes = 0
    bad_classes = 0
    bad_files: dict[Path, list[str]] = {}

    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception as exc:
            print(f"PARSE_ERROR {f.relative_to(REPO_ROOT)}: {exc}")
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            total_classes += 1
            issues = _audit_class(node, f)
            if issues:
                bad_classes += 1
                bad_files.setdefault(f, []).extend(issues)

    print(f"Audited {total_classes} classes across {len(files)} files.")
    print()
    if not bad_files:
        print("CLEAN: every INPUT_TYPES widget has a matching kwarg in "
              "its FUNCTION method.")
        return 0

    high_priority = 0
    low_priority = 0
    print("ISSUES FOUND:\n")
    for f, msgs in sorted(bad_files.items()):
        print(f"{f.relative_to(REPO_ROOT)}")
        for m in msgs:
            if "WIDGET_NOT_IN_SIGNATURE" in m or "MISSING_FUNCTION" in m:
                high_priority += 1
                print(f"  [HIGH] {m.strip()}")
            else:
                low_priority += 1
                print(f"  [LOW]  {m.strip()}")
        print()

    print(f"Summary: {high_priority} HIGH-priority issues, "
          f"{low_priority} LOW-priority issues across {bad_classes} class(es).")
    print()
    print("HIGH = TypeError waiting to happen on the next workflow run.")
    print("LOW  = function accepts param ComfyUI won't supply; usually "
          "fine if it has a default.")
    return 1 if high_priority > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
