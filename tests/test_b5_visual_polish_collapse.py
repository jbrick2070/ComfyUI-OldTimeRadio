"""S30 B5 -- delete OTR_VisualLLMSelector + collapse _POLISH_CACHE.

Five guardrails that prove the deletion stays deleted:

1. test_polish_path_uses_request_slot     -- AST scan of
   visual/llm_polish.py asserts the polish acquisition path routes
   through _otr_model_loader.request_slot("creative", ...).
2. test_no_polish_cache_symbol            -- the legacy
   _POLISH_CACHE module-level dict must NOT exist in
   visual/llm_polish.py.
3. test_visual_llm_selector_unregistered  -- the
   OTR_VisualLLMSelector mapping entry must NOT appear in the
   package __init__.py.
4. test_no_visual_llm_selector_callers    -- AST scan over nodes/,
   visual/, scripts/, tests/ for any Python identifier named
   `OTR_VisualLLMSelector` or `VisualLLMSelector`.
5. test_polish_cache_collapse_no_dup_load -- runtime test: drive
   the polish path via a mocked loader; assert only one model
   entry exists in _otr_model_loader.LLM_CACHE.

DOCUMENTED DEVIATION FROM PLAN: the plan's
`test_polish_path_uses_request_slot` named `make_polish_generate_fn`
as part of the polish path. Visual polish does NOT use
`make_polish_generate_fn` -- it builds its own chat-template prompt
and calls `model.generate(do_sample=False)` directly. The shared
piece is `request_slot` for model acquisition; the test asserts
that.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
POLISH_PATH = PACK_ROOT / "visual" / "llm_polish.py"


# ---------------------------------------------------------------------------
# 1. AST: polish path uses request_slot
# ---------------------------------------------------------------------------


def test_polish_path_uses_request_slot():
    """The polish acquisition path (the helper that fetches model +
    tokenizer for a polish call) must invoke
    _otr_model_loader.request_slot("creative", ...). AST scan over
    visual/llm_polish.py.
    """
    tree = ast.parse(POLISH_PATH.read_text(encoding="utf-8"))
    found = False
    creative_arg = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "request_slot"
        ) or (
            isinstance(fn, ast.Name) and fn.id == "request_slot"
        ):
            found = True
            # First positional arg should be the string "creative".
            if node.args:
                first = node.args[0]
                if (
                    isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value == "creative"
                ):
                    creative_arg = True
    assert found, (
        "visual/llm_polish.py must call request_slot(...) somewhere "
        "in its polish-acquisition path (B5 _POLISH_CACHE collapse)"
    )
    assert creative_arg, (
        "request_slot call in visual/llm_polish.py must use the "
        "'creative' slot per the S30 routing table"
    )


# ---------------------------------------------------------------------------
# 2. AST: no _POLISH_CACHE symbol
# ---------------------------------------------------------------------------


def test_no_polish_cache_symbol():
    """The legacy _POLISH_CACHE module-level dict must NOT exist in
    visual/llm_polish.py. AST scan for any Name / Attribute /
    Assign with that identifier (outside string / comment forensic
    contexts).
    """
    tree = ast.parse(POLISH_PATH.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_POLISH_CACHE"
        ):
            offenders.append(f"line {node.lineno}: _POLISH_CACHE = ...")
        elif isinstance(node, ast.Name) and node.id == "_POLISH_CACHE":
            offenders.append(
                f"line {getattr(node, 'lineno', '?')}: Name(_POLISH_CACHE)"
            )
    assert not offenders, (
        "visual/llm_polish.py must not carry the legacy _POLISH_CACHE "
        "symbol post-B5:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 3. AST: OTR_VisualLLMSelector unregistered
# ---------------------------------------------------------------------------


def test_visual_llm_selector_unregistered():
    """The OTR_VisualLLMSelector mapping entry must NOT appear in
    the package __init__.py. AST scan for any Constant("OTR_VisualLLMSelector")
    used as a Dict key.
    """
    init_path = PACK_ROOT / "__init__.py"
    src = init_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k in node.keys:
            if (
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and k.value == "OTR_VisualLLMSelector"
            ):
                raise AssertionError(
                    f"__init__.py still registers 'OTR_VisualLLMSelector' "
                    f"as a node mapping key at line "
                    f"{getattr(k, 'lineno', '?')}"
                )


# ---------------------------------------------------------------------------
# 4. AST: no VisualLLMSelector callers as identifiers
# ---------------------------------------------------------------------------


def _python_files() -> list[Path]:
    out: list[Path] = []
    for d in ("nodes", "visual", "scripts", "tests"):
        root = PACK_ROOT / d
        if not root.is_dir():
            continue
        out.extend(sorted(root.rglob("*.py")))
    return out


def test_no_visual_llm_selector_callers():
    """No Python identifier (Name / Attribute / FunctionDef / ClassDef)
    named `OTR_VisualLLMSelector` or `VisualLLMSelector` may exist
    under nodes/, visual/, scripts/, tests/. Forensic string-literal
    mentions are permitted.
    """
    bad_names = ("OTR_VisualLLMSelector", "VisualLLMSelector")
    offenders: list[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in bad_names:
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(node, 'lineno', '?')} [Name {node.id!r}]"
                )
            elif isinstance(node, ast.Attribute) and node.attr in bad_names:
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(node, 'lineno', '?')} [Attribute "
                    f"{node.attr!r}]"
                )
            elif (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in bad_names
            ):
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(node, 'lineno', '?')} [FunctionDef "
                    f"{node.name!r}]"
                )
            elif isinstance(node, ast.ClassDef) and node.name in bad_names:
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(node, 'lineno', '?')} [ClassDef "
                    f"{node.name!r}]"
                )
    assert not offenders, (
        "S30 B5 violation: VisualLLMSelector reintroduced as a "
        "Python identifier:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 5. Runtime: collapse to single LLM_CACHE means no dup load
# ---------------------------------------------------------------------------


def test_polish_cache_collapse_no_dup_load(monkeypatch):
    """Drive the polish acquisition path via a mocked loader; assert
    exactly one request_slot call lands per polish session, so the
    shared LLM_CACHE holds at most one resident model.
    """
    from visual import llm_polish
    # Force-import the loader first so subsequent code patches the
    # ACTUAL module that visual.llm_polish will pick up. Without this,
    # an earlier-imported module under sys.modules captures the
    # original request_slot and our monkeypatch on a fresh
    # types.ModuleType doesn't replace it.
    from nodes import _otr_model_loader as _OTRML

    request_slot_calls: list[tuple[str, str]] = []

    class _FakeCacheEntry(dict):
        pass

    def fake_request_slot(slot, model_id):
        request_slot_calls.append((slot, model_id))
        return _FakeCacheEntry({
            "model":      MagicMock(name="fake-model"),
            "tokenizer":  MagicMock(name="fake-tokenizer"),
            "device":     "cuda",
            "context_cap": 8192,
        })

    # Patch directly on the imported module so visual.llm_polish's
    # `from nodes import _otr_model_loader as _OTRML` lookup picks
    # up our fake. monkeypatch.setattr handles teardown automatically.
    monkeypatch.setattr(_OTRML, "request_slot", fake_request_slot)
    # Bypass HF token check so the helper runs end-to-end.
    monkeypatch.setattr(
        llm_polish, "ensure_hf_token", lambda: "fake-token"
    )

    result = llm_polish._acquire_polish_entry(
        "mistralai/Mistral-Nemo-Instruct-2407"
    )
    assert result is not None
    model, tokenizer, device = result
    assert model is not None
    assert tokenizer is not None
    # Single request_slot call -- no parallel polish-side cache.
    assert request_slot_calls == [
        ("creative", "mistralai/Mistral-Nemo-Instruct-2407")
    ]


# ---------------------------------------------------------------------------
# Bonus: llm_selector.py file deleted
# ---------------------------------------------------------------------------


def test_visual_llm_selector_file_deleted():
    """visual/llm_selector.py must be gone from the repo."""
    p = PACK_ROOT / "visual" / "llm_selector.py"
    assert not p.exists(), (
        f"visual/llm_selector.py still present; S30 B5 deletes it"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
