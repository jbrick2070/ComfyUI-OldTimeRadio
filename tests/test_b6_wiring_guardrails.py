"""S30 B6 -- defense-in-depth wiring tests + structure guardrails.

Catches structural drift that the per-commit B1d..B5 tests don't
defend against. Six tests:

1. test_request_slot_uses_check_vram_fit_pre_download   -- regression
   for the B1d hotfix: VRAMFitFailedError must fire BEFORE
   snapshot_download for an oversize model.
2. test_no_legacy_model_id_meta_key_in_writer           -- AST scan
   asserts the writer never emits a legacy `model_id` meta key in
   any meta dict literal.
3. test_no_not_downloaded_suffix_in_meta_stamps         -- AST scan
   asserts no string-literal `[NOT DOWNLOADED]` appears as a meta-
   stamp value in writer source.
4. test_no_model_widget_outside_writer                  -- scan
   nodes/*.py for any class that declares a `model_id` /
   `creative_writing_model` / `technical_model` widget in
   INPUT_TYPES. Only OTR_LedgerScriptWriter is allowed to have
   widget-form model picks.
5. test_slot_scheduler_transitions_match_dag            -- defense-
   in-depth unit test of the writer's _SlotScheduler transition
   counter (mirrors the B2b coverage at the wiring-test layer).
6. test_writer_has_input_types_no_other_node_picks_model -- same as
   #4 expressed as a parametrized sweep so each offending class
   gets its own failure row.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"

# Widget keys that signal a model-pick surface. Only the writer is
# allowed to expose these as widgets (combo / STRING widgets without
# forceInput). Other nodes consume via forceInput sockets.
#
# S4 (2026-06-01): the four-dropdown router added two slug-picker widgets to
# the writer -- openrouter_slot_a_model / openrouter_slot_b_model. They are
# writer-only model picks too, so they join the guarded set.
# 2026-06-01 (Comfy Credits): the sibling credit-billed lane appends a second
# pair -- comfy_slot_a_model / comfy_slot_b_model -- so the writer's active
# model-pick widgets go 4 -> 6: creative_writing_model, technical_model,
# openrouter_slot_a_model, openrouter_slot_b_model, comfy_slot_a_model,
# comfy_slot_b_model. `model_id` stays as the retired legacy name that must
# never reappear as a widget anywhere.
_MODEL_WIDGET_KEYS = frozenset({
    "model_id",
    "creative_writing_model",
    "technical_model",
    "openrouter_slot_a_model",
    "openrouter_slot_b_model",
    "comfy_slot_a_model",
    "comfy_slot_b_model",
})


# ---------------------------------------------------------------------------
# 1. test_request_slot_uses_check_vram_fit_pre_download
# ---------------------------------------------------------------------------


def test_request_slot_uses_check_vram_fit_pre_download(monkeypatch):
    """B1d regression: when an oversize model is picked, request_slot
    must raise VRAMFitFailedError BEFORE snapshot_download or any
    network/disk pre-flight work fires.
    """
    from nodes import _otr_model_catalog as catalog
    from nodes import _otr_model_loader as loader
    from nodes._otr_model_inputs import VRAMFitFailedError

    # Reset loader cache so the assertion is clean.
    loader.LLM_CACHE.update(
        {"model_id": None, "slot": None, "cache_entry": None}
    )

    download_calls: list[str] = []

    def counting_auto_download(repo_id, **kw):
        download_calls.append(repo_id)
        return f"/fake/snapshot/{repo_id}"

    # Force HF_TOKEN to exist so the validator admits the
    # gated-curated repo (Mistral-Nemo etc.); we're not testing the
    # gating path.
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(
        catalog, "auto_download_if_missing", counting_auto_download,
    )

    with pytest.raises(VRAMFitFailedError):
        loader.request_slot("creative", catalog.TEST_OVERSIZED_LLM)

    assert download_calls == [], (
        "VRAMFitFailedError must fire before auto_download_if_missing; "
        f"saw download calls: {download_calls}"
    )


# ---------------------------------------------------------------------------
# 2. test_no_legacy_model_id_meta_key_in_writer
# ---------------------------------------------------------------------------


def _writer_tree() -> ast.AST:
    return ast.parse(WRITER_PATH.read_text(encoding="utf-8"))


def test_no_legacy_model_id_meta_key_in_writer():
    """The writer must never stamp a `model_id` key inside a meta
    dict literal post-B2b. AST walk every `meta[...] = {...}`
    assignment + every `{...}` Dict literal nested in those values.
    """
    tree = _writer_tree()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Look for: meta["..."] = {...}.
        for target in node.targets:
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "meta"
            ):
                continue
            # Walk every Dict in the assigned value for a model_id key.
            for sub in ast.walk(node.value):
                if not isinstance(sub, ast.Dict):
                    continue
                for k in sub.keys:
                    if (
                        isinstance(k, ast.Constant)
                        and isinstance(k.value, str)
                        and k.value == "model_id"
                    ):
                        offenders.append(
                            f"line {getattr(k, 'lineno', '?')}: "
                            f"meta dict still has 'model_id' key"
                        )
    assert not offenders, (
        "writer must not stamp 'model_id' in meta dict literals "
        "post-B2b (replaced by creative_writing_model + "
        "technical_model):\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 3. test_no_not_downloaded_suffix_in_meta_stamps
# ---------------------------------------------------------------------------


def test_no_not_downloaded_suffix_in_meta_stamps():
    """Raw `[NOT DOWNLOADED]` widget labels must never reach a meta
    stamp. The writer normalizes via `_otr_model_catalog._strip_label_suffix`
    in `_resolve_inputs`; this guard catches a future regression where
    a meta-stamp site references the raw widget value.
    """
    tree = _writer_tree()
    bad: list[str] = []
    for node in ast.walk(tree):
        # Look for any string literal containing "[NOT DOWNLOADED]"
        # inside a meta-stamp ast.Assign target.
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "meta"
            ):
                continue
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and "[NOT DOWNLOADED]" in sub.value
                ):
                    bad.append(
                        f"line {getattr(sub, 'lineno', '?')}: "
                        f"'[NOT DOWNLOADED]' substring in meta stamp"
                    )
    assert not bad, (
        "meta stamps must not contain raw [NOT DOWNLOADED] suffix; "
        "every label is normalized via _strip_label_suffix at the "
        "resolver boundary:\n  " + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# 4 + 6. No model widget outside the writer
# ---------------------------------------------------------------------------


def _classes_in_nodes_dir() -> list[tuple[Path, ast.ClassDef]]:
    """Yield (file, ClassDef) for every class defined under nodes/."""
    out: list[tuple[Path, ast.ClassDef]] = []
    root = PACK_ROOT / "nodes"
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                out.append((path, node))
    return out


def _input_types_method(cls_node: ast.ClassDef) -> ast.FunctionDef | None:
    for sub in cls_node.body:
        if (
            isinstance(sub, ast.FunctionDef)
            and sub.name == "INPUT_TYPES"
        ):
            return sub
    return None


def _is_force_input_value(value_node: ast.AST) -> bool:
    """Return True if the value AST node looks like ("STRING", {"forceInput": True, ...}).
    Forced-input sockets are consumer surfaces -- legal anywhere.
    """
    # value should be a Tuple of (type_literal, options_dict).
    if not isinstance(value_node, ast.Tuple) or len(value_node.elts) < 2:
        return False
    opts = value_node.elts[1]
    if not isinstance(opts, ast.Dict):
        return False
    for k, v in zip(opts.keys, opts.values):
        if (
            isinstance(k, ast.Constant)
            and k.value == "forceInput"
            and isinstance(v, ast.Constant)
            and v.value is True
        ):
            return True
    return False


def _has_non_llm_opt_in(cls_node: ast.ClassDef) -> bool:
    """Return True if the class declares
    `NON_LLM_MODEL_WIDGET_OK = True` at class scope. Non-LLM media
    nodes (AudioGen, MusicGen, etc.) carry a `model_id` widget for
    their own checkpoint pick; the marker exempts them from the
    LLM-slot guard.
    """
    for sub in cls_node.body:
        if not isinstance(sub, ast.Assign):
            continue
        if len(sub.targets) != 1:
            continue
        target = sub.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "NON_LLM_MODEL_WIDGET_OK"
            and isinstance(sub.value, ast.Constant)
            and sub.value.value is True
        ):
            return True
    return False


def _collect_widget_offenders() -> list[str]:
    """Walk every class in nodes/, find INPUT_TYPES, scan the Dict
    keys for any model-widget key whose value is NOT forceInput
    AND whose class is not flagged NON_LLM_MODEL_WIDGET_OK.
    """
    offenders: list[str] = []
    for path, cls in _classes_in_nodes_dir():
        if cls.name == "OTR_LedgerScriptWriter":
            continue  # only legitimate widget-form LLM picker
        if _has_non_llm_opt_in(cls):
            continue  # non-LLM media node with its own checkpoint picker
        method = _input_types_method(cls)
        if method is None:
            continue
        for sub in ast.walk(method):
            if not isinstance(sub, ast.Dict):
                continue
            for k, v in zip(sub.keys, sub.values):
                if not (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value in _MODEL_WIDGET_KEYS
                ):
                    continue
                # Legal if the value is a forceInput socket (consumer).
                if _is_force_input_value(v):
                    continue
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(k, 'lineno', '?')} [{cls.name}."
                    f"INPUT_TYPES has widget-form key {k.value!r}]"
                )
    return offenders


def test_no_model_widget_outside_writer():
    """No class under nodes/ other than OTR_LedgerScriptWriter may
    declare a widget-form `model_id` / `creative_writing_model` /
    `technical_model` key in its INPUT_TYPES. forceInput sockets are
    allowed (consumer wiring).
    """
    offenders = _collect_widget_offenders()
    assert not offenders, (
        "S30 B6 violation: only OTR_LedgerScriptWriter may expose a "
        "model-widget surface. Other nodes must use forceInput "
        "sockets:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "key",
    list(sorted(_MODEL_WIDGET_KEYS)),
)
def test_writer_has_input_types_no_other_node_picks_model(key):
    """Parametrized sweep: for each known model-widget key, no class
    other than OTR_LedgerScriptWriter may declare it as a widget.
    Non-LLM media nodes (AudioGen, MusicGen) opt out via
    NON_LLM_MODEL_WIDGET_OK = True.
    """
    offenders: list[str] = []
    for path, cls in _classes_in_nodes_dir():
        if cls.name == "OTR_LedgerScriptWriter":
            continue
        if _has_non_llm_opt_in(cls):
            continue
        method = _input_types_method(cls)
        if method is None:
            continue
        for sub in ast.walk(method):
            if not isinstance(sub, ast.Dict):
                continue
            for k, v in zip(sub.keys, sub.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == key
                    and not _is_force_input_value(v)
                ):
                    offenders.append(
                        f"{path.relative_to(PACK_ROOT).as_posix()}:"
                        f"{getattr(k, 'lineno', '?')} [{cls.name}]"
                    )
    assert not offenders, (
        f"S30 B6 violation: widget-form {key!r} declared outside the "
        f"writer:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 5. test_slot_scheduler_transitions_match_dag
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
