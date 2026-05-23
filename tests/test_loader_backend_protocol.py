"""Sprint D D1b -- loader-backend protocol + dispatch table.

Five structural assertions:

  test_protocol_three_callables_present
      LoaderBackend protocol declares load, generate, unload.

  test_existing_safetensors_backend_signatures_unchanged
      AST signature snapshot of the legacy load_llm function the
      safetensors adapter delegates to. Fails loud if a future
      commit changes the signature without updating the adapter.

  test_existing_multimodal_text_only_backend_signatures_unchanged
      Same snapshot from the multimodal adapter's perspective.

  test_gptq_int4_adapter_constructable
      The D1c scaffold instantiates cleanly. load/generate/unload
      each raise NotImplementedError with a clear deferral message.

  test_dispatch_table_routes_loader_backend_literal_to_correct_adapter
      Every catalog row's loader_backend value resolves to the
      expected adapter class via get_backend_for_row.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_loader_backends as backends_proto  # noqa: E402
from nodes import _otr_model_catalog as catalog  # noqa: E402
from nodes import _otr_model_loader as loader  # noqa: E402
from nodes import _otr_model_runtime as runtime  # noqa: E402

# AST-pinned signature snapshot of the legacy load_llm function at
# D1b authorship time. If a future commit modifies the load_llm
# signature, the adapter delegate may break -- catches the drift.
EXPECTED_LOAD_LLM_PARAMS = (
    ("model_id",            None),     # positional
    ("device",              "cuda"),
    ("optimization_profile", "Standard"),
    ("context_cap",         None),
)


def test_protocol_three_callables_present() -> None:
    """The LoaderBackend Protocol declares load + generate + unload."""
    proto = backends_proto.LoaderBackend
    members = {name for name in dir(proto) if not name.startswith("_")}
    for required in ("load", "generate", "unload"):
        assert required in members, (
            f"LoaderBackend protocol missing required method "
            f"{required!r}; present members: {sorted(members)}"
        )


def test_existing_safetensors_backend_signatures_unchanged() -> None:
    """The legacy load_llm signature must match the D1b-time snapshot.

    The TransformersSafetensorsBackend.load delegates to load_llm via
    positional `repo_id` + the legacy function's defaults for the rest.
    A signature drift would silently break the delegate.
    """
    sig = inspect.signature(loader.load_llm)
    params = list(sig.parameters.items())
    assert len(params) == len(EXPECTED_LOAD_LLM_PARAMS), (
        f"load_llm signature has {len(params)} params; "
        f"D1b expected {len(EXPECTED_LOAD_LLM_PARAMS)}"
    )
    for (actual_name, actual_param), (expected_name, expected_default) in zip(
        params, EXPECTED_LOAD_LLM_PARAMS,
    ):
        assert actual_name == expected_name, (
            f"load_llm param order drift: expected {expected_name!r}, "
            f"got {actual_name!r}"
        )
        if expected_default is None:
            continue
        assert actual_param.default == expected_default, (
            f"load_llm param {actual_name!r} default drift: expected "
            f"{expected_default!r}, got {actual_param.default!r}"
        )


def test_existing_multimodal_text_only_backend_signatures_unchanged() -> None:
    """Same snapshot from the multimodal adapter's perspective.

    Both adapters delegate to the same load_llm function today
    (legacy monolith branches internally on multimodal vs text-only).
    The two backend classes are distinct identities so future
    per-backend hooks can land without restructuring.
    """
    multi_backend = runtime.TransformersMultimodalTextOnlyBackend()
    safe_backend = runtime.TransformersSafetensorsBackend()
    # Both adapters expose the same three callables. Inspect via
    # ast on the source so we catch a future override that changes
    # behavior even if the class still appears to "work".
    src_path = REPO_ROOT / "nodes" / "_otr_model_runtime.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    class_bodies: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_bodies[node.name] = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    # Neither concrete subclass overrides load/generate/unload at D1b;
    # both inherit from _LegacyTransformersBackendBase. A future
    # override is fine but the test surfaces it (rename the assertion
    # to expected-override and update the AST snapshot).
    assert class_bodies.get("TransformersSafetensorsBackend") == [], (
        f"TransformersSafetensorsBackend overrides methods at D1b: "
        f"{class_bodies.get('TransformersSafetensorsBackend')}; "
        f"expected pure inheritance from _LegacyTransformersBackendBase"
    )
    assert class_bodies.get("TransformersMultimodalTextOnlyBackend") == [], (
        f"TransformersMultimodalTextOnlyBackend overrides methods at "
        f"D1b: {class_bodies.get('TransformersMultimodalTextOnlyBackend')}; "
        f"expected pure inheritance from _LegacyTransformersBackendBase"
    )
    # Both adapter instances respond to the three callables.
    for adapter in (safe_backend, multi_backend):
        for method in ("load", "generate", "unload"):
            assert callable(getattr(adapter, method, None)), (
                f"{type(adapter).__name__} missing callable "
                f"{method!r}"
            )


def test_gptq_int4_adapter_constructable() -> None:
    """The D1c scaffold is constructable; load/generate/unload each
    raise NotImplementedError with a clear deferral message.

    Sprint D D4 update: load() now calls check_context_window(row)
    before raising NotImplementedError. Pass a fake row with
    context_window >= HARD_VRAM_CONTEXT_LIMIT so the precondition
    passes and the NotImplementedError surfaces as before.
    """
    from dataclasses import dataclass

    from nodes import _otr_model_catalog as _catalog

    @dataclass
    class _FakeAdequateRow:
        context_window: int = 0
        repo_id: str = "period-lm/placeholder-gptq-13b"

    fake_row = _FakeAdequateRow(
        context_window=int(_catalog.HARD_VRAM_CONTEXT_LIMIT),
    )

    adapter = runtime.TransformersGPTQInt4Backend()
    assert adapter is not None

    with pytest.raises(NotImplementedError, match="D1c"):
        adapter.load("period-lm/placeholder-gptq-13b", row=fake_row)

    with pytest.raises(NotImplementedError, match="D1c"):
        adapter.generate(model=None, messages=[])

    with pytest.raises(NotImplementedError, match="D1c"):
        adapter.unload(model=None)


def test_dispatch_table_routes_loader_backend_literal_to_correct_adapter() -> None:
    """Every catalog row resolves via get_backend_for_row to the
    adapter class matching its loader_backend literal.
    """
    expected_class_by_key: dict[str, type] = {
        "transformers_safetensors":          runtime.TransformersSafetensorsBackend,
        "transformers_multimodal_text_only": runtime.TransformersMultimodalTextOnlyBackend,
        "transformers_gptq_int4":            runtime.TransformersGPTQInt4Backend,
    }
    failures: list[str] = []
    for row in catalog.CURATED_LLM_MODELS:
        adapter = runtime.get_backend_for_row(row)
        expected_cls = expected_class_by_key.get(row.loader_backend)
        if expected_cls is None:
            failures.append(
                f"{row.repo_id!r}: loader_backend "
                f"{row.loader_backend!r} has no expected adapter class "
                f"in the test's expected_class_by_key map"
            )
            continue
        if not isinstance(adapter, expected_cls):
            failures.append(
                f"{row.repo_id!r}: get_backend_for_row returned "
                f"{type(adapter).__name__}; expected "
                f"{expected_cls.__name__}"
            )
    assert not failures, "\n  " + "\n  ".join(failures)


def test_get_backend_for_row_raises_on_unknown_loader_backend() -> None:
    """Schema drift between catalog rows and the dispatch table fails
    loud at lookup time, not silently with the wrong adapter.
    """
    class FakeRow:
        repo_id = "fake/unknown-backend"
        loader_backend = "transformers_fictional_v999"

    with pytest.raises(KeyError, match="transformers_fictional_v999"):
        runtime.get_backend_for_row(FakeRow())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
