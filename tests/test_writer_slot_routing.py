"""S30 B2b tests -- writer-side slot scheduler + per-phase LLM routing.

Five guardrails for the OTR_LedgerScriptWriter internal slot routing:

1. test_no_direct_load_llm_calls_inside_writer
2. test_writer_uses_request_slot_for_every_llm_pass
3. test_slot1_neq_slot2_split_routing_full_phase_trace
4. test_polish_routes_to_creative
5. test_slot_transition_count_matches_dag

Tests are AST-level + scheduler-unit-level so the suite does NOT
require torch / transformers / GPU. End-to-end writer runs are out of
scope for the S30 pytest gate (per the continuation plan -- no
ComfyUI Desktop runs in this sprint).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _writer_tree() -> ast.AST:
    return ast.parse(WRITER_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. test_no_direct_load_llm_calls_inside_writer
# ---------------------------------------------------------------------------


def test_no_direct_load_llm_calls_inside_writer():
    """The writer must NOT contain a direct call to load_llm(...).
    Every LLM acquisition goes through the slot scheduler ->
    _otr_model_loader.request_slot. Import-name references to load_llm
    (e.g. `_OTRML.load_llm` in a comment) are not flagged -- only
    actual `Call` nodes.
    """
    tree = _writer_tree()
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "load_llm":
            bad.append(
                f"line {getattr(node, 'lineno', '?')}: "
                f".load_llm(...)"
            )
        elif isinstance(fn, ast.Name) and fn.id == "load_llm":
            bad.append(
                f"line {getattr(node, 'lineno', '?')}: load_llm(...)"
            )
    assert not bad, (
        "writer must not call load_llm directly -- route through the "
        "slot scheduler / request_slot:\n  " + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# 2. test_writer_uses_request_slot_for_every_llm_pass
# ---------------------------------------------------------------------------


def test_writer_uses_request_slot_for_every_llm_pass():
    """The writer's _SlotScheduler invokes request_slot inside its
    _account_and_get_entry helper. AST sweep: the scheduler's
    helper must call request_slot exactly once."""
    tree = _writer_tree()
    target: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "_SlotScheduler"
        ):
            for sub in node.body:
                if (
                    isinstance(sub, ast.FunctionDef)
                    and sub.name == "_account_and_get_entry"
                ):
                    target = sub
                    break
    assert target is not None, (
        "_SlotScheduler._account_and_get_entry not found"
    )
    count = 0
    for sub in ast.walk(target):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr == "request_slot":
                count += 1
            elif isinstance(fn, ast.Name) and fn.id == "request_slot":
                count += 1
    assert count == 1, (
        f"_account_and_get_entry must call request_slot exactly once; "
        f"found {count}"
    )


def test_writer_has_slot_tag_comments_at_every_llm_call_site():
    """Defense in depth: every top-level LLM call site inside
    OTR_LedgerScriptWriter.run carries a `# LLM slot: ...` tag near
    the call, naming the slot that fires."""
    from tests.fixtures.writer_family import family_source

    src = family_source()
    # Per the continuation plan, "# LLM slot: creative" or
    # "# LLM slot: technical" must appear at every LLM call site.
    creative_hits = src.count("# LLM slot: creative")
    technical_hits = src.count("# LLM slot: technical")
    assert creative_hits >= 8, (
        f"expected >=8 '# LLM slot: creative' tags (style picker, "
        f"cast lock, outline, dialogue composer, announcer intro, "
        f"announcer outro, polish, title regen); found "
        f"{creative_hits}"
    )
    assert technical_hits >= 1, (
        f"expected >=1 '# LLM slot: technical' tag (news interpreter "
        f"in B2b; B4b adds RSS rerank); found {technical_hits}"
    )


# ---------------------------------------------------------------------------
# 3. test_slot1_neq_slot2_split_routing_full_phase_trace
# ---------------------------------------------------------------------------


class _StubCacheEntry:
    def __init__(self, model_id):
        self.model_id = model_id

    def __getitem__(self, key):
        return MagicMock()

    def get(self, key, default=None):
        return default


def _patched_scheduler(monkeypatch, *, creative_id, technical_id):
    """Build a _SlotScheduler with request_slot + the truncating
    wrapper mocked. Returns (scheduler, trace_list)."""
    import nodes.OTR_LedgerScriptWriter as W
    import nodes._otr_model_loader as ML

    trace: list[tuple[str, str]] = []

    def fake_request_slot(slot, model_id, policy=None, load_config=None):
        # S1: production threads the LLMRuntimePolicy + load_config kwargs;
        # the fake mirrors the signature (trace shape unchanged).
        trace.append((slot, model_id))
        return _StubCacheEntry(model_id)

    monkeypatch.setattr(ML, "request_slot", fake_request_slot)
    # Mock the truncating wrapper so we don't load torch.
    monkeypatch.setattr(
        W,
        "_build_truncating_generate_fn",
        lambda cache_entry, **kw: (
            lambda messages, *, temperature, max_new_tokens, stop=None: (
                f"<fake-creative-output for {cache_entry.model_id}>"
            )
        ),
    )
    monkeypatch.setattr(
        ML,
        "make_polish_generate_fn",
        lambda cache_entry: (
            lambda messages, *, temperature, max_new_tokens: (
                f"<fake-polish-output for {cache_entry.model_id}>"
            )
        ),
    )

    scheduler = W._SlotScheduler(
        creative_id=creative_id,
        technical_id=technical_id,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.03,
    )
    return scheduler, trace


def test_same_slot_id_means_zero_transitions(monkeypatch):
    """When creative_writing_model == technical_model (default), every
    call cache-hits on one resident model; the scheduler reports 0
    transitions regardless of slot interleaving.
    """
    SAME = "mistralai/Mistral-Nemo-Instruct-2407"
    scheduler, trace = _patched_scheduler(
        monkeypatch, creative_id=SAME, technical_id=SAME,
    )
    creative_fn = scheduler.for_slot("creative")
    technical_fn = scheduler.for_slot("technical")
    creative_fn([], temperature=0.8, max_new_tokens=100)
    technical_fn([], temperature=0.4, max_new_tokens=100)
    creative_fn([], temperature=0.8, max_new_tokens=100)
    technical_fn([], temperature=0.4, max_new_tokens=100)
    assert scheduler.transitions == 0
    assert all(model_id == SAME for _slot, model_id in trace)


def test_scheduler_declares_catalog_local_schema_binding(monkeypatch):
    """A local catalog row exposes lazy typed-schema binding."""
    import nodes.OTR_LedgerScriptWriter as W
    import nodes._otr_model_catalog as catalog

    local_id = "mistralai/Mistral-Nemo-Instruct-2407"
    monkeypatch.setattr(
        catalog, "_by_repo_id",
        lambda: {local_id: type("LocalRow", (), {"provider": "local"})()},
    )
    scheduler = W._SlotScheduler(
        creative_id=local_id,
        technical_id=local_id,
        top_p=.92,
        min_p=0.0,
        repetition_penalty=1.0,
    )

    slot_fn = scheduler.for_slot("creative")
    assert slot_fn._otr_local_schema_binding is True  # type: ignore[attr-defined]
    assert slot_fn._otr_openrouter is False  # type: ignore[attr-defined]
    assert slot_fn._otr_response_format is None  # type: ignore[attr-defined]
    assert callable(slot_fn._otr_bind_schema)  # type: ignore[attr-defined]


def test_scheduler_local_schema_binding_reaches_truncating_generator(monkeypatch):
    import nodes.OTR_LedgerScriptWriter as W
    import nodes._otr_model_catalog as catalog
    import nodes._otr_model_loader as loader

    local_id = "google/gemma-4-12b-it"
    monkeypatch.setattr(
        catalog, "_by_repo_id",
        lambda: {local_id: type("LocalRow", (), {"provider": "local"})()},
    )
    monkeypatch.setattr(
        loader,
        "request_slot",
        lambda *_args, **_kwargs: {"model": object(), "tokenizer": object()},
    )
    seen = {}

    def fake_build(cache_entry, **kwargs):
        seen.update(kwargs)
        return lambda messages, **call_kwargs: call_kwargs["max_new_tokens"]

    monkeypatch.setattr(W, "_build_truncating_generate_fn", fake_build)
    scheduler = W._SlotScheduler(
        creative_id=local_id,
        technical_id=local_id,
        top_p=.87,
        min_p=.04,
        repetition_penalty=1.03,
    )
    plain = scheduler.for_slot("technical")

    class ExactSchema:
        pass

    bound = plain._otr_bind_schema(ExactSchema)  # type: ignore[attr-defined]
    assert bound([], temperature=.2, max_new_tokens=77) == 77
    assert seen == {
        "schema_model": ExactSchema,
        "top_p": .87,
        "min_p": .04,
        "repetition_penalty": 1.03,
    }
    assert bound._otr_bound_schema_model is ExactSchema  # type: ignore[attr-defined]
    assert bound._otr_local_schema_binding is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. test_polish_routes_to_creative
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. test_slot_transition_count_matches_dag
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Meta key surface
# ---------------------------------------------------------------------------


def test_writer_deletes_legacy_model_id_meta_key():
    """gen_params_initial must NOT carry the legacy `model_id` key
    post-B2b. Consumers route via creative_writing_model /
    technical_model. AST scan of the assignment-site keys.
    """
    tree = _writer_tree()
    bad_keys: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "meta"
            and isinstance(node.value, ast.Dict)
        ):
            # meta["..."] = {...}; look for "model_id" key inside.
            for k in node.value.keys:
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value == "model_id"
                ):
                    bad_keys.append(
                        f"line {getattr(node, 'lineno', '?')}: "
                        f"meta dict still has 'model_id' key"
                    )
    assert not bad_keys, (
        "B2b clean break: meta dicts must not include the legacy "
        "'model_id' key -- use creative_writing_model + technical_model "
        "explicitly:\n  " + "\n  ".join(bad_keys)
    )


def test_writer_stamps_slot_transitions_meta():
    """Writer must stamp `meta["slot_transitions"]` (int) +
    `meta["slot_calls_by_slot"]` (dict) so a forensic audit can read
    the scheduler trace without re-running the writer."""
    from tests.fixtures.writer_family import family_source

    src = family_source()
    assert 'meta["slot_transitions"]' in src
    assert 'meta["slot_calls_by_slot"]' in src
    assert 'meta["gen_params_by_phase"]' in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
