"""The cascade releases a local LLM on every exit path."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ledger_obj():
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_unload_finally",
        "cast": [],
        "scenes": [],
        "shots": [],
        "beats": [],
        "lines": [],
        "music": [],
        "clips": [],
        "meta": {"episode_title": "unload finally"},
    }
    return SimpleNamespace(
        data=data,
        episode_id=data["episode_id"],
        save=lambda: None,
    )


def _disposition():
    report = SimpleNamespace(warnings=[])
    return SimpleNamespace(
        verdict="frozen_clean",
        gap_audit_pre=report,
        gap_audit_post=report,
        cleanup_receipt={"status": "clean"},
    )


def _patch_runtime(led, unload, *, assemble=None):
    from nodes import _otr_model_loader as loader
    from nodes import production_ledger as ledger_module

    if assemble is None:
        assemble = MagicMock(return_value="rebuilt script text")
    return (
        patch.object(ledger_module, "has_current_ledger", return_value=True),
        patch.object(ledger_module, "peek_ledger", return_value=led),
        patch.object(
            ledger_module, "assemble_script_text_from_ledger", assemble),
        patch.object(
            loader,
            "request_slot",
            return_value={"model": object(), "tokenizer": object()},
        ),
        patch.object(
            loader,
            "make_generate_fn",
            return_value=lambda *args, **kwargs: "",
        ),
        patch.object(loader, "unload_llm_if_local_resident", unload),
    )


def _call_node():
    from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade

    return OTR_LedgerFreezeCascade().run(
        script_text="incoming",
        script_json="{}",
        news_used="",
        estimated_minutes=15,
        technical_model="mistralai/Mistral-Nemo-Instruct-2407",
    )


def test_unload_runs_when_cascade_raises():
    from nodes import _otr_freeze_cascade as cascade

    led = _ledger_obj()
    unload = MagicMock()
    patches = _patch_runtime(led, unload)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            cascade,
            "run_freeze_cascade",
            side_effect=RuntimeError("simulated cascade explosion"),
        ):
            with pytest.raises(RuntimeError, match="cascade explosion"):
                _call_node()

    assert unload.call_count == 1


def test_clean_run_stamps_successful_unload():
    from nodes import _otr_freeze_cascade as cascade

    led = _ledger_obj()
    unload = MagicMock()
    patches = _patch_runtime(led, unload)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            cascade, "run_freeze_cascade", return_value=_disposition()):
            result = _call_node()

    assert unload.call_count == 1
    assert led.data["meta"]["freeze_unload_ok"] is True
    assert len(result) == 7


def test_unload_failure_is_stamped_without_hiding_verdict():
    from nodes import _otr_freeze_cascade as cascade

    led = _ledger_obj()
    unload = MagicMock(side_effect=RuntimeError("unload failed"))
    patches = _patch_runtime(led, unload)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            cascade, "run_freeze_cascade", return_value=_disposition()):
            result = _call_node()

    assert unload.call_count == 1
    assert led.data["meta"]["freeze_unload_ok"] is False
    assert result[4] == "frozen_clean"
    assert len(result) == 7


def test_unload_runs_when_script_assembly_falls_back():
    from nodes import _otr_freeze_cascade as cascade

    led = _ledger_obj()
    unload = MagicMock()
    assemble = MagicMock(side_effect=RuntimeError("assemble crash"))
    patches = _patch_runtime(led, unload, assemble=assemble)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            cascade, "run_freeze_cascade", return_value=_disposition()):
            result = _call_node()

    assert unload.call_count == 1
    assert result[0] == "incoming"
