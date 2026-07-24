"""The cascade exposes its final local-LLM unload receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ledger_obj():
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_unload_visible",
        "cast": [],
        "scenes": [],
        "shots": [],
        "beats": [],
        "lines": [],
        "music": [],
        "clips": [],
        "meta": {"episode_title": "unload receipt"},
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


def _run_node(*, unload_error=None):
    from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
    from nodes import _otr_freeze_cascade as cascade
    from nodes import _otr_model_loader as loader
    from nodes import production_ledger as ledger_module

    led = _ledger_obj()
    unload = MagicMock(side_effect=unload_error)
    with (
        patch.object(ledger_module, "has_current_ledger", return_value=True),
        patch.object(ledger_module, "peek_ledger", return_value=led),
        patch.object(
            ledger_module,
            "assemble_script_text_from_ledger",
            return_value="rebuilt script text",
        ),
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
        patch.object(
            loader,
            "unload_llm_if_local_resident",
            unload,
        ),
        patch.object(
            cascade,
            "run_freeze_cascade",
            return_value=_disposition(),
        ),
    ):
        result = OTR_LedgerFreezeCascade().run(
            script_text="incoming",
            script_json="{}",
            news_used="",
            estimated_minutes=15,
            technical_model="mistralai/Mistral-Nemo-Instruct-2407",
        )
    return led, unload, result


def test_freeze_unload_ok_stamp_is_visible_in_returned_json():
    led, unload, result = _run_node()

    assert len(result) == 7
    assert unload.call_count == 1
    assert json.loads(result[1])["meta"]["freeze_unload_ok"] is True
    assert led.data["meta"]["freeze_unload_ok"] is True


def test_freeze_unload_failure_stamp_is_visible_in_returned_json():
    led, unload, result = _run_node(
        unload_error=RuntimeError("simulated unload failure"))

    assert len(result) == 7
    assert unload.call_count == 1
    assert json.loads(result[1])["meta"]["freeze_unload_ok"] is False
    assert led.data["meta"]["freeze_unload_ok"] is False


def test_returned_unload_stamp_matches_live_ledger():
    led, _unload, result = _run_node()
    assert json.loads(result[1])["meta"]["freeze_unload_ok"] == (
        led.data["meta"]["freeze_unload_ok"])


def test_remote_cache_entry_skips_local_teardown():
    from nodes import _otr_model_loader as loader

    saved = dict(loader.LLM_CACHE)
    try:
        loader.LLM_CACHE.clear()
        loader.LLM_CACHE.update({
            "model_id": "openrouter:slot-b",
            "slot": "technical",
            "cache_entry": {
                "provider": "openrouter",
                "model_id": "openrouter:slot-b",
            },
        })
        with patch.object(loader, "unload_llm") as local_unload:
            assert loader.unload_llm_if_local_resident() is False
        local_unload.assert_not_called()
    finally:
        loader.LLM_CACHE.clear()
        loader.LLM_CACHE.update(saved)
