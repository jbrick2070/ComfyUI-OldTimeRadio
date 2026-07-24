"""The freeze node preserves ``news_used`` byte-for-byte."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ledger():
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_c4_test",
        "cast": [],
        "scenes": [],
        "shots": [],
        "beats": [],
        "lines": [],
        "music": [],
        "clips": [],
        "meta": {"episode_title": "c4 test"},
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


def _run_cascade(incoming_news_used: str):
    from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
    from nodes import _otr_freeze_cascade as cascade
    from nodes import _otr_model_loader as loader
    from nodes import production_ledger

    led = _ledger()
    with (
        patch.object(production_ledger, "has_current_ledger", return_value=True),
        patch.object(production_ledger, "peek_ledger", return_value=led),
        patch.object(
            production_ledger,
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
        patch.object(loader, "unload_llm_if_local_resident", return_value=True),
        patch.object(cascade, "run_freeze_cascade", return_value=_disposition()),
    ):
        return OTR_LedgerFreezeCascade().run(
            script_text="incoming",
            script_json="{}",
            news_used=incoming_news_used,
            estimated_minutes=15,
            technical_model="mistralai/Mistral-Nemo-Instruct-2407",
        )


@pytest.mark.parametrize(
    "incoming",
    [
        "headline: cool news article 12345 chars",
        "",
        "   ",
        "x" * 4096,
    ],
)
def test_news_used_passthrough_is_byte_identical(incoming):
    result = _run_cascade(incoming)

    assert len(result) == 7
    assert result[2] == incoming


def test_workflow_news_used_link_110_present_and_bypass_absent():
    import json

    workflow = json.loads(
        (Path(__file__).resolve().parents[1] / "workflows" / "otr_canonical.json")
        .read_text(encoding="utf-8")
    )
    link_ids = {link[0] for link in workflow.get("links") or []}
    assert 110 in link_ids
    assert 18 not in link_ids
