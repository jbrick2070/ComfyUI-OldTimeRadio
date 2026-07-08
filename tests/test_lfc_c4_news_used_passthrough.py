"""tests/test_lfc_c4_news_used_passthrough.py

LFC commit 12.10 -- C4 runtime news_used passthrough check.

C4 of the clean-break go-forward plan: parsed equality at three
endpoints --
    writer_output["news_used"] == cascade_output["news_used"]
    cascade_output["news_used"] == signal_lost_input["news_used"]
Catches any quiet mutation in run_freeze_cascade or the cascade
node's run() wrapper. Per B5 (LFC commit 12.1): non-empty input
is byte-identical pre/post cascade; empty input normalizes to "".

These tests exercise OTR_LedgerFreezeCascade.run() directly with
a stubbed reviewer (no live LLM) so the passthrough contract is
gated in the standard pytest sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _line(line_id, char_id, role, text):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


def _ledger_data():
    return {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_c4_test",
        "cast": [
            {"char_id": "c01", "name": "ALICE",
             "portrait_path": "ALICE.png"},
            {"char_id": "c02", "name": "BOB",
             "portrait_path": "BOB.png"},
            {"char_id": "announcer", "name": "ANNOUNCER",
             "portrait_path": "ANNOUNCER.png"},
        ],
        "scenes": [], "shots": [], "beats": [],
        "lines": [
            _line("l001", "c01", "character", "Hello there general."),
            _line("l002", "c02", "character", "Hi there friend."),
        ],
        "music": [], "clips": [],
        "meta": {
            "episode_title": "c4 test",
            "style": "mission_control_procedural",
        },
    }


def _stub_reviewer(verdict="clean_no_edits"):
    from nodes import _otr_ledger_reviewer as _OTRLR

    def fake_review(_fn, _led):
        return _OTRLR.ReviewerDisposition(
            verdict=verdict,
            pre_audit_violations=0, pre_audit_repairs_applied=0,
            doctor_edits_proposed=0, doctor_edits_applied=0,
            post_audit_violations=0,
        )
    return fake_review


def _stub_loader():
    """Stub for `_otr_model_loader` so the cascade node's run()
    doesn't try to load Mistral-Nemo. Returns a cache_entry shape
    sufficient for make_generate_fn / make_polish_generate_fn /
    unload_llm to succeed at attribute access."""
    from unittest.mock import MagicMock
    loader = MagicMock()
    loader.load_llm = MagicMock(return_value={"model": object(),
                                              "tokenizer": object()})
    loader.make_generate_fn = MagicMock(return_value=lambda *a, **k: "")
    loader.make_polish_generate_fn = MagicMock(
        return_value=lambda *a, **k: ""
    )
    loader.unload_llm = MagicMock(return_value=None)
    return loader


def _stub_pl():
    """Stub for production_ledger so the cascade picks up our
    fixture ledger via peek_ledger()."""
    from unittest.mock import MagicMock
    data = _ledger_data()
    led = SimpleNamespace(
        data=data, episode_id=data["episode_id"],
        save=lambda: None,
    )
    pl = MagicMock()
    pl.has_current_ledger = MagicMock(return_value=True)
    pl.peek_ledger = MagicMock(return_value=led)
    pl.get_ledger = MagicMock(return_value=led)
    pl.assemble_script_text_from_ledger = MagicMock(
        return_value="rebuilt script text"
    )
    return pl


# ---------------------------------------------------------------------------
# Three-endpoint parsed-equality at the C4 contract
# ---------------------------------------------------------------------------


class TestNewsUsedPassthroughParsedEquality:
    """C4: writer_output["news_used"] == cascade_output["news_used"]
              == signal_lost_input["news_used"]"""

    def _run_cascade(self, incoming_news_used: str):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        loader = _stub_loader()
        pl = _stub_pl()

        with patch("nodes.OTR_LedgerFreezeCascade._otr_model_loader",
                   loader, create=True), \
             patch("nodes.OTR_LedgerFreezeCascade.production_ledger",
                   pl, create=True), \
             patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer()):
            # The cascade node lazy-imports both _otr_model_loader
            # and production_ledger *inside* run(). Patch on the
            # module's namespace directly via import-binding doesn't
            # work in that case; use sys.modules instead.
            import sys as _sys
            from nodes import _otr_model_loader as _real_loader
            from nodes import production_ledger as _real_pl
            saved_loader = _sys.modules["nodes._otr_model_loader"]
            saved_pl = _sys.modules["nodes.production_ledger"]
            _sys.modules["nodes._otr_model_loader"] = loader
            _sys.modules["nodes.production_ledger"] = pl
            try:
                inst = OTR_LedgerFreezeCascade()
                result = inst.run(
                    script_text="x",
                    script_json="{}",
                    news_used=incoming_news_used,
                    estimated_minutes=15,
                    technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )
            finally:
                _sys.modules["nodes._otr_model_loader"] = saved_loader
                _sys.modules["nodes.production_ledger"] = saved_pl
        return result

    def test_non_empty_news_used_is_byte_identical(self):
        # Per B5 (LFC commit 12.1): non-empty input is
        # byte-identical pre/post fix.
        incoming = "headline: cool news article 12345 chars"
        result = self._run_cascade(incoming)
        # 5-slot output: (script_text, script_json, news_used,
        #                 estimated_minutes, freeze_verdict)
        cascade_news_used = result[2]
        assert cascade_news_used == incoming, (
            f"C4: non-empty news_used must pass through byte-"
            f"identical. got={cascade_news_used!r} "
            f"expected={incoming!r}"
        )

    def test_empty_news_used_normalizes_to_empty_string(self):
        # Per B5: None / "" normalize to "" -- still parsed equal.
        result = self._run_cascade("")
        assert result[2] == ""

    def test_whitespace_news_used_preserved(self):
        # A non-empty whitespace-only string is non-empty -- must
        # be preserved verbatim, NOT trimmed.
        incoming = "   "
        result = self._run_cascade(incoming)
        # The cascade's `news_used or ""` falsy-coerces ""/None
        # but "   " is truthy -- it passes through.
        assert result[2] == incoming

    def test_long_news_used_passes_through(self):
        # Soak realism: a 4kB news_used payload.
        incoming = "x" * 4096
        result = self._run_cascade(incoming)
        assert result[2] == incoming
        assert len(result[2]) == 4096


# ---------------------------------------------------------------------------
# Cascade output port = workflow news_used input contract
# ---------------------------------------------------------------------------


class TestNewsUsedWiringChainContract:
    """W2 (commit 12.1) + C4 belt-and-suspenders. The cascade's
    news_used OUTPUT is what SignalLostVideo's news_used INPUT
    receives via link 110 in the workflow JSON. Verify the
    cascade's output value matches the writer's input value
    (the previous test class) AND verify the workflow JSON link
    exists at runtime test time too."""

    def test_workflow_news_used_link_110_present(self):
        import json
        wf = json.loads(
            (Path(__file__).resolve().parents[1]
             / "workflows" / "otr_canonical.json").read_text(
                encoding="utf-8"
            )
        )
        link_ids = {l[0] for l in wf.get("links") or []}
        assert 110 in link_ids
        assert 18 not in link_ids  # legacy bypass removed
