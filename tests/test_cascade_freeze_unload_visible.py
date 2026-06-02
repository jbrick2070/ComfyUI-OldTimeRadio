"""tests/test_cascade_freeze_unload_visible.py — S34 B2 P1 proof.

S34 B2 (2026-05-15) makes the `meta.freeze_unload_ok` stamp visible
to downstream JSON consumers. The cascade's finally block stamps
the flag on `led.data` AFTER the first serialization of
`updated_script_json` -- pre-B2 the returned JSON didn't contain
the stamp. B2 adds a reserialization step between the finally
block and the return so the stamp lands in the output.

Tests prove:
  1. On a clean unload, returned `updated_script_json` carries
     `meta.freeze_unload_ok=True`.
  2. On a failed unload, returned `updated_script_json` carries
     `meta.freeze_unload_ok=False` (NOT missing).
  3. The returned JSON's stamp matches `led.data["meta"][
     "freeze_unload_ok"]` post-finally (consistency).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_lfc_b1_cascade_unload_in_finally.py)
# ---------------------------------------------------------------------------


def _ledger_obj():
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_b2_test",
        "cast": [], "scenes": [], "shots": [], "beats": [],
        "lines": [], "music": [], "clips": [],
        "meta": {"episode_title": "b2", "style": "x"},
    }
    return SimpleNamespace(
        data=data, episode_id=data["episode_id"], save=lambda: None,
    )


class _LoaderStubContext:
    """Patch `_otr_model_loader` + `production_ledger` module
    attributes the cascade node lazy-imports inside run(). The
    `unload_raises` knob simulates an unload failure so the
    `freeze_unload_ok=False` path can be tested."""

    def __init__(self, *, unload_raises: bool = False):
        self._unload_raises = unload_raises
        self.unload_mock = MagicMock(
            side_effect=(
                RuntimeError("simulated unload failure")
                if unload_raises else None
            )
        )
        self.led = _ledger_obj()
        self._patches: list = []

    def __enter__(self):
        from nodes import _otr_model_loader as _real_loader
        from nodes import production_ledger as _real_pl

        gen_fn = lambda *a, **k: ""  # noqa: E731
        polish_fn = lambda *a, **k: ""  # noqa: E731

        p_load = patch.object(
            _real_loader, "load_llm",
            return_value={"model": object(), "tokenizer": object()},
        )
        p_make = patch.object(
            _real_loader, "make_generate_fn", return_value=gen_fn,
        )
        p_polish = patch.object(
            _real_loader, "make_polish_generate_fn",
            return_value=polish_fn,
        )
        p_unload = patch.object(
            _real_loader, "unload_llm", new=self.unload_mock,
        )
        p_has = patch.object(
            _real_pl, "has_current_ledger", return_value=True,
        )
        p_peek = patch.object(
            _real_pl, "peek_ledger", return_value=self.led,
        )
        p_assemble = patch.object(
            _real_pl, "assemble_script_text_from_ledger",
            return_value="rebuilt script text",
        )

        for p in (p_load, p_make, p_polish, p_unload,
                  p_has, p_peek, p_assemble):
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *_exc):
        for p in self._patches:
            p.stop()
        return False


def _stub_reviewer():
    """Return a fake `review_ledger` that yields a clean
    ReviewerDisposition so the cascade reaches the normal-success
    path that exercises the B2 reserialization."""
    from nodes import _otr_ledger_reviewer as _OTRLR

    def fake_review(_fn, _led):
        return _OTRLR.ReviewerDisposition(
            verdict="clean_no_edits",
            pre_audit_violations=0,
            pre_audit_repairs_applied=0,
            doctor_edits_proposed=0,
            doctor_edits_applied=0,
            post_audit_violations=0,
        )
    return fake_review


# ---------------------------------------------------------------------------
# 1. Stamp visible in returned JSON on clean unload
# ---------------------------------------------------------------------------


class TestFreezeUnloadStampVisible:
    def test_freeze_unload_ok_stamp_in_returned_json(self):
        """Happy path: unload succeeds -> returned
        updated_script_json carries meta.freeze_unload_ok=True."""
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH

        with _LoaderStubContext() as ctx:
            with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                              side_effect=_stub_reviewer()):
                inst = OTR_LedgerFreezeCascade()
                result = inst.run(
                    script_text="x", script_json="{}",
                    news_used="", estimated_minutes=15,
                    technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )

        # Output slot 1 is updated_script_json per cascade RETURN_NAMES.
        assert isinstance(result, tuple) and len(result) == 7
        returned_json = result[1]
        parsed = json.loads(returned_json)
        meta = parsed.get("meta", {})
        assert "freeze_unload_ok" in meta, (
            "S34 B2 P1 regressed: returned updated_script_json does NOT "
            "contain meta.freeze_unload_ok. The reserialization after "
            "the finally block was the fix."
        )
        assert meta["freeze_unload_ok"] is True, (
            f"S34 B2 P1 regressed: meta.freeze_unload_ok in returned "
            f"JSON is {meta['freeze_unload_ok']!r}, expected True on "
            f"clean unload"
        )

    def test_freeze_unload_failure_stamp_in_returned_json(self):
        """Failure path: unload raises -> returned JSON carries
        meta.freeze_unload_ok=False (NOT missing, NOT True)."""
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH

        with _LoaderStubContext(unload_raises=True) as ctx:
            with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                              side_effect=_stub_reviewer()):
                inst = OTR_LedgerFreezeCascade()
                result = inst.run(
                    script_text="x", script_json="{}",
                    news_used="", estimated_minutes=15,
                    technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )

        # Cascade still returns its 7-tuple (the unload failure is
        # logged, stamped, and the cascade carries on; only downstream
        # video nodes might decide to bail).
        assert isinstance(result, tuple) and len(result) == 7
        returned_json = result[1]
        parsed = json.loads(returned_json)
        meta = parsed.get("meta", {})
        assert "freeze_unload_ok" in meta, (
            "S34 B2 P1 regressed: returned JSON missing "
            "freeze_unload_ok stamp on unload failure"
        )
        assert meta["freeze_unload_ok"] is False, (
            f"S34 B2 P1 regressed: meta.freeze_unload_ok in returned "
            f"JSON is {meta['freeze_unload_ok']!r}, expected False on "
            f"unload failure"
        )

    def test_freeze_unload_stamp_consistent_with_led_data(self):
        """Consistency: returned JSON's freeze_unload_ok must match
        what got stamped on `led.data` post-finally. The
        reserialization is a faithful snapshot, not a divergent
        write."""
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH

        # Run with unload that succeeds.
        with _LoaderStubContext() as ctx:
            with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                              side_effect=_stub_reviewer()):
                inst = OTR_LedgerFreezeCascade()
                result = inst.run(
                    script_text="x", script_json="{}",
                    news_used="", estimated_minutes=15,
                    technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )

            # Direct stamp on led.data
            led_stamp = ctx.led.data["meta"].get("freeze_unload_ok")
            # Stamp in returned JSON
            returned_stamp = json.loads(result[1]).get(
                "meta", {}
            ).get("freeze_unload_ok")

        assert led_stamp == returned_stamp, (
            f"S34 B2 P1 regressed: returned JSON freeze_unload_ok "
            f"({returned_stamp!r}) diverges from led.data stamp "
            f"({led_stamp!r}). The reserialization must be a faithful "
            f"snapshot of led.data after the finally block."
        )
