"""tests/test_lfc_b1_cascade_unload_in_finally.py

LFC commit 12.12 -- B1 fix regression. When run_freeze_cascade
raises, unload_llm() MUST still run (finally-block), and
meta.freeze_unload_ok must be stamped. Pre-fix the unload sat
outside the try block; on any cascade exception VRAM stayed
held.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ledger_obj():
    """Return a real ledger-shaped SimpleNamespace."""
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_b1_test",
        "cast": [], "scenes": [], "shots": [], "beats": [],
        "lines": [], "music": [], "clips": [],
        "meta": {"episode_title": "b1", "style": "x"},
    }
    return SimpleNamespace(
        data=data, episode_id=data["episode_id"], save=lambda: None,
    )


class _LoaderStubContext:
    """Context manager that patches the real _otr_model_loader +
    production_ledger module attributes the cascade node lazy-
    imports inside run(). Yields a record of unload_llm call count
    + the ledger handle so the test can inspect post-run state."""

    def __init__(self, *, unload_raises: bool = False):
        from unittest.mock import MagicMock as _MM
        self._unload_raises = unload_raises
        self.unload_mock = _MM(
            side_effect=(
                RuntimeError("simulated unload failure")
                if unload_raises else None
            )
        )
        self.led = _ledger_obj()
        self._patches: list = []

    def __enter__(self):
        from unittest.mock import patch as _patch
        from nodes import _otr_model_loader as _real_loader
        from nodes import production_ledger as _real_pl

        # Lazy generate_fn stubs.
        gen_fn = lambda *a, **k: ""  # noqa: E731
        polish_fn = lambda *a, **k: ""  # noqa: E731

        p_load = _patch.object(
            _real_loader, "load_llm",
            return_value={"model": object(), "tokenizer": object()},
        )
        p_make = _patch.object(
            _real_loader, "make_generate_fn", return_value=gen_fn,
        )
        p_polish = _patch.object(
            _real_loader, "make_polish_generate_fn",
            return_value=polish_fn,
        )
        p_unload = _patch.object(
            _real_loader, "unload_llm", new=self.unload_mock,
        )
        p_has = _patch.object(
            _real_pl, "has_current_ledger", return_value=True,
        )
        p_peek = _patch.object(
            _real_pl, "peek_ledger", return_value=self.led,
        )
        p_assemble = _patch.object(
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


# ---------------------------------------------------------------------------
# B1 — unload runs in finally
# ---------------------------------------------------------------------------


class TestB1UnloadInFinally:
    def test_unload_runs_when_cascade_raises(self):
        """run_freeze_cascade raises -> unload_llm STILL called once.

        ComfyUI's red-node failure convention: the cascade lets the
        exception propagate so the operator sees the failure on the
        canvas. The B1 contract is that VRAM is released BEFORE the
        exception propagates -- that's what the try/finally
        guarantees. Catching the exception and returning a synthetic
        verdict would silently mask cascade failures.
        """
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH

        def raising_cascade(*args, **kwargs):
            raise RuntimeError("simulated cascade explosion")

        with _LoaderStubContext() as ctx:
            with patch.object(_LFC_ORCH, "run_freeze_cascade",
                              side_effect=raising_cascade):
                inst = OTR_LedgerFreezeCascade()
                with pytest.raises(RuntimeError,
                                   match="simulated cascade explosion"):
                    inst.run(
                        script_text="x", script_json="{}",
                        news_used="", estimated_minutes=15,
 technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                    )

        # The B1 invariant: unload ran in the finally block before
        # the exception propagated.
        assert ctx.unload_mock.call_count == 1, (
            f"B1: unload_llm should run on cascade exception. "
            f"called={ctx.unload_mock.call_count} times"
        )

    def test_meta_freeze_unload_ok_stamped_true_on_clean_run(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0,
            )

        with _LoaderStubContext() as ctx:
            with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                              side_effect=fake_review):
                inst = OTR_LedgerFreezeCascade()
                inst.run(
                    script_text="x", script_json="{}",
                    news_used="", estimated_minutes=15,
 technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )

        assert ctx.led.data["meta"].get("freeze_unload_ok") is True

    def test_meta_freeze_unload_ok_stamped_false_on_unload_fail(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0,
            )

        with _LoaderStubContext(unload_raises=True) as ctx:
            with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                              side_effect=fake_review):
                inst = OTR_LedgerFreezeCascade()
                result = inst.run(
                    script_text="x", script_json="{}",
                    news_used="", estimated_minutes=15,
 technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )

        # Unload attempted (raised) -- count is 1.
        assert ctx.unload_mock.call_count == 1
        # Meta stamp reflects the failure.
        assert ctx.led.data["meta"].get("freeze_unload_ok") is False
        # Cascade still returns its verdict.
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_unload_runs_when_serialize_raises(self):
        """A failure inside assemble_script_text_from_ledger must
        still unload. The cascade catches the inner try -- this
        test verifies the finally runs on the OUTER block path."""
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR
        from nodes import production_ledger as _real_pl

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0,
            )

        with _LoaderStubContext() as ctx:
            # Override the assemble_script_text stub to raise.
            with patch.object(
                _real_pl, "assemble_script_text_from_ledger",
                side_effect=RuntimeError("assemble crash"),
            ), patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                            side_effect=fake_review):
                inst = OTR_LedgerFreezeCascade()
                inst.run(
                    script_text="x", script_json="{}",
                    news_used="", estimated_minutes=15,
 technical_model="mistralai/Mistral-Nemo-Instruct-2407",
                )

        assert ctx.unload_mock.call_count == 1
