"""tests/test_lfc_b14_unload_on_exit.py

LFC sprint commit 12.5 -- B14 fix + C7 acceptance. Verify that
OTR_LedgerFreezeCascade.run() calls the guarded _otr_model_loader
LLM-unload handoff at cascade exit so local LLM VRAM is released before
HuMo / SignalLostVideo downstream loads, while all-cloud LLM runs skip
an unnecessary torch/CUDA teardown.

These tests inspect the source file (the runtime path needs a real
LLM + GPU which we can't fake at test time). The runtime assertion
(VRAM ~= pre-cascade reading) lives in scripts/smoke_check.py from
commit 12.7.
"""
from __future__ import annotations

import re as _re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_SRC = (REPO_ROOT / "nodes" / "OTR_LedgerFreezeCascade.py").read_text(
    encoding="utf-8"
)


def _flat(text: str) -> str:
    out = _re.sub(r"\s+", " ", text)
    return out.replace('" "', "")


_FLAT = _flat(NODE_SRC)


class TestB14UnloadOnExit:
    def test_unload_llm_called_in_run(self):
        # The runtime path must invoke the guarded unload handoff somewhere
        # in the run() method body.
        assert "_OTRML.unload_llm_if_local_resident(" in _FLAT, (
            "B14 fix: OTR_LedgerFreezeCascade.run() must call "
            "_otr_model_loader.unload_llm_if_local_resident() at cascade exit"
        )

    def test_unload_after_disp_computed(self):
        # The unload call must be AFTER run_freeze_cascade returns
        # the FreezeDisposition (we need the verdict before we let
        # go of the model). Verify ordering by source position --
        # disp = _LFC_ORCH.run_freeze_cascade(...) precedes
        # _OTRML.unload_llm_if_local_resident().
        run_idx = _FLAT.find("_LFC_ORCH.run_freeze_cascade(")
        unload_idx = _FLAT.find("_OTRML.unload_llm_if_local_resident(")
        assert run_idx != -1
        assert unload_idx != -1
        assert unload_idx > run_idx, (
            "B14 fix: unload_llm_if_local_resident must come AFTER "
            "run_freeze_cascade so the FreezeDisposition is computed first"
        )

    def test_unload_wrapped_in_best_effort_try(self):
        # An unload failure must not break the cascade return path
        # (the verdict + script_json have already been computed).
        # Verify the call is inside a try/except with a log.warning
        # fallback.
        marker = "unload_llm at cascade exit"
        idx = _FLAT.find(marker)
        assert idx != -1, (
            "B14 fix: expected best-effort log.warning on unload "
            "failure with text 'unload_llm at cascade exit'"
        )
        context = _FLAT[max(0, idx - 200): idx]
        assert "log.warning" in context, (
            "B14 fix: unload failure must log at WARNING (not "
            "debug or error) so soak telemetry surfaces it"
        )

    def test_unload_imported(self):
        # _otr_model_loader unload helpers must be reachable via the
        # _OTRML module alias the cascade node lazy-imports.
        from nodes import _otr_model_loader as _OTRML
        assert hasattr(_OTRML, "unload_llm")
        assert callable(_OTRML.unload_llm)
        assert hasattr(_OTRML, "unload_llm_if_local_resident")
        assert callable(_OTRML.unload_llm_if_local_resident)


class TestB14UnloadIsCallable:
    def test_unload_llm_no_op_when_no_cache(self):
        # unload_llm must be safe to call when no model is currently
        # loaded (e.g. test environment). Should not raise.
        from nodes import _otr_model_loader as _OTRML
        try:
            _OTRML.unload_llm()
        except Exception as exc:
            pytest.fail(
                f"unload_llm raised when no model cached: {exc!r}"
            )
