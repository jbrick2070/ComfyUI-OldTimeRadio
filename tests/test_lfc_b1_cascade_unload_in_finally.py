"""The cascade releases a local LLM on every exit path and acquires none.

The freeze node validates that the technical socket is wired, runs the
deterministic cascade with no generation callback, and releases any local LLM
still resident in its ``finally`` block. These tests stub the orchestrator only
for fault injection and argument capture; the fresh-process proof that the real
cascade never generates lives in ``tests/test_freeze_policy_readonly.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TECHNICAL_ID = "mistralai/Mistral-Nemo-Instruct-2407"


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


class _AcquisitionSentinels:
    """Fail-fast stand-ins for the loader's acquisition entry points.

    The freeze node must never reach either of them; a call raises and is
    counted so the assertion names the offending entry point.
    """

    def __init__(self):
        self.calls = {"request_slot": 0, "make_generate_fn": 0}

    def _poison(self, name):
        def sentinel(*args, **kwargs):
            self.calls[name] += 1
            raise AssertionError(f"freeze must not call {name}")
        return sentinel

    @property
    def request_slot(self):
        return self._poison("request_slot")

    @property
    def make_generate_fn(self):
        return self._poison("make_generate_fn")

    def assert_untouched(self):
        assert self.calls == {"request_slot": 0, "make_generate_fn": 0}, self.calls


def _patch_runtime(led, unload, *, assemble=None, sentinels=None):
    from nodes import _otr_model_loader as loader
    from nodes import production_ledger as ledger_module

    if assemble is None:
        assemble = MagicMock(return_value="rebuilt script text")
    if sentinels is None:
        sentinels = _AcquisitionSentinels()
    return (
        patch.object(ledger_module, "has_current_ledger", return_value=True),
        patch.object(ledger_module, "peek_ledger", return_value=led),
        patch.object(
            ledger_module, "assemble_script_text_from_ledger", assemble),
        patch.object(loader, "request_slot", sentinels.request_slot),
        patch.object(loader, "make_generate_fn", sentinels.make_generate_fn),
        patch.object(loader, "unload_llm_if_local_resident", unload),
    )


def _call_node(**overrides):
    from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade

    kwargs = dict(
        script_text="incoming",
        script_json="{}",
        news_used="",
        estimated_minutes=15,
        technical_model=TECHNICAL_ID,
    )
    kwargs.update(overrides)
    return OTR_LedgerFreezeCascade().run(**kwargs)


def _capturing_cascade(disposition=None, error=None):
    """A ``run_freeze_cascade`` stub that records how it was called."""
    def cascade(*args, **kwargs):
        cascade.calls.append((args, kwargs))
        if error is not None:
            raise error
        return disposition
    cascade.calls = []
    return cascade


def _assert_cascade_called_without_model(cascade, led, *, phase_7=True, phase_8=True):
    assert len(cascade.calls) == 1
    args, kwargs = cascade.calls[0]
    assert args == (None, led), args
    assert kwargs == {
        "enable_phase_7_audio_readiness": phase_7,
        "enable_phase_8_video_readiness": phase_8,
    }, kwargs


def test_unload_runs_when_cascade_raises():
    from nodes import _otr_freeze_cascade as cascade_module

    led = _ledger_obj()
    unload = MagicMock()
    sentinels = _AcquisitionSentinels()
    stub = _capturing_cascade(error=RuntimeError("simulated cascade explosion"))
    patches = _patch_runtime(led, unload, sentinels=sentinels)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(cascade_module, "run_freeze_cascade", stub):
            with pytest.raises(RuntimeError, match="cascade explosion"):
                _call_node()

    assert unload.call_count == 1
    assert led.data["meta"]["freeze_unload_ok"] is True
    sentinels.assert_untouched()
    _assert_cascade_called_without_model(stub, led)


def test_clean_run_stamps_successful_unload():
    from nodes import _otr_freeze_cascade as cascade_module

    led = _ledger_obj()
    unload = MagicMock()
    sentinels = _AcquisitionSentinels()
    stub = _capturing_cascade(_disposition())
    patches = _patch_runtime(led, unload, sentinels=sentinels)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(cascade_module, "run_freeze_cascade", stub):
            result = _call_node(
                enable_phase_7_audio_readiness=False,
                enable_phase_8_video_readiness=True,
            )

    assert unload.call_count == 1
    assert led.data["meta"]["freeze_unload_ok"] is True
    assert len(result) == 7
    assert result[4] == "frozen_clean"
    assert result[1] == result[6]
    assert json.loads(result[1])["meta"]["freeze_unload_ok"] is True
    sentinels.assert_untouched()
    _assert_cascade_called_without_model(stub, led, phase_7=False, phase_8=True)


def test_unload_failure_is_stamped_without_hiding_verdict():
    from nodes import _otr_freeze_cascade as cascade_module

    led = _ledger_obj()
    unload = MagicMock(side_effect=RuntimeError("unload failed"))
    sentinels = _AcquisitionSentinels()
    patches = _patch_runtime(led, unload, sentinels=sentinels)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            cascade_module, "run_freeze_cascade", return_value=_disposition()):
            result = _call_node()

    assert unload.call_count == 1
    assert led.data["meta"]["freeze_unload_ok"] is False
    assert result[4] == "frozen_clean"
    assert len(result) == 7
    assert result[1] == result[6]
    assert json.loads(result[1])["meta"]["freeze_unload_ok"] is False
    sentinels.assert_untouched()


def test_unload_runs_when_script_assembly_falls_back():
    from nodes import _otr_freeze_cascade as cascade_module

    led = _ledger_obj()
    unload = MagicMock()
    sentinels = _AcquisitionSentinels()
    assemble = MagicMock(side_effect=RuntimeError("assemble crash"))
    stub = _capturing_cascade(_disposition())
    patches = _patch_runtime(led, unload, assemble=assemble, sentinels=sentinels)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(cascade_module, "run_freeze_cascade", stub):
            result = _call_node()

    assert unload.call_count == 1
    assert result[0] == "incoming"
    assert len(result) == 7
    assert json.loads(result[1])["meta"]["freeze_unload_ok"] is True
    sentinels.assert_untouched()
    _assert_cascade_called_without_model(stub, led)


# ---------------------------------------------------------------------------
# Early exits: no acquisition, no cascade, no final unload
# ---------------------------------------------------------------------------


def _early_exit_patches(sentinels, unload, *, current_ledger):
    from nodes import _otr_freeze_cascade as cascade_module
    from nodes import _otr_model_loader as loader
    from nodes import production_ledger as ledger_module

    patches = [
        patch.object(loader, "request_slot", sentinels.request_slot),
        patch.object(loader, "make_generate_fn", sentinels.make_generate_fn),
        patch.object(loader, "unload_llm_if_local_resident", unload),
        patch.object(
            cascade_module, "run_freeze_cascade",
            side_effect=AssertionError("cascade must not run on an early exit")),
    ]
    if current_ledger is None:
        patches.append(
            patch.object(ledger_module, "has_current_ledger", return_value=False))
    else:
        patches.append(
            patch.object(ledger_module, "has_current_ledger", return_value=True))
        patches.append(
            patch.object(ledger_module, "peek_ledger", return_value=current_ledger))
    return patches


def test_replay_descriptor_returns_pass_through_before_validation():
    sentinels = _AcquisitionSentinels()
    unload = MagicMock()
    replay_json = json.dumps({"meta": {"replay_from": "bundle"}})
    patches = _early_exit_patches(sentinels, unload, current_ledger=None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = _call_node(script_json=replay_json, technical_model="")

    assert len(result) == 7
    assert result[4] == "replay"
    assert result[0] == "incoming"
    assert result[1] == replay_json
    assert result[6] == replay_json
    assert result[3] == 15
    assert unload.call_count == 0
    sentinels.assert_untouched()


def test_no_current_ledger_returns_needs_full_rerun_without_acquisition():
    sentinels = _AcquisitionSentinels()
    unload = MagicMock()
    patches = _early_exit_patches(sentinels, unload, current_ledger=None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = _call_node(script_json='{"meta": {"episode_title": "x"}}')

    assert len(result) == 7
    assert result[4] == "needs_full_rerun"
    assert result[5] == 0
    assert result[1] == result[6]
    error_state = json.loads(result[1])
    assert error_state["schema_version"] == "synthetic_error_state"
    assert error_state["meta"]["freeze_verdict"] == "needs_full_rerun"
    assert error_state["meta"]["freeze_disposition"]["skipped_reason"] == (
        "no_writer_produced_ledger")
    assert unload.call_count == 0
    sentinels.assert_untouched()


def test_none_ledger_handle_returns_needs_full_rerun_without_acquisition():
    from nodes import production_ledger as ledger_module

    sentinels = _AcquisitionSentinels()
    unload = MagicMock()
    loader_and_cascade = _early_exit_patches(sentinels, unload, current_ledger=None)[:4]
    with (
        loader_and_cascade[0], loader_and_cascade[1],
        loader_and_cascade[2], loader_and_cascade[3],
        patch.object(ledger_module, "has_current_ledger", return_value=True),
        patch.object(ledger_module, "peek_ledger", return_value=None),
    ):
        result = _call_node()

    assert len(result) == 7
    assert result[4] == "needs_full_rerun"
    assert result[5] == 0
    assert result[1] == result[6]
    assert json.loads(result[1])["schema_version"] == "synthetic_error_state"
    assert unload.call_count == 0
    sentinels.assert_untouched()


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_technical_model_refuses_before_the_cascade(blank):
    from nodes._otr_model_inputs import MissingModelInputError

    led = _ledger_obj()
    sentinels = _AcquisitionSentinels()
    unload = MagicMock()
    patches = _early_exit_patches(sentinels, unload, current_ledger=led)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with pytest.raises(MissingModelInputError, match="technical"):
            _call_node(technical_model=blank)

    assert unload.call_count == 0
    assert "freeze_unload_ok" not in led.data["meta"]
    sentinels.assert_untouched()


# ---------------------------------------------------------------------------
# Serialization faults: the two dumps calls and their documented fallbacks
# ---------------------------------------------------------------------------


def _counting_json(failing_calls):
    """A stand-in for the node module's ``json`` name.

    ``loads`` is the real one. ``dumps`` delegates to the real function except
    on the numbered calls in ``failing_calls`` (1-based), where it raises.
    """
    real_dumps = json.dumps
    counter = {"dumps": 0}

    def dumps(*args, **kwargs):
        counter["dumps"] += 1
        if counter["dumps"] in failing_calls:
            raise TypeError(f"simulated serialization failure #{counter['dumps']}")
        return real_dumps(*args, **kwargs)

    return SimpleNamespace(loads=json.loads, dumps=dumps), counter


def _run_with_json_faults(failing_calls):
    import nodes.OTR_LedgerFreezeCascade as node_module
    from nodes import _otr_freeze_cascade as cascade_module

    led = _ledger_obj()
    unload = MagicMock()
    sentinels = _AcquisitionSentinels()
    faulty_json, counter = _counting_json(failing_calls)
    patches = _patch_runtime(led, unload, sentinels=sentinels)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            cascade_module, "run_freeze_cascade", return_value=_disposition()):
            with patch.object(node_module, "json", faulty_json):
                result = _call_node()
    assert counter["dumps"] == 2, counter
    assert unload.call_count == 1
    assert len(result) == 7
    assert result[1] == result[6]
    assert result[4] == "frozen_clean"
    sentinels.assert_untouched()
    return led, result


def test_first_serialization_failure_is_recovered_by_the_unload_receipt():
    led, result = _run_with_json_faults({1})
    payload = json.loads(result[1])
    assert payload["episode_id"] == led.episode_id
    assert payload["meta"]["freeze_unload_ok"] is True
    assert led.data["meta"]["freeze_unload_ok"] is True


def test_second_serialization_failure_keeps_the_pre_unload_json():
    led, result = _run_with_json_faults({2})
    payload = json.loads(result[1])
    assert payload["episode_id"] == led.episode_id
    # The first serialization precedes the finally-block stamp, so a fresh
    # fixture's returned JSON lacks the receipt while the live ledger has it.
    assert "freeze_unload_ok" not in payload["meta"]
    assert led.data["meta"]["freeze_unload_ok"] is True


def test_both_serialization_failures_keep_the_incoming_json():
    led, result = _run_with_json_faults({1, 2})
    assert result[1] == "{}"
    assert led.data["meta"]["freeze_unload_ok"] is True
