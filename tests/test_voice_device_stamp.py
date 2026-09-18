"""A cuda:1 voice stamp reaches the adapter as cuda:1 -- never card zero.

GO_FORWARD 2026-09-18: `_voice_device_from_ledger` accepted only the three
legacy names, so the ordinal CastLock stamps from `resolve_device`
(``cuda:1`` on a second card) fell through to the ``cuda`` default and the
voice loaded on card 0. Round-trip the stamp or fail loud. CPU only.
"""
from __future__ import annotations

import inspect
import json

import pytest

from nodes._otr_voice_node_common import _voice_device_from_ledger


def _ledger(device):
    return json.dumps({"meta": {"voice_device": device}, "lines": []})


@pytest.mark.parametrize("stamp", ["cuda", "cpu", "mps", "cuda:1", "cuda:0",
                                   "cuda:7", " CUDA:1 ", "xpu", "npu:1",
                                   "privateuseone"])
def test_a_resolved_stamp_round_trips_as_written(stamp):
    """Any device kind core resolved passes through; the host fails loud
    at the adapter if it lacks it. Never a guessed card."""
    assert _voice_device_from_ledger(_ledger(stamp)) == stamp.strip().lower()


def test_an_ordinal_stamp_is_not_card_zero():
    assert _voice_device_from_ledger(_ledger("cuda:1")) == "cuda:1"


@pytest.mark.parametrize("stamp", ["default", "gpu:1", "gpu", "cuda:x",
                                   "cuda:", "two words", "1cuda"])
def test_an_unresolved_or_malformed_stamp_fails_loud(stamp):
    """`default` / `gpu:N` are dropdown WORDS CastLock resolves before it
    stamps; on a ledger they mean the stamp never went through resolve_device."""
    with pytest.raises(ValueError) as caught:
        _voice_device_from_ledger(_ledger(stamp))
    assert stamp in str(caught.value)


def test_no_stamp_anywhere_keeps_the_nv50_baseline():
    assert _voice_device_from_ledger("", "") == "cuda"
    assert _voice_device_from_ledger(json.dumps({"meta": {}}), "[]") == "cuda"
    assert _voice_device_from_ledger("not json", "") == "cuda"


def test_the_script_ledger_is_the_fallback_and_an_empty_stamp_is_skipped():
    assert _voice_device_from_ledger(_ledger(""), _ledger("cuda:1")) == "cuda:1"
    assert _voice_device_from_ledger("", _ledger("cpu")) == "cpu"


def test_the_chatterbox_worker_accepts_an_ordinal_device():
    """argparse `choices` would SystemExit the worker on --device cuda:1."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "_otr_chatterbox_worker.py").read_text(encoding="utf-8")
    device_arg = src[src.index('"--device"'):]
    device_arg = device_arg[:device_arg.index(")")]
    assert "choices" not in device_arg


def test_adapters_branch_on_the_device_kind_not_the_literal():
    """A round-tripped cuda:1 must still be CUDA to musicgen's dtype pick,
    stable-audio's seeding and Bark's dtype pick."""
    from nodes import _otr_bark_lib as bark
    from nodes._otr_audio_engines import eng_musicgen, eng_stable_audio
    for module in (eng_musicgen, eng_stable_audio, bark):
        src = inspect.getsource(module)
        assert 'device == "cuda" else torch.float32' not in src, module.__name__
        assert 'if dev == "cuda":' not in src, module.__name__
        assert '.split(":", 1)[0]' in src, module.__name__
