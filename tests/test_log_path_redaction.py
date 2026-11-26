"""OTR's log lines carry no absolute paths.

ComfyUI serves its log at /internal/logs with no login. The registry's policy
reviewer signed 2.3.51 "information-disclosure" and flagged 2.3.61-2.3.63 with
no reason; full paths in ordinary log lines (the Hugging Face cache, the output
folder, model snapshots, episode files) were the leading candidate. A logging
record factory in nodes/_otr_error_text.py now cuts every path in a record from
this pack to its last component -- message, traceback and stack text. Every
other pack's records pass untouched. CPU only. UTF-8 no BOM.
"""
from __future__ import annotations

import logging
import os
import sys

import pytest

from nodes import _otr_error_text as ET

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INSIDE = os.path.join(_REPO, "nodes", "_otr_example_module.py")
_OUTSIDE = os.path.join(os.path.dirname(_REPO), "some_other_pack", "nodes.py")


@pytest.fixture
def factory():
    """Install the OTR factory for one test, then put the original back."""
    original = logging.getLogRecordFactory()
    plain = original
    while getattr(plain, "__otr_redacts_paths__", False):  # a leak from elsewhere
        plain = logging.LogRecord
    logging.setLogRecordFactory(plain)
    assert ET.install_log_redaction() is True
    yield logging.getLogRecordFactory()
    logging.setLogRecordFactory(original)


def _record(make, pathname, msg, args=(), exc_info=None):
    return make("OTR", logging.INFO, pathname, 1, msg, args, exc_info)


def test_a_record_from_this_pack_loses_its_folders(factory):
    rec = _record(factory, _INSIDE, "obs_publish OK -> %s",
                  ("/Users/jab/ComfyUI-Shared/output/otr/obs/ep_ja_final.mp4",))
    assert rec.getMessage() == "obs_publish OK -> ep_ja_final.mp4"
    win = _record(factory, _INSIDE,
                  r"HF_HOME=C:\Users\someone\ComfyUI-Models\huggingface | ok")
    assert win.getMessage() == "HF_HOME=huggingface | ok"


def test_a_record_from_another_pack_is_untouched(factory):
    text = "loaded /Users/jab/models/x.safetensors"
    assert _record(factory, _OUTSIDE, text).getMessage() == text


def test_urls_and_routes_survive(factory):
    text = "GET https://huggingface.co/Qwen/Qwen3.5-4B and /api/prompt"
    assert _record(factory, _INSIDE, text).getMessage() == text


def test_traceback_text_is_redacted_too(factory):
    try:
        raise FileNotFoundError("missing /home/someone/otr/episodes/ep1/audio.wav")
    except FileNotFoundError:
        rec = _record(factory, _INSIDE, "failed", exc_info=sys.exc_info())
    assert "audio.wav" in rec.exc_text
    assert "/home/someone" not in rec.exc_text
    assert os.path.dirname(os.path.abspath(__file__)) not in rec.exc_text


def test_a_real_logger_call_from_inside_the_pack_is_redacted(factory, caplog):
    """End to end through a real logger: this test file lives in the pack."""
    with caplog.at_level(logging.INFO, logger="OTR"):
        logging.getLogger("OTR").info("pinned output: %s", "/Users/jab/ComfyUI-Shared/output")
    assert caplog.records[-1].getMessage() == "pinned output: output"


def test_install_is_idempotent(factory):
    assert ET.install_log_redaction() is False
    assert logging.getLogRecordFactory() is factory


def test_both_entry_points_install_it_before_anything_logs():
    with open(os.path.join(_REPO, "__init__.py"), encoding="utf-8") as fh:
        init = fh.read()
    assert "_otr_install_log_redaction()" in init
    assert init.index("_otr_install_log_redaction()") < init.index("_otr_claim_pack(")
    with open(os.path.join(_REPO, "prestartup_script.py"), encoding="utf-8") as fh:
        pre = fh.read()
    assert "install_log_redaction()" in pre
    assert pre.index("install_log_redaction()") < pre.index("HF_HOME=%s")
