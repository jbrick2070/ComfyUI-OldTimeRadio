"""The detached campaign must trust the canonical terminal result, not exit 0 alone."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


HELPER = Path(__file__).parents[1] / "scripts" / "otr_campaign_receipt.ps1"


def _probe(tmp_path: Path, text: str, *, exit_code: int = 0, queue_empty: bool = True):
    stdout = tmp_path / "runner.stdout.log"
    stdout.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(HELPER),
            "-Probe",
            "-StdoutPath",
            str(stdout),
            "-ExitCode",
            str(exit_code),
            "-QueueEmpty",
            str(int(queue_empty)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return result, payload


def test_result_fail_with_zero_exit_is_not_a_pass(tmp_path: Path):
    result, receipt = _probe(
        tmp_path,
        "[canonical-api] RESULT FAIL prompt_id=failed-live-leg\n",
    )
    assert result.returncode != 0
    assert receipt["status"] == "FAIL"
    assert receipt["terminal_result"] == "FAIL"
    assert "terminal_result=FAIL" in receipt["reason"]


def test_missing_terminal_result_is_not_a_pass(tmp_path: Path):
    result, receipt = _probe(tmp_path, "runner exited without a terminal marker\n")
    assert result.returncode != 0
    assert receipt["status"] == "FAIL"
    assert receipt["terminal_result"] == ""
    assert "terminal_result=MISSING" in receipt["reason"]


def test_result_success_requires_zero_exit_and_empty_queue(tmp_path: Path):
    result, receipt = _probe(
        tmp_path,
        "[canonical-api] RESULT SUCCESS prompt_id=green-live-leg\n",
    )
    assert result.returncode == 0
    assert receipt["status"] == "PASS"
    assert receipt["terminal_result"] == "SUCCESS"

    result, receipt = _probe(
        tmp_path,
        "[canonical-api] RESULT SUCCESS prompt_id=contradictory-leg\n",
        exit_code=1,
    )
    assert result.returncode != 0
    assert receipt["status"] == "FAIL"
    assert "child_exit_code=1" in receipt["reason"]
