"""Launch the fresh canonical check outside collection-time module substitutes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_current_canonical_audio_route_and_durable_outputs(tmp_path):
    output = tmp_path / "fresh_canonical_audio"
    command = [sys.executable, str(ROOT / "scripts" / "otr_canonical_audio_check.py"),
               "--output-root", str(output)]
    env = {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt_path = output / "canonical_audio_check.json"
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    canonical = ROOT / "workflows" / "otr_canonical.json"
    assert receipt["canonical"] == str(canonical)
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert len(receipt["cases"]) == 2
    for case in receipt["cases"]:
        assert case["canonical_sha256"] == digest
        assert case["rng_seed"] == 20260910
        assert Path(case["ledger"]).is_file()
        assert hashlib.sha256(Path(case["master"]).read_bytes()).hexdigest() == case["master_sha256"]

    # A separate fresh process must reproduce the WAVs under the same fixture
    # seed; otherwise a before/after cleanup comparison would flag random hiss.
    independent_output = tmp_path / "independent_canonical_audio"
    independent = subprocess.run(command[:-1] + [str(independent_output)],
                                 cwd=ROOT, env=env, capture_output=True,
                                 text=True, encoding="utf-8", timeout=120)
    assert independent.returncode == 0, independent.stdout + independent.stderr
    independent_receipt = json.loads(
        (independent_output / "canonical_audio_check.json").read_text(encoding="utf-8"))
    assert [case["master_sha256"] for case in receipt["cases"]] == [
        case["master_sha256"] for case in independent_receipt["cases"]]

    # A rerun cannot silently reuse yesterday's artifacts as fresh evidence.
    reused = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert reused.returncode != 0
    assert "Use a fresh output root" in reused.stderr
    assert receipt_path.read_bytes() == original_receipt
