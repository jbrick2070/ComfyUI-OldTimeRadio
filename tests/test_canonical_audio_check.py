"""Launch the fresh canonical check outside collection-time module substitutes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE_LABELS = {
    "canonical_audio_opening",
    "canonical_audio_no_opening",
    "canonical_audio_silence_opening",
}
CLEAN_ENHANCE_SETTINGS = {
    "target_sample_rate": 48000,
    "spatial_width": 0.0,
    "haas_delay_ms": 0.0,
    "bass_warmth": 0.0,
    "lpf_cutoff_hz": 0.0,
    "tape_emulation": "off",
}
ZERO_PROBE = {"nonzero_samples": 0, "shape": [1, 2, 96000], "sample_rate": 48000}


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

    # Clean public contract: schema defaults, the saved canonical widgets and a
    # default call over asymmetric stereo that must come back untouched.
    clean = receipt["clean_default_check"]
    assert clean["schema_defaults"] == CLEAN_ENHANCE_SETTINGS
    assert clean["saved_node_widgets"] == CLEAN_ENHANCE_SETTINGS
    assert clean["asymmetric_stereo"] == {"shape": [1, 2, 96000],
                                          "sample_rate": 48000, "equal": True}
    # No public tape mode may add anything to silence.
    tape = receipt["public_tape_checks"]
    assert set(tape) == {"off", "subtle", "medium", "heavy"}
    assert all(probe == ZERO_PROBE for probe in tape.values()), tape

    cases = receipt["cases"]
    assert {case["episode"] for case in cases} == CASE_LABELS
    assert len(cases) == len(CASE_LABELS)
    for case in cases:
        assert case["canonical_sha256"] == digest
        assert Path(case["ledger"]).is_file()
        assert hashlib.sha256(Path(case["master"]).read_bytes()).hexdigest() == case["master_sha256"]
        if case["episode"] == "canonical_audio_silence_opening":
            assert case["measured_scene_offset_s"] is None
            assert case["nonzero_samples"] == {"scene": 0, "enhanced": 0, "master": 0}
        else:
            assert isinstance(case["measured_scene_offset_s"], float)
            assert case["nonzero_samples"]["master"] > 0

    # A separate fresh process must reproduce the WAVs byte for byte: the
    # supplied chirps are deterministic and the clean settings add nothing, so
    # no fixture seed is needed and any drift is a real production change.
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
