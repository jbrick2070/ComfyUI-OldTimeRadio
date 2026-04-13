"""
Phase 0: Audio Regression Baseline
====================================

Gates every v2 change against a known-good v1.5 audio output.
Audio is king. If this test fails, the change is reverted.

Usage:
------
1. Capture baseline (run once on clean v1.5 with fixed seed):
     python tests/v2/test_audio_byte_identical.py --capture-baseline

2. Regression gate (run after every code change):
     pytest tests/v2/test_audio_byte_identical.py -v

The baseline WAV and its SHA-256 hash are stored in:
  tests/v2/fixtures/baseline_v1.5.wav
  tests/v2/fixtures/baseline_v1.5.sha256

If fixtures are missing, tests skip with instructions to capture first.
"""

import hashlib
import os
import sys
import json
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure repo root is on sys.path so `tests.v2` resolves
# whether invoked via pytest (from repo root) or as a standalone script.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_FIXTURES = os.path.join(_HERE, "fixtures")
_BASELINE_WAV = os.path.join(_FIXTURES, "baseline_v1.5.wav")
_BASELINE_SHA = os.path.join(_FIXTURES, "baseline_v1.5.sha256")
_WORKFLOW = os.path.join(_HERE, "..", "..", "workflows", "otr_scifi_16gb_full.json")

# Fixed seed for deterministic audio output.
# These override the workflow's randomized seeds at runtime.
FIXED_SEEDS = {
    "OTR_Gemma4ScriptWriter": 42,
    "OTR_Gemma4Director": 42,
    "OTR_BatchBarkGenerator": 42,
    "OTR_KokoroAnnouncer": 42,
    "OTR_BatchAudioGenGenerator": 42,
    "OTR_MusicGenTheme": 42,
}


def sha256_file(path):
    """Compute SHA-256 of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    """Compute SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Fixture checks
# ---------------------------------------------------------------------------

_HAS_BASELINE = os.path.isfile(_BASELINE_WAV) and os.path.isfile(_BASELINE_SHA)

_SKIP_MSG = (
    "Audio baseline not captured yet. Run on your machine with GPU:\n"
    "  python tests/v2/test_audio_byte_identical.py --capture-baseline"
)


def _load_expected_hash():
    """Read the stored SHA-256 hash from the fixture file."""
    with open(_BASELINE_SHA, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Structural tests (always run, no GPU needed)
# ---------------------------------------------------------------------------

class TestBaselineFixtureIntegrity:
    """Verify the baseline fixture files are consistent."""

    @pytest.mark.skipif(not _HAS_BASELINE, reason=_SKIP_MSG)
    def test_baseline_wav_not_empty(self):
        size = os.path.getsize(_BASELINE_WAV)
        assert size > 44, (
            f"Baseline WAV is only {size} bytes - "
            "likely a header-only file or capture failure"
        )

    @pytest.mark.skipif(not _HAS_BASELINE, reason=_SKIP_MSG)
    def test_baseline_sha_matches_wav(self):
        """SHA file must match the actual WAV on disk."""
        expected = _load_expected_hash()
        actual = sha256_file(_BASELINE_WAV)
        assert actual == expected, (
            f"SHA mismatch: fixture says {expected[:16]}... "
            f"but WAV hashes to {actual[:16]}... - "
            "fixture pair is corrupt, re-capture baseline"
        )

    @pytest.mark.skipif(not _HAS_BASELINE, reason=_SKIP_MSG)
    def test_baseline_sha_is_valid_hex(self):
        h = _load_expected_hash()
        assert len(h) == 64, f"SHA-256 should be 64 hex chars, got {len(h)}"
        int(h, 16)  # Raises ValueError if not valid hex


class TestWorkflowSeedContract:
    """Verify the workflow JSON has seed-controllable nodes."""

    def test_workflow_exists(self):
        assert os.path.isfile(_WORKFLOW), f"Missing workflow: {_WORKFLOW}"

    def test_workflow_valid_json(self):
        with open(_WORKFLOW, encoding="utf-8") as f:
            wf = json.load(f)
        assert "nodes" in wf
        assert "links" in wf

    def test_all_seed_target_nodes_exist_in_workflow(self):
        with open(_WORKFLOW, encoding="utf-8") as f:
            wf = json.load(f)
        node_types = {n["type"] for n in wf["nodes"]}
        for target in FIXED_SEEDS:
            assert target in node_types, (
                f"Seed target {target} not found in workflow. "
                f"Available: {sorted(node_types)}"
            )

    def test_episode_assembler_present(self):
        """EpisodeAssembler is the final audio output node."""
        with open(_WORKFLOW, encoding="utf-8") as f:
            wf = json.load(f)
        types = {n["type"] for n in wf["nodes"]}
        assert "OTR_EpisodeAssembler" in types


class TestAudioRegressionGate:
    """The actual byte-identical regression gate.

    Requires:
    - Baseline fixtures captured
    - ComfyUI runtime with GPU
    - torch installed

    Skips gracefully when prerequisites are missing.
    """

    @pytest.mark.skipif(not _HAS_BASELINE, reason=_SKIP_MSG)
    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=(
            "Set OTR_REGRESSION_RUNTIME=1 to run full audio regression. "
            "Requires ComfyUI + GPU."
        ),
    )
    def test_audio_byte_identical_to_baseline(self):
        """Re-run the workflow with fixed seeds and compare output hash.

        This test is the Phase 0 gate. If it fails, the change broke audio.
        Revert immediately.
        """
        # Import only when actually running the regression
        # (avoids torch/comfyui import errors in CI/sandbox)
        from tests.v2._run_baseline import run_episode_and_get_audio_bytes

        audio_bytes = run_episode_and_get_audio_bytes(FIXED_SEEDS)
        actual_hash = sha256_bytes(audio_bytes)
        expected_hash = _load_expected_hash()

        assert actual_hash == expected_hash, (
            f"AUDIO REGRESSION FAILURE\n"
            f"Expected: {expected_hash}\n"
            f"Got:      {actual_hash}\n"
            f"The audio output changed. Revert the last change immediately."
        )


# ---------------------------------------------------------------------------
# CLI: baseline capture mode
# ---------------------------------------------------------------------------

def _capture_baseline():
    """Capture the v1.5 audio baseline.

    Run this once on clean v1.5 with GPU available:
      python tests/v2/test_audio_byte_identical.py --capture-baseline
    """
    print("=" * 60)
    print("Phase 0: Capturing audio baseline")
    print("=" * 60)
    print()

    # Ensure fixtures directory exists
    os.makedirs(_FIXTURES, exist_ok=True)

    try:
        from tests.v2._run_baseline import run_episode_and_save_wav
    except ImportError as e:
        print(f"Cannot import baseline runner: {e}")
        print("Make sure ComfyUI and torch are available.")
        sys.exit(1)

    wav_path = run_episode_and_save_wav(FIXED_SEEDS, _BASELINE_WAV)
    digest = sha256_file(wav_path)

    with open(_BASELINE_SHA, "w", encoding="utf-8") as f:
        f.write(digest + "\n")

    print()
    print(f"Baseline WAV: {wav_path}")
    print(f"  Size: {os.path.getsize(wav_path):,} bytes")
    print(f"  SHA-256: {digest}")
    print()
    print("Baseline captured. Commit both fixture files:")
    print(f"  {_BASELINE_WAV}")
    print(f"  {_BASELINE_SHA}")
    print()
    print("Every future code change will be gated against this hash.")


if __name__ == "__main__":
    if "--capture-baseline" in sys.argv:
        _capture_baseline()
    else:
        print("Usage:")
        print("  Capture: python tests/v2/test_audio_byte_identical.py --capture-baseline")
        print("  Test:    pytest tests/v2/test_audio_byte_identical.py -v")
