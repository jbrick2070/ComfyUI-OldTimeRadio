"""Chunk E cleanbreak tests (2026-06-08).

Validates:
  1. Workflow JSON wire state after Chunk E:
       - Link 248 (procgen-mp4 -> mux.master_audio_path) is GONE
       - Link 262 (procgen-mp4 -> render_batch.master_audio_path) is GONE
       - Link 263 (assembler.output_path -> mux.master_audio_path) present
       - Link 264 (assembler.output_path -> render_batch.master_audio_path) present
       - Link 246 (procgen-mp4 -> silent_composite.base_video_path) kept
  2. OTR_EpisodeAssembler WAV save: saves a 16-bit PCM WAV and returns
     a real file path (not the old placeholder string).
  3. validate_workflow_contract passes on the rewired JSON.
  4. Tombstoned type names caught by WorkflowDeletedNodeError.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. Workflow wire state
# ---------------------------------------------------------------------------

class TestChunkEWire:
    @pytest.fixture(autouse=True)
    def _wf(self):
        self.wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        self.link_map = {L[0]: L for L in self.wf["links"]}

    def test_link_248_cut(self):
        assert 248 not in self.link_map, (
            "Link 248 (procgen-mp4 -> mux.master_audio_path) must be cut in Chunk E"
        )

    def test_link_262_cut(self):
        assert 262 not in self.link_map, (
            "Link 262 (procgen-mp4 -> render_batch.master_audio_path) must be cut in Chunk E"
        )

    def test_link_263_present(self):
        assert 263 in self.link_map, "Link 263 (assembler.output_path -> mux) missing"
        lnk = self.link_map[263]
        assert lnk[1] == 7,  f"Link 263 src must be node 7 (EpisodeAssembler), got {lnk[1]}"
        assert lnk[2] == 1,  f"Link 263 src slot must be 1 (output_path), got {lnk[2]}"
        assert lnk[3] == 85, f"Link 263 dst must be node 85 (MasterAudioMux), got {lnk[3]}"
        assert lnk[4] == 1,  f"Link 263 dst slot must be 1 (master_audio_path), got {lnk[4]}"
        assert lnk[5] == "STRING"

    def test_link_264_present(self):
        assert 264 in self.link_map, "Link 264 (assembler.output_path -> render_batch) missing"
        lnk = self.link_map[264]
        assert lnk[1] == 7,  f"Link 264 src must be node 7, got {lnk[1]}"
        assert lnk[2] == 1,  f"Link 264 src slot must be 1 (output_path), got {lnk[2]}"
        assert lnk[3] == 92, f"Link 264 dst must be node 92 (VideoRenderBatch), got {lnk[3]}"
        assert lnk[4] == 1,  f"Link 264 dst slot must be 1 (master_audio_path), got {lnk[4]}"
        assert lnk[5] == "STRING"

    def test_link_246_kept(self):
        assert 246 in self.link_map, "Link 246 (procgen-mp4 floor -> SilentComposite) must be kept"
        lnk = self.link_map[246]
        assert lnk[1] == 12 and lnk[2] == 0 and lnk[3] == 84 and lnk[4] == 0

    def test_node7_output1_wired_to_263_264(self):
        n7 = next(n for n in self.wf["nodes"] if n["id"] == 7)
        out1_links = n7["outputs"][1].get("links") or []
        assert 263 in out1_links, "Node 7 output[1] must list link 263"
        assert 264 in out1_links, "Node 7 output[1] must list link 264"

    def test_no_duplicate_link_ids(self):
        ids = [L[0] for L in self.wf["links"]]
        assert len(ids) == len(set(ids)), f"Duplicate link IDs: {[x for x in ids if ids.count(x)>1]}"

    def test_last_link_id_updated(self):
        assert self.wf.get("last_link_id") == 264


# ---------------------------------------------------------------------------
# 2. EpisodeAssembler WAV save
# ---------------------------------------------------------------------------

class TestEpisodeAssemblerWavSave:
    """CPU-only test: patches in-flight ledger so the assembler picks a
    temp dir, then verifies a valid 16-bit PCM WAV is written and
    output_path is a real file path."""

    def _make_waveform(self, seconds=2.0, sr=24000, channels=2):
        import torch
        samples = int(seconds * sr)
        # mono -> (1, channels, samples) shape that EpisodeAssembler expects
        data = torch.zeros(1, channels, samples)
        return {"waveform": data, "sample_rate": sr}

    def test_wav_saved_and_path_returned(self, tmp_path):
        import torch
        import importlib, sys

        # Ensure fresh import in case scene_sequencer cached old state
        for key in list(sys.modules.keys()):
            if "scene_sequencer" in key:
                del sys.modules[key]

        sys.path.insert(0, str(REPO_ROOT / "nodes"))
        try:
            import scene_sequencer as sc
        except ImportError:
            pytest.skip("scene_sequencer import requires ComfyUI context")

        # Build a minimal audio input
        sr = 24000
        dur = 1.0
        samples = int(sr * dur)
        waveform = torch.zeros(1, 2, samples)
        audio_in = {"waveform": waveform, "sample_rate": sr}

        # Patch in_flight_ledger_path to return a synthetic ledger path
        # inside tmp_path so the WAV lands there.
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        ep_id = "test_ep_wav"
        fake_ledger = audio_dir / f"{ep_id}_ledger.json"
        fake_ledger.write_text("{}", encoding="utf-8")

        import nodes._otr_ledger as oled
        original_fn = oled.in_flight_ledger_path
        oled.in_flight_ledger_path = lambda: fake_ledger

        try:
            node = sc.EpisodeAssembler()
            # Call assemble with a single silent scene_audio (matches
            # EpisodeAssembler.assemble signature)
            result = node.assemble(
                scene_audio=audio_in,
                episode_title="Test Episode",
                opening_theme_audio=None,
                closing_theme_audio=None,
            )
        finally:
            oled.in_flight_ledger_path = original_fn

        _audio_out, output_path, _info, _audio_done = result

        # output_path must be a real file (not the old placeholder)
        assert os.path.isfile(output_path), (
            f"output_path must be a real WAV file, got: {output_path!r}"
        )
        assert output_path.endswith(".wav"), f"Expected .wav, got: {output_path}"

        # Validate it is a readable 16-bit PCM WAV
        with wave.open(output_path, "rb") as wf:
            assert wf.getsampwidth() == 2, "Expected 16-bit PCM"
            assert wf.getnchannels() == 2
            assert wf.getframerate() == sr
            assert wf.getnframes() == samples

    def test_wav_save_failure_non_fatal(self, tmp_path):
        """If WAV save fails, assemble must still return without raising."""
        import torch
        import importlib, sys

        for key in list(sys.modules.keys()):
            if "scene_sequencer" in key:
                del sys.modules[key]

        sys.path.insert(0, str(REPO_ROOT / "nodes"))
        try:
            import scene_sequencer as sc
        except ImportError:
            pytest.skip("scene_sequencer import requires ComfyUI context")

        sr = 24000
        samples = int(sr * 0.5)
        waveform = torch.zeros(1, 1, samples)
        audio_in = {"waveform": waveform, "sample_rate": sr}

        import nodes._otr_ledger as oled
        original_fn = oled.in_flight_ledger_path
        # Return a path whose parent doesn't exist and can't be created
        oled.in_flight_ledger_path = lambda: Path("/nonexistent_otr_path/audio/ep_ledger.json")

        try:
            node = sc.EpisodeAssembler()
            result = node.assemble(
                scene_audio=audio_in,
                episode_title="Fail Test",
                opening_theme_audio=None,
                closing_theme_audio=None,
            )
        finally:
            oled.in_flight_ledger_path = original_fn

        _audio_out, output_path, _info, _audio_done = result
        # Must not crash; output_path may be a fallback string or tmp path
        assert isinstance(output_path, str)


# ---------------------------------------------------------------------------
# 3. validate_workflow_contract on the rewired JSON
# ---------------------------------------------------------------------------

class TestWorkflowContractPostChunkE:
    def test_validate_passes(self):
        try:
            from nodes._workflow_validation import validate_workflow_contract
            import nodes as nodes_pkg
            mappings = getattr(nodes_pkg, "NODE_CLASS_MAPPINGS", {})
        except ImportError:
            pytest.skip("nodes package not importable in this env")

        wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        # Should not raise
        validate_workflow_contract(wf, mappings, strict_unknown_types=False)


# ---------------------------------------------------------------------------
# 4. Tombstoned type names
# ---------------------------------------------------------------------------

class TestTombstonesRaiseValidationError:
    def _wf_with_node(self, node_type):
        wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        wf["nodes"].append({
            "id": 9999, "type": node_type,
            "inputs": [], "outputs": [], "widgets_values": [],
        })
        return wf

    @pytest.mark.parametrize("node_type", [
        "OTR_VideoPlan",
        "OTR_ShotDurationCalculator",
        "OTR_FixedShotDurationStub",
        "OTR_RenderPlan",
    ])
    def test_tombstoned_node_raises(self, node_type):
        try:
            from nodes._workflow_validation import (
                validate_workflow_contract, WorkflowDeletedNodeError,
            )
            import nodes as nodes_pkg
            mappings = getattr(nodes_pkg, "NODE_CLASS_MAPPINGS", {})
        except ImportError:
            pytest.skip("nodes package not importable in this env")

        wf = self._wf_with_node(node_type)
        with pytest.raises(WorkflowDeletedNodeError):
            validate_workflow_contract(wf, mappings, strict_unknown_types=False)
