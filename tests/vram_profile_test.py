"""
vram_profile_test.py

Theme C / v1.4 — VRAM profile regression test.

Purpose
-------
Snapshot CUDA VRAM usage between major ComfyUI-OldTimeRadio pipeline phases and
assert that the peak never exceeds the 14.5 GB real-world ceiling on the
RTX 5080 Laptop (16 GB hardware, Blackwell sm_120).

This is an observability-first test. It does not mock the pipeline. It runs the
real nodes in sequence, calling torch.cuda.max_memory_allocated() between
phases, and records per-phase high-water marks into a JSON report that
_runtime_log consumers can ingest.

Design rules (from ROADMAP + CLAUDE.md):
  - Sequential execution only. No async CUDA streams.
  - 100% local, offline-first. No cloud calls.
  - Clean logs, meaningful names. No curse words.
  - Safe to run on a machine without CUDA: the test will skip cleanly.
  - Ceiling: 14.5 GB == 14.5 * 1024 * 1024 * 1024 bytes.

This scaffold defines the phase list, the snapshot helper, the ceiling check,
and the JSON report writer. The individual phase runners are stubs tagged
TODO(v1.4-theme-c) so the next commits can wire them into the real nodes.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import unittest
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VRAM_CEILING_BYTES: int = int(14.5 * 1024 * 1024 * 1024)  # 14.5 GB real-world target
VRAM_HARDWARE_BYTES: int = 16 * 1024 * 1024 * 1024        # 16 GB RTX 5080 Laptop

REPORT_DIR = os.path.join(os.path.dirname(__file__), "_reports")
REPORT_PATH = os.path.join(REPORT_DIR, "vram_profile_report.json")


# ---------------------------------------------------------------------------
# Torch / CUDA guard
# ---------------------------------------------------------------------------

def _torch_cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _reset_peak() -> None:
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_bytes() -> int:
    import torch
    return int(torch.cuda.max_memory_allocated())


def _current_bytes() -> int:
    import torch
    return int(torch.cuda.memory_allocated())


# ---------------------------------------------------------------------------
# Phase snapshot record
# ---------------------------------------------------------------------------

@dataclass
class PhaseSnapshot:
    name: str
    peak_bytes: int = 0
    current_bytes_after: int = 0
    wall_seconds: float = 0.0
    notes: str = ""

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / (1024.0 ** 3)


@dataclass
class VramReport:
    device_name: str = ""
    ceiling_bytes: int = VRAM_CEILING_BYTES
    hardware_bytes: int = VRAM_HARDWARE_BYTES
    phases: List[PhaseSnapshot] = field(default_factory=list)
    overall_peak_bytes: int = 0

    def add(self, snap: PhaseSnapshot) -> None:
        self.phases.append(snap)
        if snap.peak_bytes > self.overall_peak_bytes:
            self.overall_peak_bytes = snap.peak_bytes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phases"] = [asdict(p) for p in self.phases]
        d["overall_peak_gb"] = self.overall_peak_bytes / (1024.0 ** 3)
        d["ceiling_gb"] = self.ceiling_bytes / (1024.0 ** 3)
        return d


# ---------------------------------------------------------------------------
# Phase runner registry
# ---------------------------------------------------------------------------

_ctx = {}

def _run_node(node_class, **overrides):
    import inspect
    node = node_class()
    sig = node.INPUT_TYPES()
    kwargs = {}
    for group in ["required", "optional", "hidden"]:
        for k, v in sig.get(group, {}).items():
            # v is typically (TYPE_STRING, {"default": value}) or (["list", "of", "strings"], {"default": value})
            if isinstance(v[0], list):
                kwargs[k] = v[1].get("default", v[0][0]) if len(v) > 1 and isinstance(v[1], dict) else v[0][0]
            else:
                kwargs[k] = v[1].get("default", None) if len(v) > 1 and isinstance(v[1], dict) else None
    kwargs.update(overrides)
    func = getattr(node, node.FUNCTION)
    accepted = inspect.signature(func).parameters
    final_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return func(**final_kwargs)

def _run_gemma4_script_writer() -> None:
    from nodes.story_orchestrator import LLMScriptWriter
    res = _run_node(LLMScriptWriter,
                    episode_title="The Last Frequency",
                    genre_flavor="hard_sci_fi",
                    target_words=350,
                    num_characters=2,
                    model_id="google/gemma-4-E4B-it",
                    temperature=0.7,
                    arc_enhancer=True)
    _ctx["script_text"] = res[0]
    _ctx["script_json"] = res[1]

def _run_gemma4_director() -> None:
    from nodes.story_orchestrator import LLMDirector
    res = _run_node(LLMDirector, script_text=_ctx["script_text"])
    _ctx["production_plan_json"] = res[0]

def _run_kokoro_announcer() -> None:
    from nodes.kokoro_announcer import KokoroAnnouncer
    res = _run_node(KokoroAnnouncer, script_json=_ctx["script_json"])
    _ctx["announcer_audio_clips"] = res[0]

def _run_musicgen_theme() -> None:
    from nodes.musicgen_theme import MusicGenTheme
    res = _run_node(MusicGenTheme, production_plan_json=_ctx["production_plan_json"], episode_seed=42)
    _ctx["opening_theme_audio"] = res[0]
    _ctx["closing_theme_audio"] = res[1]

def _run_batch_bark_generator() -> None:
    from nodes.batch_bark_generator import BatchBarkGenerator
    res = _run_node(BatchBarkGenerator, script_json=_ctx["script_json"], production_plan_json=_ctx["production_plan_json"])
    _ctx["tts_audio_clips"] = res[0]

def _run_scene_sequencer() -> None:
    from nodes.scene_sequencer import SceneSequencer
    res = _run_node(SceneSequencer, 
                    script_json=_ctx["script_json"], 
                    production_plan_json=_ctx["production_plan_json"], 
                    tts_audio_clips=_ctx["tts_audio_clips"],
                    announcer_audio_clips=_ctx.get("announcer_audio_clips"))
    _ctx["scene_audio"] = res[0]

def _run_audio_enhance() -> None:
    from nodes.audio_enhance import AudioEnhance
    res = _run_node(AudioEnhance, audio=_ctx["scene_audio"])
    _ctx["enhanced_audio"] = res[0]

def _run_episode_assembler() -> None:
    from nodes.scene_sequencer import EpisodeAssembler
    res = _run_node(EpisodeAssembler, 
                    scene_audio=_ctx["enhanced_audio"],
                    opening_theme_audio=_ctx.get("opening_theme_audio"),
                    closing_theme_audio=_ctx.get("closing_theme_audio"))
    _ctx["episode_audio"] = res[0]

def _run_video_engine() -> None:
    from nodes.video_engine import SignalLostVideoRenderer
    res = _run_node(SignalLostVideoRenderer, 
                    audio=_ctx["episode_audio"],
                    script_json=_ctx["script_json"],
                    production_plan_json=_ctx["production_plan_json"],
                    episode_title="Test Episode")
    _ctx["video_path"] = res[0]

PHASE_RUNNERS: List[tuple] = [
    ("gemma4_script_writer", _run_gemma4_script_writer),
    ("gemma4_director",      _run_gemma4_director),
    ("kokoro_announcer",     _run_kokoro_announcer),
    ("musicgen_theme",       _run_musicgen_theme),
    ("batch_bark_generator", _run_batch_bark_generator),
    ("scene_sequencer",      _run_scene_sequencer),
    ("audio_enhance",        _run_audio_enhance),
    ("episode_assembler",    _run_episode_assembler),
    ("video_engine",         _run_video_engine),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_profile() -> VramReport:
    """Run every registered phase, snapshot VRAM, return the report."""
    report = VramReport()

    cuda_on = _torch_cuda_available()
    if cuda_on:
        import torch
        report.device_name = torch.cuda.get_device_name(0)

    for name, runner in PHASE_RUNNERS:
        snap = PhaseSnapshot(name=name)
        if cuda_on:
            _reset_peak()
        t0 = time.perf_counter()
        try:
            runner()
        except Exception as exc:  # keep the profile honest even on failure
            snap.notes = f"runner_error: {type(exc).__name__}: {exc}"
        snap.wall_seconds = time.perf_counter() - t0
        if cuda_on:
            snap.peak_bytes = _peak_bytes()
            snap.current_bytes_after = _current_bytes()
        report.add(snap)

    return report


def write_report(report: VramReport, path: str = REPORT_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# unittest entry point
# ---------------------------------------------------------------------------

class VramProfileTest(unittest.TestCase):
    """Assert pipeline peak VRAM stays under the 14.5 GB real-world ceiling."""

    def test_profile_under_ceiling(self) -> None:
        if not _torch_cuda_available():
            self.skipTest("CUDA not available; VRAM profile test requires the RTX 5080 Laptop.")

        report = run_profile()
        path = write_report(report)
        print(f"[vram_profile_test] report written: {path}")
        for p in report.phases:
            print(f"  {p.name:24s} peak={p.peak_gb:6.3f} GB  wall={p.wall_seconds:6.3f}s  {p.notes}")
        print(f"  OVERALL peak={report.overall_peak_bytes / (1024**3):6.3f} GB "
              f"(ceiling={VRAM_CEILING_BYTES / (1024**3):.1f} GB)")

        self.assertLessEqual(
            report.overall_peak_bytes,
            VRAM_CEILING_BYTES,
            msg=(
                f"Peak VRAM {report.overall_peak_bytes / (1024**3):.3f} GB "
                f"exceeds 14.5 GB ceiling on {report.device_name or 'unknown device'}."
            ),
        )


if __name__ == "__main__":
    # Allow `python tests/vram_profile_test.py` from the repo root.
    sys.exit(unittest.main(verbosity=2))
