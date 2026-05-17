"""Sprint E E12 / R-12: VideoComposite clips_dir tooltip drift guard.

Closes M5. Pre-E12 the clips_dir widget tooltip said "Directory of
per-line HuMo clips" but the actual wire (L92 in canonical workflow
JSON) sources from LTX.clips_dir, NOT HuMo. The HuMo per-line clip
lookup happens via ledger.clips[] mp4_path entries; the value at
clips_dir input is unused at execute() time and serves only as a DAG
sequencing edge.

Post-E12 the tooltip names the actual source + clarifies HuMo lookup
goes through the ledger, so a future operator reading the workflow
surface sees the correct contract.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "nodes" / "video_composite.py"


def _read_src() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_tooltip_names_ltx_source():
    src = _read_src()
    # Tooltip must name LTX as the wire source, not HuMo.
    assert "LTX clips directory" in src


def test_tooltip_explains_humo_resolution_path():
    src = _read_src()
    assert "ledger.clips[]" in src
    # The tooltip must be explicit that HuMo clips do NOT come from
    # this directory.
    assert "NOT from this directory" in src


def test_tooltip_calls_out_dag_seq_only():
    src = _read_src()
    assert "DAG sequencing only" in src
    assert "value unused at execute()" in src


def test_legacy_tooltip_text_retired():
    """The old "Directory of per-line HuMo clips" framing must be
    gone so a soak diagnostic grep for that exact string returns
    zero hits."""
    src = _read_src()
    assert "Directory of per-line HuMo clips" not in src
