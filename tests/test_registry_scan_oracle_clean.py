"""The shipped set must read clean to our replica of the registry's scan.

The Comfy Registry flags a whole version on ONE lexical finding anywhere in the
packed zip, and a flagged version never becomes Active. The replica
(`scripts/otr_registry_scan_oracle.py`) existed from 2.3.2 on, but only as a
command someone had to remember to run, and nothing in the suite called it.

Measured 2026-09-25: 2.3.4 was published with three internal plan documents
added that day and no `.comfyignore` line for any of them. One,
`apple/HF_HOME_WINDOWS_PIN.md`, quotes an environment read as prose, and the
scanner does not strip prose from data files, so the replica predicted a Flag
for the version already spent. The suite was otherwise clean, so a green
suite said nothing about the scan. This test is what makes it say something.

A clean result here is necessary, not sufficient: the replica models a private
rule and errs toward over-reporting. A finding is a line to fix -- respell it,
or keep an internal document out of the shipped set with a `.comfyignore` line.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _oracle():
    spec = importlib.util.spec_from_file_location(
        "otr_registry_scan_oracle", REPO / "scripts" / "otr_registry_scan_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_packed_set_has_no_scan_findings():
    findings = _oracle().scan()
    assert findings == [], (
        "the registry scan replica finds %d line(s) that would Flag the next "
        "version:\n%s\nRespell the line, or add an internal document to "
        ".comfyignore." % (
            len(findings),
            "\n".join("  %s:%s  %s  %s" % (f["file"], f["line"], f["rule"], f["text"])
                      for f in findings)))


def test_the_replica_still_sees_the_packed_set():
    """Non-vacuity: a scan of nothing also reports no findings."""
    shipped = _oracle()._shipped_files(False)
    assert len(shipped) > 500, (
        "the replica sees only %d shipped files; it would pass by scanning "
        "almost nothing" % len(shipped))
    assert "nodes/otr_master_audio_mux.py" in shipped
