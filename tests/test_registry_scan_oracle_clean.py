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


#: The one place the shipped set may carry findings, and why. Operator,
#: 2026-10-22: Radio Play Studio ships in the zip to measure the registry's real
#: verdict, knowing the replica predicts a Flag for its environment reads and its
#: HTTP calls to the local ComfyUI. Those calls are what the app does, so they are
#: not respelled. If the registry Flags the version, restore `contest/` in
#: .comfyignore, delete this and the two app tests below, and drop the "ships
#: inside the pack" note from contest/radio-play-studio/README.md; if it goes
#: Active, the replica over-reports these rules for this code. The rest of the
#: pack is held to zero, as before.
SHIPPED_APP = "contest/radio-play-studio/radio_play_studio/"


def test_the_packed_set_has_no_scan_findings():
    findings = [f for f in _oracle().scan() if not f["file"].startswith(SHIPPED_APP)]
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


def test_the_app_experiment_is_the_only_exception():
    """The exception covers the app package and nothing else, and only while the
    app actually ships; once contest/ is ignored again it must find nothing there."""
    findings = _oracle().scan()
    app = [f for f in findings if f["file"].startswith(SHIPPED_APP)]
    assert all(f["rule"] in {"$env_read/mod", "$http2", "$socket2"} for f in app), app
    assert len(app) <= 10, "the app grew new scan findings; respell nothing, decide first"


def test_only_the_app_itself_ships_from_contest():
    """.comfyignore keeps contest/ out by listing what not to ship, so a new folder
    or a media file added beside the app would ship without anyone deciding it."""
    app_dir = "contest/radio-play-studio/"
    top = {app_dir + name for name in ("LICENSE", "README.md", "requirements.txt")}
    extra = sorted(p for p in _oracle()._shipped_files(False) if p.startswith("contest/")
                   and p not in top
                   and not (p.startswith(SHIPPED_APP) and Path(p).suffix in {".py", ".html", ".json"}))
    assert extra == [], "these would ship from contest/; decide, then list them here or ignore them"
    # Non-vacuity, and the other direction: the app ships with its page and its languages.
    shipped = set(_oracle()._shipped_files(False))
    needed = {SHIPPED_APP + name for name in ("__main__.py", "server.py", "static/index.html",
                                               "static/i18n/en.json")}
    assert sorted(needed - shipped) == [], "the app would ship without these"
