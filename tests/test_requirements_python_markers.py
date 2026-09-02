"""requirements.txt must install on the interpreters ComfyUI actually ships.

PBUG-20260901-04: kokoro cannot be pip-installed on Python 3.13 (numpy==1.26.4
pin on 0.7.16; ``Requires-Python <3.13`` on every newer kokoro/misaki), and pip
resolves a requirements file all-or-nothing, so one bare ``kokoro>=...`` line
made a clean Python 3.13 install of the pack land NOTHING. The line now carries
a ``python_version < "3.13"`` marker. This test keeps the marker, keeps every
requirement line parseable, and keeps the marker semantics honest.
"""
from __future__ import annotations

import pathlib

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement

REPO = pathlib.Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements.txt"


def _requirement_lines() -> list[str]:
    lines = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _parsed() -> dict[str, Requirement]:
    return {Requirement(line).name.lower(): Requirement(line)
            for line in _requirement_lines()}


def test_every_requirement_line_parses():
    for line in _requirement_lines():
        Requirement(line)  # raises InvalidRequirement on a bad line


def test_kokoro_is_marker_guarded_below_python_313():
    reqs = _parsed()
    assert "kokoro" in reqs, "kokoro is the shipped default announcer voice"
    marker = reqs["kokoro"].marker
    assert marker is not None, (
        "kokoro must carry a python_version marker: on Python 3.13 it is not "
        "installable and its bare presence makes pip install NOTHING "
        "(PBUG-20260901-04)")
    assert marker.evaluate({"python_version": "3.12"}) is True
    assert marker.evaluate({"python_version": "3.13"}) is False


@pytest.mark.parametrize("python_version", ["3.12", "3.13"])
def test_requirements_resolve_to_a_nonempty_set_on_shipped_interpreters(python_version):
    """A marker may exclude a line on an interpreter; it must never exclude
    the core of the pack. transformers, soundfile and feedparser are the
    pipeline's hard floor and must stay unconditional."""
    env = {"python_version": python_version}
    active = [name for name, req in _parsed().items()
              if req.marker is None or req.marker.evaluate(env)]
    for core in ("transformers", "soundfile", "feedparser", "pyloudnorm"):
        assert core in active, f"{core} must install on Python {python_version}"


def test_marker_syntax_is_pep_508():
    Marker('python_version < "3.13"')  # the exact clause used in requirements.txt
