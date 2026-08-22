"""No launcher may pin the cast seed outside an explicit C7 branch.

WHY THIS FILE EXISTS. On 2026-08-22 a motion-module bake-off ran four legs and
every single one cast GULLIVER REEVES. The operator caught it by WATCHING THE
EPISODES -- "a name that kept popping up in every episode, I hope that was
random" -- not by reading a log, even though the writer had logged
``cast RNG seed=42 (OTR_CAST_SEED override)`` on every leg.

This is BUG-LOCAL-269 arriving through a different door. The original fix
stopped the `seed` widget pinning the cast; the launcher then grew a C7 branch
that pins it deliberately for byte-identity regression runs. A stale
``OTR_CAST_SEED`` inherited from some earlier shell reproduces the same defect
with none of the code being wrong.

Two guards, and this file is the second:

1. ``_resolve_cast_rng_seed`` warns at WARNING level when the variable is set
   and ``OTR_C7`` is not -- the combination that is always a leak.
2. THIS FILE: no shipped launcher may assign the seed variables except inside a
   branch guarded on ``OTR_C7``. A future script that pins them unconditionally
   fails here, at authoring time, instead of silently casting one actor forever.

Operator directive, 2026-08-22: *"be sure no future launchers get that 42 bug."*
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The seeds that pin an episode's identity. Pinning any of them turns every
#: episode from a server into the same episode's cast/style/source.
PINNING_VARS = ("OTR_CAST_SEED", "OTR_STYLE_SEED", "OTR_SCIFI_NEWS_PRO_SEED")

#: Scripts a human might launch a server with.
LAUNCHER_GLOBS = ("scripts/*.cmd", "scripts/*.ps1", "scripts/*.sh",
                  "scripts/*.bat")

#: `set VAR=value` (batch) or `$env:VAR = 'value'` (PowerShell) or `export`.
_ASSIGNMENTS = tuple(
    re.compile(pattern % var, re.IGNORECASE)
    for var in PINNING_VARS
    for pattern in (
        r"^\s*set\s+%s\s*=\s*\S",          # batch, with a VALUE
        r"^\s*\$env:%s\s*=\s*['\"]?\S",     # PowerShell
        r"^\s*export\s+%s=\s*\S",           # sh
    )
)


def _launchers():
    seen = []
    for glob in LAUNCHER_GLOBS:
        for path in sorted(ROOT.glob(glob)):
            # Throwaway probes are not shipped launchers.
            if path.name.startswith("_tmp_"):
                continue
            seen.append(path)
    return seen


def _guards_on_c7(lines, index):
    """Is line `index` inside a block guarded on OTR_C7?

    Deliberately generous -- it looks backwards for an OTR_C7 test and forwards
    for the close of that block. A launcher that pins seeds anywhere NEAR a C7
    check is doing the sanctioned thing; one that pins them with no C7 in sight
    is the defect.
    """
    window = lines[max(0, index - 12):index]
    return any("OTR_C7" in line for line in window)


def test_there_are_launchers_to_check():
    """A vacuous sweep is the failure mode this test class is prone to."""
    assert len(_launchers()) >= 5, [p.name for p in _launchers()]


@pytest.mark.parametrize("path", _launchers(), ids=lambda p: p.name)
def test_no_launcher_pins_an_episode_seed_outside_a_c7_branch(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    offenders = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Comments in every dialect this repo uses.
        if stripped.startswith(("rem ", "REM ", "#", "::")):
            continue
        for pattern in _ASSIGNMENTS:
            if pattern.match(line) and not _guards_on_c7(lines, i):
                offenders.append("%s:%d  %s" % (path.name, i + 1, stripped))
    assert not offenders, (
        "these launcher lines pin an episode seed with no OTR_C7 guard above "
        "them, which makes every episode from that server share one cast "
        "(BUG-LOCAL-269 -- it cast GULLIVER REEVES four legs running on "
        "2026-08-22):\n  %s" % "\n  ".join(offenders))


def test_the_soak_launcher_still_clears_the_seeds_in_its_production_branch():
    """The else branch is the thing that makes production random. If it ever
    disappears, an inherited value survives into the server."""
    launcher = ROOT / "scripts" / "_otr_soak_server_launch.cmd"
    text = launcher.read_text(encoding="utf-8", errors="replace")
    for var in PINNING_VARS:
        assert re.search(r"^\s*set\s+%s=\s*$" % var, text, re.M | re.I), (
            "%s no longer clears %s in its production branch, so a leaked "
            "value would survive into the server" % (launcher.name, var))


def test_the_production_branch_announces_itself():
    """Silence is what let the 2026-08-22 leak ride an entire bake-off: only
    the C7 branch echoed, so a production leg's log said nothing either way."""
    launcher = ROOT / "scripts" / "_otr_soak_server_launch.cmd"
    text = launcher.read_text(encoding="utf-8", errors="replace")
    assert "fresh OS entropy per episode" in text, (
        "the production branch no longer states its seed mode; a leg log must "
        "say which mode it ran in whether or not anything was pinned")


def test_the_resolver_warns_when_the_seed_is_pinned_without_c7(monkeypatch, caplog):
    """The root guard. This fires wherever the variable came from -- a
    launcher, a parent shell, or a stale terminal -- which is the point."""
    import logging

    from nodes import OTR_LedgerScriptWriter as writer

    monkeypatch.setenv("OTR_CAST_SEED", "42")
    monkeypatch.delenv("OTR_C7", raising=False)
    with caplog.at_level(logging.WARNING, logger="OTR"):
        seed, source = writer._resolve_cast_rng_seed()
    assert seed == 42
    assert source == "OTR_CAST_SEED override"
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "OTR_C7 IS NOT SET" in blob, blob
    assert "GULLIVER REEVES" in blob, (
        "the warning must name the cast it will produce -- that is the thing "
        "the operator actually recognises")


def test_a_real_c7_run_is_not_nagged(monkeypatch, caplog):
    """A deliberate byte-identity run sets both, and must stay quiet."""
    import logging

    from nodes import OTR_LedgerScriptWriter as writer

    monkeypatch.setenv("OTR_CAST_SEED", "42")
    monkeypatch.setenv("OTR_C7", "1")
    with caplog.at_level(logging.WARNING, logger="OTR"):
        seed, _source = writer._resolve_cast_rng_seed()
    assert seed == 42
    assert not [r for r in caplog.records if "OTR_C7 IS NOT SET" in r.getMessage()]


def test_production_still_draws_os_entropy(monkeypatch):
    from nodes import OTR_LedgerScriptWriter as writer

    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    monkeypatch.delenv("OTR_C7", raising=False)
    first, source = writer._resolve_cast_rng_seed()
    second, _ = writer._resolve_cast_rng_seed()
    assert source == "OS entropy"
    assert first != second, "two draws returned the same seed; entropy is dead"
