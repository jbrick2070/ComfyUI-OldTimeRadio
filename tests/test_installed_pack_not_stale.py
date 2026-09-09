"""The INSTALLED pack must not drift from this repo.

WHY THIS TEST EXISTS, and it cost real time on 2026-09-09 -- twice in one day.

On any machine where ComfyUI loads OTR from a SEPARATE COPY rather than from
this checkout (the Mac lays it out that way: the repo lives in Documents and
ComfyUI loads `custom_nodes/comfyui-old-time-radio`), the two trees drift the
moment you edit one. Two distinct failures follow, and both are convincing:

  1. **Tests lie.** A full-suite run can resolve `nodes.*` to the INSTALLED copy
     while a single-file run resolves it to the repo. Four `inspect.getsource`
     assertions reported that code plainly present in the repo was missing. It
     was missing -- from the stale copy they had actually imported.
  2. **Renders prove the wrong thing.** A live leg exercises whatever the server
     loaded, so a green suite against the repo followed by a render against a
     stale copy produces a receipt for code that was never under test.

The guide already says to run the diff after every edit you intend to render
with. That is a discipline, and a discipline is exactly what gets skipped at
the end of a long session -- so this makes it a gate.

WHERE IT DOES NOT APPLY. On the Windows boxes the repo IS the installed pack
(one directory, per CLAUDE.md section 0), so there is nothing to compare and
this skips. It also skips when no installed copy can be found at all, because
"ComfyUI is not installed here" is not a defect.
"""
from __future__ import annotations

import filecmp
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Where an installed copy might live. `OTR_INSTALLED_PACK` wins, so a machine
#: with an unusual layout can point this at the truth rather than be skipped.
_CANDIDATES = (
    "~/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio",
    "~/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio",
    "~/ComfyUI/custom_nodes/comfyui-old-time-radio",
    "~/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio",
)


def _installed_pack():
    """The installed copy, or ``None`` when there isn't a separate one."""
    override = os.environ.get("OTR_INSTALLED_PACK")
    roots = [override] if override else _CANDIDATES
    for raw in roots:
        path = Path(os.path.expanduser(raw))
        if not (path / "nodes").is_dir():
            continue
        if path.resolve() == REPO.resolve():
            return None          # the repo IS the installed pack (Windows)
        return path
    return None


def _drifted(installed: Path) -> list:
    """Source files that differ, ignoring bytecode and local droppings."""
    out = []
    for src in sorted((REPO / "nodes").rglob("*.py")):
        if "__pycache__" in src.parts:
            continue
        mirror = installed / "nodes" / src.relative_to(REPO / "nodes")
        rel = str(src.relative_to(REPO))
        if not mirror.exists():
            out.append("%s  (missing from the installed pack)" % rel)
        elif not filecmp.cmp(str(src), str(mirror), shallow=False):
            out.append("%s  (differs)" % rel)
    return out


def test_the_installed_pack_matches_this_repo():
    installed = _installed_pack()
    if installed is None:
        pytest.skip("no separate installed pack on this machine")
    drifted = _drifted(installed)
    assert not drifted, (
        "The pack ComfyUI actually loads is STALE against this repo, so a "
        "render or a full-suite run here exercises DIFFERENT CODE from the one "
        "you just edited.\n\n"
        "  installed: %s\n  repo:      %s\n\n"
        "Drifted (%d):\n  %s\n\n"
        "Fix by copying the files across and dropping the bytecode -- "
        "restarting the server does NOT sync it, only copying does:\n"
        "    cp <repo>/<file> %s/<file>\n"
        "    find %s/nodes -name __pycache__ -type d -exec rm -rf {} +\n"
        "Then restart ComfyUI so Python re-imports them."
        % (installed, REPO, len(drifted), "\n  ".join(drifted[:40]),
           installed, installed))


def test_the_installed_pack_carries_no_extra_engines():
    """A file the repo deleted must not live on in the installed copy.

    The drift check above walks the REPO, so a module deleted here but left
    behind there is invisible to it -- and a stale engine module still
    registers, which puts a retired lane back in the dropdown on the one
    machine that matters: the one actually rendering.
    """
    installed = _installed_pack()
    if installed is None:
        pytest.skip("no separate installed pack on this machine")
    orphans = []
    for mirror in sorted((installed / "nodes").rglob("*.py")):
        if "__pycache__" in mirror.parts:
            continue
        src = REPO / "nodes" / mirror.relative_to(installed / "nodes")
        if not src.exists():
            orphans.append(str(mirror.relative_to(installed)))
    assert not orphans, (
        "these files exist in the installed pack but not in this repo, so "
        "ComfyUI is loading code that no longer exists here: %s" % orphans)
