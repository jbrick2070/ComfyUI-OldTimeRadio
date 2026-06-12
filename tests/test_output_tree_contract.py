"""OH-1/OH-2 output-tree contract tests (operator law, 2026-06-11).

The contract (docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md):
``<output>/otr/`` contains EXACTLY two top-level entries -- ``episodes/``
(every episode asset of record, plus the reserved ``episodes/_shared``
system tier with cache/tmp/state) and ``obs/`` (final deliverables only,
flat). These tests pin:

* every OUTPUT helper in ``_otr_paths`` resolves under ``otr/{episodes,obs}``;
* the production write helpers RAISE LOUD on an empty/invalid episode_id
  (the ``_legacy_*`` fallback era is over);
* ``_shared`` is never an episode (helpers reject it; the ledger walker
  skips ``_``-prefixed entries);
* the portrait/still content-addressed pool lives under
  ``episodes/_shared/cache``;
* CI ban: no module outside ``_otr_paths.py`` composes
  ``comfy_output_dir()/"otr"`` (one path authority), and no nodes/ module
  hand-joins ``"otr"`` with a non-contract top-level component.

Pure CPU; no ComfyUI imports. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from nodes import _otr_paths as P
from nodes._otr_shared import portrait_ledger as PL

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- #
# Helper-contract: every OUTPUT helper lands under otr/{episodes,obs}
# --------------------------------------------------------------------------- #
_EPISODE_HELPERS = [
    P.otr_audio_dir, P.otr_stills_dir, P.otr_portraits_dir,
    P.otr_videos_dir, P.otr_composited_dir, P.otr_upscaled_dir,
]
_NOARG_HELPERS = [
    P.otr_episodes_root, P.otr_obs_dir, P.otr_state_dir,
    P.otr_shared_root, P.otr_shared_cache_dir, P.otr_shared_tmp_dir,
]


@pytest.mark.parametrize("helper", _EPISODE_HELPERS,
                         ids=lambda h: h.__name__)
def test_episode_helpers_inside_contract(helper, out_root):
    p = helper("ep_test_001")
    rel = p.resolve().relative_to((out_root / "otr").resolve())
    assert rel.parts[0] == "episodes"
    assert rel.parts[1] == "ep_test_001"


@pytest.mark.parametrize("helper", _NOARG_HELPERS,
                         ids=lambda h: h.__name__)
def test_noarg_helpers_inside_contract(helper, out_root):
    p = helper()
    rel = p.resolve().relative_to((out_root / "otr").resolve())
    assert rel.parts[0] in ("episodes", "obs")


@pytest.mark.parametrize("helper", _EPISODE_HELPERS,
                         ids=lambda h: h.__name__)
def test_empty_episode_id_raises_loud(helper, out_root):
    """OH-1: the _legacy_* fallbacks are KILLED -- empty id raises."""
    with pytest.raises(P.OtrPathContractError):
        helper("")


@pytest.mark.parametrize("bad", ["..", "a/b", "a\\b", "_shared", "_x"])
def test_invalid_episode_id_raises(bad, out_root):
    with pytest.raises(P.OtrPathContractError):
        P.otr_stills_dir(bad)


def test_shared_tier_layout(out_root):
    assert P.otr_shared_root() == out_root / "otr" / "episodes" / "_shared"
    assert P.otr_shared_cache_dir().name == "cache"
    assert P.otr_shared_tmp_dir().name == "tmp"
    assert P.otr_state_dir().name == "state"
    assert P.otr_state_dir().parent == P.otr_shared_root()


def test_is_reserved_episode_entry():
    assert P.is_reserved_episode_entry("_shared")
    assert P.is_reserved_episode_entry("_anything")
    assert not P.is_reserved_episode_entry("signal_lost_x_20260611")


# --------------------------------------------------------------------------- #
# Portrait/still content-addressed pool -> episodes/_shared/cache
# --------------------------------------------------------------------------- #
def test_portrait_pool_under_shared_cache(tmp_path):
    root = PL.stills_root(str(tmp_path))
    assert root == tmp_path / "otr" / "episodes" / "_shared" / "cache"
    p = PL.portrait_path_for_hash("ab" * 32, str(tmp_path))
    assert p.parent == root and p.suffix == ".png"


# --------------------------------------------------------------------------- #
# Walker underscore-skip: _shared is never an episode
# --------------------------------------------------------------------------- #
def test_ledger_walker_skips_reserved_entries(tmp_path):
    from nodes._otr_ledger import find_most_recent_ledger
    real = tmp_path / "ep_real" / "audio"
    real.mkdir(parents=True)
    (real / "ep_real_ledger.json").write_text("{}", encoding="utf-8")
    shared = tmp_path / "_shared" / "audio"
    shared.mkdir(parents=True)
    newer = shared / "zz_ledger.json"
    newer.write_text("{}", encoding="utf-8")
    # Make the reserved entry the NEWEST -- it must still be skipped.
    import os
    import time
    t = time.time() + 60
    os.utime(newer, (t, t))
    picked = find_most_recent_ledger([tmp_path])
    assert picked is not None
    assert picked.parent.parent.name == "ep_real"


# --------------------------------------------------------------------------- #
# Zero top-level files: helpers never produce a FILE directly under otr/
# --------------------------------------------------------------------------- #
def test_no_helper_yields_otr_toplevel_file(out_root):
    for helper in _NOARG_HELPERS:
        p = helper()
        rel = p.resolve().relative_to((out_root / "otr").resolve())
        assert len(rel.parts) >= 1 and rel.parts[0] in ("episodes", "obs")
    for helper in _EPISODE_HELPERS:
        rel = (helper("ep_x").resolve()
               .relative_to((out_root / "otr").resolve()))
        assert len(rel.parts) >= 2


# --------------------------------------------------------------------------- #
# CI ban: ONE path authority (the OH-2 narrow test)
# --------------------------------------------------------------------------- #
_BAN_COMFY_COMPOSE = re.compile(
    r"comfy_output_dir\(\)\s*/\s*[\"']otr[\"']")
#: os.path.join(..., "otr", "<top>") where <top> is NOT a contract entry.
_JOIN_OTR = re.compile(
    r"os\.path\.join\([^)\n]*[\"']otr[\"']\s*,\s*[\"'](?P<top>[^\"']+)[\"']")


def _py_sources():
    for sub in ("nodes", "scripts"):
        for p in sorted((REPO_ROOT / sub).rglob("*.py")):
            yield p
    yield REPO_ROOT / "__init__.py"


def test_ci_ban_otr_path_composition_outside_authority():
    """No module outside nodes/_otr_paths.py may compose
    comfy_output_dir()/"otr"; no module may hand-join "otr" with a
    non-contract top-level component (episodes/obs are the only ones)."""
    offenders = []
    for p in _py_sources():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel == "nodes/_otr_paths.py":
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if _BAN_COMFY_COMPOSE.search(line):
                offenders.append("%s:%d: %s" % (rel, i, line.strip()))
            m = _JOIN_OTR.search(line)
            if m and m.group("top") not in ("episodes", "obs"):
                offenders.append("%s:%d: %s" % (rel, i, line.strip()))
    assert not offenders, (
        "otr/ path composition outside the _otr_paths authority (or with a "
        "non-contract top level):\n" + "\n".join(offenders))
