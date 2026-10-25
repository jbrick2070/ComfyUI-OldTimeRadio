"""The 24/7 broadcast feed (nodes/_otr_shared/broadcast_feed.py,
scripts/otr_broadcast.py, the mux hook, the read-only route and the "On air"
tab). PLAN row 4.1, planning/creatoros-concepts/PLAN.md; operator D1 = A.

The rules under test: every publish airs by default; a pull removes only the
link and otr/obs/ stays byte-identical; a link OBS is holding is "pull pending",
never "pulled"; a name is accepted only as an exact obs file name; a feed
problem is reported and never fails a publish; no route changes the feed.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from nodes._otr_shared import broadcast_feed as BF  # noqa: E402
import otr_broadcast  # noqa: E402

A = "alpha_20261024_100000__vstb__stmo__zimg__koko__pubd__cg4e2__sa3_final.mp4"
B = "beta_20261024_110000__arch__lx8g__fk3i__koko__marc__cg4e2__sa3_final.mp4"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    otr = tmp_path / "output" / "otr"
    obs = otr / "obs"
    shared = otr / "episodes" / "_shared"
    obs.mkdir(parents=True)
    fake = types.SimpleNamespace(
        otr_obs_dir=lambda: obs,
        otr_shared_root=lambda: shared,
        otr_state_dir=lambda: shared / "state",
    )
    monkeypatch.setattr(BF, "_otr_paths", fake)
    for name, body in ((A, b"episode a"), (B, b"episode b")):
        (obs / name).write_bytes(body)
    (obs / ".otr_staging").mkdir()                    # the mux's candidate folder
    (obs / "rejected_candidate.mp4").write_bytes(b"x")  # no _final marker
    return types.SimpleNamespace(obs=obs, feed=shared / BF.FEED_DIRNAME, shared=shared)


def _digest(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


def test_sync_airs_every_published_episode_as_a_hardlink(tree):
    report = BF.sync()
    assert sorted(report["linked"]) == [A, B] and not report["errors"]
    assert sorted(p.name for p in tree.feed.iterdir()) == [A, B]
    for name in (A, B):
        assert os.path.samefile(tree.feed / name, tree.obs / name)
    assert BF.sync()["linked"] == [], "a second sync changes nothing"


def test_a_pull_removes_only_the_link_and_obs_stays_byte_identical(tree):
    BF.sync()
    before = _digest(tree.obs)
    assert BF.pull(A) == {"episode": A, "state": "off air"}
    assert not (tree.feed / A).exists()
    assert (tree.feed / B).exists()
    assert _digest(tree.obs) == before
    BF.sync()
    assert not (tree.feed / A).exists(), "a pulled episode stays off air"
    assert BF.load_pulled() == [A]
    assert BF.put_back(A) == {"episode": A, "state": "on air"}
    assert os.path.samefile(tree.feed / A, tree.obs / A)
    assert BF.load_pulled() == []
    assert _digest(tree.obs) == before


def test_a_republish_under_the_same_name_is_relinked(tree):
    BF.sync()
    fresh = tree.obs / ".otr_staging" / "candidate.mp4"
    fresh.write_bytes(b"episode a, published again")
    os.replace(fresh, tree.obs / A)                   # how the mux accepts a candidate
    assert not os.path.samefile(tree.feed / A, tree.obs / A)
    assert BF.sync()["relinked"] == [A]
    assert (tree.feed / A).read_bytes() == b"episode a, published again"


def test_a_hand_deleted_obs_file_leaves_the_feed_on_the_next_sync(tree):
    BF.sync()
    (tree.obs / B).unlink()
    assert BF.sync()["removed"] == [B]
    assert sorted(p.name for p in tree.feed.iterdir()) == [A]


def test_a_link_obs_is_playing_is_pull_pending_never_pulled(tree, monkeypatch):
    BF.sync()
    real_remove = os.remove

    def locked(path):
        if Path(path).name == A:
            raise PermissionError(32, "The process cannot access the file")
        return real_remove(path)

    monkeypatch.setattr(BF.os, "remove", locked)
    result = BF.pull(A)
    assert result["state"] == "pull pending"
    assert "next sync" in result["reason"]
    rows = {r["episode"]: r for r in BF.list_episodes()}
    assert rows[A]["state"] == "pull pending" and rows[A]["on_air"] is False
    assert BF.sync()["pending"] == [A]
    monkeypatch.setattr(BF.os, "remove", real_remove)
    report = BF.sync()
    assert report["removed"] == [A] and report["pending"] == []
    assert {r["episode"]: r["state"] for r in BF.list_episodes()}[A] == "off air"


@pytest.mark.parametrize("bad", [
    "", "../" + A, "..\\" + A, str(Path("C:/") / A), "nope_final.mp4",
    "rejected_candidate.mp4", ".otr_staging", None,
])
def test_only_an_exact_published_name_is_accepted(tree, bad):
    with pytest.raises(BF.BroadcastError):
        BF.pull(bad)
    assert BF.load_pulled() == []


def test_a_feed_on_another_volume_is_reported_once_and_obs_is_untouched(tree, monkeypatch):
    before = _digest(tree.obs)

    def cross_volume(src, dst):
        raise OSError(17, "The system cannot move the file to a different disk drive")

    monkeypatch.setattr(BF.os, "link", cross_volume)
    report = BF.sync()
    assert len(report["errors"]) == 1
    assert "one volume" in report["errors"][0] and "OTR_OBS_DIR" in report["errors"][0]
    assert _digest(tree.obs) == before


def test_a_publish_outside_the_canonical_obs_folder_leaves_the_feed_alone(tree, tmp_path):
    other = tmp_path / "someone_elses_obs"
    other.mkdir()
    report = BF.sync(published_into=str(other))
    assert report["errors"] and "canonical obs folder" in report["errors"][0]
    assert not tree.feed.exists()
    assert BF.sync(published_into=str(tree.obs))["linked"]


def test_the_listing_is_read_only_and_names_files_never_paths(tree):
    BF.sync()
    BF.pull(B)
    snapshot = sorted(p.name for p in tree.feed.iterdir())
    rows = BF.list_episodes()
    assert sorted(p.name for p in tree.feed.iterdir()) == snapshot
    assert {r["episode"] for r in rows} == {A, B}
    by ={r["episode"]: r for r in rows}
    assert by[A]["state"] == "on air" and by[A]["title"] == "alpha_20261024_100000"
    assert by[B]["state"] == "off air"
    text = repr(rows)
    assert str(tree.obs) not in text and str(tree.feed) not in text


def test_an_unreadable_pulled_list_is_reported_and_read_as_empty(tree):
    (tree.shared / "state").mkdir(parents=True)
    (tree.shared / "state" / BF.PULLED_FILENAME).write_text("{not json", encoding="utf-8")
    assert BF.load_pulled() == []


def test_the_cli_lists_pulls_and_refuses(tree, capsys):
    assert otr_broadcast.main(["sync"]) == 0
    assert "2 linked" in capsys.readouterr().out
    assert otr_broadcast.main(["pull", A]) == 0
    assert "off air" in capsys.readouterr().out
    assert otr_broadcast.main([]) == 0
    out = capsys.readouterr().out
    assert "2 published, 1 on air" in out and "off air" in out
    assert otr_broadcast.main(["pull", "../etc_final.mp4"]) == 2
    assert "refused" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        otr_broadcast.main(["pull"])


# --- the hook, the route, the tab --------------------------------------------

def test_the_mux_syncs_the_feed_after_the_obs_copy_and_only_reports():
    source = (_REPO / "nodes" / "otr_master_audio_mux.py").read_text(encoding="utf-8")
    publish = source.index("obs_copy = self._publish_to_obs(final)")
    hook = source.index("report.append(_air_on_broadcast(os.path.dirname(obs_copy)))")
    stamp = source.index("report.append(self._stamp_terminal_paths(")
    assert publish < hook < stamp


def test_the_mux_hook_never_raises(monkeypatch):
    from nodes import otr_master_audio_mux as mux

    def broken(published_into=None):
        raise RuntimeError("feed broke")

    monkeypatch.setattr(BF, "sync", broken)
    assert mux._air_on_broadcast("x") == "broadcast feed NOT synced (RuntimeError)"
    monkeypatch.setattr(BF, "sync", lambda published_into=None: {
        "linked": ["a"], "relinked": [], "removed": [], "pending": [], "errors": []})
    assert mux._air_on_broadcast("x") == "broadcast feed synced (1 linked, 0 relinked, 0 removed)"


def test_no_route_changes_the_feed():
    init = (_REPO / "__init__.py").read_text(encoding="utf-8")
    assert 'routes.get("/otr/episodes")' in init
    assert not re.search(r"routes\.(post|put|delete|patch)\(\"/otr/(episodes|broadcast)", init)
    route = init[init.index('routes.get("/otr/episodes")'):]
    route = route[:route.index("print(")]
    assert "list_episodes()" in route
    for verb in ("sync(", "pull(", "put_back(", ".act("):
        assert verb not in route, verb
    boot = init[init.index("broadcast_feed import sync"):]
    assert "_otr_feed_sync()" in boot[:400]


def _node():
    for candidate in ("node", r"C:\Program Files\nodejs\node.exe"):
        found = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if found:
            return found
    return None


def test_the_tab_ships_and_its_javascript_suite_passes():
    assert (_REPO / "js" / "episodes_tab.js").is_file()
    glue = (_REPO / "js" / "episodes_tab.js").read_text(encoding="utf-8")
    assert "registerSidebarTab" in glue and "/otr/episodes" in glue
    assert "method" not in glue, "the tab only reads"
    node = _node()
    if node is None:
        pytest.skip("node is not installed; the JS suite cannot run here")
    proc = subprocess.run([node, "--test", str(_REPO / "tests" / "js" / "episodes_tab.test.mjs")],
                          capture_output=True, text=True, cwd=str(_REPO), timeout=180)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]


def test_an_obs_file_that_is_a_symlink_is_linked_to_its_real_file(tree, tmp_path):
    """BUG-12.177: a hard link of a symlink is a link to the symlink."""
    real = tmp_path / "elsewhere_final_source.mp4"
    real.write_bytes(b"the real episode")
    name = "gamma_20261024_120000__vstb__stmo__zimg__koko__pubd__cg4e2__sa3_final.mp4"
    try:
        os.symlink(real, tree.obs / name)
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot make symlinks")
    BF.sync()
    assert not os.path.islink(tree.feed / name)
    assert (tree.feed / name).read_bytes() == b"the real episode"
