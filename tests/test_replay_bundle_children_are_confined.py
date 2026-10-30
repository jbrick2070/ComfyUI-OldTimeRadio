"""A replay bundle's children are proven inside it, and the freezer copies
nothing a link points at.

Three claims from the 2026-10-30 registry-sinks review (Codex Astra, grounded
by executed probes): `manifest.json` and `cards.json` were opened and hashed
after only their PARENT was confined, so a link planted at either name read
whatever it pointed at; the manifest's `ledger` passed a case-folded
membership check and was then opened by its raw spelling; and the freezer's
walk followed links and junctions, copying outside content into a bundle the
output tree serves. Link tests skip where the box cannot create links.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from nodes import production_ledger as PL
from nodes import _otr_producer_direction as PD
from nodes import _otr_replay_freezer as FZ


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _link_file(target: Path, link: Path) -> bool:
    try:
        os.symlink(str(target), str(link))
        return True
    except (OSError, NotImplementedError):
        return False


def _link_dir(target: Path, link: Path) -> bool:
    """A directory link. On Windows a JUNCTION first: it is what the walks
    follow (`rglob` descends into a junction and not into a symlinked
    directory on Python 3.12), so it is the shape that escaped."""
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return True
        except OSError:
            pass
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


@pytest.fixture
def outside():
    """A folder in no read root: pytest's tmp tree IS a declared root, so this
    sits beside it in the system temp directory."""
    d = Path(tempfile.mkdtemp(prefix="otr_outside_"))
    try:
        yield d
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def _bundle(root: Path, *, ledger_key="audio/ep_ledger.json", cards=False) -> Path:
    b = root / "bundle"
    (b / "audio").mkdir(parents=True)
    ledger = {"lines": [{"line_id": "b001", "text": "hello"}],
              "audio": {"master_audio_sha256": _sha(b"RIFFwav")}}
    lp = b / "audio" / "ep_ledger.json"
    lp.write_text(json.dumps(ledger), encoding="utf-8")
    master = b / "audio" / "ep_master.wav"
    master.write_bytes(b"RIFFwav")
    files = [{"path": "audio/ep_ledger.json", "bytes": lp.stat().st_size, "sha256": _sha(lp.read_bytes())},
             {"path": "audio/ep_master.wav", "bytes": master.stat().st_size, "sha256": _sha(master.read_bytes())}]
    if cards:
        cp = b / PD.CARDS_NAME
        cp.write_text(json.dumps({"version": PD.CARDS_VERSION}), encoding="utf-8")
        files.append({"path": PD.CARDS_NAME, "bytes": cp.stat().st_size, "sha256": _sha(cp.read_bytes())})
    manifest = {"schema_version": PL.REPLAY_MANIFEST_SCHEMA, "files": files,
                "ledger": ledger_key, "master_audio": "audio/ep_master.wav",
                "source_episode_id": "ep"}
    (b / PL.REPLAY_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return b


def test_membership_is_exact_and_the_manifest_carries_the_verified_spelling(tmp_path):
    good = PL.load_replay_manifest(str(_bundle(tmp_path / "a")))
    assert good["ledger"] == "audio/ep_ledger.json"
    with pytest.raises(PL.ReplayBundleError, match="never hashed"):
        PL.load_replay_manifest(str(_bundle(tmp_path / "b", ledger_key="AUDIO/EP_LEDGER.JSON")))
    # a spelling that normalizes to the listed one is admitted AS the listed one
    odd = PL.load_replay_manifest(str(_bundle(tmp_path / "c", ledger_key="audio/./ep_ledger.json")))
    assert odd["ledger"] == "audio/ep_ledger.json"


def test_a_linked_manifest_is_refused_before_it_is_parsed(tmp_path, outside):
    b = _bundle(tmp_path / "a")
    real = b / PL.REPLAY_MANIFEST_NAME
    away = outside / "manifest.json"
    away.write_bytes(real.read_bytes())
    real.unlink()
    if not _link_file(away, real):
        pytest.skip("this box cannot create a file symlink")
    with pytest.raises(ValueError):                      # OtrPathContractError
        PL.load_replay_manifest(str(b))
    with pytest.raises(ValueError):
        PD.manifest_digest(str(b))


def test_a_linked_cards_file_is_refused_and_an_unlisted_one_too(tmp_path, outside):
    b = _bundle(tmp_path / "a", cards=True)
    manifest = PL.load_replay_manifest(str(b))
    assert PD._read_cards(Path(b), manifest)["version"] == PD.CARDS_VERSION
    unlisted = {"files": [r for r in manifest["files"] if r["path"] != PD.CARDS_NAME]}
    with pytest.raises(PD.DirectionError, match="never hashed"):
        PD._read_cards(Path(b), unlisted)
    real = b / PD.CARDS_NAME
    away = outside / "cards.json"
    away.write_bytes(real.read_bytes())
    real.unlink()
    if not _link_file(away, real):
        pytest.skip("this box cannot create a file symlink")
    with pytest.raises(PD.DirectionError):
        PD._read_cards(Path(b))


def _episode(root: Path, ep_id="ep_frozen_20261030_000000") -> Path:
    ep = root / ep_id
    (ep / "audio").mkdir(parents=True)
    (ep / "stills").mkdir()
    (ep / "stills" / "scene_b001.png").write_bytes(b"\x89PNG still")
    master = ep / "audio" / (ep_id + "_master.wav")
    master.write_bytes(b"RIFFmaster")
    ledger = {"episode_id": ep_id,
              "lines": [{"line_id": "b001", "text": "hello"}],
              "audio": {"master_audio_sha256": _sha(master.read_bytes())},
              "video": {"shots": [{"shot_id": "s1", "engine_id": "x"}]}}
    (ep / "audio" / (ep_id + "_ledger.json")).write_text(json.dumps(ledger), encoding="utf-8")
    return ep


def test_the_freezer_refuses_a_linked_still_and_leaves_no_bundle(tmp_path, outside):
    ep = _episode(tmp_path / "episodes")
    secret = outside / "secret.png"
    secret.write_bytes(b"\x89PNG TOP-SECRET-OUTSIDE")
    if not _link_file(secret, ep / "stills" / "leak.png"):
        pytest.skip("this box cannot create a file symlink")
    out = tmp_path / "bundles"
    with pytest.raises(FZ.ReplayBundleError):
        FZ.freeze(ep, out)
    assert not out.exists() or not any(out.iterdir())


def test_the_freezer_refuses_a_linked_folder_and_leaves_no_bundle(tmp_path, outside):
    ep = _episode(tmp_path / "episodes")
    (outside / "jfile.png").write_bytes(b"\x89PNG outside")
    if not _link_dir(outside, ep / "stills" / "jn"):
        pytest.skip("this box cannot create a directory link")
    out = tmp_path / "bundles"
    # The walk follows a junction, so `_add` meets the outside file and the
    # freeze is refused; a walk that does not descend (a symlinked directory
    # on 3.12) builds a bundle that simply carries nothing from the link.
    # Either way nothing outside the episode is copied.
    try:
        bundle = FZ.freeze(ep, out)
    except FZ.ReplayBundleError:
        assert not out.exists() or not any(out.iterdir())
        return
    names = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    assert not any(n.startswith("stills/jn") for n in names), names


def test_a_clean_episode_still_freezes(tmp_path):
    ep = _episode(tmp_path / "episodes")
    bundle = FZ.freeze(ep, tmp_path / "bundles")
    manifest = PL.load_replay_manifest(str(bundle))
    assert {r["path"] for r in manifest["files"]} >= {"stills/scene_b001.png"}


def test_a_dot_prefixed_cards_row_is_admitted_once_and_read_once(tmp_path):
    """Codex review of 502aeeda6: admission normalized `./cards.json` to
    `cards.json` and `_read_cards` then compared the raw row. The verified
    row now carries its canonical spelling."""
    b = _bundle(tmp_path / "a", cards=True)
    mpath = b / PL.REPLAY_MANIFEST_NAME
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        if row["path"] == PD.CARDS_NAME:
            row["path"] = "./" + PD.CARDS_NAME
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = PL.load_replay_manifest(str(b))
    assert {r["path"] for r in loaded["files"]} >= {PD.CARDS_NAME}
    assert PD._read_cards(Path(b), loaded)["version"] == PD.CARDS_VERSION
