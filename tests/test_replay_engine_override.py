"""The replay engine override: a derived bundle, the import stamp and the whole-plan rewrite.

These lived in ``test_ghost_signal_stillin_lab.py`` and ran on the haunted lane and its still-in lab
sibling. That lab id was retired on 2026-10-28 with the other two AnimateDiff lab ids (operator: "ship
regions so we have two animatediff"), so they now run on the two AnimateDiff lanes that remain, the
haunted lane and the Mac's lightning lane, which are Ghost siblings (same family, roles, prompt profile
and frame contract). A retired id is refused by the override like any other unregistered one.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry

HAUNTED = "animatediff15_v3_haunted_video"
LIGHTNING = "animatediff15_lightning_video"
RETIRED_LAB = "animatediff15_v3_haunted_regions_video"


def _bundle(tmp_path, name="ep_src"):
    from nodes import production_ledger as PL
    b = tmp_path / name
    (b / "audio").mkdir(parents=True)
    ledger = {"episode_id": name, "lines": [{"line_id": "l1"}],
              "meta": {"freeze_timestamp": "t"},
              "video": {"shots": [{"shot_id": "s1", "engine_id": HAUNTED}]}}
    lp = b / "audio" / ("%s_ledger.json" % name)
    lp.write_text(json.dumps(ledger), encoding="utf-8")
    mp = b / "audio" / ("%s_master.wav" % name)
    mp.write_bytes(b"RIFF0000WAVE")
    files = []
    for rel, p in (("audio/%s_ledger.json" % name, lp), ("audio/%s_master.wav" % name, mp)):
        files.append({"path": rel, "bytes": p.stat().st_size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {"schema_version": PL.REPLAY_MANIFEST_SCHEMA, "source_episode_id": name,
                "source_episode_root": str(tmp_path / "episodes" / name), "source_commit": "abc",
                "ledger": files[0]["path"], "master_audio": files[1]["path"],
                "planned_shots": 1, "files": files}
    (b / PL.REPLAY_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return b


def test_derive_engine_bundle_is_bundle_to_bundle_and_immutable(tmp_path):
    from nodes import production_ledger as PL
    import scripts.otr_freeze_replay_bundle as fz
    src = _bundle(tmp_path)
    derived = fz.derive_engine_bundle(src, LIGHTNING)
    assert derived == tmp_path / ("ep_src__engine_%s" % LIGHTNING)
    man = PL.load_replay_manifest(str(derived))
    assert man["engine_override"] == LIGHTNING and man["derived_from"] == str(src.resolve())
    assert [f["path"] for f in man["files"]] == [f["path"] for f in
                                                 PL.load_replay_manifest(str(src))["files"]]
    assert not (src / PL.REPLAY_MANIFEST_NAME).read_text(encoding="utf-8").count("engine_override")
    with pytest.raises(PL.ReplayBundleError, match="already exists"):
        fz.derive_engine_bundle(src, LIGHTNING)
    with pytest.raises(PL.ReplayBundleError, match="ORIGINAL"):
        fz.derive_engine_bundle(derived, HAUNTED)


def _planned():
    return {
        "roles": {"character_video": HAUNTED, "announcer_visual": HAUNTED},
        "roles_effective": {"character_video": HAUNTED, "announcer_visual": HAUNTED},
        "execution_groups": [{"engine_id": HAUNTED, "role": "character_video"}],
        "shots": [{"shot_id": "s1", "engine_id": HAUNTED, "family": "text_to_video"},
                  {"shot_id": "s2", "engine_id": HAUNTED, "family": "text_to_video"}],
    }


def test_apply_replay_engine_override_rewrites_the_whole_plan_atomically():
    from nodes.otr_shot_lock import _apply_replay_engine_override
    planned = _planned()
    _apply_replay_engine_override(planned, LIGHTNING, {})
    assert set(planned["roles_effective"].values()) == {LIGHTNING}
    assert set(planned["roles"].values()) == {LIGHTNING}
    assert planned["execution_groups"][0]["engine_id"] == LIGHTNING
    assert all(s["engine_id"] == LIGHTNING and s["family"] == "text_to_video" for s in planned["shots"])
    # the reverse direction (the baseline of every A/B) works the same way
    _apply_replay_engine_override(planned, HAUNTED, {})
    assert all(s["engine_id"] == HAUNTED for s in planned["shots"])
    # a non-sibling is refused by name, and nothing moved
    with pytest.raises(ValueError, match="not a sibling"):
        _apply_replay_engine_override(planned, "ltx_8gb", {})
    assert all(s["engine_id"] == HAUNTED for s in planned["shots"])
    with pytest.raises(ValueError, match="not a registered"):
        _apply_replay_engine_override(planned, "no_such_engine", {})
    # a plan without its effective route is refused whole, never rewritten in part
    no_route = {"roles": {"character_video": HAUNTED}, "execution_groups": [],
                "shots": [{"shot_id": "s1", "engine_id": HAUNTED, "family": "text_to_video"}]}
    with pytest.raises(ValueError, match="roles_effective"):
        _apply_replay_engine_override(no_route, LIGHTNING, {})
    assert no_route["shots"][0]["engine_id"] == HAUNTED


def test_an_override_to_a_retired_lab_id_is_refused_and_nothing_moves():
    from nodes.otr_shot_lock import _apply_replay_engine_override
    planned = _planned()
    with pytest.raises(ValueError, match="not a registered"):
        _apply_replay_engine_override(planned, RETIRED_LAB, {})
    assert all(s["engine_id"] == HAUNTED for s in planned["shots"])


def test_import_replay_bundle_stamps_the_override_raw(tmp_path, monkeypatch):
    """Through the replay suite's own episode builder and freezer, so the
    imported ledger is a REAL one (lines, audio, stills) and the only new thing
    under test is the raw override stamp."""
    import importlib.util
    from nodes import production_ledger as PL
    import scripts.otr_freeze_replay_bundle as fz
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    # every ledger the test creates lands under tmp (the replay suite's redirect)
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    ep, _ledger = tcr._make_episode(tmp_path / "episodes")
    src = tcr._freeze(ep, tmp_path / "bundles")
    derived = fz.derive_engine_bundle(src, LIGHTNING)
    led = PL.import_replay_bundle(str(derived))
    meta = led.data["meta"]
    assert meta["replay_engine_override"] == LIGHTNING
    assert meta["replay_derived_from"] == str(pathlib.Path(src).resolve())
    assert meta["replay_of_episode"] == ep.name
    # an ORIGINAL bundle stamps an empty override -- the pure A/A, byte for byte
    led2 = PL.import_replay_bundle(str(src))
    assert led2.data["meta"]["replay_engine_override"] == ""
