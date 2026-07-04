"""S2 -- durable persistence contract (credits enrichment 2026-07-03).

The late OTR_CreditsRoll node can only read what is on disk, so every
credit-bearing receipt must land in the production-ledger SINGLETON:

- ``stamp_durable`` copies sections/meta from a LOCAL wire-parsed ledger
  into the singleton (the agy MUST-FIX #2 gotcha: CastLock + the image
  dispatcher stamp a local dict, never the singleton -- a bare
  ``get_ledger().save()`` would persist EMPTY/stale state).
- ``Ledger.save()`` returns None on failure and never raises (Fable
  BUILD-BREAKER #3), so ``stamp_durable`` checks the return and RAISES
  ``LedgerStampError`` in production; ``OTR_TEST_MODE=1`` is the explicit
  test-mode injection path (in-memory copy only, no disk save).
- Call sites under contract: cast_lock (cast + voice meta), the image
  dispatcher (images + meta.image_engines), stable_audio_theme
  (meta.music_engine), and otr_video_render_batch (meta.render_engines --
  previously best-effort/silent, now LOUD with no swallowing catch).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes import production_ledger as pl

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def fresh_singleton(tmp_path):
    """A known, isolated singleton ledger under tmp_path; restored after."""
    saved = pl._CURRENT
    try:
        led = pl.new_ledger("ep_s2_stamps", str(tmp_path))
        yield led
    finally:
        with pl._LEDGER_LOCK:
            pl._CURRENT = saved


# --------------------------------------------------------------------------- #
# stamp_durable core contract
# --------------------------------------------------------------------------- #
def test_stamp_durable_copies_sections_and_meta_into_singleton(fresh_singleton):
    cast = [{"char_id": "c1", "name": "MONTY", "voice_ref_id": "ref_a",
             "voice_engine": "indextts2", "commercial_clean": True}]
    ret = pl.stamp_durable(
        sections={"cast": cast},
        meta_updates={"music_engine": "stable_audio_music"},
        source="spec",
    )
    led = pl.get_ledger()
    assert led.data["cast"] == cast
    assert led.data["meta"]["music_engine"] == "stable_audio_music"
    # test-mode injection path: in-memory only, no disk save, returns None
    assert ret is None
    assert not led.data["meta"].get("paths")  # save() never ran


def test_stamp_durable_test_mode_writes_no_file(fresh_singleton, tmp_path):
    pl.stamp_durable(meta_updates={"music_engine": "x"}, source="spec")
    assert not list(tmp_path.rglob("*_ledger.json"))


def test_stamp_durable_raises_loud_on_save_none(fresh_singleton, monkeypatch):
    """Fable BUILD-BREAKER #3: save() returns None on failure, never raises --
    the stamp MUST check the return and raise in production mode."""
    monkeypatch.setenv("OTR_TEST_MODE", "0")
    monkeypatch.setattr(pl.Ledger, "save", lambda self: None)
    with pytest.raises(pl.LedgerStampError) as exc:
        pl.stamp_durable(meta_updates={"music_engine": "x"}, source="spec")
    assert "spec" in str(exc.value)


def test_stamp_durable_returns_path_on_success(fresh_singleton, monkeypatch):
    monkeypatch.setenv("OTR_TEST_MODE", "0")
    monkeypatch.setattr(pl.Ledger, "save", lambda self: "C:/fake/ledger.json")
    ret = pl.stamp_durable(meta_updates={"music_engine": "x"}, source="spec")
    assert ret == "C:/fake/ledger.json"


def test_stamp_durable_meta_merge_is_additive(fresh_singleton):
    pl.get_ledger().data["meta"] = {"episode_seed": 42}
    pl.stamp_durable(meta_updates={"image_engines": {"by_role": {}}},
                     source="spec")
    meta = pl.get_ledger().data["meta"]
    assert meta["episode_seed"] == 42            # untouched
    assert meta["image_engines"] == {"by_role": {}}


# --------------------------------------------------------------------------- #
# Call site: CastLock -- the singleton-copy gotcha
# --------------------------------------------------------------------------- #
_CHAR_CAST = [
    {"char_id": "c1", "name": "MONTY", "gender": "male",
     "voice_preset": "v2/en_speaker_1"},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
     "voice_preset": "v2/en_speaker_6"},
]


def test_cast_lock_copies_local_wire_ledger_into_singleton(fresh_singleton):
    """cast_lock.lock() stamps a LOCAL dict from load_ledger(script_json);
    S2 requires the final cast + CastLock-owned meta keys to be copied into
    the singleton (else the credits roll reads an empty ledger)."""
    from nodes.cast_lock import CastLock

    script = json.dumps({"meta": {"episode_seed": 42}, "cast": _CHAR_CAST,
                         "lines": []})
    out = CastLock().lock(script_json=script,
                          cast_voice_policy="preserve_ledger")
    wire = json.loads(out[0])

    led = pl.get_ledger()
    assert led.data["cast"] == wire["cast"]      # DELIVERED cast, not stale
    meta = led.data["meta"]
    assert meta["cast_lock_revision"] == 1
    assert meta["cast_voice_policy"] == "preserve_ledger"
    assert meta["delivery_profile_id"] == "neutral"
    assert "voice_bank_id" in meta


# --------------------------------------------------------------------------- #
# Call site: image dispatcher -- meta.image_engines + images durable
# --------------------------------------------------------------------------- #
def test_image_dispatcher_stamps_singleton(fresh_singleton, tmp_path):
    import types

    from nodes import otr_image_gen_dispatcher as disp
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_shared import role_compat as rc
    import numpy as np

    saved = dict(ireg._IMAGE_REGISTRY._registry)
    ireg._IMAGE_REGISTRY._registry.clear()
    try:
        ireg.register(types.SimpleNamespace(
            name="flux_gen1", roles=rc.ROLES, default_roles=rc.ROLES,
            commercial_clean=True, requires_flag=None,
            required_inputs=("text_prompt",), engine_version="1",
        ))
        ledger = {"episode_id": "ep_s2_img",
                  "cast": [{"char_id": "c1", "name": "BABA"}]}
        policy = {"image_models":
                  {"character_image_model": {"engine_id": "flux_gen1"}},
                  "seed": {"request_seed": 0}, "granularity": {}}
        prompts = {"version": 1, "objects": [{
            "object_id": "c1", "kind": "portrait", "role": "character_video",
            "char_id": "c1", "w": 832, "h": 1216,
            "prompt": "a spacer, station", "prompt_hash": "ph1",
            "source": "llm"}]}
        disp.dispatch_images(
            ledger, policy, prompts,
            gen_fn=lambda _req: np.full((8, 8, 3), 60, dtype=np.uint8),
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir",
        )
    finally:
        ireg._IMAGE_REGISTRY._registry.clear()
        ireg._IMAGE_REGISTRY._registry.update(saved)

    led = pl.get_ledger()
    ie = led.data["meta"]["image_engines"]
    assert ie["image_revision"] == 1
    assert ie["by_role"]                         # per-role histogram present
    assert led.data["images"]["images"]          # images section copied too


# --------------------------------------------------------------------------- #
# Call site: render batch -- previously best-effort, now LOUD
# --------------------------------------------------------------------------- #
def test_render_engines_stamp_raises_on_save_failure(fresh_singleton,
                                                     monkeypatch):
    from nodes.otr_video_render_batch import _stamp_render_engines_meta

    monkeypatch.setenv("OTR_TEST_MODE", "0")
    monkeypatch.setattr(pl.Ledger, "save", lambda self: None)
    with pytest.raises(pl.LedgerStampError):
        _stamp_render_engines_meta(
            {"engine_histogram": {}, "video_revision": 1}, 0)


def test_render_engines_stamp_lands_in_singleton(fresh_singleton):
    from nodes.otr_video_render_batch import _stamp_render_engines_meta

    _stamp_render_engines_meta(
        {"engine_histogram": {"ltx": 3}, "video_revision": 2}, 1234)
    payload = pl.get_ledger().data["meta"]["render_engines"]
    assert payload["histogram"] == {"ltx": 3}
    assert payload["video_revision"] == 2
    assert payload["vram_peak_mb"] == 1234


def test_render_batch_call_site_has_no_swallowing_catch():
    """The old call site wrapped the stamp in ``except Exception: warn`` --
    a silent fallback that defeats the LOUD contract. It must be gone."""
    src = (REPO_ROOT / "nodes" / "otr_video_render_batch.py").read_text(
        encoding="utf-8")
    assert "render_engines stamp skipped" not in src
    assert "stamp_durable" in src


# --------------------------------------------------------------------------- #
# Call site: music engine provenance (no durable path existed at all)
# --------------------------------------------------------------------------- #
def test_music_theme_stamps_engine_via_stamp_durable():
    """node 83's ``done`` output is unlinked in the workflow JSON, so the
    engine name died on the wire; stable_audio_theme must stamp
    meta.music_engine through the durable contract."""
    src = (REPO_ROOT / "nodes" / "stable_audio_theme.py").read_text(
        encoding="utf-8")
    assert "stamp_durable" in src
    assert '"music_engine"' in src
