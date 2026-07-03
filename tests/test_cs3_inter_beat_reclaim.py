"""CS-3 inter-beat reclaim -- run_episode drains the prior heavy engine before a
beat that crosses to a DIFFERENT engine, and SKIPS same-engine consecutive beats.

Two heavy engines co-resident on the 16GB card (ltx 12.5 + humo 7 = 19.5) caused
29s/it edge page-thrash. The per-beat teardown detaches but does not flush the
allocator; this selective inter-beat free (the surgical Lever-1 freer) does --
only on an engine change, so the resident-stack reuse (humo x3) is preserved.

Pure: render_shot + the residue freer are stubbed, so no GPU and no real models.
"""

from __future__ import annotations

import nodes._otr_vram_levers as _lv
from nodes._otr_video_engines import render_driver as rd


def _ledger(engine_seq):
    return {"video": {"video_revision": 1, "shots": [
        {"shot_id": "b%d" % i, "engine_id": eng, "family": "f_%s" % eng}
        for i, eng in enumerate(engine_seq)
    ]}}


def _install(monkeypatch):
    """Stub the residue freer (record reasons) + render_shot (no fallback) +
    build_request. Returns the recorded-reasons list."""
    reasons = []

    def fake_free(*, reason=""):
        reasons.append(reason)
        return {"free_gb_after": 14.0, "steps_run": [], "steps_failed": []}

    monkeypatch.setattr(_lv, "free_otr_pipeline_residue", fake_free)

    def fake_render_shot(shot, request, **kw):
        out = {"shot_id": shot["shot_id"], "engine_id": shot["engine_id"],
               "family": shot.get("family")}
        clip = {"clip_id": shot["shot_id"], "engine_id": shot["engine_id"],
                "exists": True, "size": 1}
        return clip, out, [shot["engine_id"]], 100

    monkeypatch.setattr(rd, "render_shot", fake_render_shot)
    monkeypatch.setattr(rd, "build_request",
                        lambda shot, assets, fc, canvas: {"shot_id": shot["shot_id"]})
    return reasons


def _inter_beat(reasons):
    return [r for r in reasons if "inter-beat" in r]


def test_reclaim_on_engine_change(monkeypatch):
    reasons = _install(monkeypatch)
    rd.run_episode(_ledger(["humo_1.7B", "ltx_video"]))
    inter = _inter_beat(reasons)
    assert len(inter) == 1
    assert "humo_1.7B->ltx_video" in inter[0]


def test_no_reclaim_for_same_engine_run(monkeypatch):
    # humo x3 -- the resident stack is reused; NO inter-beat reclaim churn.
    reasons = _install(monkeypatch)
    rd.run_episode(_ledger(["humo_1.7B", "humo_1.7B", "humo_1.7B"]))
    assert _inter_beat(reasons) == []


def test_reclaim_only_at_the_boundaries(monkeypatch):
    # ltx, ltx, humo, humo, wan -> reclaim only at ltx->humo and humo->wan.
    reasons = _install(monkeypatch)
    rd.run_episode(
        _ledger(["ltx_video", "ltx_video", "humo_1.7B", "humo_1.7B", "wan_i2v"]))
    inter = _inter_beat(reasons)
    assert len(inter) == 2
    assert "ltx_video->humo_1.7B" in inter[0]
    assert "humo_1.7B->wan_i2v" in inter[1]


def test_first_beat_has_no_inter_beat_reclaim(monkeypatch):
    # A single beat -> the pre-render free runs, but no inter-beat reclaim.
    reasons = _install(monkeypatch)
    rd.run_episode(_ledger(["wan_i2v"]))
    assert _inter_beat(reasons) == []
