"""The post-audio join fails LOUD at ShotLock and stays soft at the floor renderer.

PBUG-20260811-02, hardening half. Deleting the beat-id translation only helps if
the POST-AUDIO ledger actually reaches the consumer. ``overlay_audio_timing``
had four ways to silently hand back the PRE-AUDIO ledger instead -- no durable
path, an identity mismatch, a malformed file, or any other exception -- and each
one restores the exact id space the repair removed. Bug Bible 12.57: resolve the
durable owner, prove same-run identity, REJECT mismatches.

WHY ``strict`` IS A PARAMETER AND NOT A GLOBAL CONTRACT. The function has two
live callers with opposite criticality:

  * ``OTRShotLock.lock`` -- the canonical graph join, gated on ``audio_done``.
    EpisodeAssembler saves the durable ledger in the same function body
    immediately before minting that string, so a join that cannot be proved
    here is an anomaly. Fails loud.
  * ``SignalLostVideoRenderer.render_video`` (``video_engine.py``) -- calls the
    same free function with no ``audio_done`` input at all, uses the overlay for
    title-card timing, is a registered sanctioned policy-floor node, and has no
    try/except around the call. Raising there would turn a slightly-worse title
    card into a hard crash. Stays fail-soft.

A global contract is wrong for one of them whichever way it is written. Found by
the Codex and Sonnet lanes of the 2026-08-12 kibitz panel, which between them
established that both earlier proposals were wrong because both assumed a single
caller.

These tests unset ``OTR_TEST_MODE`` deliberately: the short-circuit at the top of
``overlay_audio_timing`` is OPT-OUT, and the suite already opts out elsewhere
(``tests/test_video_ledger.py``). The claim that this join is "untestable" is
false and cost this project a real defect.
"""
from __future__ import annotations

import json

import pytest

from nodes import otr_shot_lock as sl


def _wire(freeze="freeze-1", episode_id="ep_probe"):
    return {"episode_id": episode_id,
            "meta": {"freeze_timestamp": freeze} if freeze else {},
            "lines": [{"line_id": "b001", "speaker_role": "announcer"}]}


def _disk(tmp_path, payload):
    p = tmp_path / "signal_lost_probe_ledger.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def live(monkeypatch):
    """Leave TEST_MODE so the real disk join runs."""
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    return monkeypatch


# --------------------------------------------------------------------------
# fail-soft default -- the floor renderer's contract, unchanged
# --------------------------------------------------------------------------

def test_default_is_FAIL_SOFT_when_there_is_no_durable_ledger(live):
    """SignalLostVideoRenderer relies on exactly this."""
    live.setattr("nodes._otr_ledger.in_flight_ledger_path", lambda: None)
    wire = _wire()

    assert sl.overlay_audio_timing(wire) is wire


def test_default_is_FAIL_SOFT_on_an_identity_mismatch(live, tmp_path):
    live.setattr("nodes._otr_ledger.in_flight_ledger_path",
                 lambda: _disk(tmp_path, {"episode_id": "other",
                                          "meta": {"freeze_timestamp": "freeze-2"},
                                          "lines": []}))
    wire = _wire()

    assert sl.overlay_audio_timing(wire) is wire


# --------------------------------------------------------------------------
# strict -- the ShotLock contract
# --------------------------------------------------------------------------

def test_strict_REFUSES_when_no_durable_path_resolves(live):
    live.setattr("nodes._otr_ledger.in_flight_ledger_path", lambda: None)

    with pytest.raises(sl.PostAudioJoinFailed) as exc:
        sl.overlay_audio_timing(_wire(), strict=True)
    assert "post-audio join failed" in str(exc.value)


def test_strict_REFUSES_a_stale_sibling_and_SAYS_SO(live, tmp_path):
    """A different episode's audio truth is the dangerous case, and the message
    must name it as a mismatch rather than as absence."""
    live.setattr("nodes._otr_ledger.in_flight_ledger_path",
                 lambda: _disk(tmp_path, {"episode_id": "other",
                                          "meta": {"freeze_timestamp": "freeze-2"},
                                          "lines": []}))

    with pytest.raises(sl.PostAudioJoinFailed) as exc:
        sl.overlay_audio_timing(_wire(), strict=True)
    assert "stale sibling" in str(exc.value)


def test_strict_distinguishes_NO_IDENTITY_from_a_mismatch(live, tmp_path):
    """_same_frozen_episode returns False for BOTH a genuine collision and for
    'neither side carries a receipt at all'. Those are different failures and
    conflating them sends the next reader hunting a collision that never
    happened."""
    live.setattr("nodes._otr_ledger.in_flight_ledger_path",
                 lambda: _disk(tmp_path, {"episode_id": "", "meta": {}, "lines": []}))

    with pytest.raises(sl.PostAudioJoinFailed) as exc:
        sl.overlay_audio_timing(_wire(freeze="", episode_id=""), strict=True)
    message = str(exc.value)
    assert "no identity available" in message
    assert "stale sibling" not in message


def test_a_LEGACY_collision_is_a_mismatch_not_an_absence(live, tmp_path):
    """The case that broke the first classifier. No freeze_timestamp on either
    side, but two DIFFERENT non-empty episode ids -- `_same_frozen_episode`
    reports that with the same "no matching immutable identity receipt" wording
    it uses for a total absence, so sniffing the message called a real stale
    sibling an absence. The verdict must come from the ledgers."""
    live.setattr("nodes._otr_ledger.in_flight_ledger_path",
                 lambda: _disk(tmp_path, {"episode_id": "ep_other_episode",
                                          "meta": {}, "lines": []}))

    with pytest.raises(sl.PostAudioJoinFailed) as exc:
        sl.overlay_audio_timing(_wire(freeze="", episode_id="ep_probe"),
                                strict=True)
    message = str(exc.value)
    assert "stale sibling" in message
    assert "no identity available" not in message


def test_strict_turns_a_malformed_ledger_into_the_NAMED_exception(live, tmp_path):
    """Tests must assert on a type, not on incidental JSONDecodeError wording."""
    p = tmp_path / "signal_lost_probe_ledger.json"
    p.write_text("{not json", encoding="utf-8")
    live.setattr("nodes._otr_ledger.in_flight_ledger_path", lambda: p)

    with pytest.raises(sl.PostAudioJoinFailed):
        sl.overlay_audio_timing(_wire(), strict=True)


def test_strict_leaves_the_CALLERS_ledger_untouched_when_it_refuses(live, tmp_path):
    """A half-overlaid ledger is worse than no overlay: it looks joined.

    This is the CHEAP half of that guarantee -- an identity mismatch returns
    before any mutation code is reached, so it pins the early-reject path rather
    than the deep copy. The copy itself is proved by the SUCCESS test below,
    which is the one that fails if the post-copy rebind is removed."""
    live.setattr("nodes._otr_ledger.in_flight_ledger_path",
                 lambda: _disk(tmp_path, {"episode_id": "other",
                                          "meta": {"freeze_timestamp": "freeze-2"},
                                          "lines": []}))
    wire = _wire()
    before = json.dumps(wire, sort_keys=True)

    with pytest.raises(sl.PostAudioJoinFailed):
        sl.overlay_audio_timing(wire, strict=True)

    assert json.dumps(wire, sort_keys=True) == before


def test_strict_does_not_mutate_the_caller_on_SUCCESS_either(live, tmp_path):
    """The overlay returns a merged COPY under strict."""
    live.setattr("nodes._otr_ledger.in_flight_ledger_path",
                 lambda: _disk(tmp_path, {
                     "episode_id": "ep_final_title",
                     "meta": {"freeze_timestamp": "freeze-1"},
                     "lines": [{"line_id": "b001", "start_s": 9.5, "dur_s": 8.87}],
                 }))
    wire = _wire()
    before = json.dumps(wire, sort_keys=True)

    out = sl.overlay_audio_timing(wire, strict=True)

    assert out is not wire
    assert json.dumps(wire, sort_keys=True) == before, "caller's ledger mutated"
    assert out["lines"][0]["start_s"] == 9.5
    assert out["episode_id"] == "ep_final_title"


# --------------------------------------------------------------------------
# the call sites
# --------------------------------------------------------------------------

def test_lock_asks_for_strict_ONLY_once_audio_done_has_fired(monkeypatch):
    """The signal is what makes a missing join anomalous. Before it fires there
    is nothing to be strict about."""
    seen = []
    monkeypatch.setattr(sl, "overlay_audio_timing",
                        lambda led, strict=False: seen.append(strict) or led)
    node = sl.OTRShotLock()
    payload = json.dumps({"episode_id": "e", "lines": [], "meta": {}})

    for signal in ("", "   "):
        seen.clear()
        try:
            node.lock(payload, audio_done=signal)
        except Exception:
            pass                       # only the strict flag is under test here
        assert seen and seen[0] is False, "strict before audio_done fired"

    seen.clear()
    try:
        node.lock(payload, audio_done="ok")
    except Exception:
        pass
    assert seen and seen[0] is True, "lock must be strict once audio_done fired"


def test_the_floor_renderer_call_site_stays_fail_soft():
    """SignalLostVideoRenderer must not have been switched to strict. It is a
    registered policy-floor node whose call has no try/except; raising there
    crashes it. Asserted by reading the call, so a future edit trips this."""
    import inspect

    from nodes import video_engine

    src = inspect.getsource(video_engine.SignalLostVideoRenderer.render_video)
    assert "overlay_audio_timing(led)" in src, (
        "the floor renderer's overlay call changed shape -- if strict= was "
        "added here, read overlay_audio_timing's docstring first")
