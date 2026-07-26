"""``OTR_CLOUD_PIXVERSE_DURATION`` may not leave the provider's menu.

Found by the chunk-7a QA panel, fixed in 7b's first slice (2026-07-26).

``CloudWordRazzleEngine._duration_seconds`` used to open with

    env = os.environ.get("OTR_CLOUD_PIXVERSE_DURATION", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass

-- returning the operator's raw integer and skipping the ``8 if secs > 5 else
5`` bucket ladder that every other path goes through. Two defects in six lines:

1. Pixverse serves a fixed TWO-ENTRY menu, 5 s or 8 s, which this adapter
   declares as ``discrete_frames=(125, 200)`` at 25 fps. A pin of 20 sent the
   provider a length the contract says is impossible -- and by then
   ``otr_shot_lock`` had already partitioned the beat over 5 s / 8 s clips and
   minted a still per jump segment, so the plan and the request disagreed.
2. A NON-integer was silently swallowed by ``except ValueError: pass`` and fell
   through to the derived default. It was the only duration env in
   ``eng_cloud_video.py`` that failed quietly rather than loud, so a typo in a
   launch file changed the render and left nothing in the log.

This is deliberately the ONLY env refusal in this chunk. Whether an env var may
lower a declared CEILING is a separate and much harder question -- several of
those vars are documented operator knobs for VRAM-constrained boxes -- and is
left open in ``docs/2026-07-26-chunk-7b-window-prompt.md``. A menu is not a
ceiling: there is no reading of "5 s or 8 s" under which 20 is a smaller ask.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg


def _engine():
    return vreg.get_engine("word_razzle")


def _request(frames=200):
    return {"canvas": {"fps": 25}, "timing": {"target_frame_count": frames}}


# ---------------------------------------------------------------------------
# The menu is the contract, and the contract is the menu
# ---------------------------------------------------------------------------

def test_the_menu_matches_the_declared_contract():
    """Guard the guard: if these ever disagree, every test below is testing
    the wrong numbers."""
    contract = fc.frame_contract_for(_engine())
    assert contract.discrete_frames == (125, 200)
    assert contract.native_fps == 25
    assert sorted(d // 25 for d in contract.discrete_frames) == [5, 8]


@pytest.mark.parametrize("secs", [5, 8])
def test_an_on_menu_pin_is_honoured(secs, monkeypatch):
    """The knob still works for the values the provider actually serves."""
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_DURATION", str(secs))
    assert _engine()._duration_seconds(_request()) == secs


@pytest.mark.parametrize("secs", [1, 4, 6, 7, 9, 20, 300])
def test_an_off_menu_pin_REFUSES(secs, monkeypatch):
    """The defect: any of these used to be sent to the provider verbatim."""
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_DURATION", str(secs))
    with pytest.raises(fc.ContractEnvConflict, match="not on Pixverse's"):
        _engine()._duration_seconds(_request())


@pytest.mark.parametrize("raw", ["", "   "])
def test_a_blank_pin_is_not_a_pin(raw, monkeypatch):
    """Empty means unset -- it must fall through to the derived path, not
    refuse. Without this the refusal would fire on an operator who exported the
    variable and left it empty, which is a common way to disable one."""
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_DURATION", raw)
    assert _engine()._duration_seconds(_request(200)) == 8


@pytest.mark.parametrize("raw", ["eight", "5.5", "5s", "-"])
def test_a_MALFORMED_pin_refuses_LOUD_instead_of_being_swallowed(raw,
                                                                 monkeypatch):
    """The second half of the defect. ``except ValueError: pass`` meant a typo
    silently changed the render and left nothing in the log."""
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_DURATION", raw)
    with pytest.raises(fc.ContractEnvConflict, match="whole number"):
        _engine()._duration_seconds(_request())


# ---------------------------------------------------------------------------
# The derived path is untouched -- the refusal is scoped to the pin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frames,expected", [
    (0, 5),        # no timing at all -> the 5 s floor
    (25, 5),       # 1 s
    (125, 5),      # exactly 5 s
    (150, 8),      # 6 s -> the 8 s tier
    (200, 8),      # exactly 8 s
    (500, 8),      # 20 s of beat -> still 8 s, the provider's ceiling
])
def test_the_derived_duration_is_unchanged(frames, expected, monkeypatch):
    """Without this, the fix above could have been written as "always refuse"
    and every other test here would still pass.

    Note the last row: a 20-second beat still resolves to an 8-second clip, on
    the derived path, silently. That is the ``8 if secs > 5 else 5`` clamp --
    a provider-side shortfall this chunk does NOT fix. It is on 7c's list
    beside the ping-pong extender, and it is the reason the coverage plan
    partitions word_razzle beats over the menu rather than asking for one long
    clip.
    """
    monkeypatch.delenv("OTR_CLOUD_PIXVERSE_DURATION", raising=False)
    assert _engine()._duration_seconds(_request(frames)) == expected
