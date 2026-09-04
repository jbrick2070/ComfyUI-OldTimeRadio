"""Every live video engine gets a bookend scene prompt, or owes a stated reason.

**Operator ruling 2026-09-03:** *"we test all engines and lanes; if one doesn't
work we fix it or rip it, not hide it."* and *"there are no gated modes hidden
behind the scenes."*

The driver decides whether an announcer or music beat receives the style pack's
kinetic motion register by testing the engine id against a set
(`render_driver.BOOKEND_SCENE_PROMPT_ENGINES`). An engine that misses it keeps
`build_request`'s static seed -- "a 1940s radio studio, on air sign illuminated,
period broadcast set" -- with no motion, no style, and `prompt_source` never
stamped.

**That set was an inline literal and it went stale three times.** The last
occurrence reached a published episode: `whispers_in_the_park` (2026-09-03,
`otr/obs/`) rendered on `wan_ti2v` and four of its eight beats shipped that seed,
which the operator reported as "basically no movement". The set still carried
`wan_i2v` -- retired that same week and no longer registered, so the entry could
never match -- while its live replacement was absent.

A name is cheap to add and free to forget. These tests are what make forgetting
fail.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import registry as vreg


def _registered():
    return set(getattr(vreg, "CAPABILITIES", {}) or {})


#: EVERY set is checked, not just the prompt set. The first draft of this file
#: checked only `BOOKEND_SCENE_PROMPT_ENGINES`, and review promptly found two
#: tombstoned ids sitting in `SELF_COMPOSED` -- the same dead-id defect this
#: file was written to stop, reproduced inside its own fix and invisible to its
#: own test. A gate that guards one of five doors guards nothing.
_ALL_SETS = (
    ("BOOKEND_SCENE_PROMPT_ENGINES", "BOOKEND_SCENE_PROMPT_ENGINES"),
    ("BOOKEND_SCENE_PROMPT_BOUNDED", "BOOKEND_SCENE_PROMPT_BOUNDED"),
    ("BOOKEND_SCENE_PROMPT_SELF_COMPOSED", "BOOKEND_SCENE_PROMPT_SELF_COMPOSED"),
    ("BOOKEND_SCENE_PROMPT_NOT_TEXT_DRIVEN", "BOOKEND_SCENE_PROMPT_NOT_TEXT_DRIVEN"),
    ("BOOKEND_SCENE_PROMPT_KNOWN_RED", "BOOKEND_SCENE_PROMPT_KNOWN_RED"),
)


@pytest.mark.parametrize("label,attr", _ALL_SETS)
def test_every_member_of_every_set_is_a_real_registered_engine(label, attr):
    """A dead id is a silent no-op that reads as coverage.

    `wan_i2v` sat in the prompt tuple for a week after being retired: it matched
    no shot, and its presence made the set LOOK like it covered the Wan family.
    Two more (`animatediff15_video`, `animatediff15_v3_video`, tombstoned
    2026-08-23) made it into the first draft of the replacement.
    """
    unknown = sorted(e for e in getattr(rd, attr) if e not in _registered())
    assert not unknown, (
        "%s names engines that are not registered: %s -- a dead id can never "
        "match a shot, so it is coverage that does not exist. Remove it, or "
        "register the engine." % (label, unknown))


def test_every_known_red_lane_is_real_and_states_its_owed_decision():
    """A known-red entry is a DEBT, not a blessing. It must name a live lane and
    say what is owed, or it is exactly the hiding the ruling forbids."""
    for engine_id, reason in rd.BOOKEND_SCENE_PROMPT_KNOWN_RED.items():
        assert engine_id in _registered(), (
            "%s is listed known-red but is not a registered engine. Delete the "
            "entry -- a retired lane owes nothing." % engine_id)
        assert engine_id not in rd.BOOKEND_SCENE_PROMPT_ENGINES, (
            "%s is both known-red and in the prompt set; it cannot be both"
            % engine_id)
        assert len(str(reason).split()) >= 12, (
            "%s's known-red reason is too short to be a decision: %r. State "
            "what is owed and who owes it." % (engine_id, reason))
        assert "OWED" in reason or "owed" in reason or "verify" in reason, (
            "%s's known-red entry does not say what is owed. A debt with no "
            "action is a blessing wearing a debt's clothes." % engine_id)


def test_no_live_engine_falls_through_silently():
    """THE GATE THIS FILE EXISTS FOR.

    Every registered engine must be accounted for: it either composes a bookend
    scene prompt, is a Google provider or strict-text-only engine (both handled
    by their own branches of the same condition), is known-red with a stated
    debt, or does not take a text prompt at all. Anything else is an engine
    nobody decided about, which is how `wan_ti2v` shipped bland for a week.
    """
    accounted = (set(rd.BOOKEND_SCENE_PROMPT_ENGINES)
                 | set(rd.BOOKEND_SCENE_PROMPT_BOUNDED)
                 | set(rd.BOOKEND_SCENE_PROMPT_SELF_COMPOSED)
                 | set(rd.BOOKEND_SCENE_PROMPT_NOT_TEXT_DRIVEN)
                 | set(rd.BOOKEND_SCENE_PROMPT_KNOWN_RED)
                 | set(rd._GOOGLE_PROVIDER_PROMPT_ENGINES))

    unaccounted = [e for e in sorted(_registered())
                   if e not in accounted
                   and not rd._is_strict_text_only_engine(e)]

    assert not unaccounted, (
        "These registered engines are not accounted for in the bookend "
        "scene-prompt decision:\n  %s\n\n"
        "Each one silently ships build_request's static seed on announcer and "
        "music beats -- no motion, no style, prompt_source unstamped. Add it to "
        "BOOKEND_SCENE_PROMPT_ENGINES, or to BOOKEND_SCENE_PROMPT_KNOWN_RED "
        "with the decision it owes. Do not leave it undecided."
        % "\n  ".join(unaccounted))


def test_the_engine_that_shipped_the_bland_episode_is_accounted_for():
    """The regression, named, and now PAID. `wan_ti2v` rendered
    `whispers_in_the_park` and put all four bookends on the static seed.

    It is still deliberately NOT in BOOKEND_SCENE_PROMPT_ENGINES: that branch
    emits an LTX-shaped five-clause register (framing constraint, three subject
    motions, a camera move) and wan's own directive asks for "one subject, one
    action, one speed. Do not restate the set" at cfg 5.0, the highest guidance
    in the stack. What KNOWN_RED said it was OWED instead -- an engine-shaped
    formatter emitting one subject/action/speed -- is `bounded_motion_register`,
    and BOOKEND_SCENE_PROMPT_BOUNDED is where it is now paid (2026-09-03).

    The debt therefore must NOT still be sitting in KNOWN_RED: a standing
    "PROVEN DEFECT / OWED" note over a fixed defect reads as coverage exactly
    the way a dead engine id does, which is the failure this whole file exists
    to stop.
    """
    assert "wan_i2v" not in rd.BOOKEND_SCENE_PROMPT_ENGINES
    assert "wan_ti2v" not in rd.BOOKEND_SCENE_PROMPT_ENGINES
    assert "wan_ti2v" in rd.BOOKEND_SCENE_PROMPT_BOUNDED, (
        "wan_ti2v must receive the COMPACTED register -- that is the debt "
        "PBUG-20260903-06 left owed and 2026-09-03 paid")
    assert "fastwan_8gb" in rd.BOOKEND_SCENE_PROMPT_BOUNDED, (
        "fastwan_8gb inherits wan_ti2v's directive and its fix")
    assert "wan_ti2v" not in rd.BOOKEND_SCENE_PROMPT_KNOWN_RED, (
        "the debt is paid; leaving the OWED note standing reads as coverage")


@pytest.mark.parametrize("engine_id", sorted(rd.BOOKEND_SCENE_PROMPT_ENGINES))
def test_each_prompt_owning_engine_is_selectable_and_registered(engine_id):
    assert engine_id in _registered(), engine_id
    assert vreg.get_engine(engine_id) is not None, engine_id


def test_the_five_sets_are_disjoint():
    """An engine in two sets is an engine nobody decided about, twice.

    BOUNDED joined on 2026-09-03. It is a genuinely separate decision from
    ENGINES -- both receive a scene prompt, but of different SHAPES -- so an
    engine in both would be handed the five-clause LTX register and the
    one-action compaction of it, and only the last write would survive.
    """
    sets = {
        "ENGINES": set(rd.BOOKEND_SCENE_PROMPT_ENGINES),
        "BOUNDED": set(rd.BOOKEND_SCENE_PROMPT_BOUNDED),
        "SELF_COMPOSED": set(rd.BOOKEND_SCENE_PROMPT_SELF_COMPOSED),
        "NOT_TEXT_DRIVEN": set(rd.BOOKEND_SCENE_PROMPT_NOT_TEXT_DRIVEN),
        "KNOWN_RED": set(rd.BOOKEND_SCENE_PROMPT_KNOWN_RED),
    }
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sets[a] & sets[b]
            assert not overlap, "%s and %s both claim %s" % (a, b, sorted(overlap))


def test_the_ghost_family_is_never_handed_a_scene_prompt():
    """Ghost composes its own positive AND negative from the same authorities.

    Handing it a scene prompt would overwrite the positive and orphan the
    negative -- which is why the driver condition also carries
    `and not _ghost_composed`. The set is the readable statement of that.
    """
    for engine_id in rd.BOOKEND_SCENE_PROMPT_SELF_COMPOSED:
        assert engine_id not in rd.BOOKEND_SCENE_PROMPT_ENGINES, engine_id


# ---------------------------------------------------------------------------
# THE MOTION-REGISTER SELECTOR, PINNED ON THE IDS PRODUCTION ACTUALLY MINTS
#
# Nothing pinned this before. `tests/test_visual_styles_b.py` uses
# `shot_b000_music_open`, which matched the OLD `endswith` check too -- so the
# 2026-09-03 fix could be reverted whole and the suite would stay green while
# every cold open and sign-off silently fell back to `music_inter`.
#
# Measured across the music_visual shots on disk: 100% are
# `shot_music_opening_001` / `shot_music_closing_001`.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shot_id,expected", [
    ("shot_music_opening_001", "music_open"),
    ("shot_music_closing_001", "music_close"),
    ("shot_music_inter_01_001", "music_inter"),
    ("shot_b000_music_open", "music_open"),      # the synthetic opener, legacy shape
])
def test_the_selector_resolves_the_ids_production_mints(shot_id, expected):
    assert rd._ltx_motion_role_key("music_visual", shot_id, False) == expected


def test_close_is_matched_by_token_not_by_substring():
    """`"close"` is not inside `"closing"` -- that one missing letter is why
    every sign-off used the flat register. And the reverse hazard is real too:
    `"tag"` IS inside "montage" and "stage", so the list is tested against the
    id's own "_"-separated segments rather than as substrings."""
    assert rd._ltx_motion_role_key(
        "music_visual", "shot_music_closing_001", False) == "music_close"
    # A word merely CONTAINING a token must not fire it.
    for innocent in ("shot_music_montage_001", "shot_music_backstage_002"):
        assert rd._ltx_motion_role_key(
            "music_visual", innocent, False) == "music_inter", innocent


def test_a_character_beat_gets_no_register_from_the_driver():
    """The driver is what scopes the pack register to bookends -- the finalizer
    does NOT ignore a non-empty `pack_motion` by role. An earlier version of
    this test passed `pack_motion=""` twice and compared them, which would have
    passed with the whole feature removed."""
    assert rd._ltx_motion_role_key("character_video", "shot_b002", False) == ""
    assert rd._ltx_motion_role_key("announcer_visual", "shot_b001", False) == "announcer"
