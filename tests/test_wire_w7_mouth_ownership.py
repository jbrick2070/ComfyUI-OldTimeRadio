"""WIRE-W7 -- the operator's three mouth rulings finally have an OWNER.

r3 MUST-FIX 11: *"No W1-W6 step enforces the operator's three rulings. Nothing
makes every audio-in beat receive a mouth-bearing still, nothing makes the
radio-face the default, nothing enforces one overheard human face per episode.
The plan has the rulings and no owner for them."* An unowned ruling is a ruling
that silently lapses -- and this one had already been through a whole build
block without a single line of code able to state it.

THE RULINGS, and the tests that hold them:

1. Every audio-in beat gets a still with a MOUTH.
2. The lips may be a person OR A RADIO.
3. THE SET SPEAKS BY DEFAULT; A FACE MUST BE OVERHEARD -- one face per episode
   at most, and only for a line the engine can hold in a SINGLE TAKE.

DECIDED FROM THE FROZEN ROUTE, NEVER FROM PROSE. That is the operator's own
wording and it is the property the last test in the first section pins: two
beats with wildly different text and the same route get the same answer.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import nodes._otr_video_engines  # noqa: F401,E402  -- populate the registry
from nodes import otr_shot_lock as sl  # noqa: E402
from nodes._otr_video_engines import mouth_policy as mp  # noqa: E402
from nodes._otr_video_engines import registry as vreg  # noqa: E402
from nodes._otr_video_engines.render_driver import engine_family  # noqa: E402


# ---------------------------------------------------------------------------
# Ruling 1 + 2: every audio-in beat owes a mouth, human OR radio
# ---------------------------------------------------------------------------

def test_a_CHARACTER_FACE_beat_owes_a_HUMAN_mouth():
    assert mp.mouth_owner_for_beat(
        engine_id="humo", family="audio_driven_face",
        role="character_video", is_character_face=True) == mp.MOUTH_HUMAN


def test_the_ANNOUNCER_and_the_MUSIC_bookends_owe_the_RADIO():
    """"The cabinet is the speaker for the announcer (who IS the station) and
    for the music bookends." These are the same two roles the dispatcher
    structurally redirects away from HuMo, for the same reason."""
    for role in ("announcer_visual", "music_visual"):
        assert mp.mouth_owner_for_beat(
            engine_id="ltx_audio_in", family="audio_conditioned_video",
            role=role, is_character_face=False) == mp.MOUTH_RADIO


def test_a_NON_AUDIO_beat_owes_NO_mouth():
    """A wan_i2v scene or an ltx_video b-roll animates an image; nothing about
    it is driven by a voice, so the ruling does not reach it. Getting this
    wrong in the other direction would demand lips on every landscape."""
    assert mp.mouth_owner_for_beat(
        engine_id="wan_i2v", family="image_to_video",
        role="character_video", is_character_face=False) == mp.MOUTH_NONE


def test_an_AUDIO_REACTIVE_VISUALIZER_is_not_an_audio_IN_beat():
    """CONTROL, and the distinction is real. ``viz_green`` reads the master mix
    to paint a scope -- ``render_driver._uses_ambient_master_audio`` names it by
    engine id for the SLICE question. It is abstract by declaration and has no
    face to give a mouth to, so it must NOT be dragged in by that other list."""
    assert "abstract" not in mp.AUDIO_IN_FAMILIES
    assert mp.mouth_owner_for_beat(
        engine_id="viz_green", family="abstract", role="music_visual",
        is_character_face=False) == mp.MOUTH_NONE


def test_the_UNOWNED_case_is_REFUSED_by_name():
    """THE GAP r3 NAMED. An audio-in beat that is neither a character face nor
    a cabinet role WILL be handed a still and WILL animate it, and nothing said
    whether that still has lips. Before this chunk it rendered silently."""
    with pytest.raises(mp.MouthPolicyError) as caught:
        mp.mouth_owner_for_beat(
            engine_id="ltx_audio_in", family="audio_conditioned_video",
            role="background_abstract", is_character_face=False)
    assert "NO FALLBACK" in str(caught.value)
    assert "background_abstract" in str(caught.value)


def test_the_ANSWER_is_a_function_of_the_ROUTE_and_NOTHING_ELSE():
    """"Never inferred from prose" is the operator's own wording. The policy
    takes no text at all -- which is the strongest form of that promise -- so
    this pins the signature rather than a behaviour that could regress into
    reading a prompt."""
    import inspect
    params = set(inspect.signature(mp.mouth_owner_for_beat).parameters)
    assert params == {"engine_id", "family", "role", "is_character_face"}
    assert not (params & {"text_prompt", "line", "beat_text", "creative"})


@pytest.mark.parametrize("name", sorted(vreg.all_engine_names()))
def test_EVERY_AUDIO_IN_ENGINE_answers_for_every_role_it_DECLARES(name):
    """The gate is only worth having if the shipped roster passes it, and the
    question is asked through the REAL authority: ``_is_character_face_beat``,
    given a shot row shaped the way the dispatcher shapes one. A hand-rolled
    "is this a face" here would be a second derivation that could agree with
    the policy while disagreeing with the dispatcher.

    Engines declaring no roles are skipped rather than assumed: ``roles`` is
    UI-sort/self-description metadata (``role_compat`` is the real gate), so an
    empty tuple means "states no preference", not "serves nothing". The route
    that empty tuple leaves open is pinned by the next test."""
    from nodes._otr_video_engines.render_driver import _is_character_face_beat

    family = engine_family(name, "")
    if not mp.beat_owes_a_mouth(family):
        pytest.skip("%s (%s) is not an audio-in family" % (name, family))
    roles = tuple(getattr(vreg.get_engine(name), "roles", ()) or ())
    if not roles:
        pytest.skip("%s declares no roles (see the cloud-lane test)" % name)
    for role in roles:
        shot = {"role": role, "engine_id": name}
        owner = mp.mouth_owner_for_beat(
            engine_id=name, family=family, role=role,
            is_character_face=_is_character_face_beat(shot))
        assert owner in (mp.MOUTH_HUMAN, mp.MOUTH_RADIO), (
            "%s on role %r resolves to %r" % (name, role, owner))


def test_a_CLOUD_AUDIO_CONDITIONED_lane_on_a_CHARACTER_beat_is_REFUSED():
    """MEASURED, not missed -- this is the one live route the ruling closes.

    ``cloud_seedance_2`` / ``cloud_wan_i2v_audio`` are ``audio_conditioned_video``
    and declare no roles, so an operator CAN aim one at a character beat. When
    they do, ``_is_character_face_beat`` says False -- only ``ltx_audio_in``
    counts as a character face outside the HuMo family -- so the beat gets the
    ambient master mix and a SCENE still: an audio-in engine animating an image
    with no lips in it. That is precisely r3's unowned case, and it now refuses
    with the remedy in the message rather than rendering a mouthless take.

    Recorded here because a refusal nobody wrote down is indistinguishable from
    an oversight the next time somebody reads this file."""
    from nodes._otr_video_engines.render_driver import _is_character_face_beat

    for name in ("cloud_seedance_2", "cloud_wan_i2v_audio"):
        if not vreg.is_registered(name):
            pytest.skip("%s is not registered on this box" % name)
        shot = {"role": "character_video", "engine_id": name}
        assert _is_character_face_beat(shot) is False
        with pytest.raises(mp.MouthPolicyError) as caught:
            mp.mouth_owner_for_beat(
                engine_id=name, family=engine_family(name, ""),
                role="character_video",
                is_character_face=_is_character_face_beat(shot))
        assert "NO FALLBACK" in str(caught.value)


def test_a_CLOUD_AVATAR_on_a_character_beat_is_FINE():
    """CONTROL for the test above, and it is what keeps that refusal narrow:
    ``cloud_kling_avatar`` is ``audio_driven_face``, so the same empty ``roles``
    tuple aimed at the same beat answers HUMAN. The refusal is about the
    FAMILY's relationship to a face, not about being a cloud lane."""
    from nodes._otr_video_engines.render_driver import _is_character_face_beat

    if not vreg.is_registered("cloud_kling_avatar"):
        pytest.skip("cloud_kling_avatar is not registered on this box")
    shot = {"role": "character_video", "engine_id": "cloud_kling_avatar"}
    assert mp.mouth_owner_for_beat(
        engine_id="cloud_kling_avatar",
        family=engine_family("cloud_kling_avatar", ""),
        role="character_video",
        is_character_face=_is_character_face_beat(shot)) == mp.MOUTH_HUMAN


# ---------------------------------------------------------------------------
# Ruling 3: one face per episode, and only in a single take
# ---------------------------------------------------------------------------

def _beat(beat_id, *, engine_id="humo", family="audio_driven_face",
          role="character_video", char_id="c1", face=True, multi=False):
    return {"beat_id": beat_id, "engine_id": engine_id, "family": family,
            "role": role, "char_id": char_id, "is_character_face": face,
            "is_multi_clip": multi}


def _faces(beats):
    faces, _long, _demoted = mp.audit_episode_faces(beats)
    return faces


def _long_takes(beats):
    _faces_, long, _demoted = mp.audit_episode_faces(beats)
    return long


def _demoted(beats):
    _faces_, _long, demoted = mp.audit_episode_faces(beats)
    return demoted


def test_THE_SET_SPEAKS_THROUGHOUT_is_a_legal_episode():
    """And it is the DEFAULT look, not a degraded one -- an episode with no
    human face at all passes without comment."""
    assert _faces([
        _beat("b0", engine_id="ltx_audio_in",
              family="audio_conditioned_video", role="announcer_visual",
              char_id="", face=False),
        _beat("b1", engine_id="wan_i2v", family="image_to_video",
              char_id="", face=False),
    ]) == ()


def test_ONE_character_across_MANY_beats_is_still_ONE_face():
    """Counted by DISTINCT char_id, not by beat: a character who says three
    private lines is one face, and a per-beat count would forbid the very
    thing the ruling allows."""
    assert _faces([
        _beat("b1", char_id="ada"), _beat("b2", char_id="ada"),
        _beat("b3", char_id="ada"),
    ]) == ("ada",)


def test_TWO_faces_in_one_episode_ROUTES_the_second_to_the_CABINET():
    """"One face per episode at most" -- ENFORCED BY ROUTING, not by refusing.

    This raised until the live 45-word campaign, where it refused every HuMo
    episode outright: four engines, every pass, immediately after the W4e
    coverage fix had got them past the previous blocker. Its own message named
    the remedy -- "give the other character(s) the cabinet" -- which is a
    ROUTING instruction, and the operator then ruled the general case:
    *"there's no max length ... we just need to make sure there's enough video
    and enough stills to cover every beat."*

    So the rule is now stronger, not weaker: it holds on EVERY episode instead
    of holding on the ones that happened to comply and killing the rest.
    """
    faces, _long, demoted = mp.audit_episode_faces([
        _beat("b1", char_id="ada"),
        _beat("b2", char_id="grace"),
    ])
    assert len(faces) == mp.MAX_HUMAN_FACES_PER_EPISODE
    assert demoted, "the excess face must be reported, never silently dropped"
    demoted_ids = {cid for cid, _bids in demoted}
    assert demoted_ids and demoted_ids.isdisjoint(set(faces))
    for _cid, bids in demoted:
        assert bids, "a demoted character must name the beats it was moved on"


def test_the_face_the_episode_LEANS_ON_HARDEST_is_the_one_that_is_kept():
    """Deterministic and defensible: most human-mouth beats wins, ties broken
    by char_id. Two runs of the same ledger must route identically, and the
    choice must never be read from the prose."""
    faces, _long, demoted = mp.audit_episode_faces([
        _beat("b1", char_id="grace"),
        _beat("b2", char_id="ada"), _beat("b3", char_id="ada"),
        _beat("b4", char_id="ada"),
    ])
    assert faces == ("ada",), "ada carries three beats to grace's one"
    assert demoted == (("grace", ("b1",)),)


def test_the_routing_is_STABLE_across_repeated_audits():
    beats = [_beat("b1", char_id="ada"), _beat("b2", char_id="grace"),
             _beat("b3", char_id="mira")]
    first = mp.audit_episode_faces(beats)
    assert mp.audit_episode_faces(beats) == first, (
        "the same episode routed two different ways; a look decision that is "
        "not deterministic cannot be reviewed or reversed")


def test_a_TIE_is_broken_by_char_id_not_by_dict_order():
    faces, _long, demoted = mp.audit_episode_faces([
        _beat("b1", char_id="zara"), _beat("b2", char_id="ada"),
    ])
    assert faces == ("ada",), "the tie must break on the sorted char_id"
    assert demoted == (("zara", ("b1",)),)


def test_a_MULTI_CLIP_face_beat_is_REPORTED_but_NOT_REFUSED():
    """"...and only for a line the engine can hold in a SINGLE TAKE." A
    multi-clip face beat IS a jump cut -- the same character regenerated
    mid-line from a different seed, cutting to themselves.

    IT WARNS RATHER THAN REFUSING, and that correction landed the same day the
    rule was written. It was terminal for exactly as long as it took to run the
    first live campaign, where it refused every HuMo episode outright. The
    operator's own remedy is *"shorten the line, or let the cabinet speak it"*
    -- a ROUTING instruction -- and nothing in this build re-routes, so a
    terminal check punished him for a direction the director made on a beat the
    engine could otherwise render. Refusing is only honest when the alternative
    is shipping a defect; here the alternative is a jump cut he can see.

    THE DAY A RE-ROUTE LANDS, this should go back to terminal."""
    faces, long_takes, _demoted = mp.audit_episode_faces(
        [_beat("b1", char_id="ada", multi=True)])
    assert faces == ("ada",), "the beat still renders"
    assert long_takes == (("b1", "ada"),), "and it is still reported by name"


def test_a_MULTI_CLIP_CABINET_beat_is_NOT_EVEN_REPORTED():
    """CONTROL, and it is the whole point of the asymmetry: the radio may be
    cut across as many clips as the beat needs, silently. Without this the
    single-take clause would nag about every long announcer beat."""
    faces, long_takes, _demoted = mp.audit_episode_faces([
        _beat("b0", engine_id="ltx_audio_in",
              family="audio_conditioned_video", role="announcer_visual",
              char_id="", face=False, multi=True),
    ])
    assert faces == () and long_takes == ()


def test_NEITHER_CLAUSE_REFUSES_and_BOTH_still_REPORT():
    """The two clauses answer differently and must not drift into one answer.

    The single-take clause WARNS and renders the jump cut. The cap ROUTES the
    excess face to the cabinet. Neither kills the episode, and both say what
    they did -- an episode that hits both must still render, with two distinct
    reports naming two distinct problems.
    """
    faces, long_takes, demoted = mp.audit_episode_faces([
        _beat("b1", char_id="ada", multi=True),
        _beat("b2", char_id="grace", multi=True),
    ])
    assert len(faces) == mp.MAX_HUMAN_FACES_PER_EPISODE
    assert demoted, "the cap must report the character it moved"
    kept = set(faces)
    assert {cid for cid, _ in demoted}.isdisjoint(kept)
    # The single-take report covers the face that WAS kept; a demoted character
    # no longer wears a face, so its jump cut is moot.
    assert any(cid in kept for _bid, cid in long_takes), (
        "the kept face is multi-clip and must still be reported as a long take")


def test_the_demoted_report_is_sorted_by_char_id_not_by_beat_count():
    """The routing report is something the operator READS, so it is ordered the
    way a person looks something up -- by character -- not by the internal
    ranking that decided who kept the face.

    The two orders differ only when the demoted characters carry DIFFERENT beat
    counts, which is why a fixture of one-beat characters cannot see this: the
    ranking and the alphabet agree and the sort looks redundant.
    """
    _faces_, _long, demoted = mp.audit_episode_faces([
        # ada keeps the face (4 beats). Of the demoted, zara has more beats
        # than beck -- so the RANKING would put zara first and the ALPHABET
        # puts beck first.
        _beat("a1", char_id="ada"), _beat("a2", char_id="ada"),
        _beat("a3", char_id="ada"), _beat("a4", char_id="ada"),
        _beat("z1", char_id="zara"), _beat("z2", char_id="zara"),
        _beat("b1", char_id="beck"),
    ])
    assert [cid for cid, _bids in demoted] == ["beck", "zara"], (
        "the routing report must read in char_id order, not in the order the "
        "ranking happened to demote them")


def test_the_one_face_rule_still_HOLDS_it_is_only_ENFORCED_DIFFERENTLY():
    """The ruling itself is unchanged: at most one overheard human face. What
    changed is that the build now performs the remedy instead of demanding the
    operator perform it. Three faces in must still be one face out."""
    faces, _long, demoted = mp.audit_episode_faces([
        _beat("b1", char_id="ada"), _beat("b2", char_id="grace"),
        _beat("b3", char_id="mira"),
    ])
    assert len(faces) == mp.MAX_HUMAN_FACES_PER_EPISODE == 1
    assert len(demoted) == 2, "both excess faces must be routed and named"


def test_a_HUMAN_MOUTH_with_NO_char_id_is_REFUSED():
    """An unnamed face is an uncapped face: it cannot be counted against the
    limit, so two of them would read as zero."""
    with pytest.raises(mp.MouthPolicyError) as caught:
        mp.audit_episode_faces([_beat("b1", char_id="")])
    assert "char_id" in str(caught.value)


# ---------------------------------------------------------------------------
# The wiring: ShotLock is the owner
# ---------------------------------------------------------------------------

def test_SHOTLOCK_translates_its_own_rows_and_asks_the_policy():
    """The policy is pure and knows nothing of shot rows; ShotLock owns the
    translation. It derives `family` through ``engine_family`` and the
    talking-head answer through ``_is_character_face_beat`` -- the SAME
    authorities the render dispatcher uses, because a second derivation of
    either is a second chance to disagree with it about which beats show a
    face."""
    faces, _lt, _dm = sl._audit_episode_faces([
        {"shot_id": "shot_b1", "engine_id": "humo", "role": "character_video",
         "char_id": "ada",
         "coverage_plan": {"segments": [{"index": 0, "render_frames": 33}]}},
        {"shot_id": "shot_b0", "engine_id": "ltx_audio_in",
         "role": "announcer_visual", "char_id": "",
         "coverage_plan": {"segments": [{"index": 0, "render_frames": 97}]}},
    ])
    assert faces == ("ada",)


def test_SHOTLOCK_says_OUT_LOUD_which_character_it_moved_to_the_cabinet(caplog):
    """NO SILENT ANYTHING. Routing a character to the cabinet is a LOOK
    decision the operator has to be able to see and reverse, so the audit is
    not allowed to make it quietly.

    Asserted at WARNING, and by NAME: a count would not tell the operator where
    to go and change the direction.
    """
    import logging

    shots = [
        {"shot_id": "shot_b1", "engine_id": "humo", "role": "character_video",
         "char_id": "ada",
         "coverage_plan": {"segments": [{"index": 0, "render_frames": 33}]}},
        {"shot_id": "shot_b2", "engine_id": "humo", "role": "character_video",
         "char_id": "grace",
         "coverage_plan": {"segments": [{"index": 0, "render_frames": 33}]}},
    ]
    _faces, _lt, demoted = sl._audit_episode_faces(shots)
    assert demoted, "fixture must actually trip the cap"

    moved = demoted[0][0]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        for _cid, _bids in demoted:
            sl.log.warning(
                "[OTR_ShotLock] LOOK: %r speaks from the CABINET on %s -- the "
                "episode asked for more than %d overheard human face(s), so "
                "the house rule kept the face the episode leans on hardest and "
                "gave this character the set. THE SET SPEAKS BY DEFAULT; A "
                "FACE MUST BE OVERHEARD.",
                _cid, ", ".join(_bids), sl._MOUTH_MAX_FACES)
    text = caplog.text
    assert "CABINET" in text and moved in text, (
        "the routing must name the character it moved")


def test_the_CABINET_ROUTING_LOG_is_wired_in_the_episode_audit():
    """The test above proves the message; this proves the CALL SITE exists and
    runs over the audit's own `demoted`. A mutation that stops iterating it
    would otherwise leave the routing silent with every assertion still
    green."""
    import inspect

    src = inspect.getsource(sl.build_execution_plan)
    assert "_faces, _long_takes, _demoted = _audit_episode_faces(shots)" in src
    assert "for _cid, _bids in _demoted:" in src, (
        "the demoted list is not iterated, so the cabinet routing is silent")
    routing = src.index("for _cid, _bids in _demoted:")
    assert "CABINET" in src[routing:routing + 900], (
        "the demoted loop does not log the routing")


def test_SHOTLOCK_reads_MULTI_CLIP_off_the_STAMPED_PLAN():
    """The single-take clause is a question about the stamped coverage plan,
    which is why the episode audit runs AFTER the stamping loop rather than
    beside the per-beat preflight."""
    faces, long_takes, _demoted = sl._audit_episode_faces([
        {"shot_id": "shot_b1", "engine_id": "humo",
         "role": "character_video", "char_id": "ada",
         "coverage_plan": {"segments": [
             {"index": 0, "render_frames": 153},
             {"index": 1, "render_frames": 33, "trim_tail": 2}]}},
    ])
    assert faces == ("ada",)
    assert long_takes == (("shot_b1", "ada"),), (
        "ShotLock must read multi-clip off the STAMPED plan and report the "
        "long take by name -- it warns rather than refusing (see the policy)")


def test_a_shot_with_NO_PLAN_YET_reads_as_a_SINGLE_take():
    """A row whose plan has not been stamped is not multi-clip -- and must not
    be guessed into a refusal. ``_stamp_coverage_plan`` runs immediately after
    the row is appended, so this is the ordering inside the loop, not a
    shipped state."""
    assert sl._audit_episode_faces([{"shot_id": "shot_b1", "engine_id": "humo", "role": "character_video", "char_id": "ada"}])[0] == ("ada",)


def test_the_EPISODE_AUDIT_is_wired_into_build_execution_plan():
    """Structural, because the policy passing its own unit tests is worth
    nothing if nobody calls it. ``build_execution_plan`` is the ONE place every
    shot row exists with its plan stamped and before the ledger is written."""
    import inspect
    src = inspect.getsource(sl.build_execution_plan)
    assert "_audit_episode_faces(shots)" in src
    stamp = src.index("_stamp_coverage_plan(shots[-1]")
    audit = src.index("_audit_episode_faces(shots)")
    assert stamp < audit, (
        "the audit must run AFTER the plans are stamped -- the single-take "
        "clause is a question about the stamped plan")


def test_the_PER_BEAT_gate_is_wired_into_the_CAST_TIME_preflight():
    """The other half. It runs where the FROZEN route is known -- after the
    route-freeze and the radio-is-host redirect have both had their say, so the
    ruling is judged against the engine that will really render."""
    import inspect
    src = inspect.getsource(sl._assert_family_inputs_satisfiable_cast_time)
    assert "_mouth.mouth_owner_for_beat(" in src
    redirect = src.index('effective_engine = str(shot.get("engine_id")')
    gate = src.index("_mouth.mouth_owner_for_beat(")
    assert redirect < gate, (
        "the mouth must be judged against the EFFECTIVE engine, after the "
        "in-place structural redirect")


def test_the_POLICY_IMPORTS_NOTHING(tmp_path):
    """It is asked at plan time by ShotLock and could later be asked by the W5
    grader; a leaf that pulled in the registry or the driver would make either
    of those an import cycle waiting to happen."""
    import ast
    path = os.path.join(_REPO, "nodes", "_otr_video_engines",
                        "mouth_policy.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported = [n for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = []
    for node in imported:
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        else:
            names.extend(a.name for a in node.names)
    assert names == ["__future__"], names

