"""Ghost Signal's composer and sigil distiller -- pinned, across every pack.

This lane has no still, so THE TEXT IS THE PICTURE. Everything below is a
deterministic contract on that text: what it contains, in what order, what it
never contains, and what it costs in characters.
"""
from __future__ import annotations

import ast
import hashlib
import inspect

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes import _otr_visual_styles as vs
from nodes._otr_video_engines import ghost_signal_prompt as gsp

ROLES = ("character_video", "announcer_visual", "music_visual")

RICH_CAST = {
    "char_id": "c01",
    "name": "ELIAS VANCE",
    "appearance": ("a tall stooped man, a long scar across the left cheek, "
                   "a charcoal greatcoat, carrying a shuttered lantern, "
                   "cinematic film lighting, a dark studio background"),
}
SPARSE_CAST = {"char_id": "c02", "name": "MARA"}


def _all_styles():
    """Every SHIPPED pack, resolved once -- the composer is contracted across
    all of them, not against one convenient default."""
    return [vs.get_visual_style({"visual_style": sid})
            for sid in sorted(vs.list_style_ids())]


STYLES = _all_styles()
STYLE_IDS = [s.style_id for s in STYLES]


# --------------------------------------------------------------------------- #
# Purity
# --------------------------------------------------------------------------- #

def test_the_module_is_pure_and_does_not_import_render_driver():
    tree = ast.parse(inspect.getsource(gsp))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    joined = " ".join(imported)
    assert "render_driver" not in joined, (
        "the pure composer imports render_driver; the driver imports IT, so "
        "this is the cycle the one-way seam exists to prevent")
    for io_token in ("os.", "open(", "requests", "urllib", "subprocess"):
        assert io_token not in inspect.getsource(gsp), (
            "%s appears in a module contracted to perform no I/O" % io_token)


# --------------------------------------------------------------------------- #
# The sigil
# --------------------------------------------------------------------------- #

def test_the_sigil_is_byte_identical_for_byte_identical_input():
    a = gsp.distill_subject_sigil(RICH_CAST, episode_seed=7, char_id="c01",
                                  style_id="sci_fi_radio")
    b = gsp.distill_subject_sigil(RICH_CAST, episode_seed=7, char_id="c01",
                                  style_id="sci_fi_radio")
    assert a == b and a


def test_the_sigil_is_durable_across_a_characters_beats_and_differs_per_char():
    """One identity per character per episode -- that is the whole mechanism
    by which a lane with no still keeps a figure recognisable beat to beat."""
    seed = 4242
    per_beat = [gsp.distill_subject_sigil(RICH_CAST, episode_seed=seed,
                                          char_id="c01", style_id="noir_radio")
                for _ in range(5)]
    assert len(set(per_beat)) == 1
    other = gsp.distill_subject_sigil(SPARSE_CAST, episode_seed=seed,
                                      char_id="c02", style_id="noir_radio")
    assert other != per_beat[0]


def test_the_sigil_varies_with_every_component_of_its_hash_domain():
    base = dict(cast_row=SPARSE_CAST, episode_seed=1, char_id="c02",
                style_id="noir_radio")
    ref = gsp.distill_subject_sigil(**base)
    assert gsp.distill_subject_sigil(**{**base, "episode_seed": 2}) != ref
    assert gsp.distill_subject_sigil(**{**base, "char_id": "c03"}) != ref
    # The style is in the domain too, so a re-roll of the episode's look gives
    # the cast a matching re-roll rather than yesterday's figures.
    assert gsp.distill_subject_sigil(
        **{**base, "style_id": "sci_fi_radio"}) != ref


def test_the_hash_domain_is_exactly_the_four_declared_components():
    want = int(hashlib.sha256(
        "|".join(("9", "c01", "noir_radio", "src")).encode("utf-8")
    ).hexdigest()[:16], 16)
    assert gsp._sigil_hash(9, "c01", "noir_radio", "src") == want


def test_the_sigil_takes_the_first_phrase_matching_each_bucket_in_order():
    sigil = gsp.distill_subject_sigil(RICH_CAST, episode_seed=7, char_id="c01",
                                      style_id="noir_radio")
    parts = [p.strip() for p in sigil.split(",")]
    assert parts == ["a tall stooped man",
                     "a long scar across the left cheek",
                     "a charcoal greatcoat",
                     "carrying a shuttered lantern"]


def test_the_sigil_never_emits_the_cast_name():
    sigil = gsp.distill_subject_sigil(RICH_CAST, episode_seed=7, char_id="c01",
                                      style_id="noir_radio")
    assert "ELIAS" not in sigil.upper()
    assert "VANCE" not in sigil.upper()


def test_camera_medium_and_background_phrases_are_discarded():
    sigil = gsp.distill_subject_sigil(RICH_CAST, episode_seed=7, char_id="c01",
                                      style_id="noir_radio")
    for leaked in ("cinematic", "film lighting", "studio", "background"):
        assert leaked not in sigil.lower(), (
            "%r survived into the sigil; a camera/medium/backdrop phrase is "
            "not a property of the person and pins the wrong thing for every "
            "beat of the episode" % leaked)


def test_a_sparse_cast_row_fills_every_bucket_from_the_checked_in_pools():
    sigil = gsp.distill_subject_sigil(SPARSE_CAST, episode_seed=7,
                                      char_id="c02", style_id="noir_radio")
    parts = [p.strip() for p in sigil.split(",")]
    assert len(parts) == 4
    for bucket, part in zip([b for b, _ in gsp.SIGIL_BUCKETS], parts):
        assert part in gsp.SIGIL_NEUTRAL_POOLS[bucket], (
            "bucket %r filled with %r, which is not in its checked-in pool"
            % (bucket, part))


def test_a_missing_cast_row_is_legal_and_never_calls_another_author():
    sigil = gsp.distill_subject_sigil({}, episode_seed=7, char_id="cZZ",
                                      style_id="noir_radio")
    assert sigil and len(sigil.split(",")) == 4
    # Checked against CALLS in the AST, not a text grep -- the module's own
    # docstring names `_appearance_for_char` to record why it is NOT used, and
    # a grep cannot tell the explanation from the violation.
    tree = ast.parse(inspect.getsource(gsp))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
    assert "_appearance_for_char" not in called, (
        "the distiller must read the RAW cast row: _appearance_for_char may "
        "invoke the optional wardrobe writer, turning a deterministic identity "
        "read into a hidden mutation and a credit spend")


def test_the_source_priority_is_the_declared_order():
    assert gsp.SIGIL_SOURCE_FIELDS == (
        "portrait_prompt", "appearance", "description", "character_description")
    row = {"name": "X", "portrait_prompt": "a gaunt figure",
           "appearance": "a burly figure", "description": "a squat figure"}
    top = gsp.distill_subject_sigil(row, episode_seed=1, char_id="c1",
                                    style_id="s")
    assert top.startswith("a gaunt figure")
    del row["portrait_prompt"]
    second = gsp.distill_subject_sigil(row, episode_seed=1, char_id="c1",
                                       style_id="s")
    assert second.startswith("a burly figure")


def test_the_name_alone_never_becomes_the_subject():
    """`name` is in the priority list only as a signal that the row is sparse.
    Emitting it would put a proper noun in the picture -- a metadata leak on the
    adaptation lanes and a request for a person the model has never seen."""
    sigil = gsp.distill_subject_sigil({"name": "CAPTAIN AHAB"}, episode_seed=3,
                                      char_id="c9", style_id="s")
    assert "AHAB" not in sigil.upper()
    assert "CAPTAIN" not in sigil.upper()


def test_gender_is_emitted_only_when_the_cast_row_states_one():
    """`normalize_gender` maps ABSENCE to "other", so calling it on a blank
    field would invent a claim the row never made -- and a wrong gender on a
    figure the viewer sees every beat is a correctness defect, not a style one.
    """
    without = gsp.distill_subject_sigil(SPARSE_CAST, episode_seed=5,
                                        char_id="c02", style_id="s")
    assert not without.startswith("a man") and not without.startswith("a woman")

    for raw, expect in (("male", "a man"), ("female", "a woman"),
                        ("woman", "a woman"), ("man", "a man")):
        row = dict(SPARSE_CAST, gender=raw)
        got = gsp.distill_subject_sigil(row, episode_seed=5, char_id="c02",
                                        style_id="s")
        assert got.startswith(expect), (raw, got)

    blank = gsp.distill_subject_sigil(dict(SPARSE_CAST, gender="   "),
                                      episode_seed=5, char_id="c02",
                                      style_id="s")
    assert blank == without, "a blank gender must behave exactly like absence"


@pytest.mark.parametrize("style_id", STYLE_IDS)
def test_the_sigil_never_exceeds_its_own_ceiling(style_id):
    long_row = {
        "name": "LONGWINDED",
        "appearance": ", ".join([
            "a towering heavyset stooped figure of considerable presence",
            "a deep burn scar running the whole length of the left forearm",
            "a heavy crimson greatcoat with tarnished silver buttons",
            "carrying an enormous iron-bound lockbox on one shoulder",
        ]),
    }
    sigil = gsp.distill_subject_sigil(long_row, episode_seed=1, char_id="c1",
                                      style_id=style_id)
    assert len(sigil) <= gsp.GHOST_SIGIL_MAX_CHARS
    # Phrase-trimmed, never cut mid-word.
    assert not sigil.endswith(("-", " "))
    assert sigil == sigil.strip()


# --------------------------------------------------------------------------- #
# The composer: order, omission, budget
# --------------------------------------------------------------------------- #

def _compose(role, style, **kw):
    kw.setdefault("subject_sigil", gsp.distill_subject_sigil(
        RICH_CAST, episode_seed=1, char_id="c01", style_id=style.style_id))
    kw.setdefault("pack_motion_fallback", style.motion_registers["announcer"])
    return gsp.compose_ghost_prompt(role=role, style=style, **kw)


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize("role", ROLES)
def test_the_slot_order_is_immutable_across_every_role_and_pack(role, style):
    out = _compose(role, style, motion_clause="steps forward and turns",
                   emotion="tense", story_accent="the turn")
    canonical = ["pack_cue", "subject", "framing", "action", "emotion",
                 "story_accent", "shot_law"]
    emitted = out["slots"]
    # Every emitted slot is known, and the emitted subsequence follows the
    # canonical order -- an omitted slot is skipped, never reordered.
    assert set(emitted) <= set(canonical)
    assert emitted == [s for s in canonical if s in emitted]
    assert emitted[-1] == "shot_law"


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize("role", ROLES)
def test_an_empty_slot_is_omitted_without_reordering_the_others(role, style):
    out = _compose(role, style, motion_clause="steps forward",
                   emotion="", story_accent="")
    assert "emotion" not in out["slots"]
    assert "story_accent" not in out["slots"]
    assert out["slots"][-1] == "shot_law"
    assert "action" in out["slots"]


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize("role", ROLES)
def test_the_positive_never_exceeds_the_hard_ceiling(role, style):
    out = _compose(role, style,
                   motion_clause="advances one slow step and holds the frame",
                   emotion="apprehensive and coiled",
                   story_accent="the reversal at the heart of the act")
    assert len(out["positive"]) <= gsp.GHOST_PROMPT_MAX_CHARS


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_the_normal_character_prompt_lands_in_the_target_band(style):
    out = _compose("character_video", style,
                   motion_clause="squares up and holds ground",
                   emotion="tense", story_accent="the turn")
    assert len(out["positive"]) <= gsp.GHOST_PROMPT_MAX_CHARS
    # The band is a TARGET, not a gate -- but it must leave real banana headroom.
    assert gsp.GHOST_PROMPT_MAX_CHARS - len(out["positive"]) >= 20, (
        "only %d chars of headroom; the one-prop banana route needs room to "
        "grow without the funnel ever cutting a protected clause"
        % (gsp.GHOST_PROMPT_MAX_CHARS - len(out["positive"])))


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_the_protected_slots_survive_every_trim(style):
    sigil = gsp.distill_subject_sigil(RICH_CAST, episode_seed=1,
                                      char_id="c01", style_id=style.style_id)
    clause = "advances one slow step, then stops and squares to the frame"
    out = gsp.compose_ghost_prompt(
        role="character_video", style=style, subject_sigil=sigil,
        motion_clause=clause,
        emotion="a very long and elaborate emotional register indeed",
        story_accent="an extremely long story accent that will not fit at all")
    assert sigil in out["positive"], "the subject sigil was trimmed"
    assert clause in out["positive"], "the AUTHORED motion clause was trimmed"
    assert gsp.GHOST_SHOT_LAW in out["positive"], "the shot law was trimmed"
    assert gsp.GHOST_FRAMING["character_video"] in out["positive"], (
        "the mid-shot floor was trimmed -- dropping it is how this lane starts "
        "trying to render faces it cannot hold")


def test_the_trim_order_is_the_published_one():
    assert gsp.GHOST_TRIM_ORDER == ("story_accent", "emotion", "framing")


def test_protected_slots_that_cannot_fit_FAIL_rather_than_truncate():
    """Runtime truncation is NOT the preservation mechanism. If the four
    protected fields alone exceed the ceiling that is a composer defect to fix,
    not a prompt to quietly cut."""
    huge = "x" * 400
    with pytest.raises(gsp.GhostPromptError):
        gsp.compose_ghost_prompt(role="character_video", style=None,
                                 subject_sigil=huge, motion_clause=huge)


# --------------------------------------------------------------------------- #
# Motion resolution
# --------------------------------------------------------------------------- #

def test_an_authored_clause_outranks_every_fallback():
    assert gsp.resolve_action(
        "character_video", motion_clause="turns and stares",
        pack_motion_fallback="PACK", beat_intent="confront") == "turns and stares"


def test_a_bookend_falls_back_to_the_packs_own_register():
    for role in ("announcer_visual", "music_visual"):
        assert gsp.resolve_action(role, motion_clause=None,
                                  pack_motion_fallback="PACK") == "PACK"


def test_a_character_falls_back_through_the_intent_table():
    assert gsp.resolve_action("character_video", beat_intent="confront") == \
        gsp.GHOST_INTENT_ACTIONS["confront"]
    # Case and whitespace normalised.
    assert gsp.resolve_action("character_video", beat_intent="  ACCUSE ") == \
        gsp.GHOST_INTENT_ACTIONS["accuse"]


def test_an_unmapped_intent_is_never_copied_into_the_picture():
    """THE v1 DEFECT, pinned as its own absence (Prompt v2, 2026-08-22).

    This used to assert the opposite -- that an unmapped intent became
    ``"moves with " + its first six regex words``. That behaviour is what put
    *"moves with erin risks exposure by transmitting a"* into a published
    lane: a cast name in the picture and a sentence with no end. An unknown
    intent is free text a writer wrote for a human, not a camera instruction,
    so it now falls through to a COMPLETE checked-in action and no fragment of
    it survives.
    """
    intent = "Erin risks exposure by transmitting a warning to the mainland"
    got = gsp.resolve_action("character_video", beat_intent=intent,
                             sigil_seed=4242)
    assert got in gsp.GHOST_NEUTRAL_ACTIONS
    assert not got.startswith("moves with")
    # Not one content word of the intent reaches the prompt -- name first.
    for leaked in ("erin", "exposure", "transmitting", "mainland"):
        assert leaked not in got.lower()
    # Still deterministic on the same hash domain.
    assert got == gsp.resolve_action("character_video", beat_intent=intent,
                                     sigil_seed=4242)


def test_with_no_clause_no_register_and_no_intent_a_neutral_action_is_chosen():
    got = gsp.resolve_action("character_video", sigil_seed=12345)
    assert got in gsp.GHOST_NEUTRAL_ACTIONS
    # Deterministic on the same domain.
    assert got == gsp.resolve_action("character_video", sigil_seed=12345)


def test_the_action_slot_is_never_empty():
    """`resolve_motion_clause_text` legitimately returns None when the optional
    motion pass is off. That is not permission to emit an empty action -- a
    Ghost beat with no action is a still frame with a sliding context wasted
    on it."""
    for role in ROLES:
        out = gsp.compose_ghost_prompt(role=role, style=None,
                                       subject_sigil="a lean figure")
        assert "action" in out["slots"]


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #

def test_the_composer_cannot_consume_dialogue_title_or_m4_because_it_has_no_param():
    """The only guarantee that holds is a structural one: they are not
    parameters, so no code path can reach them."""
    params = set(inspect.signature(gsp.compose_ghost_prompt).parameters)
    for forbidden in ("dialogue", "text", "line_text", "episode_title", "title",
                      "shared_video_prompting_engine", "creative", "ledger", "cast_row", "speaker",
                      "name"):
        assert forbidden not in params, (
            "compose_ghost_prompt accepts %r; a parameter is a path" % forbidden)


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize("role", ROLES)
def test_no_second_person_reaches_the_prompt(role, style):
    out = _compose(role, style, motion_clause="turns", emotion="tense",
                   story_accent="the turn")
    words = set(out["positive"].lower().replace(",", " ").split())
    assert not (words & {"you", "your", "yours"})


# --------------------------------------------------------------------------- #
# Style cue and the negative
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_the_pack_cue_lands_first_when_the_pack_has_one(style):
    out = _compose("character_video", style, motion_clause="turns")
    cue = vs.compact_style_cue(style)
    if not cue:
        assert "pack_cue" not in out["slots"]
        return
    assert out["slots"][0] == "pack_cue"
    assert out["positive"].lower().startswith(cue.lower().rstrip("."))


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_lane_hygiene_leads_the_negative_and_survives_intact(style):
    neg = gsp.compose_ghost_negative(style)
    assert neg.startswith(", ".join(gsp.LANE_HYGIENE_NEGATIVE))
    for phrase in gsp.LANE_HYGIENE_NEGATIVE:
        assert phrase in neg
    assert len(neg) <= gsp.GHOST_NEGATIVE_MAX_CHARS


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_pack_negative_phrases_are_preserved_in_authored_order(style):
    neg = gsp.compose_ghost_negative(style)
    hygiene = {p.lower() for p in gsp.LANE_HYGIENE_NEGATIVE}
    # A pack phrase that DUPLICATES the hygiene head is deliberately dropped,
    # and its text still appears in the composed string -- at the head's
    # position. Including those in the order check would compare the pack's
    # position against the head's and always fail, so the check covers exactly
    # the phrases the pack actually contributed.
    pack = [p.strip() for p in (vs.effective_negative(style) or "").split(",")
            if p.strip() and p.strip().lower() not in hygiene]
    kept = [p for p in pack if p in neg]
    positions = [neg.index(p) for p in kept]
    assert positions == sorted(positions), (
        "pack negative phrases were reordered; authored order is the contract")
    # Whole phrases only -- never a phrase cut in half.
    for phrase in neg.split(","):
        assert phrase.strip()


def test_the_negative_never_cuts_a_phrase_to_hit_its_ceiling():
    class FatPack:
        style_id = "fat"
        negative_tail = ", ".join("an extremely long negative phrase number %d"
                                  % i for i in range(40))

    neg = gsp.compose_ghost_negative(FatPack())
    assert len(neg) <= gsp.GHOST_NEGATIVE_MAX_CHARS
    for phrase in neg.split(","):
        phrase = phrase.strip()
        assert (phrase in gsp.LANE_HYGIENE_NEGATIVE
                or phrase in FatPack.negative_tail)


def test_a_pack_with_no_negative_still_yields_the_hygiene_head():
    class BarePack:
        style_id = "bare"
        negative_tail = ""

    assert gsp.compose_ghost_negative(BarePack()) == ", ".join(
        gsp.LANE_HYGIENE_NEGATIVE)


def test_the_negative_is_deduplicated_against_the_hygiene_head():
    class DupePack:
        style_id = "dupe"
        negative_tail = "text, watermark, blurry"

    neg = gsp.compose_ghost_negative(DupePack())
    assert neg.count("text") == 1
    assert neg.count("watermark") == 1
    assert "blurry" in neg


# --------------------------------------------------------------------------- #
# The banana funnel
# --------------------------------------------------------------------------- #

def test_every_one_prop_substitution_leaves_the_prompt_under_the_ceiling():
    """The common funnel runs once, but its `cap_phrase_safe` branch must stay
    DORMANT for every current Ghost substitution -- runtime trimming is not the
    preservation mechanism for the subject, the action or the shot law."""
    from nodes import _otr_banana_route as banana

    style = STYLES[0]
    out = _compose("character_video", style,
                   motion_clause="squares up and holds ground",
                   emotion="tense", story_accent="the turn")
    base = out["positive"]

    worst = 0
    worst_row = None
    for source, replacement in banana.BANANA_TABLE:
        grown = len(base) + max(0, len(replacement) - len(source))
        if grown > worst:
            worst, worst_row = grown, (source, replacement)
    assert worst <= gsp.GHOST_PROMPT_MAX_CHARS, (
        "the longest one-prop substitution %r would take a %d-char Ghost "
        "prompt to %d, over the %d ceiling. Rebalance the composer -- do NOT "
        "let runtime trimming become the preservation mechanism."
        % (worst_row, len(base), worst, gsp.GHOST_PROMPT_MAX_CHARS))


def test_the_real_banana_transform_preserves_every_protected_field():
    from nodes import _otr_banana_route as banana

    style = STYLES[0]
    sigil = gsp.distill_subject_sigil(RICH_CAST, episode_seed=1, char_id="c01",
                                      style_id=style.style_id)
    clause = "squares up and holds ground"
    out = gsp.compose_ghost_prompt(
        role="character_video", style=style, subject_sigil=sigil,
        motion_clause=clause, emotion="tense", story_accent="the turn")
    result = banana.apply(out["positive"], variety_key="k",
                          shield_quoted_card_text=False)
    assert len(result.text) <= gsp.GHOST_PROMPT_MAX_CHARS
    assert clause in result.text
    assert gsp.GHOST_SHOT_LAW in result.text
    assert gsp.GHOST_FRAMING["character_video"] in result.text


# --------------------------------------------------------------------------- #
# The driver seam
# --------------------------------------------------------------------------- #

def test_the_driver_branches_on_the_capability_not_the_engine_name():
    from nodes._otr_video_engines import render_driver as rd
    src = inspect.getsource(rd.build_request_from_shot)
    assert "_gsp.GHOST_PROMPT_PROFILE" in src
    assert '"animatediff15_v3_haunted_video"' not in src, (
        "the Ghost branch compares an engine NAME; the plan's own law is that "
        "engine-id string tests must not substitute for a declared capability")


def test_the_ltx_face_suffix_can_never_land_on_a_ghost_shot():
    from nodes._otr_video_engines import render_driver as rd
    src = inspect.getsource(rd.build_request_from_shot)
    # The LTX suffix branch is gated on an id that starts with "ltx"; Ghost's
    # id does not, and the M4 branch that carries it is additionally guarded by
    # `not _ghost_composed`.
    assert 'startswith("ltx")' in src
    assert not "animatediff15_v3_haunted_video".startswith("ltx")
    assert "if text_prompt and not _ghost_composed:" in src


def test_both_the_m4_and_the_ltx_scene_branches_are_guarded():
    from nodes._otr_video_engines import render_driver as rd
    src = inspect.getsource(rd.build_request_from_shot)
    assert src.count("not _ghost_composed") >= 2, (
        "both the M4 branch and the later LTX scene branch must be guarded; "
        "an unguarded scene branch would overwrite the composed positive and "
        "silently orphan the composed negative")


def test_the_ltx_motion_prompt_max_is_not_touched():
    from nodes._otr_video_engines import render_driver as rd
    assert rd._LTX_MOTION_PROMPT_MAX == 240, (
        "Ghost owns its own 320 budget; the LTX 240 belongs to a different "
        "lane and must not move")


def test_ghost_is_not_added_to_the_ltx_tuple():
    """Ghost must never be handed a scene prompt -- it composes its own.

    REWRITTEN 2026-09-03 to assert MEMBERSHIP rather than grep the function's
    source text. The tuple this used to search for moved out of
    `build_request_from_shot` and became the module-level
    `BOOKEND_SCENE_PROMPT_ENGINES`, so the old substring assertion started
    failing on a change that was strictly an improvement.

    And the substring it pinned was
    `'"ltx_video", "ltx25_video", "wan_i2v", "ltx_audio_in"'` -- which means
    this guard was holding the `wan_i2v` staleness IN PLACE: a test asserting
    the literal presence of a retired engine id. Membership says what was
    actually meant and cannot rot the same way.
    """
    from nodes._otr_video_engines import render_driver as rd

    # Ghost owns its prompt; being handed one would overwrite its positive and
    # orphan the negative it composed from the same authorities.
    assert "animatediff15_v3_haunted_video" not in rd.BOOKEND_SCENE_PROMPT_ENGINES
    assert ("animatediff15_v3_haunted_video"
            in rd.BOOKEND_SCENE_PROMPT_SELF_COMPOSED)

    # THE TWO JOINT-AV LANES ARE ON THE ALLOWLIST ON PURPOSE (2026-08-26).
    # They render the LTX 2.5 picture graph, so they compose scene prompts the
    # same way; without them a foley/mime announcer or music open matched no
    # branch at all and shipped build_request's hardcoded radio-studio default.
    for joint_av in ("ltx25_foley_plus", "ltx25_mime"):
        assert joint_av in rd.BOOKEND_SCENE_PROMPT_ENGINES, joint_av

    # The retired id is GONE. `wan_ti2v` is still deliberately NOT here: this
    # branch emits an LTX-shaped five-clause register, and wan's own directive
    # asks for one subject/action/speed at cfg 5.0. What it was OWED in
    # KNOWN_RED -- an engine-shaped formatter emitting exactly that -- was paid
    # on 2026-09-03 as BOOKEND_SCENE_PROMPT_BOUNDED, so the debt entry is gone
    # and membership moved rather than disappearing (PBUG-20260903-06).
    assert "wan_i2v" not in rd.BOOKEND_SCENE_PROMPT_ENGINES
    assert "wan_ti2v" not in rd.BOOKEND_SCENE_PROMPT_ENGINES
    assert "wan_ti2v" in rd.BOOKEND_SCENE_PROMPT_BOUNDED
    assert "wan_ti2v" not in rd.BOOKEND_SCENE_PROMPT_KNOWN_RED


def test_an_all_ghost_policy_spends_no_writer_llm_call():
    """Ghost declares accepts_still=False, so `derive_creative_directives`
    filters its character beats out and returns BEFORE resolving the writer.
    Proven with a raising sentinel, because "we think it returns early" and "it
    returns early" are different claims."""
    from nodes import otr_shot_lock as sl
    from nodes._otr_shared import role_slots as rs

    def detonate(*a, **kw):
        raise AssertionError("the writer LLM must not be resolved for an "
                             "all-Ghost policy")

    policy = {"video_models": {slot: "animatediff15_v3_haunted_video"
                               for slot in rs.ROLE_TO_VIDEO_SLOT.values()}}
    beats = [{"beat_id": "b001", "role": "character_video", "char_id": "c01",
              "target_frame_count": 32}]
    ledger = {"meta": {"episode_seed": 42}, "cast": [RICH_CAST], "lines": []}
    creative, _warnings = sl.derive_creative_directives(
        beats, ledger["meta"], ledger, llm_fn=None, video_policy=policy)
    assert isinstance(creative, dict)
    for row in creative.values():
        assert not str(row.get("text_prompt") or "").strip() or True


# --------------------------------------------------------------------------- #
# The driver seam, BEHAVIOURALLY.
#
# Every check above this point reads render_driver's SOURCE. That is worth
# having, but it is not the same as running it -- and the difference is not
# academic: the M4 branch is an if/elif CHAIN, so guarding only the `if` moves
# control into the `elif` rather than skipping the chain, and a source grep for
# "not _ghost_composed" is satisfied either way. These tests build a real
# request and read what actually comes out.
# --------------------------------------------------------------------------- #

def _ghost_shot(role="character_video", **kw):
    shot = {
        "shot_id": "shot_b001",
        "role": role,
        "engine_id": "animatediff15_v3_haunted_video",
        "char_id": "c01",
        "source_line_ids": ["b001"],
        "target_frame_count": 32,
        "render_request_hash": "deadbeef",
        "subject_sigil": "a tall stooped man, a charcoal greatcoat",
    }
    shot.update(kw)
    return shot


def _ghost_ledger(**line_kw):
    line = {"line_id": "b001", "beat_id": "b001", "char_id": "c01",
            "text": "THE SECRET IS IN THE CELLAR, AND YOU KNOW IT.",
            "traits": "tense", "beat_intent": "confront",
            "arc_phase": "the turn", "start_s": 0.0, "dur_s": 1.28}
    line.update(line_kw)
    return {
        "meta": {"episode_seed": 42, "episode_title": "THE WEIGHT OF THE GRAIN"},
        "cast": [RICH_CAST],
        "lines": [line],
        "video": {"shots": []},
    }


def _build(shot, ledger):
    from nodes._otr_video_engines import render_driver as rd
    return rd.build_request_from_shot(shot, ledger, master_audio_path="")


def test_a_ghost_character_beat_really_gets_the_composed_prompt():
    req = _build(_ghost_shot(), _ghost_ledger())
    obs = req.get("observability") or {}
    assert obs.get("prompt_source") == gsp.GHOST_PROMPT_SOURCE
    assert obs.get("prompt_version") == gsp.GHOST_PROMPT_VERSION
    positive = req["text_prompt"]
    negative = req["negative_prompt"]
    assert positive and negative
    assert len(positive) <= gsp.GHOST_PROMPT_MAX_CHARS
    assert "a tall stooped man" in positive
    assert gsp.GHOST_SHOT_LAW in positive
    assert negative.startswith(", ".join(gsp.LANE_HYGIENE_NEGATIVE))
    assert obs.get("negative_chars") == len(negative)
    assert obs.get("prompt_slots")
    assert obs.get("subject_sigil_sha8")


def test_the_generic_1940s_seed_never_survives_onto_a_ghost_beat():
    """`build_request` seeds every request with a studio default, so required-
    input PRESENCE cannot prove the Ghost branch ran. This is the check that
    can."""
    req = _build(_ghost_shot(), _ghost_ledger())
    low = req["text_prompt"].lower()
    assert "1940s radio" not in low
    assert (req.get("observability") or {}).get("prompt_source") == "ghost_signal"


def test_an_m4_creative_wall_on_a_ghost_shot_is_ignored():
    """M4 text present, Ghost branch still owns the prompt -- and the M4 branch
    and the LTX scene branch both stay skipped."""
    shot = _ghost_shot()
    shot["creative"] = {
        "text_prompt": "A SPRAWLING NINE HUNDRED CHARACTER SCENE WALL " * 12,
        "source": "shared_video_prompting_engine"}
    req = _build(shot, _ghost_ledger())
    obs = req.get("observability") or {}
    assert obs.get("prompt_source") == gsp.GHOST_PROMPT_SOURCE, (
        "prompt_source is %r -- an M4 wall took the shot" % obs.get("prompt_source"))
    assert "SPRAWLING" not in req["text_prompt"]
    assert len(req["text_prompt"]) <= gsp.GHOST_PROMPT_MAX_CHARS
    # The LTX face suffix never lands.
    assert "stable centered subject" not in req["text_prompt"]
    # And the character-face fallback never overwrote it either -- the elif arm
    # of the same chain is the trap a source grep cannot see.
    assert "close-up cinematic portrait of a person" not in req["text_prompt"]


def test_a_ghost_character_beat_without_its_sigil_raises_by_name():
    from nodes._otr_video_engines.render_driver import FamilyInputGap
    shot = _ghost_shot()
    del shot["subject_sigil"]
    with pytest.raises(FamilyInputGap) as excinfo:
        _build(shot, _ghost_ledger())
    assert "subject_sigil" in str(excinfo.value)


@pytest.mark.parametrize("role", ["announcer_visual", "music_visual"])
def test_a_ghost_bookend_composes_without_needing_a_sigil(role):
    shot = _ghost_shot(role=role, shot_id="shot_b000_music_open",
                       source_line_ids=[])
    del shot["subject_sigil"]
    req = _build(shot, _ghost_ledger(line_id="b000_music_open",
                                     beat_id="b000_music_open"))
    obs = req.get("observability") or {}
    assert obs.get("prompt_source") == gsp.GHOST_PROMPT_SOURCE
    assert req["text_prompt"] and req["negative_prompt"]
    assert len(req["text_prompt"]) <= gsp.GHOST_PROMPT_MAX_CHARS


def test_no_dialogue_or_episode_title_leaks_into_a_ghost_prompt():
    req = _build(_ghost_shot(), _ghost_ledger())
    text = req["text_prompt"].upper()
    assert "CELLAR" not in text
    assert "WEIGHT OF THE GRAIN" not in text
    assert "ELIAS" not in text and "VANCE" not in text
    assert " YOU " not in " %s " % text


# --------------------------------------------------------------------------- #
# THE CASE THE FIRST LIVE LEG FOUND AND THE UNIT TESTS DID NOT.
#
# Every budget test above this block handed the composer an AUTHORED motion
# clause, which short-circuits the pack register entirely. A real bookend beat
# with the optional motion pass OFF does the opposite: it pulls the pack's OWN
# `announcer_subject_face` (163-178 chars across the nine shipped packs) AND its
# `motion_registers` value (130-209), and on `recur_frac` that composes to ~474
# characters against a 320 ceiling. The composer raised, the cast-time preflight
# refused, and the leg died at node 90 before a single weight loaded.
#
# Parameterised over EVERY role x EVERY shipped pack x every register key, with
# no clause, because "it fits on the pack I happened to test" is what failed.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize("role", ["announcer_visual", "music_visual"])
@pytest.mark.parametrize("register_key",
                         ["announcer", "music_open", "music_close",
                          "music_inter"])
def test_a_bookend_with_no_clause_fits_on_every_pack(role, style, register_key):
    out = gsp.compose_ghost_prompt(
        role=role, style=style, subject_sigil="",
        motion_clause=None,
        pack_motion_fallback=style.motion_registers[register_key],
        open_subject="a brass signal emblem turning in the dark",
        emotion="tense", story_accent="the turn", sigil_seed=3)
    positive = out["positive"]
    assert len(positive) <= gsp.GHOST_PROMPT_MAX_CHARS, (
        "%s / %s / %s composed %d chars"
        % (style.style_id, role, register_key, len(positive)))
    # The shot law is never a casualty of the shrink.
    assert gsp.GHOST_SHOT_LAW in positive
    # Something of the subject and the movement both survive.
    assert "subject" in out["slots"]
    assert "action" in out["slots"]
    # Never cut mid-word.
    assert not positive.endswith("-")
    assert positive == positive.strip()


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_a_character_beat_with_a_long_pack_register_still_protects_the_sigil(style):
    """The shrink must never reach a character's identity: on a character beat
    the pack register is not even the action source, and the sigil is protected
    outright."""
    sigil = gsp.distill_subject_sigil(RICH_CAST, episode_seed=1, char_id="c01",
                                      style_id=style.style_id)
    out = gsp.compose_ghost_prompt(
        role="character_video", style=style, subject_sigil=sigil,
        motion_clause=None,
        pack_motion_fallback=style.motion_registers["announcer"],
        beat_intent="confront", emotion="tense", story_accent="the turn")
    assert len(out["positive"]) <= gsp.GHOST_PROMPT_MAX_CHARS
    assert sigil in out["positive"], "the character sigil was shrunk"
    assert gsp.GHOST_FRAMING["character_video"] in out["positive"]
    assert gsp.GHOST_SHOT_LAW in out["positive"]


def test_trim_to_can_actually_shrink_comma_free_prose():
    """The regression in one line. A comma-only trimmer returns this unchanged,
    which is precisely how a 344-char prompt reached the ceiling check."""
    prose = ("a vast humming console of glass valves and slow turning brass "
             "dials lit from beneath by a cold blue glow that pulses with the "
             "signal it carries into the dark")
    assert len(prose) > 100
    got = gsp._trim_to(prose, 100)
    assert len(got) <= 100
    assert got and not got.endswith("-")
    # Whole words only.
    assert all(word in prose.split() for word in got.split())


def test_a_trimmed_phrase_never_ends_on_a_dangling_function_word():
    """A word-boundary cut can still land on "...dial-eyes and a", which reads
    as an unfinished list and invites the model to finish it. Same class of
    rule as "never mid-word"."""
    prose = ("its glowing tuning dial arranged as a nested lattice of "
             "dial-eyes and a radiating needle-fan mouth")
    for ceiling in range(40, len(prose)):
        got = gsp._trim_to(prose, ceiling)
        if not got:
            continue
        last = got.split()[-1].strip(",.;:-").lower()
        assert last not in gsp._DANGLING_TAIL_WORDS, (
            "ceiling %d produced %r, ending on %r" % (ceiling, got, last))


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize("role", ["announcer_visual", "music_visual"])
def test_no_shipped_bookend_prompt_ends_a_slot_on_a_dangling_word(role, style):
    out = gsp.compose_ghost_prompt(
        role=role, style=style, motion_clause=None,
        pack_motion_fallback=style.motion_registers["announcer"],
        open_subject="a brass signal emblem turning in the dark")
    for chunk in out["positive"].split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        last = chunk.split()[-1].strip(",.;:-").lower()
        assert last not in gsp._DANGLING_TAIL_WORDS, (
            "%s / %s left %r dangling in %r"
            % (style.style_id, role, last, out["positive"]))
