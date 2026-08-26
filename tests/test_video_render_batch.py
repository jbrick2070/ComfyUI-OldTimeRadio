"""tests/test_video_render_batch.py -- the MOTION receipt may only bill an
engine for video that reached the timeline (2026-08-26).

WHY THIS FILE EXISTS. A model refusal is meant to be a SANCTIONED GAP: the
image dispatcher records the refusal and lets the episode continue rather than
destroying eight finished beats over one declined card, and
``OTR_SilentComposite`` floors the picture for a manifest row without
``exists``. The operator ruled that an episode where every still is refused
still PUBLISHES -- full audio, correct timing, a floored picture.

**THAT PATH IS NOT COMPLETE YET, AND THESE TESTS DO NOT CLAIM IT IS.** Both
ends exist -- the dispatcher continues, the composite floors -- but nothing
between them yet MINTS a gap row: on the canonical ``still_flat`` path the
refused object is dropped from the required-target receipt and
``validate_and_repair_still_spine`` raises before any manifest exists. So every
fixture below is HAND-BUILT, and deliberately so: it pins what the accounting
must do WHEN a gap row arrives, not that one can arrive today. Read as a claim
about the live pipeline these would be false; read as the contract the control
path must satisfy, they are the reason it cannot land on wrong accounting.

``meta.render_engines`` is the durable receipt ``OTR_CreditsRoll`` reads to
state WHICH ENGINE RENDERED EACH BEAT. It was projecting every manifest row
into ``by_role`` / ``per_clip`` / the by-engine roll-up without ever consulting
``exists``, so once sanctioned gaps became survivable the card would credit
engines for motion they never delivered -- and an all-gap episode would
advertise a full slate of rendered video while containing none.

The two halves this file pins, because fixing either one alone produces a
different untruth: delivered accounting counts ONLY delivered rows, and the
gaps are COUNTED rather than quietly dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.otr_video_render_batch import (  # noqa: E402
    _build_render_engines_payload, _clip_delivered_motion)
from nodes import otr_credits_roll as cr  # noqa: E402


#: Every key the payload promises on EVERY return. A reader that has to test
#: for a key's presence before reading it cannot tell "no gaps" from "an older
#: ledger that never looked", which is the distinction the whole receipt is for.
_PAYLOAD_KEYS = {"histogram", "video_revision", "by_role", "vram_peak_mb",
                 "per_clip", "by_engine", "sanctioned_gap_count",
                 "sanctioned_gap_shot_ids"}


def _delivered(shot_id, engine_id, role="character_visual", **over):
    """A manifest row for a beat that really rendered -- the shape
    ``build_clip_manifest`` writes when the clip is on disk."""
    row = {"shot_id": shot_id, "role": role, "engine_id": engine_id,
           "path": "X:\\eps\\%s.mp4" % shot_id, "exists": True}
    row.update(over)
    return row


def _gap(shot_id, engine_id, role="character_visual", **over):
    """A SANCTIONED GAP row: the beat was planned and its engine named, and
    nothing was rendered. ``build_clip_manifest`` stamps exactly this -- the
    planned ``engine_id`` survives on the row, which is precisely why the
    accounting cannot read the row's engine as evidence of delivery."""
    row = {"shot_id": shot_id, "role": role, "engine_id": engine_id,
           "path": "", "exists": False}
    row.update(over)
    return row


def _payload(*clips, **kw):
    manifest = {"clips": list(clips)}
    manifest.update(kw)
    return _build_render_engines_payload(manifest, None)


# --------------------------------------------------------------------------- #
# The predicate, pinned directly rather than by inference
# --------------------------------------------------------------------------- #
def test_delivery_is_decided_by_exists_and_by_nothing_else():
    assert _clip_delivered_motion(_delivered("s1", "ltx_8gb")) is True
    assert _clip_delivered_motion(_gap("s1", "ltx_8gb")) is False
    # ...and a row that names an engine, a recipe and a peak is STILL a gap if
    # it did not render. A rich receipt on a beat that produced no frame is the
    # most convincing wrong answer available here.
    assert _clip_delivered_motion(
        _gap("s1", "wan_i2v", recipe="WAN_I2V_v1", vram_peak_mb=15500)) is False


# --------------------------------------------------------------------------- #
# THE ALL-GAP EPISODE -- publishable, and it must stay publishable
# --------------------------------------------------------------------------- #
def _all_gap_payload():
    return _payload(
        _gap("shot_music_opening_001", "still_flat", role="music_visual"),
        _gap("shot_body_001", "ltx_8gb"),
        _gap("shot_music_closing_001", "still_flat", role="music_visual"),
        engine_histogram={}, video_revision=4)


def test_an_all_gap_episode_credits_no_engine_with_motion():
    p = _all_gap_payload()
    assert p["by_role"] == {}
    assert p["by_engine"] == {}
    assert p["per_clip"] == []
    # the histogram was already gap-correct upstream (build_clip_manifest
    # counts a row only ``if exists``); it must arrive unchanged, not be
    # recomputed here into a second opinion.
    assert p["histogram"] == {}


def test_an_all_gap_episode_still_reports_its_gaps():
    p = _all_gap_payload()
    assert p["sanctioned_gap_count"] == 3
    assert p["sanctioned_gap_shot_ids"] == [
        "shot_music_opening_001", "shot_body_001", "shot_music_closing_001"]


def test_an_all_gap_payload_is_not_empty_so_the_credits_roll_cannot_raise():
    """THE TRAP THIS FIX HAD TO AVOID. ``OTR_CreditsRoll._require`` rejects
    None / "" / {} / [], so a payload that went empty when nothing rendered
    would turn a publishable degraded episode into a hard failure at mux time
    -- the exact outcome the sanctioned gap exists to prevent."""
    p = _all_gap_payload()
    assert set(p) == _PAYLOAD_KEYS
    assert cr._require({"render_engines": p}, "render_engines", "meta") is p


def test_an_all_gap_payload_draws_no_video_rows_rather_than_wrong_ones():
    """The reader's half: with no delivered engine there is nothing to put in
    the MODELS.VIDEO block, and it must reach that answer without crashing."""
    assert cr._video_role_rows(_all_gap_payload()) == []


# --------------------------------------------------------------------------- #
# THE MIXED EPISODE -- the shape a single refusal actually produces
# --------------------------------------------------------------------------- #
def _mixed_payload():
    return _payload(
        _gap("shot_music_opening_001", "still_flat", role="music_visual"),
        _delivered("shot_body_001", "ltx_8gb", recipe="RECIPE_LTX8_I2V_v2",
                   quant="Q8_0", vram_peak_mb=8278),
        _delivered("shot_body_002", "ltx_8gb", recipe="RECIPE_LTX8_I2V_v2",
                   quant="Q8_0", vram_peak_mb=16086),
        _gap("shot_music_closing_001", "still_flat", role="music_visual"),
        engine_histogram={"ltx_8gb": 2}, video_revision=4)


def test_a_mixed_episode_bills_only_the_beats_that_rendered():
    p = _mixed_payload()
    assert p["by_role"] == {"character_visual": {"ltx_8gb": 2}}
    assert [c["shot_id"] for c in p["per_clip"]] == [
        "shot_body_001", "shot_body_002"]
    assert set(p["by_engine"]) == {"ltx_8gb"}
    assert p["by_engine"]["ltx_8gb"]["clip_count"] == 2
    # the refused music beats must not have minted a still_flat MOTION credit
    assert "still_flat" not in p["by_engine"]
    assert "music_visual" not in p["by_role"]


def test_a_mixed_episode_names_the_beats_it_did_not_render():
    p = _mixed_payload()
    assert p["sanctioned_gap_count"] == 2
    assert p["sanctioned_gap_shot_ids"] == [
        "shot_music_opening_001", "shot_music_closing_001"]


def test_the_two_halves_add_back_up_to_the_episode():
    """What makes the gap list a receipt rather than a footnote: no beat is
    lost between the delivered rows and the gap rows."""
    p = _mixed_payload()
    assert len(p["per_clip"]) + p["sanctioned_gap_count"] == 4


def test_the_credits_card_names_only_the_engine_that_rendered():
    p = _mixed_payload()
    assert cr._video_role_rows(p) == [("character_visual", "ltx_8gb", "")]
    assert cr._recipe_suffix(p, "ltx_8gb") == "RECIPE_LTX8_I2V_v2 · Q8_0"


# --------------------------------------------------------------------------- #
# THE ROLL-UP, which is where a gap row does its quietest damage
# --------------------------------------------------------------------------- #
def test_a_gap_row_cannot_contaminate_its_engines_rollup():
    """The sharpest case, and the one a row-count assertion would miss: the
    gap shares an engine with a beat that DID render. Counted, it drags the
    engine's recipe to None and its clip_count to 2 -- so the card would print
    "mixed recipe" for an engine that ran exactly one recipe once, and the
    receipt would claim two rendered beats where one exists."""
    p = _payload(
        _delivered("s1", "ltx_8gb", recipe="RECIPE_LTX8_I2V_v2", quant="Q8_0"),
        _gap("s2", "ltx_8gb"),
    )
    row = p["by_engine"]["ltx_8gb"]
    assert row["clip_count"] == 1
    assert row["recipe"] == "RECIPE_LTX8_I2V_v2"
    assert row["varied"] == []
    assert cr._recipe_suffix(p, "ltx_8gb") == "RECIPE_LTX8_I2V_v2 · Q8_0"


def test_a_gap_row_cannot_donate_a_vram_peak_it_never_measured():
    """``vram_peak_mb`` rolls up as the WORST clip's peak, so a gap row
    carrying a stale stamp would set the episode's reported cost from work
    that produced no frame."""
    p = _payload(_delivered("s1", "wan_ti2v", vram_peak_mb=8241),
                 _gap("s2", "wan_ti2v", vram_peak_mb=16127))
    assert p["by_engine"]["wan_ti2v"]["vram_peak_mb"] == 8241


def test_the_gap_order_is_the_manifest_order_not_a_sorted_one():
    """Manifest order IS beat order, which is the only order that lets a
    reader line the gaps up against the episode they came from."""
    p = _payload(_gap("s_z", "still_flat"), _delivered("s_m", "ltx_8gb"),
                 _gap("s_a", "still_flat"))
    assert p["sanctioned_gap_shot_ids"] == ["s_z", "s_a"]


def test_a_gap_row_with_no_shot_id_is_still_counted():
    """A gap that cannot name itself is still a gap; dropping it would put the
    count back out of step with the episode."""
    p = _payload(_gap("", "still_flat"), _delivered("s1", "ltx_8gb"))
    assert p["sanctioned_gap_count"] == 1
    assert p["sanctioned_gap_shot_ids"] == ["?"]


# --------------------------------------------------------------------------- #
# CONTROLS -- these pass under the pre-fix code too, on purpose. They are what
# stops "count nothing, report everything as a gap" from looking correct.
# --------------------------------------------------------------------------- #
def test_a_clean_episode_is_unchanged_and_reports_zero_gaps():
    p = _payload(
        _delivered("s1", "ltx_8gb", recipe="R", quant="Q8_0",
                   family="image_to_video"),
        _delivered("s2", "wan_i2v", role="music_visual", recipe="W"),
        engine_histogram={"ltx_8gb": 1, "wan_i2v": 1}, video_revision=2)
    assert p["by_role"] == {"character_visual": {"ltx_8gb": 1},
                            "music_visual": {"wan_i2v": 1}}
    assert [c["delivered_engine"] for c in p["per_clip"]] == [
        "ltx_8gb", "wan_i2v"]
    assert p["by_engine"]["ltx_8gb"]["recipe"] == "R"
    assert p["by_engine"]["ltx_8gb"]["family"] == "image_to_video"
    assert p["sanctioned_gap_count"] == 0
    assert p["sanctioned_gap_shot_ids"] == []
    assert p["histogram"] == {"ltx_8gb": 1, "wan_i2v": 1}
    assert p["video_revision"] == 2


def test_an_empty_manifest_still_returns_the_whole_payload():
    p = _build_render_engines_payload({}, None)
    assert set(p) == _PAYLOAD_KEYS
    assert p["sanctioned_gap_count"] == 0
    assert p["per_clip"] == [] and p["by_engine"] == {}


def test_the_cadence_receipts_still_ride_a_delivered_row():
    """Present-key-only, and unaffected by the gap filter -- the filter chooses
    WHICH rows are projected, never which keys survive the projection."""
    p = _payload(_delivered("s1", "ltx_8gb", cadence_mode="native",
                            cadence_tail_trim=2))
    row = p["per_clip"][0]
    assert row["cadence_mode"] == "native" and row["cadence_tail_trim"] == 2
    assert "cadence_source_frame_count" not in row
