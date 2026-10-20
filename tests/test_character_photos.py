"""A person's own photo as a character's identity, and reference stills on
every lane (operator, 2026-10-19: "all lanes should generate ref images for
char stills so we have consistency"; "an architecture for other people to use
their own photos too").

The chain under test: OTR_ImageDirector parses ``Name = file`` lines into the
image policy; MetaBrief matches each name to the cast once and tags that
character's portrait object ``identity_photo`` (keeping the portrait even on a
lane that declares it never needs one); the dispatcher hands the photo to the
still engine as the PORTRAIT's reference -- never into the episode raw -- and
reports, LOUD, a photo it cannot use.

Pure CPU: no ComfyUI runtime, no GPU, no image model.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from nodes import otr_image_gen_dispatcher as disp
from nodes import otr_meta_brief_image_prompt as mbp
from nodes._otr_shared import character_photos as cp
from nodes.otr_image_director import OTRImageDirector

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


# --------------------------------------------------------------------------- #
# Parsing and matching
# --------------------------------------------------------------------------- #
def test_lines_parse_into_a_name_to_file_map():
    photos, problems = cp.parse_photo_lines(
        "Jeffrey = otr_cast_ab.png\n\n  Magnolia Rex =otr_cast_cd.jpg  ")
    assert photos == {"Jeffrey": "otr_cast_ab.png", "Magnolia Rex": "otr_cast_cd.jpg"}
    assert problems == []


@pytest.mark.parametrize("line", ["no equals sign", "= otr.png", "Ada = ../secret.png",
                                  "Ada = C:/x.png", "Ada = sub/x.png", "Ada = x.exe"])
def test_a_bad_line_is_reported_and_skipped(line):
    photos, problems = cp.parse_photo_lines(line)
    assert photos == {} and len(problems) == 1


def test_a_name_given_two_photos_uses_neither():
    photos, problems = cp.parse_photo_lines("Ada = a.png\nADA = b.png")
    assert photos == {} and "two photos" in problems[0]


def test_names_match_by_folded_name_to_exactly_one_cast_member():
    cast = [{"char_id": "c02", "name": "Jeffrey"},
            {"char_id": "c03", "name": "Magnolia  Rex"},
            {"char_id": "c04", "name": "Twin"}, {"char_id": "c05", "name": "twin"}]
    matched, problems = cp.match_cast(cast, {"jeffrey": "a.png", "magnolia rex": "b.png",
                                             "Twin": "c.png", "Nobody": "d.png"})
    assert matched == {"c02": "a.png", "c03": "b.png"}
    assert any("share that name" in p for p in problems)
    assert any("no character by that name" in p for p in problems)


def test_a_photo_resolves_only_inside_the_input_folder(tmp_path):
    (tmp_path / "otr_cast_ab.png").write_bytes(PNG)
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(PNG)
    assert cp.resolve_photo("otr_cast_ab.png", str(tmp_path)).endswith("otr_cast_ab.png")
    assert cp.resolve_photo("missing.png", str(tmp_path)) == ""
    assert cp.resolve_photo("../outside.png", str(tmp_path)) == ""
    assert len(cp.photo_hash(cp.resolve_photo("otr_cast_ab.png", str(tmp_path)))) == 64


# --------------------------------------------------------------------------- #
# OTR_ImageDirector carries the map on the policy
# --------------------------------------------------------------------------- #
def _video_policy():
    return json.dumps({"video_models": {}, "image_models": {
        "announcer_image_model": "flux3_klein", "music_image_model": "flux3_klein",
        "character_image_model": "flux3_klein"}})


def test_the_image_policy_carries_the_photos_and_reports_bad_lines():
    (raw,) = OTRImageDirector().direct(
        15, "request_hash", 42, video_policy_json=_video_policy(),
        character_photos="Ada = otr_cast_ab.png\nBob = ../x.png")
    policy = json.loads(raw)
    assert policy["character_photos"] == {"Ada": "otr_cast_ab.png"}
    assert any(w.startswith("character_photos: Bob") for w in policy["warnings"])


def test_a_graph_saved_before_the_widget_has_no_photos():
    (raw,) = OTRImageDirector().direct(15, "request_hash", 42,
                                       video_policy_json=_video_policy())
    assert json.loads(raw)["character_photos"] == {}


def test_the_widget_is_the_last_one_on_the_node():
    optional = list(OTRImageDirector.INPUT_TYPES()["optional"])
    assert optional[-1] == "character_photos"


# --------------------------------------------------------------------------- #
# MetaBrief: portraits on every lane where something reads them
# --------------------------------------------------------------------------- #
def _image_policy(engine, video="still_motion"):
    return json.dumps({
        "image_models": {"character_image_model": engine,
                         "announcer_image_model": engine, "music_image_model": engine},
        "video_models": {"announcer_video_model": {"engine_id": video},
                         "music_video_model": {"engine_id": video},
                         "character_video_model": {"engine_id": video}},
    })


def test_a_reference_still_engine_takes_back_the_portrait_free_exemption(monkeypatch):
    monkeypatch.setenv("OTR_FLUX3_REFERENCE", "1")
    policy = _image_policy("flux3_klein")
    free = mbp._portrait_free_roles_from_policy(policy)
    assert "character_video" in free          # the lane alone would skip it ...
    assert "character_video" in mbp._reference_still_roles_from_policy(policy)


def test_an_engine_without_references_keeps_the_exemption(monkeypatch):
    monkeypatch.delenv("OTR_FLUX3_REFERENCE", raising=False)
    assert mbp._reference_still_roles_from_policy(_image_policy("flux3_klein")) == set()
    assert mbp._reference_still_roles_from_policy(_image_policy("moss_image")) == set()
    assert mbp._reference_still_roles_from_policy("{not json") == set()


def test_the_subtraction_is_wired_at_the_real_call_site():
    import inspect
    src = inspect.getsource(mbp)
    assert "- _reference_still_roles_from_policy(image_policy_json)" in src
    assert "identity_photos=_identity_photos_from_policy(" in src


def test_photos_match_the_cast_through_the_policy():
    warnings = []
    policy = json.dumps({"character_photos": {"Ada": "a.png", "Zed": "z.png"}})
    got = mbp._identity_photos_from_policy(
        policy, [{"char_id": "c02", "name": "Ada"}], warnings)
    assert got == {"c02": "a.png"}
    assert any("Zed" in w for w in warnings)
    assert mbp._identity_photos_from_policy("{}", [], []) == {}


# --------------------------------------------------------------------------- #
# Dispatcher: the photo is the portrait's reference, or a LOUD report
# --------------------------------------------------------------------------- #
class _Engine:
    name = "flux3_klein"
    engine_version = "1"

    def __init__(self, accepts):
        self.accepts_reference_image = accepts

    def assert_usable(self, *a, **kw):
        return None


def _portrait_payload(photo):
    return {"version": 1, "objects": [{
        "object_id": "c02", "kind": "portrait", "role": "character_video",
        "char_id": "c02", "w": 64, "h": 64, "prompt": "a lighthouse keeper",
        "identity_photo": photo}]}


def _run(monkeypatch, tmp_path, accepts, photo="otr_cast_ab.png"):
    inputs = tmp_path / "input"
    inputs.mkdir(exist_ok=True)
    (inputs / "otr_cast_ab.png").write_bytes(PNG)
    monkeypatch.setattr(cp, "input_dir", lambda: str(inputs))
    monkeypatch.setattr(disp._ireg, "assert_usable", lambda engine_id, role: None)
    monkeypatch.setattr(disp._ireg, "get_engine", lambda engine_id: _Engine(accepts))
    seen = []

    def gen(request):
        seen.append(dict(request))
        return np.full((64, 64, 3), 128, dtype=np.uint8)

    from nodes._otr_shared import role_slots as rs
    policy = {"policy_version": 2, "seed": {"request_seed": 0},
              "video_models": {slot: "still_motion" for slot in rs.ROLE_TO_VIDEO_SLOT.values()},
              "image_models": {"character_image_model": {"engine_id": "flux3_klein"}}}
    led, _done, report, warnings = disp.dispatch_images(
        {"episode_id": "ep_photo", "cast": [{"char_id": "c02", "name": "Ada"}]},
        policy, _portrait_payload(photo), gen_fn=gen,
        output_dir=str(tmp_path / "out"), lockdir=tmp_path / "lease.lockdir")
    return led, seen, str(report) + " " + " ".join(map(str, warnings or []))


def _portrait_row(led):
    return next(r for r in led["images"]["images"] if r.get("kind") == "portrait")


def test_the_photo_is_the_portraits_reference(monkeypatch, tmp_path):
    led, seen, _blob = _run(monkeypatch, tmp_path, accepts=True)
    assert seen and seen[0]["reference_image"].endswith("otr_cast_ab.png")
    row = _portrait_row(led)
    assert row["identity_photo"] == "otr_cast_ab.png"
    assert row["portrait_anchor_mode"] == "reference_latent"


def test_an_engine_that_takes_no_reference_reports_the_photo_unused(monkeypatch, tmp_path):
    led, seen, blob = _run(monkeypatch, tmp_path, accepts=False)
    assert seen and not seen[0].get("reference_image")
    assert "NOT USED" in blob and "otr_cast_ab.png" in blob
    assert "identity_photo" not in _portrait_row(led)


def test_a_missing_photo_file_is_reported_and_the_portrait_still_mints(monkeypatch, tmp_path):
    _led, seen, blob = _run(monkeypatch, tmp_path, accepts=True, photo="gone.png")
    assert seen and not seen[0].get("reference_image")
    assert "gone.png" in blob and "not in ComfyUI's input folder" in blob


def test_a_different_photo_is_a_different_portrait(monkeypatch, tmp_path):
    _led, first, _ = _run(monkeypatch, tmp_path, accepts=True)
    (tmp_path / "input" / "otr_cast_ab.png").write_bytes(PNG + b"other")
    _led, second, _ = _run(monkeypatch, tmp_path, accepts=True)
    assert first and second, "a changed photo must mint again, not hit the cache"


# --------------------------------------------------------------------------- #
# End to end through the enumerator (Sonnet QA, 2026-10-20)
# --------------------------------------------------------------------------- #
_CAST = [{"char_id": "c02", "name": "JANE", "gender": "female",
          "portrait_prompt": "a tall weathered spacer with a scar"}]
_META = {"episode_id": "ep_photo", "episode_title": "The Toll",
         "story_brief_terms": {"setting": ["a dockside station"]}}


def _portraits(payload):
    return [o for o in payload.get("objects") or [] if o.get("kind") == "portrait"]


def test_the_photo_rides_the_portrait_object():
    payload, _ = mbp.derive_image_prompts(_CAST, _META, portrait_free_roles=set(),
                                          identity_photos={"c02": "jane.png"})
    assert [o.get("identity_photo") for o in _portraits(payload)] == ["jane.png"]


def test_a_photo_never_forces_a_portrait_onto_a_lane_that_cannot_use_it():
    """A role left in the portrait-free set has a still engine that takes no
    reference. Minting its portrait anyway would hand a refusing engine
    (Ideogram, 2026-08-22) a text-only face; the photo is reported instead."""
    payload, warnings = mbp.derive_image_prompts(
        _CAST, _META, portrait_free_roles={"character_video"},
        identity_photos={"c02": "jane.png"})
    assert _portraits(payload) == []
    assert any("jane.png" in w and "NOT USED" in w for w in warnings)


# --------------------------------------------------------------------------- #
# The other people in the frame (live proof 2026-10-20 00:30: the man beside
# MagnoliaRex was a stranger in every one of her stills)
# --------------------------------------------------------------------------- #
_BY_ID = {"c01": {"name": "The Announcer"}, "c02": {"name": "Jeffrey"},
          "c03": {"name": "MagnoliaRex"}, "c04": {"name": "Cat"},
          "c05": {"name": "Al"}}


def test_the_prompt_names_the_people_in_the_frame():
    prompt = "MagnoliaRex beside JEFFREY in a frozen Seoul; a gray cat at their feet"
    assert mbp.co_present_char_ids(prompt, _BY_ID, "c03") == ["c02", "c04"]
    assert mbp.co_present_char_ids(prompt, _BY_ID, "c02") == ["c03", "c04"]


def test_names_match_whole_words_only_and_the_count_is_capped():
    assert mbp.co_present_char_ids("a scattering of albums", _BY_ID, "c02") == []
    crowd = {f"c{i:02d}": {"name": f"Person{i}"} for i in range(10)}
    text = " ".join(f"Person{i}" for i in range(10))
    assert len(mbp.co_present_char_ids(text, crowd, "c00")) == mbp.CO_REFERENCE_LIMIT


def test_co_presence_is_wired_where_scene_stills_are_built():
    import inspect
    assert "co_present_char_ids(sprompt, _cast_by_id, _cid)" in inspect.getsource(mbp)


def test_the_other_peoples_portraits_ride_with_the_speakers(monkeypatch, tmp_path):
    from nodes._otr_shared import role_slots as rs
    monkeypatch.setattr(disp._ireg, "assert_usable", lambda engine_id, role: None)
    monkeypatch.setattr(disp._ireg, "get_engine", lambda engine_id: _Engine(True))
    seen = []

    def gen(request):
        seen.append(dict(request))
        shade = 40 + 60 * len(seen)          # a different picture per mint
        return np.full((64, 64, 3), shade % 256, dtype=np.uint8)

    def portrait(cid):
        return {"object_id": cid, "kind": "portrait", "role": "character_video",
                "char_id": cid, "w": 64, "h": 64, "prompt": "portrait of " + cid}

    scene = {"object_id": "still_b001", "kind": "scene_character",
             "role": "character_video", "char_id": "c02", "beat_id": "b001",
             "co_char_ids": ["c03"], "w": 64, "h": 64,
             "prompt": "Jeffrey and MagnoliaRex on a frozen street"}
    policy = {"policy_version": 2, "seed": {"request_seed": 0},
              "video_models": {s: "still_motion" for s in rs.ROLE_TO_VIDEO_SLOT.values()},
              "image_models": {"character_image_model": {"engine_id": "flux3_klein"}}}
    disp.dispatch_images(
        {"episode_id": "ep_co", "cast": [{"char_id": "c02", "name": "Jeffrey"},
                                         {"char_id": "c03", "name": "MagnoliaRex"}]},
        policy, {"version": 1, "objects": [portrait("c02"), portrait("c03"), scene],
                 "required_scene_targets": [{"object_id": "still_b001",
                                             "kind": "scene_character",
                                             "role": "character_video",
                                             "beat_id": "b001"}]},
        gen_fn=gen, output_dir=str(tmp_path / "out"), lockdir=tmp_path / "lease.lockdir")
    still = next(r for r in seen if r["object_id"] == "still_b001")
    assert still["reference_image"] and len(still["reference_images"]) == 1
    assert still["reference_images"][0] != still["reference_image"]
    assert not any(r.get("reference_images") for r in seen if r["object_id"] != "still_b001")


def test_klein_chains_one_reference_latent_per_portrait(monkeypatch):
    from nodes._otr_image_engines import registry as ireg
    monkeypatch.setenv("OTR_FLUX3_REFERENCE", "1")
    eng = ireg.get_engine("flux3_klein")
    p = eng._params({"prompt": "two people", "seed": 1, "reference_image": "a.png",
                     "reference_images": ["b.png", "c.png"]})
    g = eng._build_graph(p, lambda node, slot: ("wire", node, slot))
    assert [g[k]["inputs"]["image"] for k in ("load_ref", "load_ref_1", "load_ref_2")] == [
        "a.png", "b.png", "c.png"]
    assert g["ref_pos_1"]["inputs"]["conditioning"] == ("wire", "ref_pos", 0)
    assert g["ref_pos_2"]["inputs"]["conditioning"] == ("wire", "ref_pos_1", 0)
    assert g["ksampler"]["inputs"]["positive"] == ("wire", "ref_pos_2", 0)
    assert g["ksampler"]["inputs"]["negative"] == ("wire", "ref_neg_2", 0)
    # Companions never stand in for a missing speaker reference.
    lone = eng._params({"prompt": "x", "reference_images": ["b.png"]})
    assert lone["reference_images"] == [] and lone["reference_image"] == ""


def test_a_first_name_finds_a_two_word_cast_name_when_it_is_unique():
    cast = {"c02": {"name": "Jeffrey Brick"}, "c03": {"name": "Magnolia Rex"},
            "c04": {"name": "Magnolia Bell"}}
    assert mbp.co_present_char_ids("Jeffrey waves", cast, "c03") == ["c02"]
    # Two Magnolias: the bare first name is ambiguous, so neither is guessed.
    assert mbp.co_present_char_ids("Magnolia waves", cast, "c02") == []
    assert mbp.co_present_char_ids("Magnolia Rex waves", cast, "c02") == ["c03"]


def test_a_title_is_never_taken_for_a_first_name():
    cast = {"c02": {"name": "Lady Macbeth"}, "c03": {"name": "Macbeth"}}
    assert mbp.co_present_char_ids("Lady Macduff weeps", cast, "c03") == []
    assert mbp.co_present_char_ids("Lady Macbeth weeps", cast, "c03") == ["c02"]
