"""D1: a silently skipped still must name its own branch.

Regression cover for the 2026-08-03 incident: a 320-word Shakespeare leg failed
with `required scene image targets missing or unmaterialized before video
dispatch: still_b007, still_b008` and the REASON was unrecoverable. The skip
warning carried no object id, it was wire-only, the completion gate raised
before the wire was stamped, and the boot relaunch truncated the server log.

These tests pin the evidence path, not the fix -- the underlying guard behaviour
is deliberately unchanged.

Postmortem: docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md
"""
import os

import pytest

from nodes import otr_image_gen_dispatcher as disp


def _complete_video_models(*, announcer="viz_mxc_cpu", music="viz_mxc_mandala",
                           character="still_motion"):
    return {
        "announcer_video_model": {"engine_id": announcer},
        "music_video_model": {"engine_id": music},
        "character_video_model": {"engine_id": character},
    }


def _policy(**kw):
    return {
        "policy_version": 2,
        "video_models": _complete_video_models(**kw),
        "seed": {"request_seed": 0},
        "image_models": {"character_image_model": {"engine_id": "z_image_turbo"}},
    }


def _scene_object(oid, prompt, beat_id):
    return {
        "object_id": oid, "kind": "scene_character", "role": "character_video",
        "char_id": "c01", "beat_id": beat_id, "prompt": prompt,
        "w": 64, "h": 64,
    }


def _required(oid, beat_id):
    return {"object_id": oid, "kind": "scene_character",
            "role": "character_video", "beat_id": beat_id}


# --------------------------------------------------------------------------- #
# the guard's own verdict
# --------------------------------------------------------------------------- #
def test_path_guard_arm_reports_the_exact_arm_and_position():
    """The old report was prompt[:60], which shows nothing when the offending
    character sits late in a long prompt.

    WHOLE-STRING CLASSIFICATION (2026-08-05): the five rooted arms
    (drive_root/unc_root/posix_root/file_uri/explicit_relative) all anchor at
    the START of the string, so their token's index is always 0 -- a LATE
    index is only reachable through the ``image_path`` fallback, whose token
    is the trailing extension. Exercised here with a long, whitespace-free
    nested relative path -- the shape a crossed disk-path socket actually
    hands back, not prose that merely mentions a separator.
    """
    nested = "renders/queen_storm_deck_captain_lit_grain_takes_pass02_selected/"
    long_path = nested * 4 + "final_composite_master_shot.png"
    hit = disp.path_guard_arm(long_path)

    assert hit is not None
    assert hit["arm"] == "image_path"
    assert hit["token"] == ".png"
    assert hit["index"] > 200, "excerpt must be centred on a LATE match"
    assert "final_composite_master_shot.png" in hit["excerpt"]
    assert hit["prompt_len"] > 240
    # The CANONICAL sha256 the dispatcher already uses, so evidence joins to the
    # ledger's prompt_hash instead of inventing a second digest.
    assert hit["prompt_hash"] == disp._prompt_content_hash(long_path)


def test_path_guard_arm_unifies_bare_filename_and_nested_path_under_image_path():
    """RENAMED (2026-08-05, was ``..._distinguishes_extension_from_separator``):
    the old taxonomy split a bare filename (``extension_suffix``) from a
    slash-bearing path (``separator``/``alternate_separator``) even though both
    are unambiguously a path. Those three arms are GONE -- whole-string
    classification collapses a bare filename and a whitespace-free nested
    relative path into the SAME ``image_path`` arm, which is explicitly
    documented to cover "shot.png" AND "assets/shot.png" alike.
    """
    bare = disp.path_guard_arm("a_radio_host_portrait.png")
    assert bare["arm"] == "image_path"
    assert bare["token"] == ".png"

    nested = disp.path_guard_arm("output" + os.sep + "stills" + os.sep + "a.jpg")
    assert nested["arm"] == "image_path"
    assert nested["token"] == ".jpg"


def test_path_guard_arm_ordering_and_whitespace_handling_are_pinned():
    """RENAMED (2026-08-05, was ``..._preserves_the_original_predicate_exactly``).

    "Observability-only means the SET of refused prompts cannot move" was the
    OLD premise; the whole-string rewrite deliberately MOVED that set on
    purpose (prose is now clean), so that framing no longer applies verbatim.
    What still must not silently drift is pinned here instead: the ORDER the
    arms are tried in, and how incidental whitespace is handled.

    Priority: a rooted path is ALSO whitespace-free-and-ends-in-an-image-
    extension, so it would satisfy ``image_path`` too -- the ordered loop must
    still report the more specific rooted arm, not the fallback.
    """
    assert disp.path_guard_arm("C:/out/shot.png")["arm"] == "drive_root"
    assert disp.path_guard_arm("./shot.png")["arm"] == "explicit_relative"

    # Incidental OUTER whitespace around a bare path does not launder it --
    # the guard strips only the ends before classifying.
    assert disp.path_guard_arm("   shot.png")["arm"] == "image_path"
    # ...but genuine prose (whitespace THROUGHOUT, not just at the ends)
    # stays clean regardless of trailing whitespace: there is no single
    # non-whitespace run for image_path to match.
    assert disp.path_guard_arm("a portrait of a radio host.png   ") is None


def test_path_guard_arm_passes_clean_prose():
    assert disp.path_guard_arm("a normal portrait prompt, cinematic") is None
    assert disp.path_guard_arm("") is None


def test_path_guard_excerpt_is_repr_escaped_so_a_prompt_cannot_forge_log_lines():
    """A prompt is LLM-authored text. An embedded newline must not be able to
    fabricate an additional log record.

    The old fixture was bare prose with a slash, which no longer trips the
    guard at all (2026-08-05 whole-string classification) so it can no
    longer carry this case. The five rooted arms match on the STRING'S
    PREFIX only (their regexes have no end anchor), so a real rooted path
    followed by injected garbage -- including a forged newline -- still
    trips the guard; that is exactly the shape a crossed path-string socket
    could hand back, and the excerpt must still repr-escape it.
    """
    hit = disp.path_guard_arm("/var/tmp/first_half.png\nERROR forged record: a/b")
    assert hit["arm"] == "posix_root"
    assert "\\n" in hit["excerpt"]
    assert "\n" not in hit["excerpt"]


def test_assert_not_path_message_carries_the_arm():
    # A prose prompt with a slash no longer trips the guard at all (2026-08-05
    # whole-string classification); a genuine path is needed to reach the raise.
    with pytest.raises(ValueError, match="arm=drive_root"):
        disp._assert_not_path("C:/out/shot.png")


# --------------------------------------------------------------------------- #
# the completion gate carries the evidence
# --------------------------------------------------------------------------- #
def test_slash_bearing_required_target_names_its_branch_and_never_renders(tmp_path):
    """THE INCIDENT, in miniature. A PATH-bearing prompt is skipped silently;
    because the object is a REQUIRED scene target the episode then fails -- and
    the failure must now say why, not just which.

    The original fixture was prose with a slash ("a queen, lit in
    black/white"), which was the INCIDENT itself: ordinary prose refused by a
    substring guard. Whole-string classification (2026-08-05) fixed exactly
    that, so this prompt would no longer trip the guard -- a genuine path is
    needed to still exercise the skip-and-fail-loud contract this test guards.
    """
    calls = []

    def _gen(req):
        calls.append(req["object_id"])
        raise AssertionError("gen_fn must not be reached for a refused prompt")

    payload = {
        "version": 1,
        "objects": [_scene_object(
            "still_b007", "C:/renders/queen_lit_black_and_white.png", "b007")],
        "required_scene_targets": [_required("still_b007", "b007")],
    }
    with pytest.raises(disp.ImageRenderError) as ei:
        disp.dispatch_images(
            {"episode_id": "ep_d1_slash", "cast": []},
            _policy(), payload, gen_fn=_gen,
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")

    msg = str(ei.value)
    assert calls == [], "the refused prompt must never reach the engine"
    assert "still_b007" in msg
    assert "prompt_path_guard" in msg
    assert "drive_root" in msg
    assert "queen_lit_black_and_white.png" in msg, (
        "the offending text must be quoted back")

    missing = getattr(ei.value, "missing_targets", None)
    assert missing and missing[0]["object_id"] == "still_b007"
    assert missing[0]["status"] == "no_row"
    assert missing[0]["evidence"]["reason"] == "prompt_path_guard"


def test_missing_target_with_no_skip_evidence_says_so(tmp_path):
    """An object that was never dispatched at all still gets a deterministic
    status -- and must not silently imply a guard skip."""
    payload = {"version": 1, "objects": [],
               "required_scene_targets": [_required("still_b001", "b001")]}
    with pytest.raises(disp.ImageRenderError) as ei:
        disp.dispatch_images(
            {"episode_id": "ep_d1_norow", "cast": []},
            _policy(), payload, gen_fn=lambda _r: None,
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")

    msg = str(ei.value)
    assert "still_b001" in msg
    assert "no_row" in msg
    assert "no skip evidence recorded" in msg


def test_evidence_is_keyed_not_substring_matched(tmp_path):
    """still_b1 is a PREFIX of still_b12. Correlating evidence by scanning
    warning text would cross-associate them; the map is keyed for this reason.

    Both objects trip the guard, but on DIFFERENT arms, so each one's evidence
    is individually identifiable. Substring association could not tell them
    apart. Deliberately no successful render here: a real image engine's weights
    are not present in the test environment, and this contract does not need one.

    The original fixtures were prose with a slash / an os.sep, which no
    longer trip the guard (2026-08-05 whole-string classification); real path
    fixtures on two DIFFERENT rooted arms replace them so the "not
    cross-matched" proof still has two genuinely distinct arms to compare.
    """
    payload = {
        "version": 1,
        "objects": [
            _scene_object("still_b1", "C:/renders/lantern_scene.png", "b1"),
            _scene_object(
                "still_b12", "/var/renders/hall_corridor_view.png", "b12"),
        ],
        "required_scene_targets": [
            _required("still_b1", "b1"),
            _required("still_b12", "b12"),
        ],
    }
    with pytest.raises(disp.ImageRenderError) as ei:
        disp.dispatch_images(
            {"episode_id": "ep_d1_prefix", "cast": []},
            _policy(), payload, gen_fn=lambda _r: None,
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")

    by_id = {m["object_id"]: m for m in ei.value.missing_targets}
    assert set(by_id) == {"still_b1", "still_b12"}
    # Each id carries ITS OWN arm -- the proof that nothing was cross-matched.
    assert by_id["still_b1"]["evidence"]["arm"] == "drive_root"
    assert by_id["still_b1"]["evidence"]["token"] == "C:/"
    assert by_id["still_b12"]["evidence"]["arm"] == "posix_root"
    assert by_id["still_b12"]["evidence"]["token"] == "/v"
    # ...and their hashes differ, so the evidence is provably per-prompt.
    assert (by_id["still_b1"]["evidence"]["prompt_hash"]
            != by_id["still_b12"]["evidence"]["prompt_hash"])


def test_image_path_arm_is_reachable_now_that_the_safety_clause_is_retired():
    """RENAMED (2026-08-05, was ``..._extension_arm_is_reachable...``): the
    old arm name ``extension_suffix`` is gone, folded into ``image_path`` by
    the whole-string classification rewrite. The finding this test pins is
    otherwise unchanged.

    The 2026-08-03 finding this pinned has been REPEALED by its own fix.

    `dispatch_images` runs `append_visual_safety_clause` before the guard. That
    helper used to append a family-safe/no-weapons clause, so a prompt ending in
    '.png' no longer ended in '.png' by the time the guard saw it, and the
    extension-bearing arm effectively could not fire inside the dispatcher --
    which is why the live suspect for that incident was a SEPARATOR arm (also
    since retired; see test_path_guard_arm_unifies_bare_filename_and_nested_
    path_under_image_path).

    The clause was retired 2026-08-05 (operator directive: no content guardrails)
    and the helper is now a passthrough, so the `image_path` arm is reachable
    again. Note it now requires a WHITESPACE-FREE string (a bare filename, not
    a sentence that happens to end in '.png' -- "a portrait of the radio
    host.png" is explicitly the CLEAN example in path_guard_arm's own
    docstring now), so the fixture is a bare filename rather than a sentence.
    That is a strict improvement -- the guard's most specific arm is no longer
    masked by a content policy, and prose is no longer conflated with a real
    path -- and it is pinned here so the masking cannot return unnoticed.
    """
    from nodes._otr_story_brief_helpers import append_visual_safety_clause

    raw = "radio_host_portrait.png"
    assert disp.path_guard_arm(raw)["arm"] == "image_path"

    dispatched = append_visual_safety_clause(raw)
    assert dispatched == raw, "the clause helper must no longer rewrite prompts"
    assert dispatched.lower().endswith(".png")
    assert disp.path_guard_arm(dispatched)["arm"] == "image_path"


def test_skip_is_logged_at_skip_time_not_only_on_the_wire(tmp_path, caplog):
    """The wire-only warning died with the raise. The server log must carry it
    independently, because that log is what an operator actually reads.

    The original fixture ("a queen, black/white") was prose, which no longer
    trips the guard (2026-08-05 whole-string classification); a real path is
    needed to still reach the skip branch this test observes.
    """
    payload = {
        "version": 1,
        "objects": [_scene_object(
            "still_b007", "C:/renders/queen_lit_black_and_white.png", "b007")],
        "required_scene_targets": [],          # no gate -> episode would succeed
    }
    with caplog.at_level("WARNING"):
        disp.dispatch_images(
            {"episode_id": "ep_d1_log", "cast": []},
            _policy(), payload, gen_fn=lambda _r: None,
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SKIP still_b007" in logged
    assert "path guard" in logged
    assert "drive_root" in logged
