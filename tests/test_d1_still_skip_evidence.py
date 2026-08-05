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
    character sits late in a long prompt."""
    lead = "a weathered captain on a storm deck, " * 6          # ~220 chars
    hit = disp.path_guard_arm(lead + "lit in black/white, 35mm grain")

    assert hit is not None
    assert hit["arm"] == "alternate_separator"
    assert hit["token"] == "/"
    assert hit["index"] > 200, "excerpt must be centred on a LATE match"
    assert "black/white" in hit["excerpt"]
    assert hit["prompt_len"] > 240
    # The CANONICAL sha256 the dispatcher already uses, so evidence joins to the
    # ledger's prompt_hash instead of inventing a second digest.
    assert hit["prompt_hash"] == disp._prompt_content_hash(
        lead + "lit in black/white, 35mm grain")


def test_path_guard_arm_distinguishes_extension_from_separator():
    ext = disp.path_guard_arm("a portrait of a radio host.png")
    assert ext["arm"] == "extension_suffix"
    assert ext["token"] == ".png"

    sep = disp.path_guard_arm("output" + os.sep + "stills" + os.sep + "a.jpg")
    assert sep["arm"] in ("separator", "alternate_separator")


def test_path_guard_preserves_the_original_predicate_exactly():
    """Observability-only means the SET of refused prompts cannot move.

    Two ways a rewrite could have drifted, both pinned here: a stray rstrip()
    would newly refuse a trailing-space '.png', and reporting the earliest
    textual match would name an arm the original predicate never chose (it
    short-circuits separator -> alternate_separator -> extension)."""
    # trailing whitespace: endswith() fails on the RAW string, so this passes
    assert disp.path_guard_arm("a portrait of a radio host.png   ") is None
    # both separators present -> the original chain picks os.sep first, even
    # though the '/' happens to appear earlier in the text
    both = "a/b" + os.sep + "c"
    assert disp.path_guard_arm(both)["arm"] == "separator"


def test_path_guard_arm_passes_clean_prose():
    assert disp.path_guard_arm("a normal portrait prompt, cinematic") is None
    assert disp.path_guard_arm("") is None


def test_path_guard_excerpt_is_repr_escaped_so_a_prompt_cannot_forge_log_lines():
    """A prompt is LLM-authored text. An embedded newline must not be able to
    fabricate an additional log record."""
    hit = disp.path_guard_arm("first half\nERROR forged record: a/b")
    assert "\\n" in hit["excerpt"]
    assert "\n" not in hit["excerpt"]


def test_assert_not_path_message_carries_the_arm():
    with pytest.raises(ValueError, match="arm=alternate_separator"):
        disp._assert_not_path("a prompt with a black/white palette")


# --------------------------------------------------------------------------- #
# the completion gate carries the evidence
# --------------------------------------------------------------------------- #
def test_slash_bearing_required_target_names_its_branch_and_never_renders(tmp_path):
    """THE INCIDENT, in miniature. A slash-bearing prompt is skipped silently;
    because the object is a REQUIRED scene target the episode then fails -- and
    the failure must now say why, not just which."""
    calls = []

    def _gen(req):
        calls.append(req["object_id"])
        raise AssertionError("gen_fn must not be reached for a refused prompt")

    payload = {
        "version": 1,
        "objects": [_scene_object("still_b007", "a queen, lit in black/white", "b007")],
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
    assert "alternate_separator" in msg
    assert "black/white" in msg, "the offending text must be quoted back"

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
    """
    payload = {
        "version": 1,
        "objects": [
            _scene_object("still_b1", "a lantern, black/white tones", "b1"),
            _scene_object("still_b12", "a hall" + os.sep + "corridor view", "b12"),
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
    assert by_id["still_b1"]["evidence"]["arm"] == "alternate_separator"
    assert by_id["still_b1"]["evidence"]["token"] == "/"
    assert by_id["still_b12"]["evidence"]["arm"] == "separator"
    assert by_id["still_b12"]["evidence"]["token"] == os.sep
    # ...and their hashes differ, so the evidence is provably per-prompt.
    assert (by_id["still_b1"]["evidence"]["prompt_hash"]
            != by_id["still_b12"]["evidence"]["prompt_hash"])


def test_extension_arm_is_reachable_now_that_the_safety_clause_is_retired():
    """The 2026-08-03 finding this pinned has been REPEALED by its own fix.

    `dispatch_images` runs `append_visual_safety_clause` before the guard. That
    helper used to append a family-safe/no-weapons clause, so a prompt ending in
    '.png' no longer ended in '.png' by the time the guard saw it, and the
    `extension_suffix` arm effectively could not fire inside the dispatcher --
    which is why the live suspect for that incident was a SEPARATOR arm.

    The clause was retired 2026-08-05 (operator directive: no content guardrails)
    and the helper is now a passthrough, so the extension arm is reachable again.
    That is a strict improvement -- the guard's most specific arm is no longer
    masked by a content policy -- and it is pinned here so the masking cannot
    return unnoticed.
    """
    from nodes._otr_story_brief_helpers import append_visual_safety_clause

    raw = "a portrait of the radio host.png"
    assert disp.path_guard_arm(raw)["arm"] == "extension_suffix"

    dispatched = append_visual_safety_clause(raw)
    assert dispatched == raw, "the clause helper must no longer rewrite prompts"
    assert dispatched.lower().endswith(".png")
    assert disp.path_guard_arm(dispatched)["arm"] == "extension_suffix"


def test_skip_is_logged_at_skip_time_not_only_on_the_wire(tmp_path, caplog):
    """The wire-only warning died with the raise. The server log must carry it
    independently, because that log is what an operator actually reads."""
    payload = {
        "version": 1,
        "objects": [_scene_object("still_b007", "a queen, black/white", "b007")],
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
    assert "alternate_separator" in logged
