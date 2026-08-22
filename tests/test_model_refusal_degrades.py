"""A MODEL REFUSAL DEGRADES; AN ENGINE FAULT STILL HARD-FAILS.

Operator, 2026-08-22, after one refused card destroyed a finished episode:
*"why is refusing card killing the episode, i dont think thats good feature"*,
*"its an experimental stack its not perfect"*, *"i didnt want any fail on this
or that"*.

THE DISTINCTION THIS PINS. An OOM, a missing wrapper node or a decode failure
means the ENGINE IS BROKEN and must hard-fail the episode LOUD -- the 2026-06-18
NO FALLBACKS rule is untouched. A safety refusal means the engine worked
perfectly and the MODEL declined one card: it returned valid decoded pixels at
the exact requested dimensions with the graph completing. Treating those two the
same is what cost the 2026-08-21 episode seven finished stills.

WHY IT IS NOT A FALLBACK. No engine is substituted, nothing is silent, and the
beat simply has no still -- a state the pipeline already has. The prompt is
RECORDED on the way out, which the hard-fail never did: the refused card's
prompt was erased, which is why the refusal could not be diagnosed until the two
prompts were finally diffed by hand (they proved byte-identical, so the refusal
is a seed lottery on identical input -- nothing a prompt fix could reach).
"""
from __future__ import annotations

import pytest

from nodes import otr_image_gen_dispatcher as disp
from nodes._otr_image_engines.ideogram4_local import Ideogram4RefusalError


@pytest.fixture(autouse=True)
def _usable_engines(monkeypatch):
    """Stub the ADAPTER-level usability gate.

    NOT because the weights are missing -- all four ideogram artifacts and
    z_image are on this box. `assert_usable` resolves them through
    ``folder_paths``, which exists ONLY inside a running ComfyUI, so any
    off-runtime test reports "not installed" no matter what is on disk. That is
    the same trap `_resolve_model_file_by_token` documents, and it is why the
    live ideogram campaign passed while a CPU unit test cannot.

    These tests pin the DISPATCHER's handling of an exception the engine
    raised; engine installation is a different question with its own gate.
    """
    class _UsableStub:
        name = "z_image_turbo"

        def assert_usable(self, *a, **kw):
            return None

    # BOTH gates. The registry-level one is name/role only; the ADAPTER-level
    # one calls the engine's own assert_usable, which is what actually resolves
    # weights through folder_paths -- so stubbing only the first leaves the
    # second firing, which is exactly what happened on the first attempt here.
    monkeypatch.setattr(disp._ireg, "assert_usable",
                        lambda engine_id, role: None)
    monkeypatch.setattr(disp._ireg, "get_engine", lambda engine_id: _UsableStub())


def _complete_video_models():
    from nodes._otr_shared import role_slots as rs
    return {slot: "still_word" for slot in rs.ROLE_TO_VIDEO_SLOT.values()}


def _policy():
    return {
        "policy_version": 2,
        "video_models": _complete_video_models(),
        "seed": {"request_seed": 0},
        # z_image_turbo, NOT ideogram4_local: this pins the DISPATCHER's
        # handling, and the marker it branches on rides the exception instance,
        # not the engine. Naming ideogram here would make the test fail an
        # earlier weights-usability gate on any box without its 11 GB of
        # artifacts -- i.e. it would test the fixture, not the behaviour.
        "image_models": {
            "character_image_model": {"engine_id": "z_image_turbo"}},
    }


def _scene_object(oid, beat_id):
    return {
        "object_id": oid, "kind": "scene_character", "role": "character_video",
        "char_id": "c01", "beat_id": beat_id,
        "prompt": "an abstract picture evoking \"The Weight of the Grain\"",
        "w": 64, "h": 64,
    }


def _required(oid, beat_id):
    return {"object_id": oid, "kind": "scene_character",
            "role": "character_video", "beat_id": beat_id}


def _payload(*oids):
    return {
        "version": 1,
        "objects": [_scene_object(o, "b%03d" % i) for i, o in enumerate(oids, 1)],
        "required_scene_targets": [
            _required(o, "b%03d" % i) for i, o in enumerate(oids, 1)],
    }


# --------------------------------------------------------------------------- #
# The flag itself
# --------------------------------------------------------------------------- #

def test_the_refusal_declares_itself_and_a_plain_error_does_not():
    """The dispatcher cannot IMPORT the refusal class (that would be a cycle,
    swallowed by a guarded import, silently unregistering the engine), so the
    marker has to travel on the exception instance."""
    assert getattr(Ideogram4RefusalError("x"), "is_model_refusal", False) is True
    assert getattr(RuntimeError("x"), "is_model_refusal", False) is False
    assert getattr(MemoryError("cuda oom"), "is_model_refusal", False) is False


# --------------------------------------------------------------------------- #
# The behaviour
# --------------------------------------------------------------------------- #

def test_a_model_refusal_does_not_kill_the_episode(tmp_path):
    """The whole point. One refused object, and the run continues."""
    seen = []

    def gen(request):
        seen.append(request["object_id"])
        raise Ideogram4RefusalError(
            "%s: safety-refusal placeholder (min=78.0, std=10.6)"
            % request["object_id"])

    led, image_done, report, warnings = disp.dispatch_images(
        {"episode_id": "ep_refusal", "cast": []},
        _policy(), _payload("still_music_closing_001"), gen_fn=gen,
        output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")

    assert seen == ["still_music_closing_001"], "the engine was never called"
    blob = " ".join(str(w) for w in (warnings or [])) + " " + str(report)
    assert "MODEL REFUSAL" in blob, (
        "the degrade must be LOUD -- a silent skip is the thing NO FALLBACKS "
        "exists to forbid")
    assert "still_music_closing_001" in blob


def test_the_refusal_records_the_prompt_so_it_can_be_diagnosed(tmp_path):
    """The hard-fail erased the evidence: the refused card's prompt was never
    persisted, which is exactly why the 2026-08-21 refusal sat undiagnosed."""
    def gen(request):
        raise Ideogram4RefusalError("refused")

    _led, _done, report, warnings = disp.dispatch_images(
        {"episode_id": "ep_refusal_prompt", "cast": []},
        _policy(), _payload("still_music_closing_001"), gen_fn=gen,
        output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")

    blob = " ".join(str(w) for w in (warnings or [])) + " " + str(report)
    assert "Weight of the Grain" in blob, (
        "the composed prompt must ride the warning -- without it a refusal "
        "cannot be told from a content problem after the fact")


def test_one_refusal_does_not_stop_the_other_objects(tmp_path):
    """Seven finished stills must not die for the eighth."""
    calls = []

    def gen(request):
        oid = request["object_id"]
        calls.append(oid)
        if oid == "still_bad":
            raise Ideogram4RefusalError("refused")
        return b"\x89PNG\r\n\x1a\n" + b"\0" * 4096

    try:
        disp.dispatch_images(
            {"episode_id": "ep_refusal_mixed", "cast": []},
            _policy(), _payload("still_bad", "still_good"), gen_fn=gen,
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")
    except disp.ImageRenderError:
        # A later target-completeness gate may still object to the missing
        # object; what must NOT happen is the loop stopping at the refusal.
        pass
    assert "still_good" in calls, (
        "the refusal aborted the loop -- the object after it was never even "
        "attempted, which is the exact damage the operator ruled against")


# --------------------------------------------------------------------------- #
# The rule that did NOT change
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("boom", [
    MemoryError("CUDA out of memory"),
    RuntimeError("wrapper node missing: SomeLoader"),
    ValueError("decode failed"),
])
def test_a_real_engine_fault_still_hard_fails(tmp_path, boom):
    """NO FALLBACKS (operator 2026-06-18) is untouched. An OOM is not a
    blemish -- it means the engine is broken, and a broken engine must stop the
    episode rather than quietly produce a stillless render."""
    def gen(_request):
        raise boom

    with pytest.raises(disp.ImageRenderError) as ei:
        disp.dispatch_images(
            {"episode_id": "ep_real_fault", "cast": []},
            _policy(), _payload("still_b001"), gen_fn=gen,
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")
    assert "NO FALLBACK" in str(ei.value)
