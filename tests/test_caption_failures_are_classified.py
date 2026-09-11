"""Tell a broken BOX apart from a broken EPISODE, before anything rests on it.

`burn_captions_on_video` raised a bare `ValueError` for four unrelated conditions:
no ffmpeg, an ffmpeg without libass, a missing input video, and an unknown caption
style. `OTRCaptionBurn.burn` catches all four identically and, when a hero title was
planned, re-raises and ends a ~30-minute render at its last stage.

A degrade written on top of that catch was REVERTED within the hour, because
`tests/test_caption_burn_fails_closed_on_title.py` proved what it cost: an unknown
caption STYLE lands in the same branch, and degrading a misconfiguration would ship
untitled episodes forever in silence -- strictly worse than refusing.

So the classification ships FIRST, and on its own. `CaptionCapabilityGapError` is a
ValueError SUBCLASS, so every existing handler behaves exactly as before and this
change alters no behaviour at all -- which is the point. A later policy can then rest
on a probe-confirmed fact instead of a guess about what an ffmpeg exit code meant.

THE PROBE WAS ALREADY BUILT AND ALREADY TESTED. `caption_support_gap()` /
`probe_ffmpeg_capabilities()` have lived in `nodes/_otr_shared/ffmpeg.py` with their
own test file and ZERO callers outside that module -- the fourth built-and-unwired
helper found on 2026-09-11, after `select_grounding`, `select_passage` and
`SourceOverview`.
"""
from __future__ import annotations

import pytest

from nodes import otr_caption_burn as burn_mod
from nodes.otr_caption_burn import CaptionCapabilityGapError


def test_the_gap_error_IS_a_ValueError():
    """The whole no-behaviour-change claim rests on this one line. Every caller
    catches ValueError; if this were a sibling type instead of a subclass, a
    capability gap would escape the caption node entirely and kill the prompt."""
    assert issubclass(CaptionCapabilityGapError, ValueError)


def test_NO_FFMPEG_is_a_capability_gap(monkeypatch, tmp_path):
    """Nothing is wrong with the episode; the same render succeeds elsewhere."""
    monkeypatch.setattr(burn_mod, "_ffmpeg_bin", lambda _f: "")

    video = tmp_path / "ep.mp4"
    video.write_bytes(b"x")
    with pytest.raises(CaptionCapabilityGapError, match="ffmpeg not found"):
        burn_mod.burn_captions_on_video(str(video), "", str(tmp_path / "o.mp4"))


def test_a_PROBE_CONFIRMED_missing_libass_is_a_capability_gap(
        monkeypatch, tmp_path):
    """The realistic container case: an ffmpeg exists but was built minimal."""
    monkeypatch.setattr(burn_mod, "_ffmpeg_bin", lambda _f: "/usr/bin/ffmpeg")

    import nodes._otr_shared.ffmpeg as shared
    monkeypatch.setattr(
        shared, "caption_support_gap",
        lambda _p=None: "the ffmpeg at /usr/bin/ffmpeg is a minimal build "
                        "missing ass")

    video = tmp_path / "ep.mp4"
    video.write_bytes(b"x")
    with pytest.raises(CaptionCapabilityGapError, match="minimal build"):
        burn_mod.burn_captions_on_video(str(video), "", str(tmp_path / "o.mp4"))


def test_an_UNRUNNABLE_probe_does_NOT_invent_a_gap(monkeypatch, tmp_path):
    """THE GUESS THIS MUST NOT MAKE. `caption_support_gap` answers None both
    when captions will work and when the probe could not run -- an unrunnable
    probe is not evidence of a missing feature. A refusal invented here would
    kill episodes on boxes whose ffmpeg is perfectly capable."""
    monkeypatch.setattr(burn_mod, "_ffmpeg_bin", lambda _f: "/usr/bin/ffmpeg")

    import nodes._otr_shared.ffmpeg as shared
    monkeypatch.setattr(shared, "caption_support_gap", lambda _p=None: None)

    missing = tmp_path / "not_here.mp4"
    # It proceeds PAST the capability check and fails on the real next problem,
    # which is the pipeline defect -- and that one is NOT a capability gap.
    with pytest.raises(ValueError) as excinfo:
        burn_mod.burn_captions_on_video(str(missing), "", str(tmp_path / "o.mp4"))
    assert not isinstance(excinfo.value, CaptionCapabilityGapError), (
        "a missing input video was misclassified as a broken ffmpeg")
    assert "input video missing" in str(excinfo.value)


def test_a_MISSING_INPUT_is_not_a_capability_gap(monkeypatch, tmp_path):
    """A broken episode, not a broken box. Refusing is correct and must stay."""
    monkeypatch.setattr(burn_mod, "_ffmpeg_bin", lambda _f: "/usr/bin/ffmpeg")

    import nodes._otr_shared.ffmpeg as shared
    monkeypatch.setattr(shared, "caption_support_gap", lambda _p=None: None)

    with pytest.raises(ValueError) as excinfo:
        burn_mod.burn_captions_on_video(
            str(tmp_path / "gone.mp4"), "", str(tmp_path / "o.mp4"))
    assert type(excinfo.value) is ValueError, (
        "the pipeline-defect branch must stay a plain ValueError")


def test_a_PLANNED_TITLE_still_refuses_on_a_capability_gap(
        monkeypatch, tmp_path):
    """POLICY IS UNCHANGED, and this test exists to prove it.

    The reverted change made a planned title pass through. This one does not:
    classification shipped alone. When the policy question is reopened, THIS is
    the test that has to be deliberately rewritten -- which is the point of
    pinning it now."""
    monkeypatch.setattr(
        burn_mod, "_parse_title_plan",
        lambda _j: ({"card": "x"}, "plan: 1 card"))

    def gap(*_a, **_k):
        raise CaptionCapabilityGapError("OTR_CaptionBurn: no libass here")

    monkeypatch.setattr(burn_mod, "burn_captions_on_video", gap)

    import nodes._otr_paths as paths
    monkeypatch.setattr(paths, "confine_to_output_tree",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(paths, "reject_remote_paths",
                        lambda **k: None, raising=False)

    node = burn_mod.OTRCaptionBurn()
    with pytest.raises(RuntimeError, match="hero title card was planned"):
        node.burn(
            video_path="/out/otr/episodes/ep/final.mp4",
            ledger_path="",
            output_path="/out/otr/episodes/ep/final_captioned.mp4",
            burn_captions=True,
            caption_style="",
            fps=24,
            ffmpeg="",
            title_card_plan_json='{"card": "x"}',
        )


def test_the_probe_is_actually_WIRED(monkeypatch):
    """The defect class this repo keeps producing: a correct helper nothing
    calls. `caption_support_gap` sat built and tested with zero callers."""
    import inspect

    src = inspect.getsource(burn_mod.burn_captions_on_video)
    assert "caption_support_gap" in src, (
        "the capability probe is unwired again")
