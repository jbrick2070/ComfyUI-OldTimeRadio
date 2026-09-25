"""Provisioning plans every selected profile route without hidden fallback."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _provisioner():
    path = ROOT / "scripts" / "otr_provision.py"
    spec = importlib.util.spec_from_file_location(
        "_otr_provision_profile_route_tests", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_weight_video_routes_are_registered_and_asset_free():
    from nodes._otr_video_engines import registry

    provision = _provisioner()
    assert provision._NO_WEIGHT_VIDEO_ENGINES
    for engine_id in provision._NO_WEIGHT_VIDEO_ENGINES:
        assert engine_id in registry.CAPABILITIES
        assert registry.CAPABILITIES[engine_id]["model_requirements"] == []

    assert provision._REMOTE_NO_WEIGHT_VIDEO_ENGINES
    for engine_id in provision._REMOTE_NO_WEIGHT_VIDEO_ENGINES:
        assert engine_id in registry.CAPABILITIES
        assert registry.CAPABILITIES[engine_id]["model_requirements"] == []


def _with_every_visual(profile: dict, video: str) -> dict:
    """A copy of one matrix row with every video selection moved to `video`."""
    swapped = copy.deepcopy(profile)
    swapped["slot_overrides"]["video_render_engine"] = video
    for key in ("announcer_visual", "music_visual", "character_visual"):
        swapped["role_overrides"][key] = video
    return swapped


def test_remote_video_profile_skips_remote_video_download():
    """A remote-video profile must not mint a local video fetch lane."""
    provision = _provisioner()
    lanes = provision.profile_lanes("otr_cloud_deluxe_audio_in_3act")
    assert "cloud_ltx25_audio_in" not in lanes["automatic"]
    assert "cloud_ltx25_audio_in" not in lanes["manual"]


def test_razzle_ltx_8gb_profile_uses_the_ltx_8gb_weight_lane():
    provision = _provisioner()
    ltx_8gb = provision.load_profile("otr_8gb_video")
    razzle = _with_every_visual(ltx_8gb, "razzle_ltx_8gb")

    assert provision.profile_lanes(razzle) == provision.profile_lanes(ltx_8gb)
    assert "ltx_8gb" in provision.profile_lanes(razzle)["automatic"]


def test_amd_profiles_plan_their_exact_image_and_music_dependencies():
    """AMD installs with NOTHING extra -- no manual download, no third-party pack.

    Commit b1f372a9 moved AMD off `flux2_klein` (which needed a third-party
    pack and a hand-fetched 11 GB download) and onto `z_image_turbo`, which
    comes out of OTR's own fetch manifest -- "Make the AMD tiers installable
    with nothing extra". The shipped AMD row keeps that promise: its plan is
    fully automatic and its manual list is EMPTY.

    Its 6.8 GB ceiling selects `z_image_int8`, and its `stable_audio_3` music
    is the one audio engine with a fetcher lane of its own.
    """
    provision = _provisioner()

    assert provision.profile_lanes("otr_amd_still") == {
        "automatic": ["z_image_int8", "stable_audio_3"],
        "manual": [],
    }


def test_amd_machine_selector_has_a_complete_dry_run_plan(capsys):
    provision = _provisioner()

    assert provision.main(["--machine", "amd", "--list"]) == 0
    output = capsys.readouterr().out
    assert "automatic    : z_image_int8" in output or "automatic    : z_image" in output
    assert "manual       : none" in output
    assert "flux2_klein" not in output
    assert "unrecognized video engine" not in output


def test_kokoro_profiles_fail_early_on_python_313_but_bark_does_not():
    provision = _provisioner()
    kokoro = provision.load_machine_profile("8gb")
    bark = copy.deepcopy(provision.load_profile("otr_8gb_animatediff"))
    bark["slot_overrides"]["char_voice_engine"] = "bark"
    bark["slot_overrides"]["announcer_voice_engine"] = "bark"

    # 2026-09-02: 3.13 runs kokoro through kokoro-onnx; only 3.14+ is flagged.
    assert provision.profile_python_issue(kokoro, (3, 13)) == ""
    assert "no backend packaged for Python 3.14" in \
        provision.profile_python_issue(kokoro, (3, 14))
    assert provision.profile_python_issue(kokoro, (3, 12)) == ""
    assert provision.profile_python_issue(bark, (3, 13)) == ""
    assert provision.profile_python_issue(bark, (3, 14)) == ""


def test_machine_readable_plan_check_rejects_kokoro_on_python_314(
        monkeypatch, capsys):
    provision = _provisioner()
    monkeypatch.setattr(provision.sys, "version_info", (3, 14, 0))

    assert provision.main(["--machine", "8gb", "--check-plan"]) == 1
    rejected = capsys.readouterr().out
    assert "MISSING" in rejected
    assert "no backend packaged for Python 3.14" in rejected

    # A row with no Kokoro voice at all (hosted voices) is ready on 3.14.
    assert provision.main([
        "--profile", "otr_cloud_deluxe_3act", "--check-plan"
    ]) == 0
    ready = capsys.readouterr().out
    assert "READY: complete provision plan for otr_cloud_deluxe_3act" in ready


@pytest.mark.parametrize(
    "profile_id", ["otr_8gb_low", "otr_16gb_low", "otr_mac16_low"])
def test_procedural_low_rows_do_not_require_an_uninvoked_image_lane(
        profile_id):
    """Every visual is a procedural visualizer, which consumes no still.

    The rows still name an image engine per role; none may be planned. The
    Mac row names `sd15`, which has no provisioner lane at all -- planning it
    would raise, so this also proves the image selection is skipped rather
    than merely routed to nothing.
    """
    provision = _provisioner()

    assert provision.profile_lanes(profile_id) == {
        "automatic": ["stable_audio_3"],
        "manual": [],
    }


@pytest.mark.parametrize("machine_key", ["8gb", "12gb"])
def test_haunted_machine_paths_do_not_require_unconsumed_klein_weights(
        machine_key):
    provision = _provisioner()

    assert provision.profile_lanes(
        provision.load_machine_profile(machine_key)) == {
            "automatic": ["haunted"],
            "manual": [],
        }


@pytest.mark.parametrize("missing_image", ["lumina_image", "flux_gen1"])
def test_mixed_role_plan_includes_every_video_and_rejects_unowned_images(
        missing_image):
    provision = _provisioner()
    # One role moves off its procedural visualizer onto a still-consuming
    # video and pins an image engine that has no provisioner lane.
    mixed = copy.deepcopy(provision.load_profile("otr_16gb_low"))
    mixed["id"] = "mixed_role_%s" % missing_image
    mixed["role_overrides"]["character_visual"] = "still_motion"
    mixed["role_overrides"]["character_image"] = missing_image

    with pytest.raises(
            provision.ProvisionFailure,
            match="unrecognized image engine %r" % missing_image):
        provision.profile_lanes(mixed)


def test_remote_profiles_need_no_local_video_or_image_weights():
    provision = _provisioner()

    assert provision.profile_lanes("otr_cloud_deluxe_3act") == {
        "automatic": [],
        "manual": [],
    }


def test_profile_selector_rejects_path_traversal():
    provision = _provisioner()

    with pytest.raises(provision.ProvisionFailure, match="invalid profile id"):
        provision.load_profile("../otr_8gb_animatediff")


def test_profile_selector_rejects_an_id_that_is_not_a_matrix_row():
    provision = _provisioner()

    with pytest.raises(
            provision.ProvisionFailure,
            match="'requested' is not a row of config/workflow_matrix.json"):
        provision.load_profile("requested")
