"""Provisioning plans every selected profile route without hidden fallback."""
from __future__ import annotations

import importlib.util
import json
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


def test_word_razzle_profile_skips_only_its_remote_video_download():
    provision = _provisioner()

    assert provision.profile_lanes("otr_w45_word_razzle") == {
        "automatic": ["z_image", "stable_audio_3"],
        "manual": [],
    }


def test_amd_profiles_plan_their_exact_image_and_music_dependencies():
    provision = _provisioner()

    assert provision.profile_lanes("otr_amd16_rocm") == {
        "automatic": ["stable_audio_3"],
        "manual": ["flux2_klein"],
    }
    assert provision.profile_lanes("otr_amd8_rocm") == {
        "automatic": [],
        "manual": ["flux2_klein"],
    }


def test_amd_machine_selector_has_a_complete_dry_run_plan(capsys):
    provision = _provisioner()

    assert provision.main(["--machine", "amd", "--list"]) == 0
    output = capsys.readouterr().out
    assert "automatic    : none" in output
    assert "manual tiers : flux2_klein" in output
    assert "unrecognized video engine" not in output


def test_w45_profiles_without_complete_install_owners_are_explicit():
    """The pod roster may exclude these; it may never discover them silently."""
    provision = _provisioner()
    failures = {}
    for path in sorted((ROOT / "config" / "profiles").glob("otr_w45_*.json")):
        try:
            provision.profile_lanes(path.stem)
        except provision.ProvisionFailure as exc:
            failures[path.stem] = str(exc)

    assert set(failures) == {
        "otr_w45_fastwan",
        "otr_w45_ltx_audio_in",
        "otr_w45_ltx_video",
        "otr_w45_mesh_stage",
    }
    assert all("unrecognized video engine" in reason
               for reason in failures.values())


def test_default_pod_roster_is_filtered_through_the_provision_plan_owner():
    runtime = (ROOT / "scripts" / "otr_pod_runtime.sh").read_text(
        encoding="utf-8")

    assert '"$OTR_REPO_ROOT/scripts/otr_provision.py"' in runtime
    assert '--profile "$profile" --check-plan' in runtime
    assert "excluding profile without a complete provision plan" in runtime
    assert "has no complete provision plan" in runtime


def test_kokoro_profiles_fail_early_on_python_313_but_bark_does_not():
    provision = _provisioner()
    kokoro = provision.load_machine_profile("8gb")
    bark = provision.load_profile("otr_4060_floor")

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

    assert provision.main([
        "--profile", "otr_4060_floor", "--check-plan"
    ]) == 0
    ready = capsys.readouterr().out
    assert "READY: complete provision plan for otr_4060_floor" in ready


def test_procedural_4060_floor_does_not_require_an_uninvoked_image_lane():
    provision = _provisioner()

    assert provision.profile_lanes("otr_4060_floor") == {
        "automatic": [],
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


def test_mixed_role_plan_includes_every_video_and_rejects_unowned_images():
    provision = _provisioner()

    assert provision.profile_lanes("otr_sbcov_5") == {
        "automatic": ["wan_ti2v_gguf", "stable_audio_3"],
        "manual": ["flux2_klein"],
    }
    for profile_id, missing_image in (
        ("otr_soak_llmsweep_01", "flux_gen1"),
        ("otr_soak_llmsweep_02", "flux_gen1"),
    ):
        with pytest.raises(
                provision.ProvisionFailure,
                match="unrecognized image engine %r" % missing_image):
            provision.profile_lanes(profile_id)


def test_remote_profiles_need_no_local_video_or_image_weights():
    provision = _provisioner()

    assert provision.profile_lanes("google_veo_media") == {
        "automatic": [],
        "manual": [],
    }


def test_profile_selector_rejects_path_traversal():
    provision = _provisioner()

    with pytest.raises(provision.ProvisionFailure, match="invalid profile id"):
        provision.load_profile("../otr_4060_floor")


def test_profile_selector_rejects_filename_id_drift(tmp_path, monkeypatch):
    provision = _provisioner()
    profiles = tmp_path / "config" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "requested.json").write_text(
        json.dumps({"id": "different"}), encoding="utf-8"
    )
    monkeypatch.setattr(provision, "_REPO", str(tmp_path))

    with pytest.raises(
            provision.ProvisionFailure, match="profile filename/id drift"):
        provision.load_profile("requested")
