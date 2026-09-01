"""HuMo profile provisioning owns one exact automatic fetch path."""
from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_provisioner():
    path = ROOT / "scripts" / "otr_provision.py"
    spec = importlib.util.spec_from_file_location("_otr_provision_humo_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipping_humo_14b_profiles_route_to_the_complete_automatic_lane():
    provision = _load_provisioner()

    for profile_id in (
        "otr_g4_humo", "otr_w45_humo", "otr_w45_humo_14b_169",
    ):
        routes = provision.profile_lanes(provision.load_profile(profile_id))
        assert routes == {
            "automatic": ["humo", "z_image", "stable_audio_3"],
            "manual": [],
        }
    assert "humo_14b" not in provision.MANUAL_TIERS


def test_humo_1_7b_profiles_never_download_the_14b_lane():
    provision = _load_provisioner()

    for profile_id in ("otr_w45_humo_1_7b", "otr_w45_humo_1_7b_169"):
        routes = provision.profile_lanes(provision.load_profile(profile_id))
        assert "humo" not in routes["automatic"]
        assert routes["manual"] == ["humo_1_7b"]


def test_main_delegates_humo_once_for_a_selected_14b_profile(monkeypatch):
    provision = _load_provisioner()
    calls = []
    models_root = str(ROOT / "models")

    monkeypatch.setattr(provision, "comfy_root", lambda: str(ROOT.parent.parent))
    monkeypatch.setattr(provision, "models_root", lambda _comfy: models_root)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", models_root)
    monkeypatch.setattr(provision, "ensure_hf_home", lambda _root: None)
    monkeypatch.setattr(provision, "install_node_packs", lambda _comfy: None)
    monkeypatch.setattr(provision, "install_requirements", lambda: None)
    monkeypatch.setattr(provision, "fetch_lane_weights",
                        lambda lanes: calls.append(list(lanes)))
    monkeypatch.setattr(provision, "install_indextts2", lambda *_args: None)

    rc = provision.main([
        "--profile", "otr_w45_humo_14b_169", "--with-indextts2",
    ])

    assert rc == 0
    assert calls == [["humo", "z_image", "stable_audio_3"]]
