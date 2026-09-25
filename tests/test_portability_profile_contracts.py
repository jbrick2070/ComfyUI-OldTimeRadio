"""Shipping LTX 2.5 matrix rows name every artifact and safe launch fact."""
from __future__ import annotations

from nodes._otr_shared.capability_profiles import known_profile_ids, load_profile
from nodes._otr_shared.public_engines import resolve_engine_id
from nodes._otr_video_engines import eng_ltx25
from nodes._otr_video_engines import ltx25_recipe as ltx
from nodes._otr_video_engines import registry as vreg


def _ltx25_engines(profile: dict) -> list:
    """Every LTX 2.5 engine the row selects, across its roles."""
    picks = [v for k, v in (profile.get("role_overrides") or {}).items()
             if k.endswith("_visual")]
    engines = []
    for pick in picks:
        name = resolve_engine_id(pick) if pick else None
        if name and vreg.is_registered(name):
            engine = vreg.get_engine(name)
            if isinstance(engine, eng_ltx25.Ltx25VideoEngine):
                engines.append(engine)
    return engines


def test_every_shipping_ltx_profile_has_complete_model_and_launch_contract(
        monkeypatch):
    monkeypatch.delenv("OTR_LTX25_NATIVE_DIT", raising=False)
    monkeypatch.delenv("OTR_LTX25_NATIVE_TE", raising=False)
    ltx_rows = []
    for profile_id in known_profile_ids():
        profile = load_profile(profile_id)
        engines = _ltx25_engines(profile)
        if not engines or profile["status"] != "shipping":
            continue
        ltx_rows.append(profile_id)
        expected = {
            (e._dit_name(), e._text_encoder_name(), ltx.LTX25_VIDEO_VAE,
             ltx.LTX25_AUDIO_VAE, ltx.LTX25_UPSCALER_MODEL)
            for e in engines
        }
        assert len(expected) == 1, (
            f"{profile_id} selects LTX 2.5 lanes that load different weights: "
            f"{sorted(expected)}")
        models = profile["preflight"]["required_models"]
        assert tuple(models[:5]) == expected.pop(), profile_id
        assert profile["launch"] == {
            "boot_contract": "default",
            "sage_attention": False,
            "extra_args": [],
            "env": {},
        }, profile_id
    # The row on `ltx25_video` itself (public ltx25_high_video) must be among
    # them, or this walked the matrix and checked none of the lanes it names.
    assert "otr_16gb_video" in ltx_rows, ltx_rows
