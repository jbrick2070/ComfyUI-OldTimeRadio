"""S1 stills-lane cloud image adapters + canonicalize_image (2026-07-03).

Offline: the invoke bridge is monkeypatched; canonicalize_image runs on a
locally generated PIL fixture. Live provider proof stays behind the
operator-gated smokes.
"""
from pathlib import Path

import pytest

from nodes._otr_shared.cloud_media_backend import CloudErrorCode, CloudMediaError
from nodes._otr_shared.cloud_media_canonical import (
    CanonicalAsset, canonicalize_image, cloud_delivery_wh)
from nodes._otr_image_engines import eng_cloud_image as eci
from nodes._otr_image_engines import registry as ireg

_CLOUD_ROWS = ("cloud_flux_pro",
               "cloud_nano_banana_2", "cloud_seedream_2")


# --------------------------------------------------------------------------- #
# registration + menu surface
# --------------------------------------------------------------------------- #


def test_all_cloud_image_rows_registered_with_capabilities():
    for name in _CLOUD_ROWS:
        assert ireg.is_registered(name), name
        row = ireg.CAPABILITIES[name]
        assert row["cpu_ok"] is True and row["model_requirements"] == []


def test_capabilities_match_registry_exactly():
    # the registry-consistency invariant: one row per registered engine.
    assert set(ireg.CAPABILITIES) == set(ireg.all_engine_names())


def test_cloud_image_rows_never_default():
    for eng in (eci.FluxPro, eci.NanoBanana2, eci.Seedream2):
        assert tuple(eng.default_roles) == ()
    for role in ("announcer_visual", "music_visual", "character_video"):
        default = ireg.default_engine_for_role(role)
        assert not str(default).startswith("cloud_"), (role, default)


def test_required_inputs_is_text_prompt():
    for eng in (eci.FluxPro, eci.NanoBanana2, eci.Seedream2):
        assert eng.required_inputs == ("text_prompt",)


def test_prepare_does_not_crash_on_none_triple():
    # the dispatcher calls prepare(None, None, None).
    for eng in (eci.FluxPro, eci.NanoBanana2, eci.Seedream2):
        assert eng.prepare(None, None, None) == {}


def test_assert_usable_ok_with_healthy_pin(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_COMFY_CLOUD_MEDIA", raising=False)
    assert eci.FluxPro.assert_usable({}, {}) == "cloud_flux_pro"


# --------------------------------------------------------------------------- #
# partner input builders (only DECLARED, pinned kwargs)
# --------------------------------------------------------------------------- #


def _req(**over):
    r = {"prompt": "an old radio", "seed": 7, "width": 832, "height": 448}
    r.update(over)
    return r


def test_flux_pro_inputs(monkeypatch):
    # TRUE 1080p cloud stills (2026-07-03): _canvas_wh now conforms to the cloud
    # delivery canvas (orientation-preserving), so a landscape role request mints
    # at 1920x1080 -- flux_pro natively requests that from the provider.
    monkeypatch.setenv("OTR_CLOUD_STILL_CANVAS", "1920x1080")
    ins = eci.FluxPro._partner_inputs(_req())
    assert set(ins) == {"prompt", "width", "height", "prompt_upsampling", "seed"}
    # BFL /32 snap: request 1920x1088 (1080 -> 1088); canonical crops to 1080.
    assert ins["width"] == 1920 and ins["height"] == 1088
    assert ins["prompt_upsampling"] is False


def test_nano_banana_inputs_resolve_model(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_NANO_BANANA_MODEL", "nb-test-model")
    ins = eci.NanoBanana2._partner_inputs(_req())
    assert set(ins) == {"model", "prompt", "response_modalities", "seed"}
    # DYNAMICCOMBO_V3 value is a DICT. model/resolution/aspect_ratio are required
    # by the live row; thinking_level is deliberately omitted because Vertex
    # rejects thinkingConfig for this image model, and OTR's bridge has a scoped
    # GeminiNanoBanana2V2 override that omits it from the provider request.
    assert ins["model"] == {"model": "nb-test-model", "resolution": "1K",
                            "aspect_ratio": "auto"}
    assert ins["response_modalities"] == "IMAGE"


def test_seedream_inputs_resolve_model(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_SEEDREAM_MODEL", "sd-test-model")
    ins = eci.Seedream2._partner_inputs(_req())
    assert set(ins) == {"model", "prompt", "seed", "watermark"}
    # ByteDanceSeedreamNodeV2.execute bracket-reads model["model"] (a bare string
    # raises "string indices must be integers"); size_preset/width/height use
    # model.get(default) so only "model" is required (2026-07-05 fix). Its
    # Pydantic request model also caps seed at signed int32; OTR's 64-bit render
    # seeds are folded deterministically into that range.
    assert ins["model"] == {"model": "sd-test-model"} and ins["watermark"] is False
    high = eci.Seedream2._partner_inputs(_req(seed=3736360535))
    assert high["seed"] == 1588876887


def test_ideo_registered_and_inputs(monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_IDEOGRAM_SPEED", raising=False)
    monkeypatch.delenv("OTR_CLOUD_IDEOGRAM_EST_USD", raising=False)
    assert ireg.is_registered("ideo")
    assert eci.Ideo.node_key == "cloud_ideogram_v4"
    assert tuple(eci.Ideo.default_roles) == ()
    ins = eci.Ideo._partner_inputs(_req())
    assert set(ins) == {"prompt", "rendering_speed", "resolution", "seed"}
    assert ins["rendering_speed"] == "TURBO"          # v1 default = cheapest
    assert eci.Ideo._est_usd() == 0.043               # TURBO price


def test_ideogram_speed_price_map(monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_IDEOGRAM_EST_USD", raising=False)
    monkeypatch.setenv("OTR_CLOUD_IDEOGRAM_SPEED", "QUALITY")
    assert eci.Ideo._est_usd() == 0.13
    assert eci.Ideo._partner_inputs(_req())["rendering_speed"] == "QUALITY"
    monkeypatch.setenv("OTR_CLOUD_IDEOGRAM_EST_USD", "0.20")   # explicit override
    assert eci.Ideo._est_usd() == 0.20


# --------------------------------------------------------------------------- #
# canonicalize_image (real PIL on a generated fixture)
# --------------------------------------------------------------------------- #


def _provider_png(tmp_path, size=(100, 50), fmt="PNG", name="provider"):
    from PIL import Image
    ext = ".jpg" if fmt == "JPEG" else ".png"
    p = tmp_path / f"{name}{ext}"
    Image.new("RGB", size, (30, 120, 200)).save(str(p), format=fmt)
    return p


def test_canonicalize_image_conforms_to_canvas(tmp_path):
    src = _provider_png(tmp_path, size=(200, 100))
    raw = {"path": str(src), "content_type": "image/png",
           "duration_s": None, "provider_job_id": "img-9", "raw_meta": {}}
    asset = canonicalize_image(raw, {"w": 832, "h": 448, "format": "PNG"})
    assert isinstance(asset, CanonicalAsset)
    assert asset.media_type == "image" and asset.container == "png"
    assert asset.width == 832 and asset.height == 448
    assert asset.path.is_file() and asset.path.suffix == ".png"
    assert len(asset.sha256) == 64
    assert asset.provider_job_id == "img-9"
    assert any("conformed" in w for w in asset.validation_warnings)
    from PIL import Image
    with Image.open(str(asset.path)) as im:
        assert im.size == (832, 448) and im.mode == "RGB"


def test_canonicalize_image_transcodes_jpeg(tmp_path):
    src = _provider_png(tmp_path, size=(832, 448), fmt="JPEG")
    raw = {"path": str(src), "content_type": "image/jpeg",
           "duration_s": None, "provider_job_id": None, "raw_meta": {}}
    asset = canonicalize_image(raw, {"w": 832, "h": 448})
    assert asset.path.suffix == ".png" and asset.path.is_file()
    assert asset.width == 832 and asset.height == 448


def test_canonicalize_image_bad_request_fails_closed(tmp_path):
    src = _provider_png(tmp_path)
    raw = {"path": str(src), "content_type": "image/png",
           "duration_s": None, "provider_job_id": None, "raw_meta": {}}
    with pytest.raises(CloudMediaError) as ei:
        canonicalize_image(raw, {"w": 832})           # h missing
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG


def test_canonicalize_image_missing_file_fails_closed(tmp_path):
    raw = {"path": str(tmp_path / "nope.png"), "content_type": "image/png",
           "duration_s": None, "provider_job_id": None, "raw_meta": {}}
    with pytest.raises(CloudMediaError) as ei:
        canonicalize_image(raw, {"w": 832, "h": 448})
    assert ei.value.code is CloudErrorCode.CORRUPT_OUTPUT


# --------------------------------------------------------------------------- #
# render_image end-to-end (bridge monkeypatched)
# --------------------------------------------------------------------------- #


def test_render_image_returns_canonical_png_path(tmp_path, monkeypatch):
    provider = _provider_png(tmp_path, size=(300, 200), name="from_provider")
    captured = {}

    def _fake_invoke(node_key, inputs, *, timeout_s, estimated_usd=0.0):
        captured.update(node_key=node_key, inputs=inputs,
                        estimated_usd=estimated_usd)
        return {"path": str(provider), "content_type": "image/png",
                "duration_s": None, "provider_job_id": "p1", "raw_meta": {}}

    import nodes._otr_shared.cloud_media_invoke as cmi
    monkeypatch.setattr(cmi, "invoke_partner_node", _fake_invoke)
    # TRUE 1080p cloud stills: the canonical PNG conforms to the cloud delivery
    # canvas (landscape 1920x1080 here), NOT the raw role-request 640x360.
    monkeypatch.setenv("OTR_CLOUD_STILL_CANVAS", "1920x1080")
    out = eci.FluxPro.render_image(_req(width=640, height=360), {})
    assert isinstance(out, str) and out.endswith(".canon.png")
    assert Path(out).is_file()
    assert captured["node_key"] == "cloud_flux_pro"
    assert captured["estimated_usd"] > 0     # budget machine needs nonzero
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (1920, 1080)


# --------------------------------------------------------------------------- #
# cloud_delivery_wh -- the TRUE-1080p orientation-preserving cloud canvas
# --------------------------------------------------------------------------- #


def test_cloud_delivery_wh_landscape_default(monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_STILL_CANVAS", raising=False)
    monkeypatch.delenv("OTR_CLOUD_STILL_CANVAS_PORTRAIT", raising=False)
    assert cloud_delivery_wh(1472, 832, land_env="OTR_CLOUD_STILL_CANVAS",
                             port_env="OTR_CLOUD_STILL_CANVAS_PORTRAIT") == (1920, 1080)


def test_cloud_delivery_wh_portrait(monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_STILL_CANVAS_PORTRAIT", raising=False)
    # rh > rw -> portrait branch
    assert cloud_delivery_wh(480, 832, land_env="OTR_CLOUD_STILL_CANVAS",
                             port_env="OTR_CLOUD_STILL_CANVAS_PORTRAIT") == (1080, 1920)


def test_cloud_delivery_wh_square_and_zero_are_landscape():
    # square (rh not > rw) and zero/unknown both fall to landscape default
    assert cloud_delivery_wh(1024, 1024, land_env="A", port_env="B") == (1920, 1080)
    assert cloud_delivery_wh(0, 0, land_env="A", port_env="B") == (1920, 1080)


def test_cloud_delivery_wh_env_override(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_VIDEO_CANVAS", "1280x720")
    assert cloud_delivery_wh(1472, 832, land_env="OTR_CLOUD_VIDEO_CANVAS",
                             port_env="OTR_CLOUD_VIDEO_CANVAS_PORTRAIT") == (1280, 720)


def test_cloud_delivery_wh_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_STILL_CANVAS", "not-a-size")
    assert cloud_delivery_wh(1472, 832, land_env="OTR_CLOUD_STILL_CANVAS",
                             port_env="OTR_CLOUD_STILL_CANVAS_PORTRAIT") == (1920, 1080)
