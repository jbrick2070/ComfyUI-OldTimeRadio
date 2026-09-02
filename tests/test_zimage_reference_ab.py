"""CPU-only structural tests for the persistent Z-Image reference A/B."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "otr_zimage_reference_ab.py"
_SPEC = importlib.util.spec_from_file_location("otr_zimage_reference_ab", SCRIPT)
ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ab)


@pytest.fixture(autouse=True)
def _pin_recipe_env(monkeypatch):
    pins = {
        "OTR_ZIMAGE_UNET": "z_image_turbo_nvfp4.safetensors",
        "OTR_ZIMAGE_CLIP": "qwen_3_4b_fp8_mixed.safetensors",
        "OTR_ZIMAGE_VAE": "ae.safetensors",
        "OTR_ZIMAGE_CLIP_TYPE": "qwen_image",
        "OTR_ZIMAGE_LATENT_NODE": "EmptySD3LatentImage",
        "OTR_ZIMAGE_UNET_DTYPE": "default",
        "OTR_ZIMAGE_STEPS": "8",
        "OTR_ZIMAGE_CFG": "1.0",
        "OTR_ZIMAGE_SHIFT": "3.0",
        "OTR_ZIMAGE_SAMPLER": "euler",
        "OTR_ZIMAGE_SCHEDULER": "normal",
        "OTR_PORTRAIT_REF_HEIGHT": "768",
    }
    for key, value in pins.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OTR_ZIMAGE_NEGATIVE", raising=False)


def _ref(tmp_path: Path) -> Path:
    path = tmp_path / "accepted_c02.png"
    path.write_bytes(b"fixed-test-portrait")
    return path


def _available() -> set[str]:
    return {
        "UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow",
        "CLIPTextEncode", "EmptySD3LatentImage", "KSampler", "VAEDecode",
        "LoadImage", "ImageScale", "VAEEncode", "ReferenceLatent", "SaveImage",
    }


def test_off_is_exact_engine_base_graph_without_reference_latent(tmp_path):
    api, raw, params = ab.build_arm_graph("off", _ref(tmp_path), "ab/off/out")

    assert set(raw) == ab._BASE_NODES
    assert all(node["class_type"] != "ReferenceLatent" for node in api.values())
    assert raw["ksampler"]["inputs"]["positive"] == ["pos", 0]
    assert raw["ksampler"]["inputs"]["negative"] == ["neg", 0]
    assert params["reference_image"] == ""
    assert ab._graph_positive_evidence(raw, api) == {
        "graph_off_has_no_reference_latent": True,
        "graph_on_has_exact_dual_reference_chain": False,
    }


def test_on_is_exact_dual_reference_chain_and_sampler_rewire(tmp_path):
    ref = _ref(tmp_path).resolve()
    staged = ab.deterministic_load_image_input(ref)
    api, raw, params = ab.build_arm_graph("on", ref, "ab/on/out")

    assert set(raw) == ab._BASE_NODES | ab._REF_NODES
    assert raw["load_ref"]["inputs"] == {"image": staged}
    assert raw["scale_ref"]["inputs"] == {
        "image": ["load_ref", 0],
        "upscale_method": "lanczos",
        "width": 0,
        "height": 768,
        "crop": "disabled",
    }
    assert raw["encode_ref"]["inputs"] == {
        "pixels": ["scale_ref", 0], "vae": ["vae", 0]
    }
    assert raw["ref_pos"]["inputs"] == {
        "conditioning": ["pos", 0], "latent": ["encode_ref", 0]
    }
    assert raw["ref_neg"]["inputs"] == {
        "conditioning": ["neg", 0], "latent": ["encode_ref", 0]
    }
    assert raw["ksampler"]["inputs"]["positive"] == ["ref_pos", 0]
    assert raw["ksampler"]["inputs"]["negative"] == ["ref_neg", 0]
    assert api["ref_pos"]["class_type"] == "ReferenceLatent"
    assert api["ref_neg"]["class_type"] == "ReferenceLatent"
    assert params["reference_image"] == staged
    assert ab._graph_positive_evidence(raw, api) == {
        "graph_off_has_no_reference_latent": False,
        "graph_on_has_exact_dual_reference_chain": True,
    }


def test_positive_evidence_is_derived_from_graph_not_arm_label(tmp_path):
    ref = _ref(tmp_path)
    api, raw, _params = ab.build_arm_graph("on", ref, "ab/on/out")
    raw["ksampler"]["inputs"]["positive"] = ["pos", 0]
    assert ab._graph_positive_evidence(raw, api)[
        "graph_on_has_exact_dual_reference_chain"
    ] is False

    api, raw, _params = ab.build_arm_graph("on", ref, "ab/on/out")
    api["ref_pos"]["class_type"] = "NotReferenceLatent"
    assert ab._graph_positive_evidence(raw, api)[
        "graph_on_has_exact_dual_reference_chain"
    ] is False


def test_arms_match_every_pinned_generation_parameter(tmp_path):
    ref = _ref(tmp_path)
    _off_api, _off_raw, off = ab.build_arm_graph("off", ref, "ab/off/out")
    _on_api, _on_raw, on = ab.build_arm_graph("on", ref, "ab/on/out")

    assert ab._match_signature(off) == ab._match_signature(on) == ab._PINNED_PARAMS
    assert off["reference_image"] == ""
    assert on["reference_image"] == ab.deterministic_load_image_input(ref)


def test_live_resolution_fails_closed_without_reference_latent(tmp_path):
    available = _available() - {"ReferenceLatent"}
    with pytest.raises(ab.HarnessError, match="ReferenceLatent"):
        ab.build_arm_graph(
            "on", _ref(tmp_path), "ab/on/out", available=available
        )


def test_recipe_env_drift_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_ZIMAGE_CFG", "2.0")
    with pytest.raises(ab.HarnessError, match="cfg=2.0"):
        ab.build_arm_graph("off", _ref(tmp_path), "ab/off/out")


def test_artifact_dir_must_be_inside_comfy_output(tmp_path):
    output = tmp_path / "output"
    inside, prefix = ab.resolve_artifact_dir(output / "otr" / "episodes" / "ab", output)
    assert inside == (output / "otr" / "episodes" / "ab").resolve()
    assert prefix == "otr/episodes/ab"

    with pytest.raises(ab.HarnessError, match="inside ComfyUI"):
        ab.resolve_artifact_dir(tmp_path / "elsewhere", output)


def test_graph_only_cli_writes_both_explicit_receipts_without_network(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    artifact = output / "otr" / "episodes" / "zimage_ref_ab" / "stills"
    ref = _ref(tmp_path)
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(output))

    def _network_forbidden():
        raise AssertionError("graph-only contacted ComfyUI")

    monkeypatch.setattr(ab, "fetch_schemas", _network_forbidden)
    monkeypatch.setattr(ab.requests, "post", _network_forbidden)
    for arm in ("off", "on"):
        assert ab.main([
            "--arm", arm,
            "--reference-image", str(ref),
            "--artifact-dir", str(artifact),
            "--graph-only",
        ]) == 0

    off = json.loads((artifact / "off" / "receipt.json").read_text("utf-8"))
    on = json.loads((artifact / "on" / "receipt.json").read_text("utf-8"))
    assert off["status"] == on["status"] == "GRAPH_ONLY"
    assert off["match_signature"] == on["match_signature"]
    assert off["reference"] == on["reference"]
    assert off["positive_evidence"] == {
        "graph_off_has_no_reference_latent": True,
        "graph_on_has_exact_dual_reference_chain": False,
        "history_success_with_output": False,
    }
    assert on["positive_evidence"] == {
        "graph_off_has_no_reference_latent": False,
        "graph_on_has_exact_dual_reference_chain": True,
        "history_success_with_output": False,
    }
    assert off["staging"]["mode"] == "not_applicable"
    assert on["staging"]["mode"] == "would_upload"
    assert on["staging"]["requested_name"] == ab.deterministic_staged_name(ref)
    assert on["staging"]["requested_subfolder"] == ab.UPLOAD_SUBFOLDER
    assert on["staging"]["returned_name"] == ""
    assert on["staging"]["load_image_input"] == ab.deterministic_load_image_input(ref)
    off_graph = json.loads((artifact / "off" / "graph.json").read_text("utf-8"))
    on_graph = json.loads((artifact / "on" / "graph.json").read_text("utf-8"))
    assert "ref_pos" not in off_graph["api_graph"]
    assert on_graph["api_graph"]["ref_pos"]["class_type"] == "ReferenceLatent"
    assert on_graph["api_graph"]["load_ref"]["inputs"]["image"] == (
        ab.deterministic_load_image_input(ref)
    )


@pytest.mark.parametrize("arm, expected_upload_calls", [("off", 0), ("on", 1)])
def test_live_on_uploads_through_server_while_live_off_does_not(
    monkeypatch, tmp_path, arm, expected_upload_calls
):
    output = tmp_path / "output"
    artifact = output / "otr" / "episodes" / "zimage_ref_ab" / "stills"
    ref = _ref(tmp_path)
    expected_name = ab.deterministic_staged_name(ref)
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(output))
    monkeypatch.setattr(ab, "fetch_schemas", lambda: {name: {} for name in _available()})

    upload_calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "name": expected_name,
                # Deliberately server-returned rather than inferred by client:
                # proves this exact value reaches LoadImage and the receipt.
                "subfolder": "server/accepted/zimage_reference_ab",
                "type": "input",
            }

    def _post(url, *, files, data, timeout):
        upload_calls.append({
            "url": url,
            "filename": files["image"][0],
            "payload": files["image"][1].read(),
            "mime": files["image"][2],
            "data": data,
            "timeout": timeout,
        })
        return _Response()

    monkeypatch.setattr(ab.requests, "post", _post)

    submitted = []

    def _submit(api):
        submitted.append(api)
        arm_dir = artifact / arm
        (arm_dir / f"zimage_reference_{arm}_seed7_00001_.png").write_bytes(b"png")
        return "prompt-id"

    monkeypatch.setattr(ab, "submit_prompt", _submit)
    monkeypatch.setattr(ab, "poll_history", lambda *args, **kwargs: ("SUCCESS", ""))

    assert ab.main([
        "--arm", arm,
        "--reference-image", str(ref),
        "--artifact-dir", str(artifact),
    ]) == 0
    assert len(upload_calls) == expected_upload_calls
    receipt = json.loads((artifact / arm / "receipt.json").read_text("utf-8"))
    assert receipt["positive_evidence"]["history_success_with_output"] is True
    assert receipt["output_artifacts"] == [{
        "path": receipt["outputs"][0],
        "bytes": 3,
        "sha256": ab._sha256(Path(receipt["outputs"][0])),
    }]
    if arm == "on":
        call = upload_calls[0]
        assert call["url"].endswith("/upload/image")
        assert call["filename"] == expected_name
        assert call["payload"] == b"fixed-test-portrait"
        assert call["data"] == {
            "overwrite": "true",
            "type": "input",
            "subfolder": ab.UPLOAD_SUBFOLDER,
        }
        returned_input = f"server/accepted/zimage_reference_ab/{expected_name}"
        assert submitted[0]["load_ref"]["inputs"]["image"] == returned_input
        assert receipt["staging"]["mode"] == "uploaded"
        assert receipt["staging"]["requested_name"] == expected_name
        assert receipt["staging"]["returned_subfolder"] == (
            "server/accepted/zimage_reference_ab"
        )
        assert receipt["staging"]["load_image_input"] == returned_input
    else:
        assert "load_ref" not in submitted[0]
        assert receipt["staging"]["mode"] == "not_applicable"
        assert receipt["staging"]["load_image_input"] == ""
