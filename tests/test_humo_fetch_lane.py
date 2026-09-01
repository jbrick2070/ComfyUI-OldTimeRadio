"""The HuMo one-command fetch lane is exact, pinned, and fail-closed."""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_fetcher():
    path = ROOT / "scripts" / "otr_fetch_lane_weights.py"
    spec = importlib.util.spec_from_file_location("_otr_humo_fetch_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_humo_lane_is_the_complete_engine_recipe(monkeypatch):
    from nodes._otr_video_engines.eng_humo import HuMoEngine

    for name in (
        "OTR_HUMO_CKPT", "OTR_HUMO_UNET_NAME", "OTR_HUMO_LORA_NAME",
        "OTR_HUMO_CLIP_NAME", "OTR_HUMO_VAE_NAME",
        "OTR_HUMO_AUDIO_ENCODER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    fetcher = _load_fetcher()
    entries = fetcher.LANES["humo"]
    fetched = {fetcher.destination_name(row) for row in entries}
    loaded = set(HuMoEngine()._loader_names().values())

    assert fetched == loaded
    assert fetched == {
        "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "whisper_large_v3_fp16.safetensors",
        "wan_2.1_vae.safetensors",
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    }
    assert "humo_17B_fp8_e4m3fn.safetensors" not in fetched


def test_humo_lane_has_one_exact_receipt_per_artifact():
    fetcher = _load_fetcher()
    entries = [fetcher.weight_spec(row) for row in fetcher.LANES["humo"]]

    assert entries == [
        fetcher.WeightSpec(
            "Kijai/WanVideo_comfy_fp8_scaled",
            "HuMo/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            "diffusion_models/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            "033a4e487f60220b3d6e469599a6aebc46e13cee", 17_892_294_098,
            "a67ed82a7c008892f9192cdc5b23bbfe2e2a8e2f87d0b5b8dfb0226fafec022d"),
        fetcher.WeightSpec(
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "617a7633e636506f850e043bc4605f290a466a8e", 6_735_906_897,
            "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"),
        fetcher.WeightSpec(
            "Comfy-Org/HuMo_ComfyUI",
            "split_files/audio_encoders/whisper_large_v3_fp16.safetensors",
            "audio_encoders/whisper_large_v3_fp16.safetensors",
            "3a5e6947d865c3910cb2407cf2dac6a8df506b5a", 3_087_130_976,
            "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"),
        fetcher.WeightSpec(
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "split_files/vae/wan_2.1_vae.safetensors",
            "vae/wan_2.1_vae.safetensors",
            "c4f60d30c55a624e35427060fdd217579a6c1d77", 253_815_318,
            "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
        fetcher.WeightSpec(
            "Kijai/WanVideo_comfy",
            "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "8260d429d19fd7a72304cad059160b95d843913f", 738_005_744,
            "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3"),
    ]
    assert sum(row.expected_bytes for row in entries) == 28_707_153_033
    assert all(re.fullmatch(r"[0-9a-f]{40}", row.revision) for row in entries)
    assert all(re.fullmatch(r"[0-9a-f]{64}", row.expected_sha256)
               for row in entries)
    assert {row.destination.split("/", 1)[0] for row in entries} == {
        "diffusion_models", "text_encoders", "audio_encoders", "vae", "loras",
    }


def test_pinned_fetch_verifies_part_then_atomically_renames(tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    payload = b"verified HuMo fixture"
    row = fetcher.WeightSpec(
        "Example/HuMo",
        "weights/model.safetensors",
        "diffusion_models/model.safetensors",
        "1" * 40,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    seen = {}

    def fake_urlretrieve(url, destination):
        seen["url"] = url
        pathlib.Path(destination).write_bytes(payload)

    monkeypatch.setattr(fetcher.urllib.request, "urlretrieve", fake_urlretrieve)

    assert fetcher.fetch(row, str(tmp_path), False) is True
    final = tmp_path / "diffusion_models" / "model.safetensors"
    assert final.read_bytes() == payload
    assert not pathlib.Path(str(final) + ".part").exists()
    assert "/resolve/%s/weights/model.safetensors" % ("1" * 40) in seen["url"]


def test_exact_existing_file_skips_the_network(tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    payload = b"already verified"
    row = fetcher.WeightSpec(
        "Example/HuMo", "weights/model.safetensors",
        "diffusion_models/model.safetensors", "3" * 40, len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    final = tmp_path / "diffusion_models" / "model.safetensors"
    final.parent.mkdir(parents=True)
    final.write_bytes(payload)
    monkeypatch.setattr(
        fetcher.urllib.request, "urlretrieve",
        lambda *_args: (_ for _ in ()).throw(AssertionError("network called")),
    )

    assert fetcher.fetch(row, str(tmp_path), False) is True
    assert final.read_bytes() == payload


def test_pinned_fetch_rejects_bad_hash_without_publishing_final(tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    expected = b"expected bytes"
    received = b"tampered bytes"
    assert len(expected) == len(received)
    row = fetcher.WeightSpec(
        "Example/HuMo",
        "weights/model.safetensors",
        "diffusion_models/model.safetensors",
        "2" * 40,
        len(expected),
        hashlib.sha256(expected).hexdigest(),
    )

    def fake_urlretrieve(_url, destination):
        pathlib.Path(destination).write_bytes(received)

    monkeypatch.setattr(fetcher.urllib.request, "urlretrieve", fake_urlretrieve)

    assert fetcher.fetch(row, str(tmp_path), False) is False
    final = tmp_path / "diffusion_models" / "model.safetensors"
    assert not final.exists()
    assert not pathlib.Path(str(final) + ".part").exists()


def test_pinned_fetch_rejects_wrong_byte_count_without_publishing_final(
        tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    expected = b"expected bytes"
    row = fetcher.WeightSpec(
        "Example/HuMo",
        "weights/model.safetensors",
        "diffusion_models/model.safetensors",
        "5" * 40,
        len(expected),
        hashlib.sha256(expected).hexdigest(),
    )

    def fake_urlretrieve(_url, destination):
        pathlib.Path(destination).write_bytes(b"short")

    monkeypatch.setattr(fetcher.urllib.request, "urlretrieve", fake_urlretrieve)

    assert fetcher.fetch(row, str(tmp_path), False) is False
    final = tmp_path / "diffusion_models" / "model.safetensors"
    assert not final.exists()
    assert not pathlib.Path(str(final) + ".part").exists()


def test_failed_replacement_preserves_the_previous_final(tmp_path, monkeypatch):
    fetcher = _load_fetcher()
    expected = b"right"
    previous = b"wrong"
    received = b"other"
    row = fetcher.WeightSpec(
        "Example/HuMo", "weights/model.safetensors",
        "diffusion_models/model.safetensors", "4" * 40, len(expected),
        hashlib.sha256(expected).hexdigest(),
    )
    final = tmp_path / "diffusion_models" / "model.safetensors"
    final.parent.mkdir(parents=True)
    final.write_bytes(previous)

    def fake_urlretrieve(_url, destination):
        pathlib.Path(destination).write_bytes(received)

    monkeypatch.setattr(fetcher.urllib.request, "urlretrieve", fake_urlretrieve)

    assert fetcher.fetch(row, str(tmp_path), False) is False
    assert final.read_bytes() == previous
    assert not pathlib.Path(str(final) + ".part").exists()


def test_list_never_hashes_multi_gigabyte_files(tmp_path, monkeypatch, capsys):
    fetcher = _load_fetcher()
    monkeypatch.setattr(fetcher, "models_root", lambda: str(tmp_path))
    monkeypatch.setattr(
        fetcher, "_sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("--list hashed a file")),
    )
    monkeypatch.setattr(fetcher.sys, "argv", ["otr_fetch_lane_weights.py", "--list"])

    assert fetcher.main() == 0
    assert "humo" in capsys.readouterr().out
