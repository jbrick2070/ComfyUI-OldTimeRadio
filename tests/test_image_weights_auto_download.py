"""Ideogram 4, Flux.1-dev and HuMo fetch their own weights at queue time.

Operator, 2026-09-29: "all of our models should be auto-download ability".
These engines refused on a fresh install with "missing: ..." and nothing able
to fetch the files. Each now asks its own adapter which files it will load,
and the queue-time preflight downloads exactly those, pinned by SHA-256.

Ideogram's nothing-installed pick depends on the card: nvfp4 runs only on
Blackwell, so any other NVIDIA card is handed the fp8 set.
"""
from __future__ import annotations

import pytest

from nodes import _otr_visual_assets as VA
from nodes._otr_image_engines import ideogram4_local as IDEO

_NVFP4_SET = {
    ("diffusion_models", "ideogram4_nvfp4_mixed.safetensors"),
    ("diffusion_models", "ideogram4_unconditional_nvfp4_mixed.safetensors"),
    ("text_encoders", "qwen3vl_8b_nvfp4.safetensors"),
    ("vae", "flux2-vae.safetensors"),
}
_FP8_SET = {
    ("diffusion_models", "ideogram4_fp8_scaled.safetensors"),
    ("diffusion_models", "ideogram4_unconditional_fp8_scaled.safetensors"),
    ("text_encoders", "qwen3vl_8b_fp8_scaled.safetensors"),
    ("vae", "flux2-vae.safetensors"),
}


@pytest.fixture
def no_ideogram_env(monkeypatch):
    for env_var, _names, _category in IDEO._ARTIFACTS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize("blackwell,expected", [(True, _NVFP4_SET), (False, _FP8_SET)])
def test_a_fresh_install_fetches_the_set_this_card_runs(monkeypatch, no_ideogram_env,
                                                        blackwell, expected):
    monkeypatch.setattr(IDEO, "_runs_nvfp4", lambda: blackwell)
    assert VA.planned_downloads({"ideogram4_local"}) == expected


def test_every_file_it_can_fetch_is_pinned():
    for key in _NVFP4_SET | _FP8_SET | {("checkpoints", "flux1-dev-fp8.safetensors")}:
        spec = VA.MANIFEST[key]
        assert set(spec) == {"repo_id", "filename", "revision", "size", "sha256"}, key


class _Shelf:
    """A folder lookup over a real directory holding exactly ``names`` (the
    planner opens what it is told is installed, so the files must exist)."""

    def __init__(self, root, names):
        self._root = root
        self._names = set(names)
        for category, token in self._names:
            (root / category).mkdir(parents=True, exist_ok=True)
            (root / category / token).write_bytes(b"weights")

    def get_full_path(self, category, token):
        path = self._root / category / token
        return str(path) if (category, token) in self._names else None

    def get_filename_list(self, category):
        return [token for cat, token in self._names if cat == category]

    def get_folder_paths(self, category):
        return [str(self._root / category)]


def test_an_installed_precision_wins_over_the_cards_default(monkeypatch, no_ideogram_env,
                                                            tmp_path):
    """A Blackwell card that already holds the fp8 set downloads nothing."""
    monkeypatch.setattr(IDEO, "_runs_nvfp4", lambda: True)
    requests = VA.native_requests({"ideogram4_local"},
                                  folder_paths=_Shelf(tmp_path, _FP8_SET),
                                  env={}, ideogram=IDEO)
    assert {(r["category"], r["token"]) for r in requests} == _FP8_SET
    assert all(r["path"] for r in requests)


def test_the_refusal_names_the_same_files_the_fetch_downloads(monkeypatch, no_ideogram_env,
                                                              tmp_path):
    monkeypatch.setattr(IDEO, "_runs_nvfp4", lambda: False)
    named = {(cat, name) for name, _verified, cat
             in IDEO.resolve_all_artifacts(_Shelf(tmp_path, ()))}
    assert named == VA.planned_downloads({"ideogram4_local"})


def test_flux_fetches_its_all_in_one_checkpoint(monkeypatch):
    monkeypatch.delenv("OTR_FLUX_CKPT", raising=False)
    assert VA.planned_downloads({"flux_gen1"}) == {
        ("checkpoints", "flux1-dev-fp8.safetensors")}


def test_a_flux_override_with_no_allowlisted_download_refuses(monkeypatch):
    monkeypatch.setenv("OTR_FLUX_CKPT", "some-other-flux.safetensors")
    with pytest.raises(VA.VisualAssetError, match="no allowlisted download"):
        VA.planned_downloads({"flux_gen1"})


_HUMO_SHARED = {
    ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    ("vae", "wan_2.1_vae.safetensors"),
    ("audio_encoders", "whisper_large_v3_fp16.safetensors"),
}
_HUMO_14B = _HUMO_SHARED | {
    ("diffusion_models", "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors"),
    ("loras", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
}
_HUMO_17B = _HUMO_SHARED | {("diffusion_models", "humo_1.7B_fp16.safetensors")}


@pytest.fixture
def no_humo_env(monkeypatch):
    for name in ("OTR_HUMO_CKPT", "OTR_HUMO_UNET_NAME", "OTR_HUMO_LORA_NAME",
                 "OTR_HUMO_CLIP_NAME", "OTR_HUMO_VAE_NAME", "OTR_HUMO_AUDIO_ENCODER_NAME",
                 "OTR_HUMO_17B_CKPT", "OTR_HUMO_17B_UNET_NAME", "OTR_HUMO_17B_LORA_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("engine,expected", [
    ("humo", _HUMO_14B), ("humo_14B_169", _HUMO_14B),
    ("humo_1.7B", _HUMO_17B), ("humo_1.7B_169", _HUMO_17B),
])
def test_each_humo_tier_fetches_exactly_what_its_loaders_open(no_humo_env, engine, expected):
    """The 1.7B tiers run LoRA-free, so they never pull the 14B LoRA or DiT."""
    assert VA.planned_downloads({engine}) == expected


@pytest.mark.parametrize("engine,env_var", [("humo", "OTR_HUMO_CKPT"),
                                            ("humo_1.7B", "OTR_HUMO_17B_CKPT")])
def test_a_humo_dit_pinned_by_a_path_the_loader_would_not_open_refuses(
        no_humo_env, monkeypatch, tmp_path, engine, env_var):
    """Composer QA on 0bc08d22: a full-path OTR_HUMO_*CKPT passes HuMo's own
    presence check while UNETLoader opens the basename through folder_paths.
    The preflight now compares the two before any download."""
    elsewhere = tmp_path / "elsewhere" / "my_humo.safetensors"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"weights")
    monkeypatch.setenv(env_var, str(elsewhere))
    from nodes import _otr_video_engines  # noqa: F401 -- registers built-ins
    from nodes._otr_video_engines import registry as vreg
    lane = vreg.get_engine(engine)
    lane = lane() if isinstance(lane, type) else lane
    with pytest.raises(VA.VisualAssetError, match="disagrees with native loader"):
        VA.native_requests({engine}, folder_paths=_Shelf(tmp_path / "models", ()),
                           env={env_var: str(elsewhere)}, humo={engine: lane})


def test_every_humo_file_is_pinned():
    for key in _HUMO_14B | _HUMO_17B:
        spec = VA.MANIFEST[key]
        assert set(spec) == {"repo_id", "filename", "revision", "size", "sha256"}, key
