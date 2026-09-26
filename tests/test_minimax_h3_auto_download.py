"""The MiniMax H3 lanes download their own weights at queue time, at a pin.

Operator 2026-09-25: "keep them as long as they are auto download"; "MiniMax 3
is popular, so people have it". Until 2026-09-26 the two H3 lanes were the only
local video lanes that a fresh install could not run: their ~59 GB was fetched
only by ``scripts/otr_fetch_lane_weights.py``, which ``.comfyignore`` strips
from the published bundle. Now ``_otr_visual_assets`` allowlists the five
files with their exact revision, size and SHA-256, and each lane asks for
its own rows.

What these tests pin:

* each lane plans exactly the files it loads, with the pinned revision + hash;
* the planned folder and name are what the lane's own resolver -- the question
  UNETLoader / CLIPLoader / VAELoader ask -- finds after the download lands;
* the dev-tree fetcher and the queue-time planner agree, byte for byte;
* the queue-time metadata check asks for THAT revision and nothing else;
* no other lane's plan, and no other source's pinning, changed.

Offline: nothing here touches the network, a GPU or a real models folder.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from nodes import _otr_visual_assets as VA

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ID = "Comfy-Org/MiniMax-H3"
REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"

FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

#: basename -> (revision, bytes, sha256), written out literally.
PINNED = {
    FL2VA: (REVISION, 20_970_379_616,
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"),
    REF2VA: (REVISION, 20_970_379_616,
             "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"),
    ENCODER: (REVISION, 15_687_142_551,
              "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
    VIDEO_VAE: (REVISION, 5_207_808_496,
                "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
    AUDIO_VAE: (REVISION, 605_254_808,
                "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
}

#: engine -> {(category, basename)} it must plan on an empty box.
EXPECTED = {
    "minimax_h3_video": {("diffusion_models", FL2VA),
                         ("text_encoders", ENCODER),
                         ("vae", VIDEO_VAE)},
    "minimax_h3_audio_in": {("diffusion_models", REF2VA),
                            ("text_encoders", ENCODER),
                            ("vae", VIDEO_VAE),
                            ("vae", AUDIO_VAE)},
}

#: engine -> the fetcher lane (and provisioner route) holding its files.
FETCH_LANE = {"minimax_h3_video": "minimax_h3_video",
              "minimax_h3_audio_in": "minimax_h3_audio_in"}

#: The env knobs these lanes and the snapshot lanes read. Cleared per test, so
#: an operator's shell cannot change what is being pinned.
_ENV_KNOBS = tuple(
    ["OTR_MINIMAX_H3_%s_%s" % (label, kind)
     for label in ("UNET", "CLIP", "VAE", "AUDIO_VAE") for kind in ("NAME", "DIR")]
    + ["OTR_LTX25_NATIVE_DIT", "OTR_LTX25_NATIVE_TE", "OTR_LTX25_VIDEO_VAE",
       "OTR_LTX25_AUDIO_VAE", "OTR_LTX25_UPSCALER", "OTR_SA3_CKPT",
       "OTR_SA3_TEXT_ENCODER"])


@pytest.fixture(autouse=True)
def _no_weight_env(monkeypatch):
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lanes():
    from nodes._otr_video_engines.eng_minimax_h3 import (
        MiniMaxH3AudioInEngine, MiniMaxH3VideoEngine)
    return {"minimax_h3_video": MiniMaxH3VideoEngine(),
            "minimax_h3_audio_in": MiniMaxH3AudioInEngine()}


# ---------------------------------------------------------------- the plan -- #
@pytest.mark.parametrize("engine", sorted(EXPECTED))
def test_each_lane_plans_exactly_its_weights_at_the_pinned_revision(engine):
    planned = VA.planned_downloads({engine})
    assert planned == EXPECTED[engine]
    for category, name in planned:
        spec = VA.MANIFEST[(category, name)]
        assert spec["repo_id"] == REPO_ID
        # The Hub layout IS the ComfyUI category layout for this repo.
        assert spec["filename"] == "%s/%s" % (category, name)
        assert (spec["revision"], spec["size"], spec["sha256"]) == PINNED[name]


def test_both_lanes_together_request_each_shared_file_once():
    requests = VA.native_requests(set(EXPECTED), folder_paths=VA._NothingInstalled,
                                  env={}, minimax_h3=_lanes())
    tokens = [(r["category"], r["token"]) for r in requests]
    assert len(tokens) == len(set(tokens)) == 5
    assert set(tokens) == EXPECTED["minimax_h3_video"] | EXPECTED["minimax_h3_audio_in"]
    assert all(r["path"] is None and r["spec"] for r in requests)


def test_every_registered_h3_lane_is_covered():
    """A third H3 adapter that loads weights must join the set, or nothing
    fetches for it and the dropdown matrix calls it manual again."""
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines.eng_minimax_h3 import _MiniMaxH3Base
    registered = set()
    for eid in vreg.all_engine_names():
        engine = vreg.get_engine(eid)
        cls = engine if isinstance(engine, type) else type(engine)
        if issubclass(cls, _MiniMaxH3Base):
            registered.add(eid)
    assert registered == set(VA._MINIMAX_H3_WEIGHT_ENGINES) == set(EXPECTED)
    assert VA._MINIMAX_H3_WEIGHT_ENGINES <= VA._COVERED


def test_an_unallowlisted_name_override_is_refused_not_substituted(monkeypatch):
    monkeypatch.setenv("OTR_MINIMAX_H3_UNET_NAME", "my_own_h3_dit.safetensors")
    with pytest.raises(VA.VisualAssetError, match="no allowlisted download"):
        VA.planned_downloads({"minimax_h3_video"})


# ------------------------------------------- the file the loader will read -- #
@pytest.mark.parametrize("engine", sorted(EXPECTED))
def test_a_downloaded_file_is_the_file_the_lane_loads(engine, tmp_path, monkeypatch):
    """Put each planned file exactly where the queue-time fetch writes it
    (``model_type_dir(category) / token``, the folder ComfyUI reads first) and
    the lane's own resolver -- ``folder_paths.get_full_path`` with the bare
    loader token, the question its UNETLoader / CLIPLoader / VAELoader ask --
    finds every weight there, and the planner then has nothing to fetch."""
    from nodes._otr_models_root import model_type_dir

    roots = {cat: tmp_path / "models" / cat
             for cat in ("diffusion_models", "text_encoders", "vae")}
    aliases = {"unet": "diffusion_models", "clip": "text_encoders"}

    def get_full_path(category, token):
        root = roots.get(aliases.get(category, category))
        if root is None or not (root / token).is_file():
            return None
        return str(root / token)

    def get_folder_paths(category):
        return [str(roots[aliases.get(category, category)])]

    folders = ModuleType("folder_paths")
    folders.get_full_path = get_full_path
    folders.get_folder_paths = get_folder_paths
    monkeypatch.setitem(sys.modules, "folder_paths", folders)

    lane = _lanes()[engine]
    lane._comfy_root = lambda: str(tmp_path / "no_comfy_root")
    before = VA.native_requests({engine}, folder_paths=folders, env={},
                                minimax_h3={engine: lane})
    assert {(r["category"], r["token"]) for r in before} == EXPECTED[engine]
    assert all(r["path"] is None for r in before)

    written = {}
    for r in before:
        destination = model_type_dir(r["category"], folder_paths=folders) / r["token"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"h3 fixture")
        written[r["token"]] = destination

    resolved = {token: path for token, path in lane._weight_paths().values()}
    assert set(resolved) == set(written)
    for token, path in resolved.items():
        assert pathlib.Path(path) == written[token], token
    assert lane._installed()

    after = VA.native_requests({engine}, folder_paths=folders, env={},
                               minimax_h3={engine: lane})
    assert all(r["path"] is not None for r in after)


# ------------------------------------------------ fetcher / planner parity -- #
def test_the_fetcher_lanes_are_the_planned_files_at_the_same_pins():
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_otr_h3_parity_fetcher")
    provision = _load("scripts/otr_provision.py", "_otr_h3_parity_provision")
    for engine, lane in FETCH_LANE.items():
        rows = [fetcher.weight_spec(r) for r in fetcher.LANES[lane]]
        keyed = {(r.destination.split("/", 1)[0], fetcher.destination_name(r)): r
                 for r in rows}
        assert set(keyed) == VA.planned_downloads({engine}), engine
        for key, row in keyed.items():
            spec = VA.MANIFEST[key]
            assert (row.repo, row.path_in_repo, row.revision,
                    row.expected_bytes, row.expected_sha256) == (
                spec["repo_id"], spec["filename"], spec["revision"],
                spec["size"], spec["sha256"]), key
        assert provision.lane_for_engine(engine, "video") == provision.Lane(lane, False)


def test_the_fetcher_reads_the_pins_rather_than_restating_them():
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_otr_h3_pins_fetcher")
    from_planner = {
        filename.rsplit("/", 1)[-1]: (repo, filename, "%s/%s" % (
            category, filename.rsplit("/", 1)[-1]), revision, size, sha256)
        for category, repo, filename, revision, size, sha256 in VA._PINNED_SOURCES}
    assert {name: tuple(spec) for name, spec in fetcher._PINNED.items()} == from_planner


def test_the_dropdown_matrix_reads_both_h3_rows_as_auto():
    matrix = _load("scripts/otr_dropdown_matrix.py", "_otr_h3_dropdown")
    matrix._GRAPH_FETCHED_CACHE.clear()
    facts = matrix.download_facts()
    assert matrix.friction_for("minimax_h3_video", "video", facts) == (
        "auto", 38.99, "minimax_h3_video")
    assert matrix.friction_for("minimax_h3_audio_in", "video", facts) == (
        "auto", 39.55, "minimax_h3_audio_in")


# ------------------------------------------------- the queue-time pin check -- #
class _Hub:
    """``hf_hub_url`` / ``get_hf_file_metadata`` stand-ins; no network."""

    def __init__(self, **record):
        self.record = record
        self.url = mock.Mock(side_effect=lambda repo, filename, **kw:
                             "https://huggingface.co/%s/resolve/%s/%s"
                             % (repo, kw.get("revision", "main"), filename))
        self.head = mock.Mock(side_effect=lambda *_a, **_kw:
                              SimpleNamespace(**self.record))

    def pin(self, spec):
        return VA._pin_metadata(spec, hf_hub_url=self.url,
                                get_hf_file_metadata=self.head)


def _h3_spec(name=ENCODER, category="text_encoders"):
    return dict(VA.MANIFEST[(category, name)])


def test_a_pinned_source_is_asked_for_its_own_revision_once():
    revision, size, sha256 = PINNED[ENCODER]
    hub = _Hub(commit_hash=revision, etag=sha256, size=size)
    metadata = hub.pin(_h3_spec())
    assert metadata == {
        "commit": revision, "sha256": sha256, "size": size,
        "url": "https://huggingface.co/%s/resolve/%s/text_encoders/%s"
               % (REPO_ID, revision, ENCODER)}
    assert hub.head.call_count == 1
    assert hub.head.call_args.kwargs == {"token": False, "timeout": 30}
    # HEAD is never asked: every URL built names the pinned revision.
    assert all(call.kwargs.get("revision") == revision
               for call in hub.url.call_args_list)


@pytest.mark.parametrize("field, value", [
    ("commit_hash", "c" * 40),
    ("commit_hash", None),
    ("etag", "d" * 64),
    ("size", PINNED[ENCODER][1] + 1),
    ("size", True),
    ("size", None),
])
def test_a_pinned_source_that_moved_is_refused(field, value):
    revision, size, sha256 = PINNED[ENCODER]
    record = {"commit_hash": revision, "etag": sha256, "size": size}
    record[field] = value
    with pytest.raises(VA.VisualAssetError, match="pinned revision"):
        _Hub(**record).pin(_h3_spec())


def test_a_forged_pin_is_refused_before_any_metadata_call():
    hub = _Hub(commit_hash=REVISION, etag="0" * 64, size=1)
    forged = dict(_h3_spec(), sha256="0" * 64, size=1)
    with pytest.raises(VA.VisualAssetError, match="not allowlisted"):
        hub.pin(forged)
    hub.head.assert_not_called()


# --------------------------------------------- nothing else's plan changed -- #
def test_only_the_h3_sources_carry_a_pin():
    """Every other row keeps the HEAD-at-queue-time pinning it had."""
    pinned = {(category, filename.rsplit("/", 1)[-1])
              for category, _repo, filename, *_rest in VA._PINNED_SOURCES}
    assert pinned == EXPECTED["minimax_h3_video"] | EXPECTED["minimax_h3_audio_in"]
    for key, spec in VA.MANIFEST.items():
        if key in pinned:
            assert set(spec) == {"repo_id", "filename", "revision", "size", "sha256"}
        else:
            assert set(spec) == {"repo_id", "filename"}, key


#: Measured on the tree before the H3 rows existed, identical after.
_UNCHANGED_PLANS = {
    "ltx25_video": {
        ("diffusion_models", "LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors"),
        ("text_encoders", "gemma4-12b-ltx25-comfy-w4a8.safetensors"),
        ("vae", "ltx-2.5-video-vae-bf16.safetensors"),
        ("vae", "ltx-2.5-audio-vae-bf16.safetensors"),
        ("latent_upscale_models",
         "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")},
    "ltx25_foley_blackwell": {
        ("diffusion_models", "LTX25-distilled-DiT-comfy-nvfp4.safetensors"),
        ("text_encoders", "gemma4-12b-ltx25-comfy-w4a8.safetensors"),
        ("vae", "ltx-2.5-video-vae-bf16.safetensors"),
        ("vae", "ltx-2.5-audio-vae-bf16.safetensors"),
        ("latent_upscale_models",
         "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")},
    "animatediff15_v3_haunted_video": {
        ("checkpoints", "v1-5-pruned-emaonly-fp16.safetensors"),
        ("animatediff_models", "v3_sd15_mm.ckpt"),
        ("loras", "v3_sd15_adapter.ckpt")},
    "stable_audio_3": {
        ("checkpoints", "stable_audio_3_small_music_base.safetensors"),
        ("text_encoders", "t5gemma_b_b_ul2.safetensors")},
    "ltx_8gb": {
        ("checkpoints", "ltxv-2b-0.9.8-distilled.safetensors"),
        ("text_encoders", "t5xxl_fp16.safetensors")},
}


@pytest.mark.parametrize("engine", sorted(_UNCHANGED_PLANS))
def test_an_existing_lane_plans_what_it_planned_before(engine):
    assert VA.planned_downloads({engine}) == _UNCHANGED_PLANS[engine]


def test_no_other_covered_engine_plans_an_h3_file():
    h3_files = EXPECTED["minimax_h3_video"] | EXPECTED["minimax_h3_audio_in"]
    for engine in sorted(VA._COVERED - VA._MINIMAX_H3_WEIGHT_ENGINES):
        try:
            planned = VA.planned_downloads({engine})
        except VA.VisualAssetError:
            continue          # a host-env pin this box carries; not H3's concern
        assert not planned & h3_files, engine
