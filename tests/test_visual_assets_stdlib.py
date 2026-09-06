"""Independent pure visual-asset planning contracts; no real model/network I/O.

Run directly with Python -B. Fixtures are tiny temporary ordinary files. The
bridge is loaded by filename, never through the OTR/ComfyUI package initializer.
"""
import builtins
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "_otr_visual_assets.py"
_original_import = builtins.__import__


def _stdlib_only_import(name, *args, **kwargs):
    if name.split(".")[0] in {
        "torch", "transformers", "diffusers", "comfy", "folder_paths",
        "huggingface_hub", "nodes",
    }:
        raise AssertionError("pure visual bridge imported runtime dependency: " + name)
    return _original_import(name, *args, **kwargs)


_spec = importlib.util.spec_from_file_location("_asset_bridge", MODULE_PATH)
bridge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bridge
with mock.patch("builtins.__import__", side_effect=_stdlib_only_import):
    _spec.loader.exec_module(bridge)


VIDEO_SLOTS = ("announcer_video_model", "music_video_model", "character_video_model")
IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
ROLE_TO_SLOT = {
    "announcer_visual": VIDEO_SLOTS[0],
    "music_visual": VIDEO_SLOTS[1],
    "character_video": VIDEO_SLOTS[2],
}
DEFAULT_UNET = "z_image_turbo_bf16.safetensors"
DEFAULT_CLIP = "qwen_3_4b.safetensors"
DEFAULT_VAE = "ae.safetensors"
DEFAULT_CKPT = "ltxv-2b-0.9.8-distilled.safetensors"
DEFAULT_T5 = "t5xxl_fp16.safetensors"


def canonical_prompt():
    selections = {key: "ltx098_low_video (16:9)" for key in VIDEO_SLOTS}
    selections.update({key: "z_image_turbo" for key in IMAGE_SLOTS})
    selections["gate_in"] = ["63", 0]
    return {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "1": {
            "class_type": "OTR_LedgerScriptWriter",
            "inputs": {"gate_in": ["63", 0], "replay_from": ""},
        },
        "87": {"class_type": "OTR_VideoDirector", "inputs": selections},
    }


def resolve_video(pick):
    bare = pick.split(" (", 1)[0]
    return "ltx_8gb" if bare == "ltx098_low_video" else bare


def freeze_video(video_slots):
    # Shared role_slots accepts both bare IDs and director-style dictionaries.
    return {
        role: (video_slots[slot]["engine_id"] if isinstance(video_slots[slot], dict)
               else video_slots[slot])
        for role, slot in ROLE_TO_SLOT.items()
    }


class NoNetworkTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(socket, "socket", side_effect=AssertionError("network forbidden"))
        patcher.start()
        self.addCleanup(patcher.stop)


class PromptPlanTests(NoNetworkTestCase):
    def plan(self, prompt=None, **kwargs):
        options = {"resolve_video": resolve_video, "freeze_video": freeze_video}
        options.update(kwargs)
        return bridge.plan_prompt(canonical_prompt() if prompt is None else prompt, "63", **options)

    def test_exact_canonical_deduplicates_and_preserves_prompt(self):
        prompt = canonical_prompt()
        before = copy.deepcopy(prompt)
        plan = self.plan(prompt)
        self.assertEqual(plan["engines"], {"ltx_8gb", "z_image_turbo"})
        self.assertFalse(plan["replay"])
        self.assertEqual(plan["skipped"], [])
        self.assertEqual(prompt, before)

    def test_resolver_receives_three_literal_video_picks(self):
        resolver = mock.Mock(side_effect=resolve_video)
        freezer = mock.Mock(side_effect=freeze_video)
        self.plan(resolve_video=resolver, freeze_video=freezer)
        calls = [call.args[0] for call in resolver.call_args_list]
        self.assertEqual(calls.count("ltx098_low_video (16:9)"), 3)
        self.assertNotIn("z_image_turbo", calls)
        slots = freezer.call_args.args[0]
        self.assertEqual(set(slots), set(VIDEO_SLOTS))
        self.assertEqual(set(freeze_video(slots).values()), {"ltx_8gb"})

    def test_only_direct_links_to_this_validator_are_scoped(self):
        prompt = canonical_prompt()
        prompt["88"] = copy.deepcopy(prompt["87"])
        prompt["88"]["inputs"]["gate_in"] = ["different-validator", 0]
        prompt["88"]["inputs"][VIDEO_SLOTS[0]] = ["dynamic-node", 0]
        prompt["2"] = copy.deepcopy(prompt["1"])
        prompt["2"]["inputs"].update(gate_in=["different-validator", 0], replay_from="other-replay")
        self.assertEqual(self.plan(prompt)["engines"], {"ltx_8gb", "z_image_turbo"})

    def test_indirect_reroute_does_not_become_this_gate_link(self):
        prompt = canonical_prompt()
        prompt["reroute"] = {"class_type": "Reroute", "inputs": {"value": ["63", 0]}}
        prompt["88"] = copy.deepcopy(prompt["87"])
        prompt["88"]["inputs"].update(gate_in=["reroute", 0], announcer_video_model=["dynamic", 0])
        self.assertEqual(self.plan(prompt)["engines"], {"ltx_8gb", "z_image_turbo"})

    def test_each_dynamic_model_pick_is_refused(self):
        for slot in VIDEO_SLOTS + IMAGE_SLOTS:
            with self.subTest(slot=slot):
                prompt = canonical_prompt()
                prompt["87"]["inputs"][slot] = ["dynamic-string-node", 0]
                with self.assertRaises(bridge.VisualAssetError):
                    self.plan(prompt)

    def test_full_replay_skips_live_widgets_and_resolvers(self):
        prompt = canonical_prompt()
        prompt["1"]["inputs"]["replay_from"] = "  frozen-fixture-bundle  "
        prompt["87"]["inputs"][VIDEO_SLOTS[0]] = ["unused-live-dynamic", 0]
        forbidden = mock.Mock(side_effect=AssertionError("live resolver called during replay"))
        plan = self.plan(prompt, resolve_video=forbidden, freeze_video=forbidden)
        self.assertTrue(plan["replay"])
        self.assertEqual(plan["engines"], set())
        forbidden.assert_not_called()

    def test_multiple_full_replay_writers_skip(self):
        prompt = canonical_prompt()
        prompt["1"]["inputs"]["replay_from"] = "bundle-one"
        prompt["2"] = copy.deepcopy(prompt["1"])
        prompt["2"]["inputs"]["replay_from"] = "bundle-two"
        plan = self.plan(prompt)
        self.assertTrue(plan["replay"])
        self.assertEqual(plan["engines"], set())

    def test_mixed_live_and_replay_writers_refused(self):
        prompt = canonical_prompt()
        prompt["2"] = copy.deepcopy(prompt["1"])
        prompt["2"]["inputs"]["replay_from"] = "frozen-fixture-bundle"
        with self.assertRaises(bridge.VisualAssetError):
            self.plan(prompt)

    def test_dynamic_replay_not_misclassified_as_frozen_replay(self):
        prompt = canonical_prompt()
        prompt["1"]["inputs"]["replay_from"] = ["dynamic-string-node", 0]
        with self.assertRaises(bridge.VisualAssetError):
            self.plan(prompt)

    def test_whitespace_replay_is_live(self):
        prompt = canonical_prompt()
        prompt["1"]["inputs"]["replay_from"] = "  \t  "
        self.assertFalse(self.plan(prompt)["replay"])

    def test_effective_frozen_route_drives_coverage_not_original_pick(self):
        freezer = lambda _slots: {role: "fixture_remote_video" for role in ROLE_TO_SLOT}
        plan = self.plan(freeze_video=freezer)
        self.assertNotIn("ltx_8gb", plan["engines"])
        self.assertIn("z_image_turbo", plan["engines"])
        self.assertTrue(plan["skipped"])

    def test_unsupported_literals_never_generate_download_requests(self):
        prompt = canonical_prompt()
        prompt["87"]["inputs"].update({slot: "fixture_unsupported_video" for slot in VIDEO_SLOTS})
        prompt["87"]["inputs"].update({slot: "fixture_unsupported_image" for slot in IMAGE_SLOTS})
        plan = self.plan(prompt)
        self.assertTrue(plan["skipped"])
        folders = mock.Mock()
        folders.get_full_path.side_effect = AssertionError("unsupported engine resolved a model")
        self.assertEqual(bridge.native_requests(plan["engines"], folder_paths=folders, env={}), [])


class FakeFolders:
    ALIASES = {"clip": "text_encoders", "unet": "diffusion_models"}

    def __init__(self, root):
        self.roots = {key: [root / key] for key in ("diffusion_models", "text_encoders", "vae", "checkpoints")}

    def get_folder_paths(self, category):
        return [str(path) for path in self.roots[self.ALIASES.get(category, category)]]

    def get_full_path(self, category, token):
        for folder in self.get_folder_paths(category):
            candidate = Path(folder) / token
            if candidate.is_file():
                return str(candidate)
        return None

    def put(self, category, token, payload=b"tiny model fixture", root_index=0):
        destination = Path(self.get_folder_paths(category)[root_index]) / token
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination


class _NativeFixtureCase(NoNetworkTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folders = FakeFolders(self.root / "primary models")
        self.unet = DEFAULT_UNET
        self.ckpt = DEFAULT_CKPT
        self.t5 = DEFAULT_T5
        self.zimage = SimpleNamespace(
            MODEL_ENV="OTR_ZIMAGE_UNET", CLIP_ENV="OTR_ZIMAGE_CLIP", VAE_ENV="OTR_ZIMAGE_VAE",
            _DEFAULT_UNET=DEFAULT_UNET, _DEFAULT_CLIP=DEFAULT_CLIP, _DEFAULT_VAE=DEFAULT_VAE,
            _resolve_unet_name=lambda: (self.unet, bool(self.folders.get_full_path("diffusion_models", self.unet))),
        )
        self.ltx = SimpleNamespace(
            _ckpt_name=lambda: self.ckpt,
            _ckpt_path=lambda: self.folders.get_full_path("checkpoints", self.ckpt),
            _t5_name=lambda: self.t5,
            _t5_path=lambda: self.folders.get_full_path("text_encoders", self.t5),
        )

    def requests(self, engines=None, **kwargs):
        options = {"folder_paths": self.folders, "zimage": self.zimage, "ltx": self.ltx, "env": {}}
        options.update(kwargs)
        return bridge.native_requests({"z_image_turbo", "ltx_8gb"} if engines is None else engines, **options)

    def install_defaults(self):
        return [self.folders.put(category, token) for category, token in (
            ("diffusion_models", DEFAULT_UNET), ("text_encoders", DEFAULT_CLIP),
            ("vae", DEFAULT_VAE), ("checkpoints", DEFAULT_CKPT), ("text_encoders", DEFAULT_T5),
        )]


class NativeRequestTests(_NativeFixtureCase):
    def test_empty_canonical_native_store_has_exact_five_manifest_requests(self):
        rows = self.requests()
        by_token = {row["token"]: row for row in rows}
        self.assertEqual(set(by_token), {DEFAULT_UNET, DEFAULT_CLIP, DEFAULT_VAE, DEFAULT_CKPT, DEFAULT_T5})
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["path"] is None and row["spec"] is not None for row in rows))
        self.assertEqual(by_token[DEFAULT_UNET]["category"], "diffusion_models")
        self.assertEqual(by_token[DEFAULT_UNET]["spec"]["repo_id"], "Comfy-Org/z_image_turbo")
        self.assertEqual(by_token[DEFAULT_UNET]["spec"]["filename"], "split_files/diffusion_models/" + DEFAULT_UNET)
        self.assertEqual(by_token[DEFAULT_CKPT]["spec"]["repo_id"], "Lightricks/LTX-Video")
        self.assertEqual(by_token[DEFAULT_T5]["spec"]["repo_id"], "comfyanonymous/flux_text_encoders")

    def test_existing_complete_store_is_preserved_and_native_paths_returned(self):
        installed = self.install_defaults()
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in installed}
        rows = self.requests()
        self.assertEqual({Path(row["path"]) for row in rows}, set(installed))
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in installed})

    def test_each_zimage_dependency_independently_missing(self):
        for missing in (DEFAULT_UNET, DEFAULT_CLIP, DEFAULT_VAE):
            with self.subTest(missing=missing):
                self.install_defaults()
                row = next(row for row in self.requests() if row["token"] == missing)
                Path(row["path"]).unlink()
                rows = self.requests({"z_image_turbo"})
                absent = [row for row in rows if row["path"] is None]
                self.assertEqual([row["token"] for row in absent], [missing])
                self.assertIsNotNone(absent[0]["spec"])

    def test_installed_unet_precision_is_preserved_on_simulated_ada_and_blackwell(self):
        for machine in ("Ada-4060", "Blackwell-5080"):
            for precision in ("fp8", "nvfp4", "bf16", "int8_convrot"):
                with self.subTest(machine=machine, precision=precision):
                    self.unet = "z_image_turbo_" + precision + ".safetensors"
                    selected = self.folders.put("diffusion_models", self.unet)
                    rows = self.requests({"z_image_turbo"})
                    unet = next(row for row in rows if row["category"] == "diffusion_models")
                    self.assertEqual(unet["token"], self.unet)
                    self.assertEqual(Path(unet["path"]), selected)
        # These fixtures test selection preservation, not CUDA compatibility.

    def test_native_path_authority_overrides_adapter_unverified_flag(self):
        installed = self.folders.put("diffusion_models", DEFAULT_UNET)
        self.zimage._resolve_unet_name = lambda: (DEFAULT_UNET, False)
        row = next(row for row in self.requests({"z_image_turbo"}) if row["token"] == DEFAULT_UNET)
        self.assertEqual(Path(row["path"]), installed)

    def test_adapter_verified_without_native_resolution_is_not_accepted(self):
        self.unet = "custom_unregistered.safetensors"
        self.zimage._resolve_unet_name = lambda: (self.unet, True)
        with self.assertRaises(bridge.VisualAssetError):
            self.requests({"z_image_turbo"})

    def test_missing_custom_zimage_overrides_are_not_replaced_with_defaults(self):
        for env_key in ("OTR_ZIMAGE_UNET", "OTR_ZIMAGE_CLIP", "OTR_ZIMAGE_VAE"):
            with self.subTest(env_key=env_key):
                if env_key == "OTR_ZIMAGE_UNET":
                    self.unet = "custom_missing.safetensors"
                else:
                    self.unet = DEFAULT_UNET
                with self.assertRaises(bridge.VisualAssetError):
                    self.requests({"z_image_turbo"}, env={env_key: "custom_missing.safetensors"})

    def test_explicit_unet_path_disagreement_refused(self):
        self.folders.put("diffusion_models", DEFAULT_UNET)
        outside = self.root / "other" / DEFAULT_UNET
        outside.parent.mkdir()
        outside.write_bytes(b"different explicit fixture")
        with self.assertRaises(bridge.VisualAssetError):
            self.requests({"z_image_turbo"}, env={"OTR_ZIMAGE_UNET": str(outside)})

    def test_existing_token_in_extra_registered_root_is_preserved(self):
        extra = self.root / "extra models"
        self.folders.roots["diffusion_models"].append(extra)
        self.unet = "z_image_turbo_fp8.safetensors"
        selected = self.folders.put("diffusion_models", self.unet, root_index=1)
        rows = self.requests({"z_image_turbo"})
        row = next(row for row in rows if row["category"] == "diffusion_models")
        self.assertEqual(row["token"], self.unet)
        self.assertEqual(Path(row["path"]), selected)

    def test_nested_only_zimage_discovery_is_refused_not_hidden_by_download(self):
        self.folders.put("diffusion_models", "nested/" + DEFAULT_UNET)
        # The actual older ZImage resolver drops the folder prefix but returns
        # verified=True. Its basename loader cannot see that nested-only file.
        self.zimage._resolve_unet_name = lambda: (DEFAULT_UNET, True)
        with self.assertRaises(bridge.VisualAssetError):
            self.requests({"z_image_turbo"})

    def test_native_aliases_for_clip_and_unet_existing_files(self):
        unet = self.folders.put("unet", DEFAULT_UNET)
        clip = self.folders.put("clip", DEFAULT_CLIP)
        rows = {row["token"]: row for row in self.requests({"z_image_turbo"})}
        self.assertEqual(Path(rows[DEFAULT_UNET]["path"]), unet)
        self.assertEqual(Path(rows[DEFAULT_CLIP]["path"]), clip)

    def test_first_registered_duplicate_basename_wins(self):
        self.folders.roots["diffusion_models"].append(self.root / "second root")
        first = self.folders.put("diffusion_models", DEFAULT_UNET, b"first", 0)
        second = self.folders.put("diffusion_models", DEFAULT_UNET, b"second", 1)
        row = next(row for row in self.requests({"z_image_turbo"}) if row["token"] == DEFAULT_UNET)
        self.assertEqual(Path(row["path"]), first)
        self.assertEqual(second.read_bytes(), b"second")

    def test_ltx_adapter_and_native_path_disagreement_refused(self):
        self.folders.put("checkpoints", DEFAULT_CKPT)
        outside = self.root / "other.ckpt"
        outside.write_bytes(b"different checkpoint fixture")
        self.ltx._ckpt_path = lambda: str(outside)
        with self.assertRaises(bridge.VisualAssetError):
            self.requests({"ltx_8gb"})

    def test_missing_custom_ltx_tokens_are_not_substituted(self):
        for slot in ("ckpt", "t5"):
            with self.subTest(slot=slot):
                self.ckpt, self.t5 = DEFAULT_CKPT, DEFAULT_T5
                setattr(self, slot, "custom_missing.safetensors")
                with self.assertRaises(bridge.VisualAssetError):
                    self.requests({"ltx_8gb"})

    def test_zero_byte_final_fails_closed_without_a_download_request(self):
        self.folders.put("diffusion_models", DEFAULT_UNET, b"")
        with self.assertRaises(bridge.VisualAssetError):
            self.requests({"z_image_turbo"})


class MetadataTests(NoNetworkTestCase):
    def setUp(self):
        super().setUp()
        self.source = {"repo_id": "Comfy-Org/z_image_turbo",
                       "filename": "split_files/vae/ae.safetensors"}
        self.record = {"commit_hash": "a" * 40, "etag": "b" * 64, "size": 19}
        self.hub_url = mock.Mock(side_effect=lambda repo, filename, **kw:
                                 "https://huggingface.co/%s/resolve/%s/%s" %
                                 (repo, kw.get("revision", "main"), filename))
        self.head = mock.Mock(side_effect=lambda *_args, **_kwargs: SimpleNamespace(**self.record))

    def pin(self, source=None):
        return bridge._pin_metadata(self.source if source is None else source,
                                    hf_hub_url=self.hub_url, get_hf_file_metadata=self.head)

    def test_public_metadata_pinned_and_rechecked_without_auth(self):
        metadata = self.pin()
        self.assertEqual(metadata["commit"], "a" * 40)
        self.assertEqual(metadata["sha256"], "b" * 64)
        self.assertEqual(metadata["size"], 19)
        self.assertIn("/resolve/" + "a" * 40 + "/", metadata["url"])
        self.assertEqual(self.head.call_count, 2)
        self.assertTrue(all(call.kwargs == {"token": False, "timeout": 30}
                            for call in self.head.call_args_list))
        self.assertTrue(all(call.kwargs["endpoint"] == "https://huggingface.co"
                            for call in self.hub_url.call_args_list))

    def test_nonallowlisted_repository_or_filename_refused_before_metadata(self):
        for source in (
            dict(self.source, repo_id="Other/remote"),
            dict(self.source, filename="other-model.safetensors"),
        ):
            with self.subTest(source=source):
                with self.assertRaises(bridge.VisualAssetError):
                    self.pin(source)
        self.head.assert_not_called()
        self.hub_url.assert_not_called()

    def test_unpinned_commit_is_refused(self):
        for commit in (None, "main", "a" * 39, "z" * 40):
            with self.subTest(commit_type=type(commit).__name__):
                self.record["commit_hash"] = commit
                with self.assertRaises(bridge.VisualAssetError):
                    self.pin()

    def test_git_blob_etag_bad_hash_and_nonexact_size_refused(self):
        original = self.record.copy()
        for key, value in (("etag", "b" * 40), ("etag", "z" * 64),
                           ("size", None), ("size", True), ("size", 1.0),
                           ("size", 0), ("size", -1)):
            with self.subTest(key=key, value_type=type(value).__name__):
                self.record = dict(original, **{key: value})
                with self.assertRaises(bridge.VisualAssetError):
                    self.pin()

    def test_second_head_must_match_first_commit_hash_and_size(self):
        for key, value in (("commit_hash", "c" * 40), ("etag", "d" * 64), ("size", 20)):
            with self.subTest(key=key):
                self.head.side_effect = [SimpleNamespace(**self.record),
                                         SimpleNamespace(**dict(self.record, **{key: value}))]
                with self.assertRaises(bridge.VisualAssetError):
                    self.pin()


class RuntimeBridgeTests(_NativeFixtureCase):
    @contextmanager
    def runtime(self, *, forbid_hf=False, cancel=None, metadata_size=None, after_download=None):
        prefix = "_visual_assets_runtime_fixture"
        payload = b"runtime fixture, never model weights"
        trace = SimpleNamespace(heads=[], downloads=[], cancel_calls=0)

        def cancel_check():
            trace.cancel_calls += 1
            if cancel is not None:
                return cancel(trace.cancel_calls)

        def module(name, **attrs):
            value = ModuleType(name)
            value.__path__ = []
            for key, item in attrs.items():
                setattr(value, key, item)
            return value

        def hub_url(repo, filename, **kwargs):
            return "https://huggingface.co/%s/resolve/%s/%s" % (
                repo, kwargs.get("revision", "main"), filename)

        def metadata(url, **kwargs):
            trace.heads.append((url, kwargs))
            return SimpleNamespace(commit_hash="a" * 40,
                                   etag=hashlib.sha256(payload).hexdigest(),
                                   size=len(payload) if metadata_size is None else metadata_size)

        def fake_download(spec, destination, pinned, **kwargs):
            trace.downloads.append((dict(spec), Path(destination), dict(pinned)))
            kwargs["cancel"]()
            kwargs["progress"](0, len(payload))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            kwargs["progress"](len(payload), len(payload))
            if after_download is not None:
                after_download(trace, destination)
            return {"status": "downloaded", "verified": True, "bytes_verified": len(payload)}

        modules = {
            prefix: module(prefix),
            prefix + "._otr_shared": module(prefix + "._otr_shared",
                route_freeze=SimpleNamespace(freeze_role_engines=freeze_video),
                env=SimpleNamespace(snapshot=lambda: {})),
            prefix + "._otr_shared.public_engines": module(prefix + "._otr_shared.public_engines",
                resolve_engine_id=resolve_video),
            prefix + "._otr_image_engines": module(prefix + "._otr_image_engines", z_image_turbo=self.zimage),
            prefix + "._otr_video_engines": module(prefix + "._otr_video_engines"),
            prefix + "._otr_video_engines.eng_ltx_8gb": module(prefix + "._otr_video_engines.eng_ltx_8gb",
                Ltx8gbEngine=lambda: self.ltx),
            prefix + "._otr_visual_asset_download": module(prefix + "._otr_visual_asset_download",
                download_verified=fake_download),
            "folder_paths": module("folder_paths", get_full_path=self.folders.get_full_path,
                                   get_folder_paths=self.folders.get_folder_paths),
            "comfy": module("comfy", model_management=SimpleNamespace(
                throw_exception_if_processing_interrupted=cancel_check)),
            "huggingface_hub": module("huggingface_hub", hf_hub_url=hub_url, get_hf_file_metadata=metadata),
        }

        def guard(name, *args, **kwargs):
            if forbid_hf and name in ("huggingface_hub", "_otr_visual_asset_download"):
                raise AssertionError("complete store entered download import path: " + name)
            if name.split(".")[0] in {"torch", "transformers", "diffusers"}:
                raise AssertionError("real model import forbidden: " + name)
            return _original_import(name, *args, **kwargs)

        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(bridge, "__package__", prefix), \
                mock.patch.object(bridge, "__spec__", importlib.util.spec_from_file_location(
                    prefix + "._otr_visual_assets", MODULE_PATH)), \
                mock.patch("builtins.__import__", side_effect=guard), \
                mock.patch.object(bridge, "_open_stream", side_effect=AssertionError("real transport forbidden")):
            yield trace

    def test_complete_store_never_imports_hf_or_downloader(self):
        installed = self.install_defaults()
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in installed}
        with self.runtime(forbid_hf=True) as trace:
            result = bridge.ensure_prompt_visual_assets(canonical_prompt(), "63")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["receipts"], [])
        self.assertEqual(trace.heads, [])
        self.assertEqual(trace.downloads, [])
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in installed})

    def test_missing_canonical_downloads_exact_five_then_native_rechecks(self):
        with self.runtime() as trace:
            result = bridge.ensure_prompt_visual_assets(canonical_prompt(), "63")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["receipts"]), 5)
        self.assertEqual(len(trace.heads), 10)
        self.assertEqual(len(trace.downloads), 5)
        self.assertEqual(len({destination for _, destination, _ in trace.downloads}), 5)
        self.assertTrue(all(row["path"] is not None for row in self.requests()))
        self.assertTrue(all(kwargs == {"token": False, "timeout": 30} for _, kwargs in trace.heads))

    def test_existing_nvfp4_is_unchanged_and_never_enters_download_path(self):
        self.install_defaults()
        self.unet = "z_image_turbo_nvfp4.safetensors"
        selected = self.folders.put("diffusion_models", self.unet)
        original = selected.read_bytes()
        with self.runtime(forbid_hf=True) as trace:
            result = bridge.ensure_prompt_visual_assets(canonical_prompt(), "63")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(trace.downloads, [])
        self.assertEqual(selected.read_bytes(), original)
        self.assertEqual(self.unet, "z_image_turbo_nvfp4.safetensors")

    def test_invalid_metadata_stops_before_any_transfer(self):
        with self.runtime(metadata_size=0) as trace:
            with self.assertRaises(bridge.VisualAssetError):
                bridge.ensure_prompt_visual_assets(canonical_prompt(), "63")
        self.assertEqual(trace.downloads, [])
        self.assertTrue(all(row["path"] is None for row in self.requests()))

    def test_cancellation_after_metadata_stops_before_transfer(self):
        interruption = RuntimeError("fixture Comfy interruption before transfer")
        def cancel(count):
            if count == 7:  # initial check + five metadata checks + first transfer
                raise interruption
        with self.runtime(cancel=cancel) as trace:
            with self.assertRaises(RuntimeError) as caught:
                bridge.ensure_prompt_visual_assets(canonical_prompt(), "63")
        self.assertIs(caught.exception, interruption)
        self.assertEqual(len(trace.heads), 10)
        self.assertEqual(trace.downloads, [])

    def test_post_download_selection_drift_stops_before_ready(self):
        def drift(trace, destination):
            if len(trace.downloads) == 5:
                self.unet = "z_image_turbo_nvfp4.safetensors"
                self.folders.put("diffusion_models", self.unet)
        with self.runtime(after_download=drift) as trace:
            with self.assertRaisesRegex(bridge.VisualAssetError, "selection changed"):
                bridge.ensure_prompt_visual_assets(canonical_prompt(), "63")
        self.assertEqual(len(trace.downloads), 5)

    def test_replay_runtime_does_not_fetch_live_visual_choices(self):
        prompt = canonical_prompt()
        prompt["1"]["inputs"]["replay_from"] = "frozen fixture bundle"
        with self.runtime(forbid_hf=True) as trace:
            result = bridge.ensure_prompt_visual_assets(prompt, "63")
        self.assertEqual(result["status"], "not-covered")
        self.assertEqual(trace.downloads, [])
        self.assertEqual(trace.heads, [])


class TransportBoundaryTests(NoNetworkTestCase):
    def test_http_error_closes_response_and_does_not_expose_signed_url(self):
        body = io.BytesIO(b"fixture forbidden response")
        url = "https://example.invalid/fixture?signature=private-test-value"
        failure = bridge.HTTPError(url, 403, "Forbidden", {}, body)
        opener = SimpleNamespace(open=mock.Mock(side_effect=failure))
        with mock.patch.object(bridge, "build_opener", return_value=opener):
            with self.assertRaises(bridge.VisualAssetError) as caught:
                with bridge._open_stream(url):
                    self.fail("HTTP error stream yielded")
        self.assertTrue(body.closed)
        self.assertIn("403", str(caught.exception))
        self.assertNotIn("example.invalid", str(caught.exception))
        self.assertNotIn("signature", str(caught.exception))
        self.assertNotIn("private-test-value", str(caught.exception))
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
