"""A half-downloaded sharded repo must not report as on-disk.

The old rule returned True on the FIRST nonzero weight file, so a multi-shard
repo interrupted after shard 1 looked present. That short-circuits the download
that would repair it, and load_llm has no network fallback -- the cache stays
poisoned until someone clears it by hand.

These tests build synthetic snapshot directories on a tmp path. Nothing here
downloads, loads a model, imports torch, or touches the real HF cache. Both
copies of the helper are exercised: `_otr_model_catalog` and its deliberate
twin in `_otr_hf_env`.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(mod_name: str, filename: str):
    """Import one nodes/ module by path, without importing the node package."""
    path = ROOT / "nodes" / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # dataclasses needs the module registered
    spec.loader.exec_module(module)
    return module


CATALOG = _load("_shardtest_catalog", "_otr_model_catalog.py")
HF_ENV = _load("_shardtest_hf_env", "_otr_hf_env.py")

IMPLEMENTATIONS = (("catalog", CATALOG), ("hf_env", HF_ENV))


def _write(path: Path, size: int) -> None:
    path.write_bytes(b"\0" * size)


def _write_index(snapshot: Path, shard_names, name="model.safetensors.index.json"):
    weight_map = {
        f"model.layers.{i}.weight": shard
        for i, shard in enumerate(shard_names)
    }
    (snapshot / name).write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8",
    )


class ShardCompletenessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.snapshot = Path(self._tmp.name) / "snap"
        self.snapshot.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_all(self, expected, msg):
        for label, mod in IMPLEMENTATIONS:
            with self.subTest(implementation=label):
                self.assertIs(
                    mod._snapshot_has_weights(self.snapshot), expected, msg)

    # -- the defect this file exists for ---------------------------------

    def test_sharded_repo_missing_a_shard_is_not_on_disk(self):
        _write_index(self.snapshot, ["a.safetensors", "b.safetensors"])
        _write(self.snapshot / "a.safetensors", 16)
        # b.safetensors never landed -- the interrupted-download case.
        self._assert_all(False, "one materialized shard is not a usable repo")

    def test_sharded_repo_with_zero_byte_shard_is_not_on_disk(self):
        _write_index(self.snapshot, ["a.safetensors", "b.safetensors"])
        _write(self.snapshot / "a.safetensors", 16)
        _write(self.snapshot / "b.safetensors", 0)
        self._assert_all(False, "a zero-length shard is not materialized")

    def test_complete_sharded_repo_is_on_disk(self):
        _write_index(self.snapshot, ["a.safetensors", "b.safetensors"])
        _write(self.snapshot / "a.safetensors", 16)
        _write(self.snapshot / "b.safetensors", 16)
        self._assert_all(True, "every declared shard present means present")

    def test_bin_index_is_honored_too(self):
        _write_index(
            self.snapshot, ["a.bin", "b.bin"],
            name="pytorch_model.bin.index.json")
        _write(self.snapshot / "a.bin", 16)
        self._assert_all(False, "the .bin index must be read as well")

    # -- the unsharded path must not regress -----------------------------

    def test_unsharded_single_file_still_on_disk(self):
        """gemma-4-12b-it and gemma-4-E2B-it are both single-file on the real
        4060 cache, so this is the path every currently cached row takes."""
        _write(self.snapshot / "model.safetensors", 16)
        self._assert_all(True, "an unsharded repo keeps the single-blob rule")

    def test_metadata_only_snapshot_is_not_on_disk(self):
        (self.snapshot / "config.json").write_text("{}", encoding="utf-8")
        self._assert_all(False, "config without weights was never on disk")

    def test_zero_byte_unsharded_weight_is_not_on_disk(self):
        _write(self.snapshot / "model.safetensors", 0)
        self._assert_all(False, "a zero-length weight is not materialized")

    # -- fail closed ------------------------------------------------------

    def test_unreadable_index_fails_closed(self):
        (self.snapshot / "model.safetensors.index.json").write_text(
            "{not json", encoding="utf-8")
        _write(self.snapshot / "a.safetensors", 16)
        self._assert_all(
            False, "a corrupt index is evidence of a half-finished pull")

    def test_index_without_weight_map_fails_closed(self):
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 1}}), encoding="utf-8")
        _write(self.snapshot / "a.safetensors", 16)
        self._assert_all(False, "an index naming nothing cannot be trusted")

    def test_missing_snapshot_dir_fails_closed(self):
        missing = Path(self._tmp.name) / "does-not-exist"
        for label, mod in IMPLEMENTATIONS:
            with self.subTest(implementation=label):
                self.assertIs(mod._snapshot_has_weights(missing), False)

    # -- the helper's own contract ---------------------------------------

    def test_unsharded_snapshot_declares_no_shards(self):
        _write(self.snapshot / "model.safetensors", 16)
        for label, mod in IMPLEMENTATIONS:
            with self.subTest(implementation=label):
                self.assertIsNone(mod._shards_named_by_index(self.snapshot))

    def test_declared_shard_set_is_deduplicated_by_filename(self):
        """A real index maps MANY tensors onto FEW shards; the helper must
        return the distinct filenames, not one entry per tensor."""
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "t0": "a.safetensors", "t1": "a.safetensors",
                "t2": "b.safetensors"}}), encoding="utf-8")
        for label, mod in IMPLEMENTATIONS:
            with self.subTest(implementation=label):
                self.assertEqual(
                    mod._shards_named_by_index(self.snapshot),
                    {"a.safetensors", "b.safetensors"})

    def test_both_implementations_agree_on_every_case(self):
        """The two copies are duplicated deliberately; they must not drift."""
        cases = []
        _write(self.snapshot / "model.safetensors", 16)
        cases.append(("unsharded", True))
        for name, expected in cases:
            with self.subTest(case=name):
                results = {
                    label: mod._snapshot_has_weights(self.snapshot)
                    for label, mod in IMPLEMENTATIONS
                }
                self.assertEqual(len(set(results.values())), 1, results)
                self.assertIs(results["catalog"], expected)


if __name__ == "__main__":
    unittest.main()
