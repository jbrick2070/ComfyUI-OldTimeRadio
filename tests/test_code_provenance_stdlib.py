"""PBUG-20260906-02: package credits without .git; no GPU/network imports."""
import ast
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tested_code_provenance", REPO / "nodes/_otr_code_provenance.py")
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)
OID = "12345678" + "a" * 32


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "package"
        (self.repo / "nodes").mkdir(parents=True)
        self.write("__init__.py", "# package entry\n")
        self.write("nodes/otr_credits_roll.py", "# credits source\n")

    def write(self, name, value):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def test_registry_package_is_explicit_source_not_commit(self):
        label, value = provenance.code_receipt(self.repo)
        self.assertEqual(label, "SOURCE:")
        self.assertEqual(value, "sha256 " + provenance.source_fingerprint(self.repo)[:16])
        self.assertFalse((self.repo / ".git").exists())

    def test_parent_comfy_git_is_never_attributed_to_package(self):
        parent_git = self.repo.parent / ".git"
        parent_git.mkdir()
        (parent_git / "HEAD").write_text(OID, encoding="utf-8")
        self.assertEqual(provenance.code_receipt(self.repo)[0], "SOURCE:")

    def test_regular_checkout_retains_eight_character_commit(self):
        self.write(".git/HEAD", "ref: refs/heads/v2.0-alpha\n")
        self.write(".git/refs/heads/v2.0-alpha", OID + "\n")
        self.assertEqual(provenance.code_receipt(self.repo), ("COMMIT:", OID[:8]))

    def test_detached_checkout(self):
        self.write(".git/HEAD", OID)
        self.assertEqual(provenance.code_receipt(self.repo), ("COMMIT:", OID[:8]))

    def test_sha256_git_checkout(self):
        self.write(".git/HEAD", "b" * 64)
        self.assertEqual(provenance.code_receipt(self.repo), ("COMMIT:", "bbbbbbbb"))

    def test_packed_refs_match_exact_name_not_suffix(self):
        self.write(".git/HEAD", "ref: refs/heads/main")
        self.write(".git/packed-refs", "# pack-refs\n" + "b" * 40
                   + " refs/heads/prefixrefs/heads/main\n" + OID + " refs/heads/main\n")
        self.assertEqual(provenance.code_receipt(self.repo), ("COMMIT:", OID[:8]))

    def test_worktree_common_directory_reference(self):
        metadata = self.repo.parent / "git-meta"
        worktree = metadata / "worktrees/qa"
        worktree.mkdir(parents=True)
        (worktree / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
        (worktree / "commondir").write_text("../..", encoding="utf-8")
        (metadata / "packed-refs").write_text(OID + " refs/heads/main\n", encoding="utf-8")
        self.write(".git", "gitdir: ../git-meta/worktrees/qa\n")
        self.assertEqual(provenance.code_receipt(self.repo), ("COMMIT:", OID[:8]))

    def test_corrupt_git_is_not_silently_reclassified_as_package(self):
        for head in ("", "not-a-hash", "abcd1234", "ref: refs/heads/missing",
                     "ref: refs/../outside", "ref: refs/heads/bad\\name"):
            with self.subTest(head=head):
                self.write(".git/HEAD", head)
                with self.assertRaises(provenance.CodeProvenanceError):
                    provenance.code_receipt(self.repo)

    def test_empty_git_directory_stays_error(self):
        (self.repo / ".git").mkdir()
        with self.assertRaises(provenance.CodeProvenanceError):
            provenance.code_receipt(self.repo)

    def test_invalid_worktree_pointer_stays_error(self):
        self.write(".git", "wrong-pointer")
        with self.assertRaises(provenance.CodeProvenanceError):
            provenance.code_receipt(self.repo)

    def test_required_source_cannot_be_absent(self):
        (self.repo / "nodes/otr_credits_roll.py").unlink()
        with self.assertRaises(provenance.CodeProvenanceError):
            provenance.code_receipt(self.repo)

    def test_hash_changes_with_python_bytes_and_name(self):
        initial = provenance.code_receipt(self.repo)
        path = self.write("nodes/helper.py", "VALUE = 1\n")
        added = provenance.code_receipt(self.repo)
        self.assertNotEqual(added, initial)
        self.write("nodes/helper.py", "VALUE = 2\n")
        changed = provenance.code_receipt(self.repo)
        self.assertNotEqual(changed, added)
        path.rename(path.with_name("renamed.py"))
        self.assertNotEqual(provenance.code_receipt(self.repo), changed)

    def test_cache_logs_and_workflow_are_not_claimed_as_python_source(self):
        before = provenance.code_receipt(self.repo)
        for path in ("otr_runtime.log", "nodes/__pycache__/module.pyc",
                     "workflows/otr_canonical.json", "models/model.safetensors"):
            self.write(path, "unrelated to Python source")
        self.assertEqual(provenance.code_receipt(self.repo), before)

    def test_read_error_is_loud(self):
        with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(provenance.CodeProvenanceError, "unreadable"):
                provenance.code_receipt(self.repo)

    def test_directory_read_error_is_loud(self):
        def denied(*args, onerror, **kwargs):
            onerror(PermissionError("cannot enumerate source"))
        with mock.patch.object(provenance.os, "walk", side_effect=denied):
            with self.assertRaisesRegex(provenance.CodeProvenanceError, "unreadable"):
                provenance.code_receipt(self.repo)

    def test_root_directory_read_error_is_loud(self):
        with mock.patch.object(Path, "iterdir", side_effect=PermissionError("cannot enumerate root")):
            with self.assertRaisesRegex(provenance.CodeProvenanceError, "unreadable"):
                provenance.code_receipt(self.repo)

    def assert_link_rejected(self, relative, mode):
        target = self.repo / relative
        original = Path.lstat
        def metadata(path):
            if path == target:
                return types.SimpleNamespace(st_mode=mode, st_file_attributes=0x400)
            return original(path)
        with mock.patch.object(Path, "lstat", metadata):
            with self.assertRaisesRegex(provenance.CodeProvenanceError, "Linked/reparse"):
                provenance.code_receipt(self.repo)

    def test_linked_source_file_refused(self):
        self.assert_link_rejected("nodes/otr_credits_roll.py", stat.S_IFLNK)

    def test_linked_source_directory_refused(self):
        (self.repo / "nodes/external").mkdir()
        self.assert_link_rejected("nodes/external", stat.S_IFDIR)

    def test_linked_nodes_root_refused(self):
        self.assert_link_rejected("nodes", stat.S_IFDIR)


class CreditsWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Namespace only: load real credits and real leaf dependencies without
        # importing the Comfy node registry or booting any model framework.
        package = types.ModuleType("_credits_provenance_tests")
        package.__path__ = [str(REPO / "nodes")]
        cls.namespace = package.__name__
        sys.modules[cls.namespace] = package
        spec = importlib.util.spec_from_file_location(
            cls.namespace + ".otr_credits_roll", REPO / "nodes/otr_credits_roll.py")
        cls.cr = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.cr
        spec.loader.exec_module(cls.cr)
        # Reuse the real existing credits fixture without importing pytest.
        tree = ast.parse((REPO / "tests/test_credits_roll_spec.py").read_text(encoding="utf-8"))
        fixture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_led")
        scope = {}
        exec(compile(ast.Module(body=[fixture], type_ignores=[]), "credits_fixture", "exec"), scope)
        cls.ledger = staticmethod(scope["_led"])

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == cls.namespace or name.startswith(cls.namespace + "."):
                del sys.modules[name]

    def layout(self):
        with mock.patch.object(self.cr, "_sys_specs", return_value={}):
            return self.cr.build_credits_layout(self.ledger(), w=1920, h=1080, manifest={"clips": []})

    def test_real_layout_accepts_packaged_code_identity_and_retains_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nodes").mkdir()
            (root / "__init__.py").write_text("# entry", encoding="utf-8")
            source = root / "nodes/otr_credits_roll.py"
            source.write_text("# fixture source", encoding="utf-8")
            with mock.patch.object(self.cr, "__file__", str(source)):
                # Do not resolve a linked individual file into another package.
                with mock.patch.object(Path, "resolve", side_effect=AssertionError("root redirected")):
                    layout = self.layout()
                trimmed = self.cr._abridge(layout, list(self.cr._LEDGER_DROP_ORDER))
            for candidate in (layout, trimmed):
                rows = [r for kind, block in candidate["col1"] if kind == "grid" for r in block["rows"]]
                self.assertIn(provenance.code_receipt(root), rows)
                self.assertNotIn("COMMIT:", [r[0] for r in rows])

    def test_git_checkout_row_stays_same(self):
        layout = self.layout()
        rows = [r for kind, block in layout["col1"] if kind == "grid" for r in block["rows"]]
        self.assertIn(provenance.code_receipt(REPO), rows)

    def test_other_required_ledger_receipts_still_fail_closed(self):
        for key in ("episode_title", "visual_style", "source_bank", "image_engines",
                    "render_engines", "music_engine"):
            led = self.ledger()
            del led["meta"][key]
            with self.subTest(key=key), mock.patch.object(self.cr, "_sys_specs", return_value={}):
                with self.assertRaises(self.cr.CreditsDataError):
                    self.cr.build_credits_layout(led, w=1920, h=1080, manifest={})

    def test_corrupt_source_raises_credits_error_at_real_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.cr, "__file__", str(Path(directory) / "nodes/otr_credits_roll.py")):
                with self.assertRaisesRegex(self.cr.CreditsDataError, "code provenance"):
                    self.layout()


if __name__ == "__main__":
    unittest.main()
