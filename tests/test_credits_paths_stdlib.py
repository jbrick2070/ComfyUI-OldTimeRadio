"""Windows credits-path regression tests; no real media/hardware I/O.

Run with Comfy's Python -B. --repo selects the development checkout. --baseline
reads HEAD:nodes/otr_credits_roll.py with git show (no checkout or writes), then
loads that actual prior module under the same isolated test namespace. Baseline
is expected to be red: the helper is absent and the real long-path roll fails.
Only the test runner's --baseline option invokes a subprocess; media subprocesses,
Pillow, ledger access, confinement and file I/O are faked in every test.
"""
import argparse
import ast
from contextlib import ExitStack
import importlib.util
import ntpath
from pathlib import Path
import posixpath
import subprocess
import sys
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
BASELINE = False
EPISODE = "signal_lost_the_ledger_of_the_gable_name_20260906_110815"
PARENT = (r"C:\Users\guest\AppData\Local\Comfy-Desktop\ComfyUI-Installs"
          r"\ComfyUI\ComfyUI\output\otr\episodes" + "\\" + EPISODE)
BODY = ntpath.join(PARENT, EPISODE + "_silent_procgen_blended_captioned.mp4")


def units(path):
    """Independent UTF-16 unit count, including astral characters."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in ntpath.abspath(path))


def legacy_names(body, path_module=ntpath):
    # Public pre-change naming contract, not the compaction algorithm.
    base, extension = path_module.splitext(body)
    extension = extension or ".mp4"
    return (base + "_credits" + extension, base + "_credits_backdrop.png",
            base + "_with_credits" + extension)


def downstream_names(paths, path_module=ntpath):
    # Independently enumerate the real downstream suffix consumers' outputs.
    clip, backdrop, joined = paths
    return (clip, backdrop, joined, clip + ".base.png", clip + ".scroll.png",
            joined + ".concat.txt", path_module.splitext(joined)[0] + "_final.mp4")


class MemoryFilesystem:
    def __init__(self, path_module=ntpath, platform="nt"):
        self.path_module = path_module
        self.platform = platform
        self.files = {}
        self.saved = []
        self.removed = []
        self.output_mode = "ok"

    def __getattr__(self, name):
        return getattr(self.path_module, name)

    def exists(self, path):
        return path in self.files

    def getsize(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return len(self.files[path])

    def save(self, path, content=b"fake nonempty media"):
        self.saved.append(path)
        if self.output_mode == "missing":
            return
        if self.platform == "nt" and units(path) > 250:
            # Model the observed rc=0 / no Python-visible output condition.
            return
        self.files[path] = b"" if self.output_mode == "zero" else content

    def remove(self, path):
        self.removed.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]


class CreditsPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = "_credits_paths_stdlib"
        if cls.namespace in sys.modules:
            raise AssertionError("Private credits-path namespace is already in use")
        cls.addClassCleanup(cls.cleanup_namespace)
        package = types.ModuleType(cls.namespace)
        package.__path__ = [str(REPO / "nodes")]
        sys.modules[cls.namespace] = package

        class StdlibBoundary:
            def __init__(self):
                self.blocked = []

            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root != cls.namespace and root not in sys.stdlib_module_names:
                    self.blocked.append(fullname)
                    raise AssertionError("Non-stdlib credits test import: " + fullname)
                return None

        cls.boundary = StdlibBoundary()
        sys.meta_path.insert(0, cls.boundary)
        cls.addClassCleanup(sys.meta_path.remove, cls.boundary)
        source_path = REPO / "nodes/otr_credits_roll.py"
        if BASELINE:
            result = subprocess.run(
                ["git", "-C", str(REPO), "show", "HEAD:nodes/otr_credits_roll.py"],
                check=True, capture_output=True, text=True, encoding="utf-8")
            source = result.stdout
        else:
            source = source_path.read_text(encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            cls.namespace + ".otr_credits_roll", source_path)
        cls.cr = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.cr
        exec(compile(source, str(source_path), "exec"), cls.cr.__dict__)

        # Load the real pure OBS identity predicate and suffix declaration only.
        # No mux module execution, state lookup, filesystem writes or API calls.
        mux_path = REPO / "nodes/otr_master_audio_mux.py"
        mux_tree = ast.parse(mux_path.read_text(encoding="utf-8"))
        selected = []
        for node in mux_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_stem_belongs_to_episode":
                selected.append(node)
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "_PIPELINE_SUFFIXES"
                    for target in node.targets):
                selected.append(node)
        if len(selected) != 2:
            raise AssertionError("Actual mux identity/suffix declarations were not found")
        scope = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(mux_path), "exec"), scope)
        cls.belongs_to_episode = staticmethod(scope["_stem_belongs_to_episode"])
        cls.obs_suffixes = scope["_PIPELINE_SUFFIXES"]
        if cls.boundary.blocked:
            raise AssertionError("Blocked imports: " + repr(cls.boundary.blocked))

    @classmethod
    def cleanup_namespace(cls):
        for name in list(sys.modules):
            if name == cls.namespace or name.startswith(cls.namespace + "."):
                del sys.modules[name]

    def setUp(self):
        self.boundary.blocked.clear()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.fs = MemoryFilesystem()
        self.commands = []
        self.body_bytes = b"original finished body -- never overwrite"
        self.fs.files[BODY] = self.body_bytes
        self.os_proxy = types.SimpleNamespace(name="nt", path=self.fs, remove=self.fs.remove)
        self.patch(self.cr, "os", self.os_proxy)
        self.patch(self.cr, "_ffmpeg_bin", return_value="ffmpeg-NOT-EXECUTED")
        self.patch(self.cr, "otr_proc", types.SimpleNamespace(run=mock.Mock(side_effect=self.fake_run)))
        self.patch(self.cr, "_probe_video", return_value={"w": 1920, "h": 1080, "fps": 25})
        self.patch(self.cr, "build_credits_layout", return_value={
            "hero": "fixture", "col1": [], "col3_flow": []})
        self.open_mock = self.stack.enter_context(mock.patch("builtins.open", mock.mock_open()))

        ledger = types.ModuleType(self.namespace + ".production_ledger")
        ledger.get_ledger = mock.Mock(return_value=types.SimpleNamespace(data={}))
        paths = types.ModuleType(self.namespace + "._otr_paths")
        paths.confine_to_output_tree = mock.Mock()
        paths.reject_remote_paths = mock.Mock()
        self.paths = paths
        self.stack.enter_context(mock.patch.dict(sys.modules, {
            ledger.__name__: ledger, paths.__name__: paths}))

        fs = self.fs
        class FakeImage:
            def __init__(self, width, height):
                self.width, self.height = width, height

            def convert(self, mode):
                return self

            def save(self, path):
                fs.save(path, b"fake raster")

            def paste(self, image, position):
                pass

        pil = types.ModuleType("PIL")
        image = types.ModuleType("PIL.Image")
        image.new = lambda mode, size, color: FakeImage(*size)
        pil.Image = image
        self.stack.enter_context(mock.patch.dict(sys.modules, {"PIL": pil, "PIL.Image": image}))
        self.patch(self.cr, "render_static_base", return_value=FakeImage(1920, 1080))
        self.patch(self.cr, "render_scroll_canvas", return_value=FakeImage(600, 1000))

    def tearDown(self):
        self.assertEqual(self.boundary.blocked, [], "An external dependency import was attempted")

    def patch(self, owner, name, *args, **kwargs):
        return self.stack.enter_context(mock.patch.object(owner, name, *args, **kwargs))

    def fake_run(self, command, **kwargs):
        self.commands.append(tuple(command))
        self.fs.save(command[-1])
        return types.SimpleNamespace(returncode=0, stderr="p41", stdout="")

    def assert_confined_and_bounded(self, paths, parent):
        for path in downstream_names(paths):
            with self.subTest(path=path):
                self.assertEqual(ntpath.dirname(path), parent)
                self.assertLessEqual(units(path), 250)
                self.assertFalse(path.startswith("\\\\?\\"))

    def test_exact_observed_263_unit_backdrop_regression(self):
        self.assertEqual(units(BODY), 246)
        self.assertEqual(units(legacy_names(BODY)[1]), 263)
        paths = self.cr._credits_artifact_paths(BODY)
        self.assertNotEqual(paths, legacy_names(BODY))
        self.assert_confined_and_bounded(paths, PARENT)

    def test_full_episode_identity_and_real_obs_suffix_contract_are_preserved(self):
        _, _, joined = self.cr._credits_artifact_paths(BODY)
        stem = ntpath.splitext(ntpath.basename(joined))[0]
        self.assertTrue(self.belongs_to_episode(stem, EPISODE))
        self.assertEqual(stem, EPISODE + "_captioned_with_credits")
        self.assertIn(stem[len(EPISODE):], self.obs_suffixes)

    def test_scratch_names_are_unique_but_joined_episode_identity_is_stable(self):
        first = self.cr._credits_artifact_paths(BODY)
        second = self.cr._credits_artifact_paths(BODY)
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(ntpath.splitext(first[0])[0], ntpath.splitext(first[1])[0])
        self.assert_confined_and_bounded(first, PARENT)
        self.assert_confined_and_bounded(second, PARENT)

    def test_short_windows_names_and_missing_extension_remain_unchanged(self):
        for body in (r"C:\out\ep\ep_silent.mp4", r"C:\out\ep\ep_silent",
                     r"C:\out\ep\ep_captioned.mkv"):
            with self.subTest(body=body):
                self.assertEqual(self.cr._credits_artifact_paths(body), legacy_names(body))

    def test_nonwindows_long_names_remain_unchanged(self):
        body = "/tmp/" + "p" * 210 + "/episode/episode_silent_captioned.mp4"
        self.os_proxy.name = "posix"
        self.fs.path_module = posixpath
        self.assertEqual(self.cr._credits_artifact_paths(body), legacy_names(body, posixpath))

    def test_unknown_long_basename_is_not_reassigned_to_parent_episode(self):
        body = ntpath.join(PARENT, "another_episode_" + "x" * 100 + "_captioned.mp4")
        self.assertEqual(self.cr._credits_artifact_paths(body), legacy_names(body))

    def test_overdeep_parent_preserves_identity_and_honest_legacy_fallback(self):
        parent = ntpath.join("C:\\", "deep" * 60, EPISODE)
        body = ntpath.join(parent, EPISODE + "_silent_procgen_blended_captioned.mp4")
        result = self.cr._credits_artifact_paths(body)
        self.assertEqual(result, legacy_names(body))
        self.assertTrue(any(units(path) > 250 for path in downstream_names(result)))
        self.assertTrue(all(ntpath.dirname(path) == parent for path in result))

    def test_astral_unicode_is_counted_as_utf16_not_python_characters(self):
        episode = "radio_" + "\U0001f600" * 20
        parent = ntpath.join("C:\\", "r" * 100, episode)
        body = ntpath.join(parent, episode + "_silent_procgen_blended_captioned.mp4")
        old_backdrop = legacy_names(body)[1]
        self.assertLess(len(old_backdrop), 250)
        self.assertGreater(units(old_backdrop), 250)
        result = self.cr._credits_artifact_paths(body)
        self.assertNotEqual(result, legacy_names(body))
        self.assert_confined_and_bounded(result, parent)
        self.assertTrue(ntpath.basename(result[2]).startswith(episode + "_"))

    def test_actual_roll_completes_observed_long_case_and_does_not_touch_body(self):
        # This remains a meaningful red test against HEAD's actual inline naming,
        # even when that source has no _credits_artifact_paths helper yet.
        out, tail, report_text = self.cr.OTRCreditsRoll().roll(BODY, '{"clips": []}')
        report = self.cr.json.loads(report_text)
        self.assertGreater(tail, 0)
        self.assertTrue(report["ok"])
        self.assertTrue(report["credits_rendered"])
        self.assertEqual(report["output"], out)
        self.assertEqual(self.fs.files[BODY], self.body_bytes)
        self.assertNotIn(BODY, self.fs.removed)
        self.assertTrue(self.fs.exists(out))
        self.paths.reject_remote_paths.assert_called_once_with(video_path=BODY)
        self.paths.confine_to_output_tree.assert_called_once_with(out, "video_path")
        self.assertEqual(len(self.commands), 3)  # extract, render, concat
        self.assertEqual(self.commands[0][self.commands[0].index("-i") + 1], BODY)
        self.assertTrue(all(command[-1] != BODY for command in self.commands))
        generated = self.fs.saved + self.fs.removed + [out]
        generated += [call.args[0] for call in self.open_mock.call_args_list]
        generated += [ntpath.splitext(out)[0] + "_final.mp4"]
        for path in generated:
            self.assertEqual(ntpath.dirname(path), PARENT)
            self.assertLessEqual(units(path), 250)

    def test_roll_calls_actual_path_helper_once(self):
        helper = self.cr._credits_artifact_paths
        with mock.patch.object(self.cr, "_credits_artifact_paths", wraps=helper) as selected:
            out, tail, _ = self.cr.OTRCreditsRoll().roll(BODY, '{}')
        selected.assert_called_once_with(BODY)
        self.assertNotEqual(out, BODY)
        self.assertGreater(tail, 0)

    def test_zero_or_missing_rc_zero_backdrop_remains_error(self):
        for mode in ("zero", "missing"):
            with self.subTest(mode=mode):
                self.fs.output_mode = mode
                self.commands.clear()
                target = ntpath.join(PARENT, "missing_backdrop.png")
                self.fs.files.pop(target, None)
                with self.assertRaisesRegex(self.cr.CreditsDataError, r"backdrop.*rc=0.*p41"):
                    self.cr.extract_final_frame(BODY, target)
                self.assertEqual(len(self.commands), 2)
                self.assertIn("-sseof", self.commands[0])
                self.assertNotIn("-sseof", self.commands[1])

    def test_zero_or_missing_rc_zero_credits_render_remains_error(self):
        for mode in ("zero", "missing"):
            with self.subTest(mode=mode):
                self.fs.output_mode = mode
                out = ntpath.join(PARENT, "test_credits.mp4")
                self.fs.files.pop(out, None)
                with self.assertRaisesRegex(self.cr.CreditsDataError, r"console render failed.*rc=0.*p41"):
                    self.cr.render_credits_clip({"col3_flow": []}, "backdrop.png", out,
                                                w=1920, h=1080, fps=25)

    def test_zero_or_missing_rc_zero_concat_remains_error(self):
        credit = ntpath.join(PARENT, "test_credit.mp4")
        self.fs.files[credit] = b"existing credits"
        for mode in ("zero", "missing"):
            with self.subTest(mode=mode):
                self.fs.output_mode = mode
                out = ntpath.join(PARENT, "joined.mp4")
                self.fs.files.pop(out, None)
                with self.assertRaisesRegex(self.cr.CreditsDataError, r"concat.*rc=0.*p41"):
                    self.cr.append_credits(BODY, credit, out)

    def test_presentation_failure_returns_unchanged_body_zero_tail_and_false_receipt(self):
        self.fs.output_mode = "missing"
        with self.assertLogs(self.cr.log, level="ERROR"):
            out, tail, text = self.cr.OTRCreditsRoll().roll(BODY, '{}')
        report = self.cr.json.loads(text)
        self.assertEqual((out, tail), (BODY, 0.0))
        self.assertFalse(report["ok"])
        self.assertFalse(report["credits_rendered"])
        self.assertEqual(report["reason"], "presentation_failure")
        self.assertIn("rc=0", report["error"])
        self.assertEqual(self.fs.files[BODY], self.body_bytes)
        self.assertNotIn(BODY, self.fs.removed)

    def test_unknown_long_input_still_fails_presentation_honestly(self):
        body = ntpath.join(PARENT, "unrecognized_" + "x" * 100 + ".mp4")
        self.fs.files[body] = self.body_bytes
        with self.assertLogs(self.cr.log, level="ERROR"):
            out, tail, text = self.cr.OTRCreditsRoll().roll(body, '{}')
        self.assertEqual((out, tail), (body, 0.0))
        self.assertEqual(self.cr.json.loads(text)["reason"], "presentation_failure")
        self.assertEqual(self.fs.files[body], self.body_bytes)

    def test_overdeep_parent_roll_returns_original_body_with_false_receipt(self):
        parent = ntpath.join("C:\\", "deep" * 60, EPISODE)
        body = ntpath.join(parent, EPISODE + "_silent_procgen_blended_captioned.mp4")
        self.fs.files[body] = self.body_bytes
        with self.assertLogs(self.cr.log, level="ERROR"):
            out, tail, text = self.cr.OTRCreditsRoll().roll(body, '{}')
        report = self.cr.json.loads(text)
        self.assertEqual((out, tail), (body, 0.0))
        self.assertFalse(report["ok"])
        self.assertFalse(report["credits_rendered"])
        self.assertEqual(report["reason"], "presentation_failure")
        self.assertEqual(self.fs.files[body], self.body_bytes)
        self.assertNotIn(body, self.fs.removed)

    def test_confinement_failure_stays_terminal_before_media_writes(self):
        self.paths.confine_to_output_tree.side_effect = ValueError("outside output tree")
        with self.assertRaisesRegex(ValueError, "outside output tree"):
            self.cr.OTRCreditsRoll().roll(BODY, '{}')
        self.assertEqual(self.commands, [])
        self.assertEqual(self.fs.saved, [])

    def test_bad_manifest_and_ledger_truth_stay_terminal(self):
        with self.assertRaises(self.cr.CreditsDataError):
            self.cr.OTRCreditsRoll().roll(BODY, 'not JSON')
        self.assertEqual(self.commands, [])
        with mock.patch.object(self.cr, "build_credits_layout",
                               side_effect=self.cr.CreditsDataError("missing ledger receipt")):
            with self.assertRaisesRegex(self.cr.CreditsDataError, "missing ledger receipt"):
                self.cr.OTRCreditsRoll().roll(BODY, '{}')
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--baseline", action="store_true")
    options, remainder = parser.parse_known_args()
    REPO = options.repo.resolve()
    BASELINE = options.baseline
    print("Credits source: " + ("git HEAD" if BASELINE else "development working tree"))
    unittest.main(argv=[sys.argv[0], *remainder])

