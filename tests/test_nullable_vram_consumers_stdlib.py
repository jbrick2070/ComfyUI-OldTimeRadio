"""Nullable peak receipts through real pure consumers; no GPU or media I/O.

The private namespace skips node registration initializers, not consumer code.
A persistent import boundary rejects non-stdlib dependencies. The render boundary
and fake engine implement rendering/resource I/O; actual shot, beat, episode, receipt,
rollup and credits-formatting functions execute unchanged.
"""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
ABSENT = object()


class ReceiptEngine:
    name = "nullable_telemetry_test"
    family = "abstract"
    render_canvas = (64, 64)

    def __init__(self, samples):
        self.samples = iter(samples)
        self.render_calls = 0
        self.teardown_calls = 0

    def session_identity(self):
        return (self.name, "fixture", "no-weights")

    def assert_usable(self, host_caps, profile, request_template=None):
        return True

    def prepare(self, host_caps, profile, session_ctx):
        return {"no_gpu_handles": True}

    def render_clip(self, request, prepared):
        self.render_calls += 1
        raw = {"path": "clip_%d.mp4" % self.render_calls,
               "frame_count": 10, "fps": 25,
               "native_frame_count": 10, "extension_mode": "none"}
        value = next(self.samples)
        if value is not ABSENT:
            raw["vram_peak_mb"] = value
        return raw

    def canonicalize(self, raw, request, profile):
        return dict(raw, type="video")

    def teardown(self, prepared):
        self.teardown_calls += 1


class NullableVramConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = "_nullable_vram_consumers_stdlib"
        if cls.namespace in sys.modules:
            raise AssertionError("Private telemetry namespace is already in use")
        cls.addClassCleanup(cls.cleanup_namespace)
        for name, path in ((cls.namespace, REPO / "nodes"),
                           (cls.namespace + "._otr_video_engines",
                            REPO / "nodes/_otr_video_engines")):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package

        class StdlibBoundary:
            def __init__(self):
                self.blocked = []

            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root != cls.namespace and root not in sys.stdlib_module_names:
                    self.blocked.append(fullname)
                    raise AssertionError("Non-stdlib import in telemetry test: " + fullname)
                return None

        cls.boundary = StdlibBoundary()
        sys.meta_path.insert(0, cls.boundary)
        cls.addClassCleanup(sys.meta_path.remove, cls.boundary)
        cls.rd = cls.load("_otr_video_engines.render_driver")
        cls.ws = cls.load("_otr_video_engines.wan_shared")
        cls.tmp = cls.load("_otr_video_engines._tmp")
        cls.host = cls.load("_otr_shared.host_caps")
        cls.batch = cls.load("otr_video_render_batch")
        cls.credits = cls.load("otr_credits_roll")
        cls.av = cls.load("_otr_video_engines.eng_ltx_av")
        if cls.boundary.blocked:
            raise AssertionError("Blocked imports: " + repr(cls.boundary.blocked))

    @classmethod
    def load(cls, name):
        return importlib.import_module(cls.namespace + "." + name)

    @classmethod
    def cleanup_namespace(cls):
        for name in list(sys.modules):
            if name == cls.namespace or name.startswith(cls.namespace + "."):
                del sys.modules[name]

    def setUp(self):
        self.boundary.blocked.clear()
        self.post_read = self.patch(self.rd._mc, "vram_used_mb", return_value=777)
        self.patch(self.host, "build_host_caps", return_value={})
        # These are hardware/resource-I/O boundaries, not aggregation logic.
        self.patch(self.rd, "_section_has_local_video_engine", return_value=False)
        self.patch(self.rd, "_should_reclaim_between_engines", return_value=False)
        self.patch(self.tmp, "otr_engine_tmp_path", return_value="assembled.mp4")
        self.assembly = self.patch(self.ws, "assemble_beat_segments",
                                   return_value="assembled.mp4")

    def tearDown(self):
        self.assertEqual(self.boundary.blocked, [], "A forbidden import was attempted")

    def patch(self, owner, name, **kwargs):
        patcher = mock.patch.object(owner, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def install(self, samples):
        engine = ReceiptEngine(samples)
        self.patch(self.rd._vreg, "is_registered", side_effect=lambda name: name == engine.name)
        self.patch(self.rd._vreg, "get_engine", return_value=engine)

        def render(engine_name, request, *, force_oom, host_caps=None,
                   profile=None, segment=None):
            if force_oom:
                raise self.rd.OomSignal("forced telemetry fixture OOM")
            if segment is not None:
                return engine.canonicalize(
                    engine.render_clip(request, segment.begin()), request, profile)
            prepared = engine.prepare(host_caps, profile, {})
            try:
                return engine.canonicalize(
                    engine.render_clip(request, prepared), request, profile)
            finally:
                engine.teardown(prepared)

        self.patch(self.rd, "_render_one", side_effect=render)
        return engine

    def shot(self, index=0, segments=None):
        shot = {"shot_id": "shot_%d" % index, "beat_id": "b%d" % index,
                "engine_id": ReceiptEngine.name, "family": "abstract",
                "role": "character_video", "target_frame_count": 10}
        if segments is not None:
            shot["coverage_plan"] = {
                "target_visible_frames": 10 * segments, "join_mode": "jump",
                "segments": [{"index": i, "render_frames": 10,
                              "drop_head": 0, "trim_tail": 0}
                             for i in range(segments)]}
        return shot

    @staticmethod
    def request(shot, ledger, *, canvas=None, segment_index=0):
        return {"shot_id": shot["shot_id"], "segment_index": segment_index,
                "frames": 10, "asset_refs": {}, "observability": {}}

    def check_shot(self, sample, expected):
        engine = self.install([sample])
        clip, _, _, peak = self.rd.render_shot(self.shot(), {})
        self.assertEqual(peak, expected)
        self.assertEqual(clip["receipt"]["vram_peak_mb"], expected)
        self.assertEqual(engine.render_calls, 1)
        self.assertEqual(engine.teardown_calls, 1)
        self.post_read.assert_not_called()

    def test_clip_missing_stays_unknown(self):
        self.check_shot(ABSENT, None)

    def test_clip_none_stays_unknown(self):
        self.check_shot(None, None)

    def test_clip_zero_is_a_measurement(self):
        self.check_shot(0, 0)

    def test_clip_positive_is_preserved(self):
        self.check_shot(12345, 12345)

    def check_beat(self, samples, expected):
        engine = self.install(samples)
        clip, _, _, peak = self.rd.render_beat_coverage(
            self.shot(segments=len(samples)), {}, request_builder=self.request)
        self.assertEqual(peak, expected)
        self.assertEqual(clip["vram_peak_mb"], expected)
        self.assertEqual(engine.render_calls, len(samples))
        self.assertEqual(engine.teardown_calls, 1)
        self.assertFalse(clip["vram_admission"]["enforced"])
        self.assembly.assert_called_once()
        self.post_read.assert_not_called()

    def test_beat_all_unknown_stays_unknown(self):
        self.check_beat([None, ABSENT], None)

    def test_beat_zero_survives_unknown(self):
        self.check_beat([None, 0], 0)

    def test_beat_reports_maximum_successful_observation(self):
        self.check_beat([9000, None, 1000], 9000)

    def check_episode(self, samples, expected):
        engine = self.install(samples)
        ledger = {"audio": {"frozen": "unchanged"},
                  "video": {"shots": [self.shot(i) for i in range(len(samples))]}}
        before = copy.deepcopy(ledger)
        episode = self.rd.run_episode(ledger, request_builder=self.request)
        self.assertEqual(episode["vram_peak_mb"], expected)
        self.assertEqual(ledger, before)
        self.assertEqual(episode["ledger"]["audio"], before["audio"])
        self.assertEqual(engine.render_calls, len(samples))
        self.post_read.assert_not_called()

    def test_episode_all_unknown_stays_unknown(self):
        self.check_episode([None, ABSENT], None)

    def test_episode_zero_survives_unknown(self):
        self.check_episode([None, 0], 0)

    def test_episode_reports_maximum_successful_observation(self):
        self.check_episode([9000, None, 1000], 9000)

    def test_empty_episode_is_unknown_not_zero(self):
        self.check_episode([], None)

    def test_render_error_still_raises_with_unknown_telemetry(self):
        self.install([None])
        with self.assertRaises(self.rd.RenderError), self.assertLogs(self.rd._LOG, level="ERROR"):
            self.rd.render_shot(self.shot(), {}, oom_engines={ReceiptEngine.name},
                                oom_shot_id="shot_0")
        self.post_read.assert_not_called()

    def test_adapter_raw_to_clip_preserves_optional_measurement(self):
        engine = self.av.LtxAudioInEngine()
        for value in (None, 0, 12345):
            with self.subTest(value=value):
                raw = {"out_path": "clip.mp4", "frame_count": 10,
                       "vram_peak_mb": value}
                clip = engine._clip_from_raw(raw, {"shot_id": "s1"})
                self.assertEqual(clip["vram_peak_mb"], value)

    def test_render_batch_rollup_preserves_numeric_and_unknown(self):
        cases = (([None, None], None), ([None, 0], 0),
                 ([9000, None, 1000], 9000), ([None, 4096.5], 4096.5))
        for samples, expected in cases:
            with self.subTest(samples=samples):
                receipts = [{"vram_peak_mb": value} for value in samples]
                self.assertEqual(self.batch._roll_up_engine_receipts(receipts)["vram_peak_mb"], expected)

    def test_credits_formats_unknown_and_real_zero_distinctly(self):
        self.assertEqual(self.credits._fmt_gib(None), "(unknown)")
        self.assertEqual(self.credits._fmt_gib(0), "0.0 GiB")
        self.assertEqual(self.credits._fmt_gib(8192), "8.0 GiB")


if __name__ == "__main__":
    unittest.main()
