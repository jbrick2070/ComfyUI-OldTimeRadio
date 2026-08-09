"""OTR_SilentComposite's upscale cache fingerprint -- the contract tests.

Before this file existed, ``OTR_SilentComposite.IS_CHANGED`` had ZERO coverage
while being the node's entire ComfyUI cache key.

HERMETICITY IS THE HARD PART, so read this before adding a case. The suite's
shared ``folder_paths`` stub (``tests/conftest.py``) provides only
``get_output_directory`` and ``get_filename_list`` -- it has NO
``get_folder_paths``. The adapter therefore takes its ``except Exception``
branch and falls through to the repo-relative ``parents[4]`` candidate, which
on a developer box is the REAL ``<comfy_root>/models/upscale_models/``. A test
that isolates only one of those two lookups silently reads the operator's
actual checkpoint and false-passes the moment they run
``ensure_upscale_models.py`` -- that is not hypothetical, it is the defect
commit 867f16c3 was filed for. The ``hermetic`` fixture below isolates BOTH.

INJECTION IS ``monkeypatch``, NEVER ``register()``. ``EngineRegistry`` has no
unregister, so registering a fake adapter permanently pollutes the
module-global registry and makes ``test_two_engines_ship_and_only_two`` and
``test_capabilities_roster_matches_registered`` fail depending on file order.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    """Isolate BOTH model-resolution paths into tmp_path.

    Returns an object exposing ``fp_dir`` (the directory the mocked
    ``folder_paths`` advertises) and ``fallback_dir`` (the directory the
    ``parents[4]`` repo-relative fallback resolves to). Neither is the real
    models dir, and no test here ever writes to the real one.
    """
    from nodes._otr_upscale_engines import eng_spandrel_esrgan as EM

    fp_dir = tmp_path / "fp" / "upscale_models"
    fp_dir.mkdir(parents=True)

    # A 5-deep fake tree so parents[4] lands on `fake_root`, mirroring
    # <comfy_root>/custom_nodes/<pack>/nodes/_otr_upscale_engines/<file>.py.
    # The file itself need not exist: resolve() is non-strict.
    fake_root = tmp_path / "comfy_root"
    fallback_dir = fake_root / "models" / "upscale_models"
    fallback_dir.mkdir(parents=True)
    fake_file = (fake_root / "custom_nodes" / "pack" / "nodes"
                 / "_otr_upscale_engines" / "eng_spandrel_esrgan.py")
    monkeypatch.setattr(EM, "__file__", str(fake_file))

    stub = types.ModuleType("folder_paths")
    stub.get_folder_paths = lambda _name: [str(fp_dir)]
    monkeypatch.setitem(sys.modules, "folder_paths", stub)

    ns = types.SimpleNamespace(fp_dir=fp_dir, fallback_dir=fallback_dir,
                               root=fake_root)
    return ns


@pytest.fixture
def engine(hermetic):
    """A SpandrelEsrgan instance built directly -- never via the registry."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    return SpandrelEsrgan()


@pytest.fixture(autouse=True)
def _reset_log_once():
    """The log-once cache is module state; a leaked key mutes a later test."""
    from nodes import otr_silent_composite as OSC
    OSC._FINGERPRINT_LOG_ONCE_CACHE.clear()
    yield
    OSC._FINGERPRINT_LOG_ONCE_CACHE.clear()


def _write_ckpt(directory: Path, engine, body: bytes = b"weights") -> Path:
    p = directory / engine._model_filename
    p.write_bytes(body)
    return p


def _is_changed(**kw):
    # Class is OTRSilentComposite; the node TYPE string is OTR_SilentComposite.
    from nodes.otr_silent_composite import OTRSilentComposite
    base = dict(base_video_path="", clip_manifest_json="{}",
                upscale_engine="off", upscale_device="cpu")
    base.update(kw)
    return OTRSilentComposite.IS_CHANGED(**base)


# ---------------------------------------------------------------------------
# 1-2. Stability and purity -- BUG-06.07's own verify recipe
# ---------------------------------------------------------------------------
def test_identical_inputs_produce_equal_keys(hermetic, engine):
    _write_ckpt(hermetic.fp_dir, engine)
    assert engine.model_fingerprint_parts() == engine.model_fingerprint_parts()


def test_parts_are_pure_across_load_state(hermetic, engine):
    """Not one byte of this may depend on device/descriptor residency."""
    _write_ckpt(hermetic.fp_dir, engine)
    before = engine.model_fingerprint_parts()
    engine.device = "cuda:0"           # simulate load() without spandrel
    engine._descriptor = object()
    during = engine.model_fingerprint_parts()
    engine.unload()
    after = engine.model_fingerprint_parts()
    assert before == during == after


# ---------------------------------------------------------------------------
# 3-5. What must move the key
# ---------------------------------------------------------------------------
def test_engine_and_device_identity_change_the_key(hermetic):
    off_cpu = _is_changed(upscale_engine="off", upscale_device="cpu")
    off_cuda = _is_changed(upscale_engine="off", upscale_device="cuda:0")
    assert off_cpu != off_cuda
    spandrel = _is_changed(upscale_engine="spandrel_esrgan",
                           upscale_device="cpu")
    assert spandrel != off_cpu


def test_mtime_change_moves_the_key(hermetic, engine):
    ckpt = _write_ckpt(hermetic.fp_dir, engine)
    before = engine.model_fingerprint_parts()
    st0 = os.stat(ckpt)
    # A large explicit delta: a same-size rewrite can land inside the
    # filesystem's timestamp granularity and observably not move at all.
    target = st0.st_mtime_ns - 10_000_000_000
    os.utime(ckpt, ns=(target, target))
    assert os.stat(ckpt).st_mtime_ns != st0.st_mtime_ns, (
        "filesystem did not honour the utime -- the assertion below would be "
        "vacuous")
    assert engine.model_fingerprint_parts() != before


def test_size_change_moves_the_key(hermetic, engine):
    ckpt = _write_ckpt(hermetic.fp_dir, engine, b"weights")
    before = engine.model_fingerprint_parts()
    ckpt.write_bytes(b"weights-but-longer")
    assert engine.model_fingerprint_parts() != before


def test_declared_digest_change_moves_the_key(hermetic, engine, monkeypatch):
    """The pin is a CLASS attribute reached through a registry singleton.

    monkeypatch.setattr auto-reverts; mutating the class directly would leak
    into every later test in the session (Bug Bible 12.81).
    """
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    _write_ckpt(hermetic.fp_dir, engine)
    before = engine.model_fingerprint_parts()
    monkeypatch.setattr(SpandrelEsrgan, "_model_sha256", "deadbeef" * 8)
    assert engine.model_fingerprint_parts() != before


# ---------------------------------------------------------------------------
# 6. Absence is stable, non-nan, and flips when the file lands
# ---------------------------------------------------------------------------
def test_absent_model_is_stable_and_not_nan(hermetic, engine):
    first = engine.model_fingerprint_parts()
    second = engine.model_fingerprint_parts()
    assert first == second, "an unstable absent-key defeats the cache forever"
    flat = repr(first)
    assert "nan" not in flat.lower()
    assert first[0][0] == "model_absent"


def test_absent_to_present_moves_the_key(hermetic, engine):
    absent = engine.model_fingerprint_parts()
    _write_ckpt(hermetic.fp_dir, engine)
    assert engine.model_fingerprint_parts() != absent


# ---------------------------------------------------------------------------
# 7. Candidate hygiene: ordered precedence, dedupe, absolute
# ---------------------------------------------------------------------------
def test_first_candidate_wins_and_loader_agrees(hermetic, engine, monkeypatch):
    """Fingerprint and loader must select the SAME file, not merely find one."""
    from nodes._otr_upscale_engines import eng_spandrel_esrgan as EM
    second = hermetic.root / "second" / "upscale_models"
    second.mkdir(parents=True)
    monkeypatch.setattr(
        sys.modules["folder_paths"], "get_folder_paths",
        lambda _n: [str(hermetic.fp_dir), str(second)])
    first_ckpt = _write_ckpt(hermetic.fp_dir, engine, b"first")
    _write_ckpt(second, engine, b"second")
    _write_ckpt(hermetic.fallback_dir, engine, b"fallback")

    candidates, resolved = engine._resolve_model()
    assert resolved == str(first_ckpt)
    assert candidates[0] == os.path.abspath(str(hermetic.fp_dir))
    parts = engine.model_fingerprint_parts()
    assert parts[0][2] == str(first_ckpt)


def test_candidates_are_absolute_and_deduplicated(hermetic, engine,
                                                  monkeypatch):
    """folder_paths routinely returns the very dir the fallback appends."""
    dupe = str(hermetic.fallback_dir)
    monkeypatch.setattr(
        sys.modules["folder_paths"], "get_folder_paths",
        lambda _n: [dupe, dupe.upper(), str(hermetic.fp_dir) + os.sep + "."])
    candidates, _ = engine._resolve_model()
    assert len(candidates) == len(set(os.path.normcase(c) for c in candidates))
    assert all(os.path.isabs(c) for c in candidates)


# ---------------------------------------------------------------------------
# 8. Unreadable is NOT absent -- the r4 correction
# ---------------------------------------------------------------------------
def _stat_raising(monkeypatch, target: str, exc: BaseException):
    """Make os.stat raise `exc` for exactly one path, real everywhere else."""
    real_stat = os.stat

    def fake_stat(path, *a, **kw):
        if os.path.normcase(str(path)) == os.path.normcase(target):
            raise exc
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(os, "stat", fake_stat)


def test_unreadable_model_propagates_rather_than_reporting_absent(
        hermetic, engine, monkeypatch):
    """An unreadable checkpoint is a FAULT, not absence, and must not be
    laundered into the stable absent marker.

    This works without any hand-written classification because pathlib's
    `_ignore_error()` whitelists only the not-found family: `Path.is_file()`
    re-raises PermissionError. The test guards that property -- if someone
    wraps the resolver in a broad `except OSError: return None`, an unreadable
    model silently becomes 'absent' and this goes red.
    """
    target = os.path.join(str(hermetic.fp_dir), engine._model_filename)
    _stat_raising(monkeypatch, target, PermissionError(13, "denied"))
    with pytest.raises(PermissionError):
        engine.model_fingerprint_parts()


def test_not_found_is_absence_not_a_fault(hermetic, engine, monkeypatch):
    """The other half of the same line: a not-found-class error must NOT
    propagate -- it is the ordinary 'nobody downloaded it yet' state and has
    to keep yielding the stable marker."""
    target = os.path.join(str(hermetic.fp_dir), engine._model_filename)
    _stat_raising(monkeypatch, target, FileNotFoundError(2, "nope"))
    parts = engine.model_fingerprint_parts()
    assert parts[0][0] == "model_absent"


def test_is_changed_returns_bare_nan_when_fingerprint_raises(hermetic,
                                                             monkeypatch):
    from nodes import otr_silent_composite as OSC

    class Exploding:
        name = "spandrel_esrgan"

        def model_fingerprint_parts(self):
            raise OSError("disk fell over")

    monkeypatch.setattr(OSC, "_get_upscale_engine", lambda _n: Exploding())
    got = _is_changed(upscale_engine="spandrel_esrgan")
    assert isinstance(got, float) and got != got, (
        "must be a BARE top-level nan -- a nan nested inside parts would be "
        "repr'd to the characters 'nan' and compare EQUAL, i.e. a permanent "
        "cache HIT")


# ---------------------------------------------------------------------------
# 9. The roster contract -- what actually makes the member 'required'
# ---------------------------------------------------------------------------
def test_every_registered_engine_exposes_the_member(hermetic):
    """Runs under the hermetic fixture on purpose: with the real filesystem an
    unreadable models dir could make this raise, which is now correct
    behaviour and would turn a contract test into a flake."""
    from nodes._otr_upscale_engines.registry import (
        all_engine_names, get_engine)
    for name in all_engine_names():
        eng = get_engine(name)
        fn = getattr(eng, "model_fingerprint_parts", None)
        assert callable(fn), f"{name} does not expose model_fingerprint_parts"
        parts = fn()
        assert isinstance(parts, tuple), f"{name} returned {type(parts)!r}"
        for item in parts:
            assert isinstance(item, tuple), (
                f"{name} returned a FLAT tuple; IS_CHANGED extends this, so a "
                f"flat shape splices scalars into the top level")
            for value in item:
                # A NaN nested in the parts tuple is the one shape that
                # inverts fail-open: repr() renders it as the characters
                # "nan", two such strings compare EQUAL, and the engine
                # would cache a stale composite forever. The docstrings warn
                # about it; this is what actually stops engine #3 doing it.
                assert not (isinstance(value, float) and value != value), (
                    f"{name} put a NaN inside its parts tuple; that repr's to "
                    f"a stable string and PERMANENTLY defeats invalidation. "
                    f"Raise instead and let IS_CHANGED return a bare nan.")


def test_off_engine_contributes_nothing():
    from nodes._otr_upscale_engines.registry import get_engine
    assert get_engine("off").model_fingerprint_parts() == ()


# ---------------------------------------------------------------------------
# 10 + 12. Subprocess gates -- fallback stub and cold import
# ---------------------------------------------------------------------------
def test_null_off_fallback_exposes_the_member_in_subprocess():
    """The import-fallback stub is only reachable when the namespace fails to
    import. Forcing that in-process via importlib.reload would rebind shared
    classes and registry state for every later test (Bug Bible 12.81), so it
    is proven in a child instead."""
    code = (
        "import sys;"
        "sys.modules['nodes._otr_upscale_engines.registry']=None;"
        "import nodes.otr_silent_composite as m;"
        "e=m._get_upscale_engine('anything');"
        "assert e.name=='off', e.name;"
        "assert e.model_fingerprint_parts()==(), e.model_fingerprint_parts();"
        "print('NULLOFF_OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "NULLOFF_OK" in r.stdout


def test_fingerprint_does_not_import_torch_or_spandrel():
    """In-process inspection would be vacuous: sibling upscale tests import
    torch at module scope, so it is already in sys.modules by then."""
    code = (
        "import sys;"
        "from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan;"
        "SpandrelEsrgan().model_fingerprint_parts();"
        "heavy=[m for m in ('torch','spandrel') if m in sys.modules];"
        "print('HEAVY',heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"


# ---------------------------------------------------------------------------
# 11. Log-once
# ---------------------------------------------------------------------------
def test_repeated_failures_log_once_but_always_return_nan(hermetic, monkeypatch,
                                                          caplog):
    from nodes import otr_silent_composite as OSC

    class Exploding:
        name = "spandrel_esrgan"

        def model_fingerprint_parts(self):
            raise OSError("disk fell over")

    monkeypatch.setattr(OSC, "_get_upscale_engine", lambda _n: Exploding())
    with caplog.at_level("WARNING", logger="OTR"):
        first = _is_changed(upscale_engine="spandrel_esrgan")
        second = _is_changed(upscale_engine="spandrel_esrgan")
    assert first != first and second != second, "both calls must fail open"
    hits = [r for r in caplog.records
            if "upscale model fingerprint unavailable" in r.getMessage()]
    assert len(hits) == 1, f"expected exactly one diagnostic, got {len(hits)}"
    assert "spandrel_esrgan" in hits[0].getMessage(), (
        "the engine id must appear, or a typo'd profile value is invisible")


def test_log_once_cache_is_bounded():
    from nodes import otr_silent_composite as OSC
    for i in range(OSC._FINGERPRINT_LOG_ONCE_MAX + 5):
        OSC._log_fingerprint_failure_once(f"engine_{i}", OSError("x"))
    assert len(OSC._FINGERPRINT_LOG_ONCE_CACHE) <= OSC._FINGERPRINT_LOG_ONCE_MAX


# ---------------------------------------------------------------------------
# 13. Shallow checkout must not IndexError on parents[4]
# ---------------------------------------------------------------------------
def test_shallow_checkout_does_not_index_error(tmp_path, monkeypatch):
    from nodes._otr_upscale_engines import eng_spandrel_esrgan as EM
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan

    shallow = Path(os.path.abspath(os.sep)) / "a" / "eng_spandrel_esrgan.py"
    assert len(shallow.resolve().parents) <= 4, (
        f"{shallow} is not actually shallow; the test proves nothing")
    monkeypatch.setattr(EM, "__file__", str(shallow))
    stub = types.ModuleType("folder_paths")
    stub.get_folder_paths = lambda _n: []
    monkeypatch.setitem(sys.modules, "folder_paths", stub)

    candidates, resolved = SpandrelEsrgan()._resolve_model()
    assert resolved is None
    assert candidates == ()


# ---------------------------------------------------------------------------
# 14. The canonical graph must not move
# ---------------------------------------------------------------------------
def test_canonical_node_84_widget_shape_unchanged():
    """The CANONICAL workflow only -- the 45 generated variants legitimately
    carry different values. Asserted directly because
    scripts/validate_canonical_workflow.py prints 'SKIPPED
    validate_workflow_contract' and returns no-problems when it cannot import
    NODE_CLASS_MAPPINGS, so its exit code alone proves nothing."""
    wf = json.loads((REPO_ROOT / "workflows" / "otr_canonical.json")
                    .read_text(encoding="utf-8"))
    composites = [n for n in wf["nodes"]
                  if n.get("type") == "OTR_SilentComposite"]
    assert len(composites) == 1, f"expected exactly one, got {len(composites)}"
    wv = composites[0]["widgets_values"]
    assert len(wv) == 7, f"widget count drifted: {wv!r}"
    assert wv[5] == "off" and wv[6] == "cpu", (
        f"upscale widgets moved off positions 5/6: {wv!r}")
