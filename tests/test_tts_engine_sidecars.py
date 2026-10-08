"""Tests for the chatterbox Path-B sidecar and the adapter-metadata refactor
that replaced the ``_OTR_CLONE_ENGINES`` name tuple (2026-06-05).

Headless-safe: never imports the chatterbox library or spawns a worker.
Exercises the registry, adapter metadata, fail-closed ``load()``, the pure worker
helpers, the bank mirror, and the C-5 import-safety property.
"""
import importlib.util
import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script(name):
    """Import a scripts/<name> module by path (side-effect-free at import:
    the workers do the fd dance + heavy imports inside main(), not at module
    load, so this never pulls torch / chatterbox)."""
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- registry + metadata --------------------------------------------------- #
def test_chatterbox_registered_with_roles():
    from nodes import _otr_audio_engines as AE
    cbx = AE.get_engine("chatterbox")
    assert "char_voice" in cbx.roles
    assert cbx.requires_flag is None  # C6: registry IS the menu (no flag gate)
    assert cbx.commercial_clean is True
    assert cbx.sample_rate == 24000


def test_clone_engines_declare_ref_metadata():
    # NO-FALLBACK (2026-07-03): clone engines still REQUIRE a ref WAV, but a
    # missing ref now FAILS LOUD in the dispatch -- there is NO bark fallback, so
    # missing_ref_fallback is None on every cloning engine.
    from nodes import _otr_audio_engines as AE
    for name in ("indextts2", "chatterbox"):
        e = AE.get_engine(name)
        assert e.requires_voice_ref is True
        assert e.voice_ref_kind == "wav_path"
        assert e.missing_ref_fallback is None


def test_non_clone_engines_do_not_require_ref():
    from nodes import _otr_audio_engines as AE
    for name in ("bark", "kokoro"):
        e = AE.get_engine(name)
        assert getattr(e, "requires_voice_ref", False) is False
        assert getattr(e, "missing_ref_fallback", None) is None


def test_requires_voice_ref_implies_voice_ref_kind():
    # A clone engine that forgets voice_ref_kind is a config bug -> catch it here.
    from nodes import _otr_audio_engines as AE
    for role in ("char_voice", "announcer_voice", "music"):
        for name in AE.engines_for_role(role):
            e = AE.get_engine(name)
            if getattr(e, "requires_voice_ref", False):
                assert getattr(e, "voice_ref_kind", None), name


# --- fail-closed gating + load() ------------------------------------------- #
def test_optin_engine_selectable_no_flag(monkeypatch):
    # C6 -- registry IS the menu: chatterbox is selectable with NO flag gate
    # (the venv/weights checks run in load(), not assert_usable).
    from nodes import _otr_audio_engines as AE
    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    assert AE.assert_usable("chatterbox", "char_voice") == "chatterbox"


def test_optin_engine_usable_when_flagged(monkeypatch):
    from nodes import _otr_audio_engines as AE
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    assert AE.assert_usable("chatterbox", "char_voice") == "chatterbox"


def test_load_fails_closed_when_not_installed(monkeypatch):
    from nodes import _otr_audio_engines as AE
    # Point the sidecar at a venv python that does not exist -> NAMED error.
    monkeypatch.setenv("OTR_CHATTERBOX_VENV", str(REPO_ROOT / "nope" / "python.exe"))
    with pytest.raises(RuntimeError) as ei:
        AE.get_engine("chatterbox").load()
    assert "Chatterbox Path B not installed" in str(ei.value)


# --- C-5 import safety ------------------------------------------------------ #
def test_import_engines_pulls_no_sidecar_library():
    # Importing the registry must NOT import the chatterbox library (it is
    # imported only inside the isolated worker subprocess).
    import nodes._otr_audio_engines  # noqa: F401
    assert "chatterbox" not in sys.modules, "chatterbox imported at registry import (C-5)"


# --- pure worker helpers (no GPU, no model) -------------------------------- #
def test_chatterbox_worker_supported_kwargs_drops_unknown():
    w = _load_script("_otr_chatterbox_worker.py")

    def fn(text, audio_prompt_path=None, exaggeration=0.5):  # no cfg/temperature
        return None

    got = w._supported_kwargs(fn, audio_prompt_path="r", exaggeration=0.7,
                              cfg_weight=0.4, temperature=0.6)
    assert got == {"audio_prompt_path": "r", "exaggeration": 0.7}


# --- bank mirror ----------------------------------------------------------- #
def test_bank_has_a_chatterbox_pool_mirroring_indextts2():
    from nodes._otr_voice_bank import load_voice_bank
    bank, _ = load_voice_bank()
    cbx = [e for e in bank if e.engine == "chatterbox" and "char_voice" in e.roles]
    idx = [e for e in bank if e.engine == "indextts2" and "char_voice" in e.roles]
    # FLOOR, not an exact count: the bank GROWS as public-domain voices are
    # added (scripts/otr_ingest_pd_voices.py). The original mirrored pools were
    # 36 each.
    #
    # LOWERED 36 -> 20 ON 2026-08-20, and only because the OPERATOR SHRANK THE
    # BANK ON PURPOSE. He auditioned all 63 donor references and retired 21 of
    # them as dupes or voices he does not want, and every char_voice pool went
    # from 41 to 20. The mirrored-pool INVARIANT is what this test actually
    # guards: chatterbox carries the same references as indextts2 (it is
    # generated from them by scripts/_otr_mirror_clone_refs.py), so a cast that
    # works on one engine works on the other.
    #
    # THE FLOOR IS NOT A TARGET. If a future change drops the pool below 20
    # without a matching operator ruling, that is a regression and this is where
    # it surfaces. Raise it again when public-domain ingestion grows the bank.
    #
    # WORTH KNOWING FOR CASTING: the surviving split is 13 male / 7 female per
    # engine. Seven female char voices is thin for an episode that casts several
    # women, and repeats will be audible before male repeats are.
    assert len(cbx) >= 20
    assert {e.ref_path for e in cbx} == {e.ref_path for e in idx}, (
        "the pools must stay MIRRORED -- chatterbox %d vs indextts2 %d"
        % (len(cbx), len(idx)))
    assert all(e.roles == ("char_voice",) for e in cbx)


def test_caster_assigns_a_chatterbox_voice():
    from nodes._otr_voice_bank import assign_voice_for_slot, load_voice_bank
    bank, _ = load_voice_bank()
    e = assign_voice_for_slot(role="char_voice", engine="chatterbox",
                              char_id="c1", gender="female", bank=bank)
    assert e.engine == "chatterbox" and e.gender == "female"


# --- sidecar lifecycle helpers (polish round: bounded read + teardown) ------ #
class _FakeStdout:
    def __init__(self, line=None, block=False):
        self._line, self._block = line, block

    def readline(self):
        if self._block:
            import time
            time.sleep(5)
            return "late\n"
        return self._line


class _FakeProc:
    def __init__(self, line=None, block=False, alive=False):
        self.stdout = _FakeStdout(line, block)
        self.stdin = None
        self._alive = alive
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        return 0


class _FakeStderr:
    def __init__(self):
        self.closed_flag = False

    def close(self):
        self.closed_flag = True


def test_read_protocol_line_returns_and_eof():
    from nodes._otr_audio_engines import _otr_sidecar as SC
    assert SC.read_protocol_line(_FakeProc(line="hi\n"), 2.0, "x") == "hi\n"
    with pytest.raises(EOFError):
        SC.read_protocol_line(_FakeProc(line=""), 2.0, "x")


def test_read_protocol_line_times_out():
    from nodes._otr_audio_engines import _otr_sidecar as SC
    with pytest.raises(TimeoutError):
        SC.read_protocol_line(_FakeProc(block=True), 0.2, "x")


def test_close_worker_always_closes_stderr():
    from nodes._otr_audio_engines import _otr_sidecar as SC
    # live proc with no stdin -> kill path; stderr closed.
    proc, sd = _FakeProc(alive=True), _FakeStderr()
    SC.close_worker(proc, sd)
    assert proc.killed and sd.closed_flag
    # proc is None (load() never set it) -> stderr STILL closed (no leak).
    sd2 = _FakeStderr()
    SC.close_worker(None, sd2)
    assert sd2.closed_flag


def test_remove_quietly_is_safe(tmp_path):
    from nodes._otr_audio_engines import _otr_sidecar as SC
    p = tmp_path / "x.wav"
    p.write_bytes(b"0")
    SC.remove_quietly(str(p))
    assert not p.exists()
    SC.remove_quietly(None)  # must not raise
    SC.remove_quietly(str(p))  # already gone -> must not raise


# --- role-aware ref resolution ---------------------------------------------- #
def test_announcer_role_threads_to_caster():
    # _resolve_clone_ref_path now passes self.ROLE to the caster; with the
    # announcer ref's timbre the announcer-role tier wins deterministically.
    from nodes._otr_voice_bank import assign_voice_for_slot, load_voice_bank
    bank, _ = load_voice_bank()
    e = assign_voice_for_slot(role="announcer_voice", engine="chatterbox",
                              char_id="ann", gender="male",
                              timbre=("authoritative", "resonant"), bank=bank)
    assert e.voice_ref_id == "cb_announcer_male"


def test_resolve_clone_ref_path_accepts_role_kwarg():
    from nodes import _otr_voice_node_common as VC
    # Unknown gender + no matching ref -> graceful None, never a crash.
    out = VC._resolve_clone_ref_path("chatterbox", {"char_id": "x", "gender": "zzz"}, 1,
                                     role="char_voice")
    assert out is None or isinstance(out, str)


# --- polish round 2: pipe closure, double-close, timeout clamp ------------- #
class _ClosablePipe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def write(self, *_a):
        raise BrokenPipeError("dead")

    def flush(self):
        pass

    def readline(self):
        return ""


class _ProcWithPipes:
    def __init__(self):
        self.stdin = _ClosablePipe()
        self.stdout = _ClosablePipe()
        self._alive = True
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        return 0


class _RaiseOnReClose:
    def __init__(self):
        self.n = 0

    def close(self):
        self.n += 1
        if self.n > 1:
            raise ValueError("I/O operation on closed file")


def test_close_worker_closes_pipes_and_stderr():
    from nodes._otr_audio_engines import _otr_sidecar as SC
    proc, sd = _ProcWithPipes(), _FakeStderr()
    SC.close_worker(proc, sd)
    assert proc.killed
    assert proc.stdin.closed and proc.stdout.closed and sd.closed_flag


def test_close_worker_tolerates_double_close():
    from nodes._otr_audio_engines import _otr_sidecar as SC
    sd = _RaiseOnReClose()
    SC.close_worker(None, sd)
    SC.close_worker(None, sd)  # widened except -> ValueError on re-close is swallowed
    assert sd.n == 2


def test_env_float_clamps_nonpositive(monkeypatch):
    from nodes._otr_audio_engines import _otr_sidecar as SC
    monkeypatch.setenv("OTR_SIDECAR_REQUEST_TIMEOUT", "-5")
    assert SC.request_timeout() == 600.0
    monkeypatch.setenv("OTR_SIDECAR_STARTUP_TIMEOUT", "0")
    assert SC.startup_timeout() == 1800.0
    monkeypatch.setenv("OTR_SIDECAR_REQUEST_TIMEOUT", "12.5")
    assert SC.request_timeout() == 12.5


# --- polish round 3: stale-ref PD1 guard + announcer fallback -------------- #
def test_stale_ref_resolves_to_a_nonexistent_path():
    # The dispatch nulls a non-empty-but-stale clone ref so it falls through to
    # resolution + bark fallback (PD1) instead of hard-failing in the worker.
    from nodes import _otr_voice_node_common as VC
    p = VC._resolve_ref_to_disk("models/TTS/refs/indextts2/__otr_no_such_ref__.wav")
    assert (p is None) or (not os.path.exists(p))


def test_announcer_clone_engine_fails_loud_without_ref():
    # NO-FALLBACK (2026-07-03): chatterbox serves announcer_voice + requires a ref;
    # a ref-less announcer render now FAILS LOUD (no bark fallback), so
    # missing_ref_fallback is None. The dispatch raises EngineUnusable for the
    # ref-less line rather than producing bark audio.
    from nodes import _otr_audio_engines as AE
    cbx = AE.get_engine("chatterbox")
    assert "announcer_voice" in cbx.roles
    assert cbx.requires_voice_ref is True and cbx.missing_ref_fallback is None


# --- delivery-vector wiring: robust projections (QA roundtable) ------------ #
def test_chatterbox_project_robust_to_malformed_vectors():
    from nodes._otr_audio_engines import get_engine
    cbx = get_engine("chatterbox")
    assert cbx._project(None) == 0.5                # kill-switch / no delivery
    assert cbx._project(["bad"]) == 0.5             # non-dict
    assert 0.0 <= cbx._project({"calm": "bad"}) <= 1.0  # non-numeric -> no crash
    assert cbx._project({}) == 0.5                  # empty -> neutral (early return)
    assert 0.0 <= cbx._project({"calm": 1.0}) <= 1.0
    assert 0.0 <= cbx._project({"calm": 99}) <= 1.0  # out-of-range clamped


def test_deterministic_delivery_vector_is_clean_and_complete():
    from nodes._otr_delivery_vector import EMOTIONS, deterministic_delivery_vector
    v = deterministic_delivery_vector("Help! Run! Danger!", 0.5)
    assert set(v.keys()) == set(EMOTIONS)
    assert all(0.0 <= float(x) <= 1.0 for x in v.values())
    # pure / deterministic: same input -> same output
    assert v == deterministic_delivery_vector("Help! Run! Danger!", 0.5)


# --- platform-correct venv default (0c-8, 2026-09-25) ----------------------- #
def test_default_venv_python_is_platform_correct(tmp_path, monkeypatch):
    from nodes._otr_audio_engines import _otr_sidecar as SC
    root = tmp_path / "chatterbox"
    monkeypatch.setattr(SC.os, "name", "nt")
    assert SC.default_venv_python(str(root)) == os.path.join(
        str(root), ".venv", "Scripts", "python.exe")
    # posix with no provisioned Scripts entry: the real venv interpreter.
    monkeypatch.setattr(SC.os, "name", "posix")
    assert SC.default_venv_python(str(root)) == os.path.join(
        str(root), ".venv", "bin", "python")
    # posix WITH the provisioner's Scripts entry: the launcher wins. For
    # IndexTTS2 it sets the offline env (Composer QA: routing around it runs
    # the worker online on a network-less pod); for chatterbox it is an
    # equivalent symlink. This is what a provisioned pod resolves.
    launcher = root / ".venv" / "Scripts" / "python.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"launcher")
    assert SC.default_venv_python(str(root)) == os.path.join(
        str(root), ".venv", "Scripts", "python.exe")


def test_windows_venv_default_is_byte_identical(monkeypatch):
    from nodes._otr_audio_engines import _otr_sidecar as SC
    from nodes._otr_audio_engines import eng_chatterbox, eng_indextts2
    monkeypatch.setattr(SC.os, "name", "nt")
    for mod, sub in ((eng_chatterbox, "chatterbox"), (eng_indextts2, "index-tts")):
        old = os.path.join(mod._COMFY_ROOT, sub, ".venv", "Scripts", "python.exe")
        assert SC.default_venv_python(os.path.join(mod._COMFY_ROOT, sub)) == old


def test_both_sidecar_engines_delegate_to_the_shared_helper(monkeypatch):
    from nodes import _otr_audio_engines as AE
    from nodes._otr_audio_engines import _otr_sidecar as SC
    from nodes._otr_audio_engines import eng_chatterbox, eng_indextts2
    for var in ("OTR_CHATTERBOX_VENV", "OTR_INDEXTTS2_VENV"):
        monkeypatch.delenv(var, raising=False)
    for name, mod, sub in (("chatterbox", eng_chatterbox, "chatterbox"),
                           ("indextts2", eng_indextts2, "index-tts")):
        for os_name in ("nt", "posix"):
            monkeypatch.setattr(SC.os, "name", os_name)
            assert AE.get_engine(name)._venv_python() == SC.default_venv_python(
                os.path.join(mod._COMFY_ROOT, sub)), (name, os_name)


def test_venv_env_overrides_still_win(monkeypatch):
    from nodes import _otr_audio_engines as AE
    monkeypatch.setenv("OTR_CHATTERBOX_VENV", "C:/custom/cbx/python.exe")
    monkeypatch.setenv("OTR_INDEXTTS2_VENV", "C:/custom/idx/python.exe")
    assert AE.get_engine("chatterbox")._venv_python() == "C:/custom/cbx/python.exe"
    assert AE.get_engine("indextts2")._venv_python() == "C:/custom/idx/python.exe"
