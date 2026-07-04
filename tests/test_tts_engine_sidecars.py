"""Tests for the chatterbox + Dia Path-B sidecars and the adapter-metadata
refactor that replaced the ``_OTR_CLONE_ENGINES`` name tuple (2026-06-05).

Headless-safe: never imports the chatterbox / dia libraries or spawns a worker.
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
    load, so this never pulls torch / chatterbox / dia)."""
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- registry + metadata --------------------------------------------------- #
def test_chatterbox_and_dia_registered_with_roles():
    from nodes import _otr_audio_engines as AE
    cbx = AE.get_engine("chatterbox")
    dia = AE.get_engine("dia")
    assert "char_voice" in cbx.roles
    assert dia.roles == ("char_voice",)
    assert cbx.requires_flag is None  # C6: registry IS the menu (no flag gate)
    assert dia.requires_flag is None
    assert cbx.commercial_clean is True and dia.commercial_clean is True
    assert cbx.sample_rate == 24000 and dia.sample_rate == 44100


def test_clone_engines_declare_ref_metadata():
    # NO-FALLBACK (2026-07-03): clone engines still REQUIRE a ref WAV, but a
    # missing ref now FAILS LOUD in the dispatch -- there is NO bark fallback, so
    # missing_ref_fallback is None on every cloning engine.
    from nodes import _otr_audio_engines as AE
    for name in ("indextts2", "chatterbox", "dia"):
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
def test_optin_engines_selectable_no_flag(monkeypatch):
    # C6 -- registry IS the menu: chatterbox + dia are selectable with NO flag
    # gate (the venv/weights checks run in load(), not assert_usable).
    from nodes import _otr_audio_engines as AE
    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    monkeypatch.delenv("OTR_ENABLE_DIA", raising=False)
    for name in ("chatterbox", "dia"):
        assert AE.assert_usable(name, "char_voice") == name


def test_optin_engines_usable_when_flagged(monkeypatch):
    from nodes import _otr_audio_engines as AE
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    monkeypatch.setenv("OTR_ENABLE_DIA", "1")
    assert AE.assert_usable("chatterbox", "char_voice") == "chatterbox"
    assert AE.assert_usable("dia", "char_voice") == "dia"


def test_load_fails_closed_when_not_installed(monkeypatch):
    from nodes import _otr_audio_engines as AE
    # Point the sidecars at a venv python that does not exist -> NAMED error.
    monkeypatch.setenv("OTR_CHATTERBOX_VENV", str(REPO_ROOT / "nope" / "python.exe"))
    monkeypatch.setenv("OTR_DIA_VENV", str(REPO_ROOT / "nope" / "python.exe"))
    for name, needle in (("chatterbox", "Chatterbox Path B not installed"),
                         ("dia", "Dia Path B not installed")):
        with pytest.raises(RuntimeError) as ei:
            AE.get_engine(name).load()
        assert needle in str(ei.value)


# --- C-5 import safety ------------------------------------------------------ #
def test_import_engines_pulls_no_sidecar_library():
    # Importing the registry must NOT import the chatterbox / dia libraries
    # (they are imported only inside the isolated worker subprocess).
    import nodes._otr_audio_engines  # noqa: F401
    for mod in ("chatterbox", "dia"):
        assert mod not in sys.modules, "%s imported at registry import (C-5)" % mod


# --- pure worker helpers (no GPU, no model) -------------------------------- #
def test_dia_worker_build_prompt():
    w = _load_script("_otr_dia_worker.py")
    assert w._build_prompt("Hello there.", "") == "[S1] Hello there."
    assert w._build_prompt("Hi.", "Ref line.") == "[S1] Ref line. [S1] Hi."


def test_chatterbox_worker_supported_kwargs_drops_unknown():
    w = _load_script("_otr_chatterbox_worker.py")

    def fn(text, audio_prompt_path=None, exaggeration=0.5):  # no cfg/temperature
        return None

    got = w._supported_kwargs(fn, audio_prompt_path="r", exaggeration=0.7,
                              cfg_weight=0.4, temperature=0.6)
    assert got == {"audio_prompt_path": "r", "exaggeration": 0.7}


# --- bank mirror ----------------------------------------------------------- #
def test_bank_has_chatterbox_and_dia_pools():
    from nodes._otr_voice_bank import load_voice_bank
    bank, _ = load_voice_bank()
    cbx = [e for e in bank if e.engine == "chatterbox" and "char_voice" in e.roles]
    dia = [e for e in bank if e.engine == "dia"]
    # FLOOR, not an exact count: the bank GROWS as public-domain voices are added
    # (scripts/otr_ingest_pd_voices.py). The original mirrored pools were 36 each.
    assert len(cbx) >= 36
    assert len(dia) >= 36
    assert all(e.roles == ("char_voice",) for e in dia)


def test_caster_assigns_clone_voice_for_each_engine():
    from nodes._otr_voice_bank import assign_voice_for_slot, load_voice_bank
    bank, _ = load_voice_bank()
    for engine in ("chatterbox", "dia"):
        e = assign_voice_for_slot(role="char_voice", engine=engine,
                                  char_id="c1", gender="female", bank=bank)
        assert e.engine == engine and e.gender == "female"


def test_no_dangling_placeholder_chatterbox_rows():
    from nodes._otr_voice_bank import load_voice_bank
    bank, _ = load_voice_bank()
    ids = {e.voice_ref_id for e in bank}
    assert "cc_male_warm" not in ids  # the old 0-on-disk placeholders are gone
    assert "cb_announcer_male" in ids  # one real on-disk announcer ref remains


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


# --- role-aware ref resolution + Dia prompt normalization ------------------ #
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
    out = VC._resolve_clone_ref_path("dia", {"char_id": "x", "gender": "zzz"}, 1,
                                     role="char_voice")
    assert out is None or isinstance(out, str)


def test_dia_worker_normalizes_leading_speaker_tag():
    w = _load_script("_otr_dia_worker.py")
    # a transcript that already carries [S1] must not produce "[S1] [S1]".
    assert w._build_prompt("Hi.", "[S1] Ref words.") == "[S1] Ref words. [S1] Hi."
    assert w._strip_lead_tag("[S2] x") == "x"


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
def test_emo_list_robust_to_malformed_vectors():
    from nodes._otr_audio_engines import get_engine
    idx = get_engine("indextts2")
    assert idx.emo_list(None) == [0.0] * 8          # None -> flat
    assert idx.emo_list("nope") == [0.0] * 8        # non-dict -> flat
    # non-numeric + out-of-range + NaN-ish are coerced + clamped, never crash
    out = idx.emo_list({"happy": "very", "angry": 99, "sad": -4, "calm": 0.5})
    assert len(out) == 8 and all(0.0 <= v <= 1.0 for v in out)


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
