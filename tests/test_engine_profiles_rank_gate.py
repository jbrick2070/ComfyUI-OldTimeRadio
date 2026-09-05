"""Sprint 1 -- declarative profile metadata + rank-chain fallback resolver +
three-state commercial gate. Headless; the only IO is loading the shipped YAML.

These cover the NEW surface only; the legacy resolver behaviour stays pinned by
test_engine_profiles.py (unchanged). The live byte-identical dispatch is NOT
exercised here -- the rank chain is an auto-selection helper, not a rewire.
"""
from __future__ import annotations

import pytest

from nodes import _otr_engine_profiles as EP
from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason

ROLES = ("char_voice", "announcer_voice", "music")


def _resolver():
    r = EP.load_resolver()
    assert r is not None
    return r


def test_every_profile_has_sprint1_metadata():
    r = _resolver()
    for pid in r.profile_ids():
        p = r.get(pid)
        assert isinstance(p.rank, int)
        assert isinstance(p.is_default, bool)
        assert p.runtime in EP._VALID_RUNTIMES
        assert isinstance(p.needs_ref_clip, bool)
        assert isinstance(p.caps, dict)
        assert p.license_state in EP._VALID_LICENSE_STATES


def test_exactly_one_scope_default_per_role():
    r = _resolver()
    defaults = {
        role: [p.engine for p in r.for_role(role) if p.is_default]
        for role in ROLES
    }
    assert defaults["char_voice"] == ["indextts2"]
    assert defaults["announcer_voice"] == ["kokoro"]
    assert defaults["music"] == ["stable_audio_3"]


def test_rank_chain_is_sorted_by_rank():
    r = _resolver()
    chain = [p.engine for p in r.rank_chain("char_voice")]
    assert chain == ["indextts2", "chatterbox", "dia", "bark", "kokoro"]  # rank 1, 2, 2, 3, 60
    ranks = [p.rank for p in r.rank_chain("char_voice")]
    assert ranks == sorted(ranks)


def test_role_default_is_rank1_flagged():
    r = _resolver()
    assert r.role_default("char_voice").engine == "indextts2"
    assert r.role_default("announcer_voice").engine == "kokoro"
    assert r.role_default("music").engine == "stable_audio_3"


def test_fallback_returns_promoted_indextts2_default(monkeypatch):
    # indextts2 was PROMOTED 2026-06-04 to the always-usable char_voice default
    # (no flag); chatterbox stays opt-in. With every opt-in flag off the rank
    # chain still resolves to indextts2 (rank 1, always usable) -- never crashes,
    # never silently swaps. bark stays selectable as the rank-3 fallback.
    monkeypatch.delenv("OTR_ENABLE_INDEXTTS2", raising=False)
    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    r = _resolver()
    assert r.resolve_role_fallback("char_voice").engine == "indextts2"


def test_fallback_picks_indextts2_when_enabled(monkeypatch):
    # Registry-level usability only checks the flag (no install IO), so with the
    # flag on the rank-1 scope default is selected.
    monkeypatch.setenv("OTR_ENABLE_INDEXTTS2", "1")
    r = _resolver()
    assert r.resolve_role_fallback("char_voice").engine == "indextts2"


def test_music_fallback_default_is_sa3(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_STABLE_AUDIO", raising=False)
    r = _resolver()
    assert r.resolve_role_fallback("music").engine == "stable_audio_3"


def test_gate_state_maps_three_ways():
    r = _resolver()
    assert EP.gate_state(r.get("announcer_kokoro_v1")) == "clean"
    assert EP.gate_state(r.get("music_musicgen_v1")) == "warn"
    assert EP.gate_state(r.get("char_indextts2_v1")) == "warn"


def test_effective_license_state_blank_derivation():
    p_clean = EP.EngineProfile(
        profile_id="x", role="music", engine="stable_audio_3",
        commercial_clean=True,
    )
    assert EP.effective_license_state(p_clean) == "clean"
    p_gated = EP.EngineProfile(
        profile_id="y", role="music", engine="musicgen",
        commercial_clean=False,
    )
    assert EP.effective_license_state(p_gated) == "gated"


def test_license_state_mirrors_commercial_clean_for_all_rows():
    r = _resolver()
    for pid in r.profile_ids():
        p = r.get(pid)
        assert EP.effective_license_state(p) == p.license_state
        if p.license_state == "clean":
            assert p.commercial_clean is True
        elif p.license_state == "gated":
            assert p.commercial_clean is False


def test_gated_engines_carry_warn_text():
    r = _resolver()
    for pid in ("char_indextts2_v1", "char_bark_v1", "announcer_bark_v1",
                "music_musicgen_v1"):
        p = r.get(pid)
        assert p.license_state == "gated"
        assert p.warn_text.strip(), pid


def test_indextts2_runtime_is_oop_and_needs_ref_clip():
    r = _resolver()
    p = r.get("char_indextts2_v1")
    assert p.runtime == "oop_venv"
    assert p.needs_ref_clip is True


def test_unknown_license_state_gates_stop():
    # A synthetic 'unknown' profile fails closed at the gate (stop), proving the
    # third state even though no shipped row uses it yet.
    p = EP.EngineProfile(
        profile_id="x", role="music", engine="musicgen",
        commercial_clean=False, license_state="unknown",
    )
    assert EP.gate_state(p) == "stop"


def test_resolve_role_fallback_failclosed_when_nothing_usable():
    r = _resolver()
    with pytest.raises(EngineUnusable) as ei:
        r.resolve_role_fallback("no_such_role")
    assert ei.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_bad_runtime_rejected():
    with pytest.raises(Exception):
        EP.EngineProfile(
            profile_id="x", role="music", engine="musicgen",
            commercial_clean=False, runtime="cloud",
        )


def test_bad_license_state_rejected():
    with pytest.raises(Exception):
        EP.EngineProfile(
            profile_id="x", role="music", engine="musicgen",
            commercial_clean=False, license_state="maybe",
        )


def test_engine_warning_only_for_gated():
    r = _resolver()
    assert EP.engine_warning(r.get("announcer_kokoro_v1")) == ""   # clean
    w = EP.engine_warning(r.get("char_indextts2_v1"))
    assert "bilibili" in w.lower()
    assert EP.engine_warning(r.get("music_musicgen_v1"))           # non-empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
