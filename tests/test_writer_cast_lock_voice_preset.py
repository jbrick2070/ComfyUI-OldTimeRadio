"""tests/test_writer_cast_lock_voice_preset.py

P1 Gate 1 — writer cast-lock exit invariant. ``lock_cast`` must
guarantee that every returned cast row has a non-empty
``voice_preset`` starting with ``v2/``.

Today the pre-filter + reroll path already produces well-formed
presets by construction. This test pins the invariant via a direct
helper (``_assert_voice_preset_invariant``) so a future refactor
that breaks the pre-filter chain fails loud at the writer instead
of leaking an empty preset into Bark (Gate 3) or the freeze cascade
(Gate 2).
"""
from __future__ import annotations

import pytest

from nodes import _otr_casting as _otr_casting


def test_assert_voice_preset_invariant_passes_on_valid_cast():
    """Cast where every row has a v2/* voice_preset passes silently."""
    cast = [
        {"char_id": "c00", "name": "ANNOUNCER", "voice_preset": "bm_george"},
        {"char_id": "c01", "name": "LEMMY",     "voice_preset": "v2/en_speaker_8"},
        {"char_id": "c02", "name": "ASTRA",     "voice_preset": "v2/en_speaker_9"},
        {"char_id": "c03", "name": "ORION",     "voice_preset": "v2/en_speaker_3"},
    ]
    # No raise = pass.
    _otr_casting._assert_voice_preset_invariant(cast)


def test_assert_voice_preset_invariant_raises_on_empty_preset():
    """Empty / None preset on a non-ANNOUNCER row is a contract violation."""
    cast = [
        {"char_id": "c00", "name": "ANNOUNCER", "voice_preset": "bm_george"},
        {"char_id": "c01", "name": "LEMMY",     "voice_preset": "v2/en_speaker_8"},
        {"char_id": "c02", "name": "ASTRA",     "voice_preset": ""},
    ]
    with pytest.raises(_otr_casting.CastingFailedError) as exc:
        _otr_casting._assert_voice_preset_invariant(cast)
    msg = str(exc.value)
    assert "voice_preset" in msg.lower()
    assert "c02" in msg or "ASTRA" in msg


def test_assert_voice_preset_invariant_raises_on_none_preset():
    cast = [
        {"char_id": "c01", "name": "LEMMY", "voice_preset": "v2/en_speaker_8"},
        {"char_id": "c02", "name": "ASTRA", "voice_preset": None},
    ]
    with pytest.raises(_otr_casting.CastingFailedError):
        _otr_casting._assert_voice_preset_invariant(cast)


def test_assert_voice_preset_invariant_raises_on_non_v2_preset():
    """Non-v2/* preset on a Bark row is a contract violation."""
    cast = [
        {"char_id": "c01", "name": "LEMMY", "voice_preset": "v2/en_speaker_8"},
        # 'bm_george' is a Kokoro voice id; it belongs on the ANNOUNCER
        # row only. Anywhere else it's a writer bug.
        {"char_id": "c02", "name": "ASTRA", "voice_preset": "bm_george"},
    ]
    with pytest.raises(_otr_casting.CastingFailedError) as exc:
        _otr_casting._assert_voice_preset_invariant(cast)
    msg = str(exc.value)
    assert "v2/" in msg or "non-v2" in msg.lower()


def test_assert_voice_preset_invariant_excludes_announcer():
    """A non-Bark ANNOUNCER's Kokoro voice id is excluded from the check
    (no ``tts_model`` field, or any value other than ``"bark"``, reads as
    non-Bark -- the common case)."""
    cast = [
        # bf_emma is a Kokoro id — legitimately not v2/*.
        {"char_id": "c00", "name": "ANNOUNCER", "voice_preset": "bf_emma"},
        {"char_id": "c01", "name": "LEMMY",     "voice_preset": "v2/en_speaker_8"},
    ]
    _otr_casting._assert_voice_preset_invariant(cast)


def test_assert_voice_preset_invariant_INCLUDES_a_bark_announcer():
    """A Bark-delivered ANNOUNCER (2026-08-24, ``tts_model == "bark"``) must
    satisfy the same v2/* contract a character row does -- it is no longer
    exempt just for being the announcer."""
    cast = [
        {"char_id": "c00", "name": "ANNOUNCER", "tts_model": "bark",
         "voice_preset": "bf_emma"},
        {"char_id": "c01", "name": "LEMMY",     "voice_preset": "v2/en_speaker_8"},
    ]
    with pytest.raises(_otr_casting.CastingFailedError) as exc:
        _otr_casting._assert_voice_preset_invariant(cast)
    assert "v2/" in str(exc.value) or "non-v2" in str(exc.value).lower()


def test_assert_voice_preset_invariant_empty_cast_passes():
    """No rows to validate is not an error here (other gates catch empty cast)."""
    _otr_casting._assert_voice_preset_invariant([])


def test_invariant_is_in_lock_cast_export_surface():
    """The helper is part of the public __all__ so the freeze cascade can
    reuse the same logic for Gate 2."""
    assert "_assert_voice_preset_invariant" in _otr_casting.__all__
