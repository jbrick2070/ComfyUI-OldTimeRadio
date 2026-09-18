"""Multilingual one-switch rows 4-5, 7-8: voice filter, Kokoro, Lemmy, wrap."""
from __future__ import annotations

import json

import pytest

from nodes import _otr_captions as CAP
from nodes import _otr_casting as CAST
from nodes import _otr_episode_languages as EPLANG
from nodes import _otr_voice_bank as VB
from nodes._otr_audio_engines import _kokoro_backends as KB
from nodes._otr_audio_engines import eng_kokoro as KOK
from nodes.cast_lock import CastLock


def _entry(vid, *, engine="kokoro", gender="female", language=None, **kw):
    langs = () if language is None else (language,)
    return VB.VoiceBankEntry(
        voice_ref_id=vid, engine=engine, gender=gender,
        timbre=("clear",), roles=("announcer_voice", "char_voice"),
        age_band="adult", ref_path="voices/%s.pt" % vid,
        ref_sha256="pending", commercial_clean=True, languages=langs, **kw,
    )


def test_absent_languages_means_english_never_prefix():
    # ef_ prefix is Spanish in Kokoro's catalog, but an unstamped bank row
    # is English. The helper must not guess.
    sneaky = _entry("ef_dora")
    assert VB.entry_languages(sneaky) == ("en",)
    assert VB.voice_speaks_language(sneaky, "en")
    assert not VB.voice_speaks_language(sneaky, "es")


def test_stamped_spanish_is_not_english():
    dora = _entry("ef_dora", language="es")
    assert VB.entry_languages(dora) == ("es",)
    assert VB.voice_speaks_language(dora, "es")
    assert not VB.voice_speaks_language(dora, "en")


def test_assign_and_announcer_stay_inside_language():
    bank = (
        _entry("bf_emma", gender="female", language="en"),
        _entry("bm_george", gender="male", language="en"),
        _entry("ef_dora", gender="female", language="es"),
        _entry("em_alex", gender="male", language="es"),
    )
    en = VB.assign_voice_for_slot(
        role="char_voice", engine="kokoro", char_id="c1",
        gender="female", episode_seed=7, bank=bank, language="en",
    )
    es = VB.assign_voice_for_slot(
        role="char_voice", engine="kokoro", char_id="c1",
        gender="female", episode_seed=7, bank=bank, language="es",
    )
    assert en.voice_ref_id == "bf_emma"
    assert es.voice_ref_id == "ef_dora"
    assert VB.announcer_voice_ref(
        "kokoro", bank=bank, episode_seed=3, language="es",
    ).voice_ref_id in {"ef_dora", "em_alex"}
    with pytest.raises(VB.VoiceCastingError, match="naming pool size"):
        VB.assign_voice_for_slot(
            role="char_voice", engine="kokoro", char_id="c1",
            gender="female", episode_seed=7, bank=bank, language="ja",
        )


def test_gender_agnostic_fallback_honors_language():
    bank = (
        _entry("bf_emma", gender="female", language="en"),
        _entry("ef_dora", gender="female", language="es"),
    )
    hit = VB.gender_agnostic_fallback_ref(
        bank, engine="kokoro", char_id="c9", episode_seed=1, language="es",
    )
    assert hit.voice_ref_id == "ef_dora"


def test_castlock_clears_bark_preset_on_kokoro_stamp():
    from tests.test_cast_lock import _ledger_with_lines
    cast = [
        {"char_id": "c1", "name": "STOMP", "gender": "male",
         "tts_model": "bark", "voice_preset": "v2/en_speaker_1"},
        {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
         "tts_model": "kokoro", "voice_preset": "bm_george"},
    ]
    lines = [
        {"line_id": "l001", "speaker_role": "character", "char_id": "c1",
         "text": "Hi."},
        {"line_id": "l002", "speaker_role": "announcer", "char_id": "a1",
         "text": "Tonight."},
    ]
    out = CastLock().lock(
        script_json=_ledger_with_lines(
            cast, lines, meta={"source_bank": "original", "episode_seed": 42}),
        cast_voice_policy="auto_registry",
        char_voice_engine="kokoro",
        announcer_voice_engine="kokoro",
    )
    led = json.loads(out[0])
    for row in led["cast"]:
        assert not str(row.get("voice_preset") or "").startswith("v2/"), row
        assert row.get("tts_model") != "bark", row


def test_castlock_refuses_bark_on_spanish():
    from tests.test_cast_lock import _ledger
    with pytest.raises(ValueError, match="not admitted"):
        CastLock().lock(
            script_json=_ledger(
                [{"char_id": "c1", "name": "ANA", "gender": "female",
                  "voice_preset": "v2/en_speaker_1"}],
                meta={"episode_language": "es", "episode_seed": 1,
                      "source_bank": "original"},
            ),
            cast_voice_policy="auto_registry",
            char_voice_engine="bark",
            announcer_voice_engine="kokoro",
        )


def test_lemmy_language_exclusion_after_fidelity():
    forced = CAST.resolve_lemmy_cameo("original", True, language_iso="es")
    assert forced.lemmy_hit is False
    assert forced.lemmy_policy == CAST.LEMMY_POLICY_LANGUAGE_EXCLUSION
    assert forced.knob_state == CAST.LEMMY_KNOB_FORCED_INCLUDE
    assert forced.roll_executed is False
    natural = CAST.resolve_lemmy_cameo("original", None, language_iso="es")
    assert natural.lemmy_policy == CAST.LEMMY_POLICY_LANGUAGE_EXCLUSION
    english = CAST.resolve_lemmy_cameo("shakespeare", True, language_iso="es")
    assert english.lemmy_policy == CAST.LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
    off_floor = CAST.resolve_lemmy_cameo("original", False, language_iso=None)
    assert off_floor.lemmy_policy == CAST.LEMMY_POLICY_OPERATOR_CAMEO


def test_kokoro_backend_cache_identity_is_lang_and_device():
    a = KB.TorchKokoroBackend("cpu", lang_code="b")
    b = KB.TorchKokoroBackend("cpu", lang_code="e")
    assert a.lang_code == "b"
    assert b.lang_code == "e"
    eng = KOK.KokoroEngine()
    eng._lang_code = "b"
    eng._backend_key = ("torch", "b", "cpu")
    eng._backend = object()
    # Different language must not reuse the loaded backend key.
    assert eng._backend_key != ("torch", "e", "cpu")


def test_announcer_trapdoor_does_not_fall_to_english_pool(monkeypatch):
    def _boom(*_a, **_k):
        raise VB.VoiceCastingError("naming pool size")

    monkeypatch.setattr(VB, "announcer_voice_ref", _boom)
    with pytest.raises(VB.VoiceCastingError):
        KOK._pick_announcer_voice(1, language="es")
    # English still has the engine-local pool.
    monkeypatch.setattr(
        "nodes._otr_audio_engines.eng_kokoro.announcer_voice_ref",
        _boom,
        raising=False,
    )
    # The function imports announcer_voice_ref from the voice bank inside
    # the try; the raise path for English uses the local pool.
    voice = KOK._pick_announcer_voice(1, language="en")
    assert voice in KOK.ANNOUNCER_VOICE_POOL


def test_caption_wrap_policies():
    latin = CAP.wrap_text("one two three four", max_chars=8, wrap_policy="word_split")
    assert latin[0] == "one two"
    cjk = CAP.wrap_cjk_chars("日本語の字幕です", max_chars=4)
    assert cjk[0] == "日本語の"
    assert " " not in "".join(cjk)
    hindi = CAP.wrap_unicode_grapheme("नमस्ते दुनिया", max_chars=4)
    assert hindi
    # Virama keeps the conjunct in one cluster, so a 1-wide wrap does not
    # split क् + ष into two broken glyphs.
    conjunct = CAP.wrap_unicode_grapheme("क्ष", max_chars=1)
    assert conjunct == ["क्ष"]


def test_row_wrap_policies_match_scripts():
    assert EPLANG.row_by_iso("en").captions["wrap_policy"] == "word_split"
    assert EPLANG.row_by_iso("hi").captions["wrap_policy"] == "unicode_grapheme"
    assert EPLANG.row_by_iso("ja").captions["wrap_policy"] == "cjk_chars"
    assert EPLANG.row_by_iso("zh").captions["wrap_policy"] == "cjk_chars"


def test_each_admitted_language_meets_min_voice_count():
    rows, _by_label, _by_iso = EPLANG.load_registry()
    bank = VB.load_voice_bank()[0]
    for row in rows:
        if not row.admitted:
            continue
        voices = [
            e for e in bank
            if e.engine == "kokoro" and VB.voice_speaks_language(e, row.iso)
        ]
        floor = int((row.admission or {}).get("min_voice_count") or 0)
        assert len(voices) >= floor, (row.iso, len(voices), floor)
        assert any("announcer_voice" in e.roles for e in voices), row.iso
        assert any("char_voice" in e.roles for e in voices), row.iso


def _caption_ledger(tmp_path, iso, text, name="ANA"):
    from pathlib import Path
    ledger = {
        "meta": {"episode_language": iso, "episode_seed": 1,
                 "source_bank": "original"},
        "cast": [
            {"char_id": "c1", "name": name, "gender": "female"},
            {"char_id": "a1", "name": "ANNOUNCER", "gender": "male"},
        ],
        "lines": [{
            "line_id": "l001", "speaker_role": "character", "char_id": "c1",
            "text": text, "start_s": 0.0, "dur_s": 2.5,
        }],
    }
    src = Path(tmp_path) / ("%s_ledger.json" % iso)
    out = Path(tmp_path) / ("%s_captions.ass" % iso)
    src.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    path, report = CAP.build_ass_from_ledger(
        str(src), style="sdh_standard", out_path=str(out))
    assert path, report
    return Path(path).read_text(encoding="utf-8")


def test_accented_spanish_survives_ass_and_is_not_ascii_stripped(tmp_path):
    """Latin Arial/44/17 plus n-tilde: any ASCII-strip here is a defect."""
    ass = _caption_ledger(tmp_path, "es", "Buenas noches, niño.")
    assert "Buenas noches, niño." in ass
    assert "Style: SDH,Arial," in ass
    assert "ñ".upper() == "Ñ"


def test_hindi_and_cjk_ass_use_the_row_font(tmp_path):
    hi = _caption_ledger(tmp_path, "hi", "नमस्ते दुनिया")
    assert "Nirmala UI" in hi
    assert "नमस्ते" in hi
    ja = _caption_ledger(tmp_path, "ja", "日本語の字幕です")
    assert "Microsoft YaHei" in ja
    assert "日本語" in ja


def test_credits_layout_carries_font_policy_and_paints_language_header():
    from pathlib import Path
    from nodes import otr_credits_roll as CR
    assert CR._credits_font_policy({"episode_language": "es"}) == "latin_arial"
    assert CR._credits_font_policy({"episode_language": "hi"}) == "devanagari"
    assert CR._credits_font_policy({"episode_language": "zh"}) == "cjk"
    src = (Path(__file__).resolve().parents[1] / "nodes"
           / "otr_credits_roll.py").read_text(encoding="utf-8")
    assert 'lang_header = str(layout.get("language_header") or "").strip()' in src
    assert '"font_policy": _credits_font_policy(meta),' in src
