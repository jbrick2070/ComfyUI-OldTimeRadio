"""Episode language registry -- all eight Kokoro rows admitted day 1.

Day-1 dropdown (operator override of the oval's Spanish-first staging):
``Off | English | Spanish | Portuguese | Italian | French | Hindi | Japanese |
Mandarin``. Registry-only coverage: no Comfy tree, no writer widget.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from nodes import _otr_episode_languages as el

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "config" / "episode_languages.json"

# Stable COMBO API. Order is Off then sort_order.
EXPECTED_CHOICES = [
    "Off", "English", "Spanish", "Portuguese", "Italian", "French",
    "Hindi", "Japanese", "Mandarin",
]
ADMITTED_LABELS = EXPECTED_CHOICES[1:]

# Brand tokens and machine serials stay English on every row.
BRAND_TOKENS = ("SIGNAL LOST",)


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """Copy the committed registry into tmp and point the module at it."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    path = tmp_path / "episode_languages.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(el, "REGISTRY_PATH", str(path))
    el.reload_registry(path=str(path))
    return path, payload


def test_committed_registry_admits_all_eight_kokoro_languages():
    rows, by_label, by_iso = el.reload_registry()
    assert len(rows) == 8
    assert el.dropdown_choices() == EXPECTED_CHOICES
    for label in ADMITTED_LABELS:
        assert by_label[label].admitted, label
    assert [by_iso[iso].label for iso in
            ("en", "es", "pt", "it", "fr", "hi", "ja", "zh")] == ADMITTED_LABELS


def test_kokoro_lang_codes_match_the_measured_catalog():
    _rows, by_label, _by_iso = el.reload_registry()
    expected = {
        "English": "b", "Spanish": "e", "Portuguese": "p", "Italian": "i",
        "French": "f", "Hindi": "h", "Japanese": "j", "Mandarin": "z",
    }
    for label, code in expected.items():
        assert by_label[label].engines["kokoro"]["lang_code"] == code, label
    # English keeps 'a' in the character pool while 'b' announces.
    assert by_label["English"].engines["kokoro"]["lang_code_character_pool"] == ["a", "b"]


def test_every_admitted_row_declares_kokoro_first_and_google_tts():
    """Kokoro stays the dance leader (first, with a voice list); Google TTS is
    admitted beside it on every row (0i, operator 2026-09-25: "all supported
    as Kokoro"). Its entry carries no config -- Gemini TTS voices are not tied
    to a language. Any THIRD engine must be added and qualified on purpose."""
    rows, _by_label, _by_iso = el.reload_registry()
    for row in rows:
        assert list(row.engines) == ["kokoro", "google_tts"], row.label
        assert row.engines["google_tts"] == {}, row.label
        voices = row.engines["kokoro"]["voices"]
        assert voices, row.label
        assert len(set(voices)) == len(voices), "duplicate voice id on %s" % row.label


def test_min_voice_count_never_exceeds_the_row_roster():
    """French (1) and Italian (2) are admitted thin, honestly, with reuse."""
    rows, by_label, _by_iso = el.reload_registry()
    for row in rows:
        roster = row.engines["kokoro"]["voices"]
        assert row.admission["min_voice_count"] <= len(roster), row.label
    assert by_label["French"].admission["min_voice_count"] == 1
    assert by_label["Italian"].admission["min_voice_count"] == 2


def test_min_voice_count_above_the_roster_fails_closed(fresh):
    path, payload = fresh
    payload["rows"][4]["admission"]["min_voice_count"] = 9
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    with pytest.raises(el.EpisodeLanguageError, match="exceeds"):
        el.reload_registry(path=str(path))


def test_duplicate_kokoro_lang_code_fails_closed(fresh):
    path, payload = fresh
    payload["rows"][2]["engines"]["kokoro"]["lang_code"] = \
        payload["rows"][1]["engines"]["kokoro"]["lang_code"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    with pytest.raises(el.EpisodeLanguageError, match="duplicate Kokoro lang_code"):
        el.reload_registry(path=str(path))


def test_off_is_resolver_state_not_a_row():
    resolved = el.resolve_label(el.OFF_LABEL)
    assert resolved.kind == "off" and resolved.row is None and resolved.stamp is False
    assert el.resolve_ledger(el.OFF_LABEL) is None
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert all(row["label"] != el.OFF_LABEL for row in payload["rows"])


def test_blank_and_none_resolve_to_english():
    for label in ("", "   ", None):
        resolved = el.resolve_label(label)
        assert resolved.stamp is True
        assert resolved.row.iso == "en"
        assert resolved.row.label == "English"
        ledger = el.resolve_ledger(label)
        assert ledger["episode_language"] == "en"
        assert ledger["language_header"] == "English"


@pytest.mark.parametrize("label,iso,header", [
    ("English", "en", "English"),
    ("Spanish", "es", "Español"),
    ("Portuguese", "pt", "Português"),
    ("Italian", "it", "Italiano"),
    ("French", "fr", "Français"),
    ("Hindi", "hi", "हिन्दी"),
    ("Japanese", "ja", "日本語"),
    ("Mandarin", "zh", "普通话"),
])
def test_ledger_stamp_shape_per_row(label, iso, header):
    ledger = el.resolve_ledger(label)
    assert ledger["episode_language"] == iso
    assert ledger["language_header"] == header
    receipt = ledger["episode_language_receipt"]
    assert receipt["registry_id"] == "episode_languages"
    assert receipt["schema_version"] == 1
    assert receipt["label"] == label
    assert receipt["row_revision"] >= 1
    assert len(receipt["row_sha256"]) == 64


def test_unknown_label_fails_closed():
    for token in ("Klingon", "Esperanto", "es", "off", "OFF"):
        with pytest.raises(el.EpisodeLanguageError, match="unknown"):
            el.resolve_label(token)


def test_non_string_label_fails_closed():
    with pytest.raises(el.EpisodeLanguageError, match="must be a string"):
        el.resolve_label(7)


def test_inject_ninth_admitted_row_appears_without_writer_change(fresh):
    """A later row ships by itself. Synthetic fixture -- Kokoro 0.9.4 has no
    Korean lang_code, and this row is never committed."""
    path, payload = fresh
    ninth = copy.deepcopy(payload["rows"][1])
    ninth["iso"] = "ko"
    ninth["label"] = "Korean"
    ninth["native_header"] = "한국어"
    ninth["sort_order"] = 90
    ninth["engines"] = {
        "kokoro": {
            "lang_code": "k",
            "voices": ["kf_placeholder_one", "kf_placeholder_two", "km_placeholder_three"],
        }
    }
    payload["rows"].append(ninth)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    el.reload_registry(path=str(path))
    assert el.dropdown_choices(path=str(path)) == EXPECTED_CHOICES + ["Korean"]
    assert el.resolve_label("Korean", path=str(path)).row.iso == "ko"
    assert el.resolve_ledger("Korean", path=str(path))["language_header"] == "한국어"


def test_unadmitted_row_hidden_from_dropdown_and_refuses_resolve(fresh):
    path, payload = fresh
    payload["rows"][1]["admitted"] = False
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    el.reload_registry(path=str(path))
    assert el.dropdown_choices(path=str(path)) == [
        c for c in EXPECTED_CHOICES if c != "Spanish"]
    with pytest.raises(el.EpisodeLanguageError, match="not admitted"):
        el.resolve_label("Spanish", path=str(path))


def test_english_spoken_matches_today_line_composer_literals():
    row = el.row_by_label("English")
    assert row.spoken["sign_on_greeting"] == "Good evening"
    assert row.spoken["station_id_open"] == "This is SIGNAL LOST"
    assert row.spoken["station_id_close"] == "This has been SIGNAL LOST"
    assert row.spoken["sign_off_greeting"] == "Good night"
    assert row.spoken["reserved_announcer_name"] == "ANNOUNCER"
    assert row.spoken["work_line_prefix"] == "a scene from "


def test_english_credits_match_today_credits_roll_literals():
    row = el.row_by_label("English")
    assert row.credits["models_header"] == "MODELS"
    assert row.credits["production_ledger_header"] == "[ PRODUCTION LEDGER ]"
    assert row.credits["cast_voices_header"] == "CAST & VOICES"
    assert row.credits["story_spine_header"] == "[ STORY SPINE ]"
    assert row.credits["premise_label"] == "Premise:"
    assert row.credits["subject_label"] == "Subject:"
    assert row.credits["writer_llm_header"] == "[ WRITER / LLM CONFIG ]"
    assert row.credits["origin_hud"] == "ORIGIN"
    assert row.credits["more_hud"] == "+%d MORE"


def test_spanish_house_calls_from_the_fable_seed():
    row = el.row_by_label("Spanish")
    assert row.native_header == "Español"
    assert row.spoken["reserved_announcer_name"] == "LOCUTOR"
    assert row.spoken["sign_on_greeting"] == "Buenas noches"
    assert row.spoken["sign_off_greeting"] == "Buenas noches"
    assert row.spoken["station_id_open"] == "Esta es SIGNAL LOST"
    assert row.credits["story_spine_header"] == "[ ARMAZÓN ]"
    assert row.credits["writer_llm_header"] == "[ GUIONISTA / LLM ]"
    assert row.credits["origin_hud"] == "ORIGEN"
    assert row.credits["more_hud"] == "+%d MÁS"


def test_portuguese_house_calls_from_the_fable_seed():
    row = el.row_by_label("Portuguese")
    assert row.native_header == "Português"
    assert row.spoken["reserved_announcer_name"] == "LOCUTOR"
    assert row.spoken["sign_on_greeting"] == "Boa noite"
    assert row.engines["kokoro"]["voices"] == ["pf_dora", "pm_alex", "pm_santa"]


@pytest.mark.parametrize("label", ADMITTED_LABELS)
def test_every_admitted_row_carries_every_english_key_non_empty(label):
    """Admission test: same English keys on every table, no empty value."""
    english = el.row_by_label("English")
    row = el.row_by_label(label)
    for table in ("spoken", "credits", "captions", "authoring"):
        assert set(getattr(row, table)) == set(getattr(english, table)), \
            "%s.%s key drift" % (label, table)
        for key, value in getattr(row, table).items():
            assert isinstance(value, str) and value.strip(), \
                "%s.%s.%s is empty" % (label, table, key)


@pytest.mark.parametrize("label", ADMITTED_LABELS)
def test_brand_tokens_survive_on_every_row(label):
    """SIGNAL LOST is the call sign; grammar bends around it."""
    row = el.row_by_label(label)
    for token in BRAND_TOKENS:
        assert token in row.spoken["station_id_open"], label
        assert token in row.spoken["station_id_close"], label


@pytest.mark.parametrize("label", ADMITTED_LABELS)
def test_more_hud_keeps_its_percent_d_slot(label):
    assert "%d" in el.row_by_label(label).credits["more_hud"], label


@pytest.mark.parametrize("label,font,wrap,cps", [
    ("English", "latin_arial", "word_split", "latin_17"),
    ("Spanish", "latin_arial", "word_split", "latin_17"),
    ("Portuguese", "latin_arial", "word_split", "latin_17"),
    ("Italian", "latin_arial", "word_split", "latin_17"),
    ("French", "latin_arial", "word_split", "latin_17"),
    ("Hindi", "devanagari", "unicode_grapheme", "devanagari_soft"),
    ("Japanese", "cjk", "cjk_chars", "cjk_soft"),
    ("Mandarin", "cjk", "cjk_chars", "cjk_soft"),
])
def test_caption_paint_policies_per_row(label, font, wrap, cps):
    captions = el.row_by_label(label).captions
    assert captions["font_policy"] == font
    assert captions["wrap_policy"] == wrap
    assert captions["cps_policy"] == cps


@pytest.mark.parametrize("label,extras", [
    ("English", []),
    ("Spanish", []),
    ("Portuguese", []),
    ("Italian", []),
    ("French", []),
    ("Hindi", []),
    ("Japanese", ["misaki[ja]"]),
    ("Mandarin", ["misaki[zh]"]),
])
def test_readiness_extras_are_not_an_english_install_tax(label, extras):
    assert el.row_by_label(label).admission["readiness_extras"] == extras


@pytest.mark.parametrize("label,extra,module_name,dependency", [
    ("Japanese", "misaki[ja]", "misaki.ja", "pyopenjtalk"),
    ("Mandarin", "misaki[zh]", "misaki.zh", "ordered_set"),
])
def test_readiness_extra_imports_transitive_contract(
        monkeypatch, label, extra, module_name, dependency):
    """A discoverable adapter is not usable when a transitive import is absent."""
    calls = []

    def _missing_dependency(requested_module):
        calls.append(requested_module)
        raise ModuleNotFoundError("No module named %r" % dependency)

    monkeypatch.setattr(importlib, "import_module", _missing_dependency)
    assert el.readiness_extra_ok(extra) is False
    assert calls == [module_name]
    with pytest.raises(el.EpisodeLanguageError, match=r"misaki\["):
        el.assert_readiness_extras(el.row_by_label(label))


def test_readiness_extra_accepts_a_fully_importable_adapter(monkeypatch):
    calls = []

    def _available(module_name):
        calls.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", _available)
    assert el.readiness_extra_ok("misaki[zh]") is True
    assert calls == ["misaki.zh"]


@pytest.mark.parametrize("label", ADMITTED_LABELS)
def test_no_row_excludes_any_bank(label):
    """Every lane is eligible for the episode language (operator 2026-09-18)."""
    assert el.row_by_label(label).admission["source_bank_exclusions"] == [], label


_SPOKEN_CREDIT_KEYS = (
    "coda_public_domain_us", "coda_cc0", "coda_research_only", "coda_synthetic",
    "coda_named_public_domain_us", "coda_named_cc0", "coda_named_research_only",
    "coda_licensed_named", "attribution_named", "attribution_anonymous",
)


@pytest.mark.parametrize("label", ADMITTED_LABELS)
def test_every_row_authors_its_spoken_credit_sentences(label):
    """The announcer's Python-owned sentences are row data, never translated."""
    spoken = el.row_by_label(label).spoken
    for key in _SPOKEN_CREDIT_KEYS:
        assert spoken[key].strip(), (label, key)
    for key in ("coda_named_public_domain_us", "coda_named_cc0",
                "coda_named_research_only", "coda_licensed_named"):
        assert "{work_title}" in spoken[key] and "{author}" in spoken[key], (label, key)
    assert "{name}" in spoken["attribution_named"], label
    assert "{" not in spoken["attribution_anonymous"], label


@pytest.mark.parametrize("label", ADMITTED_LABELS)
def test_visual_prompt_iso_is_english_pixels_day_one(label):
    """Fully native pixels are a field flip on the row, not a second graph."""
    assert el.row_by_label(label).authoring["visual_prompt_iso"] == "en"


def test_off_label_cannot_be_committed_as_a_row(fresh):
    path, payload = fresh
    payload["rows"][0]["label"] = "Off"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    with pytest.raises(el.EpisodeLanguageError, match="resolver state"):
        el.reload_registry(path=str(path))


def test_admitted_row_missing_credits_key_fails(fresh):
    path, payload = fresh
    del payload["rows"][1]["credits"]["premise_label"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    with pytest.raises(el.EpisodeLanguageError, match="premise_label"):
        el.reload_registry(path=str(path))


def test_registry_file_is_utf8_without_bom():
    raw = REGISTRY.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert "Español" in text and "हिन्दी" in text and "日本語" in text


def test_row_sha256_changes_when_the_row_changes(fresh):
    path, payload = fresh
    before = el.resolve_ledger("Spanish", path=str(path))
    payload["rows"][1]["row_revision"] = 99
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    el.reload_registry(path=str(path))
    after = el.resolve_ledger("Spanish", path=str(path))
    assert after["episode_language_receipt"]["row_revision"] == 99
    assert (after["episode_language_receipt"]["row_sha256"]
            != before["episode_language_receipt"]["row_sha256"])
