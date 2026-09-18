"""An indicator the operator can read: iso in the filename, label on the card.

Operator 2026-09-18: "it would be nice for me to easily know what language
each one is in as I don't know Japanese or Portuguese or Chinese". Non-English
episodes publish as ``signal_lost_<title>_<iso>_<ts>.mp4`` and bake
``<TITLE> · <LANGUAGE>`` on the hero card. English, Off and legacy ledgers are
byte-identical. CPU only.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import video_engine as VE


@pytest.mark.parametrize("iso, label", [
    ("es", "Spanish"), ("pt", "Portuguese"), ("it", "Italian"), ("fr", "French"),
    ("hi", "Hindi"), ("ja", "Japanese"), ("zh", "Mandarin"),
])
def test_a_non_english_ledger_yields_its_iso_and_english_label(iso, label):
    assert VE._language_marks({"meta": {"episode_language": iso}}) == ("_" + iso, label)


@pytest.mark.parametrize("led", [
    {"meta": {"episode_language": "en"}},
    {"meta": {}},
    {},
    None,
    "not a ledger",
    {"meta": {"episode_language": "tlh"}},
])
def test_english_off_legacy_and_unreadable_add_nothing(led):
    assert VE._language_marks(led) == ("", "")


def test_the_filename_carries_the_iso_before_the_timestamp():
    src = inspect.getsource(VE)
    assert 'f"signal_lost_{safe_title}{lang_suffix}_{ts}.mp4"' in src
    assert 'f"signal_lost_{safe_title}_{ts}.mp4"' not in src


def _renderer(title, card_label=""):
    return VE._CRTRenderer(64, 64, title, [], [], [], 24, card_label=card_label)


def test_the_hero_card_carries_the_english_label_only_off_english():
    assert _renderer("El mapa prohibido")._hero_text() == "EL MAPA PROHIBIDO"
    assert _renderer("El mapa prohibido", "Spanish")._hero_text() == \
        "EL MAPA PROHIBIDO · SPANISH"
    assert _renderer("", "Japanese")._hero_text() == "SIGNAL · JAPANESE"


def test_the_label_never_reaches_the_ident_hud_or_the_scramble_seed():
    r = _renderer("El mapa prohibido", "Spanish")
    assert r.title == "El mapa prohibido"
    src = inspect.getsource(VE._CRTRenderer)
    ident = src[src.index("def _draw_ident"):]
    ident = ident[:ident.index("\n    def ")]
    assert "card_label" not in ident and "_hero_text" not in ident
    assert "rng_title=self.title" in src


def test_the_renderer_receives_the_plain_title_and_the_label_separately():
    src = inspect.getsource(VE)
    assert "_CRTRenderer(W, H, episode_title, volume, freqs, waves, fps," in src
    assert "card_label=lang_label," in src
    assert "display_title" not in src


# --------------------------------------------------------------------------- #
# the hero card and HUD draw a native title with a face that has its glyphs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("iso, policy", [
    ("hi", "devanagari"), ("ja", "cjk"), ("zh", "cjk"),
    ("es", "latin_arial"), ("en", "latin_arial"),
])
def test_the_font_policy_follows_the_row(iso, policy):
    assert VE._font_policy_for({"meta": {"episode_language": iso}}) == policy


@pytest.mark.parametrize("led", [{}, None, {"meta": {}}, {"meta": {"episode_language": "tlh"}}])
def test_off_legacy_and_unreadable_keep_the_latin_walk(led):
    assert VE._font_policy_for(led) == "latin_arial"


def test_latin_is_byte_identical_and_cached_by_size_alone():
    assert VE._load_font(20) is VE._load_font(20, "latin_arial")


@pytest.mark.parametrize("policy, windows_face", [
    ("devanagari", ("Nirmala.ttc", "Nirmala.ttf", "NirmalaUI.ttf", "mangal.ttf")),
    ("cjk", ("msyh.ttc", "msyhl.ttc", "simhei.ttf", "simsun.ttc", "YuGothM.ttc")),
])
def test_a_script_policy_resolves_a_script_face_when_the_host_has_one(policy, windows_face):
    import os, sys
    path = VE._script_font_path(policy)
    if path is None:
        if sys.platform == "win32":
            fd = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            present = [f for f in windows_face if os.path.isfile(os.path.join(fd, f))]
            assert not present, (
                "host has %s but the %s walk resolved nothing" % (present, policy))
        pytest.skip("host has no %s face; the monospace fallback is the documented path" % policy)
    font = VE._load_font(20, policy)
    assert getattr(font, "path", "") == path
    assert font is not VE._load_font(20)
    # The face the CRT measures with must carry the glyphs ASS will burn.
    family = getattr(font, "getname", lambda: ("", ""))()[0]
    expected = {"devanagari": "Nirmala", "cjk": ("Microsoft YaHei", "SimHei", "SimSun", "Yu Gothic")}[policy]
    assert any(family.startswith(e) for e in ((expected,) if isinstance(expected, str) else expected)), family


def test_the_renderer_loads_every_face_through_its_policy():
    src = inspect.getsource(VE._CRTRenderer)
    calls = [line for line in src.splitlines() if "_load_font(" in line]
    assert calls, "renderer loads no fonts?"
    assert all("self.font_policy" in line for line in calls), calls
    assert "font_policy=_font_policy_for(led)" in inspect.getsource(VE)


def test_a_script_miss_is_keyed_not_latched_for_the_life_of_the_process(monkeypatch):
    """ComfyUI runs prompts back to back in ONE process. A miss must drop
    when the configuration changes -- the same ruling `_mono_font_path`
    already carries (`test_changing_the_override_invalidates_BOTH_caches`)."""
    monkeypatch.setattr(VE, "_SCRIPT_FONT_PATH", {}, raising=False)
    monkeypatch.setattr(VE, "_SCRIPT_FONT_KEY", None, raising=False)
    walks = []

    def _walk(fd, policy):
        walks.append((fd, policy))
        return []

    monkeypatch.setitem(__import__("sys").modules, "nodes.otr_credits_roll",
                        type("M", (), {"_credits_script_font_paths": staticmethod(_walk)}))
    monkeypatch.setenv("WINDIR", r"C:\NoFontsHere")
    assert VE._script_font_path("devanagari") is None
    assert VE._script_font_path("devanagari") is None
    assert len(walks) == 1, "the miss must be cached within one configuration"
    monkeypatch.setenv("WINDIR", r"C:\Windows")
    VE._script_font_path("devanagari")
    assert len(walks) == 2, "a configuration change must drop the cached miss"


def test_a_face_that_stops_opening_evicts_instead_of_latching_none(monkeypatch):
    """A file lock or momentary I/O error may not silently downgrade every
    later render of that script."""
    monkeypatch.setattr(VE, "_SCRIPT_FONT_PATH", {"cjk": r"C:\gone.ttc"}, raising=False)
    monkeypatch.setattr(VE, "_SCRIPT_FONT_KEY", VE._script_font_key(), raising=False)
    calls = []
    real = VE.ImageFont.truetype

    def _boom(path, size, *a, **k):
        # Only the script face is broken; the monospace fallback must still
        # resolve, exactly as it would on a host with a locked font file.
        if str(path).endswith("gone.ttc"):
            calls.append(path)
            raise OSError("cannot open resource")
        return real(path, size, *a, **k)

    monkeypatch.setattr(VE.ImageFont, "truetype", _boom)
    # A size no other test has cached: a warm (policy, size) object would
    # short-circuit the open this test exists to fail.
    VE._load_font(137, "cjk")           # falls through to the mono walk
    assert calls == [r"C:\gone.ttc"], calls
    assert "cjk" not in VE._SCRIPT_FONT_PATH, "the entry must be evicted, not None"


def test_an_unknown_policy_is_announced_once_not_silently_latin(monkeypatch, caplog):
    monkeypatch.setattr(VE, "_WARNED_FONT_POLICIES", set(), raising=False)
    with caplog.at_level("WARNING"):
        VE._load_font(20, "devangari")   # a typo, not a policy
        VE._load_font(22, "devangari")
    warnings = [r for r in caplog.records if "unknown captions.font_policy" in r.message]
    assert len(warnings) == 1, "announced once per value, not per call"
    assert "devangari" in warnings[0].getMessage()


def test_every_registry_row_declares_a_known_policy():
    from nodes import _otr_episode_languages as EL
    for row in EL.load_registry()[0]:
        policy = (row.captions or {}).get("font_policy")
        assert policy in VE.KNOWN_FONT_POLICIES, (row.iso, policy)
