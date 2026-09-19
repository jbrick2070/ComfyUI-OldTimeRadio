"""The obs filename says what MADE the episode.

WHY THIS EXISTS (operator, 2026-09-03). The published copy used to inherit the
archival stem verbatim, so every episode in `otr/obs/` read

    signal_lost_<title>_<ts>_silent_procgen_blended_captioned_with_credits_final.mp4

and a file browser truncated every single row at the identical, useless point --
`..._silent_procgen_blended_captioned_wit...`. The operator sent a screenshot of
exactly that: fifteen rows, indistinguishable past the title.

Worse than useless, that tail is MISLEADING. `procgen` is a compositing stage,
not a render engine, and this very session read it as the engine and built a
whole wrong diagnosis on it ("88 of 89 episodes are static") before the ledgers
corrected it. A name that invites a wrong reading is a defect.

The obs copy now carries the five choices that produced the episode, in the
order the operator picked: episode first (so the folder still sorts by episode),
then style and video engine (the axes he compares, and the ones that must
survive truncation).

The ARCHIVAL copy in `otr/episodes/` is deliberately untouched -- its suffixes
carry pipeline provenance, `otr_caption_burn` strips those exact spellings, and
nothing that globs the archival stem may break.
"""
import os

import pytest

from nodes import otr_master_audio_mux as mux

ARCHIVAL = ("signal_lost_arms_at_the_ready_20260903_092133"
            "_silent_procgen_blended_captioned_with_credits_final.mp4")


class _Ledger:
    """Stand-in for the in-flight ledger module."""

    def __init__(self, payload):
        self.payload = payload

    def in_flight_ledger_path(self):
        return "in-memory"

    def load_ledger_safe(self, _path):
        return self.payload


def _install(monkeypatch, payload):
    import sys
    stub = _Ledger(payload)
    monkeypatch.setitem(sys.modules, "_otr_ledger", stub)
    monkeypatch.setattr(mux, "_otr_ledger", stub, raising=False)
    # The helper imports `from . import _otr_ledger`, so patch the package too.
    import nodes
    monkeypatch.setattr(nodes, "_otr_ledger", stub, raising=False)
    return stub


_FULL = {
    "meta": {"visual_style": "cartoon", "source_bank": "public_domain",
             "char_voice_engine": "indextts2",
             # Added 2026-09-07 with the writer + music fields. Both are real
             # ledger keys, confirmed against a production ledger on the 4060.
             "creative_writing_model": "Qwen/Qwen3.5-4B",
             "music_engine": "musicgen",
             "image_engines": {"by_role": {"character_video": {"z_image_turbo": 4}}}},
    "video": {"shots": [{"engine_id": "wan_ti2v"} for _ in range(8)]},
}


def test_the_pipeline_suffix_tail_is_gone(monkeypatch):
    """`_silent_procgen_blended_captioned_with_credits` is compositing noise and
    must not reach the folder the operator watches."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    for noise in ("procgen", "blended", "captioned", "silent", "with_credits"):
        assert noise not in got, (noise, got)


def test_compacted_silent_captioned_credits_does_not_leak_silent(monkeypatch):
    """Live 16 GB AnimateDiff 1-acts land as `<id>_silent_captioned_with_credits`
    (no procgen blend). Matching only `_captioned_with_credits` left `_silent`
    in the obs title, so the watch folder looked like lab leftovers."""
    _install(monkeypatch, _FULL)
    compacted = ("signal_lost_the_weight_of_lead_20260917_014122"
                 "_silent_captioned_with_credits_final.mp4")
    got = mux._obs_basename(compacted)
    assert "silent" not in got, got
    assert got.startswith("the_weight_of_lead_20260917_014122__")
    assert got.endswith("_final.mp4")


def test_the_name_carries_every_choice_as_a_short_code(monkeypatch):
    """Operator ruling 2026-09-07: four characters (five for the writer).

    Spelled in full these fields put the name at 249 of its 250-unit budget on
    a ComfyUI Desktop install -- one unit -- and two more dimensions were wanted
    in it. The codes come from `_otr_shared/shortcodes.py`, whose own tests keep
    the table complete against the live dropdowns and engine registry.
    """
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    for field in ("cart", "wti2", "zimg", "idx2", "pubd", "q354b", "mgen"):
        assert field in got, (field, got)
    # and the spelled-out forms are GONE -- that is the point of the change
    for spelled in ("cartoon", "wan_ti2v", "z_image_turbo", "indextts2",
                    "public_domain", "musicgen"):
        assert spelled not in got, (spelled, got)


def test_the_writer_and_music_dimensions_are_present(monkeypatch):
    """Added 2026-09-07. The writer LLM and the music engine were invisible in
    the published name, so two episodes differing only by writer were
    indistinguishable in the folder the operator actually watches."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert "q354b" in got, got
    assert "mgen" in got, got


def test_episode_leads_and_style_follows(monkeypatch):
    """Operator's chosen order: the folder still sorts by episode, and the two
    axes he compares sit immediately after so they survive truncation."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert got.startswith("arms_at_the_ready_20260903_092133__")
    assert got.index("cart") < got.index("wti2") < got.index("zimg")
    assert got.index("wti2") < got.index("pubd")


def test_the_final_marker_survives(monkeypatch):
    """`scripts/otr_pod_obs_bridge.py` keys on `_final` to recognise a published
    episode -- dropping it would make published work invisible to the bridge."""
    _install(monkeypatch, _FULL)
    assert mux._obs_basename(ARCHIVAL).endswith("_final.mp4")


def test_a_lane_with_no_stills_says_none_rather_than_lying(monkeypatch):
    """Ghost/AnimateDiff renders no stills, so `image_engines.by_role` is empty.
    The field must read `none`, not borrow some other episode's engine."""
    payload = {"meta": dict(_FULL["meta"], image_engines={"by_role": {}}),
               "video": {"shots": [{"engine_id": "animatediff15_v3_haunted_video"}]}}
    _install(monkeypatch, payload)
    got = mux._obs_basename(ARCHIVAL)
    assert "__none__" in got
    # The engine id is coded WHOLE. The old `_trim_engine` stripped a trailing
    # `_video`/`_image` because the field position implied the role -- but the
    # shortcode table is keyed on the engine id exactly as the registry spells
    # it, so trimming first would hand it a name it has never heard of and
    # spell the lane `unk`.
    assert "adhv" in got, got
    assert "animatediff" not in got, got


def test_it_fails_soft_to_the_archival_name(monkeypatch):
    """A publish must never die over a filename. THIS TEST EARNED ITS KEEP: the
    first cut of the helper referenced `re` without importing it, and the broad
    except swallowed the NameError -- silently disabling the whole feature while
    every publish still 'worked'."""
    import sys

    class _Boom:
        def in_flight_ledger_path(self):
            raise RuntimeError("ledger unavailable")

    monkeypatch.setitem(sys.modules, "_otr_ledger", _Boom())
    import nodes
    monkeypatch.setattr(nodes, "_otr_ledger", _Boom(), raising=False)
    assert mux._obs_basename(ARCHIVAL) == ARCHIVAL


def test_the_helper_has_its_imports(monkeypatch):
    """The guard for the bug the soft-fallback hid: exercise the REAL body and
    assert it produced a composed name, not the fallback."""
    _install(monkeypatch, _FULL)
    got = mux._obs_basename(ARCHIVAL)
    assert got != ARCHIVAL, "fell back -- the helper body raised"
    assert "__" in got


def test_fields_are_filesystem_safe():
    assert mux._obs_field("weird/name:here") == "weird-name-here"
    assert mux._obs_field("Anime") == "anime"
    assert mux._obs_field(None) == "none"
    assert mux._obs_field("", "nostyle") == "nostyle"


def test_a_very_long_title_is_capped(monkeypatch):
    _install(monkeypatch, _FULL)
    long_stem = ("signal_lost_" + ("a_very_long_episode_title_" * 8)
                 + "20260903_092133_silent_procgen_blended_captioned_with_credits_final.mp4")
    got = mux._obs_basename(long_stem)
    assert len(got) <= mux._OBS_NAME_MAX + 8, len(got)
    assert got.endswith("_final.mp4")
    assert "cart" in got, "the fields must survive the trim, not the title"


# --- 2026-09-18: the title is the operator's content, kept in its own script --
#
# `_obs_field` strips to ASCII, and the title used to go through it, so three
# shipped languages published with NO TITLE AT ALL: a Japanese episode landed
# as `_ja_20260918_190917__pori__...`, Mandarin as `_zh_...`, Hindi as
# `_-_-_-_hi_...`. Every episode in a script the operator cannot read was
# indistinguishable from the next in the folder he actually browses.
#
# Romanising was measured and rejected: `anyascii` reads 嘘の夜明け as
# "XunoYeMingke" -- the CHINESE reading of Japanese kanji -- and strips the
# vowels out of Devanagari, while `misaki.cutlet` returns IPA rather than
# romaji. A wrong romanisation is worse than none because nothing about it
# announces that it is wrong.

_JA = "\u5618\u306e\u591c\u660e\u3051"          # 嘘の夜明け
_ZH = "\u5982\u9858"                            # 如願
_HI = "\u0938\u0924\u094d\u092f \u0915\u0940 \u0930\u093e\u0924"  # सत्य की रात


@pytest.mark.parametrize("native", [_JA, _ZH, _HI, "conv\u00e9s"])
def test_a_native_script_title_survives_into_the_obs_name(native):
    stem = "%s_xx_20260918_190917" % native
    out = mux._obs_title(stem)
    kept = native.replace(" ", "_").lower()
    assert kept in out, (native, out)
    # The regression it replaces: ASCII-stripping left the language tag alone
    # at the front of the name.
    assert not out.startswith("_")


def test_the_short_code_fields_are_still_ascii_only():
    """`_obs_field` is unchanged and must stay that way -- every value that
    reaches it is one of OUR codes from a fixed table, so there is nothing
    there to preserve, and widening it would change every existing name."""
    for code in ("scif", "koko", "sa3", "vcam", "none", "q354b"):
        assert mux._obs_field(code) == code
    assert mux._obs_field(_JA, "episode") == "episode"


@pytest.mark.parametrize("hostile, want", [
    ('a/b:c*d?e"f<g>h|i', "a-b-c-d-e-f-g-h-i"),   # forbidden everywhere
    ("trailing dot.", "trailing_dot"),            # Windows drops trailing dots
    ("trailing space ", "trailing_space"),        # ... and trailing spaces
    ("", "episode"),
    ("   ", "episode"),
    ("---", "episode"),
])
def test_a_hostile_title_is_made_safe_without_being_emptied(hostile, want):
    assert mux._obs_title(hostile) == want


@pytest.mark.parametrize("reserved", ["CON", "nul.mp4", "com1", "LPT9"])
def test_a_windows_device_name_cannot_be_the_basename(reserved):
    out = mux._obs_title(reserved)
    assert out.split(".")[0].lower() not in mux._WINDOWS_RESERVED
    assert out.startswith("_")


def test_the_name_actually_round_trips_on_this_filesystem(tmp_path):
    """The claim is that the filesystem accepts these, so assert it rather
    than reasoning about it."""
    for native in (_JA, _ZH, _HI):
        name = mux._obs_title("%s_xx_1" % native) + "__scif_final.mp4"
        path = tmp_path / name
        path.write_bytes(b"x")
        assert name in [p.name for p in tmp_path.iterdir()]


# --- the production path, and the two things len() does not measure ----------

def test_obs_basename_itself_keeps_a_native_title(monkeypatch):
    """Through `_obs_basename`, not just the helper. The cursor review's note:
    a helper tested alone proves the helper, never the production path."""
    _install(monkeypatch, _FULL)
    out = mux._obs_basename(
        "signal_lost_%s_ja_20260918_190917_captioned_with_credits_final.mp4"
        % _JA)
    assert _JA in out, out
    assert out.endswith("_final.mp4")
    assert not out.startswith("_")


@pytest.mark.parametrize("budget", list(range(8, 40)))
def test_trimming_never_strands_a_combining_mark(budget):
    """Devanagari is the case: U+0930 U+093E is ra plus a vowel SIGN, and a
    codepoint slice between them leaves a bare combining mark that renders as
    a dotted circle."""
    import unicodedata
    long_hi = (_HI + " ") * 12
    out = mux._trim_title(long_hi, budget)
    assert len(out.encode("utf-8")) <= budget
    assert not (out and unicodedata.combining(out[-1])), repr(out)


@pytest.mark.parametrize("native", [_JA, _HI])
def test_the_cap_is_counted_in_bytes_not_codepoints(native):
    """A CJK codepoint is three UTF-8 bytes, so a 150-CODEPOINT cap is 450
    bytes on any filesystem that counts them (ext4, and any share landing on
    one). The cap has to be a byte cap or it is not a cap."""
    out = mux._trim_title(native * 60, mux._OBS_NAME_MAX)
    assert len(out.encode("utf-8")) <= mux._OBS_NAME_MAX
    assert out, "trimming must not empty a title outright"


def test_the_published_name_still_binds_to_its_episode(tmp_path):
    """PBUG-20260918-09, and the reason this change is not cosmetic.

    `_otr_ledger._published_obs_path` binds the published file to its episode
    by matching the obs stem against the episode id with the show prefix
    removed. ASCII-stripping the title destroyed the front of that stem, so on
    every Japanese, Mandarin and Hindi episode the published artifact was
    REFUSED and `meta.paths.obs_final` recorded nothing -- the file was on
    disk and the ledger could not say so. That is PBUG-20260904-06's exact
    failure mode (a name-bound reader refusing a renamed artifact), recurring
    silently on three languages.
    """
    from nodes import _otr_ledger as ledger
    episode_id = "signal_lost_%s_ja_20260918_190917" % _JA
    name = mux._obs_title("%s_ja_20260918_190917" % _JA) + "__pori__sa3_final.mp4"
    published = tmp_path / name
    published.write_bytes(b"x")
    assert ledger._published_obs_path(
        str(published), inferred_obs_root=tmp_path,
        episode_id=episode_id) == published.resolve()

    # And the shape it replaces does NOT bind -- pinned so a future
    # "simplification" back to ASCII cannot pass quietly.
    stripped = tmp_path / "_ja_20260918_190917__pori__sa3_final.mp4"
    stripped.write_bytes(b"x")
    assert ledger._published_obs_path(
        str(stripped), inferred_obs_root=tmp_path,
        episode_id=episode_id) is None


def test_the_byte_cap_holds_when_processing_SHRINKS_the_title(monkeypatch):
    """The bug the Sonnet post-QA reproduced: the cap was sized against the
    RAW title while the name carried the PROCESSED one.

    Processing shrinks a title whenever a run of forbidden characters or
    whitespace collapses to a single separator, so the fixed part was
    underestimated and the budget overestimated. Measured against the pre-fix formula on the
    two parametrised cases below: the CJK title published at 212 UTF-8 bytes
    and the Devanagari one at 202, both against a 150-byte cap -- silently,
    because an oversized name is still a VALID name and the fail-soft handler
    only catches crashes. (The two ASCII cases do NOT overflow pre-fix; they
    are here to pin that the fix did not break the ordinary path.)
    """
    _install(monkeypatch, _FULL)
    for title in ("::: " * 120,                      # collapses hard
                  (_JA + "::: ") * 40,               # CJK + collapsing runs
                  (_HI + "   ") * 30,                # Devanagari + whitespace
                  "a" * 400):                        # the plain long case
        got = mux._obs_basename(
            "signal_lost_%s_20260918_190917_captioned_with_credits_final.mp4"
            % title)
        assert len(got.encode("utf-8")) <= mux._OBS_NAME_MAX, (
            "%d bytes for %r" % (len(got.encode("utf-8")), got[:60]))
        assert got.endswith("_final.mp4")


# --- the two-word gloss (operator 2026-09-18) -------------------------------
#
# "this is the episode title. I need you to summarize it in two words for a
# file name, index" -- the published folder is the only one he opens, and a
# Japanese or Devanagari title there is a name he cannot read, say or type.

from nodes._otr_shared import obs_name as _N  # noqa: E402


class _GlossLedger:
    """A ledger stub carrying a gloss and a language row."""

    def __init__(self, gloss="", language="ja"):
        meta = {"visual_style": "cartoon", "source_bank": "public_domain",
                "char_voice_engine": "kokoro", "creative_writing_model": "g4",
                "music_engine": "stable_audio_3",
                "image_engines": {"by_role": {}}, "obs_title_gloss": gloss}
        if language:
            # A REAL ledger stamps the ISO here, not the label. Writing
            # "Japanese" made the first draft of these tests fail against
            # working code, and nearly sent the fix into the wrong place.
            meta["episode_language"] = language
        self.payload = {"meta": meta,
                        "video": {"shots": [{"engine_id": "wan_ti2v"}]}}

    def in_flight_ledger_path(self):
        return "in-memory"

    def load_ledger_safe(self, _path):
        return self.payload


def _with_gloss(monkeypatch, gloss, language="ja"):
    import sys
    stub = _GlossLedger(gloss, language)
    monkeypatch.setitem(sys.modules, "_otr_ledger", stub)
    monkeypatch.setattr(mux, "_otr_ledger", stub, raising=False)
    import nodes
    monkeypatch.setattr(nodes, "_otr_ledger", stub, raising=False)
    return stub


_ARCHIVAL_JA = ("signal_lost_%s_ja_20260918_190917"
                "_silent_captioned_with_credits_final.mp4" % _JA)


def test_the_gloss_replaces_the_title_and_keeps_the_language_tag(monkeypatch):
    _with_gloss(monkeypatch, "false_dawn")
    got = mux._obs_basename(_ARCHIVAL_JA)
    assert got.startswith("false_dawn_ja_20260918_190917__"), got
    assert _JA not in got
    assert got.endswith("_final.mp4")


def test_no_gloss_is_byte_identical_to_todays_name(monkeypatch):
    """The gloss is additive. An episode without one publishes exactly as it
    does today, which is what keeps every pre-existing ledger working."""
    _with_gloss(monkeypatch, "")
    got = mux._obs_basename(_ARCHIVAL_JA)
    assert got.startswith("%s_ja_20260918_190917__" % _JA), got


def test_a_ledger_with_no_language_row_invents_no_tag(monkeypatch):
    """The failure the design review caught: reading a two-letter code off the
    id makes `..._the_fall_of_it_20260918_190917` publish as ITALIAN. The iso
    comes from the ledger, so a ledger that does not name one gets no tag."""
    _with_gloss(monkeypatch, "false_dawn", language=None)
    got = mux._obs_basename(_ARCHIVAL_JA)
    assert got.startswith("false_dawn_20260918_190917__"), got
    assert "_it_" not in got and "_ja_" not in got


def test_the_published_gloss_name_binds_to_its_episode(tmp_path):
    """The seam that came apart in PBUG-20260904-06 and -09: the mux WRITES a
    name the ledger must ACCEPT. Both sides now spell the rule from
    `_otr_shared/obs_name.py`."""
    from nodes import _otr_ledger as ledger
    episode_id = "signal_lost_%s_ja_20260918_190917" % _JA
    name = "false_dawn_ja_20260918_190917__cart__sa3_final.mp4"
    published = tmp_path / name
    published.write_bytes(b"x")

    bound = ledger._published_obs_path(
        str(published), inferred_obs_root=tmp_path, episode_id=episode_id,
        obs_title_gloss="false_dawn", episode_iso="ja")
    assert bound == published.resolve()

    # NOT "exactly as strong as today" -- but a WRONG gloss and a WRONG second
    # are both still refused, which is the part that matters.
    assert ledger._published_obs_path(
        str(published), inferred_obs_root=tmp_path, episode_id=episode_id,
        obs_title_gloss="other_words", episode_iso="ja") is None
    wrong_second = tmp_path / "false_dawn_ja_20260918_999999__cart__sa3_final.mp4"
    wrong_second.write_bytes(b"x")
    assert ledger._published_obs_path(
        str(wrong_second), inferred_obs_root=tmp_path, episode_id=episode_id,
        obs_title_gloss="false_dawn", episode_iso="ja") is None


def test_the_native_forms_still_bind_when_a_gloss_exists(tmp_path):
    """Keeping the native forms costs nothing: they only ever match a file
    leading with this episode's own id, which is the no-gloss fallback name."""
    from nodes import _otr_ledger as ledger
    episode_id = "signal_lost_%s_ja_20260918_190917" % _JA
    native = tmp_path / ("%s_ja_20260918_190917__cart__sa3_final.mp4" % _JA)
    native.write_bytes(b"x")
    assert ledger._published_obs_path(
        str(native), inferred_obs_root=tmp_path, episode_id=episode_id,
        obs_title_gloss="false_dawn", episode_iso="ja") == native.resolve()


@pytest.mark.parametrize("raw, want", [
    ("false dawn", "false_dawn"),
    ("Breaking Silence", "breaking_silence"),
    ("caf\u00e9 noir", "cafe_noir"),
    ("reloj inquieto", "reloj_inquieto"),      # Latin non-English: accepted
])
def test_validate_gloss_accepts_what_it_should(raw, want):
    gloss, reason = _N.validate_gloss(raw)
    assert (gloss, reason) == (want, "")


@pytest.mark.parametrize("raw, fragment", [
    (_JA, "not English script"),
    (_HI, "not English script"),
    ("", "empty"),
    ("a two word summary of the given title", "wanted two"),
    ("title", "echoed the instruction"),
    ("supercalifragilisticexpialidocious", "over 16 characters"),
])
def test_validate_gloss_refuses_what_it_should(raw, fragment):
    gloss, reason = _N.validate_gloss(raw)
    assert gloss == ""
    assert fragment in reason, reason
