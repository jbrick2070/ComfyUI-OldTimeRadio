"""ONE STYLE AUTHORITY -- the pack-owned negative and the shared style token.

PBUG-20260817-01: `z_image_turbo` shipped a style-BLIND default negative
containing "cartoon, illustration". On a `cartoon` episode every still was
minted with the positive "bright cartoon illustration" AND that negative, so
the engine vetoed the style the episode selected. It fought four of nine packs
(anime, cartoon, sci_fi_radio, storybook_engraving) by literal phrase match.

These pin the three things that make the fix real rather than cosmetic:
the engine constant no longer carries a style opinion, the pack negative is
COMPOSED with the request negative instead of replacing it, and an empty
negative falls back to hygiene rather than to nothing.
"""
import json
import os
import pathlib
import sys

import pytest

_NODES = pathlib.Path(__file__).resolve().parents[1] / "nodes"
if str(_NODES) not in sys.path:
    sys.path.insert(0, str(_NODES))

import _otr_visual_styles as vs  # noqa: E402
from nodes._otr_image_engines import z_image_turbo as zit  # noqa: E402

_PACK_DIR = _NODES / "visual_styles"
_ALL_IDS = tuple(sorted(p.stem for p in _PACK_DIR.glob("*.json")))

#: The exact string `z_image_turbo` hardcoded before 2026-08-17. The three
#: photoreal packs carry it VERBATIM so the default lane is unchanged.
_HISTORICAL = ("oversaturated, glossy, clean digital, plastic skin, waxy skin, "
               "sterile studio lighting, cartoon, illustration, text, watermark")
_PHOTOREAL_PACKS = ("sci_fi_radio", "shakespeare_stage_realism",
                    "archival_documentary")
#: Packs whose look is non-photoreal: their negative must never veto it.
_ILLUSTRATION_PACKS = ("cartoon", "anime", "storybook_engraving",
                       "paper_origami")


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


class TestPackNegative:
    @pytest.mark.parametrize("style_id", _ALL_IDS)
    def test_every_shipped_pack_declares_a_negative(self, style_id):
        """Schema-OPTIONAL, but never absent in practice.

        The field is optional so FROZEN `embedded_visual_style_pack` ledgers
        still validate (a required key would fail them on the missing-key path,
        and injecting a default would break their sha256 receipt). Optional
        protects history; this test protects the present.
        """
        assert vs.resolve_visual_style(style_id).negative_tail.strip()

    @pytest.mark.parametrize("style_id", _PHOTOREAL_PACKS)
    def test_photoreal_packs_keep_the_historical_string_verbatim(self, style_id):
        assert vs.resolve_visual_style(style_id).negative_tail == _HISTORICAL

    @pytest.mark.parametrize("style_id", _ILLUSTRATION_PACKS)
    def test_illustration_packs_do_not_veto_themselves(self, style_id):
        """THE regression that would make this whole build cosmetic."""
        negative = vs.resolve_visual_style(style_id).negative_tail.lower()
        assert "cartoon" not in negative
        assert "illustration" not in negative

    def test_optional_field_is_known_but_not_required(self):
        """A pack WITHOUT the key still validates -- the frozen-ledger path."""
        raw = json.loads((_PACK_DIR / "cartoon.json").read_text("utf-8"))
        raw.pop("negative_tail")
        assert vs.validate_pack(raw, expected_style_id="cartoon").negative_tail == ""

    def test_optional_field_is_still_type_checked_when_present(self):
        """Optional must not mean unvalidated: a non-str would reach the
        dataclass and fail much later inside a string join."""
        raw = json.loads((_PACK_DIR / "cartoon.json").read_text("utf-8"))
        raw["negative_tail"] = ["not", "a", "string"]
        with pytest.raises(vs.VisualStyleValidationError):
            vs.validate_pack(raw, expected_style_id="cartoon")

    def test_unknown_keys_are_still_rejected(self):
        """Splitting KNOWN from REQUIRED must not weaken the typo guard."""
        raw = json.loads((_PACK_DIR / "cartoon.json").read_text("utf-8"))
        raw["negatve_tail"] = "typo"
        with pytest.raises(vs.VisualStyleValidationError):
            vs.validate_pack(raw, expected_style_id="cartoon")


class TestSelfVetoResolution:
    """Operator ruling 2026-08-17: a negative may never conflict with a style.

    Moving the negative into the packs fixed the CROSS-pack case. This is the
    SELF-veto case: `sci_fi_radio` declares "cartoon, illustration" while its
    own announcer-mouth and title-card surfaces ask for exactly those words.
    """

    @pytest.mark.parametrize("style_id", _ALL_IDS)
    def test_no_pack_suppresses_what_it_asks_for(self, style_id):
        style = vs.resolve_visual_style(style_id)
        effective = vs.effective_negative(style).lower()
        positive = " ".join(
            str(getattr(style, f, "") or "") for f in vs._POSITIVE_SURFACES
        ).lower()
        for phrase in [c.strip() for c in effective.split(",") if c.strip()]:
            if phrase in vs._ARTIFACT_PHRASES:
                continue
            assert phrase not in positive, (
                "%s suppresses %r which its own positive asks for"
                % (style_id, phrase))

    def test_the_default_pack_self_veto_is_actually_resolved(self):
        """The measured case: both conflicting phrases must be gone."""
        style = vs.resolve_visual_style("sci_fi_radio")
        assert "cartoon" in style.negative_tail        # still authored
        effective = vs.effective_negative(style).lower()
        assert "cartoon" not in effective
        assert "illustration" not in effective

    def test_resolution_keeps_the_non_conflicting_terms(self):
        """It removes the conflict, not the negative."""
        effective = vs.effective_negative(vs.resolve_visual_style("sci_fi_radio"))
        for term in ("oversaturated", "plastic skin", "waxy skin", "watermark"):
            assert term in effective

    def test_artifact_terms_are_never_stripped(self):
        """"text" appears in title-card descriptions but is anti-artifact, not
        a request for a watermark."""
        for style_id in _ALL_IDS:
            assert "watermark" in vs.effective_negative(
                vs.resolve_visual_style(style_id))

    def test_phrase_matching_not_word_matching(self):
        """"plastic skin" must survive a positive that wants realistic skin."""
        class _Style:
            style_id = "probe"
            negative_tail = "plastic skin, cartoon, watermark"
            portrait_look = "detailed realistic skin and cartoon outlines"
        effective = vs.effective_negative(_Style())
        assert "plastic skin" in effective   # phrase absent from the positive
        assert "cartoon" not in effective    # phrase present -> dropped

    def test_empty_negative_stays_empty(self):
        class _Style:
            style_id = "probe"
            negative_tail = ""
        assert vs.effective_negative(_Style()) == ""


class TestEngineNegative:
    def test_engine_constant_carries_no_style_opinion(self):
        """The style half moved into the packs; only hygiene stays engine-side."""
        hygiene = zit._HYGIENE_NEGATIVE.lower()
        for term in ("cartoon", "illustration", "clean digital"):
            assert term not in hygiene
        for term in ("text", "watermark", "plastic skin"):
            assert term in hygiene

    def test_request_negative_is_honoured(self, monkeypatch):
        """Before this, the engine read env-or-hardcoded and IGNORED the
        request, so the dispatcher's negative was computed and discarded."""
        monkeypatch.delenv("OTR_ZIMAGE_NEGATIVE", raising=False)
        assert zit._resolve_negative("photograph, film grain") == \
            "photograph, film grain"

    def test_empty_request_falls_back_to_hygiene_not_to_nothing(self, monkeypatch):
        """A legacy frozen pack resolves to "" -- it must keep its
        anti-artifact protection rather than mint with no negative at all."""
        monkeypatch.delenv("OTR_ZIMAGE_NEGATIVE", raising=False)
        assert zit._resolve_negative("") == zit._HYGIENE_NEGATIVE
        assert zit._resolve_negative(None) == zit._HYGIENE_NEGATIVE

    def test_env_override_still_wins_when_explicitly_set(self, monkeypatch):
        monkeypatch.setenv("OTR_ZIMAGE_NEGATIVE", "explicit override")
        assert zit._resolve_negative("pack terms") == "explicit override"

    def test_params_thread_the_request_negative(self, monkeypatch):
        monkeypatch.delenv("OTR_ZIMAGE_NEGATIVE", raising=False)
        params = zit.ZImageTurboEngine()._zimage_params(
            {"prompt": "a face", "seed": 1, "negative_prompt": "photograph"})
        assert params["negative"] == "photograph"


class TestStyleToken:
    def test_default_pack_is_exempt(self):
        """sci_fi_radio IS the house look and its tails are byte-pinned;
        prefixing it would churn every default-lane golden for no gain."""
        assert vs.compact_style_cue(vs.resolve_visual_style("sci_fi_radio")) == ""

    @pytest.mark.parametrize("style_id", _ILLUSTRATION_PACKS)
    def test_non_default_packs_get_a_token(self, style_id):
        assert vs.compact_style_cue(vs.resolve_visual_style(style_id))

    def test_prefix_is_positional_not_membership(self):
        """The token is usually ALREADY present at the tail (finish_visual_prompt
        appends it last), so a 'contains' test would skip exactly the prompts
        that fractured. It must front-anchor."""
        style = vs.resolve_visual_style("cartoon")
        prompt = "a man in a doorway, bright cartoon illustration"
        out = vs.prefix_style_cue(style, prompt)
        assert out != prompt
        assert out.lower().startswith("bright cartoon")

    def test_prefix_is_idempotent(self):
        style = vs.resolve_visual_style("cartoon")
        once = vs.prefix_style_cue(style, "a man in a doorway")
        assert vs.prefix_style_cue(style, once) == once

    def test_prefix_is_additive_only(self):
        """It may never drop authored text."""
        style = vs.resolve_visual_style("cartoon")
        prompt = "a man in a doorway holding a lantern"
        assert prompt in vs.prefix_style_cue(style, prompt)

    def test_empty_prompt_is_left_alone(self):
        assert vs.prefix_style_cue(vs.resolve_visual_style("cartoon"), "") == ""

    def test_no_double_punctuation_when_prompt_opens_with_the_bare_token(self):
        """Stripping only "." left `"Cartoon, a man"` -> `"cartoon style. , a
        man"`. Latent in the shipped video helper before this build."""
        class _Style:
            style_id = "paper_origami"
            positive_tail = "origami style"
            portrait_look_talking = ""
        out = vs.prefix_style_cue(_Style(), "Origami, a man in a doorway")
        assert ". ," not in out
        assert "  " not in out
