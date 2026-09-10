"""My Story input intake: what is admitted, what is refused, what is kept.

The refusals are the point. Every one of them exists because the alternative
is an episode that renders successfully and is not what the person asked for:
notes silently ignored, a frozen source replacing their words, or several
minutes of model work spent on a blank submission.

Pure / CPU. UTF-8 no BOM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_input as SI  # noqa: E402


MINE = SI.StoryInputPolicy(mode=SI.INPUT_MODE_USER_FIELDS, bank_id="my_story")
LEGACY = SI.StoryInputPolicy(mode=SI.INPUT_MODE_LEGACY, bank_id="media_archive")


def _req(**kw):
    base = dict(num_characters=2, act_count="1", include_act_breaks=True,
                source_bank_requested="my_story",
                visual_style_requested="viz_camera")
    base.update(kw)
    return SI.StoryRequest(**base)


# ---------------------------------------------------------------------------
# each field alone is enough to write from -- except the author
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["idea", "characters", "plot", "setting"])
def test_any_single_creative_field_is_enough(field):
    raw = SI.capture_raw(**{field: "something to go on"})
    assert SI.creative_input_present(raw)
    SI.check_selection(raw, MINE)


def test_the_author_alone_is_not_a_story():
    """Naming who a story is by does not supply a story.

    Worth its own test because it is the one field that looks like input and
    is not: a person who fills only this has told us the byline and nothing
    to put under it.
    """
    raw = SI.capture_raw(author="Jeffrey Brick")
    assert not SI.creative_input_present(raw)
    with pytest.raises(SI.StoryInputError) as caught:
        SI.check_selection(raw, MINE)
    assert "nothing to write from" in str(caught.value)


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", "\t \n  \r\n"])
def test_whitespace_is_not_input(blank):
    with pytest.raises(SI.StoryInputError):
        SI.check_selection(SI.capture_raw(idea=blank, plot=blank), MINE)


def test_all_four_fields_together():
    raw = SI.capture_raw(idea="a diver finds a door", characters="Ada; Tom",
                         plot="they open it", setting="1890s Cornwall")
    SI.check_selection(raw, MINE)
    payload = SI.project_payload(SI.build_bundle(raw, _req()), "2026-09-10")
    for heading in ("IDEA:", "CHARACTERS:", "PLOT:", "SETTING:"):
        assert heading in payload["full_text"]


# ---------------------------------------------------------------------------
# the raw text survives exactly as typed
# ---------------------------------------------------------------------------

def test_the_exact_words_are_preserved_beside_the_normalized_view():
    messy = "  a keeper\n\n\n\n  hears a voice   \n"
    bundle = SI.build_bundle(SI.capture_raw(idea=messy), _req())
    assert bundle.fields.idea == messy, "raw evidence must never be rewritten"
    assert bundle.normalized.idea == "a keeper\n\n  hears a voice"


def test_a_list_of_names_keeps_its_line_breaks():
    """Someone who typed one character per line meant those lines."""
    listed = "Ada, the keeper\nTom, her brother\nThe voice"
    bundle = SI.build_bundle(SI.capture_raw(characters=listed), _req())
    assert bundle.normalized.characters.count("\n") == 2


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_the_digest_is_stable_for_identical_input():
    raw, req = SI.capture_raw(idea="x"), _req()
    assert SI.build_bundle(raw, req).digest == SI.build_bundle(raw, req).digest


def test_the_digest_changes_when_any_field_changes():
    base = SI.build_bundle(SI.capture_raw(idea="x"), _req()).digest
    assert SI.build_bundle(SI.capture_raw(idea="y"), _req()).digest != base
    assert SI.build_bundle(SI.capture_raw(idea="x", plot="p"),
                           _req()).digest != base
    assert SI.build_bundle(SI.capture_raw(idea="x"),
                           _req(act_count="3")).digest != base


def test_the_digest_ignores_the_rolled_result_by_construction():
    """The identity covers what was SUBMITTED, so the validator and the writer
    agree about it even though the rolls happen between them."""
    raw = SI.capture_raw(idea="x")
    before = SI.build_bundle(raw, _req(visual_style_requested="roll (any style)"))
    after = SI.build_bundle(raw, _req(visual_style_requested="viz_camera"))
    assert before.digest != after.digest, (
        "the REQUEST is part of the identity, so a differently-captured "
        "request is a different submission -- which is exactly why the "
        "writer must reuse the pre-roll capture rather than rebuild one")


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["characters", "plot", "setting", "author"])
def test_a_creator_field_on_another_bank_is_refused_and_named(field):
    raw = SI.capture_raw(idea="a premise", **{field: "value"})
    with pytest.raises(SI.StoryInputError) as caught:
        SI.check_selection(raw, LEGACY)
    message = str(caught.value)
    assert SI.FIELD_LABELS[field] in message
    assert "media_archive" in message


def test_a_premise_alone_on_another_bank_is_ordinary():
    """`custom_premise` is shared, and typing one on the archive lane is a
    normal thing to do -- only the dedicated fields are a mismatch."""
    SI.check_selection(SI.capture_raw(idea="a premise"), LEGACY)


def test_the_roll_sentinel_counts_as_another_bank():
    sentinel = SI.StoryInputPolicy(mode=SI.INPUT_MODE_LEGACY,
                                   bank_id="roll (any eligible bank)")
    with pytest.raises(SI.StoryInputError):
        SI.check_selection(SI.capture_raw(plot="x"), sentinel)


def test_replay_is_refused_because_it_would_ignore_the_input():
    with pytest.raises(SI.StoryInputError) as caught:
        SI.check_selection(SI.capture_raw(idea="x"), MINE,
                           replay_from="/bundles/ep1")
    assert "replay_from" in str(caught.value)


def test_a_configured_snapshot_manifest_is_refused():
    with pytest.raises(SI.StoryInputError) as caught:
        SI.check_selection(SI.capture_raw(idea="x"), MINE,
                           snapshot_manifest_configured=True)
    assert "OTR_SOURCE_SNAPSHOT_MANIFEST" in str(caught.value)


def test_a_source_ref_is_refused_because_this_bank_has_no_fetcher():
    with pytest.raises(SI.StoryInputError) as caught:
        SI.check_selection(SI.capture_raw(idea="x"), MINE,
                           source_ref="folger-macbeth:act1")
    assert "source_ref" in str(caught.value)


def test_a_linked_value_where_text_belongs_is_named_not_guessed():
    """A link reaching a text field is refused BY NAME.

    Guessing is how a `[node, slot]` pair ends up inside an episode as prose.
    """
    with pytest.raises(SI.StoryInputError) as caught:
        SI.capture_raw(idea=["5", 0])
    message = str(caught.value).lower()
    assert "story input" in message and "list" in message


def test_none_reads_as_empty_rather_than_failing():
    """An absent widget on an older saved graph is empty, not an error."""
    assert SI.capture_raw(idea=None, plot=None).idea == ""


# ---------------------------------------------------------------------------
# the payload projection
# ---------------------------------------------------------------------------

def test_the_payload_satisfies_the_shared_source_contract():
    from nodes import _otr_source_payload as SP

    bundle = SI.build_bundle(SI.capture_raw(idea="a diver finds a door"), _req())
    clean = SP.validate_source_payload(
        SI.project_payload(bundle, "2026-09-10"), origin="test")
    assert set(clean) == set(SP.SOURCE_PAYLOAD_KEYS)
    assert clean["seed_text"].strip()


def test_the_headline_is_bounded_and_marked():
    long_idea = "a lighthouse keeper " * 40
    bundle = SI.build_bundle(SI.capture_raw(idea=long_idea), _req())
    headline = SI.project_payload(bundle, "2026-09-10")["headline"]
    assert headline.startswith("My Story: ")
    assert len(headline) < 120 and headline.endswith("...")


def test_a_blank_section_is_omitted_not_left_empty():
    bundle = SI.build_bundle(SI.capture_raw(idea="x"), _req())
    text = SI.project_payload(bundle, "2026-09-10")["full_text"]
    assert "IDEA:" in text
    for absent in ("CHARACTERS:", "PLOT:", "SETTING:", "BY:"):
        assert absent not in text


def test_the_seed_falls_back_to_the_first_field_that_has_text():
    """Someone may fill only the plot box and never the idea box."""
    bundle = SI.build_bundle(SI.capture_raw(plot="they open the door"), _req())
    assert SI.project_payload(bundle, "2026-09-10")["seed_text"] == \
        "they open the door"


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------

def test_a_supplied_name_is_spoken_and_printed_verbatim():
    name = "Jeffrey Brick"
    assert name in SI.attribution_sentence(name)
    assert name in SI.credits_source_line(name)
    receipt = SI.attribution_receipt(name)
    assert receipt["author"] == name
    assert receipt["source"] == "story_author widget"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_no_name_gives_a_neutral_line_and_never_invents_one(blank):
    sentence = SI.attribution_sentence(blank)
    credit = SI.credits_source_line(blank)
    assert sentence == SI.ANONYMOUS_ATTRIBUTION
    assert credit == SI.ANONYMOUS_CREDIT
    assert "listener" in sentence.lower()
    assert SI.attribution_receipt(blank)["source"] == "none supplied"


def test_the_operators_name_is_never_a_default_anywhere():
    """Operator directive 2026-09-10: the announcer must not credit him by
    default. Nothing in this module may carry a real person's name."""
    import inspect

    source = inspect.getsource(SI)
    for forbidden in ("Jeffrey", "Brick", "jbrick"):
        assert forbidden not in source, (
            "%r appears in the input module; no name may be a default"
            % forbidden)


def test_the_name_is_surrounded_but_not_reworded():
    """A pen name with odd punctuation must survive intact."""
    for name in ("K. A. Applegate", "de la Cruz", "X AE A-12"):
        assert name in SI.attribution_sentence(name)


def test_resolver_preserves_user_fields_and_the_unclamped_request(monkeypatch):
    from nodes import _otr_writer_inputs as WI, _otr_source_snapshot as SNAP
    def forbidden(*a, **kw):
        pytest.fail("My Story attempted to load a source snapshot")
    monkeypatch.setattr(SNAP, "load_snapshot_for_bank", forbidden)
    monkeypatch.delenv("OTR_SOURCE_SNAPSHOT_MANIFEST", raising=False)
    result = WI._resolve_inputs(source_bank="my_story", custom_premise="my idea",
        story_characters="Ada and Tom", story_plot="ring the bell", story_setting="a lighthouse",
        story_author="A. Listener", num_characters=99, act_count="1")
    meta = result["source_meta"]
    assert meta["requested_num_characters"] == 99
    assert meta["story_input"]["request"]["num_characters"] == 99
    assert meta["story_input"]["fields"]["plot"] == "ring the bell"
    assert result["seed_source"] == "my_story_fields"
