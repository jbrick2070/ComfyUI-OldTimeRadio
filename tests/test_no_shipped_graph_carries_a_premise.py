"""The shared graphs must not carry a premise. Written 2026-09-13 after one did.

`custom_premise` is NOT an inert creative field on banks other than my_story --
it is their SOURCE OVERRIDE. `_otr_writer_inputs._resolve_inputs` takes an
`elif custom:` branch that synthesizes a news_article from the premise text and
sets seed_source="custom_premise", so a non-empty value SKIPS the RSS fetch,
the archive manifest and the Folger fetch entirely.

A standing premise was written into the canonical's widget so a blank My Story
pick had something to perform. It did -- and it also silently seeded four of
the six banks with that same story, with no error and no refusal. The floor in
`_otr_story_input.DEFAULT_IDEA` already covers the blank-My-Story case for
every graph including a fresh node, which is exactly why the widget never
needed to carry it.

The widget stays EMPTY on every shipped graph. The floor is the only place the
standing premise lives.
"""
import json
from pathlib import Path

import pytest

from tests._support.writer_slots import value as widget_value

PACK_ROOT = Path(__file__).resolve().parent.parent

from tests._support.shipped_graphs import shipped_graphs  # noqa: E402

WORKFLOWS = shipped_graphs()

#: Slot index of custom_premise on OTR_LedgerScriptWriter, positional.
_PREMISE_SLOT = 4
_MY_STORY_CREATIVE_WIDGETS = (
    "custom_premise", "story_characters", "story_plot", "story_setting")


def _writer(graph):
    for node in graph["nodes"]:
        if node.get("type") == "OTR_LedgerScriptWriter":
            return node
    return None


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_shipped_graph_carries_a_premise(path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    node = _writer(graph)
    assert node is not None, "%s has no writer" % path.name
    value = (node["widgets_values"][_PREMISE_SLOT] or "").strip()
    assert value == "", (
        "%s ships a non-empty custom_premise (%d chars). That is a SOURCE "
        "OVERRIDE for every bank with a fetcher -- scifi_news_pro, "
        "media_archive, public_domain and shakespeare would all be seeded "
        "from it instead of fetching, silently. The standing premise belongs "
        "in _otr_story_input.DEFAULT_IDEA, which already covers a blank My "
        "Story run." % (path.name, len(value)))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
@pytest.mark.parametrize("widget", _MY_STORY_CREATIVE_WIDGETS)
def test_no_shipped_graph_carries_my_story_creative_text(path, widget):
    """A standing value in any creative My Story widget makes every run typed.

    house_source is computed from the raw widgets. A shipped plot, setting or
    cast list would disable the skip on every rolled My Story pick, the same
    class of silent seed the custom_premise guard already refuses.
    """
    graph = json.loads(path.read_text(encoding="utf-8"))
    node = _writer(graph)
    assert node is not None, "%s has no writer" % path.name
    saved = (widget_value(node, widget) or "").strip()
    assert saved == "", (
        "%s ships a non-empty %s (%d chars). That value would stamp "
        "house_source=False on every My Story run of this graph."
        % (path.name, widget, len(saved)))


def test_the_floor_still_covers_what_the_widget_was_doing():
    """Clearing the widget must not reopen the hole it was filling."""
    from nodes import _otr_story_input as si
    from nodes import _otr_story_routing as rt

    blank = si.capture_raw(idea="", characters="", plot="", setting="",
                           author="")
    row = rt.find_bank("my_story")
    policy = si.StoryInputPolicy(
        mode=rt.story_input_mode(row), bank_id="my_story")
    floored = si.with_default_idea(blank, policy)
    si.check_selection(floored, policy)            # must not raise
    assert floored.idea == si.DEFAULT_IDEA

    # ...and a fetcher bank is NOT floored, so its fetch still runs.
    news = rt.find_bank("scifi_news_pro")
    news_policy = si.StoryInputPolicy(
        mode=rt.story_input_mode(news), bank_id="scifi_news_pro")
    assert si.with_default_idea(blank, news_policy) is blank
