"""The crux kernel's pair joins with the preposition the PLACE actually takes.

Until 2026-09-05 `resolve_crux_kernel` joined its two halves with a hard-coded
``"in the"``, so every place that is a SURFACE rather than an enclosure read
wrong: "a spinning turntable in the riverbank". A riverbank, a pier, a rooftop
and a flight of stairs are things you stand ON; a crossroads, a gate and a
threshold are points you stand AT. The overwhelming majority of settings a
brief produces are enclosures and still take "in the" -- MEASURED at 7,635 of
8,408 real setting occurrences -- so the default is unchanged and these tests
pin that too: a fix that moved the common case would be a worse bug than the
one it closed.

Three narrow rules, and nothing beyond them:

* the PREPOSITION comes from the place's HEAD NOUN (its last word);
* a place that already opens with its OWN preposition ("under the pier") is
  joined bare, because a second preposition is nonsense;
* a place that already carries its OWN determiner ("his study", "British
  Columbia's Williston Reservoir") does not get a second article.

This is deliberately not a grammar engine. It is a lookup with a default, and
the default is the one that was already shipping.
"""
import pytest

from nodes._otr_video_engines import ghost_signal_author as gsa


# --------------------------------------------------------------------------- #
# 1. The connector itself.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("place, connector", [
    # THE DEFAULT, and it is most of every real brief: an enclosure.
    ("control room", "in the"),
    ("high-security archive", "in the"),
    ("film storage vault", "in the"),
    ("archival library", "in the"),
    ("industrial filing room", "in the"),
    ("forest", "in the"),
    ("petri dish", "in the"),
    # SURFACES. The measured defect: "in the riverbank".
    ("riverbank", "on the"),
    ("concrete floors", "on the"),
    ("loading dock", "on the"),
    ("old coast road", "on the"),
    ("gantry stairs", "on the"),
    ("rooftop", "on the"),
    ("hillside", "on the"),
    # POINTS.
    ("crossroads", "at the"),
    ("gate", "at the"),
    ("checkpoint", "at the"),
    # "station" IS NOT A POINT IN THIS CORPUS and is the biggest head noun in
    # it (108 occurrences): a brief writes "orbital station", "space station",
    # "research station" -- interiors. The draft had it in the AT set until the
    # corpus was read. "terminal" (46) stays AT because a brief means the
    # DEVICE: "data terminal", "computer terminal".
    ("police station", "in the"),
    ("orbital station", "in the"),
    ("secure terminal", "at the"),
    ("arctic outpost", "at the"),
    ("supply depot", "at the"),
    # THE PLACE BRINGS ITS OWN DETERMINER -> no second article.
    ("his study", "in"),
    ("British Columbia's Williston Reservoir", "in"),
    ("the Williston Reservoir", "in"),
    ("a locked ward", "in"),
    # A COMMON-NOUN POSSESSIVE IS NOT A DETERMINER -- it still takes "the".
    ("judge's bench", "in the"),
    ("governor's office", "in the"),
    ("ship's deck", "on the"),
    ("kids' area", "in the"),
    # THE PLACE BRINGS ITS OWN PREPOSITION -> no connector at all.
    ("under the pier", ""),
    ("aboard the night train", ""),
    ("beneath the reservoir", ""),
    # Degenerate input never composes a connector out of nothing.
    ("", ""),
    ("   ", ""),
    (None, ""),
])
def test_the_connector_follows_the_place(place, connector):
    assert gsa.place_preposition(place) == connector


def test_the_head_noun_decides_it_not_the_first_word():
    """A modifier never decides. "coast road" is a road; "road tunnel" is a
    tunnel, and you are IN a tunnel however the road got you there."""
    assert gsa.place_preposition("coast road") == "on the"
    assert gsa.place_preposition("road tunnel") == "in the"
    assert gsa.place_preposition("stair room") == "in the"
    assert gsa.place_preposition("room stair") == "on the"


def test_a_plural_place_resolves_like_its_singular():
    assert gsa.place_preposition("riverbanks") == gsa.place_preposition("riverbank")
    assert gsa.place_preposition("rooftops") == gsa.place_preposition("rooftop")
    assert gsa.place_preposition("gates") == gsa.place_preposition("gate")


def test_a_hyphenated_head_reads_its_last_segment():
    assert gsa.place_preposition("high-security archive") == "in the"
    assert gsa.place_preposition("service walk-path") == "on the"


# --------------------------------------------------------------------------- #
# 2. The composed kernel.
# --------------------------------------------------------------------------- #

def _kernel(objects, settings, **kw):
    meta = {"key_objects": list(objects),
            "story_brief_terms": {"setting": list(settings)}}
    kw.setdefault("role", "character_video")
    kw.setdefault("mode", "object")
    kw.setdefault("ordinal", 0)
    return gsa.resolve_crux_kernel(meta, **kw)[0]


def test_the_riverbank_pair_is_the_defect_this_row_closed():
    assert _kernel(["a spinning turntable"], ["riverbank"]) == (
        "a spinning turntable on the riverbank")


def test_the_common_enclosure_pair_is_byte_identical_to_what_shipped():
    """The fix must not move the case that was already right."""
    assert _kernel(["film canisters"], ["high-security archive"]) == (
        "film canisters in the high-security archive")
    assert _kernel(["archive reels"], ["control room"]) == (
        "archive reels in the control room")


def test_a_place_carrying_its_own_preposition_is_joined_bare():
    assert _kernel(["a brass emblem"], ["under the pier"]) == (
        "a brass emblem under the pier")


def test_a_possessive_place_takes_no_second_article():
    assert _kernel(["a bakelite radio set"],
                   ["British Columbia's Williston Reservoir"]) == (
        "a bakelite radio set in British Columbia's Williston Reservoir")


# --------------------------------------------------------------------------- #
# 3. THE POSSESSIVE IS NOT A DETERMINER -- caught in QA before this shipped.
# --------------------------------------------------------------------------- #

#: Every possessive setting in the real corpus, all 24 of them, split exactly
#: as the capital splits them. Pulled from the 4,393 unique
#: `story_brief_terms.setting` values across 2,140 episode ledgers on the dev
#: box: 9 carry a capitalised possessive and 15 a lowercase one, and the first
#: draft of this fix dropped the article from ALL of them -- which was right
#: for the 9 and wrong for the 15.
REAL_PROPER_POSSESSIVES = [
    "Victor's home", "Victor's study", "Victor's laboratory",
    "MIT's Neuroscience Lab", "Widow's Hill", "Moon's surface",
    "Olivia's estate", "NASA's Jet Propulsion Laboratory",
    "Library's National Audio-Visual Conservation Center",
]
REAL_COMMON_POSSESSIVES = [
    "judge's bench", "ship's course", "ship's parlor", "ship's deck",
    "strikers' cluster", "elders' watch", "neighbor's house",
    "governor's office", "traveler's coat", "kids' area",
    "librarian's office", "master's tent", "doctor's office",
    "elephant's den", "moon's surface",
]


@pytest.mark.parametrize("place", REAL_COMMON_POSSESSIVES)
def test_a_common_noun_possessive_keeps_its_article(place):
    """"the judge's bench", never "in judge's bench"."""
    connector = gsa.place_preposition(place)
    assert connector.endswith(" the"), (place, connector)


@pytest.mark.parametrize("place", REAL_PROPER_POSSESSIVES)
def test_a_proper_noun_possessive_takes_no_article(place):
    """"in Victor's study", never "in the Victor's study"."""
    connector = gsa.place_preposition(place)
    assert connector and not connector.endswith(" the"), (place, connector)


def test_the_capital_is_the_whole_test_and_nothing_else_is():
    """One string, two casings, two answers -- that is the entire rule."""
    assert gsa.place_preposition("moon's surface") == "on the"
    assert gsa.place_preposition("Moon's surface") == "on"


def test_the_tautology_guard_still_wins_over_the_connector():
    """A place that IS the subject drops, connector or not."""
    assert _kernel(["riverbank"], ["riverbank"]) == "riverbank"


def test_an_over_long_pair_still_drops_the_whole_place():
    kernel = _kernel(["a hand-annotated hydrographic survey chart"],
                     ["decommissioned subsurface monitoring station"])
    assert kernel == "a hand-annotated hydrographic survey chart"


def test_the_word_budget_counts_the_connector_it_actually_used():
    """A one-word connector buys the place one more word of budget.

    ``GHOST_V3_KERNEL_MAX_WORDS`` is 9. "a dented tin lantern" (4) plus
    "in the" (2) plus a four-word place is 10 and drops the place; the same
    four-word place that brings its OWN article spends one word less on the
    connector and fits at exactly 9.
    """
    assert gsa.GHOST_V3_KERNEL_MAX_WORDS == 9
    dropped = _kernel(["a dented tin lantern"],
                      ["long disused pumping shed"])
    assert dropped == "a dented tin lantern"
    kept = _kernel(["a dented tin lantern"],
                   ["the disused pumping shed"])
    assert kept == "a dented tin lantern in the disused pumping shed"


def test_every_setting_of_a_real_episode_shape_still_composes():
    """Totality is the ladder's whole claim; the connector must not break it."""
    meta = {
        "key_objects": ["film canisters", "handwritten ledgers", "ink pens"],
        "story_brief_terms": {
            "setting": ["high-security archive", "riverbank", "crossroads",
                        "under the pier", "British Columbia's Williston Reservoir"],
        },
    }
    for ordinal in range(20):
        for role, mode in (("character_video", "figure"),
                           ("announcer_visual", "object"),
                           ("music_visual", "signal")):
            kernel, source = gsa.resolve_crux_kernel(
                meta, ordinal=ordinal, role=role, mode=mode)
            assert kernel.strip(), (ordinal, role, mode)
            assert "  " not in kernel, (ordinal, role, mode, kernel)
            assert source in ("bookend_radio", "key_object", "setting",
                              "brief", "bookend")
