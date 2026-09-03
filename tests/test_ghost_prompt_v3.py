"""Ghost PROMPT v3 -- "draw the crux". The composer, the ladder, the fitter.

Prompt v3 Half A composes each beat from the EPISODE (its `key_objects`, its
setting, its light) plus the stored beat's MODE, and reads neither the stored
`motif_cue` nor the stored `drawable_beat`. These tests pin the four claims that
make that safe:

* the kernel ladder is TOTAL -- it never raises and never returns empty, because
  the measured rule on this lane is that legibility tracks concrete nouns;
* the composed text carries no costume, no authored leaf and no mode law;
* the fitter drops WHOLE units and never slices a subject;
* the world-motion pool does not repeat inside a real episode.
"""
import pytest

from nodes._otr_video_engines import ghost_signal_author as gsa
from nodes._otr_video_engines import ghost_signal_prompt as gsp
from nodes import _otr_visual_styles as vs


REAL_META = {
    "episode_seed": 4242,
    "key_objects": ["film canisters", "handwritten ledgers", "archive shelves",
                    "ink pens", "security badges"],
    "story_brief_terms": {
        "setting": ["high-security archive", "film storage vault",
                    "industrial filing room", "archival library"],
        "lighting": ["harsh fluorescent overheads", "dim pools of light",
                     "shadowy corners", "clinical white light"],
    },
    "story_brief": ("A high-security archive filled with film canisters and "
                    "dusty ledgers where an archivist and a cynical consultant "
                    "race against a security sweep."),
}

EVERY_MODE = (("character_video", "figure"),
              ("announcer_visual", "object"),
              ("music_visual", "signal"))


def _style(style_id="storybook_engraving"):
    return vs.resolve_visual_style(style_id)


# --------------------------------------------------------------------------- #
# 1. The kernel ladder is TOTAL. Every tier ends in a concrete noun.
# --------------------------------------------------------------------------- #

def test_the_first_tier_is_the_story_object_in_the_story_place():
    kernel, source = gsa.resolve_crux_kernel(REAL_META, ordinal=0,
                                             role="character_video",
                                             mode="figure")
    assert source == "key_object"
    assert "film canisters" in kernel
    assert "high-security archive" in kernel


def test_the_ladder_falls_to_setting_then_brief_then_the_bookend_radio():
    """Walked on a CHARACTER beat: a bookend takes the radio before tier 1."""
    no_objects = dict(REAL_META, key_objects=[])
    kernel, source = gsa.resolve_crux_kernel(no_objects, ordinal=0,
                                             role="character_video",
                                             mode="figure")
    assert source == "setting" and kernel

    no_terms = dict(REAL_META, key_objects=[],
                    story_brief_terms={"setting": [], "lighting": []})
    kernel, source = gsa.resolve_crux_kernel(no_terms, ordinal=0,
                                             role="character_video",
                                             mode="figure")
    assert source == "brief" and kernel

    nothing = {"key_objects": [], "story_brief": "",
               "story_brief_terms": {"setting": [], "lighting": []}}
    # a CHARACTER beat with nothing left at all still names the radio
    kernel, source = gsa.resolve_crux_kernel(nothing, ordinal=0,
                                             role="character_video",
                                             mode="figure")
    assert source == "bookend"
    assert kernel == "a bakelite radio set"


@pytest.mark.parametrize("role,mode", EVERY_MODE)
@pytest.mark.parametrize("meta", [
    {},
    {"key_objects": [], "story_brief": "", "story_brief_terms": {}},
    {"key_objects": None, "story_brief_terms": None, "story_brief": None},
    {"key_objects": ["   "], "story_brief_terms": {"setting": ["  "]},
     "story_brief": "   "},
])
def test_the_ladder_never_raises_and_never_returns_empty(meta, role, mode):
    """A brief-failed row must still NAME A THING.

    The measured pair recorded above `GHOST_MODE_LAWS_V2`: a prompt naming a
    concrete thing rendered a recognisable subject on 4 of 4 sampled beats, one
    asking for a field of texture on 0 of 4. An empty kernel would rebuild the
    0-of-4 condition on every episode whose brief failed.
    """
    kernel, source = gsa.resolve_crux_kernel(meta, ordinal=7, role=role,
                                             mode=mode)
    assert kernel.strip(), (meta, role, mode)
    assert source in ("bookend_radio", "key_object", "setting", "brief",
                      "bookend")


def test_the_kernel_is_never_word_sliced():
    """Whole units only. "data logs" must not become "data"."""
    long_pair = {
        "key_objects": ["a hand-annotated hydrographic survey chart"],
        "story_brief_terms": {
            "setting": ["a decommissioned subsurface monitoring station"]},
    }
    kernel, source = gsa.resolve_crux_kernel(long_pair, ordinal=0,
                                             role="character_video",
                                             mode="figure")
    assert source == "key_object"
    # the PLACE was dropped as one piece; the subject survives entire
    assert kernel == "a hand-annotated hydrographic survey chart"


# --------------------------------------------------------------------------- #
# 2. The composed text: no costume, no leaf, no law.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("role,mode", EVERY_MODE)
def test_the_composer_takes_no_motif_and_emits_no_law(role, mode):
    out = gsp.compose_ghost_prompt_v3(
        role=role, style=_style(), mode=mode,
        kernel="film canisters in the high-security archive",
        light="harsh fluorescent overheads", motion="drifting slowly")
    text = out["positive"]
    for law in gsp.GHOST_MODE_LAWS_V2.values():
        assert law not in text
    assert "coat" not in text and "satchel" not in text
    assert out["components"]["vantage"] == gsp.GHOST_VANTAGE_V3[mode]
    assert "kernel" in out["slots"]


def test_the_signature_cannot_be_handed_a_motif_or_a_leaf():
    """The costume is not reachable from here, by construction."""
    with pytest.raises(TypeError):
        gsp.compose_ghost_prompt_v3(
            role="character_video", style=_style(), mode="figure",
            kernel="k", light="", motion="",
            motif_cue="a lean figure in a charcoal coat")  # noqa: E1123


def test_an_empty_kernel_is_refused_rather_than_composed():
    with pytest.raises(gsp.GhostPromptError):
        gsp.compose_ghost_prompt_v3(role="character_video", style=_style(),
                                    mode="figure", kernel="", light="x",
                                    motion="y")


def test_a_bookend_never_composes_figure_mode():
    with pytest.raises(gsp.GhostPromptError):
        gsp.compose_ghost_prompt_v3(role="announcer_visual", style=_style(),
                                    mode="figure", kernel="k", light="",
                                    motion="")


def test_the_light_slot_is_dropped_on_signal_mode():
    """One lighting statement per prompt.

    The `signal` vantage already says "lit against the dark, the light moving";
    composing the pack's own lighting term beside it produced two contradictory
    statements on the same beat in the design prototype.
    """
    assert gsa.resolve_world_light(REAL_META, ordinal=0, mode="signal") == ""
    assert gsa.resolve_world_light(REAL_META, ordinal=0, mode="object")


# --------------------------------------------------------------------------- #
# 3. Variety: the odometer, and a pool that does not exhaust.
# --------------------------------------------------------------------------- #

def test_the_kernel_odometer_does_not_repeat_the_pair_every_few_beats():
    """Cycling both wheels on the same index repeated 7 times in 29 beats."""
    seen = [gsa.resolve_crux_kernel(REAL_META, ordinal=i,
                                    role="character_video", mode="figure")[0]
            for i in range(20)]
    assert len(set(seen)) == 20


def test_the_world_motion_pool_survives_the_longest_real_episode():
    """29 shots is the longest planned episode observed on this lane.

    The 2026-08-30 incident exhausted a six-clause pool on `figure` alone in a
    five-act episode, and under v3 this pool fires on EVERY beat rather than
    only on a failed batch -- so the claim being pinned is that a full episode's
    worth of beats in one bucket produces no repeat.
    """
    for mode, pool in gsa.GHOST_WORLD_MOTION_V3.items():
        assert len(pool) >= 32, mode
        picked = [gsa.resolve_world_motion(mode=mode, episode_seed=99,
                                           ordinal=i) for i in range(len(pool))]
        assert len(set(picked)) == len(pool), mode


def test_two_episodes_do_not_open_on_the_same_motion():
    a = gsa.resolve_world_motion(mode="object", episode_seed=1, ordinal=0)
    b = gsa.resolve_world_motion(mode="object", episode_seed=2, ordinal=0)
    assert a != b


def test_no_pool_clause_carries_its_own_subject():
    """A clause with a subject re-introduces the figure it was meant to remove.

    `GHOST_FALLBACK_CLAUSES` entries are whole sentences ("a figure turns a page
    and holds the paper to the lamp"), so appending one after a crux kernel
    would read "a vast cold water reservoir, a figure turns a page...". Every v3
    clause is a bare verb phrase.
    """
    banned = ("a figure", "the figure", "it ", "he ", "she ", "they ")
    for mode, pool in gsa.GHOST_WORLD_MOTION_V3.items():
        for clause in pool:
            low = clause.casefold()
            for word in banned:
                assert not low.startswith(word), (mode, clause, word)
            for human in gsa._HUMAN_WORDS:
                assert human not in low.split(), (mode, clause, human)


# --------------------------------------------------------------------------- #
# 4. The finalizer: fits, never raises on a real row, banana once.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("style_id", sorted(vs.list_style_ids()))
@pytest.mark.parametrize("role,mode", EVERY_MODE)
def test_every_pack_and_mode_finalizes_without_raising(style_id, role, mode):
    out = gsa.finalize_ghost_prompt_v3(role=role, style=_style(style_id),
                                       mode=mode, ledger_meta=REAL_META,
                                       ordinal=3)
    assert out["positive"].strip()
    assert out["negative"].strip()
    assert out["kernel_source"] == (
        "key_object" if role == "character_video" else "bookend_radio")
    if out["measured"]:
        assert out["positive_clip_windows"] == 1
        assert out["positive_clip_tokens"] <= gsa.GHOST_CLIP_WINDOW_TOKENS


@pytest.mark.parametrize("role,mode", EVERY_MODE)
def test_a_failed_brief_still_finalizes_on_the_cueless_default_pack(role, mode):
    """`sci_fi_radio` contributes no pack cue, so this is the thinnest prompt
    the lane can produce: kernel, motion, vantage and nothing else."""
    empty = {"key_objects": [], "story_brief": "", "episode_seed": 7,
             "story_brief_terms": {"setting": [], "lighting": []}}
    out = gsa.finalize_ghost_prompt_v3(role=role, style=_style("sci_fi_radio"),
                                       mode=mode, ledger_meta=empty, ordinal=0)
    assert out["kernel_source"] in ("bookend", "bookend_radio")
    assert out["components"]["kernel"].strip()
    assert out["components"]["pack_cue"] == ""


def test_the_fitter_drops_whole_units_in_order_and_keeps_the_subject():
    huge = {
        "episode_seed": 1,
        "key_objects": ["a hand-annotated hydrographic survey chart " * 3],
        "story_brief_terms": {
            "setting": ["a decommissioned subsurface monitoring station"],
            "lighting": ["a failing bank of sodium emergency lamps " * 3]},
    }
    out = gsa.finalize_ghost_prompt_v3(role="announcer_visual", style=_style(),
                                       mode="object", ledger_meta=huge,
                                       ordinal=0)
    if not out["measured"]:
        pytest.skip("no installed tokenizer; the fitter cannot run")
    assert out["dropped"], "an over-budget prompt must have shed something"
    # whatever survived is a WHOLE unit, never a slice
    for name, text in out["components"].items():
        assert text == "" or text in out["positive"], name
    # and the drop order is a subsequence of the declared order
    order = list(gsa.GHOST_V3_DROP_ORDER)
    assert [u for u in order if u in out["dropped"]] == list(out["dropped"])


def _word_measure(limit=1):
    """An INJECTED measurer, so the fitter is exercised in CI.

    The installed SD1 tokenizer is unavailable under `OTR_TEST_MODE`, which
    makes the drop-to-fit path -- the one piece of v3 that can change the sent
    text -- the piece least likely to be covered. `resolve_token_measure`
    documents that an injected measurer always gates, so a word counter scaled
    to trip the budget runs the real loop deterministically.
    """
    return lambda text: (len(str(text).split()) * limit, 1)


def test_the_fitter_runs_the_declared_drop_order_under_an_injected_measure():
    out = gsa.finalize_ghost_prompt_v3(
        role="character_video", style=_style(), mode="figure",
        ledger_meta=REAL_META, ordinal=0,
        # every word costs 6 tokens, so the full prompt is far over 69 and the
        # fitter must shed units until it is not
        token_measure_fn=_word_measure(6))
    assert out["dropped"], "an over-budget prompt must shed something"
    order = list(gsa.GHOST_V3_DROP_ORDER)
    assert [u for u in order if u in out["dropped"]] == list(out["dropped"])
    # the subject survived whole, and the pack cue with it
    assert "film canisters" in out["positive"]
    assert out["components"]["kernel"]
    assert out["components"]["kernel"] in out["positive"]


@pytest.mark.parametrize("cost", [2, 3, 4, 6, 10])
@pytest.mark.parametrize("role,mode", EVERY_MODE)
def test_the_fitter_drops_IN_ORDER_and_then_stops(cost, role, mode):
    """Whatever the budget, the units go in the declared order and no further.

    Asserting an exact drop list would be asserting the tokenizer's arithmetic;
    the CONTRACT is that the fitter walks `GHOST_V3_DROP_ORDER` from the front,
    stops the moment the prompt fits, and never reaches past the units it
    needed.

    A SUBSEQUENCE, not a prefix: a unit that was already absent is stepped over
    without being receipted, because `light` is structurally empty on `signal`
    mode and the kernel's setting half is absent whenever the brief gave no
    place. Receipting those would publish a budget decision that was never made.
    """
    measure = _word_measure(cost)
    out = gsa.finalize_ghost_prompt_v3(
        role=role, style=_style(), mode=mode,
        ledger_meta=REAL_META, ordinal=0, token_measure_fn=measure)
    order = list(gsa.GHOST_V3_DROP_ORDER)
    assert [u for u in order if u in out["dropped"]] == list(out["dropped"])
    if mode == "signal":
        assert "light" not in out["dropped"]
    # it stopped because it fitted, or because it ran out of units to drop
    fits = measure(out["positive"])[0] <= gsa.GHOST_AUTHOR_TOKEN_TARGET
    assert fits or len(out["dropped"]) == len(order)
    # and the subject is still there whole, whatever was shed
    assert out["components"]["kernel"]
    assert out["components"]["kernel"] in out["positive"]


def test_a_generous_budget_drops_nothing():
    out = gsa.finalize_ghost_prompt_v3(
        role="character_video", style=_style(), mode="figure",
        ledger_meta=REAL_META, ordinal=0, token_measure_fn=_word_measure(1))
    assert out["dropped"] == []
    assert set(out["slots"]) == {"pack_cue", "kernel", "light", "motion",
                                 "vantage"}


def test_the_banana_route_is_applied_exactly_once():
    on = gsa.finalize_ghost_prompt_v3(role="announcer_visual", style=_style(),
                                      mode="object", ledger_meta=REAL_META,
                                      ordinal=0, banana_enabled=True)
    off = gsa.finalize_ghost_prompt_v3(role="announcer_visual", style=_style(),
                                       mode="object", ledger_meta=REAL_META,
                                       ordinal=0, banana_enabled=False)
    assert on["banana_gate"] is True and off["banana_gate"] is False
    assert on["banana_receipt"] and off["banana_receipt"]
    twice = gsa.finalize_ghost_prompt_v3(role="announcer_visual",
                                         style=_style(), mode="object",
                                         ledger_meta=REAL_META, ordinal=0,
                                         banana_enabled=True)
    assert twice["positive"] == on["positive"]


def test_the_slot_receipts_name_only_slots_that_are_present():
    out = gsa.finalize_ghost_prompt_v3(role="music_visual", style=_style(),
                                       mode="signal", ledger_meta=REAL_META,
                                       ordinal=0)
    assert "light" not in out["slots"]          # dropped on signal mode
    assert set(out["slot_tokens"]) <= set(gsp.GHOST_V3_SLOTS)
    for name in out["slots"]:
        assert out["components"][name]


def test_the_version_constant_is_not_the_capability_token():
    """Bumping `GHOST_PROMPT_PROFILE` would unregister every peer from its lane."""
    assert gsp.GHOST_PROMPT_VERSION_V3 == "ghost_signal_v3"
    assert gsp.GHOST_PROMPT_PROFILE == "ghost_signal_v1"
    assert gsp.GHOST_PROMPT_VERSION_V3 != gsp.GHOST_PROMPT_VERSION_V2


# ---------------------------------------------------------------------------
# The brief's own vocabulary reaches the prompt as prose, not as identifiers
#
# Measured 2026-09-03 across the 1,955 episodes on disk: 273 of them (14.0%)
# carried at least one setting term nothing downstream could say out loud, and
# 8.2% of composed kernels named one. `snake_case` was 690 of the 773 bad terms
# -- `control_room`, `film_reel`, `petri_dish`, `concrete_floors` -- and the
# composer emitted them verbatim, so a real episode rendered
# "archive_reels in the concrete_floors". The rate fell to ~3% once the writer
# changed in August, which is why this is normalisation and not a campaign: the
# terms are RIGHT, their punctuation is the schema's, and a replay of any of
# those 273 episodes still composes its prompt at render time.
# ---------------------------------------------------------------------------

IDENTIFIER_META = {
    "episode_seed": 99,
    "key_objects": ["archive_reels", "clipboards"],
    "story_brief_terms": {"setting": ["control_room", "concrete_floors"]},
}


@pytest.mark.parametrize("raw, spoken", [
    ("control_room", "control room"),
    ("concrete_floors", "concrete floors"),
    ("petri_dish", "petri dish"),
    ("candlelit_period_chamber", "candlelit period chamber"),
    ("high-security archive", "high-security archive"),   # prose is untouched
    ("  padded  spacing  ", "padded spacing"),
    ("", ""),
    (None, ""),
])
def test_spoken_term_says_the_word_without_the_punctuation(raw, spoken):
    assert gsa._spoken_term(raw) == spoken


def test_setting_terms_are_spoken_not_identifiers():
    assert gsa._setting_terms(IDENTIFIER_META) == ["control room", "concrete floors"]


def test_kernel_carries_no_underscore_from_either_list():
    """Both halves of the pair are normalised -- objects leak too."""
    kernel, source = gsa.resolve_crux_kernel(IDENTIFIER_META, ordinal=0,
                                             role="character_video", mode="object")
    assert source == "key_object"
    assert "_" not in kernel
    assert kernel == "archive reels in the control room"


def test_no_composed_kernel_of_a_real_episode_shape_carries_an_underscore():
    for ordinal in range(14):
        for role, mode in EVERY_MODE:
            kernel, _source = gsa.resolve_crux_kernel(
                IDENTIFIER_META, ordinal=ordinal, role=role, mode=mode)
            assert "_" not in kernel, (ordinal, role, mode, kernel)


def test_a_term_that_is_both_the_object_and_the_place_drops_the_place():
    """"petri dish in the petri dish" was composed on a real episode.

    A brief may list one term in `key_objects` and in `setting` both, and the
    pair then says a thing is inside itself. The SUBJECT is the half the beat is
    about, so the place drops -- the same resolution the over-long pair takes.
    """
    same = {"key_objects": ["petri_dish"],
            "story_brief_terms": {"setting": ["petri_dish"]}}
    kernel, source = gsa.resolve_crux_kernel(same, ordinal=0,
                                             role="character_video", mode="object")
    assert kernel == "petri dish"
    assert source == "key_object"


def test_the_tautology_guard_ignores_case_and_keeps_a_genuine_pair():
    cased = {"key_objects": ["Control Room"],
             "story_brief_terms": {"setting": ["control_room", "loading dock"]}}
    assert gsa.resolve_crux_kernel(cased, ordinal=0, role="character_video",
                                   mode="object")[0] == "Control Room"
    assert gsa.resolve_crux_kernel(REAL_META, ordinal=0, role="character_video",
                                   mode="object")[0] == (
        "film canisters in the high-security archive")
