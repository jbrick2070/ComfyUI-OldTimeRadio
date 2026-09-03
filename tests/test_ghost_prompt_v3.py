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
import re

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
    # `trailing_style` joined the set 2026-09-03: a generous budget keeps the
    # pack's own style vocabulary at BOTH ends of the prompt.
    assert set(out["slots"]) == {"pack_cue", "kernel", "light", "motion",
                                 "vantage", "trailing_style"}


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
    # BUMPED 2026-09-03 for the trailing style clause. The bump is REQUIRED,
    # not cosmetic: `OTRVideoRenderBatch.IS_CHANGED` folds this constant into
    # its cache key, so without it a resident ComfyUI session re-serves clips
    # rendered by the previous composer and the change looks like a no-op.
    assert gsp.GHOST_PROMPT_VERSION_V3 == "ghost_signal_v3.1"
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


# ---------------------------------------------------------------------------
# THE STYLE BRACKETS THE PROMPT (operator ruling 2026-09-03)
#
# "visual style - key objects per beat story - + movement", and "if you can
# spend more at start and end of prompt to highlight visual style but NO EXTRA
# OBJECTS because that's where it gets tripped up".
#
# The packs authored more style vocabulary than v3 was asking for: anime wrote
# "anime style, expressive linework, cel-shaded color" and the prompt carried
# "anime style". The remainder now rides at the END, bounded, and no object
# vocabulary is added anywhere.
# ---------------------------------------------------------------------------

from nodes import _otr_visual_styles as _vs


def _style_obj(style_id):
    return _vs.get_visual_style({"visual_style": style_id})


def test_every_pack_but_the_house_style_brackets_the_prompt():
    for style_id in _vs.list_style_ids():
        out = gsa.finalize_ghost_prompt_v3(
            role="character_video", style=_style_obj(style_id), mode="figure",
            ledger_meta=REAL_META, ordinal=0)
        positive = out["positive"]
        if style_id == _vs.DEFAULT_STYLE_ID:
            continue
        front = _vs.compact_style_cue(_style_obj(style_id))
        tail = _vs.trailing_style_cue(_style_obj(style_id))
        assert positive.startswith(front), (style_id, positive[:40])
        assert positive.endswith(tail), (style_id, positive[-40:])
        assert "trailing_style" in out["slots"], style_id


def test_the_house_style_gains_no_style_text_at_either_end():
    """`sci_fi_radio` IS the house look and must emit none, front or back.

    This is the defect the r2 reviewer caught in the spec before it shipped:
    deriving the tail by subtracting the front cue from `positive_tail` hands
    back the WHOLE tail when the front cue is empty, which is exactly the
    default style's case -- it would have emitted "cinematic, 35mm film look,
    subtle film grain, volumetric lighting" at the back of every house-look
    prompt and churned every default-lane golden.
    """
    house = _style_obj(_vs.DEFAULT_STYLE_ID)
    assert _vs.compact_style_cue(house) == ""
    assert _vs.trailing_style_cue(house) == ""
    out = gsa.finalize_ghost_prompt_v3(role="character_video", style=house,
                                       mode="figure", ledger_meta=REAL_META,
                                       ordinal=0)
    assert "trailing_style" not in out["slots"]
    assert out["positive"].startswith("film canisters")
    for word in ("cinematic", "35mm", "film grain", "volumetric"):
        assert word not in out["positive"], word


def test_the_trailing_cue_keeps_whole_units_and_never_slices_one():
    """A sliced unit changes what is asked for: "cel-shaded" is not a colour."""
    for style_id in _vs.list_style_ids():
        style = _style_obj(style_id)
        tail = _vs.trailing_style_cue(style)
        if not tail:
            continue
        authored = str(style.positive_tail or "")
        for unit in tail.split(", "):
            assert unit in authored, (style_id, unit)
        assert len(tail.split()) <= _vs.TRAILING_STYLE_MAX_WORDS, style_id


def test_the_two_longest_packs_are_capped_rather_than_dropped():
    """`recur_frac` (22 words authored) and `video_art` (14) are the operator's
    two packs of interest and the only ones that overflowed. They must be
    TRIMMED to whole units, not left whole and then dropped by the fitter."""
    for style_id in ("recur_frac", "video_art"):
        style = _style_obj(style_id)
        authored = len(str(style.positive_tail or "").split())
        tail = _vs.trailing_style_cue(style)
        assert tail, style_id
        assert len(tail.split()) < authored, style_id
        assert len(tail.split()) <= _vs.TRAILING_STYLE_MAX_WORDS, style_id


def test_the_style_enrichment_is_surrendered_before_any_earning_slot():
    """Adding the trailing clause must never COST a slot that was already there.

    It is first in `GHOST_V3_DROP_ORDER` for exactly this reason: under budget
    pressure the prompt reverts to what it emitted before the clause existed,
    rather than trading away light or framing to keep decoration.
    """
    assert gsa.GHOST_V3_DROP_ORDER[0] == "trailing_style"
    out = gsa.finalize_ghost_prompt_v3(
        role="character_video", style=_style(), mode="figure",
        ledger_meta=REAL_META, ordinal=0, token_measure_fn=_word_measure(3))
    assert out["dropped"], "an over-budget prompt must shed something"
    assert out["dropped"][0] == "trailing_style"
    assert "trailing_style" not in out["slots"]


def test_movement_outlives_framing_under_budget_pressure():
    """Operator ruling 2026-09-03: the budget buys style, objects and MOVEMENT.

    `motion` used to be dropped second, which was harmless only because the
    ladder never fired (v3 measures ~32 tokens against a target of 69). The
    trailing clause lengthens prompts, so the order now sheds framing before
    movement -- a shot that still moves but is less precisely staged beats a
    precisely staged still one.
    """
    order = list(gsa.GHOST_V3_DROP_ORDER)
    assert order.index("vantage") < order.index("motion")
    assert order.index("motion") < order.index("kernel_setting")

    # Squeeze hard enough to shed several units, and confirm motion is still
    # standing after framing has gone.
    out = gsa.finalize_ghost_prompt_v3(
        role="character_video", style=_style(), mode="figure",
        ledger_meta=REAL_META, ordinal=0, token_measure_fn=_word_measure(5))
    dropped = list(out["dropped"])
    if "vantage" in dropped and "motion" not in dropped:
        assert out["components"]["motion"] in out["positive"]
    assert dropped == [u for u in order if u in dropped], dropped


def test_the_lighting_term_is_spoken_like_the_setting_and_the_objects():
    """Found live by a source-bank sweep, on a real episode, after the first fix.

    `resolve_crux_kernel` and `_setting_terms` were normalised for
    PBUG-20260903-04; `resolve_world_light` reads the SAME LLM-authored brief,
    a different key, and was missed. Measured 481 of 7,978 lighting terms (6.0%)
    carry an underscore, and the composer emitted them verbatim:
    "handheld bronze communicator in the forest, storm_light, ...".
    """
    meta = dict(REAL_META)
    meta["story_brief_terms"] = dict(meta["story_brief_terms"])
    meta["story_brief_terms"]["lighting"] = ["storm_light", "dim_glow"]

    assert gsa.resolve_world_light(meta, ordinal=0, mode="figure") == "storm light"

    for style_id in ("anime", "sci_fi_radio"):
        out = gsa.finalize_ghost_prompt_v3(
            role="character_video", style=_style_obj(style_id), mode="figure",
            ledger_meta=meta, ordinal=0)
        assert "_" not in out["positive"], (style_id, out["positive"])
        assert "storm light" in out["positive"], style_id


# ---------------------------------------------------------------------------
# THE BOOKENDS GET THE PACK'S OWN KINETIC DIRECTION (operator report 2026-09-03)
#
# He watched a published episode and said the announcer and music beats "had
# basically no movement". They were composing from GHOST_WORLD_MOTION_V3, whose
# clauses are atmospheric by design ("cooling into shadow"), while every style
# pack had already authored a kinetic register for exactly those four roles,
# ending in a camera move -- which the live v3 path never read.
# ---------------------------------------------------------------------------

def test_a_bookend_prefers_the_packs_kinetic_register_over_the_generic_pool():
    style = _style_obj("storybook_engraving")
    pack = _vs.bounded_motion_register(
        dict(style.motion_registers)["announcer"])
    assert pack, "the pack authored an announcer register"

    generic = gsa.finalize_ghost_prompt_v3(
        role="announcer_visual", style=style, mode="object",
        ledger_meta=REAL_META, ordinal=1)
    kinetic = gsa.finalize_ghost_prompt_v3(
        role="announcer_visual", style=style, mode="object",
        ledger_meta=REAL_META, ordinal=1, pack_motion=pack)

    assert generic["components"]["motion"] != kinetic["components"]["motion"]
    assert kinetic["components"]["motion"] == pack
    assert pack in kinetic["positive"]


def test_a_character_beat_is_untouched_by_the_register():
    """Scoped to bookends on purpose -- character motion is a separate report."""
    style = _style_obj("storybook_engraving")
    before = gsa.finalize_ghost_prompt_v3(
        role="character_video", style=style, mode="figure",
        ledger_meta=REAL_META, ordinal=2)
    after = gsa.finalize_ghost_prompt_v3(
        role="character_video", style=style, mode="figure",
        ledger_meta=REAL_META, ordinal=2, pack_motion="")
    assert before["positive"] == after["positive"]


def test_an_absent_or_oversized_register_still_leaves_the_beat_moving():
    """A missing register must never cost the beat its motion slot."""
    style = _style_obj("storybook_engraving")
    out = gsa.finalize_ghost_prompt_v3(
        role="announcer_visual", style=style, mode="object",
        ledger_meta=REAL_META, ordinal=1, pack_motion="")
    assert out["components"]["motion"], "the generic pool must still supply one"
    assert "motion" in out["slots"]

    assert _vs.bounded_motion_register(None) == ""
    assert _vs.bounded_motion_register("") == ""
    assert _vs.bounded_motion_register("Continuous shot, same console throughout.") == ""


def test_the_register_drops_the_static_framing_line_and_keeps_the_camera():
    """The leading "Continuous shot" clause is a shot RULE, not a movement, and
    telling a model to hold still is the opposite of the defect being fixed."""
    got = _vs.bounded_motion_register(
        "Continuous shot, same console throughout. Etched dial needle glides. "
        "Hand-tinted highlights shimmer. Paper grain breathes softly. "
        "Slow illustrated dolly forward.")
    # The static camera instruction goes...
    assert "continuous shot" not in got.lower()
    # ...but the SUBJECT it named is carried onto the kinetic clause rather than
    # discarded with it, or the dial has no antecedent (operator 2026-09-03).
    assert got.startswith("radio console etched dial needle glides")
    assert got.endswith("slow illustrated dolly forward")
    assert len(got.split()) <= _vs.MOTION_REGISTER_MAX_WORDS


def test_the_compacted_register_never_orphans_the_dial():
    """Operator, 2026-09-03: *"'dial' -- you'd have to say radio system dial."*

    The packs author "Continuous shot, same console throughout. Dial needle
    sweeps in crisp arcs..." -- the framing sentence carries the ANTECEDENT.
    Dropping it as a damping instruction (which it is) left "dial needle sweeps"
    attachable to a telephone, a clock face or a gauge. The subject is now
    re-anchored onto the kinetic clause instead of being discarded with it.
    """
    for style_id in _vs.list_style_ids():
        style = _style_obj(style_id)
        registers = dict(getattr(style, "motion_registers", {}) or {})
        for key, raw in registers.items():
            got = _vs.bounded_motion_register(raw)
            if not got:
                continue
            if "dial" not in got.lower():
                continue
            assert re.search(r"\b(radio|console|set)\b", got.lower()), (
                "%s/%s emits a bare dial with no radio antecedent: %r"
                % (style_id, key, got))


def test_the_anchor_check_matches_whole_words_not_substrings():
    """"set" is inside "settles" -- the first cut of this guard read
    "dial settles" as already-anchored and left it bare. It is the same defect
    as `"close" in "closing"` in the driver's register selector, committed
    minutes after fixing that one."""
    got = _vs.bounded_motion_register(
        "Continuous shot, same console throughout. Dial settles. "
        "Slow dolly pull back.")
    assert got.startswith("radio console"), got
    assert "settles" in got
