"""The TWO independent operator randomizers (`nodes/_otr_rolls.py`).

Operator directive 2026-07-31: `source_bank` and `visual_style` are separate
randomizers that can be turned on or off individually. The load-bearing
claims pinned here are (a) each surface is switched ONLY by its own
sentinel, (b) each replays from its OWN seed env, (c) a manual pick is
byte-identical and leaves NO receipt, and (d) a roll never reaches for an
LLM or for a hardcoded id list.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import OTR_LedgerScriptWriter as WRITER
from nodes import _otr_lane_specs as LANES
from nodes import _otr_rolls as ROLLS
from nodes import _otr_story_routing as ROUTING
from nodes import _otr_visual_styles as STYLES


def _req(target_words: int = 320) -> "LANES.RollRequest":
    return LANES.RollRequest(target_words=target_words)


def _factory(target_words: int = 320):
    return lambda: _req(target_words)


@pytest.fixture(autouse=True)
def _fresh_registry():
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None


# ---------------------------------------------------------------------------
# the two surfaces are INDEPENDENT
# ---------------------------------------------------------------------------

def test_the_two_sentinels_are_distinct_and_are_not_registry_rows():
    assert ROLLS.BANK_SENTINEL != ROLLS.STYLE_SENTINEL
    assert ROLLS.BANK_SENTINEL not in ROUTING.list_bank_ids()
    assert ROLLS.STYLE_SENTINEL not in STYLES.list_style_ids()
    # And neither sentinel is mistaken for the other's command.
    assert ROLLS.is_bank_sentinel(ROLLS.STYLE_SENTINEL) is False
    assert ROLLS.is_style_sentinel(ROLLS.BANK_SENTINEL) is False


def test_the_two_surfaces_use_separate_seed_envs():
    """Either roll must be replayable ALONE."""
    assert ROLLS.BANK_SEED_ENV != ROLLS.STYLE_SEED_ENV
    # OTR_STYLE_SEED is the narrative arc-shape seed -- never reused here.
    assert ROLLS.STYLE_SEED_ENV == "OTR_VISUAL_STYLE_SEED"
    assert "OTR_STYLE_SEED" not in (ROLLS.BANK_SEED_ENV, ROLLS.STYLE_SEED_ENV)


def test_rolling_the_bank_does_not_roll_the_style():
    bank, bank_receipt = ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, request_factory=_factory(), env={})
    style, style_receipt = ROLLS.resolve_style_selection(
        "sci_fi_radio", env={})
    assert bank_receipt is not None and bank in ROUTING.list_bank_ids()
    assert style == "sci_fi_radio" and style_receipt is None


def test_rolling_the_style_does_not_roll_the_bank():
    bank, bank_receipt = ROLLS.resolve_bank_selection(
        "scifi_news", request_factory=_factory(), env={})
    style, style_receipt = ROLLS.resolve_style_selection(
        ROLLS.STYLE_SENTINEL, env={})
    assert bank == "scifi_news" and bank_receipt is None
    assert style_receipt is not None and style in STYLES.list_style_ids()


def test_both_surfaces_can_roll_in_the_same_run_with_separate_seeds():
    env = {ROLLS.BANK_SEED_ENV: "11", ROLLS.STYLE_SEED_ENV: "22"}
    _bank, bank_receipt = ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, request_factory=_factory(), env=env)
    _style, style_receipt = ROLLS.resolve_style_selection(
        ROLLS.STYLE_SENTINEL, env=env)
    assert bank_receipt.seed == 11 and style_receipt.seed == 22
    assert bank_receipt.surface == "source_bank"
    assert style_receipt.surface == "visual_style"


# ---------------------------------------------------------------------------
# the manual path stays byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("picked", ["scifi_news", "shakespeare", "anything"])
def test_a_manual_bank_pick_returns_unchanged_with_no_receipt(picked):
    assert ROLLS.resolve_bank_selection(
        picked, request_factory=_factory(), env={}) == (picked, None)


@pytest.mark.parametrize("picked", ["sci_fi_radio", "anime", "anything"])
def test_a_manual_style_pick_returns_unchanged_with_no_receipt(picked):
    assert ROLLS.resolve_style_selection(picked, env={}) == (picked, None)


def test_a_manual_pick_never_builds_the_request():
    """Gate-first ordering: a malformed target must not raise BEFORE the
    unknown/non-runnable bank gate gets to speak."""
    def _explode():
        raise AssertionError("the request was built on the manual path")

    assert ROLLS.resolve_bank_selection(
        "scifi_news", request_factory=_explode, env={}) == ("scifi_news", None)


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------

def test_bank_pool_is_runnable_and_lane_compatible_and_sorted():
    order = ROLLS.eligible_bank_ids(_req(320))
    banks = ROUTING._ensure_loaded().banks
    assert list(order) == sorted(order), "pool must not depend on row order"
    assert order, "at least one bank must be rollable at a normal target"
    for bank_id in order:
        assert banks[bank_id].runnable is True
    # custom_source_bank is the shipped non-runnable row -- never rollable.
    assert "custom_source_bank" not in order


def test_bank_pool_drops_a_lane_that_cannot_build_the_request():
    """A bank whose lane would reject the target must not win the roll and
    then fail -- filtering BEFORE the draw is the whole point."""
    wide = set(ROLLS.eligible_bank_ids(_req(320)))
    narrow = set(ROLLS.eligible_bank_ids(_req(5000)))
    assert narrow < wide, "the 30..900 lane should drop out at 5000 words"
    dropped = wide - narrow
    assert dropped, "expected at least one lane-constrained bank"
    for bank_id in dropped:
        bank = ROUTING.get_bank(bank_id)
        assert not LANES.is_roll_compatible(bank, _req(5000))


def test_bank_pool_is_derived_from_the_live_registry_not_a_literal():
    """A client bank admitted at runtime must be an ordinary peer."""
    order = ROLLS.eligible_bank_ids(_req(320))
    runnable = {b.source_bank_id for b in ROUTING._ensure_loaded().banks.values()
                if b.runnable}
    assert set(order) <= runnable
    assert set(order) == {
        b for b in runnable
        if LANES.is_roll_compatible(ROUTING.get_bank(b), _req(320))
    }


def test_style_pool_is_every_registered_style_sorted():
    """No filter, by design: every registered style is fully live."""
    order = ROLLS.eligible_style_ids()
    assert list(order) == sorted(STYLES.list_style_ids())
    assert len(order) == len(STYLES.list_style_ids())


def test_an_empty_bank_pool_fails_loud_and_names_both_filters(monkeypatch):
    monkeypatch.setattr(LANES, "is_roll_compatible",
                        lambda _bank, _req: False, raising=True)
    with pytest.raises(ROLLS.RollError) as caught:
        ROLLS.resolve_bank_selection(
            ROLLS.BANK_SENTINEL, request_factory=_factory(), env={})
    message = str(caught.value)
    assert "runnable=false" in message
    assert "lane incompatibility" in message
    assert "no eligible bank" in message


# ---------------------------------------------------------------------------
# seeds and the draw
# ---------------------------------------------------------------------------

def test_the_draw_is_deterministic_over_a_synthetic_pool():
    """Pinned on a SYNTHETIC pool on purpose -- asserting that a live seed
    lands on a NAMED bank breaks every time a bank lands."""
    pool = ("alpha", "bravo", "charlie", "delta", "echo")
    for seed in (0, 1, 7, 123456, 2 ** 32 - 1):
        assert ROLLS.draw(pool, seed) == ROLLS.draw(pool, seed)
    assert set(ROLLS.draw(pool, s) for s in range(200)) == set(pool)


def test_the_draw_refuses_an_empty_pool():
    with pytest.raises(ROLLS.RollError, match="empty pool"):
        ROLLS.draw((), 1)


@pytest.mark.parametrize("env_var", ["OTR_BANK_SEED", "OTR_VISUAL_STYLE_SEED"])
def test_an_env_seed_is_recorded_as_an_override_and_replays(env_var):
    seed, source = ROLLS.resolve_seed(env_var, {env_var: "424242"})
    assert seed == 424242
    assert source == f"{env_var} override"


@pytest.mark.parametrize("env_var", ["OTR_BANK_SEED", "OTR_VISUAL_STYLE_SEED"])
def test_no_env_seed_mints_from_os_entropy_but_still_replays(env_var):
    seed, source = ROLLS.resolve_seed(env_var, {})
    assert source == "OS entropy"
    assert 0 <= seed < 2 ** 32
    # SystemRandom only MINTS; the draw itself is always seeded, so the
    # recorded seed reproduces the selection.
    pool = ("a", "b", "c", "d")
    assert ROLLS.draw(pool, seed) == ROLLS.draw(pool, seed)


@pytest.mark.parametrize("bad", ["nope", "", " 12x", "1.5", "-1", "4294967296"])
def test_a_malformed_or_out_of_range_seed_raises_rather_than_degrading(bad):
    """An operator who typed a seed asserted replay intent."""
    if not bad.strip():
        # blank is "unset", not malformed -- it falls through to entropy.
        seed, source = ROLLS.resolve_seed(ROLLS.BANK_SEED_ENV,
                                          {ROLLS.BANK_SEED_ENV: bad})
        assert source == "OS entropy"
        return
    with pytest.raises(ROLLS.RollError):
        ROLLS.resolve_seed(ROLLS.BANK_SEED_ENV, {ROLLS.BANK_SEED_ENV: bad})


def test_a_seeded_bank_roll_replays_from_its_own_receipt():
    env = {ROLLS.BANK_SEED_ENV: "987654"}
    selected, receipt = ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, request_factory=_factory(), env=env)
    assert ROLLS.draw(receipt.eligible_order, receipt.seed) == selected
    assert receipt.selected == selected


def test_an_unseeded_style_roll_replays_from_its_own_receipt():
    """The unseeded path is proved by REPLAY, not by predicting the id."""
    selected, receipt = ROLLS.resolve_style_selection(
        ROLLS.STYLE_SENTINEL, env={})
    assert receipt.seed_source == "OS entropy"
    assert ROLLS.draw(receipt.eligible_order, receipt.seed) == selected


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------

def test_receipt_meta_shape_is_json_ready_and_names_its_surface():
    _selected, receipt = ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, request_factory=_factory(),
        env={ROLLS.BANK_SEED_ENV: "5"})
    row = receipt.to_meta()
    assert row["receipt_version"] == ROLLS.RECEIPT_VERSION
    assert row["surface"] == "source_bank"
    assert row["requested"] == ROLLS.BANK_SENTINEL
    assert row["selected"] == receipt.selected
    assert row["seed"] == 5 and isinstance(row["seed"], int)
    assert row["seed_source"] == "OTR_BANK_SEED override"
    assert isinstance(row["eligible_order"], list)   # a tuple is not JSON
    assert row["eligible_order"] == list(receipt.eligible_order)


def test_style_receipt_names_its_own_surface_and_seed_source():
    _selected, receipt = ROLLS.resolve_style_selection(
        ROLLS.STYLE_SENTINEL, env={ROLLS.STYLE_SEED_ENV: "9"})
    row = receipt.to_meta()
    assert row["surface"] == "visual_style"
    assert row["requested"] == ROLLS.STYLE_SENTINEL
    assert row["seed_source"] == "OTR_VISUAL_STYLE_SEED override"


# ---------------------------------------------------------------------------
# a pinned source and the roll are mutually exclusive
# ---------------------------------------------------------------------------

def test_the_bank_roll_refuses_a_pinned_source_ref():
    with pytest.raises(ROLLS.RollError, match="pins a specific source"):
        ROLLS.resolve_bank_selection(
            ROLLS.BANK_SENTINEL, request_factory=_factory(),
            source_ref="folger:mac.2.2", env={})


def test_a_blank_source_ref_does_not_block_the_roll():
    for blank in ("", "   ", None):
        selected, receipt = ROLLS.resolve_bank_selection(
            ROLLS.BANK_SENTINEL, request_factory=_factory(),
            source_ref=blank, env={})
        assert receipt is not None and selected


def test_a_pinned_source_ref_is_fine_on_a_manual_pick():
    assert ROLLS.resolve_bank_selection(
        "shakespeare", request_factory=_factory(),
        source_ref="folger:mac.2.2", env={}) == ("shakespeare", None)


# ---------------------------------------------------------------------------
# widget wiring
# ---------------------------------------------------------------------------

def test_both_dropdowns_carry_their_sentinel_as_choice_zero():
    optional = WRITER.OTR_LedgerScriptWriter.INPUT_TYPES()["optional"]
    bank_choices = optional["source_bank"][0]
    style_choices = optional["visual_style"][0]
    assert bank_choices[0] == ROLLS.BANK_SENTINEL
    assert bank_choices[1:] == list(ROUTING.list_bank_ids())
    assert style_choices[0] == ROLLS.STYLE_SENTINEL
    assert style_choices[1:] == list(STYLES.list_style_ids())


def test_the_shipped_defaults_are_still_concrete_ids_not_the_roll():
    """Zero canonical-JSON diff: the saved graph's VALUES do not change."""
    optional = WRITER.OTR_LedgerScriptWriter.INPUT_TYPES()["optional"]
    assert optional["source_bank"][1]["default"] == "scifi_news"
    assert optional["visual_style"][1]["default"] == "sci_fi_radio"


def test_no_new_widget_was_added_for_either_roll():
    """The rolls are UI COMMANDS inside existing dropdowns. A new widget
    would shift positional widgets_values (BUG-LOCAL-097)."""
    optional = WRITER.OTR_LedgerScriptWriter.INPUT_TYPES()["optional"]
    for suspicious in ("roll", "randomize", "bank_roll", "style_roll",
                       "randomizer"):
        assert suspicious not in optional


# ---------------------------------------------------------------------------
# the roll is mechanical
# ---------------------------------------------------------------------------

def test_the_roll_module_imports_nothing_that_could_call_a_model():
    """Selection under a declared uniform policy over an enumerated pool is
    MECHANICAL. Pinned on the import graph rather than on source text: a
    model call needs a backend, and the roll imports none. If a later draft
    introduces one, this fails and that draft is wrong.
    """
    import ast

    source = (Path(__file__).resolve().parents[1]
              / "nodes" / "_otr_rolls.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    # loop var named `imp` NOT `alias` -- the B7 forbidden sweep flags `alias`
    # as a runtime marker (CW-6 gotcha), same as test_visual_styles_3a.py and
    # test_wire_w2_deferred_image_gap.py.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(imp.name.split(".")[0] for imp in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            imported.update(imp.name for imp in node.names)
    allowed = {
        # stdlib
        "__future__", "annotations",
        "os", "random", "dataclasses", "typing",
        "dataclass", "Any", "Callable", "Mapping", "Sequence",
        # the three registries a roll is allowed to read
        "_otr_lane_specs", "_otr_story_routing", "_otr_visual_styles",
        "_LANES", "_ROUTING", "_STYLES", "",
    }
    assert imported <= allowed, f"unexpected imports: {imported - allowed}"


def test_the_roll_never_mutates_process_env():
    before = dict(__import__("os").environ)
    ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, request_factory=_factory(), env={})
    ROLLS.resolve_style_selection(ROLLS.STYLE_SENTINEL, env={})
    assert dict(__import__("os").environ) == before


def test_the_rng_factory_is_a_real_seam():
    """Pinned so the draw can be exercised without OS entropy."""
    calls = []

    def _fake(seed):
        calls.append(seed)
        return random.Random(seed)

    ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, request_factory=_factory(),
        env={ROLLS.BANK_SEED_ENV: "3"}, rng_factory=_fake)
    assert calls == [3]
