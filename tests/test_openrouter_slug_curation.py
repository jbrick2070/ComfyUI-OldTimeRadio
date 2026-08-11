"""Slug-curation guard (2026-08-07).

WHY THIS FILE EXISTS. `tencent/hy3:free` sat in the dropdown until its promo
ended and the slug stopped resolving upstream. Two things had been written down
and neither could enforce itself: a comment saying the pin was "temporary
through 2026-07-21", and the implicit assumption that someone would notice. The
date passed unnoticed by 17 days. A comment cannot fail; a test can.

WHAT THIS GUARD DELIBERATELY DOES NOT CLAIM:
  * It forbids free-MARKER ids. It CANNOT detect an arbitrarily-named model that
    happens to be promo-priced -- that is dated review evidence, not an
    automatable property.
  * It sees only what this pack SHIPS. Slugs supplied at runtime through
    OTR_OPENROUTER_FAVORITES, OTR_OPENROUTER_SLOT_x_DEFAULT or
    OTR_OPENROUTER_MODEL_ALLOWLIST are the operator's business and are invisible
    here.
  * It proves catalog CURATION, not that any slug generates successfully.

A NOTE ON HOW IT IS WRITTEN. It imports the constants BY NAME. An earlier draft
iterated the module's ``__all__``, which does not export any slug constant --
that guard would have iterated nothing, passed forever, and reproduced the exact
silent decay it exists to prevent.
"""
from __future__ import annotations

import datetime
import re

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb

#: ids advertising a price promise inside the identifier. A promise expires;
#: an identifier does not.
_FREE_MARKERS = (":free", "-free")


def _shipped_concrete_ids() -> set[str]:
    """Every CONCRETE (non-alias) OpenRouter id this pack ships.

    Aliases need no date -- they resolve upstream at request time, which is the
    whole reason the curation prefers them. Concrete ids are version claims that
    can quietly stop being true, so they are exactly the set that must be dated.
    """
    concrete = {
        orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT,
        orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT,
    }
    concrete.update(
        row["id"] for row in getattr(cat, "CURATED_CREATIVE_ROWS", ())
    )
    # The auto-routers (2026-08-10). They carry no '~', so they are concrete by
    # this rule and must be dated like any other pin. Included HERE rather than
    # left to a test of their own, because a curated list the dating guard
    # cannot see is precisely the second-list-that-must-agree defect the rest of
    # this suite exists to prevent.
    concrete.update(getattr(cat, "OPENROUTER_CURATED_ROUTERS", ()))
    # Computed from the '~' prefix rather than a hand-kept list, so a future
    # alias-valued default (chunk B) does not demand a nonsensical date entry.
    return {mid for mid in concrete if not mid.startswith("~")}


def _all_shipped_ids() -> set[str]:
    ids = set(cat.OPENROUTER_CURATED_ALIASES) | _shipped_concrete_ids()
    ids.update(row["id"] for row in getattr(cat, "CURATED_CREATIVE_ROWS", ()))
    ids.update(getattr(cat, "OPENROUTER_CURATED_ROUTERS", ()))
    return ids


# ---------------------------------------------------------------------------
# Auto-routers (operator, 2026-08-10). Offered, AND now the default for both
# slots -- see test_both_slots_default_to_the_auto_router for the decision.
# ---------------------------------------------------------------------------
def test_exactly_the_two_json_capable_routers_are_offered():
    """MEASURED against live /api/v1/models on 2026-08-10, not chosen by taste.
    bodybuilder / fusion / pareto-code declare NO supported_parameters at all,
    so they cannot be told to return JSON and cannot serve a schema-constrained
    writer pass. Listing them would put three dead entries in a dropdown."""
    assert cat.OPENROUTER_CURATED_ROUTERS == (
        "openrouter/auto", "openrouter/auto-beta")


@pytest.mark.parametrize("dead", ["openrouter/bodybuilder", "openrouter/fusion",
                                  "openrouter/pareto-code"])
def test_the_json_incapable_routers_are_never_shipped(dead):
    assert dead not in _all_shipped_ids()


def test_both_slots_default_to_the_auto_router():
    """OPERATOR DECISION 2026-08-10, recorded rather than argued with.

    This test REPLACED one asserting the exact opposite ("no router is a
    recommended default"), written hours earlier under the previous rule. The
    reasoning behind that rule was not wrong and is preserved in
    `_otr_openrouter_backend`: a router resolves differently week to week, and
    the 2026-08-04 directive had settled prose variance. The operator was told
    that twice and chose the router anyway.

    The test is kept -- pointing the other way -- because a DEFAULT that changes
    which model writes the show is exactly the kind of thing that should never
    move silently. If someone reverts it, they do so deliberately.
    """
    assert orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT == "openrouter/auto"
    assert orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT == "openrouter/auto"
    for default in (orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT,
                    orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT):
        assert default in cat.OPENROUTER_CURATED_ROUTERS, default


def test_a_default_router_is_still_dated_and_still_json_capable():
    """The default is now a concrete id, so the dating rule must still cover it
    -- and it must be one of the two routers measured able to return JSON, not
    one of the three that declare no parameters at all."""
    for default in (orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT,
                    orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT):
        assert default in cat.OPENROUTER_VERIFIED_ON_BY_ID, default
        assert default in ("openrouter/auto", "openrouter/auto-beta"), default


def test_routers_are_dated_like_any_other_concrete_pin():
    for router in cat.OPENROUTER_CURATED_ROUTERS:
        assert not router.startswith("~"), router
        assert router in cat.OPENROUTER_VERIFIED_ON_BY_ID, router


def test_routers_are_not_smuggled_into_the_pointer_set():
    """The curated ALIAS tuple is `~...` pointers only; a router in there would
    fail test_curated_aliases_are_all_routing_pointers_and_unique, and putting
    it there would also hide it from the dating guard."""
    for router in cat.OPENROUTER_CURATED_ROUTERS:
        assert router not in cat.OPENROUTER_CURATED_ALIASES, router


def test_routers_appear_in_both_slot_dropdowns_and_auto_leads(monkeypatch):
    """Both routers reachable in A and B, with `openrouter/auto` LEADING each --
    it is the recommended default for both slots now."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OTR_OPENROUTER_MODEL_ALLOWLIST", raising=False)
    monkeypatch.delenv("OTR_OPENROUTER_PROVIDER_FILTER", raising=False)
    for slot in ("a", "b"):
        choices = cat.openrouter_catalog_dropdown_choices(slot)
        if len(choices) <= 2:
            pytest.skip("cold catalog cache on this box; dropdown is sentinel-led")
        for router in cat.OPENROUTER_CURATED_ROUTERS:
            assert router in choices, (slot, router)
        # `openrouter/auto` now LEADS both slots -- it is the recommended
        # default. It must still appear exactly once despite being both the lead
        # and a curated-router entry; `_add` de-duplicates, and this is what
        # proves it does.
        assert choices[1] == "openrouter/auto", (slot, choices[:3])
        assert choices.count("openrouter/auto") == 1, (slot, choices)


def test_no_shipped_slug_carries_a_free_marker():
    """The HY3 defect, made unrepeatable."""
    for mid in _all_shipped_ids():
        for marker in _FREE_MARKERS:
            assert marker not in mid, (
                f"{mid!r} carries {marker!r}. A ':free' id is a PRICE PROMISE "
                f"baked into an IDENTIFIER; promises expire and identifiers do "
                f"not. Prefer a '~family-latest' alias."
            )


def test_every_concrete_id_is_dated():
    """Keys must be EXACTLY the shipped concrete ids -- so a new pin cannot be
    added without a date, and a removed pin cannot leave a stale entry."""
    assert set(cat.OPENROUTER_VERIFIED_ON_BY_ID) == _shipped_concrete_ids()


def test_verified_on_dates_are_real_iso_dates():
    """`fromisoformat`, not a regex: a regex accepts 2026-02-31."""
    for mid, raw in cat.OPENROUTER_VERIFIED_ON_BY_ID.items():
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw), (
            f"{mid}: {raw!r} is not YYYY-MM-DD"
        )
        datetime.date.fromisoformat(raw)  # raises on an impossible date


def test_curated_aliases_are_all_routing_pointers_and_unique():
    # NOTE ON NAMING: the bare identifier `alias` is on this repo's
    # forbidden-symbol extinction list (tests/_s28_forbidden_sweep.py, the
    # no-legacy-back-compat directive), so loop variables here are `slug`.
    # OPENROUTER_CURATED_ALIASES / `aliases` do not match `\balias\b` and are
    # fine; the singular does. Different concept, same word -- the sweep cannot
    # tell them apart, and the retired word loses.
    aliases = cat.OPENROUTER_CURATED_ALIASES
    assert len(set(aliases)) == len(aliases), "duplicate entry in the curated set"
    for slug in aliases:
        assert slug.startswith("~"), (
            f"{slug!r} is in the curated ROUTING-POINTER set but is not a "
            f"'~...' pointer. A concrete id belongs in CURATED_CREATIVE_ROWS "
            f"with a verified-on date."
        )


def test_curated_pointer_spellings_are_pinned_literally():
    """INDEPENDENT golden values -- the one assertion in this suite that does
    not derive its expectation from the constant it is checking.

    Every other contract test builds `expected` FROM
    OPENROUTER_CURATED_ALIASES, so it proves the dropdown echoes the tuple
    faithfully but cannot see a typo INSIDE the tuple -- expected and actual
    share a root. The deleted recent-tier test used to carry literal spellings;
    without something like this, `~x-ai/grok-latest` could be silently
    misspelled and the whole suite would stay green.

    Each spelling below was verified against live /api/v1/models -- the first
    ten on 2026-08-07, the deepseek flash pointer on 2026-08-09.
    """
    pinned = {
        "~anthropic/claude-opus-latest",
        "~anthropic/claude-sonnet-latest",
        "~anthropic/claude-haiku-latest",
        "~anthropic/claude-fable-latest",
        "~openai/gpt-latest",
        "~openai/gpt-mini-latest",
        "~google/gemini-pro-latest",
        "~google/gemini-flash-latest",
        "~moonshotai/kimi-latest",
        "~x-ai/grok-latest",
        "~deepseek/deepseek-v4-flash-latest",
    }
    assert set(cat.OPENROUTER_CURATED_ALIASES) == pinned, (
        "The curated pointer set changed. That is allowed -- but update these "
        "literals IN THE SAME COMMIT and re-verify the new spelling against "
        "live /api/v1/models, because nothing else in this suite can catch a "
        "misspelt pointer."
    )


def test_neither_default_is_a_stale_able_version_pin():
    """SUPERSEDES `test_creative_default_is_a_routing_pointer_not_a_pin`.

    The original rule: the creative default must be a `~family-latest` pointer.
    It existed because the default had been `anthropic/claude-opus-4.8` and was
    ALREADY a version behind when someone noticed -- opus-5 was live at the
    identical price while the pin still said 4.8, and nothing could see it. A
    pin has no way to report that it has gone stale.

    THE UNDERLYING REQUIREMENT WAS NEVER "starts with ~". It was "the default
    must not be a frozen version claim". A `~latest` pointer satisfies that by
    resolving upstream. An AUTO-ROUTER satisfies it more completely: it has no
    version in it at all, so there is nothing to go stale -- it is the most
    evergreen id on the board, in the operator's sense of the word.

    So the test now asserts the requirement rather than the old spelling: a
    default may be a pointer OR a router, and may never be a bare concrete pin.
    """
    for label, default in (
        ("creative", orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT),
        ("technical", orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT),
    ):
        evergreen = (default.startswith("~")
                     or default in cat.OPENROUTER_CURATED_ROUTERS)
        assert evergreen, (
            f"{label} default {default!r} is a frozen version pin. Use a "
            f"'~family-latest' pointer or a curated auto-router: a pin cannot "
            f"notice it has gone stale, and replay is unaffected because the "
            f"ledger stamps the RESOLVED model."
        )
        offered = (cat.OPENROUTER_CURATED_ALIASES
                   + cat.OPENROUTER_CURATED_ROUTERS)
        assert default in offered, (
            f"the {label} default must also be OFFERED in the curated set, or "
            f"the dropdown's leading pick is a slug the dropdown does not list"
        )


def test_a_cheap_pointer_is_offered_so_the_cheap_slot_needs_no_pin():
    """Chunk B: the cheap option is an ALIAS, which is the whole point.

    The two cheaper candidates -- qwen/qwen3.7-flash and
    inclusionai/ling-2.6-flash -- are concrete ids whose authors publish no
    `~latest` resolver, so shipping either would re-create the hy3 defect to
    save a rounding error. This asserts the curated set keeps at least one
    pointer meaningfully cheaper than the frontier creative default, so nobody
    has to reach for a pin to get a cheap slot.
    """
    assert "~deepseek/deepseek-v4-flash-latest" in cat.OPENROUTER_CURATED_ALIASES
    for banned in ("qwen/qwen3.7-flash", "inclusionai/ling-2.6-flash"):
        assert banned not in _all_shipped_ids(), (
            f"{banned!r} is a CONCRETE id with no '~latest' resolver published "
            f"by its author. Cheap is not worth a slug that can vanish -- use "
            f"~deepseek/deepseek-v4-flash-latest."
        )


def test_retired_synthesis_machinery_stays_retired():
    """`~x-ai/grok-latest` replaced ~30 lines that synthesised an author's
    newest concrete slug. If any of these reappear, the alias set is being
    worked around instead of extended."""
    for symbol in ("OPENROUTER_NO_LATEST_AUTHORS", "_newest_concrete_for_author",
                   "_NON_FRONTIER_MARKERS", "_OPENROUTER_RECENT_COUNT",
                   "_PINNED_CREATIVE_CONTENDER_ROWS", "OPENROUTER_FRONTIER_LATEST"):
        assert not hasattr(cat, symbol), (
            f"{symbol} is back. It was deleted 2026-08-07 because OpenRouter "
            f"now publishes the alias it emulated, or because it was uncurated "
            f"cache spill."
        )
