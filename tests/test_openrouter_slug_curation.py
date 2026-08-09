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
    # Computed from the '~' prefix rather than a hand-kept list, so a future
    # alias-valued default (chunk B) does not demand a nonsensical date entry.
    return {mid for mid in concrete if not mid.startswith("~")}


def _all_shipped_ids() -> set[str]:
    ids = set(cat.OPENROUTER_CURATED_ALIASES) | _shipped_concrete_ids()
    ids.update(row["id"] for row in getattr(cat, "CURATED_CREATIVE_ROWS", ()))
    return ids


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


def test_creative_default_is_a_routing_pointer_not_a_pin():
    """Chunk B (2026-08-09). The creative default must stay an alias.

    It was `anthropic/claude-opus-4.8` and was ALREADY a version behind when
    that was noticed -- opus-5 was live at the identical price while the pin
    still said 4.8. Nothing could see it: a pin has no way to report that it has
    gone stale, and the dated-pin guard above only proves someone WROTE a date,
    not that the date is still true. Making the default a pointer removes the
    version claim entirely, and this test stops it being quietly re-pinned.

    The TECHNICAL default is deliberately NOT covered: the only DeepSeek pointer
    is the flash tier, so aliasing it would be a capability drop on the slot
    that most needs reliable structured output.
    """
    assert orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT.startswith("~"), (
        f"creative default {orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT!r} is a "
        f"concrete pin. Use a '~family-latest' pointer: a pin cannot notice it "
        f"has gone stale, and replay is unaffected because the ledger stamps "
        f"the RESOLVED model (tests/test_openrouter_resolved.py)."
    )
    assert orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT in cat.OPENROUTER_CURATED_ALIASES, (
        "the creative default must also be OFFERED in the curated set, or the "
        "dropdown's leading pick is a slug the dropdown does not list"
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
