"""nodes/_otr_rolls.py -- the operator-facing randomizers.

TWO INDEPENDENT ROLLS, one mechanism (operator directive 2026-07-31):

* `source_bank`  -> which story bank writes the episode
* `visual_style` -> which visual pack dresses it

Each is switched on by picking that dropdown's own SENTINEL row instead of a
concrete id, so they are controlled SEPARATELY: roll the bank and pin the
style, pin the bank and roll the style, roll both, or roll neither. Nothing
here is a global "randomize everything" flag, and neither roll can turn the
other on.

The sentinel is a UI COMMAND, never a registry row. It is prepended to the
existing dropdown's choices, so no new widget appears, no positional
`widgets_values` slot shifts (BUG-LOCAL-097 is not engaged), and the
canonical workflow JSON does not change: choices are not persisted, only the
selected value is.

ZERO LLM CALLS IN THE ROLL ITSELF. A roll reads a pool of eligible ids,
filters, and draws deterministically from a seed. Selection under a declared
uniform policy over an enumerated pool is mechanical. Selecting a style lane
(such as `visual_storybased`) may later trigger an LLM authoring pass during
script reflection, but the roll mechanism itself never calls a model.

SEEDS. Each surface owns its own env override so either roll can be replayed
ALONE: `OTR_BANK_SEED` and `OTR_VISUAL_STYLE_SEED`. (`OTR_STYLE_SEED` is the
narrative arc-shape seed and is deliberately NOT reused.) Both the override
and the entropy path end in a seeded `random.Random`, so the recorded seed
always replays: `SystemRandom` only ever MINTS a seed, it never draws with
one. A malformed or out-of-range override RAISES -- an operator who typed a
seed asserted replay intent, and quietly ignoring it is the silent-degrade
class this repo bans.

RECEIPTS. `meta["bank_roll"]` / `meta["style_roll"]` are written ONLY when
that surface actually rolled. On a manual pick the key is ABSENT -- not
null, not a stub -- so "was this rolled?" is answerable from the frozen
ledger without a convention about empty values.

Import direction stays one-way:
`_otr_story_routing / _otr_visual_styles / _otr_lane_specs <- _otr_rolls <- OTR_LedgerScriptWriter`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

try:
    from . import _otr_lane_specs as _LANES
    from . import _otr_story_routing as _ROUTING
    from . import _otr_visual_styles as _STYLES
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_lane_specs as _LANES  # type: ignore
    import _otr_story_routing as _ROUTING  # type: ignore
    import _otr_visual_styles as _STYLES  # type: ignore


RECEIPT_VERSION = 1

#: The `source_bank` dropdown's roll command.
BANK_SENTINEL = "roll (any eligible bank)"
#: The `visual_style` dropdown's roll command.
STYLE_SENTINEL = "roll (any style)"

BANK_SEED_ENV = "OTR_BANK_SEED"
STYLE_SEED_ENV = "OTR_VISUAL_STYLE_SEED"

_SEED_BITS = 32
_SEED_CEILING = 2 ** _SEED_BITS


class RollError(_ROUTING.StoryRoutingError):
    """A roll could not produce a selection. Always loud, never a fallback."""


@dataclass(frozen=True)
class RollReceipt:
    """What a roll did, in the exact terms needed to replay it."""

    surface: str
    """Which dropdown rolled: "source_bank" or "visual_style"."""

    requested: str
    """The sentinel the operator actually picked."""

    selected: str
    """The concrete id the draw returned."""

    seed: int
    seed_source: str
    eligible_order: "tuple[str, ...]"
    """The EXACT ordered pool handed to the draw, sorted by id.

    Sorted so the outcome depends on registry CONTENT, never on row order:
    another window reordering banks.json must not change what a replayed
    seed selects."""

    def to_meta(self) -> dict:
        """The ledger shape. `eligible_order` becomes a JSON list."""
        return {
            "receipt_version": RECEIPT_VERSION,
            "surface": self.surface,
            "requested": self.requested,
            "selected": self.selected,
            "seed": int(self.seed),
            "seed_source": self.seed_source,
            "eligible_order": list(self.eligible_order),
        }


def is_bank_sentinel(value: Any) -> bool:
    return str(value or "").strip() == BANK_SENTINEL


def is_style_sentinel(value: Any) -> bool:
    return str(value or "").strip() == STYLE_SENTINEL


def _parse_seed(raw: str, env_var: str) -> int:
    try:
        seed = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise RollError(
            f"{env_var}={raw!r} is not an integer. Unset it to roll on OS "
            f"entropy, or give it a replayable seed."
        ) from exc
    if not 0 <= seed < _SEED_CEILING:
        raise RollError(
            f"{env_var}={seed} is out of range; expected 0 <= seed < "
            f"{_SEED_CEILING}."
        )
    return seed


def resolve_seed(
    env_var: str, env: "Mapping[str, str] | None" = None,
) -> "tuple[int, str]":
    """(seed, seed_source) for one surface, from the SUPPLIED mapping.

    Reads the mapping it is handed rather than reaching for `os.environ`
    itself, so a test never has to mutate process state to pin a roll.
    """
    # snapshot() is a COPY and the single read below is immediate, so it
    # cannot disagree with the live mapping. A caller-supplied mapping is
    # still used as given -- that is this function's whole point.
    mapping = otr_env.snapshot() if env is None else env
    raw = str(mapping.get(env_var, "") or "").strip()
    if raw:
        return _parse_seed(raw, env_var), f"{env_var} override"
    return random.SystemRandom().getrandbits(_SEED_BITS), "OS entropy"


def draw(
    eligible_order: "Sequence[str]",
    seed: int,
    rng_factory: "Callable[[int], Any]" = random.Random,
) -> str:
    """Pick one id from an ordered pool. PURE -- the whole test seam.

    Given the same `(eligible_order, seed)` this always returns the same id,
    which is what makes the receipt a replay instruction rather than a note.
    """
    pool = list(eligible_order)
    if not pool:
        raise RollError("cannot draw from an empty pool")
    return pool[rng_factory(seed).randrange(len(pool))]


# ---------------------------------------------------------------------------
# source_bank
# ---------------------------------------------------------------------------

def eligible_bank_ids() -> "tuple[str, ...]":
    """Bank ids the roll may select, sorted.

    ONE filter (operator ruling 2026-07-12: the roll does NOT filter on
    rights, and no `rights_class` field exists -- understanding the terms
    under which AI output is used is the end user's call, not a gate inside
    the writer): `bank.runnable`, the ONE curation surface.

    The second filter -- "the lane's declared request compatibility" -- was
    removed 2026-08-14 with the word authority. It existed solely so a lane
    could refuse a `target_words` outside its band, and there is no longer a
    target to refuse. See `_otr_lane_specs`.
    """
    banks = _ROUTING._ensure_loaded().banks
    return tuple(sorted(
        bank_id for bank_id, bank in banks.items() if bank.runnable
    ))


def _bank_pool_diagnosis() -> str:
    """Name every filter that emptied the pool, and how many each removed."""
    banks = _ROUTING._ensure_loaded().banks
    not_runnable = [b for b in banks.values() if not b.runnable]
    return (
        f"{len(banks)} registered bank(s): "
        f"{len(not_runnable)} removed by runnable=false "
        f"({sorted(b.source_bank_id for b in not_runnable)})"
    )


def resolve_bank_selection(
    requested: str,
    *,
    source_ref: str = "",
    env: "Mapping[str, str] | None" = None,
    rng_factory: "Callable[[int], Any]" = random.Random,
) -> "tuple[str, RollReceipt | None]":
    """(concrete bank id, receipt-or-None). The ONE writer of `bank_roll`.

    A non-sentinel value returns byte-identically with NO receipt.

    A nonblank `source_ref` alongside the sentinel is REFUSED, loudly. A
    pinned source belongs to one specific bank (a Folger scene ref means
    nothing to the archive lane), so "roll the bank but keep my pinned
    source" has no coherent answer -- and silently rolling anyway would hand
    a lane a reference it cannot resolve, then blame the lane.
    """
    if not is_bank_sentinel(requested):
        return str(requested), None
    pinned = str(source_ref or "").strip()
    if pinned:
        raise RollError(
            f"source_bank is set to the roll while source_ref={pinned!r} "
            f"pins a specific source. A pinned reference belongs to ONE "
            f"bank: pick that bank directly, or clear source_ref to roll."
        )
    order = eligible_bank_ids()
    if not order:
        raise RollError(
            f"the source_bank roll has no eligible bank. "
            f"{_bank_pool_diagnosis()}. Pick a bank directly, or change "
            f"the request so a lane can run it."
        )
    seed, seed_source = resolve_seed(BANK_SEED_ENV, env)
    selected = draw(order, seed, rng_factory)
    return selected, RollReceipt(
        surface="source_bank",
        requested=BANK_SENTINEL,
        selected=selected,
        seed=seed,
        seed_source=seed_source,
        eligible_order=order,
    )


# ---------------------------------------------------------------------------
# visual_style
# ---------------------------------------------------------------------------

DYNAMIC_STYLE_ID: str = "visual_storybased"


def eligible_style_ids() -> "tuple[str, ...]":
    """Style ids the roll may select, sorted.

    Includes all registered disk packs plus the dynamic `visual_storybased`
    lane at equal odds (10 total, 10% each per Operator Ruling R1).
    """
    return tuple(sorted((*_STYLES.list_style_ids(), DYNAMIC_STYLE_ID)))


def floor_style_ids() -> "tuple[str, ...]":
    """Style ids eligible for floor fallback when dynamic visual_storybased fails.

    STRICTLY excludes `visual_storybased` (9 total) -- drawing from the floor pool
    prevents re-selecting dynamic on fallback.
    """
    return tuple(sorted(_STYLES.list_style_ids()))


def resolve_style_selection(
    requested: str,
    *,
    env: "Mapping[str, str] | None" = None,
    rng_factory: "Callable[[int], Any]" = random.Random,
) -> "tuple[str, RollReceipt | None]":
    """(concrete style id, receipt-or-None). The ONE writer of `style_roll`.

    Independent of the bank roll in every respect: its own sentinel, its own
    seed env, its own receipt. Rolling one surface never implies the other.
    """
    if not is_style_sentinel(requested):
        return str(requested), None
    order = eligible_style_ids()
    if not order:
        raise RollError(
            "the visual_style roll has no eligible style: the visual-style "
            "registry is empty. Pick a style directly, or repair the pack "
            "directory."
        )
    seed, seed_source = resolve_seed(STYLE_SEED_ENV, env)
    selected = draw(order, seed, rng_factory)
    return selected, RollReceipt(
        surface="visual_style",
        requested=STYLE_SENTINEL,
        selected=selected,
        seed=seed,
        seed_source=seed_source,
        eligible_order=order,
    )


__all__ = [
    "BANK_SEED_ENV",
    "BANK_SENTINEL",
    "DYNAMIC_STYLE_ID",
    "RECEIPT_VERSION",
    "RollError",
    "RollReceipt",
    "STYLE_SEED_ENV",
    "STYLE_SENTINEL",
    "draw",
    "eligible_bank_ids",
    "eligible_style_ids",
    "floor_style_ids",
    "is_bank_sentinel",
    "is_style_sentinel",
    "resolve_bank_selection",
    "resolve_seed",
    "resolve_style_selection",
]
