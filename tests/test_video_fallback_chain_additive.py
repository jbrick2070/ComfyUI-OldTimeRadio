"""Additive CPU coverage for the pure fallback-CHAIN resolver (A-S6).

New, self-contained edge-case tests for nodes/_otr_shared/fallback.py that
complement tests/test_video_retry_taxonomy.py (which walks the real declared
chain). These pin the resolver's pure contract: a single-linked chain resolves
in order, terminates at a declared floor, and FAILS CLOSED on a cycle, on a
runaway that never terminates, and on an empty start. No GPU, no model load --
stdlib + the module under test only. UTF-8, no BOM, ASCII-only, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared.fallback import (
    DEFAULT_MAX_HOPS,
    FallbackChainError,
    FallbackCycleError,
    chain_terminates_at_floor,
    resolve_fallback_chain,
)


def _chain_map(mapping):
    """Return a fallback_of(name) -> next-or-None lookup over a plain dict."""
    return lambda name: mapping.get(name)


def test_single_terminal_returns_just_start():
    assert resolve_fallback_chain("still_kenburns", _chain_map({})) == [
        "still_kenburns"
    ]


def test_linear_chain_resolves_in_order():
    mapping = {"humo": "humo_1.7B", "humo_1.7B": "still_kenburns"}
    assert resolve_fallback_chain("humo", _chain_map(mapping)) == [
        "humo",
        "humo_1.7B",
        "still_kenburns",
    ]


def test_chain_terminates_at_floor_true_and_false():
    mapping = {"humo": "humo_1.7B", "humo_1.7B": "still_kenburns"}
    fb = _chain_map(mapping)
    assert chain_terminates_at_floor("humo", fb, {"still_kenburns"}) is True
    # ends at humo_1.7B, which is not a declared floor name
    assert chain_terminates_at_floor("humo_1.7B", fb, {"abstract"}) is False


def test_cycle_raises_cycle_error():
    mapping = {"a": "b", "b": "c", "c": "a"}
    with pytest.raises(FallbackCycleError):
        resolve_fallback_chain("a", _chain_map(mapping))


def test_cycle_error_is_a_chain_error():
    assert issubclass(FallbackCycleError, FallbackChainError)


def test_runaway_chain_raises_chain_error_at_max_hops():
    # never repeats (no cycle) and never terminates -> must trip max_hops.
    def fb(name):
        return name + "x"

    with pytest.raises(FallbackChainError) as exc:
        resolve_fallback_chain("e", fb, max_hops=8)
    # a pure cycle would be a FallbackCycleError; a runaway is the base class
    assert not isinstance(exc.value, FallbackCycleError)


def test_empty_start_raises_value_error():
    with pytest.raises(ValueError):
        resolve_fallback_chain("", _chain_map({}))


def test_exact_max_hops_chain_is_allowed():
    # a chain whose length == DEFAULT_MAX_HOPS that then terminates resolves OK.
    n = DEFAULT_MAX_HOPS
    names = ["e%d" % i for i in range(n)]
    mapping = {names[i]: names[i + 1] for i in range(n - 1)}
    chain = resolve_fallback_chain(names[0], _chain_map(mapping))
    assert chain == names
    assert len(chain) == n


def test_custom_max_hops_boundary():
    mapping = {"a": "b", "b": "c", "c": "d"}
    fb = _chain_map(mapping)
    # resolves fine with a generous bound
    assert resolve_fallback_chain("a", fb, max_hops=16) == ["a", "b", "c", "d"]
    # but a tiny bound trips before termination
    with pytest.raises(FallbackChainError):
        resolve_fallback_chain("a", fb, max_hops=2)
