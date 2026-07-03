"""Pure fallback-CHAIN resolver (A-S6) -- shared, dependency-free.

A video role can degrade at render time when its chosen engine is unusable (OOM,
missing weights, a load failure): the 14B ``audio_driven_face`` engine HuMo
degrades to its lighter ``humo_1.7B`` tier, which in turn degrades to the
in-process radio-floor ``still_motion``. Each engine declares a SINGLE next hop
via a ``fallback_engine`` attribute (``None`` / absent = terminal radio floor);
this module walks that single-linked chain to the floor.

It is the CPU, planning-time core of the spec's ``resolve_next_fallback(shot,
host_caps, visited)``: it proves a declared chain RESOLVES, is ACYCLIC, and
TERMINATES at the radio floor (SUBPROJECT_A: "Fallback chains -- resolve +
acyclic + the radio-floor terminates"). The render-time SUB-DAG re-resolution
(host_caps gating + the provider sub-DAG + the durable ledger restamp) layers on
top in the GPU render node; this stays a pure, side-effect-free walk so it is
testable on the CPU box and reusable by the audio / image / 3D namespaces.

Dependency-free + fail-closed: the caller injects ``fallback_of`` (an
engine-name -> next-engine-name-or-None lookup) so this module never imports any
registry. A chain that loops raises :class:`FallbackCycleError`; a chain that
will not terminate within ``max_hops`` raises :class:`FallbackChainError`. UTF-8,
no BOM, ASCII-only source.
"""
from __future__ import annotations

from typing import Callable, List, Optional

#: A safety bound: no sane fallback chain is longer than this. A chain that does
#: not terminate within it is treated as malformed (fail-closed) rather than
#: walked forever.
DEFAULT_MAX_HOPS = 16


class FallbackChainError(ValueError):
    """A malformed fallback chain (a cycle, or a runaway that never terminates)."""


class FallbackCycleError(FallbackChainError):
    """The chain revisits an engine (a cycle): it would never reach the floor."""


def resolve_fallback_chain(
    start: str,
    fallback_of: Callable[[str], Optional[str]],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> List[str]:
    """Walk the single-linked ``fallback_engine`` chain from ``start`` to the floor.

    ``fallback_of(name)`` returns the next engine name in the chain, or ``None``
    when ``name`` is terminal (the zero-VRAM radio floor, or simply has no
    declared fallback). Returns the ordered list of engine names visited,
    beginning with ``start`` and ending at the terminal engine.

    Fail-closed: a chain that revisits a name (a cycle) raises
    :class:`FallbackCycleError`; a chain that has not terminated after
    ``max_hops`` raises :class:`FallbackChainError` (a missing terminal). A falsy
    ``start`` raises ``ValueError``.
    """
    if not start:
        raise ValueError("resolve_fallback_chain needs a non-empty start engine")
    chain: List[str] = []
    seen = set()
    current: Optional[str] = start
    while current is not None:
        if current in seen:
            raise FallbackCycleError(
                "fallback chain cycles at '%s' (visited: %s)"
                % (current, " -> ".join(chain))
            )
        chain.append(current)
        seen.add(current)
        if len(chain) > max_hops:
            raise FallbackChainError(
                "fallback chain from '%s' exceeded %d hops without terminating: %s"
                % (start, max_hops, " -> ".join(chain))
            )
        current = fallback_of(current)
    return chain


def chain_terminates_at_floor(
    start: str,
    fallback_of: Callable[[str], Optional[str]],
    floor_names,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> bool:
    """True iff the chain from ``start`` resolves (acyclic, bounded) and ENDS at
    one of ``floor_names`` (the radio-floor engines that always succeed). A
    convenience over :func:`resolve_fallback_chain` for the convergence gate."""
    chain = resolve_fallback_chain(start, fallback_of, max_hops=max_hops)
    return chain[-1] in set(floor_names)


__all__ = [
    "DEFAULT_MAX_HOPS",
    "FallbackChainError",
    "FallbackCycleError",
    "resolve_fallback_chain",
    "chain_terminates_at_floor",
]
