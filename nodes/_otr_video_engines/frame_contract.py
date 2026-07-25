"""THE DECLARATION SURFACE: what an adapter can render, and how it joins.

Multi-clip beat coverage (2026-07-25, chunk 2). The operator's requirement is
that a beat be covered by enough REAL rendered clips to be moving video, with
chaining preferred and a jump cut acceptable. That is only decidable if each
adapter DECLARES two things the partitioner can reason about without ever
loading a model:

* :class:`FrameContract` -- the legal render lengths. Pure, STATIC numbers:
  never live VRAM, never mutable environment. Stills are minted before the
  render phase, so a partition computed from runtime state would be a partition
  the image phase could not have planned for.
* :data:`CONTINUITY_MODES` -- whether segment N+1 can actually begin on segment
  N's terminal frame. "Accepts a still" is NOT "guarantees first-frame
  continuity": ``accepts_still`` only controls minting (``motion_common.py:393``),
  HuMo's reference is a soft identity hint per the Bug Bible, and Veo's
  ``lastFrame`` is first/last INTERPOLATION inside ONE clip, not chaining.
  Engines without strict first-frame support get a JUMP CUT, which is legal and
  honest; silently pretending they chain is neither.

EVERY ADAPTER IS ``single_only`` UNTIL IT PROVES OTHERWISE. An adapter that
declares nothing resolves to :data:`SINGLE_ONLY`, so this whole module is inert
until an adapter opts in. That is deliberate: the multi-clip path must be
opt-in per engine and provable per engine, never inherited by default.

Cold-import clean: stdlib only, no registry import at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field


#: Segment N+1 starts EXACTLY on segment N's terminal frame; the adapter
#: guarantees first-frame lock. Only this earns a CHAIN.
CONTINUITY_STRICT_FIRST_FRAME = "strict_first_frame"

#: The adapter accepts a reference image but does not guarantee it as frame 0
#: (HuMo's identity hint, Veo's interpolation endpoints). NOT chainable --
#: these take a jump cut.
CONTINUITY_SOFT_REFERENCE = "soft_reference"

#: No continuity mechanism at all (procedural visualizers, text-to-video).
CONTINUITY_NONE = "none"

CONTINUITY_MODES = (
    CONTINUITY_STRICT_FIRST_FRAME,
    CONTINUITY_SOFT_REFERENCE,
    CONTINUITY_NONE,
)


class FrameContractError(ValueError):
    """An adapter declared a frame contract that cannot be satisfied."""


@dataclass(frozen=True)
class FrameContract:
    """The STATIC, PURE set of render lengths an adapter will accept.

    ``min_frames`` / ``max_frames``
        Inclusive bounds on ONE render call. ``max_frames`` is the adapter's
        own hard ceiling, not a VRAM guess.
    ``quantum``
        Legal lengths are ``min_frames + k * quantum``. An ``8n+1`` engine
        declares ``min_frames=9, quantum=8``. ``1`` means any length in range.
    ``discrete_durations``
        For providers that only accept a fixed menu of durations (Veo's 4/6/8
        seconds). When non-empty this REPLACES the min/max/quantum arithmetic
        and the lengths are exactly these values.
    ``allow_tail_trim``
        Whether the assembler may render the smallest covering length and trim
        the excess at canonicalization. Discrete-duration lanes REQUIRE this --
        4/6/8s durations cannot sum to an arbitrary beat.
    ``supports_multi_clip``
        The opt-in. False (the default) means this engine renders a beat in one
        call and the partitioner must never split it.
    ``continuity``
        One of :data:`CONTINUITY_MODES`.

    Frozen, so a contract cannot be mutated after an adapter publishes it --
    the partitioner and the image phase must read the same numbers.
    """

    min_frames: int = 1
    max_frames: int = 0            # 0 == "unbounded / not declared"
    quantum: int = 1
    discrete_durations: tuple = field(default_factory=tuple)
    allow_tail_trim: bool = False
    supports_multi_clip: bool = False
    continuity: str = CONTINUITY_NONE

    def __post_init__(self):
        if self.continuity not in CONTINUITY_MODES:
            raise FrameContractError(
                "continuity %r is not one of %r" % (self.continuity,
                                                    CONTINUITY_MODES))
        if int(self.min_frames) < 1:
            raise FrameContractError(
                "min_frames must be >= 1, got %r" % (self.min_frames,))
        if int(self.quantum) < 1:
            raise FrameContractError(
                "quantum must be >= 1, got %r" % (self.quantum,))
        if self.max_frames and int(self.max_frames) < int(self.min_frames):
            raise FrameContractError(
                "max_frames %r is below min_frames %r"
                % (self.max_frames, self.min_frames))
        if self.discrete_durations:
            if any(int(d) < 1 for d in self.discrete_durations):
                raise FrameContractError(
                    "discrete_durations must all be >= 1, got %r"
                    % (self.discrete_durations,))
            if not self.allow_tail_trim:
                # A fixed menu of lengths cannot sum to an arbitrary beat, so
                # without tail trimming there are beats this engine could never
                # cover exactly -- that is a contract that lies.
                raise FrameContractError(
                    "discrete_durations requires allow_tail_trim=True: a fixed "
                    "duration menu cannot sum to an arbitrary beat length")
        if self.supports_multi_clip and not self.max_frames \
                and not self.discrete_durations:
            # Multi-clip exists to cover a beat LONGER than one render. An
            # engine with no ceiling has nothing to split at.
            raise FrameContractError(
                "supports_multi_clip=True requires a declared max_frames (or "
                "discrete_durations); an engine with no ceiling never splits")

    # -- pure queries -------------------------------------------------- #

    def is_legal_length(self, n: int) -> bool:
        """True iff ``n`` is a length this adapter will accept in one call."""
        n = int(n)
        if self.discrete_durations:
            return n in tuple(int(d) for d in self.discrete_durations)
        if n < int(self.min_frames):
            return False
        if self.max_frames and n > int(self.max_frames):
            return False
        return (n - int(self.min_frames)) % int(self.quantum) == 0

    def legal_lengths(self) -> tuple:
        """Every legal length, ascending. Empty when unbounded (nothing to
        enumerate -- callers must use :meth:`is_legal_length` instead)."""
        if self.discrete_durations:
            return tuple(sorted(int(d) for d in self.discrete_durations))
        if not self.max_frames:
            return ()
        return tuple(range(int(self.min_frames), int(self.max_frames) + 1,
                           int(self.quantum)))

    def largest_legal_at_most(self, n: int):
        """The largest legal length <= ``n``, or None if there is none."""
        n = int(n)
        if self.discrete_durations:
            fits = [int(d) for d in self.discrete_durations if int(d) <= n]
            return max(fits) if fits else None
        if n < int(self.min_frames):
            return None
        capped = min(n, int(self.max_frames)) if self.max_frames else n
        steps = (capped - int(self.min_frames)) // int(self.quantum)
        return int(self.min_frames) + steps * int(self.quantum)

    def smallest_legal_at_least(self, n: int):
        """The smallest legal length >= ``n``, or None if it exceeds the
        ceiling. Used with ``allow_tail_trim`` to cover a remainder exactly."""
        n = max(int(n), int(self.min_frames))
        if self.discrete_durations:
            fits = [int(d) for d in self.discrete_durations if int(d) >= n]
            return min(fits) if fits else None
        steps = -(-(n - int(self.min_frames)) // int(self.quantum))  # ceil div
        candidate = int(self.min_frames) + steps * int(self.quantum)
        if self.max_frames and candidate > int(self.max_frames):
            return None
        return candidate


#: The default every adapter gets until it declares otherwise: ONE render per
#: beat, no chaining. Nothing in the coverage path may split an engine that
#: resolves to this.
SINGLE_ONLY = FrameContract()


def frame_contract_for(engine) -> FrameContract:
    """Resolve one adapter's declared contract, defaulting to SINGLE_ONLY.

    Accepts either a ``frame_contract()`` method or a ``frame_contract``
    attribute, matching how the rest of the adapter surface duck-types. An
    adapter that declares nothing, or whose declaration raises, is treated as
    ``single_only`` -- the safe direction, because the worst case is that a
    beat renders exactly the way it does today.
    """
    if engine is None:
        return SINGLE_ONLY
    declared = getattr(engine, "frame_contract", None)
    if declared is None:
        return SINGLE_ONLY
    try:
        value = declared() if callable(declared) else declared
    except Exception:  # noqa: BLE001 -- a broken declaration is single_only
        return SINGLE_ONLY
    return value if isinstance(value, FrameContract) else SINGLE_ONLY


def supports_multi_clip(engine) -> bool:
    """True iff this adapter has opted in to multi-clip coverage."""
    return bool(frame_contract_for(engine).supports_multi_clip)


def can_chain(engine) -> bool:
    """True iff segment N+1 may begin exactly on segment N's terminal frame."""
    contract = frame_contract_for(engine)
    return bool(contract.supports_multi_clip
                and contract.continuity == CONTINUITY_STRICT_FIRST_FRAME)
