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

EVERY ENGINE GETS THE SAME TERMS (chunk 7a, 2026-07-26). This module used to
open with "EVERY ADAPTER IS ``single_only`` UNTIL IT PROVES OTHERWISE" and
carried a ``supports_multi_clip`` flag that each adapter had to set for itself.
The operator ended that design: "this architecture should work with all video
and still models. There's no gate with opt in or opt out. If there is, we need
to remove that. Everything gets an equal term... I don't like any hidden
opt-ins. It either works or it fails."

So the flag is gone and multi-clip is universal. What an adapter must still
EARN, per engine and from evidence, is the CHAIN -- see :data:`CONTINUITY_MODES`
below. An engine that cannot prove first-frame lock gets an honest jump cut,
which is legal; it does not get a pretended seam.

:data:`SINGLE_ONLY` survives as the resolution for an engine that declares
NOTHING, and ``tests/test_engine_contract_roster.py`` fails by name if any
registered engine ever resolves to it again.

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


class ContractEnvConflict(RuntimeError):
    """The ENVIRONMENT asked for something a static declaration forbids.

    Deliberately NOT a :class:`FrameContractError`. That one means "the
    declaration itself is incoherent" and is raised at construction, so it can
    never reach a render. This one means "the declaration is fine and the box
    was told something else", which is a runtime condition an operator caused
    and an operator can fix -- so it carries the variable name and the legal
    values, and is caught by nothing.

    SCOPE, HONESTLY (2026-07-26). Only ``word_razzle`` raises this today, for
    an env duration outside its provider's fixed menu. The wider question --
    whether an env var may lower a declared CEILING at all -- is deliberately
    NOT answered here: several of those vars are documented operator knobs for
    VRAM-constrained boxes (the WAN 8GB launch contract depends on one), so
    refusing them outright would break a real production path. The fork is
    written up in ``docs/2026-07-26-chunk-7b-window-prompt.md`` under OPEN
    DECISION and is the operator's to settle.
    """


@dataclass(frozen=True)
class FrameContract:
    """The STATIC, PURE set of render lengths an adapter will accept.

    ``min_frames`` / ``max_frames``
        Inclusive bounds on ONE render call. ``max_frames`` is the adapter's
        own hard ceiling, not a VRAM guess.
    ``quantum``
        Legal lengths are ``min_frames + k * quantum``. An ``8n+1`` engine
        declares ``min_frames=9, quantum=8``. ``1`` means any length in range.
    ``discrete_frames``
        For providers that only accept a fixed menu of clip lengths (Veo's
        4/6/8 SECONDS). When non-empty this REPLACES the min/max/quantum
        arithmetic and the legal lengths are exactly these values.

        THE UNIT IS FRAMES, NOT SECONDS. ``coverage_plan`` compares these
        numbers against ``target_visible_frames`` directly, so a provider menu
        expressed in seconds MUST be multiplied by :attr:`native_fps` at the
        declaration site. Veo's 4/6/8 s at 24 fps declares ``(96, 144, 192)``.
        The field was named ``discrete_durations`` until 2026-07-26; it was
        renamed because the old name invited exactly that seconds-for-frames
        substitution, and a validator cannot catch it -- ``(4, 6, 8)`` is a
        perfectly well-formed frame menu, just a wrong one.
    ``native_fps``
        The frame rate these frame numbers are expressed at, or ``0`` for an
        engine that renders at whatever the canvas asks for. Static and
        informational: it is what lets the matrix report seconds, and what
        tells a reader that a 24 fps provider is being planned against a
        25 fps canvas. It is NOT read by the partitioner.
    ``allow_tail_trim``
        Whether the assembler may render the smallest covering length and trim
        the excess at canonicalization. Discrete-duration lanes REQUIRE this --
        4/6/8s durations cannot sum to an arbitrary beat.
    ``continuity``
        One of :data:`CONTINUITY_MODES`.

    Frozen, so a contract cannot be mutated after an adapter publishes it --
    the partitioner and the image phase must read the same numbers.
    """

    min_frames: int = 1
    max_frames: int = 0            # 0 == "unbounded / not declared"
    quantum: int = 1
    discrete_frames: tuple = field(default_factory=tuple)
    native_fps: int = 0            # 0 == "renders at the canvas rate"
    allow_tail_trim: bool = False
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
        if int(self.native_fps) < 0:
            raise FrameContractError(
                "native_fps must be >= 0, got %r" % (self.native_fps,))
        if self.discrete_frames:
            if any(int(d) < 1 for d in self.discrete_frames):
                raise FrameContractError(
                    "discrete_frames must all be >= 1, got %r"
                    % (self.discrete_frames,))
            if not self.allow_tail_trim:
                # A fixed menu of lengths cannot sum to an arbitrary beat, so
                # without tail trimming there are beats this engine could never
                # cover exactly -- that is a contract that lies.
                raise FrameContractError(
                    "discrete_frames requires allow_tail_trim=True: a fixed "
                    "length menu cannot sum to an arbitrary beat length")

    # -- pure queries -------------------------------------------------- #

    def is_legal_length(self, n: int) -> bool:
        """True iff ``n`` is a length this adapter will accept in one call."""
        n = int(n)
        if self.discrete_frames:
            return n in tuple(int(d) for d in self.discrete_frames)
        if n < int(self.min_frames):
            return False
        if self.max_frames and n > int(self.max_frames):
            return False
        return (n - int(self.min_frames)) % int(self.quantum) == 0

    def legal_lengths(self) -> tuple:
        """Every legal length, ascending. Empty when unbounded (nothing to
        enumerate -- callers must use :meth:`is_legal_length` instead)."""
        if self.discrete_frames:
            return tuple(sorted(int(d) for d in self.discrete_frames))
        if not self.max_frames:
            return ()
        return tuple(range(int(self.min_frames), int(self.max_frames) + 1,
                           int(self.quantum)))

    def largest_legal_at_most(self, n: int):
        """The largest legal length <= ``n``, or None if there is none."""
        n = int(n)
        if self.discrete_frames:
            fits = [int(d) for d in self.discrete_frames if int(d) <= n]
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
        if self.discrete_frames:
            fits = [int(d) for d in self.discrete_frames if int(d) >= n]
            return min(fits) if fits else None
        steps = -(-(n - int(self.min_frames)) // int(self.quantum))  # ceil div
        candidate = int(self.min_frames) + steps * int(self.quantum)
        if self.max_frames and candidate > int(self.max_frames):
            return None
        return candidate


#: What an engine that declares NOTHING resolves to: unbounded, quantum 1, no
#: chaining. Every length is legal, so it never splits and never refuses -- the
#: pre-7a behaviour of the whole tree, kept as the honest answer for a stub, a
#: custom slot, or an adapter that failed to import. No REGISTERED engine may
#: resolve to it; ``tests/test_engine_contract_roster.py`` fails by name if one
#: does.
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


def can_split(engine) -> bool:
    """True iff a beat could ever need more than one clip on this adapter.

    REPLACES ``supports_multi_clip(engine)`` (chunk 7a, 2026-07-26), which read
    a per-adapter opt-in flag that no longer exists. The question is no longer
    "did this engine volunteer" -- every engine is in -- but the strictly
    arithmetic "does this engine have a ceiling to exceed". An unbounded engine
    (the visualizers, the still families, mesh_stage) accepts any length, so no
    beat can ever overflow one render and no split can ever be needed.
    """
    contract = frame_contract_for(engine)
    return bool(contract.max_frames or contract.discrete_frames)


def can_chain(engine) -> bool:
    """True iff segment N+1 may begin exactly on segment N's terminal frame.

    The one thing still EARNED per adapter. Multi-clip is universal; a seamless
    join is not, because only an engine that genuinely locks frame 0 to a
    supplied image can deliver one. Everything else jump cuts, honestly.
    """
    return frame_contract_for(engine).continuity == CONTINUITY_STRICT_FIRST_FRAME
