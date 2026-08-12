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

from dataclasses import dataclass, field, replace


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


def declares_continuity_kwarg(engine) -> bool:
    """True iff a ``FrameContract(...)`` call in this engine's class body passes
    ``continuity`` as a KEYWORD -- i.e. the join mode was DECIDED, not defaulted.

    The value alone cannot answer this. ``CONTINUITY_NONE`` is the dataclass
    default, so a lane that never thought about chaining and a lane that
    concluded NONE after reading its own render path are byte-identical at
    runtime. That is exactly why G3.3 asks the question at all (lesson L3).

    AST-BASED, AND THAT IS THE POINT (lane 12, 2026-08-11). The check used to be
    a substring search for ``"continuity="`` over the class's SOURCE TEXT. It
    worked while nobody discussed continuity in a comment, and it silently
    rotted the moment lanes 10-12 began WRITING DOWN why each lane's value is
    NONE -- every one of those comments contains that literal, so the check
    would have gone green for a lane whose real declaration had been deleted,
    satisfied by the paragraph explaining the declaration it no longer had.
    Found by a post-coding QA pass on a test written minutes earlier.

    Comments are not AST nodes, so prose cannot satisfy this. Lives here rather
    than in the preflight suite because the gate and every lane's own test must
    ask the SAME question -- two readers of one invariant is how they drift.

    Never raises: unreadable or unparseable source answers False, which is the
    safe direction (it reports "not declared", never a false pass).
    """
    if engine is None:
        return False
    import ast
    import inspect
    import textwrap

    for base in type(engine).__mro__:
        if base is object:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(base)))
        except Exception:  # noqa: BLE001 -- unreadable source is not a pass
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name == "FrameContract" and any(
                    kw.arg == "continuity" for kw in node.keywords):
                return True
    return False


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


# ---------------------------------------------------------------------------
# THE EFFECTIVE CONTRACT (B3, 2026-07-26) -- a tier ceiling that PLANS
# ---------------------------------------------------------------------------

#: The ONLY engines whose tier-pinned ``max_render_frames`` is a coverage
#: PLANNING cap. This is a deliberate allowlist of ONE, not a rollout.
#:
#: HISTORY, AND THE REASONING EXPIRED -- read this before trusting the paragraph
#: that follows it. The original argument was: ``max_render_frames`` is not a
#: general planning cap because WAN reads 17 from ``config/profiles/
#: otr_8gb_wan.json``, renders a short native clip and PING-PONGS it up to the
#: beat's full length (``eng_wan_ti2v._floor_length`` ->
#: ``wrapper_bridge.extend_frames_to_target``), so narrowing WAN's contract
#: before ``partition_beat`` would turn every WAN beat into a pile of 17-frame
#: renders.
#:
#: **THAT MECHANISM NO LONGER EXISTS.** ``extend_frames_to_target`` was DELETED
#: under the operator's no-mirror ruling and ``eng_wan_ti2v`` now REFUSES a beat
#: it cannot render in one affordable pass rather than padding it. Which is
#: exactly why ``wan_ti2v`` was ADDED to this list on 2026-08-02: with the mirror
#: gone the ceiling had to become a PLANNING cap, or the engine would simply
#: refuse the beats the mirror used to absorb. The comment argued against a
#: decision the file itself made thirty lines below.
#:
#: What survives is the SHAPE of the rule, not its example: membership here is a
#: per-engine decision because a ceiling that plans and a ceiling that merely
#: caps a render are different things. ``ltx_8gb`` plans real coverage, so its
#: ceiling belongs to the PLANNER.
#:
#: Adding an id here is a per-engine decision with a live proof attached, never
#: a convenience. ``tests/test_multiclip_effective_contract.py`` pins WAN's
#: topology as unmoved by a pinned ceiling.
#: ``fastwan_8gb`` ADDED 2026-08-01, with the live proof this comment demands.
#: The WAN reasoning above holds for ``wan_ti2v`` and is deliberately left alone,
#: but it does NOT survive contact with a COVERAGE-PLANNED beat. Both adapters
#: declare ``strict_first_frame`` continuity, which makes them CHAINABLE, so
#: ``partition_beat`` plans multi-clip segments up to the contract max (177) --
#: and a multi-clip beat never reaches ``_floor_length``, so the ping-pong that
#: makes the ceiling harmless on the single-clip path never runs.
#:
#: THE LIVE PROOF (canonical run, 2026-08-01, this profile): the render refused
#: by name -- "fastwan_8gb was handed a coverage-planned segment of 177 frame(s)
#: but this tier pins its render ceiling at 17" -- exactly the contradiction
#: ``_planned_length`` documents and exactly the remedy it names ("Raise
#: max_render_frames for this tier or route the beat to a single-clip engine").
#: The tier now pins 81, the highest rung the four-arm bench MEASURED at this
#: canvas (6563.1 / 6531.1 / 6563.1 MiB at 17 / 49 / 81 -- flat), and listing the
#: engine here lets the planner SEE that cap so plan and contract cannot disagree.
#: Capping the planner at 81 is not the "pile of 17-frame renders" the WAN note
#: warns about: 81 frames is 3.24 s of real motion per segment.
#: ``wan_ti2v`` ADDED 2026-08-02. Its adapter-side ping-pong was deleted under
#: the operator's no-mirror ruling, and that mirror was load-bearing: without
#: coverage planning the engine simply REFUSES any beat it cannot render in one
#: VRAM-affordable pass. Listing it lets the planner split those beats into
#: affordable NATIVE segments, which is what "original video for every second of
#: audio" requires.
PLANNING_CAP_ENGINES = ("ltx_8gb", "fastwan_8gb", "wan_ti2v")


class PlanningCapError(ValueError):
    """A tier pinned a render-length ceiling no legal segment fits under.

    Deliberately NOT a :class:`FrameContractError` (the declaration is fine)
    and deliberately NOT a ``coverage_plan.CoveragePlanError`` (the beat is
    coverable; the CONFIGURATION is not). It is a terminal operator-caused
    condition: it names the ceiling, the engine and the floor it fell under,
    and nothing catches it. A clamp to the nearest legal rung would be a silent
    fallback -- the tier asked for something the adapter cannot render, and the
    honest answer is to say so before any weights load.
    """


def normalized_planning_ceiling(value) -> int:
    """THE ONE normalization of a ``max_render_frames`` value. 0 == unpinned.

    Exists so the planner, the ledger stamp and the render boundary cannot
    drift. Before this, ``max(0, int(x or 0))`` was written out at each site;
    three literal copies of one rule is how the stamped receipt and the
    re-derived one start disagreeing for reasons that have nothing to do with
    the ceiling. Never raises -- a malformed stamp reads as UNPINNED, matching
    ``motion_common.profile_max_render_frames``, because a ceiling nobody can
    parse must not silently become a small number.
    """
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def effective_frame_contract(engine_id, contract, max_render_frames):
    """The contract the PLANNER must partition against, ceiling included.

    Returns ``contract`` UNCHANGED for every engine outside
    :data:`PLANNING_CAP_ENGINES`, for an unpinned ceiling, and for a ceiling at
    or above the adapter's own declared maximum (a ceiling of 240 on an engine
    that stops at 161 is not binding and must not pretend to be).

    Otherwise it returns a copy narrowed to the largest LEGAL rung at or below
    the ceiling -- ``9 + 8k`` for ``ltx_8gb``, so a ceiling of 100 yields 97,
    not 100. Narrowing to the raw ceiling would emit segments the adapter
    refuses; that is the failure this function exists to prevent, one level up.

    CALL THIS OUTSIDE ANY BROAD ``except``. It raises
    :class:`PlanningCapError`, which is a ``ValueError``, and both call sites
    sit next to a pre-existing ``except Exception`` that means "this engine is
    unbuildable, give it no plan". Folding the call into one of those turns a
    loud misconfiguration into a beat with no coverage plan at all -- the same
    shape as chunk 1a's four swallowed fail-closed sites, found here by three
    independent reviewers before a line was written.

    Pure. No environment, no registry, no I/O -- the image phase plans stills
    against the same numbers, so this must answer identically at plan time and
    at render time.
    """
    ceiling = normalized_planning_ceiling(max_render_frames)
    if not ceiling or str(engine_id) not in PLANNING_CAP_ENGINES:
        return contract
    if contract.discrete_frames:
        # A fixed provider menu has no ``max_frames`` to narrow -- lowering it
        # would not change ``is_legal_length``, which consults the menu FIRST,
        # so a "narrowed" discrete contract would be one that lies. No engine
        # in the allowlist declares a menu; refusing beats inventing an answer
        # for a combination that can only arrive by a future declaration.
        #
        # A ceiling at or above the whole menu is NOT binding, though, and the
        # documented guarantee at the top of this function has to hold for the
        # discrete shape too (2026-07-27, post-code panel): refusing there
        # would fail a tier that never constrained the engine at all.
        if ceiling >= max(int(d) for d in contract.discrete_frames):
            return contract
        raise PlanningCapError(
            "%s declares a discrete frame menu %r, so a render-length ceiling "
            "of %d has no legal rung to narrow to. A tier ceiling is only a "
            "planning cap for a ladder contract."
            % (engine_id, tuple(contract.discrete_frames), ceiling))
    rung = contract.largest_legal_at_most(ceiling)
    if rung is None:
        raise PlanningCapError(
            "tier ceiling max_render_frames=%d is below %s's shortest legal "
            "render (%d frames), so no segment this adapter accepts fits under "
            "it. NO FALLBACK -- clamping to an illegal length would emit a "
            "plan the adapter refuses at render time, after the weights load."
            % (ceiling, engine_id, int(contract.min_frames)))
    if contract.max_frames and rung >= int(contract.max_frames):
        return contract
    return replace(contract, max_frames=int(rung))


def coverage_contract_receipt(engine_id, contract, max_render_frames):
    """The durable sibling receipt for a NARROWED contract, or ``None``.

    ``None`` when nothing narrowed -- which is every engine outside
    :data:`PLANNING_CAP_ENGINES`, every unpinned tier, and every non-binding
    ceiling. That ``None`` is meaningful: the render boundary re-derives this
    and requires EXACT equality, so a receipt appearing or disappearing between
    plan time and render time is itself the refusal. A shot whose engine was
    swapped after the stamp (the legacy force-map path) compares a stamped dict
    against a re-derived ``None`` and fails closed, which is correct: the plan
    was built under a ceiling the new engine's lane does not honour.

    Carries the ceiling that CAUSED the narrowing, not just its result, so a
    receipt read out of a stored ledger explains itself without the profile.

    ``contract`` MUST be the adapter's DECLARED contract, never one already
    narrowed by :func:`effective_frame_contract`. A narrowed contract narrows
    to itself, compares equal, and returns ``None`` -- the receipt then
    silently never exists and the render boundary has nothing to check. The
    first draft of the stamp site made exactly that mistake by rebinding one
    variable.
    """
    ceiling = normalized_planning_ceiling(max_render_frames)
    effective = effective_frame_contract(engine_id, contract, ceiling)
    if effective == contract:
        return None
    return {
        "engine_id": str(engine_id),
        "min_frames": int(effective.min_frames),
        "max_frames": int(effective.max_frames),
        "quantum": int(effective.quantum),
        "native_fps": int(effective.native_fps),
        "allow_tail_trim": bool(effective.allow_tail_trim),
        "continuity": str(effective.continuity),
        "max_render_frames": int(ceiling),
    }
