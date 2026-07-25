"""THE PARTITIONER: how many real clips cover one beat, and exactly how long.

Multi-clip beat coverage (2026-07-25, chunk 3). The operator's requirement is
that a beat be covered by enough REAL rendered clips to be moving video --
chain (segment N+1 begins on segment N's terminal frame) preferred, jump cut
acceptable, and NEVER a mirror/ping-pong or a held last frame.

PURE AND STATIC. This module reads no environment, no VRAM, no clock, and no
registry. It takes a beat's visible frame target and one
:class:`~.frame_contract.FrameContract` and returns a :class:`CoveragePlan`.
That is a hard requirement, not a style preference: stills are minted BEFORE
the render phase, so a partition that depended on runtime state would be one
the image phase could not have planned stills for.

THE SEAM ARITHMETIC, which is the subtle part. Under CHAIN the successor's
first frame IS the predecessor's terminal frame, so concatenating both whole
would show that frame twice. Each successor therefore drops its head frame,
and the invariant every plan must satisfy is::

    sum(render_frames - drop_head - trim_tail) == target_visible_frames

exactly. Not approximately. A beat whose assembled length drifts from its
audio is the defect this build exists to remove, so an inexact plan is a
terminal error rather than a rounding.

WHY A SEARCH AND NOT A GREEDY WALK. Legal lengths form the arithmetic ladder
``min + k*quantum``, so taking the largest legal chunk first can strand a
remainder that is not itself legal (an ``8n+1`` engine covering 170 visible
frames strands 1 frame, and 1 is not a legal render). Solving for the segment
COUNT first always finds an exact partition when one exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .frame_contract import (
    CONTINUITY_STRICT_FIRST_FRAME,
    FrameContract,
)

#: Segment N+1 begins exactly on segment N's terminal frame. Preferred.
JOIN_CHAIN = "chain"
#: Segments are independent renders butted together. Legal and honest for any
#: engine that cannot lock frame 0.
JOIN_JUMP = "jump"
#: One render, the whole beat. Every single_only engine.
JOIN_SINGLE = "single"

JOIN_MODES = (JOIN_CHAIN, JOIN_JUMP, JOIN_SINGLE)


class CoveragePlanError(ValueError):
    """A beat cannot be covered exactly by this adapter's frame contract."""


@dataclass(frozen=True)
class CoverageSegment:
    """One real render call inside a beat.

    ``render_frames``  what the adapter is asked to render (always legal).
    ``drop_head``      frames removed from the FRONT at assembly. 1 for a
                       chained successor (it duplicates its predecessor's
                       terminal frame), else 0.
    ``trim_tail``      frames removed from the END at assembly. Non-zero only
                       when the contract allows tail trimming and the ladder
                       cannot land on the target exactly.
    ``index``          0-based position within the beat.
    """

    index: int
    render_frames: int
    drop_head: int = 0
    trim_tail: int = 0

    @property
    def visible_frames(self) -> int:
        """What this segment actually contributes to the assembled beat."""
        return int(self.render_frames) - int(self.drop_head) - int(self.trim_tail)


@dataclass(frozen=True)
class CoveragePlan:
    """The complete, durable description of how one beat gets covered."""

    target_visible_frames: int
    join_mode: str
    segments: tuple = field(default_factory=tuple)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def total_visible_frames(self) -> int:
        return sum(s.visible_frames for s in self.segments)

    @property
    def is_multi_clip(self) -> bool:
        return len(self.segments) > 1

    def to_dict(self) -> dict:
        """JSON-safe form for the durable ledger stamp."""
        return {
            "target_visible_frames": int(self.target_visible_frames),
            "join_mode": str(self.join_mode),
            "segments": [
                {"index": int(s.index), "render_frames": int(s.render_frames),
                 "drop_head": int(s.drop_head), "trim_tail": int(s.trim_tail)}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data) -> "CoveragePlan":
        data = data or {}
        return cls(
            target_visible_frames=int(data.get("target_visible_frames") or 0),
            join_mode=str(data.get("join_mode") or JOIN_SINGLE),
            segments=tuple(
                CoverageSegment(
                    index=int(row.get("index") or 0),
                    render_frames=int(row.get("render_frames") or 0),
                    drop_head=int(row.get("drop_head") or 0),
                    trim_tail=int(row.get("trim_tail") or 0),
                )
                for row in (data.get("segments") or ())
            ),
        )


def join_mode_for(contract: FrameContract, target_visible_frames: int) -> str:
    """Which join a beat gets, from the adapter's own declaration.

    ``single`` unless the adapter opted in to multi-clip AND the beat actually
    exceeds one legal render. Only ``strict_first_frame`` earns a chain: a soft
    identity reference (HuMo) or interpolation endpoints (Veo ``lastFrame``) do
    not lock frame 0, so those lanes jump-cut rather than pretend.
    """
    target = int(target_visible_frames)
    if not contract.supports_multi_clip:
        return JOIN_SINGLE
    if contract.is_legal_length(target):
        return JOIN_SINGLE
    # Fits inside ONE render but not on the ladder -- still one clip, with the
    # tail trim covering the remainder. The existence check is load-bearing
    # (2026-07-25 QA): this used to return SINGLE whenever the target was
    # merely <= max_frames, but the smallest legal length AT OR ABOVE the
    # target can still exceed the ceiling. A min=1 quantum=2 max=12 engine
    # asked for 12 frames has no single legal render (the ladder is odd:
    # 1,3,..,11) yet 11 + 1 covers it exactly as two clips -- and the old test
    # declared it SINGLE, then refused it. Found by a differential sweep after
    # the first two math fixes, missed by the panel.
    if contract.allow_tail_trim \
            and contract.smallest_legal_at_least(target) is not None:
        return JOIN_SINGLE
    return (JOIN_CHAIN
            if contract.continuity == CONTINUITY_STRICT_FIRST_FRAME
            else JOIN_JUMP)


def _ladder_partition(total_render_frames, contract, count):
    """Split ``total_render_frames`` into exactly ``count`` legal lengths.

    Legal lengths are ``min + quantum*a`` with ``0 <= a <= a_max``, so a split
    exists iff the residue is divisible by ``quantum`` and the required number
    of quantum steps fits inside ``count`` segments. Fills each segment toward
    the ceiling in order, which keeps the plan deterministic and puts the short
    segment last -- where a viewer expects a beat to end.
    """
    min_f, q = int(contract.min_frames), int(contract.quantum)
    residue = int(total_render_frames) - count * min_f
    if residue < 0 or residue % q:
        return None
    steps = residue // q
    max_steps_each = (((int(contract.max_frames) - min_f) // q)
                      if contract.max_frames else steps)
    if max_steps_each < 0 or steps > count * max_steps_each:
        return None
    out = []
    for _ in range(count):
        take = min(steps, max_steps_each)
        out.append(min_f + take * q)
        steps -= take
    return out


def _candidate_totals(required, contract, count):
    """Total render values worth trying for a TRIMMED cover, cheapest first.

    For a ladder contract the total of ``count`` legal lengths is always
    ``count*min + quantum*A``, so the reachable totals are exactly the values
    congruent to ``count*min`` modulo ``quantum`` and bounded by ``count*max``.
    That makes the smallest total at or above ``required`` fully determined --
    exactly one candidate, nothing to guess.

    A discrete menu is not an arithmetic progression, so there the reachable
    totals are scanned upward. The scan is bounded by the largest menu entry:
    trimming more than one whole segment's worth would mean the last segment
    contributes nothing, which the validator rejects anyway.
    """
    required = int(required)
    if contract.discrete_durations:
        span = max(int(d) for d in contract.discrete_durations)
        ceiling = count * span
        return range(required, min(required + span, ceiling) + 1)
    min_f, q = int(contract.min_frames), int(contract.quantum)
    base = count * min_f
    total = base if required <= base else (
        base + -(-(required - base) // q) * q)        # ceil to the next step
    if contract.max_frames and total > count * int(contract.max_frames):
        return ()
    return (total,)


def _discrete_partition(total_render_frames, contract, count):
    """Exact split over a fixed duration menu (Veo's 4/6/8s), or None.

    MEMOIZED, and that is not an optimization -- it is a correctness-of-service
    fix (2026-07-25 QA). The bare recursive descent explores up to
    ``len(menu) ** count`` nodes, so with a four-value menu it took 18s at
    count=14 and was still running past 20s at count=16. Since
    :func:`partition_beat` walks counts up to ``max_segments`` (64), an
    unsatisfiable target would hang the calling thread indefinitely rather than
    refuse -- a render node that never returns is worse than one that fails.
    Memoizing on ``(remaining, left)`` bounds the search to
    ``total * count`` states, which is small and predictable.
    """
    menu = sorted({int(d) for d in contract.discrete_durations}, reverse=True)
    if not menu:
        return None
    smallest = menu[-1]
    memo = {}

    def _walk(remaining, left):
        if left == 0:
            return [] if remaining == 0 else None
        # Bound the branch before recursing: `left` segments can cover at most
        # left*largest and at least left*smallest.
        if remaining < left * smallest or remaining > left * menu[0]:
            return None
        key = (remaining, left)
        if key in memo:
            return memo[key]
        result = None
        for value in menu:
            if value > remaining:
                continue
            rest = _walk(remaining - value, left - 1)
            if rest is not None:
                result = [value] + rest
                break
        memo[key] = result
        return result

    return _walk(int(total_render_frames), count)


def partition_beat(target_visible_frames, contract, *, join_mode=None,
                   max_segments=64) -> CoveragePlan:
    """THE ENTRY POINT: cover ``target_visible_frames`` exactly.

    Raises :class:`CoveragePlanError` when no exact cover exists, rather than
    returning a plan whose assembled length would drift from the beat's audio.
    """
    target = int(target_visible_frames)
    if target < 1:
        raise CoveragePlanError(
            "target_visible_frames must be >= 1, got %r" % (target_visible_frames,))

    mode = join_mode or join_mode_for(contract, target)
    if mode not in JOIN_MODES:
        raise CoveragePlanError("unknown join_mode %r" % (mode,))

    # ---- one render, the whole beat -------------------------------------- #
    if mode == JOIN_SINGLE:
        if contract.is_legal_length(target):
            return CoveragePlan(target, mode,
                                (CoverageSegment(0, target),))
        if contract.allow_tail_trim:
            render = contract.smallest_legal_at_least(target)
            if render is not None:
                return CoveragePlan(
                    target, mode,
                    (CoverageSegment(0, render, trim_tail=render - target),))
        raise CoveragePlanError(
            "beat of %d visible frame(s) is not a legal single render for this "
            "contract (min=%s max=%s quantum=%s discrete=%s allow_tail_trim=%s) "
            "and the adapter has not opted in to multi-clip coverage"
            % (target, contract.min_frames, contract.max_frames,
               contract.quantum, contract.discrete_durations,
               contract.allow_tail_trim))

    # ---- multi-clip ------------------------------------------------------- #
    # Under CHAIN every successor duplicates its predecessor's terminal frame,
    # so covering `target` visible frames with k segments requires rendering
    # `target + (k - 1)` frames in total.
    drop = 1 if mode == JOIN_CHAIN else 0
    splitter = (_discrete_partition if contract.discrete_durations
                else _ladder_partition)

    for count in range(2, int(max_segments) + 1):
        total_render = target + drop * (count - 1)
        lengths = splitter(total_render, contract, count)
        if lengths is not None:
            return _build(target, mode, lengths, drop)

    # ---- exact failed; trim the tail if the contract permits it ----------- #
    #
    # THE SEARCH RANGE IS DERIVED, NOT GUESSED (2026-07-25 QA fix). This loop
    # used to try only ``extra in range(1, quantum + 1)``, on the assumption
    # that any shortfall could be bridged within one quantum step. That is
    # false whenever ``count * min_frames`` overshoots the required total by
    # more than one step -- which happens routinely just above ``max_frames``
    # when ``min_frames`` is large relative to the gap. An adversarial sweep
    # found 832 beats that WERE coverable by trimming and were refused anyway.
    # Smallest repro: min=4 max=5 quantum=1 target=6, where [4, 4] with a
    # 2-frame tail trim is a perfectly legal cover.
    #
    # The honest range comes from the ladder itself: for a given segment count
    # the total render must be congruent to ``count * min_frames`` modulo the
    # quantum, so the smallest legal total at or above the required total is
    # fully determined. There is nothing left to guess.
    if contract.allow_tail_trim:
        for count in range(2, int(max_segments) + 1):
            required = target + drop * (count - 1)
            for total in _candidate_totals(required, contract, count):
                lengths = splitter(total, contract, count)
                if lengths is None:
                    continue
                trim = total - required
                # The trim comes off the LAST segment, which must still
                # contribute at least one visible frame after its head drop.
                if lengths[-1] - drop - trim < 1:
                    continue
                return _build(target, mode, lengths, drop, trim_tail=trim)

    raise CoveragePlanError(
        "no exact multi-clip cover of %d visible frame(s) exists for this "
        "contract (min=%s max=%s quantum=%s discrete=%s allow_tail_trim=%s) "
        "within %d segments -- refusing to emit a plan whose assembled length "
        "would drift from the beat audio"
        % (target, contract.min_frames, contract.max_frames, contract.quantum,
           contract.discrete_durations, contract.allow_tail_trim, max_segments))


def _build(target, mode, lengths, drop, trim_tail=0):
    segments = []
    for index, render in enumerate(lengths):
        segments.append(CoverageSegment(
            index=index,
            render_frames=int(render),
            drop_head=(drop if index else 0),
            trim_tail=(int(trim_tail) if index == len(lengths) - 1 else 0),
        ))
    plan = CoveragePlan(target, mode, tuple(segments))
    validate_coverage_plan(plan, None)   # arithmetic self-check, contract-free
    return plan


def validate_coverage_plan(plan: CoveragePlan, contract):
    """Validate a plan at a BOUNDARY. Raises :class:`CoveragePlanError`.

    Called on BOTH sides of the wire -- where the plan is serialized and again
    before it is executed -- because a plan that survives serialization but not
    execution is exactly the class of defect a durable stamp is supposed to
    prevent. ``contract`` may be ``None`` to check the arithmetic alone.
    """
    if not plan.segments:
        raise CoveragePlanError("coverage plan has no segments")
    if plan.join_mode not in JOIN_MODES:
        raise CoveragePlanError("unknown join_mode %r" % (plan.join_mode,))

    for index, seg in enumerate(plan.segments):
        if seg.index != index:
            raise CoveragePlanError(
                "segment %d carries index %d -- segment order is the assembly "
                "order and must be dense and ascending" % (index, seg.index))
        if seg.render_frames < 1:
            raise CoveragePlanError(
                "segment %d renders %d frame(s)" % (index, seg.render_frames))
        if seg.drop_head < 0 or seg.trim_tail < 0:
            raise CoveragePlanError(
                "segment %d has a negative trim" % index)
        if seg.visible_frames < 1:
            raise CoveragePlanError(
                "segment %d contributes %d visible frame(s) after trims -- a "
                "segment that contributes nothing must not be rendered"
                % (index, seg.visible_frames))
        expected_head = (1 if (plan.join_mode == JOIN_CHAIN and index) else 0)
        if seg.drop_head != expected_head:
            raise CoveragePlanError(
                "segment %d drops %d head frame(s), expected %d for join_mode "
                "%r -- under a chain every successor drops exactly the "
                "duplicated terminal frame, and nothing else may"
                % (index, seg.drop_head, expected_head, plan.join_mode))

    if plan.join_mode == JOIN_SINGLE and plan.segment_count != 1:
        raise CoveragePlanError(
            "join_mode 'single' with %d segments" % plan.segment_count)

    total = plan.total_visible_frames
    if total != int(plan.target_visible_frames):
        raise CoveragePlanError(
            "coverage plan assembles to %d visible frame(s) but the beat "
            "target is %d. The assembled clip would drift from the beat audio; "
            "a plan is exact or it is not a plan."
            % (total, plan.target_visible_frames))

    if contract is not None:
        for seg in plan.segments:
            if not contract.is_legal_length(seg.render_frames):
                raise CoveragePlanError(
                    "segment %d renders %d frame(s), which this adapter cannot "
                    "accept (min=%s max=%s quantum=%s discrete=%s)"
                    % (seg.index, seg.render_frames, contract.min_frames,
                       contract.max_frames, contract.quantum,
                       contract.discrete_durations))
        if plan.is_multi_clip and not contract.supports_multi_clip:
            raise CoveragePlanError(
                "a %d-segment plan was built for an adapter that has not opted "
                "in to multi-clip coverage" % plan.segment_count)
        if plan.join_mode == JOIN_CHAIN \
                and contract.continuity != CONTINUITY_STRICT_FIRST_FRAME:
            raise CoveragePlanError(
                "join_mode 'chain' requires continuity %r, but this adapter "
                "declares %r -- it cannot guarantee that a successor begins on "
                "its predecessor's terminal frame, so this beat must jump cut"
                % (CONTINUITY_STRICT_FIRST_FRAME, contract.continuity))
        if any(s.trim_tail for s in plan.segments) and not contract.allow_tail_trim:
            raise CoveragePlanError(
                "plan trims a tail but the adapter declares allow_tail_trim=False")
    return plan


# --------------------------------------------------------------------------- #
# JUMP-STILL REQUESTS -- what the IMAGE phase owes a jump-cut beat (chunk 4)
# --------------------------------------------------------------------------- #
#
# A CHAIN successor begins on its predecessor's terminal frame, so it needs no
# new still. A JUMP successor is an INDEPENDENT render with nothing to begin
# from, and the image phase mints exactly ONE still per beat -- so without this
# every segment after the first would render from no init image at all. That is
# the text-only / dark-floor degradation this build exists to remove, which is
# why chunk 4 exists at all: WITHOUT IT A JUMP CUT HAS NO STILL.
#
# ONE AUTHORITY, THREE CONSUMERS, NO RE-DERIVATION. OTR_ShotLock mints these
# rows and stamps them durably on the shot; the image dispatcher and the still
# spine READ them off the ledger rather than recomputing an id from a beat id.
# That is deliberate. The beat id a SHOT renders and the beat id an image
# OBJECT was keyed under pass through a canonicalizing remap (the positioned
# music opener, ``render_driver._canonical_visual_beat_id``), so two
# independent derivations are two chances to disagree about one string -- the
# mirror class chunk 1a collapsed for routing.

#: The ledger ``kind`` every jump-segment still carries. Deliberately NOT a
#: ``scene_*`` kind: ``render_driver._still_index`` and
#: ``_still_spine_row_for_beat`` both select scene rows BY BEAT with a
#: last-write-wins / plate precedence, so a segment still wearing a scene kind
#: would shadow the beat's own still and segment 0 would render from the LAST
#: segment's image. A distinct kind makes the segment rows invisible to every
#: existing consumer, which is what keeps this chunk behaviour-inert.
JUMP_STILL_KIND = "jump_segment"

#: Object-id prefix for a jump-segment still.
JUMP_STILL_ID_PREFIX = "jumpstill_"


def jump_still_object_id(beat_id, segment_index) -> str:
    """The durable object id of one jump segment's still. Pure."""
    return "%s%s_s%d" % (JUMP_STILL_ID_PREFIX, str(beat_id or ""),
                         int(segment_index))


def jump_still_requests(plan, beat_id, *, role="", engine_id="", char_id=""):
    """Every still this beat's plan owes the image phase, in segment order.

    EMPTY for a single-clip beat and for a CHAIN. One request per segment
    AFTER the first under a JUMP: segment 0 uses the beat's existing scene
    still, which the image phase already mints and the spine already validates,
    so minting a second still for it would orphan the first.

    Returns plain JSON-safe dicts because they ride the durable ledger.
    """
    if plan is None or plan.join_mode != JOIN_JUMP or not plan.is_multi_clip:
        return ()
    beat = str(beat_id or "")
    if not beat:
        raise CoveragePlanError(
            "a jump-cut beat needs a beat_id to key its per-segment stills -- "
            "an unkeyed request could never be matched to a rendered still")
    out = []
    for seg in plan.segments[1:]:
        row = {
            "object_id": jump_still_object_id(beat, seg.index),
            "kind": JUMP_STILL_KIND,
            "beat_id": beat,
            "segment_index": int(seg.index),
            "role": str(role or ""),
            "engine_id": str(engine_id or ""),
        }
        if char_id:
            row["char_id"] = str(char_id)
        out.append(row)
    return tuple(out)
