"""nodes/_otr_lane_specs.py -- the ONE story-lane authority.

A source bank's `default_story_pipeline` decides HOW its episode executes:
either a DISPATCHED lane (a dedicated runner module) or this writer's own
INLINE body. Before this module that authority lived inside
`OTR_LedgerScriptWriter` as `_RUNNER_BY_PIPELINE` /
`_LEGACY_INLINE_PIPELINES` / `_resolve_lane_runner`, which meant anything
that is NOT the writer -- the bank randomizer, a future replay tool, a
checker -- had to import the writer (and therefore ComfyUI) to ask a
question about lanes. It lives here now. There is exactly ONE table; no
view, no shadow copy.

Two entry points, one policy (kibitz r3 headline: a single function cannot
BOTH re-raise a runner's native error and raise one catchable type):

* `assert_supported` -- the writer's entry gate. Re-raises the lane's OWN
  exception, unwrapped and unreworded, so a direct pick fails exactly as it
  always did.
* `is_roll_compatible` -- the randomizer's filter. Resolves only the lane's
  DECLARED compat errors and returns False for those. An ImportError,
  AttributeError or genuine runner defect PROPAGATES: a bare `except` would
  silently turn a bug into "ineligible", which is the silent-degrade class
  this repo bans.

LaneSpec stores NAMES, never callables and never exception CLASSES.
Building the table out of imported objects would drag every runner module
into ComfyUI startup (the writer imports this module at top level), which
defeats the lazy-import contract the runner wrappers exist to keep.

Import direction is one-way and acyclic:
`_otr_story_routing <- _otr_lane_specs <- _otr_bank_roll <- OTR_LedgerScriptWriter`.
This module never imports the writer.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

try:
    from . import _otr_story_routing as _ROUTING
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_story_routing as _ROUTING  # type: ignore


class UnknownLanePipelineError(_ROUTING.StoryRoutingError):
    """A pipeline id that is neither a dispatched lane nor a known inline one.

    A runnable bank whose pipeline has no registered execution lane is a
    WIRING bug, not a runtime condition to degrade around. Routing's
    `executable` flag stays metadata-only and never gates this.
    """


@dataclass(frozen=True)
class RollRequest:
    """The request fields a lane may legitimately refuse.

    Lives here rather than in `_otr_bank_roll` so the import edge between
    the two stays one-way: the roll imports lane specs, never the reverse.

    Only `target_words` today. It is deliberately NOT a general "everything
    the operator typed" bag -- a lane may decline a REQUEST SHAPE it cannot
    execute, and nothing else. Word count is never a quality gate here
    (docs: "we never chase word count"); this is purely "would this lane
    raise before it wrote a word".
    """

    target_words: int


@dataclass(frozen=True)
class LaneSpec:
    """One dispatched lane, by NAME. Nothing here is imported eagerly."""

    module: str
    """Runner module, relative to this package (e.g. "_otr_scifi_codex")."""

    runner_attr: str
    """The lane entry point inside `module`."""

    compat_attr: str
    """Target-only preflight predicate inside `module`; "" = unconstrained.

    Empty is a DECLARATION, not an absence: every dispatched lane states its
    policy explicitly, because "no hook found" silently meaning "compatible"
    is how an incompatible lane wins a roll it should never have entered.
    """

    compat_error_attrs: "tuple[str, ...]" = ()
    """Exception attribute names in `module` that mean "this lane declines
    this request". ONLY these are caught by `is_roll_compatible`."""


# The dispatched lanes. The key IS the pipeline id from pipelines.json.
# A pipeline absent from BOTH this table and INLINE_PIPELINES raises.
LANE_SPECS: "dict[str, LaneSpec]" = {
    "scifi_news_pro_multipass": LaneSpec(
        module="_otr_scifi_fable2",
        runner_attr="run_scifi_fable2_episode",
        # Unconstrained BY DECLARATION: the lane accepts any target the
        # shared episode budget already accepted (it only refuses < 1, and
        # the budget floor of 30 is strictly tighter).
        compat_attr="",
        compat_error_attrs=(),
    ),
    # 2026-07-19: base scifi_codex (v1) was retired and the v4 lane became
    # the direct scifi_news runner.
    "scifi_news_circuit": LaneSpec(
        module="_otr_scifi_codex",
        runner_attr="run_scifi_codex_episode",
        compat_attr="assert_supported_target_words",
        compat_error_attrs=("CodexTargetRangeError",),
    ),
}

# The pipelines whose execution lane is the writer's own inline body.
INLINE_PIPELINES: frozenset = frozenset({
    "legacy_many_pass",
    "legacy_many_pass_adapt",
    "original_multi_pass",
})


def _load(module_name: str) -> Any:
    """Import a runner module lazily, package-first then flat."""
    try:
        return importlib.import_module(f".{module_name}", __package__)
    except ImportError:  # pragma: no cover - standalone / test load
        return importlib.import_module(module_name)


def _unknown(pipeline_id: str) -> UnknownLanePipelineError:
    return UnknownLanePipelineError(
        f"story pipeline {pipeline_id!r} has no registered execution lane "
        f"(_otr_lane_specs.LANE_SPECS) and is not a known inline lane "
        f"{sorted(INLINE_PIPELINES)}. The pipeline's execution lane is not "
        f"built; there is no fallback. Register the lane in the SAME change "
        f"that flips the bank runnable."
    )


def is_dispatched(pipeline_id: str) -> bool:
    """True iff this pipeline runs through a dedicated runner module."""
    return pipeline_id in LANE_SPECS


def runner_for(pipeline_id: str) -> "Callable[..., Any] | None":
    """The lane runner, or None for a known INLINE lane. Unknown -> RAISE.

    None is a real answer ("the writer's own body runs this"), never a
    fallback: an unregistered pipeline raises instead of quietly running
    the legacy branch under a name that promised something else.
    """
    spec = LANE_SPECS.get(pipeline_id)
    if spec is not None:
        return getattr(_load(spec.module), spec.runner_attr)
    if pipeline_id in INLINE_PIPELINES:
        return None
    raise _unknown(pipeline_id)


def _compat_hook(spec: LaneSpec):
    """The lane's declared preflight predicate, or None if unconstrained."""
    if not spec.compat_attr:
        return None
    return getattr(_load(spec.module), spec.compat_attr)


def assert_supported(bank: Any, req: RollRequest) -> None:
    """Writer entry gate: raise the LANE'S OWN error, unwrapped.

    A direct pick that a lane cannot execute must fail with exactly the
    exception type and message it has always produced -- the gate moves
    WHEN the failure happens (before any source work), never WHAT it is.
    An inline lane is a no-op; an unregistered pipeline raises.
    """
    pipeline_id = str(getattr(bank, "default_story_pipeline", "") or "")
    spec = LANE_SPECS.get(pipeline_id)
    if spec is None:
        if pipeline_id in INLINE_PIPELINES:
            return
        raise _unknown(pipeline_id)
    hook = _compat_hook(spec)
    if hook is not None:
        hook(req.target_words)


def is_roll_compatible(bank: Any, req: RollRequest) -> bool:
    """Randomizer filter: would this lane refuse this request outright?

    Catches ONLY the lane's DECLARED compat errors. Anything else -- an
    ImportError, an AttributeError, a defect inside the preflight --
    propagates, because a randomizer that silently drops a bank whenever its
    module is broken hides the breakage behind a smaller pool.
    """
    pipeline_id = str(getattr(bank, "default_story_pipeline", "") or "")
    spec = LANE_SPECS.get(pipeline_id)
    if spec is None:
        # An inline lane is compatible; an unregistered pipeline is NOT
        # eligible-by-default -- it is a wiring bug, and the roll says so.
        if pipeline_id in INLINE_PIPELINES:
            return True
        return False
    hook = _compat_hook(spec)
    if hook is None:
        return True
    module = _load(spec.module)
    declared = tuple(getattr(module, name) for name in spec.compat_error_attrs)
    if not declared:
        hook(req.target_words)
        return True
    try:
        hook(req.target_words)
    except declared:
        return False
    return True


__all__ = [
    "INLINE_PIPELINES",
    "LANE_SPECS",
    "LaneSpec",
    "RollRequest",
    "UnknownLanePipelineError",
    "assert_supported",
    "is_dispatched",
    "is_roll_compatible",
    "runner_for",
]
