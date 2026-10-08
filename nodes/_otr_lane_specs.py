"""nodes/_otr_lane_specs.py -- the ONE story-lane authority.

A source bank's `default_story_pipeline` decides HOW its episode executes:
either a DISPATCHED lane (a dedicated runner module) or this writer's own
INLINE body. This authority lives here, not in `OTR_LedgerScriptWriter`, so
anything that is NOT the writer -- the bank randomizer, a future replay tool,
a checker -- can ask a question about lanes without importing the writer (and
therefore ComfyUI). There is exactly ONE table; no view, no shadow copy.

A lane that genuinely cannot build a requested SHAPE fails loudly when it
tries.

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
class LaneSpec:
    """One dispatched lane, by NAME. Nothing here is imported eagerly."""

    module: str
    """Runner module, relative to this package (e.g. "_otr_scifi_news_pro")."""

    runner_attr: str
    """The lane entry point inside `module`."""


# The dispatched lanes. The key IS the pipeline id from pipelines.json.
# A pipeline absent from BOTH this table and INLINE_PIPELINES raises.
LANE_SPECS: "dict[str, LaneSpec]" = {
    "scifi_news_pro_multipass": LaneSpec(
        module="_otr_scifi_news_pro",
        runner_attr="run_scifi_news_pro_episode",
    ),
    "my_story_multipass": LaneSpec(
        module="_otr_my_story",
        runner_attr="run_my_story_episode",
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


# The bank roll filters on `bank.runnable` alone -- see
# `_otr_rolls.eligible_bank_ids`.


__all__ = [
    "INLINE_PIPELINES",
    "LANE_SPECS",
    "LaneSpec",
    "UnknownLanePipelineError",
    "is_dispatched",
    "runner_for",
]
