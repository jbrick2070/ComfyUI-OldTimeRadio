"""S16.1: production workflow JSON must have no legacy Director-era
widget names on any nested surface (widget.name, node title, S&R
identifier).

The legacy LLMDirector node + its production_plan_json / director_json
output sockets were retired in voice-path-cleanbreak S2 (2026-05-12);
this guard prevents them from sneaking back.

Sprint 10B Wave 1 Agent D (2026-05-27) introduces a NEW dormant
writers'-room Director agent (OTR_DirectorBrief) per design Section 4
Wave 1; that surface is the writers'-room director (typed
DirectorBrief output, no production_plan_json / director_json
sockets) and is allowed via the ALLOWED_DIRECTOR_SUBSTRINGS list
below. The FORBIDDEN regex still rejects the legacy LLMDirector
patterns + the bare 'Director' / 'director_*' strings that would
indicate the legacy node returning.
"""
from __future__ import annotations

import json
import pathlib
import re


# Strict legacy patterns. The bare 'director' word is intentionally
# kept in the regex so a future re-introduction of the legacy node
# (named just 'Director' / 'OTR_LLMDirector') fails loud. The
# ALLOWED_DIRECTOR_SUBSTRINGS escape hatch below covers the new
# Sprint 10B writers'-room Director agent by substring -- every
# allowed substring is checked AFTER the FORBIDDEN regex match, so
# the regex still catches any future Director surface that is not
# explicitly the writers'-room agent.
FORBIDDEN = re.compile(
    r"director|production_plan_json|director_json", re.IGNORECASE
)

# Sprint 10B Wave 1 Agent D escape hatch. Each entry is a substring;
# if the matched-on string contains any of these, the FORBIDDEN match
# is allowed. The list is intentionally tight: only the exact
# writers'-room Director surface names that Wave 1 ships dormant.
ALLOWED_DIRECTOR_SUBSTRINGS = (
    # justification: Sprint 10B Wave 1 Agent D class name; the
    # ComfyUI node class registration (NODE_CLASS_MAPPINGS key)
    # and properties["Node name for S&R"] both carry this exact
    # string -- it is the new writers'-room Director agent, NOT
    # the deleted legacy LLMDirector node.
    "OTR_DirectorBrief",
    # justification: Sprint 10B Wave 1 Agent D node title; the
    # human-readable workflow title for the dormant Director node
    # ("Director Brief (dormant, Wave 1 Agent D)") -- the parent
    # design (Section 4 Wave 1) frames it as Director Brief.
    "Director Brief",
    # justification: Sprint 10B Wave 1 Agent D output socket name;
    # the typed DirectorBrief dict is emitted on this slot. Wave 2
    # Agent F's Story Room loop wires consumers to this socket.
    "director_brief",
    # justification: v2.0 model-agnostic video platform (Phase 1 wiring,
    # 2026-06-08). OTR_VideoDirector is the per-role A/B/C VIDEO engine
    # selector emitting video_policy_json -- NOT the retired legacy
    # LLMDirector; it carries none of the forbidden production_plan_json /
    # director_json sockets. S&R name + node title carry this class string.
    "OTR_VideoDirector",
    # justification: v2.0 model-agnostic image platform (Phase 1 wiring,
    # 2026-06-08). OTR_ImageDirector is the per-role IMAGE engine + grain
    # selector emitting image_policy_json -- NOT the retired legacy
    # LLMDirector; it carries none of the forbidden production_plan_json /
    # director_json sockets. S&R name + node title carry this class string.
    "OTR_ImageDirector",
)


def _is_allowed_director_surface(value: str) -> bool:
    """Return True iff `value` matches the new writers'-room
    Director surface (Sprint 10B Wave 1 Agent D) rather than the
    retired legacy LLMDirector. Substring-based check; keep the
    list tight so a stray 'Director' anywhere else still fails."""
    if not value:
        return False
    return any(s in value for s in ALLOWED_DIRECTOR_SUBSTRINGS)

WORKFLOW_PATHS = (
    pathlib.Path("workflows/otr_canonical.json"),
)


def test_no_director_surfaces_in_production_workflow():
    """Scan node.title, nested widget.name, and properties['Node name
    for S&R'] for legacy Director-era substrings. Any hit that is not
    on the ALLOWED_DIRECTOR_SUBSTRINGS escape hatch fails.

    The escape hatch covers Sprint 10B Wave 1 Agent D's new
    writers'-room Director agent (OTR_DirectorBrief) -- the new
    surface is NOT the retired legacy LLMDirector; it carries the
    typed DirectorBrief output, not the deleted production_plan_json
    / director_json sockets.
    """
    for wf_path in WORKFLOW_PATHS:
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        for node in wf.get("nodes") or []:
            title = node.get("title") or ""
            if FORBIDDEN.search(title) and not _is_allowed_director_surface(title):
                raise AssertionError(
                    f"{wf_path}: node id={node.get('id')} title "
                    f"{title!r} carries a legacy Director-era substring."
                )
            for inp in node.get("inputs") or []:
                if not isinstance(inp, dict):
                    continue
                widget = inp.get("widget") or {}
                if not isinstance(widget, dict):
                    continue
                w_name = widget.get("name") or ""
                if FORBIDDEN.search(w_name) and not _is_allowed_director_surface(w_name):
                    raise AssertionError(
                        f"{wf_path}: node id={node.get('id')} input "
                        f"{inp.get('name')!r} has widget.name={w_name!r} "
                        f"with a legacy Director-era substring."
                    )
            sr_name = (node.get("properties") or {}).get(
                "Node name for S&R"
            ) or ""
            if FORBIDDEN.search(sr_name) and not _is_allowed_director_surface(sr_name):
                raise AssertionError(
                    f"{wf_path}: node id={node.get('id')} S&R name "
                    f"{sr_name!r} carries a legacy Director-era substring."
                )
