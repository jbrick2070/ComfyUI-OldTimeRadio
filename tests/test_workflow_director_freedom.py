"""S16.1: production workflow JSON must have no Director-era widget
names on any nested surface (widget.name, node title, S&R identifier).
"""
from __future__ import annotations

import json
import pathlib
import re


FORBIDDEN = re.compile(
    r"director|production_plan_json|director_json", re.IGNORECASE
)

WORKFLOW_PATHS = (
    pathlib.Path("workflows/otr_scifi_16gb_full.json"),
)


def test_no_director_surfaces_in_production_workflow():
    """Scan node.title, nested widget.name, and properties['Node name
    for S&R'] for Director-era substrings. Any hit fails."""
    for wf_path in WORKFLOW_PATHS:
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        for node in wf.get("nodes") or []:
            title = node.get("title") or ""
            assert not FORBIDDEN.search(title), (
                f"{wf_path}: node id={node.get('id')} title "
                f"{title!r} carries a Director-era substring."
            )
            for inp in node.get("inputs") or []:
                if not isinstance(inp, dict):
                    continue
                widget = inp.get("widget") or {}
                if not isinstance(widget, dict):
                    continue
                w_name = widget.get("name") or ""
                assert not FORBIDDEN.search(w_name), (
                    f"{wf_path}: node id={node.get('id')} input "
                    f"{inp.get('name')!r} has widget.name={w_name!r} "
                    f"with a Director-era substring."
                )
            sr_name = (node.get("properties") or {}).get(
                "Node name for S&R"
            ) or ""
            assert not FORBIDDEN.search(sr_name), (
                f"{wf_path}: node id={node.get('id')} S&R name "
                f"{sr_name!r} carries a Director-era substring."
            )
