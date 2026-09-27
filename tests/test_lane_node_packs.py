"""`js/lane_node_packs.json` is pinned to the engines it describes.

The frontend names a missing node pack when a workflow loads (2026-09-27): a
lane such as AnimateDiff builds its nodes in Python, so the saved graph alone
cannot tell ComfyUI the pack is missing. The JS reads a small table; this file
proves the table says exactly what the engines' own `_node_candidates` and
`wrapper_bridge._PACK_FOR_PREFIX` say, so the two can never drift. It also runs
the JS suite for the pure core, like tests/test_workflow_schema_boundary.py.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TABLE = json.loads((_ROOT / "js" / "lane_node_packs.json").read_text(encoding="utf-8"))


def _expected_table():
    """Menu id -> (pack name, node types) derived from the engines themselves."""
    import nodes._otr_video_engines  # noqa: F401 -- registers every engine
    from nodes._otr_shared import public_engines as PE
    from nodes._otr_video_engines import registry as R
    from nodes._otr_video_engines import wrapper_bridge as WB

    lanes = {}
    for internal in R.all_engine_names():
        table = getattr(R.get_engine(internal), "_node_candidates", None)
        if not callable(table):
            continue
        needed = {}
        for names in dict(table()).values():
            for name in names:
                for prefix, pack, _url in WB._PACK_FOR_PREFIX:
                    if name.startswith(prefix):
                        needed.setdefault(pack, set()).add(name)
        assert len(needed) <= 1, (internal, needed)   # one pack per lane today
        for pack, types in needed.items():
            menu_ids = {internal} | {pub for pub, inner in PE._PUBLIC_ENGINES.items()
                                     if inner == internal}
            for menu_id in menu_ids:
                lanes[menu_id] = (pack, sorted(types))
    return lanes


def test_the_table_matches_every_engine_that_needs_another_pack():
    expected = _expected_table()
    assert expected, "no engine needs another pack -- then this table should go"
    got = {lane: (row["pack"], sorted(row["node_types"]))
           for lane, row in _TABLE["lanes"].items()}
    assert got == expected


def test_every_pack_in_the_table_has_a_registry_id():
    """The id the frontend's card routes to in Node Manager. Measured
    2026-09-27: Manager installs AnimateDiff-Evolved as
    `comfyui-animatediff-evolved`."""
    for row in _TABLE["lanes"].values():
        assert _TABLE["packs"].get(row["pack"]), row["pack"]
    assert _TABLE["packs"]["ComfyUI-AnimateDiff-Evolved"] == "comfyui-animatediff-evolved"


def test_the_glue_feeds_the_frontends_missing_node_list():
    """The hint is WIRED, not just written: the extension implements the hook
    the frontend awaits before it scans the graph, and pushes into its list."""
    glue = (_ROOT / "js" / "lane_node_packs.js").read_text(encoding="utf-8")
    assert "app.registerExtension(" in glue
    assert 'from "./lane_node_packs_core.js"' in glue
    assert "async beforeConfigureGraph(graphData, missingNodeTypes)" in glue
    assert "missingNodeTypes.push(entry)" in glue
    assert 'new URL("./lane_node_packs.json", import.meta.url)' in glue


def _node_exe():
    for candidate in ("node", r"C:\Program Files\nodejs\node.exe"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=30,
                           check=True)
            return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def test_the_javascript_suite_passes():
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS suite cannot run here")
    proc = subprocess.run(
        [node, "--test", str(_ROOT / "tests" / "js" / "lane_node_packs.test.mjs")],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=180)
    assert proc.returncode == 0, "%s\n%s" % (proc.stdout[-3000:], proc.stderr[-2000:])
