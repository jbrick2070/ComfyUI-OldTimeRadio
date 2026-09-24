# -*- coding: utf-8 -*-
r"""The saved-graph schema boundary: shipped, wired, and inert on current graphs.

WHY THE BOUNDARY EXISTS. LiteGraph restores widget values POSITIONALLY -- the nth
saved value goes to the nth widget the node declares today. So a node that drops
a widget shifts every later value up by one on load: no error, no warning, a
graph that looks fine and renders something else. A server-side check cannot
help, because by the time a prompt arrives the wrong values are already in the
widgets. `js/workflow_schema.js` owns the call to `app.loadGraphData` and
reconciles BY NAME first, or refuses and leaves the open canvas alone.

WHAT THIS FILE COVERS, and what it does not. The reconciliation LOGIC is tested
in JavaScript against the real shipped module -- `tests/js/workflow_schema.test.mjs`,
run here through `node --test` so it cannot rot unnoticed in a Python-only suite.
This file additionally proves the two things that test cannot see: that the
extension is actually SHIPPED and REGISTERED, and that it is INERT on every graph
this pack currently ships.

THE INERTNESS CHECK IS THE ONE THAT WOULD CATCH A REAL DISASTER. A boundary that
"migrates" a current graph is worse than no boundary at all, because it rewrites
values nobody asked it to touch. All 25 shipped graphs must come back byte-equal.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_JS = _ROOT / "js"


def _node_exe():
    """The node binary, or None. Absence is a SKIP, never a silent pass."""
    for candidate in ("node", r"C:\Program Files\nodejs\node.exe"):
        found = shutil.which(candidate) or (
            candidate if os.path.isfile(candidate) else None)
        if found:
            return found
    return None


def test_the_extension_ships_and_is_registered():
    """A frontend extension nobody serves is a file, not a feature.

    ComfyUI only serves a pack's web assets when the package exports
    ``WEB_DIRECTORY``. Without that line the reconciliation never loads and every
    other test here would still pass.
    """
    assert (_JS / "workflow_schema.js").is_file()
    assert (_JS / "workflow_schema_core.js").is_file()

    init = (_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert 'WEB_DIRECTORY = "./js"' in init, (
        "__init__.py must export WEB_DIRECTORY or ComfyUI will not serve js/")
    assert '"WEB_DIRECTORY"' in init, "WEB_DIRECTORY should be in __all__"


def test_the_glue_owns_the_loader_and_holds_no_logic():
    """It must intercept ``loadGraphData``, not ``beforeConfigureGraph``.

    The frontend runs extension hooks through ``invokeExtensionsAsync``, which
    CATCHES what a hook throws and merely logs it -- so throwing from
    ``beforeConfigureGraph`` cannot stop a stale graph. Only owning the loader
    call can refuse one. Asserted structurally because the alternative is a live
    browser, and this is the one claim a browser check would not make obvious.
    """
    glue = (_JS / "workflow_schema.js").read_text(encoding="utf-8")
    assert "app.loadGraphData" in glue, "the glue must own the loader call"
    # NOT MERELY ABSENT AS A WORD -- the file explains at length why it does
    # not use that hook, so a bare substring check fails on its own docstring.
    # What must be absent is the hook being DEFINED: `beforeConfigureGraph(` or
    # `beforeConfigureGraph:` as an extension property.
    import re
    registered_hook = re.search(
        r"beforeConfigureGraph\s*[(:]", glue)
    assert not registered_hook, (
        "the extension defines beforeConfigureGraph; its exceptions are "
        "swallowed by invokeExtensionsAsync, so refusal must live at the "
        "loader boundary instead. Found: %r" % (registered_hook.group(0),))
    # On refusal the original loader must NOT run. The early `return` before the
    # forwarding call is the whole mechanism.
    assert "return;" in glue
    assert "original.call(this" in glue, "the original must be forwarded to"
    assert "...rest" in glue, (
        "forward every argument: loadGraphData takes (graphData, clean, "
        "restore_view, workflow, options) and that list may grow")


def test_the_javascript_suite_passes():
    """Run the real JS tests, so they cannot rot inside a Python-only suite."""
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS suite cannot run here")
    proc = subprocess.run(
        [node, "--test", str(_HERE / "js" / "workflow_schema.test.mjs")],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=180)
    assert proc.returncode == 0, (
        "the JS reconciliation suite failed:\n%s\n%s"
        % (proc.stdout[-3000:], proc.stderr[-2000:]))
    assert "# fail 0" in proc.stdout.replace("ℹ", "#") or "fail 0" in proc.stdout


def test_every_shipped_graph_round_trips_unchanged():
    """THE INERTNESS PROOF. A boundary that rewrites a current graph is a bug.

    Runs the real core against the real live schema and every graph this pack
    ships -- the canonical plus all variants. Any of them coming back `changed`
    or `refused` means the boundary would be rewriting or blocking files that are
    correct today, which is strictly worse than not having it.
    """
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS core cannot run here")

    from nodes._otr_workflow_apply import build_offline_schemas
    schemas = build_offline_schemas()
    registry = {t: {"nodeData": {"input": s.get("input", {})}}
                for t, s in schemas.items()}

    graphs = [_ROOT / "workflows" / "otr_canonical.json"]
    graphs += sorted((_ROOT / "workflows" / "variants").glob("otr_*.json"))
    assert len(graphs) >= 20, "expected the canonical plus the variants, got %d" % len(graphs)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = pathlib.Path(tmp) / "registry.json"
        reg_path.write_text(json.dumps(registry), encoding="utf-8")
        script = pathlib.Path(tmp) / "roundtrip.mjs"
        script.write_text(
            'import fs from "node:fs";\n'
            'import { pathToFileURL } from "node:url";\n'
            'const core = await import(pathToFileURL(process.argv[2]).href);\n'
            'const reg = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));\n'
            'const out = [];\n'
            'for (const f of process.argv.slice(4)) {\n'
            '  const g = JSON.parse(fs.readFileSync(f, "utf8"));\n'
            '  const before = JSON.stringify(g);\n'
            '  const r = core.reconcileGraph(g, reg);\n'
            '  if (!r.ok) out.push([f, "refused", r.reason + ": " + r.detail]);\n'
            '  else if (r.changed) out.push([f, "changed", r.notes.join("; ")]);\n'
            '  else if (JSON.stringify(g) !== before) out.push([f, "mutated-input", ""]);\n'
            '}\n'
            'console.log(JSON.stringify(out));\n',
            encoding="utf-8")
        proc = subprocess.run(
            [node, str(script), str(_JS / "workflow_schema_core.js"),
             str(reg_path)] + [str(g) for g in graphs],
            capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr[-2000:]
    problems = json.loads(proc.stdout.strip().splitlines()[-1])
    assert problems == [], (
        "the boundary is not inert on shipped graphs -- it would rewrite or "
        "block files that are correct today:\n%s"
        % "\n".join("  %s: %s %s" % tuple(p) for p in problems))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
