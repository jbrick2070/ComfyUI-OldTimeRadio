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
import re
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
    # NOT A SUBSTRING CHECK ON THE RAW FILE -- the glue explains at length why
    # it does not use that hook, so its own comment would trip one. And not a
    # regex for `name(` or `name:` either: a QA pass defeated that with an
    # ordinary quoted key, `"beforeConfigureGraph": function(g){}`, and again
    # with a computed one. Strip the comments, then require the identifier to be
    # absent from the CODE, which no legitimate spelling can evade.
    code = re.sub(r"/\*.*?\*/", "", glue, flags=re.S)       # block comments
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)        # line comments
    assert "beforeConfigureGraph" not in code, (
        "the extension references beforeConfigureGraph in code; exceptions "
        "from that hook are swallowed by invokeExtensionsAsync, so a refusal "
        "must live at the loader boundary instead")

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



# The reconciliation is run in node, against the real shipped core, and the
# result is handed back as JSON so the assertions live in Python where a failure
# prints something a reader can act on. Written with real newlines rather than
# escapes: a JS source string full of backslash-n inside a Python literal is how
# this file got mangled once already.
_MIGRATE_MJS = """import fs from "node:fs";
import { pathToFileURL } from "node:url";
const core = await import(pathToFileURL(process.argv[2]).href);
const reg = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const old = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const first = core.reconcileGraph(old, reg);
if (!first.ok) {
  console.log(JSON.stringify({ok: false, why: first.reason + ": " + first.detail}));
} else {
  const again = core.reconcileGraph(first.graph, reg);
  console.log(JSON.stringify({ok: true, changed: first.changed, notes: first.notes,
                              graph: first.graph, changedAgain: again.changed}));
}
"""


def test_a_real_pre_removal_graph_migrates_to_exactly_what_we_ship_today():
    """THE MIGRATION PROOF, on a real saved graph instead of a constructed one.

    The JS suite proves the algorithm on small hand-built nodes; the inertness
    test proves the boundary leaves today's graphs alone. Neither covers the case
    the boundary was BUILT for -- a real file somebody saved before a widget was
    removed, opened in the build that removed it.

    The fixture is the canonical graph as it stood at the commit before the
    writer's two GGUF widgets came out, captured with ``git show`` rather than
    authored. Migrating it must land on EXACTLY the values ``build_variants.py``
    emits today, because that is what a correct migration means: the user's saved
    graph ends up where a freshly generated one already is.
    """
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS core cannot run here")

    fixture = _HERE / "fixtures" / "pre_gguf_removal_canonical.json"
    saved = json.loads(fixture.read_text(encoding="utf-8"))
    writer = next(n for n in saved["nodes"]
                  if n.get("type") == "OTR_LedgerScriptWriter")
    stale = [s["widget"]["name"] for s in writer.get("inputs", [])
             if s.get("widget", {}).get("name", "").startswith("gguf")]

    # THE FIXTURE GUARD. Without it, re-capturing this file from the CURRENT
    # canonical would leave a test that migrates nothing and passes anyway --
    # green, meaningless, and indistinguishable from the real thing.
    assert stale, (
        "the fixture no longer carries the widgets it was captured for, so this "
        "test would pass without migrating anything. Re-capture it from the "
        "commit before the removal; never regenerate it from the canonical.")

    from nodes._otr_workflow_apply import build_offline_schemas
    schemas = build_offline_schemas()
    spec = schemas["OTR_LedgerScriptWriter"]["input"]
    declared = set(spec.get("required", {})) | set(spec.get("optional", {}))
    assert not declared & set(stale), (
        "this build still declares %s, so the fixture is not stale relative to "
        "it and nothing would be migrated" % sorted(declared & set(stale)))

    registry = {t: {"nodeData": {"input": s.get("input", {})}}
                for t, s in schemas.items()}

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = pathlib.Path(tmp) / "registry.json"
        reg_path.write_text(json.dumps(registry), encoding="utf-8")
        script = pathlib.Path(tmp) / "migrate.mjs"
        script.write_text(_MIGRATE_MJS, encoding="utf-8")
        proc = subprocess.run(
            [node, str(script), str(_JS / "workflow_schema_core.js"),
             str(reg_path), str(fixture)],
            capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ok"], "a real older graph was refused: %s" % out.get("why")
    assert out["changed"], (
        "the older graph should have been migrated, not passed through")
    assert out["changedAgain"] is False, (
        "migration is not idempotent -- loading the migrated graph again "
        "changed it")

    migrated = out["graph"]
    mig_writer = next(n for n in migrated["nodes"]
                      if n.get("type") == "OTR_LedgerScriptWriter")

    survivors = {s["widget"]["name"] for s in mig_writer.get("inputs", [])
                 if s.get("widget", {}).get("name")}
    assert not survivors & set(stale), (
        "a widget this build removed survived the migration: %s"
        % sorted(survivors & set(stale)))

    current = json.loads((_ROOT / "workflows" / "otr_canonical.json")
                         .read_text(encoding="utf-8"))
    cur_writer = next(n for n in current["nodes"]
                      if n.get("type") == "OTR_LedgerScriptWriter")
    assert mig_writer["widgets_values"] == cur_writer["widgets_values"], (
        "a migrated older graph must land on the same values a freshly "
        "generated one carries.\n  migrated : %r\n  canonical: %r"
        % (mig_writer["widgets_values"], cur_writer["widgets_values"]))

    # EVERY link is checked, not only the one that moved. ``dst_slot`` indexes
    # the node's ``inputs`` array, so a descriptor removed anywhere shifts every
    # later socket. The invariant proving the repair is identity-based rather
    # than arithmetic is simply: the slot a link points at is the slot holding
    # it.
    by_id = {n["id"]: n for n in migrated["nodes"]}
    for link in migrated["links"]:
        link_id, _src, _src_slot, dst_id, dst_slot = link[:5]
        dst = by_id.get(dst_id)
        if dst is None:
            continue
        slots = dst.get("inputs") or []
        assert 0 <= dst_slot < len(slots), (
            "link %s points past the end of %s's inputs (%s of %s)"
            % (link_id, dst.get("type"), dst_slot, len(slots)))
        assert slots[dst_slot].get("link") == link_id, (
            "link %s lands on %s slot %s, which holds link %r -- the "
            "destination slots were not repaired by identity"
            % (link_id, dst.get("type"), dst_slot, slots[dst_slot].get("link")))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
