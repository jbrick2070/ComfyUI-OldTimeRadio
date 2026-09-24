/**
 * Tests for js/workflow_schema_core.js -- the saved-graph reconciliation logic.
 *
 * THESE RUN THE REAL SHIPPED MODULE, not a re-implementation. The core is split
 * out from the frontend glue precisely so it can be imported by a plain
 * `node --test` run with no ComfyUI present: a test that mimicked the algorithm
 * would prove the mimicry correct and leave the shipped file untested, which is
 * a failure mode this repo has already recorded.
 *
 * The glue in workflow_schema.js holds no decisions -- it wraps
 * app.loadGraphData, forwards to the core, and either calls the original loader
 * or does not. That wiring is what the browser check proves; everything below is
 * the logic.
 *
 * Run: node --test tests/js/workflow_schema.test.mjs
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ""));
const CORE = pathToFileURL(
    path.join(here, "..", "..", "js", "workflow_schema_core.js")).href;
const { reconcileGraph } = await import(CORE);

/** The live-schema lookup the core reads, shaped like LiteGraph's registry. */
const registry = {};
function reconcile(graph) { return reconcileGraph(graph, registry); }

/** Declare a node type's live widget order, as /object_info would. */
function declare(type, widgetNames, extra = {}) {
    const required = {};
    for (const n of widgetNames) required[n] = ["STRING", {}];
    Object.assign(required, extra);
    registry[type] = { nodeData: { input: { required } } };
}

/** A saved node: widget descriptors interleaved with link sockets, as on disk. */
function savedNode(id, type, pairs, sockets = []) {
    const inputs = [];
    for (const [name] of pairs) inputs.push({ name, widget: { name }, link: null });
    for (const s of sockets) inputs.push(s);
    return { id, type, inputs, widgets_values: pairs.map(([, v]) => v) };
}

test("a modern graph round-trips completely unchanged", () => {
    declare("W", ["a", "b", "c"]);
    const graph = { nodes: [savedNode(1, "W", [["a", "A"], ["b", "B"], ["c", "C"]])], links: [] };
    const before = structuredClone(graph);
    const out = reconcile(graph);
    assert.equal(out.ok, true);
    assert.equal(out.changed, false, "an unchanged graph must not be rewritten");
    assert.deepEqual(graph, before, "the input must not be mutated");
});

test("a graph lacking any migration marker is NOT refused", () => {
    declare("W", ["a"]);
    const out = reconcile({ nodes: [savedNode(1, "W", [["a", "A"]])], links: [] });
    assert.equal(out.ok, true);
    assert.equal(out.changed, false);
});

test("a removed widget is dropped and the SURVIVING values keep their names", () => {
    // The file has four; this build declares three. 'gone' is removed.
    declare("W", ["first", "second", "fourth"]);
    const graph = {
        nodes: [savedNode(1, "W", [
            ["first", "MARKER_1"], ["second", "MARKER_2"],
            ["gone", "MARKER_REMOVED"], ["fourth", "MARKER_4"]])],
        links: [],
    };
    const out = reconcile(graph);
    assert.equal(out.ok, true);
    assert.equal(out.changed, true);
    const n = out.graph.nodes[0];
    // DISTINCT MARKERS, so a positional shift is visible rather than plausible:
    // a naive splice would leave MARKER_4 sitting in 'gone''s old slot.
    assert.deepEqual(n.widgets_values, ["MARKER_1", "MARKER_2", "MARKER_4"]);
    assert.deepEqual(n.inputs.map((s) => s.widget.name), ["first", "second", "fourth"]);
    assert.equal(graph.nodes[0].widgets_values.length, 4, "original untouched");
});

test("values are rebuilt in THIS build's order, not the file's", () => {
    declare("W", ["c", "a"]);          // live order differs from saved order
    const out = reconcile({
        nodes: [savedNode(1, "W", [["a", "A"], ["gone", "X"], ["c", "C"]])],
        links: [],
    });
    assert.equal(out.ok, true);
    assert.deepEqual(out.graph.nodes[0].widgets_values, ["C", "A"]);
});

test("non-widget sockets survive, and link slots are repaired BY IDENTITY", () => {
    declare("W", ["keep_a", "keep_b"]);
    // 'gone' sits BEFORE the socket, so removing it shifts the socket's index.
    const node = savedNode(1, "W",
        [["keep_a", "A"], ["gone", "X"], ["keep_b", "B"]],
        [{ name: "gate_in", link: 279 }]);
    const graph = { nodes: [node], links: [[279, 9, 0, 1, 3, "STRING"]] };
    const out = reconcile(graph);
    assert.equal(out.ok, true);
    const n = out.graph.nodes[0];
    const socketIdx = n.inputs.findIndex((s) => s.link === 279);
    assert.equal(socketIdx, 2, "socket should follow the two surviving widgets");
    assert.equal(out.graph.links[0][4], socketIdx,
        "dst_slot must be recomputed from inputs[i].link, not by subtracting");
});

test("REFUSES when a link is wired into a removed widget", () => {
    declare("W", ["kept"]);
    const node = savedNode(1, "W", [["kept", "K"], ["gone", "X"]]);
    node.inputs[1].link = 42;          // the removed widget is connected
    const graph = { nodes: [node], links: [[42, 9, 0, 1, 1, "STRING"]] };
    const before = structuredClone(graph);
    const out = reconcile(graph);
    assert.equal(out.ok, false, "an ambiguous link must refuse, not silently drop");
    assert.match(out.reason, /link/i);
    assert.deepEqual(graph, before, "the caller's graph must be untouched on refusal");
});

test("REFUSES a short value vector rather than guessing the alignment", () => {
    declare("W", ["a", "b"]);
    const node = savedNode(1, "W", [["a", "A"], ["b", "B"]]);
    node.widgets_values = ["A"];       // one value, two names
    const out = reconcile({ nodes: [node], links: [] });
    assert.equal(out.ok, false);
    assert.match(out.reason, /count/i);
});

test("REFUSES duplicate widget names", () => {
    declare("W", ["a"]);
    const node = savedNode(1, "W", [["a", "1"], ["a", "2"]]);
    const out = reconcile({ nodes: [node], links: [] });
    assert.equal(out.ok, false);
    assert.match(out.reason, /duplicate/i);
});

test("migration is IDEMPOTENT -- loading the result again changes nothing", () => {
    declare("W", ["first", "fourth"]);
    const once = reconcile({
        nodes: [savedNode(1, "W", [["first", "A"], ["gone", "X"], ["fourth", "D"]])],
        links: [],
    });
    assert.equal(once.changed, true);
    const twice = reconcile(once.graph);
    assert.equal(twice.ok, true);
    assert.equal(twice.changed, false, "a second pass must be a no-op");
    assert.deepEqual(twice.graph.nodes[0].widgets_values, ["A", "D"]);
});

test("an unknown node type is left alone, not refused", () => {
    // A graph may contain nodes from a pack that is not installed. That is the
    // loader's business to report; pre-empting it would block a legitimate file.
    const out = reconcile({
        nodes: [savedNode(1, "SomeoneElsesNode", [["x", "1"]])], links: [],
    });
    assert.equal(out.ok, true);
    assert.equal(out.changed, false);
});

test("a node with no saved widget names is left alone", () => {
    declare("W", ["a"]);
    const out = reconcile({
        nodes: [{ id: 1, type: "W", inputs: [], widgets_values: ["A"] }], links: [],
    });
    assert.equal(out.ok, true, "no names to reconcile by: leave it, do not guess");
    assert.equal(out.changed, false);
});

test("a null or empty graph is handled without throwing", () => {
    assert.equal(reconcile(null).ok, true);
    assert.equal(reconcile({}).ok, true);
    assert.equal(reconcile({ nodes: [] }).ok, true);
});

// --- cases a review pass found the suite could not distinguish ---------------

test("REFUSES an OVER-count too, not only an under-count", () => {
    // Mutation-proven gap: with only the under-count case covered, changing the
    // check from `!==` to `>` left all twelve tests green while the boundary
    // silently accepted a node with MORE values than names -- the shape a
    // seed-style companion slot produces.
    declare("W", ["a", "b"]);
    const node = savedNode(1, "W", [["a", "A"], ["b", "B"]]);
    node.widgets_values = ["A", "B", "EXTRA"];
    const out = reconcile({ nodes: [node], links: [] });
    assert.equal(out.ok, false, "more values than names must refuse");
    assert.match(out.reason, /count/i);
});

test("REFUSES a node with values, no names, and a count this build cannot fit", () => {
    // ComfyUI's own core nodes serialise with ZERO widget descriptors -- the
    // shipped templates save KSampler with seven values and no names at all. An
    // earlier cut returned "unchanged" for that shape, so a count mismatch
    // passed straight through: the exact positional corruption this boundary
    // exists to stop, going through the middle of it.
    declare("W", ["only_one"]);
    const out = reconcile({
        nodes: [{ id: 1, type: "W", widgets_values: ["A", "B"] }], links: [],
    });
    assert.equal(out.ok, false);
    assert.match(out.reason, /no widget names/i);
});

test("a node with values, no names, and a MATCHING count passes through", () => {
    // The other half of the same rule: when the count still fits, positional
    // restore is correct and the node must not be touched or blocked.
    declare("W", ["a", "b"]);
    const graph = {
        nodes: [{ id: 1, type: "W", widgets_values: ["A", "B"] }], links: [],
    };
    const before = structuredClone(graph);
    const out = reconcile(graph);
    assert.equal(out.ok, true);
    assert.equal(out.changed, false);
    assert.deepEqual(graph, before);
});

test("widgets_values_named loses the dropped key, so the two agree", () => {
    // The frontend writes this map on every save and PREFERS it when
    // LiteGraph.namedValuesRestore is on. Leaving a stale entry would make the
    // named and positional forms disagree, and the one left unfixed would win
    // the moment that setting flipped.
    declare("W", ["kept"]);
    const node = savedNode(1, "W", [["kept", "K"], ["gone", "X"]]);
    node.widgets_values_named = { kept: "K", gone: "X" };
    const out = reconcile({ nodes: [node], links: [] });
    assert.equal(out.ok, true);
    assert.equal(out.changed, true);
    assert.deepEqual(out.graph.nodes[0].widgets_values_named, { kept: "K" });
});

test("REFUSES duplicate node ids rather than repairing against the wrong node", () => {
    // LiteGraph never emits these, so it takes a hand-edited file -- but the id
    // map would collapse last-write-wins and a link would be repaired against
    // the wrong node's inputs, which is silently wrong rather than merely odd.
    declare("W", ["kept"]);
    const a = savedNode(7, "W", [["kept", "A"], ["gone", "X"]]);
    const b = savedNode(7, "W", [["kept", "B"], ["gone", "Y"]]);
    const out = reconcile({ nodes: [a, b], links: [[1, 9, 0, 7, 0, "STRING"]] });
    assert.equal(out.ok, false);
    assert.match(out.reason, /duplicate node id/i);
});
