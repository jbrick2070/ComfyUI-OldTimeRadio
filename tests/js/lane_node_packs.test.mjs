/**
 * Tests for js/lane_node_packs_core.js against the REAL shipped table and the
 * REAL shipped workflows -- not fixtures written to agree with the code.
 *
 * Run: node --test tests/js/lane_node_packs.test.mjs
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ""));
const ROOT = path.join(here, "..", "..");
const { missingPackEntries } = await import(
    pathToFileURL(path.join(ROOT, "js", "lane_node_packs_core.js")).href);
const TABLE = JSON.parse(readFileSync(path.join(ROOT, "js", "lane_node_packs.json"), "utf8"));
const workflow = (name) => JSON.parse(
    readFileSync(path.join(ROOT, "workflows", `${name}.json`), "utf8"));

/** A ComfyUI with core nodes and OTR registered, and no AnimateDiff-Evolved. */
const withoutAde = { KSampler: {}, OTR_VideoDirector: {}, OTR_LedgerScriptWriter: {} };
const withAde = { ...withoutAde, ADE_AnimateDiffLoaderGen1: {},
                  ADE_StandardStaticContextOptions: {} };

test("every AnimateDiff workflow names the pack when it is missing", () => {
    for (const name of ["otr_8gb_animatediff", "otr_16gb_animatediff",
                        "otr_mac16_animatediff"]) {
        const entries = missingPackEntries(workflow(name), withoutAde, TABLE);
        assert.deepEqual(entries.map((e) => e.type).sort(),
                         ["ADE_AnimateDiffLoaderGen1", "ADE_StandardStaticContextOptions"],
                         name);
        for (const e of entries) {
            assert.equal(e.cnrId, "comfyui-animatediff-evolved", name);
            assert.equal(e.isReplaceable, false);
            // The node the hint points at is the OTR node that picked the lane.
            const node = workflow(name).nodes.find((n) => String(n.id) === e.nodeId);
            assert.ok(node && node.type.startsWith("OTR_"), name);
        }
    }
});

test("nothing is pushed once the pack is installed", () => {
    assert.deepEqual(missingPackEntries(workflow("otr_8gb_animatediff"), withAde, TABLE), []);
});

test("a workflow on no AnimateDiff lane gets no hint", () => {
    for (const name of ["otr_8gb_low", "otr_8gb_still", "otr_canonical", "otr_app"]) {
        assert.deepEqual(missingPackEntries(workflow(name), withoutAde, TABLE), [], name);
    }
});

test("an unknown registry pushes nothing rather than everything", () => {
    const graph = workflow("otr_8gb_animatediff");
    assert.deepEqual(missingPackEntries(graph, null, TABLE), []);
    assert.deepEqual(missingPackEntries(graph, {}, TABLE), []);
});

test("a lane picked on the canonical is found wherever its widget sits", () => {
    const graph = workflow("otr_canonical");
    const director = graph.nodes.find((n) => n.type === "OTR_VideoDirector");
    director.widgets_values = [...director.widgets_values];
    director.widgets_values[0] = "animatediff15_lightning_video (16:9)";
    const entries = missingPackEntries(graph, withoutAde, TABLE);
    assert.equal(entries.length, 2);
    assert.ok(entries.every((e) => e.nodeId === String(director.id)));
});

test("a lane id in a non-OTR node is not a selection", () => {
    const graph = { nodes: [{ id: 5, type: "Note",
                              widgets_values: ["animatediff15_v3_haunted_video"] }] };
    assert.deepEqual(missingPackEntries(graph, withoutAde, TABLE), []);
});
