/**
 * Frontend glue for the missing-pack hint (2026-09-27). GLUE ONLY -- the
 * decision lives in `lane_node_packs_core.js`, covered by
 * `tests/js/lane_node_packs.test.mjs`.
 *
 * The frontend hands every extension its missing-node list in
 * `beforeConfigureGraph`, BEFORE it scans the graph's own nodes, and shows
 * whatever that list holds in its standard missing-node card -- App Mode
 * included -- with the route to the pack in Node Manager. A video lane like
 * AnimateDiff builds its nodes in Python, so the graph alone cannot say the
 * pack is missing; this names it when the workflow opens instead of at Run.
 *
 * ADVISORY, and deliberately a separate extension from `workflow_schema.js`:
 * that file REFUSES stale graphs and must own `loadGraphData`, because
 * `invokeExtensionsAsync` swallows whatever a hook throws. A hint needs no
 * such power -- if this hook fails, the workflow loads exactly as before and
 * the queue-time gate still names the pack.
 */
import { app } from "../../scripts/app.js";
import { missingPackEntries } from "./lane_node_packs_core.js";

const TAG = "[OldTimeRadio] node packs";

/** The lane -> node pack table, fetched once. A failed fetch is logged and
 *  reads as an empty table: a hint is never a reason to block a load. */
let lanePacksTable = null;
function loadLanePacksTable() {
    if (!lanePacksTable) {
        lanePacksTable = fetch(new URL("./lane_node_packs.json", import.meta.url))
            .then((r) => (r.ok ? r.json() : {}))
            .catch((err) => {
                console.warn(`${TAG}: lane node-pack table unavailable`, err);
                return {};
            });
    }
    return lanePacksTable;
}

app.registerExtension({
    name: "OldTimeRadio.NodePackHints",
    async beforeConfigureGraph(graphData, missingNodeTypes) {
        try {
            if (!Array.isArray(missingNodeTypes)) return;
            const entries = missingPackEntries(
                graphData, globalThis.LiteGraph?.registered_node_types,
                await loadLanePacksTable());
            const already = new Set(missingNodeTypes.map((m) =>
                (typeof m === "string" ? m : m?.type)));
            for (const entry of entries) {
                if (!already.has(entry.type)) missingNodeTypes.push(entry);
            }
            if (entries.length) {
                console.log(`${TAG}: this workflow's video lane needs ` +
                            [...new Set(entries.map((e) => e.cnrId || e.type))].join(", "));
            }
        } catch (err) {
            console.warn(`${TAG}: missing-pack hint skipped`, err);
        }
    },
});
