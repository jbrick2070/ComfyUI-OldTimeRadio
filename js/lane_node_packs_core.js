/**
 * Which node packs a loaded workflow needs and does not have. PURE: imports
 * nothing from ComfyUI, so `tests/js/lane_node_packs.test.mjs` runs it with
 * plain `node --test`. The glue that feeds the result to the frontend lives in
 * `lane_node_packs.js`.
 *
 * WHY. A video lane such as `animatediff15_v3_haunted_video` builds its
 * AnimateDiff-Evolved nodes inside OTR's Python, so the saved graph holds no
 * node from that pack and ComfyUI cannot tell the pack is missing. A person
 * only found out by pressing Run (measured 2026-09-27 on a fresh portable).
 * Naming the pack in the frontend's own missing-node list when the workflow
 * opens puts the fix in front of them first, with Node Manager one click away.
 *
 * A lane is SELECTED when its id is a string value of an OTR node's widget --
 * lane ids are unique tokens, so this needs no widget names and survives a
 * widget moving. A dropdown saves a LABEL, "<id> (16:9)", so the id is the text
 * before the first " (" -- the same first step as the backend's
 * `public_engines.resolve_engine_id`. The table lists every menu id that
 * resolves to a lane, so no second mapping step is needed here.
 */

/** The menu id inside a saved dropdown label (`resolve_engine_id` step 1). */
export function laneIdOf(value) {
    return typeof value === "string" ? value.split(" (", 1)[0] : "";
}

/**
 * Entries for the frontend's missing-node list, in the shape its own
 * `collectMissingNodes` pushes: `{type, nodeId, cnrId, isReplaceable}`.
 *
 * @param {object} graphData  the workflow being loaded (litegraph JSON)
 * @param {object|null} registered  LiteGraph.registered_node_types; null or
 *     empty means "cannot tell", which pushes nothing rather than everything
 * @param {object} table  the parsed `lane_node_packs.json`
 * @returns {Array<{type:string,nodeId:string,cnrId:(string|undefined),isReplaceable:boolean}>}
 */
export function missingPackEntries(graphData, registered, table) {
    if (!registered || typeof registered !== "object"
            || Object.keys(registered).length === 0) {
        return [];
    }
    const lanes = (table && table.lanes) || {};
    const packs = (table && table.packs) || {};
    const out = [];
    const seen = new Set();
    for (const node of (graphData && graphData.nodes) || []) {
        if (typeof node?.type !== "string" || !node.type.startsWith("OTR_")) {
            continue;
        }
        // A muted (2) or bypassed (4) node never runs, so its lane is not a
        // selection -- the frontend skips such nodes the same way.
        if (node.mode === 2 || node.mode === 4) continue;
        for (const value of Array.isArray(node.widgets_values) ? node.widgets_values : []) {
            const id = laneIdOf(value);
            const lane = id && Object.prototype.hasOwnProperty.call(lanes, id)
                ? lanes[id] : undefined;
            if (!lane) continue;
            for (const type of lane.node_types || []) {
                if (type in registered || seen.has(type)) continue;
                seen.add(type);
                out.push({
                    type,
                    nodeId: String(node.id),
                    cnrId: packs[lane.pack],
                    isReplaceable: false,
                });
            }
        }
    }
    return out;
}
