/**
 * Reconcile a saved graph against the node schema this build declares.
 *
 * PURE. No ComfyUI imports, no DOM, no globals except `LiteGraph` read through
 * an injectable lookup. That is deliberate: this half holds every decision worth
 * testing, so it must be importable by a plain `node --test` run. The frontend
 * glue that wraps `app.loadGraphData` lives in `workflow_schema.js` and holds no
 * logic of its own.
 *
 * WHY ANY OF THIS EXISTS. LiteGraph restores widget values POSITIONALLY: the nth
 * saved value goes to the nth widget the node declares today. So when a node
 * drops a widget, every later value shifts up by one -- silently, with no error,
 * producing a graph that looks loaded and renders something else. A server-side
 * check cannot help; by then the wrong values are already in the widgets.
 */

/** Refusals carry the reason a human needs, not a stack trace. */
function refuse(reason, detail) {
    return { ok: false, reason, detail };
}

/**
 * The widget names this build declares for a node type, in saved order.
 *
 * `lookup` is injected so tests can declare a schema without a frontend. In the
 * browser the caller passes the real `LiteGraph.registered_node_types`.
 *
 * Returns null for an unknown type: a graph may legitimately contain nodes from
 * a pack that is not installed, and that is the loader's business to report, not
 * ours to pre-empt.
 */
export function liveWidgetNames(nodeType, lookup) {
    const input = lookup?.[nodeType]?.nodeData?.input;
    if (!input) return null;
    const names = [];
    for (const group of ["required", "optional"]) {
        const entries = input[group];
        if (!entries) continue;
        for (const [name, spec] of Object.entries(entries)) {
            const opts = Array.isArray(spec) && spec.length > 1 ? spec[1] : null;
            // A forceInput field is a socket, never a saved widget value.
            if (opts && opts.forceInput) continue;
            const typeDef = Array.isArray(spec) ? spec[0] : spec;
            // A COMBO is declared as its list of choices rather than a type name.
            const isWidget = Array.isArray(typeDef)
                || ["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"].includes(typeDef);
            if (!isWidget) continue;
            names.push(name);
        }
    }
    return names;
}

/**
 * The widget names a SAVED node carries, in its own saved order.
 *
 * Read from the file rather than derived from a schema: the entire point is to
 * learn what the file thinks it has, which is what may differ from what we
 * declare today.
 */
export function savedWidgetNames(node) {
    const out = [];
    for (const slot of node.inputs || []) {
        if (slot && slot.widget && typeof slot.widget.name === "string") {
            out.push(slot.widget.name);
        }
    }
    return out;
}

/** Reconcile ONE node in place. Caller works on a copy. */
function reconcileNode(node, lookup) {
    const live = liveWidgetNames(node.type, lookup);
    if (live === null) return { ok: true, changed: false };

    const saved = savedWidgetNames(node);
    const values = node.widgets_values;

    if (!Array.isArray(values)) return { ok: true, changed: false };

    // A NODE CAN CARRY VALUES AND NO NAMES, AND THAT SHAPE IS NOT RARE.
    // ComfyUI's own core nodes serialise this way -- the shipped workflow
    // templates save KSampler with all seven widget values and ZERO widget
    // descriptors in `inputs`. An earlier cut of this file returned "unchanged"
    // for that shape, which meant a count mismatch passed through silently:
    // exactly the positional corruption this boundary exists to stop, sailing
    // through the middle of it. Found by a review pass that read the installed
    // frontend's own templates rather than only this pack's graphs.
    //
    // There is nothing to reconcile BY here -- inferring which value is which
    // from a bare list is the guess we refuse to make. But we CAN tell whether
    // the file still fits: if the count disagrees with what this build
    // declares, positional restore is guaranteed to put values in the wrong
    // widgets, so refuse. If it agrees, positional restore is correct and the
    // node passes through untouched.
    if (saved.length === 0) {
        if (values.length !== live.length) {
            return refuse(
                "a node carries values but no widget names, and the count no " +
                "longer matches this build",
                `${node.type}: ${values.length} value(s) saved, ` +
                `${live.length} widget(s) declared. Without names there is ` +
                `nothing to reconcile by, and restoring positionally would put ` +
                `values in the wrong widgets.`);
        }
        return { ok: true, changed: false };
    }

    // THE 1:1 MAPPING IS CHECKED, NEVER ASSUMED. Every node in this pack's
    // canonical graph maps widget descriptors to saved values one for one
    // (measured across all 21). A node carrying a seed-style companion value
    // would not, and mapping it positionally would corrupt it -- so that refuses
    // rather than guesses.
    if (saved.length !== values.length) {
        return refuse(
            "value count does not match the saved widget names",
            `${node.type}: ${saved.length} saved widget name(s) but ` +
            `${values.length} value(s). This node cannot be reconciled by name.`);
    }

    const seen = new Set();
    for (const name of saved) {
        if (seen.has(name)) {
            return refuse("duplicate widget name in the saved graph",
                          `${node.type}: '${name}' appears twice`);
        }
        seen.add(name);
    }

    const byName = new Map();
    saved.forEach((name, i) => byName.set(name, values[i]));

    const liveSet = new Set(live);
    const dropped = saved.filter((n) => !liveSet.has(n));
    if (dropped.length === 0) {
        // Nothing this build lacks. An unchanged modern graph round-trips
        // untouched: it is NOT refused for missing a migration marker, and it is
        // not rewritten either.
        return { ok: true, changed: false };
    }

    // A LINK INTO A REMOVED DESCRIPTOR IS AMBIGUOUS, SO IT REFUSES. Silently
    // deleting a connection the user drew is worse than declining the file --
    // they can still open it in a build that has the widget and see what it fed.
    for (const slot of node.inputs || []) {
        const nm = slot?.widget?.name;
        if (nm && dropped.includes(nm)
            && slot.link !== null && slot.link !== undefined) {
            return refuse("a link is connected to a widget this build removed",
                          `${node.type}: '${nm}' is wired (link ${slot.link})`);
        }
    }

    node.inputs = (node.inputs || []).filter(
        (slot) => !(slot?.widget?.name && dropped.includes(slot.widget.name)));
    // Rebuilt in the order THIS build declares. A widget we declare that the
    // file does not carry is left out, for the node's own default to fill.
    node.widgets_values = live
        .filter((name) => byName.has(name))
        .map((name) => byName.get(name));

    // KEEP `widgets_values_named` CONSISTENT IF THE FILE CARRIES IT. The
    // frontend writes this map on every save and, when `LiteGraph
    // .namedValuesRestore` is on, PREFERS it over the positional list. That
    // setting is off by default in the installed build, which is why the
    // positional repair above is the load-bearing half -- but leaving a stale
    // entry for a widget this build no longer has would make the two
    // representations disagree, and the one we did not fix would win the moment
    // the setting flipped. Found by a review that read the frontend bundle.
    if (node.widgets_values_named
        && typeof node.widgets_values_named === "object") {
        for (const name of dropped) delete node.widgets_values_named[name];
    }

    return { ok: true, changed: true, dropped };
}

/**
 * Repair every link's destination slot BY IDENTITY.
 *
 * `dst_slot` indexes the node's `inputs` array, which holds link sockets and
 * widget descriptors together -- so removing a descriptor shifts every later
 * slot. Recomputing from `inputs[i].link === linkId` is self-correcting and
 * cannot be double-applied, which subtracting an offset can.
 */
export function repairLinkSlots(graph) {
    // DUPLICATE NODE IDS WOULD COLLAPSE THIS MAP and repair a link against the
    // wrong node. LiteGraph never emits them, so this can only come from a
    // hand-edited file -- but a silently wrong slot is worse than a stop.
    const byId = new Map();
    for (const node of graph.nodes || []) {
        if (byId.has(node.id)) return { duplicateId: node.id };
        byId.set(node.id, node);
    }
    for (const link of graph.links || []) {
        const [linkId, , , dstNodeId] = link;   // [id, src, srcSlot, dst, dstSlot, type]
        const node = byId.get(dstNodeId);
        if (!node) continue;
        const idx = (node.inputs || []).findIndex((s) => s && s.link === linkId);
        if (idx >= 0) link[4] = idx;
    }
}

/**
 * Reconcile a whole graph on a COPY.
 *
 * Returns `{ok:true, graph, changed, notes}` or a refusal. The caller passes the
 * copy to the loader only on success, so a refusal leaves the open canvas as it
 * was.
 */
export function reconcileGraph(graphData, lookup) {
    if (!graphData || !Array.isArray(graphData.nodes)) {
        return { ok: true, graph: graphData, changed: false, notes: [] };
    }
    const copy = structuredClone(graphData);
    let changed = false;
    const notes = [];
    for (const node of copy.nodes) {
        const verdict = reconcileNode(node, lookup);
        if (!verdict.ok) return verdict;
        if (verdict.changed) {
            changed = true;
            notes.push(`${node.type}: dropped ${verdict.dropped.join(", ")}`);
        }
    }
    if (changed) {
        const problem = repairLinkSlots(copy);
        if (problem?.duplicateId !== undefined) {
            return refuse("duplicate node id in the saved graph",
                          `node id ${problem.duplicateId} appears more than ` +
                          `once, so a link cannot be repaired unambiguously`);
        }
    }
    return { ok: true, graph: copy, changed, notes };
}
