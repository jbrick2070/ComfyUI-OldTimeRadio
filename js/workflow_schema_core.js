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

/**
 * A COMBO widget's live options, or null. `/object_info` declares a COMBO as its
 * list of choices; a V2 spec carries them as `["COMBO", {options}]`.
 */
export function comboOptions(nodeType, widgetName, lookup) {
    const input = lookup?.[nodeType]?.nodeData?.input;
    for (const group of ["required", "optional"]) {
        const spec = input?.[group]?.[widgetName];
        if (!Array.isArray(spec)) continue;
        if (Array.isArray(spec[0])) return spec[0];
        if (spec[0] === "COMBO" && Array.isArray(spec[1]?.options)) return spec[1].options;
    }
    return null;
}

const LEGACY_LABEL_TAGS = [" [NOT DOWNLOADED]", " [LOCAL HF]"];

/**
 * A writer label as the backend reads it: the same steps, in the same order, as
 * `_otr_model_catalog._strip_label_suffix` -- the retired download-state tags,
 * then a trailing parenthetical badge, cut at its LAST " (" -- followed by
 * `_canonical_qwen_id`'s one retired spelling. A Hugging Face repo id cannot
 * contain " (", so nothing real is cut.
 */
export function writerName(value) {
    let s = String(value).trim();
    for (const tag of LEGACY_LABEL_TAGS) {
        if (s.endsWith(tag)) s = s.slice(0, -tag.length).trimEnd();
    }
    if (s.endsWith(LEGACY_LABEL_TAGS[1])) s = s.slice(0, -LEGACY_LABEL_TAGS[1].length).trimEnd();
    if (s.endsWith(")") && s.includes(" (")) s = s.slice(0, s.lastIndexOf(" (")).trimEnd();
    // `_otr_model_catalog.DEFAULT_LLM_NF4` -> `QWEN_LLM` (pinned by a test).
    return s === "Qwen/Qwen3.5-4B:nf4" ? "Qwen/Qwen3.5-4B" : s;
}

/**
 * Saved VALUES this build offers under another spelling, by node type and
 * widget name. The writer dropdowns offered "repo (8.7 GB download, ...)" until
 * 2026-10-14 and the bare repo id since; an older graph's value is rewritten to
 * the option it names before the loader sees it, so the widget shows the pick
 * instead of a value its list does not hold (the backend's VALIDATE_INPUTS
 * would run the old spelling anyway -- this is for the person looking at it).
 */
const VALUE_RESPELLINGS = {
    OTR_LedgerScriptWriter: {
        creative_writing_model: writerName,
        technical_model: writerName,
    },
};

/** Respell ONE node's saved values in place, by widget name. Caller works on a copy. */
function respellValues(node, lookup) {
    const rules = VALUE_RESPELLINGS[node.type];
    const values = node.widgets_values;
    if (!rules || !Array.isArray(values)) return [];
    // Which value is which: the file's own names when it carries them one for
    // one, else this build's order when the count agrees -- never a guess.
    const saved = savedWidgetNames(node);
    const live = liveWidgetNames(node.type, lookup);
    const names = saved.length
        ? (saved.length === values.length ? saved : null)
        : (live && live.length === values.length ? live : null);
    if (!names) return [];
    const notes = [];
    for (const [widget, respell] of Object.entries(rules)) {
        const i = names.indexOf(widget);
        if (i < 0 || typeof values[i] !== "string") continue;
        const options = comboOptions(node.type, widget, lookup);
        if (!options || options.includes(values[i])) continue;
        const next = respell(values[i]);
        if (next === values[i] || !options.includes(next)) continue;
        notes.push(`${node.type}.${widget}: '${values[i]}' -> '${next}'`);
        values[i] = next;
        if (node.widgets_values_named && typeof node.widgets_values_named === "object"
            && widget in node.widgets_values_named) {
            node.widgets_values_named[widget] = next;
        }
    }
    return notes;
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
    // descriptors in `inputs`. Returning "unchanged" for that shape would let a
    // count mismatch pass through silently: exactly the positional corruption
    // this boundary exists to stop.
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
    let dropped = false;
    for (const node of copy.nodes) {
        // A respelling touches one value and no input, so it needs no link
        // repair -- and must not start refusing a file the loader takes today.
        const respelled = respellValues(node, lookup);
        if (respelled.length) {
            changed = true;
            notes.push(...respelled);
        }
        const verdict = reconcileNode(node, lookup);
        if (!verdict.ok) return verdict;
        if (verdict.changed) {
            changed = true;
            dropped = true;
            notes.push(`${node.type}: dropped ${verdict.dropped.join(", ")}`);
        }
    }
    if (dropped) {
        const problem = repairLinkSlots(copy);
        if (problem?.duplicateId !== undefined) {
            return refuse("duplicate node id in the saved graph",
                          `node id ${problem.duplicateId} appears more than ` +
                          `once, so a link cannot be repaired unambiguously`);
        }
    }
    return { ok: true, graph: copy, changed, notes };
}
