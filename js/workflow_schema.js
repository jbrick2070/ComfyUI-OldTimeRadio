/**
 * Frontend glue for the saved-graph schema boundary. GLUE ONLY -- every
 * decision lives in `workflow_schema_core.js`, which imports nothing from
 * ComfyUI and is covered by `tests/js/workflow_schema.test.mjs`.
 *
 * WHY THE LOADER BOUNDARY AND NOT `beforeConfigureGraph`. The frontend runs
 * extension hooks through `invokeExtensionsAsync`, which CATCHES whatever a hook
 * throws and merely logs it. So throwing from `beforeConfigureGraph` cannot stop
 * a stale graph: the load continues and the exception becomes a console line
 * nobody reads. Only owning the call to `loadGraphData` can actually refuse one.
 */
import { app } from "../../scripts/app.js";
import { reconcileGraph } from "./workflow_schema_core.js";

const TAG = "[OldTimeRadio] workflow schema";

app.registerExtension({
    name: "OldTimeRadio.WorkflowSchema",
    async setup() {
        const original = app.loadGraphData;
        if (typeof original !== "function") {
            console.warn(`${TAG}: app.loadGraphData is not a function on this ` +
                         `frontend; schema reconciliation is INACTIVE.`);
            return;
        }

        // Preserve the signature and the `this` binding, and forward with
        // ...rest so a longer argument list keeps working -- the real one is
        // (graphData, clean, restore_view, workflow, options).
        //
        // THE RETURN VALUE IS FORWARDED ON EVERY PATH BUT ONE: a refusal returns
        // undefined where the original would return a Promise. Every call site
        // in the installed frontend `await`s this and none chains .then() or
        // reads the result, so awaiting undefined is harmless there -- but the
        // asymmetry is real and is stated rather than glossed, because a future
        // caller that does chain would find it the hard way.
        app.loadGraphData = function (graphData, ...rest) {
            let verdict;
            try {
                verdict = reconcileGraph(
                    graphData, LiteGraph?.registered_node_types);
            } catch (err) {
                // A fault in THIS code must not cost a user their graph. Log
                // loudly and fall back to the behaviour they had before this
                // file existed.
                console.error(
                    `${TAG}: reconciliation crashed; loading unreconciled.`, err);
                return original.call(this, graphData, ...rest);
            }

            if (!verdict.ok) {
                const msg =
                    `This workflow was saved with a node layout this build no ` +
                    `longer has, and it cannot be migrated safely.\n\n` +
                    `${verdict.reason}\n${verdict.detail}\n\n` +
                    `Nothing was loaded and your open workflow is unchanged. ` +
                    `Re-generate it with scripts/build_variants.py, or open it ` +
                    `in a build that still has that widget.`;
                console.error(
                    `${TAG}: refused -- ${verdict.reason}: ${verdict.detail}`);
                app.ui?.dialog?.show?.(msg);
                return;                      // the original loader never runs
            }

            if (verdict.changed) {
                console.log(`${TAG}: migrated an older graph -- ` +
                            `${verdict.notes.join("; ")}`);
            }
            return original.call(this, verdict.graph, ...rest);
        };

        console.log(`${TAG}: active`);
    },
});
