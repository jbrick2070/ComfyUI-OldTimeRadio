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

        // Preserve the signature, the `this` binding and the return value. The
        // real one is (graphData, clean, restore_view, workflow, options) and is
        // async; forwarding with ...rest keeps working if that list grows.
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
