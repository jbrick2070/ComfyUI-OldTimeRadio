/**
 * The roll pools as clickable checkboxes (2026-09-28). GLUE ONLY -- the
 * decisions live in `roll_pickers_core.js`, covered by
 * `tests/js/roll_pickers.test.mjs`.
 *
 * WHY THIS SHAPE. The app view cannot draw a native multi-select (measured on
 * frontend 1.52.7: a blank box), which is why 2.3.9 made the pools typed text.
 * It DOES draw a DOM widget. So each pool's text widget is replaced, IN PLACE
 * and under the SAME NAME, by a checklist whose value is the same typed list.
 * Measured on the sandbox before this was written: the node keeps its 38 saved
 * values and the pool's slot, the queued job carries the checked names, a saved
 * value comes back checked on load, and a save and reload keeps it -- the app
 * form needs no change because the row's widget name never changes.
 *
 * The choices come from the pool's own input spec (`otr_choices`, set by
 * OTR_LedgerScriptWriter.INPUT_TYPES), so the list is exactly what the backend
 * accepts. If this file does not load, the text boxes are simply what people
 * see, and the typed list still works.
 */
import { app } from "../../scripts/app.js";
import { readPool, writePool, choiceLabel } from "./roll_pickers_core.js";

const TAG = "[OldTimeRadio] roll pickers";
const WRITER = "OTR_LedgerScriptWriter";
const POOLS = ["language_roll_pool", "bank_roll_pool", "style_roll_pool"];

function choicesFor(node, name) {
    const spec = node.constructor?.nodeData?.input?.optional?.[name];
    const choices = spec?.[1]?.otr_choices;
    return Array.isArray(choices) && choices.length ? choices.map(String) : null;
}

function makeChecklist(name, choices, initial) {
    let value = initial == null ? "" : String(initial);
    const root = document.createElement("div");
    root.className = "otr-roll-picker";
    root.dataset.pool = name;
    root.style.cssText =
        "display:flex;flex-wrap:wrap;gap:4px 14px;padding:4px 2px;" +
        "font:13px/1.4 system-ui,sans-serif;color:var(--input-text,#ddd)";
    const note = document.createElement("div");
    note.style.cssText = "flex-basis:100%;font-size:12px;color:var(--error-text,#e88)";
    note.hidden = true;
    const boxes = choices.map((choice) => {
        const label = document.createElement("label");
        label.style.cssText = "display:inline-flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap";
        const box = document.createElement("input");
        box.type = "checkbox";
        box.value = choice;
        box.addEventListener("change", () => {
            const checked = new Set(boxes.filter((b) => b.checked).map((b) => b.value));
            value = writePool(checked, choices);
            note.hidden = true;
        });
        label.append(box, choiceLabel(choice));
        root.append(label);
        return box;
    });
    root.append(note);
    const paint = () => {
        const { checked, unknown } = readPool(value, choices);
        for (const box of boxes) box.checked = checked.has(box.value);
        note.hidden = unknown.length === 0;
        note.textContent = unknown.length
            ? `Not a choice here, so it is ignored: ${unknown.join(", ")}` : "";
    };
    paint();
    return {
        root,
        get: () => value,
        set: (v) => { value = v == null ? "" : String(v); paint(); },
    };
}

app.registerExtension({
    name: "OldTimeRadio.RollPickers",
    nodeCreated(node) {
        if (node?.comfyClass !== WRITER || !Array.isArray(node.widgets)) return;
        for (const name of POOLS) {
            try {
                const choices = choicesFor(node, name);
                const index = node.widgets.findIndex((w) => w.name === name);
                if (!choices || index < 0 || typeof node.addDOMWidget !== "function") continue;
                const picker = makeChecklist(name, choices, node.widgets[index].value);
                const widget = node.addDOMWidget(name, "otr_roll_picker", picker.root, {
                    getValue: picker.get,
                    setValue: picker.set,
                    getMinHeight: () => 26 * Math.ceil(choices.length / 3) + 8,
                });
                // addDOMWidget appends; move the checklist into the text box's
                // slot so the saved value's position never changes.
                node.widgets.splice(node.widgets.indexOf(widget), 1);
                node.widgets.splice(index, 1, widget);
            } catch (err) {
                console.warn(`${TAG}: left ${name} as a text box`, err);
            }
        }
    },
});
