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
 * OTR_LedgerScriptWriter.INPUT_TYPES and OTR_VideoDirector.INPUT_TYPES), so
 * the list is exactly what the backend accepts. If this file does not load,
 * the text boxes are simply what people see, and the typed list still works.
 *
 * CLOSED BY DEFAULT (2026-10-14). Each list sits behind a one-line header that
 * says what the pool holds ("All eligible", or the ticked names) and opens on
 * a click or Enter/Space. Opening and closing touch no saved value. A saved
 * name that is not a choice is reported under the header, open or closed.
 *
 * HOW THE ROW IS SIZED, measured on frontend 1.53.10 in all three places a
 * pool is drawn (App form, Vue-node graph, classic canvas graph):
 * - The frontend stretches the widget's element to the row it was given, so
 *   the row's height is read from an inner `body` that is never stretched;
 *   that is what lets the row shrink again when the list closes.
 * - The classic canvas takes the widget's `margin` (10 by default) off the
 *   top and the bottom of the row, which left a 26 px header a 6 px slot.
 *   `margin: 0` makes the row exactly the body. Nothing clips, so a frontend
 *   that ignores the option draws the old way: content running into the gap.
 * - Only the classic canvas (the widget sits in a `.dom-widget` box there)
 *   takes a node's height from its `size`. A click there grows or shrinks the
 *   node by exactly what the list gained or lost, before the next frame, so
 *   the rows above keep their height and closing undoes opening. Anything
 *   else that changes the list's height there (switching back from the App
 *   form, which lays it out at another width; a note appearing; a narrower
 *   node) is covered by litegraph itself, which grows a node whose rows
 *   overflow (`_arrangeWidgets`) and never clips. The Vue-node graph sizes
 *   the node to its content on its own, and the App form does not draw it.
 */
import { app } from "../../scripts/app.js";
import { readPool, writePool, choiceLabel, poolSummary, POOL_HELP }
    from "./roll_pickers_core.js";

const TAG = "[OldTimeRadio] roll pickers";
// Each node's pools. The video director's pair narrows its Randomize video and
// Randomize stills switches (2026-09-28); the writer's three narrow its
// language, bank and style rolls.
const POOLS_BY_NODE = {
    OTR_LedgerScriptWriter: ["language_roll_pool", "bank_roll_pool", "style_roll_pool"],
    OTR_VideoDirector: ["video_roll_pool", "still_roll_pool"],
};
// The name each closed list leads with. The graph draws no label beside a DOM
// widget, so without it five closed lists would all read "All eligible".
// The same words as the App rows (config/app_mode.json labels, "... to roll").
const POOL_NAMES = {
    language_roll_pool: "Languages",
    bank_roll_pool: "Banks",
    style_roll_pool: "Styles",
    video_roll_pool: "Video lanes",
    still_roll_pool: "Still models",
};

function choicesFor(node, name) {
    const spec = node.constructor?.nodeData?.input?.optional?.[name];
    const choices = spec?.[1]?.otr_choices;
    return Array.isArray(choices) && choices.length ? choices.map(String) : null;
}

let listCount = 0;

function makeChecklist(name, choices, initial, onToggle) {
    let value = initial == null ? "" : String(initial);
    let open = false;
    const root = document.createElement("div");
    root.className = "otr-roll-picker";
    root.dataset.pool = name;
    root.style.cssText = "font:13px/1.4 system-ui,sans-serif;color:var(--input-text,#ddd)";
    const body = document.createElement("div");
    body.style.cssText = "padding:1px 8px 3px";
    const head = document.createElement("button");
    head.type = "button";
    head.title = POOL_HELP;
    head.style.cssText =
        "display:flex;align-items:baseline;gap:6px;width:100%;margin:0;padding:2px 4px;" +
        "border:0;border-radius:4px;background:transparent;color:inherit;font:inherit;" +
        "text-align:left;cursor:pointer";
    const arrow = document.createElement("span");
    arrow.setAttribute("aria-hidden", "true");
    const poolName = POOL_NAMES[name] ?? choiceLabel(name);
    const nameTag = document.createElement("span");
    nameTag.textContent = `${poolName}:`;
    nameTag.style.cssText = "flex:none;opacity:.75";
    const summary = document.createElement("span");
    summary.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    head.append(arrow, nameTag, summary);
    const help = document.createElement("div");
    help.textContent = POOL_HELP;
    help.style.cssText = "padding:0 4px 3px;font-size:12px;opacity:.75";
    const list = document.createElement("div");
    list.id = `otr-roll-pool-${++listCount}`;
    list.setAttribute("role", "group");
    list.setAttribute("aria-label", poolName);
    head.setAttribute("aria-controls", list.id);
    const note = document.createElement("div");
    note.style.cssText = "padding:2px 4px 0;font-size:12px;color:var(--error-text,#e88)";
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
            paint();
        });
        label.append(box, choiceLabel(choice));
        list.append(label);
        return box;
    });
    body.append(head, help, list, note);
    root.append(body);
    // `hidden` would lose to the list's own display:flex, so set display.
    const show = () => {
        head.setAttribute("aria-expanded", String(open));
        arrow.textContent = open ? "\u25BE" : "\u25B8";
        help.hidden = !open;
        list.style.cssText = "flex-wrap:wrap;gap:4px 14px;padding:2px 4px;"
            + `display:${open ? "flex" : "none"}`;
    };
    const paint = () => {
        const { checked, unknown } = readPool(value, choices);
        for (const box of boxes) box.checked = checked.has(box.value);
        summary.textContent = poolSummary(value, choices).text;
        note.hidden = unknown.length === 0;
        note.textContent = unknown.length
            ? `Not a choice here, and the run will stop on it: ${unknown.join(", ")}. `
              + (open ? "Click any box to clear it." : "Open the list and click any box to clear it.")
            : "";
    };
    // The row's height: the drawn body, or before it is drawn (or while the
    // node is off screen) the last height seen in this state, or a guess.
    const seen = new Map();
    const height = () => {
        const key = `${open}|${note.hidden}`;
        const drawn = body.offsetHeight;
        if (drawn > 0) seen.set(key, drawn);
        return seen.get(key) ?? (26 + (note.hidden ? 0 : 34)
            + (open ? 22 + 24 * Math.ceil(choices.length / 2) : 0));
    };
    // Measured on both sides of the toggle in the same task: reading
    // offsetHeight lays the list out now, so the node is refitted before the
    // canvas draws its next frame.
    head.addEventListener("click", () => {
        const before = body.offsetHeight;
        open = !open;
        show();
        paint();
        const after = body.offsetHeight;
        if (before > 0 && after > 0 && after !== before) onToggle?.(before, after);
    });
    show();
    paint();
    return {
        root,
        height,
        get: () => value,
        set: (v) => { value = v == null ? "" : String(v); paint(); },
    };
}

// Grow or shrink the node by what the list just gained or lost, so the rows
// under it move with it and no blank shell is left behind. Never below the
// node's own minimum, never narrower than it was.
function fitNode(node, before, after) {
    // `size` is a typed array on litegraph nodes, not an Array.
    if (typeof node.setSize !== "function" || !(node.size?.length >= 2)) return;
    const min = typeof node.computeSize === "function" ? node.computeSize() : [0, 0];
    const width = Math.max(node.size[0], min[0]);
    const height = Math.max(min[1], node.size[1] + after - before);
    if (width === node.size[0] && height === node.size[1]) return;
    node.setSize([width, height]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "OldTimeRadio.RollPickers",
    nodeCreated(node) {
        const pools = POOLS_BY_NODE[node?.comfyClass];
        if (!pools || !Array.isArray(node.widgets)) return;
        for (const name of pools) {
            try {
                const choices = choicesFor(node, name);
                const index = node.widgets.findIndex((w) => w.name === name);
                if (!choices || index < 0 || typeof node.addDOMWidget !== "function") continue;
                const picker = makeChecklist(name, choices, node.widgets[index].value,
                    (before, after) => {
                        if (picker.root.closest(".dom-widget")) fitNode(node, before, after);
                    });
                const widget = node.addDOMWidget(name, "otr_roll_picker", picker.root, {
                    getValue: picker.get,
                    setValue: picker.set,
                    // Exactly the list's own height, open or closed: a taller
                    // node gives its spare room to other rows, never to a
                    // blank band under the checklist.
                    getMinHeight: picker.height,
                    getMaxHeight: picker.height,
                    margin: 0,
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
