/**
 * The roll pools as checkboxes -- the decisions, with no ComfyUI imports, so
 * `tests/js/roll_pickers.test.mjs` runs them under plain `node --test`. The
 * glue that draws them is `roll_pickers.js`.
 *
 * The saved value stays what it always was: the typed list the backend parses
 * (`nodes/_otr_rolls.py::parse_roll_pool`). These helpers only translate that
 * text to checked boxes and back, matching the backend's forgiveness: case,
 * and spaces, hyphens or underscores, do not matter.
 *
 * NEVER "FINE" WHERE THE BACKEND STOPS. Every value the backend refuses reads
 * here as an entry that is not a choice, so the list warns before the run
 * does (tests/test_roll_pickers.py runs both on the same inputs). The one
 * difference left is the safe direction: a few hand-typed Python forms that
 * literal_eval accepts and this does not read (a name with a backslash
 * escape, an r/u prefix or triple quotes inside a single-quoted list) warn
 * although the run would go; any click rewrites them as plain names.
 */

/** How a pool entry is compared -- the same rule as the backend's `_pool_key`. */
export function poolKey(name) {
    return String(name).trim().replace(/[\s_-]+/g, "_")
        .replace(/^_+|_+$/g, "").toLowerCase();
}

/** The entries of a saved pool value: a typed list, a JSON list, or the
 *  str() of a list saved by 2.3.8 ("['anime', 'cartoon']"). Text that opens
 *  with "[" is a list or nothing, as in the backend: anything it would refuse
 *  ("[anime, cartoon]", an empty or blank entry, a missing bracket) comes
 *  back whole, as one entry that matches no choice. */
export function poolEntries(value) {
    if (Array.isArray(value)) return value.map(String);
    const text = value == null ? "" : String(value).trim();
    if (!text) return [];
    if (text.startsWith("[")) return bracketEntries(text) ?? [text];
    return text.split(/[,;\n]/).map((s) => s.trim()).filter(Boolean);
}

// A bracketed value read the way `parse_roll_pool` reads it -- json.loads,
// then literal_eval -- as its trimmed names, or null where it raises.
function bracketEntries(text) {
    let items;
    try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) return null;
        // str() of a non-string is never a choice; any spelling of it will do
        items = parsed.map((v) => (typeof v === "string" ? v : JSON.stringify(v)));
    } catch {
        // The Python repr of a list of plain names: quoted, no escapes inside,
        // one trailing comma allowed ("['anime',]"), no empty slot ("[,]").
        if (!text.endsWith("]")) return null;
        const parts = text.slice(1, -1).split(",").map((s) => s.trim());
        if (parts.length === 1 && parts[0] === "") return [];
        if (parts.length > 1 && parts[parts.length - 1] === ""
            && parts[parts.length - 2] !== "") parts.pop();
        const quoted = /^(['"])([^'"\\]*)\1$/;
        items = [];
        for (const part of parts) {
            const m = quoted.exec(part);
            if (!m) return null;
            items.push(m[2]);
        }
    }
    const names = items.map((s) => s.trim());
    // a blank name is not a choice to the backend: it stops the run
    return names.some((s) => !s) ? null : names;
}

/**
 * Which choices a saved value checks, and which entries it names that are not
 * choices at all (shown to the person, dropped on their next click).
 * @returns {{checked: Set<string>, unknown: string[]}}
 */
export function readPool(value, choices) {
    const byKey = new Map(choices.map((c) => [poolKey(c), c]));
    const checked = new Set();
    const unknown = [];
    for (const entry of poolEntries(value)) {
        const hit = byKey.get(poolKey(entry));
        if (hit) checked.add(hit);
        else unknown.push(entry);
    }
    return { checked, unknown };
}

/** The value to save for a set of checked choices, in the choices' own order. */
export function writePool(checked, choices) {
    return choices.filter((c) => checked.has(c)).join(", ");
}

/** A choice as a person reads it: `video_art` -> `video art`. */
export function choiceLabel(choice) {
    return String(choice).replace(/_/g, " ");
}

/** The empty-pool rule, said once, the same on Graph and App. */
export const EMPTY_POOL_RULE = "No selections = all eligible.";

/** The rule in full, between the header and the boxes of an open list, and on
 *  the header's hover. Ticks narrow the draw; they never force a choice this
 *  run cannot take (a video lane this machine cannot run is left out,
 *  `nodes/_otr_lane_rolls.py`). */
export const POOL_HELP = EMPTY_POOL_RULE
    + " Ticks narrow the roll to the ticked choices still eligible for this run.";

/**
 * The one-line summary a COLLAPSED checklist shows (2026-10-14). Read-only:
 * it never normalizes or rewrites the saved value -- opening or closing the
 * list changes nothing that is serialized.
 *
 * An empty pool is "All eligible (none ticked)" -- the empty-pool rule said
 * where a closed list can be read -- never "none": the roll then draws from
 * every eligible choice. A pool with ticks names them, in the choices' own
 * order, and stops listing after `maxNames`. "Ticked", not "will run": a
 * ticked choice this run cannot take is left out of the draw. A saved value
 * naming only things that are not choices is NOT empty -- the backend stops
 * the run on it -- so it never reads "All eligible".
 * @returns {{text: string, count: number, checked: Set<string>, unknown: string[]}}
 */
export function poolSummary(value, choices, maxNames = 3) {
    const { checked, unknown } = readPool(value, choices);
    const picked = choices.filter((c) => checked.has(c));
    if (picked.length === 0) {
        return { text: unknown.length ? "No valid choice ticked" : "All eligible (none ticked)",
                 count: 0, checked, unknown };
    }
    const shown = picked.slice(0, maxNames).map(choiceLabel).join(", ");
    const more = picked.length > maxNames ? `, +${picked.length - maxNames} more` : "";
    return { text: `${picked.length} ticked: ${shown}${more}`,
             count: picked.length, checked, unknown };
}
