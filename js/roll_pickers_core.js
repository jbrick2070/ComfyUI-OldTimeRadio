/**
 * The roll pools as checkboxes -- the decisions, with no ComfyUI imports, so
 * `tests/js/roll_pickers.test.mjs` runs them under plain `node --test`. The
 * glue that draws them is `roll_pickers.js`.
 *
 * The saved value stays what it always was: the typed list the backend parses
 * (`nodes/_otr_rolls.py::parse_roll_pool`). These helpers only translate that
 * text to checked boxes and back, matching the backend's forgiveness: case,
 * and spaces, hyphens or underscores, do not matter. One known difference: a
 * saved 2.3.8 list is split on commas, so a quoted entry containing a comma
 * would split where the backend's literal_eval would not. No choice contains a
 * comma today.
 */

/** How a pool entry is compared -- the same rule as the backend's `_pool_key`. */
export function poolKey(name) {
    return String(name).trim().replace(/[\s_-]+/g, "_")
        .replace(/^_+|_+$/g, "").toLowerCase();
}

/** The entries of a saved pool value: a typed list, a JSON list, or the
 *  str() of a list saved by 2.3.8 ("['anime', 'cartoon']"). Text that opens
 *  with "[" is a list or nothing, as in the backend: an unquoted name inside
 *  brackets ("[anime, cartoon]") is not a list there and stops the run, so it
 *  comes back whole, as one entry that matches no choice. */
export function poolEntries(value) {
    if (Array.isArray(value)) return value.map(String);
    let text = value == null ? "" : String(value).trim();
    if (!text) return [];
    if (text.startsWith("[")) {
        const items = text.endsWith("]") ? text.slice(1, -1).split(",")
            .map((s) => s.trim()).filter(Boolean) : null;
        const quoted = /^(['"])[^'"]*\1$/;
        if (!items || !items.every((s) => quoted.test(s))) return [text];
        return items.map((s) => s.slice(1, -1).trim()).filter(Boolean);
    }
    return text.split(/[,;\n]/).map((s) => s.trim()).filter(Boolean);
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
