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
 *  str() of a list saved by 2.3.8 ("['anime', 'cartoon']"). */
export function poolEntries(value) {
    if (Array.isArray(value)) return value.map(String);
    let text = value == null ? "" : String(value).trim();
    if (!text) return [];
    if (text.startsWith("[") && text.endsWith("]")) {
        text = text.slice(1, -1);
        return text.split(",").map((s) => s.trim().replace(/^['"]|['"]$/g, ""))
            .filter(Boolean);
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
