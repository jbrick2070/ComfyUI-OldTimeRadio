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
 * does. Bracketed text is read by ONE grammar on both sides -- this file and
 * `_otr_rolls._bracket_names` -- so there they agree exactly, both ways.
 * Typed text is trusted only in printable ASCII plus TAB/LF/CR (every choice
 * in every pool is [0-9A-Za-z_], and in that alphabet JS's and Python's
 * trimming, case-folding and `\s` agree); typed text with any other
 * character -- a pasted no-break space, say -- warns although the backend's
 * more forgiving reader may run it. tests/test_roll_pickers.py runs both
 * readers over the same few thousand values.
 */

/** Same limit as the backend's `POOL_TEXT_LIMIT`. */
export const POOL_TEXT_LIMIT = 4096;
const TRUSTED = /^[\x20-\x7e\t\n\r]*$/;
const trimWs = (s) => s.replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "");

/** How a pool entry is compared -- the same rule as the backend's `_pool_key`. */
export function poolKey(name) {
    return String(name).trim().replace(/[\s_-]+/g, "_")
        .replace(/^_+|_+$/g, "").toLowerCase();
}

/** The entries of a saved pool value: a typed list, a JSON list of strings,
 *  or the str() of a list saved by 2.3.8 ("['anime', 'cartoon']"). Anything
 *  the backend would refuse, or text outside the trusted alphabet, comes back
 *  whole, as one entry that matches no choice. (The widget always holds a
 *  string: roll_pickers.js stringifies whatever a graph saved.) */
export function poolEntries(value) {
    const { names, refused } = readEntries(value);
    return refused === null ? names : [refused];
}

// `refused` is the whole text when it cannot be read as names at all; it is
// then reported as is and never matched against a choice (poolKey's own trim
// would otherwise forgive a no-break space the backend does not).
function readEntries(value) {
    const text = trimWs(value == null ? "" : String(value));
    if (!text) return { names: [], refused: null };
    if (text.length > POOL_TEXT_LIMIT || !TRUSTED.test(text)) return { names: [], refused: text };
    if (text.startsWith("[")) {
        const names = bracketNames(text);
        return names ? { names, refused: null } : { names: [], refused: text };
    }
    return { names: text.split(/[,;\n]/).map(trimWs).filter(Boolean), refused: null };
}

// The Python repr of a list of plain names: quoted, no quote, backslash or
// control character inside; one trailing comma allowed ("['anime',]"), no
// empty slot ("[,]"). Mirrors `_otr_rolls._repr_names`.
function reprNames(text) {
    if (!text.endsWith("]")) return null;
    const parts = text.slice(1, -1).split(",").map(trimWs);
    if (parts.length === 1 && parts[0] === "") return [];
    if (parts.length > 1 && parts[parts.length - 1] === ""
        && parts[parts.length - 2] !== "") parts.pop();
    const out = [];
    for (const part of parts) {
        const m = /^(['"])([^'"\\\x00-\x1f\x7f]*)\1$/.exec(part);
        if (!m) return null;
        out.push(m[2]);
    }
    return out;
}

// A bracketed value read by the backend's grammar (`_bracket_names`): a JSON
// list of strings, else the plain repr -- its trimmed names, or null where the
// backend raises. Each name is checked against the alphabet BEFORE trimming
// (a decoded "\ufeffanime" is refused by both; trimmed first, JS would keep it).
function bracketNames(text) {
    let items;
    try {
        items = JSON.parse(text);
    } catch {
        items = reprNames(text);
    }
    if (!Array.isArray(items) || !items.every((v) => typeof v === "string")) return null;
    const names = [];
    for (const raw of items) {
        if (!TRUSTED.test(raw)) return null;
        const name = trimWs(raw);
        if (!name) return null;
        names.push(name);
    }
    return names;
}

/**
 * Which choices a saved value checks, and which entries it names that are not
 * choices at all (shown to the person, dropped on their next click).
 * @returns {{checked: Set<string>, unknown: string[]}}
 */
export function readPool(value, choices) {
    const byKey = new Map(choices.map((c) => [poolKey(c), c]));
    const checked = new Set();
    const { names, refused } = readEntries(value);
    if (refused !== null) return { checked, unknown: [refused] };
    const unknown = [];
    for (const entry of names) {
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
