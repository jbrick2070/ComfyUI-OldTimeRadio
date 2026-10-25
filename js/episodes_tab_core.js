/**
 * The "On air" sidebar tab -- the decisions, with no ComfyUI imports, so
 * `tests/js/episodes_tab.test.mjs` runs them under plain `node --test`. The
 * glue that draws them is `episodes_tab.js`. (CreatorOS row 4.1.)
 *
 * The tab only READS: `GET /otr/episodes` lists every published episode and
 * whether it is on the 24/7 broadcast feed. Pulling and putting back are
 * `scripts/otr_broadcast.py`, never a route, because ComfyUI routes need no
 * login; each row offers the exact command to copy.
 */

/** The command that pulls an episode from air, or puts it back. Double quotes
 *  around the name work in PowerShell, cmd and POSIX shells alike; a name
 *  holding a quote is refused rather than escaped, since obs names never do. */
export function commandFor(episode, onAir) {
    const name = String(episode || "");
    if (!name || /["\r\n]/.test(name)) return null;
    const action = onAir ? "pull" : "put_back";
    return `python scripts/otr_broadcast.py ${action} "${name}"`;
}

/** "3 of 5 on air", or the reason the list is empty. */
export function summaryLine(rows) {
    const list = Array.isArray(rows) ? rows : [];
    if (!list.length) return "No published episodes yet.";
    const onAir = list.filter((r) => r && r.on_air).length;
    return `${onAir} of ${list.length} on air`;
}

/** One row as the tab shows it. Pure. */
export function rowView(row) {
    const r = row || {};
    const state = String(r.state || "off air");
    const when = Number.isFinite(r.published_at)
        ? new Date(r.published_at * 1000).toISOString().slice(0, 16).replace("T", " ")
        : "";
    return {
        title: String(r.title || r.episode || ""),
        episode: String(r.episode || ""),
        state,
        when,
        onAir: state === "on air",
        pending: state === "pull pending",
        command: commandFor(r.episode, state !== "off air"),
        actionLabel: state === "off air" ? "Copy put-back command" : "Copy pull command",
    };
}

/** The response body as rows, or an error message to show instead. */
export function readResponse(body) {
    if (!body || body.ok !== true || !Array.isArray(body.episodes)) {
        return { rows: [], error: (body && body.reason) || "The episode list could not be read." };
    }
    return { rows: body.episodes.map(rowView), error: null };
}
