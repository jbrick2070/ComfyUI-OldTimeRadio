/**
 * The "On air" sidebar tab (CreatorOS row 4.1). GLUE ONLY -- the decisions live
 * in `episodes_tab_core.js`, covered by `tests/js/episodes_tab.test.mjs`.
 *
 * It lists every published episode and whether it is on the 24/7 broadcast
 * feed, from the read-only `GET /otr/episodes`. It changes nothing: each row's
 * button copies the `scripts/otr_broadcast.py` command that pulls the episode
 * from air or puts it back. If `registerSidebarTab` is missing on this
 * frontend the tab is simply absent and the script still works.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { readResponse, summaryLine } from "./episodes_tab_core.js";

const TAG = "[OldTimeRadio] On air tab";

async function draw(el) {
    el.replaceChildren();
    const box = document.createElement("div");
    box.style.cssText = "padding:8px;display:flex;flex-direction:column;gap:6px;overflow:auto;height:100%";
    const head = document.createElement("div");
    const refresh = document.createElement("button");
    refresh.textContent = "Refresh";
    refresh.addEventListener("click", () => draw(el));
    const status = document.createElement("span");
    status.style.marginLeft = "8px";
    head.append(refresh, status);
    box.append(head);
    el.append(box);

    let parsed;
    try {
        const resp = await api.fetchApi("/otr/episodes");
        parsed = readResponse(await resp.json());
    } catch (err) {
        parsed = { rows: [], error: "The episode list could not be read." };
    }
    if (parsed.error) {
        status.textContent = parsed.error;
        return;
    }
    status.textContent = summaryLine(parsed.rows);
    for (const row of parsed.rows) {
        const item = document.createElement("div");
        item.style.cssText = "border-top:1px solid var(--border-color,#444);padding-top:4px";
        const title = document.createElement("div");
        title.textContent = `${row.title}  (${row.state}${row.when ? ", " + row.when : ""})`;
        title.title = row.episode;
        title.style.fontWeight = row.onAir ? "600" : "400";
        item.append(title);
        if (row.command) {
            const copy = document.createElement("button");
            copy.textContent = row.actionLabel;
            copy.title = row.command;
            copy.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(row.command);
                    copy.textContent = "Copied";
                } catch (err) {
                    copy.textContent = row.command;
                }
            });
            item.append(copy);
        }
        box.append(item);
    }
}

app.registerExtension({
    name: "OldTimeRadio.EpisodesTab",
    async setup() {
        const manager = app.extensionManager;
        if (!manager || typeof manager.registerSidebarTab !== "function") {
            console.warn(`${TAG}: registerSidebarTab is not available on this frontend; ` +
                         "the tab is not shown (scripts/otr_broadcast.py still works).");
            return;
        }
        manager.registerSidebarTab({
            id: "otr-on-air",
            icon: "pi pi-video",
            title: "On air",
            tooltip: "Old Time Radio: what the 24/7 broadcast is airing",
            type: "custom",
            render: (el) => { draw(el); },
        });
    },
});
