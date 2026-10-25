/**
 * Tests for js/episodes_tab_core.js -- the real shipped module.
 *
 * Run: node --test tests/js/episodes_tab.test.mjs
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ""));
const { commandFor, summaryLine, rowView, readResponse } = await import(
    pathToFileURL(path.join(here, "..", "..", "js", "episodes_tab_core.js")).href);

const NAME = "wire_conduction_20261015_082547__vstb__stmo__zimg__koko__pubd__cg4e2__sa3_final.mp4";

test("an episode on air gets the pull command, one off air the put-back", () => {
    assert.equal(commandFor(NAME, true), `python scripts/otr_broadcast.py pull "${NAME}"`);
    assert.equal(commandFor(NAME, false), `python scripts/otr_broadcast.py put_back "${NAME}"`);
});

test("a name that would break the quoting is refused, not escaped", () => {
    assert.equal(commandFor('a"b_final.mp4', true), null);
    assert.equal(commandFor("a\nb_final.mp4", true), null);
    assert.equal(commandFor("", true), null);
});

test("the summary counts what is on air", () => {
    assert.equal(summaryLine([]), "No published episodes yet.");
    assert.equal(summaryLine([{ onAir: true }, { onAir: false }, { onAir: true }]),
                 "2 of 3 on air");
    assert.equal(summaryLine(null), "No published episodes yet.");
});

test("a pending pull is still offered as a pull", () => {
    const row = rowView({ episode: NAME, title: "wire_conduction_20261015_082547",
                          state: "pull pending", published_at: 0 });
    assert.equal(row.pending, true);
    assert.equal(row.onAir, false);
    assert.ok(row.command.includes(" pull "));
    assert.equal(row.when, "1970-01-01 00:00");
});

test("an off-air row offers the put-back command", () => {
    const row = rowView({ episode: NAME, title: "t", state: "off air", published_at: 1 });
    assert.ok(row.command.includes(" put_back "));
    assert.equal(row.actionLabel, "Copy put-back command");
});

test("a failed or malformed response shows its reason and no rows", () => {
    assert.deepEqual(readResponse({ ok: false, reason: "see the console" }),
                     { rows: [], error: "see the console" });
    assert.deepEqual(readResponse(null), { rows: [], error: "The episode list could not be read." });
    const ok = readResponse({ ok: true, episodes: [{ episode: NAME, state: "on air", on_air: true }] });
    assert.equal(ok.error, null);
    assert.equal(ok.rows[0].onAir, true);
    // The count the tab prints is taken from the rows it draws.
    assert.equal(summaryLine(ok.rows), "1 of 1 on air");
});
