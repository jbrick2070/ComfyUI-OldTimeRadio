/**
 * Tests for js/roll_pickers_core.js -- the real shipped module.
 *
 * Run: node --test tests/js/roll_pickers.test.mjs
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ""));
const { poolKey, poolEntries, readPool, writePool, choiceLabel, poolSummary,
        EMPTY_POOL_RULE, POOL_HELP } = await import(
    pathToFileURL(path.join(here, "..", "..", "js", "roll_pickers_core.js")).href);

const STYLES = ["anime", "cartoon", "recur_frac", "video_art"];
const LANGS = ["English", "Spanish", "French", "Japanese"];

test("the key matches the backend: case and separators do not matter", () => {
    assert.equal(poolKey("Video Art"), "video_art");
    assert.equal(poolKey("  RECUR-frac "), "recur_frac");
    assert.equal(poolKey("English-"), "english");
});

test("a typed list, a JSON list and a 2.3.8 saved list read the same", () => {
    const want = ["anime", "cartoon"];
    assert.deepEqual(poolEntries("anime, cartoon"), want);
    assert.deepEqual(poolEntries("anime;cartoon"), want);
    assert.deepEqual(poolEntries('["anime", "cartoon"]'), want);
    assert.deepEqual(poolEntries("['anime', 'cartoon']"), want);
    // what roll_pickers.js hands over for a graph that saved a real list
    assert.deepEqual(poolEntries(String(["anime", "cartoon"])), want);
    assert.deepEqual(poolEntries(""), []);
    assert.deepEqual(poolEntries(null), []);
});

test("a saved value checks its choices, forgiving case and spaces", () => {
    const { checked, unknown } = readPool("english, JAPANESE", LANGS);
    assert.deepEqual([...checked].sort(), ["English", "Japanese"]);
    assert.deepEqual(unknown, []);
    assert.deepEqual([...readPool("Video Art", STYLES).checked], ["video_art"]);
});

test("a name that is not a choice is reported, not checked", () => {
    const { checked, unknown } = readPool("anime, oil_painting", STYLES);
    assert.deepEqual([...checked], ["anime"]);
    assert.deepEqual(unknown, ["oil_painting"]);
});

test("the saved value lists checked choices in the choices' own order", () => {
    assert.equal(writePool(new Set(["video_art", "anime"]), STYLES), "anime, video_art");
    assert.equal(writePool(new Set(), STYLES), "");
});

test("reading what was written gives back the same boxes", () => {
    const checked = new Set(["Spanish", "French"]);
    const again = readPool(writePool(checked, LANGS), LANGS).checked;
    assert.deepEqual([...again].sort(), [...checked].sort());
});

test("labels read as words", () => {
    assert.equal(choiceLabel("video_art"), "video art");
    assert.equal(choiceLabel("English"), "English");
});

test("an empty pool summarizes as all eligible, never as none", () => {
    for (const empty of ["", null, "  ", "[]"]) {
        const s = poolSummary(empty, STYLES);
        assert.equal(s.text, "All eligible (none ticked)");
        assert.equal(s.count, 0);
        assert.deepEqual(s.unknown, []);
    }
    assert.match(EMPTY_POOL_RULE, /No selections = all eligible/);
    assert.ok(POOL_HELP.startsWith(EMPTY_POOL_RULE));
    assert.match(POOL_HELP, /still eligible/);
});

test("one, several and many ticks summarize in the choices' own order", () => {
    assert.equal(poolSummary("video_art", STYLES).text, "1 ticked: video art");
    assert.equal(poolSummary("video_art, anime", STYLES).text, "2 ticked: anime, video art");
    const many = poolSummary("anime, cartoon, recur_frac, video_art", STYLES, 2);
    assert.equal(many.text, "4 ticked: anime, cartoon, +2 more");
    assert.equal(many.count, 4);
});

test("an unknown saved entry is reported by the summary too", () => {
    const s = poolSummary("anime, oil_painting", STYLES);
    assert.equal(s.text, "1 ticked: anime");
    assert.deepEqual(s.unknown, ["oil_painting"]);
    // only unknown names: the backend stops the run, so never "All eligible"
    const bad = poolSummary("oil_painting", STYLES);
    assert.deepEqual(bad.unknown, ["oil_painting"]);
    assert.equal(bad.count, 0);
    assert.equal(bad.text, "No valid choice ticked");
});

test("the summary reports the boxes a saved value checks", () => {
    const s = poolSummary("Video Art ;  ANIME", STYLES);
    assert.deepEqual([...s.checked].sort(), ["anime", "video_art"]);
    assert.equal(s.count, 2);
});

test("bracketed text is read by the backend's one grammar, or reported whole", () => {
    // _otr_rolls._bracket_names: a JSON list of strings, else the plain repr.
    // Anything it refuses -- an unquoted name, an empty or blank slot, a
    // missing bracket, a non-string item, an escape inside a repr, a name with
    // a character outside the alphabet -- stops the run, so the list must not
    // read it as ticks or as an empty pool.
    for (const bad of ["[anime, cartoon]", '["anime", cartoon]', "[anime",
                       '[""]', '[" "]', '["", "anime"]', "[,]", '["anime",,"cartoon"]',
                       "[1]", "[null]", '[["anime"]]', "['anime\\n']", "[r'anime']",
                       '["\\ufeffanime"]', '["anime\\u00a0"]', '["anime"] x']) {
        assert.deepEqual(poolEntries(bad), [bad], bad);
        const s = poolSummary(bad, STYLES);
        assert.equal(s.text, "No valid choice ticked", bad);
        assert.deepEqual(s.unknown, [bad]);
    }
    assert.deepEqual(poolEntries('[ "anime" , "cartoon" ]'), ["anime", "cartoon"]);
    assert.deepEqual(poolEntries('["anime\\t", "cartoon"]'), ["anime", "cartoon"]);
    assert.deepEqual(poolEntries('["\\u0061nime"]'), ["anime"]);
    assert.deepEqual(poolEntries("['anime',]"), ["anime"]);
    assert.deepEqual(poolEntries("[ ]"), []);
});

test("a long run of spaces is read in linear time", () => {
    const value = "x" + " ".repeat(200000) + "x";
    const t0 = Date.now();
    assert.equal(poolSummary(value, STYLES).text, "No valid choice ticked");
    assert.ok(Date.now() - t0 < 1000, "reading took " + (Date.now() - t0) + " ms");
});

test("typed text is trusted only in printable ASCII", () => {
    for (const odd of ["anime\u00a0", "\ufeffanime", "anime,\u3000cartoon", "anime\x0b"]) {
        assert.deepEqual(poolEntries(odd).length, 1, JSON.stringify(odd));
        assert.equal(poolSummary(odd, STYLES).text, "No valid choice ticked");
    }
    assert.deepEqual(poolEntries("x".repeat(4097)), ["x".repeat(4097)]);
});
