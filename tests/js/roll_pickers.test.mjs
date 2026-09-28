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
const { poolKey, poolEntries, readPool, writePool, choiceLabel } = await import(
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
    assert.deepEqual(poolEntries(["anime", "cartoon"]), want);
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
