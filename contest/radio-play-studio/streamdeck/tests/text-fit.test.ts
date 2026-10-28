// SPDX-License-Identifier: MIT
// The fitter's own edges: nothing, a word wider than any line, more words
// than the lines can hold, and a box too small for even an ellipsis.
import { describe, expect, it } from "vitest";
import { FIT_SAFETY, fitText, textWidth, truncateToWidth } from "../src/gfx/text-fit.js";
import { CELL, renderNodeSvg } from "../src/gfx/svg-renderer.js";
import { texts } from "./helpers/face-measure.js";

describe("fitting text to a box", () => {
  it("fits nothing as no lines", () => {
    expect(fitText("", { maxWidth: 100, maxSize: 20, minSize: 10 })).toEqual({ lines: [], size: 20, truncated: false });
    expect(fitText("   ", { maxWidth: 100, maxSize: 20, minSize: 10 }).lines).toEqual([]);
  });

  it("shrinks before it wraps, and wraps before it cuts", () => {
    const one = fitText("Shakespeare", { maxWidth: 160, maxSize: 25, minSize: 13, weight: 700, maxLines: 2 });
    expect(one.lines).toEqual(["Shakespeare"]);
    expect(one.size).toBeGreaterThanOrEqual(20);
    const two = fitText("Shakespeare stage realism", { maxWidth: 160, maxSize: 25, minSize: 13, weight: 700, maxLines: 2, maxWrapSize: 17 });
    expect(two.lines).toEqual(["Shakespeare", "stage realism"]);
    expect(two.truncated).toBe(false);
  });

  it("cuts a word wider than any line with an ellipsis, inside the line", () => {
    const fitted = fitText("Supercalifragilisticexpialidocious", { maxWidth: 80, maxSize: 14, minSize: 12, maxLines: 2 });
    expect(fitted.truncated).toBe(true);
    for (const line of fitted.lines) {
      expect(textWidth(line, { size: fitted.size })).toBeLessThanOrEqual(80 * FIT_SAFETY);
    }
    expect(fitted.lines.at(-1)!.endsWith("…")).toBe(true);
  });

  it("keeps to its line count when the words run over, and ends the last line with an ellipsis", () => {
    const long = "project 2026-10-28-wanderer is waiting to hear how its last run ended; open it and look again";
    const fitted = fitText(long, { maxWidth: 100, maxSize: 13, minSize: 13, maxLines: 2 });
    expect(fitted.lines).toHaveLength(2);
    expect(fitted.truncated).toBe(true);
    expect(fitted.lines[1].endsWith("…")).toBe(true);
    for (const line of fitted.lines) {
      expect(textWidth(line, { size: 13 })).toBeLessThanOrEqual(100 * FIT_SAFETY);
    }
  });

  it("gives an ellipsis alone when not even one letter fits", () => {
    expect(truncateToWidth("Portuguese", 4, { size: 20 })).toBe("…");
    expect(truncateToWidth("Hi", 1000, { size: 20 })).toBe("Hi");
  });

  it("never cuts a character in half (an emoji in a server reason)", () => {
    const cut = truncateToWidth("ok 🎙🎙🎙🎙🎙🎙🎙🎙", 60, { size: 13 });
    expect(() => encodeURIComponent(cut)).not.toThrow();
  });

  it("draws a dash for a blank value, never nothing", () => {
    const els = texts(renderNodeSvg({ title: "Language", value: "   " }, CELL));
    expect(els.some((t) => t.text === "—")).toBe(true);
  });
});
