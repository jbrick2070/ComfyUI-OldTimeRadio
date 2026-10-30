// SPDX-License-Identifier: MIT
// The committed key art in key-art/ is what people put on their keys, so it must
// be what scripts/key-art.ts draws: an edit to a glyph without `npm run key-art`
// fails here. The PNGs are checked for size, not bytes -- rasterizing depends on
// the installed Segoe UI build, the SVG does not.
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { KEY_ART, keyArtPreviewSvg, keyArtSvg } from "../scripts/key-art";

const DIR = join(process.cwd(), "key-art");

function pngSize(path: string): [number, number] {
  const b = readFileSync(path);
  expect(b.subarray(1, 4).toString("latin1")).toBe("PNG");
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

describe("key art", () => {
  it("has the six faces, labelled exactly", () => {
    expect(KEY_ART.map((a) => [a.name, a.label])).toEqual([
      ["studio", "STUDIO"],
      ["logblaster", "LOGS"],
      ["comfyui", "GRAPH"],
      ["spanish", "ESPAÑOL"],
      ["reel", "REEL"],
      ["clips", "CLIPS"]
    ]);
  });

  it("commits each SVG as the module draws it", () => {
    for (const art of KEY_ART) {
      expect(readFileSync(join(DIR, "svg", `${art.name}.svg`), "utf8"), art.name).toBe(keyArtSvg(art));
    }
    expect(readFileSync(join(DIR, "preview.svg"), "utf8")).toBe(keyArtPreviewSvg());
  });

  it("commits every face at every size, and nothing else", () => {
    const names = KEY_ART.map((a) => a.name).sort();
    for (const [folder, size, ext] of [["svg", 0, ".svg"], ["png", 1024, ".png"], ["key120", 120, ".png"], ["key72", 72, ".png"]] as const) {
      expect(readdirSync(join(DIR, folder)).sort(), folder).toEqual(names.map((n) => n + ext).sort());
      if (size) {
        for (const n of names) {
          expect(pngSize(join(DIR, folder, n + ext)), `${folder}/${n}`).toEqual([size, size]);
        }
      }
    }
  });
});
