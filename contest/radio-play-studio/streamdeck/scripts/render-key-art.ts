// SPDX-License-Identifier: MIT
// Writes key-art/: each face as SVG, as a 1024 px PNG master, and at the two key
// sizes (120 px for a Stream Deck +, 72 px for the classic decks), plus the
// contact sheet. Run with `npm run key-art`. The labels are Segoe UI, so render on
// Windows, where the plugin runs and CI builds.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Resvg } from "@resvg/resvg-js";
import { KEY_ART, keyArtPreviewSvg, keyArtSvg } from "./key-art.js";

const out = join(process.cwd(), "key-art");
const font = { loadSystemFonts: true, defaultFontFamily: "Segoe UI" };
const png = (svg: string, width?: number) =>
  new Resvg(svg, { font, ...(width ? { fitTo: { mode: "width" as const, value: width } } : {}) }).render().asPng();

for (const sub of ["svg", "png", "key120", "key72"]) {
  mkdirSync(join(out, sub), { recursive: true });
}
for (const art of KEY_ART) {
  const svg = keyArtSvg(art);
  writeFileSync(join(out, "svg", `${art.name}.svg`), svg);
  writeFileSync(join(out, "png", `${art.name}.png`), png(svg, 1024));
  writeFileSync(join(out, "key120", `${art.name}.png`), png(svg, 120));
  writeFileSync(join(out, "key72", `${art.name}.png`), png(svg, 72));
}
const preview = keyArtPreviewSvg();
writeFileSync(join(out, "preview.svg"), preview);
writeFileSync(join(out, "preview.png"), png(preview));
console.log(`key art: ${KEY_ART.map((a) => a.name).join(", ")} -> ${out}`);
