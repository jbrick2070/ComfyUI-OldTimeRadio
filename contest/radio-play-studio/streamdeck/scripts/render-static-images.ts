// SPDX-License-Identifier: MIT
// Writes the images Stream Deck shows before the plugin has drawn anything --
// each action's default key image -- from the same renderer the live faces
// use, so the store's tile, the action list and the deck all look like one
// thing. Run with `npm run images` after changing the faces.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { renderMakeKeySvg, renderNodeKeySvg, SOCKET_COLORS } from "../src/gfx/svg-renderer.js";

const root = join(process.cwd(), "com.jbrick2070.otr.sdPlugin", "imgs", "actions");
const files: Record<string, string> = {
  "make/key.svg": renderMakeKeySvg({ headline: "MAKE", wave: true, caption: "the episode" }),
  "choice/key.svg": renderNodeKeySvg({
    title: "Story bank",
    value: "Surprise me",
    hint: "press: next\nhold: reset",
    socket: SOCKET_COLORS.bank
  }),
  "machine/key.svg": renderNodeKeySvg({
    title: "Machine",
    value: "16 GB card",
    hint: "foley",
    socket: SOCKET_COLORS.machine
  })
};
for (const [rel, svg] of Object.entries(files)) {
  writeFileSync(join(root, rel), svg + "\n", "utf8");
  console.log("wrote", rel);
}
