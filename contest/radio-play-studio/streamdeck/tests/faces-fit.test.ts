// SPDX-License-Identifier: MIT
// Every word on every face fits its face. Each <text> the renderers write is
// measured in Segoe UI (the same advances the fitter uses, read from the real
// font) and must sit inside its box: a node's body, or the Make key's clear
// disc at that line's height. Rows must not overlap, and no real choice value
// may be cut short -- only a server's long error text may end in an ellipsis.
import { describe, expect, it } from "vitest";
import { displayValue } from "../src/actions/choice-math.js";
import {
  CELL,
  DialCellInput,
  MakeKeyInput,
  NODE_KEY,
  renderMakeKeySvg,
  renderNodeSvg,
  SOCKET_COLORS
} from "../src/gfx/svg-renderer.js";
import { MACHINE_NAMES } from "../src/state/studio-store.js";
import { PHASE_WORDS } from "../src/actions/faces.js";
import { BANKS, checkKey, checkNode, LANGUAGES, LONG_ERROR, STYLES, WAYS } from "./helpers/face-measure.js";

const ACTS = ["1 act", "7 acts"];

const VALUES: Array<[string, string]> = [
  ...LANGUAGES.map((v): [string, string] => ["Language", displayValue(v)]),
  ...BANKS.map((v): [string, string] => ["Story bank", displayValue(v)]),
  ...STYLES.map((v): [string, string] => ["Visual style", displayValue(v)]),
  ...ACTS.map((v): [string, string] => ["Acts", v]),
  ...Object.values(MACHINE_NAMES).map((v): [string, string] => ["Machine", v]),
  ...WAYS.map((v): [string, string] => ["Video", v]),
  ["Language", "Off air"],
  ["Story bank", "Tuning in…"],
  ["Visual style", "Not offered"],
  ["Machine", "Its own graph"],
  ["Video", "Its own graph"]
];

type Hint = { hint: string; pending?: boolean; tone?: DialCellInput["tone"]; progress?: number; title?: string };
const COMMON_HINTS: Hint[] = [
  { hint: "on air", tone: "muted", progress: 0.42 },
  { hint: "nothing on port 65535", tone: "muted" },
  { hint: "reading the form…", tone: "muted" },
  { hint: "no machine menu", tone: "muted" },
  { hint: "not yet proven" },
  ...Object.values(MACHINE_NAMES).map((m) => ({ hint: m })),
  ...WAYS.map((w) => ({ hint: w })),
  { hint: LONG_ERROR, tone: "muted" }
];
/** What faces.ts writes on a strip cell, and on a key. */
const HINTS: Record<"strip cell" | "node key", Hint[]> = {
  "strip cell": [
    ...COMMON_HINTS,
    { hint: "press to reset" },
    ...WAYS.map((w) => ({ hint: `${w}\npress to switch`, pending: true })),
    ...Object.values(MACHINE_NAMES).map((m) => ({ hint: `${m}\npress to switch`, pending: true })),
    { hint: "not yet proven\npress to switch", pending: true }
  ],
  "node key": [
    ...COMMON_HINTS,
    { hint: "press: next\nhold: reset" },
    ...WAYS.map((w) => ({ hint: w, pending: true, title: "Hold to use" })),
    ...Object.values(MACHINE_NAMES).map((m) => ({ hint: m, pending: true, title: "Hold to use" })),
    { hint: "not yet proven", pending: true, title: "Hold to use" }
  ]
};

describe("every word fits its face", () => {
  for (const [name, g] of [["strip cell", CELL], ["node key", NODE_KEY]] as const) {
    it(`fits every real value and hint on a ${name}`, () => {
      let faces = 0;
      for (const [fieldTitle, value] of VALUES) {
        for (const h of HINTS[name]) {
          const title = h.title ?? fieldTitle;
          const input: DialCellInput = {
            value,
            position: "11/11",
            socket: SOCKET_COLORS.style,
            ...h,
            title
          };
          const els = checkNode(renderNodeSvg(input, g), g, `${name} ${title}=${value} / ${h.hint}`);
          // A real value is never cut; only the long server text may be.
          const valueEls = els.filter((t) => t.weight === 700 && t.y > g.nodeY + g.titleH);
          expect(valueEls.map((t) => t.text).join(" "), `${name}: ${value} was cut`).toBe(value);
          expect(els.some((t) => t.text.toUpperCase() === title.toUpperCase()), `${name}: title ${title} was cut`).toBe(true);
          if (h.hint !== LONG_ERROR) {
            const hintText = els.filter((t) => t.weight === 400).map((t) => t.text).join(" ");
            expect(hintText, `${name}: hint was cut`).toBe(h.hint.replace(/\n/g, " "));
          }
          faces += 1;
        }
      }
      expect(faces).toBe(VALUES.length * HINTS[name].length);
    });
  }

  it("fits every Make key face, at every phase, with long minutes and long reasons", () => {
    const faces: Array<[string, MakeKeyInput]> = [
      ["ready", { headline: "MAKE", wave: true, caption: "the episode" }],
      ["off", { headline: "OFF AIR", caption: ["start the", "Studio"], tone: "muted" }],
      ["loading", { glyph: "dots", caption: "tuning in…", tone: "muted" }],
      ["starting", { glyph: "dots", caption: "on the way…" }],
      ["done", { glyph: "check", caption: "press to play", tone: "good", progress: 1 }],
      ["busy", { glyph: "pause", caption: "a stage is running", tone: "muted" }],
      ["busy 409", { glyph: "pause", caption: LONG_ERROR, tone: "muted" }],
      ["error", { glyph: "cross", caption: "ComfyUI is not answering on http://127.0.0.1:8188", tone: "alert" }],
      ["form error", { glyph: "cross", caption: LONG_ERROR, tone: "alert" }],
      ...Object.values(PHASE_WORDS).flatMap((word): Array<[string, MakeKeyInput]> => [
        [`running ${word}`, { headline: "100%", caption: [word, "120 min left"], progress: 0.99 }],
        [`running ${word} 0%`, { headline: "0%", caption: [word], progress: 0 }]
      ]),
      ["running external", { headline: "57%", caption: ["from the page", "45 min left"], progress: 0.57 }]
    ];
    for (const [label, input] of faces) {
      const els = checkKey(renderMakeKeySvg(input), label);
      const caption = Array.isArray(input.caption) ? input.caption.join(" ") : input.caption ?? "";
      if (caption && caption !== LONG_ERROR && !caption.includes("http")) {
        const drawn = els.filter((t) => t.weight === 400).map((t) => t.text).join(" ");
        expect(drawn, `${label}: a caption was cut`).toBe(caption);
      }
      if (input.headline) {
        expect(els.filter((t) => t.text === input.headline), `${label}: headline was cut`).toHaveLength(2);
      }
    }
  });
});
