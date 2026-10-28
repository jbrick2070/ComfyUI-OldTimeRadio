// SPDX-License-Identifier: MIT
// Every word on every face fits its face. Each <text> the renderers write is
// measured in Segoe UI (the same advances the fitter uses, read from the real
// font) and must sit inside its box: a node's body, or the Make key's clear
// disc at that line's height. Rows must not overlap, and no real choice value
// may be cut short -- only a server's long error text may end in an ellipsis.
import { describe, expect, it } from "vitest";
import { displayValue } from "../src/actions/choice-math.js";
import {
  ASCENT,
  CELL,
  DESCENT,
  DialCellInput,
  KEY,
  keyHalfChord,
  MakeKeyInput,
  NODE_KEY,
  NodeGeometry,
  PALETTE,
  renderMakeKeySvg,
  renderNodeSvg,
  SOCKET_COLORS
} from "../src/gfx/svg-renderer.js";
import { textWidth, Weight } from "../src/gfx/text-fit.js";
import { MACHINE_NAMES } from "../src/state/studio-store.js";
import { PHASE_WORDS } from "../src/actions/faces.js";

interface TextEl {
  x: number;
  y: number;
  size: number;
  weight: Weight;
  anchor: string;
  spacing: number;
  fill: string;
  text: string;
}

function unescape(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&");
}

function texts(svg: string): TextEl[] {
  const out: TextEl[] = [];
  for (const m of svg.matchAll(/<text ([^>]*)>([^<]*)<\/text>/g)) {
    const attr = (name: string): string | undefined => new RegExp(`${name}="([^"]*)"`).exec(m[1])?.[1];
    out.push({
      x: Number(attr("x")),
      y: Number(attr("y")),
      size: Number(attr("font-size")),
      weight: Number(attr("font-weight") ?? 400) as Weight,
      anchor: attr("text-anchor") ?? "start",
      spacing: Number(attr("letter-spacing") ?? 0),
      fill: attr("fill") ?? "",
      text: unescape(m[2])
    });
  }
  return out;
}

function span(t: TextEl): { left: number; right: number; top: number; bottom: number } {
  const w = textWidth(t.text, { size: t.size, weight: t.weight, letterSpacing: t.spacing });
  const left = t.anchor === "middle" ? t.x - w / 2 : t.anchor === "end" ? t.x - w : t.x;
  return { left, right: left + w, top: t.y - t.size * ASCENT, bottom: t.y + t.size * DESCENT };
}

function checkNode(svg: string, g: NodeGeometry, label: string): TextEl[] {
  const els = texts(svg);
  expect(els.length, label).toBeGreaterThan(0);
  const left = g.nodeX + g.pad - 0.5;
  const right = g.nodeX + g.nodeW - g.pad + 0.5;
  for (const t of els) {
    const s = span(t);
    expect(s.left, `${label}: "${t.text}" runs off the left (${s.left.toFixed(1)} < ${left})`).toBeGreaterThanOrEqual(left);
    expect(s.right, `${label}: "${t.text}" runs off the right (${s.right.toFixed(1)} > ${right})`).toBeLessThanOrEqual(right);
    expect(s.top, `${label}: "${t.text}" above the node`).toBeGreaterThanOrEqual(g.nodeY);
    expect(s.bottom, `${label}: "${t.text}" below the node`).toBeLessThanOrEqual(g.nodeY + g.nodeH);
  }
  // Rows (one per baseline) stack without touching; the title row holds two.
  const rows = [...new Map(els.map((t) => [t.y, t])).values()].sort((a, b) => a.y - b.y);
  for (let i = 1; i < rows.length; i += 1) {
    expect(span(rows[i]).top, `${label}: "${rows[i].text}" overlaps "${rows[i - 1].text}"`).toBeGreaterThanOrEqual(
      span(rows[i - 1]).bottom - 1
    );
  }
  return els;
}

function checkKey(svg: string, label: string): TextEl[] {
  const els = texts(svg);
  for (const t of els) {
    const s = span(t);
    const half = keyHalfChord(s.top, s.bottom);
    const reach = Math.max(KEY.cx - s.left, s.right - KEY.cx);
    expect(reach, `${label}: "${t.text}" leaves the ring's clear disc (${reach.toFixed(1)} > ${half.toFixed(1)})`).toBeLessThanOrEqual(
      half + 0.5
    );
  }
  const main = els.filter((t) => t.fill !== PALETTE.pink).sort((a, b) => a.y - b.y);
  for (let i = 1; i < main.length; i += 1) {
    expect(span(main[i]).top, `${label}: "${main[i].text}" overlaps "${main[i - 1].text}"`).toBeGreaterThanOrEqual(
      span(main[i - 1]).bottom - 1
    );
  }
  return els;
}

// The longest real values each control can show (nodes/_otr_shared/shortcodes.py,
// config/episode_languages.json, runner.PHASE_BY_CLASS, config.MACHINES).
const STYLES = [
  "roll (any style)", "anime", "archival_documentary", "cartoon", "paper_origami", "recur_frac",
  "sci_fi_radio", "shakespeare_stage_realism", "storybook_engraving", "video_art", "visual_storybased"
];
const BANKS = ["roll (any eligible bank)", "media_archive", "original", "scifi_news_pro", "public_domain", "shakespeare"];
const LANGUAGES = ["roll (any language)", "Off", "English", "Portuguese", "Mandarin", "Japanese", "Spanish"];
const ACTS = ["1 act", "7 acts"];
const WAYS = ["foley", "video", "stills"];
const LONG_ERROR =
  "project 2026-10-28-wanderer is waiting to hear how its last run ended; open it and look again, or forget that run";

const VALUES: Array<[string, string]> = [
  ...LANGUAGES.map((v): [string, string] => ["Language", displayValue(v)]),
  ...BANKS.map((v): [string, string] => ["Story bank", displayValue(v)]),
  ...STYLES.map((v): [string, string] => ["Visual style", displayValue(v)]),
  ...ACTS.map((v): [string, string] => ["Acts", v]),
  ...Object.values(MACHINE_NAMES).map((v): [string, string] => ["Machine", v]),
  ["Language", "Off air"],
  ["Story bank", "Tuning in…"],
  ["Visual style", "Not offered"],
  ["Machine", "Its own graph"]
];

type Hint = { hint: string; pending?: boolean; tone?: DialCellInput["tone"]; progress?: number; title?: string };
const COMMON_HINTS: Hint[] = [
  { hint: "on air", tone: "muted", progress: 0.42 },
  { hint: "nothing on port 65535", tone: "muted" },
  { hint: "reading the form…", tone: "muted" },
  { hint: "no machine menu", tone: "muted" },
  ...WAYS.map((w) => ({ hint: `${w} (unproven)` })),
  { hint: LONG_ERROR, tone: "muted" }
];
/** What faces.ts writes on a strip cell, and on a key. */
const HINTS: Record<"strip cell" | "node key", Hint[]> = {
  "strip cell": [
    ...COMMON_HINTS,
    { hint: "press to reset" },
    ...WAYS.map((w) => ({ hint: `${w} (unproven)\npress to switch`, pending: true }))
  ],
  "node key": [
    ...COMMON_HINTS,
    { hint: "press: next\nhold: reset" },
    ...WAYS.map((w) => ({ hint: `${w} (unproven)`, pending: true, title: "Hold to use" }))
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
