// SPDX-License-Identifier: MIT
// The faces, in the Studio's own colours (the page's palette and the OTR
// icon: deep purple, a faint grid, the green lamp, heavy white lettering with
// a hot-pink shadow) and in ComfyUI's own shape: every dial is a node, with a
// title bar, typed sockets and a wire. The Stream Deck + strip lays its four
// cells edge to edge, so the wires meet and the strip reads as one graph --
// Language -> Story bank -> Visual style -> Acts -- and while an episode is
// being made each node is outlined green, the way ComfyUI marks the node it
// is running.
//
// Plain SVG strings, so the tests can read them, and only what the Stream
// Deck's SVG renderer draws: no filters, no masks, no web fonts. Every string
// is laid out by text-fit.ts against its box -- the node's body, or the chord
// of the key's ring at that height -- so nothing runs off a face.
import { fitText, textWidth, truncateToWidth, FIT_SAFETY, Weight } from "./text-fit.js";

/** The page's colours (radio_play_studio/static/index.html :root), plus the
 *  hot pink of the OTR icon's lettering shadow. */
export const PALETTE = {
  bg: "#12092F",
  panel: "#1C1240",
  strip: "#24184C",
  ink: "#F4EDFB",
  muted: "#B9ACDC",
  line: "#3B2B70",
  fieldLine: "#7A64B8",
  accent: "#7C3AED",
  accentText: "#B794FF",
  lamp: "#50F898",
  bad: "#FF8A7A",
  bar: "#2E2060",
  grid: "#483088",
  pink: "#FF3EA5"
} as const;

/** ComfyUI's link colours, one per kind of choice, so each dial's sockets
 *  read like a typed input. */
export const SOCKET_COLORS = {
  language: "#FFD500",
  bank: "#FFA931",
  style: "#64B5F6",
  acts: "#FF9CF9",
  machine: "#B39DDB"
} as const;

export type Tone = "normal" | "muted" | "alert" | "good";

export const FONT_FAMILY = "Segoe UI, Arial, sans-serif";
export const BLACK_FAMILY = "Segoe UI Black, Segoe UI, Arial, sans-serif";

export function escapeXml(raw: string): string {
  return raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function toSvgDataUri(svg: string): string {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function toneColor(tone: Tone): string {
  switch (tone) {
    case "alert":
      return PALETTE.bad;
    case "good":
      return PALETTE.lamp;
    case "muted":
      return PALETTE.muted;
    default:
      return PALETTE.ink;
  }
}

interface TextRun {
  x: number;
  y: number;
  text: string;
  size: number;
  fill: string;
  weight?: Weight;
  anchor?: "start" | "middle" | "end";
  letterSpacing?: number;
}

function textSvg(run: TextRun): string {
  const weight = run.weight ?? 400;
  const attrs = [
    `x="${run.x.toFixed(1)}"`,
    `y="${run.y.toFixed(1)}"`,
    `font-family="${weight === 900 ? BLACK_FAMILY : FONT_FAMILY}"`,
    `font-size="${run.size}"`,
    weight !== 400 ? `font-weight="${weight}"` : "",
    `fill="${run.fill}"`,
    run.anchor && run.anchor !== "start" ? `text-anchor="${run.anchor}"` : "",
    run.letterSpacing ? `letter-spacing="${run.letterSpacing}"` : ""
  ].filter(Boolean);
  return `<text ${attrs.join(" ")}>${escapeXml(run.text)}</text>`;
}

/** The icon's lettering: the word in ink over the same word in hot pink,
 *  nudged down and right. */
const SHADOW = 2.5;
function shadowedTextSvg(run: TextRun): string {
  return textSvg({ ...run, x: run.x + SHADOW, y: run.y + SHADOW, fill: PALETTE.pink }) + textSvg(run);
}

function clampShare(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
}

/** The page's background: deep purple under a faint square grid. */
function gridSvg(width: number, height: number, step: number): string {
  const lines: string[] = [];
  for (let x = step; x < width; x += step) {
    lines.push(`M${x} 0V${height}`);
  }
  for (let y = step; y < height; y += step) {
    lines.push(`M0 ${y}H${width}`);
  }
  return (
    `<rect width="${width}" height="${height}" fill="${PALETTE.bg}"/>` +
    `<path d="${lines.join("")}" stroke="${PALETTE.grid}" stroke-opacity="0.45" stroke-width="1" fill="none"/>`
  );
}

// -- a ComfyUI node: the touch-strip cell, or a key --------------------------

/** Where a node sits on its face. The sockets sit on the node's edges at
 *  `socketY`, and the wire runs the face's whole width at that height; on the
 *  strip the four cells touch, so each wire meets the next cell's. */
export interface NodeGeometry {
  width: number;
  height: number;
  nodeX: number;
  nodeY: number;
  nodeW: number;
  nodeH: number;
  titleH: number;
  /** The title's largest and smallest sizes, and its letter-spacing. */
  titleSize: number;
  titleMin: number;
  titleSpacing: number;
  socketY: number;
  pad: number;
  /** The middle of the value's block, and its largest sizes. */
  valueMid: number;
  valueMax: number;
  valueWrapMax: number;
  valueMin: number;
  /** The hint: the baseline of its last line, its sizes, and how many
   *  lines it may take (a key is narrow, so its hint may wrap). */
  hintY: number;
  hintMax: number;
  hintMin: number;
  hintLines: number;
  /** The grid's square. */
  grid: number;
}

/** A Stream Deck + touch-strip cell, 200x100. */
export const CELL: NodeGeometry = {
  width: 200,
  height: 100,
  nodeX: 10,
  nodeY: 4,
  nodeW: 180,
  nodeH: 92,
  titleH: 24,
  titleSize: 12,
  titleMin: 10,
  titleSpacing: 0.5,
  socketY: 62,
  pad: 12,
  valueMid: 49,
  valueMax: 25,
  valueWrapMax: 17,
  valueMin: 13,
  hintY: 90,
  hintMax: 12,
  hintMin: 11,
  hintLines: 2,
  grid: 20
};

/** A key, 144x144: the same node, taller, for decks without dials. */
export const NODE_KEY: NodeGeometry = {
  width: 144,
  height: 144,
  nodeX: 9,
  nodeY: 8,
  nodeW: 126,
  nodeH: 128,
  titleH: 30,
  titleSize: 14,
  titleMin: 11,
  titleSpacing: 0.3,
  socketY: 92,
  pad: 10,
  valueMid: 70,
  valueMax: 26,
  valueWrapMax: 20,
  valueMin: 14,
  hintY: 129,
  hintMax: 15,
  hintMin: 14,
  hintLines: 2,
  grid: 24
};

export interface DialCellInput {
  title: string;
  value: string;
  /** The small line under the value: what a press does, or why it cannot. */
  hint?: string;
  tone?: Tone;
  /** "3/11": where the pick sits in its list. */
  position?: string;
  /** The control is resting on a row a press has not taken yet: drawn as a
   *  selected node, with a white outline. */
  pending?: boolean;
  /** 0..1 while an episode is being made: the node is outlined green and a
   *  bar runs along its title, as ComfyUI draws the node it is running. */
  progress?: number | null;
  /** The sockets' colour (SOCKET_COLORS). */
  socket?: string;
}

/** The text box inside a node's body. */
export function nodeTextBox(g: NodeGeometry): { left: number; right: number } {
  return { left: g.nodeX + g.pad, right: g.nodeX + g.nodeW - g.pad };
}

/** A node: title bar with the field's name and the pick's place in its list,
 *  the pick in large letters (two lines if it needs them), a hint under it,
 *  typed sockets on both edges and the wire through them. */
export function renderNodeSvg(input: DialCellInput, g: NodeGeometry): string {
  const P = PALETTE;
  const { width, height, nodeX, nodeY, nodeW, nodeH, titleH, socketY } = g;
  const tone = input.tone ?? "normal";
  const progress = clampShare(input.progress);
  const running = progress !== null;
  const socket = input.socket ?? P.accentText;
  const box = nodeTextBox(g);
  const inner = box.right - box.left;
  const parts: string[] = [];

  // The wire behind everything, then the node over it.
  parts.push(gridSvg(width, height, g.grid));
  parts.push(
    `<line x1="0" y1="${socketY}" x2="${width}" y2="${socketY}" stroke="${P.bg}" stroke-width="7"/>` +
      `<line x1="0" y1="${socketY}" x2="${width}" y2="${socketY}" stroke="${running ? P.lamp : socket}" stroke-width="3.5" stroke-opacity="${tone === "muted" && !running ? 0.45 : 0.95}"/>`
  );
  const outline = input.pending ? P.ink : running ? P.lamp : P.line;
  const outlineWidth = input.pending || running ? 2.5 : 1.5;
  const r = 9;
  const titlePath =
    `M${nodeX} ${nodeY + titleH}V${nodeY + r}Q${nodeX} ${nodeY} ${nodeX + r} ${nodeY}` +
    `H${nodeX + nodeW - r}Q${nodeX + nodeW} ${nodeY} ${nodeX + nodeW} ${nodeY + r}V${nodeY + titleH}Z`;
  parts.push(
    `<g${tone === "muted" && !running ? ` opacity="0.62"` : ""}>` +
      `<rect x="${nodeX}" y="${nodeY}" width="${nodeW}" height="${nodeH}" rx="${r}" fill="${P.panel}"/>` +
      `<path d="${titlePath}" fill="${P.strip}"/>` +
      `<line x1="${nodeX}" y1="${nodeY + titleH}" x2="${nodeX + nodeW}" y2="${nodeY + titleH}" stroke="${P.line}" stroke-width="1"/>` +
      `<circle cx="${nodeX + 12}" cy="${nodeY + titleH / 2}" r="4" fill="${socket}"/>`
  );

  // The title bar: the field's name, and the pick's place in its list when
  // there is room for both. The name shrinks before the place is dropped,
  // and is cut only if it cannot fit alone at its smallest size.
  const titleLeft = nodeX + 21;
  const titleRoom = box.right - titleLeft;
  const titleText = input.title.toUpperCase();
  const position = input.position ?? "";
  const positionStyle = { size: g.titleSize - 1, weight: 700 as Weight };
  const positionWidth = position ? textWidth(position, positionStyle) + 6 : 0;
  const titleStyleAt = (size: number) => ({ size, weight: 700 as Weight, letterSpacing: g.titleSpacing });
  const fitsAt = (size: number, reserve: number): boolean =>
    textWidth(titleText, titleStyleAt(size)) <= (titleRoom - reserve) * FIT_SAFETY;
  let titleSize = 0;
  let showPosition = false;
  for (let size = g.titleSize; size >= g.titleMin && !titleSize; size -= 1) {
    if (position && fitsAt(size, positionWidth)) {
      titleSize = size;
      showPosition = true;
    }
  }
  for (let size = g.titleSize; size >= g.titleMin && !titleSize; size -= 1) {
    if (fitsAt(size, 0)) {
      titleSize = size;
    }
  }
  const titleStyle = titleStyleAt(titleSize || g.titleMin);
  const title = titleSize ? titleText : truncateToWidth(titleText, titleRoom * FIT_SAFETY, titleStyle);
  const titleBaseline = nodeY + titleH / 2 + g.titleSize * 0.36;
  parts.push(textSvg({ x: titleLeft, y: titleBaseline, text: title, ...titleStyle, fill: P.ink }));
  if (showPosition) {
    parts.push(textSvg({ x: box.right, y: titleBaseline, text: position, ...positionStyle, fill: P.accentText, anchor: "end" }));
  }

  // The value: one line as large as fits, else two smaller ones.
  const value = fitText(input.value || "—", {
    maxWidth: inner,
    maxSize: g.valueMax,
    minSize: g.valueMin,
    weight: 700,
    maxLines: 2,
    maxWrapSize: g.valueWrapMax
  });
  const lineHeight = Math.round(value.size * 1.1);
  const firstBaseline =
    value.lines.length > 1 ? g.valueMid - lineHeight / 2 + value.size * 0.36 : g.valueMid + value.size * 0.36;
  value.lines.forEach((line, i) => {
    parts.push(
      textSvg({
        x: width / 2,
        y: firstBaseline + i * lineHeight,
        text: line,
        size: value.size,
        weight: 700,
        fill: toneColor(tone),
        anchor: "middle"
      })
    );
  });

  if (input.hint) {
    const hint = fitHint(input.hint, inner, g);
    // The block sits on the hint's baseline: one line where the last would be.
    const hintLineHeight = Math.round(hint.size * 1.15);
    const firstHint = g.hintY - (hint.lines.length - 1) * hintLineHeight;
    hint.lines.forEach((line, i) => {
      parts.push(
        textSvg({
          x: width / 2,
          y: firstHint + i * hintLineHeight,
          text: line,
          size: hint.size,
          fill: input.pending ? P.ink : P.muted,
          anchor: "middle"
        })
      );
    });
  }
  parts.push(`</g>`);

  // The node's outline and sockets sit over the body, at full strength.
  parts.push(
    `<rect x="${nodeX}" y="${nodeY}" width="${nodeW}" height="${nodeH}" rx="${r}" fill="none" stroke="${outline}" stroke-width="${outlineWidth}"/>`
  );
  if (running) {
    const barW = (nodeW - 2 * r) * (progress ?? 0);
    parts.push(
      `<rect x="${nodeX + r}" y="${nodeY + titleH - 3}" width="${(nodeW - 2 * r).toFixed(1)}" height="3" fill="${P.bar}"/>` +
        `<rect x="${nodeX + r}" y="${nodeY + titleH - 3}" width="${barW.toFixed(1)}" height="3" fill="${P.lamp}"/>`
    );
  }
  for (const cx of [nodeX, nodeX + nodeW]) {
    parts.push(
      `<circle cx="${cx}" cy="${socketY}" r="6" fill="${running ? P.lamp : socket}" stroke="${P.bg}" stroke-width="2.5"/>`
    );
  }

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    ...parts,
    `</svg>`
  ].join("");
}

/** A node's hint. A "\n" in it marks where its lines break -- each phrase
 *  stays whole -- and every line is fitted at the one size all of them fit
 *  at; without one, the words wrap wherever they fall. */
function fitHint(hint: string, inner: number, g: NodeGeometry): { lines: string[]; size: number } {
  const given = hint.split("\n").map((l) => l.trim()).filter(Boolean);
  if (given.length < 2) {
    const fitted = fitText(hint, { maxWidth: inner, maxSize: g.hintMax, minSize: g.hintMin, maxLines: g.hintLines });
    return { lines: fitted.lines, size: fitted.size };
  }
  const lines = given.slice(0, g.hintLines);
  for (let size = g.hintMax; size >= g.hintMin; size -= 1) {
    if (lines.every((l) => textWidth(l, { size }) <= inner * FIT_SAFETY)) {
      return { lines, size };
    }
  }
  const size = g.hintMin;
  return { lines: lines.map((l) => truncateToWidth(l, inner * FIT_SAFETY, { size })), size };
}

/** A dial's touch-strip cell. */
export function renderDialCellSvg(input: DialCellInput): string {
  return renderNodeSvg(input, CELL);
}

/** The same node on a key, for decks without dials. */
export function renderNodeKeySvg(input: DialCellInput): string {
  return renderNodeSvg(input, NODE_KEY);
}

// -- the Make key ----------------------------------------------------------

/** The ring: centre, the radius of its stroke's middle, and its width. Text
 *  stays inside `inner`, the clear disc the ring encloses, `pad` from it. */
export const KEY = { size: 144, cx: 72, cy: 72, ring: 63, stroke: 8, inner: 58, pad: 4 } as const;

/** Half the width the key's clear disc has for a row whose glyphs run from
 *  `top` to `bottom` (y, in px). */
export function keyHalfChord(top: number, bottom: number): number {
  const dy = Math.max(Math.abs(top - KEY.cy), Math.abs(bottom - KEY.cy));
  if (dy >= KEY.inner) {
    return 0;
  }
  return Math.max(0, Math.sqrt(KEY.inner * KEY.inner - dy * dy) - KEY.pad);
}

/** How far a line of text reaches above and below its baseline, as a share
 *  of its size (Segoe UI: cap height 0.70, descender 0.21). */
export const ASCENT = 0.72;
export const DESCENT = 0.22;

export interface MakeKeyInput {
  /** The big word or number in the middle, in the icon's lettering. */
  headline?: string;
  /** Under the middle: a string wraps onto up to two lines; an array, or a
   *  "\n" in the string, is one line per item. */
  caption?: string | string[];
  tone?: Tone;
  /** 0..1 fills the ring; absent leaves it empty. */
  progress?: number | null;
  glyph?: "check" | "cross" | "pause" | "dots" | null;
  /** The icon's green waveform under the middle. */
  wave?: boolean;
}

function ringSvg(share: number | null): string {
  const { cx, cy, ring, stroke } = KEY;
  const track = `<circle cx="${cx}" cy="${cy}" r="${ring}" fill="none" stroke="${PALETTE.bar}" stroke-width="${stroke}"/>`;
  if (share === null || share <= 0.001) {
    return track;
  }
  // The lamp's glow: a wide, faint stroke under the bright one.
  const arc = (width: number, opacity: number): string => {
    if (share >= 0.999) {
      return `<circle cx="${cx}" cy="${cy}" r="${ring}" fill="none" stroke="${PALETTE.lamp}" stroke-width="${width}" stroke-opacity="${opacity}"/>`;
    }
    const a = share * 2 * Math.PI - Math.PI / 2;
    const x = cx + ring * Math.cos(a);
    const y = cy + ring * Math.sin(a);
    const large = share > 0.5 ? 1 : 0;
    return (
      `<path d="M ${cx} ${cy - ring} A ${ring} ${ring} 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}" ` +
      `fill="none" stroke="${PALETTE.lamp}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-linecap="round"/>`
    );
  };
  return track + arc(stroke + 7, 0.25) + arc(stroke, 1);
}

/** The icon's waveform: flat chatter either side of one tall peak. */
function waveSvg(cy: number, halfWidth: number, color: string): string {
  const pts: string[] = [];
  const steps = 48;
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps; // 0..1 across
    const x = 72 - halfWidth + t * halfWidth * 2;
    const u = (t - 0.5) * 2; // -1..1
    const peak = Math.exp(-u * u * 40) * 12;
    const dip = Math.exp(-(u * u) * 9) * 5;
    const chatter = Math.sin(t * Math.PI * 14) * 1.6 * (1 - Math.exp(-u * u * 30));
    const y = cy - peak + dip * 0.6 + chatter;
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return `<polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
}

/** A glyph in a box centred on (72, cy), `h` px tall. */
function glyphSvg(glyph: NonNullable<MakeKeyInput["glyph"]>, color: string, cy: number, h: number): string {
  const s = h / 40;
  const X = (dx: number): string => (72 + dx * s).toFixed(1);
  const Y = (dy: number): string => (cy + dy * s).toFixed(1);
  const w = (n: number): string => (n * s).toFixed(1);
  switch (glyph) {
    case "check":
      return (
        `<polyline points="${X(-18)},${Y(1)} ${X(-5)},${Y(14)} ${X(19)},${Y(-14)}" fill="none" stroke="${color}" stroke-opacity="0.25" stroke-width="${w(15)}" stroke-linecap="round" stroke-linejoin="round"/>` +
        `<polyline points="${X(-18)},${Y(1)} ${X(-5)},${Y(14)} ${X(19)},${Y(-14)}" fill="none" stroke="${color}" stroke-width="${w(8)}" stroke-linecap="round" stroke-linejoin="round"/>`
      );
    case "cross":
      return (
        `<line x1="${X(-14)}" y1="${Y(-14)}" x2="${X(14)}" y2="${Y(14)}" stroke="${color}" stroke-width="${w(8)}" stroke-linecap="round"/>` +
        `<line x1="${X(14)}" y1="${Y(-14)}" x2="${X(-14)}" y2="${Y(14)}" stroke="${color}" stroke-width="${w(8)}" stroke-linecap="round"/>`
      );
    case "pause":
      return (
        `<rect x="${X(-14)}" y="${Y(-18)}" width="${w(10)}" height="${w(36)}" rx="${w(2.5)}" fill="${color}"/>` +
        `<rect x="${X(4)}" y="${Y(-18)}" width="${w(10)}" height="${w(36)}" rx="${w(2.5)}" fill="${color}"/>`
      );
    case "dots":
      return [-14, 0, 14]
        .map((dx) => `<circle cx="${X(dx)}" cy="${Y(0)}" r="${w(4.5)}" fill="${color}"/>`)
        .join("");
    default:
      return "";
  }
}

interface Row {
  top: number;
  bottom: number;
  baseline: number;
}

function rowAt(top: number, size: number): Row {
  return { top, bottom: top + size * (ASCENT + DESCENT), baseline: top + size * ASCENT };
}

/** The widest a row may be on the key: the disc's chord at that row, less
 *  `extra` (the headline's shadow). */
function rowWidth(row: Row, extra = 0): number {
  return Math.max(0, keyHalfChord(row.top, row.bottom) * 2 - extra);
}

/** The Make key: a ring that fills with the lamp's green as the episode is
 *  made, the icon's lettering or a glyph in the middle, the waveform under it
 *  when it is ready, and up to two caption lines, each fitted to the ring's
 *  clear disc at its own height. */
export function renderMakeKeySvg(input: MakeKeyInput): string {
  const P = PALETTE;
  const tone = input.tone ?? "normal";
  const color = toneColor(tone);
  // An array, or a "\n" in the string, sets the lines; otherwise it wraps.
  const rawCaption = input.caption ?? "";
  const explicitLines = Array.isArray(rawCaption) || rawCaption.includes("\n");
  const captionItems = (Array.isArray(rawCaption) ? rawCaption : explicitLines ? rawCaption.split("\n") : [rawCaption])
    .map((c) => c.trim())
    .filter(Boolean);
  const hasCaption = captionItems.length > 0;

  // Rows from the top: the middle (headline or glyph), the wave, the caption.
  const middleCentre = hasCaption ? (input.wave ? 52 : 56) : 72;
  const parts: string[] = [];

  let middleBottom = middleCentre;
  if (input.glyph) {
    const h = hasCaption ? 34 : 46;
    parts.push(glyphSvg(input.glyph, color, middleCentre, h));
    middleBottom = middleCentre + h / 2;
  } else if (input.headline) {
    let placed = false;
    for (let size = 36; size >= 16 && !placed; size -= 1) {
      const row = rowAt(middleCentre - (size * ASCENT) / 2, size);
      if (textWidth(input.headline, { size, weight: 900 }) <= rowWidth(row, SHADOW) * FIT_SAFETY) {
        parts.push(
          shadowedTextSvg({ x: 72, y: row.baseline, text: input.headline, size, weight: 900, fill: color, anchor: "middle" })
        );
        middleBottom = row.baseline + SHADOW;
        placed = true;
      }
    }
    if (!placed) {
      const size = 16;
      const row = rowAt(middleCentre - (size * ASCENT) / 2, size);
      const text = truncateToWidth(input.headline, rowWidth(row, SHADOW) * FIT_SAFETY, { size, weight: 900 });
      parts.push(shadowedTextSvg({ x: 72, y: row.baseline, text, size, weight: 900, fill: color, anchor: "middle" }));
      middleBottom = row.baseline + SHADOW;
    }
  }

  let captionTop = middleBottom + 7;
  if (input.wave) {
    const waveY = middleBottom + 15;
    parts.push(waveSvg(waveY, rowWidth({ top: waveY - 12, bottom: waveY + 4, baseline: waveY }) / 2, P.lamp));
    captionTop = waveY + 9;
  }

  if (hasCaption) {
    const captionFill = tone === "alert" ? P.ink : P.muted;
    const lineGap = 1.16;
    let lines: { text: string; size: number }[] = [];
    if (explicitLines) {
      // One line per item, all at one size: the largest every item fits at.
      let size = 15;
      for (; size > 12; size -= 1) {
        const fits = captionItems.slice(0, 2).every((item, i) => {
          const row = rowAt(captionTop + i * size * lineGap, size);
          return textWidth(item, { size }) <= rowWidth(row) * FIT_SAFETY;
        });
        if (fits) {
          break;
        }
      }
      lines = captionItems.slice(0, 2).map((item, i) => {
        const row = rowAt(captionTop + i * size * lineGap, size);
        return { text: truncateToWidth(item, rowWidth(row) * FIT_SAFETY, { size }), size };
      });
    } else {
      const text = captionItems[0];
      let fitted: ReturnType<typeof fitText> | null = null;
      for (let size = 15; size >= 13 && !fitted; size -= 1) {
        const attempt = fitText(text, {
          maxWidth: (i) => rowWidth(rowAt(captionTop + i * size * lineGap, size)),
          maxSize: size,
          minSize: size,
          maxLines: 2
        });
        if (!attempt.truncated) {
          fitted = attempt;
        }
      }
      if (!fitted) {
        const size = 13;
        fitted = fitText(text, {
          maxWidth: (i) => rowWidth(rowAt(captionTop + i * size * lineGap, size)),
          maxSize: size,
          minSize: size,
          maxLines: 2
        });
      }
      lines = fitted.lines.map((t) => ({ text: t, size: fitted!.size }));
    }
    lines.forEach((line, i) => {
      const row = rowAt(captionTop + i * line.size * lineGap, line.size);
      parts.push(textSvg({ x: 72, y: row.baseline, text: line.text, size: line.size, fill: captionFill, anchor: "middle" }));
    });
  }

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${KEY.size}" height="${KEY.size}" viewBox="0 0 ${KEY.size} ${KEY.size}">`,
    gridSvg(KEY.size, KEY.size, 24),
    `<circle cx="${KEY.cx}" cy="${KEY.cy}" r="${KEY.ring}" fill="${P.panel}" fill-opacity="0.85"/>`,
    ringSvg(clampShare(input.progress)),
    ...parts,
    `</svg>`
  ].join("");
}
