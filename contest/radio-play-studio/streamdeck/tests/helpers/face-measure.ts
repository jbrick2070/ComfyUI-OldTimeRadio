// SPDX-License-Identifier: MIT
// Reading a face back and measuring it: every <text> the renderers write,
// measured in Segoe UI (the same advances the fitter uses, read from the real
// font), checked against its box -- a node's body, or the Make key's clear
// disc at that line's height -- and against the rows around it.
import { expect } from "vitest";
import { ENGINE_NAMES, WAY_WORDS } from "../../src/actions/faces.js";
import { ASCENT, DESCENT, KEY, keyHalfChord, NodeGeometry, PALETTE } from "../../src/gfx/svg-renderer.js";
import { textWidth, Weight } from "../../src/gfx/text-fit.js";

export interface TextEl {
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

export function texts(svg: string): TextEl[] {
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

export function span(t: TextEl): { left: number; right: number; top: number; bottom: number } {
  const w = textWidth(t.text, { size: t.size, weight: t.weight, letterSpacing: t.spacing });
  const left = t.anchor === "middle" ? t.x - w / 2 : t.anchor === "end" ? t.x - w : t.x;
  return { left, right: left + w, top: t.y - t.size * ASCENT, bottom: t.y + t.size * DESCENT };
}

export function checkNode(svg: string, g: NodeGeometry, label: string): TextEl[] {
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

export function checkKey(svg: string, label: string): TextEl[] {
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
export const STYLES = [
  "roll (any style)", "anime", "archival_documentary", "cartoon", "paper_origami", "recur_frac",
  "sci_fi_radio", "shakespeare_stage_realism", "storybook_engraving", "video_art", "visual_storybased"
];
export const BANKS = ["roll (any eligible bank)", "media_archive", "original", "scifi_news_pro", "public_domain", "shakespeare"];
export const LANGUAGES = ["roll (any language)", "Off", "English", "Portuguese", "Mandarin", "Japanese", "Spanish"];
/** Every name a menu control gives a way: each engine, and each way's word. */
export const WAYS = [...new Set([...Object.values(ENGINE_NAMES), ...Object.values(WAY_WORDS)])];
/** A server reason longer than any face: the one text a face may cut. */
export const LONG_ERROR =
  "project 2026-10-28-wanderer is waiting to hear how its last run ended; open it and look again, or forget that run";
