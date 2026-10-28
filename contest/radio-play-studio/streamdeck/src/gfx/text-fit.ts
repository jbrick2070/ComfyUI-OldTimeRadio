// SPDX-License-Identifier: MIT
// Making a string fit its box on a face. Widths come from the real Segoe UI
// advances (font-metrics.ts), so "fits" is measured, not guessed: a face
// shrinks its text, then wraps it, and only as a last resort cuts it with an
// ellipsis.
import {
  BLACK_FALLBACK,
  BLACK_WIDTHS,
  BOLD_FALLBACK,
  BOLD_WIDTHS,
  REGULAR_FALLBACK,
  REGULAR_WIDTHS
} from "./font-metrics.js";

/** 400 regular, 700 bold, 900 black (Segoe UI Black, the lettering of the
 *  OTR icon and the page's masthead). */
export type Weight = 400 | 700 | 900;

/** The share of a box a line may fill: room for the renderer's own rounding,
 *  and for Arial standing in where Segoe UI is missing (Arial runs wider). */
export const FIT_SAFETY = 0.9;

export interface TextStyle {
  size: number;
  weight?: Weight;
  /** Extra space after every character, as SVG letter-spacing. */
  letterSpacing?: number;
}

/** The width in pixels `text` takes in Segoe UI at this style. */
export function textWidth(text: string, style: TextStyle): number {
  const weight = style.weight ?? 400;
  const table = weight === 900 ? BLACK_WIDTHS : weight === 700 ? BOLD_WIDTHS : REGULAR_WIDTHS;
  const fallback = weight === 900 ? BLACK_FALLBACK : weight === 700 ? BOLD_FALLBACK : REGULAR_FALLBACK;
  let units = 0;
  let count = 0;
  for (const ch of text) {
    units += table[ch] ?? fallback;
    count += 1;
  }
  return (units / 1000) * style.size + (style.letterSpacing ?? 0) * count;
}

export interface FitOptions {
  /** The widest a line may be at line `index` (0-based) of `lines`. */
  maxWidth: number | ((index: number, lines: number) => number);
  maxSize: number;
  minSize: number;
  weight?: Weight;
  letterSpacing?: number;
  /** Lines the text may wrap onto once one line will not do at minSize. */
  maxLines?: number;
  /** The largest size a wrapped block may use (wrapping at the single-line
   *  maximum would make a two-line block taller than the box). */
  maxWrapSize?: number;
}

export interface Fitted {
  lines: string[];
  size: number;
  /** True when the text had to be cut to fit. */
  truncated: boolean;
}

function limit(opts: FitOptions, index: number, lines: number): number {
  const raw = typeof opts.maxWidth === "function" ? opts.maxWidth(index, lines) : opts.maxWidth;
  return Math.max(0, raw * FIT_SAFETY);
}

function style(opts: FitOptions, size: number): TextStyle {
  return { size, weight: opts.weight, letterSpacing: opts.letterSpacing };
}

/** Greedy word wrap into at most `lines` lines; null when the words do not
 *  all fit. */
function wrap(words: string[], opts: FitOptions, size: number, lines: number): string[] | null {
  const out: string[] = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (textWidth(candidate, style(opts, size)) <= limit(opts, out.length, lines)) {
      current = candidate;
      continue;
    }
    if (!current) {
      return null; // one word wider than the line
    }
    out.push(current);
    if (out.length >= lines) {
      return null;
    }
    current = word;
    if (textWidth(current, style(opts, size)) > limit(opts, out.length, lines)) {
      return null;
    }
  }
  if (current) {
    out.push(current);
  }
  return out.length <= lines ? out : null;
}

/** Cut `text` to the widest prefix that, with an ellipsis, fits `width`. */
export function truncateToWidth(text: string, width: number, s: TextStyle): string {
  if (textWidth(text, s) <= width) {
    return text;
  }
  const chars = Array.from(text);
  let lo = 0;
  let hi = chars.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    const head = chars.slice(0, mid).join("").trimEnd() + "…";
    if (textWidth(head, s) <= width) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo === 0 ? "…" : chars.slice(0, lo).join("").trimEnd() + "…";
}

/** The largest size, and the fewest lines, at which `text` fits. */
export function fitText(text: string, opts: FitOptions): Fitted {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) {
    return { lines: [], size: opts.maxSize, truncated: false };
  }
  for (let size = opts.maxSize; size >= opts.minSize; size -= 1) {
    if (textWidth(clean, style(opts, size)) <= limit(opts, 0, 1)) {
      return { lines: [clean], size, truncated: false };
    }
  }
  const maxLines = Math.max(1, opts.maxLines ?? 1);
  const words = clean.split(" ");
  for (let lines = 2; lines <= maxLines; lines += 1) {
    for (let size = Math.min(opts.maxWrapSize ?? opts.maxSize, opts.maxSize); size >= opts.minSize; size -= 1) {
      const wrapped = wrap(words, opts, size, lines);
      if (wrapped) {
        return { lines: wrapped, size, truncated: false };
      }
    }
  }
  // It does not fit whole: fill the lines at the smallest size and cut the last.
  const size = opts.minSize;
  const s = style(opts, size);
  const out: string[] = [];
  let rest = words;
  for (let i = 0; i < maxLines && rest.length > 0; i += 1) {
    const width = limit(opts, i, maxLines);
    if (i === maxLines - 1) {
      out.push(truncateToWidth(rest.join(" "), width, s));
      rest = [];
      break;
    }
    let take = 0;
    while (take < rest.length && textWidth(rest.slice(0, take + 1).join(" "), s) <= width) {
      take += 1;
    }
    if (take === 0) {
      out.push(truncateToWidth(rest[0], width, s));
      rest = rest.slice(1);
    } else {
      out.push(rest.slice(0, take).join(" "));
      rest = rest.slice(take);
    }
  }
  return { lines: out, size, truncated: true };
}
