// SPDX-License-Identifier: MIT
// The faces: a 200x100 touch-strip cell for the dials and a 144x144 key for
// Make. Plain SVG strings, so the tests can read them.

export const PALETTE = {
  bgTop: "#140F0A",
  bgBottom: "#1F1710",
  track: "#3A2E22",
  amber: "#F59E0B",
  amberDeep: "#B45309",
  cream: "#FDF6E3",
  muted: "#C9B79C",
  dim: "#7C6A55",
  green: "#22C55E",
  red: "#EF4444"
} as const;

export type Tone = "normal" | "muted" | "alert" | "good";

export function escapeXml(raw: string): string {
  return raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function truncateLabel(text: string, maxChars: number): string {
  const clean = text.trim();
  if (clean.length <= maxChars) {
    return clean;
  }
  return clean.slice(0, Math.max(1, maxChars - 1)).trimEnd() + "…";
}

export function toSvgDataUri(svg: string): string {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const FONT = "font-family=\"Segoe UI, Arial, sans-serif\"";

function toneColor(tone: Tone): string {
  switch (tone) {
    case "alert":
      return PALETTE.red;
    case "good":
      return PALETTE.green;
    case "muted":
      return PALETTE.dim;
    default:
      return PALETTE.cream;
  }
}

export interface DialCellInput {
  title: string;
  value: string;
  hint?: string;
  tone?: Tone;
  /** 0..1 draws a bar along the bottom; absent draws none. */
  progress?: number | null;
}

/** A dial's touch-strip cell: the field's name, the pick in big letters, a
 *  hint underneath, and a progress bar while an episode is being made. */
export function renderDialCellSvg(input: DialCellInput): string {
  const P = PALETTE;
  const tone = input.tone ?? "normal";
  const value = input.value.trim();
  const valueSize = value.length > 14 ? 15 : value.length > 9 ? 19 : 24;
  const safeValue = escapeXml(truncateLabel(value || "—", 22));
  const safeTitle = escapeXml(truncateLabel(input.title, 24));
  const safeHint = escapeXml(truncateLabel(input.hint ?? "", 30));
  const progress =
    typeof input.progress === "number" && Number.isFinite(input.progress)
      ? Math.max(0, Math.min(1, input.progress))
      : null;
  const bar =
    progress === null
      ? ""
      : `<rect x="0" y="95" width="200" height="5" fill="${P.track}"/>` +
        `<rect x="0" y="95" width="${(progress * 200).toFixed(1)}" height="5" fill="${P.amber}"/>`;
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">`,
    `<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">`,
    `<stop offset="0" stop-color="${P.bgTop}"/><stop offset="1" stop-color="${P.bgBottom}"/>`,
    `</linearGradient></defs>`,
    `<rect width="200" height="100" fill="url(#bg)"/>`,
    `<text x="10" y="20" ${FONT} font-size="11" fill="${P.muted}" letter-spacing="0.5">${safeTitle.toUpperCase()}</text>`,
    `<text x="100" y="60" ${FONT} font-size="${valueSize}" font-weight="700" fill="${toneColor(tone)}" text-anchor="middle">${safeValue}</text>`,
    safeHint
      ? `<text x="100" y="86" ${FONT} font-size="10" fill="${P.dim}" text-anchor="middle">${safeHint}</text>`
      : "",
    bar,
    `</svg>`
  ].join("");
}

export interface MakeKeyInput {
  /** The big word or number in the middle. */
  headline: string;
  /** A line under it; the reason, the phase, the minutes left. */
  caption?: string;
  tone?: Tone;
  /** 0..1 draws the ring; absent draws a resting ring. */
  progress?: number | null;
  /** The glyph in the middle instead of the headline. */
  glyph?: "play" | "check" | "cross" | "pause" | null;
}

function ringPath(cx: number, cy: number, r: number, share: number): string {
  if (share >= 0.999) {
    return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${PALETTE.amber}" stroke-width="9"/>`;
  }
  if (share <= 0.001) {
    return "";
  }
  const a = share * 2 * Math.PI - Math.PI / 2;
  const x = cx + r * Math.cos(a);
  const y = cy + r * Math.sin(a);
  const large = share > 0.5 ? 1 : 0;
  return (
    `<path d="M ${cx} ${cy - r} A ${r} ${r} 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}" ` +
    `fill="none" stroke="${PALETTE.amber}" stroke-width="9" stroke-linecap="round"/>`
  );
}

function glyphSvg(glyph: MakeKeyInput["glyph"], color: string): string {
  switch (glyph) {
    case "play":
      return `<polygon points="60,48 60,96 102,72" fill="${color}"/>`;
    case "check":
      return `<polyline points="48,74 66,92 98,54" fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>`;
    case "cross":
      return (
        `<line x1="54" y1="54" x2="90" y2="90" stroke="${color}" stroke-width="10" stroke-linecap="round"/>` +
        `<line x1="90" y1="54" x2="54" y2="90" stroke="${color}" stroke-width="10" stroke-linecap="round"/>`
      );
    case "pause":
      return `<rect x="54" y="52" width="12" height="40" rx="3" fill="${color}"/><rect x="78" y="52" width="12" height="40" rx="3" fill="${color}"/>`;
    default:
      return "";
  }
}

/** The Make key: a ring that fills as the episode is made, with a word, a
 *  number or a glyph in the middle and one line under it. */
export function renderMakeKeySvg(input: MakeKeyInput): string {
  const P = PALETTE;
  const tone = input.tone ?? "normal";
  const color = toneColor(tone);
  const progress =
    typeof input.progress === "number" && Number.isFinite(input.progress)
      ? Math.max(0, Math.min(1, input.progress))
      : null;
  const headline = escapeXml(truncateLabel(input.headline, 9));
  const headSize = input.headline.length > 6 ? 20 : input.headline.length > 4 ? 26 : 32;
  const caption = escapeXml(truncateLabel(input.caption ?? "", 16));
  const middle = input.glyph
    ? glyphSvg(input.glyph, color)
    : `<text x="72" y="${caption ? 78 : 83}" ${FONT} font-size="${headSize}" font-weight="700" fill="${color}" text-anchor="middle">${headline}</text>`;
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="144" height="144" viewBox="0 0 144 144">`,
    `<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">`,
    `<stop offset="0" stop-color="${P.bgTop}"/><stop offset="1" stop-color="${P.bgBottom}"/>`,
    `</linearGradient></defs>`,
    `<rect width="144" height="144" rx="18" fill="url(#bg)"/>`,
    `<circle cx="72" cy="72" r="58" fill="none" stroke="${P.track}" stroke-width="9"/>`,
    progress === null ? "" : ringPath(72, 72, 58, progress),
    middle,
    caption
      ? `<text x="72" y="${input.glyph ? 122 : 104}" ${FONT} font-size="12" fill="${P.muted}" text-anchor="middle">${caption}</text>`
      : "",
    `</svg>`
  ].join("");
}
