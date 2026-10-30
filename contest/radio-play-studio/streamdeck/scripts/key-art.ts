// SPDX-License-Identifier: MIT
// The six static key faces for the launcher and media keys beside the plugin's
// own actions: STUDIO, LOGS, GRAPH, ESPAÑOL, REEL and CLIPS. Original glyphs in
// the palette of jbrick2070/streamdeck-fadester (at 7cf72073e1a5498b4d756d7a47d24785acbd5b79).
// MAKE, Machine and the dials are drawn live by the plugin and are not here.
// `npm run key-art` writes these to key-art/; tests/key-art.test.ts keeps the
// committed SVGs equal to what this module draws.

export const KEY_ART_COLORS = {
  bg: "#08070B",
  panel: "#24152F",
  purple: "#9146FF",
  light: "#C084FC",
  yellow: "#FFF04B",
  white: "#FAF7FF"
} as const;

const C = KEY_ART_COLORS;

export interface KeyArt {
  /** File name without extension. */
  name: string;
  /** The label baked into the face; the key's own title is hidden. */
  label: string;
  /** What the key opens in the demo layout. */
  opens: string;
  glyph: string;
}

export const KEY_ART: readonly KeyArt[] = [
  {
    name: "studio",
    label: "STUDIO",
    opens: "the Studio page, http://127.0.0.1:8765/",
    glyph: `
    <rect x="45" y="17" width="30" height="43" rx="14" fill="${C.purple}" stroke="${C.light}" stroke-width="3.5"/>
    <path d="M53 29h14m-14 9h14m-14 9h14" stroke="${C.yellow}" stroke-width="4"/>
    <path d="M35 47v4a25 25 0 0 0 50 0v-4M60 76v-1m-14 2h28" stroke="${C.light}" stroke-width="4.5"/>
    <path d="M60 76v-8" stroke="${C.light}" stroke-width="4.5"/>`
  },
  {
    name: "logblaster",
    label: "LOGS",
    opens: "LogBlaster, http://127.0.0.1:8188/logblaster",
    glyph: `
    <path d="M24 40h14l20-15v48L38 58H24Z" fill="${C.purple}" stroke="${C.light}" stroke-width="3.5"/>
    <path d="M34 61l4 15h12l-4-10" fill="${C.panel}" stroke="${C.light}" stroke-width="3.5"/>
    <path d="M71 31h22M71 48h28M71 65h18" stroke="${C.yellow}" stroke-width="5"/>`
  },
  {
    name: "comfyui",
    label: "GRAPH",
    opens: "the ComfyUI graph, http://127.0.0.1:8188/",
    glyph: `
    <path d="M45 34h24M82 46v18H60" stroke="${C.light}" stroke-width="4"/>
    <rect x="20" y="23" width="28" height="23" rx="5" fill="${C.purple}" stroke="${C.light}" stroke-width="3.5"/>
    <rect x="69" y="23" width="28" height="23" rx="5" fill="${C.purple}" stroke="${C.light}" stroke-width="3.5"/>
    <rect x="33" y="59" width="29" height="23" rx="5" fill="${C.yellow}" stroke="${C.yellow}" stroke-width="3.5"/>`
  },
  {
    name: "spanish",
    label: "ESPAÑOL",
    opens: "a Spanish-language episode (a local video file)",
    glyph: `
    <path d="M28 76V45a32 27 0 0 1 64 0v31Z" fill="${C.panel}" stroke="${C.light}" stroke-width="4"/>
    <path d="M40 38h25M40 48h25M40 58h25" stroke="${C.purple}" stroke-width="5"/>
    <circle cx="79" cy="43" r="5" fill="${C.yellow}"/>
    <path d="M48 59h29v23H48Z" fill="${C.purple}" stroke="${C.light}" stroke-width="3"/>
    <path d="m57 65 11 6-11 6Z" fill="${C.yellow}"/>
    <path d="M32 79h9m41 0h8" stroke="${C.light}" stroke-width="4"/>`
  },
  {
    name: "reel",
    label: "REEL",
    opens: "a highlight reel (a local video file)",
    glyph: `
    <circle cx="60" cy="49" r="32" fill="${C.purple}" stroke="${C.light}" stroke-width="3.5"/>
    <g fill="${C.bg}" stroke="none"><circle cx="60" cy="28" r="6"/><circle cx="80" cy="43" r="6"/><circle cx="72" cy="66" r="6"/><circle cx="48" cy="66" r="6"/><circle cx="40" cy="43" r="6"/></g>
    <path d="m54 40 15 9-15 9Z" fill="${C.yellow}"/>`
  },
  {
    name: "clips",
    label: "CLIPS",
    opens: "a folder of clips",
    glyph: `
    <rect x="40" y="23" width="28" height="38" rx="3" fill="${C.panel}" stroke="${C.light}" stroke-width="3" transform="rotate(-14 54 42)"/>
    <rect x="66" y="22" width="25" height="39" rx="3" fill="${C.panel}" stroke="${C.light}" stroke-width="3" transform="rotate(12 78 42)"/>
    <path d="M23 44V33h25l10 11" fill="${C.yellow}" stroke="${C.yellow}" stroke-width="3.5"/>
    <path d="M22 45h76l-8 34H30Z" fill="${C.purple}" stroke="${C.light}" stroke-width="4"/>
    <path d="m54 53 14 9-14 9Z" fill="${C.yellow}"/>`
  }
];

/** One face as a standalone SVG, 1024 px at its natural size (viewBox 120). */
export function keyArtSvg(art: KeyArt): string {
  const size = art.name === "spanish" ? 16.5 : 18;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 120 120"><title>${art.label}</title>
  <rect width="120" height="120" fill="${C.bg}"/>
  <rect x="3" y="3" width="114" height="114" rx="14" fill="${C.bg}" stroke="${C.purple}" stroke-width="3"/>
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">${art.glyph}</g>
  <text x="60" y="103" text-anchor="middle" fill="${C.white}" font-family="Segoe UI" font-size="${size}" font-weight="800">${art.label}</text></svg>`;
}

/** The contact sheet: all six large, then a row at 72 px to check they still read. */
export function keyArtPreviewSvg(): string {
  const inner = (art: KeyArt) => keyArtSvg(art).replace(/<svg[^>]*>|<\/svg>/g, "");
  const large = KEY_ART.map(
    (art, i) => `<g transform="translate(${36 + (i % 3) * 252},${76 + Math.floor(i / 3) * 244}) scale(1.8)">${inner(art)}</g>`
  ).join("");
  const small = KEY_ART.map((art, i) => `<g transform="translate(${36 + i * 122},600) scale(0.6)">${inner(art)}</g>`).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="792" height="714" viewBox="0 0 792 714"><rect width="792" height="714" fill="#141118"/><text x="36" y="39" fill="${C.white}" font-family="Segoe UI" font-size="22" font-weight="700">RADIO PLAY STUDIO · FADESTER STYLE</text>${large}<text x="36" y="582" fill="#CFC0DF" font-family="Segoe UI" font-size="14">72 px readability check</text>${small}</svg>`;
}
