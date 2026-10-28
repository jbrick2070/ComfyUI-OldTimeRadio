// SPDX-License-Identifier: MIT
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const PLUGIN_DIR = join(ROOT, "com.jbrick2070.otr.sdPlugin");

function walkFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (["node_modules", ".git", "bin", "release", "logs"].includes(entry.name)) {
      continue;
    }
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walkFiles(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

describe("manifest, layout, icons and repository hygiene", () => {
  it("declares SDK 3, Node 24, Stream Deck 7.1 and the three actions with their files", () => {
    const manifest = JSON.parse(readFileSync(join(PLUGIN_DIR, "manifest.json"), "utf8")) as Record<string, unknown>;
    expect(manifest.UUID).toBe("com.jbrick2070.otr");
    expect(manifest.SDKVersion).toBe(3);
    expect((manifest.Nodejs as { Version: string }).Version).toBe("24");
    expect((manifest.Software as { MinimumVersion: string }).MinimumVersion).toBe("7.1");
    const actions = manifest.Actions as Array<Record<string, unknown>>;
    expect(actions.map((a) => a.UUID)).toEqual([
      "com.jbrick2070.otr.choice",
      "com.jbrick2070.otr.machine",
      "com.jbrick2070.otr.make"
    ]);
    // Choice and Machine go on a dial or, on a deck without dials, a key.
    expect(actions[0].Controllers).toEqual(["Encoder", "Keypad"]);
    expect(actions[1].Controllers).toEqual(["Encoder", "Keypad"]);
    expect(actions[2].Controllers).toEqual(["Keypad"]);
    for (const a of actions) {
      expect(statSync(join(PLUGIN_DIR, String(a.PropertyInspectorPath))).size).toBeGreaterThan(0);
      expect(statSync(join(PLUGIN_DIR, `${String(a.Icon)}.svg`)).size).toBeGreaterThan(0);
      const states = a.States as Array<{ Image: string }>;
      expect(statSync(join(PLUGIN_DIR, `${states[0].Image}.svg`)).size).toBeGreaterThan(0);
      const encoder = a.Encoder as { layout: string } | undefined;
      if (encoder) {
        expect(statSync(join(PLUGIN_DIR, encoder.layout)).size).toBeGreaterThan(0);
      }
    }
    expect(statSync(join(PLUGIN_DIR, "imgs", "plugin", "plugin-icon.png")).size).toBeGreaterThan(64);
    expect(statSync(join(PLUGIN_DIR, "imgs", "plugin", "plugin-icon@2x.png")).size).toBeGreaterThan(64);
    expect(statSync(join(PLUGIN_DIR, "imgs", "plugin", "category-icon.svg")).size).toBeGreaterThan(64);
  });

  it("gives the dials a full 200x100 canvas", () => {
    const layout = JSON.parse(readFileSync(join(PLUGIN_DIR, "layouts", "cell.json"), "utf8")) as {
      controller: string;
      items: Array<{ key: string; type: string; rect: number[] }>;
    };
    expect(layout.controller).toBe("Encoder");
    expect(layout.items).toHaveLength(1);
    expect(layout.items[0]).toMatchObject({ key: "canvas", type: "pixmap", rect: [0, 0, 200, 100] });
  });

  it("ships UTF-8 without BOM, no empty files, SPDX headers, and no banned word", () => {
    const banned = ["dum", "my"].join("");
    for (const file of walkFiles(ROOT)) {
      const bytes = readFileSync(file);
      expect(bytes.length, `empty file: ${file}`).toBeGreaterThan(0);
      const bom = bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf;
      expect(bom, `BOM in ${file}`).toBe(false);
      if (/\.(ts|js|mjs|html|css)$/.test(file)) {
        const text = bytes.toString("utf8");
        expect(text.includes("SPDX-License-Identifier: MIT"), `no SPDX header in ${file}`).toBe(true);
        expect(text.toLowerCase().includes(banned), `banned word in ${file}`).toBe(false);
      }
    }
  });
});

describe("the property inspector speaks every phase", () => {
  it("names each StudioPhase in plain words, never the bare key", () => {
    const store = readFileSync(join(ROOT, "src", "state", "studio-store.ts"), "utf8");
    const union = store.slice(store.indexOf("export type StudioPhase"), store.indexOf("export interface RunState"));
    const phases = [...union.matchAll(/\| "([a-z_]+)"/g)].map((m) => m[1]);
    expect(phases.length).toBeGreaterThanOrEqual(9);
    const pi = readFileSync(join(PLUGIN_DIR, "ui", "pi-common.js"), "utf8");
    const words = pi.slice(pi.indexOf("var PHASE_WORDS"), pi.indexOf("function showState"));
    for (const phase of phases) {
      const named = words.split(/\r?\n/).some((l) => l.trim().startsWith(`${phase}: [`));
      expect(named, `pi-common.js has no words for "${phase}"`).toBe(true);
    }
  });
});
