// SPDX-License-Identifier: MIT
// The packed plugin in dist/ is what gets installed, so it must be this tree:
// every file in the plugin folder byte for byte, and bin/plugin.js exactly what
// a build of src/ makes now. A change to the source without `npm run pack`
// fails here, instead of shipping last week's plugin under today's README.
import { describe, expect, it } from "vitest";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { rollup, type OutputOptions, type RollupOptions } from "rollup";
import { readZip } from "./helpers/zip-read";

const ROOT = process.cwd();
const FOLDER = "com.jbrick2070.otr.sdPlugin";
const PLUGIN_DIR = join(ROOT, FOLDER);
const PACKAGE = join(ROOT, "dist", "com.jbrick2070.otr.streamDeckPlugin");
const BUILT = `${FOLDER}/bin/plugin.js`;

/** The plugin folder's own files, as the packer takes them: bin/ is the
 *  build's (checked against a fresh build below), and logs and source maps
 *  are left out by the Elgato CLI's default ignores. */
function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (dir === PLUGIN_DIR && (entry.name === "bin" || entry.name === "logs")) {
        continue;
      }
      out.push(...sourceFiles(full));
    } else if (!/\.(log|js\.map)$/.test(entry.name) && entry.name !== ".sdignore") {
      out.push(`${FOLDER}/${relative(PLUGIN_DIR, full).split(sep).join("/")}`);
    }
  }
  return out;
}

async function buildNow(): Promise<string> {
  const config = (await import(pathToFileURL(join(ROOT, "rollup.config.mjs")).href)).default as RollupOptions & {
    output: OutputOptions;
  };
  const { output: outputOptions, ...inputOptions } = config;
  const bundle = await rollup(inputOptions);
  try {
    const { output } = await bundle.generate(outputOptions);
    return output[0].code;
  } finally {
    await bundle.close();
  }
}

describe("the packed plugin in dist/", () => {
  const entries = existsSync(PACKAGE)
    ? readZip(readFileSync(PACKAGE)).filter((e) => !e.name.endsWith("/"))
    : [];
  const byName = new Map(entries.map((e) => [e.name, e.data]));

  it("exists, and holds the plugin folder and nothing else", () => {
    expect(existsSync(PACKAGE), "run `npm run pack`").toBe(true);
    const expected = [...sourceFiles(PLUGIN_DIR), BUILT].sort();
    expect([...byName.keys()].sort()).toEqual(expected);
  });

  it("carries every file of the folder unchanged", () => {
    for (const name of sourceFiles(PLUGIN_DIR)) {
      const packed = byName.get(name);
      expect(packed, `${name} is not in the package`).toBeDefined();
      const source = readFileSync(join(ROOT, name));
      if (name === `${FOLDER}/manifest.json`) {
        // The packer writes the manifest back as it reads it; what it says
        // must match, not its final newline.
        expect(JSON.parse(packed!.toString("utf8")), "the package's manifest is stale").toEqual(
          JSON.parse(source.toString("utf8"))
        );
      } else {
        expect(packed!.equals(source), `${name} in the package is stale: run \`npm run pack\``).toBe(true);
      }
    }
  });

  it("carries the plugin a build of src/ makes now", async () => {
    const packed = byName.get(BUILT);
    expect(packed, "bin/plugin.js is not in the package").toBeDefined();
    const code = await buildNow();
    expect(code.includes("sourceMappingURL"), "the shipped build points at a source map it does not carry").toBe(false);
    expect(packed!.toString("utf8") === code, "bin/plugin.js in the package is stale: run `npm run pack`").toBe(true);
  }, 120_000);
});
