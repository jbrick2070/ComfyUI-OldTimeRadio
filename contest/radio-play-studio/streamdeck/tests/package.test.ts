// SPDX-License-Identifier: MIT
// The packed plugin in dist/ is what gets installed, so it must be this tree:
// every file in the plugin folder byte for byte, and bin/ exactly what a build
// of src/ makes now. A change to the source without `npm run pack` fails here,
// instead of shipping last week's plugin under today's README.
import { describe, expect, it } from "vitest";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { pathToFileURL } from "node:url";
import ignore from "ignore";
import { rollup, type OutputOptions, type RollupOptions } from "rollup";
import { readZip } from "./helpers/zip-read";

const ROOT = process.cwd();
const FOLDER = "com.jbrick2070.otr.sdPlugin";
const PLUGIN_DIR = join(ROOT, FOLDER);
const PACKAGE = join(ROOT, "dist", "com.jbrick2070.otr.streamDeckPlugin");
const BIN = `${FOLDER}/bin/`;

/** What the Elgato CLI leaves out of a package: these defaults (its
 *  `defaultIgnorePatterns`, pinned to the installed CLI below) and each line of
 *  an `.sdignore` in the plugin folder, matched by the same `ignore` library it
 *  uses -- case-insensitively, as that library does by default. */
const PACKER_DEFAULT_IGNORES = [".sdignore", ".git", "/.env*", "*.log", "*.js.map"];

function packerIgnores(): (path: string) => boolean {
  const rules = ignore().add(PACKER_DEFAULT_IGNORES);
  const own = join(PLUGIN_DIR, ".sdignore");
  if (existsSync(own)) {
    rules.add(readFileSync(own, "utf8"));
  }
  return (path) => rules.ignores(path);
}

/** Every file the packer takes from the plugin folder, as package names. */
function packedNames(): string[] {
  const ignored = packerIgnores();
  return readdirSync(PLUGIN_DIR, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => relative(PLUGIN_DIR, join(entry.parentPath, entry.name)).split(sep).join("/"))
    .filter((path) => !ignored(path))
    .map((path) => `${FOLDER}/${path}`);
}

/** The ignore list the installed CLI carries, read out of its own code. */
function cliIgnoreLists(): { readable: string[] | null; shipped: string[] | null } {
  const cli = join(ROOT, "node_modules", "@elgato", "cli");
  const readable = /defaultIgnorePatterns = \[streamDeckIgnoreFilename, ([^\]]*)\]/.exec(
    readFileSync(join(cli, "dist", "index.js"), "utf8")
  );
  // `streamdeck pack` runs bin/streamdeck.mjs, a minified build of the same
  // source: `const <a>=".sdignore",<b>=[<a>, ...]` with backtick strings.
  const shipped = /const (\w+)=`\.sdignore`,\w+=\[\1,((?:`[^`]*`,?)*)\]/.exec(
    readFileSync(join(cli, "bin", "streamdeck.mjs"), "utf8")
  );
  return {
    readable: readable ? [".sdignore", ...[...readable[1].matchAll(/"([^"]*)"/g)].map((m) => m[1])] : null,
    shipped: shipped ? [".sdignore", ...[...shipped[2].matchAll(/`([^`]*)`/g)].map((m) => m[1])] : null
  };
}

/** Each file a release build writes into bin/ -- plugin.js, and the
 *  package.json that makes it an ES module -- by package name. */
async function buildNow(): Promise<Map<string, Buffer>> {
  // The config is a function of rollup's flags; none is the release build,
  // whatever ROLLUP_WATCH this shell happens to carry.
  const makeConfig = (await import(pathToFileURL(join(ROOT, "rollup.config.mjs")).href)).default as (
    flags: Record<string, unknown>
  ) => RollupOptions & { output: OutputOptions };
  const { output: outputOptions, ...inputOptions } = makeConfig({});
  const bundle = await rollup(inputOptions);
  try {
    const { output } = await bundle.generate(outputOptions);
    return new Map(
      output.map((file) => [`${BIN}${file.fileName}`, Buffer.from(file.type === "chunk" ? file.code : file.source)])
    );
  } finally {
    await bundle.close();
  }
}

describe("the packed plugin in dist/", () => {
  const entries = existsSync(PACKAGE) ? readZip(readFileSync(PACKAGE)) : [];
  const byName = new Map(entries.map((e) => [e.name, e.data]));
  // One build for every test that needs it. bin/ on disk may be stale or, on a
  // fresh clone, absent, and the package must carry the build regardless.
  let build: Promise<Map<string, Buffer>> | null = null;
  const built = (): Promise<Map<string, Buffer>> => (build ??= buildNow());
  const folderOnly = async (): Promise<string[]> => {
    const fromBuild = await built();
    return packedNames().filter((name) => !fromBuild.has(name));
  };

  it("leaves out what the installed Elgato CLI leaves out", () => {
    // A CLI that changes its defaults fails here, so the list above is re-read
    // from the CLI rather than trusted.
    const { readable, shipped } = cliIgnoreLists();
    expect(readable, "the CLI no longer names defaultIgnorePatterns").toEqual(PACKER_DEFAULT_IGNORES);
    expect(shipped, "the CLI's pack command no longer carries this ignore list").toEqual(PACKER_DEFAULT_IGNORES);
  });

  it("exists, and holds the plugin folder and the build, each file once, and nothing else", async () => {
    expect(existsSync(PACKAGE), "run `npm run pack`").toBe(true);
    const expected = [...(await folderOnly()), ...(await built()).keys()].sort();
    // Every entry as listed, so a duplicate name or a folder entry -- which a
    // map would fold away -- still fails.
    expect(entries.map((e) => e.name).sort()).toEqual(expected);
  }, 120_000);

  it("carries every file of the folder unchanged", async () => {
    for (const name of await folderOnly()) {
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
  }, 120_000);

  it("is read by a reader that refuses a local header disagreeing with the directory", () => {
    const tampered = Buffer.from(readFileSync(PACKAGE));
    const nameAt = 30; // the first entry's local header starts the file
    const name = tampered.subarray(nameAt, nameAt + tampered.readUInt16LE(26)).toString("utf8");
    expect(byName.has(name)).toBe(true);
    tampered[nameAt + name.length - 1] ^= 0x01; // same length, another name
    expect(() => readZip(tampered)).toThrow(/local header/);
  });

  it("carries what a build of src/ makes now", async () => {
    const fromBuild = await built();
    const code = fromBuild.get(`${BIN}plugin.js`);
    expect(code, "the build wrote no plugin.js").toBeDefined();
    expect(code!.includes("sourceMappingURL"), "the shipped build points at a source map it does not carry").toBe(false);
    // plugin.js is an ES module whatever package.json sits above the folder
    // the Stream Deck installs it in.
    expect(JSON.parse(fromBuild.get(`${BIN}package.json`)?.toString("utf8") ?? "{}")).toEqual({ type: "module" });
    for (const [name, data] of fromBuild) {
      expect(byName.get(name)?.equals(data), `${name} in the package is stale: run \`npm run pack\``).toBe(true);
    }
  }, 120_000);
});
