// SPDX-License-Identifier: MIT
import commonjs from "@rollup/plugin-commonjs";
import nodeResolve from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";

// A source map only while watching (`rollup -c -w`): the shipped package leaves
// maps out, so a built plugin.js should not point at one. The mode is rollup's
// own --watch flag, not the environment, so tests/package.test.ts can ask for
// the release build by calling this with no flags.
export default (flags = {}) => ({
  input: "src/plugin.ts",
  output: {
    file: "com.jbrick2070.otr.sdPlugin/bin/plugin.js",
    format: "esm",
    sourcemap: Boolean(flags.watch)
  },
  external: [
    "node:crypto",
    "node:events",
    "node:http",
    "node:https",
    "node:net",
    "node:os",
    "node:path",
    "node:stream",
    "node:tls",
    "node:url",
    "node:util",
    "node:zlib"
  ],
  plugins: [
    typescript({
      tsconfig: "./tsconfig.json",
      include: ["src/**/*.ts"]
    }),
    nodeResolve({
      browser: false,
      exportConditions: ["node"],
      preferBuiltins: true
    }),
    commonjs(),
    {
      // plugin.js is an ES module wherever it is installed. Node takes a
      // file's module type from the nearest package.json above it, and the
      // installed bin/ has none of ours, so a stray `"type": "commonjs"` one
      // above the Stream Deck's Plugins folder would stop the plugin loading
      // (Sonnet QA). Elgato's own template emits this same file.
      name: "emit-module-package-file",
      generateBundle() {
        this.emitFile({ fileName: "package.json", source: `{ "type": "module" }`, type: "asset" });
      }
    }
  ]
});
