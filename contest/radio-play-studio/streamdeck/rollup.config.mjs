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
    commonjs()
  ]
});
