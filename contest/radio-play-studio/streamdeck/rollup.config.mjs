// SPDX-License-Identifier: MIT
import commonjs from "@rollup/plugin-commonjs";
import nodeResolve from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";

// A source map only while watching: the shipped package leaves maps out, so a
// built plugin.js should not point at one.
const watching = Boolean(process.env.ROLLUP_WATCH);

export default {
  input: "src/plugin.ts",
  output: {
    file: "com.jbrick2070.otr.sdPlugin/bin/plugin.js",
    format: "esm",
    sourcemap: watching
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
};
