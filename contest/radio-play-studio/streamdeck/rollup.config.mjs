// SPDX-License-Identifier: MIT
import commonjs from "@rollup/plugin-commonjs";
import nodeResolve from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";

export default {
  input: "src/plugin.ts",
  output: {
    file: "com.jbrick2070.otr.sdPlugin/bin/plugin.js",
    format: "esm",
    sourcemap: true
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
