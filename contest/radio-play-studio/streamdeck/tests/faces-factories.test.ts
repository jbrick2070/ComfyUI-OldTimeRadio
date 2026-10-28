// SPDX-License-Identifier: MIT
// The same measure as faces-fit.test.ts, through the real face factories.
// faces.ts writes the words and the renderers lay them out; driving the
// factories with every state a store can be in means built-in copy -- a hint,
// a busy reason, a placeholder, a counter -- is measured as written, not as a
// test author retyped it. (Codex QA, 2026-10-28: "this workflow has no such
// choice" was cut on a key, and the pending machine counter read 0/N, both
// invisible to a test that fed the renderer its own strings.)
import { describe, expect, it } from "vitest";
import { offeredChoices } from "../src/actions/choice-math.js";
import { PHASE_WORDS, renderChoice, renderMachine, renderMakeKey } from "../src/actions/faces.js";
import { CELL, NODE_KEY } from "../src/gfx/svg-renderer.js";
import { MachineEntry, RunState, STAGE_BUSY, StudioPhase, StudioStore } from "../src/state/studio-store.js";
import { StudioField } from "../src/types/contract.js";
import { DIAL_FIELDS } from "../src/types/settings.js";
import { BANKS, checkKey, checkNode, LANGUAGES, LONG_ERROR, STYLES, TextEl } from "./helpers/face-measure.js";

interface StubState {
  phase: StudioPhase;
  fields?: Record<string, StudioField>;
  values?: Record<string, string>;
  formError?: string;
  busy?: string;
  entries?: MachineEntry[];
  current?: number;
  pending?: number | null;
  run?: Partial<RunState> | null;
}

/** A store in a given state, answering exactly what faces.ts asks. Entries
 *  are fresh objects on every call, as the real store's are. */
function stubStore(st: StubState): StudioStore {
  const locked = ["running", "busy", "off", "starting"].includes(st.phase);
  const entries = (): MachineEntry[] => (st.entries ?? []).map((e) => ({ machine: e.machine, way: { ...e.way } }));
  const run = st.run
    ? ({ id: "0123456789ab", external: false, value: 0, phase: "", minutesLeft: null, finished: false, seen: 0, ...st.run } as RunState)
    : null;
  return {
    getPhase: () => st.phase,
    getClient: () => ({ getPort: () => 65535 }),
    getField: (k: string) => st.fields?.[k],
    getValue: (k: string) => st.values?.[k],
    isLocked: () => locked,
    getFormError: () => st.formError ?? "",
    getBusyReason: () => st.busy ?? "",
    getMachineEntries: entries,
    getMachineIndex: () => st.current ?? 0,
    getPendingIndex: () => (st.pending ?? null),
    getPendingMachine: () => (st.pending === null || st.pending === undefined ? null : entries()[st.pending]),
    getRun: () => run
  } as unknown as StudioStore;
}

const choiceField = (key: string, choices: string[]): StudioField => ({
  key, label: key, group: "The show", help: "", kind: "choice", default: choices[0], choices
});
const REAL_FIELDS: Record<string, StudioField> = {
  language: choiceField("language", LANGUAGES),
  bank: choiceField("bank", [...BANKS, "my_story"]),
  style: choiceField("style", STYLES),
  acts: choiceField("acts", ["1", "2", "3", "4", "5", "6", "7"])
};
/** config.MACHINES, the unproven rows as config.UNPROVEN names them, and a
 *  machine the deck has no name for yet. */
const REAL_ENTRIES: MachineEntry[] = (
  [
    ["nvidia8", "foley", true], ["nvidia8", "video", true], ["nvidia8", "stills", true],
    ["nvidia16", "foley", true], ["nvidia16", "video", true], ["nvidia16", "stills", true],
    ["mac16", "video", true], ["mac16", "stills", true], ["amd", "stills", true],
    ["comfy_cloud", "foley", false], ["comfy_cloud", "video", false], ["google_cloud", "stills", true],
    ["a_machine_added_later", "stills", false]
  ] as Array<[string, string, boolean]>
).map(([machine, way, proven]) => ({ machine, way: { key: way, workflow: `${machine}_${way}.json`, proven } }));
const ALL_PHASES: StudioPhase[] = ["off", "loading", "form_error", "ready", "starting", "running", "busy", "done", "error"];
/** The only words a face ends with an ellipsis on purpose. */
const MEANT_ELLIPSIS = new Set(["Tuning in…", "reading the form…", "tuning in…", "on the way…"]);

function cutWords(els: TextEl[]): string[] {
  return els.filter((t) => t.text.endsWith("…") && !MEANT_ELLIPSIS.has(t.text)).map((t) => t.text);
}

describe("every face the factories draw fits, and only a server's long text is ever cut", () => {
  it("choice controls: every field, value, phase and mode, on the form or not", () => {
    let faces = 0;
    for (const mode of ["dial", "key"] as const) {
      const g = mode === "dial" ? CELL : NODE_KEY;
      for (const key of DIAL_FIELDS) {
        const field = REAL_FIELDS[key];
        for (const value of offeredChoices(field)) {
          for (const phase of ALL_PHASES) {
            for (const present of [true, false]) {
              const st: StubState = {
                phase,
                fields: present ? { [key]: field } : {},
                values: { [key]: value },
                run: phase === "running" ? { value: 0.42 } : null
              };
              const label = `${mode} ${key}=${value} ${phase}${present ? "" : " (not on form)"}`;
              const els = checkNode(renderChoice(stubStore(st), key, mode), g, label);
              expect(cutWords(els), label).toEqual([]);
              faces += 1;
            }
          }
        }
      }
    }
    expect(faces).toBeGreaterThan(500);
  });

  it("machine controls: every row chosen or resting, every phase and mode, and a counter from 1", () => {
    for (const mode of ["dial", "key"] as const) {
      const g = mode === "dial" ? CELL : NODE_KEY;
      for (const phase of ALL_PHASES) {
        for (let i = 0; i < REAL_ENTRIES.length; i += 1) {
          for (const pending of [null, (i + 3) % REAL_ENTRIES.length]) {
            const st: StubState = {
              phase,
              entries: REAL_ENTRIES,
              current: i,
              pending,
              run: phase === "running" ? { value: 0.7 } : null
            };
            const label = `${mode} machine row ${i} resting ${pending} ${phase}`;
            const els = checkNode(renderMachine(stubStore(st), mode), g, label);
            expect(cutWords(els), label).toEqual([]);
            for (const t of els.filter((e) => /^\d+\/\d+$/.test(e.text))) {
              const [at, of] = t.text.split("/").map(Number);
              expect(at, `${label}: counter ${t.text}`).toBeGreaterThanOrEqual(1);
              expect(at, `${label}: counter ${t.text}`).toBeLessThanOrEqual(of);
              expect(of).toBe(REAL_ENTRIES.length);
            }
            // The resting row's own counter, not the chosen row's.
            if (pending !== null && mode === "dial" && !["running", "busy", "off", "starting"].includes(phase)) {
              expect(els.some((t) => t.text === `${pending + 1}/${REAL_ENTRIES.length}`), label).toBe(true);
            }
          }
        }
      }
      for (const phase of ALL_PHASES) {
        const label = `${mode} no menu ${phase}`;
        expect(cutWords(checkNode(renderMachine(stubStore({ phase }), mode), g, label)), label).toEqual([]);
      }
    }
  });

  it("the Make key: every phase word, minutes to three digits, known and long reasons", () => {
    const runs: Array<Partial<RunState>> = [
      ...Object.keys(PHASE_WORDS).flatMap((phase) => [
        { phase, value: 0.07, minutesLeft: 120 },
        { phase, value: 0.99, minutesLeft: 1 },
        { phase, value: 0.5, minutesLeft: null }
      ]),
      { phase: "phase.something_new", value: 0.3, minutesLeft: 9 },
      { phase: "", value: 0.3, external: true, minutesLeft: 45 },
      { phase: "", value: 0, minutesLeft: null }
    ];
    for (const run of runs) {
      const label = `running ${JSON.stringify(run)}`;
      const els = checkKey(renderMakeKey(stubStore({ phase: "running", run })), label);
      expect(cutWords(els), label).toEqual([]);
      expect(els.some((t) => t.text.startsWith("phase.")), `${label}: raw phase key shown`).toBe(false);
    }
    const others: Array<[string, StubState, boolean]> = [
      ["off", { phase: "off" }, false],
      ["loading", { phase: "loading" }, false],
      ["starting", { phase: "starting" }, false],
      ["ready", { phase: "ready" }, false],
      ["done", { phase: "done", run: { finished: true, value: 1, done: { run: "r", file: "f" } } }, false],
      ["busy stage", { phase: "busy", busy: STAGE_BUSY }, false],
      ["busy 409", { phase: "busy", busy: LONG_ERROR }, true],
      ["error short", { phase: "error", run: { finished: true, error: "the run failed" } }, false],
      ["error long", { phase: "error", run: { finished: true, error: LONG_ERROR } }, true],
      ["form error long", { phase: "form_error", formError: LONG_ERROR }, true],
      ["form error empty", { phase: "form_error" }, false]
    ];
    for (const [label, st, serverText] of others) {
      const els = checkKey(renderMakeKey(stubStore(st)), label);
      if (!serverText) {
        expect(cutWords(els), label).toEqual([]);
      }
    }
  });
});
