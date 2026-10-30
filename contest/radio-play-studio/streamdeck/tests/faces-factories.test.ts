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
import { ENGINE_NAMES, PHASE_WORDS, renderChoice, renderMakeKey, renderMenu } from "../src/actions/faces.js";
import { CELL, NODE_KEY } from "../src/gfx/svg-renderer.js";
import { MachineEntry, MenuPart, RunState, STAGE_BUSY, StudioPhase, StudioStore } from "../src/state/studio-store.js";
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
  // Both menu controls get the same rows here: a face only reads its own.
  const entries = (_part?: MenuPart): MachineEntry[] => (st.entries ?? []).map((e) => ({ machine: e.machine, way: { ...e.way } }));
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
    getMenuEntries: entries,
    getMenuIndex: () => st.current ?? 0,
    getPendingIndex: () => (st.pending ?? null),
    getPendingEntry: () => (st.pending === null || st.pending === undefined ? null : entries()[st.pending]),
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
/** config.MACHINES with the lane each graph saves, the unproven rows as
 *  config.UNPROVEN names them, a lane the deck has no name for yet, a way read
 *  without node definitions, and a machine the deck has no name for yet. */
const REAL_ENTRIES: MachineEntry[] = (
  [
    ["nvidia8", "foley", true, "ltx25_foley_8gb"], ["nvidia8", "video", true, "ltx098_low_video"],
    ["nvidia8", "stills", true, "still_motion"], ["nvidia8", "animatediff", true, "animatediff15_v3_haunted_video"],
    ["nvidia8", "mime", true, "ltx25_mime_8gb"], ["nvidia8", "audio_in", true, "ltx25_audio_in_8gb"],
    ["nvidia8", "visualiser", true, "viz_camera"],
    ["nvidia16", "foley", true, "ltx25_foley_16gb (16:9)"], ["nvidia16", "video", true, "ltx25_high_video"],
    ["nvidia16", "stills", true, "still_motion"], ["nvidia16", "animatediff", true, "animatediff15_v3_haunted_video"],
    ["nvidia16", "mime", false, "ltx25_mime_16gb"], ["nvidia16", "visualiser", true, "viz_camera"],
    ["mac16", "video", true, "ltx098_low_video"], ["mac16", "stills", true, "still_motion"],
    ["mac16", "animatediff", true, "animatediff15_lightning_video"], ["mac16", "visualiser", true, "viz_camera"],
    ["amd", "stills", true, "still_motion"],
    ["comfy_cloud", "foley", false, "cloud_ltx25_foley_plus"], ["comfy_cloud", "video", false, "cloud_vidu_q2_pro_fast_720p"],
    ["google_cloud", "stills", false, "a_lane_added_later"],
    ["a_machine_added_later", "audio_in", false, ""]
  ] as Array<[string, string, boolean, string]>
).map(([machine, way, proven, video]) => ({
  machine,
  way: { key: way, workflow: `${machine}_${way}.json`, proven, ...(video ? { uses: { video } } : {}) }
}));
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

  it("machine and video controls: every row chosen or resting, every phase and mode, and a counter from 1", () => {
    for (const [part, mode] of [["machine", "dial"], ["machine", "key"], ["engine", "dial"], ["engine", "key"]] as const) {
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
            const label = `${part} ${mode} row ${i} resting ${pending} ${phase}`;
            const els = checkNode(renderMenu(stubStore(st), part, mode), g, label);
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
        const label = `${part} ${mode} no menu ${phase}`;
        expect(cutWords(checkNode(renderMenu(stubStore({ phase }), part, mode), g, label)), label).toEqual([]);
      }
    }
  });

  it("the video control names each engine, the machine control the engine it would film with", () => {
    const at = (workflow: string): number => REAL_ENTRIES.findIndex((e) => e.way.workflow === workflow);
    const words = (svg: string): string => [...svg.matchAll(/>([^<]+)</g)].map((m) => m[1]).join(" ");
    const face = (part: MenuPart, current: number): string =>
      words(renderMenu(stubStore({ phase: "ready", entries: REAL_ENTRIES, current }), part, "dial"));
    expect(face("engine", at("nvidia8_animatediff.json"))).toContain("AnimateDiff");
    expect(face("engine", at("mac16_animatediff.json"))).toContain("AnimateDiff Lightning");
    expect(face("engine", at("nvidia16_foley.json"))).toContain(ENGINE_NAMES.ltx25_foley_16gb);
    expect(face("engine", at("nvidia16_mime.json"))).toContain("not yet proven");
    // A lane the deck has no name for yet reads as its way; a way read without
    // node definitions too.
    expect(face("engine", at("google_cloud_stills.json"))).toContain("stills");
    expect(face("engine", at("a_machine_added_later_audio_in.json"))).toContain("audio-in");
    expect(face("machine", at("nvidia8_visualiser.json"))).toContain("8 GB card");
    expect(face("machine", at("nvidia8_visualiser.json"))).toContain("Visualiser");
    expect(face("machine", at("nvidia16_mime.json"))).toContain("not yet proven");
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
