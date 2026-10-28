// SPDX-License-Identifier: MIT
// What each control shows for the store's state. Pure: state in, SVG out.
import { displayValue, offeredChoices } from "./choice-math.js";
import {
  DialCellInput,
  MakeKeyInput,
  renderDialCellSvg,
  renderMakeKeySvg,
  renderNodeKeySvg,
  SOCKET_COLORS
} from "../gfx/svg-renderer.js";
import { MACHINE_NAMES, StudioStore } from "../state/studio-store.js";
import { DIAL_FIELD_TITLES, DialField } from "../types/settings.js";

/** A choice or machine control is a dial on a Stream Deck + and a key on a
 *  deck without dials; the face says what that control's touch does. */
export type ControlMode = "dial" | "key";

/** The server names the phase by its page string key (runner.PHASE_BY_CLASS);
 *  the key face has room for one short word. */
export const PHASE_WORDS: Record<string, string> = {
  "phase.writing": "writing",
  "phase.casting": "casting",
  "phase.voicing": "voices",
  "phase.music": "music",
  "phase.pictures": "pictures",
  "phase.planning": "planning",
  "phase.filming": "filming",
  "phase.sound": "mixing",
  "phase.finishing": "finishing"
};

export function phaseWord(phase: string): string {
  if (!phase) {
    return "";
  }
  return PHASE_WORDS[phase] ?? phase.replace(/^phase\./, "").replace(/_/g, " ");
}

function runProgress(store: StudioStore): number | null {
  const run = store.getRun();
  return run && !run.finished ? run.value : null;
}

/** "3 acts", "1 act": the acts dropdown holds bare numbers. */
function fieldValueText(field: DialField, value: unknown): string {
  if (field === "acts" && /^\d+$/.test(String(value))) {
    return `${value} act${String(value) === "1" ? "" : "s"}`;
  }
  return displayValue(value);
}

const OFF_HINT = (store: StudioStore): string => `nothing on port ${store.getClient().getPort()}`;

/** A choice control's node for one field. */
export function choiceFace(store: StudioStore, field: DialField, mode: ControlMode = "dial"): DialCellInput {
  const title = DIAL_FIELD_TITLES[field];
  const socket = SOCKET_COLORS[field];
  const phase = store.getPhase();
  if (phase === "off") {
    return { title, socket, value: "Off air", tone: "muted", hint: OFF_HINT(store) };
  }
  const form = store.getField(field);
  if (!form) {
    return phase === "loading"
      ? { title, socket, value: "Tuning in…", tone: "muted", hint: "reading the form" }
      : { title, socket, value: "Not offered", tone: "muted", hint: store.getFormError() || "not in this workflow" };
  }
  const current = store.getValue(field);
  const value = fieldValueText(field, current);
  const offered = form.kind === "choice" ? offeredChoices(form) : [];
  const at = offered.indexOf(String(current));
  const position = at >= 0 && offered.length > 1 ? `${at + 1}/${offered.length}` : undefined;
  if (store.isLocked()) {
    return { title, socket, value, position, tone: "muted", hint: "on air", progress: runProgress(store) };
  }
  const hint = mode === "dial" ? "press to reset" : "press: next\nhold: reset";
  return { title, socket, value, position, hint };
}

/** The machine control's node: the card in large letters, how it films under. */
export function machineFace(store: StudioStore, mode: ControlMode = "dial"): DialCellInput {
  const title = "Machine";
  const socket = SOCKET_COLORS.machine;
  const phase = store.getPhase();
  if (phase === "off") {
    return { title, socket, value: "Off air", tone: "muted", hint: OFF_HINT(store) };
  }
  const entries = store.getMachineEntries();
  if (entries.length === 0) {
    return phase === "loading"
      ? { title, socket, value: "Tuning in…", tone: "muted", hint: "reading the form" }
      : { title, socket, value: "Its own graph", tone: "muted", hint: "no machine menu" };
  }
  const currentIndex = Math.max(0, store.getMachineIndex());
  const current = entries[currentIndex];
  const wayText = (e: (typeof entries)[number]): string => `${e.way.key}${e.way.proven ? "" : " (unproven)"}`;
  // A machine the deck has no name for yet reads as words, and wraps like one.
  const nameOf = (e: (typeof entries)[number]): string => MACHINE_NAMES[e.machine] ?? displayValue(e.machine);
  const position = (i: number): string => `${i + 1}/${entries.length}`;
  if (store.isLocked()) {
    return {
      title,
      socket,
      value: nameOf(current),
      position: position(currentIndex),
      tone: "muted",
      hint: wayText(current),
      progress: runProgress(store)
    };
  }
  const pending = store.getPendingMachine();
  const pendingIndex = store.getPendingIndex();
  if (pending && pendingIndex !== null) {
    const i = pendingIndex;
    // A key is small: the instruction takes the title bar, where it is big.
    return mode === "dial"
      ? { title, socket, value: nameOf(pending), position: position(i), pending: true, hint: `${wayText(pending)}\npress to switch` }
      : { title: "Hold to use", socket, value: nameOf(pending), pending: true, hint: wayText(pending) };
  }
  if (phase === "loading") {
    return { title, socket, value: nameOf(current), position: position(currentIndex), tone: "muted", hint: "tuning in…" };
  }
  return { title, socket, value: nameOf(current), position: position(currentIndex), hint: wayText(current) };
}

/** The Make key's face. */
export function makeKeyFace(store: StudioStore): MakeKeyInput {
  const run = store.getRun();
  switch (store.getPhase()) {
    case "off":
      return { headline: "OFF AIR", caption: ["start the", "Studio"], tone: "muted" };
    case "loading":
      return { glyph: "dots", caption: "tuning in…", tone: "muted" };
    case "starting":
      return { glyph: "dots", caption: "on the way…" };
    case "form_error":
      return { glyph: "cross", caption: store.getFormError() || "no form", tone: "alert" };
    case "busy":
      return { glyph: "pause", caption: store.getBusyReason() || "busy", tone: "muted" };
    case "running": {
      const value = run?.value ?? 0;
      const word = phaseWord(run?.phase ?? "") || (run?.external ? "from the page" : "on air");
      const minutes = run?.minutesLeft;
      const caption = typeof minutes === "number" && minutes > 0 ? [word, `${minutes} min left`] : [word];
      return { headline: `${Math.round(value * 100)}%`, caption, progress: value };
    }
    case "done":
      return { glyph: "check", caption: "press to play", tone: "good", progress: 1 };
    case "error":
      return { glyph: "cross", caption: run?.error || "it failed", tone: "alert" };
    default:
      return { headline: "MAKE", wave: true, caption: "the episode" };
  }
}

export function renderChoice(store: StudioStore, field: DialField, mode: ControlMode): string {
  const face = choiceFace(store, field, mode);
  return mode === "dial" ? renderDialCellSvg(face) : renderNodeKeySvg(face);
}

export function renderMachine(store: StudioStore, mode: ControlMode): string {
  const face = machineFace(store, mode);
  return mode === "dial" ? renderDialCellSvg(face) : renderNodeKeySvg(face);
}

export function renderMakeKey(store: StudioStore): string {
  return renderMakeKeySvg(makeKeyFace(store));
}
