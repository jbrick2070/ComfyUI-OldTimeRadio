// SPDX-License-Identifier: MIT
// What each control shows for the store's state. Pure: state in, SVG out.
import { displayValue } from "./choice-math.js";
import { DialCellInput, MakeKeyInput, renderDialCellSvg, renderMakeKeySvg } from "../gfx/svg-renderer.js";
import { StudioStore } from "../state/studio-store.js";
import { DIAL_FIELD_TITLES, DialField } from "../types/settings.js";

function runProgress(store: StudioStore): number | null {
  const run = store.getRun();
  return run && !run.finished ? run.value : null;
}

/** A choice dial's cell for one field. */
export function choiceDialFace(store: StudioStore, field: DialField): DialCellInput {
  const title = DIAL_FIELD_TITLES[field];
  const phase = store.getPhase();
  if (phase === "off") {
    return { title, value: "Studio off", tone: "muted", hint: `port ${store.getClient().getPort()}` };
  }
  if (phase === "loading" && !store.getField(field)) {
    return { title, value: "loading…", tone: "muted" };
  }
  if (!store.getField(field)) {
    return { title, value: "not on this form", tone: "muted", hint: store.getFormError() };
  }
  const value = displayValue(store.getValue(field));
  if (store.isLocked()) {
    return { title, value, tone: "muted", hint: "making an episode", progress: runProgress(store) };
  }
  return { title, value, hint: "turn: choose · push: default" };
}

/** The machine dial's cell. */
export function machineDialFace(store: StudioStore): DialCellInput {
  const title = "Machine";
  const phase = store.getPhase();
  if (phase === "off") {
    return { title, value: "Studio off", tone: "muted", hint: `port ${store.getClient().getPort()}` };
  }
  const entries = store.getMachineEntries();
  if (phase === "loading" && entries.length === 0) {
    return { title, value: "loading…", tone: "muted" };
  }
  if (entries.length === 0) {
    return { title, value: "not on the menu", tone: "muted", hint: "the Studio runs its own graph" };
  }
  const pending = store.getPendingMachine();
  if (pending) {
    return { title, value: pending.label, hint: "push: switch to it" };
  }
  const current = entries[store.getMachineIndex()] ?? entries[0];
  if (phase === "loading") {
    return { title, value: current.label, tone: "muted", hint: "reading the form…" };
  }
  if (store.isLocked()) {
    return { title, value: current.label, tone: "muted", hint: "making an episode", progress: runProgress(store) };
  }
  return { title, value: current.label, hint: "turn: choose · push: switch" };
}

/** The Make key's face. */
export function makeKeyFace(store: StudioStore): MakeKeyInput {
  const run = store.getRun();
  switch (store.getPhase()) {
    case "off":
      return { headline: "Studio", caption: "not running", tone: "muted" };
    case "loading":
      return { headline: "…", caption: "reading form", tone: "muted" };
    case "form_error":
      return { headline: "form", caption: store.getFormError() || "could not read", tone: "alert" };
    case "busy":
      return { headline: "", glyph: "pause", caption: store.getBusyReason(), tone: "muted" };
    case "running": {
      const value = run?.value ?? 0;
      const minutes = run?.minutesLeft;
      const caption =
        typeof minutes === "number" && minutes > 0
          ? `${run?.phase || "making"} · ${minutes} min`
          : run?.phase || (run?.external ? "from the page" : "making");
      return { headline: `${Math.round(value * 100)}%`, caption, progress: value };
    }
    case "done":
      return { headline: "", glyph: "check", caption: "press: play", tone: "good", progress: 1 };
    case "error":
      return { headline: "", glyph: "cross", caption: run?.error || "failed", tone: "alert" };
    default:
      return { headline: "", glyph: "play", caption: "make the episode" };
  }
}

export function renderChoiceDial(store: StudioStore, field: DialField): string {
  return renderDialCellSvg(choiceDialFace(store, field));
}

export function renderMachineDial(store: StudioStore): string {
  return renderDialCellSvg(machineDialFace(store));
}

export function renderMakeKey(store: StudioStore): string {
  return renderMakeKeySvg(makeKeyFace(store));
}
