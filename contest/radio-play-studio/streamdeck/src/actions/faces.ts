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
import { MACHINE_NAMES, MachineEntry, MenuPart, StudioStore } from "../state/studio-store.js";
import { MenuWay } from "../types/contract.js";
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

/** The video engine each lane a menu graph saves is, named for a person;
 *  the page's MODEL_NAMES, short enough for a key. */
export const ENGINE_NAMES: Record<string, string> = {
  ltx25_foley_8gb: "LTX 2.5 foley",
  ltx25_foley_16gb: "LTX 2.5 foley",
  ltx25_foley_24gb: "LTX 2.5 foley",
  ltx25_high_video: "LTX 2.5",
  ltx098_low_video: "LTX 0.9.8",
  ltx_8gb: "LTX 2",
  ltx25_mime_8gb: "LTX 2.5 mime",
  ltx25_mime_16gb: "LTX 2.5 mime",
  ltx25_audio_in_8gb: "LTX 2.5 audio-in",
  animatediff15_v3_haunted_video: "AnimateDiff",
  animatediff15_lightning_video: "AnimateDiff Lightning",
  still_motion: "Stills",
  still_flat: "Flat stills",
  viz_camera: "Visualiser",
  cloud_ltx25_foley_plus: "LTX 2.5 foley",
  cloud_vidu_q2_pro_fast_720p: "Vidu Q2",
  cloud_ltx25_audio_in: "LTX 2.5 audio-in"
};

/** The menu's ways (config.WAYS) in a word, for a graph whose engine is not
 *  named above or was not read. */
export const WAY_WORDS: Record<string, string> = {
  foley: "foley",
  video: "video",
  stills: "stills",
  animatediff: "AnimateDiff",
  mime: "mime",
  audio_in: "audio-in",
  visualiser: "visualiser"
};

/** What a way films with: the engine its graph saves, else the way itself. */
export function engineName(way: MenuWay): string {
  // The director's label is the lane and its suffixes, each starting " (":
  // "viz_camera (16:9) (audio-reactive, no scene image)" (otr_video_director._label_for).
  const lane = String(way.uses?.video ?? "").split(" (")[0];
  return ENGINE_NAMES[lane] ?? WAY_WORDS[way.key] ?? displayValue(way.key);
}

/** A menu control's node. The machine control: the card in large letters,
 *  what it would film with under. The video control: the engine in large
 *  letters, the card it runs on under. */
export function menuFace(store: StudioStore, part: MenuPart, mode: ControlMode = "dial"): DialCellInput {
  const title = part === "machine" ? "Machine" : "Video";
  const socket = part === "machine" ? SOCKET_COLORS.machine : SOCKET_COLORS.video;
  const phase = store.getPhase();
  if (phase === "off") {
    return { title, socket, value: "Off air", tone: "muted", hint: OFF_HINT(store) };
  }
  const entries = store.getMenuEntries(part);
  if (entries.length === 0) {
    return phase === "loading"
      ? { title, socket, value: "Tuning in…", tone: "muted", hint: "reading the form" }
      : { title, socket, value: "Its own graph", tone: "muted", hint: "no machine menu" };
  }
  const currentIndex = Math.max(0, store.getMenuIndex(part));
  const current = entries[currentIndex];
  // A machine the deck has no name for yet reads as words, and wraps like one.
  const machineName = (e: MachineEntry): string => MACHINE_NAMES[e.machine] ?? displayValue(e.machine);
  const main = (e: MachineEntry): string => (part === "machine" ? machineName(e) : engineName(e.way));
  // An engine's name and "(unproven)" do not fit one line of a key together;
  // a row with no published episode says so in place of the other half.
  const sub = (e: MachineEntry): string =>
    !e.way.proven ? "not yet proven" : part === "machine" ? engineName(e.way) : machineName(e);
  const position = (i: number): string => `${i + 1}/${entries.length}`;
  if (store.isLocked()) {
    return {
      title,
      socket,
      value: main(current),
      position: position(currentIndex),
      tone: "muted",
      hint: sub(current),
      progress: runProgress(store)
    };
  }
  const pending = store.getPendingEntry(part);
  const pendingIndex = store.getPendingIndex(part);
  if (pending && pendingIndex !== null) {
    const i = pendingIndex;
    // A key is small: the instruction takes the title bar, where it is big.
    return mode === "dial"
      ? { title, socket, value: main(pending), position: position(i), pending: true, hint: `${sub(pending)}\npress to switch` }
      : { title: "Hold to use", socket, value: main(pending), pending: true, hint: sub(pending) };
  }
  if (phase === "loading") {
    return { title, socket, value: main(current), position: position(currentIndex), tone: "muted", hint: "tuning in…" };
  }
  return { title, socket, value: main(current), position: position(currentIndex), hint: sub(current) };
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

export function renderMenu(store: StudioStore, part: MenuPart, mode: ControlMode): string {
  const face = menuFace(store, part, mode);
  return mode === "dial" ? renderDialCellSvg(face) : renderNodeKeySvg(face);
}

export function renderMakeKey(store: StudioStore): string {
  return renderMakeKeySvg(makeKeyFace(store));
}
