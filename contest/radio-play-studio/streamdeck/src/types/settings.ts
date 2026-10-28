// SPDX-License-Identifier: MIT

/** The form choices a dial may be assigned to. `my_story` needs typed text,
 *  so the bank dial never offers it (choice-math.ts). */
export const DIAL_FIELDS = ["language", "bank", "style", "acts"] as const;
export type DialField = (typeof DIAL_FIELDS)[number];

export const DIAL_FIELD_TITLES: Record<DialField, string> = {
  language: "Language",
  bank: "Story bank",
  style: "Visual style",
  acts: "Acts"
};

export interface StudioGlobalSettings {
  [key: string]: string | number | boolean | undefined;
  /** The Studio's port on loopback. The host is always 127.0.0.1: the server
   *  answers no other Host, so there is nothing else to offer. */
  port?: number;
  /** The picks, as JSON: they are what the dials show, so they survive a
   *  Stream Deck restart. */
  picksJson?: string;
}

export const DEFAULT_GLOBAL_SETTINGS: Required<StudioGlobalSettings> = {
  port: 8765,
  picksJson: "{}"
};

export function normalizePort(raw: unknown): number {
  const n = Number(raw);
  return Number.isInteger(n) && n > 0 && n < 65536 ? n : DEFAULT_GLOBAL_SETTINGS.port;
}

export interface ChoiceDialSettings {
  [key: string]: string | number | boolean | undefined;
  field?: DialField;
}

export const DEFAULT_CHOICE_DIAL_SETTINGS: Required<ChoiceDialSettings> = {
  field: "bank"
};

export function normalizeField(raw: unknown): DialField {
  return (DIAL_FIELDS as readonly string[]).includes(String(raw))
    ? (raw as DialField)
    : DEFAULT_CHOICE_DIAL_SETTINGS.field;
}

/** How long a key or dial is held before it counts as a long press. */
export const HOLD_THRESHOLD_MS = 800;
