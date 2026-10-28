// SPDX-License-Identifier: MIT
// Turning a dial through a form field's choices. Pure functions, so the tests
// can turn without a Stream Deck.
import { StudioField } from "../types/contract.js";

/** The bank that takes typed text (form.MY_STORY_BANK); a dial cannot type. */
export const MY_STORY_BANK = "my_story";

/** The choices a dial offers for a field: the server's, less the one a dial
 *  cannot use. */
export function offeredChoices(field: StudioField): string[] {
  const choices = Array.isArray(field.choices) ? field.choices.map(String) : [];
  return field.key === "bank" ? choices.filter((c) => c !== MY_STORY_BANK) : choices;
}

/** The value a dial shows for a field with no pick yet. */
export function defaultValue(field: StudioField): string | number {
  if (field.kind === "choice") {
    const offered = offeredChoices(field);
    const fallback = String(field.default ?? "");
    return offered.includes(fallback) ? fallback : offered[0] ?? fallback;
  }
  if (field.kind === "int" || field.kind === "float") {
    return clampNumber(field, Number(field.default));
  }
  return String(field.default ?? "");
}

/** Whether `value` is one the field offers now; a pick that is not is
 *  dropped when the form is read again (another machine, another menu). */
export function isOffered(field: StudioField, value: unknown): boolean {
  if (field.kind === "choice") {
    return offeredChoices(field).includes(String(value));
  }
  if (field.kind === "int" || field.kind === "float") {
    const n = Number(value);
    return Number.isFinite(n) && clampNumber(field, n) === n;
  }
  return typeof value === "string";
}

/** The next value `steps` ticks away: through the list for a choice, by one
 *  for a number, held at the ends. */
export function cycleValue(field: StudioField, current: unknown, steps: number): string | number {
  if (steps === 0) {
    return isOffered(field, current) ? (current as string | number) : defaultValue(field);
  }
  if (field.kind === "choice") {
    const offered = offeredChoices(field);
    if (offered.length === 0) {
      return String(current ?? "");
    }
    const at = offered.indexOf(String(current));
    if (at < 0) {
      return steps > 0 ? offered[0] : offered[offered.length - 1];
    }
    const n = offered.length;
    return offered[(((at + steps) % n) + n) % n];
  }
  if (field.kind === "int" || field.kind === "float") {
    const base = Number.isFinite(Number(current)) ? Number(current) : Number(field.default) || 0;
    return clampNumber(field, base + steps);
  }
  return String(current ?? "");
}

function clampNumber(field: StudioField, n: number): number {
  let out = Number.isFinite(n) ? n : 0;
  if (field.kind === "int") {
    out = Math.round(out);
  }
  if (typeof field.min === "number") {
    out = Math.max(field.min, out);
  }
  if (typeof field.max === "number") {
    out = Math.min(field.max, out);
  }
  return out;
}

/** The words on the touch strip for a value: `public_domain` reads as
 *  "public domain", a model id keeps its last path piece. */
export function displayValue(value: unknown): string {
  const text = String(value ?? "");
  const tail = text.includes("/") ? text.slice(text.lastIndexOf("/") + 1) : text;
  return tail.replace(/_/g, " ");
}
