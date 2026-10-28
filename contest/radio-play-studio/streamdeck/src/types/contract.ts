// SPDX-License-Identifier: MIT
// What Radio Play Studio's server answers, as the page reads it. The server is
// ../radio_play_studio/server.py; API-CONTRACT.md names each route. Nothing
// here is invented: a field the server may leave out is optional here.

export type FieldKind = "choice" | "int" | "float" | "bool" | "text";

/** One row of the form, from GET /api/options (form.describe_form). */
export interface StudioField {
  key: string;
  label: string;
  group: string;
  help: string;
  kind: FieldKind;
  default: unknown;
  choices?: unknown[] | null;
  min?: number | null;
  max?: number | null;
}

/** One way a machine films: a workflow file beside the server's own. */
export interface MenuWay {
  key: string;
  workflow: string;
  proven: boolean;
}

export interface MenuMachine {
  key: string;
  ways: MenuWay[];
}

/** The machine menu, present only when the server's own workflow is on it. */
export interface StudioMenu {
  machines: MenuMachine[];
  detected: string;
  current: { machine: string; way: string; workflow: string };
}

export interface OptionsResponse {
  fields: StudioField[];
  photo_stills?: string[];
  menu?: StudioMenu | null;
}

/** A row of GET /api/episodes (Runs.listing): an episode, or a Scene Producer
 *  stage run, which carries its project_id. */
export interface RunListing {
  id: string;
  finished: boolean;
  started: number;
  project_id?: string;
  stage?: string;
}

/** The events on GET /api/episodes/<id>/events (runner.Studio.stream). */
export type StudioEvent =
  | { type: "submitted" }
  | {
      type: "progress";
      value: number;
      message?: string;
      nodes_done?: number;
      nodes_total?: number;
      minutes_left?: number;
      overrun?: boolean;
      phase?: string;
    }
  | { type: "step"; value: number; step?: number | null; steps?: number | null; phase?: string }
  | { type: "status"; status: string }
  | { type: "output"; name: string }
  | { type: "note"; message: string }
  | { type: "done"; path: string; run: string; file: string }
  | { type: "error"; message: string };

/** The choices this plugin posts. Every key is a form.FIELDS key, except
 *  `workflow`, which runner.WORKFLOW_CHOICE reads to pick the graph. */
export type Choices = Record<string, string | number>;

/** An answer the server refused, with its JSON `error`. */
export class StudioHttpError extends Error {
  public readonly status: number;

  public constructor(status: number, message: string) {
    super(message);
    this.name = "StudioHttpError";
    this.status = status;
  }
}

/** The Studio is not answering on its port at all. */
export class StudioUnreachable extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "StudioUnreachable";
  }
}
