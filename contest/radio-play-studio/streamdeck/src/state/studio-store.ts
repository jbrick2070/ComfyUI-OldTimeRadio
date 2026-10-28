// SPDX-License-Identifier: MIT
// Everything the deck knows about the Studio, in one place: whether it answers,
// the form and menu it offers, the picks the dials made, and the run being
// followed. Actions read it and draw; it talks to the server.
import { EventEmitter } from "node:events";
import { cycleValue, defaultValue, isOffered } from "../actions/choice-math.js";
import { EventStreamHandle, StudioClient } from "../studio/studio-client.js";
import {
  Choices,
  MenuWay,
  RunListing,
  StudioEvent,
  StudioField,
  StudioHttpError,
  StudioMenu,
  StudioUnreachable
} from "../types/contract.js";
import { DIAL_FIELDS } from "../types/settings.js";

/** What the Make key is for right now. */
export type StudioPhase =
  | "off" // the Studio is not answering on its port
  | "loading" // the form is being read (about 40 s while ComfyUI renders)
  | "form_error" // the form could not be read: ComfyUI down, or a graph fault
  | "ready" // the form is in hand and nothing is being made
  | "running" // an episode is being made, this deck's or the page's
  | "busy" // the GPU is held by something the deck cannot follow
  | "done" // the last run finished; the key opens its episode
  | "error"; // the last run failed; the key says why

export interface RunState {
  id: string;
  /** Started by the page, not this deck. */
  external: boolean;
  /** The highest progress seen, 0..1: the server's share only moves forward
   *  on the page too, and it caps below 1 until `done`. */
  value: number;
  phase: string;
  minutesLeft: number | null;
  message: string;
  finished: boolean;
  done?: { run: string; file: string };
  error?: string;
  /** How many events have arrived; the next Last-Event-ID on a reconnect. */
  seen: number;
}

/** One row of the machine dial: a machine and one way it films. */
export interface MachineEntry {
  machine: string;
  way: MenuWay;
  label: string;
}

const MACHINE_NAMES: Record<string, string> = {
  nvidia8: "8 GB",
  nvidia16: "16 GB",
  mac16: "Mac 16 GB",
  amd: "AMD",
  comfy_cloud: "Comfy Cloud",
  google_cloud: "Google Cloud"
};

export function machineLabel(machine: string, way: MenuWay): string {
  const name = MACHINE_NAMES[machine] ?? machine;
  return `${name} · ${way.key}${way.proven ? "" : " ?"}`;
}

export interface StoreOptions {
  /** The Studio is asked again this often while it is off or its form failed. */
  offPollMs?: number;
  /** ... and this often while idle, to notice a run the page started. */
  idlePollMs?: number;
  /** A lost event stream is opened again after this long. */
  reconnectMs?: number;
  /** Called with the picks whenever a dial changes them, to persist them. */
  onPicksChanged?: (picks: Choices) => void;
  setTimer?: typeof setTimeout;
  clearTimer?: typeof clearTimeout;
}

export class StudioStore extends EventEmitter {
  private readonly client: StudioClient;
  private readonly opts: Required<StoreOptions>;
  private fields: StudioField[] = [];
  private menu: StudioMenu | null = null;
  private picks: Choices = {};
  private pendingMachine: number | null = null;
  private run: RunState | null = null;
  private stream: EventStreamHandle | null = null;
  private reachable = false;
  private loading = false;
  private formError = "";
  private busyReason = "";
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private started = false;
  private ticking: Promise<void> | null = null;

  public constructor(client: StudioClient, opts: StoreOptions = {}) {
    super();
    this.client = client;
    this.opts = {
      offPollMs: opts.offPollMs ?? 5_000,
      idlePollMs: opts.idlePollMs ?? 10_000,
      reconnectMs: opts.reconnectMs ?? 2_000,
      onPicksChanged: opts.onPicksChanged ?? (() => undefined),
      setTimer: opts.setTimer ?? setTimeout,
      clearTimer: opts.clearTimer ?? clearTimeout
    };
  }

  // -- lifecycle ----------------------------------------------------------
  public start(): void {
    this.started = true;
    void this.tick();
  }

  public stop(): void {
    this.started = false;
    this.clearPoll();
    if (this.reconnectTimer !== null) {
      this.opts.clearTimer(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.closeStream();
  }

  public getClient(): StudioClient {
    return this.client;
  }

  /** The persisted picks, read back at start-up. */
  public setPicks(picks: Choices): void {
    this.picks = { ...picks };
    this.reconcilePicks();
    this.changed();
  }

  public getPicks(): Choices {
    return { ...this.picks };
  }

  // -- what the deck shows --------------------------------------------------
  public getPhase(): StudioPhase {
    if (!this.reachable) {
      return "off";
    }
    if (this.run && !this.run.finished) {
      return "running";
    }
    if (this.busyReason) {
      return "busy";
    }
    if (this.run?.finished) {
      return this.run.done ? "done" : "error";
    }
    if (this.loading) {
      return "loading";
    }
    if (this.formError || this.fields.length === 0) {
      return "form_error";
    }
    return "ready";
  }

  /** The dials turn only while nothing is being made. */
  public isLocked(): boolean {
    const phase = this.getPhase();
    return phase === "running" || phase === "busy" || phase === "off";
  }

  public getRun(): RunState | null {
    return this.run ? { ...this.run } : null;
  }

  public getFormError(): string {
    return this.formError;
  }

  public getBusyReason(): string {
    return this.busyReason;
  }

  public getField(key: string): StudioField | undefined {
    return this.fields.find((f) => f.key === key);
  }

  /** What a dial shows for a field: its pick, else the form's default. */
  public getValue(key: string): string | number | undefined {
    const field = this.getField(key);
    if (!field) {
      return undefined;
    }
    const pick = this.picks[key];
    return pick !== undefined && isOffered(field, pick) ? pick : defaultValue(field);
  }

  public getMenu(): StudioMenu | null {
    return this.menu;
  }

  /** The machine dial's rows, in menu order; empty without a menu. */
  public getMachineEntries(): MachineEntry[] {
    if (!this.menu) {
      return [];
    }
    const out: MachineEntry[] = [];
    for (const machine of this.menu.machines) {
      for (const way of machine.ways) {
        out.push({ machine: machine.key, way, label: machineLabel(machine.key, way) });
      }
    }
    return out;
  }

  /** The row the next POST would run on. */
  public getMachineIndex(): number {
    const entries = this.getMachineEntries();
    const workflow = this.currentWorkflow();
    return entries.findIndex((e) => e.way.workflow === workflow);
  }

  /** The row the dial is resting on before it is pushed, if any. */
  public getPendingMachine(): MachineEntry | null {
    const entries = this.getMachineEntries();
    return this.pendingMachine === null ? null : entries[this.pendingMachine] ?? null;
  }

  public studioUrl(): string {
    const base = `${this.client.baseUrl()}/`;
    // The page opens on the run its address names (index.html, #run=), so
    // this is how the page is made to show what the deck started.
    return this.run ? `${base}#run=${encodeURIComponent(this.run.id)}` : base;
  }

  /** Where the finished episode plays; empty until a run has `done`. */
  public episodeUrl(): string {
    const done = this.run?.done;
    if (!done) {
      return "";
    }
    return `${this.client.baseUrl()}/episodes/${encodeURIComponent(done.run)}/${encodeURIComponent(done.file)}`;
  }

  // -- the dials ------------------------------------------------------------
  /** Turn a field `steps` ticks; false when the dial is locked or the field
   *  is not on this form. */
  public cycle(key: string, steps: number): boolean {
    const field = this.getField(key);
    if (!field || this.isLocked()) {
      return false;
    }
    this.picks[key] = cycleValue(field, this.getValue(key), steps);
    this.picksChanged();
    return true;
  }

  public resetPick(key: string): boolean {
    if (this.isLocked() || !this.getField(key)) {
      return false;
    }
    delete this.picks[key];
    this.picksChanged();
    return true;
  }

  public rotateMachine(steps: number): boolean {
    const entries = this.getMachineEntries();
    if (entries.length === 0 || this.isLocked()) {
      return false;
    }
    const from = this.pendingMachine ?? Math.max(0, this.getMachineIndex());
    const n = entries.length;
    this.pendingMachine = (((from + steps) % n) + n) % n;
    this.changed();
    return true;
  }

  /** Make the resting row the one that runs: the form is read again for it,
   *  and picks it does not offer fall back to its defaults. */
  public async applyMachine(): Promise<boolean> {
    const pending = this.getPendingMachine();
    this.pendingMachine = null;
    if (!pending || this.isLocked()) {
      this.changed();
      return false;
    }
    if (pending.way.workflow === this.currentWorkflow()) {
      this.changed();
      return true;
    }
    this.picks.workflow = pending.way.workflow;
    this.picksChanged();
    await this.refresh();
    return true;
  }

  // -- the key ------------------------------------------------------------
  /** What a press of the Make key does in the current phase: "open" leaves
   *  the finished run in place for episodeUrl(); the key calls clearRun()
   *  once it has opened it. */
  public async press(): Promise<"made" | "open" | "cleared" | "refused"> {
    switch (this.getPhase()) {
      case "done":
        return "open";
      case "error":
        this.clearRun();
        return "cleared";
      case "ready":
        await this.make();
        return "made";
      default:
        return "refused";
    }
  }

  /** Let go of a finished run, so the key is back to making. A run still
   *  going is kept: there is nothing else to show for it. */
  public clearRun(): void {
    if (this.run?.finished) {
      this.run = null;
      this.changed();
    }
  }

  /** The choices as the next POST sends them: what the four dials show, for
   *  the fields this form has, and the menu workflow when there is a menu. */
  public postableChoices(): Choices {
    const out: Choices = {};
    for (const key of DIAL_FIELDS) {
      const value = this.getValue(key);
      if (value !== undefined) {
        out[key] = value;
      }
    }
    const workflow = this.currentWorkflow();
    if (this.menu && workflow) {
      out.workflow = workflow;
    }
    return out;
  }

  public async make(): Promise<void> {
    this.busyReason = "";
    this.run = null;
    this.changed();
    try {
      const id = await this.client.startEpisode(this.postableChoices());
      this.attach(id, false);
    } catch (err) {
      if (err instanceof StudioUnreachable) {
        this.lost();
        return;
      }
      const refusal = err as StudioHttpError;
      if (refusal.status === 409) {
        // Something is being made. An episode the page started is followed;
        // anything else (a stage, a project waiting to hear how its last run
        // ended) is shown as the server put it, and the page sorts it out.
        await this.tick();
        if (!this.run) {
          this.busyReason = refusal.message;
          this.changed();
        }
        return;
      }
      this.run = {
        id: "",
        external: false,
        value: 0,
        phase: "",
        minutesLeft: null,
        message: "",
        finished: true,
        error: refusal.message,
        seen: 0
      };
      this.changed();
    }
  }

  // -- talking to the server ------------------------------------------------
  /** Look at the Studio: is it there, is anything being made. The form is
   *  read when it has not been, or when the Studio has just come back. */
  public tick(): Promise<void> {
    if (!this.ticking) {
      this.ticking = this.doTick().finally(() => {
        this.ticking = null;
        this.schedulePoll();
      });
    }
    return this.ticking;
  }

  private async doTick(): Promise<void> {
    let listing: RunListing[];
    try {
      listing = await this.client.episodes();
    } catch (err) {
      if (err instanceof StudioUnreachable) {
        this.lost();
        return;
      }
      throw err;
    }
    const cameBack = !this.reachable;
    this.reachable = true;
    const unfinished = listing.filter((r) => !r.finished);
    const episode = unfinished.find((r) => !r.project_id);
    const stage = unfinished.find((r) => Boolean(r.project_id));
    if (episode) {
      if (!this.run || this.run.id !== episode.id) {
        this.attach(episode.id, true);
      }
    } else if (this.run && !this.run.finished && !this.stream) {
      // The run the deck was following is gone from the listing and no stream
      // is open to say how it ended (the Studio restarted).
      this.run = null;
    }
    this.busyReason = stage && !episode ? "The Scene Producer is making a stage" : "";
    this.changed();
    if ((cameBack || this.fields.length === 0) && !this.loading) {
      await this.refresh();
    }
  }

  /** Read the form for the workflow the deck is on. */
  public async refresh(): Promise<void> {
    this.loading = true;
    this.formError = "";
    this.changed();
    try {
      const body = await this.client.options(this.menu ? this.currentWorkflow() : undefined);
      this.fields = Array.isArray(body.fields) ? body.fields : [];
      this.menu = body.menu ?? null;
      this.reconcilePicks();
      const starting = this.startingWorkflow();
      if (starting && starting !== this.menu?.current.workflow) {
        // The page opens on the detected machine's own graph, never a 16 GB
        // graph on an 8 GB card; so does the deck, once, when it has no pick.
        this.picks.workflow = starting;
        this.picksChanged();
        const again = await this.client.options(starting);
        this.fields = Array.isArray(again.fields) ? again.fields : [];
        this.menu = again.menu ?? this.menu;
        this.reconcilePicks();
      }
    } catch (err) {
      if (err instanceof StudioUnreachable) {
        this.lost();
        return;
      }
      this.formError = (err as Error).message || "the form could not be read";
    } finally {
      this.loading = false;
    }
    this.changed();
  }

  private startingWorkflow(): string | null {
    if (!this.menu || this.picks.workflow) {
      return null;
    }
    const detected = this.menu.machines.find((m) => m.key === this.menu?.detected);
    if (!detected || detected.key === this.menu.current.machine) {
      return null;
    }
    const way = detected.ways.find((w) => w.key === this.menu?.current.way) ?? detected.ways[0];
    return way ? way.workflow : null;
  }

  private currentWorkflow(): string {
    const pick = this.picks.workflow;
    if (typeof pick === "string" && pick && this.getMachineEntries().some((e) => e.way.workflow === pick)) {
      return pick;
    }
    return this.menu?.current.workflow ?? "";
  }

  /** Drop picks this form does not offer, and a workflow not on this menu. */
  private reconcilePicks(): void {
    let dropped = false;
    for (const key of Object.keys(this.picks)) {
      if (key === "workflow") {
        if (this.menu && !this.getMachineEntries().some((e) => e.way.workflow === this.picks.workflow)) {
          delete this.picks.workflow;
          dropped = true;
        }
        continue;
      }
      const field = this.getField(key);
      if (this.fields.length > 0 && (!field || !isOffered(field, this.picks[key]))) {
        delete this.picks[key];
        dropped = true;
      }
    }
    if (dropped) {
      this.opts.onPicksChanged(this.getPicks());
    }
  }

  private attach(id: string, external: boolean): void {
    this.closeStream();
    this.run = {
      id,
      external,
      value: 0,
      phase: "",
      minutesLeft: null,
      message: "",
      finished: false,
      seen: 0
    };
    this.busyReason = "";
    this.changed();
    this.openStream();
  }

  private openStream(): void {
    const run = this.run;
    if (!run || run.finished) {
      return;
    }
    this.stream = this.client.events(
      run.id,
      (event, eventId) => this.onEvent(run, event, eventId),
      (error) => this.onStreamEnd(run, error),
      run.seen > 0 ? run.seen - 1 : undefined
    );
  }

  private onEvent(run: RunState, event: StudioEvent, eventId: number): void {
    if (this.run !== run) {
      return;
    }
    run.seen = Math.max(run.seen, eventId + 1);
    switch (event.type) {
      case "progress": {
        const value = Number(event.value);
        if (Number.isFinite(value)) {
          run.value = Math.max(run.value, Math.min(1, Math.max(0, value)));
        }
        if (event.phase) {
          run.phase = event.phase;
        }
        run.minutesLeft =
          typeof event.minutes_left === "number" && Number.isFinite(event.minutes_left)
            ? event.minutes_left
            : run.minutesLeft;
        break;
      }
      case "step":
        if (event.phase) {
          run.phase = event.phase;
        }
        break;
      case "note":
        run.message = event.message;
        break;
      case "done":
        run.finished = true;
        run.value = 1;
        run.minutesLeft = 0;
        if (typeof event.run === "string" && typeof event.file === "string" && event.run && event.file) {
          run.done = { run: event.run, file: event.file };
        } else {
          run.error = "the run finished without an episode file";
        }
        break;
      case "error":
        run.finished = true;
        run.error = event.message || "the run failed";
        break;
      default:
        break;
    }
    this.changed();
  }

  private onStreamEnd(run: RunState, error?: Error): void {
    this.stream = null;
    if (this.run !== run) {
      return;
    }
    if (run.finished) {
      this.changed();
      return;
    }
    if (error instanceof StudioUnreachable) {
      this.lost();
      return;
    }
    if (error instanceof StudioHttpError && error.status === 404) {
      // The Studio no longer knows this run: it restarted.
      this.run = null;
      this.changed();
      return;
    }
    // The stream closed under a run still going (the server cuts a quiet
    // socket after its timeout): open it again where it left off.
    if (this.started) {
      this.reconnectTimer = this.opts.setTimer(() => {
        this.reconnectTimer = null;
        this.openStream();
      }, this.opts.reconnectMs);
    }
  }

  private lost(): void {
    this.reachable = false;
    this.closeStream();
    this.changed();
  }

  private closeStream(): void {
    if (this.stream) {
      const stream = this.stream;
      this.stream = null;
      stream.close();
    }
  }

  private schedulePoll(): void {
    this.clearPoll();
    if (!this.started) {
      return;
    }
    const phase = this.getPhase();
    const wait = phase === "off" || phase === "form_error" ? this.opts.offPollMs : this.opts.idlePollMs;
    this.pollTimer = this.opts.setTimer(() => {
      this.pollTimer = null;
      void this.tick();
    }, wait);
  }

  private clearPoll(): void {
    if (this.pollTimer !== null) {
      this.opts.clearTimer(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private picksChanged(): void {
    this.opts.onPicksChanged(this.getPicks());
    this.changed();
  }

  private changed(): void {
    this.emit("changed");
  }
}
