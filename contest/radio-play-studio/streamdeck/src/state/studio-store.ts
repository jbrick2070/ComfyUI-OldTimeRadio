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
  OptionsResponse,
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
  | "starting" // the Make press is on its way to the Studio
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
  /** The page's string key for the phase (`phase.writing` ...). */
  phase: string;
  minutesLeft: number | null;
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
}

/** The menu's machine keys (config.MACHINES) as the dial names them. */
export const MACHINE_NAMES: Record<string, string> = {
  nvidia8: "8 GB card",
  nvidia16: "16 GB card",
  mac16: "Mac 16 GB",
  amd: "AMD card",
  comfy_cloud: "Comfy Cloud",
  google_cloud: "Google Cloud"
};

/** The busy reason when a Scene Producer stage holds the GPU. */
export const STAGE_BUSY = "a stage is running";

export interface StoreOptions {
  /** The Studio is asked again this often while it is off or its form failed. */
  offPollMs?: number;
  /** ... and this often while idle, to notice a run the page started. */
  idlePollMs?: number;
  /** A lost event stream is opened again after this long. */
  reconnectMs?: number;
  /** Called with the picks whenever they change, to persist them. */
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
  private submitting = false;
  private formError = "";
  private busyReason = "";
  /** Bumped by every form read and by a port change: an answer that comes
   *  back under an older number is dropped, so a slow read for one machine
   *  cannot land on top of a quicker read for the next. */
  private formRequest = 0;
  /** Bumped whenever the run the deck follows is replaced or let go: a
   *  listing asked for before that describes a world that has moved on (a
   *  listing taken just before Make does not have the new run in it). */
  private runGeneration = 0;
  /** The press in flight: an answer for an older one (the port changed
   *  under it) neither attaches a run nor releases the lock. */
  private submitToken = 0;
  /** The look in flight: a port change starts a new one at once instead of
   *  queueing behind a form read that can take a minute and a half. */
  private tickToken = 0;
  /** False until the Studio has answered or failed to, once: the faces say
   *  "tuning in", not "off air", while the first look is on its way. */
  private looked = false;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private started = false;
  private ticking: Promise<void> | null = null;

  public constructor(client: StudioClient, opts: StoreOptions = {}) {
    super();
    // Every control on every page listens; a full XL is 32 of them.
    this.setMaxListeners(0);
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
    this.clearReconnect();
    this.closeStream();
  }

  public getClient(): StudioClient {
    return this.client;
  }

  /** The picks as persisted. Read back before start(); when they arrive
   *  later and name another workflow, that workflow's form is read. */
  public setPicks(picks: Choices): void {
    const workflowBefore = this.picks.workflow;
    this.picks = { ...picks };
    if (this.fields.length > 0) {
      if (this.picks.workflow !== workflowBefore) {
        void this.refresh();
      } else {
        this.reconcilePicks();
      }
    }
    this.changed();
  }

  public getPicks(): Choices {
    return { ...this.picks };
  }

  /** The Studio moved to another port: nothing the old one said holds. */
  public endpointChanged(): void {
    this.formRequest += 1;
    this.runGeneration += 1;
    this.submitToken += 1;
    this.tickToken += 1;
    this.ticking = null;
    this.submitting = false;
    this.looked = false;
    this.clearPoll();
    this.clearReconnect();
    this.closeStream();
    this.run = null;
    this.fields = [];
    this.menu = null;
    this.pendingMachine = null;
    this.reachable = false;
    this.loading = false;
    this.formError = "";
    this.busyReason = "";
    this.changed();
    void this.tick();
  }

  // -- what the deck shows --------------------------------------------------
  public getPhase(): StudioPhase {
    if (!this.reachable) {
      return this.looked ? "off" : "loading";
    }
    if (this.submitting) {
      return "starting";
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
    return phase === "running" || phase === "busy" || phase === "off" || phase === "starting" || !this.reachable;
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

  /** The machine dial's rows, in menu order; empty without a menu. */
  public getMachineEntries(): MachineEntry[] {
    if (!this.menu) {
      return [];
    }
    const out: MachineEntry[] = [];
    for (const machine of this.menu.machines) {
      for (const way of machine.ways) {
        out.push({ machine: machine.key, way });
      }
    }
    return out;
  }

  /** The row the next POST would run on. */
  public getMachineIndex(): number {
    const workflow = this.currentWorkflow();
    return this.getMachineEntries().findIndex((e) => e.way.workflow === workflow);
  }

  /** Where the dial is resting, as an index into getMachineEntries(); null
   *  when it is not resting anywhere. */
  public getPendingIndex(): number | null {
    return this.getPendingMachine() ? this.pendingMachine : null;
  }

  /** The row the dial is resting on before it is pressed, if any. */
  public getPendingMachine(): MachineEntry | null {
    const entries = this.getMachineEntries();
    return this.pendingMachine === null ? null : entries[this.pendingMachine] ?? null;
  }

  public studioUrl(): string {
    const base = `${this.client.baseUrl()}/`;
    // The page opens on the run its address names (index.html, #run=), so
    // this is how the page is made to show what the deck started.
    return this.run?.id ? `${base}#run=${encodeURIComponent(this.run.id)}` : base;
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
      this.runGeneration += 1;
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

  /** Start an episode from the dials. One at a time: the press is locked out
   *  from the moment it is made until the Studio has answered it. */
  public async make(): Promise<void> {
    if (this.submitting || this.getPhase() !== "ready") {
      return;
    }
    const token = ++this.submitToken;
    this.submitting = true;
    this.busyReason = "";
    this.runGeneration += 1;
    this.run = null;
    this.changed();
    const current = (): boolean => token === this.submitToken;
    try {
      const id = await this.client.startEpisode(this.postableChoices());
      if (current()) {
        this.attach(id, false);
      }
    } catch (err) {
      if (!current()) {
        return;
      }
      if (err instanceof StudioUnreachable) {
        this.lost();
        return;
      }
      const status = err instanceof StudioHttpError ? err.status : 0;
      const message = (err as Error).message || "the Studio refused it";
      if (status === 409) {
        // Something is being made. Ask what, now -- not from a look that was
        // already on its way before the press -- and keep the key locked until
        // the answer is in. An episode the page started is followed; anything
        // else (a stage, a project waiting to hear how its last run ended) is
        // shown as the server put it, and the page sorts it out.
        try {
          const listing = await this.client.episodes();
          if (current()) {
            this.followListing(listing);
          }
        } catch {
          // the next look will tell
        }
        if (current() && !this.run && !this.busyReason) {
          this.busyReason = message;
        }
        return;
      }
      this.run = {
        id: "",
        external: false,
        value: 0,
        phase: "",
        minutesLeft: null,
        finished: true,
        error: message,
        seen: 0
      };
    } finally {
      if (current()) {
        this.submitting = false;
        this.changed();
      }
    }
  }

  // -- talking to the server ------------------------------------------------
  /** Look at the Studio: is it there, is anything being made. The form is
   *  read when it has not been, when the last read failed, or when the
   *  Studio has just come back. Never rejects: what goes wrong is state. */
  public tick(): Promise<void> {
    if (!this.ticking) {
      const token = ++this.tickToken;
      const request = this.formRequest;
      this.ticking = this.doTick()
        .catch((err: unknown) => {
          if (request !== this.formRequest) {
            return; // an answer from before a port change
          }
          // A listing that fails in a way the client did not class (a 500, a
          // body that is not JSON) is a Studio that is not answering properly.
          this.reachable = false;
          this.looked = true;
          this.formError = (err as Error)?.message || "the Studio answered with an error";
          this.changed();
        })
        .finally(() => {
          if (token === this.tickToken) {
            this.ticking = null;
            this.schedulePoll();
          }
        });
    }
    return this.ticking;
  }

  private async doTick(): Promise<void> {
    const request = this.formRequest;
    const runGeneration = this.runGeneration;
    let listing: RunListing[];
    try {
      listing = await this.client.episodes();
    } catch (err) {
      if (request === this.formRequest && err instanceof StudioUnreachable) {
        this.lost();
        return;
      }
      throw err;
    }
    if (request !== this.formRequest) {
      return; // the port changed under this answer
    }
    const cameBack = !this.reachable;
    this.reachable = true;
    this.looked = true;
    // A press, an attach or a clear since the listing was asked for: it does
    // not know about the run the deck now follows, so it decides nothing.
    if (runGeneration === this.runGeneration && !this.submitting) {
      this.followListing(listing);
    }
    this.changed();
    if ((cameBack || this.fields.length === 0 || this.formError) && !this.loading) {
      await this.refresh();
    }
  }

  /** Square the run the deck follows with what the Studio lists. */
  private followListing(listing: RunListing[]): void {
    const tracked = this.run?.id ? listing.find((r) => r.id === this.run?.id) : undefined;
    if (this.run && this.run.id && !this.run.finished) {
      if (!tracked) {
        // The Studio no longer knows it: it restarted.
        this.closeStream();
        this.run = null;
      } else if (!this.stream && this.reconnectTimer === null) {
        // Followed, but nothing is listening (the stream dropped while the
        // Studio was away). Open it again; a run that ended meanwhile replays
        // its last events from where the deck left off.
        this.openStream();
      }
    }
    const unfinished = listing.filter((r) => !r.finished);
    const episode = unfinished.find((r) => !r.project_id);
    const stage = unfinished.find((r) => Boolean(r.project_id));
    if (episode && (!this.run || (this.run.finished && this.run.id !== episode.id))) {
      this.attach(episode.id, true);
    }
    this.busyReason = stage && !episode ? STAGE_BUSY : "";
  }

  /** Read the form for the workflow the deck is on: its saved pick when it
   *  has one, else the Studio's own. */
  public async refresh(): Promise<void> {
    const request = ++this.formRequest;
    this.loading = true;
    this.formError = "";
    this.changed();
    try {
      let body = await this.readForm(this.wantedWorkflow(), request);
      if (request !== this.formRequest) {
        return;
      }
      this.apply(body);
      const starting = this.startingWorkflow();
      if (starting) {
        // The page opens on the detected machine's own graph, never a 16 GB
        // graph on an 8 GB card; so does the deck, once, when it has no pick.
        this.picks.workflow = starting;
        this.opts.onPicksChanged(this.getPicks());
        body = await this.readForm(starting, request);
        if (request !== this.formRequest) {
          return;
        }
        this.apply(body);
      }
    } catch (err) {
      if (request !== this.formRequest) {
        return;
      }
      if (err instanceof StudioUnreachable) {
        this.loading = false;
        this.lost();
        return;
      }
      this.formError = (err as Error).message || "the form could not be read";
    }
    this.loading = false;
    this.changed();
  }

  /** The form for `workflow`; a saved workflow the Studio no longer offers
   *  (400) is dropped and the Studio's own form read instead -- but only by
   *  the newest read, and only if that workflow is still the pick: a refusal
   *  that a later switch has overtaken must not erase the later pick. */
  private async readForm(workflow: string | undefined, request: number): Promise<OptionsResponse> {
    try {
      return await this.client.options(workflow);
    } catch (err) {
      if (!workflow || !(err instanceof StudioHttpError) || err.status !== 400 || request !== this.formRequest) {
        throw err;
      }
      if (this.picks.workflow === workflow) {
        delete this.picks.workflow;
        this.opts.onPicksChanged(this.getPicks());
      }
      return await this.client.options(undefined);
    }
  }

  private apply(body: OptionsResponse): void {
    this.fields = Array.isArray(body.fields) ? body.fields : [];
    this.menu = body.menu ?? null;
    this.reconcilePicks();
  }

  private wantedWorkflow(): string | undefined {
    const pick = this.picks.workflow;
    return typeof pick === "string" && pick ? pick : undefined;
  }

  private startingWorkflow(): string | null {
    if (!this.menu || this.wantedWorkflow()) {
      return null;
    }
    const menu = this.menu;
    const detected = menu.machines.find((m) => m.key === menu.detected);
    if (!detected || detected.key === menu.current.machine) {
      return null;
    }
    const way = detected.ways.find((w) => w.key === menu.current.way) ?? detected.ways[0];
    return way ? way.workflow : null;
  }

  private currentWorkflow(): string {
    const pick = this.wantedWorkflow();
    if (pick && this.getMachineEntries().some((e) => e.way.workflow === pick)) {
      return pick;
    }
    return this.menu?.current.workflow ?? "";
  }

  /** Drop picks this form does not offer, and a workflow not on this menu. */
  private reconcilePicks(): void {
    let dropped = false;
    for (const key of Object.keys(this.picks)) {
      if (key === "workflow") {
        if (!this.menu || !this.getMachineEntries().some((e) => e.way.workflow === this.picks.workflow)) {
          delete this.picks.workflow;
          dropped = true;
        }
        continue;
      }
      const field = this.getField(key);
      if (!field || !isOffered(field, this.picks[key])) {
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
    this.clearReconnect();
    this.run = {
      id,
      external,
      value: 0,
      phase: "",
      minutesLeft: null,
      finished: false,
      seen: 0
    };
    this.busyReason = "";
    this.changed();
    this.openStream();
  }

  private openStream(): void {
    const run = this.run;
    if (!run || run.finished || !run.id) {
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
        if (typeof event.minutes_left === "number" && Number.isFinite(event.minutes_left)) {
          run.minutesLeft = event.minutes_left;
        }
        break;
      }
      case "step":
        if (event.phase) {
          run.phase = event.phase;
        }
        break;
      case "done":
        run.finished = true;
        run.value = 1;
        run.minutesLeft = 0;
        if (typeof event.run === "string" && typeof event.file === "string" && event.run && event.file) {
          run.done = { run: event.run, file: event.file };
        } else {
          run.error = "it finished without an episode file";
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
    if (this.run !== run) {
      return;
    }
    this.stream = null;
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
      this.clearReconnect();
      this.reconnectTimer = this.opts.setTimer(() => {
        this.reconnectTimer = null;
        if (this.run === run && !this.stream) {
          this.openStream();
        }
      }, this.opts.reconnectMs);
    }
  }

  private lost(): void {
    this.reachable = false;
    this.looked = true;
    this.clearReconnect();
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

  private clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      this.opts.clearTimer(this.reconnectTimer);
      this.reconnectTimer = null;
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
