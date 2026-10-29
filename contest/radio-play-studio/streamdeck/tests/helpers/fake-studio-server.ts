// SPDX-License-Identifier: MIT
// A stand-in for radio_play_studio/server.py with the rules the plugin must
// meet: a loopback Host only, JSON posts only, 202 {id} for an episode, 409
// while one is being made, and the event stream with ids and keep-alives.
import { createServer, IncomingMessage, Server, ServerResponse } from "node:http";
import { AddressInfo } from "node:net";

export interface FakeRun {
  id: string;
  finished: boolean;
  started: number;
  project_id?: string;
  stage?: string;
  events: Array<Record<string, unknown>>;
}

export interface FakeStudioOptions {
  /** The form for each workflow name (the server's own under ""). */
  forms?: Record<string, { fields: unknown[]; menu?: unknown }>;
  /** Answer /api/options with this status and error instead. */
  optionsFailure?: { status: number; error: string };
  /** Answer POST /api/episodes with this instead of starting a run. */
  postFailure?: { status: number; error: string };
  /** Answer GET /api/episodes with this instead of the listing. */
  listFailure?: { status: number; error: string };
  /** Hold the /api/options answer for a workflow this long (ms). */
  optionsDelayMs?: Record<string, number>;
  /** Hold the POST /api/episodes answer this long (ms), the run made. */
  postDelayMs?: number;
  /** Hold the GET /api/episodes answer this long (ms); the listing is taken
   *  when the request arrives, as the real server's is. */
  listDelayMs?: number;
}

const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]"]);

export const SAMPLE_FIELDS = [
  { key: "language", label: "Language of the play", group: "The show", help: "", kind: "choice",
    default: "en", choices: ["en", "fr", "de"] },
  { key: "bank", label: "Story bank", group: "The show", help: "", kind: "choice",
    default: "roll", choices: ["roll", "shakespeare", "public_domain", "my_story"] },
  { key: "acts", label: "Acts", group: "The show", help: "", kind: "choice",
    default: "3", choices: ["1", "2", "3", "4", "5"] },
  { key: "cast_size", label: "Cast size", group: "The show", help: "", kind: "int",
    default: 4, min: 2, max: 8 },
  { key: "style", label: "Visual style", group: "The show", help: "", kind: "choice",
    default: "noir", choices: ["noir", "technicolor", "woodcut"] },
  { key: "my_plot", label: "What happens", group: "My story", help: "", kind: "text", default: "" }
];

export const SAMPLE_MENU = {
  machines: [
    { key: "nvidia8", ways: [
      { key: "foley", workflow: "otr_8gb_ltx25_foley.json", proven: true },
      { key: "stills", workflow: "otr_8gb_still.json", proven: true }
    ] },
    { key: "nvidia16", ways: [
      { key: "foley", workflow: "otr_canonical.json", proven: true },
      { key: "video", workflow: "otr_16gb_video.json", proven: false }
    ] }
  ],
  detected: "nvidia16",
  current: { machine: "nvidia16", way: "foley", workflow: "otr_canonical.json" }
};

export class FakeStudioServer {
  public readonly runs: FakeRun[] = [];
  public readonly posts: Array<{ host: string; contentType: string; body: unknown }> = [];
  public readonly optionsRequests: string[] = [];
  public listRequests = 0;
  /** What happened, in order, so a test can prove the interleaving it sets up
   *  actually took place rather than assume the timers fell that way:
   *  `listing <n> taken: <ids or none>`, `listing <n> answered`,
   *  `run made: <id>`, `post answered: <id>`, `events <id>: no such run`. */
  public readonly timeline: string[] = [];
  public readonly eventRequests: Array<{ id: string; lastEventId: string | undefined }> = [];
  public refusedHosts = 0;
  private server: Server | null = null;
  private opts: FakeStudioOptions;
  private waiters: Array<() => void> = [];
  private readonly openStreams = new Set<ServerResponse>();
  private postGate: Promise<void> | null = null;
  private listGate: Promise<void> | null = null;

  public constructor(opts: FakeStudioOptions = {}) {
    this.opts = opts;
  }

  public configure(opts: FakeStudioOptions): void {
    this.opts = { ...this.opts, ...opts };
  }

  /** Listen on `port` (0: any free one); the runs survive a stop and a start,
   *  as they would not on the real server, so a blip can be tested. */
  public async start(port = 0): Promise<number> {
    this.server = createServer((req, res) => this.handle(req, res));
    await new Promise<void>((resolve) => this.server!.listen(port, "127.0.0.1", resolve));
    return (this.server.address() as AddressInfo).port;
  }

  public async stop(): Promise<void> {
    if (!this.server) {
      return;
    }
    const server = this.server;
    this.server = null;
    this.wake();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  /** Hold every POST /api/episodes from now on, after its body is read and
   *  before its run exists, until the returned function is called: the window
   *  where the real server checks the choices against ComfyUI. */
  public holdPosts(): () => void {
    let release!: () => void;
    this.postGate = new Promise<void>((resolve) => (release = resolve));
    return () => {
      this.postGate = null;
      release();
    };
  }

  /** Hold the answer to every GET /api/episodes from now on until the returned
   *  function is called; each listing is still taken when its request arrives. */
  public holdListings(): () => void {
    let release!: () => void;
    this.listGate = new Promise<void>((resolve) => (release = resolve));
    return () => {
      this.listGate = null;
      release();
    };
  }

  /** A run the page started, listed as unfinished. */
  public addRun(run: Partial<FakeRun> & { id: string }): FakeRun {
    const full: FakeRun = { finished: false, started: Date.now() / 1000, events: [], ...run };
    this.runs.push(full);
    return full;
  }

  /** Append an event to a run; `done`/`error` finish it. */
  public emit(id: string, event: Record<string, unknown>): void {
    const run = this.runs.find((r) => r.id === id);
    if (!run) {
      throw new Error(`no run ${id}`);
    }
    run.events.push(event);
    if (event.type === "done" || event.type === "error") {
      run.finished = true;
    }
    this.wake();
  }

  /** Close every open event stream mid-run, as the server does to a socket
   *  that has been quiet past its timeout. */
  public cutStreams(): void {
    for (const res of this.openStreams) {
      res.destroy();
    }
    this.openStreams.clear();
  }

  /** End every open event stream cleanly mid-run, as a server does that
   *  closes a quiet stream: the client sees an ordinary end, not an error. */
  public endStreams(): void {
    for (const res of this.openStreams) {
      res.end();
    }
    this.openStreams.clear();
  }

  private wake(): void {
    const waiters = this.waiters;
    this.waiters = [];
    for (const w of waiters) {
      w();
    }
  }

  private hostOk(req: IncomingMessage): boolean {
    const host = (req.headers.host ?? "").trim().toLowerCase();
    const name = host.startsWith("[") ? host.slice(0, host.indexOf("]") + 1) : host.split(":")[0];
    return LOOPBACK.has(name);
  }

  private json(res: ServerResponse, status: number, body: unknown): void {
    const raw = Buffer.from(JSON.stringify(body), "utf8");
    res.writeHead(status, {
      "Content-Type": "application/json; charset=utf-8",
      "Content-Length": String(raw.length),
      "Cache-Control": "no-store"
    });
    res.end(raw);
  }

  private handle(req: IncomingMessage, res: ServerResponse): void {
    if (!this.hostOk(req)) {
      this.refusedHosts += 1;
      this.json(res, 403, { error: "this app only answers on localhost" });
      return;
    }
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (req.method === "GET") {
      this.get(url, req, res);
      return;
    }
    if (req.method === "POST") {
      this.post(url, req, res);
      return;
    }
    this.json(res, 404, { error: "not found" });
  }

  private get(url: URL, req: IncomingMessage, res: ServerResponse): void {
    if (url.pathname === "/api/options") {
      const workflow = url.searchParams.get("workflow") ?? "";
      this.optionsRequests.push(workflow);
      if (this.opts.optionsFailure) {
        this.json(res, this.opts.optionsFailure.status, { error: this.opts.optionsFailure.error });
        return;
      }
      const forms = this.opts.forms ?? { "": { fields: SAMPLE_FIELDS, menu: SAMPLE_MENU } };
      const own = forms[""];
      const ownMenu = own?.menu as typeof SAMPLE_MENU | undefined;
      const offered = !workflow || workflow in forms ||
        Boolean(ownMenu?.machines.some((m) => m.ways.some((w) => w.workflow === workflow)));
      const form = forms[workflow] ?? own;
      const delay = this.opts.optionsDelayMs?.[workflow] ?? 0;
      const answer = (status: number, body: unknown): void => {
        if (delay > 0) {
          setTimeout(() => this.json(res, status, body), delay);
        } else {
          this.json(res, status, body);
        }
      };
      if (!form || !offered) {
        answer(400, { error: `${workflow} is not one of the workflows this page offers` });
        return;
      }
      const menu = form.menu as typeof SAMPLE_MENU | undefined;
      const body: Record<string, unknown> = { fields: form.fields, photo_stills: [] };
      if (menu) {
        const current = workflow
          ? menu.machines.flatMap((m) => m.ways.map((w) => ({ machine: m.key, way: w.key, workflow: w.workflow })))
              .find((e) => e.workflow === workflow) ?? menu.current
          : menu.current;
        body.menu = { ...menu, current };
      }
      answer(200, body);
      return;
    }
    if (url.pathname === "/api/episodes") {
      if (this.opts.listFailure) {
        this.json(res, this.opts.listFailure.status, { error: this.opts.listFailure.error });
        return;
      }
      this.listRequests += 1;
      const n = this.listRequests;
      const listing = {
        episodes: this.runs
          .slice()
          .sort((a, b) => b.started - a.started)
          .map(({ events: _events, ...rest }) => rest)
      };
      this.timeline.push(`listing ${n} taken: ${listing.episodes.map((r) => r.id).join(",") || "none"}`);
      const answer = (): void => {
        this.timeline.push(`listing ${n} answered`);
        this.json(res, 200, listing);
      };
      const listDelay = this.opts.listDelayMs ?? 0;
      const gate = this.listGate;
      if (gate) {
        void gate.then(answer);
      } else if (listDelay > 0) {
        setTimeout(answer, listDelay);
      } else {
        answer();
      }
      return;
    }
    const events = /^\/api\/episodes\/([0-9a-f]+)\/events$/.exec(url.pathname);
    if (events) {
      this.eventStream(events[1], req, res);
      return;
    }
    this.json(res, 404, { error: "not found" });
  }

  private post(url: URL, req: IncomingMessage, res: ServerResponse): void {
    if (url.pathname !== "/api/episodes") {
      this.json(res, 404, { error: "not found" });
      return;
    }
    const contentType = (req.headers["content-type"] ?? "").split(";")[0].trim().toLowerCase();
    if (contentType !== "application/json") {
      this.json(res, 415, { error: "send application/json" });
      return;
    }
    const parts: Buffer[] = [];
    req.on("data", (c: Buffer) => parts.push(c));
    req.on("end", () => {
      let body: unknown;
      try {
        body = JSON.parse(Buffer.concat(parts).toString("utf8") || "{}");
      } catch {
        this.json(res, 400, { error: "bad request: not JSON" });
        return;
      }
      this.posts.push({ host: req.headers.host ?? "", contentType, body });
      if (this.opts.postFailure) {
        this.json(res, this.opts.postFailure.status, { error: this.opts.postFailure.error });
        return;
      }
      // Busy is refused before the choices are checked and again after, as
      // server.py's Runs._launch does: a run may start while they are.
      const busy = (): boolean => {
        if (this.runs.some((r) => !r.finished)) {
          this.json(res, 409, { error: "an episode is already being made; the page shows its progress" });
          return true;
        }
        return false;
      };
      if (busy()) {
        return;
      }
      const start = (): void => {
        if (!busy()) {
          this.startRun(res);
        }
      };
      const gate = this.postGate;
      if (gate) {
        void gate.then(start);
      } else {
        start();
      }
    });
  }

  private startRun(res: ServerResponse): void {
    const run = this.addRun({ id: (this.runs.length + 1).toString(16).padStart(12, "0") });
    this.timeline.push(`run made: ${run.id}`);
    const answer = (): void => {
      this.timeline.push(`post answered: ${run.id}`);
      this.json(res, 202, { id: run.id });
    };
    const delay = this.opts.postDelayMs ?? 0;
    if (delay > 0) {
      setTimeout(answer, delay);
    } else {
      answer();
    }
  }

  private eventStream(id: string, req: IncomingMessage, res: ServerResponse): void {
    const run = this.runs.find((r) => r.id === id);
    if (!run) {
      this.timeline.push(`events ${id}: no such run`);
      this.json(res, 404, { error: "no such episode" });
      return;
    }
    const lastId = req.headers["last-event-id"];
    this.eventRequests.push({ id, lastEventId: Array.isArray(lastId) ? lastId[0] : lastId });
    let sent = Math.max(0, Number(lastId ?? -1) + 1);
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-store" });
    this.openStreams.add(res);
    res.on("close", () => this.openStreams.delete(res));
    const pump = (): void => {
      if (res.destroyed || res.writableEnded || !this.server) {
        return;
      }
      while (sent < run.events.length) {
        res.write(`id: ${sent}\ndata: ${JSON.stringify(run.events[sent])}\n\n`);
        sent += 1;
      }
      if (run.finished) {
        res.end();
        return;
      }
      res.write(": keep-alive\n\n");
      this.waiters.push(pump);
    };
    pump();
  }
}
