// SPDX-License-Identifier: MIT
import { afterEach, describe, expect, it } from "vitest";
import { ChoiceDialController, ControlSink } from "../src/actions/choice-dial.js";
import { MachineDialController } from "../src/actions/machine-dial.js";
import { MakeKeyController } from "../src/actions/make-key.js";
import { SettingsSync } from "../src/state/settings-sync.js";
import { STAGE_BUSY, StudioStore } from "../src/state/studio-store.js";
import { StudioClient } from "../src/studio/studio-client.js";
import { Choices } from "../src/types/contract.js";
import { FakeStudioOptions, FakeStudioServer, SAMPLE_FIELDS, SAMPLE_MENU } from "./helpers/fake-studio-server.js";

function decodeSvgUri(uri: string): string {
  const prefix = "data:image/svg+xml;utf8,";
  return uri.startsWith(prefix) ? decodeURIComponent(uri.slice(prefix.length)) : uri;
}

async function until(check: () => boolean, ms = 3000): Promise<void> {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > ms) {
      throw new Error("timed out waiting");
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}

const quietSink = (): ControlSink => ({ draw: async () => undefined, showAlert: async () => undefined });

describe("the deck against a Studio", () => {
  let server: FakeStudioServer | null = null;
  let store: StudioStore | null = null;
  const extraServers: FakeStudioServer[] = [];

  afterEach(async () => {
    store?.stop();
    store = null;
    await server?.stop();
    server = null;
    for (const s of extraServers.splice(0)) {
      await s.stop();
    }
  });

  async function boot(
    opts: FakeStudioOptions = {},
    before?: (store: StudioStore) => void
  ): Promise<{ server: FakeStudioServer; store: StudioStore; saved: Choices[]; port: number }> {
    server = new FakeStudioServer(opts);
    const port = await server.start();
    const saved: Choices[] = [];
    store = new StudioStore(new StudioClient(port), {
      offPollMs: 40,
      idlePollMs: 40,
      reconnectMs: 20,
      onPicksChanged: (p) => saved.push(p)
    });
    before?.(store);
    store.start();
    await until(() => store!.getPhase() === "ready");
    return { server, store, saved, port };
  }

  it("passes the server's Host and JSON rules as a plain Node client, with no server change", async () => {
    const { server, store } = await boot();
    await store.make();
    expect(server.refusedHosts).toBe(0);
    expect(server.posts).toHaveLength(1);
    expect(server.posts[0].host).toMatch(/^127\.0\.0\.1:\d+$/);
    expect(server.posts[0].contentType).toBe("application/json");
  });

  it("posts what the dials show, in the form's own keys, plus the menu workflow", async () => {
    const { server, store, saved } = await boot();
    expect(store.cycle("bank", 1)).toBe(true);
    expect(store.cycle("style", 2)).toBe(true);
    expect(store.cycle("acts", -1)).toBe(true);
    expect(saved.at(-1)).toEqual({ bank: "shakespeare", style: "woodcut", acts: "2" });
    await store.make();
    expect(server.posts[0].body).toEqual({
      choices: { language: "en", bank: "shakespeare", style: "woodcut", acts: "2", workflow: "otr_canonical.json" }
    });
  });

  it("follows the run's events: progress only climbs, 100% comes with done, and the key opens the file", async () => {
    const { server, store } = await boot();
    const opened: string[] = [];
    const images: string[] = [];
    const key = new MakeKeyController(
      store,
      {
        setImage: async (img) => {
          images.push(decodeSvgUri(img));
        },
        showAlert: async () => undefined,
        openUrl: async (url) => {
          opened.push(url);
        }
      },
      1,
      50
    );
    key.appear();
    key.keyDown();
    await key.keyUp();
    await until(() => store.getPhase() === "running");
    const id = store.getRun()!.id;
    expect(store.isLocked()).toBe(true);
    expect(store.cycle("bank", 1)).toBe(false);

    server.emit(id, { type: "submitted" });
    server.emit(id, { type: "progress", value: 0.4, phase: "phase.writing", minutes_left: 12 });
    server.emit(id, { type: "progress", value: 0.2 });
    server.emit(id, { type: "step", value: 0.9, step: 9, steps: 10 });
    await until(() => store.getRun()!.value === 0.4);
    // The phase arrives as the page's string key and is shown as a word.
    await until(() => images.some((svg) => svg.includes(">40%<") && svg.includes(">writing<") && svg.includes(">12 min left<")));
    expect(images.some((svg) => svg.includes("phase."))).toBe(false);

    server.emit(id, { type: "done", path: "x", run: "n16f_abc", file: "episode final.mp4" });
    await until(() => store.getPhase() === "done");
    expect(store.getRun()!.value).toBe(1);
    const url = `${store.getClient().baseUrl()}/episodes/n16f_abc/episode%20final.mp4`;
    expect(store.episodeUrl()).toBe(url);

    key.keyDown();
    await key.keyUp();
    expect(opened).toEqual([url]);
    expect(store.getPhase()).toBe("ready"); // played once; the next press makes another
    key.disappear();
  });

  it("shows a failed run's reason and a press clears it without making another", async () => {
    const { server, store } = await boot();
    await store.make();
    const id = store.getRun()!.id;
    server.emit(id, { type: "error", message: "ComfyUI said no" });
    await until(() => store.getPhase() === "error");
    expect(store.getRun()!.error).toBe("ComfyUI said no");
    expect(await store.press()).toBe("cleared");
    expect(store.getPhase()).toBe("ready");
    expect(server.posts).toHaveLength(1);
  });

  it("follows an episode the page started, and treats a 409 with nothing to follow as busy", async () => {
    const { server, store } = await boot();
    const theirs = server.addRun({ id: "aaaaaaaaaaaa" });
    await until(() => store.getPhase() === "running");
    expect(store.getRun()!.external).toBe(true);
    expect(store.getRun()!.id).toBe(theirs.id);
    server.emit(theirs.id, { type: "done", path: "x", run: "r", file: "f.mp4" });
    await until(() => store.getPhase() === "done");

    server.configure({ postFailure: { status: 409, error: "project p1 is waiting to hear how its last run ended" } });
    store.clearRun();
    expect(store.getPhase()).toBe("ready");
    await store.make();
    expect(store.getPhase()).toBe("busy");
    expect(store.getBusyReason()).toContain("project p1");
  });

  it("treats a Scene Producer stage as busy and never attaches to it", async () => {
    const { server, store } = await boot();
    server.addRun({ id: "bbbbbbbbbbbb", project_id: "p1", stage: "stills" });
    await until(() => store.getPhase() === "busy");
    expect(store.getBusyReason()).toBe(STAGE_BUSY);
    expect(store.getRun()).toBeNull();
    expect(store.cycle("bank", 1)).toBe(false);
  });

  it("switches machine on the press, re-reads the form for it, and drops picks it does not offer", async () => {
    const { server, store, saved } = await boot({
      forms: {
        "": { fields: SAMPLE_FIELDS, menu: SAMPLE_MENU },
        "otr_8gb_still.json": {
          fields: SAMPLE_FIELDS.map((f) => (f.key === "style" ? { ...f, choices: ["noir"] } : f)),
          menu: SAMPLE_MENU
        }
      }
    });
    store.cycle("style", 1); // technicolor
    const dial = new MachineDialController(store, quietSink(), 1);
    expect(store.getMachineIndex()).toBe(2); // 16 GB foley, the server's own
    await dial.rotate(-1);
    expect(store.getPendingMachine()!.way.workflow).toBe("otr_8gb_still.json");
    await dial.push();
    expect(store.getPicks().workflow).toBe("otr_8gb_still.json");
    expect(server.optionsRequests.at(-1)).toBe("otr_8gb_still.json");
    expect(store.getValue("style")).toBe("noir");
    expect(saved.at(-1)).toEqual({ workflow: "otr_8gb_still.json" });
    await store.make();
    expect((server.posts[0].body as { choices: Choices }).choices.workflow).toBe("otr_8gb_still.json");
  });

  it("with no menu, the machine dial has nothing to turn and no workflow is posted", async () => {
    const { server, store } = await boot({ forms: { "": { fields: SAMPLE_FIELDS } } });
    expect(store.rotateMachine(1)).toBe(false);
    await store.make();
    expect((server.posts[0].body as { choices: Choices }).choices.workflow).toBeUndefined();
  });

  it("opens on the detected machine's own graph when the Studio's default is another card's", async () => {
    const { server, store } = await boot({
      forms: { "": { fields: SAMPLE_FIELDS, menu: { ...SAMPLE_MENU, detected: "nvidia8" } } }
    });
    expect(store.getPicks().workflow).toBe("otr_8gb_ltx25_foley.json");
    expect(server.optionsRequests).toEqual(["", "otr_8gb_ltx25_foley.json"]);
  });

  it("goes off when the Studio is gone and comes back when it answers", async () => {
    const { server, store } = await boot();
    await server.stop();
    await until(() => store.getPhase() === "off");
    const again = new FakeStudioServer();
    extraServers.push(again);
    const port = await again.start();
    store.getClient().setPort(port);
    store.endpointChanged();
    await until(() => store.getPhase() === "ready");
  });

  it("resumes a cut event stream where it left off", async () => {
    const { server, store } = await boot();
    await store.make();
    const id = store.getRun()!.id;
    server.emit(id, { type: "progress", value: 0.3 });
    await until(() => store.getRun()!.value === 0.3);
    server.cutStreams();
    await until(() => server.eventRequests.length >= 2);
    expect(server.eventRequests[1].lastEventId).toBe("0");
    server.emit(id, { type: "done", path: "x", run: "r", file: "f" });
    await until(() => store.getPhase() === "done");
  });

  it("draws the dial as a node: its name, the pick, and a green bar while making", async () => {
    const { server, store } = await boot();
    const frames: string[] = [];
    const dial = new ChoiceDialController(
      store,
      { draw: async (uri) => void frames.push(decodeSvgUri(uri)), showAlert: async () => undefined },
      { field: "bank" },
      1
    );
    dial.appear();
    await dial.rotate(2);
    await until(() => frames.some((f) => f.includes(">Public domain<")));
    expect(frames.at(-1)).toContain(">STORY BANK<");
    expect(frames.at(-1)).toContain(">3/3<");
    await store.make();
    server.emit(store.getRun()!.id, { type: "progress", value: 0.5 });
    await until(() => frames.some((f) => f.includes(">on air<") && f.includes(`width="81.0" height="3" fill="#50F898"`)));
    dial.disappear();
  });

  // -- the review's findings (Codex and Sonnet, 2026-10-28), each pinned ------

  it("reads the saved picks before the first form, and asks for the saved workflow's form", async () => {
    const { server, store } = await boot({}, (s) => s.setPicks({ workflow: "otr_8gb_still.json", bank: "shakespeare" }));
    expect(server.optionsRequests[0]).toBe("otr_8gb_still.json");
    expect(store.getValue("bank")).toBe("shakespeare");
    await store.make();
    expect((server.posts[0].body as { choices: Choices }).choices).toMatchObject({
      bank: "shakespeare",
      workflow: "otr_8gb_still.json"
    });
  });

  it("drops a saved workflow the Studio no longer offers and reads its own form", async () => {
    const { server, store, saved } = await boot({}, (s) => s.setPicks({ workflow: "otr_gone.json" }));
    expect(server.optionsRequests.slice(0, 2)).toEqual(["otr_gone.json", ""]);
    expect(store.getPicks().workflow).toBeUndefined();
    expect(saved.some((p) => p.workflow === undefined)).toBe(true);
  });

  it("loads the saved port and picks through SettingsSync, and writes both back whole", async () => {
    server = new FakeStudioServer();
    const port = await server.start();
    const client = new StudioClient(8765);
    const written: Array<Record<string, unknown>> = [];
    const sync = new SettingsSync(client, async (s) => void written.push(s));
    store = new StudioStore(client, { offPollMs: 40, idlePollMs: 40, onPicksChanged: (p) => sync.picksChanged(p) });
    sync.attach(store);
    sync.load({ port, picksJson: JSON.stringify({ style: "woodcut" }) });
    store.start();
    await until(() => store!.getPhase() === "ready");
    expect(client.getPort()).toBe(port);
    expect(store.getValue("style")).toBe("woodcut");
    store.cycle("bank", 1);
    expect(written.at(-1)).toEqual({ port, picksJson: JSON.stringify({ style: "woodcut", bank: "shakespeare" }) });
  });

  it("drops a slow form answer that a later machine switch has overtaken", async () => {
    const { server, store } = await boot({
      forms: {
        "": { fields: SAMPLE_FIELDS, menu: SAMPLE_MENU },
        "otr_8gb_still.json": { fields: SAMPLE_FIELDS.map((f) => (f.key === "style" ? { ...f, choices: ["old"] } : f)), menu: SAMPLE_MENU },
        "otr_16gb_video.json": { fields: SAMPLE_FIELDS.map((f) => (f.key === "style" ? { ...f, choices: ["new"] } : f)), menu: SAMPLE_MENU }
      },
      optionsDelayMs: { "otr_8gb_still.json": 150 }
    });
    store.rotateMachine(-1); // 8 GB stills, slow
    const slow = store.applyMachine();
    await until(() => server.optionsRequests.includes("otr_8gb_still.json"));
    store.rotateMachine(2); // 16 GB video, quick
    const quick = store.applyMachine();
    await Promise.all([slow, quick]);
    await new Promise((r) => setTimeout(r, 200));
    expect(store.getPicks().workflow).toBe("otr_16gb_video.json");
    expect(store.getValue("style")).toBe("new");
  });

  it("locks the key while the press is on its way, so two presses make one POST", async () => {
    const { server, store } = await boot({ postDelayMs: 120 });
    const first = store.press();
    await until(() => store.getPhase() === "starting");
    expect(await store.press()).toBe("refused");
    expect(store.cycle("bank", 1)).toBe(false);
    await first;
    expect(server.posts).toHaveLength(1);
    expect(store.getPhase()).toBe("running");
  });

  it("picks a run back up after the Studio blinks out and returns, and sees it finish", async () => {
    const { server, store, port } = await boot();
    await store.make();
    const id = store.getRun()!.id;
    server.emit(id, { type: "progress", value: 0.25 });
    await until(() => store.getRun()!.value === 0.25);
    await server.stop();
    await until(() => store.getPhase() === "off");
    server.emit(id, { type: "progress", value: 0.6 });
    server.emit(id, { type: "done", path: "x", run: "r1", file: "f.mp4" });
    await server.start(port);
    await until(() => store.getPhase() === "done");
    expect(store.getRun()!.id).toBe(id);
    expect(store.episodeUrl()).toContain("/episodes/r1/f.mp4");
  });

  it("recovers from a failed form read on the next look, with the old form still in hand", async () => {
    const { server, store } = await boot();
    server.configure({ optionsFailure: { status: 503, error: "ComfyUI is not answering" } });
    await store.refresh();
    expect(store.getPhase()).toBe("form_error");
    expect(store.getFormError()).toBe("ComfyUI is not answering");
    expect(await store.press()).toBe("refused");
    server.configure({ optionsFailure: undefined });
    await until(() => store.getPhase() === "ready");
  });

  it("starts over against a new port: the old server's form, run and stream are dropped", async () => {
    const { store } = await boot();
    await store.make();
    expect(store.getPhase()).toBe("running");
    const other = new FakeStudioServer({ forms: { "": { fields: SAMPLE_FIELDS.slice(0, 2) } } });
    extraServers.push(other);
    const port = await other.start();
    store.getClient().setPort(port);
    store.endpointChanged();
    await until(() => store.getPhase() === "ready");
    expect(store.getRun()).toBeNull();
    expect(other.optionsRequests).toEqual([""]);
    expect(store.getField("style")).toBeUndefined();
  });

  it("never lets a failed look escape as a rejection: a 500 listing is shown and retried", async () => {
    const { server, store } = await boot();
    server.configure({ listFailure: { status: 500, error: "internal error" } });
    await expect(store.tick()).resolves.toBeUndefined();
    expect(store.getPhase()).toBe("off");
    server.configure({ listFailure: undefined });
    await until(() => store.getPhase() === "ready");
  });

  it("a dial turned while held does not reset on release; a plain press does", async () => {
    const { store } = await boot();
    const dial = new ChoiceDialController(store, quietSink(), { field: "style" }, 1);
    dial.dialDown();
    await dial.rotate(1, true);
    await dial.push();
    expect(store.getValue("style")).toBe("technicolor");
    dial.dialDown();
    await dial.push();
    expect(store.getValue("style")).toBe("noir");
  });

  it("on a deck without dials, a key press steps to the next choice and a hold resets it", async () => {
    const { store } = await boot();
    const frames: string[] = [];
    const key = new ChoiceDialController(
      store,
      { draw: async (uri) => void frames.push(decodeSvgUri(uri)), showAlert: async () => undefined },
      { field: "language" },
      1,
      "key",
      60
    );
    key.appear();
    key.keyDown();
    await key.keyUp();
    expect(store.getValue("language")).toBe("fr");
    await until(() => frames.some((f) => f.includes(">Fr<") && f.includes(">hold: reset<") && f.includes('width="144"')));
    key.keyDown();
    await new Promise((r) => setTimeout(r, 100));
    await key.keyUp();
    expect(store.getValue("language")).toBe("en");
    key.disappear();
  });

  it("on a key, the machine steps on a press and switches on a hold", async () => {
    const { server, store } = await boot();
    const key = new MachineDialController(store, quietSink(), 1, "key", 60);
    key.keyDown();
    await key.keyUp();
    expect(store.getPendingMachine()!.way.workflow).toBe("otr_16gb_video.json");
    key.keyDown();
    await new Promise((r) => setTimeout(r, 100));
    await key.keyUp();
    await until(() => store.getPicks().workflow === "otr_16gb_video.json");
    expect(server.optionsRequests.at(-1)).toBe("otr_16gb_video.json");
  });
});
