// SPDX-License-Identifier: MIT
import { afterEach, describe, expect, it } from "vitest";
import { ChoiceDialController } from "../src/actions/choice-dial.js";
import { MachineDialController } from "../src/actions/machine-dial.js";
import { MakeKeyController } from "../src/actions/make-key.js";
import { StudioStore } from "../src/state/studio-store.js";
import { StudioClient } from "../src/studio/studio-client.js";
import { Choices } from "../src/types/contract.js";
import { FakeStudioServer, SAMPLE_FIELDS, SAMPLE_MENU } from "./helpers/fake-studio-server.js";

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

describe("the deck against a Studio", () => {
  let server: FakeStudioServer | null = null;
  let store: StudioStore | null = null;

  afterEach(async () => {
    store?.stop();
    store = null;
    await server?.stop();
    server = null;
  });

  async function boot(opts = {}): Promise<{ server: FakeStudioServer; store: StudioStore; saved: Choices[] }> {
    server = new FakeStudioServer(opts);
    const port = await server.start();
    const saved: Choices[] = [];
    store = new StudioStore(new StudioClient(port), {
      offPollMs: 50,
      idlePollMs: 50,
      reconnectMs: 20,
      onPicksChanged: (p) => saved.push(p)
    });
    store.start();
    await until(() => store!.getPhase() === "ready");
    return { server, store, saved };
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
      choices: {
        language: "en",
        bank: "shakespeare",
        style: "woodcut",
        acts: "2",
        workflow: "otr_canonical.json"
      }
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
    server.emit(id, { type: "progress", value: 0.4, phase: "writing", minutes_left: 12 });
    server.emit(id, { type: "progress", value: 0.2 });
    server.emit(id, { type: "step", value: 0.9, step: 9, steps: 10 });
    await until(() => store.getRun()!.value === 0.4);
    expect(store.getRun()!.phase).toBe("writing");
    await until(() => images.some((svg) => svg.includes(">40%<") && svg.includes("writing · 12 min")));

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
    expect(store.getRun()).toBeNull();
    expect(store.cycle("bank", 1)).toBe(false);
  });

  it("switches machine on the push, re-reads the form for it, and drops picks it does not offer", async () => {
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
    const dial = new MachineDialController(store, { setFeedback: async () => undefined, showAlert: async () => undefined }, 1);
    expect(store.getMachineIndex()).toBe(2); // 16 GB foley, the server's own
    await dial.rotate(-1);
    expect(store.getPendingMachine()!.label).toBe("8 GB · stills");
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
      forms: {
        "": { fields: SAMPLE_FIELDS, menu: { ...SAMPLE_MENU, detected: "nvidia8" } }
      }
    });
    expect(store.getPicks().workflow).toBe("otr_8gb_ltx25_foley.json");
    expect(server.optionsRequests).toEqual(["", "otr_8gb_ltx25_foley.json"]);
  });

  it("reports the form's failure and will not make until it reads", async () => {
    const { server, store } = await boot();
    server.configure({ optionsFailure: { status: 503, error: "ComfyUI is not answering" } });
    await store.refresh();
    expect(store.getPhase()).toBe("form_error");
    expect(store.getFormError()).toBe("ComfyUI is not answering");
    expect(await store.press()).toBe("refused");
  });

  it("goes off when the Studio is gone and comes back when it answers", async () => {
    const { server, store } = await boot();
    await server.stop();
    await until(() => store.getPhase() === "off");
    const again = new FakeStudioServer();
    const port = await again.start();
    store.getClient().setPort(port);
    await until(() => store.getPhase() === "ready");
    await again.stop();
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

  it("draws the dial: the field's name, the pick, and a bar while making", async () => {
    const { server, store } = await boot();
    const frames: string[] = [];
    const dial = new ChoiceDialController(
      store,
      { setFeedback: async (fb) => { frames.push(decodeSvgUri(fb.canvas)); }, showAlert: async () => undefined },
      { field: "bank" },
      1
    );
    dial.appear();
    await dial.rotate(2);
    await until(() => frames.some((f) => f.includes("public domain")));
    expect(frames.at(-1)).toContain("STORY BANK");
    await store.make();
    server.emit(store.getRun()!.id, { type: "progress", value: 0.5 });
    await until(() => frames.some((f) => f.includes("making an episode") && f.includes("width=\"100.0\"")));
    dial.disappear();
  });
});
