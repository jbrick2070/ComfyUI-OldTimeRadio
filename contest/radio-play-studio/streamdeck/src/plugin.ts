// SPDX-License-Identifier: MIT
import streamDeck from "@elgato/streamdeck";
import { ChoiceDialAction } from "./actions/choice-dial.js";
import { MachineDialAction } from "./actions/machine-dial.js";
import { MakeKeyAction } from "./actions/make-key.js";
import { StudioStore } from "./state/studio-store.js";
import { StudioClient } from "./studio/studio-client.js";
import { Choices } from "./types/contract.js";
import { DEFAULT_GLOBAL_SETTINGS, normalizePort, StudioGlobalSettings } from "./types/settings.js";

const client = new StudioClient(DEFAULT_GLOBAL_SETTINGS.port);
let globalSettings: Required<StudioGlobalSettings> = { ...DEFAULT_GLOBAL_SETTINGS };

const store = new StudioStore(client, {
  onPicksChanged: (picks: Choices) => {
    globalSettings = { ...globalSettings, picksJson: JSON.stringify(picks) };
    void streamDeck.settings.setGlobalSettings(globalSettings);
  }
});

function readPicks(json: unknown): Choices {
  try {
    const parsed = JSON.parse(String(json ?? "{}")) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const out: Choices = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof value === "string" || typeof value === "number") {
          out[key] = value;
        }
      }
      return out;
    }
  } catch {
    // unreadable picks start over
  }
  return {};
}

function sendStateToPropertyInspector(): void {
  if (!streamDeck.ui.action) {
    return;
  }
  const run = store.getRun();
  void streamDeck.ui.sendToPropertyInspector({
    event: "studioState",
    phase: store.getPhase(),
    port: client.getPort(),
    studioUrl: store.studioUrl(),
    formError: store.getFormError(),
    busyReason: store.getBusyReason(),
    runError: run?.error ?? "",
    picks: store.getPicks()
  });
}

store.on("changed", sendStateToPropertyInspector);

streamDeck.settings.onDidReceiveGlobalSettings<StudioGlobalSettings>((ev) => {
  const port = normalizePort(ev.settings.port);
  globalSettings = { port, picksJson: String(ev.settings.picksJson ?? "{}") };
  const portChanged = port !== client.getPort();
  client.setPort(port);
  store.setPicks(readPicks(globalSettings.picksJson));
  if (portChanged) {
    void store.tick();
  }
  sendStateToPropertyInspector();
});

streamDeck.ui.onDidAppear(() => sendStateToPropertyInspector());

streamDeck.ui.onSendToPlugin<{ event?: string; port?: string | number }>((ev) => {
  const payload = ev.payload ?? {};
  if (payload.event === "piReady") {
    sendStateToPropertyInspector();
  } else if (payload.event === "updatePort") {
    const port = normalizePort(payload.port);
    globalSettings = { ...globalSettings, port };
    void streamDeck.settings.setGlobalSettings(globalSettings);
    client.setPort(port);
    void store.tick();
    sendStateToPropertyInspector();
  } else if (payload.event === "openStudio") {
    void streamDeck.system.openUrl(store.studioUrl());
  }
});

streamDeck.actions.registerAction(new ChoiceDialAction(store));
streamDeck.actions.registerAction(new MachineDialAction(store));
streamDeck.actions.registerAction(new MakeKeyAction(store, (url) => streamDeck.system.openUrl(url)));

void streamDeck.connect().then(() => {
  void streamDeck.settings.getGlobalSettings();
  store.start();
});
