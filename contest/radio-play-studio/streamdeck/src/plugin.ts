// SPDX-License-Identifier: MIT
import streamDeck from "@elgato/streamdeck";
import { ChoiceDialAction } from "./actions/choice-dial.js";
import { MachineDialAction } from "./actions/machine-dial.js";
import { MakeKeyAction } from "./actions/make-key.js";
import { SettingsSync } from "./state/settings-sync.js";
import { StudioStore } from "./state/studio-store.js";
import { StudioClient } from "./studio/studio-client.js";
import { DEFAULT_GLOBAL_SETTINGS, StudioGlobalSettings } from "./types/settings.js";

const client = new StudioClient(DEFAULT_GLOBAL_SETTINGS.port);
const settings = new SettingsSync(client, (s) => streamDeck.settings.setGlobalSettings(s));
const store = new StudioStore(client, { onPicksChanged: (picks) => settings.picksChanged(picks) });
settings.attach(store);

const openUrl = (url: string): Promise<void> => streamDeck.system.openUrl(url);

function sendStateToPropertyInspector(): void {
  if (!streamDeck.ui.action) {
    return;
  }
  const run = store.getRun();
  streamDeck.ui
    .sendToPropertyInspector({
      event: "studioState",
      phase: store.getPhase(),
      port: client.getPort(),
      studioUrl: store.studioUrl(),
      formError: store.getFormError(),
      busyReason: store.getBusyReason(),
      runError: run?.error ?? "",
      picks: store.getPicks()
    })
    .catch(() => undefined);
}

store.on("changed", sendStateToPropertyInspector);

streamDeck.ui.onDidAppear(() => sendStateToPropertyInspector());

streamDeck.ui.onSendToPlugin<{ event?: string; port?: string | number }>((ev) => {
  const payload = ev.payload ?? {};
  if (payload.event === "piReady") {
    sendStateToPropertyInspector();
  } else if (payload.event === "updatePort") {
    settings.setPort(payload.port);
    sendStateToPropertyInspector();
  } else if (payload.event === "openStudio") {
    openUrl(store.studioUrl()).catch(() => undefined);
  }
});

streamDeck.actions.registerAction(new ChoiceDialAction(store));
streamDeck.actions.registerAction(new MachineDialAction(store));
streamDeck.actions.registerAction(new MakeKeyAction(store, openUrl));

// The saved port and picks are read before the store starts, so its first
// form read and the first press already use them. (getGlobalSettings answers
// with the settings; it does not raise onDidReceiveGlobalSettings, which in
// SDK 3 fires only for changes made elsewhere.)
void streamDeck
  .connect()
  .then(async () => {
    try {
      settings.load(await streamDeck.settings.getGlobalSettings<StudioGlobalSettings>());
    } catch {
      settings.load(undefined);
    }
  })
  .finally(() => store.start());
