// SPDX-License-Identifier: MIT
// The plugin's global settings (the Studio's port, the dials' picks) and the
// store, kept in step. Loaded once before the store starts, so the first
// form read and the first press already use what was saved; written back
// whole every time either changes, so one never overwrites the other with a
// default.
import { StudioStore } from "./studio-store.js";
import { StudioClient } from "../studio/studio-client.js";
import { Choices } from "../types/contract.js";
import { DEFAULT_GLOBAL_SETTINGS, normalizePort, StudioGlobalSettings } from "../types/settings.js";

export function readPicks(json: unknown): Choices {
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

export class SettingsSync {
  private readonly client: StudioClient;
  private readonly persist: (settings: Required<StudioGlobalSettings>) => Promise<void>;
  private store: StudioStore | null = null;
  private current: Required<StudioGlobalSettings> = { ...DEFAULT_GLOBAL_SETTINGS };

  public constructor(client: StudioClient, persist: (settings: Required<StudioGlobalSettings>) => Promise<void>) {
    this.client = client;
    this.persist = persist;
  }

  public attach(store: StudioStore): void {
    this.store = store;
  }

  public get(): Required<StudioGlobalSettings> {
    return { ...this.current };
  }

  /** What getGlobalSettings returned, before the store starts. */
  public load(saved: StudioGlobalSettings | undefined): void {
    this.current = {
      port: normalizePort(saved?.port),
      picksJson: typeof saved?.picksJson === "string" ? saved.picksJson : DEFAULT_GLOBAL_SETTINGS.picksJson
    };
    this.client.setPort(this.current.port);
    this.store?.setPicks(readPicks(this.current.picksJson));
  }

  /** The store's picks changed: save them beside the port. */
  public picksChanged(picks: Choices): void {
    this.current = { ...this.current, picksJson: JSON.stringify(picks) };
    this.save();
  }

  /** The port was changed in an action's settings. */
  public setPort(raw: unknown): void {
    const port = normalizePort(raw);
    if (port === this.current.port) {
      return;
    }
    this.current = { ...this.current, port };
    this.client.setPort(port);
    this.save();
    this.store?.endpointChanged();
  }

  private save(): void {
    this.persist(this.get()).catch(() => undefined);
  }
}
