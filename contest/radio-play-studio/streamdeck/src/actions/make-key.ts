// SPDX-License-Identifier: MIT
import {
  action,
  KeyDownEvent,
  KeyUpEvent,
  SingletonAction,
  WillAppearEvent,
  WillDisappearEvent
} from "@elgato/streamdeck";
import { CellRateLimiter } from "../gfx/cell-rate-limiter.js";
import { toSvgDataUri } from "../gfx/svg-renderer.js";
import { StudioStore } from "../state/studio-store.js";
import { HOLD_THRESHOLD_MS } from "../types/settings.js";
import { renderMakeKey } from "./faces.js";

export interface KeySink {
  setImage(image: string): Promise<void>;
  showAlert(): Promise<void>;
  openUrl(url: string): Promise<void>;
}

/** The key that makes the episode and shows how it is going. A press makes
 *  one, or plays the one just made, or clears a failure; a hold opens the
 *  Studio page on whatever the deck is following. */
export class MakeKeyController {
  private readonly store: StudioStore;
  private readonly sink: KeySink;
  private readonly limiter: CellRateLimiter<string>;
  private readonly holdMs: number;
  private holdTimer: ReturnType<typeof setTimeout> | null = null;
  private longPressFired = false;
  private visible = false;

  private readonly onChanged = (): void => {
    this.render();
  };

  public constructor(store: StudioStore, sink: KeySink, rateLimitMs = 100, holdMs = HOLD_THRESHOLD_MS) {
    this.store = store;
    this.sink = sink;
    this.holdMs = holdMs;
    this.limiter = new CellRateLimiter<string>(async (svg) => {
      await this.sink.setImage(toSvgDataUri(svg));
    }, rateLimitMs);
  }

  public appear(): void {
    this.visible = true;
    this.store.on("changed", this.onChanged);
    this.render();
  }

  public disappear(): void {
    this.visible = false;
    this.clearHold();
    this.limiter.dispose();
    this.store.off("changed", this.onChanged);
  }

  public keyDown(): void {
    this.clearHold();
    this.longPressFired = false;
    this.holdTimer = setTimeout(() => {
      this.holdTimer = null;
      this.longPressFired = true;
      void this.sink.openUrl(this.store.studioUrl());
    }, this.holdMs);
  }

  public async keyUp(): Promise<void> {
    this.clearHold();
    if (this.longPressFired) {
      this.longPressFired = false;
      return;
    }
    const did = await this.store.press();
    if (did === "open") {
      const url = this.store.episodeUrl();
      // Played once; the next press makes another.
      this.store.clearRun();
      await this.sink.openUrl(url);
    } else if (did === "refused") {
      await this.sink.showAlert();
    }
  }

  public svg(): string {
    return renderMakeKey(this.store);
  }

  private render(): void {
    if (this.visible) {
      this.limiter.push(this.svg());
    }
  }

  private clearHold(): void {
    if (this.holdTimer !== null) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
  }
}

@action({ UUID: "com.jbrick2070.otr.make" })
export class MakeKeyAction extends SingletonAction {
  private readonly store: StudioStore;
  private readonly keys = new Map<string, MakeKeyController>();
  private readonly openUrl: (url: string) => Promise<void>;

  public constructor(store: StudioStore, openUrl: (url: string) => Promise<void>) {
    super();
    this.store = store;
    this.openUrl = openUrl;
  }

  public override onWillAppear(ev: WillAppearEvent): void {
    if (!ev.action.isKey()) {
      return;
    }
    const key = ev.action;
    const ctrl = new MakeKeyController(this.store, {
      setImage: (img) => key.setImage(img),
      showAlert: () => key.showAlert(),
      openUrl: this.openUrl
    });
    this.keys.set(key.id, ctrl);
    ctrl.appear();
  }

  public override onWillDisappear(ev: WillDisappearEvent): void {
    this.keys.get(ev.action.id)?.disappear();
    this.keys.delete(ev.action.id);
  }

  public override onKeyDown(ev: KeyDownEvent): void {
    this.keys.get(ev.action.id)?.keyDown();
  }

  public override async onKeyUp(ev: KeyUpEvent): Promise<void> {
    await this.keys.get(ev.action.id)?.keyUp();
  }
}
