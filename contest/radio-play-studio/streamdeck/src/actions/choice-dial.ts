// SPDX-License-Identifier: MIT
import {
  action,
  DialRotateEvent,
  DialUpEvent,
  DidReceiveSettingsEvent,
  SingletonAction,
  TouchTapEvent,
  WillAppearEvent,
  WillDisappearEvent
} from "@elgato/streamdeck";
import { CellRateLimiter } from "../gfx/cell-rate-limiter.js";
import { toSvgDataUri } from "../gfx/svg-renderer.js";
import { StudioStore } from "../state/studio-store.js";
import { ChoiceDialSettings, DialField, normalizeField } from "../types/settings.js";
import { renderChoiceDial } from "./faces.js";

export interface DialSink {
  setFeedback(feedback: { canvas: string }): Promise<void>;
  showAlert(): Promise<void>;
}

/** One dial on the deck, turning one field of the form. */
export class ChoiceDialController {
  private readonly store: StudioStore;
  private readonly sink: DialSink;
  private field: DialField;
  private readonly limiter: CellRateLimiter<string>;
  private visible = false;

  private readonly onChanged = (): void => {
    this.render();
  };

  public constructor(store: StudioStore, sink: DialSink, settings?: ChoiceDialSettings, rateLimitMs = 100) {
    this.store = store;
    this.sink = sink;
    this.field = normalizeField(settings?.field);
    this.limiter = new CellRateLimiter<string>(async (svg) => {
      await this.sink.setFeedback({ canvas: toSvgDataUri(svg) });
    }, rateLimitMs);
  }

  public getField(): DialField {
    return this.field;
  }

  public appear(): void {
    this.visible = true;
    this.store.on("changed", this.onChanged);
    this.render();
  }

  public disappear(): void {
    this.visible = false;
    this.limiter.dispose();
    this.store.off("changed", this.onChanged);
  }

  public updateSettings(settings?: ChoiceDialSettings): void {
    this.field = normalizeField(settings?.field);
    this.render();
  }

  public async rotate(ticks: number): Promise<void> {
    if (ticks === 0) {
      return;
    }
    if (!this.store.cycle(this.field, ticks)) {
      await this.sink.showAlert();
    }
  }

  public async push(): Promise<void> {
    if (!this.store.resetPick(this.field)) {
      await this.sink.showAlert();
    }
  }

  /** The left half of the strip goes back one, the right half forward one. */
  public rotateByTap(tapX: number): Promise<void> {
    return this.rotate(tapX < 100 ? -1 : 1);
  }

  public svg(): string {
    return renderChoiceDial(this.store, this.field);
  }

  private render(): void {
    if (this.visible) {
      this.limiter.push(this.svg());
    }
  }
}

@action({ UUID: "com.jbrick2070.otr.choice" })
export class ChoiceDialAction extends SingletonAction<ChoiceDialSettings> {
  private readonly store: StudioStore;
  private readonly dials = new Map<string, ChoiceDialController>();

  public constructor(store: StudioStore) {
    super();
    this.store = store;
  }

  public override onWillAppear(ev: WillAppearEvent<ChoiceDialSettings>): void {
    if (!ev.action.isDial()) {
      return;
    }
    const dial = ev.action;
    const ctrl = new ChoiceDialController(
      this.store,
      { setFeedback: (fb) => dial.setFeedback(fb), showAlert: () => dial.showAlert() },
      ev.payload.settings
    );
    this.dials.set(dial.id, ctrl);
    ctrl.appear();
  }

  public override onWillDisappear(ev: WillDisappearEvent<ChoiceDialSettings>): void {
    this.dials.get(ev.action.id)?.disappear();
    this.dials.delete(ev.action.id);
  }

  public override onDidReceiveSettings(ev: DidReceiveSettingsEvent<ChoiceDialSettings>): void {
    this.dials.get(ev.action.id)?.updateSettings(ev.payload.settings);
  }

  public override async onDialRotate(ev: DialRotateEvent<ChoiceDialSettings>): Promise<void> {
    await this.dials.get(ev.action.id)?.rotate(ev.payload.ticks);
  }

  public override async onDialUp(ev: DialUpEvent<ChoiceDialSettings>): Promise<void> {
    await this.dials.get(ev.action.id)?.push();
  }

  public override async onTouchTap(ev: TouchTapEvent<ChoiceDialSettings>): Promise<void> {
    await this.dials.get(ev.action.id)?.rotateByTap(ev.payload.tapPos[0]);
  }
}
