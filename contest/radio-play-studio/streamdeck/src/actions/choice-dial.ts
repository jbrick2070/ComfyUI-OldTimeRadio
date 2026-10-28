// SPDX-License-Identifier: MIT
import {
  action,
  DialDownEvent,
  DialRotateEvent,
  DialUpEvent,
  DidReceiveSettingsEvent,
  KeyDownEvent,
  KeyUpEvent,
  SingletonAction,
  TouchTapEvent,
  WillAppearEvent,
  WillDisappearEvent
} from "@elgato/streamdeck";
import { CellRateLimiter } from "../gfx/cell-rate-limiter.js";
import { toSvgDataUri } from "../gfx/svg-renderer.js";
import { StudioStore } from "../state/studio-store.js";
import { ChoiceDialSettings, DialField, HOLD_THRESHOLD_MS, normalizeField } from "../types/settings.js";
import { ControlMode, renderChoice } from "./faces.js";

export interface DialSink {
  setFeedback(feedback: { canvas: string }): Promise<void>;
  showAlert(): Promise<void>;
}

export interface KeyFaceSink {
  setImage(image: string): Promise<void>;
  showAlert(): Promise<void>;
}

/** What a control draws with and how it complains: a dial's strip cell or a
 *  key's image. */
export interface ControlSink {
  draw(svgUri: string): Promise<void>;
  showAlert(): Promise<void>;
}

export function dialControlSink(sink: DialSink): ControlSink {
  return { draw: (canvas) => sink.setFeedback({ canvas }), showAlert: () => sink.showAlert() };
}

export function keyControlSink(sink: KeyFaceSink): ControlSink {
  return { draw: (image) => sink.setImage(image), showAlert: () => sink.showAlert() };
}

/** Telling a short press from a hold, for a key or a dial's push. */
export class PressTimer {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private held = false;
  private readonly holdMs: number;

  public constructor(holdMs = HOLD_THRESHOLD_MS) {
    this.holdMs = holdMs;
  }

  /** Down: `onHold` runs if it is still down after the hold time. */
  public down(onHold: () => void): void {
    this.cancel();
    this.held = false;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.held = true;
      onHold();
    }, this.holdMs);
  }

  /** Up: true when this was a short press (the hold has not fired). */
  public up(): boolean {
    const short = this.timer !== null && !this.held;
    this.cancel();
    this.held = false;
    return short;
  }

  public cancel(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}

/** One control turning one field of the form: a dial on a Stream Deck +, or
 *  a key on a deck without dials. A dial turns through the choices and its
 *  press goes back to the default; a key steps to the next choice on a press
 *  and goes back to the default on a hold. */
export class ChoiceDialController {
  private readonly store: StudioStore;
  private readonly sink: ControlSink;
  private readonly mode: ControlMode;
  private field: DialField;
  private readonly limiter: CellRateLimiter<string>;
  private readonly press: PressTimer;
  private visible = false;
  /** The dial was turned while held down: its release is not a press. */
  private turnedWhileDown = false;

  private readonly onChanged = (): void => {
    this.render();
  };

  public constructor(
    store: StudioStore,
    sink: ControlSink,
    settings?: ChoiceDialSettings,
    rateLimitMs = 100,
    mode: ControlMode = "dial",
    holdMs = HOLD_THRESHOLD_MS
  ) {
    this.store = store;
    this.sink = sink;
    this.mode = mode;
    this.field = normalizeField(settings?.field);
    this.press = new PressTimer(holdMs);
    this.limiter = new CellRateLimiter<string>(async (svg) => {
      await this.sink.draw(toSvgDataUri(svg));
    }, rateLimitMs);
  }

  public appear(): void {
    this.visible = true;
    this.store.on("changed", this.onChanged);
    this.render();
  }

  public disappear(): void {
    this.visible = false;
    this.press.cancel();
    this.limiter.dispose();
    this.store.off("changed", this.onChanged);
  }

  public updateSettings(settings?: ChoiceDialSettings): void {
    this.field = normalizeField(settings?.field);
    this.render();
  }

  public async rotate(ticks: number, pressed = false): Promise<void> {
    if (ticks === 0) {
      return;
    }
    if (pressed) {
      this.turnedWhileDown = true;
    }
    if (!this.store.cycle(this.field, ticks)) {
      await this.sink.showAlert();
    }
  }

  public dialDown(): void {
    this.turnedWhileDown = false;
  }

  /** A dial's push: back to the default, unless the dial was turned while
   *  it was held. */
  public async push(): Promise<void> {
    if (this.turnedWhileDown) {
      this.turnedWhileDown = false;
      return;
    }
    if (!this.store.resetPick(this.field)) {
      await this.sink.showAlert();
    }
  }

  /** The left half of the strip goes back one, the right half forward one. */
  public rotateByTap(tapX: number): Promise<void> {
    return this.rotate(tapX < 100 ? -1 : 1);
  }

  public keyDown(): void {
    this.press.down(() => {
      void this.push();
    });
  }

  public async keyUp(): Promise<void> {
    if (this.press.up()) {
      await this.rotate(1);
    }
  }

  public svg(): string {
    return renderChoice(this.store, this.field, this.mode);
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
  private readonly controls = new Map<string, ChoiceDialController>();

  public constructor(store: StudioStore) {
    super();
    this.store = store;
  }

  public override onWillAppear(ev: WillAppearEvent<ChoiceDialSettings>): void {
    const target = ev.action;
    let ctrl: ChoiceDialController;
    if (target.isDial()) {
      ctrl = new ChoiceDialController(this.store, dialControlSink(target), ev.payload.settings, 100, "dial");
    } else if (target.isKey()) {
      ctrl = new ChoiceDialController(this.store, keyControlSink(target), ev.payload.settings, 100, "key");
    } else {
      return;
    }
    this.controls.set(target.id, ctrl);
    ctrl.appear();
  }

  public override onWillDisappear(ev: WillDisappearEvent<ChoiceDialSettings>): void {
    this.controls.get(ev.action.id)?.disappear();
    this.controls.delete(ev.action.id);
  }

  public override onDidReceiveSettings(ev: DidReceiveSettingsEvent<ChoiceDialSettings>): void {
    this.controls.get(ev.action.id)?.updateSettings(ev.payload.settings);
  }

  public override async onDialRotate(ev: DialRotateEvent<ChoiceDialSettings>): Promise<void> {
    await this.controls.get(ev.action.id)?.rotate(ev.payload.ticks, ev.payload.pressed);
  }

  public override onDialDown(ev: DialDownEvent<ChoiceDialSettings>): void {
    this.controls.get(ev.action.id)?.dialDown();
  }

  public override async onDialUp(ev: DialUpEvent<ChoiceDialSettings>): Promise<void> {
    await this.controls.get(ev.action.id)?.push();
  }

  public override async onTouchTap(ev: TouchTapEvent<ChoiceDialSettings>): Promise<void> {
    await this.controls.get(ev.action.id)?.rotateByTap(ev.payload.tapPos[0]);
  }

  public override onKeyDown(ev: KeyDownEvent<ChoiceDialSettings>): void {
    this.controls.get(ev.action.id)?.keyDown();
  }

  public override async onKeyUp(ev: KeyUpEvent<ChoiceDialSettings>): Promise<void> {
    await this.controls.get(ev.action.id)?.keyUp();
  }
}
