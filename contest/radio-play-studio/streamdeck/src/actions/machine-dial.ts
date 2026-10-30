// SPDX-License-Identifier: MIT
import {
  action,
  DialDownEvent,
  DialRotateEvent,
  DialUpEvent,
  KeyDownEvent,
  KeyUpEvent,
  SingletonAction,
  TouchTapEvent,
  WillAppearEvent,
  WillDisappearEvent
} from "@elgato/streamdeck";
import { CellRateLimiter } from "../gfx/cell-rate-limiter.js";
import { toSvgDataUri } from "../gfx/svg-renderer.js";
import { MenuPart, StudioStore } from "../state/studio-store.js";
import { HOLD_THRESHOLD_MS } from "../types/settings.js";
import { ControlSink, dialControlSink, keyControlSink, PressTimer } from "./choice-dial.js";
import { ControlMode, renderMenu } from "./faces.js";

/** A control that walks one half of the machine menu: which card (the
 *  machine control), or which video engine that card films with (the video
 *  control). A turn (a dial) or a press (a key) only rests on a row; a dial's
 *  press or a key's hold makes it the one that runs. */
export class MachineDialController {
  private readonly store: StudioStore;
  private readonly sink: ControlSink;
  private readonly mode: ControlMode;
  private readonly part: MenuPart;
  private readonly limiter: CellRateLimiter<string>;
  private readonly press: PressTimer;
  private visible = false;
  /** The dial was turned while held down: its release is not a press, so it
   *  does not switch machines (a switch re-reads the form, up to a minute). */
  private turnedWhileDown = false;

  private readonly onChanged = (): void => {
    this.render();
  };

  public constructor(
    store: StudioStore,
    sink: ControlSink,
    rateLimitMs = 100,
    mode: ControlMode = "dial",
    holdMs = HOLD_THRESHOLD_MS,
    part: MenuPart = "machine"
  ) {
    this.store = store;
    this.sink = sink;
    this.mode = mode;
    this.part = part;
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

  public async rotate(ticks: number, pressed = false): Promise<void> {
    if (ticks === 0) {
      return;
    }
    if (pressed) {
      this.turnedWhileDown = true;
    }
    if (!this.store.rotateMenu(this.part, ticks)) {
      await this.sink.showAlert();
    }
  }

  public dialDown(): void {
    this.turnedWhileDown = false;
  }

  public async push(): Promise<void> {
    if (this.turnedWhileDown) {
      this.turnedWhileDown = false;
      return;
    }
    if (!(await this.store.applyMenu(this.part))) {
      await this.sink.showAlert();
    }
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
    return renderMenu(this.store, this.part, this.mode);
  }

  private render(): void {
    if (this.visible) {
      this.limiter.push(this.svg());
    }
  }
}

/** The Stream Deck side of a menu control, on a dial or a key. */
class MenuControlAction extends SingletonAction {
  private readonly store: StudioStore;
  private readonly part: MenuPart;
  private readonly controls = new Map<string, MachineDialController>();

  public constructor(store: StudioStore, part: MenuPart) {
    super();
    this.store = store;
    this.part = part;
  }

  public override onWillAppear(ev: WillAppearEvent): void {
    const target = ev.action;
    let ctrl: MachineDialController;
    if (target.isDial()) {
      ctrl = new MachineDialController(this.store, dialControlSink(target), 100, "dial", HOLD_THRESHOLD_MS, this.part);
    } else if (target.isKey()) {
      ctrl = new MachineDialController(this.store, keyControlSink(target), 100, "key", HOLD_THRESHOLD_MS, this.part);
    } else {
      return;
    }
    this.controls.set(target.id, ctrl);
    ctrl.appear();
  }

  public override onWillDisappear(ev: WillDisappearEvent): void {
    this.controls.get(ev.action.id)?.disappear();
    this.controls.delete(ev.action.id);
  }

  public override async onDialRotate(ev: DialRotateEvent): Promise<void> {
    await this.controls.get(ev.action.id)?.rotate(ev.payload.ticks, ev.payload.pressed);
  }

  public override onDialDown(ev: DialDownEvent): void {
    this.controls.get(ev.action.id)?.dialDown();
  }

  public override async onDialUp(ev: DialUpEvent): Promise<void> {
    await this.controls.get(ev.action.id)?.push();
  }

  public override async onTouchTap(ev: TouchTapEvent): Promise<void> {
    await this.controls.get(ev.action.id)?.push();
  }

  public override onKeyDown(ev: KeyDownEvent): void {
    this.controls.get(ev.action.id)?.keyDown();
  }

  public override async onKeyUp(ev: KeyUpEvent): Promise<void> {
    await this.controls.get(ev.action.id)?.keyUp();
  }
}

@action({ UUID: "com.jbrick2070.otr.machine" })
export class MachineDialAction extends MenuControlAction {
  public constructor(store: StudioStore) {
    super(store, "machine");
  }
}

@action({ UUID: "com.jbrick2070.otr.video" })
export class VideoDialAction extends MenuControlAction {
  public constructor(store: StudioStore) {
    super(store, "engine");
  }
}
