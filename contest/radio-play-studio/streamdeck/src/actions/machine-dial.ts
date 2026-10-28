// SPDX-License-Identifier: MIT
import {
  action,
  DialRotateEvent,
  DialUpEvent,
  SingletonAction,
  TouchTapEvent,
  WillAppearEvent,
  WillDisappearEvent
} from "@elgato/streamdeck";
import { CellRateLimiter } from "../gfx/cell-rate-limiter.js";
import { toSvgDataUri } from "../gfx/svg-renderer.js";
import { StudioStore } from "../state/studio-store.js";
import { DialSink } from "./choice-dial.js";
import { renderMachineDial } from "./faces.js";

/** The dial that walks the machine menu: which card, and how it films. The
 *  turn only rests on a row; the push makes it the one that runs. */
export class MachineDialController {
  private readonly store: StudioStore;
  private readonly sink: DialSink;
  private readonly limiter: CellRateLimiter<string>;
  private visible = false;

  private readonly onChanged = (): void => {
    this.render();
  };

  public constructor(store: StudioStore, sink: DialSink, rateLimitMs = 100) {
    this.store = store;
    this.sink = sink;
    this.limiter = new CellRateLimiter<string>(async (svg) => {
      await this.sink.setFeedback({ canvas: toSvgDataUri(svg) });
    }, rateLimitMs);
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

  public async rotate(ticks: number): Promise<void> {
    if (ticks !== 0 && !this.store.rotateMachine(ticks)) {
      await this.sink.showAlert();
    }
  }

  public async push(): Promise<void> {
    if (!(await this.store.applyMachine())) {
      await this.sink.showAlert();
    }
  }

  public svg(): string {
    return renderMachineDial(this.store);
  }

  private render(): void {
    if (this.visible) {
      this.limiter.push(this.svg());
    }
  }
}

@action({ UUID: "com.jbrick2070.otr.machine" })
export class MachineDialAction extends SingletonAction {
  private readonly store: StudioStore;
  private readonly dials = new Map<string, MachineDialController>();

  public constructor(store: StudioStore) {
    super();
    this.store = store;
  }

  public override onWillAppear(ev: WillAppearEvent): void {
    if (!ev.action.isDial()) {
      return;
    }
    const dial = ev.action;
    const ctrl = new MachineDialController(this.store, {
      setFeedback: (fb) => dial.setFeedback(fb),
      showAlert: () => dial.showAlert()
    });
    this.dials.set(dial.id, ctrl);
    ctrl.appear();
  }

  public override onWillDisappear(ev: WillDisappearEvent): void {
    this.dials.get(ev.action.id)?.disappear();
    this.dials.delete(ev.action.id);
  }

  public override async onDialRotate(ev: DialRotateEvent): Promise<void> {
    await this.dials.get(ev.action.id)?.rotate(ev.payload.ticks);
  }

  public override async onDialUp(ev: DialUpEvent): Promise<void> {
    await this.dials.get(ev.action.id)?.push();
  }

  public override async onTouchTap(ev: TouchTapEvent): Promise<void> {
    await this.dials.get(ev.action.id)?.push();
  }
}
