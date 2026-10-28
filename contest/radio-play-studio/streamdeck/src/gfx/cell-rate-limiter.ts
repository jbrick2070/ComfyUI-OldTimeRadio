// SPDX-License-Identifier: MIT
// A face is redrawn at most every `minIntervalMs`; the newest payload wins.
// (The same limiter as streamdeck-fadester's, where it keeps the Stream Deck's
// own frame budget from being flooded by a dial turned fast.)

export class CellRateLimiter<TPayload> {
  private readonly minIntervalMs: number;
  private readonly onFlush: (payload: TPayload) => void | Promise<void>;
  private lastFlushTime = -Infinity;
  private pendingPayload: TPayload | undefined;
  private hasPending = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  public constructor(onFlush: (payload: TPayload) => void | Promise<void>, minIntervalMs = 100) {
    // A face that fails to draw (the Stream Deck gone mid-send) is skipped,
    // never left as a rejection nobody is waiting on: the next change redraws.
    this.onFlush = async (payload) => {
      try {
        await onFlush(payload);
      } catch {
        // dropped on purpose; see above
      }
    };
    this.minIntervalMs = Math.max(1, minIntervalMs);
  }

  public push(payload: TPayload, now = Date.now()): void {
    const elapsed = now - this.lastFlushTime;
    if (elapsed >= this.minIntervalMs && this.timer === null) {
      this.lastFlushTime = now;
      this.hasPending = false;
      this.pendingPayload = undefined;
      void this.onFlush(payload);
      return;
    }

    this.pendingPayload = payload;
    this.hasPending = true;
    if (this.timer === null) {
      const waitMs = Math.max(1, this.minIntervalMs - elapsed);
      this.timer = setTimeout(() => {
        this.timer = null;
        if (this.hasPending && this.pendingPayload !== undefined) {
          const toSend = this.pendingPayload;
          this.pendingPayload = undefined;
          this.hasPending = false;
          this.lastFlushTime = Date.now();
          void this.onFlush(toSend);
        }
      }, waitMs);
    }
  }

  public dispose(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.hasPending = false;
    this.pendingPayload = undefined;
  }
}
