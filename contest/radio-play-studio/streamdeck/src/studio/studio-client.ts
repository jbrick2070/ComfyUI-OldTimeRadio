// SPDX-License-Identifier: MIT
// The plain HTTP side of talking to Radio Play Studio: JSON in, JSON out, and
// the event stream. The server (server.py) answers only a loopback Host and
// only JSON posts; node:http sends `Host: 127.0.0.1:<port>` for a request made
// to that address, so both rules are met without the server changing.
import { request as httpRequest, IncomingMessage } from "node:http";
import { SseParser } from "./sse.js";
import {
  Choices,
  OptionsResponse,
  RunListing,
  StudioEvent,
  StudioHttpError,
  StudioUnreachable
} from "../types/contract.js";

export const STUDIO_HOST = "127.0.0.1";

/** Seconds a request may take. /api/options reads ComfyUI's node definitions,
 *  which takes about 40 s while ComfyUI is rendering (index.html says so). */
const REQUEST_TIMEOUT_MS = 90_000;

export interface EventStreamHandle {
  close(): void;
}

export class StudioClient {
  private port: number;

  public constructor(port: number) {
    this.port = port;
  }

  public setPort(port: number): void {
    this.port = port;
  }

  public getPort(): number {
    return this.port;
  }

  public baseUrl(): string {
    return `http://${STUDIO_HOST}:${this.port}`;
  }

  public options(workflow?: string): Promise<OptionsResponse> {
    const query = workflow ? `?workflow=${encodeURIComponent(workflow)}` : "";
    return this.json<OptionsResponse>("GET", `/api/options${query}`);
  }

  public async episodes(): Promise<RunListing[]> {
    const body = await this.json<{ episodes?: RunListing[] }>("GET", "/api/episodes");
    return Array.isArray(body.episodes) ? body.episodes : [];
  }

  /** Start an episode: the run id, or a StudioHttpError with the server's
   *  reason (409 busy, 503 ComfyUI unreachable, 400 a choice it refused). */
  public async startEpisode(choices: Choices): Promise<string> {
    const body = await this.json<{ id: string }>("POST", "/api/episodes", { choices });
    return String(body.id);
  }

  /** Follow a run's events until it ends. Each event comes with the stream's
   *  `id` for it (0, 1, 2 ...), the Last-Event-ID a reconnect resumes after.
   *  `onEnd` is called once, with the reason the stream closed when it was
   *  not the run finishing. */
  public events(
    runId: string,
    onEvent: (event: StudioEvent, eventId: number) => void,
    onEnd: (error?: Error) => void,
    lastEventId?: number
  ): EventStreamHandle {
    let ended = false;
    const end = (error?: Error): void => {
      if (!ended) {
        ended = true;
        onEnd(error);
      }
    };
    const parser = new SseParser((message) => {
      let event: StudioEvent;
      try {
        event = JSON.parse(message.data) as StudioEvent;
      } catch {
        return; // a line that is not JSON is not one of the server's events
      }
      const id = Number(message.id);
      onEvent(event, Number.isInteger(id) ? id : -1);
    });
    const headers: Record<string, string> = { Accept: "text/event-stream" };
    if (lastEventId !== undefined) {
      headers["Last-Event-ID"] = String(lastEventId);
    }
    const req = httpRequest(
      {
        host: STUDIO_HOST,
        port: this.port,
        method: "GET",
        path: `/api/episodes/${encodeURIComponent(runId)}/events`,
        headers
      },
      (res) => {
        if (res.statusCode !== 200) {
          void readBody(res).then((text) => end(this.refusal(res.statusCode ?? 0, text)));
          return;
        }
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => parser.feed(chunk));
        res.on("end", () => end());
        res.on("error", (err) => end(err));
      }
    );
    req.on("error", (err) => end(new StudioUnreachable(err.message)));
    req.end();
    return {
      close: () => {
        ended = true;
        req.destroy();
      }
    };
  }

  private json<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const payload = body === undefined ? undefined : Buffer.from(JSON.stringify(body), "utf8");
      const headers: Record<string, string> = { Accept: "application/json" };
      if (payload) {
        headers["Content-Type"] = "application/json";
        headers["Content-Length"] = String(payload.length);
      }
      const req = httpRequest(
        { host: STUDIO_HOST, port: this.port, method, path, headers, timeout: REQUEST_TIMEOUT_MS },
        (res) => {
          void readBody(res).then((text) => {
            const status = res.statusCode ?? 0;
            if (status < 200 || status >= 300) {
              reject(this.refusal(status, text));
              return;
            }
            try {
              resolve((text ? JSON.parse(text) : {}) as T);
            } catch {
              reject(new StudioHttpError(status, "the Studio answered with something that is not JSON"));
            }
          }, reject);
        }
      );
      req.on("timeout", () => req.destroy(new Error("the Studio took too long to answer")));
      req.on("error", (err) => reject(new StudioUnreachable(err.message)));
      if (payload) {
        req.write(payload);
      }
      req.end();
    });
  }

  private refusal(status: number, text: string): StudioHttpError {
    let message = `HTTP ${status}`;
    try {
      const parsed = JSON.parse(text) as { error?: unknown };
      if (typeof parsed.error === "string" && parsed.error) {
        message = parsed.error;
      }
    } catch {
      // not JSON: the status is all we know
    }
    return new StudioHttpError(status, message);
  }
}

function readBody(res: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    res.on("data", (chunk: Buffer) => parts.push(chunk));
    res.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
    res.on("error", reject);
  });
}
