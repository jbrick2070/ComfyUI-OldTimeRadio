// SPDX-License-Identifier: MIT
// A text/event-stream reader for what server.py's _events writes: `id: N`,
// `data: {json}`, a blank line, and `: keep-alive` comments every HEARTBEAT_S.

export interface SseMessage {
  id: string;
  data: string;
}

export class SseParser {
  private buffer = "";
  private id = "";
  private data: string[] = [];
  private readonly onMessage: (message: SseMessage) => void;

  public constructor(onMessage: (message: SseMessage) => void) {
    this.onMessage = onMessage;
  }

  /** Feed the next piece of the stream; a message may span pieces. */
  public feed(chunk: string): void {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      this.line(line);
      newline = this.buffer.indexOf("\n");
    }
  }

  private line(line: string): void {
    if (line === "") {
      if (this.data.length > 0) {
        this.onMessage({ id: this.id, data: this.data.join("\n") });
      }
      this.data = [];
      return;
    }
    if (line.startsWith(":")) {
      return; // a comment: the keep-alive
    }
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "data") {
      this.data.push(value);
    } else if (field === "id") {
      this.id = value;
    }
  }
}
