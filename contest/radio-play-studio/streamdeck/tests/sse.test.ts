// SPDX-License-Identifier: MIT
import { describe, expect, it } from "vitest";
import { SseMessage, SseParser } from "../src/studio/sse.js";

describe("reading the Studio's event stream", () => {
  it("yields each id/data pair, ignores keep-alive comments, and survives split chunks", () => {
    const got: SseMessage[] = [];
    const parser = new SseParser((m) => got.push(m));
    parser.feed("id: 0\ndata: {\"type\": \"submitted\"}\n\n: keep-alive\n\nid: 1\nda");
    parser.feed("ta: {\"type\": \"progress\", \"value\": 0.5}\n\n");
    expect(got).toEqual([
      { id: "0", data: "{\"type\": \"submitted\"}" },
      { id: "1", data: "{\"type\": \"progress\", \"value\": 0.5}" }
    ]);
  });

  it("accepts CRLF line ends and a data line with no space after the colon", () => {
    const got: SseMessage[] = [];
    const parser = new SseParser((m) => got.push(m));
    parser.feed("id:7\r\ndata:{}\r\n\r\n");
    expect(got).toEqual([{ id: "7", data: "{}" }]);
  });
});
