// SPDX-License-Identifier: MIT
import { describe, expect, it } from "vitest";
import { cycleValue, defaultValue, displayValue, isOffered, offeredChoices } from "../src/actions/choice-math.js";
import { StudioField } from "../src/types/contract.js";
import { SAMPLE_FIELDS } from "./helpers/fake-studio-server.js";

const field = (key: string): StudioField => SAMPLE_FIELDS.find((f) => f.key === key) as StudioField;

describe("turning a dial through a field", () => {
  it("walks a dropdown forward and back, wrapping at the ends", () => {
    const style = field("style");
    expect(cycleValue(style, "noir", 1)).toBe("technicolor");
    expect(cycleValue(style, "woodcut", 1)).toBe("noir");
    expect(cycleValue(style, "noir", -1)).toBe("woodcut");
    expect(cycleValue(style, "noir", 5)).toBe("woodcut");
  });

  it("never offers my_story on the bank dial, and skips it when turning", () => {
    const bank = field("bank");
    expect(offeredChoices(bank)).toEqual(["roll", "shakespeare", "public_domain"]);
    expect(cycleValue(bank, "public_domain", 1)).toBe("roll");
    expect(isOffered(bank, "my_story")).toBe(false);
  });

  it("treats acts as the dropdown the server describes, not as a number", () => {
    const acts = field("acts");
    expect(acts.kind).toBe("choice");
    expect(cycleValue(acts, "3", 1)).toBe("4");
    expect(cycleValue(acts, "5", 1)).toBe("1");
  });

  it("steps an int field by one and holds at min and max", () => {
    const cast = field("cast_size");
    expect(cycleValue(cast, 4, 1)).toBe(5);
    expect(cycleValue(cast, 8, 1)).toBe(8);
    expect(cycleValue(cast, 2, -1)).toBe(2);
    expect(cycleValue(cast, "nonsense", 1)).toBe(5);
  });

  it("starts from the first or last choice when the current value is not offered", () => {
    const style = field("style");
    expect(cycleValue(style, "sepia", 1)).toBe("noir");
    expect(cycleValue(style, "sepia", -1)).toBe("woodcut");
  });

  it("falls back to a choice the dial can show when the default is off the dial", () => {
    const bank: StudioField = { ...field("bank"), default: "my_story" };
    expect(defaultValue(bank)).toBe("roll");
  });

  it("reads underscores as spaces and a model id by its last piece", () => {
    expect(displayValue("public_domain")).toBe("public domain");
    expect(displayValue("google/gemma-4-12b-it")).toBe("gemma-4-12b-it");
    expect(displayValue(3)).toBe("3");
  });
});
