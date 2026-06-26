<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The proposed news-bleed policy is explicitly defeated by existing roster logic, and the Layer 3 JSON design contradicts the text-only plumbing it claims to reuse.

MUST-FIX BEFORE BUILD:

1. [Layer 2 - News-bleed] `build_allowed_roster` defeats the banned-noun detector.
   *Defect:* The spec asks to verify if `_otr_line_composer.build_allowed_roster` whitelists news terms. It DOES. Lines 368-370 explicitly loop over `key_terms` (journalistic terms from the news brief) and add them to the `allowed_roster`. If the news abstraction emits a banned source noun, but it was passed as a `key_term`, it is whitelisted.
   *Fix:* Modify `build_allowed_roster` to accept a `banned_terms` set and explicitly exclude them, or stop passing raw news `key_terms` to the roster builder entirely.

2. [Layer 3 - Optional online LLM cleaner] JSON output breaks existing writer plumbing.
   *Defect:* The spec claims Layer 3 will output JSON (`{clean_text, removed_spans...}`) while reusing "the EXISTING writer LLM plumbing... NO new workflow-JSON node." The existing plumbing (`compose_line_draft` / `compose_line`) returns a raw string, aggressively strips formatting (which will mangle JSON braces/quotes), and its system prompt strictly enforces "Only the words the character speaks out loud". It does not parse JSON.
   *Fix:* Layer 3 must use `_otr_structured_call.py` (which requires a Pydantic schema and handles `json_syntax_repair`), OR it must output plain text dialogue exactly like the existing `polish_line` pass.

3. [Layer 1 - Upstream prompt] Prompt instruction contradicts news grounding.
   *Defect:* Adding "no real-world proper names" to the compose prompt directly conflicts with the existing `key_terms` injection, which passes real-world news entities (e.g., CERN, NASA) into the `NAMED ENTITIES` block. This will cause the LLM to hallucinate fake names for legitimate news targets or refuse to output.
   *Fix:* Change the instruction to: "no real-world proper names *unless listed in NAMED ENTITIES*."

SHOULD-FIX:

1. [Layer 2 - Caps-name vocative] `scrub_self_vocative` does not cover roster names.
   *Defect:* The spec asks to verify if `scrub_self_vocative` already covers the "YUKI MARTIN" vocative leak. It does NOT. `scrub_self_vocative` (line 62, `_otr_line_hygiene.py`) only checks `speaker_name` (the character's *own* name). It will ignore a character yelling another character's name in all-caps.
   *Fix:* Implement a new `scrub_roster_vocative` function that iterates over the full `allowed_roster` to strip/title-case all-caps vocative matches.

2. [Layer 2 - Malformed quotes] Ambiguity in "internal odd quote" fail-closed logic.
   *Defect:* "fail-CLOSED to recompose on any INTERNAL odd quote" is dangerous if applied to all quote characters, as it will trigger false positives on legitimate apostrophes and single quotes (e.g., "don't").
   *Fix:* Explicitly restrict the fail-closed recompose trigger to unbalanced *double quotes* (`"`), reusing the existing `segment_double_quotes` helper which already normalizes and counts them safely.

CUT THESE (over-engineering):

1. [Layer 3 - Optional online LLM cleaner]
   *Why to cut:* Layer 2 already introduces deterministic, regex-backed extract-or-fail rules for the exact four leak classes identified. Adding a synchronous, JSON-structured LLM repair pass introduces severe latency, cost, and a massive failure surface (JSON decode errors, schema validation errors requiring `_otr_repair_prompts.py` loops) for an edge case Layer 2 is already designed to catch. Rely on Layer 2's deterministic rules + the existing recompose budget.