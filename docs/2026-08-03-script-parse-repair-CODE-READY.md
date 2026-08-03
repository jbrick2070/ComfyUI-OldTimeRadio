# Script-parse repair: code-ready plan (r2-hardened)

**2026-08-03, HEAD `ad2f8f6a`.** Supersedes `-FINAL.md` (method) and `-PLAN.md`
(premise, wrong). Panel: `kibitz-runs/2026-08-03-writer-scaffolding/r1/` (method)
and `kibitz-runs/2026-08-03-script-parse-repair/r2/` (coding plan). r2 returned
twelve must-fixes; all are folded in below and the two decisive ones were
re-verified in the code by the driver.

## The law this obeys (operator, 2026-07-22)

`_otr_ledger_cleanup.py:39-41`: **"length, language, style, visual vocabulary,
craft and quality may never fail a story. Structure may. Content is repaired;
structure is named."**

The markup ladder breaks it: markdown scaffolding is FORMATTING -- content --
and the ladder names it structural and kills the episode. `_otr_ledger_cleanup`
would have repaired it but never runs, because the death is upstream of any
ledger. This plan applies the existing law one stage earlier.

## Scope: exposure is per-TRANSPORT, not per-bank

Lanes ARE source banks.

| pipeline / bank lane | runner | script transport | exposed |
|---|---|---|---|
| `scifi_news_pro_multipass` | `_otr_scifi_fable2` | RAW TEXT markup (`:1577`) | **YES** |
| `scifi_news_circuit` | `_otr_scifi_codex` | JSON + bound schema (`:1933-1948`) | no |
| `legacy_many_pass`, `legacy_many_pass_adapt`, `original_multi_pass` | writer inline | UNKNOWN | audit A1 |

A schema-bound lane cannot express `**SCENE 5**` as a speaker. Exactly one lane
is raw-text today, which is why the durable protection is a PREFLIGHT RULE, not
a per-bank patch.

## STATUS -- increment 0 is BUILT; increments 1-5 are NOT build-ready

**Increment 0 shipped** (fourth balanced shape, transport-gated, ordered last)
and is green. It stands alone: it fixes the `**SCENE 5**` / `**TITLE: ...**`
defect present in BOTH dead legs, and needs none of the machinery below.

**Increments 1-5 are BLOCKED on a redesign.** r3 (wiring) returned seven
must-fixes; the load-bearing ones invalidate the call/trace design written
below, so treat everything after this block as a DRAFT to be corrected, not a
spec to implement:

1. **The adjudication design creates more trace rows than calls.** It appends a
   `row_adjudication` row and then a separate `accepted` row after the
   re-parse -- but the re-parse is not a `creative_fn` invocation, and
   `box["calls"] == len(p3_attempts)` is enforced (`:414-425`, `:2870-2877`).
   **Fix: exactly one trace row per invocation. When adjudication's re-parse is
   clean, the ADJUDICATION call's own row becomes the selected/final one.**
   Never synthesize a second row.
2. **The capacity fallback breaks the five-call ceiling.** Attempting the
   draft-carrying repair, catching a pre-call refusal, then re-calling
   defects-only for the SAME rung consumes two invocations. Fix: consume the
   refused rung with a `capacity_prompt_no_room` trace and schedule
   defects-only on the NEXT rung.
3. **Capacity exceptions are provider-specific, not one type.** Local raises
   `PromptContextOverflowError` (`OTR_LedgerScriptWriter.py:888-890`),
   OpenRouter wraps as `OpenRouterConfigError`
   (`_otr_openrouter_backend.py:1190-1223`), and Comfy Credits, Google API,
   GGUF and the local loader each wrap differently. "Catch
   `GenerationContextOverflowError`" is wrong. Fix: one cause-chain classifier
   over the two shared capacity phases.
4. **`ProviderCapacityMessages` with a finite `max_new_tokens` fails pre-call
   by construction** -- it sets `reserve_remaining` and `bounded_capacity`, so
   backends try to reserve the whole context beside a nonempty prompt
   (`_otr_generation_budget.py:13-27`). Adjudication needs its own bounded
   message type.
5. **The eligible-defect predicate contradicts itself** (requires all defects in
   the allowlist, then permits derivatives alongside). Needs one executable
   predicate plus eligible-only / eligible+derivative / derivative-only
   fixtures.
6. **`scene_index` has no producer.** `parse_fable2_markup` returns `None` on
   any defect and scene membership lives only in private `_Parse.scenes`
   (`:227-264`, `:417-485`). A public immutable diagnostic parse result is
   required first.
7. **`apply_verdicts(raw, verdicts)` cannot enforce its own safety property** --
   `RowVerdict` carries no line number, terminator or original text. Signature
   becomes `apply_verdicts(raw, rows, verdicts)` with an exact row-ID bijection
   and per-action compatibility checks.

Also: `_strip_conversational_wrapper` itself uses `splitlines()` + `"\n".join()`
(`:1570-1617`), so increment 1's "lossless" claim is false wherever a wrapper is
removed; and the increment 5 seam owner is
`nodes/story_packs/scifi_news_pro/scifi_news_pro.json:12`, not the reader at
`:1751`.

## THE CALL-ACCOUNTING CONTRACT (r2 MUST-FIX 1+2 -- read first)

`_counting` wraps `creative_fn` and `box["calls"] != len(p3_attempts)` raises
`"P3 attempt/call count drift"` (`_otr_scifi_fable2.py:2870-2877`). Adjudication
is a `creative_fn` call, therefore **it MUST append exactly one trace row**, and
the trace model must be able to represent it. This is not deferrable.

**State machine, exact:**

* Rungs remain FOUR (`_MARKUP_LADDER_TEMPS`, `:184-190`). Adjudication does not
  replace a rung.
* Adjudication fires **at most ONCE per ladder**, on the FIRST rung whose
  defects are all in the eligible allowlist. Never twice, whatever later rungs do.
* **Maximum `creative_fn` calls per ladder = 5** (4 rungs + 1 adjudication).
* Adjudication appends one `PassAttemptTrace` with the next `attempt` number,
  `selected=False`, and outcome `row_adjudication` (accepted verdicts applied)
  or `row_adjudication_rejected` (malformed reply / unusable verdicts).
* Slot: **`creative_fn`**, not a new "one-call slot" -- deciding whether a line
  is dialogue or scaffolding is creative authorship, and only creative and
  technical slots exist (r2 SHOULD-FIX 1). Temperature: the current rung's.
* If adjudication succeeds and the re-parse is clean, the LADDER returns; the
  accepted result is that re-parse, recorded as a normal `accepted` row -- so
  the single-selected-last invariant (`:656-657`, `:2066-2071`) still holds.

**Trace changes required IN INCREMENT 2, one change, all together:**
`outcome` Literal + `valid_outcomes` (`:617`, `:625`, `:634-637`); the
`selected == (outcome == "accepted")` invariant (`:656-657`) becomes
`selected` true iff `accepted`; `_validate_attempt_sequence` (`:2066-2071`);
`_attempt_payload` / `structural_retries`; exhaustion reporting; **final-draft
seal re-baseline** (traces are hashed into `artifact_hash`, `:2040`,
`:2104-2110`); and the trace tests (`tests/test_fable2_artifacts.py:206-236`).

## Increment 0 -- deterministic transport completion

`nodes/_otr_fable2_markup.py`, `_canonicalize_transport_line` (`:52-109`).

Fourth balanced shape (wrapper spans the whole line) as the LAST branch of the
colon path, replacing the bare `return s, ()` at `:108`:

* **Order**: shapes 1-3 first, else `**BO NI:** Hello **world**` is mangled.
* **Interior balance**: after removing outer markers the remainder must contain
  an EVEN count of that marker, else untouched and loud.
* **Transport grammar only**: unwrap only if the result matches
  `_RE_TITLE|_RE_MUSIC|_RE_SCENE|_RE_CODA|_RE_END` (`:37-41`).

**Empty `SCENE n:` -- RULED (r2 MUST-FIX 9).** `_RE_SCENE` requires a nonempty
setting (`:39`), so `SCENE 5:` is `BAD_LINE_SHAPE`. It is NOT repaired
deterministically -- the following row is ambiguous (it may be dialogue, a
setting, or absent). It becomes an eligible quarantine class handled by the
typed `scene_setting` verdict in increment 2. One owner, named.

**Test expectations, corrected (r2 MUST-FIX 10):**
`**BO NI:** Hello **world**` -> `BO NI: Hello **world**` -- the PAYLOAD's markers
survive; the line is NOT byte-identical. The docstring says exactly this
(`:63-67`); my earlier spec's "byte-identical" was wrong.

Fixtures: fourth shape unwraps; the BO NI case per above; unbalanced interior
untouched; wrapped NON-transport (`**She turns: the room is empty**`) still
reaches UNKNOWN_SPEAKER.

## Increment 1 -- retain the rejected draft, losslessly

`_run_markup_ladder`: keep the post-`_strip_conversational_wrapper` `raw`
(`:1699`) per attempt -- the STRIPPED text, because `parse_fable2_markup` numbers
lines of that string (`:1700`; `_otr_fable2_markup.py:430`).

**Line preservation (r2 MUST-FIX 11):** all row extraction and re-assembly uses
`splitlines(keepends=True)` and preserves untouched segments exactly. Never
`splitlines()` + `"\n".join()` -- that rewrites CRLF, blank rows and the
terminal newline. Tests cover CRLF/LF, blank rows, and presence/absence of a
final newline.

## Increment 2 -- bounded row adjudication (the core)

New pure module `nodes/_otr_script_row_repair.py` -- testable without the ladder,
reusable by any future raw-text lane.

**Eligible defect allowlist (r2 MUST-FIX 4), exact:**

    UNKNOWN_SPEAKER
    BAD_LINE_SHAPE          (includes the empty SCENE n: header)

`SKELETON_BREAK` and `CAST_MEMBER_SILENT` are DERIVATIVE / whole-document
(`_otr_fable2_markup.py:305-415`, `:468-485`) and are NOT eligible on their own.
A line carrying a derivative defect ALONGSIDE an eligible one is still
quarantined; a rung whose defects are wholly derivative goes straight to
increment 3. Genuine skeleton corruption never reaches a DROP classifier.

**Data model (r2 MUST-FIX 3, 5, 6):**

    @dataclass(frozen=True)
    class QuarantineRow:
        row_id: str                    # stable, e.g. "L24"
        line_no: int
        text: str                      # original, without terminator
        terminator: str                # "" | "\n" | "\r\n"  -- preserved
        defects: tuple[ParseDefect, ...]   # ONE row may carry several
        scene_no: int | None
        scene_speakers: tuple[str, ...]    # cast already speaking in that scene

    @dataclass(frozen=True)
    class RowVerdict:
        row_id: str
        action: Literal["dialogue_by", "drop", "scene_setting", "unresolved"]
        speaker: str | None            # required iff action == "dialogue_by"
        setting: str | None            # required iff action == "scene_setting"

`quarantine_rows(raw, defects, roster, scene_index)` -- roster and scene
membership MUST be passed in; `prev_line`/`next_line` cannot establish "already
speaking in this scene" and `_Parse.scenes` is private (`:227-264`, `:401-414`).
Extract a module-level speaker resolver shared with `_Parse._speaker_key`
(`:261-264`) rather than duplicating identity rules (r2 SHOULD-FIX 4).

`parse_adjudication(reply, rows)` is STRICT and pure: exactly one verdict per
row_id, no extras, no duplicates, `speaker` must be an exact roster name,
whitespace/casing normalized once, any violation raises
`AdjudicationReplyError`. **That exception is caught by the ladder and routed to
increment 3** -- one malformed micro-call must never kill the ladder
(r2 MUST-FIX 6).

`apply_verdicts(raw, verdicts)` rewrites ONLY quarantined rows; every other byte
is copied. **That is the safety property -- structural, not tuned.**

**Prompt/response shape (r2 SHOULD-FIX 2, 3):** rows are sent as JSON (a
length-prefixed structure), not raw text with delimiters -- rejected text can
contain colons and verdict-like tokens. Response bounded by a finite
`max_new_tokens` derived from row count and roster-name length, plus a
completion marker. Test the maximum-row capacity case.

**No announcer defaulting, ever**: an ANNOUNCER line mid-scene closes the scene
and pushes to `_POSTAMBLE` (`:388-400`), and `base_user` already restricts
announcer to intro/outro (`_otr_scifi_fable2.py:1557`). Orphans go to a cast
member in `scene_speakers`, else `unresolved`.

## Increment 3 -- true repair fallback, with its own message builder

**Do NOT reuse `format_example` (r2 MUST-FIX 8).** That branch frames its
assistant turn as *"show the exact output FORMAT once, as a tiny example
episode"* (`:1685-1690`) -- re-supplying a rejected draft there presents
malformed output as the exemplar. Write a dedicated builder producing:

    [system, user=base assignment, assistant=rejected draft, user=repair instruction]

and **delete the dead `format_example` parameter and branch** (`:1655`,
`:1681-1692`; no production caller) rather than repurposing it (r2 CUT 1).

**Capacity contract (r2 MUST-FIX 7).** `_run_markup_ladder` has no context cap
or tokenizer, and `ProviderCapacityMessages` exposes no "room" query
(`_otr_generation_budget.py:12-26`). Therefore: **do not predict -- catch.**
Attempt the draft-carrying repair; on `GenerationContextOverflowError` with
phase `prompt_no_room` (permanently non-rerollable, `:119-129`, `:174-180`) or
the OpenRouter pre-call wrap (`_otr_openrouter_backend.py:1190-1223`), fall back
to defects-only for that rung and record `capacity_mode` in the trace. Both
backends' error shapes are named in the test matrix.

## Increment 4 -- telemetry only

Per-character character-word ratios logged on every adjudicated acceptance.
Accepted side is computed from `parsed.scenes[].lines[]` -- NOT
`parsed.character_word_count`, which is a TOTAL (`:198-211`, `:487-504`)
(r2 SHOULD-FIX 4). Rejected-side floor via the shared module-level speaker
resolver.

**ENFORCE NOTHING.** `CAST_MEMBER_SILENT` already makes total deletion loud, and
increment 2's structural preservation bounds partial deletion to quarantined
rows. The future-enforcement mutation checklist is CUT from this build
(r2 CUT 2); revisit only after telemetry establishes a threshold.

## Increment 5 -- prevention

One line in the FORMAT REMINDER inside `_script_user_prompt` (`:1553-1558`) and
the `fable2_script_system` seam (`:1751`): plain text only, no markdown
emphasis, no headings beyond the transport lines shown. The `format_example`
route is inactive -- and by increment 3 it is deleted.

## Exit criteria -- EVERY increment (r2 MUST-FIX 12)

Per `CLAUDE.md:77-86, 121-124`: focused tests -> full regression suite -> Bug
Bible -> commit AND push -> HEAD == origin. No increment is "done" without all
five.

**PBUG + Bible**: record the two formatting failures as ONE new root-cause PBUG
carrying both live artifacts. Do **not** merge into `PBUG-20260802-02`, which
documents a different silent-second-character defect
(`docs/PROD_BUG_LOG.md:3049-3095`) (r2 SHOULD-FIX 7). Promote with a portable
`BUG_BIBLE.yaml` entry plus executable coverage, since the verify condition is
automatable.

## Proof obligations

**Unit tests are the safety proof.** Template
`tests/test_45word_failure_regressions.py:90-115` -- `creative_fn` is injected
and the suite drives it with a scripted response iterator. **Supply enough
responses for the maximum call count (now 5)**; a short iterator raises
`StopIteration` and masks the assertion.

Happy paths: adjudication repairs quarantined rows, non-quarantined lines
byte-identical, full parse clean, every cast member still speaking;
scaffolding-only removal ACCEPTED (the false-positive path -- a gate that
refuses correct repairs is a new outage).

Strict-parser and ladder-failure coverage (r2 SHOULD-FIX 5), each its own test:
missing / duplicate / unknown / extra verdict IDs; unknown roster name;
malformed action; zero quarantined rows; adjudication raises;
`prompt_no_room`; `output_limit`; maximum call count; CRLF/LF and
final-newline preservation.

**Live, after unit-green**: re-run `ltx_audio_in` and `viz_mxc_cpu` through
`workflows/otr_canonical.json` with the selective reset and watchdog
(`CLAUDE.md:129-169`); require RESULT SUCCESS + `obs_publish OK` + canonical
assets. **Liveness alone does not prove adjudication ran** (r2 SHOULD-FIX 6) --
telemetry must identify transport-only acceptance vs adjudicated acceptance vs
retained-draft fallback, with call count and capacity mode. `viz_mxc_cpu`
carried THREE defect classes (`_w45_server.log:10983-10999`) and is not a clean
isolation test.

## The durable protection: preflight rule

`docs/SOURCE_BANK_PREFLIGHT.md`, **Gate 3 (LLM slots, prompts, and
authorship)**:

> **Script transport must be schema-bound, or carry row repair.**
> A bank whose script pass returns FREE TEXT parsed by pattern is exposed to
> model formatting drift: decoration the parser does not recognise becomes a
> structural defect and kills the episode, however good the story. Prefer a
> schema-bound pass (`scifi_news_circuit` is the reference). A raw-text lane
> MUST provide (a) a documented closed grammar for what is stripped as
> transport, (b) bounded row adjudication over defect-bearing rows only,
> preserving all other bytes, and (c) fixtures proving a decorated draft is
> repaired rather than rejected. Live evidence: 2026-08-03, two sweep legs lost
> in the writer to `**SCENE 5**`-class decoration.

## Audit A1 -- separate campaign, NOT an acceptance criterion

`_otr_ledger_cleanup.py` is the general post-writer completion pass and is sound
by design. What is NOT audited: every other seam where an LLM writes a value a
deterministic consumer later reads -- the three inline pipelines' transport, and
any pass emitting a field consumed by TTS, slicing, shot direction, captions,
credits or `obs_publish`. The question at each: does malformed model output fail
LOUD, get REPAIRED, or PASS THROUGH into a consumer? Only the third is a bug.
Scoped separately; it does not block this repair (r2 CUT 3).
