# QA brief -- `scripts/otr_api.py`, the canonical workflow, and two OPEN live defects

Written for a fan-out QA pass (Antigravity + Sonnet 5), 2026-08-12. Everything
below is grounded against the real Windows files at
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
(branch `v2.0-alpha`, HEAD `bf1d02a1` plus uncommitted work described here).

**Use the REAL Windows files.** Do not read a Linux-mount copy; it lags and
shows stale/truncated content. Python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.

**A campaign is RUNNING while you read this.** `scripts/otr_writer_bank_gate.py`
is spawning one subprocess per bank, and each subprocess imports
`scripts/otr_api.py`. **Do not edit any file. Review only.** Editing
`otr_api.py` mid-run is exactly how the `public_domain` leg died today (see
defect C).

---

## 1. What changed in `scripts/otr_api.py`, and why

`poll_history` used to return `str(status.get("messages"))[:500]` on failure.
`messages` is a list of `[event_name, payload]` pairs -- mostly timestamps and
cache lists -- so the 500-character cut landed INSIDE the traceback:

    'traceback': ['  File "C

The failing node, the exception type, and every frame naming our code were past
the cut. That cost the diagnosis of BOTH live writer failures today
(PBUG-20260812-02 and -03); each had to be re-derived from an unrelated server
log. **A campaign leg costs 2-20 minutes, so a lost traceback is a lost leg.**

Replaced with `describe_execution_error(messages)` (`otr_api.py:716`), which
pulls the named fields instead of truncating a repr, keeps the TAIL of the
traceback (our frames are at the end, bounded by `_ERROR_TRACEBACK_FRAMES = 12`),
handles `execution_interrupted` as a distinct event, and falls back to the old
`str(messages)[:500]` for any shape it does not recognise.

Field names were checked against the authority --
`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py`
`handle_execution_error` (line 686), which builds exactly:
`prompt_id, node_id, node_type, executed, exception_message, exception_type,
traceback, current_inputs, current_outputs`.

Tests: `tests/test_api_execution_error_diagnosis.py` (15, green).

### What I want attacked here

1. **Is the fallback actually unreachable-safe?** `describe_execution_error`
   must NEVER raise and never return empty -- a diagnosis helper that throws
   turns a reported failure into a silent one, strictly worse than what it
   replaced. Find an input shape that gets past the `try`/`except` or returns
   `""`.
2. **Is returning a MULTI-LINE string safe for every caller?** The old return
   was a single truncated line. `scripts/otr_canonical_api_run.py:249` prints it
   with an f-string; `scripts/otr_writer_bank_gate.py` and
   `scripts/otr_w45_campaign.py` parse leg logs. Does any consumer assume the
   ERROR line is exactly one line? Check log parsers, receipt writers, and
   anything doing `.splitlines()` or regex-per-line over a leg log.
3. **Is `_ERROR_TRACEBACK_FRAMES = 12` enough** to always reach an OTR frame,
   given ComfyUI's executor depth plus the `comfyui-lora-manager` metadata hook
   that wraps `_async_map_node_over_list` (seen in today's server log at
   `metadata_hook.py:177`)? Count the real frame depth in
   `tmp/_bankgate_server.log` around line 1847.
4. Any other caller of `poll_history` whose contract I changed.

---

## 2. The canonical workflow

`workflows/otr_canonical.json` is the SOURCE OF TRUTH (CLAUDE.md section 0).
litegraph schema: `nodes[]` + `links[]` + `last_node_id` + `last_link_id`; a
link is `[link_id, src_node, src_slot, dst_node, dst_slot, type]`;
`widgets_values` is POSITIONAL -- a new optional widget may only be APPENDED at
the END, because inserting mid-list shifts every saved value (BUG-LOCAL-097).

### What I want attacked here

5. **Widget-count vs live `INPUT_TYPES` audit** for every OTR node in the graph.
   A widget count that disagrees with the node's live `INPUT_TYPES` is silent
   value drift. Run it; report any node where they disagree.
6. **Link referential integrity** -- every `link_id` referenced by a node's
   `inputs`/`outputs` exists in `links[]` and vice versa; no dangling ids; every
   wired input NAME exists in that node's `INPUT_TYPES`.
7. **Is anything wired that no longer exists**, or registered-but-UNWIRED? Code
   that is not wired into this JSON is DEAD (the 2026-06-13 miss: a node plus a
   new blend input shipped and tested but unwired, and ran dormant in
   production). Cross-check `NODE_CLASS_MAPPINGS` against the graph.

---

## 3. TWO OPEN LIVE DEFECTS -- the fan-out I actually want fixes for

Both surfaced today in `scripts/otr_writer_bank_gate.py`, which pins the source
bank instead of rolling it, on the cheapest visual profile
(`otr_w45_still_flat`, 45 words). Results so far: `media_archive` PASS 8.5 min,
`original` PASS 8.7 min, and:

### DEFECT A -- `scifi_news_pro`: the writer invents off-roster characters

Died in `OTR_LedgerScriptWriter` after 4 attempts, 4.0 min. From
`tmp/_bankgate_server.log:1718`:

    [scifi_fable2] pass 'script' failed after 4 attempt(s):
    markup ladder exhausted; last defects:
    - UNKNOWN_SPEAKER: INMATE #347 (line 23)
    - SKELETON_BREAK: announcer outro missing before the CODA (line 43)
    - SKELETON_BREAK: character line (Officer Martinez) after the last scene (line 45)
    - SKELETON_BREAK: ANNOUNCER line after the CODA (line 47)

`INMATE #347` and `Officer Martinez` are NOT on the treatment's cast roster --
the model invented speakers. This is NOT the stage-direction defect fixed today
(PBUG-20260812-03); `_standalone_stage_direction_repair_note`
(`nodes/_otr_scifi_fable2.py:1700`) DELIBERATELY stays silent for an undecorated
unknown name, because telling the model to fold or drop a real character's line
is worse advice than the failure it replaces.

So the repair rung gets only the generic *"Repair only the malformed FORMAT
defects below"*, and the model re-offends for 4 attempts.

**Questions:**
- A1. Does the REPAIR prompt state the legal roster explicitly? Read
  `_run_markup_ladder` (`_otr_scifi_fable2.py:1793`) and `_script_user_prompt`.
  `base_user` is re-sent every attempt -- does it name the legal speaker labels
  in a form the model can obey, or only describe the cast prose-style?
- A2. Is *"the ONLY legal speaker labels are: X, Y, Z, ANNOUNCER -- re-attribute
  any other line to one of these or remove it"* SAFE advice to add to the repair
  rung? Argue against it too: could it cause the model to silently drop story
  content? Note the sprint lessons require returning ambiguous placement to the
  model and failing closed, and that we do NOT do deterministic Python folding.
- A3. Is the real defect upstream instead -- i.e. should the TREATMENT have
  included these characters? `cast_shapes` is capped at `max_length=8`
  (`_otr_scifi_fable2.py:297`). If the story needs a bit-part speaker, is the
  cap forcing the writer to either invent or distort? Check whether a
  bit-part/one-line speaker has any legal representation at all.
- A4. Note the ordering: `announcer outro missing before the CODA` may be the
  ROOT and the other three consequential. Is the ladder reporting a cascade as
  four independent defects, and would repairing only the first fix all four?

### DEFECT B -- `public_domain`: ledger save fails after visual_style embedding

    RuntimeError: failed to save ledger after visual_style pack embedding
      at nodes/OTR_LedgerScriptWriter.py:6394, in _run_writer_tail
      (called from OTR_LedgerScriptWriter.py:5997, in run)

Prompt ran 123.66 s then raised. The message names no path and no cause.

**Questions:**
- B1. Read `_run_writer_tail` around line 6394. What operation actually failed,
  and WHY does the raise discard it? A `RuntimeError` with no underlying
  exception, no path and no errno is nearly undiagnosable -- exactly the class
  of defect section 1 of this brief is about.
- B2. Is this bank-specific? `public_domain` carries `meta.provenance.*`
  (license label/url, source label/url, `blocks_publish`,
  `commercial_use_allowed`) that other banks do not -- see the ledger at
  `output/otr/episodes/signal_lost_entangled_at_twing_hall_20260812_070242/`.
  Does the visual_style pack embedding interact with those fields?
- B3. Is it a WRITE failure (disk, permissions, path length, a locked file, the
  janitor/PendingSweep deleting a pending dir mid-write -- see
  `[PendingSweep] BUG-LOCAL-290` firing repeatedly in the server log) or a
  SERIALIZE failure (a non-JSON-serializable value, like PBUG-20260812-02's
  bound method)? Say which, with evidence.
- B4. Does it leave a partial/corrupt ledger on disk?

### DEFECT C -- my own, already understood, listed so nobody re-diagnoses it

The `public_domain` leg ALSO shows
`NameError: name 'describe_execution_error' is not defined` at
`otr_api.py:749`. That is NOT a code defect: I edited `otr_api.py` between two
edits while the gate was spawning subprocesses that import it, and that leg
imported a half-applied file. The file is consistent now (verified: both
functions defined, call site correct, 15 tests green). **The leg must be
re-run**; do not diagnose this one.

---

## 4. Ground rules for your review

- **Every claim must cite a real file and line you actually read.** Label each
  CONFIRMED / SUSPECTED / UNVERIFIABLE. Unverified claims are hypotheses.
- **Do not propose content guardrails.** Operator directive: no profanity or
  violence filtering in the generation path, and the source's own language is
  carried as written.
- **Prompt-only fixes must stay prompt-only** -- nothing may turn an invalid
  script valid or weaken an acceptance gate.
- **Behavioural tests, not lexical ones.** Lesson L26: this project has been
  bitten repeatedly by tests asserting on strings/comments that pass while the
  code is wrong -- including twice today, once by a fixture shape no producer
  emits. Prefer fixtures derived from the real parser/producer.
- Report as a numbered list, most severe first, with file:line and a concrete
  failure scenario. No praise, no padding.
