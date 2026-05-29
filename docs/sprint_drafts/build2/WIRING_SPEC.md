# Build 2 Wiring Spec -- Slot-Formatted Output + Deterministic Tier-A Gate

**Date:** 2026-05-28
**Plan:** `workflows/GO_FORWARD_PLAN_v10_four_builds_2026-05-28.md` (Build 2)
**Status:** DRAFT for the integration session. Nothing here has been wired
into `nodes/` or the workflow JSON yet. These files are staged under
`docs/sprint_drafts/build2/` and do NOT auto-import into ComfyUI.

This spec tells the integration session exactly how to (a) make the
Story Room writer EMIT one `d###|SPEAKER: text` block per voiced slot,
(b) run `_otr_craft_floor` BEFORE `OTR_StoryRoomCommit` and hard-fail on
any Tier-A failure, and (c) keep the one-in / one-out commit invariant.

Build 2 MUST land before Build 4's `compose_exchange` (critique 2: the
integrity gate goes in before any exchange writing can make a mess).

---

## 0. What Build 2 is and is NOT

- **IS:** a deterministic format/integrity gate. Slot count, slot order,
  speaker match, empty line, per-line word floor, parse error, duplicate
  slot. Same input -> same verdict. No LLM, no taste, no semantics.
- **IS NOT:** the Tier-B semantic validator (costly-choice realized,
  scene-turn, EXPOSITION_DUMP). That is explicitly deferred in the plan
  and needs labeled false-positive/false-negative rates before any check
  is promoted to hard-fail. Do not add semantic checks to
  `_otr_craft_floor`.

---

## 1. Files staged in this folder

| File | Role |
|------|------|
| `_otr_craft_floor.py` | The deterministic Tier-A gate. Pure module. |
| `test_craft_floor.py` | pytest suite (clean pass, every failure code, determinism). |
| `WIRING_SPEC.md` | This document. |

### Promotion checklist when moving `_otr_craft_floor.py` to `nodes/`

1. Copy `_otr_craft_floor.py` to `nodes/_otr_craft_floor.py`.
2. **Add `from __future__ import annotations` back as the first import.**
   The staged copy omits it only to dodge a Python-3.10 dataclasses +
   pytest-import-hook race when the module is loaded by file path. Inside
   the `nodes/` package the normal import path is used, so the future
   import is safe and matches repo convention.
3. The module has **no `model_id` widget** and **no `INPUT_TYPES`** -- it
   is a helper module, not a node. It will not trip the Sprint 28
   forbidden-pattern sweep (`docs/_s28_forbidden_sweep.py`). No new LLM
   call site -> nothing to tag under project rule 6.
4. Update `test_craft_floor.py`'s loader block to
   `from nodes import _otr_craft_floor as cf` (the staged file-path loader
   block is documented in-line for exactly this swap), and move the test
   to `tests/test_craft_floor.py` so it joins the standard suite.

---

## 2. Public surface of `_otr_craft_floor`

```
parse_slot_lines(raw) -> list[(slot_id, speaker, text)]
normalize_slot_line(slot_id, speaker, text) -> str
evaluate_tier_a(raw, manifest, word_floor=4) -> CraftFloorResult
FAILURE_CODES  # tuple of the seven stable code strings
DEFAULT_WORD_FLOOR = 4
```

- `manifest` is the ordered expected-slot list. Accepts either
  `{"slot_id"/"dialogue_slot_id", "speaker"}` dicts OR
  `(slot_id, speaker)` tuples. **Pass the in-flight ledger's voiced
  lines straight through** -- they carry `dialogue_slot_id` + `speaker`,
  which `_normalize_manifest` reads natively.
- `CraftFloorResult`:
  - `.passed: bool` -- True iff zero failures.
  - `.failures: list[SlotFailure]` -- each has `code`, `slot_id`,
    `index`, `detail`. Order is deterministic.
  - `.failure_codes: list[str]` -- convenience.
  - `.parsed_slots: list[ParsedSlot]`.
  - `.word_floor: int`.
  - `.to_dict()` -- JSON-friendly for stamping onto `meta`.

### Failure-code vocabulary (the public contract)

| Code | Fires when |
|------|------------|
| `SLOT_COUNT_MISMATCH` | parsed block count != manifest slot count |
| `SLOT_ORDER_MISMATCH` | parsed slot ids (in order) != manifest slot ids |
| `SPEAKER_MISMATCH` | a row's speaker != manifest speaker for that slot id |
| `EMPTY_LINE` | a voiced row has empty text (more specific than the word floor; the same row is NOT also flagged `BELOW_WORD_FLOOR`) |
| `BELOW_WORD_FLOOR` | a non-empty row's word count < `word_floor` |
| `PARSE_ERROR` | a block opens with `d###\|` but does not match `d###\|SPEAKER: text` |
| `DUPLICATE_SLOT` | the same slot id appears in more than one parsed block |

Word counting mirrors `OTR_StoryRoomCommit._commit_dialogue` /
`production_ledger._word_count` (`[A-Za-z][A-Za-z0-9'-]*`) so a row that
passes the floor here produces the same `word_count` the commit node
stamps.

---

## 3. (a) Make the Story Room writer EMIT `d###|SPEAKER` one block per slot

Today the writer (`nodes/_otr_story_room.py`) emits free-form
speaker-prefixed PROSE (`SPEAKER NAME: speech`, one paragraph each), and
a separate **narrow Extract LLM pass**
(`_otr_story_room_extract.extract_dialogue_only` ->
`OTR_StoryRoomExtract`) maps that prose onto the ledger's
`dialogue_slot_id` list. Build 2 keeps that topology; it only tightens
the **output shape** so the gate has a clean, per-slot string to read.

Two acceptable wiring options -- pick ONE and write it down in the
sprint plan so the choice is auditable:

### Option A (preferred): gate the Extract output, no writer prompt change

The narrow Extract path already returns rows shaped
`{dialogue_slot_id, speaker, text, beat_id}` keyed to the ledger's
ordered `voice_slot_ids`. Serialize those rows to the slot-block string
with `normalize_slot_line` and run the gate on that string.

- **Where:** a new helper invoked from `OTR_StoryRoomExtract.run`
  (`nodes/OTR_StoryRoomExtract.py`) right after `extract_dialogue_only`
  returns `dialogue_rows`, OR from a thin new node placed between Extract
  and Commit (see section 4). Either is fine; the gate input is the same.
- **Build the raw string** in slot order:
  ```python
  raw = "\n".join(
      normalize_slot_line(r["dialogue_slot_id"], r["speaker"], r["text"])
      for r in dialogue_rows
  )
  ```
- **Build the manifest** from the in-flight ledger's voiced lines
  (already read in `OTR_StoryRoomExtract.run` as `voice_slot_ids`; pull
  the parallel `speaker` from the same `ledger_lines`):
  ```python
  manifest = [
      {"slot_id": str(ln.get("dialogue_slot_id") or "").strip(),
       "speaker": str(ln.get("speaker") or "").strip()}
      for ln in ledger_lines
      if str(ln.get("dialogue_slot_id") or "").strip()
  ]
  ```
- **No writer prompt change, no new LLM call.** This is the lowest-risk
  path and keeps Audio-is-king: the legacy/dormant paths are untouched.

### Option B: make the writer emit slot blocks directly

Change the writer system prompt (`_otr_story_room._WRITER_SYSTEM_PROMPT`,
the OUTPUT FORMAT section) so each speech is prefixed with its
`d###|SPEAKER:` slot id rather than `SPEAKER:`. This requires the writer
to KNOW the slot ids up front (pass the ordered voiced-slot manifest into
`build_writer_user_prompt`), and it removes the narrow Extract pass.
**Higher risk** -- it touches the creative writer prompt and the
reproducibility anchor. Defer to Build 4 unless the team explicitly wants
the writer to own slot ids now. If chosen, the writer turn stays
`# LLM slot: creative` (no new slot, no new widget; rule 6 unchanged).

> **Recommendation:** Option A for Build 2. It satisfies "one slot =
> exactly one committed text block" (critique 8) at the gate boundary
> without disturbing the writer or the audio baseline.

---

## 4. (b) Run `_otr_craft_floor` BEFORE `OTR_StoryRoomCommit`, hard-fail on Tier-A

The gate must run on the **same rows that Commit will join**, BEFORE the
ledger is written. There are two placements; pick ONE:

### Placement 1 (preferred): inside `OTR_StoryRoomExtract.run`

Run the gate right after `extract_dialogue_only` returns and BEFORE the
node emits its `story_room_extraction` payload. On a Tier-A failure,
emit a **failure sentinel** (the same `{"status": "failed", ...}` shape
the node already emits on `ExtractionCallFailedError`) with the failure
codes folded into `reason`. `OTR_StoryRoomCommit` already treats
`status != "ok"` as a hard-fail under `commit=True` (it raises
`StoryRoomCommitError` rather than silently falling back to legacy --
see `OTR_StoryRoomCommit._extraction_block_reason` -> `status_failed`).
So the red-graph behavior comes for free with no commit-node change.

- File: `nodes/OTR_StoryRoomExtract.py`, in `run`, after the
  `extract_dialogue_only(...)` call succeeds.
- Pseudocode:
  ```python
  from ._otr_craft_floor import evaluate_tier_a, normalize_slot_line
  raw = "\n".join(
      normalize_slot_line(r["dialogue_slot_id"], r["speaker"], r["text"])
      for r in dialogue_rows
  )
  manifest = [
      {"slot_id": str(ln.get("dialogue_slot_id") or "").strip(),
       "speaker": str(ln.get("speaker") or "").strip()}
      for ln in ledger_lines
      if str(ln.get("dialogue_slot_id") or "").strip()
  ]
  verdict = evaluate_tier_a(raw, manifest)   # word_floor default 4
  if not verdict.passed:
      payload = {
          "status": "failed",
          "reason": "Tier-A craft floor: " + ", ".join(verdict.failure_codes),
          "tier_a": verdict.to_dict(),
          "cast": [], "beats": [], "dialogue": [],
          "audio_cues": [], "running_facts": [], "arc": None, "premise": "",
      }
      return (json.dumps(payload, ensure_ascii=False, indent=2),)
  ```
- This keeps the gate deterministic and OFF the audio path until
  `commit=True`. When `commit=False` the existing dormant pass-through
  still holds (Audio-is-king / PD1 byte-identity unaffected).

### Placement 2: a new gate node between Extract and Commit

Add `OTR_StoryRoomCraftFloor` (a STRING-in / STRING-out node) wired
between `OTR_StoryRoomExtract.story_room_extraction` and
`OTR_StoryRoomCommit.story_room_extraction`. It parses the extraction,
rebuilds the manifest from the in-flight ledger, runs `evaluate_tier_a`,
and either passes the payload through unchanged (on pass) or rewrites it
to a `status:"failed"` sentinel (on fail). This adds one node + two links
to the workflow JSON (PD3). Prefer Placement 1 unless the team wants the
gate visible as its own node for operator inspection.

> **Whichever placement:** the gate is a HARD fail under `commit=True`.
> No silent fallback to legacy lines (matches the existing Commit
> contract -- integrity over recovery). The `to_dict()` of the verdict
> SHOULD be stamped onto the failure sentinel / `meta` so the operator
> sees exactly which slot broke and why on the red graph.

---

## 5. (c) Keep the one-in / one-out commit invariant

`OTR_StoryRoomCommit._commit_dialogue` already enforces the invariant
that Build 2 protects:

- draft row count == voiced slot count, else `SLOT_COUNT_MISMATCH`-style
  `StoryRoomCommitError`;
- draft slot id sequence == voiced slot id sequence (position-by-position);
- non-empty text per voiced slot;
- one ledger row written per voiced slot, no add/skip, no fallback.

The craft floor is a **pre-check with the same shape** so failures
surface BEFORE the commit attempt (clearer codes, earlier in the graph).
**Do NOT weaken or remove the Commit-node checks** -- they remain the
last line of defense. The two layers agree by construction (same slot-id
join, same word-count tokenizer). Build 2 adds no new commit path and
must not change the existing "commits N/N rows with no fallback" behavior
on a known-good draft.

---

## 6. Regression gates (run after wiring; do not skip)

Per project rules, run after every code change, without being asked:

```bash
# Bug Bible regression (primary quality gate)
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v

# Core
pytest tests/test_core.py -v

# Audio byte-identity (Audio is king -- must stay byte-identical to baseline)
pytest tests/test_audio_byte_identical.py -v

# This build's gate suite (after promotion to tests/)
pytest tests/test_craft_floor.py -v
```

Build 2 gate (from the plan):
- extract + commit still clean (rows committed == draft rows, no
  fallback) on a known-good draft;
- the validator produces **zero false-reds** on a known-good draft
  (`test_clean_pass_zero_failures` pins this);
- 100% deterministic -- same input -> same verdict
  (`test_determinism_*` pins this).

Audio must stay byte-identical: the gate only runs on the `commit=True`
Story Room path, which is opt-in. The dormant/legacy compose path is
untouched. If wiring the gate destabilizes the commit -> audio bridge,
revert immediately (Audio is king).

---

## 7. Workflow JSON (PD3)

- **Placement 1 (preferred):** NO workflow JSON change. The gate lives
  inside `OTR_StoryRoomExtract.run`; the node's class name, inputs,
  widgets, and output socket are unchanged. Verify the JSON still points
  at the same surfaces after the edit (it will).
- **Placement 2:** add the `OTR_StoryRoomCraftFloor` node + two links
  (Extract output -> gate input, gate output -> Commit's
  `story_room_extraction` input). Register the class in
  `nodes/__init__.py`. Update the workflow JSON: new node, two links,
  no new widgets, single STRING in / STRING out. Re-validate the JSON
  against the current node surface before calling the change done.

---

## 8. Open questions for integration (decide and record)

1. **Option A vs B (section 3)** and **Placement 1 vs 2 (section 4)** --
   pick one of each and note it in the sprint plan so the choice is
   auditable. The draft recommends Option A + Placement 1.
2. **Word floor value.** Default is 4. Confirm against the canonical
   episode: ANNOUNCER bookends and terse character lines may legitimately
   run short. If a real bookend dips below 4 words, either raise the
   bookend or lower the floor -- the floor is a parameter on
   `evaluate_tier_a` precisely so this is a one-line tune, not a code
   change.
3. **Should ANNOUNCER bookends be exempt from the word floor?** The
   current module applies the floor uniformly. If bookends should be
   exempt, that is a small additive change (skip the floor when
   `speaker == "ANNOUNCER"`), but it is a POLICY call -- left out of the
   draft so the integration session decides explicitly.
4. **Manifest speaker source.** This draft assumes the in-flight ledger
   line carries a `speaker` field parallel to `dialogue_slot_id`. Confirm
   that field name on the live ledger (the Extract node reads
   `dialogue_slot_id` from `ledger.lines`; verify `speaker` is present on
   the same rows). If the speaker lives under a different key, adjust the
   manifest-builder; the gate itself is agnostic.
5. **Where to stamp the verdict.** Recommend `meta.story_room_craft_floor`
   (mirroring `meta.story_room_commit`) so the proof survives onto the
   committed ledger for downstream forensic reads. Confirm the key name.
