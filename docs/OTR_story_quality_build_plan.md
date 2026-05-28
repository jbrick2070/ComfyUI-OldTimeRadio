# OTR Story-Quality Build Plan -- Path Forward

**Audited and refined 2026-05-27 against live source** (`v2.0-alpha` @ `bcfe8a5`).
Each sprint runs **REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT** and is shaped as a
**Subagent Contract**: a fresh subagent with no prior context can read one sprint
section, execute it end to end, and ship a green-tests commit.

**Canonical decision (locked):** beat sheet is the single story engine and feeds
the writers' room (Path B) as the canonical renderer. Legacy per-line composer
(Path A) is a constraint-repair fallback only, never a silent quality fallback.

**Order is non-negotiable:** 1 (integrity) -> 2 (beat engine) -> 3 (arc-aware lines)
-> 4 (best-of-N) -> 5 (editor downgrade). Doing 2-5 before 1 = tuning prose that
never reaches the bus.

---

## Cross-sprint conventions

1. **Branch:** `v2.0-alpha` only. Never `main`.
2. **Python:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
3. **Regression gate after every code change** (no exceptions, no asking):
   ```cmd
   cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q
   C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q
   ```
   Baseline: **3597 passed / 21 skipped / 0 failed.** New tests added per sprint
   raise this floor; the count never decreases.
4. **Forbidden-pattern sweep after each sprint:**
   ```cmd
   C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
   ```
   Must exit 0. (Enforces PD6: no `model_id` widget on consumer nodes.)
5. **Word ban:** never use "dummy" in code, comments, fixtures, or commit
   messages. Use "placeholder" / "stub" / a descriptive name.
6. **Audio byte-identity:** verify legacy compose path stays byte-identical
   when `use_story_room=False` (PD1). Audit-only when commit toggles ledger
   writes.
7. **Git push:** Desktop Commander `cmd` shell only. Never PowerShell, never
   sandbox git. Commit message via `.git\COMMIT_EDITMSG` + `git commit -F`.
   Verify after push: `local HEAD == origin HEAD`, no 0-byte files, no BOM,
   all node classes registered in `__init__.py`.
8. **Wire every code change into the workflow JSON.** A Python-only change
   without a matching `workflows/otr_scifi_16gb_full.json` audit + edit is
   not done. After any node-side surface change (input name, output socket,
   widget name, class rename) re-run:
   ```cmd
   C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -c "import json,pathlib; json.loads(pathlib.Path('workflows/otr_scifi_16gb_full.json').read_text(encoding='utf-8')); print('JSON OK')"
   ```
9. **LLM slot tagging:** every LLM call site carries `# LLM slot: creative`
   or `# LLM slot: technical` with one-sentence reason. Model id arrives via
   STRING socket from the writer's broadcast outputs -- never a local widget.

---

## Sprint 0 -- Baseline proof (no code; prove the bugs before fixing)

**Subagent contract**
- Inputs: read access to `output/pending_*` and `output/episodes/*`.
- Outputs: one markdown file `docs/2026-05-27-otr-quality-baseline.md`.
- Acceptance: file commits cleanly; three numbers stamped.

**REVIEW**
1. Read `nodes/_otr_outline.py` lines 83-150 (`Beat` schema) and confirm whether
   `intent` is a real objective-with-turn or a flat label. (Field is currently
   a single sentence in `[4, 200]` chars -- it is a flat label; this becomes
   Sprint 2's motivation.)
2. List the 5 most recent ledgers under `output/pending_*` AND under
   `output/episodes/*`:
   ```cmd
   cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   dir /b /o-d output\pending_* output\episodes\*.json 2>nul
   ```
3. For each ledger, read `meta.story_room_commit.rows_committed` and
   `meta.story_room_commit.rows_skipped`. Tabulate.
4. Count episodes where `meta.story_room_editor_verdicts[0].pass_decision ==
   true` AND no further verdicts exist (rubber-stamp rate).

**CODE / WIRE** -- none.

**REGRESS** -- n/a (measurement only).

**COMMIT** -- `docs/2026-05-27-otr-quality-baseline.md` with:
- Total ledgers inspected (>= 5)
- Total `rows_committed` vs `rows_skipped` across all
- Rubber-stamp count (cycle-0 pass)
- Plain-prose verdict: if `sum(rows_skipped) > 0`, Sprint 1 is the
  highest-priority sprint and slot-id integrity is the keystone.

Commit message: `docs: Sprint 0 -- baseline proof captured`.

**Done-when**: file exists, three numbers present, commit pushed.

---

## Sprint 1 -- Delivery integrity (make the good draft land or crash loud)

Every prompt improvement is fake progress until this is true. Highest priority
by unanimous round-robin agreement.

### Sprint 1 -- corrections from the audit

The handoff suggested `_otr_outline.py` for slot-id work; the audit shows the
Story Room (Path B) consumes `_otr_stage1_plan.Stage1Beat`, not `_otr_outline.Beat`.
Slot ids live on `Stage1Beat` for Path B integrity. `_otr_outline.Beat` is the
Path A surface and gets a mirrored field for symmetry only when Sprint 4's
best-of-N selector reads from it.

### Sprint 1 -- surface map (audited)

| File | Symbol | Role in slot-id work |
|------|--------|----------------------|
| `nodes/_otr_stage1_plan.py` | `Stage1Beat` (line 144) | Add `dialogue_slot_id: Optional[str]` field |
| `nodes/_otr_stage1_plan.py` | `Stage1Plan` (line 259), `validate_plan_semantics` (line 328), `parse_and_validate_plan` (line 403) | Stamp `dialogue_slot_id` post-parse on voiced beats (`speaker != "MUSIC"` — covers cast-name speakers + ANNOUNCER bookends) in beat order: `d001`, `d002`, ... |
| `nodes/_otr_story_room_extract.py` | `StoryRoomExtractionSchema` (line 226), `_DialogueRow` (line 153) | Add `dialogue_slot_id` to dialogue row schema; new narrow `DialogueOnlySchema` |
| `nodes/_otr_story_room_extract.py` | `extract_from_transcript` (line 515) | Split into `extract_dialogue_only` returning `list[DialogueRow]` |
| `nodes/OTR_StoryRoomExtract.py` | `run` (line 215) | Pull Stage 1 plan from in-flight ledger; call narrow path; reassemble `StoryRoomExtraction` in-node |
| `nodes/OTR_StoryRoomCommit.py` | `_commit_dialogue` (line 179) | Join by `dialogue_slot_id`; fail loud on count mismatch (raise `StoryRoomCommitError`); no silent fallback |
| `nodes/_otr_story_room.py` | `build_writer_user_prompt` (line 325) | Tell the Writer how many voiced slots the episode has so its prose carries one beat per slot |
| `workflows/otr_scifi_16gb_full.json` | StoryRoom (id 75), Extract (id 76), Commit (id 77) | Confirm `commit=true`, `use_story_room=true` widgets and no socket renames |

### Sprint 1 -- subagent contract

- Inputs: branch `v2.0-alpha` at HEAD. Repo at the path above.
- Outputs: 4 source files edited, 1 workflow JSON re-validated, N new pytest
  cases under `tests/test_dialogue_slot_id.py`, 1 commit tagged
  `BUG-WAVE3-INTEGRITY`.
- Acceptance: pytest pass count >= baseline+8 (the new slot-id cases),
  forbidden-pattern sweep exits 0, workflow JSON parses, no `dummy` token in
  diff, no `model_id` widget added to consumer nodes.

### Sprint 1 REVIEW

Read top-to-bottom:
- `nodes/_otr_stage1_plan.py` (especially `Stage1Beat` + `validate_plan_semantics`).
- `nodes/_otr_story_room_extract.py` end-to-end.
- `nodes/OTR_StoryRoomCommit.py` end-to-end.
- `nodes/_otr_story_room.py` `build_writer_user_prompt` (line 325) +
  `_call_writer`.

Confirm with one paragraph in the commit body:
- Which beat schemas are touched.
- Which call sites currently look up by raw `beat_id` (the bug).
- Where ledger `lines[*]` rows get their `beat_id` (so the new
  `dialogue_slot_id` column has a mirror row to join against).

### Sprint 1 CODE

1. `nodes/_otr_stage1_plan.py` -- add to `Stage1Beat`:
   ```python
   dialogue_slot_id: Optional[str] = Field(
       default=None,
       pattern=r"^d\d{3}$",
       description=(
           "Sequence id for voiced beats only (d001, d002, ...), "
           "assigned in voiced-beat order. None on non-voiced beats "
           "(music_*, sfx, narrator-only). Sprint 1 keystone: Extract "
           "and Commit join on this, not raw beat_id."
       ),
   )
   ```
2. Same file -- in `parse_and_validate_plan` (or a post-parse helper
   `stamp_dialogue_slot_ids(plan)`), after `validate_plan_semantics`:
   ```python
   def stamp_dialogue_slot_ids(plan: Stage1Plan) -> Stage1Plan:
       """Stamp d001..dNNN on voiced beats in declaration order.

       Voiced determination on Stage1Beat: any beat whose speaker is
       NOT the literal "MUSIC". This covers both cast-name speakers
       (character) and "ANNOUNCER" bookends (also voiced -- rendered
       by Kokoro on a separate bus per Stage1CastMember docstring).
       Non-voiced beats (speaker == "MUSIC", i.e. music_inter slots)
       keep dialogue_slot_id = None.

       Note: Stage1Beat has NO speaker_role field; voicedness comes
       from the speaker value vs RESERVED_SPEAKERS ({"ANNOUNCER",
       "MUSIC"}). _otr_outline.Beat (Path A) does have speaker_role
       and a mirrored stamp lives there for Sprint 4 best-of-N
       symmetry.
       """
       counter = 1
       for beat in plan.beats:
           if beat.speaker != "MUSIC":  # ANNOUNCER + cast names = voiced
               beat.dialogue_slot_id = f"d{counter:03d}"
               counter += 1
       return plan
   ```
   Call `stamp_dialogue_slot_ids(plan)` at the end of `parse_and_validate_plan`.
3. **Adapter wire-through (the real `init_lines_from_outline` lives in
   `nodes/production_ledger.py` line 671, not `_otr_ledger.py`).**

   `init_lines_from_outline` reads beat attrs via `getattr(beat, "X",
   default)` for: `beat_id`, `speaker`, `speaker_role`, `mood`, `sfx_cue`,
   `intent`, `arc_phase`, `target_words`. Stage1Beat exposes
   `beat_id` + `speaker` + `intent` natively, but lacks
   `speaker_role` / `mood` / `target_words` / `arc_phase` / `sfx_cue`
   (Stage1Beat has `length_target_words` + `emotional_register` instead).

   Two ways to wire `dialogue_slot_id` through:

   **Option A (preferred):** add `dialogue_slot_id` to Stage1Beat AND
   add a one-liner getter inside `init_lines_from_outline` that reads
   it via `_g("dialogue_slot_id", None)` and writes it onto the line row.
   The existing Path A (`_otr_outline.Beat`) is the surface
   `init_lines_from_outline` was originally written for, so it already
   tolerates beats missing fields (the `_g` defaulting handles drift).

   **Option B:** introduce a `Stage1Beat.to_outline_beat()` adapter that
   maps `length_target_words -> target_words`, `emotional_register ->
   mood`, derives `speaker_role` from `speaker` (`"MUSIC" -> "music_inter"`,
   `"ANNOUNCER" -> "announcer"`, else `"character"`), and forwards
   `dialogue_slot_id` through.

   Pick Option A unless reading `init_lines_from_outline`'s call sites
   reveals existing Path-B consumers that already do the field-name
   translation -- in which case Option B is the right collapse point.
   The Sprint 1 owner reads `_otr_legacy_to_stage1_adapter.py` and the
   OTR_LedgerScriptWriter `run` method to decide. Stamp the decision in
   the commit message.

   Then:
   ```python
   # Inside init_lines_from_outline, alongside the existing _g calls:
   dialogue_slot_id = _g("dialogue_slot_id", None)
   # Inside the row dict:
   "dialogue_slot_id": dialogue_slot_id,
   ```
4. `nodes/_otr_story_room_extract.py` -- add narrow schema + function:
   ```python
   class _DialogueOnlyRow(BaseModel):
       dialogue_slot_id: str = Field(..., pattern=r"^d\d{3}$")
       speaker: str = Field(..., min_length=1, max_length=40)
       text: str = Field(..., min_length=1, max_length=2000)

   class DialogueOnlySchema(BaseModel):
       dialogue: List[_DialogueOnlyRow] = Field(..., min_length=1, max_length=64)

   def extract_dialogue_only(
       transcript_payload: Dict[str, Any],
       *,
       generate_fn: Callable[..., str],
       voice_slot_ids: List[str],
       cast_names: Optional[List[str]] = None,
       news_seed: str = "",
       max_attempts: int = 2,
   ) -> List[Dict[str, Any]]:
       """Extract ONLY the dialogue array from the transcript.

       The voice_slot_ids list pins how many rows the LLM must emit
       and what their dialogue_slot_id values are -- the row count
       must equal len(voice_slot_ids) exactly. The full
       StoryRoomExtraction is reassembled in OTR_StoryRoomExtract from
       the in-flight Stage 1 plan + this dialogue list.
       """
       # ... constrained-decode binding to DialogueOnlySchema, 2 attempts,
       # # LLM slot: technical
   ```
   The prompt block is reduced to: writer draft + cast hint + slot id list +
   instruction "Emit exactly N rows, one per slot id, in order." Output cap
   ~500-800 tokens (vs ~3-4k tokens for the full schema). Per-attempt time
   drops from 5+ min to 30-60 sec on Mistral-Nemo NF4.
5. `nodes/OTR_StoryRoomExtract.py` -- rewrite `run` to:
   - Resolve the in-flight ledger; read its `meta.stage1_plan` (or call
     `load_stage1_plan_from_ledger`) -> `Stage1Plan` (already slot-stamped).
   - Compute `voice_slot_ids = [b.dialogue_slot_id for b in plan.beats if b.dialogue_slot_id]`.
   - Call `extract_dialogue_only(...)` with `voice_slot_ids`.
   - Reassemble a full `StoryRoomExtraction` in-node from `plan.cast`,
     `plan.beats`, `plan.arc`, `plan.premise`, `plan.running_facts` + the
     LLM dialogue list.
   - Output the serialized JSON exactly as before.
6. `nodes/OTR_StoryRoomCommit.py` -- replace `_commit_dialogue` lookup:
   ```python
   class StoryRoomCommitError(RuntimeError):
       """Raised when dialogue commit cannot land cleanly. Halts the
       node (no silent fallback to legacy lines). Sprint 1 keystone:
       integrity over recovery."""

   def _commit_dialogue(self, ledger_dict, dialogue_rows):
       lines = ledger_dict.get("lines") or []
       voice_lines = [
           ln for ln in lines
           if (ln.get("dialogue_slot_id") or "").strip()
       ]
       voice_slot_ids = [ln["dialogue_slot_id"] for ln in voice_lines]
       draft_slot_ids = [
           str(row.get("dialogue_slot_id") or "").strip()
           for row in dialogue_rows
       ]
       if len(draft_slot_ids) != len(voice_slot_ids):
           raise StoryRoomCommitError(
               f"slot count mismatch: draft={len(draft_slot_ids)} "
               f"voice={len(voice_slot_ids)}"
           )
       if draft_slot_ids != voice_slot_ids:
           raise StoryRoomCommitError(
               f"slot order mismatch: draft={draft_slot_ids[:8]}... "
               f"voice={voice_slot_ids[:8]}..."
           )
       idx = {ln["dialogue_slot_id"]: ln for ln in voice_lines}
       committed = []
       for row in dialogue_rows:
           sid = row["dialogue_slot_id"]
           text = (row.get("text") or "").strip()
           if not text:
               raise StoryRoomCommitError(
                   f"empty text for slot {sid}; fail loud (no skip)"
               )
           target = idx[sid]
           target["text"] = text
           target["char_count"] = len(text)
           import re as _re
           target["word_count"] = len(_re.findall(
               r"[A-Za-z][A-Za-z0-9'\-]*", text
           ))
           committed.append(sid)
       return {
           "commit_mode": "dialogue_slot_order",
           "draft_rows": len(draft_slot_ids),
           "voice_slots": len(voice_slot_ids),
           "rows_committed": len(committed),
           "rows_skipped": 0,
           "fallback_to_legacy": False,
           "committed_slot_ids": committed,
       }
   ```
   Wrap the call site so `StoryRoomCommitError` halts the node (red graph in
   ComfyUI) instead of pass-through.
7. `nodes/_otr_story_room.py` -- in `build_writer_user_prompt` (line 325),
   append after the cast block:
   ```
   VOICED SLOTS: this episode has N voiced lines.
   Write each line as a single character utterance. Do NOT add extra
   characters or merge lines. The Editor will check that exactly N
   spoken lines exist in your draft.
   ```
   Where N is `len(voice_slot_ids)` computed from the Director-brief context
   or passed in as a new kwarg.
8. Stamp proof block (Commit node already builds this; just confirm shape):
   ```json
   {
     "enabled": true,
     "commit_mode": "dialogue_slot_order",
     "draft_rows": 18,
     "voice_slots": 18,
     "rows_committed": 18,
     "rows_skipped": 0,
     "fallback_to_legacy": false
   }
   ```

### Sprint 1 WIRE

- Open `workflows/otr_scifi_16gb_full.json` and confirm:
  - Node id 75 (`OTR_StoryRoom`) has `use_story_room=true` widget value.
  - Node id 76 (`OTR_StoryRoomExtract`) inputs unchanged.
  - Node id 77 (`OTR_StoryRoomCommit`) has `commit=true` widget value.
  - No socket names changed (Extract still emits `story_room_extraction`,
    Commit still has `script_json` + `story_room_extraction` inputs).
- Re-run the JSON parse sanity above.

### Sprint 1 REGRESS

Add new test file `tests/test_dialogue_slot_id.py` with at least 8 cases:
1. `stamp_dialogue_slot_ids` assigns d001..dN on every Stage1Beat whose
   speaker != "MUSIC" (covers cast names + ANNOUNCER); leaves None on
   MUSIC-speaker beats.
2. `Stage1Beat` rejects malformed `dialogue_slot_id` (e.g. `"d1"`, `"D001"`,
   `""`).
3. `parse_and_validate_plan` returns a plan with slot ids stamped.
4. `DialogueOnlySchema` validates a happy-path payload.
5. `DialogueOnlySchema` rejects an empty dialogue list.
6. `extract_dialogue_only` exhausts retries -> `ExtractionCallFailedError`.
7. `OTR_StoryRoomCommit._commit_dialogue` raises `StoryRoomCommitError` on
   slot-count mismatch.
8. `OTR_StoryRoomCommit._commit_dialogue` raises `StoryRoomCommitError` on
   slot-order mismatch.
9. Happy-path commit: exact slot match, 18 voice slots, 18 draft rows ->
   `rows_committed=18`, `rows_skipped=0`, `fallback_to_legacy=false`.
10. Stage 1 plan migration: a pre-Sprint-1 ledger lacking `dialogue_slot_id`
    on lines should fail the commit loudly (operator sees the red graph and
    knows to regenerate Stage 1).

Run:
```cmd
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/test_dialogue_slot_id.py -v
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
```

### Sprint 1 COMMIT

Stage everything:
```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git add nodes\_otr_stage1_plan.py nodes\_otr_story_room_extract.py nodes\OTR_StoryRoomExtract.py nodes\OTR_StoryRoomCommit.py nodes\_otr_story_room.py nodes\_otr_ledger.py tests\test_dialogue_slot_id.py docs\OTR_story_quality_build_plan.md
```
Write commit message via file tool to `.git\COMMIT_EDITMSG`:
```
sprint-1: dialogue_slot_id keystone + extract scope reduction (BUG-WAVE3-INTEGRITY)

- Stage1Beat carries dialogue_slot_id; stamped on voiced beats only.
- extract_dialogue_only narrow schema replaces full StoryRoomExtractionSchema
  for the LLM call; the full extraction is reassembled in-node from the
  in-flight Stage 1 plan + LLM dialogue.
- OTR_StoryRoomCommit joins by dialogue_slot_id, raises StoryRoomCommitError
  on count or order mismatch. No silent fallback to legacy lines.
- Writer prompt told the voiced-slot count so the prose carries one beat
  per slot.

Tests: tests/test_dialogue_slot_id.py adds 10 cases; full pytest passes;
Bug Bible regression passes; forbidden-pattern sweep exits 0.

Path A unchanged (legacy compose still byte-identical at use_story_room=False).
```
Commit + push:
```cmd
git commit -F .git\COMMIT_EDITMSG
git push origin v2.0-alpha
git log -1 --format=%H%n%s
```

### Sprint 1 LIVE SOAK (operator-driven, post-commit)

Jeffrey runs 5 episodes in ComfyUI Desktop with `use_story_room=true` +
`commit=true`. For each ledger:
- `meta.story_room_commit.rows_skipped == 0`
- `meta.story_room_commit.fallback_to_legacy == false`
- Extract per-attempt time logged in console drops from 5+ min to 30-60 sec.

If any of the 5 episodes shows `rows_skipped > 0` or a `StoryRoomCommitError`
fires unexpectedly: revert + open a Bug Bible candidate entry. Do NOT
proceed to Sprint 2 until 5/5 episodes prove full commit.

**Done-when**: 5/5 live ledgers carry the proof block above with rows_skipped=0.

---

## Sprint 2 -- Beat engine (replace 350-char seed with dramatic state object)

Where quality is actually born. Biggest ceiling-mover. Do after integrity so
the effect is measurable.

### Sprint 2 -- surface map

| File | Symbol | Role |
|------|--------|------|
| `nodes/_otr_news_wiring.py` | `build_news_briefs` | Currently writes `script_brief` (350 chars). Augment to also produce a `DramaticState` object stamped on the ledger. |
| `nodes/_otr_stage1_plan.py` | `Stage1Beat` (line 144) | Add objective/obstacle/turn/tactics_used/state_before/state_after/subtext/tension/next_turn fields (all optional during Sprint 2; required in Sprint 2.1). |
| `nodes/_otr_stage1_plan.py` | `Stage1Plan` (line 259) | Add top-level `dramatic_state: Optional[DramaticState]` field. |
| New file `nodes/_otr_dramatic_state.py` | `DramaticState` Pydantic | dramatic_question, character_a_wants, character_b_wants, costly_choice_beat (dialogue_slot_id), ending_change. |
| New file `nodes/_otr_beat_validators.py` | `validate_beat_sheet(plan)` | Structural validators only (no taste): dead-beat detect (state_before == state_after), costly_choice resolves, ending_change != opening state. Returns a list of structural defects; empty list = pass. Triggers structural re-roll of the BEAT SHEET (not a line re-roll) when defects exist. |
| `nodes/OTR_LedgerScriptWriter.py` / `_otr_director_brief.py` | brief construction | Read DramaticState from news_interpreter output; thread it into Director brief + Writer prompt. |

### Sprint 2 -- subagent contract

- Inputs: Sprint 1 committed and live-soak verified (5/5 clean). Branch
  `v2.0-alpha`.
- Outputs: 2 new modules, 4 edited modules, 1 workflow JSON re-validated,
  >= 15 new pytest cases, 1 commit tagged `SPRINT-2-DRAMATIC-STATE`.
- Acceptance: pytest pass count >= Sprint 1 floor + 15; validators reject
  hand-crafted dead-beat fixtures, pass good fixtures; 5/5 live episodes
  produce schema-valid beat sheets with resolvable costly_choice_beat.

### Sprint 2 REVIEW
- Read `nodes/_otr_news_wiring.py`: identify where `script_brief` is written
  to `meta.news` and which retry path applies.
- Read `nodes/_otr_director_brief.py`: where the Director brief is built and
  what fields it carries today.
- Read Writer prompt assembly (`_otr_story_room.build_writer_user_prompt`):
  where to inject the new state object.

### Sprint 2 CODE
1. `nodes/_otr_dramatic_state.py` (new):
   ```python
   class DramaticState(BaseModel):
       dramatic_question: str = Field(..., min_length=10, max_length=240)
       character_a_wants: str = Field(..., min_length=4, max_length=120)
       character_b_wants: str = Field(..., min_length=4, max_length=120)
       costly_choice_beat: str = Field(..., pattern=r"^d\d{3}$")
       ending_change: str = Field(..., min_length=4, max_length=200)
   ```
2. `nodes/_otr_stage1_plan.py` -- extend `Stage1Beat` with optional fields:
   `objective`, `obstacle`, `turn`, `tactics_used`, `state_before`,
   `state_after`, `subtext`, `tension` (int 1-5), `next_turn`. Field caps
   tight (<=120 chars each) so the constrained-decode budget stays small.
3. `nodes/_otr_beat_validators.py` (new): `validate_beat_sheet(plan)` returns
   `list[str]` defect messages. Three rules:
   - Dead beat: `state_before` and `state_after` are normalized-string-equal.
   - Costly choice resolves: `dramatic_state.costly_choice_beat` exists as a
     `dialogue_slot_id` on a beat, AND that beat's `state_after` != its
     `state_before`.
   - Ending change: `dramatic_state.ending_change` is reflected by the
     final voiced beat's `state_after != Stage1Plan.beats[0].state_before`.
4. Wire the validator into the beat-sheet generator's retry loop:
   structural defects trigger a structural re-roll (regenerate the beat
   sheet at a slightly higher temperature) up to 2 retries, then fail loud.
5. Token-budget redirect: ~30% planning / ~50% beat+draft / ~20%
   validation. (Verify by inspecting `_otr_episode_budget` allocations and
   the Stage 1/2/3 call temperatures.)

### Sprint 2 WIRE
- Director brief: add `dramatic_state` dict field; thread into the existing
  `raw_brief` block (also keep the legacy `dramatic_question` /
  `opposed_desires` fields for back-compat during this sprint).
- Writer prompt: add a `DRAMATIC STATE` block above the cast block.

### Sprint 2 REGRESS
Add `tests/test_dramatic_state.py` and `tests/test_beat_validators.py`:
- Pydantic happy/sad path on `DramaticState`.
- `validate_beat_sheet` rejects dead-beat fixture (state_before ==
  state_after), passes good fixture.
- `validate_beat_sheet` rejects unchanged-ending fixture.
- `validate_beat_sheet` rejects missing-costly-choice fixture.
- Integration: Stage 1 pipeline produces a Plan whose `dramatic_state` is
  populated and `validate_beat_sheet` returns `[]`.

### Sprint 2 COMMIT
Commit message file:
```
sprint-2: dramatic state object replaces 350-char seed; structural validators land

- New nodes/_otr_dramatic_state.py + nodes/_otr_beat_validators.py.
- Stage1Beat carries objective/obstacle/turn/tactics_used/state_before/
  state_after/subtext/tension/next_turn (optional Sprint 2; required Sprint 2.1).
- Stage1Plan carries dramatic_state.
- Structural re-roll of the beat sheet fires when state_before==state_after,
  costly_choice unresolved, or ending unchanged. Not taste; structure only.
- Writer prompt + Director brief read the new state.

Tests: 15+ new cases. Full pytest + Bug Bible green. Forbidden sweep clean.
```
Tag: `SPRINT-2-DRAMATIC-STATE`.

**Done-when**: pytest green, 5/5 live episodes have populated dramatic_state +
empty `validate_beat_sheet` defect list.

---

## Sprint 3 -- Arc-aware line generation (kill immediate-context bias)

### Sprint 3 -- surface map

| File | Symbol | Role |
|------|--------|------|
| `nodes/_otr_line_composer.py` | `LineRequest` (line 555), `_build_user_prompt` (line 964) | Path A only. Add DRAMATIC QUESTION / THIS BEAT / NEXT BEAT MUST REVEAL block above the rolling window (last 2 lines, line 1459). |
| `nodes/_otr_line_composer.py` | `compose_line` (line 1900), `compose_line_draft` (line 1692) | Thread `dramatic_state` + `beat_state` (objective/obstacle/turn/subtext/tension/next_turn) onto `LineRequest`. |
| `nodes/_otr_line_composer.py` | output prompt | Append: "Write 1 spoken line. Do not summarize the objective. Do not explain the turn. Perform the objective indirectly. The situation must be different after this line." |

### Sprint 3 -- subagent contract

- Inputs: Sprint 2 committed.
- Outputs: 1 file edited (Path A only), >= 6 new pytest cases for prompt
  shape + no-restate heuristic, 1 commit tagged `SPRINT-3-ARC-LINES`.
- Acceptance: pytest pass count >= Sprint 2 floor + 6; pin-prompt tests
  verify byte-stable static prefix (KV reuse preserved); no-restate
  heuristic holds on at least 4 fixtures.

### Sprint 3 CODE

In `_build_user_prompt` insert BEFORE the cast-card block, AFTER the static
system header (which is the KV-stable prefix -- do NOT touch its bytes):
```
DRAMATIC QUESTION: {dramatic_state.dramatic_question}
THIS BEAT:
  Objective: {beat.objective}
  Obstacle:  {beat.obstacle}
  Turn:      {beat.turn}
  Subtext:   {beat.subtext}
  Tension:   {beat.tension}/5
NEXT BEAT MUST REVEAL: {next_beat.next_turn or "(end of episode)"}
```
Then the existing rolling window of last 2 lines stays as the magnetic pole
directly above the generation slot. Append the output constraint sentence
verbatim:
```
Write 1 spoken line. Do not summarize the objective. Do not explain
the turn. Perform the objective indirectly. The situation must be
different after this line.
```

### Sprint 3 REGRESS

`tests/test_line_composer_arc.py`:
- Pin-prompt: `_build_user_prompt` produces the static system prefix
  byte-identical to a recorded fixture (KV reuse pin).
- The DRAMATIC QUESTION / THIS BEAT block renders all 5 fields in order.
- The NEXT BEAT MUST REVEAL line carries the right `next_turn` text.
- No-restate heuristic: when the LLM (mocked) outputs verbatim
  `{beat.objective}` or `{beat.turn}`, the validator flags it. (Run as a
  post-generation lint, not a re-roll trigger.)

### Sprint 3 COMMIT

Tag `SPRINT-3-ARC-LINES`. Commit message file:
```
sprint-3: arc-aware line composer prompt (Path A only)

- Prepend DRAMATIC QUESTION / THIS BEAT (objective/obstacle/turn/subtext/
  tension) / NEXT BEAT MUST REVEAL above the rolling window in
  _build_user_prompt.
- Append output constraint: 1 spoken line, no summary, no explanation,
  perform indirectly, situation different after.
- Path B unchanged (StoryRoom Writer drafts whole episode against brief).
- Static system prefix bytes preserved for KV reuse.

Tests: 6+ new cases. Pin-prompt fixture committed.
```

**Done-when**: pytest green, pin-prompt fixture committed, no-restate
heuristic test passes.

---

## Sprint 4 -- Best-of-N at the structure (not at line revision)

### Sprint 4 -- surface map

| File | Symbol | Role |
|------|--------|------|
| `nodes/_otr_stage1_call.py` | beat-sheet entry | Fan out N=3 calls with diversity-knob system prompts (A: moral-dilemma, B: bureaucratic-absurd, C: intimate personal-cost). |
| New file `nodes/_otr_beat_selector.py` | `select_winning_beat_sheet(candidates)` | Validate each via Sprint 2 `validate_beat_sheet`; only validated candidates eligible. Selector returns dict per design: `{winner, reason, scores: {clear_opposed_desires, costly_choice_present, each_beat_changes_situation, ending_changed_from_beginning, no_alarm_countdown_rescue}}`. Tie by summed structural score. |
| New file `nodes/OTR_BeatSelector.py` | ComfyUI wrapper node | Sits between the beat-engine fan-out and the Writer. One new node. |
| `workflows/otr_scifi_16gb_full.json` | new node id 78 | Wire OTR_BeatSelector between Stage 1 + StoryRoom. |

### Sprint 4 -- subagent contract

- Inputs: Sprint 3 committed.
- Outputs: 2 new modules, 1 new node, 1 workflow JSON edit (new node + 4
  links: 3 candidate inputs + 1 selected output), >= 8 new pytest cases,
  1 commit tagged `SPRINT-4-BEST-OF-N`.
- Acceptance: pytest pass count >= Sprint 3 floor + 8; selector picks the
  known-best fixture in >= 80% of cases; all-invalid input raises
  `NoValidBeatSheetError` (no silent shipping of a dead sheet).

### Sprint 4 CODE

1. `nodes/_otr_beat_selector.py` (new): pure mechanical scorer:
   ```python
   class BeatSelectorScores(BaseModel):
       clear_opposed_desires: int           # 0 or 1
       costly_choice_present: int
       each_beat_changes_situation: int
       ending_changed_from_beginning: int
       no_alarm_countdown_rescue: int       # penalty axis

   class NoValidBeatSheetError(RuntimeError): ...

   def select_winning_beat_sheet(
       candidates: list[Stage1Plan],
   ) -> tuple[Stage1Plan, dict]:
       """Return (winner, audit_dict). Raises NoValidBeatSheetError if
       all candidates fail validate_beat_sheet."""
   ```
2. `nodes/_otr_stage1_call.py`: fan out 3 calls with distinct system
   prompts. Use the structured_call ladder per existing convention.
3. `nodes/OTR_BeatSelector.py`: ComfyUI wrapper. 3 STRING inputs (candidate
   plans serialized), 1 STRING output (winning plan + audit). PD6: no
   model_id widget (selector is pure Python, no LLM call).

### Sprint 4 WIRE
- Insert OTR_BeatSelector as new node id 78 in
  `workflows/otr_scifi_16gb_full.json`.
- Stage 1 fan-out emits 3 STRING outputs (candidate_a, candidate_b,
  candidate_c). Connect to selector. Selector winning_plan -> StoryRoom.

### Sprint 4 REGRESS

`tests/test_beat_selector.py`:
- Mechanical scoring: known-best fixture wins, ties broken by total score.
- All-invalid input -> `NoValidBeatSheetError`.
- One-valid-one-invalid -> valid one wins regardless of score.
- Selector emits the audit dict with all 5 scoring axes.

### Sprint 4 COMMIT

Tag `SPRINT-4-BEST-OF-N`. Commit message file:
```
sprint-4: best-of-N beat sheet selection (parallel, not serial)

- Stage 1 fans out 3 candidate beat sheets with diversity knobs
  (moral-dilemma / bureaucratic-absurd / intimate personal-cost).
- New nodes/_otr_beat_selector.py + nodes/OTR_BeatSelector.py: pure
  mechanical scorer (no LLM call) judges visible structure only.
- Sprint 2 validators run per candidate; only validated candidates
  eligible. All-invalid input -> NoValidBeatSheetError (no silent
  shipping of a dead sheet).
- Workflow JSON wires new node id 78 between Stage 1 and StoryRoom.

Tests: 8+ new cases. Selector picks known-best fixture > 80% of time.
```

**Done-when**: pytest green; selector wins on fixtures; workflow JSON parses.

---

## Sprint 5 -- Editor downgrade (constraint checker only; stop the churn)

### Sprint 5 -- surface map

| File | Symbol | Role |
|------|--------|------|
| `nodes/_otr_editor_pass.py` | `_EDITOR_SYSTEM_PROMPT` (line 280), `build_editor_prompt` (line 338) | Strip "make it better / improve pacing / more drama" verbs. Limit checks to 5 categories. |
| `nodes/_otr_editor_pass.py` | `EditorVerdict` (line 172), `EditorVerdictSchema` (line 204) | Add `failing_constraints: list[str]` mapping to the 5-category enum. Keep `pass_decision` + `cycle`. Drop `per_axis_notes` taste field (or keep but stop rendering it into the Writer revision prompt). |
| `nodes/_otr_editor_pass.py` | `run_editor` (line 432) | Same call shape; new rubric subset. |
| `nodes/_otr_story_room.py` | `run_story_room` (line 498), constants `DEFAULT_MAX_EDITOR_CYCLES` + `DEFAULT_MAX_TOTAL_TURNS` (lines 56-71) | Cap `max_editor_cycles=1` (hard). |
| `nodes/OTR_StoryRoom.py` | widget defaults | Mirror the cap in the node's widget. |

### Sprint 5 -- subagent contract

- Inputs: Sprint 4 committed.
- Outputs: 2 source files edited, 1 widget default changed, >= 6 new
  pytest cases, 1 commit tagged `SPRINT-5-EDITOR-DOWNGRADE`.
- Acceptance: editor flags injected defects on fixtures, passes clean
  fixtures, does NOT trigger revision on a structurally-valid-but-plain
  draft.

### Sprint 5 CODE

1. New `_EDITOR_SYSTEM_PROMPT`:
   ```
   You are the Editor. You are NOT a taste editor. You check 5
   constraints only:
     1. WRONG_SPEAKER -- a line is attributed to a name not in the cast.
     2. PHANTOM_CHARACTER -- a non-cast name speaks or is named.
     3. MISSING_COSTLY_CHOICE -- the costly_choice_beat does not resolve.
     4. NO_FINAL_THIRD_TURN -- the final third of the draft has no
        state change (state_before == state_after across the run).
     5. FORMAT_FAILURE -- malformed JSON / off-schema output.

   You do not improve pacing. You do not request "more drama." You do
   not rewrite. You return a verdict only.
   ```
2. `EditorVerdictSchema` gets a `failing_constraints: List[str]`
   constrained to the 5 enum values above.
3. `run_story_room` caps `max_editor_cycles` to 1; on Editor fail, run
   ONE targeted repair Writer turn that addresses only the failing
   constraints, then re-check, then ship or fail loud (no open-ended
   improvement loop).

### Sprint 5 REGRESS

`tests/test_editor_constraints.py`:
- Editor flags WRONG_SPEAKER on a fixture where dialogue uses a name not
  in cast.
- Editor flags PHANTOM_CHARACTER on a fixture with a stranger name.
- Editor flags MISSING_COSTLY_CHOICE on a fixture where
  costly_choice_beat's state_before == state_after.
- Editor flags NO_FINAL_THIRD_TURN on a fixture where the last third
  beats are all dead.
- Editor passes a clean fixture.
- Editor does NOT trigger revision on a structurally-valid-but-plain
  fixture (no taste flag).

### Sprint 5 COMMIT

Tag `SPRINT-5-EDITOR-DOWNGRADE`. Commit message file:
```
sprint-5: editor downgrade to constraint checker; max_editor_cycles=1

- _EDITOR_SYSTEM_PROMPT strips taste verbs. Editor checks 5 constraints:
  WRONG_SPEAKER / PHANTOM_CHARACTER / MISSING_COSTLY_CHOICE /
  NO_FINAL_THIRD_TURN / FORMAT_FAILURE.
- EditorVerdictSchema.failing_constraints replaces taste rubric axes
  in the revision-prompt rendering.
- run_story_room caps max_editor_cycles at 1; one targeted repair turn
  then ship or fail loud. Quality comes from Sprint 4 selection, not
  editor cycles.

Tests: 6+ new cases.
```

**Done-when**: pytest green; editor does not trigger revision on plain
but valid drafts.

---

## Done = all five green + one human listen-test

Structural validators are the automated regression gate (judge visible
structure, never taste). The **only** true quality gate is a human A/B
listen test (the existing Wave 3 mechanism): legacy Path A vs the new
beat-engine + best-of-N output. Ship the new path as default only when
it wins the listen test. Until then, integrity proof + structural
validators keep the pipeline honest.

---

## Subagent dispatch protocol

A subagent picking up any sprint receives:
1. This document (whole, not excerpted).
2. The branch state (HEAD commit hash + working tree clean check).
3. A scratch file `docs/sprint_<n>_log.md` to write its REVIEW
   paragraph, test counts, and post-commit verification into.

The subagent's contract:
- Read the named sprint section in full.
- Execute REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT.
- Run the cross-sprint regression gate after EVERY edit, not at the end.
  (Detect breakage cheaply.)
- Stop and surface if any of:
  - pytest count drops below baseline.
  - forbidden-pattern sweep exits non-zero.
  - workflow JSON fails to parse.
  - any LLM call site lacks the `# LLM slot:` tag.
  - any new code uses the word "dummy".
  - any new consumer node has a `model_id` widget.
- After commit + push, write a one-paragraph status update to
  `docs/sprint_<n>_log.md`: HEAD hash, new test count, time elapsed,
  any open questions for the next sprint.

---

## Open carryovers (decide BEFORE Sprint 2 starts)

1. **Writer halt on news-brief exhaustion** (Jeffrey 2026-05-27): if
   `build_news_briefs` exhausts retries, the writer must HALT and re-roll
   news, not continue with `meta["news"] = None`. The resolver band-aid
   (BUG-294 cross-ledger walker) hides this. Decision: add as a Sprint 2
   sub-section ("News briefs are mandatory; writer raises on exhaustion;
   queue retry re-rolls news"). Land it alongside the DramaticState lift.
2. **DramaticState storage** -- attach to Stage 1 plan (top-level
   `dramatic_state: Optional[DramaticState]`). The state object becomes the
   reproducibility anchor.
3. **Sprint 5 cap-to-1 timing** -- defer to Sprint 5 as written; do not
   pre-empt during Sprints 2-3.

---

## Resume instructions for the morning

In a fresh window with the OldTimeRadio folder selected, attach
`docs/OTR_story_quality_build_plan.md` and `session_handoff.md` and say:

"Read the build plan. Run the cross-sprint regression gate. If green,
start Sprint 1. Stop after Sprint 1 commit + push and hand me the live-soak
checklist for the 5-episode test."
