# OTR Story-Quality Build Plan — Path Forward

Synthesis of the round-robin (3 model responses + the code review). No fluff.
Each sprint runs **REVIEW → CODE → WIRE → REGRESS → COMMIT**. Do them in order;
each unblocks the next. Sprints slot in after current Sprint H work.

**Canonical decision (locked):** the beat sheet is the single story engine and feeds
the writers' room (Path B) as the canonical renderer. The legacy per-line composer
(Path A) stays only as a constraint-repair fallback, never a silent quality fallback.
All three reviewers agreed: integrity first, then upstream structure, never more
serial review.

---

## Sprint 0 — Baseline proof (no code; prove the bugs before fixing)

**REVIEW**
- Read `_otr_outline.py` (not yet reviewed). Confirm whether per-beat `intent` is a
  real objective-with-turn or a flat label.
- Open 5 recent ledgers: read `meta.story_room_commit` → record `rows_committed` vs
  `rows_skipped`. High skip = Path B draft is being dropped (the dominant suspect).
- Count Story Room episodes terminating at editor cycle 0 with `pass_decision=True`
  on the first read = rubber-stamp rate.

**CODE / WIRE** — none.

**REGRESS** — n/a. This is the measurement baseline everything else is judged against.

**COMMIT** — `docs/2026-05-2x-otr-quality-baseline.md` with the three numbers. If
`rows_skipped` is high, Sprint 1 is confirmed as #1 priority.

---

## Sprint 1 — Delivery integrity (make the good draft land or crash loud)

Every prompt improvement is fake progress until this is true. This is the highest-
priority sprint by unanimous agreement.

**REVIEW**
- `OTR_StoryRoomCommit._commit_dialogue` (joins by raw `beat_id`).
- `_otr_story_room_extract.py` (numbers rows `b001+` in draft order, voiced only).
- `_otr_outline` beat-id assignment (numbers across ALL beats incl. SFX/music/announcer).
  Confirm the two numbering schemes do not align.

**CODE**
- Introduce a new primitive `dialogue_slot_id` (`d001`, `d002`, …), assigned **only to
  voiced dialogue beats**, separate from `beat_id` (which keeps covering SFX/music/
  bookends). Both reviewers invented this independently — it is the keystone.
- `Extract` emits rows keyed by `dialogue_slot_id` in draft order, not `beat_id`.
- `Commit` joins on `dialogue_slot_id` (or strict slot order), not raw `beat_id`.
- **Fail loud:** if `len(draft_rows) != len(voice_slots)` or `rows_skipped > 0`, raise
  `StoryRoomCommitError` and halt the node (turn the ComfyUI graph red). No silent
  fallback to legacy lines.
- Stamp the proof block:
  ```json
  "story_room_commit": {
    "enabled": true, "commit_mode": "dialogue_slot_order",
    "draft_rows": 18, "voice_slots": 18,
    "rows_committed": 18, "rows_skipped": 0, "fallback_to_legacy": false
  }
  ```

**Extract scope reduction (paired with `dialogue_slot_id`)**

Live observation 2026-05-27 (run `pending_20260527_223452`): `OTR_StoryRoomExtract`
attempt 1/2 at temp=0.20 is taking 5+ minutes per attempt. Math: ~3000-4000 tokens of
constrained JSON output (the full `StoryRoomExtractionSchema` — cast + beats + dialogue
+ audio_cues + running_facts + arc + premise) × Mistral-Nemo 12B NF4 under constrained
decode at ~10-15 tok/sec = 5+ minutes wall-clock. With `_EXTRACT_MAX_NEW_TOKENS=16384`
(BUG-293 cap) there's no early termination either — the model fills the budget.

Root cause: **Extract is re-extracting structure Stage 1 already produced.** Cast,
beats, arc, premise, running_facts are all known to the in-flight ledger from Stage 1.
The genuinely new content from the Story Room transcript is the dialogue lines and
nothing else.

Sprint 1 fix (slots into the `dialogue_slot_id` keystone work above):
- Split `extract_from_transcript()` into a narrow `extract_dialogue_only()` that emits
  `dialogue: [{dialogue_slot_id, speaker, text}]` and nothing else. Schema becomes a
  Pydantic mirror with one field.
- `OTR_StoryRoomExtract` reuses the existing Stage 1 plan (cast / beats / arc /
  premise / running_facts) from the in-flight ledger; only the dialogue array comes
  from the LLM. The full `StoryRoomExtraction` dataclass is reassembled in-node from
  ledger + LLM dialogue.
- Output drops from ~3000-4000 tokens → ~500-800 tokens. Per-attempt time drops from
  5+ min → 30-60 sec. Editor cycles benefit on the same axis (less context to validate).
- Same constraint as the `dialogue_slot_id` keystone: row count equality is the
  invariant. If `len(dialogue) != len(voice_slots)`, fail loud — same path as
  `StoryRoomCommitError`.

This is one CODE change at the same call site as `dialogue_slot_id`. Do them together.

**WIRE**
- No new graph edges; the Commit node already sits between writer `script_json` and the
  freeze cascade. Confirm `commit=True` and `use_story_room=True` on the canonical graph.

**REGRESS**
- Unit: slot-count mismatch raises; exact-match commits all rows.
- E2E: run 5 episodes; assert `rows_skipped == 0` and `fallback_to_legacy == false` in
  every ledger. Baseline skip-rate must drop to zero.

**COMMIT** — only when 5/5 episodes prove full commit. Tag `BUG-WAVE3-INTEGRITY`.

---

## Sprint 2 — Beat engine (replace the 350-char seed with a dramatic state object)

Where quality is actually born. Biggest ceiling-mover. Do after integrity so its effect
is measurable.

**REVIEW**
- `news_interpreter` `script_brief` (≤350 chars — the postage-stamp arc).
- `_otr_outline` beat construction and the `Beat` dataclass fields.

**CODE**
- Add an episode-level dramatic state object: `dramatic_question`, `character_a_wants`,
  `character_b_wants` (must oppose), `costly_choice_beat` (a `dialogue_slot_id`),
  `ending_change`.
- Upgrade each beat to a real schema, constrained-decoded: `dialogue_slot_id`, `speaker`,
  `objective`, `obstacle`, `turn`, `tactics_used` (verb phrase), `state_before`,
  `state_after`, `subtext`, `tension` (int), `next_turn`.
- **Structural validator (not taste):** post-generation, if `state_before` ≈ `state_after`
  (semantically identical) the beat has no turn → trigger a **structural re-roll of the
  beat sheet**, not a line re-roll. Also assert `costly_choice_beat` resolves and
  `ending_change != opening state`.
- Keep the brief ≤ its char budget but redirect token budget upstream: ~30% planning /
  ~50% beat+draft / ~20% validation.

**WIRE**
- `_otr_outline` consumes the new state object; Director/Writer prompts read beats from
  it. The dramatic state object becomes the reproducibility anchor.

**REGRESS**
- Validator suite: reject any beat sheet with a dead beat (`state_before==state_after`),
  missing costly choice, or unchanged ending. Assert re-roll fires and resolves.
- E2E: 5 episodes each produce a schema-valid beat sheet with a resolvable costly-choice
  beat.

**COMMIT** — when the validator rejects hand-crafted dead-beat fixtures and passes good
ones, and 5/5 live episodes validate.

---

## Sprint 3 — Arc-aware line generation (kill immediate-context bias)

**REVIEW**
- `_otr_line_composer._build_user_prompt`, `LineRequest`, the rolling-window block.

**CODE**
- Inject above the cast cards, per call: `DRAMATIC QUESTION`; `THIS BEAT` (objective /
  obstacle / turn / subtext / tension); `NEXT BEAT MUST REVEAL: {next_turn}`; then the
  existing last-3-lines window.
- Output constraints: "Write 1 spoken line. Do not summarize the objective. Do not
  explain the turn. Perform the objective indirectly. **The situation must be different
  after this line.**" The forced state-change line is the anti-decorative lever.
- The `next_turn`/`state_after` target sits directly above the generation slot as the
  magnetic pole (immediate-context bias works for you when the pull is local).

**WIRE**
- Writer threads the new beat fields onto `LineRequest`. Path A only — Path B's Writer
  already drafts whole-episode against the brief.

**REGRESS**
- Pin-prompt unit tests for the new block order (byte-stable static prefix preserved for
  KV reuse).
- Heuristic check: line does not verbatim-restate `objective`/`turn` text.

**COMMIT** — when prompt-shape tests pass and the no-restate heuristic holds on fixtures.

---

## Sprint 4 — Best-of-N at the structure (replace serial revision with parallel selection)

This is the §5b principle made real: take more shots at a good story, do not repair a
bad one.

**REVIEW**
- `run_story_room` loop budget; confirm serial-revision turns are capped (see Sprint 5).

**CODE**
- Generate **N=3 beat sheets** with diversity knobs, not 3 revisions:
  `A: moral-dilemma`, `B: bureaucratic-absurd`, `C: intimate personal-cost`.
- **Mechanical selector** (judges visible structure, never taste):
  ```json
  { "winner": "B", "reason": "Only B has an irreversible choice before the ending.",
    "scores": { "clear_opposed_desires":1, "costly_choice_present":1,
      "each_beat_changes_situation":1, "ending_changed_from_beginning":1,
      "no_alarm_countdown_rescue":1 } }
  ```
- Selector ties broken by highest summed structural score; winner enters the draft path.
- Run validation (Sprint 2) per candidate first; only validated candidates are eligible.
  (Validation = floor, best-of-N = ceiling. Do both, per the round-robin's open question.)

**WIRE**
- Beat-engine node fans out N candidates → selector node → single winning beat sheet into
  the Writer. One new selector node.

**REGRESS**
- Selector unit tests on fixtures where the structurally-strongest candidate is known;
  assert it wins. Assert all-invalid input fails loud rather than shipping a dead sheet.

**COMMIT** — when selector picks the known-best fixture ≥ target rate and never ships an
unvalidated sheet.

---

## Sprint 5 — Editor downgrade (constraint checker only; stop the churn)

Honors the core principle: revision cannot originate quality. Strip the editor of taste.

**REVIEW**
- `_otr_editor_pass` `_EDITOR_SYSTEM_PROMPT`, `run_editor`, `pass_decision`.

**CODE**
- Editor checks **only**: wrong speaker, phantom character, missing costly-choice beat,
  no turn in the final third, format failure. Remove "make it better / more drama /
  improve pacing."
- Cap serial revision turns hard (e.g. 1). Quality now comes from Sprint 4 selection, not
  from editor cycles.
- Editor returns failing constraints → one targeted repair turn → re-check → ship or
  fail loud. No open-ended improvement loop.

**WIRE** — none (same loop, narrower job).

**REGRESS**
- Editor flags injected constraint defects (wrong speaker, phantom name, no final-third
  turn) on fixtures and passes clean ones. Assert it no longer triggers revision on a
  structurally-valid-but-plain draft (that is Sprint 4's job, not the editor's).

**COMMIT** — when the constraint suite passes and revision-churn rate drops to the cap.

---

## Done = all five green + one human listen-test

Structural validators are the automated regression gate (judge visible structure, never
taste — the only thing small models score reliably). The **only** true quality gate is a
human A/B listen test (the existing Wave 3 mechanism): legacy Path A vs the new
beat-engine + best-of-N output. Ship the new path as default only when it wins the
listen test. Until then, integrity proof + structural validators keep the pipeline honest.

**Order is non-negotiable:** 1 (integrity) → 2 (beat engine) → 3 (arc-aware lines) →
4 (best-of-N) → 5 (editor downgrade). Doing 2–5 before 1 means tuning prose that never
reaches the bus.
