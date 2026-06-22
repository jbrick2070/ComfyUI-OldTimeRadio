<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The wiring plan still targets the wrong integration point for role_mismatch, duplicates existing LineRequest arc fields, and leaves schema migration/validation ordering ambiguous enough to break cue rows and cast/TTS handoff.

MUST-FIX BEFORE BUILD:

1. [FIX 5 / R3 hand-in] Defect: role_mismatch source is misidentified as “upstream writer stamping an engine name.” Grounding identifies the source as `nodes/_otr_ledger_reviewer.py:500` inside `_render_cast_contract_for_audit`:
   `role = row.get("speaker_role") or row.get("tts_model") or ""`.
   This fallback converts `tts_model` values like `kokoro`/`bark` into role values and then repair rejects them. Concrete fix: change that line to use only an explicit role field, e.g. `role = row.get("speaker_role") or ""`, and fail validation if `speaker_role` is absent where required. Do not spend R3 tracing for an upstream writer unless this reviewer fallback is first removed.

2. [FIX 5] Defect: proposed schema says `speaker_role in {character, announcer}` and `cue_type in {music_*, sfx}`, but the grounded current reviewer repair allow-list is `_ALLOWED_SPEAKER_ROLES = {character, announcer, music_open, music_close, music_inter, sfx}`. If you add fail-closed validation before migrating existing cue rows, all existing music/sfx rows encoded as role values will fail. Concrete fix: sequence schema migration before validation:
   1. If legacy `role`/`speaker_role` is `music_open|music_close|music_inter|sfx`, move it to `cue_type`.
   2. Set `speaker_role=""` or omit it for cue rows.
   3. Require `speaker_role in {character, announcer}` only for spoken rows.
   4. Update `OTR_LedgerReviewer` and deterministic repairs to use the same normalized schema.

3. [FIX 5 / Sequencing + acceptance] Defect: “Fail-closed validation at the FIRST ledger-construction boundary AND in `OTR_LedgerReviewer`” is under-specified and can be ordered too early. If validation runs before legacy role migration, it will reject existing valid cue rows. Concrete fix: define the exact order as:
   legacy row normalization/migration -> schema validation -> cast_lock -> freeze/reviewer audit -> TTS.
   The “first boundary” must be after normalization, not before it.

4. [FIX 3 / R3 hand-in] Defect: the plan says to add a new `SceneArcContext` to `LineRequest`, but grounding shows `LineRequest` already carries `arc_phase`, `dramatic_question`, `beat_objective`, `beat_obstacle`, `beat_turn`, `beat_subtext`, `beat_tension`, `next_turn`, `last_lines`, `outline_spine`, and `current_beat_block`. Adding another context object duplicates the interface and risks prompt divergence between old and new arc fields. Concrete fix: do not add `SceneArcContext` as a new interface. Re-point FIX 3 to the beat-planning/outline/slot-drama contract code that populates the existing `LineRequest` arc fields. Only add missing scene-level continuity constraints if they are not already expressible through the existing fields.

5. [FIX 1 / critic call sites] Defect: the plan names `review_ledger(..., scope_line_ids=...)`, but grounding says the actual wiring point is `run_story_critic(...)` at:
   - `nodes/_otr_freeze_cascade.py:754` initial whole-episode pass
   - `nodes/_otr_reroll.py:621` reroll-loop pass
   If only the internal reviewer function is changed, the reroll caller cannot pass the scope and the freeze path may break on a signature mismatch. Concrete fix: add `scope_line_ids: set[str] | None = None` to `run_story_critic`, thread it down to `review_ledger`, pass `None` from freeze-cascade, and pass the patched target line set from `_otr_reroll.py`.

6. [FIX 1] Defect: “flags only `scope_line_ids` + their continuity neighbors” depends on how neighbors are selected, but no interface contract is defined. If neighbors are selected by list index after patching, deleted/reordered lines or duplicate IDs can produce unstable scope. Concrete fix: define neighbor selection against the canonical ledger order after patch application, keyed by stable `line_id`, and validate duplicate/missing `line_id` before invoking scoped critic.

7. [FIX 1 / FIX 4] Defect: structured critic output adds `failed_dimension`, but the reroll hint path is not explicitly wired to consume it. The plan says “the `hint` names WHICH dimension,” but does not name the producer/consumer boundary. Concrete fix: update the critic flag schema, parser, and reroll target construction together so every flagged row carries `line_id`, existing reason text, and optional/required `failed_dimension`. If the LLM omits or emits an invalid enum, map it to a named parse/validation error or a deterministic fallback dimension before reroll.

8. [FIX 2 / workflow node surface] Defect: voice preset validation is specified in `cast_lock.py`, but the actual workflow has cast node id=80 feeding TTS nodes id=81 `OTR_BatchCharacterVoices` and id=82 `OTR_AnnouncerVoice`. If cast_lock only logs/repairs but still emits rows with missing `voice_preset`, TTS receives invalid input. Concrete fix: make node 80’s output contract fail-closed: after cast-lock, before output to nodes 81/82, assert all `speaker_role in {character, announcer}` rows have non-empty `voice_preset`; cue rows must not be sent to character/announcer TTS.

9. [FIX 2] Defect: “Persist `cast_seed` into `meta.cast_contract` at cast time” is not enough to guarantee replay if downstream code reads `cast_seed` from a different location. Grounding only confirms the workflow node surface, not the metadata access path. Concrete fix: verify every reader of cast seed and standardize on one path. If `meta.cast_contract.cast_seed` is canonical, update cast_lock and any replay path to read that location. verify: exact metadata key currently used by `cast_lock.py`.

10. [Sequencing + acceptance / R3 hand-in] Defect: “sequence the workflow-JSON / node changes” conflicts with grounding: `critic/reroll/reviewer run INSIDE the writer pipeline`, and `FIX 1/4/5 are code-internal`; `FIX 2` is inside `OTR_CastLock` node 80. No JSON/node/widget change is required. Concrete fix: explicitly state “no workflow JSON changes, no node/widget additions” for these fixes, and add a regression check that `workflows/otr