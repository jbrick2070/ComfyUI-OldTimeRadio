<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still asks R2 to build fixes that grounding says already exist, and the real gaps lack implementable APIs/data shapes.

MUST-FIX BEFORE BUILD:
1. [S1 / §1] Defect: “decouple prose from metadata” is contradicted by grounding. `_otr_line_composer.py compose_line_draft()` already generates only spoken dialogue; metadata is prebuilt on `LineRequest`. Building S1 as written will waste effort or regress the current architecture. Concrete fix: replace S1 with the real gap: per-line isolation. Define either:
   - `compose_scene_draft(scene_context, line_requests) -> {line_id: text}`, or
   - augment existing `compose_line()` with a `SceneArcContext` containing neighboring beats, escalation plan, prior unresolved pressure, and intended state change.
   Do not ask the LLM to emit metadata JSON.

2. [S3 / §3] Defect: “critic emits concrete correction_instruction” and “reroll targeted patch” are already implemented per grounding: `RerollTarget {line_id, hint}` and `_otr_reroll.py` patches stable `line_id`s. The missing part is critic scope and monotonic convergence. Concrete fix: change S3 to specify a critic interface that accepts a review scope, e.g. `review_ledger(..., scope_line_ids: set[str] | None, neighbor_window: int = 1)`, and returns flags only for `scope_line_ids + neighbors`. Add reroll-loop logic: track previous unresolved target count/set; stop as non-convergent if the scoped flagged count does not decrease after a cycle.

3. [S2 / §2] Defect: the “flat” definition is not codeable as written. It references `line_job`, `hidden_pressure`, “relative to prior lines,” and “change knowledge / pressure / relationship,” but the plan does not define where these fields live in the ledger, whether they are nullable, or how a deterministic test computes them. Grounding says `FlatLine.reason` is just a free string and composer/critic do not share a definition. Concrete fix: define a shared `FlatRubric` input/output contract:
   - inputs: `line_id`, `char_id`, `speaker_role`, `arc_phase`, `line_job`, `hidden_pressure`, previous/next neighbor text, current text;
   - output: `is_flat: bool`, `failed_dimension: enum`, `reason: str`, `minimal_fix: str`.
   Also define fallback behavior when `line_job` or `hidden_pressure` is missing.

4. [S2 / §2] Defect: per-role rules are underspecified against the actual data model. The plan says announcer/music/sfx are exempt from dialogue pressure, but grounding says the critic currently filters to character rows around `_otr_story_critic.py` L291-304. It is unclear whether music/sfx rows ever enter flat-line review. Concrete fix: explicitly define critic inclusion rules:
   - `speaker_role == character`: flat-line rubric applies;
   - `speaker_role == announcer`: frame/transition rubric only;
   - `cue_type in music_* | sfx`: excluded from flat-line review.
   This requires the `speaker_role` / `cue_type` split to exist before the critic relies on it.

5. [S5 / §5] Defect: `voice_preset REQUIRED` is correct but not implementable as stated because the failure paths are specific: `cast_lock.py` returns early when `cast_seed is None`, and unmatched `char_id`s preserve `None`. Concrete fix: add a postcondition after cast locking, regardless of seed:
   - collect all characters plus announcer requiring TTS;
   - if `voice_preset is None`, either deterministically assign from the valid preset list or raise a hard validation error before freeze/TTS;
   - define the source of the valid preset list and whether presets must be unique per character.
   Do not rely on `cast_seed` being present.

6. [S5 / §5] Defect: role-mismatch source is explicitly unverified. Grounding says `cast_lock` only reads `speaker_role or role`; the write that puts an engine name into the role field is upstream. “Hardcode the engine->role separation in casting/reviewer code” is not enough to locate or fix the write. Concrete fix: add an R3 trace task before implementation: identify every writer of `speaker_role`, `role`, `tts_model`, `voice_preset`, and `archetype`. Then add fail-closed validation at the first ledger construction boundary and again in `OTR_LedgerReviewer`.

7. [S5 / §5] Defect: the proposed schema split is incomplete. `speaker_role`, `cue_type`, `tts_model`, `voice_preset`, and `archetype` are named, but no migration or compatibility rule is given for existing rows using `role`. Concrete fix: define the row schema:
   - `speaker_role: "character" | "announcer" | null`;
   - `char_id: str | null`;
   - `cue_type: "music_intro" | "music_bridge" | "sfx" | ... | null`;
   - `tts_model: engine_roster enum | null`;
   - `voice_preset: preset enum | null`;
   - `archetype: "lead" | "foil" | "support" | null`.
   Add invariant matrix: character/announcer require `voice_preset`; cue rows must not.

8. [S4 / §4] Def