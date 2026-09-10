# My Story -- D1 independent bank design

Date: 2026-09-10. Status: Sprint 4 implementation under final verification.
Sprint 5 native App and live publication qualification remain pending.

Baseline: `v2.0-alpha` HEAD `75bb405b` == origin. Canonical SHA-256
`b24221b6f672a99cfe6448684ef79d08dc832051dea7c66c050c04b6f4465629`
(23 nodes, 62 links, last_link_id 290, writer node 1 carries 33 widgets and
34 input descriptors, gate_in at input slot 32, replay_from at 33).

Scope authority: `docs/2026-09-10-my-story-app-scope.md` (R1-R4 converged) plus
two operator additions received 2026-09-10 while D0 was in flight:

1. **Attribution.** A listener may name who the story is by. The announcer
   and the credits use that name when given and a neutral phrase when not.
   The operator's own name never appears by default anywhere in the episode.
2. The README, the source-bank guides and a new My Story guide may be
   refreshed freely, and the preflight may gain My Story-specific checks.

## 0. Exposure disclosure (Gate 1 honesty)

Before this design was written, the driver read the ORIGINAL lane's
input branch (`nodes/_otr_writer_inputs.py` original branch, `operator_hint`)
and its creative front (`nodes/_otr_original_radio.py::build_original_briefs`)
in order to understand what "premise" currently means, and read the
SCI-FI lane (`nodes/_otr_scifi_news_pro.py`) as the only existing example of
the dispatched-lane INTERFACE (runner signature, tail parts, the ledger row
contract, authorship receipt, content-owned freeze). No before-comparison
receipt is claimed. The independence comparison in section 9 is made against
both lanes with that exposure on the record. Shared INTERFACES are reused on
purpose (they are the production contract); the pass graph, artifacts,
prompts and ledger-write strategy below are this bank's own.

## 1. Registry rows

### 1.1 `nodes/story_packs/banks.json` -- new row

```json
{
  "source_bank_id": "my_story",
  "label": "My Story",
  "source_kind": "user_story",
  "interpreter": "",
  "fetcher": "",
  "default_story_model": "my_story",
  "default_story_pipeline": "my_story_multipass",
  "defaults": {
    "story_form_label": "radio drama from a listener's idea",
    "source_material_label": "Listener story idea",
    "title_form_label": "radio drama",
    "hud_origin_label": "STORY ORIGIN",
    "credits_source_line": "a story from a listener's own idea, produced by machine for this broadcast",
    "style_pool_class": "generic",
    "story_scaffold": "off",
    "story_input_mode": "user_fields_v1",
    "auto_select": false
  },
  "required_seams": [],
  "runnable": true,
  "guide_ref": "Type your idea in Story input, optionally add characters, plot, setting and who the story is by, pick a look, and run. docs/MY_STORY_GUIDE.md."
}
```

Decisions, each with its reason:

* `fetcher` and `interpreter` are EMPTY: the listener's fields are the source.
  The registry law (b) in `_crossref_bank` today accepts a runnable bank with
  empty ids when its pipeline is not source-contract and `executable=true`.
  This design ADDS a stricter rule for `user_fields_v1`: fetcher and
  interpreter MUST be empty, so a user-input bank can never quietly also
  fetch.
* `required_seams: []` and NO `line_composer_system` in the pack. That is what
  routes the freeze to `content_owned_readonly`
  (`_otr_freeze_cascade.resolve_freeze_policy`), which verifies the runner's
  sealed authorship receipt instead of running the legacy inline mutators.
* `story_scaffold: off` for the same reason the original bank sets it: the
  listener's idea is the premise; a catalog premise beside it would fight it.
* `credits_source_line` is the NEUTRAL fallback. When the listener supplies
  `story_author` the writer overrides this stamp (section 6). The row never
  names the operator.
* `runnable: true` lands in the SAME chunk as the pack, runner, lane spec,
  input route, canonical wiring and tests (scope: "No new source bank is
  marked runnable before ... land together"). Live publication proof is
  sprint 5; the offline closure tests in section 10 are the gate for the
  flag in sprint 4.

### 1.2 `nodes/story_packs/pipelines.json` -- new row

```json
{
  "story_pipeline_id": "my_story_multipass",
  "label": "My Story (listener idea -> interpretation -> treatment -> acts -> frame)",
  "executable": true,
  "requires_source_contract": false,
  "declared_seams": [
    "my_story_interpret_system",
    "my_story_treatment_system",
    "my_story_act_system",
    "my_story_frame_system"
  ],
  "passes": [
    {"pass_id": "interpret",  "slot": "technical", "seam_refs": ["my_story_interpret_system"],
     "description": "P0 technical: the raw field bundle -> StoryInterpretation (requirements vs mentions, named cast, assumptions, conflicts, cast-size plan)."},
    {"pass_id": "treatment", "slot": "creative",  "seam_refs": ["my_story_treatment_system"],
     "description": "P1 creative: interpretation -> StoryTreatment (title, dramatic question, setting, cast with voice traits, per-act plan, ending)."},
    {"pass_id": "acts",      "slot": "creative",  "seam_refs": ["my_story_act_system"],
     "description": "P2 creative, one call per act: treatment + prior act digest -> ActScript (the scene's spoken lines)."},
    {"pass_id": "frame",     "slot": "creative",  "seam_refs": ["my_story_frame_system"],
     "description": "P3 creative: treatment + attribution -> StoryFrame (announcer intro/outro, coda, music cues)."},
    {"pass_id": "voices",    "slot": "technical", "seam_refs": [],
     "description": "P4 pure Python: deterministic voice-preset assignment from the open voice pool by gender and timbre; no LLM call."},
    {"pass_id": "assemble",  "slot": "technical", "seam_refs": [],
     "description": "P5 pure Python: cast/scenes/shots/beats/lines/music rows, authorship receipt, TTS delivery text, word-delivery and cast-contract stamps."}
  ],
  "notes": [
    "executable:true = nodes/_otr_my_story.py::run_my_story_episode via _otr_lane_specs.LANE_SPECS.",
    "The listener's fields are the sole source authority; no spark deck, no RSS, no manifest is consulted.",
    "Failure never routes to another pipeline."
  ]
}
```

`slot` for the two Python passes is declared `technical` because the registry
schema requires one of the two values; the description says no LLM runs.

### 1.3 Registry validation additions (`nodes/_otr_story_routing.py`)

* `_parse_bank`: validate `defaults.story_input_mode` in `{"legacy",
  "user_fields_v1"}` (absent = legacy) and `defaults.auto_select` is a bool
  (absent = true). Reject `user_fields_v1` with `auto_select` true.
* `_crossref_bank` rule (b) gains: when `story_input_mode == "user_fields_v1"`
  the bank must declare NO fetcher and NO interpreter, and its pipeline must
  be non-source-contract with `executable=true`.
* New pure helpers: `story_input_mode(bank) -> str` and
  `effective_auto_select(bank) -> bool` take a resolved `SourceBank` OR
  `None` and answer `"legacy"` / `True` for `None`. New NON-RAISING lookup
  `find_bank(source_bank_id) -> SourceBank | None` (`reg.banks.get`), distinct
  from `get_bank` and `require_runnable_bank`, which keep raising
  `UnknownBankError`. The roll sentinel and any unknown id resolve to `None`
  through `find_bank` (review fold: `get_bank` raises on the sentinel, which
  is the canonical default). No new top-level keys; client rows are unchanged
  when they omit both defaults.

### 1.4 Roll pool (`nodes/_otr_rolls.py`)

`eligible_bank_ids()` becomes `bank.runnable and effective_auto_select(bank)`,
still sorted by id. `_bank_pool_diagnosis()` names the second filter. The
five current banks keep `auto_select` absent (= true) so the pool is unchanged
for them; `my_story` is manual-only. The blank automatic canonical run
(source_bank = "roll (any eligible bank)") therefore never lands on My Story.

### 1.5 Lane spec (`nodes/_otr_lane_specs.py`)

`LANE_SPECS["my_story_multipass"] = LaneSpec(module="_otr_my_story",
runner_attr="run_my_story_episode")`. Lazy, by name, exactly like the sci-fi
row. `INLINE_PIPELINES` is unchanged.

## 2. The writer's input surface

### 2.1 Widgets (append-only, canonical + INPUT_TYPES + run() together)

Appended AFTER `replay_from`, in this order, all `STRING`, default `""`:

| widget | multiline | App label (sprint 5) | meaning |
|---|---|---|---|
| `story_characters` | yes | Character names and notes | names, relationships, roles, voice or identity details |
| `story_plot` | yes | Plot ideas | events, conflicts, an ending |
| `story_setting` | yes | Setting | place, era, atmosphere, world |
| `story_author` | no | Story by (optional) | who the story is attributed to |

`custom_premise` keeps its slot (4) and is the general idea ("Story input").
`num_characters` (1) and `act_count` (6) keep their meaning.

Widget vector: 33 -> 37 values on node 1; input descriptors 34 -> 38 (four
`{"widget": {"name": ...}}` entries appended after `replay_from`). The scope
said three fields; the attribution request makes it four. `gate_in` stays
input slot 32 and link 279 keeps that destination; no link is retargeted.
Every test that pins 33 moves to 37 with the reason in the comment.

### 2.2 The raw bundle and its policy (`nodes/_otr_story_input.py`, pure)

```
RawStoryFields(idea, characters, plot, setting, author)      # exact strings
StoryInputPolicy(mode: "legacy"|"user_fields_v1", bank_id)     # from the bank row
StoryRequest(num_characters_raw, act_count, include_act_breaks,
             source_bank_requested, visual_style_requested)   # captured before rolls
StoryInputBundle(schema_version="my_story_input_v1", fields, normalized,
                 request, digest)                              # digest = sha256 of
                                                               # canonical JSON of
                                                               # fields + request
```

Functions (no I/O, no imports of the writer):

* `capture_raw(**widget_values) -> RawStoryFields` -- strings as received;
  non-string values are a `StoryInputError` naming the field (the validator
  passes only literals; links are handled separately, see 2.4).
* `dedicated_fields_present(raw) -> bool` -- any of characters/plot/setting/
  author non-blank after strip.
* `creative_input_present(raw) -> bool` -- any of idea/characters/plot/setting
  non-blank. `author` alone is NOT an idea.
* `check_selection(raw, policy, *, source_bank_widget, source_ref, replay_from,
  snapshot_manifest_configured)` -- raises `StoryInputError` with guidance for
  each of: dedicated fields on a non-My-Story selection (including the roll
  sentinel); My Story with `replay_from` non-blank; My Story with a
  source-snapshot manifest configured; My Story with non-blank `source_ref`;
  My Story with no creative input. Returns nothing on success. This ONE
  function is called at both check sites (2.3) so the two cannot disagree.
  `snapshot_manifest_configured` is a bool supplied by the caller from a NEW
  non-raising probe `_otr_source_snapshot.manifest_configured() -> bool`
  (reads the env var only; review fold: `load_snapshot_for_bank` RAISES
  `SourceSnapshotError` for a bank absent from a configured manifest, so it
  cannot be the detector).
* `build_bundle(raw, request) -> StoryInputBundle` -- deterministic; timestamps
  are NOT part of the digest.
* `project_payload(bundle, today) -> dict` -- the seven-string source payload:
  `headline` = "My Story: " + first 80 chars of the idea (or of the first
  non-blank creative field), `summary` = "", `full_text` = the labelled
  projection `IDEA:\n...\n\nCHARACTERS:\n...\n\nPLOT:\n...\n\nSETTING:\n...\n\nBY:\n...`
  with blank sections omitted, `source` = "My Story (listener idea)",
  `date` = today, `link` = "", `seed_text` = the normalized idea or the first
  non-blank creative field. The bundle rides beside it in `source_meta`; the
  projection never replaces the raw evidence.

### 2.3 Writer `run()` -- two check sites, one contract

1. **FIRST statement of `run()`, before the replay shortcut, before the
   rolls:** capture `raw` and `request` from the widget arguments (no I/O;
   `request.source_bank_requested` and `request.visual_style_requested` are
   the LITERAL widget values, which may be the roll sentinels), resolve
   `policy` with `find_bank(source_bank)` (the roll sentinel and an unknown
   id give `None`, hence `legacy`; an unknown id still fails later at
   `require_runnable_bank` exactly as today), and call `check_selection(...)`.
   This is where "dedicated fields on the wrong bank", "My Story plus
   replay_from" and "My Story plus snapshot manifest" fail, cheaply, before
   anything is imported or rolled.
2. **Immediately after `_source_bank_row` is bound and before the LLM
   preflight, the scaffold env mutation, budget resets and
   `_resolve_inputs`:** if `story_input_mode(_source_bank_row) ==
   "user_fields_v1"`, call `check_selection(...)` again with the bound row
   (identical answer by construction), build the bundle FROM THE SAME `raw`
   AND `request` OBJECTS CAPTURED AT SITE 1 -- never from the run()-locals
   `source_bank` / `visual_style`, which the two rolls have already rebound
   (review fold: the style roll runs before the row is bound, so a
   recomputed request would change the digest between validator and writer)
   -- and call the draft coordinator (2.5) to persist or verify it. Stamp
   nothing yet; the ledger does not exist. A test rolls `visual_style` with
   My Story selected and asserts the writer's digest equals the validator's.

`_resolve_inputs` gains keyword-only `story_characters=""`, `story_plot=""`,
`story_setting=""`, `story_author=""`, `story_request=None` after the
existing `*` (`story_request` is the pre-roll `StoryRequest` from `run()`;
a direct call without it builds one from the arguments it was given). The
user-fields branch is decided BEFORE the snapshot load, and the snapshot load
itself is SKIPPED on that branch (review fold: `load_snapshot_for_bank` is
called unconditionally today and raises for a bank absent from a configured
manifest):

```
_rb_bank = _otr_story_routing.get_bank(source_bank or "scifi_news_pro")
_user_fields = _otr_story_routing.story_input_mode(_rb_bank) == "user_fields_v1"
_source_snapshot = (None if _user_fields
                    else _otr_source_snapshot.load_snapshot_for_bank(...))
source_document = None
if _user_fields:
    bundle = _otr_story_input.build_bundle(raw, request)   # same pure code as run()
    news_article = _otr_source_payload.validate_source_payload(
        _otr_story_input.project_payload(bundle, today),
        origin="_resolve_inputs my_story")
    news_seed = news_article["seed_text"]; seed_source = "my_story_fields"
    source_meta = {"kind": "user_story", "story_input": bundle.as_dict(),
                   "draft_digest": bundle.digest,
                   "requested_num_characters": <raw int before the clamp>,
                   "story_author": bundle.normalized.author}
    source_rights = {"license_label": "listener original idea"}
elif _source_snapshot is not None: ...            # unchanged
elif _bank_has_no_source_contract(_rb_bank): ...  # unchanged (original lane)
```

`_bank_has_no_source_contract` is NOT changed; the new branch simply runs
first, so My Story can never inherit the spark draw. A direct
`_resolve_inputs` call with dedicated fields on a legacy bank raises the same
`StoryInputError` (defensive; `run()` already refused).

`source_rights` carries a label only. No licence claim is invented; with
`provenance_normalize` absent the eligibility receipt records
`rights_not_stamped`, which is informational, so an original listener story
is publishable by the existing rule.

### 2.4 Validator (`nodes/_otr_workflow_validator.py`) -- before assets, both branches

`validate()` gains, before `ensure_prompt_visual_assets` in BOTH the
`validate_anyway=False` branch and the normal branch, a call to
`_otr_story_input.check_queued_prompt(prompt, unique_id, persist=...)`:

* Find every `OTR_LedgerScriptWriter` node in the queued `prompt` whose
  `gate_in` input is a link to THIS validator (`[unique_id, 0]`). Unrelated
  writers are ignored. Multiple dependent writers are each checked.
* For each such writer, read `source_bank`, `custom_premise`,
  `story_characters`, `story_plot`, `story_setting`, `story_author`,
  `source_ref`, `replay_from`, `num_characters`, `act_count`,
  `include_act_breaks`, `visual_style` from `prompt[node]["inputs"]`.
  A value that is a real Comfy link (`[node_id, slot]`) marks that field
  DEFERRED to the writer's evaluated check. A missing new field is the empty
  string (an older saved graph), never a link. Any other non-string on a
  STRING field is a `StoryInputError`.
* If `source_bank` is a literal that resolves to `user_fields_v1`, or any
  dedicated field is a non-blank literal, run `check_selection` on the
  literal values (deferred fields count as blank for the mismatch test and
  as "unknown" for the blank test: a bundle whose only creative field is
  linked is deferred, not refused). On failure RAISE before any download.
* On a literal My Story admission, persist the draft through the coordinator
  with `(prompt_id, writer node id)` from the execution context; the writer
  later verifies the same digest. `validate_anyway=False` still admits and
  persists: the flag skips the contract audit, not the listener's input.

The no-download promise is stated for the canonical/App literal-input route;
a linked creative field defers to the writer, which still fails before any
model work.

### 2.5 Drafts (`nodes/_otr_story_drafts.py`, the ONE storage owner)

* Path: `otr_state_dir() / "story_drafts" / <digest> / "input.json"`, which
  resolves under `<output>/otr/episodes/_shared/state/story_drafts/`. No new
  top-level output entry.
* `ensure_draft(bundle, *, context=None, caller=None) -> DraftReceipt(path,
  digest, status in {"created", "verified"})`:
  * identity: explicit `context` (a `prompt_id`/`node_id` pair) wins; else
    `comfy_execution.utils.get_executing_context()` imported LAZILY inside the
    function; else `caller` (a test or script name) is required, and a call
    with none of the three raises `StoryDraftError` -- no invented identity.
  * write: `json.dumps(sort_keys=True, ensure_ascii=True)` to a sibling
    temp file, `os.replace` into place (same directory, atomic). The file
    carries the bundle plus `created_at` (UTC) and `submitted_by`
    (`{"prompt_id", "node_id", "caller"}`), which are OUTSIDE the digest.
  * exists: re-read, recompute the digest from the stored fields + request,
    compare; equal -> "verified"; different -> `StoryDraftError` (a corrupted
    or hand-edited draft is never silently reused).
  * any OSError -> `StoryDraftError`; the caller aborts generation. "Saved"
    is never claimed on failure.
* The writer stamps `meta["story_draft"] = {"schema_version",
  "digest", "path", "status"}` at D.1 when the bank is My Story.
* Never called from `IS_CHANGED`. Cancellation after admission leaves the
  file in place (it was written before generation began).

## 3. The pass graph (independent design)

All LLM passes go through `_otr_structured_call.structured_call` (schema,
retry ladder, typed repair) with the slot fns the writer hands the runner.
Per-pass budgets are `max_new_tokens`; the transport's own
`fit_output_tokens` decides whether the measured prompt leaves room. A
`prompt_no_room` failure is TERMINAL and is re-raised as
`MyStoryInputTooLongError` naming the longest field and the measured
prompt/context numbers. Nothing is truncated; no fixed character quota is
invented.

### P0 `interpret` (technical slot, structured, max_new_tokens 1200)

Input: the labelled projection of the bundle, `num_characters` (raw request),
`act_count`, `include_act_breaks`.

```
StoryInterpretation
  requirements: list[Requirement{id, text, kind: cast|relationship|setting|
                                 outcome|event|tone|other, source_field,
                                 strength: required|preferred}]
  named_cast:   list[NamedCast{name, notes, stated_gender: ""|male|female,
                               speaking: bool, required: bool}]
  cast_plan:    CastPlan{requested: int, planned: int, exclusive: bool,
                         reason: str}
  setting_brief: str
  assumptions:   list[str]         # material gaps the story will fill
  conflicts:     list[Conflict{requirement_id, why, resolution}]
```

Rules in the seam: distinguish requirements from incidental mentions,
examples, typos and brainstorming alternatives; never infer gender from a
name (`stated_gender` only when the text states it); `exclusive` is true only
when the listener says only these people; record assumptions; a constraint
that cannot be honoured is recorded in `conflicts` with the resolution the
story will take, never dropped silently.

Post-validator (Python): every `required` named cast member with
`speaking=true` is unique by name; `planned` = `len(required speaking)` when
`exclusive`, else `max(len(required speaking), requested)`; `planned` must be
>= 1 and <= the open voice pool size (`config.cast_pools.open_voice_pool(set())`
length) -- a plan that exceeds the voice stock is a terminal
`MyStoryCastError` BEFORE any creative call, with the number in the message.

### P1 `treatment` (creative slot, structured, max_new_tokens 1800)

Input: the interpretation (JSON), the raw projection, `act_count`,
`include_act_breaks`.

```
StoryTreatment
  title: str
  logline: str
  dramatic_question: str
  setting: str
  time_of_day: str
  cast: list[CastMember{name, role, character_description,
                        gender: male|female, age_band, register, timbre}]
  acts: list[ActPlan{n, purpose, scene_setting, turns: list[str], ending_state}]
  ending: str
```

Post-validator: `len(cast) == cast_plan.planned`; every required speaking
name is present verbatim; when the listener stated a gender it is honoured;
`len(acts) == act_count`; act numbers 1..N in order. `gender` is canonicalised
through `_otr_roster_gender.canonical_bank_gender` before the Literal check
(shared vocabulary, not a new map).

### P2 `acts` (creative slot, structured, ONE CALL PER ACT, max_new_tokens 1600)

Input per call: the treatment (JSON), the act plan for THIS act, a bounded
digest of the previous act (its last four lines and its `ending_state`), the
cast names, and the target shape (a single scene per act).

```
ActScript
  n: int
  scene_setting: str
  lines: list[Line{speaker, text}]        # nonempty; 6..40 is prompt guidance only
```

Post-validator: `n` matches; every `speaker` is a treatment cast name
(exact); at least two distinct speakers unless the treatment has one cast
member; no empty text; consecutive same-speaker lines are allowed (merged at
assembly). A failed act retries its own call only; earlier accepted acts are
kept.

Cast coverage (review fold: the freeze's Phase 10 hard-fails any cast row
with zero lines, the early repair pass is inline-only, and the sci-fi lane
closes the same gap with its own equality gate): every act call receives the
list of cast members NOT YET HEARD; for the LAST act that list is a
post-validator REQUIREMENT (every unheard member must speak), so the ladder
and the typed repair run before anything is assembled. After the last act is
accepted, a pure check asserts the union of speakers across all acts equals
the treatment cast (minus the announcer); a violation is a terminal
`MyStoryCastError` naming the silent character, raised before P3/P4/P5.

### P3 `frame` (creative slot, structured, max_new_tokens 700)

Input: title, logline, the attribution sentence (section 6), whether inter-act
music cues are wanted (`include_act_breaks and act_count > 1`).

```
StoryFrame
  announcer_intro: list[str]   # nonempty; 1..3 is prompt guidance only
  announcer_outro: list[str]   # nonempty; 1..2 is prompt guidance only
  coda: str                    # one closing line
  music_open: str              # cue description
  music_close: str
  music_inter: list[str]       # exactly act_count-1 cues when wanted, else []
```

Post-validator: attribution sentence present verbatim in intro or outro when
supplied (mechanically provable); `music_inter` length as required.

### P4 `voices` (pure Python)

`assign_voices(treatment.cast, rng) -> list[cast rows]`:
announcer row c01 from `config.cast_pools.pick_announcer(rng)`; characters
c02.. in treatment order; for each, pick from
`config.cast_pools.open_voice_pool(taken)` with the shared
`_otr_casting.python_assign_voice_preset` (gender + timbre + age_band, seeded
rng), then `_otr_casting._assert_unique_bark_voices(rows)`. `tts_model`
"bark" for characters as the other content-owned lane records (the
per-machine voice engine is resolved downstream by the cast lock and the
profile, exactly as it is for the sci-fi lane). No LLM chooses a voice.

Cameo: My Story does NOT run the house cameo roll. The listener's cast is
the authority, and a cameo the listener did not ask for would override it.
The contract is stamped with `content_owned_cast_contract(decision=None)`,
which records `lemmy_policy = "content_owned_cast_no_cameo_roll"` and
`lemmy_hit = False`. A non-default `lemmy_cameo` widget on this bank is
logged once as not applicable and recorded at
`meta["my_story"]["lemmy_knob_ignored"]`, never silently dropped.
`tests/test_cast_lock_policy_repin.py::BANK_CAMEO_POLICY` gains
`"my_story": "no_cameo_roll"` -- a FOURTH label added to that test file's
own closed vocabulary in the same change (`known` set and a new
`_NO_CAMEO_ROLL_POLICIES` bucket whose cross-check is
`_source_bank_excludes_lemmy(bank) is False`), because the existing three
labels answer a different question (may Lemmy appear) and none is honest for
a lane that never rolls (review fold: the design first wrote a mis-spelt
ledger string into that map, which would have failed
`test_every_policy_value_is_one_we_defined`). The ledger constant
`_otr_casting.CONTENT_OWNED_NO_CAMEO_ROLL == "content_owned_cast_no_cameo_roll"`
is asserted in the My Story runner test, where it belongs.

### P5 `assemble` (pure Python)

The ledger row contract (shared with every lane; this is the production
schema, not a copied pass):

* cast rows via `led.set_cast` (c01 announcer, c02.. characters).
* `shot_000` preamble: music_open sentinel line (`line_id "shot_000_music"`,
  `char_id == speaker_role == "music_open"`, text ""), then announcer intro
  rows (`char_id "announcer"`, `speaker_role "announcer"`, `boundary
  "shot_start"` then `"beat_start"`).
* one scene + one shot per act: `s01`/`shot_001` ...; consecutive same-speaker
  lines merged into one row; rows `char_id cNN`, `speaker_role "character"`,
  first row `boundary "shot_start"`, later `"beat_start"`; one beat per row
  with `line_ids [line_id]`.
* after act k < N when inter cues are wanted: `music_inter` sentinel line +
  music row `placement "interstitial"`, `cue_id "inter_NN"`, `anchor_line_id`.
* postamble shot: announcer outro rows, the coda row, then the music_close
  sentinel + closing music row (`placement "closing"`). Opening music row
  `placement "opening"` anchors the opening sentinel.
* `led.set_scenes/set_shots/set_beats/set_lines/set_music`, incremental
  `led.save()` after the preamble and after each act.
* `_otr_content_authorship.stamp_receipt(led.data, owner_bank="my_story",
  accepted_artifacts={"interpretation": interpretation.model_dump(),
  "treatment": treatment.model_dump(), "acts": [a.model_dump() for a in acts],
  "frame": frame.model_dump()})` -- a MAPPING of artifact id to the accepted
  object (review fold: the first draft wrote a set literal, which has no
  `.items()`). Every voiced row's text is a verbatim constituent of an
  accepted artifact (the merge joins lines with one space, the same
  normalisation the receipt hashes). This is what the read-only freeze
  verifies.
* `_otr_readiness.stamp_text_for_tts_delivery(led)`.
* `_otr_word_delivery.stamp_contract(meta, owner="my_story")` at runner
  entry and `stamp_actual(led.data, stage="my_story_assembled")` after
  assembly.
* `meta["cast_contract"] = content_owned_cast_contract(...)` with
  `num_characters_request` = the RAW request (from `source_meta`) and
  `num_characters_locked` = the locked count; `meta["my_story"]` receipt:
  `{"schema_version": "my_story_v1", "seed", "interpretation", "treatment",
  "acts_accepted": N, "frame", "cast_plan", "attribution", "pass_receipts",
  "delivery_telemetry"}`.
* `_require_ledger_save(led, what)` after the receipts -- My Story's OWN
  small guard in `_otr_my_story.py` raising `MyStoryError` (review fold:
  the only existing `require_ledger_save` is private to the sci-fi module and
  its exception prefixes every message with `[scifi_news_pro]`).

Return `MyStoryTailParts(outline_view=MyStoryOutlineView(premise=
dramatic_question, title=title, setting=setting), canon=
episode_canon_from_outline_dict({title, premise, setting, time_of_day,
sound_palette: []}), final_title_override=title, run_story_spine=False,
tail_finalizer=None)`. The writer builds `WriterTailContext`; the tail keeps
its precedence (typed `episode_title` > override > regen).

### Retry and failure boundaries

| pass | retry | terminal |
|---|---|---|
| interpret | structured ladder (3 attempts, typed repair) | ladder exhausted; cast plan exceeds voice stock; prompt no room |
| treatment | ladder | ladder exhausted; prompt no room |
| act k | ladder for act k only | ladder exhausted for act k; prompt no room |
| frame | ladder | ladder exhausted (attribution missing after repair); prompt no room |
| voices/assemble | none (deterministic) | invariant violation (a Python bug, raised loud) |

No pass falls back to another pipeline. `output_limit` and
`decode_degeneracy` re-roll inside the ladder as today; `prompt_no_room`
never re-rolls.

## 4. Sizes and budgets

* Story size is `act_count` 1..6 only. Each act is one scene and one call.
* Prompt sizing is real arithmetic at the transport (`fit_output_tokens`
  with the resolved context: HF tokenizer length, GGUF `gguf_n_ctx`, or the
  provider estimate). The runner adds NO second quota. A no-room refusal
  surfaces the measured numbers and the longest field so the listener can
  shorten it.
* Reference per-pass ceilings (`max_new_tokens`): interpret 1200, treatment
  1800, act 1600, frame 700. These are output ceilings, never targets;
  actual words are telemetry (`word_budget.policy = actual_count_only`).

## 5. Character count and act breaks

* `num_characters` stays a REQUEST. Raw value preserved at
  `source_meta.requested_num_characters` before the legacy clamp; the clamp
  stays for every other consumer.
* Precedence: explicit named speaking cast > numeric request. `exclusive`
  cast pins the count to the names. Otherwise the LLM fills unnamed roles up
  to `max(named, requested)`. Requested and locked counts are both recorded
  in `cast_contract`; the plan's `reason` explains a difference in prose.
* Incidental names are not forced into speaking roles (`speaking=false`).
* `include_act_breaks` is honoured: true and `act_count > 1` -> one
  interstitial cue between acts; false -> none. The original lane's inline
  behaviour is not inherited.

## 6. Attribution (operator addition)

* `story_author` is optional free text. Blank means "a listener".
* The attribution sentence is authored by Python, not the model:
  `"Tonight's story is by {author}."` when supplied, else
  `"Tonight's story comes from one of our listeners."` The frame pass must
  carry it verbatim in the intro (preferred) or outro; the post-validator
  checks it mechanically.
* Printed credit: `meta["credits_source_line"]` is overridden to
  `"a story by {author}, produced by machine for this broadcast"` when
  supplied; otherwise the bank default (section 1.1) stands. The override is
  stamped AFTER the writer's existing unconditional bank-default stamp
  (`OTR_LedgerScriptWriter.py:3284-3289` at the review baseline), never at
  the D.1 anchor used for the other stamps (review fold: that block
  overwrites the key whenever the row defines a default, and My Story's row
  does). The credits roll already renders `credits_source_line` with no bank
  branch.
* `meta["story_attribution"] = {"author": <normalized or "">, "sentence":
  <the sentence>, "source": "story_author widget"}`.
* The operator's name is never a default anywhere. No source of attribution
  other than the widget exists.

## 7. Delivery intent and the terminal wire

### 7.1 Writer stamp (every new run, both lanes)

At D.1, right after `meta["source_bank"]`:

```
meta["delivery_intent"] = {
  "schema_version": "otr_delivery_intent_v1",
  "source_bank": <bank id>,
  "publication_required": <story_input_mode == user_fields_v1>,
  "draft_digest": <bundle digest or "">,
  "delivery_token": uuid4().hex,          # fresh per run
}
```

The replay shortcut returns before D.1 and stamps nothing (a replayed bundle
keeps its source ledger's meta; older bundles carry no intent).

### 7.2 Mux (`nodes/otr_master_audio_mux.py`)

* New optional forceInput `script_json` (STRING, default "") appended at
  `inputs[10]` after `foley_receipts_json`; Python default `None` for legacy
  direct calls (`mux(..., script_json=None)`).
* Semantics:
  * `None` -> legacy absent wire: today's behaviour, unchanged.
  * `""` or non-JSON or non-object -> `ValueError` (present but unusable).
  * object with no `meta.delivery_intent` -> a legacy ledger on the wire
    (replay bundles, older runs): `publication_required` treated as false,
    logged once. This is a deliberate reading of the scope's "present
    empty/malformed input is an error": an older ledger is neither.
  * intent present -> validate shape/version; malformed -> `ValueError`.
  * `meta.source_bank` resolving to a `user_fields_v1` bank with NO intent or
    with `publication_required` false -> `ValueError` (contradictory).
* Binding: when `publication_required` is true, the in-flight ledger
  (`_inflight_episode_for_stem`) must exist and its
  `meta.delivery_intent.delivery_token` and `source_bank` must equal the
  wire's; otherwise `ValueError("unrelated ledger")`. The token survives the
  `rename_episode` in `video_engine`; the pending episode id is never
  compared.
* After production: when required, `decision.publishable` must be true, the
  OBS copy must exist on disk (`os.path.isfile(obs_copy)`), and the terminal
  stamp must have written `meta.obs_final_path`; otherwise `ValueError`
  ("My Story delivery failed: ...") -- the archival final stays on disk and
  the run is a FAILED run. The stamp check is STRUCTURAL: after
  `_stamp_terminal_paths(...)` (which stays best-effort and never raises for
  the other banks), the mux re-loads the in-flight ledger from disk
  (`in_flight_ledger_path` + `load_ledger_safe`) and compares
  `meta["obs_final_path"]` to the published path itself (review fold: the
  helper's return value is prose, not a status). Existing banks keep the
  withheld-but-successful branch.
* `IS_CHANGED` adds `sha256(script_json or "")` and, when the wire requires
  publication, a fingerprint of the published file the in-flight ledger names
  (`path|size|mtime_ns`, or `"missing"`), so a deleted or moved OBS file
  re-executes the node. Cache checks stay read-only.
* `RETURN_TYPES` stay `("STRING", "STRING")` in sprint 4. The UI envelope
  and the four tuple-unpacking test migrations are sprint 5.

### 7.3 Canonical wiring (same chunk as the code)

* Writer node 1: `widgets_values` += `["", "", "", ""]`; `inputs` +=
  four widget descriptors (`story_characters`, `story_plot`,
  `story_setting`, `story_author`) after `replay_from`.
* Mux node 85: `inputs` += `{"name": "script_json", "type": "STRING",
  "link": 291}` at index 10; writer output slot 1 (`script_json`) `links`
  becomes `[230, 291]`; new link `[291, 1, 1, 85, 10, "STRING"]`;
  `last_link_id` 290 -> 291. Inputs 8 (link 287) and 9 (link 288) are
  untouched.
* Then `OTR_WorkflowValidator` direct run, JSON round-trip,
  `tests/test_widget_value_alignment.py`,
  `tests/test_canonical_widget_input_parity.py`,
  `tests/test_workflow_link_target_indexes.py`, `scripts/build_variants.py
  --check` (no committed variants: soft-skip is expected and is said so).

## 8. Field ownership and the final text boundary

| field | owner |
|---|---|
| raw fields, bundle, digest | `_otr_story_input` (pure) via the writer's capture |
| draft file | `_otr_story_drafts` only |
| payload projection, `source_meta`, `source_rights` | `_resolve_inputs` (new branch) |
| `meta.delivery_intent`, `meta.story_draft`, `meta.story_attribution` | writer D.1 stamps (right after `meta.source_bank`) |
| `credits_source_line` attribution override | writer, AFTER the bank-default credits stamp (section 6) |
| interpretation / treatment / acts / frame artifacts, `meta.my_story` | the runner |
| cast rows, voice presets | runner P4 |
| scenes/shots/beats/lines/music, authorship receipt, TTS delivery text, word delivery, cast contract | runner P5 |
| title binding, canon write, visual plan, story-brief reflection, freeze-side stamps | the shared tail (unchanged) |
| publication eligibility receipt | the freeze (unchanged, sole writer) |
| OBS path stamps, publication, delivery enforcement | the mux |

Final text: the accepted `ActScript` lines and the frame's announcer lines
ARE the spoken text; assembly merges same-speaker runs with a single space
and stamps the authorship receipt over the accepted artifacts. Two different
mutation windows exist downstream and only one is skipped: the FREEZE's
inline safety cleanup does not run on this lane (`content_owned_readonly`),
but the writer TAIL's `run_ledger_clean` / `run_ledger_cleanup` window runs
on EVERY lane, wrapped in `_otr_clean_transaction` which re-seals a proven
rewrite as an authorized transition or rolls the ledger back to the accepted
text (review fold). My Story therefore inherits exactly the sci-fi lane's
exposure: its text can be touched inside that window, and the protection is
reseal-or-rollback, not a skip. Making a listener's text untouchable end to
end would be a change to shared tail code and is NOT part of this design.
The `text_for_tts` projection is a separate field stamped from the sealed
text. The hashes recorded by `stamp_actual` and the authorship receipt are
what the freeze re-verifies.

## 9. Independence comparison (six dimensions)

| dimension | original lane | scifi_news_pro | My Story (this design) |
|---|---|---|---|
| source / input authority | random spark deck + optional operator hint | fetched science article | the listener's five fields; nothing random or fetched |
| pass graph | concept x3 -> select -> compat briefs -> INLINE outline/beat/line composer | dossier -> pitch -> treatment -> news read -> whole-play markup -> casting LLM -> assemble | interpret -> treatment -> one structured call PER ACT -> frame -> Python voices -> assemble |
| slot assignment | creative: concept, select; technical: briefs; then inline | technical: dossier, news read, casting; creative: pitch, treatment, script | technical: interpret; creative: treatment, acts, frame; Python: voices, assemble |
| artifacts | ConceptPitches, SelectedConcept, OriginalBriefsModel (interpreter-compat shape) | Dossier, Pitch, Treatment, FinalDraft (markup), CastingVoices | StoryInterpretation, StoryTreatment, ActScript[N], StoryFrame, cast rows (no LLM casting artifact) |
| retry boundaries | ladder per pass; inline composer per line | ladder per pass; markup ladder over the whole play | ladder per pass; an act retries alone; frame attribution is a post-validator |
| ledger-write strategy | writer inline: skeleton lines then per-line updates | runner assembles all five hierarchies after the whole play parses | runner assembles preamble, then per act incrementally, then postamble; authorship receipt over four typed artifacts |

The interfaces shared on purpose: the runner keyword signature, the tail
parts duck type, `structured_call`, the ledger row contract, the content
authorship receipt, `content_owned_cast_contract`, the voice pool and preset
picker, `stamp_text_for_tts_delivery`, word delivery, the eligibility receipt.

## 10. Tests (sprint 4, all offline)

New:

* `tests/test_my_story_input.py` -- capture, blank/whitespace bundles, each
  field alone, combinations, exact raw preservation, digest determinism and
  timestamp independence, `check_selection` on every refusal (wrong bank,
  sentinel, replay_from, snapshot manifest, source_ref, no creative input,
  author-only), payload projection, non-string values.
* `tests/test_my_story_drafts.py` -- create, verify, mismatch, OSError,
  explicit context vs lazy context vs caller, no identity refuses, path under
  the state dir, atomicity (temp file gone), idempotent repeat.
* `tests/test_my_story_validator.py` -- queued prompt with a literal My
  Story writer (admits, persists, no asset call), blank (raises before
  assets), wrong-bank dedicated fields (raises), linked creative field
  (deferred, no raise), unrelated writer ignored, two dependent writers,
  `validate_anyway=False` still admits.
* `tests/test_my_story_runner.py` -- fake slot fns returning canned JSON:
  pass order and slots, requirement retention (named cast and constraints
  survive into treatment/acts), cast plan precedence, voice-stock terminal
  error, per-act retry isolation, attribution sentence required, act-break
  cues, assembly row contract (roles, boundaries, sentinels, music rows),
  authorship receipt validates, `stamp_actual` present, cast contract with
  `decision=None`, read-only freeze structural validation passes on the
  assembled ledger, tail parts shape.
* `tests/test_my_story_delivery.py` -- writer intent stamp on every bank,
  mux: None/""/non-JSON/legacy ledger/malformed intent/contradictory intent,
  token binding across a renamed episode, required publication withheld ->
  raise, deleted OBS file -> IS_CHANGED differs, unchanged rerun stable.
* `tests/test_my_story_registry.py` -- bank/pipeline rows validate,
  `story_input_mode`/`auto_select` parsing and rejection, roll pool excludes
  My Story, manual selection runnable, lane spec resolves lazily.

Updated: the 33 -> 37 pins (`test_source_bank_widget_2c.py` x2,
`test_source_ref_widget.py`, `test_otr_api_companions.py`,
`test_visual_style_widget_3c.py`, `test_workflow_json_guardrails.py` with
its count-history docstring); roster lists (`test_bank_variants.py`
LIVE_BANKS and its exact-tuple assertion, `test_user_bank_admission.py`,
`test_user_bank_bundles.py`, `test_production_ledger.py`,
`test_provenance_v4.py`, `test_scene_guard_v4.py`,
`test_bank_scalar_defaults.py`, `test_freeze_policy_readonly.py`
content-owned list, `test_lane_specs.py`, `test_cast_lock_policy_repin.py`
with the fourth label); `test_workflow_link_target_indexes.py` runs
unchanged against the new link. Tooltips updated in the same change: the
validator's `validate_anyway` tooltip and module docstring say the listener
input admission always runs; the writer's `custom_premise` tooltip names the
My Story meaning.

## 11. Sprint 5 contract (recorded here, built later)

* App layout saved on the canonical (`extra.linearData` for node 1 controls
  in the order: source_bank, custom_premise, num_characters, act_count,
  story_characters, story_plot, story_setting, story_author, visual_style;
  `extra.linearMode` false so graph stays the initial view). Native rename
  changes the shared widget label; descriptions carry the My Story-only
  guidance.
* Mux returns `{"ui": {"gifs": [{filename, subfolder, type: "output",
  format: "video/mp4"}], "files": [<episode report .txt>]}, "result":
  (final, report_text)}`; the `.mp4` descriptor is the OBS AAC copy resolved
  against `folder_paths.get_output_directory()`; an unservable OBS path
  emits no descriptor and the report says so. The four success unpackers
  migrate atomically.
* Live proof: one-act, then three-act, then six-act My Story canonical runs
  on the 5080 with the file in `otr/obs/`; browser proof of App/Graph
  switching, saved parameters, playback, history; README instructions
  written from the proven flow.

## 12a. Review fold (2026-09-10)

One finished-design review, six independent Sonnet 5 readers with distinct
lenses (r1 arc x2, r2 coding x2, r3 wiring x2), each grounded on the live
Windows files; the driver verified every claim against the same files.
Roster stated exactly: no Codex, no agy, no Cursor lane ran; the scope
campaign's own R1-R4 (Gemini + Cursor) preceded this design. 19 findings,
0 refuted, all folded above and marked "review fold":

| # | severity | disposition |
|---|---|---|
| credits override clobbered by the bank-default stamp | BLOCKER | folded (section 6, section 8) |
| cameo test-map label breaks the closed vocabulary; mis-spelt constant | BLOCKER + 3 MAJOR | folded (section 3 P4): fourth label in the test, ledger constant asserted in the runner test |
| `get_bank` raises on the roll sentinel | BLOCKER | folded: `find_bank` non-raising lookup (1.3, 2.3) |
| digest recomputed after the style roll | BLOCKER | folded: same pre-roll capture threaded to both sites, test added (2.3) |
| `accepted_artifacts` written as a set | BLOCKER | folded (P5) |
| tail clean window runs on every lane | MAJOR | folded: honest exposure note (section 8), no shared-code change |
| no cast-coverage gate before the freeze | MAJOR | folded: unheard-cast requirement on the last act plus a union gate (P2) |
| `validate_source_payload` missing `origin` | MAJOR + MINOR | folded (2.3) |
| snapshot detector would raise | MAJOR | folded: `manifest_configured()` probe and the load skipped on the user-fields branch (2.2, 2.3) |
| `require_ledger_save` is sci-fi-private | MAJOR | folded: own guard (P5) |
| three test files missing from the inventory | 2 MAJOR | folded (section 10) |
| mux stamp check has no structural signal | MAJOR | folded: re-load and compare (7.2) |
| `validate_anyway` tooltip would lie | MINOR | folded (section 10) |

## 12. Documentation deliverables

* D0 currentization (in flight): README story-source and preflight sections,
  `docs/EXTENDING_OTR.md`, `docs/SOURCE_BANK_GUIDE.md`,
  `docs/SOURCE_BANK_PREFLIGHT.md`, `docs/PRODUCTION_SPRINT_LESSONS.md`
  supersession notes, `docs/SOAK_LEG_GUIDE.md` sections 3-5, `docs/README.md`.
* New `docs/MY_STORY_GUIDE.md` (user-facing: what to type, what the controls
  mean, attribution, what "size" means, where the draft and the movie land,
  limits) and a My Story block in `docs/SOURCE_BANK_PREFLIGHT.md` (input
  authority, manual-only availability, draft persistence, delivery
  enforcement, App round-trip, player).
