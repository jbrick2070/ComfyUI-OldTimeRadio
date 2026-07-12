# dynamic_story -- Story-Derived Visual Direction: Scoped Design (docs-only)

Date: 2026-07-12 (rev 4 -- r1 arc + r2 coding plan + r3 wiring, each hardened by
a kibitz round: codex gpt-5.6-sol @ ultra + antigravity gemini-3.5-pro; driver
anchors, panel reviews and judgments in
kibitz-runs/2026-07-12-dynamic-story-visual/{r1,r2,r3}/)
Author: Claude (Cowork), docs-only architecture owner for this feature.
Implementation owner: Codex (all code, tests, registries, prompts, workflow JSON).
Status: DESIGN ONLY. Nothing in this document is implemented. No code, test,
prompt, registry, or workflows/otr_canonical.json change accompanies it.

Scope contract honored here:

- One additional option (`dynamic_story`) in the EXISTING visual-style dropdown.
- When selected, an LLM reads the final accepted/frozen story and authors an
  episode-specific, evidence-bound `visual_direction` artifact that every
  downstream visual composer consumes.
- An explicit user-selected named visual pack keeps current behavior,
  byte-identical, and always wins. `dynamic_story` is opt-in per episode.

Every code claim below is grounded `file:line` against the real Windows repo
(v2.0-alpha working tree, 2026-07-12). This design is GRADED against
docs/PRODUCTION_SPRINT_LESSONS.md; the lesson each requirement discharges is
named inline.

---

## 1. Grounded current dataflow (what exists today)

### 1.1 The visual-style selector and its one stamping point

- The dropdown lives on OTR_LedgerScriptWriter: choices come LIVE from the
  visual-style pack registry (`list_style_ids()`) at INPUT_TYPES
  (nodes/OTR_LedgerScriptWriter.py:2871-2889), default `sci_fi_radio`.
- The id is gated fail-loud at the top of `run()` beside the source-bank gate:
  `_otr_visual_styles.resolve_visual_style(visual_style)` -- unknown id raises
  `UnknownVisualStyleError` before ANY story work
  (nodes/OTR_LedgerScriptWriter.py:3334-3339).
- The ONE authoritative stamp: `meta["visual_style"] = resolved["visual_style"]`
  (nodes/OTR_LedgerScriptWriter.py:3651-3655). The comment there is the
  threading law: "every downstream visual composer reads meta['visual_style']
  via get_visual_style(meta) off the serialized ledger".
- Channel separation is explicit and load-bearing: `meta.style` is the STORY
  grammar slug, `meta.visual_style` is the visual prompt-pack selector; the two
  are never crossed (nodes/OTR_LedgerScriptWriter.py:216-222, 2249-2261).
- The dropdown surface is TEST-PINNED to the registry:
  `test_choices_are_exactly_the_registry` asserts
  `choices == list(vs.list_style_ids())`
  (tests/test_visual_style_widget_3c.py:62-66), and `visual_style` is
  positionally pinned at INPUT_TYPES index 24 (:54-60) and at
  `widgets_values[24] == "sci_fi_radio"` in the canonical workflow.

### 1.2 The pack registry (the resolution seam dynamic_story must enter through)

- Packs are JSON files under `nodes/visual_styles/<style_id>.json`; the module
  is pure behavior, lazy, fail-loud (nodes/_otr_visual_styles.py:3-13, 50).
- `DEFAULT_STYLE_ID = "sci_fi_radio"` (nodes/_otr_visual_styles.py:53).
- Schema v2 is a FIXED field inventory: v1 fields (style_id, label, the four
  tails, forbidden_terms, ...) + 11 required look/subject strings + 4 dict
  fields with exact key sets (open_subjects, motion_registers,
  still_word_typography, still_word_backdrop)
  (nodes/_otr_visual_styles.py:60-102). Validation is strict: template
  placeholder rules, mouth-vocabulary contract, 240-char motion budget,
  forbidden-terms lint (nodes/_otr_visual_styles.py:109-123, 184+).
- Resolution: `resolve_visual_style(style_id)` -- unknown id = hard error, no
  fallback (nodes/_otr_visual_styles.py:367-375); `get_visual_style(meta)` --
  absent/empty `meta["visual_style"]` = production default; PRESENT-but-unknown
  fails LOUD (nodes/_otr_visual_styles.py:378-390). Both are META-ONLY seams --
  they never see `lines`/`cast` (r1 finding; drives 5.3).
- The loaded packs are CACHED module-globally (`_STYLES`,
  nodes/_otr_visual_styles.py:170, 355-359). A per-episode dynamic pack must
  NEVER enter that cache.
- The geometry-vs-look law (nodes/_otr_visual_styles.py:7-9): packs own ONLY
  look/subject vocabulary; framing/headroom/mouth-safety GEOMETRY stays in
  Python. `dynamic_story` obeys the same law -- and the LLM does not even author
  every pack LOOK field (2.4).

### 1.3 The story freeze boundary (what "final accepted story" means)

- OTR_LedgerScriptWriter -> OTR_LedgerFreezeCascade; Phase 10 runs the
  deterministic gap audit and, on success, stamps `meta.cleanup_locked = True`,
  `meta.freeze_timestamp` (ISO-8601 UTC), and `meta.freeze_verdict =
  frozen_clean | frozen_with_warns` (nodes/_otr_ledger_freeze.py:758-819,
  stamps at 806-811). Critical gaps raise `FreezeAssertionError` and stamp
  `needs_full_rerun` (nodes/_otr_ledger_freeze.py:787-795).
- Freeze locks story CONTENT, not the whole rows. OTR_CastLock assigns voice
  presets AFTER the freeze (acknowledged inside the freeze audit itself,
  nodes/_otr_ledger_freeze.py:493-502); OTR_ShotLock overlays per-line audio
  timing IN PLACE (nodes/otr_shot_lock.py:169-221) and stamps a whole
  `ledger['video']` section + `meta.video_revision` (:1104-1131). Consequence:
  the RAW `lines`/`cast` arrays are NOT stable post-freeze -- any staleness check
  must hash an authored-content DTO, never the raw rows (2.5).
- The cascade unloads a locally-resident writer LLM in a `finally` block and
  stamps `meta.freeze_unload_ok` (nodes/OTR_LedgerFreezeCascade.py:377-387,
  453-478). That receipt is a PRECONDITION for this feature (5.2): a resident
  writer LLM plus a fresh creative LLM is an OOM on a 16 GB card.
- **There is NO top-level `beats` array.** Beats are DERIVED from lines inside
  ShotLock, with `beat_id = line_id` (nodes/otr_shot_lock.py:260-288), and the
  synthetic opening-music beat `OPENING_MUSIC_BEAT_ID = "b000_music_open"` is
  created LATER still, by `derive_opening_music_beat(ledger, fps)` (:291-301).
  Nothing keyed to a beat exists at direction time.

### 1.4 Canonical wiring (workflows/otr_canonical.json, live graph)

Read out of the real file: `last_node_id = 95`, `last_link_id = 283`, 23 nodes,
57 links, `groups: []` (empty -- no bounding box constrains placement), every
node `mode == 0` (nothing muted).

- Writer (node 1) `script_json` -> link 230 -> FreezeCascade (node 62).
  `visual_style` is combined widget slot 24 (`widgets_values[24] ==
  "sci_fi_radio"`; `inputs[24]` carries `"widget": {"name": "visual_style"}`
  with `link: null` -- the widget-as-input serialization, not a forceInput).
  The 34-entry vector is APPEND-ONLY (BUG-LOCAL-097); this feature does not
  touch it.
- **FreezeCascade (62) out[1] `script_json` fans out to SIX links:**
  `[16, 231, 232, 233, 252, 255]` --
  `[16, 62, 1, 12, 2]` -> OTR_SignalLostVideo (12);
  `[231, 62, 1, 81, 0]` -> OTR_BatchCharacterVoices (81);
  `[232, 62, 1, 82, 0]` -> OTR_AnnouncerVoice (82);
  `[233, 62, 1, 83, 0]` -> OTR_StableAudioTheme (83);
  `[252, 62, 1, 90, 0]` -> OTR_ShotLock (90);
  `[255, 62, 1, 89, 0]` -> OTR_MetaBriefImagePromptGen (89).
  Only 252 and 255 are the VISUAL-lane consumers. **16 / 231 / 232 / 233 MUST
  keep reading the RAW freeze json** -- the audio trio is pinned to it by
  tests/test_full_workflow_v2_audio_wiring.py:220-232, and node 80 OTR_CastLock
  keeps sourcing 62.out[6] (`v2_ledger_json`, link 234). The direction node is
  inserted in the VISUAL lane ONLY.
- OTR_VideoDirector (87) -> link 270 -> OTR_ImageDirector (88) in[0]; -> link
  251 -> ShotLock in[2]. ImageDirector -> link 254 -> MetaBrief in[1]; -> link
  257 -> OTR_ImageGenDispatcher (91) in[1]. MetaBrief out[0] -> link 258 ->
  Dispatcher in[2]. ShotLock out[0] -> link 256 -> Dispatcher in[0]; out[4]
  `episode_id` -> link 268 -> Dispatcher in[4]. OTR_EpisodeAssembler (7) out[3]
  `audio_done` -> link 253 -> ShotLock in[1] and -> link 259 -> Dispatcher
  gate_in. Dispatcher out[0] -> link 260 -> OTR_VideoRenderBatch (92) in[0];
  out[1] `image_done` -> link 267 -> RenderBatch in[2].
- Node 89: `widgets_values [False]`; inputs `script_json`(255),
  `image_policy_json`(254), `gate_in`(null), `consistency_gate_warn_only`(widget).
  Node 90: `widgets_values [False]`; inputs `script_json`(252),
  `audio_done`(253), `video_policy_json`(251), `image_done`(null),
  `gate_in`(null), `consistency_gate_warn_only`(widget); **its `done` output
  (out[3]) ships UNWIRED (`links: []`)** -- the precedent for the new node's own
  unwired `done`.
- Free placement rectangle for a new node: x 1060-1440, y 760-1000 is empty
  (62 is at x 620; 89 at x 1080 y 1040; 90 at x 1129 y 1200; 91 at x 1560).

The Dispatcher receives its ledger via ShotLock's whole-ledger re-serialization
(parse at nodes/otr_shot_lock.py:1040, re-emit at :1142), so a meta artifact
stamped upstream reaches it with no extra wire.

### 1.5 Where the style is actually consumed (THREE independent resolvers)

- **Image prompts (MetaBrief).** `derive_image_prompts` resolves the style ONCE
  at entry -- `_vstyle = _resolve_style(meta)` (:1601-1609) -- and threads it
  down. **Its signature takes NO ledger:** `derive_image_prompts(cast, meta, *,
  llm_fn=None, ..., lines=None, ...)` (:1570-1574), and `generate()` parses the
  ledger but passes only `cast` / `meta` / `lines` (:2137-2168). Downstream of
  the threaded pack: portrait fallback + LLM instruction builders (:1148-1238),
  beat-aware character scene builder (:1241-1366), radio-host prompts (:356-444),
  mesh fodder + background plate (:1437-1496), still_word cards (:942-1013), and
  the aspect anchors (:171-188).
- **Video prompts (ShotLock).** Every talking-head prompt is finished through
  `finish_visual_prompt(meta, text_prompt)` -- inside a BARE
  `except Exception: pass` (nodes/otr_shot_lock.py:626-636), so a style failure
  is silently swallowed today (7.2).
- **Video render (render_driver).** It resolves the style ITSELF:
  `_vstyle = get_visual_style((ledger or {}).get("meta") or {})`
  (nodes/_otr_video_engines/render_driver.py:1248) -- a THIRD resolution point
  that raises on the sentinel -- and one branch builds its prompt core from the
  BRIEF, not the pack: `core = get_story_brief_ltx(_meta)` (:2069) then
  `finish_visual_prompt(...)` (:2080).
- The helper family (`_resolve_style` :330, `get_era_tail` :344,
  `compose_still_prompt` :590, `finish_visual_prompt` :667 in
  nodes/_otr_story_brief_helpers.py) resolves the pack from meta.
- LOOK-AUTHORITY REALITY: the era tail is BRIEF-FIRST -- the brief's
  `atmosphere_line`/`visual_palette`/lighting win, and the pack's `era_tail` is
  only the fallback default (nodes/_otr_story_brief_helpers.py:356-370, 401,
  414, 428). A design that authors a new palette in the pack alone would be
  silently shadowed by the brief, in BOTH lanes. 5.4 resolves the precedence.
- ShotLock stamps per-shot `creative` sidecars with `prompt_hash` +
  `request_hash` into `ledger['video'].shots[]` (:637-647, 913-948).

### 1.6 Image dispatch, cache, and persistence (the replay substrate)

- Dispatcher cache key: `request_cache_key(role, object_id, prompt_hash, seed,
  engine_id, engine_version, kind, w, h)` -- "a change in ANY field -> new key ->
  regen" (nodes/otr_image_gen_dispatcher.py:117-129). It keys the PROMPT HASH: a
  change to rationale or a receipt that does not alter a composed prompt does NOT
  invalidate the cache.
- **A cache HIT CLONES the previous row** (`fresh = dict(ref_row or {})`) and
  overwrites only `provenance` (:627-632) -- so any provenance field the design
  adds would be INHERITED STALE on a hit unless it is set explicitly on BOTH
  paths (7.5).
- Results land in `ledger["images"]` and persist via
  `stamp_durable(sections={"images": ...})` (:796-826;
  nodes/production_ledger.py:408-422). `OTR_TEST_MODE=1` makes `stamp_durable`
  SKIP the disk write (:408-452; conftest sets it, tests/conftest.py:38).
- **A canonical re-queue is never a replay.** The writer and the cascade both
  return `time.time()` from `IS_CHANGED` ("always re-execute",
  nodes/OTR_LedgerScriptWriter.py:3023-3028;
  nodes/OTR_LedgerFreezeCascade.py:269-272), so re-queuing writes a FRESH story
  and a FRESH freeze. Replay is a test-bench property, not a live-smoke leg (9.3).

### 1.7 Ledger save/merge ownership (what survives, what gets dropped)

`Ledger.save()` merges in-memory state with the on-disk ledger
(nodes/production_ledger.py:1287-1346, 1357-1513):

- TOP-LEVEL: only `TOP_PRESERVE = (schema_version, audio_gates, transitions,
  radio_bookend_path)` is copied forward (:1387-1393). An unknown top-level key
  absent from a later in-memory save is DROPPED. Top-level is NOT ledger-safe.
- META: per-key merge -- disk wins only where in-mem lacks the key or holds an
  empty value (:1403-1413). A namespaced meta key written once SURVIVES every
  later save by construction.
- ROWS: row-level ownership. `_MERGE_OWNED_ROW_FIELDS` is a 19-field frozenset
  (text, char_count, word_count, traits, boundary, char_id, speaker_role,
  arc_phase, compose_flags, beat_intent, target_words, dialogue_slot_id, shot_id,
  beat_id, skip, tts_skip_reason, reviewer_skip_reason, reviewer_note,
  needs_render_realign + the music-cue spec fields) -- never resurrected from
  disk (:1441-1459). The merge boundary controls DISK COPY-FORWARD only; it is
  NOT a runtime mutation validator, and it is NOT the same field set as the
  direction DTO.
- The freeze audit hard-requires the fixed top-level list set and per-line shape
  (nodes/_otr_ledger_freeze.py:118-129, 252-404); it does not govern namespaced
  meta keys.
- Source-bank law: "Evidence maps and authorship receipts live in typed artifacts
  or namespaced `meta`; the fixed line schema contains no ad hoc provenance
  fields" (docs/SOURCE_BANK_PREFLIGHT.md:184-186).

---

## 2. The typed artifact: `meta.visual_direction` (schema `vd-1`)

### 2.1 TWO models, not one (Lesson 2)

A model cannot author its own `prompt_sha256`. The design has two typed models:

- **`VisualDirectionDraftV1`** / **`VisualDirectionShotsBatchV1`** -- EXACTLY
  what the LLM returns. Pydantic `BaseModel`, `extra="forbid"`, closed nested
  rows. These are the `schema=` passed to `structured_call`.
- **`VisualDirectionArtifactV1`** -- what Python ASSEMBLES and stamps: the drafts
  + the vetted safety base + the derived receipts + the seals. Never generated,
  never repaired, never LLM-touched.

`nodes/_otr_visual_direction.py` is therefore NOT "pure stdlib" -- it imports
pydantic, lazily, like every other structured pass.

**No `max_length` on any authored string (r3 codex M2 -- Lesson 3).**
`validate_tolerant_data` catches a `ValidationError`, CLAMPS over-long top-level
strings to their declared `max_length`, re-validates, and PROCEEDS with only a
`log.warning` (nodes/_otr_structured_call.py:422-434). On an authored visual
string that is a SILENT COERCION OF AUTHORED CONTENT -- precisely what Lesson 3
forbids. Because the clamp arm fires only on a `ValidationError`, declaring NO
`max_length` on authored strings disarms it completely. Every length bound is
enforced in the `post_validator` instead, which raises `PostValidationError` and
routes the failure to the TYPED REPAIR rung (4.5). No change to the shared
machinery; authored text either comes back correct or fails closed.

### 2.2 `VisualDirectionDraftV1` -- pass P-A's surface (closed set)

```json
{
  "style_language": "<one-paragraph episode visual thesis>",
  "era_cues": ["<cue>", "..."],
  "motifs": [ { "motif": "<recurring visual motif>",
                "evidence": [ {"id": "line:l0007", "quote": "<verbatim <=120 chars>"} ] } ],
  "clue_visual": { "treatment": "<how the clue mechanism reads on screen>",
                   "evidence": [ {"id": "line:l0012", "quote": "..."} ] },

  "look": {
    "label": "...", "era_tail": "...", "positive_tail": "...",
    "image_grade_tail": "...", "broadcast_tail": "...",
    "portrait_look": "...", "portrait_instruction_look": "...",
    "scene_instruction_look": "...", "radio_object_look": "...",
    "plate_look": "...", "still_word_title_mood_style": "..."
  },

  "field_evidence": {
    "look.era_tail":      { "evidence": [ {"id": "line:l0003", "quote": "..."} ],
                            "kind": "factual" },
    "look.portrait_look": { "evidence": [ {"id": "cast:c01", "quote": "..."} ],
                            "kind": "factual" },
    "style_language":     { "evidence": [ {"id": "title"} ], "kind": "rationale" },
    "era_cues":           { "evidence": [ {"id": "brief:atmosphere_line"} ],
                            "kind": "rationale" }
  }
}
```

- `look` is the CLOSED authored whitelist -- exactly those 11 keys, no more, no
  fewer. `still_word_typography` / `still_word_backdrop` are NOT authored (2.4).
- **`field_evidence` keys are a CLOSED ENUM** of literal key names: the 11
  `look.*` keys + `style_language` + `era_cues`. They are NOT free-form JSON
  pointers or schema paths (r3 codex M12 -- rev 3 forbade schema-path keys in the
  prompt while requiring them in the schema).
- **Row-shaped items carry their evidence INLINE only** (`motifs[]`,
  `clue_visual`) -- never duplicated into `field_evidence`.
- An `evidence` item is `{id, quote?}`. A `factual` entry MUST carry a `quote`
  (<= 120 chars) that is a VERBATIM SUBSTRING of the cited item's text in the
  source DTO -- machine-checked in the `post_validator`. ID existence alone does
  not prove support (r1 codex M5; r3 codex M12, light form). A `rationale` entry
  needs only the `id`.
- `shots[]` is NOT in this model -- it is pass P-B (2.3, 4.4), whose draft is
  `VisualDirectionShotsBatchV1`:
  `{ "shots": [ { "line_id": "...", "subject_note": "...", "mood": "<one register word>",
  "evidence": [ {"id": "line:<the same line_id>", "quote": "..."} ] } ] }`.
  Every row must cite its OWN line.

### 2.3 `VisualDirectionArtifactV1` -- what Python stamps

```json
{
  "schema_version": "vd-1",
  "created_utc": "2026-07-12T00:00:00+00:00",
  "writer": { "node": "OTR_DynamicStoryDirection", "node_rev": "<git short>" },

  "model_receipt": {
    "slot": "creative",
    "requested_model": "<the handle requested>",
    "resolved_model": "<provider-reported concrete model, or null>",
    "provider": "<transformers | gguf | openrouter | comfy_credits | google_api>",
    "runtime_policy": { "...": "policy_from_meta snapshot" },
    "effective_sampling": { "...": "PROVIDER-EFFECTIVE values, or null where the
                            provider does not report them (4.3)" },
    "prompt_version": "vd-look-1 / vd-shots-1",
    "token_budget": { "context_cap": 8192, "pa_input": 0, "pa_reserved": 0,
                      "pb_batches": [], "repair_envelope": 0 },
    "attempt_journal": [ { "pass": "look", "batch": null, "rung": "base",
                           "outcome": "ok", "reason": null,
                           "prompt_sha256": "...", "accepted_sha256": "..." } ],
    "source_recheck": "clean"
  },

  "story_binding": {
    "episode_id": "<ledger episode_id>",
    "freeze_verdict": "frozen_clean",
    "freeze_timestamp": "<meta.freeze_timestamp verbatim>",
    "source_sha256": "<canonical sha256 of DirectionSourceV1, 2.5>"
  },

  "rationale": { "style_language": "...", "era_cues": [ ... ],
                 "motifs": [ ... ], "clue_visual": { ... } },

  "style_pack": {
    "style_id": "dynamic_story", "schema_version": "v2", "is_dynamic": true,
    "...": "the COMPLETE v2 field set -- ASSEMBLED from the vetted safety base
            + the draft's `look` -- validated by the SAME _validate_row rules
            (nodes/_otr_visual_styles.py:184+)"
  },
  "authored_fields": ["era_tail", "positive_tail", "..."],
  "field_evidence": { "<the draft map, verbatim>": {} },

  "shots": [ { "line_id": "...", "subject_note": "...", "mood": "...",
               "evidence": [ ... ] } ],

  "semantic_sha256": "<see below>",
  "artifact_sha256": "<canonical sha256 of the whole object minus this field>"
}
```

**Hash discipline (enumerated key-for-key).** `binding` below is a DERIVED
two-key object built from `story_binding` -- it is NOT the `story_binding` object
(r3 antigravity M1). `freeze_timestamp` and `freeze_verdict` are volatile across
a re-freeze of the same content, so they are EXCLUDED from the semantic seal and
INCLUDED in the envelope seal:

```
semantic_sha256 = canonical_sha256({
    "schema_version":  <str>,
    "style_pack":      <the assembled v2 pack>,
    "authored_fields": <sorted list>,
    "field_evidence":  <map>,
    "shots":           <list, source-line order>,
    "rationale":       <object>,
    "binding": { "episode_id": <str>, "source_sha256": <str> }
})
artifact_sha256 = canonical_sha256(<whole artifact minus artifact_sha256>)
```

Both are RECOMPUTED on every read (7.1). `created_utc` and `model_receipt` sit
outside the semantic hash.

The canonicalizer must be STRICT JSON-only: it REJECTS non-JSON values, NaN,
infinity, and non-string object keys. The existing helper
(nodes/production_ledger.py:292-302) falls back to `repr()` and never raises --
correct for a log line, wrong for a fail-closed seal.

`content_mutations` is CUT as a stored field (it would always be the constant 0).
The CHECK stays (5.2 step 6, refuse the stamp on any delta); its outcome is
recorded once as `model_receipt.source_recheck`.

### 2.4 Who authors which pack field (the safety split)

- **Vetted safety base (checked-in Python constant, operator-owned; not
  LLM-touched):** `portrait_look_talking` (S4b lip-sync law -- bright/frontal/
  warm, nodes/otr_meta_brief_image_prompt.py:160-168), `announcer_subject_face`,
  `announcer_subject_ltx_mouth` (mouth-vocabulary contract,
  nodes/_otr_visual_styles.py:117-119), `announcer_subject_object`,
  `open_subjects` ({form} templates), `motion_registers` (240-char engine budget,
  :121-123), `non_character_emblem_fallback`, `allow_radio_tails`,
  `forbidden_terms`, **and `still_word_typography` + `still_word_backdrop` in
  their entirety (D5 CLOSED)** -- every nested dict the LLM authors is another
  wrong-depth failure surface (4.5) and another 200-400 output tokens against an
  8192 cap. Reopen post-v1.
- **LLM-authored look whitelist:** exactly the 11 keys of
  `VisualDirectionDraftV1.look`.
- `authored_fields` is DERIVED (Python writes it from the whitelist) and is the
  machine-checkable proof the assembly obeyed the split.
- The anti-geometry lint over authored fields uses a PYTHON-OWNED fixed
  vocabulary (framing/headroom/crop/close-up/mouth-visibility terms), never the
  artifact's own `forbidden_terms` (an LLM must not author its own guard).

### 2.5 `DirectionSourceV1` -- the ONE source DTO, and its per-pass VIEWS

ONE closed DTO is simultaneously (a) the hash preimage and (b) the evidence
universe. Nothing may be cited that is not in it.

| Collection | EXACT fields (closed set) | null normalization |
|---|---|---|
| `meta` | episode_title, style, source_bank, visual_style | str: `""` |
| `brief` | atmosphere_line, visual_palette, key_objects, lighting, story_brief_terms | str: `""`; list: `[]` |
| `cast[]` | char_id, name, gender, traits, character_description | str: `""`; list: `[]` |
| `lines[]` | line_id, text, char_id, speaker_role, boundary, skip, tts_skip_reason | str: `""`; **bool `skip`: `false`** |

Rows in ledger order; keys sorted; per-type normalization (a blanket `null -> ""`
would corrupt the boolean `skip` and the list-valued brief fields -- r3 codex M4).
`story_brief_terms` lives under `brief` ONLY (rev 3 duplicated it under `meta`).
Any field NOT listed is omitted -- audio timing, voice presets, word counts,
reviewer notes are all post-freeze or non-authored, and excluding them BY DESIGN
is what makes the hash stable across ShotLock's in-place timing overlay
(nodes/otr_shot_lock.py:169-221) and CastLock's post-freeze voice assignment
(nodes/_otr_ledger_freeze.py:493-502).

`source_sha256 = canonical_sha256(DirectionSourceV1)`. Writer and readers compute
it with ONE shared pure helper so the two sides can never drift.
**No `beats` row** -- there is no such array (1.3).

**Per-pass VIEWS (each a NAMED, DEFINED projection of the DTO):**

- `DirectionSourceView_PA`: `meta` + `brief` + full `cast[]` + a LINE SPINE
  (`line_id`, `char_id`, `speaker_role`, `boundary`, `skip`, and `text` in full
  if the exact-preflight fits it, else a deterministic head truncation to the
  measured budget -- the truncation POINT is recorded in the receipt).
- `DirectionSourceView_PB(batch)`: the authored `look` from P-A + FULL `text` for
  the lines in this batch + the cast rows those lines reference.

**Cite-only-what-you-saw:** an evidence `id` must resolve in the DTO AND be
present in the VIEW the citing pass received. A pass cannot cite a line it never
read.

### 2.6 Evidence grammar, and the `shots[]` universe

IDs: `line:<line_id>`, `cast:<char_id>`, `meta:<key>`, `brief:<key>`, `title`.
`beat:` is RETIRED -- there is no ledger beat at direction time (1.3).

**The `shots[]` universe is the STILL-TARGET SET, not "every line" (r3
antigravity M4).** A row exists ONLY for a line that actually yields a scene
still or a word card, and that set is computed by the SAME pure target-derivation
helper MetaBrief uses (`derive_scene_still_targets` / `_iter_beat_lines`,
nodes/otr_meta_brief_image_prompt.py:1016+). Exactly one row per target, in
source-line order, no omissions, unique `line_id`. `b000_music_open` is OUT OF
SCOPE (it does not exist at direction time; the opening-music scene keeps its
pack-level treatment). Consumption is MANDATORY and per-kind (7.4) -- an authored
note that nothing consumes is dead scope.

### 2.7 Deliberately OUT of vd-1 (cuts, judged)

- `scenes[]` (no canonical scene key); `global.continuity` (`meta.continuity`
  already has one owner, nodes/OTR_LedgerScriptWriter.py:4721-4746; character look
  is carried by the cast rows, nodes/otr_shot_lock.py:116-153); wardrobe
  (`OTR_OUTFIT_LOCK` stays the authority, :143-153); executable
  `composition_rules` (geometry law) -- all CUT in r1.
- `rationale.composition_notes` -- non-executable BY CONSTRUCTION, so pure schema
  + context budget. CUT (r2).
- `story_binding.content_mutations` -- an always-zero stored constant. CUT (r3);
  the check survives as `model_receipt.source_recheck`.
- still_word typography/backdrop authorship -- CUT from v1 (2.4).
- Reroll/revision machinery, and with it ARTIFACT HISTORY -- CUT from v1 (7.5).
  The provenance digest is scoped to the CURRENT artifact; "which historical
  artifact authored this cached pixel" is explicitly not a v1 promise.
- Credits/dossier integration -- CUT.

`rationale.motifs` / `clue_visual` / `era_cues` are KEPT but NON-EXECUTABLE:
nothing composes prompts from them; they are the evidence-bound reasoning behind
the pack tails, and the audit walk reads them (7.6).

---

## 3. Ownership (Lesson 1)

### 3.1 Actors

| Actor | Role | Grounding |
|---|---|---|
| OTR_DynamicStoryDirection (NEW node) | SOLE writer of `meta.visual_direction`. Post-freeze, reads the frozen ledger read-only, stamps the artifact, forwards patched `script_json`. Pure pass-through when the sentinel is not selected. | Insertion at the 62.out[1] VISUAL fan-out (links 255/252, 1.4) |
| OTR_LedgerScriptWriter | Writes `meta.visual_style` only (now possibly the sentinel). Never writes `visual_direction`. | :3651-3655 |
| `resolve_visual_direction(ledger) -> ResolvedDirection` (NEW seam) | The ONE reader. Validates the artifact against the ledger and returns a typed BUNDLE: `{style: VisualStyle, shots_by_line_id: dict, semantic_sha256: str, artifact_meta: dict}` -- a bare `VisualStyle` cannot carry the mandatory shots index or the digest (r3 codex M7). `_resolve_style`/`get_visual_style` stay meta-only for named packs. | 5.3 |
| OTR_MetaBriefImagePromptGen | Consumer + MANDATORY consumer of `shots[]` per the 7.4 matrix. **Signature change required** -- `derive_image_prompts` today takes no ledger and resolves meta-only. Also gains a `finally` LLM teardown. | :1570-1574, 1601-1609, 2087-2096, 2137-2168 |
| OTR_ShotLock | Consumer; resolves BEFORE the beat loop, outside the fail-soft block. Also gains a `finally` LLM teardown. | :626-636, 651-697 |
| render_driver | THIRD independent resolver -- must resolve through the new seam and honor the dynamic look authority in its brief-core branch. | :1248, 2069, 2080 |
| OTR_ImageGenDispatcher | Persists the direction digest + composed prompt on image rows, on BOTH the cache-miss and cache-HIT paths (a hit CLONES the old row, :627-632). | :117-129, 603-643, 796-826 |

### 3.2 Field-level ownership (authored | derived | measured)

| Field | Class | Writer |
|---|---|---|
| `style_pack.<11 look keys>` | AUTHORED | direction LLM (P-A) |
| `style_pack.<all other v2 keys>` | AUTHORED (offline) | operator, in the checked-in safety base |
| `rationale.*` | AUTHORED | direction LLM (P-A), non-executable |
| `field_evidence` | AUTHORED | direction LLM (P-A) |
| `shots[]` | AUTHORED | direction LLM (P-B, batched) |
| `authored_fields` | DERIVED | Python (from the whitelist) |
| `story_binding.source_sha256` | DERIVED | Python |
| `story_binding.episode_id / freeze_verdict / freeze_timestamp` | DERIVED | Python (verbatim from meta) |
| `model_receipt.*` (provider, effective sampling, token_budget, attempt_journal, source_recheck) | MEASURED | Python (from the cache entry, the preflight, and the ladder) |
| `semantic_sha256` / `artifact_sha256` | DERIVED | Python |
| `created_utc` | MEASURED | Python |

### 3.3 Closed nested-row field sets

- `shots[]` row: EXACTLY `{line_id, subject_note, mood, evidence}`.
- `field_evidence` entry: EXACTLY `{evidence: [EvidenceItem], kind:
  "factual"|"rationale"}`; its KEY is from the closed enum (2.2).
- `EvidenceItem`: EXACTLY `{id, quote?}`; `quote` REQUIRED when `kind ==
  "factual"`.
- `rationale.motifs[]` row: EXACTLY `{motif, evidence}`.
- `clue_visual`: EXACTLY `{treatment, evidence}`.
- Any extra key at any depth = schema failure (`extra="forbid"`).

---

## 4. The LLM derivation contract (Lessons 2, 3, 4, 5)

### 4.1 Five representations, in lockstep

| # | Representation | Where it lives |
|---|---|---|
| 1 | base prompt | NEW `nodes/_otr_visual_direction_prompts.py` -- a FEATURE prompt module, matching the two existing `*_prompts.py` modules. **NOT a story-pack seam:** pack seams are PER-SOURCE-BANK (`PRODUCTION_SEAM_ALLOWLIST`, nodes/_otr_story_pack.py:27-44; unknown seam = `UnknownSeamError`, :146-151), and visual direction is orthogonal to the bank -- all 11 packs would otherwise have to author it. |
| 2 | typed schema | `VisualDirectionDraftV1` + `VisualDirectionShotsBatchV1` (pydantic, `extra="forbid"`, NO `max_length` on authored strings -- 2.1). |
| 3 | worked fixture | **TWO fixtures, one per pass**, under `tests/fixtures/`, each EMBEDDED verbatim in its own base prompt. |
| 4 | parser + validator | `parse_first_json_object` (nodes/_otr_json.py:81) / `parse_validate_tolerant` (nodes/_otr_structured_call.py:442) + a per-pass `post_validator`. **Never hand-roll brace slicing** -- `_parse_directives` (nodes/otr_shot_lock.py:429-451) is the anti-pattern: it slices `find("{")`/`rfind("}")` and returns `{}` SILENTLY. |
| 5 | repair prompt | Per failure class, PER PASS (4.5). |

Message assembly follows the proven pattern
(nodes/_otr_scifi_codex.py:1156-1160): seam text + `schema_shape_instruction`
(nodes/_otr_structured_call.py:195) as SYSTEM; a deterministic sorted-key JSON
envelope (the pass VIEW + the JSON schema + the worked fixture) as USER.

**The prompt must explicitly FORBID the known pseudo-shapes** (Lesson 2):
numbered fields (`era_tail_2`), `_secondary`/`_tertiary` variants, invented
schema-path keys, singular-vs-list aliases (`shot` vs `shots`), and **valid
collections nested at the WRONG DEPTH**. That last class is the highest-
probability live failure of this feature and it is already logged TWICE, this
week, on the second local family: PBUG-20260712-02 (nested `causal_steps` inside
`caller_threads` rows) and PBUG-20260712-03 (nested `shots` inside `scenes` rows)
-- docs/PROD_BUG_LOG.md. Each pass's prompt states ITS OWN collections' exact
top-level ownership (P-A owns `motifs` / `field_evidence`; P-B owns `shots`), and
the repair ladder carries the deterministic relocation rung (4.5).

### 4.2 Slot and model (D2 CLOSED -- `creative`)

The slot comment (nodes/OTR_LedgerScriptWriter.py:405-411) reads "technical =
structured passes", which naively argues `technical` for any JSON pass. But the
operative rule is PASS NATURE, and fable2 proves it: P0 dossier (extraction) runs
on **technical** (nodes/_otr_scifi_fable2.py:1129-1137), while P1 pitch room, P2b
treatment and P3 whole-play markup -- all authorship, all schema-constrained JSON
through `structured_call` -- run on **creative** (:1166-1174, :1201-1209, :1394).
Visual direction is authorship. **`creative`.**

- Model id: `meta["creative_writing_model"]` (stamped
  nodes/OTR_LedgerScriptWriter.py:1421-1422; the read-from-meta idiom is
  nodes/otr_shot_lock.py:663-665), resolved fail-loud through
  `require_model(model_id, slot="creative")` (nodes/_otr_model_inputs.py:72).
- Entry: `request_slot("creative", model_id, policy=policy_from_meta(meta))`
  (nodes/_otr_model_loader.py:790, 821-824; nodes/_otr_shared/llm_policy.py:131).
- **Two signatures, do not mix them.** `structured_call` needs
  `slot_fn(messages, *, temperature, max_new_tokens) -> str` (the
  `make_generate_fn` shape). ShotLock's `llm_fn` is the DIFFERENT
  `callable(prompt: str) -> str` (nodes/otr_shot_lock.py:513-516). The test
  injection for this node uses the GenerateFn shape.

### 4.3 The provider-effective interface (r3 codex M6 -- Lesson 5)

`_otr_model_loader.make_generate_fn`'s local lane (:1108-1137) applies the chat
template, tokenizes, and calls `model.generate` -- with **no `max_input_tokens`
computation, no truncation warning, and no must-fit honoring**. `context_cap` sits
on the cache entry and is IGNORED there. The guard exists ONLY inside the writer's
own slot wrapper (nodes/OTR_LedgerScriptWriter.py:664-699): `max_input_tokens =
max(64, context_cap - int(max_new_tokens))` (:681), then either
`raise PromptContextOverflowError("... refusing to left-truncate an unsliceable
provenance prompt")` when the messages object carries the must-fit marker
(:684-690; marker `_PromptMustFitMessages`, nodes/_otr_scifi_codex.py:308-311) or
a LEFT-truncating `PROMPT_GUARD` (:691-699). PROMPT_GUARD truncation of a typed
repair is the root-cause chain in PBUG-20260712-03.

And the guard cannot simply be copied, because **remote lanes have no HF
tokenizer to measure with**, OpenRouter/Comfy floor the output reservation, and
providers may override temperature/caps
(nodes/_otr_openrouter_backend.py:905-985; nodes/_otr_comfy_backend.py:426-480;
nodes/_otr_google_api/llm.py:115-162; nodes/_otr_gguf_backend.py:393-402).

**REQUIREMENT: ONE provider-effective interface**, used identically by the
preflight, the invocation, and the receipt, exposing per cache entry:
`context_cap`, `count_tokens(messages) -> int` (HF tokenizer locally; the
provider's own accounting or a conservative estimator remotely),
`effective_max_new_tokens(requested) -> int` (honoring provider floors), and
`effective_sampling() -> dict | null`. The direction pass RAISES on overflow and
NEVER truncates. Unknown values land in the receipt as `null`, not as invented
numbers.

**Constrained generation** is a LANE feature, not a slot feature:
`make_constrained_generate_fn` (nodes/_otr_constrained_generate.py:161) binds
lm-format-enforcer on the local HF lane (:262-269) and maps the same pydantic
schema to `response_format` on openrouter / comfy_credits / gguf_native
(:207-238). **Google has NO branch** -- so the cloud qualification leg (9.2)
either adds one or explicitly runs unconstrained with the repair ladder as its
only defense. State which.

### 4.4 Budgets, and why it is TWO passes (D8 CLOSED)

`DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"`
(nodes/_otr_model_catalog.py:32) backs both slots by default; `resolve_context_cap`
(:1258) clamps to `HARD_VRAM_CONTEXT_LIMIT`, default **8192** (:1207-1217), and
Mistral-Nemo is CURATED to 8192 (:1226-1234).

One vd-1 object cannot fit: 11 look strings + rationale + evidence + one `shots[]`
row per still target (a 420-word episode carries roughly 40-60 lines) -- and the
input carries every line's text + the brief + the schema + the fixture. Sizing
from `target_words` is exactly what Lesson 5 forbids.

**Two pass classes, one artifact, one hash:**

- **P-A (look).** ONE call. Input: `DirectionSourceView_PA`. Output:
  `VisualDirectionDraftV1`.
- **P-B (shots).** BATCHED over still targets, mirroring the existing batching
  seam `derive_creative_directives(..., batch_size: int = 15, ...)`
  (nodes/otr_shot_lock.py:499-508). Input: `DirectionSourceView_PB(batch)`.
  Output: `VisualDirectionShotsBatchV1`.

**Preflight, in the only order that can execute (r3 codex M5 + antigravity M2):**

1. `request_slot` FIRST (inside the teardown-protected region, 5.2) -- there is
   no `context_cap` and no tokenizer before a cache entry exists.
2. EXACT preflight of P-A: `count_tokens(P-A messages) +
   effective_max_new_tokens(pa_reserve) + REPAIR_ENVELOPE <= context_cap`.
3. SCHEMA-MAXIMUM preflight of P-B BEFORE any generation: the worst legal batch
   (batch_size rows, each at the maximum legal `subject_note` length + evidence +
   JSON overhead), so an impossible budget fails before the first token is spent.
4. Run P-A. Then EXACT-preflight each real P-B batch before it is sent.

`REPAIR_ENVELOPE` is COMPUTED, not guessed: the typed-repair call re-sends the
original prompt + the failed output echo + the directive and must itself generate
a full response, so the worst-case rung needs roughly
`input + 2 * max_new_tokens + len(directive)`. Measure the directive; do not
hard-code a magic number.

`batch_size` and the output reservations are DERIVED FROM THE MEASURED MAXIMA of
that arithmetic. (The starting figures 15 / ~900 tokens are a PLACEHOLDER for
Codex's measurement, not a specification: 15 legal 240-char notes plus evidence
and JSON overhead already exceed 900 tokens.)

Bounds (enforced in `post_validator`, never as `max_length` -- 2.1):
`subject_note <= 240` chars (mirrors the motion budget,
nodes/_otr_visual_styles.py:121-123); `quote <= 120` chars; every authored string
bounded; assembled artifact `<= 64 KB` canonical (a STORAGE bound, not a context
budget).

### 4.5 The repair ladder (Lessons 3, 4) -- reuse `structured_call`

`structured_call(*, prompt, schema, slot_fn, base_temperature,
structural_retry_temperature, repair_prompt_factory, post_validator,
max_new_tokens, max_attempts, helper_name)` (nodes/_otr_structured_call.py:551).
Its rungs ARE Lesson 4:

1. base attempt at `base_temperature` (:668-689);
2. structural retry -- SAME prompt, LOWER temperature, ONLY on
   `json.JSONDecodeError` (:700-721); a schema/content failure deliberately SKIPS
   this rung (:691-699);
3. typed repair at `_REPAIR_TEMPERATURE = 0.10` (:83, :724-775); the factory
   receives the original prompt, the failed raw output, and the exception
   (nodes/_otr_repair_prompts.py:128-152);
4. repair-syntax retry -- re-sends the EXACT cached repair prompt once (:783-811,
   floor `_REPAIR_SYNTAX_RETRY_FLOOR = 0.25` at :89).

`max_attempts = 3` (`_DEFAULT_MAX_ATTEMPTS`, :69). Entry invariant, fails loud:
`structural_retry_temperature` MUST be strictly lower than `base_temperature`
(:640-648). Exhaustion raises `StructuredCallFailedError` (:97, :819-823) -- never
a sentinel. Existing factories + dispatcher: nodes/_otr_repair_prompts.py:164,
:184, :204, :231, :250, :271, :290, :321, `make_dispatching_repair_factory`:402.

`post_validator` (typed `PostValidationError`, :128, raised :435-438) carries
EVERY deterministic CONTENT check: evidence-ID resolution against the DTO **and
against the pass's own VIEW**, the quote-substring check, the Python geometry
lint, the authored-vs-safety-base collision check, all length bounds, `line_id`
membership/uniqueness against the still-target set, and P-B aggregate set equality
(every target covered, none invented).

| Failure class | Rung | Deterministic repair (no LLM call, :750-761) |
|---|---|---|
| undecodable JSON | structural retry (same prompt, lower temp) | no |
| wrong-depth collection, WITHIN THIS PASS's own collections (P-A: `motifs`, `field_evidence`; P-B: `shots`) | typed repair naming the exact top-level ownership | **YES, and only here:** an authoritative top-level collection wins; a nested one is lifted VERBATIM only when top-level is absent/empty AND the destination is unique (or all candidates are byte-identical). The factory must then validate the COMPLETE contract -- schema AND post_validator -- before returning; a partially valid object must NOT consume the rung (Lesson 3; PBUG-20260712-02/-03 fix pattern). Cross-pass relocation is impossible and must not be attempted. |
| other schema/field shape | typed repair (`schema_field_repair` style) | no |
| unresolvable evidence ID, or an ID outside this pass's VIEW | typed repair naming the invariant + the owning key | no |
| missing or non-substring `quote` on a `factual` entry | typed repair naming the cited line's text | no |
| over-long authored string (note, quote, tail) | **typed repair** -- reached via `post_validator`, never a silent clamp (2.1) | no |
| geometry / forbidden term in an authored field | typed repair naming the term + the field | no |
| authored field outside the whitelist | typed repair naming the whitelist | **no** -- stripping it would accept an LLM write where it must not |
| P-B batch misses a target, invents a `line_id`, or duplicates one | typed repair naming the exact legal id set for THIS batch | no |
| ladder exhausted | `StructuredCallFailedError` -> named domain error -> episode ABORTS | fail closed |

**Attempt journal.** `structured_call` today returns only the accepted model and
exposes no rung/outcome hook, so a caller-side wrapper can hash raw calls but
cannot know which rung fired or why (r3 codex M8). The design adds an OPTIONAL
attempt-event sink to `structured_call` (an additive, backward-compatible
parameter -- section 8) that emits `{pass, batch, rung, outcome, reason,
effective_config, prompt_sha256, accepted_sha256}` per attempt. Receipts store
ACCEPTED-state hashes and reasons -- never rejected raw model output.

---

## 5. Lifecycle and storage

### 5.1 Location: `meta.visual_direction`. Nothing else is ledger-safe.

Top-level is ruled out by the merge code (only `TOP_PRESERVE` survives,
nodes/production_ledger.py:1387-1393) and by the freeze audit's top-level pin
(nodes/_otr_ledger_freeze.py:118-129). Line/cast rows are ruled out by row
ownership (:1441-1459) and the preflight law
(docs/SOURCE_BANK_PREFLIGHT.md:184-186). META survives by construction
(:1403-1413) -- exactly how `meta.visual_style`, `meta.freeze_*`,
`meta.gap_audit_*` persist today.

### 5.2 Write path (order is load-bearing -- r3 codex M9)

1. Preconditions (ALL fail-closed): `meta.visual_style == "dynamic_story"`;
   `meta.cleanup_locked is True`; `meta.freeze_verdict` in
   `{frozen_clean, frozen_with_warns}` (`needs_full_rerun` refuses,
   nodes/_otr_ledger_freeze.py:787-811); **`meta.freeze_unload_ok is True`** (a
   still-resident writer LLM plus a fresh creative LLM is an OOM on 16 GB;
   nodes/OTR_LedgerFreezeCascade.py:453-478); non-empty `lines`/`cast`.
2. Build `DirectionSourceV1`; compute `source_sha256` (2.5). Derive the
   still-target set with MetaBrief's own helper (2.6).
3. `request_slot` -- and from here to step 6 everything runs inside the
   teardown-protected region.
4. Preflight P-A exactly + P-B at schema maximum (4.4). Overflow raises HERE.
5. P-A, then the P-B batches, via `structured_call` (4.5).
6. Assemble the pack (safety base + authored `look`); validate the whole artifact:
   embedded pack through `_validate_row` (nodes/_otr_visual_styles.py:184+),
   evidence + quote resolution, geometry lint, whitelist collision, shots coverage
   against the target set. REBUILD `DirectionSourceV1` from the live ledger and
   REFUSE to stamp on any delta (`model_receipt.source_recheck`).
7. **`finally`: LLM teardown** (5.5). If teardown RAISES: persist the failed
   receipt and RE-RAISE -- do NOT proceed to GPU image work with a resident LLM.
8. ONLY AFTER a successful teardown: seal (`semantic_sha256`, `artifact_sha256`),
   stamp `meta.visual_direction` on the wire ledger, persist via
   `stamp_durable(meta_updates={"visual_direction": ...})`
   (nodes/production_ledger.py:408-422), re-serialize `script_json`, emit `done`.
   Before the durable stamp: `peek = peek_ledger()` -- it can return `None`
   (:372-397) -- and raise if `peek is None` or its `episode_id` differs from the
   wire ledger's.

### 5.3 Read path: `resolve_visual_direction(ledger) -> ResolvedDirection`

`get_visual_style(meta)` cannot enforce the staleness matrix -- it never sees the
arrays (nodes/_otr_visual_styles.py:378-390). ONE new ledger-aware function
validates the artifact and returns a TYPED BUNDLE:
`{style: VisualStyle, shots_by_line_id: dict[str, ShotNote], semantic_sha256: str,
artifact_meta: dict}`. A bare `VisualStyle` would force consumers to re-read
unvalidated `meta` for the notes and the digest.

Three consumer entries; **two need a signature change**:

- **MetaBrief:** `generate()` holds the parsed `led`
  (nodes/otr_meta_brief_image_prompt.py:2137-2144) -- it resolves there and passes
  the bundle down. `derive_image_prompts` gains `style=None` and
  `shots_by_line_id=None` instead of calling `_resolve_style(meta)` itself
  (:1570-1574, :1609). The threaded style must reach EVERY prompt branch.
- **ShotLock:** holds the whole `led` (nodes/otr_shot_lock.py:1040); resolves ONCE
  before the beat loop, outside the fail-soft block (7.2).
- **render_driver:** resolves once at the episode entry and threads the bundle
  through the shot-request builders instead of calling `get_visual_style(meta)` at
  :1248.

REJECTED (r1): reading the `peek_ledger()` process singleton inside
`get_visual_style` -- consumers operate on the WIRE ledger, and the singleton can
lag or be absent.

The dynamic `VisualStyle` NEVER enters the module pack cache (`_STYLES`,
nodes/_otr_visual_styles.py:170, 355-359); it is built per resolve from the
artifact.

### 5.4 Look authority on the dynamic lane (BOTH lanes)

The brief outranks the pack for the era tail today
(nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428), and the VIDEO lane has
a second brief-first core: `core = get_story_brief_ltx(_meta)`
(nodes/_otr_video_engines/render_driver.py:2069).

**Rule: on the dynamic lane the artifact pack is the SOLE final-look authority in
BOTH lanes; the brief is evidence INPUT to the direction LLM, not a runtime
override.** `get_era_tail`, the palette reads in the still/portrait profiles, and
render_driver's brief-core branch must all use the pack-authored tail when the
resolved style `is_dynamic`. The derivation prompt receives the brief verbatim (it
is IN `DirectionSourceV1`), so brief specifics reach the final look THROUGH the
authored pack, with evidence recorded.

D10 CLOSED: the dynamic lane is signalled by the `is_dynamic` BOOLEAN on the
resolved `VisualStyle`, not by a `style_id == "dynamic_story"` string compare
scattered through the helper family.

### 5.5 VRAM: a teardown BARRIER, not one node's `finally`

The direction node's own unload does NOT keep VRAM at baseline before image
dispatch: MetaBrief immediately RE-RESOLVES the writer LLM
(`_resolve_writer_llm(meta, warnings)`, nodes/otr_meta_brief_image_prompt.py:
2087-2096, called at :2158, delegating to ShotLock's resolver at
nodes/otr_shot_lock.py:651-697), and neither node unloads before returning.

Contract: **OTR_DynamicStoryDirection, OTR_MetaBriefImagePromptGen AND
OTR_ShotLock** each tear down in a `finally` after their last LLM call, mirroring
the cascade (nodes/OTR_LedgerFreezeCascade.py:377-387, 453-478):
`unload_llm_if_local_resident()`, a `*_unload_ok` stamp, loud logging, and an
ABORT before GPU image work if unloading raises. "No local model was ever loaded"
and "teardown failed" are DISTINCT receipts. Lazy imports throughout.

---

## 6. Dropdown / override semantics

- The writer dropdown gains ONE entry: `dynamic_story`, appended CODE-SIDE as a
  sentinel next to `list_style_ids()` (the `ADD_CUSTOM` idiom,
  nodes/otr_video_director.py:35). `list_style_ids()` itself stays REGISTRY-ONLY.
  The placeholder-pack-file alternative is REJECTED (r1) -- and because no file is
  added, the registry sweep (nodes/_otr_visual_styles.py:329-336) needs no
  exemption.
- **The dropdown test changes in the same commit:**
  `test_choices_are_exactly_the_registry` asserts
  `choices == list(vs.list_style_ids())` (tests/test_visual_style_widget_3c.py:
  62-66). It becomes "registry PLUS exactly one sentinel" at the WRITER surface;
  the registry-only property is asserted on `list_style_ids()` itself.
- **Sentinel gate fix:** the writer's run() gate `resolve_visual_style(...)` raises
  on any id without a pack file (nodes/OTR_LedgerScriptWriter.py:3334-3339) --
  selecting `dynamic_story` today would kill the run BEFORE the story exists. The
  gate must special-case the sentinel; the stamp mechanics stay unchanged
  (:3651-3655).
- **Explicit named pack always wins, byte-identical:** when
  `meta.visual_style != "dynamic_story"` the direction node is a pure pass-through
  and every resolver behaves exactly as today. Absent/empty `visual_style` keeps
  resolving to `sci_fi_radio` (nodes/_otr_visual_styles.py:386-389). **Every new
  receipt/provenance field is DYNAMIC-LANE-ONLY** -- adding a field to image rows
  unconditionally would break the named-pack byte-identity baseline.
- **The sentinel VALUE is the only trigger.** Artifact presence never activates
  dynamic styling -- a stale `meta.visual_direction` under a named pack is inert.
- Precedence: (1) named pack -> current behavior, dynamic machinery inert;
  (2) `dynamic_story` -> artifact mandatory, fail-closed; (3) nothing -> production
  default pack. No env override, no silent fallback between lanes.

---

## 7. Failure, stale data, replay, and audit

### 7.1 Fail-closed matrix (enforced in `resolve_visual_direction`)

With `meta.visual_style == "dynamic_story"`, ANY of the following aborts the
episode loudly (named error, never a fallback to a named pack):

- `meta.visual_direction` absent or not a dict; `schema_version` not `vd-1`.
- `story_binding.episode_id` / `.freeze_timestamp` / `.freeze_verdict` differ from
  the live meta (a re-frozen or foreign ledger).
- REBUILT `DirectionSourceV1` hash differs from the bound `source_sha256` --
  authored story content changed after direction. (The DTO, not the raw arrays:
  post-freeze timing/voice mutations are EXPECTED and must not false-fail.)
- RECOMPUTED `semantic_sha256` or `artifact_sha256` mismatch (a checksum against
  corruption -- the digest lives inside the mutable ledger, so it is not
  tamper-proofing).
- Embedded `style_pack` fails v2 validation, or an `authored_fields` entry names a
  safety-base field.
- Any evidence ID fails to resolve; any `factual` quote is not a substring of its
  cited text; any authored key in the closed enum lacks a `field_evidence` entry.
- The `shots[]` set is not exactly the still-target set (a missing target, an
  unknown `line_id`, or a duplicate).

Postures mirrored from: nodes/otr_image_director.py:428-456,
nodes/otr_shot_lock.py:1053-1061, nodes/_otr_visual_styles.py:367-375.

### 7.2 The ShotLock swallow

ShotLock's prompt finisher wraps `finish_visual_prompt` in a bare
`except Exception: pass` (nodes/otr_shot_lock.py:626-636) -- on the dynamic lane
that would silently convert every 7.1 abort into an unstyled prompt. Dynamic-lane
contract: resolve ONCE, before the beat loop, OUTSIDE any fail-soft block, and pass
`style=` into the finisher. `VisualStyleError` and vd-1 validation errors must
PROPAGATE.

### 7.3 Story immutability

Protection is the SOURCE-DTO COMPARISON (write-time re-hash refusing to stamp on
delta, 5.2 step 6; read-time re-hash, 7.1) -- NOT the merge ownership boundary,
which only controls disk copy-forward (nodes/production_ledger.py:1426-1459), and
NOT the freeze audit, which has already run.

### 7.4 Per-kind consumption matrix for `shots[]` (r3 antigravity M4 -- MANDATORY)

The mood seam rev 3 cited (nodes/otr_meta_brief_image_prompt.py:1004-1008) lives
INSIDE `compose_still_word_prompt` and fires for `character_video` WORD CARDS only.
Appending the notes there would leave character-scene stills and beat stills --
the very shots the notes describe -- ignoring them entirely. That is dead LLM
output, which this design forbids. The consumption matrix is therefore explicit:

| Prompt kind | Composer | `subject_note` | `mood` |
|---|---|---|---|
| scene still, character beat | `_compose_char_scene_prompt` (:1241-1366) | **YES** -- appended as a subject clause | **YES** |
| scene still, non-character beat | `compose_still_prompt` (nodes/_otr_story_brief_helpers.py:590) | **YES** | **YES** |
| still_word card | `compose_still_word_prompt` (:942-1013) | no (a word card has no subject) | **YES** -- overrides the derived mood adjective at :1004-1008 on character cards |
| portrait | portrait builders (:1148-1238) | no (per-CHARACTER, not per-line) | no |
| radio-host object | `build_radio_host_prompt` (:356-444) | no (per-EPISODE) | no |
| mesh fodder / background plate | (:1437-1496) | no (deliberately subject-free; `plate_look` is pack-level) | no |
| talking-head video prompt | ShotLock `finish_visual_prompt` | no in v1 (state explicitly) | no in v1 |

Consequently `shots[]` rows exist ONLY for the still-target set (2.6), every row IS
consumed, and nothing authored is dead. Notes are ADDITIVE clauses: they can never
replace pack-level fields, are bounded (4.4), and are linted against the safety-base
`forbidden_terms` + the Python geometry vocabulary at validation time.

Engine-safety continuity is structural: talking/mouth/motion/subject/typography
fields all come from the vetted base (2.4), so no per-episode LLM output can degrade
lip-sync or motion; `render_driver`'s `motion_registers` reads never vary with LLM
output.

### 7.5 Replay, reruns, and the reroll cut

- **A canonical re-queue is NOT a replay** (1.6): the writer and cascade
  `IS_CHANGED` return `time.time()`. Replay is proven at the RESOLVER/CACHE seams in
  deterministic tests over a CAPTURED frozen ledger -- never as a live leg, and never
  with a test-only workflow or mutation node.
- The new node declares NO `IS_CHANGED`, so ComfyUI may serve it from cache when its
  input `script_json` is byte-identical. That is INTENTIONAL and desirable
  (immutable-per-source), and it is tested; it is reachable only when the upstream did
  not re-execute.
- Replay property that DOES hold: an unchanged STORED artifact composes byte-identical
  prompts -> dispatcher cache HITs; any PROMPT-AFFECTING change flows into
  `prompt_hash` -> new cache keys -> regeneration
  (nodes/otr_image_gen_dispatcher.py:117-129). A change to `rationale` or a receipt
  alone does NOT invalidate the cache -- the key is the prompt hash.
- Reroll/revision is CUT from v1, and with it ARTIFACT HISTORY. The provenance digest
  identifies the artifact that authored the prompt for the CURRENT run; "which
  historical artifact authored this cached pixel" is explicitly not a v1 promise.
- REQUIRED: on the dynamic lane the Dispatcher writes `visual_direction_semantic_sha256`
  AND the composed prompt onto image rows **on BOTH the cache-miss and the cache-HIT
  path**. A hit CLONES the previous row (`fresh = dict(ref_row or {})`, :627-632), so an
  inherited digest would be STALE ATTRIBUTION. ShotLock does the same for its video
  `creative` sidecars, and on the dynamic lane the video section is persisted fail-loud
  rather than left on the wire.

### 7.6 Debug/audit walk ("why does this shot look like this?")

1. Rendered asset -> `ledger['images'].images[]` row (or
   `ledger['video'].shots[].creative`) -> composed `prompt` + `prompt_hash` +
   `visual_direction_semantic_sha256`.
2. That digest -> the `meta.visual_direction` that authored the prompt.
3. Prompt tail vocabulary -> `style_pack` fields (verbatim in the artifact) + the
   per-line `shots[]` note; `authored_fields` says whether the LLM or the safety base
   wrote each.
4. Each authored key -> its `field_evidence` entry -> the cited id AND the verbatim
   quote -> the exact `DirectionSourceV1` text that motivated it (factual vs rationale
   typed).
5. `model_receipt` gives provider, requested + resolved model, slot, effective
   sampling, the token budget, and the attempt journal (which rung ran, and why);
   `story_binding` proves which frozen story it derived from.

Pixels -> prompt -> pack field -> evidence quote -> frozen story text.

---

## 8. Code / workflow surfaces -- ALL "not implemented"

1. **NEW node `nodes/otr_dynamic_story_direction.py` (`OTR_DynamicStoryDirection`).**
   `CATEGORY = "OldTimeRadio/v2/visual"`, `FUNCTION = "direct"`,
   `OUTPUT_NODE = False`, `VALIDATE_INPUTS -> True`, display name
   `" Dynamic Story Direction (story-derived visual pack)"` (leading space + Title
   Case, the registry convention). `INPUT_TYPES`: required `script_json` (STRING,
   `multiline`, `default "{}"`, `forceInput: True`) -- **and nothing else.** The
   optional `gate_in` is CUT (no producer, no sequencing use; the `script_json` edge
   already orders the node).
   `RETURN_TYPES = ("STRING", "STRING", "STRING")`,
   `RETURN_NAMES = ("patched_ledger_json", "direction_report", "done")` -- the
   CastLock/ShotLock idiom (nodes/otr_shot_lock.py:966-969). `done` is the standard
   opaque ordering STRING and ships UNWIRED, exactly as ShotLock's own `done` does
   today (`links: []`). `direction_report` also ships unwired -- and is therefore
   ALSO written to the log, so the observability claim is true either way. Zero
   widgets.
2. **Registration in the LITERAL `_NODE_MODULES` dict in `__init__.py`** (:119-325;
   one tuple entry supplies BOTH the class mapping and the display name, written by
   the loader loop at :362-363). **NOT** via `nodes/_otr_class_registry.py`: the
   canonical-workflow contract test builds its node-class mappings by AST-parsing the
   literal `_NODE_MODULES` dict (tests/test_workflow_contract_validation.py:41) and
   never executes the class-registry merge (__init__.py:335-349) -- a registry-only
   node is INVISIBLE to the workflow gate.
3. **Workflow JSON delta -- the literal record (same change as the code, CLAUDE.md
   section 0).** Counters: `last_node_id: 95 -> 96`, `last_link_id: 283 -> 284`.
   REPOINT existing link ids; never renumber.

```json
{ "id": 96, "type": "OTR_DynamicStoryDirection",
  "pos": [1060, 760], "size": [379.96875, 239.96875],
  "flags": {}, "order": <topological -- after 62, before 89/90>, "mode": 0,
  "inputs": [ { "name": "script_json", "type": "STRING", "link": 284 } ],
  "outputs": [
    { "name": "patched_ledger_json", "type": "STRING", "links": [252, 255],
      "slot_index": 0 },
    { "name": "direction_report", "type": "STRING", "links": [] },
    { "name": "done", "type": "STRING", "links": [] } ],
  "properties": { "Node name for S&R": "OTR_DynamicStoryDirection" },
  "widgets_values": [] }
```

   Links: ADD `[284, 62, 1, 96, 0, "STRING"]`; REPOINT
   `[252, 96, 0, 90, 0, "STRING"]` and `[255, 96, 0, 89, 0, "STRING"]`.
   Node 62 `outputs[1].links`: `[16, 231, 232, 233, 252, 255]` ->
   `[16, 231, 232, 233, 284]`. Nodes 89/90 keep `inputs[0].link = 255 / 252` (the ids
   do not change, only their SOURCE); no widget change anywhere.
   **Re-baseline procedure, in order:** `OTR_WorkflowValidator`
   (nodes/_otr_workflow_validator.py:183, `validate()` :394-480 -- note `_assert_stamp`
   at :296-392 is the `semantic_master_hash` tripwire and `validate_anyway` can never
   skip it, :400-408) -> JSON round-trip -> widget/`INPUT_TYPES` audit
   (`_expected_slot_count`, :140-155) -> link/output-fan-out reconciliation -> master
   hash stamp.
4. **`nodes/_otr_visual_direction.py`** (lazy; pydantic + stdlib): the typed models,
   `DirectionSourceV1` + its per-pass views + the strict canonical hasher (shared by
   writer and readers), evidence + quote resolution, geometry lint, pack assembly, the
   fail-closed matrix, `resolve_visual_direction(ledger) -> ResolvedDirection`.
5. **`nodes/_otr_visual_direction_prompts.py`**: the P-A and P-B base prompts, the two
   worked fixtures, and the vd-1 typed-repair directives (4.5).
6. **`nodes/_otr_visual_direction_base.py`**: the vetted safety base as a PYTHON
   CONSTANT (D9 CLOSED -- a JSON file under `visual_styles/` would need a registry-sweep
   exemption, nodes/_otr_visual_styles.py:329-336, and could be picked up as a
   selectable pack by accident).
7. **`nodes/_otr_structured_call.py`**: an ADDITIVE, backward-compatible attempt-event
   sink parameter (4.5). No behavior change for existing callers.
8. **A provider-effective config + token-counting interface** (4.3) -- wherever Codex
   lands it (`_otr_model_loader` is the natural home). Includes the must-fit guard.
9. **`nodes/_otr_visual_styles.py`**: code-side sentinel exposure for the dropdown
   (registry unchanged); `visual_style_from_payload(dict)` funneling through
   `_validate_row`; the `is_dynamic` flag; dynamic instances bypass `_STYLES`.
10. **`nodes/_otr_story_brief_helpers.py`**: dynamic-lane look precedence in
    `get_era_tail` / the palette reads (5.4).
11. **`nodes/OTR_LedgerScriptWriter.py`**: sentinel-aware dropdown + run() gate (6).
12. **`nodes/otr_meta_brief_image_prompt.py`**: `generate()` resolves via the ledger;
    `derive_image_prompts` takes `style=` + `shots_by_line_id=` and threads them per the
    7.4 matrix; `finally` LLM teardown.
13. **`nodes/otr_shot_lock.py`**: hoist style resolution above the beat loop (7.2);
    `finally` LLM teardown; persist the video section fail-loud on the dynamic lane;
    stamp the direction digest into the `creative` sidecars.
14. **`nodes/_otr_video_engines/render_driver.py`**: resolve once through the new seam
    instead of `get_visual_style(meta)` (:1248); honor the dynamic look authority in the
    brief-core branch (:2069-2080).
15. **`nodes/otr_image_gen_dispatcher.py`**: dynamic-lane `visual_direction_semantic_sha256`
    + composed prompt on image rows, on BOTH the miss and the HIT path (7.5).

---

## 9. Test + live-smoke plan + sprint receipt

### 9.1 Unit (CPU, `OTR_TEST_MODE=1`, injected GenerateFn)

Injection shape is `slot_fn(messages, *, temperature, max_new_tokens)` -- NOT
ShotLock's `callable(prompt)->str` (4.2). Fakes modelled on
tests/test_video_platform_aseam.py:401-500.

1. **Schema round-trip, PER PASS:** a valid draft validates; each required field's
   absence fails named; `extra="forbid"` rejects an unknown key at every depth; the
   assembled pack goes through `_validate_row` verbatim; an `authored_fields` entry
   naming a safety-base field fails.
2. **No-silent-clamp:** an over-long `subject_note` / `era_tail` reaches TYPED REPAIR
   (via `post_validator`) and is NEVER truncated-and-accepted. This test exists
   specifically to pin the 2.1 decision against
   `_clamp_overlong_strings` (nodes/_otr_structured_call.py:422-434).
3. **Wrong-depth battery, PER PASS** (the PBUG-20260712-02/-03 class): P-A with
   `evidence` nested inside a `look` value, or `field_evidence` nested inside `motifs`;
   P-B with `shots` nested inside a row, a singular `shot` alias, a numbered
   `era_tail_2`. Each must either be deterministically relocated (unique destination
   ONLY, and the factory must pass the COMPLETE contract before returning) or advance
   the LLM repair rung -- never be silently accepted.
4. **Repair-ladder accounting:** a syntax failure runs the structural rung; a schema or
   post-validation failure goes STRAIGHT to typed repair; exhaustion raises
   `StructuredCallFailedError` and the episode aborts. The `attempt_journal` records
   every rung with its reason.
5. **Context preflight:** a fixture whose P-A prompt exceeds `context_cap -
   reserve - repair_envelope` RAISES before any generation, naming the real numbers; a
   schema-maximum P-B batch that cannot fit RAISES before P-A is even run.
6. **Evidence + quote:** an unresolvable id fails; a `factual` entry whose quote is not
   a substring of the cited text fails; an id valid in the DTO but absent from the
   citing pass's VIEW fails.
7. **Fail-closed matrix:** one test per row of 7.1 -- all raise; NONE fall back to a
   named pack.
8. **Inert-path byte-identity, PARAMETERIZED over every registered pack +
   absent/default style:** serialized ledger, composed prompts, prompt_hashes and
   dispatcher request keys are byte-identical before vs after the feature.
   **Capture those four baselines as committed fixtures BEFORE any code lands** -- once
   the code changes there is no immutable "before".
9. **Channel isolation:** run the SAME mocked story twice, changing ONLY
   `visual_style` (`sci_fi_radio` vs `dynamic_story`). Every pre-freeze story-model
   message must be byte-identical; only the post-freeze direction lane and the selector
   receipt may differ. This proves the sentinel never leaks into the story channel.
10. **Source-DTO stability:** ShotLock's timing overlay + CastLock's voice-preset
    assignment do NOT change `source_sha256`; a one-word text edit DOES.
11. **Merge survival, SPLIT in two:** (a) in-memory stamping under `OTR_TEST_MODE=1`
    (where `stamp_durable` skips the disk write, nodes/production_ledger.py:408-452);
    (b) a real `Ledger.save()` merge in an isolated tmpdir asserting restoration from
    disk (mirror of tests/test_ledger_merge_ownership.py).
12. **Story immutability:** a mutating fake is refused; `source_recheck` is not
    `clean`.
13. **Hash determinism:** same fixture + same injected output => identical
    `semantic_sha256`; `created_utc` / receipt variation changes ONLY the envelope hash.
    (A property of the STORED artifact, not of the LLM: the local lane hardcodes
    `do_sample=True` with no seed, nodes/_otr_model_loader.py:1122-1129, so
    re-derivation is NOT reproducible.)
14. **Per-kind consumption matrix (7.4):** for each kind, assert the note/mood clause
    IS or IS NOT present in the composed prompt, exactly as the matrix says; assert
    every `shots[]` row is consumed by at least one prompt; assert the `shots[]` set
    equals the still-target set.
15. **Look authority:** parameterized `get_era_tail` / palette tests asserting the
    pack-authored tail wins verbatim and the brief is IGNORED when `is_dynamic` -- in
    the still lane AND in render_driver's brief-core branch.
16. **Consumer propagation:** with dynamic selected and a broken artifact, MetaBrief,
    ShotLock AND render_driver all RAISE (do not swallow, do not fall back).
17. **Teardown:** each of the three LLM-touching post-freeze nodes unloads in `finally`;
    a teardown failure RAISES and never proceeds to GPU work.
18. **Dispatcher provenance:** on the dynamic lane, a cache HIT row carries the CURRENT
    digest and prompt -- not the cloned row's (pins :627-632); on a named pack, image
    rows are byte-identical to the pre-feature baseline (no new fields).
19. **Replay at the seam** (not live): a captured frozen ledger + an unchanged artifact
    recomposes byte-identical prompts and identical dispatcher request keys.
20. **Topology test:** the canonical JSON wires `62:1 -> 96:0 -> {89:0, 90:0}`; node 62's
    remaining fan-out is exactly `[16, 231, 232, 233, 284]`;
    `OTR_DynamicStoryDirection` is in the literal `_NODE_MODULES`.
21. **Look-QA rubric fixtures:** a small multi-genre fixture set (noir / western /
    sci-fi source DTOs) with an operator rubric -- specificity, palette/medium coherence,
    recurring identity, talking-lane safety, measurable difference from the fixed-pack
    control. Schema validity alone does not prove the direction is any good.

Suite discipline: full Windows regression + Bug Bible after every code chunk (CLAUDE.md
section 3). The conftest hard-fails the session on ANY new failed nodeid
(tests/conftest.py:219-286), so this is a real gate. Three-File Contract for any new bug
class.

**Tests that BREAK and must change in the same commit:**
`tests/test_visual_style_widget_3c.py:62-66` (choices == registry -> registry +
sentinel); `tests/test_google_video_sfx_workflow.py:41` (`last_link_id == 283` -> 284).
**Generic gates that must stay green:** `tests/test_workflow_graph_integrity_guards.py`
(widget-vector drift; link source-slot bounds; **output-link reconciliation** -- if the
delta repoints 252/255 in `links[]` but leaves them in `62.outputs[1].links`, this is
what fires); `tests/test_workflow_live_passes_validator.py:41-47`;
`tests/test_core.py:410-415` (id ceilings); `tests/test_workflow_link_target_indexes.py`;
`tests/test_full_workflow_v2_audio_wiring.py:149, 220-232` (the audio trio + CastLock stay
on the raw freeze outputs).
**Caveat:** the contract test runs with `strict_unknown_types=False`, so a class that
fails to IMPORT in the bare test env is silently SKIPPED there. A green contract test is
NOT proof the node registered -- the topology test (item 20) is.

### 9.2 Model-diversity qualification ladder (Lesson 6)

Both slots default to Mistral-Nemo (`DEFAULT_LLM`, nodes/_otr_model_catalog.py:32). A
prompt proven on one family is not qualified.

1. unit fixtures + full Windows suite + Bug Bible;
2. canonical 30-word end-to-end on **two local families** --
   `mistralai/Mistral-Nemo-Instruct-2407` and `google/gemma-4-E4B-it [LOCAL HF]` (the
   family behind PBUG-20260712-02 and -03, i.e. the one that demonstrably fails
   DIFFERENTLY) -- plus **one configured cloud/frontier creative lane**. The cloud lane
   is DECLARED before the run, with its concrete model id, its auth path (OpenRouter is
   key-gated, nodes/_otr_openrouter_backend.py:272), whether constrained generation is
   available on it (4.3 -- Google has NO branch today), and its rate policy. It is never
   silently skipped.
3. the same three pairings at 120 words;
4. only then any 720-word qualification / bake-off.

Record per leg: concrete model label, provider, slot, prompt_version, repair-rung counts
from the attempt journal, measured token budget, ledger path, episode asset path,
published asset path.

### 9.3 Live smoke (5080, headless :8000, reset per CLAUDE.md section 4)

Every leg loads the REAL `workflows/otr_canonical.json`.

1. **Control leg:** 30-word episode, `visual_style="sci_fi_radio"` -- prompts byte-match
   the committed pre-feature baseline (9.1 item 8).
2. **Dynamic legs:** the 9.2 ladder. Each asserts: the artifact is on the DISK ledger
   with full receipts; the attempt journal shows the rungs; still prompts carry the
   authored vocabulary AND the per-line notes per the 7.4 matrix; **no local LLM is
   resident when the Dispatcher's GPU work begins** (which requires MetaBrief's and
   ShotLock's teardown too, 5.5); assets exist at `otr\episodes\<ep>\` (Test-Path);
   `obs_publish OK` **AND** the final file exists under `otr\obs\` (Lesson 7 --
   `obs_publish OK` alone is not proof).
3. **Stale-source and replay:** NOT live legs (1.6, 7.5) -- proven deterministically over
   a captured frozen ledger.

Any failure in a LIVE run (smoke, soak, or published episode) gets an append-only
`PBUG-<YYYYMMDD>-<NN>` entry in docs/PROD_BUG_LOG.md using the template at :15-26
(surfaced / symptom / root cause / fix / verify idea / bible-worthy / confidence /
status). Dev-only catches are fixed and tested but NOT logged. Bible promotion happens at
the operator-triggered fan-out (Lesson 9).

### 9.4 Sprint receipt (fill at close)

```text
SPRINT RECEIPT: PASS | FAIL
scope:                    dynamic_story visual direction (vd-1)
authoritative_writers:    OTR_DynamicStoryDirection -> meta.visual_direction
durable_artifacts:        meta.visual_direction (+ semantic/artifact seals);
                          visual_direction_semantic_sha256 + composed prompt on image rows
                          and video creative sidecars (dynamic lane only)
canonical_workflow_hash:  <after the node 96 / link 284 delta>
focused_tests:            <9.1 items 1-21>
full_suite:               <passed/skipped/xfail>
bug_bible:                <result>
model_pairings:           mistral-nemo | gemma-4-E4B | <declared cloud creative lane>
30_word_receipts:         <ledger + asset paths per pairing>
120_word_receipts:        <ledger + asset paths per pairing>
720_word_receipts:        <only after 120w is green>
live_ledgers:             <paths>
published_assets:         <otr\obs\ paths, Test-Path verified>
prod_bug_entries:         <PBUG ids, or none>
head:                     <sha>
origin:                   <sha -- must equal head>
remaining_risks:
```

---

## 10. Remaining decisions for Codex

CLOSED by r1: sentinel = code-side, no placeholder pack (D1); no reroll in v1 (D4);
wardrobe orthogonal (D7); per-shot notes artifact-side with mandatory consumption (D3);
talking lane pinned to the safety base (D6).

CLOSED by r2: **D2** = `creative` slot (fable2 precedent, 4.2). **D5** = still_word
typography/backdrop pinned to the safety base in v1 (2.4). **D8** = the budget equation
+ the P-A/P-B split (4.4). **D9** = Python constant module (8.6). **D10** = an
`is_dynamic` flag on the resolved `VisualStyle` (5.4).

CLOSED by r3: the workflow delta (node 96 / link 284, repoint-don't-renumber, 8.3);
`gate_in` cut from the node; no `max_length` on authored strings (2.1); the per-kind
consumption matrix (7.4); `content_mutations` cut (2.3).

**I4 -- the direction/audio ordering edge. JUDGE'S RULING: NOT in v1.** The r3 panel
proposed a `96.done -> 81.gate_in` edge (link 285) to force the direction pass ahead of
the audio lane. Rejected for v1 on evidence the panel did not weigh: ComfyUI executes
nodes SERIALLY, the direction node unloads its LLM in `finally`, and the LLM-after-audio
pattern ALREADY exists in production -- MetaBrief re-resolves the writer LLM
(nodes/otr_meta_brief_image_prompt.py:2087-2096, :2158) with no audio ordering gate. The
feature introduces no new interleaving hazard, and an inbound edge from the visual lane
into the audio lane is a coupling not worth adding on an assumption. **Reopen trigger:**
a live smoke that OOMs with an LLM and an audio model co-resident. The `done` output
exists precisely so this edge is a one-link change if that happens.

Still open -- implementation choices, not design forks:

- **I1 -- where the must-fit / provider-effective helper lives** (4.3):
  `_otr_model_loader` is the natural home. Requirement: fail-loud on overflow, never
  truncate, and the SAME interface feeds preflight, invocation and receipt.
- **I2 -- constrained generation on/off for P-A** (4.3): lm-format-enforcer binds the
  schema at token level on the local HF lane and is the strongest defense against the
  wrong-depth class, but it costs latency and has no Google branch. Codex measures and
  decides; the typed-repair ladder is mandatory either way.
- **I3 -- P-B batch size and output reservation** (4.4): DERIVE from the measured
  maxima. The batching itself is not optional.

---

END. Docs-only deliverable; no code, tests, prompts, registries, or
workflows/otr_canonical.json were touched. Codex owns everything in section 8.
Kibitz artifacts: kibitz-runs/2026-07-12-dynamic-story-visual/{r1,r2,r3}/
(driver_anchor.md, codex.md, antigravity.md, judgment.md, final.md).
