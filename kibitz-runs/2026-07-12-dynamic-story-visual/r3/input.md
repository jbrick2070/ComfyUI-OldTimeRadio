# dynamic_story -- Story-Derived Visual Direction: Scoped Design (docs-only)

Date: 2026-07-12 (rev 3 -- r1 arc + r2 coding plan, both hardened by kibitz:
codex gpt-5.6-sol @ ultra + antigravity gemini-3.5-pro; driver anchors +
grounding in kibitz-runs/2026-07-12-dynamic-story-visual/{r1,r2}/)
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
  they never see `lines`/`cast` (r1 finding; drives section 5.3).
- The loaded packs are CACHED module-globally (`_STYLES`,
  nodes/_otr_visual_styles.py:170, 355-359). A per-episode dynamic pack must
  NEVER enter that cache (r1 antigravity S3).
- The geometry-vs-look law (nodes/_otr_visual_styles.py:7-9): packs own ONLY
  look/subject vocabulary; framing/headroom/mouth-safety GEOMETRY stays in
  Python. `dynamic_story` obeys the same law -- and r1 tightened it further: the
  LLM does not even author every pack LOOK field (2.4).

### 1.3 The story freeze boundary (what "final accepted story" means)

- OTR_LedgerScriptWriter -> OTR_LedgerFreezeCascade; Phase 10 runs the
  deterministic gap audit and, on success, stamps `meta.cleanup_locked = True`,
  `meta.freeze_timestamp` (ISO-8601 UTC), and `meta.freeze_verdict =
  frozen_clean | frozen_with_warns` (nodes/_otr_ledger_freeze.py:758-819,
  stamps at 806-811). Critical gaps raise `FreezeAssertionError` and stamp
  `needs_full_rerun` (nodes/_otr_ledger_freeze.py:787-795).
- Freeze locks story CONTENT, not the whole rows: post-freeze extension is an
  established pattern. OTR_CastLock assigns voice presets AFTER the freeze
  (acknowledged inside the freeze audit itself,
  nodes/_otr_ledger_freeze.py:493-502); OTR_ShotLock overlays per-line audio
  timing IN PLACE (nodes/otr_shot_lock.py:169-221) and stamps a whole
  `ledger['video']` section + `meta.video_revision`
  (nodes/otr_shot_lock.py:1104-1131). Consequence (r1 must-fix): the RAW
  `lines`/`cast` arrays are NOT stable post-freeze -- any staleness check must
  hash an authored-content PROJECTION, never the raw rows (2.5).
- The cascade unloads a locally-resident writer LLM in a `finally` block and
  stamps `meta.freeze_unload_ok` (nodes/OTR_LedgerFreezeCascade.py:377-387,
  453-478) -- the VRAM handoff pattern (5.5).
- **There is NO top-level `beats` array.** Beats are DERIVED from lines inside
  ShotLock, with `beat_id = line_id` (nodes/otr_shot_lock.py:260-288), and the
  synthetic opening-music beat `OPENING_MUSIC_BEAT_ID = "b000_music_open"` is
  created LATER still, by `derive_opening_music_beat(ledger, fps)` (:291-301).
  Nothing keyed to a beat exists at direction time (r2 codex M6).

### 1.4 Canonical wiring (workflows/otr_canonical.json, live graph)

Grounded by walking `nodes[]`/`links[]` of the real file (litegraph schema;
`last_node_id = 95`, `last_link_id = 283`):

- Writer (node 1) `script_json` -> link 230 -> FreezeCascade (node 62).
  The writer's `visual_style` widget is combined widget slot 24
  (`widgets_values[24] == "sci_fi_radio"`; `inputs[24]` carries
  `"widget": {"name": "visual_style"}` with `link: null`).
- **FreezeCascade (62) out[1] `script_json` fans out to SIX links:**
  `[16, 231, 232, 233, 252, 255]` --
  16 -> OTR_SignalLostVideo (12) in[2]; 231 -> OTR_BatchCharacterVoices (81)
  in[0]; 232 -> OTR_AnnouncerVoice (82) in[0]; 233 -> OTR_StableAudioTheme (83)
  in[0]; 252 -> OTR_ShotLock (90) in[0]; 255 -> OTR_MetaBriefImagePromptGen (89)
  in[0]. Only 252 and 255 are the VISUAL-lane consumers. The other four MUST
  keep reading the RAW freeze json (tests/test_full_workflow_v2_audio_wiring.py
  pins 81/82/83 to it) -- the direction node is inserted in the visual lane ONLY.
- OTR_VideoDirector (87) `video_policy_json` -> link 270 -> OTR_ImageDirector
  (88) in[0], and -> link 251 -> ShotLock in[2].
- ImageDirector `image_policy_json` -> link 254 -> MetaBrief in[1], and ->
  link 257 -> OTR_ImageGenDispatcher (91) in[1].
- MetaBrief out[0] `image_prompts_json` -> link 258 -> Dispatcher in[2].
- ShotLock out[0] `patched_ledger_json` -> link 256 -> Dispatcher in[0];
  ShotLock out[4] `episode_id` -> link 268 -> Dispatcher in[4].
- OTR_EpisodeAssembler (7) out[3] `audio_done` -> link 253 -> ShotLock in[1] and
  -> link 259 -> Dispatcher gate_in.
- Dispatcher out[0] -> link 260 -> OTR_VideoRenderBatch (92) in[0];
  Dispatcher out[1] `image_done` -> link 267 -> RenderBatch in[2].
- Node 89 has an UNWIRED optional `gate_in` (in[2], link null); node 90 has
  unwired `image_done` (in[3]) and `gate_in` (in[4]). Both carry
  `widgets_values == [False]`.

The Dispatcher receives its ledger via ShotLock's whole-ledger re-serialization
(parse at nodes/otr_shot_lock.py:1040, re-emit at :1142), so a meta artifact
stamped upstream reaches it with no extra wire (r1 anchor A1).

### 1.5 Where the style is actually consumed (three consumers, three shapes)

- **Image prompts (MetaBrief).** `derive_image_prompts` resolves the style ONCE
  at entry -- `_vstyle = _resolve_style(meta)` -- and threads it down; "an
  unknown meta['visual_style'] stops the episode HERE, before any prompt is
  composed" (nodes/otr_meta_brief_image_prompt.py:1601-1609). **Its signature
  takes NO ledger:** `derive_image_prompts(cast, meta, *, llm_fn=None, ...,
  lines=None, ...)` (:1570-1574), and the node's `generate()` parses the ledger
  but passes only `cast` / `meta` / `lines` (:2137-2168). Consumers of the
  threaded pack: portrait fallback + LLM instruction builders (:1148-1238),
  beat-aware character scene builder (:1241-1366), radio-host prompts (:356-444),
  mesh fodder + background plate (:1437-1496), still_word cards (:942-1013), and
  the aspect anchors (:171-188).
- **Video prompts (ShotLock).** M4 creative derivation finishes every
  talking-head prompt through `finish_visual_prompt(meta, text_prompt)` -- but
  that call sits inside a BARE `except Exception: pass`
  (nodes/otr_shot_lock.py:626-636), so a style-resolution failure there is
  silently swallowed today. On the dynamic lane the style must be resolved
  BEFORE the beat loop, outside that fail-soft block (7.2).
- **Video render (render_driver).** It resolves the style ITSELF:
  `_vstyle = get_visual_style((ledger or {}).get("meta") or {})`
  (nodes/_otr_video_engines/render_driver.py:1248) -- a THIRD independent
  resolution point that raises on the sentinel -- and one branch builds its
  prompt core from the BRIEF, not the pack: `core = get_story_brief_ltx(_meta)`
  (:2069) then `finish_visual_prompt(...)` (:2080). (r2 codex M7 / antigravity
  M1; MISSING from rev 2 entirely.)
- The helper family (`_resolve_style` :330, `get_era_tail` :344,
  `compose_still_prompt` :590, `finish_visual_prompt` :667 in
  nodes/_otr_story_brief_helpers.py) resolves the pack from meta.
- LOOK-AUTHORITY REALITY (r1 codex M3): the era tail is BRIEF-FIRST -- the
  brief's `atmosphere_line`/`visual_palette`/lighting win, and the pack's
  `era_tail` is only the fallback default
  (nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428). Any design that
  authors a new palette in the pack alone would be silently shadowed by the
  brief, in BOTH the still lane and the video lane. 5.4 resolves the precedence.
- ShotLock stamps per-shot `creative` sidecars with `prompt_hash` +
  `request_hash` into `ledger['video'].shots[]`
  (nodes/otr_shot_lock.py:637-647, 913-948).

### 1.6 Image dispatch, cache, and persistence (the replay substrate)

- Dispatcher cache key: `request_cache_key(role, object_id, prompt_hash, seed,
  engine_id, engine_version, kind, w, h)` -- "a change in ANY field -> new key ->
  regen" (nodes/otr_image_gen_dispatcher.py:117-129). Note it keys the PROMPT
  HASH: a change to rationale or a receipt that does not alter a composed prompt
  does NOT invalidate the cache (r2 codex M12).
- Seeds: `resolve_object_seed` (request_hash mode; pinned bookend seed)
  (:132-162).
- Results land in `ledger["images"] = {image_revision, images[], cache_index}`
  and persist to the DISK ledger via `stamp_durable(sections={"images": ...})`
  (nodes/otr_image_gen_dispatcher.py:796-826; nodes/production_ledger.py:408-422).
- **A canonical re-queue is never a replay.** The writer and the cascade both
  return `time.time()` from `IS_CHANGED` ("always re-execute",
  nodes/OTR_LedgerScriptWriter.py:3023-3028;
  nodes/OTR_LedgerFreezeCascade.py:269-272), so re-queuing writes a FRESH story
  and a FRESH freeze (r2 codex M12). Replay is a test-bench property, not a
  live-smoke leg (9.3).

### 1.7 Ledger save/merge ownership (what survives, what gets dropped)

`Ledger.save()` merges in-memory state with the on-disk ledger
(nodes/production_ledger.py:1287-1346, 1357-1513):

- TOP-LEVEL: only `TOP_PRESERVE = (schema_version, audio_gates, transitions,
  radio_bookend_path)` is copied forward from disk (:1387-1393). An unknown
  top-level key present on disk but absent from a later in-memory save is
  DROPPED. Top-level is NOT ledger-safe for a new artifact.
- META: per-key merge -- disk wins only where in-mem lacks the key or holds an
  empty value; in-mem wins where it has a real value (:1403-1413). A namespaced
  meta key written once SURVIVES every later save by construction.
- ROWS: row-level ownership. `_MERGE_OWNED_ROW_FIELDS` is a **19-field**
  frozenset (text, char_count, word_count, traits, boundary, char_id,
  speaker_role, arc_phase, compose_flags, beat_intent, target_words,
  dialogue_slot_id, shot_id, beat_id, skip, tts_skip_reason,
  reviewer_skip_reason, reviewer_note, needs_render_realign + the music-cue spec
  fields) -- never resurrected from disk (:1441-1459); out-of-band DURABLE
  renderer fields copy forward only on a content-identity match (:1477-1491).
  The merge boundary controls DISK COPY-FORWARD only -- it is NOT a runtime
  mutation validator (story-immutability protection is the projection
  comparison, 7.3). **It is also not the same field set as the direction
  projection** (r2 codex M3 corrected rev 2's claim that it was).
- The freeze audit hard-requires the fixed top-level list set and per-line shape
  (nodes/_otr_ledger_freeze.py:118-129, 252-404); it does not govern namespaced
  meta keys.
- Source-bank law, same conclusion: "Evidence maps and authorship receipts live
  in typed artifacts or namespaced `meta`; the fixed line schema contains no ad
  hoc provenance fields" (docs/SOURCE_BANK_PREFLIGHT.md:184-186).
- `OTR_TEST_MODE=1` makes `stamp_durable` SKIP the disk write
  (nodes/production_ledger.py:408-452; conftest sets it, tests/conftest.py:38) --
  so an in-memory stamping test and a real merge-survival test are two DIFFERENT
  tests (r2 codex S3).

---

## 2. The typed artifact: `meta.visual_direction` (schema `vd-1`)

### 2.1 TWO models, not one (r2 codex M1 -- Lesson 2)

Rev 2 showed one JSON blob containing both LLM-authored content AND
Python-derived receipts/hashes. That is unimplementable: a model cannot author
its own `prompt_sha256`. The design has TWO typed models:

- **`VisualDirectionDraftV1`** -- EXACTLY what the LLM returns. Pydantic
  `BaseModel`, `extra="forbid"`, every string and list bounded. Contains ONLY
  authored content. This is the `schema=` passed to `structured_call`.
- **`VisualDirectionArtifactV1`** -- what Python ASSEMBLES and stamps: the draft
  + the vetted safety base + the derived receipts + the seals. Never generated,
  never repaired, never LLM-touched.

Consequence: `nodes/_otr_visual_direction.py` is NOT "pure stdlib" (rev 2's
claim). It imports pydantic, lazily, like every other structured pass.

### 2.2 `VisualDirectionDraftV1` -- the LLM surface (closed set)

```json
{
  "style_language": "<one-paragraph episode visual thesis>",
  "motifs": [ { "motif": "<recurring visual motif>",
                "evidence": ["line:l0007"] } ],
  "clue_visual": { "treatment": "<how the clue mechanism reads on screen>",
                   "evidence": ["line:l0012", "line:l0003"] },
  "era_cues": ["<cue>", "..."],

  "look": {
    "label": "...", "era_tail": "...", "positive_tail": "...",
    "image_grade_tail": "...", "broadcast_tail": "...",
    "portrait_look": "...", "portrait_instruction_look": "...",
    "scene_instruction_look": "...", "radio_object_look": "...",
    "plate_look": "...", "still_word_title_mood_style": "..."
  },

  "field_evidence": {
    "look.era_tail":      { "evidence": ["line:l0003", "meta:episode_title"],
                            "kind": "factual" },
    "look.portrait_look": { "evidence": ["cast:c01"], "kind": "factual" },
    "style_language":     { "evidence": ["title"], "kind": "rationale" }
  }
}
```

- `look` is the CLOSED authored whitelist -- exactly the 11 keys above, no more,
  no fewer. `still_word_typography` / `still_word_backdrop` are NOT authored in
  v1 (D5 closed, 2.4).
- `field_evidence` is keyed by DOTTED PATH into this draft (not a JSON pointer):
  every key of `look`, plus `style_language`, plus `motifs[i].motif` and
  `clue_visual.treatment` -- one entry each, mandatory.
- `shots[]` is NOT in this model. It is authored by a separate batched pass
  (4.4) with its own draft model, `VisualDirectionShotsBatchV1`:
  `{ "shots": [ { "line_id": "...", "subject_note": "<=240 chars>",
  "mood": "<one register word>", "evidence": ["line:..."] } ] }`.

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
    "runtime_policy": { "...": "policy_from_meta snapshot" },
    "effective_sampling": { "base_temperature": 0.35,
                            "structural_retry_temperature": 0.15,
                            "repair_temperature": 0.10,
                            "do_sample": true, "top_p": 0.92,
                            "max_new_tokens_look": 1400,
                            "max_new_tokens_shots": 900 },
    "attempt_journal": [ { "pass": "look", "rung": "base", "outcome": "ok",
                           "reason": null, "prompt_sha256": "...",
                           "response_sha256": "..." } ],
    "prompt_version": "vd-look-1 / vd-shots-1"
  },

  "story_binding": {
    "episode_id": "<ledger episode_id>",
    "freeze_verdict": "frozen_clean",
    "freeze_timestamp": "<meta.freeze_timestamp verbatim>",
    "source_sha256": "<canonical sha256 of DirectionSourceV1, 2.5>",
    "content_mutations": 0
  },

  "rationale": {
    "style_language": "<from the draft>",
    "motifs": [ ... ], "clue_visual": { ... }, "era_cues": [ ... ]
  },

  "style_pack": {
    "style_id": "dynamic_story",
    "schema_version": "v2",
    "...": "the COMPLETE v2 field set -- ASSEMBLED from the vetted safety base
            + the draft's `look` -- validated by the SAME _validate_row rules
            (nodes/_otr_visual_styles.py:184+)"
  },
  "authored_fields": ["era_tail", "positive_tail", "..."],
  "field_evidence": { "<same map, re-keyed onto the assembled pack>": {} },

  "shots": [ { "line_id": "<ledger line_id>", "subject_note": "...",
               "mood": "...", "evidence": ["line:..."] } ],

  "semantic_sha256": "<see below>",
  "artifact_sha256": "<canonical sha256 of the whole object minus this field>"
}
```

**Hash discipline (r2 codex M4 -- rev 2 contradicted itself).** The semantic
preimage is enumerated KEY FOR KEY, and `story_binding.freeze_timestamp` is a
timestamp, so it is EXCLUDED:

```
semantic_sha256 = canonical_sha256({
    "schema_version":  <str>,
    "style_pack":      <the assembled v2 pack>,
    "authored_fields": <sorted list>,
    "field_evidence":  <map>,
    "shots":           <list, source-line order>,
    "rationale":       <object>,
    "binding": { "episode_id":   <str>,
                 "source_sha256": <str> }        # NO timestamps, NO verdict
})
artifact_sha256 = canonical_sha256(<whole artifact minus artifact_sha256>)
```

`created_utc`, `model_receipt` (incl. the attempt journal), `freeze_timestamp`
and `freeze_verdict` are OUTSIDE the semantic hash and INSIDE the envelope hash.
Both are RECOMPUTED on every read (7.1).

The canonicalizer must be STRICT JSON-only: it REJECTS non-JSON values, NaN,
infinity, and non-string object keys. The existing helper
(nodes/production_ledger.py:292-302) falls back to `repr()` and never raises --
correct for a log line, wrong for a fail-closed seal (r2 codex S2).

### 2.4 Who authors which pack field (the safety split)

The pack is ASSEMBLED from two sources and then validated whole:

- **Vetted safety base (checked-in Python constant, operator-owned; not
  LLM-touched):** `portrait_look_talking` (S4b lip-sync law -- bright/frontal/
  warm, nodes/otr_meta_brief_image_prompt.py:160-168), `announcer_subject_face`,
  `announcer_subject_ltx_mouth` (mouth-vocabulary contract,
  nodes/_otr_visual_styles.py:117-119), `announcer_subject_object`,
  `open_subjects` ({form} templates), `motion_registers` (240-char engine
  budget, :121-123), `non_character_emblem_fallback`, `allow_radio_tails`,
  `forbidden_terms`, **and (D5 CLOSED, r2) `still_word_typography` +
  `still_word_backdrop` in their entirety** -- every nested dict the LLM authors
  is another wrong-depth failure surface (see 4.5) and another 200-400 output
  tokens against an 8192 cap; typography variety is a low-yield surface next to
  an abort. Reopen post-v1.
- **LLM-authored look whitelist (the creative surface):** exactly the 11 keys of
  `VisualDirectionDraftV1.look` (2.2).
- `authored_fields` is DERIVED (Python writes it from the whitelist) and is the
  machine-checkable proof that the assembly obeyed the split.
- The anti-geometry lint over authored fields uses a PYTHON-OWNED fixed
  vocabulary (framing/headroom/crop/close-up/mouth-visibility terms), never the
  artifact's own `forbidden_terms` (an LLM must not author its own guard).

### 2.5 `DirectionSourceV1` -- the ONE source DTO (r2 codex M3 -- Lessons 1, 5)

Rev 2 had three inconsistent notions of "the story input": a projection whose
field list was wrongly claimed to equal the merge-owned set, a brief that was
fed to the prompt but absent from the projection, and a `beats` array that does
not exist. Replaced by ONE closed DTO that is simultaneously (a) the prompt
input, (b) the hash preimage, and (c) the evidence universe. Nothing may be
cited that is not in it; nothing in the prompt is outside it.

| Collection | EXACT fields (closed set) |
|---|---|
| `meta` | episode_title, style, source_bank, visual_style, story_brief_terms |
| `brief` | atmosphere_line, visual_palette, key_objects, lighting (the brief fields the era tail reads today: nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428) |
| `cast[]` | char_id, name, gender, traits, character_description |
| `lines[]` | line_id, text, char_id, speaker_role, boundary, skip, tts_skip_reason |

Rules: rows in ledger order; keys sorted; `null` normalized to the empty string;
any field NOT listed is omitted (audio timing, voice presets, word counts,
reviewer notes -- all post-freeze or non-authored, all excluded BY DESIGN, which
is what makes the hash stable across ShotLock's in-place timing overlay
(nodes/otr_shot_lock.py:169-221) and CastLock's post-freeze voice assignment
(nodes/_otr_ledger_freeze.py:493-502)).

`source_sha256 = canonical_sha256(DirectionSourceV1)`. Writer and readers
compute it with ONE shared pure helper so the two sides can never drift.

**No `beats` row.** There is no such array (1.3).

### 2.6 Evidence ID grammar

`line:<line_id>`, `cast:<char_id>`, `meta:<key within the meta whitelist>`,
`brief:<key within the brief whitelist>`, `title`. Every ID must resolve against
`DirectionSourceV1`; an unresolvable ID is a validation error (fail-closed).
`beat:` is RETIRED -- there is no ledger beat at direction time (1.3).

Every LLM-authored prompt-bearing field MUST carry a `field_evidence` entry,
typed `factual` (cites story content) or `rationale` (creative justification) --
ID existence proves traceability, the `kind` split keeps the audit honest about
what is derivation vs taste.

`shots[].line_id` must be a NON-SKIPPED line in `DirectionSourceV1.lines`.
`b000_music_open` is explicitly OUT OF SCOPE for authored rows (it does not
exist yet); the opening-music scene keeps its current pack-level treatment.

### 2.7 Deliberately OUT of vd-1 (cuts, judged)

- `scenes[]` -- no canonical scene key exists. CUT (r1).
- `global.continuity` -- `meta.continuity` already has one owner
  (nodes/OTR_LedgerScriptWriter.py:4721-4746); character look continuity is
  carried by the cast rows the appearance lookup reads
  (nodes/otr_shot_lock.py:116-153). CUT (r1).
- Wardrobe -- `OTR_OUTFIT_LOCK` stays the one wardrobe authority
  (nodes/otr_shot_lock.py:143-153). CUT (r1, D7).
- Executable `composition_rules` -- geometry law. CUT as an authority (r1).
- `rationale.composition_notes` -- non-executable BY CONSTRUCTION, so it is pure
  schema + context budget. CUT ENTIRELY (r2 codex CUT 4).
- Reroll/revision machinery -- CUT from v1 (7.5).
- Credits/dossier integration -- CUT.
- `still_word_typography` / `still_word_backdrop` authorship -- CUT from v1
  (2.4, D5 closed).

`rationale.motifs` / `clue_visual` / `era_cues` are KEPT (the operator product
intent names them) but they are NON-EXECUTABLE: nothing downstream composes
prompts from them; they are the evidence-bound reasoning that produced the pack
tails, and the audit walk reads them (7.6).

---

## 3. Ownership (Lesson 1)

### 3.1 Actors

| Actor | Role | Grounding |
|---|---|---|
| OTR_DynamicStoryDirection (NEW node -- not implemented) | SOLE writer of `meta.visual_direction`. Runs post-freeze, reads the frozen ledger read-only, stamps the artifact, forwards patched `script_json`. Pure pass-through when the sentinel is not selected. | Insertion at the FreezeCascade out[1] visual fan-out (links 255/252, 1.4) |
| OTR_LedgerScriptWriter | Writes `meta.visual_style` only (now possibly the sentinel). Never writes `visual_direction`. | nodes/OTR_LedgerScriptWriter.py:3651-3655 |
| `resolve_visual_direction(ledger)` (NEW seam -- not implemented) | The ONE reader that validates the artifact against the ledger and returns the assembled `VisualStyle`. Called at each consumer ENTRY. `_resolve_style`/`get_visual_style` stay meta-only for named packs. | 5.3 |
| OTR_MetaBriefImagePromptGen | Consumer (portraits, scene stills, radio-host, mesh fodder, plates, still_word) + MANDATORY consumer of `shots[]` notes. **Requires a signature change** -- `derive_image_prompts` today receives no ledger and resolves meta-only. | nodes/otr_meta_brief_image_prompt.py:1570-1574, 1601-1609, 2137-2168 |
| OTR_ShotLock | Consumer; on the dynamic lane resolves the style BEFORE the beat loop, outside the fail-soft finisher block. | nodes/otr_shot_lock.py:626-636 |
| render_driver | **Third independent resolver** -- calls `get_visual_style(meta)` itself and builds one prompt core from the BRIEF. Must resolve through the new seam and honor the dynamic look authority. | nodes/_otr_video_engines/render_driver.py:1248, 2069, 2080 |
| OTR_ImageGenDispatcher | Indirect consumer: prompts arrive pre-composed; its cache key reacts to any PROMPT change; it receives the artifact via ShotLock's whole-ledger re-serialization. Gains a `visual_direction_semantic_sha256` provenance stamp on image rows. | nodes/otr_image_gen_dispatcher.py:117-129, 796-826; nodes/otr_shot_lock.py:1040, 1142 |

### 3.2 Field-level ownership (authored | derived | measured)

| Field | Class | Writer | Lifecycle |
|---|---|---|---|
| `style_pack.<11 look keys>` | AUTHORED | direction LLM (P-A) | post-freeze, immutable once stamped |
| `style_pack.<all other v2 keys>` | AUTHORED (offline) | operator, in the checked-in safety base | build-time constant |
| `rationale.*` | AUTHORED | direction LLM (P-A) | post-freeze, non-executable |
| `field_evidence` | AUTHORED | direction LLM (P-A) | validated against DirectionSourceV1 |
| `shots[]` | AUTHORED | direction LLM (P-B, batched) | post-freeze; MANDATORY consumption by MetaBrief |
| `authored_fields` | DERIVED | Python (from the whitelist) | proof of the safety split |
| `story_binding.source_sha256` | DERIVED | Python (`canonical_sha256(DirectionSourceV1)`) | recomputed on every read |
| `story_binding.episode_id / freeze_verdict / freeze_timestamp` | DERIVED | Python (copied verbatim from meta) | compared on every read |
| `story_binding.content_mutations` | MEASURED | Python (re-hash after the LLM pass) | must be 0 or the stamp is refused |
| `model_receipt.*` incl. `attempt_journal` | MEASURED | Python (from the cache entry + the ladder) | provider-truthful; unknown values are `null` |
| `semantic_sha256` / `artifact_sha256` | DERIVED | Python | recomputed on every read |
| `created_utc` | MEASURED | Python | outside the semantic hash |

### 3.3 Closed nested-row field sets

- `shots[]` row: EXACTLY `{line_id, subject_note, mood, evidence}`. Zero or one
  row per non-skipped line; unique `line_id`; source-line order; every supplied
  row MUST be consumed.
- `field_evidence` entry: EXACTLY `{evidence: [str], kind: "factual"|"rationale"}`.
- `rationale.motifs[]` row: EXACTLY `{motif, evidence}`.
- `clue_visual`: EXACTLY `{treatment, evidence}`.
- Any extra key at any depth = schema failure (`extra="forbid"`).

---

## 4. The LLM derivation contract (Lessons 2, 3, 4, 5)

### 4.1 Five representations, in lockstep

| # | Representation | Where it lives |
|---|---|---|
| 1 | base prompt | NEW `nodes/_otr_visual_direction_prompts.py` -- a FEATURE prompt module, matching the two existing `*_prompts.py` modules (nodes/_otr_period_prompts.py, nodes/_otr_repair_prompts.py). **NOT a story-pack seam:** pack seams are PER-SOURCE-BANK (`PRODUCTION_SEAM_ALLOWLIST`, nodes/_otr_story_pack.py:27-44; unknown seam = `UnknownSeamError`, :146-151), and visual direction is orthogonal to the bank -- all 11 packs would otherwise have to author it. |
| 2 | typed schema | `VisualDirectionDraftV1` / `VisualDirectionShotsBatchV1` (pydantic, `extra="forbid"`), in `nodes/_otr_visual_direction.py`. |
| 3 | worked fixture | A valid vd-1 draft under `tests/fixtures/`, EMBEDDED verbatim in the base prompt. |
| 4 | parser + validator | `parse_first_json_object` (nodes/_otr_json.py:81) / `parse_validate_tolerant` (nodes/_otr_structured_call.py:442) + a `post_validator`. **Never hand-roll brace slicing** -- `_parse_directives` (nodes/otr_shot_lock.py:429-451) is the anti-pattern: it `find("{")`/`rfind("}")` slices and returns `{}` SILENTLY. |
| 5 | repair prompt | Per failure class, 4.5. |

Message assembly follows the proven pattern
(nodes/_otr_scifi_codex.py:1156-1160): the seam text + `schema_shape_instruction`
(nodes/_otr_structured_call.py:195) as SYSTEM; a deterministic sorted-key JSON
envelope (`DirectionSourceV1` + the JSON schema + the worked fixture) as USER.

**The prompt must explicitly FORBID the known pseudo-shapes** (Lesson 2):
numbered fields (`era_tail_2`), `_secondary`/`_tertiary` variants, schema-path
strings used as field names, singular-vs-list aliases (`shot` vs `shots`), and
**valid collections nested at the WRONG DEPTH**. That last class is the highest-
probability live failure of this feature and it is already logged TWICE, this
week, on the second local family: PBUG-20260712-02 (nested `causal_steps` inside
`caller_threads` rows) and PBUG-20260712-03 (nested `shots` inside `scenes`
rows) -- docs/PROD_BUG_LOG.md. vd-1's draft carries `motifs[]`, `field_evidence{}`
and (in P-B) `shots[]`: the prompt states each collection's exact top-level
ownership, and the repair ladder carries the deterministic relocation rung (4.5).

### 4.2 Slot and model (D2 CLOSED -- `creative`)

The slot comment (nodes/OTR_LedgerScriptWriter.py:405-411) reads "technical =
structured passes", which naively argues `technical` for any JSON pass. But the
operative rule is PASS NATURE, not output format, and fable2 proves it: P0
dossier (extraction) runs on **technical**
(nodes/_otr_scifi_fable2.py:1129-1137), while P1 pitch room, P2b treatment and
P3 whole-play markup -- all authorship, all schema-constrained JSON through
`structured_call` -- run on **creative** (:1166-1174, :1201-1209, :1394). Visual
direction is authorship. **`creative`.**

Mechanics:

- Model id: `meta["creative_writing_model"]` (stamped
  nodes/OTR_LedgerScriptWriter.py:1421-1422; the read-from-meta idiom is
  nodes/otr_shot_lock.py:663-665), resolved fail-loud through
  `require_model(model_id, slot="creative")` (nodes/_otr_model_inputs.py:72).
- Entry: `request_slot("creative", model_id, policy=policy_from_meta(meta))`
  (nodes/_otr_model_loader.py:790, 821-824;
  nodes/_otr_shared/llm_policy.py:131).
- Constrained generation is a LANE feature, not a slot feature:
  `make_constrained_generate_fn` (nodes/_otr_constrained_generate.py:161) binds
  lm-format-enforcer on the local HF lane (:262-269) and maps the same pydantic
  schema to `response_format` on remote lanes (:207-238). A creative-slot
  structured pass may use it.
- **Two signatures, do not mix them.** `structured_call` needs
  `slot_fn(messages, *, temperature, max_new_tokens) -> str` (the
  `make_generate_fn` shape). ShotLock's `llm_fn` is the DIFFERENT
  `callable(prompt: str) -> str` (nodes/otr_shot_lock.py:513-516). The test
  injection for this node uses the GenerateFn shape (r2 codex M8).

### 4.3 The context guard the seam does NOT give you (Lesson 5)

`_otr_model_loader.make_generate_fn`'s local lane
(nodes/_otr_model_loader.py:1108-1137) applies the chat template, tokenizes, and
calls `model.generate` -- with **no `max_input_tokens` computation, no
truncation warning, and no must-fit honoring**. `context_cap` sits on the cache
entry and is IGNORED on that path. The guard exists ONLY inside the writer's own
slot wrapper (nodes/OTR_LedgerScriptWriter.py:664-699):
`max_input_tokens = max(64, context_cap - int(max_new_tokens))` (:681), then
either `raise PromptContextOverflowError("... refusing to left-truncate an
unsliceable provenance prompt")` when the messages object carries the must-fit
marker (:684-690, marker `_PromptMustFitMessages` at
nodes/_otr_scifi_codex.py:308-311), or a LEFT-truncating `PROMPT_GUARD` warning
(:691-699).

A direction prompt is the most provenance-sensitive prompt in the repo. On the
raw ShotLock idiom it would be silently over-run or left-truncated -- losing the
SYSTEM/schema prefix FIRST. That is precisely the root-cause chain in
PBUG-20260712-03.

**REQUIREMENT:** the direction node's `slot_fn` is MUST-FIT capable -- it reads
`cache_entry["context_cap"]`, measures the tokenized input, and RAISES on
overflow. It never truncates. (Codex's implementation choice: lift the writer's
wrapper into a shared helper, or write the equivalent guard in the direction
module. The DESIGN requirement is fail-loud, not the location.)

### 4.4 Budgets, and why it is TWO passes (D8 CLOSED)

Real numbers: `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"`
(nodes/_otr_model_catalog.py:32) backs both slots by default;
`resolve_context_cap` (:1258) clamps to `HARD_VRAM_CONTEXT_LIMIT`, default
**8192** (:1207-1217), and Mistral-Nemo is CURATED to 8192 (:1226-1234).
Input budget = `context_cap - max_new_tokens`.

One vd-1 object would need: 11 look strings + rationale + one `field_evidence`
entry per authored field + **one `shots[]` row per non-skipped line**. A
420-word episode carries roughly 40-60 lines; at ~40 tokens per row that is
1600-2400 output tokens for `shots[]` ALONE. Input carries every line's text +
the brief + the schema + the fixture. At 420 words -- let alone the 720-word
bake-off target -- input + output does not fit in 8192, and a 30-word smoke will
NOT reveal it. Sizing from `target_words` is exactly what Lesson 5 forbids.

**Two pass classes, one artifact, one hash:**

- **P-A (look).** ONE call. Input: `DirectionSourceV1` with the LINE SPINE ONLY
  (line_id + char_id + speaker_role + a truncated text preview), plus meta,
  brief, and full cast. Output: `VisualDirectionDraftV1`.
  `max_new_tokens_look = 1400`.
- **P-B (shots).** BATCHED over non-skipped lines, mirroring the existing
  batching seam `derive_creative_directives(..., batch_size: int = 15, ...)`
  (nodes/otr_shot_lock.py:499-508). Input: P-A's authored `look` (so notes stay
  coherent with the pack) + this batch's FULL line text. Output:
  `VisualDirectionShotsBatchV1` for this batch only.
  `max_new_tokens_shots = 900` per batch of 15.

Preflight, BEFORE any generation: for each pass, tokenize the assembled prompt
and assert `input_tokens + max_new_tokens <= context_cap`; if the WORST-CASE
repair envelope (original prompt + failed output echo + directive) would not
fit, raise then -- not mid-ladder. Bounds: `subject_note <= 240` chars (mirrors
the motion budget, nodes/_otr_visual_styles.py:121-123); every authored string
bounded in the schema; assembled artifact `<= 64 KB` canonical (a STORAGE bound,
not a context budget).

### 4.5 The repair ladder (Lessons 3, 4) -- reuse `structured_call`

Rev 2's "bounded attempts (2, matching `max_reseed`)" is the wrong mechanism AND
numerically wrong (`max_reseed=2` executes up to three calls). The repo's single
structured-JSON entrypoint is `structured_call(*, prompt, schema, slot_fn,
base_temperature, structural_retry_temperature, repair_prompt_factory,
post_validator, max_new_tokens, max_attempts, helper_name)`
(nodes/_otr_structured_call.py:551), whose rungs ARE Lesson 4:

1. base attempt at `base_temperature` (:668-689);
2. structural retry -- SAME prompt, LOWER temperature, ONLY on
   `json.JSONDecodeError` (:700-721); a schema/content failure deliberately
   SKIPS this rung (:691-699);
3. typed repair at `_REPAIR_TEMPERATURE = 0.10` (:83, :724-775); the factory
   receives the original prompt, the failed raw output, and the exception
   (nodes/_otr_repair_prompts.py:128-152);
4. repair-syntax retry -- re-sends the EXACT cached repair prompt once
   (:783-811, floor `_REPAIR_SYNTAX_RETRY_FLOOR = 0.25` at :89).

`max_attempts = 3` (`_DEFAULT_MAX_ATTEMPTS`, :69). Entry invariant, fails loud:
`structural_retry_temperature` MUST be strictly lower than `base_temperature`
(:640-648). Exhaustion raises `StructuredCallFailedError` (:97, :819-823) --
never a sentinel. Existing factories + dispatcher:
nodes/_otr_repair_prompts.py:164, :184, :204, :231, :250, :271, :290, :321,
`make_dispatching_repair_factory`:402.

Lesson 3's line maps onto `post_validator` (typed `PostValidationError`,
nodes/_otr_structured_call.py:128, raised :435-438), which carries every
deterministic CONTENT check: evidence-ID resolution against `DirectionSourceV1`,
the Python geometry lint, the authored-vs-safety-base collision check, the
240-char caps, `line_id` membership and uniqueness.

| Failure class | Rung | Deterministic repair (no LLM call, :750-761) |
|---|---|---|
| undecodable JSON | structural retry (same prompt, lower temp) | no |
| wrong-depth collection (`shots` inside a row, `evidence` inside `look`) | typed repair naming the exact top-level ownership | **YES, and ONLY here:** an authoritative top-level collection wins; a nested one is lifted VERBATIM only when top-level is absent/empty and the destination is unique -- then the FULL schema + post_validator must pass or the LLM repair runs (the PBUG-20260712-02/-03 fix pattern) |
| other schema/field shape | typed repair (`schema_field_repair` style) | no |
| unresolvable evidence ID | typed repair naming the invariant + the owning dotted path | no |
| geometry / forbidden term in an authored field | typed repair naming the term + the field | no |
| authored field outside the whitelist | typed repair naming the whitelist | **no** -- silently stripping it would accept an LLM write where it must not |
| `line_id` not in the source, or duplicated | typed repair naming the legal ids | no |
| ladder exhausted | `StructuredCallFailedError` -> named domain error -> episode ABORTS | fail closed |

Every rung that runs is recorded in `model_receipt.attempt_journal` (pass, rung,
outcome, reason, prompt/response sha256) -- Lesson 4's "log which rung ran and
why".

---

## 5. Lifecycle and storage

### 5.1 Location: `meta.visual_direction`. Nothing else is ledger-safe.

- Top-level is ruled out by the merge code: only `TOP_PRESERVE` survives a later
  in-memory save (nodes/production_ledger.py:1387-1393); the freeze audit also
  pins the top-level shape (nodes/_otr_ledger_freeze.py:118-129).
- Line/cast rows are ruled out by row ownership (:1441-1459) and the preflight
  law (docs/SOURCE_BANK_PREFLIGHT.md:184-186).
- META survives by construction (:1403-1413) -- exactly how `meta.visual_style`,
  `meta.freeze_*`, `meta.gap_audit_*` persist today.

### 5.2 Write path

1. Preconditions (ALL fail-closed): `meta.visual_style == "dynamic_story"`;
   `meta.cleanup_locked is True`; `meta.freeze_verdict` in
   `{frozen_clean, frozen_with_warns}` (`needs_full_rerun` refuses,
   nodes/_otr_ledger_freeze.py:787-811); non-empty `lines`/`cast`.
2. Build `DirectionSourceV1` and compute `source_sha256` (2.5).
3. Context preflight for P-A and P-B, worst-case repair envelope included (4.4).
   Overflow raises HERE, before any generation.
4. P-A via `structured_call` on the creative slot with the must-fit slot_fn
   (4.2, 4.3, 4.5). Then P-B, batched.
5. Assemble the pack (safety base + authored `look`); validate the whole vd-1
   artifact: embedded pack through `_validate_row`
   (nodes/_otr_visual_styles.py:184+), evidence resolution, geometry lint,
   whitelist collision, shots coverage.
6. REBUILD `DirectionSourceV1` from the live ledger and refuse to stamp on any
   delta (`content_mutations` must be 0).
7. Seal (`semantic_sha256`, `artifact_sha256`), stamp `meta.visual_direction` on
   the wire ledger, and persist via
   `stamp_durable(meta_updates={"visual_direction": ...})`
   (nodes/production_ledger.py:408-422). Before stamping durably, verify
   `peek_ledger()`'s episode_id matches the wire ledger's (:372-397) -- the
   process singleton can be stale (r2 codex S6).
8. `finally`: LLM teardown (5.5).

### 5.3 Read path: `resolve_visual_direction(ledger) -> VisualStyle`

`get_visual_style(meta)` cannot enforce the staleness matrix -- it never sees the
arrays (nodes/_otr_visual_styles.py:378-390). ONE new ledger-aware function
validates the artifact and returns the assembled `VisualStyle`. It is called at
each consumer ENTRY and the result is threaded down the existing `style=`
parameters -- the resolve-once contract already in force
(nodes/otr_meta_brief_image_prompt.py:1601-1609).

Three consumer entries, and **two of them need a signature change** (r2):

- **MetaBrief:** `generate()` holds the parsed `led`
  (nodes/otr_meta_brief_image_prompt.py:2137-2144) -- it resolves there and
  passes the result down. `derive_image_prompts` gains a `style=None` (and a
  shots-note index) parameter instead of calling `_resolve_style(meta)` itself
  (:1570-1574, :1609). The threaded style must reach EVERY branch that composes
  a prompt -- portrait, character scene, radio-host, mesh fodder, plate,
  still_word -- not just the still_word mood seam.
- **ShotLock:** holds the whole `led` (nodes/otr_shot_lock.py:1040); resolves
  ONCE before the beat loop, outside the fail-soft block (7.2).
- **render_driver:** resolves once in the episode entry and threads the result
  through the shot-request builders instead of calling `get_visual_style(meta)`
  at :1248.

REJECTED alternative (r1): reading the `peek_ledger()` process singleton inside
`get_visual_style` -- consumers operate on the WIRE ledger, and the singleton can
lag or be absent in that seam.

The dynamic `VisualStyle` instance NEVER enters the module pack cache (`_STYLES`,
nodes/_otr_visual_styles.py:170, 355-359); it is built per resolve from the
artifact.

### 5.4 Look authority on the dynamic lane (both lanes)

Today the brief outranks the pack for the era tail
(nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428) -- correct for named
packs, but on the dynamic lane it would shadow exactly the palette the LLM just
authored, leaving TWO competing look authorities. And the VIDEO lane has the same
problem in a second place: `core = get_story_brief_ltx(_meta)`
(nodes/_otr_video_engines/render_driver.py:2069) is the runtime prompt core for
one branch.

**Rule: on the dynamic lane the artifact pack is the SOLE final-look authority in
BOTH lanes; the brief is evidence INPUT to the direction LLM, not a runtime
override.** Concretely: `get_era_tail` (and the palette reads inside the
still/portrait profiles, and render_driver's brief-core branch) must, when the
resolved style is dynamic, use the pack-authored tail. The derivation prompt
receives the brief verbatim (it is IN `DirectionSourceV1`), so brief specifics
reach the final look THROUGH the authored pack, with evidence recorded.

D10 CLOSED: the dynamic lane is signalled by a BOOLEAN FLAG on the resolved
`VisualStyle` (e.g. `is_dynamic`), not by a `style_id == "dynamic_story"` string
compare scattered through the helper family.

### 5.5 VRAM: a teardown BARRIER, not one node's `finally` (r2 codex M9)

Rev 2 claimed the direction node's own unload keeps VRAM at baseline before image
dispatch. FALSE: MetaBrief immediately RE-RESOLVES the writer LLM
(`_resolve_writer_llm(meta, warnings)`, nodes/otr_meta_brief_image_prompt.py:
2087-2096, called at :2158, delegating to ShotLock's resolver at
nodes/otr_shot_lock.py:651-697), and neither node unloads before returning.

Contract: EVERY post-freeze node that touches an LLM tears down in a `finally`
after its last LLM call, mirroring the cascade
(nodes/OTR_LedgerFreezeCascade.py:377-387, 453-478): `unload_llm_if_local_
resident()`, a `*_unload_ok` stamp, loud logging, and an ABORT before GPU image
work if unloading raises. "No local model was ever loaded" and "teardown failed"
are DISTINCT receipts. The artifact is re-serialized AFTER teardown so the
receipt is durable. Lazy imports throughout (module top level imports stdlib +
pydantic only).

---

## 6. Dropdown / override semantics

- The writer dropdown gains ONE entry: `dynamic_story`, appended CODE-SIDE as a
  sentinel next to `list_style_ids()` (the `ADD_CUSTOM` idiom,
  nodes/otr_video_director.py:35). `list_style_ids()` itself stays
  REGISTRY-ONLY. The placeholder-pack-file alternative is REJECTED (r1): a
  renderable placeholder pack is a silent-fallback hazard -- and because no file
  is added, the registry sweep (nodes/_otr_visual_styles.py:329-336) needs no
  exemption.
- **The dropdown test changes in the same commit:**
  `test_choices_are_exactly_the_registry` asserts
  `choices == list(vs.list_style_ids())` (tests/test_visual_style_widget_3c.py:
  62-66). It becomes "registry PLUS exactly one sentinel", asserted at the writer
  surface; the registry-only property is asserted on `list_style_ids()` itself.
- **Sentinel gate fix (r1 must-fix):** the writer's run() gate
  `resolve_visual_style(visual_style)` raises on any id without a pack file
  (nodes/OTR_LedgerScriptWriter.py:3334-3339) -- selecting `dynamic_story` today
  would kill the run BEFORE the story exists. The gate must special-case the
  sentinel (accept it without registry resolution); the stamp mechanics stay
  unchanged (:3651-3655).
- **Explicit named pack always wins, byte-identical:** when
  `meta.visual_style != "dynamic_story"`, the direction node is a pure
  pass-through and every resolver behaves exactly as today. Absent/empty
  `visual_style` keeps resolving to `sci_fi_radio`
  (nodes/_otr_visual_styles.py:386-389).
- **The sentinel VALUE is the only trigger.** Artifact presence never activates
  dynamic styling -- a stale `meta.visual_direction` under a named pack is inert.
- Precedence: (1) named pack -> current behavior, dynamic machinery inert;
  (2) `dynamic_story` -> artifact mandatory, fail-closed; (3) nothing ->
  production default pack. No env override, no silent fallback between lanes.

---

## 7. Failure, stale data, replay, and audit

### 7.1 Fail-closed matrix (enforced in `resolve_visual_direction(ledger)`)

With `meta.visual_style == "dynamic_story"`, ANY of the following aborts the
episode loudly (named error, never a fallback to a named pack):

- `meta.visual_direction` absent or not a dict.
- `schema_version` not in the known set (`vd-1`).
- `story_binding.episode_id`, `.freeze_timestamp`, or `.freeze_verdict` differs
  from the live meta (a re-frozen or foreign ledger).
- `story_binding.content_mutations != 0`.
- REBUILT `DirectionSourceV1` hash differs from the bound `source_sha256` --
  authored story content changed after direction. (The SOURCE DTO, not the raw
  arrays: post-freeze timing/voice mutations are EXPECTED and must not
  false-fail.)
- RECOMPUTED `semantic_sha256` or `artifact_sha256` mismatch (corruption
  detection -- the digest lives inside the mutable ledger, so this is a checksum,
  not tamper-proofing).
- Embedded `style_pack` fails v2 validation, or an `authored_fields` entry names
  a safety-base field (the LLM wrote where it must not).
- Any evidence ID fails to resolve; any authored prompt-bearing field lacks a
  `field_evidence` entry.
- Any `shots[].line_id` is absent from the source or duplicated.

Postures mirrored from: nodes/otr_image_director.py:428-456,
nodes/otr_shot_lock.py:1053-1061, nodes/_otr_visual_styles.py:367-375.

### 7.2 The ShotLock swallow (r1 must-fix)

ShotLock's prompt finisher wraps `finish_visual_prompt` in a bare
`except Exception: pass` (nodes/otr_shot_lock.py:626-636) -- on the dynamic lane
that would silently convert every 7.1 abort into an unstyled prompt. Dynamic-lane
contract: resolve the style ONCE, before the beat loop, OUTSIDE any fail-soft
block, and pass `style=` into the finisher. `VisualStyleError` and vd-1
validation errors must PROPAGATE.

### 7.3 Story immutability

Protection is the SOURCE-DTO COMPARISON (write-time re-hash refusing to stamp on
delta, 5.2 step 6; read-time re-hash, 7.1) -- NOT the merge ownership boundary,
which only controls disk copy-forward (nodes/production_ledger.py:1426-1459), and
NOT the freeze audit, which has already run.

### 7.4 Scene-by-scene drift prevention

- ONE pack per episode; the resolve-once threading contract
  (nodes/otr_meta_brief_image_prompt.py:1601-1609) puts the SAME instance behind
  every prompt.
- `shots[].subject_note/mood` are ADDITIVE clauses appended at the same seam the
  beat mood token uses today (nodes/otr_meta_brief_image_prompt.py:1004-1008);
  they can never replace pack-level fields, are capped at 240 chars, and are
  linted against the safety-base `forbidden_terms` + the Python geometry
  vocabulary at validation time.
- On the dynamic lane MetaBrief's consumption of `shots[]` is MANDATORY (an
  authored note that is never consumed is dead scope; a `line_id` present in
  `shots[]` but absent from the ledger fails validation).
- Engine-safety continuity is structural: talking/mouth/motion/subject/typography
  fields come from the vetted base (2.4), so no per-episode LLM output can
  degrade lip-sync or motion behavior. `render_driver`'s `motion_registers` reads
  therefore never vary with LLM output.

### 7.5 Replay, reruns, and the reroll cut

- **A canonical re-queue is NOT a replay** (1.6): the writer and cascade
  `IS_CHANGED` return `time.time()`, so a requeue writes a fresh story, freeze and
  direction. Replay is therefore proven at the RESOLVER/CACHE seams in
  deterministic tests over a CAPTURED frozen ledger -- never as a live leg, and
  never with a test-only workflow or mutation node (which would violate the
  canonical-workflow law).
- Replay property that DOES hold: an unchanged STORED artifact composes
  byte-identical prompts -> dispatcher cache HITs; any PROMPT-AFFECTING artifact
  change flows into `prompt_hash` -> new cache keys -> regeneration
  (nodes/otr_image_gen_dispatcher.py:117-129). A change to `rationale` or a
  receipt alone does NOT invalidate the cache -- the key is the prompt hash, not
  the artifact hash.
- Reroll/revision machinery is CUT from v1. When it returns, the lever is a
  mutating widget (`request_seed`-style) to force the cache miss, plus
  persisted-artifact reuse keyed on `source_sha256 + prompt_version +
  requested_model`.
- REQUIRED (promoted from should, r2 codex S4): stamp
  `visual_direction_semantic_sha256` (short form) onto the dispatcher's image
  rows and ShotLock's video `creative` sidecars, so an asset links to the exact
  direction that authored its prompt even after a later re-direction (today's
  rows carry prompt/render identity only,
  nodes/otr_image_gen_dispatcher.py:117-129, 796-826;
  nodes/otr_shot_lock.py:637-647, 940-947). Without it the 7.6 audit walk breaks
  the moment a re-direction happens.

### 7.6 Debug/audit walk ("why does this shot look like this?")

1. Rendered asset -> `ledger['images'].images[]` row (or
   `ledger['video'].shots[].creative`) -> `prompt` + `prompt_hash` +
   `visual_direction_semantic_sha256`.
2. That digest -> the exact `meta.visual_direction` that authored the prompt.
3. Prompt tail vocabulary -> `style_pack` fields (verbatim in the artifact) + the
   per-line `shots[]` note; `authored_fields` says whether the LLM or the safety
   base wrote each.
4. Each authored field -> its `field_evidence` entry -> the exact
   `DirectionSourceV1` content that motivated it (factual vs rationale typed).
5. `model_receipt` gives requested + resolved model, slot, effective sampling,
   and the attempt journal (which rung ran, and why); `story_binding` proves which
   frozen story it derived from.

Pixels -> prompt -> pack field -> evidence -> frozen story text.

---

## 8. Code / workflow surfaces -- ALL "not implemented"

1. **NEW node `nodes/otr_dynamic_story_direction.py`
   (`OTR_DynamicStoryDirection`)**. `INPUT_TYPES`: required `script_json`
   (STRING, forceInput), optional `gate_in` (STRING, forceInput, opaque).
   `RETURN_TYPES/RETURN_NAMES`: `("STRING","STRING","STRING")` /
   `("patched_ledger_json", "direction_report", "done")` -- the CastLock/ShotLock
   idiom (nodes/otr_shot_lock.py:966-969). `direction_report` is the operator's
   only human-readable surface for a taste feature; `done` is the standard opaque
   ordering STRING (ShotLock's own `done` ships UNWIRED, `links: []`, in the live
   canonical file). Both ship UNWIRED in v1 -- stated explicitly so the link audit
   expects nothing. `FUNCTION`, `CATEGORY`, `VALIDATE_INPUTS`. Zero widgets ->
   `widgets_values: []`.
2. **Registration in the LITERAL `_NODE_MODULES` dict in `__init__.py`**
   (:119-325; one tuple entry supplies BOTH the class mapping and the display
   name, written by the loader loop at :362-363). **NOT** via
   `nodes/_otr_class_registry.py`: the canonical-workflow contract test builds its
   node-class mappings by AST-parsing the literal `_NODE_MODULES` dict
   (tests/test_workflow_contract_validation.py:41) and never executes the class-
   registry merge (__init__.py:335-349) -- a registry-only node is INVISIBLE to
   the workflow gate.
3. **Workflow JSON delta (same change as the code, CLAUDE.md section 0).**
   Node id **96**, new link id **284**. Add `[284, 62, 1, 96, 0, "STRING"]`;
   REPOINT the existing links rather than renumbering:
   `[252, 96, 0, 90, 0, "STRING"]` and `[255, 96, 0, 89, 0, "STRING"]`. Node 62
   `outputs[1].links` becomes `[16, 231, 232, 233, 284]` (the audio trio + the
   signal-lost read STAY on the raw freeze json). New node `outputs[0].links` =
   `[252, 255]`. `last_node_id = 96`, `last_link_id = 284`. `widgets_values: []`.
   r3 owns the FULL record (pos/size/order/properties) and the test-pin sweep.
4. **`nodes/_otr_visual_direction.py`** (lazy; pydantic + stdlib): the two typed
   models, `DirectionSourceV1` + the canonical hasher (shared by writer and
   readers), evidence resolution, geometry lint, pack assembly, the fail-closed
   matrix, `resolve_visual_direction(ledger)`.
5. **`nodes/_otr_visual_direction_prompts.py`**: the P-A and P-B base prompts,
   the worked fixture reference, and the vd-1 typed-repair directives (4.5).
6. **`nodes/_otr_visual_direction_base.py`**: the vetted safety base as a PYTHON
   CONSTANT (D9 CLOSED -- a JSON file under `visual_styles/` would need a
   registry-sweep exemption, nodes/_otr_visual_styles.py:329-336, and could be
   picked up as a selectable pack by accident).
7. **`nodes/_otr_visual_styles.py`**: code-side sentinel exposure for the
   dropdown (registry itself unchanged); `visual_style_from_payload(dict)`
   funneling through `_validate_row`; the `is_dynamic` flag; dynamic instances
   bypass `_STYLES`.
8. **`nodes/_otr_story_brief_helpers.py`**: dynamic-lane look precedence in
   `get_era_tail` / the palette reads (5.4).
9. **`nodes/OTR_LedgerScriptWriter.py`**: sentinel-aware dropdown + run() gate (6).
10. **`nodes/otr_meta_brief_image_prompt.py`** (r2 -- MISSING from rev 2):
    `generate()` resolves via the ledger; `derive_image_prompts` takes `style=`
    + the shots-note index and threads them into EVERY prompt branch.
11. **`nodes/otr_shot_lock.py`**: hoist style resolution above the beat loop on
    the dynamic lane (7.2); stamp the direction digest into the `creative`
    sidecars.
12. **`nodes/_otr_video_engines/render_driver.py`** (r2 -- MISSING from rev 2):
    resolve once through the new seam instead of `get_visual_style(meta)` at
    :1248; honor the dynamic look authority in the brief-core branch (:2069-2080).
13. **`nodes/otr_image_gen_dispatcher.py`**: `visual_direction_semantic_sha256`
    provenance on image rows (7.5).
14. **A shared must-fit slot_fn helper** (4.3), wherever Codex lands it.

---

## 9. Test + live-smoke plan + sprint receipt

### 9.1 Unit (CPU, `OTR_TEST_MODE=1`, injected GenerateFn)

The injection shape is `slot_fn(messages, *, temperature, max_new_tokens)` --
NOT ShotLock's `callable(prompt)->str` (4.2). Fakes modelled on
tests/test_video_platform_aseam.py:401-500.

1. **Schema round-trip:** a valid `VisualDirectionDraftV1` validates; each
   required field's absence fails named; `extra="forbid"` rejects an unknown key
   at every depth; the assembled pack goes through `_validate_row` verbatim; an
   `authored_fields` entry naming a safety-base field fails.
2. **Wrong-depth battery (the PBUG-20260712-02/-03 class):** `evidence` nested
   inside a `look` value; `shots` nested inside a `motifs` row; a singular `shot`
   alias; a numbered `era_tail_2`. Each must either be deterministically relocated
   (only in the unique-destination case) AND then pass the full schema +
   post_validator, or advance the LLM repair rung -- never be silently accepted.
3. **Repair-ladder accounting:** a fake that fails syntax once then succeeds runs
   the structural rung; a fake that fails schema goes STRAIGHT to typed repair
   (never the structural rung); ladder exhaustion raises
   `StructuredCallFailedError` and the episode aborts. The `attempt_journal`
   records each rung.
4. **Context preflight:** a fixture whose prompt exceeds `context_cap -
   max_new_tokens` RAISES before any generation and is never truncated; assert the
   raised error names the real numbers.
5. **Fail-closed matrix:** one test per row of 7.1 -- all raise; NONE fall back
   to a named pack.
6. **Inert-path byte-identity, PARAMETERIZED over every registered pack +
   absent/default style:** serialized ledger, composed prompts, prompt_hashes and
   dispatcher request keys are byte-identical before vs after the feature.
   **Capture those four baselines as committed fixtures BEFORE any code lands**
   (r2 codex S5) -- once the code changes there is no immutable "before".
7. **Source-DTO stability:** ShotLock's timing overlay + CastLock's voice-preset
   assignment do NOT change `source_sha256`; a one-word text edit DOES.
8. **Merge survival, SPLIT in two** (r2 codex S3): (a) in-memory stamping under
   `OTR_TEST_MODE=1` (where `stamp_durable` skips the disk write,
   nodes/production_ledger.py:408-452); (b) a real `Ledger.save()` merge in an
   isolated tmpdir asserting restoration from disk
   (mirror of tests/test_ledger_merge_ownership.py).
9. **Story immutability:** a mutating fake is refused with
   `content_mutations != 0`.
10. **Hash determinism:** same fixture + same injected output => identical
    `semantic_sha256`; `created_utc` / receipt variation changes ONLY the envelope
    hash. (This is a property of the STORED artifact, not of the LLM: the local
    lane hardcodes `do_sample=True` with no seed,
    nodes/_otr_model_loader.py:1122-1129, so re-derivation is NOT reproducible.)
11. **Drift + geometry guards:** a `subject_note` with a geometry/forbidden term
    fails; a `shots[].line_id` not in the ledger fails; a duplicate `line_id`
    fails; all pack-level look tokens identical across beats.
12. **Consumer propagation:** with dynamic selected and a broken artifact,
    MetaBrief, ShotLock AND render_driver all RAISE (do not swallow, do not fall
    back) -- pins 7.2 and the two newly-added surfaces.
13. **Replay at the seam** (not live): a captured frozen ledger + an unchanged
    artifact recomposes byte-identical prompts and identical dispatcher request
    keys.
14. **Look-QA rubric fixtures:** a small multi-genre fixture set (noir / western
    / sci-fi source DTOs) with an operator rubric -- specificity, palette/medium
    coherence, recurring identity, talking-lane safety, measurable difference from
    the fixed-pack control. Schema validity alone does not prove the direction is
    any good.

Suite discipline: full Windows regression + Bug Bible after every code chunk
(CLAUDE.md section 3). The conftest hard-fails the session on ANY new failed
nodeid (tests/conftest.py:219-286), so this is a real gate. Three-File Contract
for any new bug class.

**Tests that BREAK and must change in the same commit:**
`tests/test_visual_style_widget_3c.py:62-66` (choices == registry -> registry +
sentinel); `tests/test_google_video_sfx_workflow.py:41`
(`last_link_id == 283` -> 284); plus the generic gates that must stay green
(`tests/test_workflow_graph_integrity_guards.py` widget-vector drift + output-link
reconciliation; `tests/test_core.py:410-415` id ceilings). r3 owns the full sweep.

### 9.2 Model-diversity qualification ladder (Lesson 6)

Both slots default to Mistral-Nemo (`DEFAULT_LLM`,
nodes/_otr_model_catalog.py:32). A prompt proven on one family is not qualified.

1. unit fixtures + full Windows suite + Bug Bible;
2. canonical 30-word end-to-end on **two local families** --
   `mistralai/Mistral-Nemo-Instruct-2407` and `google/gemma-4-E4B-it [LOCAL HF]`
   (the family behind PBUG-20260712-02 and -03, i.e. the one that demonstrably
   fails DIFFERENTLY) -- plus **one configured cloud/frontier creative lane**
   (`openrouter:slot-a` or `google_api:slot-a`; OpenRouter is key-gated,
   nodes/_otr_openrouter_backend.py:272, so this leg is operator-env dependent and
   is DECLARED, never silently skipped);
3. the same three pairings at 120 words;
4. only then any 720-word qualification / bake-off.

Record per leg: concrete model label, slot, prompt_version, repair-rung counts
from the attempt journal, ledger path, episode asset path, published asset path.

### 9.3 Live smoke (5080, headless :8000, reset per CLAUDE.md section 4)

Every leg loads the REAL `workflows/otr_canonical.json`.

1. **Control leg:** 30-word episode, `visual_style="sci_fi_radio"` -- prompts
   byte-match the committed pre-feature baseline (9.1 item 6).
2. **Dynamic legs:** the 9.2 ladder. Each asserts: the artifact is on the DISK
   ledger with full receipts; the attempt journal shows the rungs; still prompts
   carry the authored vocabulary; **no local LLM is resident when the Dispatcher's
   GPU work begins** (which requires MetaBrief's and ShotLock's teardown too, 5.5);
   assets exist at `otr\episodes\<ep>\` (Test-Path); `obs_publish OK` AND the final
   file exists under `otr\obs\` (Lesson 7 -- `obs_publish OK` alone is not proof).
3. **Stale-source leg:** NOT a live leg (1.6, 7.5) -- proven deterministically over
   a captured frozen ledger.

Any failure in a LIVE run (smoke, soak, or published episode) gets an append-only
`PBUG-<YYYYMMDD>-<NN>` entry in docs/PROD_BUG_LOG.md using the template at
:15-26 (surfaced / symptom / root cause / fix / verify idea / bible-worthy /
confidence / status). Dev-only catches are fixed and tested but NOT logged. Bible
promotion happens at the operator-triggered fan-out (Lesson 9).

### 9.4 Sprint receipt (fill at close)

```text
SPRINT RECEIPT: PASS | FAIL
scope:                    dynamic_story visual direction (vd-1)
authoritative_writers:    OTR_DynamicStoryDirection -> meta.visual_direction
durable_artifacts:        meta.visual_direction (+ semantic/artifact seals);
                          visual_direction_semantic_sha256 on image rows + video creative sidecars
canonical_workflow_hash:  <after the node 96 / link 284 delta>
focused_tests:            <9.1 items 1-14>
full_suite:               <passed/skipped/xfail>
bug_bible:                <result>
model_pairings:           mistral-nemo | gemma-4-E4B | <cloud creative lane>
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

CLOSED by r1: sentinel = code-side, no placeholder pack (D1); no reroll in v1
(D4); wardrobe orthogonal (D7); per-shot notes artifact-side with mandatory
MetaBrief consumption (D3); talking lane pinned to the safety base (D6).

CLOSED by r2: **D2** = `creative` slot (fable2 precedent, 4.2). **D5** =
still_word typography/backdrop pinned entirely to the safety base in v1 (2.4).
**D8** = the budget equation + the P-A/P-B split (4.4). **D9** = Python constant
module (8.6). **D10** = an `is_dynamic` flag on the resolved `VisualStyle` (5.4).

Still open -- implementation choices, not design forks:

- **I1 -- where the must-fit slot_fn helper lives** (4.3): lift the writer's
  wrapper (nodes/OTR_LedgerScriptWriter.py:664-699) into a shared module, or
  reimplement the guard in the direction module. Requirement: fail-loud on
  overflow, never truncate.
- **I2 -- constrained generation on/off for P-A** (4.2): lm-format-enforcer binds
  the schema at token level on the local HF lane and is the strongest defense
  against the wrong-depth class, but it costs latency on an 11-string nested
  object. Codex measures and decides; the typed-repair ladder is mandatory either
  way.
- **I3 -- P-B batch size** (4.4): 15 mirrors ShotLock. Codex may tune it from the
  measured token budget; the batching itself is not optional.

---

END. Docs-only deliverable; no code, tests, prompts, registries, or
workflows/otr_canonical.json were touched. Codex owns everything in section 8.
Kibitz artifacts: kibitz-runs/2026-07-12-dynamic-story-visual/{r1,r2}/
(driver_anchor.md, codex.md, antigravity.md, judgment.md, final.md).
