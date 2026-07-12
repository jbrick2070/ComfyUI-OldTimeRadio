# dynamic_story -- Story-Derived Visual Direction: Scoped Design (docs-only)

Date: 2026-07-12 (rev 2 -- hardened by r1 kibitz: codex gpt-5.6-sol @ ultra +
antigravity gemini-3.5-pro; driver anchor + grounding in
kibitz-runs/2026-07-12-dynamic-story-visual/r1/)
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

Every code claim below is grounded `file:line` against the real Windows repo at
the time of writing (v2.0-alpha working tree, 2026-07-12).

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
  grammar slug (from the story contract), `meta.visual_style` is the visual
  prompt-pack selector; the two are never crossed
  (nodes/OTR_LedgerScriptWriter.py:216-222, 2249-2261).

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
  they never see `lines`/`cast`/`beats` (r1 finding; drives section 4.3).
- The loaded packs are CACHED module-globally (`_STYLES`,
  nodes/_otr_visual_styles.py:170, 355-359). A per-episode dynamic pack must
  NEVER enter that cache (r1 antigravity S3; section 5).
- The geometry-vs-look law (nodes/_otr_visual_styles.py:7-9): packs own ONLY
  look/subject vocabulary; framing/headroom/mouth-safety GEOMETRY stays in
  Python. `dynamic_story` must obey the same law -- and r1 tightened it
  further: the LLM does not even author every pack LOOK field (section 2.2).

### 1.3 The story freeze boundary (what "final accepted story" means)

- OTR_LedgerScriptWriter -> OTR_LedgerFreezeCascade; Phase 10 runs the
  deterministic gap audit and, on success, stamps
  `meta.cleanup_locked = True`, `meta.freeze_timestamp` (ISO-8601 UTC), and
  `meta.freeze_verdict = frozen_clean | frozen_with_warns`
  (nodes/_otr_ledger_freeze.py:758-819, stamps at 806-811). Critical gaps raise
  `FreezeAssertionError` and stamp `needs_full_rerun`
  (nodes/_otr_ledger_freeze.py:787-795).
- Freeze locks story CONTENT, not the whole rows: post-freeze extension is an
  established pattern. OTR_CastLock assigns voice presets AFTER the freeze
  (acknowledged inside the freeze audit itself,
  nodes/_otr_ledger_freeze.py:493-502); OTR_ShotLock overlays per-line audio
  timing IN PLACE (nodes/otr_shot_lock.py:169-221) and stamps a whole
  `ledger['video']` section + `meta.video_revision`
  (nodes/otr_shot_lock.py:1104-1131). Consequence (r1 must-fix): the RAW
  `lines`/`cast` arrays are NOT stable post-freeze -- any staleness check must
  hash an authored-content PROJECTION, never the raw rows (section 2.3).
- The cascade unloads a locally-resident writer LLM in a `finally` block and
  stamps `meta.freeze_unload_ok` (nodes/OTR_LedgerFreezeCascade.py:377-387,
  453-478) -- the VRAM handoff any post-freeze LLM node must reproduce
  (section 4.5).

### 1.4 Canonical wiring (workflows/otr_canonical.json, live graph)

Grounded by walking `nodes[]`/`links[]` of the real file (litegraph schema,
single-line JSON; node ids below are the live ids):

- Writer (node 1) `script_json` -> link 230 -> FreezeCascade (node 62).
  The writer's `visual_style` widget is combined widget slot 24
  (`widgets_values[24] == "sci_fi_radio"` in the canonical file; input name
  `visual_style` at in[24]).
- FreezeCascade out[1] `script_json` -> link 255 -> OTR_MetaBriefImagePromptGen
  (node 89) in[0], and -> link 252 -> OTR_ShotLock (node 90) in[0].
- OTR_VideoDirector (node 87) `video_policy_json` -> link 270 ->
  OTR_ImageDirector (node 88) in[0], and -> link 251 -> ShotLock in[2].
- ImageDirector `image_policy_json` -> link 254 -> MetaBrief in[1], and ->
  link 257 -> OTR_ImageGenDispatcher (node 91) in[1].
- MetaBrief out[0] `image_prompts_json` -> link 258 -> Dispatcher in[2].
- ShotLock out[0] `patched_ledger_json` -> link 256 -> Dispatcher in[0];
  ShotLock out[4] `episode_id` -> link 268 -> Dispatcher in[4].
- OTR_EpisodeAssembler (node 7) out[3] `audio_done` -> link 253 -> ShotLock
  in[1] and -> link 259 -> Dispatcher gate_in.
- Dispatcher out[0] -> link 260 -> OTR_VideoRenderBatch (node 92) in[0];
  Dispatcher out[1] `image_done` -> link 267 -> RenderBatch in[2].

The frozen ledger fans out on FreezeCascade out[1]; every visual composer sees
`meta.visual_style` on that wire. The Dispatcher receives its ledger via
ShotLock's whole-ledger re-serialization (parse at nodes/otr_shot_lock.py:1040,
re-emit at :1142), so a meta artifact stamped upstream reaches it with no
extra wire (r1 anchor A1).

### 1.5 Where the style is actually consumed (resolve-once contract)

- Image prompts: `derive_image_prompts` resolves the style ONCE at entry --
  `_vstyle = _resolve_style(meta)` -- and threads it down; "an unknown
  meta['visual_style'] stops the episode HERE, before any prompt is composed"
  (nodes/otr_meta_brief_image_prompt.py:1601-1609). Consumers of the threaded
  pack include: portrait fallback + LLM instruction builders
  (nodes/otr_meta_brief_image_prompt.py:1148-1238), beat-aware character scene
  builder (:1241-1366), radio-host prompts (`build_radio_host_prompt`,
  :356-444), mesh fodder + background plate (:1437-1496), still_word cards
  (`compose_still_word_prompt`, :942-1013), and the aspect anchors
  (`_style_anchor_for_aspect`, :171-188).
- Video prompts: ShotLock's M4 creative derivation finishes every talking-head
  prompt through `finish_visual_prompt(meta, text_prompt)` -- but that call
  sits inside a BARE `except Exception: pass`
  (nodes/otr_shot_lock.py:626-636), so today a style-resolution failure there
  is silently swallowed. r1 must-fix: on the dynamic lane the style must be
  resolved BEFORE the beat loop, outside that fail-soft block (section 6.2).
- The helper family (`_resolve_style` :330, `get_era_tail` :344,
  `compose_still_prompt` :590, `finish_visual_prompt` :667 in
  nodes/_otr_story_brief_helpers.py) resolves the pack from meta.
  render_driver reads pack `motion_registers`
  (nodes/_otr_video_engines/render_driver.py, visual_style reads).
- LOOK-AUTHORITY REALITY (r1 codex M3): the era tail is BRIEF-FIRST -- the
  brief's `atmosphere_line`/`visual_palette`/lighting win, and the pack's
  `era_tail` is only the fallback default
  (nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428). Any design that
  authors a new palette in the pack alone would be silently shadowed by the
  brief. Section 5.3 resolves the precedence for the dynamic lane.
- ShotLock stamps per-shot `creative` sidecars with `prompt_hash` +
  `request_hash` into `ledger['video'].shots[]`
  (nodes/otr_shot_lock.py:637-647, 913-948).

### 1.6 Image dispatch, cache, and persistence (the replay substrate)

- Dispatcher cache key: `request_cache_key(role, object_id, prompt_hash, seed,
  engine_id, engine_version, kind, w, h)` -- "a change in ANY field -> new key
  -> regen" (nodes/otr_image_gen_dispatcher.py:117-129). Seeds:
  `resolve_object_seed` (request_hash mode; pinned bookend seed) (:132-162).
- Results land in `ledger["images"] = {image_revision, images[], cache_index}`
  and are persisted to the DISK ledger via
  `stamp_durable(sections={"images": ...})`
  (nodes/otr_image_gen_dispatcher.py:796-826;
  nodes/production_ledger.py:408-422).

### 1.7 Ledger save/merge ownership (what survives, what gets dropped)

`Ledger.save()` merges in-memory state with the on-disk ledger
(BUG-LOCAL-108 lineage, nodes/production_ledger.py:1287-1346, 1357-1513):

- TOP-LEVEL: only `TOP_PRESERVE = (schema_version, audio_gates, transitions,
  radio_bookend_path)` is copied forward from disk
  (nodes/production_ledger.py:1387-1393). An unknown top-level key present on
  disk but absent from a later in-memory save is DROPPED. Top-level is
  therefore NOT ledger-safe for a new artifact.
- META: per-key merge -- disk wins only where in-mem lacks the key or holds an
  empty value; in-mem wins where it has a real value
  (nodes/production_ledger.py:1403-1413). A namespaced meta key written once
  SURVIVES every later save by construction.
- ROWS (lines/clips/music): row-level ownership. `_MERGE_OWNED_ROW_FIELDS`
  (19 owned content/state fields) are never resurrected from disk
  (nodes/production_ledger.py:1441-1459); out-of-band DURABLE renderer fields
  copy forward only on a content-identity match
  (nodes/production_ledger.py:1477-1491, identity at :305-347). NOTE (r1): the
  merge boundary controls DISK COPY-FORWARD only -- it is not a runtime
  mutation validator; story-immutability protection comes from the projection
  comparison in section 6.3.
- The freeze audit hard-requires the fixed top-level list set and per-line
  shape (nodes/_otr_ledger_freeze.py:118-129, 252-404); it does not govern
  namespaced meta keys.
- Source-bank law, same conclusion from the docs side: "Evidence maps and
  authorship receipts live in typed artifacts or namespaced `meta`; the fixed
  line schema contains no ad hoc provenance fields"
  (docs/SOURCE_BANK_PREFLIGHT.md:184-186). Freeze policy selection + receipts
  precedent: docs/SOURCE_BANK_PREFLIGHT.md:187-194, 277-288.

---

## 2. Proposed typed artifact: `meta.visual_direction` (schema `vd-1`)

One JSON object, stamped once per episode, versioned, hash-sealed. It contains
a synthesized visual-style pack (v2 schema) plus the evidence and receipts
around it, so every existing consumer keeps consuming a `VisualStyle` and never
learns a second prompt-composition path.

### 2.1 Shape

```json
{
  "schema_version": "vd-1",
  "created_utc": "2026-07-12T00:00:00+00:00",
  "writer": { "node": "OTR_DynamicStoryDirection", "node_rev": "<git short>" },

  "model_receipt": {
    "slot": "creative",
    "requested_model": "<the stable handle requested>",
    "resolved_model": "<provider-reported concrete model, when available>",
    "runtime_policy": { "...": "policy_from_meta snapshot" },
    "effective_sampling": { "temperature": 0.1, "do_sample": true,
                            "top_p": "<as configured>", "max_new_tokens": 0 },
    "attempts_used": 1,
    "prompt_sha256": "<sha256 of the full derivation prompt>",
    "response_sha256": "<sha256 of the raw accepted model output>"
  },

  "story_binding": {
    "episode_id": "<ledger episode_id>",
    "freeze_verdict": "frozen_clean",
    "freeze_timestamp": "<meta.freeze_timestamp verbatim>",
    "story_projection_sha256": "<canonical sha256 of the PROJECTION, 2.3>",
    "content_mutations": 0
  },

  "rationale": {
    "style_language": "<one-paragraph episode visual thesis>",
    "motifs": [ { "motif": "<recurring visual motif>",
                  "evidence": ["line:l0007"] } ],
    "clue_visual": { "treatment": "<how the clue mechanism reads on screen>",
                     "evidence": ["beat:b003", "line:l0012"] },
    "era_cues": ["<cue>", "..."],
    "composition_notes": ["<non-executable note>", "..."]
  },

  "style_pack": {
    "style_id": "dynamic_story",
    "schema_version": "v2",
    "...": "the COMPLETE v2 field set -- assembled per 2.2 from the vetted
            safety base + the LLM-authored look whitelist -- validated by the
            SAME _validate_row rules (nodes/_otr_visual_styles.py:184+)"
  },
  "authored_fields": ["era_tail", "positive_tail", "..."],
  "field_evidence": {
    "/style_pack/era_tail": { "evidence": ["line:l0003", "meta:episode_title"],
                              "kind": "factual" },
    "/style_pack/portrait_look": { "evidence": ["cast:c01", "beat:b002"],
                                   "kind": "factual" },
    "/rationale/style_language": { "evidence": ["title"],
                                   "kind": "rationale" }
  },

  "shots": [
    { "beat_id": "<line_id>",
      "subject_note": "<what this beat's still should feature, <=240 chars>",
      "mood": "<one register word>",
      "evidence": ["line:<line_id>"] }
  ],

  "semantic_sha256": "<canonical sha256 over style_pack + authored_fields +
                      field_evidence + shots + rationale + story_binding
                      (timestamps and model_receipt EXCLUDED)>",
  "artifact_sha256": "<canonical sha256 of the whole object minus this field>"
}
```

Hash discipline (r1 codex M6): `semantic_sha256` covers only authored content
and its binding -- NO timestamps, NO model receipt -- so "same story, same
direction" is checkable across runs; `artifact_sha256` seals the full envelope
for corruption detection. `created_utc` never contaminates the semantic hash.

### 2.2 Who authors which pack field (r1 codex M4 -- the safety split)

The LLM does NOT author the full v2 pack. The pack is ASSEMBLED from two
sources and then validated whole:

- **Vetted safety base (checked-in, operator-owned; not LLM-touched):**
  `portrait_look_talking` (S4b lip-sync law -- bright/frontal/warm,
  nodes/otr_meta_brief_image_prompt.py:160-168), `announcer_subject_face`,
  `announcer_subject_ltx_mouth` (mouth-vocabulary contract,
  nodes/_otr_visual_styles.py:117-119), `announcer_subject_object`,
  `open_subjects` ({form} templates), `motion_registers` (240-char engine
  budget, :121-123), `non_character_emblem_fallback`, `allow_radio_tails`,
  `forbidden_terms`.
- **LLM-authored look whitelist (the creative surface):** `label`,
  `era_tail`, `positive_tail`, `image_grade_tail`, `broadcast_tail`,
  `portrait_look`, `portrait_instruction_look`, `scene_instruction_look`,
  `radio_object_look`, `plate_look`, `still_word_title_mood_style`, and the
  VALUES of `still_word_typography` / `still_word_backdrop` (the genre
  SELECTOR stays Python-locked,
  nodes/otr_meta_brief_image_prompt.py:997-1002; the LLM authors the selected
  genre's row richly and fills the remaining fixed keys with the episode
  default row).
- `authored_fields` records exactly which fields the LLM wrote; everything
  else provably came from the safety base.
- The anti-geometry lint over authored fields uses a PYTHON-OWNED fixed
  vocabulary (framing/headroom/crop/close-up/mouth-visibility terms), never
  the artifact's own `forbidden_terms` (an LLM must not author its own guard).

### 2.3 The story projection (r1 codex M1 / antigravity M1 -- what gets hashed)

Raw `lines`/`cast` mutate post-freeze by design (timing overlay,
nodes/otr_shot_lock.py:169-221; voice assignment,
nodes/_otr_ledger_freeze.py:493-502). The binding therefore hashes an
IMMUTABLE AUTHORED-CONTENT PROJECTION -- exactly the fields the merge layer
already declares author-owned (nodes/production_ledger.py:1441-1448):

| Array | Projected fields |
|---|---|
| lines | line_id, text, char_id, speaker_role, beat_id, traits, boundary, skip, tts_skip_reason |
| cast  | char_id, name, gender, traits, character_description |
| beats | beat_id, and its line-reference field(s) as shipped |
| meta (whitelist) | episode_title, style, source_bank, visual_style, story_brief_terms |

`story_projection_sha256` = canonical sha256 (pattern:
nodes/production_ledger.py:292-302) over that projection. Writer and readers
compute it with ONE shared pure helper so the two sides can never drift.
Evidence VALUES cited in `field_evidence` resolve against the same projection,
so a cited line's text is covered by the hash.

### 2.4 Evidence ID grammar

`line:<line_id>`, `beat:<beat_id>`, `cast:<char_id>`, `meta:<dotted.path
within the meta whitelist>`, `title`. Every ID must resolve against the bound
projection; an unresolvable ID is a validation error (fail-closed). Every
LLM-authored prompt-bearing field MUST have a `field_evidence` entry (keyed by
JSON pointer); entries are typed `factual` (cites story content) or
`rationale` (creative justification) -- ID existence proves traceability, the
`kind` split keeps the audit honest about what is derivation vs taste
(r1 codex M5). `shots[].beat_id` must match the ShotLock beat-id scheme
(line_id / synthetic `b000_music_open`; nodes/otr_shot_lock.py:279-296,
nodes/otr_meta_brief_image_prompt.py:1016-1023).

### 2.5 Deliberately OUT of vd-1 (r1 cuts, judged)

- `scenes[]` -- no canonical scene key exists on the visual wire; per-beat
  `shots[]` covers the intent. CUT.
- `global.continuity` -- `meta.continuity` already has one owner (the writer's
  continuity ledger, nodes/OTR_LedgerScriptWriter.py:4721-4746) and character
  look continuity is already carried by the cast rows the appearance lookup
  reads (nodes/otr_shot_lock.py:116-153). The direction node CITES those as
  evidence; it never re-authors them. CUT.
- Wardrobe -- `OTR_OUTFIT_LOCK` stays the one wardrobe authority
  (nodes/otr_shot_lock.py:143-153). Orthogonal. CUT (D7 resolved).
- Executable `composition_rules` -- geometry law; survives only as
  non-executable `rationale.composition_notes`. CUT as an authority.
- Reroll/revision machinery -- cut from v1 wholesale (section 6.5).
- Credits/dossier integration -- not part of the first build. CUT.

`rationale.motifs` / `clue_visual` are KEPT (the operator product intent names
them) but they are NON-EXECUTABLE: nothing downstream composes prompts from
them; they exist as the evidence-bound reasoning that produced the pack tails.

---

## 3. Ownership: exactly one writer, read-only consumers

| Actor | Role | Grounding |
|---|---|---|
| OTR_DynamicStoryDirection (NEW node -- not implemented) | SOLE writer of `meta.visual_direction`. Runs post-freeze, reads the frozen ledger read-only, stamps the artifact, forwards patched `script_json`. Pure pass-through when the sentinel is not selected. | Insertion at the FreezeCascade out[1] fan-out (links 255/252, section 1.4) |
| OTR_LedgerScriptWriter | Writes `meta.visual_style` only (now possibly the sentinel value). Never writes `visual_direction`. | nodes/OTR_LedgerScriptWriter.py:3651-3655 |
| `resolve_visual_direction(ledger)` (NEW ledger-aware seam -- not implemented) | The ONE reader that validates the artifact against the ledger and returns the assembled `VisualStyle`. Called at each consumer ENTRY (which all hold the parsed ledger); `_resolve_style`/`get_visual_style` stay meta-only for named packs. | replaces the naive branch; see 4.3 |
| OTR_MetaBriefImagePromptGen | Consumer (portraits, scene stills, radio-host, mesh fodder, plates, still_word) + MANDATORY consumer of `shots[]` notes for beat stills on the dynamic lane. | nodes/otr_meta_brief_image_prompt.py:1601-1609, 2134-2168 |
| OTR_ShotLock | Consumer; on the dynamic lane resolves the style BEFORE the beat loop, outside the fail-soft finisher block. | nodes/otr_shot_lock.py:626-636 (the block to hoist past) |
| render_driver | Consumer of pack `motion_registers` -- which on the dynamic lane come from the SAFETY BASE, so engine motion never varies with LLM output. | section 2.2 |
| OTR_ImageGenDispatcher | Indirect consumer: prompts arrive pre-composed; its cache key already reacts to any prompt change; it receives the artifact via ShotLock's whole-ledger re-serialization. | nodes/otr_image_gen_dispatcher.py:117-129; nodes/otr_shot_lock.py:1040, 1142 |

---

## 4. Lifecycle and storage decision (grounded, not guessed)

### 4.1 Location: `meta.visual_direction`. Nothing else is ledger-safe.

- Top-level is ruled out by the merge code: only `TOP_PRESERVE` survives a
  later in-memory save (nodes/production_ledger.py:1387-1393); the freeze
  audit also pins the top-level shape (nodes/_otr_ledger_freeze.py:118-129).
- Line/cast rows are ruled out by row ownership
  (nodes/production_ledger.py:1441-1459) and the preflight law
  (docs/SOURCE_BANK_PREFLIGHT.md:184-186).
- META survives by construction: per-key merge keeps a real in-memory value
  and back-fills from disk when a later save lacks the key
  (nodes/production_ledger.py:1403-1413) -- exactly how `meta.visual_style`,
  `meta.freeze_*`, `meta.gap_audit_*` persist today.

### 4.2 Write path

1. Preconditions (ALL fail-closed): `meta.visual_style == "dynamic_story"`;
   `meta.cleanup_locked is True`; `meta.freeze_verdict` in
   `{frozen_clean, frozen_with_warns}` (`needs_full_rerun` refuses;
   nodes/_otr_ledger_freeze.py:787-811); non-empty `lines`/`cast`.
2. Compute `story_projection_sha256` FIRST (section 2.3).
3. Run the LLM derivation on the CREATIVE slot (recommended -- this is a taste
   task; final call is D2) via the existing seam
   (`request_slot`/`make_generate_fn` + `policy_from_meta`,
   nodes/otr_shot_lock.py:677-694), bounded attempts (2, matching
   `max_reseed`, nodes/otr_shot_lock.py:507); exhaustion fails closed
   (docs/SOURCE_BANK_PREFLIGHT.md:132-133).
4. Assemble the pack (safety base + authored whitelist, 2.2); validate vd-1 +
   the embedded pack + evidence resolution + the Python geometry lint.
5. Re-compute the projection hash and refuse to stamp on any delta
   (`content_mutations: 0` receipt).
6. Stamp `meta.visual_direction` on the wire ledger AND persist via
   `stamp_durable(meta_updates={"visual_direction": ...})`
   (nodes/production_ledger.py:408-422).

### 4.3 Read path (the r1-corrected seam)

`get_visual_style(meta)` cannot enforce the staleness matrix -- it never sees
the arrays (nodes/_otr_visual_styles.py:378-390). The design therefore adds
ONE ledger-aware function, `resolve_visual_direction(ledger) -> VisualStyle`,
called at each consumer ENTRY, where the parsed ledger is already in hand:
`derive_image_prompts` receives `cast` + `lines` + `meta` from the node
(nodes/otr_meta_brief_image_prompt.py:2134-2168), and ShotLock holds the whole
`led` (nodes/otr_shot_lock.py:1040). The entry resolves ONCE and threads the
returned `VisualStyle` down the existing `style=` parameters -- the
resolve-once contract already in force (:1601-1609). REJECTED alternative
(r1): reading the `peek_ledger()` process singleton inside `get_visual_style`
-- consumers operate on the WIRE ledger, and the singleton can lag or be
absent in that seam.

The dynamic `VisualStyle` instance NEVER enters the module pack cache
(`_STYLES`, nodes/_otr_visual_styles.py:170, 355-359); it is built per
resolve from the artifact (r1 antigravity S3).

### 4.4 Look authority on the dynamic lane (r1 codex M3)

Today the brief outranks the pack for the era tail
(nodes/_otr_story_brief_helpers.py:356-370, 401, 414, 428) -- correct for
named packs (the brief carries the episode's specifics), but on the dynamic
lane it would shadow exactly the palette the LLM just authored, leaving TWO
competing look authorities. Rule: **on the dynamic lane the artifact pack is
the SOLE final-look authority; the brief is evidence INPUT to the direction
LLM, not a runtime override.** Concretely: `get_era_tail` (and the palette
reads inside the still/portrait profiles) must, when the resolved style is
dynamic, return the pack-authored tail instead of the brief-derived one. The
derivation prompt receives the brief verbatim, so brief specifics reach the
final look THROUGH the authored pack, with evidence recorded.

### 4.5 VRAM handoff (r1 codex M8 / antigravity S1)

The direction node reloads an LLM after the cascade deliberately unloaded it
(nodes/OTR_LedgerFreezeCascade.py:377-387, 453-478). The node must mirror
that contract: `unload_llm_if_local_resident()` in a `finally`, a
`meta.visual_direction_unload_ok` stamp on failure, loud logging -- so image
dispatch never starts with a resident writer LLM on the 16 GB card. Lazy
imports throughout (module top level imports stdlib only -- the repo's
import-isolation posture; nodes/_otr_visual_styles.py:11-13).

---

## 5. Dropdown / override semantics

- The writer dropdown gains ONE entry: `dynamic_story`, appended CODE-SIDE as
  a sentinel next to `list_style_ids()` (the `ADD_CUSTOM` idiom,
  nodes/otr_video_director.py:35). The placeholder-pack-file alternative is
  REJECTED (r1): a renderable placeholder pack is a silent-fallback hazard.
- **Sentinel gate fix (r1 must-fix, was understated in rev 1):** the writer's
  run() gate `resolve_visual_style(visual_style)` raises on any id without a
  pack file (nodes/OTR_LedgerScriptWriter.py:3334-3339) -- selecting
  `dynamic_story` today would kill the run BEFORE the story exists. The gate
  must special-case the sentinel (accept it without registry resolution); the
  stamp mechanics stay unchanged (:3651-3655).
- **Explicit named pack always wins, byte-identical:** when
  `meta.visual_style != "dynamic_story"`, the direction node is a pure
  pass-through and every resolver behaves exactly as today. Absent/empty
  `visual_style` keeps resolving to `sci_fi_radio`
  (nodes/_otr_visual_styles.py:386-389).
- **The sentinel VALUE is the only trigger.** Artifact presence never
  activates dynamic styling -- a stale `meta.visual_direction` under a named
  pack is inert (r1 anchor B3).
- Precedence summary: (1) named pack -> current behavior, dynamic machinery
  inert; (2) `dynamic_story` -> artifact mandatory, fail-closed; (3) nothing
  -> production default pack. No env override, no silent fallback between
  lanes.

---

## 6. Failure, stale data, replay, and audit behavior

### 6.1 Fail-closed matrix (enforced in `resolve_visual_direction(ledger)`)

With `meta.visual_style == "dynamic_story"`, ANY of the following aborts the
episode loudly (named error, never a fallback to a named pack):

- `meta.visual_direction` absent or not a dict.
- `schema_version` not in the known set (`vd-1`).
- `story_binding.freeze_timestamp != meta.freeze_timestamp` or episode_id
  mismatch (a re-frozen or foreign ledger).
- Recomputed `story_projection_sha256` (section 2.3) differs from the bound
  value -- authored story content changed after direction. (Projection, not
  raw arrays: post-freeze timing/voice mutations are EXPECTED and must not
  false-fail -- r1 must-fix.)
- `artifact_sha256` mismatch (corruption detection -- the digest lives inside
  the mutable ledger, so this is a checksum, not tamper-proofing).
- Embedded `style_pack` fails v2 validation, or an `authored_fields` entry
  names a safety-base field (the LLM wrote where it must not).
- Any evidence ID fails to resolve; any authored prompt-bearing field lacks a
  `field_evidence` entry.

Postures mirrored from: nodes/otr_image_director.py:428-456,
nodes/otr_shot_lock.py:1053-1061, nodes/_otr_visual_styles.py:367-375.

### 6.2 The ShotLock swallow (r1 must-fix)

ShotLock's prompt finisher wraps `finish_visual_prompt` in a bare
`except Exception: pass` (nodes/otr_shot_lock.py:626-636) -- on the dynamic
lane that would silently convert every 6.1 abort into an unstyled prompt. The
dynamic-lane contract: resolve the style ONCE, before the beat loop, OUTSIDE
any fail-soft block, and pass `style=` into the finisher.
`VisualStyleError` and vd-1 validation errors must propagate.

### 6.3 Story immutability

Protection is the PROJECTION COMPARISON (write-time re-hash refusing to stamp
on delta, 4.2 step 5; read-time re-hash, 6.1) -- NOT the merge ownership
boundary, which only controls disk copy-forward
(nodes/production_ledger.py:1426-1459), and NOT the freeze audit, which has
already run. (r1 correction of rev 1's over-claim.)

### 6.4 Scene-by-scene drift prevention

- ONE pack per episode; the resolve-once threading contract
  (nodes/otr_meta_brief_image_prompt.py:1601-1609) puts the SAME instance
  behind every prompt.
- `shots[].subject_note/mood` are ADDITIVE clauses appended at the same seam
  the beat mood token uses today
  (nodes/otr_meta_brief_image_prompt.py:1004-1008); they can never replace
  pack-level fields, are capped at 240 chars, and are linted against the
  safety-base `forbidden_terms` + the Python geometry vocabulary at
  validation time.
- On the dynamic lane MetaBrief's consumption of `shots[]` is MANDATORY
  (an authored note that is never consumed is dead scope; a beat_id present
  in `shots[]` but absent from the ledger fails validation).
- Engine-safety continuity is structural: talking/mouth/motion/subject fields
  come from the vetted base (2.2), so no per-episode LLM output can degrade
  lip-sync or motion behavior.

### 6.5 Replay, reruns, and the reroll cut (r1 codex M6 / antigravity M4)

- Reroll/revision machinery is CUT from v1. A ComfyUI re-queue re-executes the
  writer and cascade anyway (fresh story, fresh freeze), and a node whose
  inputs are unchanged is served from cache -- so "re-roll just the
  direction" has no state source in v1. When it returns post-v1, the lever is
  a mutating widget (`request_seed`-style) to force the cache miss, plus
  persisted-artifact reuse keyed on
  `story_projection_sha256 + prompt version + requested model`.
- Replay behavior that DOES hold in v1: an unchanged artifact composes
  byte-identical prompts -> dispatcher cache HITs; any artifact change flows
  into `prompt_hash` -> new cache keys -> regeneration
  (nodes/otr_image_gen_dispatcher.py:117-129). Seed scheme untouched
  (:132-162).
- SHOULD (observability): stamp `semantic_sha256` (short form) onto the
  dispatcher's image rows' provenance so an asset links to the exact
  direction that authored its prompt even after a later re-direction
  (today's rows carry prompt/render identity only, :117-129, 796-826).

### 6.6 Debug/audit walk ("why does this shot look like this?")

1. Rendered asset -> `ledger['images'].images[]` row (or
   `ledger['video'].shots[].creative`) -> `prompt` + `prompt_hash`
   (nodes/otr_image_gen_dispatcher.py:796-826;
   nodes/otr_shot_lock.py:940-947).
2. Prompt tail vocabulary -> `style_pack` fields (verbatim in the artifact) +
   the per-beat `shots[]` note keyed by the same beat_id;
   `authored_fields` says whether the LLM or the safety base wrote each.
3. Each authored field -> its `field_evidence` pointer entry -> the exact
   frozen projection content that motivated it (factual vs rationale typed).
4. `model_receipt` gives requested + resolved model, slot, effective sampling
   (the local lane samples -- temp 0.1 floor, do_sample=True, no seed:
   nodes/otr_shot_lock.py:687-692 -- so "attempts", not "reseeds");
   `story_binding` proves which frozen story it derived from.

Pixels -> prompt -> pack field -> evidence -> frozen story text.

---

## 7. Likely future code/workflow surfaces -- ALL "not implemented"

1. **NEW node `nodes/otr_dynamic_story_direction.py`
   (`OTR_DynamicStoryDirection`)** -- not implemented. Full ComfyUI node
   contract: `INPUT_TYPES` (`script_json` forceInput; optional `gate_in`),
   `RETURN_TYPES/RETURN_NAMES` (`patched_ledger_json`, `direction_report`,
   `done` -- the standard opaque ordering STRING, the CastLock/ShotLock
   idiom, nodes/otr_shot_lock.py:966-969), `FUNCTION`, `CATEGORY`,
   `VALIDATE_INPUTS`. Registration in the package init's `_NODE_MODULES` /
   class + display mappings (`__init__.py`). Lazy heavy imports; VRAM
   teardown per 4.5.
2. **Workflow JSON delta (same change as the node code, CLAUDE.md
   section 0)** -- not implemented. THREE link records, not two: one NEW link
   FreezeCascade(62) out[1] -> Direction in[0], plus links 255/252 re-sourced
   from Direction out[0] to MetaBrief(89) in[0] and ShotLock(90) in[0];
   update both nodes' `inputs[].link` and the source nodes' `outputs[].links`
   fan-out lists + `last_node_id`/`last_link_id`. `widgets_values`
   append-only. Re-validate: OTR_WorkflowValidator + JSON round-trip +
   link/widget audit.
3. **`nodes/_otr_visual_direction.py`** (pure/stdlib, lazy) -- not
   implemented. vd-1 schema validation, the story projection + canonical
   hashing helper (shared by writer and readers), evidence resolution,
   geometry lint, pack assembly from safety base + whitelist.
4. **`nodes/_otr_visual_styles.py`** -- not implemented. Code-side sentinel
   exposure for the dropdown; `visual_style_from_payload(dict)` funneling
   through `_validate_row`; dynamic instances bypass `_STYLES`.
5. **`nodes/_otr_story_brief_helpers.py`** -- not implemented.
   `resolve_visual_direction(ledger)` entry seam + the dynamic-lane look
   precedence in `get_era_tail`/palette reads (4.4).
6. **`nodes/OTR_LedgerScriptWriter.py`** -- not implemented. Sentinel-aware
   dropdown + run() gate (5).
7. **`nodes/otr_shot_lock.py`** -- not implemented. Hoist style resolution
   above the beat loop on the dynamic lane (6.2).
8. **Vetted safety-base pack data** (checked-in, likely
   `nodes/visual_styles/_dynamic_safety_base.json` or a Python constant --
   Codex's call; if a JSON file lives in `visual_styles/` it must be excluded
   from the registry sweep, which currently rejects unexpected non-pack files,
   nodes/_otr_visual_styles.py:329-336) -- not implemented.

---

## 8. Focused test + live-smoke plan (for Codex to implement)

Unit (CPU, `OTR_TEST_MODE=1`, injected `llm_fn` -- the established pattern,
nodes/otr_shot_lock.py:499-528):

1. **Schema round-trip:** valid `vd-1` validates; each required field's
   absence fails named; embedded pack goes through `_validate_row` verbatim;
   an `authored_fields` entry naming a safety-base field fails.
2. **Fail-closed matrix:** one test per row of 6.1 -- all raise; NONE fall
   back to a named pack.
3. **Inert-path byte-identity, PARAMETERIZED over every registered pack +
   absent/default style** (not just sci_fi_radio): serialized ledger, composed
   prompts, prompt_hashes, and dispatcher request keys are byte-identical
   before vs after the feature. Boundaries defined at those four surfaces
   BEFORE any GPU-asset comparison.
4. **Projection stability:** timing overlay
   (nodes/otr_shot_lock.py:169-221-style mutations) + voice-preset assignment
   do NOT change `story_projection_sha256`; a one-word text edit DOES.
5. **Merge survival:** stamp, save from an in-memory ledger lacking the key,
   assert restoration (nodes/production_ledger.py:1403-1413; mirror of
   test_ledger_merge_ownership).
6. **Story immutability:** a mutating fake `llm_fn` scenario is refused with
   `content_mutations != 0`.
7. **Determinism of hashing:** same fixture + same injected output =>
   identical `semantic_sha256`; `created_utc` variation changes ONLY the
   envelope hash.
8. **Drift + geometry guards:** shots[] note with an exclusion/geometry term
   fails; shots[] beat_id not in the ledger fails; all pack-level look tokens
   identical across beats.
9. **ShotLock propagation:** with dynamic selected and a broken artifact, the
   ShotLock path RAISES (does not swallow) -- pins the 6.2 hoist.
10. **Look-QA rubric fixtures (r1 codex should-fix):** a small multi-genre
    fixture set (noir / western / sci-fi story projections) with an operator
    rubric -- specificity, palette/medium coherence, recurring identity,
    talking-lane safety, measurable difference from the fixed-pack control.
    Schema validity alone does not prove the direction is any good.

Suite discipline: full regression + Bug Bible after every code chunk
(CLAUDE.md section 3); Three-File Contract for any new bug class.

Live smoke (5080, headless :8000, reset per CLAUDE.md section 4):

1. **Control leg:** 30-word episode, `visual_style="sci_fi_radio"` -- prompts
   byte-match the pre-feature baseline.
2. **Dynamic leg:** `visual_style="dynamic_story"` -- artifact present on the
   disk ledger with receipts; VRAM returns to baseline after the direction
   stamp (the 4.5 teardown) BEFORE image dispatch; still prompts carry the
   authored vocabulary; assets at `otr\episodes\<ep>\` (Test-Path),
   `obs_publish OK`.
3. **Stale-evidence leg:** edit one frozen line's text on the wire copy ->
   dynamic resolve aborts loudly (6.1 in vivo).
4. **Replay leg:** re-queue unchanged -> dispatcher reports cache HITs (6.5).

---

## 9. Unresolved decisions for Codex

Resolved by r1 (no longer open): sentinel = code-side, no placeholder pack
(old D1); no reroll in v1 (old D4); wardrobe orthogonal (old D7); per-shot
notes artifact-side with mandatory MetaBrief consumption (old D3); talking
lane pinned to the safety base (old D6 -- strongest form adopted).

Still open:

- **D2 -- LLM slot.** Recommendation: `creative` (taste task; codex r1
  concurs). Alternative: `technical` for schema-heavy structured output.
  Whichever is chosen, the receipt records requested + resolved identity
  (virtual vs provider-resolved ids demonstrably differ on the OpenRouter and
  Google lanes).
- **D5 -- still_word typography scope.** Design says: Python keeps the genre
  SELECTOR; the LLM authors the selected genre's typography/backdrop VALUES
  and fills the other fixed keys with the default row (2.2). Confirm, or pin
  still_word entirely to the safety base for v1.
- **D8 -- size budgets.** Proposed: notes <= 240 chars (mirrors the motion
  budget, nodes/_otr_visual_styles.py:121-123), `shots[]` <= one row per
  non-skipped line, total artifact <= 64 KB canonical, plus an output-token
  reservation sized from line count (docs/SOURCE_BANK_PREFLIGHT.md:140-141).
  Confirm numbers.
- **D9 (new) -- safety-base packaging.** JSON file under `visual_styles/`
  (needs a registry-sweep exemption, nodes/_otr_visual_styles.py:329-336) vs
  a Python-side constant module. Codex's call at build time.
- **D10 (new) -- dynamic-lane era-tail mechanics.** 4.4 fixes the PRECEDENCE;
  the concrete seam (a style-aware branch inside `get_era_tail` vs a
  pack-flag the helper honors) is an implementation choice.

---

END. Docs-only deliverable; no code, tests, prompts, registries, or
workflows/otr_canonical.json were touched. Codex owns everything in section 7.
r1 kibitz artifacts: kibitz-runs/2026-07-12-dynamic-story-visual/r1/
(driver_anchor.md, codex.md, antigravity.md, judgment.md, final.md).
