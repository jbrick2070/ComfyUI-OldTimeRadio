# dynamic_story -- Story-Derived Visual Direction: Scoped Design (docs-only)

Date: 2026-07-12
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
  fails LOUD (nodes/_otr_visual_styles.py:378-390).
- The geometry-vs-look law (nodes/_otr_visual_styles.py:7-9): packs own ONLY
  look/subject vocabulary; framing/headroom/mouth-safety GEOMETRY stays in
  Python. `dynamic_story` must obey the same law -- the LLM authors LOOK, never
  geometry.

### 1.3 The story freeze boundary (what "final accepted story" means)

- OTR_LedgerScriptWriter -> OTR_LedgerFreezeCascade; Phase 10 runs the
  deterministic gap audit and, on success, stamps
  `meta.cleanup_locked = True`, `meta.freeze_timestamp` (ISO-8601 UTC), and
  `meta.freeze_verdict = frozen_clean | frozen_with_warns`
  (nodes/_otr_ledger_freeze.py:758-819, stamps at 806-811). Critical gaps raise
  `FreezeAssertionError` and stamp `needs_full_rerun`
  (nodes/_otr_ledger_freeze.py:787-795).
- Post-freeze extension is an established pattern, not a violation: OTR_CastLock
  assigns voice presets AFTER the freeze (acknowledged inside the freeze audit
  itself, nodes/_otr_ledger_freeze.py:493-502), and OTR_ShotLock stamps a whole
  `ledger['video']` section + `meta.video_revision` onto the frozen ledger
  (nodes/otr_shot_lock.py:1104-1131). Freeze locks story CONTENT (lines / cast
  / beats), not the ledger's ability to gain new namespaced sections.

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

So the frozen ledger fans out on FreezeCascade out[1]; every visual composer
sees `meta.visual_style` on that wire.

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
  prompt through `finish_visual_prompt(meta, text_prompt)`
  (nodes/otr_shot_lock.py:626-636); the helper family
  (`_resolve_style` :330, `get_era_tail` :344, `compose_still_prompt` :590,
  `finish_visual_prompt` :667 in nodes/_otr_story_brief_helpers.py) resolves
  the pack from meta. render_driver reads pack `motion_registers` (grep:
  nodes/_otr_video_engines/render_driver.py carries `visual_style` reads).
- ShotLock stamps per-shot `creative` sidecars with `prompt_hash` +
  `request_hash` into `ledger['video'].shots[]`
  (nodes/otr_shot_lock.py:637-647, 913-948).

### 1.6 Image dispatch, cache, and persistence (the replay substrate)

- Dispatcher cache key: `request_cache_key(role, object_id, prompt_hash, seed,
  engine_id, engine_version, kind, w, h)` -- "a change in ANY field -> new key
  -> regen" (nodes/otr_image_gen_dispatcher.py:117-129). Seeds:
  `resolve_object_seed` (request_hash mode; pinned bookend seed)
  (:132-162).
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
  (nodes/production_ledger.py:1477-1491, identity at :305-347).
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
a COMPLETE synthesized visual-style pack (v2 schema) plus the evidence and
receipts around it, so every existing consumer keeps consuming a `VisualStyle`
and never learns a second code path.

```json
{
  "schema_version": "vd-1",
  "direction_revision": 1,
  "created_utc": "2026-07-12T00:00:00+00:00",
  "writer": { "node": "OTR_DynamicStoryDirection", "node_rev": "<git short>" },

  "model_receipt": {
    "slot": "technical",
    "model_id": "<meta.technical_model at call time>",
    "runtime_policy": { "...": "policy_from_meta snapshot" },
    "temperature": 0,
    "reseeds_used": 0,
    "prompt_sha256": "<sha256 of the full derivation prompt>",
    "response_sha256": "<sha256 of the raw accepted model output>"
  },

  "story_binding": {
    "episode_id": "<ledger episode_id>",
    "freeze_verdict": "frozen_clean",
    "freeze_timestamp": "<meta.freeze_timestamp verbatim>",
    "lines_sha256": "<canonical sha256 of ledger.lines>",
    "cast_sha256": "<canonical sha256 of ledger.cast>",
    "beats_sha256": "<canonical sha256 of ledger.beats>",
    "content_mutations": 0
  },

  "global": {
    "style_language": "<one-paragraph episode visual thesis>",
    "palette": ["ink black", "sodium amber", "..."],
    "medium": "<e.g. gouache-on-board period illustration>",
    "composition_rules": ["<rule>", "..."],
    "era_cues": ["<cue>", "..."],
    "motifs": [
      { "motif": "<recurring visual motif>", "evidence": ["line:l0007"] }
    ],
    "clue_visual": { "treatment": "<how the clue mechanism reads on screen>",
                     "evidence": ["beat:b003", "line:l0012"] },
    "continuity": {
      "characters": { "<char_id>": { "notes": "<locked look notes>",
                                     "evidence": ["cast:c01"] } },
      "environments": { "<env_key>": { "notes": "<locked environment>",
                                       "evidence": ["meta:story_brief_terms.setting"] } }
    },
    "exclusions": ["<term the episode must never render>", "..."],
    "evidence": { "palette": ["line:l0003", "meta:episode_title"],
                  "medium": ["meta:story_brief_terms.atmosphere"] }
  },

  "style_pack": {
    "style_id": "dynamic_story",
    "schema_version": "v2",
    "label": "<episode-specific label>",
    "...": "EVERY required v2 field (nodes/_otr_visual_styles.py:60-102),
            authored by the LLM, validated by the SAME _validate_row rules"
  },

  "scenes": [
    { "scene_key": "<scene/act key or beat range>",
      "look_delta": "<bounded additive mood language>",
      "evidence": ["beat:b002"] }
  ],
  "shots": [
    { "beat_id": "<line_id>",
      "subject_note": "<what this beat's still/clip should feature>",
      "mood": "<one register word>",
      "evidence": ["line:<line_id>"] }
  ],

  "artifact_sha256": "<canonical sha256 of this object minus this field>"
}
```

Schema rules (all fail-closed at write AND at read):

- `story_binding` hashes use the existing canonical-JSON digest pattern
  (`_canonical_sha256`, nodes/production_ledger.py:292-302).
- `style_pack` must pass the UNMODIFIED v2 validator (`_validate_row`,
  nodes/_otr_visual_styles.py:184+): non-empty rule, exact dict key sets,
  `{form}`/`{base}` placeholder rules, mouth vocabulary, 240-char motion
  budget, forbidden-terms lint. The LLM gets no schema latitude.
- Evidence ID grammar: `line:<line_id>`, `beat:<beat_id>`, `cast:<char_id>`,
  `meta:<dotted.path>`, `title`. Every ID must resolve against the bound frozen
  ledger; an unresolvable ID is a validation error (fail-closed, never
  warn-and-continue).
- `shots[].beat_id` must match the ShotLock beat-id scheme (line_id / synthetic
  `b000_music_open`; nodes/otr_shot_lock.py:279-296,
  nodes/otr_meta_brief_image_prompt.py:1016-1023) so per-shot notes join the
  same rows the image and video planners key by.
- Geometry law: no field of `vd-1` may carry framing/headroom/mouth-safety
  language; the existing Python geometry constants remain the only geometry
  source (nodes/otr_meta_brief_image_prompt.py:124-134,
  nodes/_otr_visual_styles.py:7-9). Enforced by extending the forbidden-terms
  lint over `global` and `scenes`/`shots` string leaves.

---

## 3. Ownership: exactly one writer, read-only consumers

| Actor | Role | Grounding |
|---|---|---|
| OTR_DynamicStoryDirection (NEW node -- not implemented) | SOLE writer of `meta.visual_direction`. Runs post-freeze, reads the frozen ledger read-only, stamps the artifact, forwards patched `script_json`. | Insertion point: FreezeCascade out[1] fan-out (links 255/252, section 1.4) |
| OTR_LedgerScriptWriter | Writes `meta.visual_style` only (may now stamp the value `dynamic_story`). Never writes `visual_direction`. | nodes/OTR_LedgerScriptWriter.py:3651-3655 |
| `_resolve_style` / `get_visual_style` seam | The ONE reader that turns the artifact's `style_pack` into a `VisualStyle` for every composer. | nodes/_otr_story_brief_helpers.py:330; nodes/_otr_visual_styles.py:378-390 |
| OTR_MetaBriefImagePromptGen | Consumer (portraits, scene stills, radio-host, mesh fodder, plates, still_word) + consumer of `shots[].subject_note/mood` for beat stills. | nodes/otr_meta_brief_image_prompt.py:1601-1609 and section 1.5 |
| OTR_ShotLock | Consumer via `finish_visual_prompt` on every talking-head prompt; optional consumer of `shots[]` notes in the M4 derivation instruction. | nodes/otr_shot_lock.py:626-636 |
| render_driver | Consumer of pack `motion_registers` (unchanged -- it reads the resolved pack). | nodes/_otr_video_engines/render_driver.py (visual_style reads) |
| OTR_ImageGenDispatcher | Indirect consumer only: prompts arrive pre-composed; its cache key already reacts to any prompt change. Never reads the artifact directly. | nodes/otr_image_gen_dispatcher.py:117-129 |
| Credits / dossier | Read-only provenance display (may quote `model_receipt` + `global.style_language`). | meta reads only |

No second writer exists. Re-direction (operator wants a re-roll) is a full
replacement stamp by the SAME node with `direction_revision += 1`; consumers
always read the latest whole object -- there is no partial patching of the
artifact.

---

## 4. Lifecycle and storage decision (grounded, not guessed)

**Location: `meta.visual_direction` (namespaced meta key). Nothing else is
ledger-safe.**

- Top-level is ruled out by the actual merge code: `_merge_with_disk` preserves
  only the four `TOP_PRESERVE` keys from disk
  (nodes/production_ledger.py:1387-1393); any other top-level key written by
  one node is silently dropped by the next in-memory `Ledger.save()`. The
  freeze audit also pins the top-level shape to a fixed list set
  (nodes/_otr_ledger_freeze.py:118-129).
- Line/cast rows are ruled out by the ownership model: rows carry only owned
  composition fields + durable render fields
  (nodes/production_ledger.py:1441-1459) and the preflight law forbids ad hoc
  provenance fields on the line schema (docs/SOURCE_BANK_PREFLIGHT.md:184-186).
- META survives by construction: the per-key merge keeps a real in-memory value
  and back-fills from disk when a later save lacks the key
  (nodes/production_ledger.py:1403-1413). This is exactly how
  `meta.visual_style`, `meta.freeze_*`, `meta.gap_audit_*` persist today.

**Lifecycle:**

1. Preconditions (ALL fail-closed): `meta.visual_style == "dynamic_story"`;
   `meta.cleanup_locked is True`; `meta.freeze_verdict` in
   `{frozen_clean, frozen_with_warns}` (`needs_full_rerun` refuses;
   nodes/_otr_ledger_freeze.py:787-811); non-empty `lines`/`cast`.
2. The direction node computes `lines/cast/beats` canonical hashes FIRST, runs
   the LLM derivation (bounded reseeds, temp=0 -- mirroring the V-11
   writer-slot protocol of `_resolve_writer_llm`,
   nodes/otr_shot_lock.py:651-697), validates `vd-1` + the embedded pack,
   re-verifies the story hashes AFTER the call (proving `content_mutations: 0`
   -- the capability-receipt pattern already ratified for fable2's freeze
   boundary), and stamps `meta.visual_direction`.
3. Persistence: stamp on the WIRE ledger (patched `script_json` out) AND to the
   disk ledger via `stamp_durable(meta_updates={"visual_direction": ...})`
   (nodes/production_ledger.py:408-422) so disk-overlay readers and post-run
   forensics see the same object.
4. Consumers read it wherever `_resolve_style(meta)` runs today. The artifact
   is immutable for the rest of the run; a re-roll replaces it wholesale with
   `direction_revision` incremented and a fresh `artifact_sha256`.

**Ordering note (design decision for Codex):** the direction node sits on the
frozen-ledger wire BEFORE MetaBrief and ShotLock (section 7). MetaBrief runs
pre-audio in graph order; ShotLock is gated on `audio_done` (link 253) -- both
therefore see the stamped artifact if the node is wired at the FreezeCascade
fan-out. No audio-side node reads visual style (grep section 1.5), so stamping
immediately post-freeze is safe.

---

## 5. Dropdown / override semantics

- The writer dropdown gains ONE entry: `dynamic_story`. Today the list is
  `list_style_ids()` = pack files on disk
  (nodes/OTR_LedgerScriptWriter.py:2871-2873;
  nodes/_otr_visual_styles.py:362-364), and the run() gate
  `resolve_visual_style(visual_style)` raises on any id without a pack file
  (nodes/OTR_LedgerScriptWriter.py:3334-3339). Two implementation options are
  left to Codex (section 9, D1); both preserve the fail-loud law.
- `meta["visual_style"]` is stamped with the literal `dynamic_story` -- the
  existing single stamping point, unchanged mechanics
  (nodes/OTR_LedgerScriptWriter.py:3651-3655).
- **Explicit named pack always wins, byte-identical:** when
  `meta.visual_style != "dynamic_story"`, the direction node is a PURE
  PASS-THROUGH (no LLM call, no stamp, `script_json` forwarded verbatim) and
  `_resolve_style` behaves exactly as today. The dynamic path is reachable ONLY
  through the explicit sentinel. Absent/empty `visual_style` keeps resolving to
  `sci_fi_radio` (nodes/_otr_visual_styles.py:386-389) -- the default is NOT
  dynamic.
- `_resolve_style`/`get_visual_style` gains one branch: id == `dynamic_story`
  => build the `VisualStyle` from `meta.visual_direction.style_pack` through
  the SAME validator; anything missing/stale => raise (section 6). All other
  ids: unchanged code path.
- Precedence summary: (1) named pack selected -> current behavior, dynamic
  machinery inert; (2) `dynamic_story` selected -> artifact mandatory,
  fail-closed; (3) nothing selected -> production default pack. There is no
  environment override and no silent fallback between lanes (repo no-fallback
  law, e.g. nodes/otr_meta_brief_image_prompt.py:387-390).

---

## 6. Failure, stale data, replay, and audit behavior

### 6.1 Fail-closed matrix (read side, enforced in the `_resolve_style` branch)

With `meta.visual_style == "dynamic_story"`, ANY of the following aborts the
episode loudly (named error, no fallback to sci_fi_radio):

- `meta.visual_direction` absent or not a dict.
- `schema_version` not in the known set (`vd-1`).
- `story_binding.freeze_timestamp != meta.freeze_timestamp` or
  `story_binding.episode_id` mismatch (a re-frozen or foreign ledger).
- Recomputed `lines/cast/beats` canonical hashes differ from
  `story_binding.*_sha256` (story changed after direction -- stale evidence).
- `artifact_sha256` mismatch (tampered/corrupted artifact).
- Embedded `style_pack` fails v2 validation.
- Any evidence ID fails to resolve against the bound ledger.

This mirrors the repo's existing fail-closed postures: the ImageDirector's
required wired policy (nodes/otr_image_director.py:428-456), ShotLock's
policy_version gate (nodes/otr_shot_lock.py:1053-1061), and the unknown-style
hard error (nodes/_otr_visual_styles.py:367-375).

### 6.2 Story immutability

The direction node never mutates `lines`/`cast`/`beats`: (a) it re-hashes the
three arrays after the LLM call and refuses to stamp on any delta
(`content_mutations: 0` receipt); (b) even a buggy write of owned row fields
would be caught by the merge ownership boundary
(nodes/production_ledger.py:1441-1459) and by Phase-10-class invariants
(nodes/_otr_ledger_freeze.py:252-404). Visual generation consumes text; it
never edits it (LLM-first law: Python judges, the LLM writes -- and here the
LLM writes VISUAL direction only).

### 6.3 Scene-by-scene drift prevention

- ONE global `style_pack` per episode; the resolve-once threading contract
  already in production (entry-resolved `_vstyle` threaded to every helper,
  nodes/otr_meta_brief_image_prompt.py:1601-1609) guarantees every prompt in
  the episode is finished through the SAME pack instance.
- `scenes[].look_delta` and `shots[].subject_note/mood` are ADDITIVE clauses
  only: consumers append them to the composed prompt at the same seam the beat
  mood token uses today (nodes/otr_meta_brief_image_prompt.py:1004-1008); they
  can never replace pack-level palette/medium/grade fields.
- Every additive clause is linted against `global.exclusions` + the pack
  `forbidden_terms` at artifact-validation time, so a per-scene note cannot
  smuggle a contradictory look.
- Continuity is pinned where identity lives: `continuity.characters` keys are
  `char_id`s (the same alias-safe join the appearance lookup uses,
  nodes/otr_shot_lock.py:116-153), so a character's locked look follows the
  row, not a display name.

### 6.4 Cache and replay

- No new cache machinery. The dispatcher key already contains `prompt_hash`
  (nodes/otr_image_gen_dispatcher.py:117-129): an unchanged artifact composes
  byte-identical prompts -> cache HITs; a re-rolled artifact changes prompts ->
  new keys -> regeneration. Seeds are untouched
  (`resolve_object_seed`, :132-162), preserving episode reproducibility under
  `OTR_CAST_SEED`-style overrides.
- ShotLock's per-shot `prompt_hash`/`request_hash` sidecars
  (nodes/otr_shot_lock.py:637-647) give the video lane the same property.

### 6.5 Debug/audit walk ("why does this shot look like this?")

1. Rendered asset -> `ledger['images'].images[]` row (or
   `ledger['video'].shots[].creative`) -> `prompt` + `prompt_hash`
   (nodes/otr_image_gen_dispatcher.py:796-826; nodes/otr_shot_lock.py:940-947).
2. Prompt tail vocabulary -> `meta.visual_direction.style_pack` fields
   (verbatim in the artifact) + the per-beat `shots[]` note keyed by the same
   beat_id.
3. Each pack/global field -> its `evidence` ID list -> the exact frozen
   line/beat/cast row text that motivated it.
4. `model_receipt` says which model on which slot produced it, with prompt and
   response hashes; `story_binding` proves which frozen story revision it was
   derived from; `created_utc`/`direction_revision` order multiple rolls.

That is the full deterministic chain: pixels -> prompt -> pack field ->
evidence -> frozen story text.

---

## 7. Likely future code/workflow surfaces -- ALL "not implemented"

Marked explicitly; sizes are indicative for Codex's planning only.

1. **NEW node `nodes/otr_dynamic_story_direction.py`
   (`OTR_DynamicStoryDirection`)** -- not implemented. In: `script_json`
   (forceInput, from FreezeCascade out[1]), optional `gate_in`. Out:
   `patched_ledger_json`, `direction_report`, `done` (opaque STRING gate,
   mirroring the ShotLock/CastLock done-gate idiom,
   nodes/otr_shot_lock.py:966-969). Pass-through when the sentinel is not
   selected. LLM via the existing slot seam
   (`request_slot`/`make_generate_fn`, nodes/otr_shot_lock.py:677-694) -- no
   new model_id widget (V-11).
2. **Workflow JSON delta (same change as the node code, CLAUDE.md section 0)**
   -- not implemented. Rewire links 255 and 252 so FreezeCascade out[1] feeds
   the new node and its `patched_ledger_json` feeds MetaBrief(89) in[0] and
   ShotLock(90) in[0]. `widgets_values` append-only; re-validate with
   OTR_WorkflowValidator + link/widget audit.
3. **`nodes/_otr_visual_styles.py`** -- not implemented. (a) Expose the
   dropdown sentinel per D1; (b) a `visual_style_from_payload(dict)`
   constructor that funnels an in-memory pack dict through the SAME
   `_validate_row` path (no second validator).
4. **`nodes/_otr_story_brief_helpers.py`** -- not implemented. The
   `_resolve_style`/`get_visual_style` dynamic branch + the fail-closed matrix
   of section 6.1.
5. **`nodes/OTR_LedgerScriptWriter.py`** -- not implemented. Dropdown choice
   list + the run() gate accept the sentinel (stamp mechanics unchanged).
6. **Artifact validation module** (likely `nodes/_otr_visual_direction.py`,
   pure/stdlib, lazy -- registry posture of `_otr_visual_styles.py:11-13`) --
   not implemented. Schema, evidence-ID resolution, hash sealing, receipts.
7. **Optional consumer enrichments** -- not implemented, separately gateable:
   MetaBrief beat stills consume `shots[].subject_note/mood`; ShotLock's M4
   batch instruction quotes `global.style_language`. Both are additive-clause
   only (section 6.3).

---

## 8. Focused test + live-smoke plan (for Codex to implement)

Unit (CPU, `OTR_TEST_MODE=1`, injected `llm_fn` -- the established pattern,
nodes/otr_shot_lock.py:499-528):

1. **Schema round-trip:** valid `vd-1` validates; each required field's
   absence fails with a named error; embedded pack reuses `_validate_row`
   verbatim (assert same exception types as
   nodes/_otr_visual_styles.py:184+).
2. **Fail-closed matrix:** one test per row of section 6.1 (absent artifact,
   version skew, freeze-timestamp skew, lines-hash skew, artifact-hash skew,
   bad pack, dangling evidence ID) -- all raise; NONE fall back to
   sci_fi_radio.
3. **Byte-identity when a named pack is selected:** golden-prompt comparison
   of `derive_image_prompts` + `compose_still_word_prompt` +
   `build_radio_host_prompt` output with `visual_style="sci_fi_radio"` before
   vs after the feature lands (the sentinel machinery must be provably inert).
4. **Merge survival:** stamp `meta.visual_direction`, run a `Ledger.save()`
   cycle from an in-memory ledger lacking the key, assert the disk merge
   restores it (per-key rule, nodes/production_ledger.py:1403-1413); mirror of
   `test_ledger_merge_ownership`.
5. **Story immutability:** direction pass over a fixture ledger leaves
   `lines/cast/beats` hashes unchanged; a mutating fake `llm_fn` scenario is
   refused with `content_mutations != 0`.
6. **Determinism:** same frozen fixture + same injected `llm_fn` output =>
   identical `artifact_sha256`; identical downstream `prompt_hash`es across
   two runs.
7. **Drift guard:** two beats in different scenes share every pack-level look
   token; a `look_delta` containing an `exclusions` term fails validation.
8. **Evidence resolution:** every grammar form (`line:`, `beat:`, `cast:`,
   `meta:`, `title`) resolves; unknown line_id fails.

Suite discipline: full regression + Bug Bible after every code chunk
(CLAUDE.md section 3); Three-File Contract if any new bug class is minted.

Live smoke (5080, headless :8000, reset per CLAUDE.md section 4):

1. **Control leg:** 30-word episode, `visual_style="sci_fi_radio"` -- prompts
   and assets byte-match the pre-feature baseline.
2. **Dynamic leg:** same seed envelope, `visual_style="dynamic_story"` --
   assert `meta.visual_direction` present on the disk ledger with receipts;
   every still row's prompt carries the artifact's pack vocabulary; assets at
   `otr\episodes\<ep>\` (Test-Path), `obs_publish OK`.
3. **Stale-evidence leg:** hand-edit one frozen line on the wire copy ->
   dynamic resolve must abort loudly (proves 6.1 in vivo).
4. **Replay leg:** re-queue the dynamic episode unchanged -> dispatcher
   reports cache HITs (proves 6.4).

---

## 9. Unresolved decisions for Codex

- **D1 -- Sentinel mechanics for the dropdown.** (a) Append `dynamic_story` as
  a code-side sentinel next to `list_style_ids()` and teach the writer gate to
  accept it, or (b) ship a `nodes/visual_styles/dynamic_story.json` pack whose
  fields are placeholders and OVERLAY the artifact's pack at resolve time.
  (a) keeps "packs on disk are real"; (b) keeps the dropdown/gate code
  untouched but weakens the fail-loud story (a placeholder pack could
  accidentally render). Recommend (a); decide.
- **D2 -- LLM slot.** `technical` (mirrors ShotLock M4,
  nodes/otr_shot_lock.py:677-685) vs `creative` (this IS a taste task). One
  slot must be named in the receipt either way.
- **D3 -- Where per-shot notes live at render time.** Artifact-only (MetaBrief
  reads them at compose time) vs ALSO mirrored into
  `ledger['video'].shots[].creative` by ShotLock for the video lane. Mirroring
  duplicates data but makes the video sidecar self-contained.
- **D4 -- Re-roll policy.** Bounded reseeds inside one stamp (recommend 2,
  matching `max_reseed`, nodes/otr_shot_lock.py:507) and then hard abort -- or
  an operator-facing re-direction widget. Exhaustion must fail closed
  (docs/SOURCE_BANK_PREFLIGHT.md:132-133).
- **D5 -- Interaction with the still_word per-episode genre lock.** The genre
  SELECTOR is deliberately Python (operator lettering-consistency directive,
  nodes/otr_meta_brief_image_prompt.py:997-1002). Does dynamic_story's pack
  merely supply the typography VALUES (recommended -- keeps the lock), or may
  the LLM also pick the genre key?
- **D6 -- Talking-lane conservatism.** Talking portraits' only style surface is
  `portrait_look_talking` (S4b lip-sync law,
  nodes/otr_meta_brief_image_prompt.py:160-168). Recommend the artifact's
  authored value be additionally linted for darkness/palette terms (the proof8
  failure class) or pinned to the conservative default; decide the lint list.
- **D7 -- Wardrobe lock overlap.** `OTR_OUTFIT_LOCK` already LLM-locks
  per-character wardrobe (nodes/otr_shot_lock.py:143-153). Fold wardrobe into
  `continuity.characters` (one authority) or keep the two features orthogonal?
- **D8 -- Artifact size budget.** meta rides every serialized ledger wire; a
  420-word episode's `shots[]` could be sizeable. Cap (e.g. notes <= 240 chars,
  mirroring the motion-register budget, nodes/_otr_visual_styles.py:121-123)
  or leave unbounded?

---

END. Docs-only deliverable; no code, tests, prompts, registries, or
workflows/otr_canonical.json were touched. Codex owns everything in section 7.
