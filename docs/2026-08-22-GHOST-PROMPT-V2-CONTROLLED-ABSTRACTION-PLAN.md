# Ghost Prompt v2 -- Controlled Abstraction Plan

Date: 2026-08-22
Status: implementation-ready for Opus; R1--R3 completed and the operator stopped the remaining R4 panel
Problem source: `docs/2026-08-22-GHOST-PROMPT-PROBLEM-STATEMENT.md`

## 1. Outcome

Ghost Signal should produce a short, drawable, beat-specific SD1.5 prompt across every engine that declares `prompt_profile="ghost_signal_v1"`, including the official AnimateDiff v3 peer, without pretending that this lane can maintain a photographic face. The episode should feel intentionally related from beat to beat, but not show the same mediocre person in every shot.

The visual grammar is **controlled abstraction**:

- a stable low-bandwidth motif per character (silhouette, color, and/or prop);
- a changing symbolic or kinetic visual derived from each ledger beat;
- three coordinated representations: `figure`, `object`, and `signal`;
- no face-identity promise, no close-up floor, and no pure random prompt lottery.

The existing style cue and effective negative are held fixed. Live operator review says anime looks anime and archival looks archival. Richer positive style tails are an explicit operator-owned **WONTFIX for this sprint**: this work must not conflate good style conditioning with bad content authoring.

## 2. Grounded defect

The bad examples are not CLIP truncation. A probe of the eight live Ghost requests in `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\_shared\state\node_episode_input.json` measured roughly 40--65 SD1 tokenizer tokens including special tokens. ComfyUI's installed SD1 encoder also chunks overlength text rather than dropping everything past one 77-token window.

The observed fragment is composer-owned. `ghost_signal_prompt.resolve_action()` takes the first six regex words of an unmapped free-text `beat_intent` and prefixes `moves with`. It therefore emits incomplete phrases such as:

> moves with erin risks exposure by transmitting a

The same route leaks cast names. `_trim_to()` knows how to remove dangling function words, but this unmapped-intent branch does not call it. Raw `arc_phase="scene"` is also emitted even though it adds no visual information.

Current live prompts are approximately 208--291 characters. The 320-character ceiling is a recipe/quality choice and banana-funnel headroom, not a proven tokenizer limit. Prompt v2 should be deliberately shorter because the target model benefits from clear visual clauses, not because a fictitious hard CLIP cutoff requires it.

## 3. Operator decisions that constrain the design

1. Preserve the style system that is visibly working. Positive style cue and negative prompt remain byte-identical for the same style pack.
2. Prefer abstract coolness over rigid character likeness.
3. Consistency means a recurring motif, not a recurring face.
4. Use a compact LLM pass tailored to this model, then stamp its result into the ledger before rendering.
5. Never feed that pass raw dialogue, episode title, M4 scene wall, second-person text, or cast-name metadata.
6. The real workflow JSON stays untouched: this is an internal ShotLock/ledger/composer change with no node, link, or widget change.
7. Preserve the current clip and cadence contract: one beat is one Ghost clip and one prompt. AnimateDiff's 16-frame value is an internal sliding-context length, not a clip cap. Do not introduce 32/64-frame microclips; split only if a later live quality proof establishes that the current long-clip path cannot hold a beat.

## 4. Prompt v2 contract

The final positive prompt has four conceptual parts:

`existing style cue, recurrence cue, authored visual beat, representation-specific shot law`

There is exactly one authored visual beat per current Ghost shot/clip. The current live episode therefore has eight prompt bodies for eight beats, including the opening and closing bookends. If an already-existing coverage plan ever contains multiple render segments, the render contract must not silently reuse one segment's object under another index, but this sprint does not create such segmentation for Ghost.

It must be a compact comma-separated visual instruction, not prose and not a tag landfill. The authored beat should usually be 5--14 words and at most 96 normalized characters. It must name something drawable and one visible change/action.

Examples of the intended content layer, before the unchanged style cue is applied:

- `object`: `rust-red bracelet motif, microfilm coils into the clasp under a passing shadow`
- `signal`: `amber key motif, radio bands crush a phone glow into concentric static`
- `figure`: `lean silhouette in an olive coat, one hand tears a bracelet into the light`

These examples are illustrative only. They must never become canned episode output.

Representation rules:

- `figure`: a mid-shot-or-wider silhouette may appear; the compact figure motif is included; faces are obscured or visually irrelevant.
- `object`: the stable color/prop motif recurs as an isolated object or emblem.
- `signal`: the stable color/prop motif recurs through light, static, shadow, waveform, reflection, or geometric motion.

All positive clauses are affirmative. The composer must not insert `no people`, `without humans`, or similar positive-channel negations. Object/signal absence is expressed by affirmative subject/framing choices, while the unchanged negative remains the style pack's established exclusion authority.

Python assigns representation before the LLM call using an episode-seed-keyed rotation over the global Ghost timeline. Character beats cycle through `figure`, `object`, and `signal`; bookends use role-appropriate `object`/`signal` assignments. No more than two consecutive Ghost clips may use the same representation, and at least half of three-or-more character clips are `object` or `signal`. The LLM authors only the leaf for the already-assigned mode; it never selects or echoes the mode.

## 5. Authoring boundary

Add a Ghost-specific, batch authoring surface adjacent to the pure composer. Keep `ghost_signal_prompt.py` free of I/O and model loading.

ShotLock is the authority:

1. Extract beats and resolve the effective role-to-engine map once.
2. Build the existing durable `subject_sigil` map, then derive a separate structured recurrence-motif map from canonical allowlisted silhouette, costume/color, and prop tokens. Whole cast prose, gender-as-person, face/jaw/brow/hair landmarks, field labels, and names are not motif tokens. Keep `subject_sigil` byte-stable as a compatibility/forensic receipt; Prompt v2 renders only the new selected `motif_cue`.
3. Identify every beat whose resolved engine declares the unchanged capability token `prompt_profile == ghost_signal_v1`. This includes character, announcer, music, official-v3, cadence, and haunted peers. Do not bump the capability string; bump only composer/author versions.
4. Assign each Ghost beat's representation deterministically, then build a safe episode-level request from ledger fields only.
5. Resolve the already-selected writer/technical model through the existing ShotLock loader seam; add no widget and no model selection surface. Make one batched call for the normal episode and unload that LLM explicitly before image/video work.
6. Validate exact beat coverage, schema, safety, leaf length, and the complete composed prompt's installed-SD1 token count. Retry the failed batch once as a fresh bounded call; this is not an open-ended conversation.
7. Pass the validated Ghost map into `build_execution_plan()`. After sigils exist and before cast-time family preflight, attach the same object to the temporary preflight shot; stamp it unchanged on the durable row. Preflight and render must consume the identical composer version and leaf.
8. Declare `ghost_prompt: Optional[dict] = None` on the extra-forbid `ShotRow` schema in the same change. Absence means non-Ghost/legacy. A deliberate deterministic result is represented only by `source="deterministic_fallback"` plus a nonempty `fallback_reason`; there is no `fallback` boolean.
9. If no model is configured (unit-test/legacy local path), stamp a deterministic complete fallback object. If a requested live model fails to load, keep the existing fail-loud model policy.

The model sees only:

- opaque `shot_id`;
- normalized role;
- sanitized, lower-case `beat_intent` with every known cast name removed;
- normalized emotion when present;
- a mapped arc cue only when the value is meaningful (`scene` is omitted);
- the already-selected representation and its mechanically distilled motif as read-only context;
- short output rules for one `drawable_beat` field.

The model never sees `line.text`, episode title, story brief/M4 wall, raw cast row, or a character's name. The complete set of known cast names is removed before request construction. Second-person tokens and field-label debris such as `Face:` are removed/rejected at this boundary. The model is not asked to return the motif, mode, style, framing, law, IDs beyond the opaque lookup key, or any other Python-owned field. The returned `drawable_beat` is rejected if it contains a known cast name, second-person address, lettering instructions, camera/style boilerplate, a dangling function-word tail, or content outside its assigned representation.

## 6. Durable ledger schema and replay

Every Ghost shot receives:

```json
{
  "ghost_prompt": {
    "schema_version": 1,
    "author_version": "ghost_drawable_beat_v1",
    "mode": "object",
    "motif_cue": "rust-red bracelet motif",
    "drawable_beat": "microfilm coils into the clasp under a passing shadow",
    "source": "writer_llm",
    "model_id": "<ledger-selected model>",
    "request_sha256": "<full hash>",
    "output_sha256": "<full hash>",
    "fallback_reason": ""
  }
}
```

`source` is one of `writer_llm`, `replay`, or `deterministic_fallback`. The exact strings used for rendering are durable; render time does not call an LLM.

The request hash is SHA-256 over compact, sorted canonical JSON with exactly these keys: `author_version`, `beat_id`, `mapped_arc`, `mode`, `model_id`, `motif_cue`, `motif_sha256`, `normalized_emotion`, `ordinal`, `role`, `sanitized_intent`, `schema_version`, and `template_sha256`. `motif_sha256` is exactly `sha256(motif_cue.encode("utf-8")).hexdigest()`. `ordinal` is the zero-based position in the current ordered Ghost author batch (`g000` = 0, `g001` = 1), never a global durable-shot index. `template_sha256` covers the exact system/user template, output envelope, temperature, and batch-size output-token formula, so a generation-contract change invalidates replay. `beat_id` is the canonical ledger identity, including `b000_music_open`; neither cast-time `shot_id=beat_id` nor durable `shot_id="shot_"+beat_id` is hashed. Raw dialogue/title/M4/cast prose, style-pack text, negative text, and `render_request_hash` are excluded. A changed safe input, motif, author version, template/generation contract, or model invalidates reuse. If the input ledger already carries a valid prior `ghost_prompt` with the same request hash, ShotLock reuses the accepted leaf without a model call; this sprint does not invent a hidden disk-ledger lookup. Replay joins by `source_line_ids[0]` when present, otherwise by exact durable `shot_id == "shot_" + beat_id`, which covers the synthetic opening.

The authored object must **not** alter `render_request_hash` or the derived video seed: the same-seed A/B depends on that separation. Ghost's engine cache identity already includes the final composed positive and negative hashes, so a prompt change cannot hit a stale clip cache. Add the leaf/source hashes to receipts without folding them into the seed domain.

## 7. Composer changes

Prompt v2 consumes only the stamped Ghost object plus existing role/style authorities. Its pure scalar interface is `compose_ghost_prompt_v2(role, style, mode, motif_cue, drawable_beat)`; it has no `open_subject` parameter and does not consume raw `beat_intent`, raw traits, raw `arc_phase`, a pack motion register, or the optional LTX motion-clause object. The render driver reads `shot.ghost_prompt`, validates the declared schema/version, and passes those scalar fields. A present-but-malformed object fails closed. A truly legacy Ghost row with no object may use an explicitly versioned v1 compatibility path; every newly built Ghost preflight/row must carry v2.

- Retain the current style cue function and negative composer unchanged.
- Replace the current four-cue face-adjacent identity emphasis with a compact motif appropriate to the selected representation. In particular, pooled jaw/brow/face landmarks must not be the recurrence mechanism for `object` or `signal` modes.
- Preserve a complete deterministic action for cast-time preflight and no-model fallback. Delete the `moves with <first six words>` behavior; an unknown intent is never copied as a fragment.
- Use affirmative representation-specific framing/shot-law text. `object` uses an isolated object/emblem filling the composition; `signal` uses an abstract signal field filling the composition; `figure` stays mid-shot or wider and does not request a visible face. The positive channel never says `no people` or otherwise attends to an excluded human concept.
- Keep `GHOST_PROMPT_PROFILE="ghost_signal_v1"` as the stable capability token. Bump the composer prompt version and extend trace receipts with author version, exact model ID, source/fallback disposition, mode, request hash prefix, drawable-beat hash prefix/text, final installed-token count/section count, and final positive/negative hashes.
- Retain a hard safety ceiling for downstream banana substitution, but establish the new operating band from measured composed prompts and A/B output rather than asserting a tokenizer cutoff. The preliminary target is 150--230 characters; implementation measurements and the A/B proof may tighten it.
- Exclude every `prompt_profile="ghost_signal_v1"` shot from the later optional `_otr_motion_clause` authoring pass. That pass receives names/raw dialogue and its result would be ignored by Prompt v2. Render-time performs zero LLM calls; engine motion-source receipts name the stored Ghost drawable leaf instead of `ledger_motion_clause`.

## 8. Deterministic fallback

Fallback is not the old free-text slice. It must always be a complete drawable clause.

- Mode follows the same episode-seed-keyed rotation used by the authored path.
- Known compact intent enums use complete checked-in actions.
- Unknown free-text intent uses a checked-in representation-specific neutral kinetic clause. Do not reintroduce regex noun/verb extraction as a new truncation surface.
- Bookends receive phase-specific complete abstract clauses rather than the same long pack motion register on every episode.
- Every fallback is stamped and receipted exactly like authored output.

## 9. Tests and proof

Focused CPU tests must prove:

1. the exact live long-form character intents no longer emit names or dangling fragments;
2. `Face:` and similar field labels cannot survive motif distillation;
3. raw dialogue/title/M4/name values cannot enter the LLM request or final prompt;
4. deterministic mode scheduling plus JSON coverage/length/safety validation and one bounded retry;
5. requested-model failure remains loud; no-model test path stamps deterministic fallback;
6. replay with the same request hash spends no LLM call and is byte-identical;
7. style cue and negative are byte-identical before/after for every shipped pack;
8. object/signal positive strings use affirmative non-human subject laws and contain no human/face request; figure prompts contain no face/close-up request; pixel-level absence remains an A/B eyeball question, not a string-level overclaim;
9. changing the stored drawable beat changes engine cache identity but leaves `render_request_hash` and actual video seed unchanged;
10. every Ghost trace row carries the new prompt receipts;
11. no node, link, widget, or `workflows/otr_canonical.json` change occurs.

The frozen-control audit is complete. Across all nine shipped packs and all eight live shots, maximum full-positive counts are 63--69 installed SD1 tokens including BOS/EOS; negatives are 27--37; every negative phrase fits; and the scan found zero positive/negative self-conflicts. The live cadence is exactly eight beats to eight clips at 25 fps/hold-2, with the unbounded `max_frames=0` contract and exact delivered targets of 250, 248, 36, 43, 70, 72, 339, and 200 frames. No separate cadence/style/negative repair is warranted.

Prompt v2 adds a first-window quality admission guard: after replacing the old action/emotion/raw-arc block with the authored leaf, the complete final positive must remain at or below 77 installed SD1 tokens including BOS/EOS. This is a salience/recipe guard, not a claim that ComfyUI transport discards later chunks. An over-window candidate rejects/retries or falls back by changing the model-owned content leaf only; style cue, recurrence cue, framing law, and negative are never trimmed or rewritten to make it pass.

Then run the full Windows regression suite and the separate Bug Bible regression.

Live proof is a same-seed A/B on the current frozen Ghost episode:

- run and archive the full canonical v1 A arm before replacing the composer; there is no existing same-script Ghost A/B and no qualification-grade pruned replay shortcut;
- same ledger/script, engine, style pack, negative prompt, frame counts, scheduler/settings, and video seeds;
- A uses the current v1 content prompt; B uses the stamped v2 content prompt;
- first compare exact prompts and receipts, then render the full small episode or a representative span including figure/object/signal modes;
- publish proof assets directly to the canonical episode/OBS paths;
- operator eyeball judges beat relevance, interestingness, recurrence without face repetition, temporal coherence, and preservation of the already-good style.

The 320-character ceiling may change only after this A/B demonstrates that the shorter recipe is at least as coherent and more beat-relevant. Golden/still-carried lanes are untouched.

## 10. Documentation/admission

The live node input plus the checked-in composer reproduces a name-leaking, incomplete **character** prompt route, but no archived full prompt proves that exact string reached a published render. The problem statement's announcer example is unreachable because announcer/music resolve a nonempty pack register first. Treat this as static/dev coverage under the already-admitted dangling-preservation class; do not create a new production PBUG or misattribute the older 344-character bookend refusal (`BUG_BIBLE.yaml` 12.127) as this symptom.

This sprint should extend executable coverage for the existing general rule that prompt preservation cannot mean shipping a dangling fragment. A new Bug Bible entry is warranted only if the final grounded history review establishes a genuinely new portable failure class.

## 11. R1 decisions carried into coding review

R1 closed the following questions:

1. Author in ShotLock after effective route/sigil/motif resolution and before cast-time preflight; temporary and durable shots consume the same object.
2. Keep `subject_sigil` byte-stable for compatibility/forensics; derive a separate non-face `motif_cue` that Prompt v2 owns.
3. Python assigns representation deterministically; the LLM owns only `drawable_beat`.
4. Preserve the 320-character banana ceiling and add the measured one-window installed-token admission gate; retarget the normal character band only after A/B.
5. Capture a full canonical v1 baseline before code, then run the pinned canonical v2 arm and abort the comparison unless semantic ledger/audio/shot/seed/style/negative controls match.

## 12. R2 code-level blueprint

R2 reviews the following concrete implementation, not a choice among broad
architectures.

### 12.1 Pure author and composer surfaces

Add `nodes/_otr_video_engines/ghost_signal_author.py`. It is stdlib-only until
the explicitly lazy tokenizer loader runs, and it never loads an LLM. It owns:

- `GHOST_AUTHOR_SCHEMA_VERSION = 1`,
  `GHOST_AUTHOR_VERSION = "ghost_drawable_beat_v1"`, the exact batch-template
  text, `GHOST_TEMPLATE_SHA256` (the SHA-256 of the canonical system/user
  template, JSON envelope, temperature, and output-token formula), the
  96-character and hard 5--14-word leaf bounds (the model is
  instructed to target 6--10), allowed source/mode
  enums, safe-input normalization, strict output parsing, request/output hashes,
  deterministic fallback pools, and stored-object validation;
- a deterministic episode-seed-keyed mode scheduler. Character beats cycle
  `figure -> object -> signal` from a hashed offset; non-character bookends
  alternate `object`/`signal`; a deterministic collision correction prevents a
  run longer than two without changing clip planning;
- `build_ghost_author_specs(...)`, which receives only ShotLock-projected rows
  and gives the model opaque IDs `g000`, `g001`, ... rather than ledger IDs;
- `build_batch_prompt(specs)` and `parse_batch_response(raw, expected_ids)`. The
  only accepted envelope is
  `{"shots":[{"id":"g000","drawable_beat":"..."}]}` with no duplicate,
  unknown, missing, or extra fields. One exact enclosing markdown JSON fence may
  be removed as transport wrapping; prose, duplicate keys, trailing objects, and
  schema extras remain invalid. A first invalid response retries the whole batch
  once atomically; no rows are salvaged across attempts. A second invalid batch
  receives a complete deterministic batch;
- `validate_drawable_beat(...)`, which enforces shape and boundary safety, not a
  Python story-vocabulary judge: no cast name/id, second-person address, field
  label, lettering instruction, style/camera boilerplate, line break, dangling
  function-word tail, or person/face request in `object`/`signal`; and
- a cached local SD1 tokenizer measurer. It lazily instantiates the installed
  `comfy.sd1_clip.SD1Tokenizer`, asserts its `clip_l` contract is max-length 77
  with real start/end tokens, and calls
  `tokenize_with_weights(text, return_word_ids=True)`. Payload tuples have
  nonzero word IDs; the published count is payload plus two BOS/EOS tokens per
  returned section (`payload + 2 * windows`) and windows is the number of
  returned 77-token rows. It never trusts Hugging Face's 8192
  `model_max_length` metadata or counts padded row length. Production cannot
  replace a missing tokenizer with a whitespace estimate. The authoring API
  takes `token_measure_fn=None` as an explicit test injection point; one
  integration test exercises the installed Comfy tokenizer.
- `finalize_ghost_prompt_v2(...)`, the shared author/render finalizer. It lives
  in this module, not `ghost_signal_prompt.py`, because this module owns the
  banana-route and tokenizer admission. It imports the pure v2 composer from
  `ghost_signal_prompt.py` and the stdlib-only transform from
  `_otr_banana_route.py`; it never imports `render_driver` and creates no cycle.

Keep `ghost_signal_prompt.py` pure. Preserve the existing v1 composer as the
explicit missing-object compatibility path and add:

- one internal structured sigil-component distiller shared by the existing
  byte-stable `distill_subject_sigil(...)` and the new motif reducer. It exposes
  bucket choices before the 110-character sigil join/trim, so v2 can emit only
  canonical allowlisted silhouette/color/prop tokens without losing a trailing
  prop. It never copies a cast phrase, name, gender noun, landmark, hair, jaw,
  brow, or face token. Golden tests pin the old sigil bytes. For one character
  the three render cues share color+prop, with figure also carrying silhouette;
- role+mode constants for beats with no character sigil:
  `announcer_visual/object = "radio dial emblem"`,
  `announcer_visual/signal = "radio dial signal"`,
  `music_visual/object = "broadcast console emblem"`, and
  `music_visual/signal = "broadcast waveform signal"`. These compact radio
  anchors recur while the unchanged pack cue supplies anime/archive/material
  style; an empty Ghost bookend motif is invalid;
- `compose_ghost_prompt_v2(role, style, mode, motif_cue, drawable_beat)`. Its
  only ordered pieces are the unchanged `_prefix_pack_cue` result, motif,
  drawable leaf, and an affirmative mode law. Figure is mid-shot-or-wider and
  silhouette-led; object is an isolated emblem filling the composition; signal
  is an abstract field filling it. It returns those exact protected components
  alongside the joined string so post-transform survival is checkable without
  re-parsing prose. The unchanged negative composer is called verbatim; and
- `GHOST_PROMPT_VERSION_V2 = "ghost_signal_v2"` while
  `GHOST_PROMPT_PROFILE = "ghost_signal_v1"` remains the peer capability.

The old v1 `resolve_action()` also loses its six-word free-text copy: mapped
enums remain complete checked-in actions and every unknown becomes a complete
deterministic neutral action. This keeps a legacy row from reproducing the
name-leaking fragment without pretending a legacy replay is v2.

### 12.2 ShotLock transaction and model lifecycle

The smallest safe insertion is inside `build_execution_plan()` at the existing
single-authority route seam:

1. construct `engine_for` exactly once as today;
2. build the byte-stable `subject_sigils` map;
3. select all beats whose resolved registered engine has
   `prompt_profile == GHOST_PROMPT_PROFILE`;
4. reduce character sigils to motifs, project safe line metadata, schedule
   modes, and author one Ghost map keyed by `beat_id`;
5. pass both maps to cast-time preflight; then
6. stamp the same validated object on the durable row before coverage stamps.

Extend `_assert_family_inputs_satisfiable_cast_time(..., subject_sigils=None,
ghost_prompts=None)` and attach a deep copy of `ghost_prompts[beat_id]` to its
temporary shot before `build_request_from_shot()`. Extend
`build_execution_plan(..., warnings=None)` only to carry loud authoring
dispositions back to `lock()`; do not move route resolution to a second helper
or build a second engine map.

Every real and synthetic beat already has a canonical `beat_id`; the synthetic
opening is `b000_music_open`. Author maps and request hashes key only on that
canonical value. They never hash the cast-time temporary `shot_id=beat_id` or
the durable `shot_id="shot_"+beat_id`, which intentionally differ. If
`ledger is None`, the author map is empty and direct unit fixtures exercise the
explicit legacy path; a real ledger with Ghost beats must author/stamp every one.
If any resolved registered engine has the Ghost prompt profile, including a
bookend-only episode, `meta.episode_seed` is required before scheduling.

Refactor `_resolve_writer_llm` through one internal
`_resolve_writer_llm_binding(meta, warnings)` returning the raw message-based
generate function plus the exact normalized
`cache_entry.get("model_id", requested_model_id)`. Existing M4 callers keep the
one-argument prompt callable API and their current call site is unchanged.
The binding preserves the current `OTR_TEST_MODE==1` and empty-model early
returns and the configured-model fail-loud behavior. Ghost alone calls the raw
message binding with `[{"role":"user","content": batch_prompt}]`,
`temperature=0.1`, and `max_new_tokens=64 + 48 * len(specs)`; it never passes a
`stop` argument. These values and the exact message split are included in
`template_sha256`. Ghost requests the same technical slot, policy, and
GGUF load config. If a configured model cannot resolve/load, preserve the
current fail-loud behavior. Generation/parse failure receives one bounded retry
then a receipted deterministic fallback. A `finally` block calls
the existing `_otr_model_loader.unload_llm_if_local_resident()` once around the
whole episode batch before any
preflight/image/video work and then asserts
`has_local_resident_llm() is False`; remote providers do not import torch merely
to clear an empty cache.

Before loading, validate any incoming `ledger.video.shots[*].ghost_prompt`
against its expected request hash. A valid match is replayed without a call; a
same-hash malformed object fails closed; a changed safe input/model/template
produces a different hash and is reauthored. There is no hidden file lookup.
Normalize the requested model ID first through the pure model-catalog
`validate_model_id()` path; do not call `request_slot()` until at least one row
actually needs authoring. After a load, assert the cache-entry model ID agrees
with that normalized identity. A prior `writer_llm` object changes only
`source` to `replay`; a prior deterministic fallback retains
`source=deterministic_fallback` and its nonempty `fallback_reason` so reuse can
never launder it into proof eligibility. Mode, motif, leaf, model,
request/output hashes, composed prompt, and seed remain byte-identical.
`output_sha256` hashes the exact accepted `drawable_beat` UTF-8 bytes, not the
disposition-bearing wrapper, so source handling is not a hash contradiction.

The per-shot request hash uses the exact key set and serialization in section 6;
there is no second spelling such as normalized mood or internal shot ID.

### 12.3 Ledger, render, and later-pass wiring

Add `ghost_prompt: Optional[dict] = None` beside `subject_sigil` on the
extra-forbid `ShotRow`; do not change `VideoRequest` or the canonical workflow.
The exact object has these fields and no others:

```json
{
  "schema_version": 1,
  "author_version": "ghost_drawable_beat_v1",
  "mode": "object",
  "motif_cue": "rust lantern emblem",
  "drawable_beat": "microfilm coils into the clasp under a passing shadow",
  "source": "writer_llm",
  "model_id": "<exact normalized model>",
  "request_sha256": "<64 lowercase hex>",
  "output_sha256": "<64 lowercase hex>",
  "fallback_reason": ""
}
```

In `render_driver.build_request_from_shot()`:

- branch on `ghost_prompt` before the legacy character-sigil requirement. A
  valid present object -> v2 composer only and requires its nonempty motif, not
  its legacy sigil; it never reads raw line intent,
  traits, arc, optional motion clause, or pack motion register;
- absent object -> explicitly logged/stamped v1 compatibility composer, where
  the existing character `subject_sigil` requirement remains;
- present malformed object -> `FamilyInputGap`, never a v1 downgrade;
- v2 publishes no phrase-trimming banana budget. Author-time candidate admission
  and render call one shared `finalize_ghost_prompt_v2(...)` helper with the
  same role, style, mode, motif, leaf, ledger meta, `freeze_timestamp`, video
  banana gate, and tokenizer. It composes, applies
  `banana.apply(..., shield_quoted_card_text=False)` when gated on, transforms
  each protected component through the same fixed table, and validates those
  post-banana style/motif/leaf/law components against the literal result.
  Banana vocabulary substitution inside the leaf is allowed and is not treated
  as leaf loss. The helper returns final positive/negative, component, banana,
  token, and window receipts. It rejects a positive over 320 characters or a
  positive/negative over one 77-token window and never trims/repairs;
- the v2 request installs that helper's banana receipt and marks only the local
  build as already finalized so the generic banana funnel does not apply again
  and overwrite a real substitution receipt with a zero-substitution receipt.
  `_apply_visual_safety_prompt()` is provably inert because every Ghost peer is
  local; a focused assertion/test fails if a future peer violates that premise.
  Legacy v1 retains the current common-funnel behavior and cap;
- observability adds author/schema version, exact model, source/fallback reason,
  mode, request/output hash prefixes, accepted leaf text/hash, final installed
  token/section count, and existing final positive/negative hashes. A v2 row
  stamps `prompt_version = GHOST_PROMPT_VERSION_V2` (`ghost_signal_v2`), while
  the adapter capability remains `GHOST_PROMPT_PROFILE = ghost_signal_v1`.
  Exact new
  trace-copy keys are `author_version`, `ghost_schema_version`,
  `ghost_source`, `ghost_fallback_reason`, `ghost_model_id`, `ghost_mode`,
  `ghost_request_sha8`, `ghost_output_sha8`, `ghost_drawable_beat`,
  `ghost_drawable_beat_sha8`, `positive_clip_tokens`,
  `positive_clip_windows`, `negative_clip_tokens`, `negative_clip_windows`,
  `clip_window_max`, and `clip_counter`; `clip_counter` names the shared
  measurer/version and is written by that helper, not merely allowlisted; and
- engine cache identity continues to see final positive/negative while
  `render_request_hash` and the derived video seed remain untouched.

Update the trace-copy allowlist and change the Ghost engine declaration from
`motion_source = "ledger_motion_clause"` to
`motion_source = "ledger_ghost_drawable_beat"`.

The cast-time preflight first asserts exact `ghost_prompts` coverage for every
registered Ghost-profile beat and attaches a deep copy to the temporary shot;
it may not silently fall through to v1 when a real-ledger map entry is absent.

The later optional motion-clause path must not load a writer before it discovers
that every target is Ghost. Add a `skip_shot` predicate and lazy
`generate_fn_factory` to `generate_motion_clauses()`: node 92 passes a
capability-based predicate for `prompt_profile == GHOST_PROMPT_PROFILE` and the
uninvoked `_mc_fn` factory while retaining the existing `_mc_on()` guard and the
backward-compatible `generate_fn=` path. The factory is invoked only on the
first eligible shot that also has dialogue and a character. Ghost shots continue
without writing `motion_clause`. Mixed episodes generate clauses only for
eligible non-Ghost rows; an all-Ghost replay performs zero render-time LLM
loads/calls. Node 92 wraps the whole enabled motion block in `try/finally`, calls
`unload_llm_if_local_resident()`, and verifies absence before
`run_real_episode()`.

### 12.4 Retry, fallback, and token admission

One normal episode is one batch call. A rejected/missing/duplicate leaf rejects
the whole batch and receives one fresh whole-batch retry. If that also fails,
the whole batch receives unique, complete, mode-specific checked-in clauses
selected from the episode+beat+mode hash domain with deterministic collision
probing;
opening and closing have different phase-specific clauses. No raw free-text
noun/verb slicing occurs.

Before any LLM call, every shipped pack/mode/longest-motif mechanical shell plus
the shortest legal fallback must prove it can fit; a failure is a composer
constant defect, not a model retry. Each candidate is composed with the real
style and passed through the shared finalizer, including the actual video banana
gate and episode variety key. Its literal final positive targets at most 69
installed SD1 tokens including BOS/EOS and one section, preserving the measured
v1 worst-case as headroom. A candidate above that target retries/falls back by
replacing only the whole leaf. Style cue, motif, law, hard 320-character ceiling,
and negative are immutable. The final literal positive and separate negative
must each be one window and at most 77 tokens; the render boundary refuses any
violation and never trims/repairs it.

The shared measurer lazily uses installed
`SD1Tokenizer.tokenize_with_weights(..., return_word_ids=True)`, counts nonzero
word-ID payload plus BOS/EOS per returned section, and publishes both count and
section count. Production fails closed if that tokenizer cannot load. Under
`OTR_TEST_MODE`, only a missing installed-tokenizer gate may be skipped; leaf
shape, safety, character, component, and 320-character validation all remain.
An integration test uses the installed tokenizer. The engine may call the same
shared measurer immediately before `CLIPTextEncode`; it must not introduce a
second loaded-CLIP counting algorithm that can disagree after model load.

Character modes use an unmodified cyclic sequence from a hashed offset; this
alone satisfies the non-figure quota and never repeats. Bookends independently
alternate object/signal from a hashed offset. Scan the merged Ghost timeline
left-to-right; only if a bookend would create a third identical mode, flip that
bookend to the other non-figure mode. Character assignments are never rewritten.

Production may receipt the deterministic batch after two invalid semantic
attempts. The same-seed v2 proof arm is invalid unless every row says
`source=writer_llm`, has an empty fallback reason, and exact full coverage; it
aborts rather than presenting fallback output as the LLM treatment.

### 12.5 Code/test footprint

Expected production edits are limited to:

- `nodes/_otr_video_engines/ghost_signal_author.py` (new),
- `nodes/_otr_video_engines/ghost_signal_prompt.py`,
- `nodes/otr_shot_lock.py`,
- `nodes/_otr_video_engines/schemas.py`,
- `nodes/_otr_video_engines/render_driver.py`,
- `nodes/_otr_motion_clause.py`,
- `nodes/otr_video_render_batch.py`, and
- `nodes/_otr_video_engines/eng_ghost_signal.py`.

Focused tests extend `test_ghost_signal_prompt.py`,
`test_ghost_signal_lane.py`, `test_motion_clause.py`, and ShotLock/preflight
coverage. They pin one-call batching, safe request projection, exact coverage,
one retry, unique fallback, replay/no-spend, explicit unload, malformed refusal,
preflight/durable equality, v1 absence compatibility, style/negative byte
identity across all packs, <=77 actual tokens, seed stability/cache invalidation,
one beat/clip/prompt, and no workflow diff.
