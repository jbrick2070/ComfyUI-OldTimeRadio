# The instrument -- durable prompt and seed receipts, and a canonical replay mode (campaign item 0)

**Driver: Claude (Fable 5.1, Cowork), 2026-09-02, HEAD `208b6b00`. Design item: full arc before code
(CLAUDE.md 2026-08-15 routing; every fresh read of the campaign said the same). This anchor is
docs only; no node file changes until the round has run and one window codes it.** The campaign
statement is `../PROBLEM_STATEMENT.md`; the judgment with all six reads folded is
`../fresh-eyes/pass01_judgment.md`.

## 1. The facts (read at HEAD today)

* **The planned prompt is wire-only.** `OTR_ShotLock` builds `video.shots[]` (creative, sigil,
  `render_request_hash`, `target_frame_count`, coverage plan, the whole `ghost_prompt` object)
  and returns it as `patched_ledger_json` (`nodes/otr_shot_lock.py:3060-3076`); it never saves.
* **The disk merge drops it.** `Ledger._merge_with_disk` keeps only `schema_version, audio,
  audio_gates, transitions, radio_bookend_path` from disk (`nodes/production_ledger.py:1592-1595`)
  and `Ledger.data` has no `video` key; 0 of the 60 newest ledgers carry `video.shots`.
* **Node 92 already has a durable write path.** `OTR_VideoRenderBatch` stamps
  `audio_motion_profiles` into the in-flight on-disk ledger through
  `_otr_ledger.in_flight_ledger_path()` + `load_ledger_safe` / `save_ledger_safe`
  (`nodes/otr_video_render_batch.py:330-345`). The pattern the trace needs exists.
* **The seed.** `request_seed = _seed_from_hash(render_request_hash, shot_id)`
  (`render_driver.py:548`, applied unconditionally at `:3767`); `render_request_hash` =
  `_content_hash([brief_hash, cast_hash, beat_id, char_id])` (`otr_shot_lock.py:1314-1318`).
  Same ledger -> same hashes -> same seeds. The final positive/negative text, the engine's
  resolved adapter strength, denoise and context settings exist only inside `render_driver`
  and the engine's `_build_render_request`, and are logged as a sha8 plus ~100 chars.
* **Nothing can hand the same ledger to the render phase twice.** `OTR_LedgerScriptWriter.IS_CHANGED`
  returns `time.time()` (`OTR_LedgerScriptWriter.py:2789-2793`), so node 1 always re-executes and
  writes a new ledger with a fresh brief and cast roll. `preserve_ledger` is a cast policy
  (`cast_lock.py:67`), not a replay. The canonical chain is node 1 -> 80 (cast) -> voices ->
  sequencer -> 87/88/89/90 (directors, brief, ShotLock) -> 91 (stills) -> 92 (video) -> 94/95
  (scopes, credits) -> composite -> publish (`workflows/otr_canonical.json` links).
* **The visual smoke submits only node 92** (`scripts/otr_visual_smoke.py:8-19`): it proves a
  render, not the composite-to-obs tail. `otr/obs/` is the success signal (operator rule).
* **Blinding is a house method** (`docs/PRODUCTION_SPRINT_LESSONS.md` lesson 40): unlabeled
  candidates, the key withheld in a separate file, the verdict before the reveal, "no meaningful
  difference" accepted.

## 2. What the instrument must deliver (acceptance, before any pixel arm)

1. A published episode whose on-disk ledger carries, per beat: the PLANNED shot (the `ghost_prompt`
   object and `creative`), and the ACTUAL render (final positive and negative text, `prompt_sha8`,
   the `request_seed` that reached the sampler, `comparison_seed_hash` = today's
   `render_request_hash`, a separate `actual_request_sha` over everything that reached the
   sampler, engine id, recipe id, adapter id and RESOLVED strength, denoise, context and
   injection settings, still content hash when a still was used, model file digests, per-clip
   peak VRAM and wall seconds). Per-clip receipts in `meta.render_engines.per_clip` gain
   `prompt_sha8` and `request_seed`.
2. A replay of that episode's frozen ledger through the canonical graph that publishes a NEW
   episode to `otr/obs/` whose every `render_request_hash` and `request_seed` equals the
   original's (the A/A null), with the audio byte-identical to the frozen master.
3. Two A/A nulls and one candidate published under neutral titles, a key file outside obs, and a
   scorecard (style, face, setting, motion-to-speech, then "overall better").

## 3. Driver decisions (D1-D6) -- the reviewer pressure-tests these

**D1. Two records, two owners.** ShotLock writes its PLANNED `video` section to the in-flight
on-disk ledger (the same `load_ledger_safe` / `save_ledger_safe` shape node 92 already uses) and
`TOP_PRESERVE` gains `"video"`; `OTR_VideoRenderBatch` appends a bounded `render_trace[]` row per
rendered clip or segment (the ACTUAL record above) and `TOP_PRESERVE` gains `"render_trace"`.
The trace is append-only within a run and replaced on a re-render of the same episode dir. The
planned and the actual are never merged into one object: the difference between them is the
finding when a render deviates from its plan.

**D2. `render_driver` returns what it sampled.** Today the final text and settings die inside the
driver; `render_shot` returns the clip path and a peak. It returns the trace row too (positive,
negative, seed, resolved adapter strength, denoise, context, injection, still hash, recipe id),
and the engine's `_build_render_request` result is the single source of those values, so the
trace cannot describe a render that did not happen.

**D3. Replay enters at the writer, and the audio passes through frozen.** Shape A: node 1 gains
ONE trailing optional STRING widget, `replay_from` (default `""`), appended at the END of
`widgets_values` (the positional rule; regenerate variants; the four workflow tests). When set to
a frozen bundle directory, the writer short-circuits authorship and emits the bundle's ledger and
script as its outputs, with `meta.replay_from`, `meta.replay_of_episode` and a new `episode_id`
(`<slug>_replay_<stamp>`) so the output dir and the obs file are distinct. Downstream, CastLock
runs in `preserve_ledger` (unchanged rows), the voice nodes and the sequencer see
`meta.replay_from` and copy the frozen line WAVs and master mix instead of rendering (the ledger
timings are already frozen), the still dispatcher hits its content-addressed cache for rows whose
files exist in the bundle (and mints for a peer engine that declares a new still plan), ShotLock
re-plans deterministically (same brief, same cast -> same hashes), and node 92 renders through
the real tail to `obs_publish OK`. *Alternative shape B (Codex's "node-92 ingress bundle"):* replay
at node 92 only, ignoring its wire inputs. Cheaper to build, but the upstream nodes still execute
and author a throwaway episode on every replay (about ten minutes of writer and TTS per A/B arm),
and the frozen audio never reaches the composite, so the published file is not the same show. The
driver proposes A and asks the reviewer to break it: the voice-node pass-through is the part with
the most surface.

**D4. The bundle is frozen by a script, not by a node.** `scripts/otr_freeze_replay_bundle.py
<episode_dir>` copies the ledger, the per-line WAVs, the master mix, the stills and portraits, and
`episode_canon.json` into `<output>/otr/episodes/_replay/<episode_id>/` and writes
`manifest.json` with SHA-256 per file and the source commit; replay refuses a bundle whose
digests do not match its manifest. Freezing is explicit and operator-visible; nothing freezes
itself.

**D5. Blinding lives in the harness, not in the nodes.** `otr_canonical_api_run.py` gains
`--replay-from <bundle>` and `--label <code>` (the existing `--title` patch carries the code
onto the title card); the key file `_blind_key.json` is written beside the bundle, never in
obs. Two nulls and one candidate per comparison.

**D6. Tests, all offline.** `TOP_PRESERVE` keeps `video` and `render_trace` across a save; the
trace row shape and the `actual_request_sha` recipe; the same ledger reproduces every
`render_request_hash` and `request_seed` (pure functions, no server); the writer's replay
short-circuit emits the bundle's ledger unchanged and refuses a digest mismatch; the widget is
appended at the end (`build_variants --check`, `test_widget_value_alignment`,
`test_canonical_widget_input_parity`, `test_workflow_link_target_indexes`); the voice pass-through
copies rather than renders when `meta.replay_from` is set. Live proof: freeze
`signal_lost_the_tectal_echo_20260902_131902`, replay it twice, publish both, diff the seeds
(all equal) and the master audio (byte-identical).

## 4. Files this touches (one window codes them; the 5080 owns `nodes/`)

`nodes/production_ledger.py` (TOP_PRESERVE), `nodes/otr_shot_lock.py` (durable planned write),
`nodes/otr_video_render_batch.py` (trace + receipt fields + peak VRAM per clip),
`nodes/_otr_video_engines/render_driver.py` (return the sampled values),
`nodes/OTR_LedgerScriptWriter.py` (the `replay_from` widget and short-circuit), the voice nodes and
the sequencer (frozen-audio pass-through), `nodes/otr_image_gen_dispatcher.py` (cache hit on
bundle rows), `workflows/otr_canonical.json` + variants (the trailing widget),
`scripts/otr_freeze_replay_bundle.py` (new), `scripts/otr_canonical_api_run.py` (replay + label),
tests. Out of scope: any engine recipe, any pixel arm, any change to the 4060 profiles.

## 5. Questions for the round

1. D3 A vs B: is the voice-node pass-through the right seam, or is there a cheaper place where
   the frozen master can enter the sequencer without touching each TTS node?
2. Should `render_trace` live at the ledger's top level (needs `TOP_PRESERVE`) or under `meta`
   (recursively merged already, no preserve change)? What breaks each way?
3. Is the writer-side widget the right entry, given `IS_CHANGED` and the per-run seed rolls
   (bank roll, style roll) that must NOT re-roll on replay?
4. Which digests belong in `actual_request_sha` so two arms differing only in a graph setting
   get different SHAs while two A/A nulls get the same one?
5. Anything in the ownership split (which node writes which key) that would let a re-render
   silently overwrite a planned record or vice versa?

## 6. r1 fold (Antigravity, Gemini 3.7 Flash High, `kibitz-runs/2026-09-02-replay-instrument/r1`) -- every claim re-read at the files

Verdict NO with four must-fixes; all four grounded, all four taken. The design changes shape:

* **The audio chain is longer than D3 said, and replay must bypass ALL of it at one seam.** The
  canonical audio path is node 1 -> 62 `OTR_LedgerFreezeCascade` -> 80 `OTR_CastLock` -> 81/82
  (voices) and 83 `OTR_StableAudioTheme` -> 3 `OTR_SceneSequencer` -> 4 `OTR_AudioEnhance` ->
  7 `OTR_EpisodeAssembler`, and node 7 owns the master mix (`scene_sequencer.py:1451`,
  `RETURN_NAMES = (episode_audio, output_path, episode_info, audio_done)`). Node 62 loads the
  technical LLM and re-mints `meta.freeze_timestamp` (`_otr_ledger_freeze.py:1097`), and both
  the still dispatcher (`otr_image_gen_dispatcher.py:590-597`) and ShotLock
  (`otr_shot_lock.py:180-188`) hard-reject a wire/disk freeze mismatch; node 83 loads the music
  model; node 4 re-applies DSP. **D3 becomes:** when `meta.replay_from` is set, nodes 62, 80,
  81, 82, 83, 3 and 4 return pass-through stubs (no model load, no meta mutation, the frozen
  `freeze_timestamp` preserved), and node 7 copies the bundle's master WAV into the new
  episode dir, assigns `output_path`, loads it as `episode_audio`, and emits `audio_done`.
  Per-line WAV copying is CUT: `OTR_VideoRenderBatch` already slices beat audio from the
  master (`otr_video_render_batch.py:420-426`). The driver's audio-cache idea (the cache key
  carries `episode_seed`, `cast_lock_revision`, `line_id`; `OTR_AUDIO_CACHE_DIR` overrides the
  dir) would only ever cover the voice nodes and still leave 62, 83, 3 and 4 running; it is
  dropped in favour of the node-7 seam.
* **ShotLock calls an LLM.** `_resolve_writer_llm` + `llm_fn(prompt)` derive the creative
  directives (`otr_shot_lock.py:1312-1338`), so a replayed ShotLock is NOT deterministic.
  **D3 gains:** on `meta.replay_from`, ShotLock skips `llm_fn` and the creative derivation and
  reuses the PLANNED `video` section from the bundle ledger, re-verifying route consistency
  against the live video policy only. (This is also why D1 must persist the planned section:
  without it there is nothing to reuse.)
* **Write through the singleton, loudly.** `stamp_durable(sections=..., source=...)`
  (`production_ledger.py:527`) copies wire-parsed sections into the process singleton and
  saves, raising `LedgerStampError` on failure; disk-only `save_ledger_safe` would desync the
  singleton that later `Ledger.save()` calls (e.g. `OTR_SignalLostVideo`) merge from. **D1
  becomes:** ShotLock stamps `{"video": section}` through `stamp_durable`; `TOP_PRESERVE` gains
  `"video"` so intermediate saves keep it.
* **The writer's replay branch must mint the workspace.** `production_ledger.new_ledger(
  episode_id=<slug>_replay_<stamp>)` before the randomizer rolls, populate `led.data` from the
  bundle, keep the original bank/style rolls and `freeze_timestamp`, record
  `meta.replay_from` / `meta.replay_of_episode`, save the skeleton, then emit the wire outputs;
  otherwise `in_flight_ledger_path()` binds to a stale directory.

Should-fixes taken: the ACTUAL trace lives under `meta` (`meta.render_trace[]`, stamped by
`OTR_VideoRenderBatch` beside the `meta.render_engines` it already stamps through
`stamp_durable`), so `render_trace` needs no `TOP_PRESERVE` entry and no schema bump -- only
`"video"` does; CastLock forces `preserve_ledger` internally on replay regardless of the
`auto_registry` widget (node 80 in the canonical), logging the enforcement; **item 0 is scoped
to `--replay-from` and the A/A equality proof** -- the blinding key file and the scorecard are
the evaluation campaign's, not this item's. Optional taken: the freeze script verifies every
referenced file exists and is non-empty before writing the manifest.

## 7. Revised decisions after r1 (what r2 plans the code for)

* **D1'** ShotLock -> `stamp_durable({"video": ...})`; `TOP_PRESERVE` += `"video"`.
* **D2'** `OTR_VideoRenderBatch` builds `meta.render_trace[]` from the request objects it hands
  to `render_shot` (`req["text_prompt"]`, `req["negative_prompt"]`, `req["seed_bundle"]`,
  `req["observability"]` -- `prompt_sha8`, `negative_sha8`, `prompt_version` are already set
  by the driver at `render_driver.py:1674-1681, 2911-2925`) plus the clip dict the engine
  returns (`recipe`, `domain_adapter_strength`, `render_canvas`, cadence fields) and the
  per-clip peak; adds `prompt_sha8` and `request_seed` to `per_clip`. `render_driver` is not
  edited.
* **D3'** Writer replay branch (workspace + wire outputs from the bundle) + pass-through stubs
  on 62 / 80 / 81 / 82 / 83 / 3 / 4 + the node-7 seam (master WAV copy) + ShotLock reuse of the
  planned section + CastLock forced `preserve_ledger`.
* **D4'** `scripts/otr_freeze_replay_bundle.py` with existence/size checks and SHA-256 manifest.
* **D5'** `otr_canonical_api_run.py --replay-from <bundle>` only; `--title` already exists.
* **D6'** Tests as before, plus: every bypassed node returns its stub without touching CUDA
  (assert no model load under `OTR_TEST_MODE`), the freeze timestamp survives replay unchanged,
  ShotLock on replay never calls `llm_fn`, and the replayed ledger's `render_request_hash` and
  `request_seed` equal the original's for every beat.

Open for r2: the exact stub return values for each bypassed node's `RETURN_TYPES` (AUDIO
batches, strings, done signals) so downstream sockets stay typed; whether the still dispatcher
(91) reuses bundle rows or re-mints (it re-resolves and must find the SAME freeze receipt, so
the bundle's `images[]` and files must be copied into the new episode dir before 91 runs).

## 8. r2 fold (Codex gpt-5.6-sol, coding plan, `kibitz-runs/2026-09-02-replay-instrument/r2`) -- nine must-fixes, all grounded, eight taken, one scoped

* **D2 was wrong to leave `render_driver` alone.** Node 92 calls `run_real_episode` once and never
  sees a request; requests are built inside the driver, segment requests inside
  `render_beat_coverage` (`render_driver.py:4192-4197`), and the driver returns only
  `{ledger, clips, trace, vram_peak_mb, audio_motion_rows}` (`:4870-4873`). **D2''**: the driver
  emits one normalized ACTUAL receipt per rendered segment at the adapter's final graph
  boundary and returns them; node 92 persists what it is handed and reconstructs nothing.
* **The receipt is versioned and canonical.** `receipt_version`, a common envelope (engine id,
  recipe id, implementation version, shot id, segment index, role, final positive and negative
  text, seed, frames, canvas, fps, denoise, adapter id and resolved strength, context and
  injection values, still content hash or null, `model_artifacts` digest references,
  `comparison_seed_hash`, `render_run_id`, completion status, monotonic wall seconds, segment
  peak VRAM) plus an engine-specific `sampler_inputs` dict with explicit nulls;
  `actual_request_sha` = SHA-256 over the canonical JSON of that receipt. Model files are
  hashed ONCE per run into a `model_artifacts` table (never per clip). Peak VRAM and wall time
  are per physical segment; beat and episode aggregates are derived.
* **Nodes 7 and 4 cannot read `meta`.** Node 7 takes AUDIO, a title, themes, the cue manifest
  and the video policy (`scene_sequencer.py:1151-1232`, optional `forceInput` sockets already
  exist at :1192/:1215); node 4 takes AUDIO only (`audio_enhance.py:294-348`). **D3''**: node 7
  gains ONE appended optional `forceInput` socket (`replay_descriptor`, STRING) wired from the
  freeze cascade's `v2_ledger_json` (or the writer's `script_json`) in the canonical -- a link
  addition, so the four workflow tests and `build_variants --check` run; node 3 emits a tiny
  typed CPU AUDIO placeholder on replay; node 4 is NOT branched (it processes the placeholder);
  node 7 byte-copies the frozen master, verifies its SHA against the manifest BEFORE emitting
  `audio_done`, and loads the copied file only for its AUDIO return -- never a decode/re-encode.
* **CastLock's replay return is before the revision increment.** `cast_lock.py:331-365` bumps
  `cast_lock_revision` and reassigns Bark voices before the policy branch, so "forced
  preserve_ledger" would still mutate. The replay return sits before the freeze gate, the
  increment, the voice assignment and the model resolution, and returns the original ledger,
  the original revision, an explicit replay report and a non-empty done token.
* **The clone is a validated import, not a dict assignment.** The ledger already owns
  `_rebase_episode_local_paths` (`production_ledger.py:371`), `_rebase_publication_eligibility`
  (:736) and `rename_episode` (:774); the publisher fails closed on a receipt for another
  episode (`otr_master_audio_mux.py:808+`). **D3''** adds ONE `import_replay_bundle(bundle, new_id)`
  operation: deep-copy, set the new root `episode_id`, keep the source identity under
  `meta.replay_of_episode`, materialize the bundle's assets into the new episode dir, rebase every
  episode-local path, re-evaluate publication eligibility for the new id, clear the source's
  terminal / obs pointers, reset run-volatile telemetry, CLEAR the source `meta.render_trace`,
  then save atomically. `freeze_timestamp` is kept for the freeze-receipt consumers, and the
  merge's `_same_durable_run` (:1585-1587) is checked so two workspaces cannot be mistaken for one
  run -- a separate `content_freeze_id` if they can (r3 verifies).
* **`meta.render_trace` is built complete and stamped ONCE**, after every segment succeeds, with
  `render_run_id`, `shot_id`, `segment_index` and status on every row; `stamp_durable` is a
  shallow meta update (:527-558), never an append; a failed trace stamp fails the publish.
  `video_revision` rides in the same ShotLock `stamp_durable` call as `meta_updates`.
* **Bundle images do not hit the cache by themselves.** The dispatcher follows `cache_index`,
  prefers `pool_path` over `path` (`otr_image_gen_dispatcher.py:1528-1543`), regenerates on a
  missing file, and bumps `image_revision` (:1160, :2016). **D3''**: the import materializes the
  bundle's image bytes into the new episode's stills/portraits dirs, rebases every row and cache
  reference, and node 91 on replay verifies those exact bytes and re-stamps them WITHOUT calling
  `gen_fn`. Peer-engine re-minting is outside item 0 (it belongs to the candidate arm).
* **The manifest is a safe import format**: `schema_version`, `source_episode_id`, source
  commit, normalized RELATIVE paths only (absolute paths, `..`, escaping links / reparse points
  and case-folded duplicates rejected), sizes and SHA-256, built in a temporary sibling directory
  and renamed into place only after every check; replay consumes manifested files only.
* **Scoped, not taken as written (must-fix 9):** the local profile's LTX 2.5 two-stage evidence
  applies when that adapter's graph boundary changes; item 0 edits the driver's receipt
  assembly generically and no engine adapter. The acceptance therefore adds one leg of the
  shipping canonical defaults (still_flat, no diffusion) proving receipts stamp on the cheap
  family too, and the LTX loader/decode receipts are required if r3 finds the LTX boundary
  touched. Tests add the negative paths Codex listed (traversal, digest mismatch, zero-byte
  assets, copy failure, no CUDA/LLM/TTS/image calls on replay, multi-segment trace order,
  recomputable `actual_request_sha`, receipt rebasing, bundle immutability, publish refused
  when the trace stamp fails). Optional taken: a standalone verifier script that recomputes
  manifest hashes, `actual_request_sha`, seeds and original-vs-replay equality without ComfyUI.
Cuts agreed: no node-4 branch; no peer re-minting in item 0; no per-line WAV copy; no top-level
`render_trace` / `TOP_PRESERVE` entry for it (only `"video"` is added).

## 9. What r3 (wiring) must settle

1. The exact canonical edit: node 7's appended optional `forceInput` socket and its link source
   (`62.v2_ledger_json` vs `1.script_json`), the widget/input index it lands at, and the four
   workflow tests plus `build_variants --check` passing; the writer's trailing `replay_from`
   widget and its `CREATIVE_WHITELIST` entries in both copies (`scripts/otr_api.py:831`,
   `nodes/_otr_workflow_apply.py:681`).
2. The stub return values, typed, for nodes 62 / 80 / 81 / 82 / 83 / 3 on replay, and how each
   reads the replay flag (from its ledger-json input; 81/82/83 get `ledger_json` from 62/80).
3. The receipt's `sampler_inputs` for the haunted engine (the eight fields of section 2's
   recipe table) and for the cheap family (null sampler, still path hash).
4. Whether `freeze_timestamp` may identify two workspaces (`_same_durable_run`), and every
   consumer of it on the replay path.
5. Node 91's verify-and-restamp path for materialized rows, and the `_still_index` preference on
   rows whose `pool_path` and `path` now both point inside the new episode dir.

## 10. r3 fold (Cursor grok-4.6-high, wiring, `kibitz-runs/2026-09-02-replay-instrument/r3`) -- eight must-fixes, all grounded at the canonical and the nodes, all taken

* **Node 7's cue-pair check raises before any replay branch could run.** `assemble()` raises
  `CueManifestError` when `music_cue_audio` is present and the manifest is blank
  (`scene_sequencer.py:1244-1250`), and links 282/283 always deliver node 83's AUDIO. **The
  replay branch sits BEFORE that check.** The socket: a trailing optional `forceInput` STRING
  `replay_descriptor` at INPUT_TYPES optional end, canonical input index 10 (today's last is
  `video_policy_json` at 9, link 286); new link `[289, 62, 6, 7, 10, "STRING"]` from
  `OTR_LedgerFreezeCascade.v2_ledger_json` (output slot 6), `last_link_id` 288 -> 289; no
  `widgets_values` slot (forceInput); `replay_descriptor=""` added to `assemble()` so
  `tests/test_input_types_signature_parity.py` stays green. Not from node 1's `script_json`
  (pre-freeze; it fans only to 62 via link 230).
* **The writer's replay branch is the FIRST statement of `run()`**, before the bank and style
  rolls (`:2919-2924`), `require_runnable_bank` (`:2925`), the visual-style resolve
  (`:2930-2931`) and `_preflight_llm_selection` (`:2955-2964`), all of which bind the live
  widgets. The widget appends AFTER `gate_in` in optional; `gate_in` is canonical input index
  32 (link 279), the new widget-backed input is index 33; `widgets_values` gains one trailing
  `""` (today 32 values ending `[14.5, 4096, "Q8_0"]`; the exact index is asserted by the
  four workflow tests and `build_variants --check`, never assumed). `replay_from=""` on `run()`
  before the hidden auth kwargs; `replay_from` added to BOTH `CREATIVE_WHITELIST` copies
  (equality pinned by `tests/test_workflow_apply.py`); `otr_canonical_api_run.py --replay-from`
  patches the widget through `patch_creative` and bypasses `_parse_value` (a path that happens
  to parse as JSON must stay a string). `IS_CHANGED` stays `time.time()` (a digest-less hash
  would serve a stale short-circuit); ShotLock's fingerprint and the freeze cascade's
  `time.time()` invalidate 90 on the new episode id.
* **Node 89 (`OTR_MetaBriefImagePromptGen`) joins the bypass list.** It runs after ShotLock
  (link 255, 90 -> 89) and always resolves the writer LLM (`otr_meta_brief_image_prompt.py:2530`)
  -- an LLM/VRAM hit on every replay and a `prompt_hash` cache miss at 91. On `meta.replay_from`
  89 returns a typed JSON stub with no LLM; 91 verifies and re-stamps the imported
  `ledger["images"]` files first and never iterates MetaBrief objects or calls `gen_fn`.
* **Identity: a workspace id, not a second freeze receipt.** `_same_durable_run`
  (`production_ledger.py:444-466`) and `_same_frozen_episode` (`otr_shot_lock.py:167-186`)
  compare `freeze_timestamp` alone when either side has one; a replay that keeps the source
  receipt would be the source's durable run at ShotLock's strict post-audio overlay
  (`:2928`) and at every `stamp_durable`. A second freeze timestamp is CUT (it would break the
  banana variety at `otr_image_gen_dispatcher.py:1172-1176` and the freeze-mismatch rejects).
  **Fix:** `import_replay_bundle` stamps a workspace-unique `meta.replay_workspace_id`; both
  identity helpers require it to match when either side carries it; `freeze_timestamp` stays
  byte-identical; `new_ledger(new_id)` rebinds `_CURRENT` before any downstream peek.
* **Node 3's placeholder must be DSP-safe.** `empty_audio_batch` (`[1,1,0]`,
  `_otr_resolved_request.py:183-187`) cannot feed node 4's resample + Haas + LPF
  (`audio_enhance.py:388-408`, canonical Haas 0.8 ms). Node 3 returns a CPU float32 AUDIO
  `{waveform: [1, C, T], sample_rate: 48000}` with T large enough for that DSP; 81 / 82 / 83 may
  return `empty_audio_batch` with `done = "replay:passthrough"` because 3 and 7 do not consume
  their audio, and 83's `cue_manifest_json` may be `""` only because node 7's replay branch
  precedes the cue check.
* **One owner for `<ep>_master.wav`.** The import rebases stills / portraits / ledger paths only;
  node 7 copies the manifest's master entry onto the canonical filename it derives today
  (`scene_sequencer.py:1436-1451`), verifies SHA-256 against the manifest, THEN emits
  `audio_done`; on a mismatch it raises and never emits `audio_done` (ShotLock is gated by link
  253); the copy is loaded only for the AUDIO return; never re-encoded.
* **`actual_request_sha` hashes causal sampler inputs only:** final positive and negative text,
  seed, frames, canvas, fps, denoise, adapter id and resolved strength, context / injection
  values, still content hash, `model_artifacts` digests, `comparison_seed_hash`, engine /
  recipe / implementation ids, `sampler_inputs` (explicit nulls on the cheap family) -- never
  wall seconds, peak VRAM, `render_run_id` or timestamps, or two A/A nulls would differ by
  construction. The haunted `sampler_inputs` field list is read from the live adapter's
  `_build_render_request` and recipe receipt at coding time (checkpoint, motion module, adapter
  + strength, sampler, scheduler, steps, cfg, denoise, canvas, context length / overlap /
  fuse, seed), not from the campaign statement. `meta.render_trace` is stamped once, after every
  segment, and `LedgerStampError` is not caught in node 92.
* **The live proof needs a NEW frozen episode.** No ledger on disk carries `video.shots`
  today, so Tectal Echo cannot exercise planned-section reuse. Sequence: ship D1' first,
  render one canonical episode, freeze THAT, replay it twice. `TOP_PRESERVE` gains only
  `"video"`.

Should-fixes taken: the import rebases `path` (not only `pool_path`) because `_still_index`
reads `im["path"]` (`render_driver.py:618`) and node 92 would otherwise open the source
episode's stills; skip `_materialize_episode_copy` when the source already equals the hashed
destination (Windows `copyfile` onto itself fails); the freeze-cascade stub returns the same
JSON on slots 1 and 6 (HEAD already does at `:400-408`) and returns BEFORE the
`needs_full_rerun` check (`:213-227`) and before Phase 10; the CastLock replay return sits
BEFORE `_enforce_freeze_gate` and the revision increment (`cast_lock.py:336, 349-355`); the
freeze script resolves its output root through the same resolver as `_default_out_dir`
(the live tree is `C:\Users\jeffr\Documents\ComfyUI\output\otr\...`), so the bundle root is
`<output>/otr/episodes/_replay/<episode_id>/`; regenerate variants and run the four workflow
tests after the two canonical edits, repairing links by identity; `--title` cannot survive the
short-circuit, so a distinct title card (a later arm's need) is stamped by the import, not the
widget. Optional taken: the replay descriptor on node 7 is a tiny `{replay_from, episode_id,
replay_workspace_id}` JSON (parsed from `v2_ledger_json`'s meta, so no new upstream field);
the replay episode id carries microseconds (`rename_episode` hard-fails on an existing dir);
`video_revision` is left unchanged on replay.

## 11. The coding contract (what r4 converges on and one window builds)

Files and the one change each carries:
1. `nodes/production_ledger.py`: `TOP_PRESERVE += ("video",)`; `import_replay_bundle(bundle_dir,
   new_episode_id)` (validated import: manifest checks, deep copy, new root id,
   `meta.replay_of_episode`, `meta.replay_from`, `meta.replay_workspace_id`, asset
   materialization into the new dir, `path` and `pool_path` rebased, publication eligibility
   re-evaluated for the new id, source terminal / obs pointers cleared, `meta.render_trace`
   cleared, atomic save); `_same_durable_run` honours `replay_workspace_id`.
2. `nodes/otr_shot_lock.py`: `_same_frozen_episode` honours `replay_workspace_id`; on
   `meta.replay_from` skip `llm_fn` and the creative derivation and reuse the planned `video`
   section; always `stamp_durable(sections={"video": section}, meta_updates={"video_revision":
   revision}, source="OTR_ShotLock")`.
3. `nodes/_otr_video_engines/render_driver.py`: one versioned ACTUAL receipt per rendered
   segment (`receipt_version`, envelope, `sampler_inputs`, `actual_request_sha` over causal
   inputs), returned in `ep["receipts"]`; `model_artifacts` hashed once per run.
4. `nodes/otr_video_render_batch.py`: `_stamp_render_trace(ep["receipts"])` once after every
   segment via `stamp_durable(meta_updates={"render_trace": rows})`; `per_clip` gains
   `prompt_sha8` and `request_seed`.
5. `nodes/OTR_LedgerScriptWriter.py`: trailing optional `replay_from` widget; the replay branch
   as the first statement of `run()`: `import_replay_bundle` -> `new_ledger` rebound -> wire
   outputs `(script_text, script_json, news_used, estimated_minutes, technical_model)` from the
   imported ledger.
6. `nodes/OTR_LedgerFreezeCascade.py`, `nodes/cast_lock.py`, `nodes/batch_character_voices.py`,
   `nodes/announcer_voice.py`, `nodes/stable_audio_theme.py`, `nodes/scene_sequencer.py`
   (SceneSequencer + EpisodeAssembler), `nodes/otr_meta_brief_image_prompt.py`,
   `nodes/otr_image_gen_dispatcher.py`: the typed replay returns described above, each reading
   the flag from its own ledger-json input, none touching CUDA, an LLM, TTS or an image model.
7. `workflows/otr_canonical.json` (+ variants regenerated): node 1's trailing widget and input;
   node 7's input 10 and link 289.
8. `scripts/otr_api.py` + `nodes/_otr_workflow_apply.py`: `replay_from` in both whitelists;
   `scripts/otr_canonical_api_run.py --replay-from`.
9. `scripts/otr_freeze_replay_bundle.py` (new) and `scripts/otr_verify_replay.py` (new, offline
   verifier of manifest, receipts, seeds and original-vs-replay equality).
10. Tests: the offline set in D6' plus the negative paths, the workflow four, the whitelist
    parity, the signature parity; live proof: render one canonical episode on the shipping
    defaults, freeze it, replay it twice, publish all three, verify seeds equal and the master
    byte-identical, and that the cheap family's receipts stamp too.

## 12. r4 convergence (Sonnet 5, in-process; roster: Antigravity r1, Codex r2, Cursor r3, Sonnet r4 -- one seat per round, as the operator ruled 2026-09-02)

CONVERGED. All 21 must-fixes (4 + 9 + 8) traced into sections 6 / 8 / 10 / 11 as taken, with r2 must-fix 9 scoped and the reason confirmed at the files (the per-segment request loop is engine-generic; no LTX boundary is edited). No contradiction between the folds. No new must-fix: no other node between 62 and 92 calls the writer LLM or mints a freeze timestamp; the dispatcher has no IS_CHANGED override and the replay path short-circuits before its cache lookup; ShotLock strict post-audio overlay passes only because the import rebinds the singleton before any peek, which section 10 already requires. **The one thing to settle at coding time:** the haunted engine sampler_inputs field list is read from the live `_build_render_request` and `_recipe_receipt` of `eng_ghost_signal.py` / `eng_ghost_signal_official.py` and pinned in section 13 before `actual_request_sha` is written -- a wrong field set is exactly how two A/A nulls would differ by construction.

## 13. Coding receipts (2026-09-02, one window, in the `instrument` git worktree)

### 13.1 The pinned haunted `sampler_inputs` (the r4 open item)

`GhostSignalEngine.sampler_inputs_for(request)` returns exactly these keys, every one read from
the SAME constant or resolver the graph builder uses (no second copy of a number anywhere):

| key | source |
|---|---|
| `checkpoint` | `GHOST_CHECKPOINT_NAME` |
| `motion_module` | `self.motion_module_name` |
| `adapter`, `adapter_strength` | `self.lora_name` / `self.lora_strength` (`None` strength when the lane carries no adapter -- the base engine) |
| `steps`, `cfg`, `sampler`, `scheduler`, `denoise`, `beta_schedule` | `GHOST_STEPS`, `GHOST_CFG`, `GHOST_SAMPLER_NAME`, `GHOST_SCHEDULER`, `GHOST_DENOISE`, `GHOST_BETA_SCHEDULE` |
| `canvas_w`, `canvas_h` | `GHOST_CANVAS_W`, `GHOST_CANVAS_H` |
| `context_length`, `context_overlap`, `context_fuse_method`, `context_use_on_equal_length`, `context_start_percent`, `context_guarantee_steps` | the six `GHOST_CONTEXT_*` constants |
| `source_fps`, `target_fps`, `hold_factor` | `GHOST_SOURCE_FPS`, `self.target_fps`, `self.hold_factor` |
| `source_request`, `unique_source_count` | `self._build_render_request(request)` -- the plan the builder itself renders from |
| `latent`, `init_image` | the literals `"EmptyLatentImage"` / `None` (the lane is text-to-video; a still-in peer will put a hash here, which is the point of carrying the key now) |

`model_artifacts()` returns `[("checkpoint", path), ("motion_module", path)]` plus `("adapter",
path)` when `lora_name` is set. Both are what `build_actual_receipt` hashes into
`actual_request_sha` (through `sampler_inputs` and `model_artifacts`) alongside the request's
text, negative, seed, target frame count, canvas and still hash. Wall time, peak VRAM and the
run id are stamped on the receipt but excluded from the hash by construction
(`_RECEIPT_CAUSAL_KEYS`). `tests/test_render_receipts.py` pins that two identical requests hash
equal and a changed cell hashes different.

### 13.2 What shipped, against the contract in section 11

Items 1-9 as written. Deviations, each deliberate:

* Section 11 item 6 names `nodes/batch_character_voices.py` and `nodes/announcer_voice.py`; the
  pass-through lives once in `nodes/_otr_voice_node_common.py::generate`, which both nodes call
  -- one seam, both nodes covered (`tests/test_canonical_replay.py` exercises both classes).
* `EpisodeAssembler` grew the `replay_descriptor` forceInput socket in INPUT_TYPES as well as
  the `assemble()` parameter (the first cut declared only the parameter; the contract validator
  refused the canonical's link 289 as a rogue socket -- caught by
  `tests/test_workflow_contract_validation.py`, fixed before merge).
* The writer's `replay_from` is the LAST optional entry, after the `gate_in` socket, not before
  `gguf_quant` where the first cut put it. INPUT_TYPES order IS the widgets_values order; the
  first placement would have rebound `gguf_quant`'s saved value to `replay_from` on every graph
  (BUG-LOCAL-097). Caught by `tests/test_workflow_json_guardrails.py::TestWidgetOrderVsInputTypes`
  and the S5 tail pins, which now pin `order[33] == "replay_from"`, `len(order) == 34`, and the
  saved vector at 33.
* `workflows/otr_story_only.json` is hand-maintained (not a `build_variants` output) and is the
  one other graph carrying the writer; it got the same trailing descriptor + `""` slot.
* The four hand-kept `workflows/variants/*.env.json` recipes carry a copy of their variant's
  `master_hash`; regenerating the variants moved every hash, so the copies were re-synced by
  exact string replacement (`tests/test_remaining_video_contracts.py` is the pin).
* Every replay check that parses `meta` off a wire treats a non-dict wire (the legacy parser
  list) as "not a replay" and falls through to the node's historical loud path -- the sequencer
  test `test_sequencer_legacy_list_raises` caught an `AttributeError` shadowing the ValueError.

### 13.3 The build defect that never reached the tree

The replay-node patch script tested "already applied" by the ANCHOR still being present. An
insert-style hunk keeps its anchor inside the replacement, so every re-run (there were several,
fixing anchors) re-inserted the block: four copies of the replay block in `production_ledger.py`,
three in `scene_sequencer.py` and `otr_image_gen_dispatcher.py`, two in `OTR_LedgerFreezeCascade.py`.
Python took the last definition each time, so the targeted tests were green over duplicated
code. Found by counting definitions across the diff; fixed by reverting the ten files to HEAD
and re-applying once with the correct test (the REPLACEMENT text present == applied). Rule
recorded in memory; the receipt is `grep -c "def _assemble_replay"` == 1 and friends.

### 13.4 Worktree-only test artefacts (verified in the main checkout after the merge, not here)

* `tests/test_credits_roll_spec.py` (44): `_git_short_sha` reads `.git/HEAD`; a worktree's
  `.git` is a file.
* `tests/test_workflow_json_guardrails.py::TestWidgetOrderVsInputTypes`: `_resolve_ncm` imports
  `custom_nodes.ComfyUI-OldTimeRadio` from `PACK_ROOT.parent.parent`, which from the worktree
  is the MAIN checkout's registry (proven: it resolved
  `custom_nodes\ComfyUI-OldTimeRadio\__init__.py` and reported no `replay_from`).
* `tests/test_w45_campaign_bank_pinning.py` writes under `tmp/`, an untracked dir.

### 13.5 Tests run before the merge

Offline: `tests/test_canonical_replay.py` (24), `tests/test_render_receipts.py` (12), the
workflow four (`build_variants --check` 91/0, link-target indexes, widget-value alignment,
canonical widget-input parity), the whitelist parity (`tests/test_workflow_apply.py`), and every
test file touching a changed node module (284 files; the only failures left are the 13.4
artefacts). The full suite and the Bible run in the main checkout after the merge, and the live
proof (render, freeze, replay twice, verify) runs when the adapter sweep releases the GPU.

**After the merge (main checkout, 969f2578 on v2.0-alpha, pushed 17:45 at the sweep's leg-3
boundary):** the Bible regression is green (22 passed, 27 skipped, 3 xfailed); the full suite
ran to completion with exactly four failures, all append canaries that pin what sits LAST on
node 7 / node 62 / the link counter and fired, as designed, on the additive edit:
`test_freeze_cascade_v2_ports` (v2_ledger_json now fans out to CastLock AND the assembler),
`test_google_video_sfx_workflow` (`last_link_id` 288 -> 289), and two in `test_ltx25_foley_bed`
(`video_policy_json` is second to last on the assembler, `replay_descriptor` last). The pins
were moved with dated notes, re-run green (38 passed), and the full suite re-run on the fixed
tree (dc4beea5) is the receipt: **12921 passed, 122 skipped, 1 xfailed, 0 failed in 10:19**
(summary line read from the run file, not inferred). The three worktree-only artefacts of 13.4
did not reproduce in the main checkout.

### 13.6 Finished-diff review

Sonnet 5, one pass, scoped to the named functions of the diff (roster: Antigravity r1, Codex r2,
Cursor r3, Sonnet r4, Sonnet QA). Verdict: findings, eight. Grounded and disposed:

1-3. The writer widget order and the duplicated blocks (13.2, 13.3) -- the reviewer read the
   pre-dedupe tree; both were already caught by the tests and fixed. Its independent check that
   the first placement would have handed `"Q8_0"` to `replay_from` on every normal render (so
   every canonical run would have died in `import_replay_bundle`) is the right reading of
   BUG-LOCAL-097 and is why the placement rule is now a comment beside the widget.
4. `EpisodeAssembler.__doc__` lost: the patch inserted the two replay methods as the first
   statements of the class body, ahead of the docstring. TAKEN -- docstring moved back above
   them, `ast.get_docstring` verified.
5. The writer's replay branch returned a float on the INT `estimated_minutes` slot. TAKEN --
   `int(round(...))`.
6. Node 7's `replay_descriptor` and node 1's `replay_from` descriptors lacked the siblings'
   `shape: 7` / `localized_name`. TAKEN -- normalised in the canonical and in
   `otr_story_only.json`, variants regenerated, `--check` 91/0.
7. `find_master`'s fallback glob could freeze a stale `pending_*_master.wav`. TAKEN --
   `pending_*` excluded, the episode-id-named master preferred over mtime.
8. `import_replay_bundle` rebinds the singleton before the asset copies, so a failed import
   leaves `_CURRENT` on a half-built workspace. DECLINED, with the reviewer's own reason: the
   graph aborts on the same exception and the next writer run rebinds through `new_ledger`
   anyway; a rollback would add a second code path to the seam for a state nothing reads.

Verified correct by the reviewer (kept as the receipt): the base engine carries `lora_name`,
`lora_strength` and `_lora_path`, so `sampler_inputs_for` never raises on the plain lane;
`still_sha256` hashes content, `model_artifacts` is an ordered list, so `actual_request_sha` is
run-stable; every engine's `canonicalize()` returns a dict, so no clip is a bare path at the
receipt; `_stamp_render_trace` is safe under `OTR_TEST_MODE` and on an empty receipt list, and
lets `LedgerStampError` propagate as the neighbouring stamps do; the popped `meta.paths` is
rebuilt unconditionally by `Ledger.save`; both `_same_durable_run` and `_same_frozen_episode`
gate on `replay_workspace_id` before the timestamp; every pass-through returns the arity and
types its node declares and runs before any model, LLM or GPU work; link 289 appends to node 62
output 6 rather than replacing it.
