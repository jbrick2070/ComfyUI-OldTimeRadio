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
