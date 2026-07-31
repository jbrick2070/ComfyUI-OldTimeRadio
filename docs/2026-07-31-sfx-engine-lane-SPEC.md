# Generated SFX Engine Lane -- BUILD SPEC

**Status: CONTRACTS SETTLED, BLOCKED ON A PREREQUISITE. Not yet code-ready.**

Product of a local kibitz r1->r4 arc, 2026-07-31 (operator request). Panel =
Codex `gpt-5.6-sol` @ high + Antigravity `Gemini 3.6 Flash (High)`; Claude as
code-grounded anchor panelist and sole judge. Supersedes
`docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`, whose arc
was wrong in four structural places against the tree as it now stands.

Round artifacts (judgments, per-round plans, both agents' reviews) live in
`kibitz-runs/2026-07-31-sfx-engine/`, which is GITIGNORED. This file is the
tracked output and carries contracts only -- the review narrative is deliberately
left behind.

**HONEST STATUS.** r4 did NOT return convergence: Codex's final verdict is still
"no", and it raised four new must-fixes that are folded in below. Antigravity was
absent from r4 -- it hit a hard quota wall (`429 RESOURCE_EXHAUSTED`), so the
final round is one-seat and the convergence check is correspondingly weaker.
A fifth round is NOT recommended: what remains is not a review problem.

**THE PREREQUISITE THAT BLOCKS EVERYTHING.** No SFX checkpoint exists on this
host -- only `stable_audio_3_small_music.safetensors` (2.11 GiB) and
`stable-audio-open-1.0.safetensors` (4.52 GiB) under
`C:\ComfyUI-Models\checkpoints`. No audio engine anywhere declares an `sfx` role
(roster: `char_voice`, `announcer_voice`, `music`) and the role gate is enforced
(`_otr_audio_engines/registry.py:162-165`). `eng_stable_audio_3` is
`roles=("music",)` with music-specific prompt shaping and a `..._music`
default checkpoint -- it cannot be pointed at an SFX model. Until a real
checkpoint is pilot-proven with a MEASURED peak-VRAM number, no engine ID, no
tier label and no VRAM claim in this spec is anything but a guess.

---

## 1. Topology

    80 OTR_CastLock ──ledger_json──────────▶ 96 OTR_CueDirector
    96 ──sfx_plan_json───────────────────────▶ 97 OTR_SfxCueRenderer
     3 OTR_SceneSequencer ──timed_ledger_json▶ 97      (NEW 3rd output)
     3 ──scene_audio────────────────────────▶ 97      (true clock + length)
    97 ──sfx_cue_audio─────────────────────▶ 7 OTR_EpisodeAssembler
    97 ──sfx_cue_manifest_json─────────────▶ 7

**Two nodes, because authoring and placement need different inputs at different
times.** Cue AUTHORING reads the frozen ledger and has no timing. Cue PLACEMENT
needs `lines[].start_s`/`dur_s`, which are seeded `None`
(`production_ledger.py:1179-1180`) and first written by SceneSequencer
(`scene_sequencer.py:1021-1027` computes, `:1117-1129` patches).

**Node 97 MUST be WIRED from node 3, never read the ledger from disk.** Without
a LiteGraph edge, ComfyUI's DAG sorter may schedule 97 first and every
eligibility check evaluates against `None`. `in_flight_ledger_path` also falls
back to a newest-mtime walker (`_otr_ledger.py:414-419`) whose own docstring
records it returning a 6-day-old episode on a live soak.

**SceneSequencer gains a third output**, appended so nothing shifts:
`RETURN_TYPES = ("AUDIO","STRING","STRING")`,
`RETURN_NAMES = ("scene_audio","render_log","timed_ledger_json")`. It returns the
ACTUALLY PATCHED ledger and fails LOUD when required timing could not be
produced; today that write-back is best-effort and suppresses failures
(`:1081-1155`). "Required timing" = every unique, non-skipped
character/announcer line selected by the run receives finite non-negative
scene-space timing; partial `start_line`/`end_line` runs need their own stated
rule.

## 2. Node contracts

**Node 96 `OTR_CueDirector`** -- required force-input `ledger_json`; no widgets;
`FUNCTION="direct"`; `RETURN_TYPES=("STRING","STRING")` named
`("sfx_plan_json","cue_report")`.

**Node 97 `OTR_SfxCueRenderer`** -- required inputs in serialized order
`sfx_plan_json`, `timed_ledger_json`, `scene_audio`, then the engine COMBO;
`FUNCTION="render"`; returns AUDIO, manifest STRING, render-log STRING.
**No `done` output** (the AUDIO/manifest edges already gate EpisodeAssembler and
teardown must complete before return). **No seed widget** -- per-cue seeds derive
from `meta.episode_seed` and live in the authored spec; two seed authorities
create replay ambiguity.

Both register through `nodes/_otr_class_registry.py` (`NewNodeSpec`), merged into
`NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` by `__init__.py:329-363`, with
`test_class_registry.py` extended BEFORE any workflow node is added.

## 3. CueDirector is an LLM pass on the CREATIVE slot

`PRODUCTION_SPRINT_LESSONS` §3 reserves sound-design decisions for a model, so
this is not left open. Bind to `meta.creative_writing_model` through the existing
resolver pattern -- `request_slot("creative", ...)`, `policy_from_meta(meta)`,
`load_config_from_meta(meta, "creative")` (`otr_shot_lock.py:867-882`). No new
widget and no new slot.

It owes the full bounded-pass contract: all five representations in lockstep
(`PRODUCTION_SPRINT_LESSONS` §2:38-45 -- base prompt, typed schema, worked
fixture, parser+validator, repair prompt), a retry ladder by failure class (§4),
a context budget sized from the real artifact (§5), the finite slot ladder
(§32:534), and explicit named transitions for malformed output, repair
exhaustion, auth/rate-limit, cancellation, and structural corruption.

**Node 97 must evict CueDirector's LLM residency before loading the SFX model** --
nodes 3 and 96 are sibling branches with no fixed execution order.

## 4. Reading the ledger

- Beats come from top-level **`ledger["beats"]`**. `meta.outline.beats` does not
  exist -- `meta["outline"]` is never written anywhere, and the fallback was
  deliberately deleted (`_otr_ledger_freeze.py:285-292`).
- Beats rows carry **`line_ids` (PLURAL LIST)** and exactly eight keys; they have
  no `speaker_role` and no `beat_intent`, and `set_beats` drops anything else
  (`production_ledger.py:1088-1097`, `:1188-1197`). Iterate `beats[].line_ids`,
  resolve into the unique `lines[].line_id` map, and read role from the LINE.
- **Join on `line_id`.** `dialogue_slot_id` is `None` on every non-voiced beat by
  construction (`_otr_outline.py:1593-1598`) and has no uniqueness validation
  anywhere -- the freeze validator checks `line_id` only (`:311-314`).
- On the current writer path `line_id === beat_id === beats[].line_ids[0]`
  identically (`:1169`, `:1171`, `:1194`). That is ONE writer's property, not the
  schema (`set_lines` reads them independently). **Never lean on it.**

## 5. Identity envelope

`sfx_plan_json` carries `episode_id`, `meta.freeze_timestamp` (there is no
`freeze_id` -- zero matches repo-wide; identity is compared by
`_same_durable_run`, `production_ledger.py:456-457`), and
**`sfx_authoring_source_sha256`**: a hash over an IMMUTABLE PROJECTION --
episode id, freeze timestamp, and the authored beat/line fields, EXCLUDING
timing, audio, paths and mutable telemetry. A full content hash is unsatisfiable
because node 3 legitimately mutates the ledger between 96 and 97. Node 97
recomputes the same projection and fails LOUD on any mismatch.

## 6. Clock and coordinates

Node 97 canonicalizes every engine result to **CPU float32 mono, resampled to
`scene_audio.sample_rate`** BEFORE the post-render bounds check, packing, hashing
and WAV persistence -- so coordinates and counts are both scene-domain and
comparable. Take the clock from the `scene_audio` TENSOR, never a constant:
SceneSequencer is hardcoded 48 kHz (`:859`) but AudioEnhance may emit 24-96 kHz
(`audio_enhance.py:293-294`).

Cue coordinates are resolved and PERSISTED in **scene_audio space**, then
converted to **master_mix space** by EpisodeAssembler in the same pass that
already converts `lines[]`, `music[]` and `clips[]` (`:1552-1584`). This is what
prevents every cue landing ~10 s early, since EA prepends the opening theme
(`:1299-1304`). EA may resample once more to `main_waveform` and scale
start/count under one stated rounding rule; device, dtype and channel count must
match before slice-add.

## 7. Eligibility

With `anchor_start`/`anchor_end` from `lines[]` and `cue_samples` at the scene
rate:

- `before`: `start = anchor_start - offset_samples - cue_samples`
- `during`: `start = anchor_start + offset_samples`
- `after`:  `start = anchor_end + offset_samples`

Renderable iff `0 <= start` and `start + cue_samples <= base_scene_samples`.
Otherwise a NAMED non-rendering disposition -- `underruns_scene_start` /
`overruns_scene_end` -- recorded with its computed coordinates. **Never clamped,
never crashed.** Structural problems (unknown `line_id`, identity mismatch,
malformed plan) fail LOUD.

**Bounds are checked TWICE**: on `requested_duration_s` before inference, and
again on the ACTUAL post-render sample count, because an engine may return a
different duration. An actual overrun is omitted from the AUDIO batch and
recorded with a named disposition.

## 8. Manifest (new -- the music one cannot be reused)

`_otr_cue_manifest.PLACEMENTS = ("opening","closing","interstitial")` with a hard
raise at `:171-175`, plus `_REQUIRED_ROW_FIELDS` (`:29-37`) and contiguous
`batch_index` coverage (`:210-219`). A `before`/`during`/`after` row fails it.

Add `SFX_PLACEMENTS = ("before","during","after")` with
`validate_sfx_manifest()` / `parse_sfx_manifest()` alongside the music validator.
Required per row: `batch_index`, `cue_id`, `anchor_line_id`, `placement`,
`sample_rate`, ACTUAL `sample_count`, `resolved_start_sample`,
`authored_spec_sha256`, `episode_id`, `freeze_timestamp`. **No `disposition` on
manifest rows** -- the manifest maps ACTUAL audio rows; the ledger owns
non-rendering and failed-attempt dispositions.

`pack_audio_batch` right-pads to the longest clip and consumes the true `T` as a
loop local without returning it (`_otr_audio_engines/base.py:150-153`), and
refuses mixed rates (`:136-140`) -- which is exactly why actual counts ride the
manifest.

**Zero-cue contract:** `empty_audio_batch()` is `[1,1,0]`, while the music
validator equates row count with `waveform.shape[0]`. `validate_sfx_manifest`
special-cases `T == 0`: require zero rendered rows; otherwise require
`B == row_count`.

## 9. Durable persistence -- Node 97 is the sole ledger writer

Section: **`ledger["audio"]["generated_sfx"]`**, scoped strictly to this new
generated lane.

Node 97 validates the plan, materializes authored rows and preflight
dispositions, atomically persists an attempt receipt BEFORE inference, then
persists render-owned fields or the failure receipt before returning or raising.
It resolves the target through the timed ledger's exact
`meta.paths.ledger_path`, **never** `in_flight_ledger_path()`. EpisodeAssembler
may only reconcile and shift matching render-owned fields.

Authored fields and render-owned fields are enumerated separately, with a
complete state enum. Reconciliation recomputes authored identity and updates only
render-owned fields (`BUG-12.65` / `PBUG-20260721-13`: reconcile the rendered
manifest into the ledger BEFORE timeline mutation).

**`_merge_with_disk` must gain a same-freeze, `cue_id`-keyed merge**
(`production_ledger.py:1477-1539`). Today `audio` is preserved as ONE BULK OBJECT
(`:1513-1519`): disk `audio` is adopted only if in-memory is absent or falsy, so
one non-empty field discards the whole disk object. `meta` already received the
nested fix (`:1520-1539`). **This is a latent defect today** -- the SFX lane is
simply the second `audio.*` writer that would expose it. Test stale-wire/new-disk
AND new-wire/stale-disk.

Paths resolve against `ledger.meta.paths.episode_root`
(`_otr_ledger.py:224-227`; note `_build_meta_paths` is private and only
`ledger_path`/`episode_root`/`audio_dir` are unconditional), stored
**episode-relative**, with containment validation and a deterministic safe
`cue_id` pattern. Relative paths remove the `BUG-12.66` rename coupling by
construction rather than depending on the rebaser.

## 10. Mixing and coexistence

Mix into EpisodeAssembler's `main_waveform` BEFORE theme segmentation and BEFORE
`_master_loudness`. Mixing earlier would push SFX through AudioEnhance's
speech-oriented resampling, stereo widening, low-pass and tape processing
(`audio_enhance.py:278-326`).

EpisodeAssembler gains `sfx_cue_audio` (AUDIO) and `sfx_cue_manifest_json`
(STRING), **appended at the very END of `optional`** so no saved positional
widget vector shifts. Declaration and signature land in the SAME commit.

Overlap gain is a fixed episode-level `1 / max_simultaneous_renderable_cues`.
**Assert finiteness and bounds on the SFX BUS ALONE** -- a `[-1,1]` assertion on
the combined scene is unsatisfiable, since a bounded bus plus a near-unity main
waveform legitimately exceeds unity. The existing tanh soft limiter
`_master_loudness` (`scene_sequencer.py:171`, applied `:1355-1363`) does the
mastering; do not clamp before it.

**A provider-stem SFX lane already ships and must not be disturbed:** four
`google_vid_sfx_*` engines extract provider audio to `.sfx.wav` stems
(`eng_google_vid_sfx.py:403,439-441`, registered `registry.py:539-561`), plus
`eng_cloud_video.py:916-918`; truth rides `clip_manifest_json` and on-disk stems
(the ledger has NO SFX keys -- `sfx[]` was ripped 2026-07-01,
`production_ledger.py:1490-1492`); `OTR_MasterAudioMux` mixes the bed at the
TERMINAL node, positioned from VIDEO timing and `adelay`-quantized to integer
milliseconds. The two lanes are independent and additive: generated manifests go
ONLY to EpisodeAssembler, provider stems ONLY to MasterAudioMux. **Do not exempt
generated SFX from the mux's PCM integrity gate** -- it builds its reference from
the already-generated master plus the provider bed, so generated audio already
inside the master does not invalidate the comparison.

## 11. `IS_CHANGED`

Node 96: the immutable authoring projection plus prompt/schema/model/policy/
load-config versions. Node 97: plan + timing identity, profile, adapter, prompt
version, pinned checkpoint digest with path/size/mtime, and durable cue-file
existence/hash. **Never hash multi-GiB weights during validation.**

## 12. Engine adapter

Follow `eng_stable_audio_3`'s structure: no heavy or optional imports and no
filesystem/model loading at module import; native ComfyUI loaders so model
management owns residency; teardown in `finally` before return, with
cancellation checks between cues. **Do not copy
`stable_audio_theme._write_cue_wav`** (`:421-452`) -- bare whole-body `except`
returning `""` unlogged, while this spec declares those paths durable truth.
Registration touches the capability row, `_LEGACY_FIRST_ENGINES`,
`config/audio_engine_profiles.yaml` and `_otr_audio_engines/__init__.py`.

**An empty roster is not a shippable intermediate state.** `build_engine_combo`
ends `"default": engines[0]` (`_otr_voice_node_common.py:274`) -- an unguarded
IndexError inside `INPUT_TYPES`, i.e. at node-registration time. There is NO null
engine and no silent-stub fallback.

## 13. Workflow wiring

Nodes from **96** (`last_node_id = 95`). Links from **285** (`last_link_id = 284`
and link 284 EXISTS: `[284,12,0,90,4,"STRING"]` =
`OTR_SignalLostVideo.video_path -> OTR_ShotLock.gate_in`, a SEQUENCING gate).

| link | from | to |
|---|---|---|
| 285 | 80:0 | 96:0 `ledger_json` |
| 286 | 96:0 | 97:0 `sfx_plan_json` |
| 287 | 3:2 | 97:1 `timed_ledger_json` |
| 288 | 3:0 | 97:2 `scene_audio` |
| 289 | 97:0 | 7:9 `sfx_cue_audio` |
| 290 | 97:1 | 7:10 `sfx_cue_manifest_json` |

`last_node_id = 97`, `last_link_id = 290`. Node 96 `widgets_values=[]`; node 97
`[qualified_engine_id]`. Node 7's new inputs are socket-only and add no
`widgets_values`. Every source output's `links` list must be updated.

## 14. Build order

0. **PREREQUISITE:** pilot-prove an SFX checkpoint -- identity, license/source
   receipt, SHA, text encoder, native loader calls, supported controls, duration
   granularity, sample rate/channels, output shape, and MEASURED peak VRAM
   against the 16 GiB budget. Fix the exact max-cues / per-cue-seconds /
   per-episode-seconds constants here, enforced deterministically in CueDirector
   output validation AND again before any engine load. An LLM instruction is not
   back-pressure.
1. `_merge_with_disk` `cue_id`-keyed merge + tests (independently valuable -- it
   fixes a latent defect today).
2. SFX manifest validator + tests.
3. Resolve `docs/2026-07-31-NEWBUG-scene-sequencer-music-sockets.md`, then add
   the registry-wide INPUT_TYPES-to-FUNCTION signature-parity test. **In that
   order** -- the test fails immediately on the existing mismatch.
4. Node 96 CueDirector + tests.
5. SceneSequencer third output + tests.
6. Node 97 renderer + eligibility + persistence + tests.
7. EpisodeAssembler inputs, mixing, coordinate conversion + tests.
8. Canonical workflow wiring + validator/link/widget/signature audits.

Steps 1-3 do NOT depend on step 0 and may proceed immediately. Only node 97's
registration and the canonical wiring are gated on the engine.

## 15. Tests

New: `tests/test_sfx_contracts.py`, `test_cue_director.py`,
`test_sfx_cue_renderer.py`, `test_sfx_episode_assembler.py`.
Extend: `test_cue_manifest.py`, `test_sequencer_ledger.py`,
`test_ledger_merge_ownership.py`, `test_production_ledger.py`,
`test_audio_engine_registry.py`, `test_engine_profiles.py`,
`test_class_registry.py`, `test_full_workflow_v2_audio_wiring.py`,
`test_workflow_json_wiring_invariants.py`.
Plus a combined-lane regression: generated SFX enters only EpisodeAssembler,
provider stems only MasterAudioMux, both additive, neither mixed twice.

## 16. Verify-at-build

- Checkpoint identity, license/source receipt, SHA, encoder, loader calls,
  controls, duration granularity, rate/channels, output shape.
- Peak VRAM vs the 16 GiB budget; teardown returns near baseline across two
  consecutive cue batches.
- Identical per-cue seeds reproduce the promised determinism; record any
  nondeterminism honestly.
- Exercise zero-cue `[1,1,0]`, one-cue, multi-cue, actual-duration
  underrun/overrun, and mixed-rate rejection.
- `IS_CHANGED` invalidates on profile/checkpoint/prompt/artifact change and
  reuses an unchanged durable render.
- Package startup with no heavy import side effects; a non-empty registered SFX
  combo.
- `OTR_WorkflowValidator`, JSON round-trip, link integrity, widget-vs-live
  `INPUT_TYPES` audit, signature parity, canonical-link assertions.
- Focused tests, full Windows suite, Bug Bible.
- Reset the box, run the REAL canonical workflow: `RESULT SUCCESS`,
  `obs_publish OK`, asset under `otr/episodes/<ep>/`, final under `otr/obs/`.
- Title rename/reload preserves every generated-SFX row and resolves every
  relative asset under the renamed episode root.
