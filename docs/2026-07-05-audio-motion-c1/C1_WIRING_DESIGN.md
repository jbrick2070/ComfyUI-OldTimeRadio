# S-C C1 -- audio_motion_profile: producer wiring design (the real fork)

**Status 2026-07-05 (operator at yoga):** EXTRACTION CORE SHIPPED @d60bf371
(`nodes/_otr_audio_motion.py` + `tests/test_audio_motion_profile.py`, 14 tests,
read-only, byte-identity proven). What remains is the PRODUCER wiring -- and it
is a genuine fork with frozen-JSON stakes, so it is kibitz-gated per the handoff
("kibitz if the producer-node design is a real fork").

## What C1 says
`audio_motion_profile` extraction + ledger stamp: "schema field, producer node,
IS_CHANGED/cache key, PROVE conditioning WAVs never replace the master." C2
(per-engine consumers + HuMo phrase-chunking) is DEFERRED. So nothing consumes
the profile yet -- this chunk only produces + stamps it.

## The core (already shipped, wiring-agnostic)
`nodes/_otr_audio_motion.py`:
- `analyze_wav(path) -> {duration_s, rms_dbfs, peak_dbfs, dynamic_range_db,
  silence_ratio, onset_s, brightness, speech_vs_music}` (READ-ONLY; soundfile).
- `build_audio_motion_profiles(rows, resolver)` -> per-row profile list
  (`ok`/`reason` bookkeeping; deterministic; no disk writes).
- `stamp_ledger_audio_motion(ledger, profiles)` -> sets
  `ledger["audio_motion_profiles"]` + `meta.audio_motion_profile` ONLY.
- `PROFILE_VERSION = "amp-1"`.

Either wiring below just supplies a `resolver(row)->wav_path` and calls these.

## Grounded graph facts (real workflows/otr_scifi_16gb_full.json, 2026-07-05)
**CORRECTED after kibitz r1 (codex-found, verified):** the chain is
ShotLock -> ImageGenDispatcher -> VideoRenderBatch, NOT ShotLock -> VideoRenderBatch.
- link 256 = node90 `OTR_ShotLock`[out0] -> node91 `OTR_ImageGenDispatcher`[in0].
- link 260 = node91[out0 patched_ledger_json] -> node92 `OTR_VideoRenderBatch`[in0].
- link 267 = node91[out1 image_done] -> node92[in2].
- link 264 = node7 `OTR_EpisodeAssembler`[out1 output_path] -> node92[in1 master_audio_path].
- node 92 OUT `clip_manifest_json`=[261,271,275]. `last_node_id`=95, `last_link_id`=276.
- `render_driver._slice_master_audio(master_path, start_s, dur_s, master_hash)`
  ALREADY slices the frozen master per beat, read-only, cache-keyed by
  `ledger['audio']['master_audio_sha256']` (`SLICER_VERSION="2"`). This is the
  natural `resolver` -- per-beat conditioning WAV without re-inventing slicing.

## Option A -- dedicated producer node (what the plan literally says) [HARDENED r1]
New `OTR_AudioMotionProfile` (id 96) inserted **91 ImageGenDispatcher -> [96] ->
92 VideoRenderBatch**: re-point link 260 through 96, add a master_audio_path fan
(from node 7) to 96, output patched ledger to 92; keep node 91's image_done
(link 267) straight to 92. `OUTPUT_NODE=False`; NO custom IS_CHANGED in v1 (rely
on input-hash caching + the existing slice cache key -- codex CUT, accepted).
- Row universe: ONE profile per VIDEO SHOT (`ledger["video"]["shots"]`), timing
  from each shot's start_s/dur_s; skip (ok=False) on missing timing.
- resolver = `render_driver._slice_master_audio(master, start_s, dur_s, sha)`
  (standalone, READ-ONLY, cache-keyed on master_audio_sha256).
- Durable stamp: resolve the in-flight ledger path
  (`_otr_ledger.in_flight_ledger_path()`) + `save_ledger_safe()` (fail-soft),
  AND return the mutated wire JSON downstream. Acceptance = OTR_WorkflowValidator
  + JSON round-trip + link/widget audit (+ operator eyeball on the frozen graph).
- PRO: matches the spec; ALWAYS runs (even procgen-only episodes with no motion
  render); a clean standalone stamp; independent cache key.
- CON: edits the FROZEN production JSON (new node + re-pointed link 260 + a new
  link + positional widget slot) -- exactly the surface that burned this repo
  (BUG-LOCAL-097 positional drift; 2026-06-13 unwired-node miss). Needs the
  workflow validator + link/widget audit + an operator graph eyeball.

## Option B -- fold into VideoRenderBatch (zero JSON change)
Node 92 already holds BOTH the patched ledger AND master_audio_path AND already
slices per beat. Compute the profile from the slice it already makes, stamp the
ledger in-node, save. NO new node, NO JSON edit, live immediately.
- PRO: zero frozen-JSON risk; reuses the existing slice (no double ffmpeg);
  ships without an operator graph eyeball.
- CON: only runs when a per-beat MOTION render happens (procgen-only / still
  episodes produce no profile); diverges from the "producer node" wording; the
  cache key rides VideoRenderBatch's (OUTPUT_NODE=True, always-runs anyway).

## Recommendation (Claude, anchored)
Lean Option A ONLY if the operator wants the profile on EVERY episode regardless
of render mode; otherwise Option B is the senior-pragmatic, zero-risk choice and
the frozen-JSON edit is avoided. Because C2 has no consumer yet, the robustness
gap of B (no profile on procgen-only runs) costs nothing today. My default: B
now (safe, live), promote to A when C2 lands and a consumer needs the profile on
non-motion episodes -- at which point the operator is present to eyeball the JSON.
KIBITZ to converge; operator eyeball gates any JSON edit.
