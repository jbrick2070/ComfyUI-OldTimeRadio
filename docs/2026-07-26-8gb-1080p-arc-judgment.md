# JUDGMENT -- 8 GB -> 1080p for `ltx_8gb` (kibitz r1 -> r4, + one Fable pass)

Full 4-round local arc, 8 agent calls: codex `gpt-5.6-sol` (high) + agy
`Gemini 3.6 Flash (High)`, both pinned and VERIFIED every round. Plus ONE Fable
call, operator-requested, on the viewer question. $0 external. Driver anchor
written before the r1 fan-out. Grounded against the real Windows files at HEAD
`78df72b9`. Run: `kibitz-runs/2026-07-26-8gb-1080p/r{1,2,3,4}/`.

**No code written. This is the architecture.**

---

## 1. THE CANVAS IS 512x288, AND EVERY SOURCE AGREES

**Fable, on the viewer question:** *"Softness is a state; a motion reset is an
event. Viewers habituate to states in about a minute -- soft becomes 'the look'.
The visual system never habituates to events... Soft reads as OLD; stutter reads
as BROKEN. Old is your brand."*

At 512x288 a beat plays as ONE continuous shot. At 1024x576 the same beat
becomes four or five stitched clips -- a motion restart roughly every 1.5
seconds, hundreds per episode, under dry narration. The 4x pixel advantage is
largely nominal: an 8-step distilled 2B model does not put real detail into
576p, so the choice is between two grades of soft while the join count scales
exactly as advertised.

Independently, codex found the arithmetic reason: **512x288 and 1024x576 are the
only exact-16:9 rungs that are also /32-clean.** `832x480` is 26:15 -- delivered
it becomes 1872x1080 with side bars inside the 1920 frame.

**So `config/profiles/otr_8gb_ltx.json`'s 512x288 was right all along, for a
reason nobody had written down, and the "quality floor is 832x480" instinct from
BUG-LOCAL-412 would have introduced pillarboxing on every episode.**

Pillarbox: never (all three). If 832 ever wins on composition, render 832x480
and CROP to 832x468 (exact 16:9) -- but that contingency is CUT from this build
(codex r4 CUT 1); 512x288 is selected.

Acceptance is mechanical: `render_canvas` equals 512x288, the final probe reports
1920x1080, and the scale computation requires **zero pad area**.

## 2. THE BLOCKER NOBODY HAD NOTICED -- IT OUTRANKS THE CANVAS

**`session_identity` appears in exactly ONE file, `beat_session.py`. No adapter
declares it.** Verified by exhaustive grep over `nodes/`.

`BeatSession.open()` (`:147-164`) refuses -- before the weights land, "NO
FALLBACK" -- any multi-segment beat whose engine cannot say what its handles ARE.
**So every multi-segment beat is refused today, for all 31 engines.** The
canvas fix is necessary and not sufficient; a long beat is rejected before the
canvas is ever consulted.

Second half, also confirmed: `Ltx8gbEngine.load()` only resolves node CLASSES,
and the graph carries its own `ckpt`/`clip` loader nodes, so every segment
re-executes `CheckpointLoaderSimple` + `CLIPLoader`. "ONE model load per beat"
is not true for this adapter yet. That is O3 from the 7b judgment, now on the
critical path rather than after it.

## 3. THE CROSS-TIER TRAP -- this would have broken WAN

**`max_render_frames` is NOT a planning cap.** Its shipped contract is a
per-clip NATIVE-RENDER ceiling that leaves beat length unchanged: **WAN reads
17, renders a short clip, then PING-PONGS it to the full beat.** Applying it
before `partition_beat()` would turn every WAN beat into a pile of 17-frame
renders and silently rewrite the tier `PBUG-20260723-02` just fixed.

So: effective-contract derivation is scoped **strictly to
`engine_id == "ltx_8gb"`**; WAN keeps its static planning contract AND its
adapter-side cap + ping-pong; and a WAN regression ships in the same commit.

**Corollary, and it vindicates the operator's instinct: ripping ping-pong is
LANE-SPECIFIC.** It is a correctness hole for `ltx_8gb` -- it lets a short
render masquerade as a planned segment and pass the count gate -- and it is
load-bearing for WAN today. Ping-pong is wrong exactly where a coverage plan has
already promised a length.

## 4. THE BUILD

### B1 -- `run_graph(..., external_results=..., on_result=...)`

Four defects, not one:

1. **Kahn's algorithm reports a CYCLE.** `_topo_order` initializes
   `satisfied = set()`; external keys are not in `graph`, so `deps[n] <=
   satisfied` is never true and every downstream node fails with "graph has a
   cycle". Accept `external_keys`, seed `satisfied` with them, allow wires to
   reference them. (agy r4 -- the sharpest single finding of the round)
2. **`free_after_use` DELETES the hoisted handles.** `run_graph` drops nodes
   whose consumer count reaches 0 (`:366-373`); if external keys are not in
   `keep`, segment 0's cleanup evicts `results["ckpt"]` and **segment 1 dies**.
   Auto-add every external key to `keep`. (agy r3)
3. **Shape + collisions.** Reject ID collisions, normalize values to tuples.
4. **Keep it engine-agnostic.** `wrapper_bridge` must NOT hard-code `ckpt` /
   `clip` / `modelsampling` shapes. Generic API + an `on_result` callback;
   validate the three LTX arities in `Ltx8gbEngine.prepare()`. (codex r4)

**Partial-failure transaction:** `run_graph` raises without exposing prior
results, so use `on_result` to register each detachable handle as it lands, then
unwind in REVERSE order, clear external refs, reclaim, release the lease -- one
idempotent `finally`. Inject a failure after each loader in tests. Remove
per-segment patcher discovery from `render_clip()`: `prepare()` registers once,
teardown detaches once.

### B2 -- `resolve_session_config(profile)`, then identity

**An ordering bug sits under this.** `BeatSession` asks `session_identity()` at
`:156`, but `MotionEngineBase` does not set `_active_profile` until `prepare()`
at `:431`, so any profile-derived field would differ before and after loading.
Fix: ONE pure `resolve_session_config(profile)` computed BEFORE the first
identity check, frozen, and passed to identity, `prepare`, `assert_usable` and
graph construction. Three accessors must not independently reread state.

**And the identity must describe what the loaders ACTUALLY load** (codex r4
MUST-FIX 5). `_ckpt_path()` accepts `OTR_LTX_8GB_CKPT` and directory overrides,
but `_build_graph()` passes only BASENAMES to `CheckpointLoaderSimple` /
`CLIPLoader` -- so an override can be receipted while a different registered
basename loads. Resolve through the same `folder_paths` token each loader uses,
and reject unsupported path overrides. Validate existence immediately; a missing
checkpoint or T5 raises BEFORE receipts, identity, or `prepare()`.

Identity: engine id, recipe, checkpoint path + receipt, T5 path + receipt, T5
device, hoisted model-sampling params. EXCLUDE per-segment prompt, seed, frame
count, canvas, tiled-decode. **Receipt = (canonical path, size, mtime_ns)** --
never hash the checkpoint on every `begin_segment()`.

**`ltx_8gb` only.** No package-wide rollout (both seats, twice).

### B3 -- ONE effective contract, threaded to where planning happens

`_stamp_coverage_plan()` reads the static contract and `build_execution_plan()`
runs BEFORE `video.max_render_frames` is stamped; `assert_coverage_plans()`
independently reloads the static one. Thread policy into `_stamp_coverage_plan()`,
derive the LTX effective contract there, stamp an explicit LTX-only sibling
receipt (`shot.coverage_contract`: engine id, min/max, quantum, fps, tail-trim,
continuity), and re-derive the IDENTICAL value at render validation with exact
equality required.

Normalize the cap to the largest legal LTX rung (`9 + 8k`) at or below the
ceiling; **raise a named error if the ceiling is < 9** rather than clamping to an
illegal segment (agy r4).

### B4 -- delete ping-pong, and the assertion it was hiding

**`render_driver.py:2982` already enforces `if got != segment.render_frames:
raise RenderError`,** and `_ltx8_frame_length` snaps to `(length-1)//8*8+1`.
Ping-pong is what currently papers over a non-`8n+1` segment. Delete it and any
such segment becomes a hard RenderError.

So the planner must emit only `8n+1` lengths under the effective contract, with
`drop_head=1` on successors, verified in `assert_coverage_plans()` before
execution. Tail formula (agy r4): for remaining `T_rem`,
`N_k = min(C_max, 8*ceil(T_rem/8) + 1)`; chained coverage
`C = N_0 + sum(N_k - 1) >= T`; `allow_tail_trim` slices `C` down to `T`.

Update the `max_render_frames` tooltip, which currently PROMISES ping-pong
extension, to state that `ltx_8gb` uses it as a coverage-PLANNING cap while WAN
uses it as an adapter-side native cap, and 0 leaves the engine unpinned.

### B5 -- the canvas seam, fail-closed

Derive and validate `(w, h)` from `ledger.video.canonical_canvas` after route
locking, thread it through every segment request, and SUPPRESS the
`OTR_VIDEO_LANDSCAPE_CANVAS` overwrite when a stamp is present. Reject unless
positive, /32, exactly 16:9, and 25 fps.

**Validate BEFORE `BeatSession` opens** (codex r4 MUST-FIX 2). Today
`render_driver.py:2902-2905` opens the session -- which prepares, i.e. loads --
and per-segment `assert_usable()` only runs later at `:2760-2765`. After B1 that
means loading a 6.34 GiB checkpoint before rejecting a bad canvas. Resolve and
validate the frozen config, ledger canvas, fps, model files and the segment-0
request before entering the session; keep per-segment validation before each
`begin_segment()`.

**agy's `wan_shared._dims` change is REJECTED.** It noticed the shared fallback
is 832x480 (not 16:9) and proposed changing that default to 512x288 -- but
`_dims` is shared with the WAN engines, and editing a shared default to satisfy
an LTX gate is precisely the cross-lane damage section 3 forbids. The gate
applies only where the `ltx_8gb` stamp is consumed, and the fallback is
unreachable there because a missing stamp is already terminal. Leave it alone.

### B6 -- the 8 GB levers have no profile channel: FREEZE THEM IN THE RECIPE

The profile schema accepts only `device_policy`, `dtype_policy`,
`max_render_frames`. So "profile beats env" for T5 device / tiled VAE / sampling
has **no end-to-end channel**, and this build forbids new widgets. Per
`PBUG-20260723-02` the env vars cannot bind on a production leg either
(`otr_8gb_ltx.json`'s `launch.env` is `{}`).

**Freeze the MEASURED T5 / tiling / sampling selection into a versioned
`ltx_8gb` recipe in CODE, and demote the env vars to prequalification-only.**
Code binds on every leg regardless of how the server booted -- which is the whole
point of that PBUG. `resolve_session_config(profile)` owns the resolution; the
generic `_get_engine_setting` accessor is CUT because it would preserve hidden
production env channels (codex r4 CUT 3). Log a WARNING whenever an env override
is honoured during prequalification (agy r4).

Already built and defaulted off, and this is where they get decided: tiled VAE
decode exists with schema-verified knobs (`:296-330`, default OFF because "core
VAEDecode handles the 8GB peak AT THE SMOKE CANVAS"), and T5 already offloads to
CPU (`:216-221`) because `t5xxl_fp16` alone is ~9 GB -- that one is load-bearing,
not an optimization.

## 5. THE ACCEPTANCE CASE IS 237, NOT 169 -- and that CUTS an open blocker

Both seats independently. The canonical `OTR_EpisodeAssembler` ships
`opening_duration_sec=10.0` and `crossfade_ms=500`
(`scene_sequencer.py:1232-1233`), and the synthetic opening beat derives from the
persisted first-line `start_s`. So the canonical produces
`round((10 - 0.5) * 25) = 237` frames. 169 would require
`opening_duration_sec = 7.26`, which is why it needed profile-schema work.

**237 is the canonical case, is multi-segment, and NECESSARILY exercises tail
trim** -- a strictly stronger test than 169, which was chosen precisely because
it divides evenly. Verified arithmetic, at `C_max = 65`:

```
[65, 65, 65, 49]  ->  241 chained visible (drop_head=1 x3)  ->  trim 4  ->  237
every segment is 8n+1: 65 = 8*8+1, 49 = 8*6+1
```

**This CUTS O4 from the 7b judgment entirely** -- no `opening_duration_sec` /
`crossfade_ms` profile schema, flattening, or widget mapping is needed. An open
blocker disappeared by choosing the case the canonical already produces.

Acceptance: target 237, >= 2 segments, every render length `8n+1`, total visible
237, tail trim 4, no ping-pong, `RESULT SUCCESS` + `obs_publish OK` + assets on
disk.

## 6. WHAT FABLE ADDED THAT NO OTHER SOURCE DID

Recorded as future decisions, NOT folded into this build:

1. **"Cut, don't chain."** A filmmaker covers a 7-second beat with two different
   compositions and a deliberate cut on the line's rhythm. A cut is grammar and
   reads as invisible; a mid-shot momentum reset is neither cut nor continuity,
   which is why it itches. **The build already supports jump-cut segments with
   their own stills** (chunk 4/6b); `ltx_8gb` declares `strict_first_frame`, i.e.
   chain-only. Fable is arguing the jump cut may be the BETTER viewer experience,
   not merely the cheaper one. Worth a real look once 7d proves the machinery.
2. **Render at 12.5 fps, interpolate to 25** (RIFE-class). Doubles
   seconds-per-clip at any canvas and halves the joins; slow atmospheric drift is
   best-case content for interpolation.
3. **Grain after the upscale, back off the unsharp.** Sharpening a 3.75x upscale
   buys halos; grain buys perceived texture and turns "soft" into "archival".
4. **Salt in high-res stills with a slow drift** -- infinitely sharp, join-free,
   VRAM-free, and literally how 1940s documentaries look. The build has
   `still_pan`.
5. **Verify the COMPOSITION floor, not just sharpness.** These models have a
   point below their training bucket where geometry degrades -- melting
   architecture, wandering shapes. If 512x288 is merely soft, take it; if it is
   structurally degenerate, the answer is the 832x468 crop, NOT 1024.

## 7. MEASUREMENT PROTOCOL

* **Build mechanics first, measure second, freeze the recipe third, run canonical
  acceptance last.** Declare numeric VRAM reserve and wall-clock pass limits
  BEFORE testing.
* Fresh canonical boot per cell; never walk rungs in one resident server; discard
  any process that OOMs (CUDA OOM corrupts the allocator; CLAUDE.md section 4).
* Through the canonical path only: `OTR_VideoRenderBatch._render_episode()` ->
  `run_real_episode()` -> `run_episode()` -> `render_beat_coverage()`. Calling
  `render_beat_coverage()` directly bypasses wiring, still-spine checks and
  publication -- and `render_single` resolves its canvas through a channel
  production never touches.
* `fraction = 8192.0 / detected_total_mib`, set at subprocess entrypoint BEFORE
  any torch.cuda / ComfyUI import (it raises once the context exists), verified
  with the getter. **0.48 on a 16,303 MiB card is 7,825 MiB, not 8,192.**
* Start the VRAM probe BEFORE `BeatSession` -- once loaders hoist into
  `prepare()`, a probe started inside `render_clip()` MISSES the load peaks.
  Sample NVML through session close, read the allocator peak separately, report
  both after teardown. Add `segment_receipts[]` (index, requested/decoded frames,
  canvas, recipe, native peak) -- today the loop copies the final segment's clip,
  so "do not overwrite per-segment receipts" is otherwise not implementable.
* **Label it prequalification.** ComfyUI still sees 16 GB and may choose
  different residency. "8 GB qualified" needs one real 8 GB host.

## 8. WALL-CLOCK RISK

`BeatSession` is per-BEAT and the loaders live inside it, so a 40-beat all-LTX
episode may reload the 6.34 GiB checkpoint and ~9 GiB T5 **forty times**. Measure
cold `prepare()` separately and set a numeric episode ceiling before comparing
architectures. Cross-segment conditioning caching is CUT for now -- hoisting
ckpt/T5/modelsampling closes the residency defect; caching `pos`/`neg` adds
prompt-invalidation rules without being needed for correctness.

## 9. CLAIMS REFUTED

**Driver, by the panel:** that `max_render_frames` could serve as a general
planning cap (it would have broken WAN); that the r2 "profile beats env" fix had
a channel (it does not).

**Panel, by the driver:** agy's `wan_shared._dims` default change (cross-lane
damage); agy r1's `torch.cuda.set_per_process_memory_fraction(0.48)` as an
8192 MiB ceiling (it is 7,825 MiB on this card -- codex caught the arithmetic,
agy caught that it must be set before CUDA init; both corrections adopted).

**Operator direction re-examined and respectfully declined, 4 of 4:** ping-pong
on this lane. The operator permitted it and his reasoning was sound as far as it
went (no lip-sync to reverse), but it lets a short native render pass the
segment-count gate looking like a full segment -- a hole in the coverage proof
rather than a style choice. It also buys nothing: chained segments are already
seamless by construction and yield new motion. **Recommendation: keep the
permission in reserve if coverage planning proves too expensive at 8 GB.**

## 10. PROCESS GAP FOUND

`PBUG-20260723-02` declares itself **bible-worthy** (`PROD_BUG_LOG.md:2679-2713`)
-- *"a contract declared only in a process-launch environment cannot bind work
submitted to an already-running server"* -- but **no matching entry exists in
`comfyui-custom-node-survival-guide/BUG_BIBLE.yaml`.** That rule has now
explained three separate defects in four days (C1, C1b, the canvas, and the 8 GB
levers). It belongs in the Bible with executable coverage.

## 11. VERIFY-AT-BUILD

- [ ] Applied `otr_8gb_ltx` carries the selected `video.max_render_frames`;
      `render.frame_budget` is never treated as episode authority.
- [ ] Checkpoint/T5 receipt paths match the files the ComfyUI loader tokens
      actually load; files exist and are non-zero.
- [ ] Exactly ONE checkpoint load, ONE T5 load, ONE model-sampling construction
      per multi-segment beat.
- [ ] Failure injected after each loader detaches accumulated owners, clears
      external refs, releases the lease, returns VRAM to baseline.
- [ ] `_topo_order` accepts external keys without a false cycle;
      `free_after_use` does not evict them.
- [ ] Canonical 237-frame opening: legal `8n+1` segments, `drop_head=1`,
      `trim_tail=4`, exactly 237 assembled frames.
- [ ] No `extend_frames_to_target` for `ltx_8gb`; **WAN's extension still
      active**, and `max_render_frames=17` does not alter WAN's coverage-plan
      topology (both the plan test and the existing `_floor_length` regression).
- [ ] Every LTX request renders 512x288 @ 25 fps; final ffprobe reports
      1920x1080 with ZERO pad area.
- [ ] Prequalification cells set the verified fraction before CUDA init and
      record allocator + NVML peaks.
- [ ] 40-beat canonical run meets the predeclared wall-clock ceiling; cold
      `prepare()` reported separately.
- [ ] Focused suite + full Windows suite + Bug Bible; `OTR_WorkflowValidator`,
      JSON round-trip, widget/input/link audit; `RESULT SUCCESS`,
      `obs_publish OK`, assets on disk; pushed, HEAD == origin, no BOM, AST parse.
