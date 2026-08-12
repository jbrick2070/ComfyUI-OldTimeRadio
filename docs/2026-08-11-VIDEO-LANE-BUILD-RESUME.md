# RESUME HERE -- video lane build

## WINDOW HANDOFF 2026-08-12 (later) -- LANE 20 CLOSED

**20 of 21 packets done. Lane 21 is the last one, and it is UNLIKE every lane
before it: a standalone runner that is NOT registered.** Box CLEAN (no resident
server, port 8000 clear, VRAM 1,283 MiB). HEAD == origin at `2362353b`.

* `2362353b` -- **lane 20, `h3_low_audio_in` / `minimax_h3_audio_in`.** 7/7 green
  on arrival, live smoke PASS on the FIRST attempt. Receipt:
  `docs/evidence/lane_receipts/lane20-h3_low_audio_in.md`.

**Baselines now:** suite **10114 passed / 110 skipped / 1 xfailed**, nothing
deselected. Bug Bible **20/24/3 at 272**. `build_variants.py --check` **48
variants, 0 failures**.

### WHAT LANE 21 IS, AND WHY IT IS NOT LIKE 19 OR 20

`h3_low_mime` -- a **standalone runner**, explicitly **NOT registered this
build**. Its row: "G5.2 keeps-audio exemption, clip/stem receipts, durable
output path, solo-runner QA."

**The interesting part is `keeps-audio`.** Lanes 19 and 20 both prove SILENCE on
their emitted file (V-1: only `OTR_MasterAudioMux` ever adds audio), and
`canonicalize` ffprobes it. The mime runner is the EXEMPTION -- it keeps H3's
natively generated audio, which is the whole reason it exists. So it must NOT
reuse `_MiniMaxH3Base.canonicalize` unexamined, and G5's lexical gate does not
apply to an unregistered runner. Read the G5.2 spec before assuming either way.

**The lab has the ground truth**, and both legs PASSED cold with native audio:
`vram-recipe-lab/recipes/h3_mime_i2v.json` (864x480, model f90 = 3.750 s, 7.28
GiB, 178.9 s, audio at -27.5 LUFS) and `h3_mime_r2v.json` (7.23 GiB, -40.5
LUFS). Note f90 is BELOW the 124 trained floor and on the 17k+5 grid -- so a
mime clip is deliberately shorter than anything the registered lanes may render,
which is one reason it is a separate runner rather than a third adapter.

**Reuse, do not re-derive:** `eng_minimax_h3` already gives you the grid math
(`align_frame_count`, `model_rungs`, `canvas_frames_for_model`,
`canvas_index_map`), the weight resolution, the boot contract and the recipe.
A runner can import them without registering anything.

### THEN: the 45-word render of EVERY visual path

The operator's acceptance gate, and it runs LAST -- after lane 21, because 21 is
the final path. It supersedes row 22's single end-to-end episode. Every visual
path gets a 45-word render; the `otr_w45_*` profiles are the shape.

---

## WINDOW HANDOFF 2026-08-12 -- lane 19 CLOSED, two Lemmy fixers CLOSED

**19 of 21 packets done. Lane 20 is next and it is the CHEAPEST lane left**,
because lane 19 already wrote the module it lives in. Everything below is what
lane 20 needs and would otherwise have to rediscover. Box left CLEAN (no
resident server, port 8000 clear, VRAM 1,345 MiB). HEAD == origin at `38d15fa9`.

**What landed, both pushed and lockstep-verified:**

* `be4aadff` -- **lane 19, `h3_low_video` / `minimax_h3_video`.** 7/7 gates green
  on arrival, live smoke PASS. Receipt:
  `docs/evidence/lane_receipts/lane19-h3_low_video.md`.
* `38d15fa9` -- **Lemmy chunks A1 and B**, per the operator's "lemmy fixers
  first" directive, plus `PBUG-20260812-01`.

**Baselines now:** full suite **10067 passed / 110 skipped / 1 xfailed**, nothing
deselected (was 9985 at the lane-18 wrap; +58 for lane 19's rows across the
parametrized roster suites, +24 for the two Lemmy test files). Bug Bible
**20/24/3 at 272**. `build_variants.py --check` **47 variants, 0 failures**
(was 46; lane 19 added one).

### THE OPERATOR'S ORDERING, given 2026-08-12 before sleeping

Three messages, and they reorder the tail of the queue:

1. **"we need a 45 word render of every visual path so that's probably first"**
   -- that sweep is the acceptance gate, superseding row 22's single episode.
2. **"lemmy fixers first" / "i want the lemmy fixed before a big video test"**.
3. **"try to get all 21 done"**.

**Order this window took, and the one to continue:** lane 19 -> Lemmy fixers ->
**lanes 20 + 21** -> **the 45-word sweep LAST**. Lanes 20 and 21 each ADD a
visual path, so sweeping before them means sweeping twice.

### LANE 20 -- everything already known, so do not re-derive it

**It is a SECOND registration out of `nodes/_otr_video_engines/eng_minimax_h3.py`,
not a new module.** Same weights root, same boot contract, same frame grid, same
canvas, same preflight. Suggested shape: extract the shared half into a base
(weights, `session_identity`, `assert_usable`, `_assert_boot_contract`, `load`,
`canonicalize`, `render_clip`) and leave per-adapter only `name` / `family` /
`required_inputs` / `frame_contract` / `_node_candidates` / `_build_graph`.

* **Weights, all present:** DiT `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
  (note **ref2va**, not fl2va), the same `qwen3vl_32b_minimax_h3_nvfp4_awq`
  encoder, the same `minimax_h3_video_vae_fp16`, **plus**
  `minimax_h3_audio_vae_fp32.safetensors` -- which lane 19 deliberately does NOT
  load and lane 20 does.
* **The node is `MiniMaxH3ReferenceToVideo`**, confirmed live. Its reference
  sockets are `COMFY_AUTOGROW_V3`. In the lab's API graph they serialize DOTTED
  (`"ref_images.ref_image_0"`, `"ref_audios.ref_audio_0"`); `execute` consumes
  plain dicts (`(ref_images or {}).values()`), so an in-process call through
  `wrapper_bridge` should pass `ref_images={"ref_image_0": Wire(...)}`.
  **THIS IS THE ONE UNVERIFIED THING -- prove it on the live server before
  building around it.** `_iter_wires` does recurse dicts, and lane 19 proved
  V3 node classes execute fine through `run_graph`.
* **Ground-truth recipe:** `vram-recipe-lab/recipes/
  h3_jobd_lipsync_refaudio_seed43_f192.json` -- a lip-sync ref2va leg at
  832x480 f192 seed 43 that PASSED cold at 6.88 GiB in 436 s. Its prompt uses
  the `<Picture 1>` / `<Audio 1>` tag convention the node's docstring requires.
* **Continuity is `soft_reference`, NOT strict** -- and the contrast with lane 19
  is the point. Lane 19 wires `first_frame`, a keyframe pinned at
  `resolved_frame_index 0` and re-injected every step. ref2va's `ref_images` are
  IDENTITY references with no frame-0 guarantee, which is exactly what
  `CONTINUITY_SOFT_REFERENCE` means. Do not copy lane 19's answer.
* **THE MOUTH POLICY, and it must land in the SAME commit.**
  `render_driver.py:1548` (`_is_character_face_beat`) tests
  `engine_id == "ltx_audio_in"` by EQUALITY; it becomes a membership test
  including `minimax_h3_audio_in`. Without it every H3 character beat raises
  `MouthPolicyError` at plan time -- `mouth_owner_for_beat` refuses an audio-in
  beat that is neither a character face nor a cabinet role. **Do NOT mint a new
  family to dodge this:** a family outside `content_oracle.MOTION_FAMILIES`
  makes frozen clips motion-EXEMPT. Check `render_driver.py:769`
  (`_still_spine_requires_scene`) too -- another `ltx_audio_in` literal list.
* **Profile:** seed 43 in `seed_policy`, `boot_contract: "h3"`, canvas 864x480,
  modelled on `config/profiles/otr_h3_low_video.json`.
* **Roster tests it WILL turn red** (all of them fixed at the fixture, never by
  weakening a gate -- see lane 19's L23/lane 8's lesson): `test_public_engines`
  (`_TIER`), `test_still_plan_parity` (regenerate the fixture with
  `python tests/test_still_plan_parity.py --regenerate`, then CHECK the diff is
  additive), `test_still_plan_layer2_parity` (framing_geometry must be VERBATIM
  the producer constant), `test_multiclip_session_identity_roster`,
  `test_ltx_8gb_session_identity.py::_ENGINES_WITH_A_SESSION`,
  `test_boot_contracts`, and `tests/_s28_forbidden_sweep` (never name a local
  variable `alias`).

### LEMMY -- what is left after this window

`GO_FORWARD_PLAN.md` "WHAT REMAINS ON THE LEMMY SPRINT" is updated. A1 and B are
**DONE**. Still open, none blocking:

* **Chunk C** -- and READ THIS BEFORE STARTING IT. The UNIT half is already
  covered: `tests/test_lemmy_index_rate_to_bus.py` proves duration, pitch, the
  no-op-only-when-equal rule, CPU/float32, and every shipped engine rate landing
  on the 48000 bus; `tests/test_voice_mixed_rate_resample.py` covers the mixed-rate
  pack. What is genuinely uncovered is the **SceneSequencer INTEGRATION** path --
  driving `scene_sequencer`'s dialogue assembly with 22050 clips, since
  `tests/test_sequencer_ledger.py` carries no 22050 fixture. Scope chunk C to
  that, not to the unit.
* **Chunks A2 -> A3 -> A4** -- v2 identity + replay bridge. The operator
  pre-ruled **AUTO-PROMOTE on a clean replay**: if A4 reproduces all six frozen
  clip hashes, A2/A3 rewrite the receipt's identity fields with no second
  sign-off.
* **Chunk E** -- release/OBS editorial-recast audit. **Operator only.**
* **Branch B stays unbuilt** -- it existed for a G1 failure, and G1 passed.
* **A live forced-Lemmy render would settle chunk B's status.** The resolver bug
  is proved by the filesystem and by replaying the old code, but NOT by a
  captured live failure, so it is deliberately not in `PROD_BUG_LOG.md`.

---

# The 2026-08-11 handoff follows, unedited

Read this, then `docs/GO_FORWARD_PLAN.md` item 5 (the VIDEO LANE QUEUE table).
Everything else you need is linked from those two.

## Where it stands

**18 of 21 lane packets confirmed working and pushed. Lane 19
(`h3_low_video` / `minimax_h3_video`) is NEXT, and it is the first NEW-ENGINE
packet -- lanes 10-18 repaired lanes that already existed. Nothing is in
flight; the working tree carries only other windows' files.**

**THE WHOLE CHEAP SHELF IS GREEN** (lanes 11-18: four visualizers, four still
families). All eight closed G2 by declaring their profile canvas channel INERT
rather than by declaring a canvas -- see **L19**, and treat a declaration on a
procedural lane as the anomaly that owes an argument. All four still families
now REFUSE a missing still instead of painting a black beat.

| # | Lane | State |
|---|---|---|
| 0 | scaffolding | DONE `49adc824` |
| 1 | `wan22_high_i2v` (`wan_i2v`) | DONE `b303afa3` |
| 2 | `humo14_high_audio_in_wide` (`humo_14B_169`) | DONE `e19dd473` |
| 3 | `humo17_high_audio_in_portrait` + `_wide` | DONE `d226bea5` |
| 4 | `humo14_high_audio_in_portrait` (`humo`) | DONE `b53ca2f1` |
| 5 | `wan22_high_video` (`wan_ti2v`) | DONE `d0536e72` |
| 6 | `wan22_high_fast` (`fastwan_8gb`) | DONE `930e3bda` |
| 7 | `ltx23_low_audio_in` (`ltx_audio_in`) | DONE `57665ee8` -- live 1024x576 f193 |
| 7b | `ltx_audio_in` headroom | DONE `310437ae` -- MARGINAL, 115 MB, diet boot shipped |
| 8 | `ltx098_low_video` (`ltx_8gb`) | DONE `c6a99764` -- live 512x288 f161 |
| 9 | `ltx23_high_video` (`ltx_video`) | DONE -- both legs run; floor 169 -> 9 |
| 9b | `ltx_video` headroom | **OPEN** -- no diet boot ever tried, no headroom at f169 |
| 10 | `mesh_stage` | DONE `8e1f02bf` -- 4 red gates closed; the lane was DEAD on this box (L1) |
| 11 | `viz_green` | DONE `28b4e1b5` -- a canvas declaration REVERTED after a Codex consult (L19) |
| 12 | `viz_camera` | DONE `8699fe29` -- + the G3 gate its own docs had disabled (L20) |
| 13 | `viz_mxc_cpu` | DONE `f44993de` |
| 14 | `viz_mxc_mandala` | DONE `eb3f8412` -- visualizer family closed |
| 15 | `still_motion` | DONE `95b6b8ca` -- **the black-beat defect** closed (L21) |
| 16 | `still_pan` | DONE `fc7812dd` |
| 17 | `still_flat` | DONE `b79af369` -- still shelf closed; the floor is now an unoccupied control (L22) |
| 18 | `still_word` | DONE -- verification lane; **the whole cheap shelf is green** |
| 19 | `h3_low_video` (`minimax_h3_video`) | **NEXT -- DIAGNOSED, not started.** Both 21 GB weights + all four node classes installed, so it is smokeable. See the LANE 19 DIAGNOSIS block in GO_FORWARD |
| 20-21 | `h3_low_audio_in`, the standalone mime runner | NOT STARTED |
| 22 | 30-word end-to-end episode gate | NOT RUN |

**Baselines at the lane-18 wrap:** full suite **9985 passed / 109 skipped /
1 xfailed**, nothing deselected; Bug Bible **20/24/3 at 272**;
`build_variants.py --check` **0 failures** (its "46 variants" count is
workstation-dependent -- `git ls-files` counts 45; compare the FAILURES, not
the count). Box left clean.

**Two things wait on the OPERATOR, neither blocking:** the leaked
`otr_sbcov_*` variants that make `--check` crash on a fresh clone, and that
variant-count caveat. Both are written up in `docs/GO_FORWARD_PLAN.md`.

## THREE THINGS LANE 7 CHANGED FOR EVERY LANE AFTER IT

**1. Run `scripts/build_variants.py --check` BEFORE starting a lane.** Lane 7
inherited five RED variants from lane 5 -- they still carried `wan_8gb (16:9)`
in node 87, because lane 5 regenerated only the variant whose profile it edited
-- and had to separate them from its own drift. A red at the start of a lane
belongs to whoever caused it. It is 46 variants / 0 failures right now.

**2. A solo smoke means something different now.** `render_single` -- the path
EVERY lane smoke runs through -- never consulted `declared_render_canvas`. It
derived the canvas from `render_aspect` instead, so lanes 1-6 all validated the
aspect default rather than their own declaration. Invisible for six lanes
because all six declared exactly what that path already produced. Fixed;
pinned by
`test_ltx_8gb_canonical_canvas.py::test_render_single_takes_the_DECLARATION_not_the_aspect_default`.

**3. A lane's VRAM peak now reaches disk.** The `_clip_summary` passthrough
(relayed from the concurrent window, landed in lane 7's commit) means a smoke
report carries `vram_peak_mb` / `recipe` / `quant` / `render_canvas` /
`native_frame_count` / `extension_mode`. **One re-smoke each recovers the
`wan_ti2v` and `fastwan` peaks** that were measured and dropped -- no
measurement campaign, and it unblocks queue row 5a.

## QUEUE ROW 7b -- RESOLVED 2026-08-11 (`310437ae`)

Operator ruled: prove the lever, do not wave through 0.24%. Done. Both legs
cold, same recipe/canvas/frames/still, only the boot differing:

* `default` -- 14,465 MB absolute / 11,952 MB net -- margin 35 MB
* `ltx_av_diet` -- **14,385 MB** absolute / 11,872 MB net -- margin **115 MB**

Clears, but UNDER the 0.3 GiB threshold, so decision rule 2 applied: the diet
contract ships and the lane is flagged **MARGINAL** in the manifest with both
numbers. Output is BYTE-IDENTICAL between boots, so the diet is free. It bought
only 80 MB because `reserve_vram_gb` is INERT on this lane -- the adapter's own
in-process 4.0 GB reserve dominates any boot value -- so HuMo's ~1.9 GiB does
NOT transfer and the lever is nearly exhausted here.

"Confirmed working" = built, 7/7 preflight gates green, a live render smoked and
PROBED (canvas, exact frame count, silence, no trim), full suite green, pushed.

## LANE 9 CLOSED 2026-08-11 -- both legs run, and they moved the contract

Full write-up: `docs/evidence/lane_receipts/lane09-ltx23_high_video.md`.

* **The lane could not have reported an honest number.** `_render_clip_hq` --
  the path this box's dev-family unet actually routes to -- had no
  `VramPeakProbe` and never called `_clip_telemetry`, and `_clip_from_raw`
  dropped five of seven receipt fields. `render_driver` substitutes an
  instantaneous sample for a missing peak, so the marker leg would have
  reported **4,124 MB** instead of **15,916 MB**. New lesson **L15**.
* **The marker:** 15,916 MB absolute / **13,313 MB net**, cold, 1024x576x169,
  147.5 s. State the surface: absolute is OVER the 14.5 GiB ceiling (97.6% of
  the card), net is under. `high` is measured against its SIBLING
  (`ltx23_low_audio_in`, 11,872 net), not against a card.
* **S8b-14 resolved at its root.** The band at 1024x576 is OPEN --
  f9/f49/f97/f121/f137 all decode clean, including the exact pair that FAILS at
  1472x832. The floor was a canvas-dependent decode constraint, not a look
  choice: 169 -> 9, contract now `min_frames=9, max_frames=169, quantum=8`.
  New lesson **L16**. A 2 s beat renders 49 frames in 75.4 s instead of 169 in
  147.5 s and discards nothing.
* **Operator gated the look change on eyes:** the solo smoke is a SHORT-BEAT
  A/B (`ab_BEFORE_vs_AFTER_f49.mp4` in the lane smoke dir). A short render is a
  re-paced COMPLETE arc, not the truncation of a long one. One constant
  reverses it.
* Naming MOVE `ltx23_16gb_video` -> `_LEGACY_ENGINE_ALIASES`, retiring the
  **last `16gb` token**; verified live on the server (28 menu rows, all four
  spellings resolve, retired id absent from the menu).

## The lane 9 material that used to live here -- MEASUREMENT, not gates

Its preflight rows were already green before the lane started, so there is no
gate to flip. What landed in `7afe40e5`, all of it prerequisite:

- the canvas moved to **1024x576** by operator ruling. Its HQ two-stage path
  had the SAME fixed-x2 illegal-stage-A defect as lane 7 and the corpus never
  flagged it -- 832x480 halves to 416x240 and `240 % 32 == 16`. It hid because
  the OLD landscape default (1472x832) is also /64, so the path was legal until
  the lane was reconfigured. Lesson L13.
- a PREQUALIFICATION consent act (`OTR_LTX_VIDEO_PREQUALIFICATION`): this was
  the last frozen LTX adapter with no sanctioned way to measure, and its own
  env refusal correctly blocks moving the decode floor. Default off, LOUD, and
  its clips stamp `+prequalification[...]`.
- the L4 receipt fields it never produced. Without them a measurement has
  nowhere to stamp departures and no peak to report.

**BOTH LEGS ARE NOW RUN -- the list below is kept as the record of what was
asked for, and the section above records what each answered:**

1. the **decode band at 1024x576** under the consent act, to answer S8b-14 at
   its ROOT. The 169 floor is canvas-dependent and was measured at 1472x832, so
   a 50-frame beat renders 169 and discards 119 -- ~3.4x the GPU work. The
   floor is deliberately UNMOVED and the trim ratio is logged loud until this
   leg answers it. Do NOT bundle a guess into the canvas fix.
2. a **single-render VRAM leg** for the low/high marker, which the corpus
   forbids shipping as a guess -- its only datapoint is a chained diagnostic
   that cannot decide it.

NOTE: the operator's ruling cited a lab 1024x576x193 warm row (7.36 GiB). That
row is recorded under `ltx_audio_in`, NOT this lane. The geometry argument
stands on its own, but this lane still has NO measurement at its new canvas.

Then lane 10 `mesh_stage` -- the most defective lane left, 4 red gates.

## The per-lane loop (do not skip step 1)

1. **Read `docs/LANE_BUILD_LESSONS.md` top to bottom** and check the lane
   against every entry. This is not ceremony -- lane 2 found HuMo carrying the
   identical defect that killed wan_i2v by doing exactly this, before writing
   any code.
2. Run `tests/test_lane_preflight_matrix.py`. The lane's RED rows are its work
   list. Quick matrix dump:
   `python -c "import sys;sys.path[:0]=['.','tests'];import test_lane_preflight_matrix as M;[print(n.ljust(28), ' '.join({'pass':'ok','exempt':'n/a','expected_red':'RED*','unexpected_pass':'??','RED':'RED'}[M.evaluate(g,n)[0]].ljust(4) for g in M.GATES)) for n in M.ENGINE_NAMES]"`
3. Code. 4. Wire (profile, boot lane, node-87 strings GENERATED never typed).
5. Regress: AST, dead-ref grep, full suite, Bug Bible.
6. Preflight row flips GREEN -- red row means no smoke, no commit.
7. Smoke the lane ALONE on the boot IT declares.
8. Commit AND push. 9. Append what bit you to the lessons ledger.

## Things that will bite you, learned the hard way

- **The strict unexpected-pass gate is your friend.** When you fix a lane, its
  `EXPECTED_RED` entry in `tests/test_lane_preflight_matrix.py` must be
  DELETED in the same commit or the suite fails and tells you to. That fired
  correctly on every lane so far.
- **Renames break tests that hardcode the aspect suffix or a bare engine id.**
  Grep for `(16:9)` near naming tables and for the internal id used as an
  expected widget VALUE. Derive from `_aspect_suffix(internal)`; assert that a
  saved value RESOLVES rather than that it is spelled a particular way.
- **`git add` BY NAME.** The tree carries other windows' dirty files
  (`tmp/*.ps1`, `docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md`, the
  `config/profiles/otr_sbcov_*.json` set, `kibitz/`, `uv.lock`). None of them
  are yours.
- **Reset the GPU before every smoke**, selectively by CommandLine -- never a
  blanket python kill, which also takes the Claude MCP extension pythons.
  Helper: the scratchpad `boot_lane.ps1 -Lane WAN|LTX|HUMO -Contract default|diet`.
- **Commit messages: use `git commit -F <file>`.** A backtick in a `-m` string
  gets command-substituted by bash and the commit dies mid-message.
- **Pushes**: one attempt, then hand the operator a PowerShell block. A DNS
  blip on the Wi-Fi path cost one push tonight; the work was already committed
  so nothing was at risk.

## Naming: SETTLED 2026-08-11

`wan22_high_i2v` is correct and stands. The spec's `wan21` was a single
mistyped version number that every downstream document inherited -- the naming
itself was never in doubt. Spec and transplant plan corrected; no code moved,
because the build had used the right name from the start. The retired spelling
keeps a legacy-alias row so a paste from a stale copy of any reviewed doc still
resolves instead of erroring.

## ~~One thing the operator still owes a decision on~~ -- SETTLED (`f2470e31`)

The cold-peaks-vs-corpus-headline question is answered by the NET ruling:
derive cost rows from **NET** peaks (each leg's absolute minus its OWN pre-queue
baseline), never from absolute device totals, and seed a row ONLY from a true
`VramPeakProbe` maximum -- an `nvidia-smi` sample is a LOWER BOUND and would
make the row under-predict, admitting renders that then OOM. Every leg in
`docs/evidence/video_evidence_manifest.json` now records both surfaces.

Nothing is machine-qualified yet. `QUALIFIED_COST_ROWS` is still empty and the
manifest says "admission NOT enforced" per lane, in words. Deliberate
(standing default Q3), not an omission -- and `ltx_audio_in` additionally reads
**MARGINAL** there, with both boot numbers.

## Open rows that are NOT lanes

- **2b** -- move the boot-contract check from `assert_usable` into the ShotLock
  preflight. Needs `boot_contract` plumbed into the frozen director policy.
- **5b** -- S7 WAN retention. Instrument the post-close boundary, collect
  telemetry on a live chained leg, THEN pick a release branch from what it
  names. A measurement campaign, not a code change.
- **5a re-smoke (cheap, unblocks a queue row)** -- `_clip_summary` used to drop
  the telemetry, so `wan_ti2v` and `fastwan` smoked with no peak on disk. The
  passthrough landed in lane 7, so ONE re-smoke each recovers both. No
  measurement campaign.
- **8 GB re-measure** -- `otr_8gb_wan` went 17 -> 81 frames because 17 was
  narrowing the planner into 0.68 s segments. 81 has NOT been proved to fit on
  real 8 GB hardware. If it does not, the answer is a measured ceiling, not a
  return to 17.
- `docs/2026-08-10-FINAL-QA-video-build-corpus.md` still carries its ORIGINAL
  header verdict ("NOT IMPLEMENTATION-READY") while the master spec says that
  pass re-ran and cleared lane 1. Its 21-lane plan is what this build follows.
  One edit would stop the next window stopping at the wrong gate.
