# VIDEO_LANE_PREFLIGHT receipt -- lane 13, `viz_mxc_cpu`

`VIDEO_LANE_PREFLIGHT receipt: viz_mxc_cpu | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane13_viz_mxc_cpu/ | verdict PASS -- 7/7`

The third visualizer, and the third time the same two answers were right. What
makes it a lane rather than a copy-paste is that the L19 premise was re-derived
against THIS engine's painter before either answer was reused.

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | PASS | PASS | no weights on this lane |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT |
| G3 contract vs runtime | **RED** | **PASS** | `continuity=CONTINUITY_NONE` at this lane's own contract |
| G4 / G6 | n/a | n/a | exempt -- procedural CPU lane |
| G5 audio law (V-1) | PASS | PASS | already probed its own emitted mp4 |
| G7 public surface | PASS | PASS | `ENGINE_MATRIX.md` unchanged (correct -- nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.

## G2 -- the premise, re-derived rather than inherited

L19's runnable check is to grep this engine's render path for any dimension NOT
derived from the request. Done, and it comes back clean:

* `render_clip` reads `w, h` from `_canvas_dims(request)`;
* `paint_rainbow_frame(w, h, ...)` lays the radio dial out through
  `ring_geom(w, h)`;
* the scanline table, the vignette and `_small_font(h)` are all built from that
  same pair;
* `encode_silent_mp4` is handed the same `w, h`.

No latent grid, no trained input size, no canvas-dependent constant. So the
1472x832 an episode hands this lane is `OTR_VIDEO_LANDSCAPE_CANVAS`'s default --
an operator lever -- and not a property of the engine. Since
`declared_render_canvas` is applied LAST, declaring would silently make this the
one visualizer that ignores that lever.

**And it would be worst on this lane specifically.** Its module docstring says
its purpose is to run "on ANY box (AMD / Mac / Intel), no GPU, no shaders". Seven
profiles select it, including `otr_amd16_rocm`, `otr_amd8_rocm` and
`otr_mac_mps` -- the portability tiers. Pinning a canvas on the lane that exists
to be portable is the opposite of what it is for.

Channel declared INERT in `PROFILE_CANVAS_DOCUMENTED_DEAD` with the mechanism
written out; inert, not dead (L18). As with lanes 11-12, that also makes the
untracked `otr_sbcov_2` profile irrelevant to this row rather than blocking it.

**Nothing had to move in `test_ltx_8gb_canonical_canvas.py`, and the precise
reason is worth stating because the first draft of this receipt got it wrong.**
That file holds TWO different things and they are easy to conflate:

* `test_a_SIBLING_lane_still_takes_the_landscape_default` -- the real
  differential control, which drives `build_request_from_shot`. It is pinned to
  **`still_pan` alone** and has been since lane 11 moved it off `mesh_stage`.
  `viz_mxc_cpu` is not in it and never was.
* `test_engines_that_declare_NOTHING_are_left_alone` -- a weaker list assertion
  (`declared_render_canvas(x) is None`) that DOES include this lane, alongside
  `still_pan`.

So this lane declaring nothing means the list entry stays valid; it does not
mean this lane was carrying the control. The proof that `viz_mxc_cpu` really
reaches the landscape default now lives in this lane's OWN
`test_the_landscape_lever_still_reaches_this_lane`, which drives the same real
builder. Lane 16 is the packet that will have to move the control, when
`still_pan` gains a declaration.

## G3 -- one line, this lane only

`continuity=CONTINUITY_NONE` passed at this lane's own contract, with the
reason: `render_clip` paints every frame from the beat's own audio analysis and
a per-beat rng key and reads no predecessor frame.

`viz_mxc_mandala` (lane 14) keeps both its rows. It is NOT covered by lanes
11-13, and it is the one visualizer with a NAMED pycairo dependency, so its
render path could genuinely differ -- lane 14 re-checks rather than assumes.

## The solo smoke -- LIVE PASS

Stock `default` boot, box reset per CLAUDE.md section 4 first. Real audio: the
same 4-second mono 24 kHz slice, copied into this lane's own smoke directory, so
the visualizer runs REACTIVE rather than idle.

| Item | Value |
|---|---|
| Harness | `_otr_single_engine_smoke.py --engine viz_mxc_cpu --frames 100 --audio <slice>` |
| Prompt id | `065f3cc7-883a-41af-8218-c8a5664e894e` |
| Wall time | **1.9 s** |
| Canvas PROBED | **832x480** -- `render_single`'s wide default, the shipped path for a lane that declares nothing |
| Frames PROBED | **100**, exactly the ask; 100/25 = 4.000 s against the 4 s slice |
| Rate / codec | 25/1, h264, yuv420p, bt709 |
| Audio | **zero audio streams** -- silence proved on the emitted file |
| Artifact | `.../_lane_smokes/lane13_viz_mxc_cpu/lane13_viz_mxc_cpu_smoke_f100.mp4` |
| sha256 | `090436a3fed684d15ba9c5f94f4e8026a3e8fa9283e0124f3b5b3394d2d9e324` |

One leg, for the reason lane 12 recorded: the lever behaviour is a DRIVER fact,
already proved live in lane 11 and now pinned CPU-side per lane.

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/ffmpeg lane, G4 exempt.

**No profile, variant, workflow or `ENGINE_MATRIX.md` change** -- nothing was
declared, so nothing downstream moved.

**`viz_mxc_mandala` was not touched.** Its answers are probably these two again;
"probably" is what L19 says to check.
