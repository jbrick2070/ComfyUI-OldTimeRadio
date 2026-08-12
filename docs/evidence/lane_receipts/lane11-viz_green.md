# VIDEO_LANE_PREFLIGHT receipt -- lane 11, `viz_green`

`VIDEO_LANE_PREFLIGHT receipt: viz_green | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane11_viz_green/ | verdict PASS -- 7/7`

Two red gates, both closed. The interesting part of this lane is that **its
first draft closed G2 the wrong way, a Codex consult broke the framing, and the
fix was to REMOVE the code the draft had added.**

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | PASS | PASS | no weights on this lane |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT, with the reason |
| G3 contract vs runtime | **RED** | **PASS** | `continuity=CONTINUITY_NONE` at this lane's own declaration |
| G4 admission honesty | n/a | n/a | exempt -- procedural lane, no VRAM envelope |
| G5 audio law (V-1) | PASS | PASS | already probed its own emitted mp4 |
| G6 guards | n/a | n/a | exempt -- no diffusion model, no boot contract |
| G7 public surface | PASS | PASS | `ENGINE_MATRIX.md` unchanged (correctly -- nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures** (but
see the count caveat at the end -- 46 is workstation-dependent).

## G2 -- the draft, the consult, and why the fix was a DELETION

**What the draft did.** Both request-building paths were measured against the
real driver first:

```
build_request_from_shot -> (1472, 832)     [the ledger/profile asked 832x480]
render_single           -> (832, 480)
```

Two different sizes for one lane depending only on which builder ran. That
looked exactly like lane 7's finding ("every solo lane smoke was validating the
ASPECT DEFAULT rather than the declaration"), so the draft declared
`render_canvas = (1472, 832)`, moved the three TRACKED profiles to match,
regenerated their variants, and pinned it with two tests.

**Why that was wrong, and the check that catches it.** 1472x832 is not a
property of this lane at all. `eng_visualizer.render_clip` paints and encodes at
exactly the size the request carries -- no latent grid, no fixed model input, no
canvas-dependent constant. The 1472x832 an episode hands it is the DEFAULT OF
`OTR_VIDEO_LANDSCAPE_CANVAS`, an operator lever, applied by the driver to every
non-face family. And `declared_render_canvas` is applied LAST, overruling every
earlier channel -- so the declaration would have made `viz_green` the one
visualizer in its family that silently ignores that lever, and pinned the smoke
path too. A real behaviour change wearing a documentation label.

**Lesson L2's own precision note is exactly this check** -- "a canvas
declaration must either agree with those overrides or the lane must state that
they are unsupported; check the OVERRIDE PATH, not just the default" -- and the
draft walked straight past it. Recorded as L19.

**The shipped position:** this lane declares NOTHING, and its profile canvas
channel is declared INERT in `PROFILE_CANVAS_DOCUMENTED_DEAD` with the precise
mechanism. Inert, not dead (L18): the profile number IS carried -- profile ->
applier -> node-87 director widgets -> `request["canvas"]` -- and is then
overwritten by the landscape default. It cannot decide this lane's size, so
reconciling it to a different number would be equally unable to decide it.

**Tests changed shape with the decision.** The two canvas pins were replaced by
the property this lane actually has: it declares nothing and paints any size it
is handed (four sizes asserted), plus a test driving the REAL
`build_request_from_shot` that fails if the `OTR_VIDEO_LANDSCAPE_CANVAS` lever
ever stops reaching this lane -- so a future declaration cannot be added without
something saying what it costs.

**And it dissolved a blocker rather than fighting it.** The draft's G2.3 row was
held red by `config/profiles/otr_sbcov_4.json`, an UNTRACKED file this window
must not edit or adopt. Documenting the channel inert makes G2.3 skip the
profile comparison entirely, so the row is green on its own merits and the
untracked file is irrelevant to it.

## G3 -- one line, this lane only

`frame_contract` never passed `continuity=`, so `CONTINUITY_NONE` was a dataclass
default while the comment above it had asserted "CONTINUITY none" since the
engine was written. NONE is true here for a reason this lane can state:
`render_clip` paints every frame from the beat's own audio analysis and reads no
predecessor frame, so there is no terminal state a successor could inherit.

**Scoped to this lane on purpose -- the L13 judgment call.** Lane 10's identical
fix landed on `_CheapFamilyBase` and flipped four still lanes green for free
because they SHARE one contract object. The four visualizers do not: each
declares its own `FrameContract` in its own module. There is no shared mechanism
to sweep, so lanes 12, 13 and 14 each still owe their own one-line declaration
and their `EXPECTED_RED` G3 rows are untouched.

## The solo smoke -- LIVE PASS, two legs

Stock `default` boot, box reset per CLAUDE.md section 4 before each leg.

**The audio is real.** `viz_green` paints audio-reactive scopes, so a smoke on
silence exercises the idle path and proves nothing about the lane's job. A
4-second mono 24 kHz slice was cut (read-only) from a published episode in
`otr/obs/` into the lane's own smoke directory.

| | LEG A -- shipped default | LEG B -- the lever moved |
|---|---|---|
| Boot env | stock | `OTR_VIDEO_RENDER_CANVAS=1472x832` |
| Prompt id | `8cac677c-a083-4a36-9d3f-f2f2491137aa` | `f0fe8fb4-7961-4ddf-9d80-e7315606a8d2` |
| Wall time | 1.4 s | 3.7 s |
| Canvas PROBED | **832x480** | **1472x832** |
| Frames PROBED | 100 (= 4.000 s at 25 fps, the audio slice's length) | 100 |
| Rate / codec | 25/1, h264, yuv420p, bt709 | 25/1, h264, yuv420p, bt709 |
| Audio | **zero audio streams** | **zero audio streams** |
| sha256 | `e9680b022c48a8dc995fa17bed6989cc6f5c9b47bc88502b13718de5990f9da3` | `db5c5eabaee5b7069401c11ff1daf9e829737f194d3673f1d3309b83154efa9c` |

**Two legs because one would have proved the wrong thing.** The lane's property
is that it honours whatever canvas it is handed, so a single size cannot
demonstrate it. Leg B also settles the argument that decided this lane: the
operator lever really does reach `viz_green` under shipped code.

**The strongest single piece of evidence here is an sha256 collision.** Leg B's
digest is **byte-identical** to the render produced by the draft's declaration
at the same canvas. Same frames, same encode -- which proves the declaration was
purely a canvas-SELECTION change and never affected how this engine paints. The
draft therefore bought nothing and cost an operator lever, measured rather than
argued.

## The consult

Opened per the review routing (Codex CLI is the consult of record for a genuine
fork). It ran ~25 minutes, mostly idle, and returned a grounded answer that was
right on both counts that mattered:

* **It refuted the declaration** on the override-path argument above.
* **It found the sbcov provenance I had missed** -- `tmp/_gen_profiles.py`
  (2026-07-20) states in its own docstring that those six profiles are
  "throwaway smoke config ... DELETED after the sweep. **NOT committed**". So
  the six COMMITTED `otr_sbcov_*` variants are LEAKED artifacts of a finished
  sweep, not orphaned build inputs -- which inverts the operator question from
  "adopt six sources?" to "delete twelve leaked artifacts?". Verified against
  the real file before being folded in.

Its claim that the tracked variant count is **45**, not 46, also checks out:
`git ls-files` counts 45, and the 46th on disk is the untracked
`otr_upscale_ltx_probe.json` from another window. **The "46 variants" baseline
quoted in `GO_FORWARD_PLAN.md` is workstation-dependent** and is corrected there.

Not everything was adopted: its suggestion to remove the leaked variants in a
repair commit is left to the operator, because deleting committed artifacts
changes what ships.

## Deliberately NOT done here

**No VRAM number, no cost row.** This lane holds nothing in VRAM (numpy + PIL +
ffmpeg); G4 is exempt for that reason and the smoke's `vram_peak_mb: null` is
correct rather than a gap.

**The other three visualizers were untouched.** Lanes 12-14 own their own G2 and
G3 rows; this lane changed no shared code.

**The canonical workflow and every profile and variant are untouched** -- the
draft's profile/variant edits were reverted in full, so this lane's diff is the
engine comment, one keyword, the gate tables, tests and docs.
