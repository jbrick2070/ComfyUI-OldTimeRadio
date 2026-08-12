# VIDEO_LANE_PREFLIGHT receipt -- lane 15, `still_motion`

`VIDEO_LANE_PREFLIGHT receipt: still_motion | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane15_still_motion/ | verdict PASS -- 7/7`

The first still lane, and the first packet in this run that changed BEHAVIOUR
rather than declarations. It closes the historical black-beat defect.

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | PASS | PASS | no weights on this lane |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT |
| G3 contract vs runtime | PASS | PASS | already green -- **lane 10's shared-base fix**, inherited free |
| G4 / G6 | n/a | n/a | exempt -- CPU/ffmpeg lane |
| G5 audio law (V-1) | PASS | PASS | already probes its own emitted mp4 |
| G7 public surface | PASS | PASS | `still_plan` declared + audit-clean; matrix unchanged |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.

## S8b-12(b) -- THE BLACK-BEAT DEFECT, closed

`render_clip` emitted a dark lavfi floor when the base still was missing: a
silent, black, structurally VALID clip that the composite then positioned like
any other beat. The spec calls it "the historical black-beat defect, still
reachable"; **NO FALLBACKS** (operator 2026-07-02) says a failure must be loud.

`StillMotionFamily._require_still = True` now. **The argument against it was
stale, and that was verified rather than assumed** -- which is the whole reason
this was safe to do without asking:

* This family was once the terminus of the `humo -> humo_1.7B -> still_motion`
  degrade chain. That chain was **RIPPED on 2026-07-02**: `UNIVERSAL_FLOOR`,
  `FLOOR_NAMES` and `make_fallback_of` survive only as comments recording the
  rip, and `default_roles` is empty.
* So this engine renders only because an operator SELECTED it, and
  `accepts_still = True` means the image dispatcher mints its still. **A missing
  still therefore means MINTING FAILED** -- precisely the case that must not
  ship as a watchable black beat.
* `check_ltx_open_health` was checked and left alone: it detects a degraded OPEN
  after the fact and is about engine SELECTION, not about a still that failed to
  mint. It still says the procgen floor stays the safety net, and that remains
  true for the lanes that have one.

**Scoped to this lane.** The base default stays `False`, so `still_pan` (16) and
`still_flat` (17) are byte-identical until their own packets decide -- pinned by
`test_the_other_still_families_are_UNCHANGED_by_that_refusal`, so the next lane
can tell at a glance whether its family already refuses.

## S8b-12(a) -- the ffmpeg preflight gate, a SHARED-BASE fix

`_CheapFamilyBase.assert_usable` returned the name UNCONDITIONALLY, under a
comment saying "the real ffmpeg check runs in render_clip". True, and that is
the defect: `render_clip` runs mid-beat, after the writer, the TTS, the master
freeze and the stills are already paid for. Every `viz_*` lane gates ffmpeg at
BOTH boundaries; these four gated it at neither.

It now refuses at preflight through the same `scope_draw.find_ffmpeg` probe the
visualizers use, so there is ONE answer to "is ffmpeg here" across the CPU shelf.
Per **L13** this lands once on the base and covers **all four** still families,
and the twin assertion iterates all four rather than just this lane -- a
shared-mechanism fix has to be shown to reach every adapter sharing it.

**One test went red and it was the correct kind** (lane 8's lesson):
`test_each_family_renders_silent_clip[still_motion]` called `render_clip` with no
`asset_refs`, so it had quietly become an assertion that this family renders from
nothing -- never its subject, which is the CLIP CONTRACT. Fixed at the FIXTURE by
handing every still-consuming family a real still, never by weakening the gate.

## G2 -- INERT, premise re-checked on a path the visualizers never touch

Six profiles set `render.canvas_w/h` here. L19 says verify the premise per
engine, and this lane reaches ffmpeg through
`wrapper_bridge.ffmpeg_still_motion_cmd`, **not** `scope_draw`'s encoder -- so
lanes 11-14's check did not cover it. It still holds: the builder takes the
caller's width/height and scales the still to COVER them, so the canvas is
whatever the request carried.

**The one difference found in the whole family sweep, recorded honestly:** these
builders pass the dims through `even_dim()` first (round to nearest even >= 2).
That is a **yuv420p mod-2 CODEC requirement applied to whatever it is given** --
the same class as the shared encoder's nvenc minimum that lane 12 checked and
discounted -- not a native canvas. It is a no-op at every canvas in play, since
1472x832 and 832x480 are both already even.

So the 1472x832 an episode hands this lane is `OTR_VIDEO_LANDSCAPE_CANVAS`'s
default -- an operator lever -- and declaring would overrule it for this lane
alone. Channel declared INERT; inert, not dead (L18).

## S8b-15 -- `still_plan` is read by NOTHING in production, and stays that way

Confirmed still true at `eb3f8412`: `grep -rl still_plan nodes/` returns only
`still_plan_helpers.py`, the adapters that DECLARE it, and the audit. G7.4 is
GREEN because the plan is declared and audit-clean -- which is exactly the trap
lesson **L6** describes, so it is written down here rather than left implied:

> **A green G7 row does NOT mean `still_plan` is wired.** It means the
> declaration parses and passes audit. Nothing in the production render path
> reads it today.

Deliberately NOT wired by this lane: giving it a consumer is a design change
across every adapter that declares one, not a still-lane packet. No test asserts
"nothing reads it" either -- that would block the wiring instead of enabling it.

## The solo smoke -- LIVE PASS, two legs, and the second one is the point

Stock `default` boot; box reset per CLAUDE.md section 4 first.

| | LEG A -- real still | LEG B -- the REFUSAL |
|---|---|---|
| Harness | `--engine still_motion --frames 100 --portrait <png>` | `--engine still_motion --frames 100 --expect-fail "requires a base still"` |
| Prompt id | `b856aa4e-af82-408d-a459-46af4e17ca3d` | fail-closed, NAMED |
| Canvas PROBED | **832x480** | n/a |
| Frames PROBED | **100** exactly | n/a |
| Rate / codec | 25/1, h264, yuv420p, bt709 | n/a |
| Audio | **zero audio streams** | n/a |
| sha256 | `3692f155b93b5f87ecf52c51f97ba24a2ed55dd644f72f70e380a2dd5ed6a3fa` | n/a |

**Leg B fired the new refusal on the live server**, verbatim:

> `RuntimeError: still_motion requires a base still but none was provided/exists
> (asset_refs still/init_image='') -- refusing the dark floor (NO FALLBACKS).
> The image phase must mint this beat's still before the video render.`

That is the black-beat defect closed in production shape, not just in a unit
test. **A refusal that has never been fired is a refusal nobody has tested** --
and before this lane, that exact code path would have returned a clean black mp4
and an `ok: true`.

## What the QA pass found, and it was a behaviour lane so it mattered more

Four findings, all fixed before the push:

1. **A coverage loss the fixture fix caused.** Staging a still for every
   still-consuming family meant `test_each_family_renders_silent_clip` stopped
   driving `render_clip`'s `else:` branch -- the synthesized dark lavfi floor --
   end to end, because all three parametrized families set `uses_still`. That
   branch is STILL LIVE for `still_pan` and `still_flat`. It now has its own
   parametrized case, which doubles as the BEHAVIOURAL half of the scope guard:
   one test asserts their `_require_still` is False, the other proves what that
   still buys them.
2. **`eng_humo.py` was asserting the opposite of this lane's premise.**
   `HuMo17BEngine`'s docstring still ended "Degrades on to the zero-VRAM still
   floor (humo -> humo_1.7B -> still_motion)" -- **four lines above**
   `fallback_engine = None  # NO FALLBACKS`. Lane 15 had to grep that chain's
   machinery to prove it was gone before the refusal could ship, and the
   sentence claiming otherwise was sitting one file over, ready to talk the next
   window out of the same fix. Corrected in place.
3. + 4. Two stale passages in `GO_FORWARD_PLAN.md`: the packet count was not
   bumped, and a "Lanes 15-18" block written before this lane closed now
   contradicted the paragraph above it. Both fixed -- and this file is where
   lane 16's instructions live, so a contradiction there is not cosmetic.

Also confirmed by the pass, and worth recording because a behaviour change lives
or dies on it: **the refusal is genuinely loud.** `_render_one` is the only
production caller of the per-instance `assert_usable`; `render_shot` catches
broadly but RE-RAISES as `RenderError`, and `classify_failure` maps
`EngineUnusable` to `DEPENDENCY_MISSING`, which escalates. Nothing swallows the
`RuntimeError` from `render_clip` into a degraded clip.

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/ffmpeg lane, G4 exempt.

**`still_pan` / `still_flat` / `still_word` behaviour unchanged.** They inherit
the ffmpeg preflight gate (shared base, and that is the point of L13) and
nothing else.

**No profile, variant, workflow or `ENGINE_MATRIX.md` change** -- nothing
declared.
