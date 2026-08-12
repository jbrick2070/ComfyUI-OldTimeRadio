# VIDEO_LANE_PREFLIGHT receipt -- lane 16, `still_pan`

`VIDEO_LANE_PREFLIGHT receipt: still_pan | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane16_still_pan/ | verdict PASS -- 7/7`

The second still lane. Its two answers are lane 15's, taken on **this lane's own
evidence** rather than inherited -- which is the whole reason lane 15 left the
call open instead of making it for three lanes at once.

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | PASS | PASS | no weights on this lane |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT |
| G3 contract vs runtime | PASS | PASS | already green -- lane 10's shared-base fix |
| G4 / G6 | n/a | n/a | exempt -- CPU/ffmpeg lane |
| G5 audio law (V-1) | PASS | PASS | already probes its own emitted mp4 |
| G7 public surface | PASS | PASS | matrix unchanged (nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.
The ffmpeg PREFLIGHT gate arrived free with lane 15's shared-base fix.

## S8b-12(b) -- the black-beat defect, closed here too

`_require_still = True` on `StillPanFamily`. The evidence is the same as lane
15's and was re-run rather than cited:

* `default_roles` is empty and **nothing routes here automatically** -- the
  `UNIVERSAL_FLOOR` / `FLOOR_NAMES` / `make_fallback_of` machinery was ripped
  2026-07-02 and only comments recording the rip remain.
* `accepts_still = True`, so the image dispatcher mints this lane's still. A
  missing still therefore means **MINTING FAILED** -- and the dark lavfi floor
  turned that into a silent black beat the composite positioned like any other.

**`still_flat` is deliberately still untouched.** The base default stays False;
lane 17 makes its own call on its own evidence. It is now the LAST family
carrying the dark floor.

## The scope guard from lane 15 fired, and that is the ledger working

Lane 15 wrote `test_the_other_still_families_are_UNCHANGED_by_that_refusal`
asserting `still_pan._require_still is False`. Lane 16 flipped it and **that
test went red** -- exactly its job. A behaviour change spreading across a shared
shelf has to be an explicit act per lane, never a side effect of a base default.

It is rewritten as `test_which_still_families_REFUSE_a_missing_still_is_pinned_per_lane`,
carrying the state as a LEDGER of who has ruled:

| family | refuses? | ruled by |
|---|---|---|
| `still_word` | True | always did (Sprint B -- a word card cannot be black) |
| `still_motion` | True | lane 15 |
| `still_pan` | True | **lane 16** |
| `still_flat` | False | **lane 17's call, not yet made** |
| the BASE | False | so a new cheap family opts IN, never inherits it |

The companion floor test's parameter list shrank the same way -- it is now the
live scope of the dark floor, and it says in its own docstring that **if that
list ever empties while the `else:` branch survives, the branch is dead code and
should be deleted, not left as a floor nobody reaches.**

## G2 -- INERT, and the smoke PROVED the premise rather than asserting it

Three profiles set `render.canvas_w/h` here. `_still_motion` defaults True on
this family, so it renders through the same `wrapper_bridge.ffmpeg_still_motion_cmd`
lane 15 verified: the caller's width/height, the still scaled to COVER them, no
native canvas. The `even_dim()` snap is a yuv420p mod-2 codec requirement, not a
size of its own.

**And the two lanes' smokes came out BYTE-IDENTICAL, which is the proof:**

```
otr_floor_still_motion_s93obx_o.mp4  sha256 3692f155...5ed6a3fa   (lane 15)
otr_floor_still_pan_nrrbf8jr.mp4     sha256 3692f155...5ed6a3fa   (lane 16)
```

Two distinct files on disk, same digest, same still and canvas and frame count.
That is what "they share one builder" means in evidence rather than in prose --
and it is the claim the G2 justification rests on, so it was worth checking
instead of asserting.

## The solo smoke -- LIVE PASS, two legs

Stock `default` boot; box reset per CLAUDE.md section 4 first.

| | LEG A -- real still | LEG B -- the REFUSAL |
|---|---|---|
| Harness | `--engine still_pan --frames 100 --portrait <png>` | `--engine still_pan --frames 100 --expect-fail "requires a base still"` |
| Prompt id | `da93851a-9aa7-423a-80b7-16280f1ed20f` | fail-closed, NAMED |
| Canvas PROBED | **832x480** | n/a |
| Frames PROBED | **100** exactly | n/a |
| Rate / codec | 25/1, h264, yuv420p | n/a |
| Audio | **zero audio streams** | n/a |
| sha256 | `3692f155b93b5f87ecf52c51f97ba24a2ed55dd644f72f70e380a2dd5ed6a3fa` | n/a |

Leg B fired the new refusal on the live server. Before this lane that path
returned a clean black mp4 and `ok: true`.

## Stale text corrected while here

Three sentences that had gone false and would have argued against this fix:

* `StillPanFamily`'s own docstring said the family "always renders". It does
  not, when its still is missing.
* `render_driver.py` said `still_word` has no floor "Unlike still_pan/still_flat"
  -- true when written, wrong for two of the three now.
* `tests/test_video_cheap_render.py`'s module docstring still called this shelf
  "the always-succeeds radio floor ... the fallback-chain terminus the A-S6
  chain humo -> humo_1.7B -> still_motion converges on" -- false twice over.

## What the QA pass found

Three things, all fixed before the push, and two of them are the same stale
framing this lane was already hunting:

1. **`wrapper_bridge.ffmpeg_lavfi_floor_cmd`'s docstring** still ended "so the
   radio floor ALWAYS renders (the fallback-chain terminus)" -- the fourth
   instance of that claim, in the very function the `else:` branch calls. I had
   found three and missed the one closest to the mechanism.
2. **`test_video_render_driver_additive.py`** had an aside saying a missing
   still leaves `init_image` empty "(the cheap family will draw its floor)".
   Its assertions are one layer up and stay correct; only the aside was wrong.
3. **The scope guard was not generic** -- it enumerated four engines by name, so
   a NEW cheap family could inherit or hand-set the flag unnoticed.

Fixing (3) surfaced something worth recording: **`mesh_stage` also extends
`_CheapFamilyBase`.** It takes the frame contract and the canvas/still helpers,
but OVERRIDES `render_clip` completely (hy3d -> Blender), so it never reaches
the branch `_require_still` guards and **the flag is INERT there**. It refuses a
missing still by its own `FileNotFoundError` instead. The guard now says so and
asserts BOTH halves -- that the flag is unread in that override, and that
something else does the refusing -- because "this flag does not apply here" is
only safe when it is paired with what does.

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/ffmpeg lane, G4 exempt.
**`still_flat` and `still_word` behaviour unchanged.**
**No profile, variant, workflow or `ENGINE_MATRIX.md` change** -- nothing declared.
