# The multi-clip maths for EVERY local video model -- verify these numbers

> ## SUPERSEDED FOR NUMBERS -- 2026-08-06
>
> **Do not read any figure in this document as current.** It is kept as the
> dated problem statement it was; its QUESTIONS are still good and several are
> still open. Its NUMBERS have moved.
>
> * **Live per-model numbers:** [`ENGINE_MATRIX.md`](ENGINE_MATRIX.md) --
>   generated from the live registry and drift-gated
>   (`python tools/engine_matrix.py --check`, a suite test).
> * **Live analysis, fix list and status:**
>   [`2026-08-02-FINAL-all-engine-maths-and-stills.md`](2026-08-02-FINAL-all-engine-maths-and-stills.md).
>
> **What is provably wrong here, verified 2026-08-06:**
>
> | this doc says | live registry |
> |---|---|
> | `humo (portrait)` -- `33-177/4`, **3 segments** | jump, **5**: 97, 97, 97, 97, 57 |
> | `humo_14B_169` -- `33-49/4`, **10 segments** of 1.96 s, NINE minted stills | jump, **5**: 97, 97, 97, 97, 57 |
> | the "49-frame ceiling" this doc argues about | `_HUMO_14B_SAFE_RENDER_FRAMES = 97` (`eng_humo.py:106`) |
> | section 3's canvas table (six engines inherit 1472x832) | superseded by the 2026-08-02 effective-canvas retraction; humo resolves its own `_native_dims`, `ltx_audio_in` takes a driver env branch |
> | section 2: `humo stills_minted=2` | the generated re-mint column reports **0** for every humo row -- UNRESOLVED, see the FINAL doc section 3 |
>
> **The lesson this document became:** it hand-typed numbers the generated
> matrix already owned, and four days later it was the only one lying. A
> hand-maintained doc must cite the generated matrix, never copy it.


**Operator, 2026-08-02:** "did you check the maths on all local video models
including humo? if not that's a big miss -- we need to see their maths for
multi beats, seconds, all that, well thought out."

It WAS a miss. I had verified two of ten. Everything below is measured from the
live registry, and the panel's job is to find where the numbers or my reading
of them are wrong. Dropdown names throughout, as the operator asked.

## 1. COVERAGE ARITHMETIC -- measured, all ten

A 442-frame beat (17.68 s at 25 fps), partitioned by each engine's own contract:

**STALE TABLE -- superseded 2026-08-06. The `humo` rows below are wrong (see the
banner at the top). Live numbers: [`ENGINE_MATRIX.md`](ENGINE_MATRIX.md).**

| dropdown name | contract | mode | segments | seconds each | total |
|---|---|---|---|---|---|
| `wan_8gb (16:9)` | 17-177/4 | chain | 3 | 7.08, 7.04, 3.56 | 17.68 s |
| `fastwan_8gb (16:9)` | 17-177/4 | chain | 3 | 7.08, 7.04, 3.56 | 17.68 s |
| `ltx_8gb (16:9)` | 9-161/8 | chain | 3 | 6.44, 6.40, 4.84 | 17.68 s |
| `ltx23_16gb_video (16:9)` | 169-169/8 | chain | 3 | 6.76, 6.72, 4.20 | 17.68 s |
| `ltx23_16gb_audio_in (16:9)` | 9-497/8 | single | 1 | 17.68 | 17.68 s |
| `humo (portrait)` | 33-177/4 | **jump** | 3 | 7.08, 7.08, 3.52 | 17.68 s |
| `humo_1.7B (portrait)` | 33-177/4 | **jump** | 3 | 7.08, 7.08, 3.52 | 17.68 s |
| `humo_1.7B_169 (16:9)` | 33-177/4 | **jump** | 3 | 7.08, 7.08, 3.52 | 17.68 s |
| `humo_14B_169 (16:9)` | 33-49/4 | **jump** | **10** | 1.96 x 9, 0.08 | 17.68 s |
| `wan_i2v (16:9)` | 33-177/4 | chain | 3 | 7.08, 7.04, 3.56 | 17.68 s |

**Every engine totals EXACTLY 17.68 s.** No drift at 33/49/177/250/442 frames on
any engine. The coverage arithmetic itself appears sound -- that is the first
thing to check, because everything below assumes it.

## 2. THE FINDING I MOST WANT BROKEN: humo JUMP-CUTS, it does not chain

The four `humo` variants declare `soft_reference` continuity, not
`strict_first_frame`, so `join_mode_for` resolves their multi-segment beats to
**JUMP**, never CHAIN:

    humo           jump   segments=3   stills_minted=2   drop_head=[0,0,0]
    humo_14B_169   jump   segments=10  stills_minted=9   drop_head=[0,...]
    wan_i2v        chain  segments=3   stills_minted=0   drop_head=[0,1,1]
    wan_8gb        chain  segments=3   stills_minted=0   drop_head=[0,1,1]

So a long `humo` beat is not one continuous performance -- it is N separately
minted stills, each animated independently, cut together. For a TALKING HEAD
that is a much stronger claim than for a scene engine: the character's face is
re-minted at every cut.

**`humo_14B_169 (16:9)` is the extreme case: a 17.68 s beat becomes TEN clips of
1.96 s with NINE freshly minted portrait stills.** Ten jump cuts in eighteen
seconds, on the same character, mid-speech.

Questions the panel must answer, grounded:
* Is jump-cut segmentation ACCEPTABLE for an audio-driven face engine at all, or
  does a re-minted portrait visibly change the character between cuts?
* Do the minted stills for one beat share a seed / portrait identity, or can
  each cut drift in appearance? Where is that pinned?
* `humo` is lip-synced. Does each segment receive only ITS OWN audio slice, so
  the mouth matches, and is the slice arithmetic the same one WIRE-W4b fixed for
  the chain engines?
* Should `humo_14B_169`'s 49-frame ceiling simply disqualify it from beats over
  ~2 s rather than producing ten cuts?

## 3. THE CANVAS GAP -- six of ten declare nothing

**STALE TABLE -- superseded 2026-08-06 by the effective-canvas retraction. The
generated matrix resolves each engine's real canvas (humo uses its own
`_native_dims`; `ltx_audio_in` takes a driver env branch). Live numbers:
[`ENGINE_MATRIX.md`](ENGINE_MATRIX.md).**

`declared_render_canvas` returns None for six engines, so they inherit the shared
1472x832 landscape default:

| declares a canvas | inherits 1472x832 |
|---|---|
| `wan_8gb` 832x480 | `ltx23_16gb_audio_in` |
| `fastwan_8gb` 832x480 | `humo`, `humo_1.7B` |
| `ltx_8gb` 512x288 | `humo_1.7B_169`, `humo_14B_169` |
| `ltx23_16gb_video` 832x480 | `wan_i2v` |

This is the dead-channel class that cost `wan_8gb` a 268-minute leg: its profile
said 832x480 and nothing carried that to the render. **Is the same true for the
six above -- are they rendering at a canvas their profiles never asked for?**
`humo` is portrait, so 1472x832 is landscape-wrong for it specifically.

## 4. THE COST MODEL IS THE BROKEN PART, and I already misused it

`FRAME_COST_MODEL` has ONE row (`wan_ti2v`); every other engine falls back to
`_DEFAULT_FRAME_COST = (7000, 185)`. At each engine's effective canvas that
prices only 29 affordable frames at the landscape default, 91 at 832x480, 247 at
512x288 -- against contract maxima of 177/169/497.

**But the row is demonstrably wrong.** `fastwan_8gb` has SHIPPED a published
episode rendering 177-frame segments at 832x480, which the row says is
impossible (91). And the live refusal that killed a leg reported
`affordable 20 frames (free=9675 MB)` -- free VRAM measured AFTER the model was
hoisted, so residency was subtracted from free AND charged again as overhead.
**Double-counted.**

I added an affordability assert to the planned path and it broke `fastwan_8gb`,
an engine that was green. Reverted the same hour. The r3 panel had already given
the order -- "update motion_common with the measured fit coefficients BEFORE
enabling the planning cap or planned path budget checking" -- and I did it
backwards.

Panel: is the double-count real, and where exactly should hoist credit enter?

## 5. WHAT THE PANEL MUST VERIFY

1. **The coverage arithmetic in section 1** -- recompute independently. Any beat
   length, any engine, where visible != beat.
2. **The humo jump-cut behaviour** (section 2) -- is it correct, is it
   acceptable for a lip-synced face, and is `humo_14B_169`'s ten-cut result a
   bug or a legitimate consequence of a 49-frame ceiling?
3. **The canvas gap** (section 3) -- for each of the six, what canvas does its
   profile ask for and what does it actually render at?
4. **The cost-model double-count** (section 4) -- confirm or refute, and name
   the correct accounting.
5. **`ltx23_16gb_audio_in` renders a 17.68 s beat as ONE 442-frame segment.**
   Its contract allows 497. Is that genuinely affordable at its canvas, or is it
   the largest unexploded shell in the set?
6. **What have I not measured at all?** `still_*`, `viz_*`, `mesh_stage`,
   `word_razzle` are unbounded/procedural -- do they need coverage maths, or is
   "any length" genuinely true for them?

## CONSTRAINTS

Every second of audio gets ORIGINAL video; no mirrors or ping-pong (the
`wan_8gb` mirror was deleted 2026-08-02). Fail loud. `wan_8gb`'s sampler recipe
is frozen. The only workflow JSON is `workflows/otr_canonical.json`. 16 GB
RTX 5080, 14.5 GB real-world ceiling. 100% local. **Do not launch renders or
boot a server.**
