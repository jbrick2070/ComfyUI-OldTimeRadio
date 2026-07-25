# PER-BEAT STILLS -- r1 architecture judgment

**Run:** `kibitz-runs/2026-07-25-per-beat-stills/r1/`. HEAD `22e65a07`,
`v2.0-alpha`. Panel: codex `gpt-5.6-sol` (high, pin verified -- the FIRST
launch silently drifted to `gpt-5.5` and was killed before it produced
anything) + agy `Gemini 3.6 Flash (High)` (DISPLAY name verified). Claude is
the grounded anchor panelist and sole judge. Anchor: `driver_anchor.md`,
written before either review was read.

**Both seats returned VERDICT: no.** Neither ratified Option C, and both
rejected it for the SAME reason, reached independently.

---

## 1. THE HEADLINE: the premise was false, 2 of 2, and it was MY premise

Both panelists independently found that **nothing in this build renders more
than one clip per beat today**, and that the two pieces of evidence I put in
the brief as "the machinery may already exist" do not mean what their names
suggest. I verified both corrections against the code myself:

- **Veo's `last_frame` is NOT clip chaining.** CONFIRMED at
  `eng_google_veo_video.py:277-293`: `last_frame` is paired with `init_image`
  and sent as `instance["lastFrame"]`, with a hard error if `last_frame` is
  supplied without `init_image`. That is **first-frame/last-frame
  interpolation WITHIN ONE CLIP** -- "generate a clip that starts here and
  ends there". No code extracts clip k's output or schedules clip k+1.
- **WAN does not segment a beat; it PING-PONGS.** CONFIRMED at
  `eng_wan_ti2v.py:349-388` and `:521-535`. `_floor_length` predicts a
  VRAM-bounded render length, and the render tail then calls
  `_wb.extend_frames_to_target(frames, target_frames)` -- the docstring is
  explicit: *"ping-pong-extend the VRAM-bounded short render up to the beat's
  audio-derived target so the composite fills the beat with motion instead of
  holding the last frame (the 0.68s-then-freeze bug)."*
- **One clip is stored per shot.** CONFIRMED at `render_driver.py:2627`:
  `clips[out_shot["shot_id"]] = clip` -- a dict keyed by shot id, one entry.

**Consequence, and it is the whole finding:** the 177-frame beat on the 8-GB
WAN tier does not become 11 clips of 17 frames. It becomes ONE 17-frame render
ping-ponged out to 177 frames. `max_render_frames` is a *render workload*
ceiling, not a segmenter.

So **"as many stills as it takes to cover the beat" is a NEW CAPABILITY being
requested, not an existing pattern to standardise.** codex: *"decide
explicitly whether per-beat still coverage replaces WAN clip-fill."* agy:
*"multi-clip rendering per beat is a new feature requirement, not an existing
pipeline pattern."*

That reframes the operator's ask from "wire up the thing that's already
there" to "replace boomerang fill with real multi-clip rendering" -- a much
larger and much more valuable piece of work, and one he should authorise with
its true size visible.

## 2. Where both seats converge -- this is the architecture of record

Independently reached, and I confirmed each against the code:

1. **CUT `StillPlanRow`, its closed enums, and the 31 declarations.** Both.
   Unchanged from the prior R1 -- that decision survives this round.
2. **CUT the `cardinality` enum specifically.** Both, and the operator's
   doctrine is what settles it. The apparent contradiction (43 `per_beat` vs
   14 `per_subject`) is NOT a contradiction: **asset cardinality and per-clip
   assignment are different axes.** CONFIRMED at `render_driver.py:1531-1536`
   -- face lanes resolve `init_image` from `_portrait_index(ledger)[char_id]`,
   i.e. BY CHARACTER, while scene assets resolve by beat. Several clips may
   reference one stable identity asset. The operator's rule holds without
   touching those 14 rows.
3. **ONE pre-validation pass resolves FINAL effective engine per shot, before
   `validate_and_repair_still_spine`.** Both. It must absorb BOTH the force
   map (`render_driver.py:2783`) and the radio-host redirect
   (`render_driver.py:1413`, called per shot at `:1510` -- i.e. AFTER
   validation at `otr_video_render_batch.py:322`). Render may verify the
   snapshot; it may never mutate the engine.
4. **Freeze at SHOT granularity, not episode.** Both, and this is my anchor's
   M1 confirmed: "frozen" and "per-beat" stop conflicting when the unit of
   capture is the shot. ShotLock is the natural owner -- it is already the
   single `ledger["video"]` authority and already owns audio-bound frame
   planning.
5. **Per-adapter continuation strategy, NOT a uniform rule.** Both, on the
   same grounds I anchored: chaining a drifting last frame through an
   audio-driven FACE engine causes identity drift within a beat, which is the
   BUG-LOCAL-129 / 2026-06-30 "generic human host" defect class again. Scene
   i2v wants chain; face wants reuse; `viz_*` wants none.
6. **Fail closed on a malformed force map.** Both. `render_driver.py:2798`
   currently logs `IGNORED (parse)` and returns the ledger unchanged.
7. **`+ Add Custom Model` must fail closed** (`otr_video_director.py:443-481`).
   Both.
8. **`render_aspect` and provider-side become required, explicit facts**, with
   no silent portrait default (`still_plan_helpers.py:177-189`) and cloud
   classification going through the real predicate `_is_cloud_video_engine`
   (`render_driver.py:1274-1295`), never a bare `getattr`. Both.
9. **Per-beat LTX recipe survives as an explicit SHOT-OWNED override**, not an
   ambient env re-read. Both. This answers the operator's original LTX
   question: he keeps the capability, and it stops escaping the freeze.

## 3. The one real split, and my judge call

**Coverage arithmetic. agy is wrong; codex is right. I reject agy's fix.**

agy's SHOULD-FIX 2 proposes
`still_count = ceil(beat_duration_frames / engine_max_render_frames)`.
codex explicitly forbids exactly that formula. codex has the better of it on
evidence I verified:

- `max_render_frames` bounds RENDER WORKLOAD, not delivered duration. With
  ping-pong fill in place, rendered frames and visible frames are different
  numbers, so the quotient is meaningless as a still count.
- `_floor_length` calls `_MC.compute_real_frame_budget(_MC.free_vram_mb(), ...)`
  -- it reads **live free VRAM** (`eng_wan_ti2v.py:378-388`). A still count
  derived from it would be computed in the image phase from a VRAM reading
  that will not hold by render time. That is an unstable plan, and stills
  would already be minted against it.

**Adopted instead (codex):** each adapter exposes a PURE legal-frame contract
(fixed/discrete lengths, min, max, quantization) over the frozen route and
profile -- no VRAM reads at plan time -- and ONE shared partitioner produces
the render/visible segment plan with deterministic final-segment trim.
Runtime resource loss must FAIL that shot, never silently revise the segment
count after stills are minted.

**Secondary split: reuse `ExecutionGroup` for coverage clips?** codex says
cut it; agy lists it as optional. **codex wins, and it corrects my anchor.**
CONFIRMED: `build_execution_plan` (`otr_shot_lock.py:1058-1089`) emits one
consumer group **per ROLE** -- *"CW-1 emits one consumer group per role that
has beats (no base-clip providers yet -> no edges)"* -- and `run_episode`
renders directly from shots without consulting them. Coverage clips are
sequential pieces of ONE shot, not provider/consumer engine topology. A linear
chain does not need a general DAG. **Adopted: an ordered `clips[]` /
`segments[]` plan on the shot** carrying clip id, render frames, visible
frames, timeline offset, init assignment, and optional predecessor id.

## 4. Corrections to MY OWN anchor -- stated plainly

- **My M6 coverage example was wrong.** I wrote that a 177-frame beat at a
  17-frame ceiling is "a coverage problem". It is currently a ping-pong
  problem. Both seats caught it. The requirement is still real; my
  illustration of it was not.
- **My M9 was wrong twice.** I first over-claimed the execution-group DAG as
  half-built machinery to reuse, then corrected myself mid-round to "a
  designed, empty extension point at role granularity". codex went further and
  is right: it is the wrong home entirely.
- **My M8 cited Veo's `last_frame` as evidence a chaining concept exists.** It
  is not that. Both seats corrected it.

The anchor's M1 (shot-granular freeze), M2 (one mutation pass before
validation), M3/M4 (aspect + provider_side explicit), M5 (fail closed) and M8's
per-adapter *conclusion* all survive on both seats' agreement.

## 5. DECISIONS THE OPERATOR OWNS -- no code until these are answered

**D1. Does multi-clip-per-beat rendering get built at all?** It is a new
capability, not a wiring job. Building it means: a per-adapter legal-frame
contract, a shared partitioner, per-segment audio slicing for
audio-conditioned engines, per-segment render, chain-frame extraction and
validation, concatenation with deterministic trim, and one-row-per-beat
downstream manifest preserved with subclip receipts under it. That is a
multi-day block, not a chunk. **Alternative he may prefer:** keep ping-pong
fill where it is adequate and build multi-clip only for engines where the
boomerang is visibly wrong.

**D2. HuMo: chain or reuse?** His instruction was "similar for all HuMos, WAN
8GB, LTXes and the other cloud engines". Read as CAPABILITY (every engine gets
a per-clip still contract) both seats and I agree. Read as STRATEGY (everyone
chains) it collides with identity drift on the face family. **Recommendation:
universal capability, `reuse` for audio-driven face, `chain` for scene i2v and
cloud, `none` for `viz_*`.** One token per adapter; his call to override.

**D3. Does the ROUTING FREEZE still ship first and alone?** Yes -- and this is
now stronger, not weaker. The shot-granular route lock is independently
shippable with ONE clip per shot, delivers the real bug fix (spine validated
against the engine that actually renders), and is the precondition for any
coverage work. Multi-clip lands after, on top of it.

## 6. Ordering of record (revised)

1. **Route lock** -- ShotLock stamps final engine/family/aspect/provider-side/
   recipe per shot; force map + radio-host redirect resolve THERE; spine
   validation and MetaBrief read the snapshot; render verifies, never mutates.
   Ships alone, with the forced-route live proof. One clip per shot.
2. **Compact per-adapter descriptor + materializer + post-registration audit**
   replacing `StillPlanRow` (keep the registry-parity invariant from
   `test_still_plan_audit.py:87-114`; it catches adapters silently lost behind
   the guarded imports).
3. **Teardown** of the table, enums and 31 declarations, with an explicit
   enumerated removal list.
4. **Multi-clip coverage** -- only if D1 is yes.
5. **Prompt hook** last.

**One process note from codex worth keeping (SHOULD-FIX 4):** the
validation/mutation ordering defect is CODE-CONFIRMED but has no live
production artifact, and this project's own doctrine reserves production-bug
status for a reproduced failure. It should be described as a code-confirmed
defect with a forced-route proof owed -- not as a live PBUG row -- until that
leg runs.
