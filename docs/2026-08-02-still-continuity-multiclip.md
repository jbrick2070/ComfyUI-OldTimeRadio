# Still continuity across a multi-clip beat -- what actually happens, and what should

**Operator, 2026-08-02:** "we need to get the right still continuity for
clip-to-clip beats."

This document supersedes the still-related claims in
`docs/2026-08-02-all-local-engines-multiclip-maths.md`. Two of those claims were
WRONG, and the correction is the most useful thing here.

Dropdown names throughout. Do not launch renders or boot a server.

## 1. THE CORRECTION -- I measured the planner, not production

I reported that `humo (portrait)` mints 2 per-segment stills and
`humo_14B_169 (16:9)` mints 9. **Both numbers are wrong.** They came from calling
the pure planner `coverage_plan.jump_still_requests()` directly. Production never
reaches it for those engines: `otr_shot_lock._stamp_coverage_plan` returns before
minting when `_lane_consumes_a_still` is false, and that predicate is false for
the whole `audio_driven_face` family.

Measured across the live registry at a 442-frame beat:

| dropdown name | family | join | segs | consumes a still | stills RE-MINTED |
|---|---|---|---|---|---|
| `fastwan_8gb (16:9)` | image_to_video | chain | 3 | yes | **0** |
| `ltx_8gb (16:9)` | image_to_video | chain | 3 | yes | **0** |
| `ltx23_16gb_video (16:9)` | text_to_video | chain | 3 | no | **0** |
| `ltx23_16gb_audio_in (16:9)` | audio_conditioned_video | single | 1 | yes | **0** |
| `humo (portrait)` | audio_driven_face | jump | 3 | **no** | **0** |
| `humo_1.7B (portrait)` | audio_driven_face | jump | 3 | **no** | **0** |
| `humo_1.7B_169 (16:9)` | audio_driven_face | jump | 3 | **no** | **0** |
| `humo_14B_169 (16:9)` | audio_driven_face | jump | **10** | **no** | **0** |
| `wan_i2v (16:9)` | image_to_video | chain | 3 | yes | **0** |
| `wan_ti2v (16:9)` | image_to_video | chain | 3 | yes | **0** |
| `mesh_stage`, `still_*`, `viz_*` | various | single | 1 | mixed | **0** |

**No local engine ever re-mints a per-segment still.** The re-mint path needs
JUMP *and* a still-consuming lane, and nothing is both: every JUMP engine is a
face lane that consumes no scene still, and every still-consuming lane resolves
to chain or single. `jump_still_requests` returns nothing for a CHAIN plan.

That makes the re-mint machinery in `otr_image_gen_dispatcher.py:650-690`
**unreachable on every local engine today** -- including its deliberate
seed-drop, which reasons at length about what a jump cut should look like and
currently governs nothing. Panel: confirm or refute that it is dead. If dead, is
it a latent trap for the first cloud lane that lands on JUMP + still-consuming,
or is it correct-and-waiting?

## 2. SO WHAT IS THE REAL CONTINUITY STORY? Two shapes, not three

**CHAIN (`fastwan_8gb`, `ltx_8gb`, `ltx23_16gb_video`, `wan_i2v`, `wan_ti2v`).**
Segment 0 renders from the beat's minted scene still. Every successor overwrites
`asset_refs["init_image"]` with the REAL terminal frame its predecessor ended on
(`render_driver.py:869`). Continuity is frame-exact and needs no still at all.
This looks right. Verify it is.

**JUMP (all four `humo` variants).** No per-segment still exists, so every
segment renders from the SAME character portrait as `ref_image`. Per the engine's
own note (`eng_humo.py:215-218`), HuMo wires `LoadImage -> WanHuMoImageToVideo`
`ref_image`, NOT `start_image`: the reference is an **identity hint, not a
first-frame lock**, which is exactly why the contract declares `soft_reference`
and refuses to pretend it can chain.

**Consequence: identity is preserved across humo's cuts -- but POSE resets.**
Each segment restarts from the same reference portrait. The character does not
change face; the character SNAPS BACK to the reference pose at every cut.

## 3. THE ONE THAT WORRIES ME: `humo_14B_169 (16:9)`

Its 49-frame ceiling makes a 17.68 s beat into **ten segments of 1.96 s**. Same
face throughout -- and ten pose resets in eighteen seconds. Roughly two seconds
of performance, restart, two seconds, restart.

* Is that a legitimate consequence of a 49-frame ceiling, or does it disqualify
  `humo_14B_169` from beats longer than about 2 s?
* Is the 49-frame ceiling even real? The other three variants carry 33-177 on the
  same architecture. Where does 49 come from, and is it a VRAM measurement or an
  inherited guess?
* If the tier stays, should long character beats route AWAY from it -- and to
  what, given it is the only 14B face tier?

## 4. THE DESIGN QUESTION THE OPERATOR IS ASKING

For a beat that needs N clips, what SHOULD the still continuity be? Candidates,
and each needs breaking:

1. **Chain where the engine can, jump where it cannot** (today's behaviour).
   Honest, but leaves humo with pose resets.
2. **Feed the previous segment's terminal frame as the next segment's
   `ref_image`.** Tempting -- it would make humo's pose flow. But `ref_image` is
   an identity hint, so conditioning it on a rendered frame feeds generation
   output back as identity, which drifts by construction and compounds over ten
   segments. Does it drift, and is the drift worse than the reset?
3. **Cap the beat length per engine** so multi-clip is rare on face lanes.
4. **Mint one still per beat and share it across segments** -- already what humo
   does, stated as a rule rather than a side effect.

Whatever wins must hold the operator's constraint: **every second of audio gets
ORIGINAL video. No mirrors, no ping-pong, no held frames.** Option 4 must not
become "the same clip twice."

## 5. WHAT THE PANEL MUST VERIFY

1. **The re-mint path is dead on all local engines** (section 1). Confirm by
   reading the mint gate and the JUMP/CHAIN resolution, not by trusting my table.
2. **CHAIN continuity is frame-exact** -- the terminal-frame extraction really
   overwrites `init_image`, really uses the LAST rendered frame, and `drop_head`
   does not double-count it.
3. **`ref_image` is an identity hint, not a first-frame lock.** This is the load-
   bearing fact for the whole design. If it is actually a first-frame lock, humo
   should chain and everything above changes.
4. **The pose-reset claim.** Does a HuMo segment genuinely restart at the
   reference pose, or does audio conditioning carry motion phase across?
5. **`humo_14B_169`'s 49-frame ceiling** -- provenance and whether ten cuts is
   acceptable.
6. **Which option in section 4 is right**, and what breaks if we adopt it.

## CONSTRAINTS

100% local, open source, offline-first. 16 GB RTX 5080, 14.5 GB real-world
ceiling. `wan_8gb`'s sampler recipe is FROZEN. The only workflow JSON is
`workflows/otr_canonical.json`. Every second of audio gets original video. Fail
loud, no fallbacks. **Do not launch renders or boot a server.**
