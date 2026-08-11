# VIDEO_LANE_PREFLIGHT receipt -- lane 8, `ltx098_low_video` (`ltx_8gb`)

`VIDEO_LANE_PREFLIGHT receipt: ltx_8gb | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane08_ltx098_low_video/ | verdict PASS`

The lane whose public marker had been provisional since it was written, because
nothing had ever measured it on this box. It has a number now.

## Matrix row

7/7 GREEN. Both `EXPECTED_RED` entries (`G2`, `G6`) deleted in this commit.

## S8b-13 -- the missing Sage gate, on the family BUG-070 was written for

`ltx_8gb` is an **LTX-Video 0.9.8** engine and it was the only one of the three
LTX lanes with **no `assert_sage_not_patched` call at all**. Both siblings have
it (`eng_ltx_video.py`, `eng_ltx_av.py`). int8-PV SageAttention process-ABORTS
LTX with no traceback, so "no gate" does not mean "degraded output" -- it means
the failure mode is a dead process instead of a named refusal.

**Ordered FIRST, before any weight is resolved.** A refusal that costs nothing
beats one that costs a checkpoint load, and the test proves the ordering rather
than assuming it: it makes weight resolution raise `RuntimeError`, so if the
Sage gate did not run first the test fails with the wrong exception type.

**The node gate was the same hole one level down.** `assert_usable` checked the
render knobs, the checkpoint, its integrity and the T5 -- and no node classes.
So a missing LTXV class surfaced inside `load()`, mid-render, after the
checkpoint had already been paid for. It now resolves every class at preflight,
collects **every** miss before raising (naming one at a time turns a fresh
install into a sequence of failed renders), and reads the **active** candidate
set -- the tiled-VAE knob swaps `VAEDecode` for `VAEDecodeTiled`, so a fixed
list would gate the wrong class in one of the two configurations.

**Adding the gates turned six existing tests red, and that is worth recording.**
Six "CONTROL ... still passes" / "stays usable" checks in
`test_ltx_8gb_assert_usable_single_clip.py` and
`test_ltx_8gb_dir_override_tripwire.py` call the REAL `assert_usable` on a CPU
box where ComfyUI's registry is empty, so the new node gate refused before their
actual subject -- loader tokens, DIR overrides, the integrity floor -- was ever
reached. Fixed at the fixtures (hand them a mapping in which every candidate
exists), never by weakening the gate. That is lesson L9's shape again: when a
gate gets stricter, the checks that were using it as a proxy for something else
have to say so out loud.

Why the sibling never hit this: the CPU suite never calls
`LtxVideoEngine.assert_usable()` at all, so its identical node gate is simply
never exercised there. `ltx_8gb` has a dedicated `assert_usable` suite, which is
why the hole was visible here and invisible next door.

## S8b-11 -- two profiles describing a render that never happened

`otr_g4_ltx_8gb.json` and `otr_w45_ltx_8gb.json` both set
`render.canvas_w/h = 832x480` on a lane that **declares 512x288**. The
declaration is applied LAST and overruled them, so the render was always
512x288 -- but the config an operator reads said something 2.7x larger, on the
tier that exists precisely because it cannot afford those pixels.

**512x288 stands, and it was already right.** It is the 2026-07-26 arc
judgment's ruled canvas: exact 16:9, /32-clean, zero pad area to 1920x1080, and
chosen because at 512x288 a beat plays as ONE continuous shot. The profiles
moved to the declaration; nothing about the render changed.

## The `low` marker, which is the reason this lane needed a measurement

The evidence manifest said, in its own words, "**NO measurement of any kind on
this box, which is also why its low/high public marker is still provisional**",
and the corpus told this lane to run a lab-first measurement **before** final
naming. So the smoke came first and the rename rode on its result.

`ltx_8gb` was an IDENTITY row -- public id == internal id -- like lane 6's
`fastwan_8gb`, so it needs **no** `_LEGACY_ENGINE_ALIASES` entry on the way
out: a bare internal id already passes through `resolve_engine_id` step 3, and
adding one would imply an internal rename that never happened. The internal id
KEEPS `8gb`; only the public surface loses it.

**What the label deliberately does NOT say.** It states the measured cost --
"6.8 GiB net at 512x288x161" -- and not "runs on an 8 GB card". Net cost is not
a fit promise on a card whose desktop already eats some of it, and asserting
otherwise would repeat the exact mistake lane 5 retired the `8gb` token for.

## G8.1 solo smoke

| Item | Value |
|---|---|
| Boot | stock `default` contract on the `LTX` token (Sage-free, BUG-070) |
| Harness | `_otr_single_engine_smoke.py --engine ltx_8gb --frames 161` with a real scene still |
| Prompt id | `a33531d1-5985-48ce-a709-9b309ca85b3c` |
| Recipe | `ltx098_distilled_2b_i2v_single_pass_v2`, no quant |
| Wall time | **22.1 s** |
| Canvas PROBED | **512x288** -- equals the declaration |
| Frames PROBED | **161** counted, duration 6.440 s = 161/25 exactly |
| Rate | 25/1 |
| Codec / pixfmt | h264 / yuv420p |
| Audio | **zero audio streams** -- silence PROVED on the emitted file |
| Trim | none; 161 asked, 161 delivered, `extension_mode: none` |
| Peak, ABSOLUTE | **9,106 MB device-total, COLD, `VramPeakProbe` MAXIMUM** |
| Peak, NET | **6,835 MB** (minus this leg's own 2,271 MB pre-queue baseline) |
| Headroom | **5,394 MB** under the 14,500 MB ceiling |
| Artifact | `.../lane08_ltx098_low_video/ltx_8gb_512x288_f161_default_smoke.mp4` |
| sha256 | `65d0dc199d09d5239bc7874ec3b93bb6173bb5785047b5f20e845fee0ba834bd` |

**The cheapest lane in the roster, by a distance.** Against lane 7's
`ltx23_low_audio_in` on the same box, same boot, same day: **1.75x cheaper by
NET** (6,835 vs 11,952 MB) and **13.8x faster per beat** (22.1 s vs 303.8 s).
`low` is not a hedge on this lane; it is the measurement.

## Live menu check, on the running server

`ltx098_low_video (16:9)` is the generated option; `ltx_8gb`, `ltx_8gb (16:9)`
and `ltx098_low_video` all resolve to `ltx_8gb`. Public table stays at 10 rows
-- a MOVE plus an identity retirement, so the bijection assert never had a
chance to fire.

## Deliberately NOT done here

**No cost row.** The marker is qualified; the ENVELOPE is not. One cold leg at
one rung is a datapoint, and `QUALIFIED_COST_ROWS` stays empty with the manifest
saying "admission NOT enforced" for this lane, in words.

**The f161 ceiling is not re-derived.** The contract declares `max_frames=161`
and this leg rendered exactly that, so the top of the ladder is now measured at
the declared canvas -- but only the top. Nothing here says the rungs below it
scale linearly, and the LTX ladder is documented non-monotonic against pixel
scaling, which is why the affine FIT is CUT for this family.
