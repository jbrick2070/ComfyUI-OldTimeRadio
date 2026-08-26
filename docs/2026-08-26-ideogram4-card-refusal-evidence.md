# LIVE EVIDENCE -- `ideogram4_local` refuses CARD beats, and the render dies later

Extracted 2026-08-26 from the bank/engine sweep's server log. The log is under
`tmp/`, which the launcher ROTATES on every reboot -- the same mechanism that
nearly destroyed the 2026-08-25 dossier evidence -- so the numbers are copied
here where they survive.

Source run: `scripts/otr_bank_engine_sweep.py`, 8 legs, one act each,
`google/gemma-4-E2B-it (3.0 GB)` in BOTH the creative and technical writer
slots. Receipt: `docs/2026-08-26-bank-engine-e2b-sweep-receipt.json`.
Rotated server log at time of extraction:
`tmp/otr_leg_dossier_e2b_20260826.20260826_024217_695.log`.

## The split is along ONE axis, and it is not the bank or the model

| bank | `otr_soak_llmsweep_01` | `otr_soak_llmsweep_02` |
|---|---|---|
| media_archive | PASS | FAIL |
| original | PASS | FAIL |
| public_domain | PASS | FAIL |
| shakespeare | PASS | FAIL |

The two profiles differ ONLY in which image engines the three role dropdowns
name. Profile 01 routes music to `flux_gen1`; profile 02 routes music to
`ideogram4_local`. Every leg that routed a music beat to `ideogram4_local`
died; every leg that did not, published.

So `gemma-4-E2B-it` driving both writer slots produced a complete published
episode on ALL FOUR banks. The failures are one image engine.

## Per-engine outcome across the whole sweep

| engine | minted | refused |
|---|---|---|
| flux2_klein | 35 | 0 |
| z_image_turbo | 32 | 0 |
| flux_gen1 | 16 | 0 |
| lumina_image | 8 | 0 |
| **ideogram4_local** | **2** | **6** |

91 mints from the other four engines, zero refusals. `ideogram4_local`
succeeded 2 of 8 attempts -- 25%.

## The six refusal events, verbatim

Every one is a MUSIC beat. There is no third object in the list.

```
ideogram4_local still_music_opening_001 min=79.0 std=10.5
ideogram4_local still_music_closing_001 min=80.0 std=10.5
ideogram4_local still_music_closing_001 min=78.0 std=10.2
ideogram4_local still_music_opening_001 min=80.0 std=10.2
ideogram4_local still_music_closing_001 min=80.0 std=10.3
ideogram4_local still_music_opening_001 min=87.0 std=10.5
```

The detector's own reference: *"a real card measures min~0, std~27-41"*. Every
refused frame lands at min 78-87 and std 10.2-10.5 -- a near-uniform mid-gray,
tightly clustered across six independent seeds on four different banks. That
clustering is the point: this is one reproducible output shape, not six
unrelated bad draws.

It is NOT seed-deterministic -- the engine did mint 2 cards -- so a retry at a
new seed would sometimes work. But at a 25% success rate, any episode routing a
required beat to this engine will usually die.

## The prompt that was refused

Benign, and worth recording so nobody hunts a content cause:

```
archival documentary. a vintage tabletop tube radio receiver glowing warmly,
aged vacuum tubes and worn dials, a dusty archive room, a theater projection
area, cinematic three-quarter framing, people shown with full heads and ...
negative: oversaturated, glossy, clean digital, plastic skin, waxy skin,
sterile studio lighting, cartoon, illustration, text, watermark
seed: 3356364758
```

The engine's own message attributes it to the card: *"The card text or its
styling was refused by the model, not by OTR."*

**CORRECTION, 2026-08-26, operator-prompted.** An earlier revision of this file
claimed music beats are "title cards carrying rendered TEXT". **That is WRONG,
and it was inferred from the beat NAME rather than read from the prompt.** The
refused prompt is a PROSE SCENE description and it explicitly forbids text:

```
archival documentary. a vintage tabletop tube radio receiver glowing warmly,
aged vacuum tubes and worn dials, a dusty archive room, a theater projection
area, cinematic three-quarter framing, people shown with full heads and clear
headroom inside frame, faces unobstructed, balanced composition, ...
sepia tones, aged paper, archival documentary still, ... no on-screen text
```

So the engine was handed the three things it is worst at, at once: a PROSE
shape (which `docs/2026-08-21-ideogram4-verdict.md` ROUND 4 proves is what it
refuses -- shape, not content), a request for FACES, and an explicit
`no on-screen text` instruction that forbids the typography this engine is
better at than anything else shipped. A 75% refusal rate against that is not
surprising.

That correction does not change the PBUG's mechanism or its numbers -- the
sanctioned-gap contradiction and the 6/2 refusal split stand. It changes the
REMEDY space: the option nobody had considered is to give this engine a
CARD-shaped prompt when it holds the music slot, rather than benching it or
restricting it to scenes.

## The contradiction that turns a missing picture into a dead episode

Two components disagree about whether a sanctioned gap is survivable:

* `OTR_ImageGenDispatcher` records the refusal and states
  *"The episode CONTINUES with no still for this object"* and
  *"the episode continues (operator 2026-08-22)"*.
* `render_driver.validate_and_repair_still_spine` then raises
  `RenderError: still-spine handoff missing materialized scene still for shot
  shot_music_opening_001 beat music_opening_001 engine still_flat`.

So the gap is sanctioned at the point it is created and fatal at the point it
is consumed. The episode dies AFTER paying for the script, the voices, the
audio mux and every other still -- roughly 10-11 minutes of a ~11 minute leg.

This is the OPEN item already carrying an r1 judgment at
`kibitz-runs/2026-08-25-model-refusal-required-still/r1/judgment.md`, with an
operator question outstanding on the all-refused-episode case. This document
adds the frequency data that judgment did not have: it is not a rare edge, it
is one engine failing 75% of the time on one beat type.
