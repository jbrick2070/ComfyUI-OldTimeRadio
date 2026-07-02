# Proof #7 verdict -- talking register + 832x448 + warm still (2026-07-02)

Episode: `signal_lost_lab_race_against_time_20260702_150943` (30w, prompt 4d4bade1,
commit 8cc5d09f). Final: `otr\obs\signal_lost_lab_race_against_time_20260702_150943_silent_procgen_blended_final.mp4`
(62.1 MB, obs_publish OK, audio_byte_identical OK 614c692b7c38).

## Scores (raw shot clips, slice audio muxed for eval; bar: speech >= 2.0)

| beat | role            | init still                  | motion | r-onset | verdict |
|------|-----------------|-----------------------------|--------|---------|---------|
| b000 | music_open      | radio console               | 3.29   | 0.06    | exempt (console register by design) |
| b001 | announcer       | radio mouth (face-forward)  | **4.62** | 0.286 | **PASS** |
| b002 | character (c03) | scene still (landscape)     | 0.62   | 0.146   | FAIL |
| b003 | character (c02) | scene still (landscape)     | 0.32   | 0.205   | FAIL |
| b004 | character (c03) | scene still (landscape)     | 1.27   | 0.171   | re-roll band |
| b005 | announcer       | radio mouth (face-forward)  | **5.51** | 0.226 | **PASS** |

Canonical isolation reference was 3.32; the announcers now BEAT the canonical
(4.62 / 5.51). The talking register + verbatim prompt + fixed 1920x1088 guide
chain fully land for the radio bookends.

## The character residual is the INIT, not the prompt

The register FIRED on every character beat (log: "IA2V TALKING register:
character beat bNNN M4 wall -> compact talking prompt (202 chars)"), but the
driver logs its own smoking gun:

    ltx_audio_in: beat b002 conditioning on scene still still_b002_*.png
    (landscape; portrait never used)

Character beats condition on WIDE landscape scene stills -- the face is a
small fraction of the canvas, so the mouth region carries too few pixels for
the audio coupling to grab. Announcers win because their init IS a
face-forward close-up. This is exactly the S4 prediction (antigravity r2:
wide-portrait-only init routing + LOUD aspect guard).

Portrait-vs-wide A/B: run on the isolation harness same day (one variable =
init still; b002's exact slice audio + exact 202-char production prompt +
832x448@25fps/6.1s): legs `pw_ab_scene` vs `pw_ab_portrait` under
`output\otr\episodes\canonical_ia2v_probe\`. Results appended below.

## Timing / VRAM (operator ask)

- Total episode: **31:19** (vs ~28.5 min proof5 at 512x288, ~27 min old
  recipe) -- the 832x448 upsampled two-stage costs ~3 extra minutes.
- Per-clip cadence (slice->slice): 4:16, 4:24, 3:22, 3:15, 3:02, 3:50
  (~3-4.5 min/clip; long announcer beats are the 4min+ ones).
- Rendered canvas: 832x448 (16:9 snap of the 832x480 default: 832/16*9=468
  -> /32 floor = 448).
- VRAM: device peak ~13.3 GB observed (< 14.5 GB ceiling, no guard trip);
  torch-side peak 8.1 GB.

## A/B result (portrait vs wide init) -- DECISIVE

Isolation harness, ONE variable (the init still); b002's exact slice audio +
the exact 202-char production prompt + 832x448@25fps/6.1s:

| leg | init | motion | r-onset | lag |
|-----|------|--------|---------|-----|
| pw_ab_scene    | wide scene still (production behavior) | 0.57 | 0.124 | -1 |
| pw_ab_portrait | c03 in-character portrait              | **2.86** | 0.208 | **0** |

The scene leg REPRODUCES production b002 (0.57 vs 0.62) on the known-good
harness -- the init is the whole residual. The portrait leg clears the bar
with perfect lag alignment. Frames: `pw_ab_scene_frame.png` (dark, small
face) vs `pw_ab_portrait_frame.png` (full-frame 16:9 center-crop, mid-speech,
NO pillarbox -- the 2026-06-20 pillarbox failure mode does not occur under
the ia2v canvas-independent guide chain).

## S4 SHIPPED same session

render_driver: under the ia2v talking register a `character_video`
ltx_audio_in beat conditions on the cast member's PORTRAIT
(init_source=`character_portrait_ia2v`, LOUD swap log w/ dims aspect guard);
a missing cast portrait fails LOUD (NO FALLBACK to the too-small scene
still). Announcer/music bookends + all single-pass recipes keep the scene
still (the 2026-06-20 directive holds outside RECIPE_IA2V). 4 new tests;
legacy wide-still character contract pinned to distilled_native.

Videos for the operator eyeball:
- `docs\2026-07-02-canonical-ia2v\side_by_side_proof4_vs_proof7_announcer.mp4`
  (proof4 OLD vs proof7 NEW, announcer open, sound on)
- proof7 final: `otr\obs\signal_lost_lab_race_against_time_20260702_150943_silent_procgen_blended_final.mp4`
- A/B raw legs: `output\otr\episodes\canonical_ia2v_probe\ia2v_smoke_00010_.mp4`
  (scene) / `ia2v_smoke_00011_.mp4` (portrait)
