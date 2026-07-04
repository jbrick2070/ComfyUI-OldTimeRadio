# Talking-Radio bookends via ltx_audio_in (LTX-2.3) -- PLAN (for kibitz)

## The idea (grounded)
LTX-2.3 image-audio-to-video is a LIP-SYNC model that animates ANY still that reads as a
face-with-a-mouth -- the official comfy.org template lip-syncs a NON-human mossy creature with
big rubbery lips (`comfy.org/workflows/video_ltx2_3_ia2v`). So feed `ltx_audio_in` a
mouth-forward ANTHROPOMORPHIC RADIO still and the radio TALKS/SINGS to the audio -- "the radio IS
the host", literally, on the lighter engine (no HuMo needed for bookends).

The routing already exists: `OTR_LTX_RADIO_FACE=1` (the 2026-07-01 addendum) mints a wide
radio-face still per bookend role and feeds it to `ltx_audio_in`'s init on announcer/music beats,
with the master-mix slice as `audio_ref`. What's missing: (A) LTX quality via the official
two-stage latent upsampler, and (B) a still that gives the radio a PROMINENT MOUTH to lip-sync.

## THREE DISTINCT SUB-PLANS (each a SELF-CONTAINED coder-window task)
The operator codes in SEPARATE windows. A and B are INDEPENDENT (different files, no shared
edit) and can run in parallel; C is the integration eyeball + go/no-go and depends on A+B landing.
Each sub-plan is green-gated on its own (suite + Bug Bible + B7 + workflow re-validate if JSON
touched; commit AND push per green chunk to v2.0-alpha; UTF-8 no BOM; SFW; NO FALLBACKS).

---

### SUB-PLAN A -- LTX-AV two-stage latent upsampler (video engine only)
**File:** `nodes/_otr_video_engines/eng_ltx_av.py` (+ `tests/`). No other production file. No image
prompt, no workflow-JSON node change (env-gated, default OFF = byte-identical).
**Why:** your `eng_ltx_av.py` renders a SINGLE small pass (512x288) -- the official LTX-2.3 recipe
renders a base latent then UPSAMPLES the latent + light refine for sharpness. Missing nodes vs the
official recipe (grounded): `LTXVLatentUpsampler`, `LatentUpscaleModelLoader`, `LTXVPreprocess`,
`LTXVCropGuides`, `ResizeImagesByLongerEdge` (your engine has NONE of these).
**Build:**
- Add an OPTIONAL second stage behind `OTR_LTX_AV_UPSCALE` (env, default "0" = today's single-pass,
  byte-identical). ON: after the base AV-latent sample, run `LTXVLatentUpsampler` (needs the LTX
  latent-upscale model via `LatentUpscaleModelLoader`) + a short refine sample, then `VAEDecodeTiled`.
- Add the new node ids to `eng_ltx_av._node_candidates` (capture the real `class_type` + inputs from a
  live `/object_info` FIRST -- do not guess the socket names; the AV-latent nodes were captured this
  way originally). Lazy-import / fail-loud if the upscale model is absent (assert_usable probe).
- VRAM: base pass stays small; the upscaler + refine must keep render-phase NVML <= 14.5 GB (the hard
  ceiling) -- tiled decode + the small base help; add a per-beat VRAM observability line (you already
  have `VramPeakProbe`). If it can't fit, the upscale stage fails LOUD, never silently downgrades.
- Evaluate `LTXVCropGuides` for CORRECTNESS (the official uses it on the concat-AV path) -- if it
  changes framing/sync, wire it even in the single-pass path; if purely cosmetic, keep it upscale-only.
**Acceptance:** OFF = byte-identical to today (a regression test proving the single-pass trace is
unchanged). ON = a sharper clip at the same base VRAM budget (eyeball) with NVML <= 14.5 GB stamped.
Grounded node list from `comfy.org/workflows/download/adca306765ce.json` (official) +
`RCWorkflows/011326-LTX2-AudioSync-i2v-WIP.json` (WIP, has the same upsampler).

---

### SUB-PLAN B -- mouth-forward radio face (image prompt only)
**File:** `nodes/otr_meta_brief_image_prompt.py` (`build_radio_host_prompt` + the console/head
constants + negatives) (+ `tests/test_brief_radio_host.py`). No video engine, no workflow-JSON.
**Why:** LTX lip-syncs whatever has an obvious MOUTH (the mossy creature's big lips). The current
`console_face` leans on "two round dial-eyes and a radiating needle-fan mouth" -- push the MOUTH to be
unmistakable so LTX has a clear mouth to drive.
**Build:**
- Rework `_RADIO_CONSOLE_FACE` to lead with a prominent mouth: e.g. "a vintage radio whose two round
  tuning dials are eyes and whose wide speaker grille is an expressive rubbery MOUTH/lips, a
  face-forward anthropomorphic radio that opens and closes its grille-mouth" -- eyes present, but the
  MOUTH is the dominant, animatable feature (mirror the mossy-creature lips lesson).
- `radio_head_person` (announcer): same principle -- the radio HEAD's grille is the mouth.
- Keep it brief-driven (form via `radio_form_from_meta`) + the overtness mix + the "no human" /
  "no baby" negatives + the "still" era-tail story flair. Deterministic; seed-pinned (unchanged).
**Acceptance:** the console prompt contains an explicit MOUTH/lips token (not just "needle-fan"); unit
tests updated; brief-driven form + negatives intact; no human in console_face. (Purely a prompt-string
change -- audio/video paths unaffected, `test_audio_byte_identical` green.)

---

### SUB-PLAN C -- talking-radio eyeball + go/no-go (integration; depends on A+B)
**Scope:** NO new production code beyond a doc + (if the eyeball passes) a decision. Uses the EXISTING
`OTR_LTX_RADIO_FACE=1` routing + sub-plans A and B once they land.
**Build:**
- Render a bookend with `OTR_LTX_RADIO_FACE=1` + `OTR_LTX_AV_UPSCALE=1` + the mouth-forward still,
  force `ltx_audio_in` on announcer/music (`OTR_FORCE_ENGINE_MAP`), and EYEBALL: does the radio's
  grille-mouth lip-sync to the announcer/music audio, or only ambient-drift? (Render straight to
  otr/episodes/<ep>/, obs to otr/obs/; Test-Path the obs.)
- Record the verdict in `docs/2026-07-01-talking-radio/EYEBALL.md` + a HANDOFF_LOG.md line.
- GO/NO-GO decision (operator-gated, not auto): if the radio TALKS -> `ltx_audio_in` becomes the
  bookend host and `OTR_ENABLE_HUMO_HOSTS` can retire for bookends (separate follow-up commit,
  operator-gated). If it only drifts -> keep it as the moving-console look; HuMo-hosts stays the
  face path. Either way audio byte-identical; no black; LOUD on missing still.
**Acceptance:** an eyeball verdict + a recorded decision; no silent behavior change.

## Invariants (all sub-plans)
Single resident heavy <= 14.5 GB; audio byte-identical (`test_audio_byte_identical` GREEN); determinism
seed-keyed; NO FALLBACKS (LOUD on missing still / unfit upscale model / OOM); workflow-JSON edited in
the SAME change ONLY if a node/widget changes (A/B/C do NOT change node-87 -- env-gated); UTF-8 no BOM;
SFW; suite + Bug Bible + B7 green + push per green chunk.

## Ask for the panel
1. Sub-plan A: is the two-stage upsampler safe within 14.5 GB on the 5080, and is `LTXVCropGuides`
   correctness-relevant (must it be wired even single-pass) or cosmetic (upscale-only)?
2. Sub-plan B: does a "grille = mouth" still risk breaking the mesh/faceless-radio invariant elsewhere
   (the same `build_radio_host_prompt` output feeds the HuMo `radio_host_portrait` AND the ltx A/B
   stills -- confirm the mouth-forward change is safe for both consumers)?
3. Are A/B truly independent (no shared symbol) so two windows can build them in parallel?
4. Any reason `ltx_audio_in` lip-sync on a non-face radio still would fail where the mossy creature
   succeeded (init strength, crop guides, face-detection preprocessing in the LTX node)?
