# Talking-Radio bookends via ltx_audio_in -- HARDENED PLAN (kibitz r1)

Panel: Codex + Antigravity (grounded) + Claude anchor/judge (the Claude Code lane skipped -- the
driver IS Claude). Both agents converged; every claim below verified against the real files.

## The reframe that reorders everything (Codex, GROUNDED)
The premise "LTX lip-syncs a radio mouth" is UNPROVEN in THIS repo and contradicts its own notes:
`eng_ltx_av.py` has no LTX lip-sync parameter (conditions on still + audio only), `render_driver.py`
documents `OTR_LTX_RADIO_FACE` as AMBIENT motion (not lip-sync), and the shipped A/B addendum says
HuMo is the only true talking host. The official comfy.org LTX-2.3 template DOES lip-sync a NON-human
mossy creature -- so the MODEL can; the open question is whether OUR engine invokes it that way.
Antigravity explains the mechanism: `LTXVImgToVideo` has NO face/landmark detector -- it relies on the
UNET's learned mouth representation, so a still with a CLEAR mouth region gets driven, one without
only drifts. => The mouth-forward still is the right lever, but the whole idea must be PROVEN with a
live probe BEFORE any engine surgery. Sharpness (the upsampler) is a SEPARATE quality question, not
the proof.

## SEQUENCED, DISTINCT sub-plans (one coder window at a time -- CLAUDE.md serialization)
Distinct + self-contained (drop each into its own window), but SEQUENCED, not parallel (CLAUDE.md:
"one coder window in the code at a time"). Order = B -> C -> A. Each green-gated on its own (suite +
Bug Bible + B7; commit+push per green chunk; UTF-8 no BOM; SFW; NO FALLBACKS; audio byte-identical).

---

### SUB-PLAN B (FIRST) -- mouth-forward LTX radio-face prompt, SPLIT from the HuMo host
**File:** `nodes/otr_meta_brief_image_prompt.py` (+ `tests/test_brief_radio_host.py`). Prereq for C.
**Codex MUST-FIX #4 (the split):** `build_radio_host_prompt(style="console_face")` currently feeds
BOTH the HuMo `radio_host_portrait` (:1078-1092) AND the ltx A/B stills (:1095-1118). A mouth-forward
change tuned for LTX can HARM HuMo face-readability. So introduce a DISTINCT LTX style -- e.g.
`style="ltx_radio_mouth"` (or a `_RADIO_CONSOLE_MOUTH` constant) used ONLY by the ltx radio-face mint;
leave the HuMo `console_face`/`radio_head_person` looks UNCHANGED.
**Build:** the LTX radio-face subject leads with a PROMINENT MOUTH: e.g. "a vintage radio whose two
round tuning dials are eyes and whose wide speaker grille is an expressive rubbery mouth/lips that
opens and closes -- a face-forward anthropomorphic radio." Big-rubbery-lips is explicitly fine
(operator). Keep brief-driven form (`radio_form_from_meta`) + the overtness mix + the correct negative
by style (Antigravity SHOULD-FIX #1: `console_face`/LTX = `RADIO_CONSOLE_NEG` "no human" (no baby
needed -- no person); `radio_head_person` = `RADIO_HEAD_PERSON_NEG` with baby).
**Safe for other consumers (Antigravity SHOULD-FIX #2, GROUNDED):** the 3D mesher `_mesh_fodder_subject`
uses `radio_form_from_meta` (faceless) -- the new mouth face terms never reach it. Confirm HuMo host
unchanged with a test on both consumers.
**Acceptance:** LTX radio-face prompt carries an explicit MOUTH token; HuMo host prompts byte-unchanged;
`test_audio_byte_identical` green (prompt-only).

---

### SUB-PLAN C (SECOND) -- PROVE the talking radio (live probe + defined criterion)
**Scope:** a live one-beat probe + a written verdict. NO production routing change (override-only).
**Codex MUST-FIX #1/#2 + Ask#4:** before any engine surgery, PROVE it. Steps:
1. Capture the ltx_audio_in nodes' real `class_type`/inputs from a live `/object_info` (do not guess).
2. Render ONE bookend beat with the sub-plan-B mouth-forward still, frozen audio, force
   `ltx_audio_in` via `OTR_FORCE_ENGINE_MAP` (Antigravity NICE-TO-HAVE: ship a one-line script so no
   manual dispatch edit), side-by-side `OTR_LTX_RADIO_FACE=0/1`. Render to otr/episodes/<ep>/, obs to
   otr/obs/; Test-Path the obs.
3. WRITTEN criterion (not "looks like it talks"): the grille-mouth open/close correlates with
   speech/music transients across the clip. Record verdict in `docs/2026-07-01-talking-radio/EYEBALL.md`
   + a HANDOFF_LOG.md line + a manifest "ambient-vs-lipsync-expectation" stamp (Codex optional) so a
   non-talking face is not misread as a broken render.
**GO/NO-GO (operator-gated):** talks -> proceed to A (quality) + a SEPARATE, same-commit workflow-JSON
change if you promote ltx_audio_in as the bookend default (Codex SHOULD-FIX #3); only-drifts -> keep the
moving-console look, HuMo stays the face path. **"Retire OTR_ENABLE_HUMO_HOSTS" stays OUT of scope**
until C passes (Codex CUT #2).

---

### SUB-PLAN A (LAST, OPTIONAL) -- LTX-AV two-stage latent upsampler (quality only)
**File:** `nodes/_otr_video_engines/eng_ltx_av.py` (+ tests). Build ONLY if C proves the concept
(Codex CUT #1: sharpness does not answer the mouth-motion question). Env-gated `OTR_LTX_AV_UPSCALE`
(default 0 = today, byte-identical).
**Corrections + engineering (both agents, GROUNDED):**
- Baseline is **832x480** (default; up to 1472x832), NOT 512x288 (Codex/Antigravity: eng_ltx_av.py
  :58-59/:771-777 + node-87 canvas). VRAM estimate must use the real dims.
- **Downscale the BASE pass** when upscaling (Antigravity #1): halve base spatial dims (e.g. ~416x240)
  in `_build_graph`, then `LTXVLatentUpsampler` + a short refine sample bring it to target -- else
  upscaling a full-res base OOMs past 14.5 GB.
- **Re-concat the AV latent for the refine** (Antigravity #2): after Separate -> upsample the VIDEO
  latent, run `LTXVConcatAVLatent` AGAIN with the original AUDIO latent before the refine
  `SamplerCustomAdvanced`.
- **Define the upscale model** (Antigravity #3): `OTR_LTX_AV_UPSCALE_MODEL` env + `_weight_paths` sanity
  floor + `_node_candidates` -- resolved/required ONLY when `OTR_LTX_AV_UPSCALE=1` (Codex SHOULD-FIX #2:
  else OFF stops being byte-identical on installs lacking the nodes).
- **`LTXVCropGuides` upscale-path-only** (Antigravity #4 / Codex): wiring it single-pass would break
  byte-identity; keep it behind the flag unless a probe proves it correctness-relevant (then update the
  regression baseline deliberately).
- **CUT** `ResizeImagesByLongerEdge` (engine pre-snaps dims) + `LTXVPreprocess` (redundant unless proven)
  from the candidate list (Antigravity CUTs).
**Acceptance:** OFF byte-identical (regression test on the single-pass trace); ON = sharper clip at the
same budget, render-phase NVML <= 14.5 GB stamped, upscale-model-absent fails LOUD.

## Net
Build order **B -> C -> A**, one window at a time. C is the pivot: prove the radio actually
lip-syncs before investing in the upsampler or any routing change. Big rubbery grille-mouth = the
lever (no face detector in LTX -> it drives whatever reads as a mouth).
