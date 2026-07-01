# r1 JUDGMENT -- HuMo improvements (grounded; STRONG, plan reshaped)

Panel: Codex + Antigravity (both "no" -- deep grounded corrections). Claude anchor. Claude-Code absent.
r1 did FAR more than arc -- it caught 3 code-grounded facts that rewrite the plan.

## GROUNDED FACTS THAT RESHAPE THE PLAN
1. **Loop-fill ALREADY EXISTS + looping a TALKING FACE is WRONG.** `otr_silent_composite.py:243`
   `_should_loop_fill` (default-on `OTR_CLIP_FILL=1`), reports `looped_fill` vs `held_last_frame`
   (L410-447), and L734-750 already treats `held_last_frame` as a legibility failure. So the mush is NOT
   "build loop-fill". CRITICAL (Antigravity #1, unanimous): looping/ping-ponging a HuMo speaking mouth
   while dialogue plays DESYNCS lip-sync. => (a) EXCLUDE `audio_driven_face` from composite loop-fill;
   (b) the real HuMo underrun fix = **PHRASE-CHUNKING** (render long dialogue in speech-aligned chunks
   within the 177-frame cap) -- ELEVATE to a GOAL; (c) diagnose WHY the eyeballed beat held_last_frame
   (fill off? or HuMo already excluded but no chunking?).
2. **The radio-not-face guard is DEAD CODE.** `is_never_humo_role`/`_NEVER_HUMO_ROLES`
   (`_otr_speaker_role.py:96,168`) is defined + exported but has ZERO callers (grep-confirmed). So
   announcer/music are NOT actually blocked from HuMo today. FIX: WIRE `is_never_humo_role` into the
   render_driver dispatch AND remove announcer/music from HuMo's `roles` (eng_humo.py:89/94) so the
   dropdown/capability never offers it. The host = radio is a REQUIREMENT, not a decision.
3. **Most "quality knobs" are hardcoded, not exposed.** Only steps/cfg are env-backed (eng_humo.py:148).
   sampler=`uni_pc`, scheduler=`simple`, LoRA strength=`1.0`, shift=`8.0` are HARDCODED; audio-conditioning
   strength is NOT passed to `WanHuMoImageToVideo`. CLASSIFY every lever before build: already-tunable
   (steps/cfg + the INIT PORTRAIT = biggest VRAM-free lever) / requires-code-change (LoRA str, sampler,
   shift) / verify-wrapper-supports (audio-cond strength).

## LOCKED (build-ready direction)
- **Clip-fill:** exclude audio_driven_face from loop-fill; keep loop-fill for non-face engines; keep the
  legibility guard. HuMo underrun -> phrase-chunking GOAL (scope carefully -- deepest change).
- **Host=radio:** wire the dead guard + strip HuMo host roles. Route announcer/music bookends to the
  EXISTING `ltx_audio_in` + `radio_bookend.png` path (render_driver.py:1835 already expects LTX-family
  open beats). CUT the "new bespoke radio animator" engine and CUT the "face" option.
- **Labels:** add a registry `vram_tier_label` metadata prop + append it in otr_video_director `_label_for`
  (today L113 only appends aspect). Ground the number to the TRUE single-resident peak (registry 7000MB,
  NOT the stale 3.3GB comment). Auto-derived; no custom label string.
- **1.7B vs 14B:** node 87 currently selects `humo_14B_169`. This sprint = 1.7B quality + labels for BOTH
  tiers; do NOT silently demote the workflow default (operator-gated). Split "1.7B quality" from "14B
  option clarity".
- **Isolation smoke = PREREQUISITE for the portrait knob A/B** (not for the guard/label/clip-fill fixes).
  SPLIT: ship the deterministic fixes first; gate subjective portrait tuning behind building the smoke.

## r1-ONLY sufficiency: the deterministic items are build-ready from r1. Phrase-chunking is the one piece
that could use an r2 coding pass -- flag it to the coder; the rest is coder-ready.
