# R1 Claude Anchor Review — model-selection coherence

VERDICT: The model-selection arc is sound and grounded. Two decisions are
firmly supportable from the code; three need explicit operator calls. No
model choice here breaks an invariant.

## MUST-FIX (model-selection level)
1. **Collapse flash/tts into ONE engine (CONFIRMED).** partner_nodes.yaml shows
   both rows are class `ElevenLabsTextToSpeech`, identical input sets, and
   PRICING.md confirms price is flat and tier is quality-only. Registering two
   OTR engines for one node = drift risk (BUG-LOCAL-097 class) and a confusing
   dropdown. One `elevenlabs` engine with `model` (the DYNAMICCOMBO tier) as a
   default_param is the correct shape. MUST resolve the V3 DYNAMICCOMBO
   expansion at pin time to enumerate the real `model` tier options — the static
   pin is shallow (CONFIRMED: `model: COMFY_DYNAMICCOMBO_V3`).
2. **Delivery mapping must target `stability` + `seed` ONLY (CONFIRMED).** The
   pinned node exposes `stability` FLOAT and `seed` INT and nothing else
   expressive. Any scope language promising a similarity_boost/style/speed
   mapping is writing against inputs that do not exist on the pinned row. Scope
   the delivery vector -> `stability` (and byte-stable = high stability, fixed
   seed). Re-confirm at V3 expansion whether more knobs hide in the combo.

## SHOULD-FIX
3. **Voice pool must be a checked-in manifest (CONFIRMED path exists).**
   voice_reference_bank.json + the entry schema already exist and CastLock's
   scorer keys on episode_seed; an ElevenLabs entry drops in with voice_id in
   the `ref_path` slot. This is the determinism answer — a curated pool, not the
   live library. Low risk; the schema is already there.
4. **Sonilo = music v1 default (CONFIRMED on cost + inputs).** Cheaper, native
   `duration`, marked BEST. Stability-audio as the selectable alt. Solid.

## UNVERIFIABLE (verify-at-build)
- Whether the ElevenLabs `model` DYNAMICCOMBO actually contains distinct
  quality tiers vs just voice-model variants — needs the live V3 expansion
  (same GOTCHA flagged for seedance in the cloud-engines memory).
- Whether Sonilo honors a short (<10s) duration precisely, or clamps — provider
  behavior, confirm at first live cue.
- ToS/license cleanliness of specific ElevenLabs premade voices for commercial
  use — operator review item, not code.

## Invariant check
- Audio SPINE frozen: both lanes produce per-line WAV / cue AUDIO consumed by
  the SAME assembler; no change to master mix / mux-LAST. OK.
- No fallback: dropdown-is-enable + fail-loud is consistent with the existing
  registry-IS-the-menu design. OK.
- Determinism: curated pool + OTR_CAST_SEED reproduces; music seed where
  supported. OK.
