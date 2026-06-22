# R1 CONVERGED -- capability-based routing (refined, NON-REGRESSIVE)

> **>>> OPERATOR CORRECTION (2026-06-22 -- SUPERSEDES any audio-gating framing below):** audio-in is
> NOT a role requirement. It is an OPTIONAL capability that ONLY the audio-driven specials (HuMo,
> LTX-AV) consume. A b-roll video/still model (wan_i2v, ltx_video, flux stills) needs only a
> **text_prompt** -- the still is ALWAYS DERIVED from that prompt per beat and consumed OPTIONALLY -- so
> it fits EVERY role (music + announcer included), exactly like ltx_video. Operator: "audio-in is not
> required for a video or still model; wan for the music role can just use a text prompt just like ltx
> (the non-audio-in version)."
> **=> CODE IMPLICATION (the real fix):** a b-roll engine declares `required_inputs=("text_prompt",)` +
> `optional_inputs=("init_image",)` -- NOT init_image-REQUIRED. wan_i2v's current `required_inputs =
> ("init_image",)` is the mis-declaration; align it to ltx_video's `("text_prompt",)` with the still
> optional. Then role-fit = `required_inputs <= role_available_inputs` makes EVERY b-roll engine fit
> EVERY role, and ONLY HuMo/LTX-AV (audio_ref-required) stay limited to audio-supplying roles -- BY
> CAPABILITY. The hardcoded `roles` whitelist then mostly DISAPPEARS (the "optional override" in the
> design below is a fallback for any genuine creative restriction, not the primary mechanism). This is
> the cleanest "declare capabilities once, model-agnostic downstream." R2 builds THIS.

Panel: gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro + grok-4.3 + Claude grounded. The DIRECTION
(capability routing) is endorsed; the NAIVE "drop the whitelist, pure input-subset" is REJECTED
(over-matches + ignores aspect). Converged on a refined, strictly-additive design:

## Design (R2 input)
1. **`roles` becomes an OPTIONAL strict override, not a required gate** (Gemini, the key insight). In
   `engine_fits_role`: if descriptor `roles` is explicitly set + non-empty -> ENFORCE it (keeps
   specialized engines like `station_card` from over-matching); if empty/None (wan_i2v, ltx_video) ->
   BYPASS the whitelist, route on pure capability (`required_inputs <= role_available_inputs`). Wan
   (empty roles) unblocks to every capability-compatible role incl. announcer; set-roles engines are
   UNCHANGED -> strictly additive, satisfies the non-regression bar.
2. **Decouple `default_roles` from eligibility** (unanimous). The descriptor `roles` key is most likely
   populated from `default_roles` (empty for wan) -> THAT conflation is the actual bug. `default_roles`
   = auto-default PICK only, never an eligibility gate. Confirm the descriptor-builder source (OPEN).
3. **ASPECT is the real non-input constraint** (DeepSeek/Grok/GPT). wan/ltx = wide, HuMo = portrait;
   pure input-subset ignores it. The director already derives per-role aspect downstream -- CONFIRM
   aspect compat is enforced there (and is NOT what `roles` was silently encoding); if not, add an
   explicit `supported_aspects` / role aspect. Resolve in R2/R3.
4. **Declare-once across BOTH gates.** render_driver `_assert_family_inputs_satisfiable`
   (FAMILY_REQUIRED_INPUTS, by family) must DERIVE from the engine `required_inputs` (or assert-equal in
   a test). KEEP the render gate -- it checks the actual request's token PRESENCE; role-fit checks the
   role's theoretical SUPPLY (different jobs, one source).
5. **Non-regression proof = a GENERATED test**, not a static doc: enumerate ALL registry engines x all 5
   roles; assert `before(engine_fits_role)=True => after=True`; list the additive False->True deltas
   (wan_i2v -> announcer_visual expected).
6. **Eligibility != auto-selection** (GPT/DeepSeek). A bigger eligible pool must NOT change which engine
   auto-PICKS for an existing slot. Test: default pick per existing slot unchanged; no unrequested
   substitution.
7. **Scope v1 = VIDEO engines only.** DEFER image engines to a separate build item (ungrounded blast
   radius; the live wall is video wan_i2v -> announcer).
8. Use existing token names (text_prompt / init_image / audio_ref / base_clip_ref). `required_inputs`
   IS the capability declaration -- no parallel `image_in/audio_in` vocab.

## OPEN -> R2 (coding)
- Confirm the descriptor `roles` source in `otr_video_director.py` (the real bug site).
- Aspect: enforced downstream already, or needs an explicit capability dimension?
- Can FAMILY_REQUIRED_INPUTS derive from `required_inputs` (are there multiple engines per family with
  different inputs)?
- `optional_inputs` field (engines that CAN use init_image but don't require it -> still eligible where
  it's absent).

## Invariants (carried)
Strict SUPERSET of today's routing (every current fit preserved; only additive deltas; before/after
test); no-silent-swap LOUD on a real mismatch; audio specials (HuMo/LTX-AV/character_3d) stay gated by
capability; deterministic CPU tests; UTF-8 no BOM; SFW.
