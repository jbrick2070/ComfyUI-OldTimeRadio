# R1 judgment -- voice-casting architecture

## Convergence: the architecture is RESOLVED (panel said "open questions"; the
## judge resolved them into a concrete, grounded, build-ready design).

## Accepted (grounded CONFIRMED)
- HYBRID casting (LLM proposes a voice_ref_id from the engine's voice cards;
  Python validates + falls closed to the seeded scorer). [GPT, Gemini, DeepSeek, anchor]
- Two-lane identity: voice_ref_id (cloners) / voice_preset (bark fallback). CONFIRMED
  vs _resolve_clone_ref_path (uses voice_ref_id/gender, never voice_preset). [GPT#5, Gemini#2]
- Gender/voice are PURE PYTHON (Sprint 3D) -- my problem statement was wrong;
  corrected. Keep precompute_ensemble_slots (balance); add LLM fit. CONFIRMED vs
  _otr_casting L14-26. [GPT#3, Gemini#1]
- Library coverage gate + male-light/`other` remediation. [GPT#6/#7, Gemini SHOULD#1]
- meta.cast_voice_slots stamp (CastLock reads timbre/age_band that lock_cast doesn't
  stamp). CONFIRMED vs _auto_registry entry.get('timbre'). [GPT#4]
- Reproducibility stamp meta.voice_cast_decision. [GPT#9]
- v2<->ref same-gender map so a bark-fallback identity resolves on a cloner. [Gemini#2]

## Rejected / cut
- Pure-LLM voice assignment (can't meet determinism/no-collision/commercial-clean). [all]
- Default freeze-halt for stage-direction-only lines -- conflict resolved toward PD1
  (non-blocking diagnostic; QA-flag opt-in). [GPT#8 over Gemini#3]
- "voice_preset is the single universal identity" -- two lanes instead. [GPT#5]

## Verify-at-build (panel claims downgraded / to confirm in code)
- My STEP 3 v2/* fallback does NOT silently drop a cloner to bark (the gender-agnostic
  last resort catches it) -- the refine (stamp voice_ref_id) is an improvement, not a
  bug fix. CONFIRMED vs _resolve_clone_ref_path L116-130.
- _apply_llm_slot_fill + diversify_speech_signatures cuts -- pre-existing + orthogonal;
  confirm at build whether to remove (not a blocker).
- voice_preset max_length=80 -> 255 (Gemini SHOULD#2) -- confirm the field + consumers.

## Panel mechanics
All three answered at --max-tokens 12000 (GPT 5607 / Gemini 3908 / DeepSeek 2925 out
tokens). No truncation. Spend ~$0.21.

## NEXT
The architecture (pass01_plan.md) is build-ready as a 5-chunk plan. R2 (coding) ->
R3 (wiring) -> R4 (convergence) + the build are a large follow-on the operator
schedules. This is a NEW workstream, distinct from the now-complete STORY+CAST FIX.
