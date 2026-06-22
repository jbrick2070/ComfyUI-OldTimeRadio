# Voice-casting architecture -- CONVERGED design (roundtable R1)

Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude code-grounded judge.
R1 spend ~$0.21. All three returned "no -- resolve the open questions"; the judge's
job is to RESOLVE them, which this doc does. Every decision is grounded against the
real source; two panel claims were verified + corrected below.

## Grounding correction (the panel caught my problem-statement error)
- **Gender + voice are PURE PYTHON, not LLM (Sprint 3D).** `_otr_casting.py` L14-26:
  `precompute_ensemble_slots` (Python) owns the gender/timbre/role distribution;
  `llm_write_description` writes ONLY the prose; `python_assign_voice_preset` (Python)
  picks the voice. So "the LLM picks gender" was WRONG. The operator's directive =
  ADD LLM voice-FIT back WITHOUT losing the deterministic ensemble BALANCE.
- **My STEP 3 `v2/*` fallback does NOT break clone engines (but is imprecise).**
  `_otr_voice_node_common._resolve_clone_ref_path` resolves a clone voice by
  `voice_ref_id` -> gender -> ANY-ref-for-engine (L116-130, "a clone engine must
  still get a REAL voice rather than silently dropping to bark"). It never reads
  `voice_preset`. So a STEP-3-repaired row still gets a real clone voice -- but the
  identity contract is genuinely TWO-LANE (below), not "voice_preset is universal".

## DECISION C -- casting intelligence = HYBRID (LLM proposes, Python disposes)
Keep `precompute_ensemble_slots` UNCHANGED (Python ensemble BALANCE -> the gender
mix the operator wants is preserved + deterministic). Add LLM voice-FIT at the
EXISTING per-character `llm_write_description` call (NO new LLM call):
- Hand the LLM the selected engine's VOICE CARDS for the character's PRECOMPUTED
  gender slot: {voice_ref_id, age_band, timbre, role tags, style tags, short curated
  description, commercial flag} -- NO file paths, NO character names (I-9).
- The LLM returns a ranked best-fit `voice_ref_id` (within its gender slot).
- `python_assign_voice_preset` becomes a VALIDATOR: accept the LLM's id iff
  in-library + engine-correct + gender-consistent + non-colliding; else FALL CLOSED
  to the deterministic `assign_voice_for_slot` scorer. Pure-LLM assignment is CUT
  (can't meet determinism / no-collision / commercial-clean alone).
- Reproducibility stamp: `meta.voice_cast_decision[char_id] = {policy_version,
  bank_sha, engine, model_id, prompt_version, seed, candidate_ids, proposed_id,
  accepted_id, fallback_reason}`. Python validation is the reproducibility boundary;
  a no-LLM / LLM-failure run is byte-identical to today.

## DECISION Q4 -- identity = TWO explicit LANES
- `voice_ref_id` = PRIMARY identity for bank/cloner engines (indextts2 / chatterbox
  / dia / kokoro). `voice_preset` = bark / universal FALLBACK identity.
- The caster/LLM chooses a `voice_ref_id` from the ACTIVE engine for cloners; bark
  uses a `v2/*` preset. REFINE STEP 3: a repaired character row stamps a real
  `voice_ref_id` from the active engine's bank (not only a bark preset).
- Add a deterministic `v2/en_speaker_* <-> same-gender voice_ref_id` map so a bark-
  fallback identity always resolves to a same-gender clone ref (closes the Gemini
  "fallback degrades to bark" concern at the contract level).
- Raise the `voice_preset` / `voice_ref_id` max_length (80 -> 255) for verbose ids.

## DECISION B -- library solidity = coverage bar + remediation
- Deterministic CI/test gate over the loaded bank (config/voice_reference_bank.json,
  137 refs today): each APPROVED engine must have >= 3 distinct voices per
  (gender x age_band), >= 1 announcer ref, and enough unique voices for a worst-case
  5-character no-reuse cast. Fail the gate -> the engine is not "approved".
- Remediate the MALE-LIGHT imbalance (cloners ~14-15 M vs ~22 F) by adding male refs.
- `gender="other"` policy: `_otr_casting` permits it but the bank is male/female only
  -> add androgynous bank entries OR a deterministic other->voice rule with a LOUD
  report (today it silently degrades).
- Anthology distinctness: an OPTIONAL deterministic recent-voice-exclusion salt for
  cross-episode variety, OR explicitly state cross-episode reuse is allowed.

## DECISION A -- robustness = keep the net + NON-BLOCKING diagnostic
Keep the shipped two-layer net (spine recompose -> per-line 0.30s silence). Resolve
the Gemini(halt)-vs-GPT(don't-halt) conflict toward PD1: a stage-direction-only line
is a NON-BLOCKING mechanical DIAGNOSTIC counted in
`meta.delivery_hygiene_report` (already added) -- NOT a freeze halt by default. The
spine RECOMPOSE is the "writer fixes it" path; the silence is the last resort. A
strict-QA env flag MAY opt into a halt. Add an acceptance test (character + announcer
lanes: a stage-direction-only line -> recompose or 0.30s silence, P-OBS warning, no
crash, line stays in timing).

## DECISION (enabling) -- stamp the voice-fit inputs CastLock needs
`cast_lock._auto_registry` already reads `entry.get("timbre")` / `age_band`, but
`lock_cast` stamps only gender / description / speech_signature. Stamp
`meta.cast_voice_slots[char_id] = {gender, timbre, role, age_band, speech_signature,
description_digest}` (frozen-safe meta) and have CastLock read THAT.

## Build order (R2 coding -> R3 wiring -> R4 when the operator builds)
1. Two-lane identity + STEP-3 refine (voice_ref_id for cloners) + v2<->ref map.
2. `meta.cast_voice_slots` stamp + CastLock reads it.
3. Library coverage gate (test) + male-light/`other` remediation.
4. HYBRID LLM voice-fit folded into `llm_write_description` + the validator +
   `meta.voice_cast_decision`. (Default-on, $0 deterministic fallback.)
5. Robustness acceptance test + the non-blocking diagnostic count.

## CUT / verify-at-build
- Pure-LLM voice assignment -- CUT (HYBRID covers it safely).
- Default freeze-halt for stage-direction-only lines -- CUT (PD1; behind a QA flag).
- `_apply_llm_slot_fill` (a 2nd LLM naming call) + `diversify_speech_signatures`
  (overwrites LLM intent on a collision) -- panel flagged both as over-engineering;
  VERIFY-AT-BUILD whether to cut (orthogonal cleanup, not a blocker for this build).
- Visual/portrait contract in the casting prompt -- out of voice scope.

## Invariants honored
Deterministic + seed-keyed (C7). 100% local. Fail-soft (audio is king). Ledger
{cast,lines,meta} frozen (all new casting fields ride free-form meta). Identity is
voice_ref_id / voice_preset, never the character name (I-9). Suite + Bug Bible green
per chunk. Default-ON in otr_scifi_16gb_full.json; any wiring goes IN that JSON.
