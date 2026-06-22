<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan is still a decision memo, not a buildable architecture; the core “LLM-best casting” goal conflicts with current ownership of gender/voice, CastLock has no LLM surface, and the identity contract is ambiguous between `voice_ref_id` and `voice_preset`.

MUST-FIX BEFORE BUILD:
1. [Q1 Casting intelligence] The plan asks “pure-LLM vs deterministic vs HYBRID” but never chooses an architecture. This is not build-ready because every downstream choice depends on it: where the LLM runs, what it sees, what it stamps, and how fallback works. Concrete fix: choose one target. Smallest coherent target: HYBRID — LLM proposes a ranked list of `voice_ref_id`s from the selected engine’s voice cards; Python validates gender/engine/availability/no-reuse and falls closed to `_otr_voice_bank.assign_voice_for_slot`.

2. [Q1 Casting intelligence] Literal “LLM chooses best voice from the selected engine’s library” cannot currently run at CastLock without new wiring: `CastLock.INPUT_TYPES` has `script_json`, `voice_bank`, `cast_voice_policy`, `delivery_profile`, `allow_voice_reuse`, `gate_in`; it has no `generate_fn`, LLM handle, model input, or prompt slot. Concrete fix: either (a) add a dedicated local LLM input/node before or inside CastLock and update `otr_scifi_16gb_full.json` in the same change, or (b) revise the goal to “LLM-informed tags, deterministic voice choice” and fold tag extraction into the existing writer description call.

3. [Grounded facts C] + [_otr_casting module doc] The document says “The LLM picks the CAST (names + gender + description),” but the grounded code says the opposite: `_otr_casting` Sprint 3D explicitly moved gender and voice out of the LLM; `precompute_ensemble_slots` decides gender/timbre/role in Python, and `llm_write_description` writes description only. Concrete fix: correct the ownership model. If Python owns gender, the LLM must not be described as choosing “right GENDER.” If the LLM is allowed to challenge gender, specify the deterministic validator and how it preserves ensemble balance.

4. [Q1 Casting intelligence] + [_otr_casting.lock_cast] + [cast_lock._auto_registry] The plan wants voice selection based on age/persona/register, but the cast rows available to CastLock do not carry the necessary structured slot data. In `lock_cast`, open rows stamp `gender`, `character_description`, and `speech_signature`, but not `timbre`, `role`, or `age_band`; yet `cast_lock._auto_registry` calls `entry.get("timbre")` and `entry.get("age_band")`. Concrete fix: stamp voice-fit inputs in frozen-safe metadata, e.g. `meta.cast_voice_slots[char_id] = {gender, timbre, role, age_band, speech_signature, character_description_digest}`, and make CastLock read that instead of missing cast fields.

5. [Q4 Engine-agnostic identity] The identity contract is internally confused. The prose says `voice_preset` is the universal ID every adapter maps, but the cloner path actually works through `voice_ref_id`/bank refs: `cast_lock._stamp` writes `voice_ref_id` and `voice_engine`; `_otr_voice_node_common._resolve_clone_ref_path` resolves clone refs from `voice_ref_id` or gender, not from `voice_preset`. Concrete fix: define two lanes explicitly: `voice_ref_id` is the primary identity for bank/cloner engines; `voice_preset` is the Bark/universal fallback identity. The LLM must choose `voice_ref_id` for the selected engine, never a character name; Python stamps/keeps `voice_preset` only as fallback.

6. [Q2 Library solidity] “137 refs” and per-engine gender totals are not a solidity bar. The plan needs a pass/fail approval rule before implementation. Concrete fix: add CI/approval gates over the loaded bank: minimum non-reject, on-disk refs per `(engine, gender, age_band)`; minimum unique voices for worst-case 5-character no-reuse casts; at least one announcer ref for the active announcer engine; and a male-light remediation threshold. Without this, “approved voice model” is a subjective label.

7. [Q2 Library solidity] + [_otr_voice_bank.assign_voice_for_slot] The plan does not handle `gender="other"` coherently. `_otr_casting` permits `other`; the grounded bank summary only gives male/female counts; `_otr_voice_bank.assign_voice_for_slot` has a hard gender floor and no gender match means no castable ref. That means “right gender + voice” silently degrades for `other` characters. Concrete fix: either add `other`/androgynous bank entries and coverage bars, or define an explicit deterministic `other`-to-voice policy with loud report metadata.

8. [Q3 Robustness] The proposed critic/freeze gate for stage-direction-only lines risks contradicting the invariant “fail-soft / audio is king.” `_otr_voice_node_common._render_per_line` already emits 0.30s silence and continues. Concrete fix: make stage-direction-only detection a non-blocking mechanical diagnostic in meta/report by default; only halt under an explicit strict QA flag. Do not route ordinary renderability through a subjective critic freeze halt.

9. [Invariants] + [Q1 Casting intelligence] Deterministic LLM voice choice is underspecified. A local LLM call is not automatically reproducible unless the prompt, model ID/version, decoding params, bank SHA, candidate ordering, and retry behavior are stamped. Concrete fix: define `meta.voice_cast_decision = {policy_version, bank_sha, engine, model_id, prompt_version, seed, candidate_ids, proposed_ids, accepted_id, fallback_reason}` and make Python validation the reproducibility boundary.

SHOULD-FIX:
1. [cast_lock._auto_registry] Runtime fail-soft is weaker than the prose claims. `load_voice_bank()` is called without a local catch in `_auto_registry`; malformed/missing bank JSON can abort CastLock before the later fail-soft voice repair. Concrete fix: decide the boundary: CI fails hard on malformed approved banks, but runtime falls back to `preserve_ledger`/Bark identities with a loud report if bank load fails.

2. [Q2 Library solidity] “Distinct casts per episode” needs an anthology-level anti-reuse policy, not only within-episode no-collision. Current `used` in `_auto_registry` is per cast only. Concrete fix: add an optional deterministic episode-history salt or recent-voice exclusion set in metadata/config, or explicitly state that cross-episode reuse is allowed.

3. [Q1 Casting intelligence] The LLM cannot make an acoustic “best” choice from IDs alone. [ASSUMPTION] Unless the local LLM can hear reference clips, it will only rank text tags. Concrete fix: create voice cards from bank metadata: `voice_ref_id`, gender, age_band, timbre, roles, quality_tier, style_tags, short curated description, and commercial flag. Do not expose file paths or character names as decision anchors.

4. [Q4 Engine-agnostic identity] Bark is outside the bank but is also used as the universal fallback namespace. That makes “coherent with cloner banks” impossible unless there is a mapping policy. Concrete fix: add deterministic fallback mapping from each accepted `voice_ref_id` to a same-gender `v2/en_speaker_*`, or state that Bark fallback is only gender-safe, not acoustically equivalent.

5. [Q3 Robustness] The silence guard is per-line and engine-agnostic, but the plan should name the expected user-facing behavior: the line remains in timing with 0.30s silence and logs a P-OBS warning. Concrete fix: add this as an acceptance test across character and announcer lanes.

6. [Invariants] “Ledger wire format frozen; new fields ride free-form meta” conflicts with existing code reading optional cast-row fields like `timbre` and `age_band` in `cast_lock._auto_registry`. Concrete fix: standardize all new casting intelligence fields under `meta`, and only mirror to cast rows if already tolerated by consumers.

OPTIONAL / NICE-TO-HAVE:
- Add an offline “casting audition report” artifact listing each character, description, candidate voices, accepted voice, fallback reason, and collision decisions.
- Add a seed-matrix simulation that proves no no-reuse failures for 3–5 character casts under the approved engines.
- Add per-engine male/female/other dashboards generated from `voice_reference_bank.json`.

CUT THESE (scope / over-engineering):
1. [Q1 Casting intelligence] Cut pure-LLM voice assignment. It cannot satisfy determinism, no-collision, engine availability, commercial-clean gating, or fail-soft requirements without Python validation; HYBRID covers the goal safely.

2. [Q3 Robustness] Cut default freeze-halting for stage-direction-only lines. The shipped two-layer net already protects audio; halting belongs behind a strict QA flag, not the default render path.

3. [_otr_casting._build_user_prompt] Cut visual/portrait requirements from this voice-casting build scope. The `CHARACTER VISUAL CONTRACT` may serve another subsystem, but it does not help voice selection and bloats the casting prompt surface for this workstream.

4. [_otr_casting._apply_llm_slot_fill] Cut LLM naming overlay from this voice-casting architecture decision. It is orthogonal to selecting solid voices and adds another LLM behavior surface that can mask whether voice casting itself works.

5. [Q4 Engine-agnostic identity] Cut any attempt to make `voice_preset` the single universal identity for all engines. The grounded cloner path already uses `voice_ref_id`; forcing one ID field across Bark presets and ref-WAV cloners will keep producing ambiguous fallback behavior.