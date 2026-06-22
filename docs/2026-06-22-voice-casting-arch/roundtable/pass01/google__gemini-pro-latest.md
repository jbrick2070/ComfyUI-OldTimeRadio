<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The codebase explicitly blocks the primary operator directive (Goal C) by hardcoding voice and gender selection in pure Python, and the "engine-agnostic" fallback identity is an unmapped Bark-specific string that will force clone engines to degrade to Bark.

MUST-FIX BEFORE BUILD:
1. [Narrative / arc coherence - Goal C vs `_otr_casting.py`] The operator directive (2026-06-22) demands the LLM make the "BEST casting call -- the right GENDER + VOICE". However, `_otr_casting.py` (Sprint 3D) explicitly boasts about doing the exact opposite: "Sprint 3D moves balance and voice selection out of the LLM... precompute_ensemble_slots -- PURE PYTHON". You cannot fulfill the directive without reverting this.
   *Fix:* Implement a HYBRID approach in `cast_one_character`. Pass the available bank's semantic traits (gender, timbre, age) to the LLM. The LLM proposes the ideal traits for the character description it just wrote. Python then deterministically scores the pre-filtered `available_voices` against the LLM's proposed traits (falling back to the seed-keyed scorer on a miss).

2. [Missing pieces / Correctness - `CastLock._fallback_voice_identity` vs `_otr_voice_node_common.py`] `CastLock` assigns `v2/en_speaker_*` to orphans, claiming it is an "engine-agnostic identity every approved adapter maps from". But `_resolve_clone_ref_path` only looks up by `voice_ref_id` or gender; it contains no logic to map Bark presets to native clone references. Consequently, any fallback row sent to a clone engine (IndexTTS2/Chatterbox) will fail to find a reference and hard-drop to the Bark fallback engine.
   *Fix:* Define a strict mapping matrix in `_otr_voice_bank.py` that maps the 10 `v2/en_speaker_*` identities to 10 guaranteed `voice_ref_id`s in every approved engine's library. Update `_resolve_clone_ref_path` to query this map before falling back to gender-random.

3. [Robustness - `_otr_voice_node_common._render_per_line`] Emitting a 0.30s silence for stage-direction-only lines prevents a TTS crash, but it masks a structural LLM hallucination (outputting stage directions as dialogue). 
   *Fix:* The empty-content check must also exist upstream in the freeze/critic cascade. If `clean_spoken_text` empties a line, the critic must flag it as `freeze_verdict='needs_full_rerun'` (mechanical defect) so the writer fixes it, rather than shipping an episode with missing dialogue replaced by dead air.

SHOULD-FIX:
1. [Library solidity (B) - `_otr_voice_bank.py`] 14-22 voices per gender per engine is too small to avoid collisions across a 5-character cast without heavy reuse, especially when filtered by age/timbre. [ASSUMPTION] The operator assumes 137 refs is enough for an anthology, but fragmentation across 4 engines means the actual pool per episode is dangerously shallow.
   *Fix:* Enforce a minimum coverage bar in `load_voice_bank`: reject any engine bank that does not have at least 3 distinct voices per (gender x age_band) matrix. 

2. [Engine-agnostic identity - `_otr_casting.py`] `CastingResponse` enforces `voice_preset` max_length=80. If an LLM or auto-registry selects a deeply nested `voice_ref_id` path for a local clone model, it may easily exceed 80 characters.
   *Fix:* Increase `voice_preset` and `voice_ref_id` max_length limits to 255 to accommodate absolute file paths or verbose UUIDs.

CUT THESE (scope / over-engineering):
1. `_apply_llm_slot_fill` and `_build_pass1_prompt` in `_otr_casting.py`. This requires an entirely separate, second LLM call just to rename characters and add "texture notes" after Python already rolled names. It violates the "one LLM call per character" constraint and adds massive failure surface. Cut it; let the primary description LLM call handle naming.
2. `diversify_speech_signatures` in `_otr_casting.py`. It blindly overwrites speech signatures with a round-robin list ("clipped and terse", "warm and rambling") if there is a collision. This completely destroys the LLM's creative intent for the character. Cut it and handle uniqueness via the LLM prompt.

[ASSUMPTION] The architecture assumes that `stereo_policy="mono_safe"` (naive downmixing) will not cause phase cancellation or audio artifacts with all future TTS engines.
[ASSUMPTION] The plan assumes the LLM can consistently output valid JSON matching `DescriptionResponse` without a fallback regex parser, relying entirely on `_otr_json.parse_first_json_object`.