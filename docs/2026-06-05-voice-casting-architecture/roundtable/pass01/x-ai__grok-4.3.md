<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Mismatches between cast_lock.py auto_registry, _otr_voice_node_common.py resolver, bank commercial_clean flags, and adapter declarations will produce silent bark fallbacks + non-deterministic rates even when refs exist on disk.

MUST-FIX BEFORE BUILD:
1. [cast_lock.py:280] _resolve_char_engine returns "indextts2" (via legacy_first_engines) but IndexTTS2Engine.commercial_clean=False while bank entries stamp true; fix: make CastLock read the adapter's commercial_clean attr (or centralize the flag in the bank schema) and gate the stamp.
2. [cast_lock.py:240] default cast_voice_policy="preserve_ledger" + _auto_registry only runs on explicit "auto_registry"; with IndexTTS2 now default this yields the observed None->bark path in _resolve_clone_ref_path; fix: change default to "auto_registry" when target_engine in _OTR_CLONE_ENGINES.
3. [_otr_voice_node_common.py:312] _resolve_clone_ref_path and line 340 both do cast.get("gender") with no population guarantee; fix: require CastLock._stamp (or _assign_bark_voices) to always write "gender" from the ledger cast row.
4. [eng_indextts2.py:30] commercial_clean=False hard-coded on adapter while bank vz_* rows have true and the release gate reads the bank; fix: delete the adapter flag or make it read-only from the resolved VoiceBankEntry.
5. [_otr_voice_node_common.py:220] ref_field dispatch only handles "voice_ref_path" vs. anything-else; kokoro's "voice_ref_id" path is untested for char_voice; fix: add explicit test that voice_ref_id reaches generate_voice for kokoro.

SHOULD-FIX:
1. [Question 3 / _otr_voice_bank.py:140] _LADDER hard-codes gender as floor with no "unspecified" tier; add a final tier that accepts gender="" when bank has an "unspecified" entry so non-binary slots don't immediately VoiceCastingError.
2. [_otr_voice_node_common.py:370] resample_audio only called on bark fallback; a future 16000 Hz engine would still hit pack_audio_batch with mixed rates inside the primary loop; add a post-generate resample to adapter.sample_rate for every clip.
3. [cast_lock.py:310] announcer_voice_ref pins only on lowest voice_ref_id with no engine-specific announcer_voice_ref override; document the kokoro begin_episode path that bypasses it.

OPTIONAL / NICE-TO-HAVE:
- Emit structured WARNING from assign_voice_for_slot when allow_voice_reuse=True causes reuse (Question 8).
- Add voice_bank_entry_schema.json "gender" enum including "unspecified".

CUT THESE (over-engineering):
- [Question 2] The mixed voice_ref_field contract is fine; collapsing everything to a single canonical voice_ref_id + per-adapter resolver adds indirection with no determinism gain once CastLock always stamps.
- [Question 1] The four-field "add engine" checklist (bank tag + voice_ref_field + generate_voice + sample_rate) is already minimal; no need for extra interface methods.

[ASSUMPTION] All claims above verified only against the five provided grounding files; no other source files were inspected.