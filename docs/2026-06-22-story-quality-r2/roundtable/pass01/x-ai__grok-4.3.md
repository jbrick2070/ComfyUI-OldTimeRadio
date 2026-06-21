<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The document is a problem statement + open questions, not a concrete, buildable fix set; multiple defects in the identified code paths would still ship the exact warts described in A/B.

MUST-FIX BEFORE BUILD:
1. [A, _otr_outline.py _assemble_outline:1408-1422] music_inter beat always emits the literal placeholder intent string; this flows through stamp_dialogue_slot_ids (which only skips slot_id, not the intent) and production_ledger.init_lines_from_outline. Fix: add a `voiced=False` or `suppress_caption=True` field on Beat (or special-case speaker_role=="music_inter" in the line composer) and change the intent to "" or a non-text cue; keep the beat for timing.
2. [B, _otr_outline.py _assemble_outline:1450-1458] announcer close intent is the hardcoded summary string "Close the episode and tag the broadcast." which the announcer composer turns into thesis lines. Fix: change it to a concrete final-image instruction (or make the close beat carry an explicit `final_beat_image` field) before any prompt change.
3. [C, whole Path C flow] no per-beat enforcement of opposed-wants or stage-business ban exists in Stage 3 (_build_beat_user_prompt) or the line composer; the adjacency window only carries previous_intent, not the DramaticState opposed wants. Fix: inject the two wants + costly_choice_beat into every Stage-3 prompt (or add the cheap per-line hygiene pass) or the weak-model collapse remains.
4. [Q3, _otr_dramatic_state.py derive_dramatic_state_from_meta] the helper only populates defaults when script_brief lacks a '?'; this leaves the structural validators with no opposed-wants signal on most runs. Fix: always derive from cast + ending_change or require the news_interpreter to emit the DramaticState object.

SHOULD-FIX:
1. [_otr_outline.py:1507-1519 (old line numbers referenced in spec) + current _assemble_outline] music_inter still receives arc_phase=phase_name even though it is non-voiced; validator #5 walks only voiced beats but the field is still required, creating an unnecessary ordering constraint. Drop arc_phase for music_* beats.
2. [_otr_line_hygiene.py is_truncated + detect_narration_self_address] these run only on character lines; announcer close lines (the source of B) are excluded by design. Extend the truncation/narration scrub to announcer beats or the thesis-tag problem is untouched.
3. [OutlineRequest.__post_init__ + generate_outline Stage 2] the deterministic fallback skeleton is only used on cast drift; weak models that produce meandering stage business in Stage 3 still succeed. Add a post-Stage-3 "no stage-business" validator that forces a deterministic reroll on filler verbs.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line `in_world_cue` optional field on music_inter beats so a future in-world cue can be supplied without changing the schema.
- Make HEDGE_LIST and _UNRESOLVED_MARKERS in _otr_dramatic_state.py a shared constant imported by both the measurement scan and the F3 outro composer.

CUT THESE (over-engineering):
1. The full 1-beat adjacency window + phase_summary in every Stage-3 prompt (lines ~1280-1310); the previous_intent alone is sufficient for the opposed-wants goal and halves prompt tokens on weak models.
2. The _phase_cast_phantom_repair Levenshtein path; the deterministic round-robin fallback already guarantees cast safety, so the extra import and threshold logic can be removed without changing observable behavior.