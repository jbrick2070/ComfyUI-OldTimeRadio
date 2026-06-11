# Pass 02 judgment -- look-QA round 5 (3-model panel, ~$0.03)

Panel: gpt-5.5 (no), gemini-3.1-pro (yes-with-fixes), deepseek-v4-pro
(yes-with-fixes). All claims grounded against render_driver.py,
otr_silent_composite.py, otr_shot_lock.py, eng_ltx_video.py + live-ledger probes.

## ACCEPTED (grounded CONFIRMED)

1. **_env_int NameError** (all three): the helper does not exist in
   eng_ltx_video.py -- write it (parse os.environ, default 121, clamp to
   _LTX_MIN_FRAMES, LOUD on invalid). Trivial but build-fatal as drafted.
2. **Diversity gate had no data path** (GPT-2): run_episode's `trace` rows carry
   only shot_id/attempts/final_engine (verified in tonight's /history node-92
   report). Fix: text-engine trace rows gain `prompt_sha8`, `prompt_source`,
   `prompt_chars`; the diversity check reads the trace (durable in /history).
3. **Synthetic b000 silently kills positioned timeline** (GPT-3) -- CONFIRMED
   sharper than claimed: `build_clip_manifest` (render_driver L771-777) resolves
   start_s from LINES only -> b000 row start_s=None; `plan_timeline_segments`
   (otr_silent_composite L222-223) requires ALL rows positioned -> tonight's
   episode fell back to SEQUENTIAL assembly. It LOOKED right only because the
   beats are contiguous and b000 is order-0. Any real inter-beat gap would
   mis-place everything after it. Fix: manifest start_s falls back to
   `shot.get("start_s")` (the 435ba0a synthetic stamp).
4. **Shot-row char_id is ignored by the driver** (GPT-4, Gem-2): driver L350-352
   reads the LINE only. Fix: ShotLock `build_execution_plan` stamps
   `"char_id": b["char_id"]` on shot rows; driver resolves
   `shot.char_id or line.char_id` (shot first -- it carries the normalized
   announcer id); the missing-portrait warning fires on the RESOLVED value.
5. **Manifest rows lack char_id/init for the face-acceptance check** (GPT-5):
   add `char_id` + `init_image` to manifest rows (pure function, cheap).
6. **Anchor must cover ALL M4 prompt paths** (GPT-6/7): one `subject_anchor`
   prepended before the consistency guard for every path (llm_text L483,
   composed L486, deterministic L494); `_prompt_is_consistent` stays
   role-blind -- it is only called inside the char_beats loop (document that;
   Gem optional folded as the docstring note).
7. **Ambiguity wording conflict** (GPT-8): acceptance re-scoped -- no UNRESOLVED
   self-vocative ships from a 2-character cast; 3+-character ambiguity is
   LOUD-logged and ships (operator eyeball remains final).
8. **Re-attribution must move every speaker-identity field and run BEFORE
   casting/voice mint** (GPT-9, Gem S-1): build-step verification item -- lines
   carry char_id + speaker_role (probe); the repair updates both and runs
   pre-casting so the minted voice matches the corrected speaker.
9. **Synthetic-open detector** (GPT S-1): empty `source_line_ids` OR the
   OPENING_MUSIC_BEAT_ID suffix -- not role alone. **Log beat ids via
   `_beat_id_for_shot`** (GPT S-2). **Env-override prompts exempt the diversity
   gate (warn-only)** (GPT S-3). **cap=120 -> snap 113 test** (GPT S-6).
   **Anchor tokens near the prompt head** (GPT S-7, first 160 chars).
10. **YAVG stddev becomes a diagnostic, not a hard gate** (GPT S-5/cut-2, DS
    optional): the enforced gates are the cap log line, prompt-sha diversity,
    per-beat face spot-frames, and the operator eyeball.
11. **Unknown beat_intent fallback** (DS S-1): unmapped intent -> "a beat of
    {intent}" + one INFO line, never silent skip. **Portrait warning gated to
    talking-head engine family** (DS S-3 + Gem S-2: `audio_driven_face`).

## REJECTED (with reason)

- **DS S-2 false-positive heuristic for "John, I think..."**: MISREAD -- the
  detector is speaker-conditioned (fires only when the vocative equals the
  SPEAKER's own name); "John, ..." spoken by Mary never matches. No verb
  heuristic needed.
- **Wan cap this round** (GPT cut-1 vs DS-1): CUT -- the wan lane is
  operator-gated and ungrounded here; left as a one-line verify note, no code.
- **Dropping the unpushed-commits line** (GPT cut-3): kept, relabeled "release
  procedure" -- it is the operator's standing rule for this build, not a code
  invariant.

## VERIFY-AT-BUILD (carried)

- Writer exchange structure for the interlocutor rule: the CAST table is the
  data source (exactly-2-character cast -> the other one); no scene table needed
  (DS-3 resolved).
- beat_intent/arc_phase presence on fresh episodes (DS-2): degrade-silent +
  fallback clause covers absence; fixture tests pin both directions.
- eng_wan_i2v ask path (GPT S-4): inspect only, no speculative cap.
- HuMo text-dominance empirics: the re-render is the test.

## Convergence

Pass02 items are spec-precision (helper, data paths, wording) -- no architecture
changes. Folded into pass02_plan.md (FINAL). Pass03 = convergence check.
