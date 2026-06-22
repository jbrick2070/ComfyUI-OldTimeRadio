<!-- Claude code-grounded anchor review, R1 (voice-casting architecture) -->
VERDICT: yes-with-fixes. The robustness net (A) is sound + shipped; the real new
work is (C) LLM-informed casting as a HYBRID over the existing deterministic
scorer, plus a measured (B) library coverage bar. Do NOT rip out the deterministic
caster -- extend it.

MUST-FIX / KEY DECISIONS:
1. [C -- casting intelligence] HYBRID, not pure-LLM. The LLM PROPOSES a voice per
   character from the SELECTED engine's library (given the character's
   description/age/persona/gender); a deterministic validator accepts only an
   in-library, gender-consistent, non-colliding choice and FAILS CLOSED to the
   existing seeded scorer (`assign_voice_for_slot`). This mirrors the project's
   proven "LLM proposes, deterministic disposes" pattern (STEP 2) and keeps a
   no-LLM / LLM-failure run fully cast. CONFIRMED seam: the WRITER already runs the
   LLM with the cast descriptions + gender and persists `meta.cast_contract`;
   CastLock already owns voice application by replaying that contract. So the LLM
   voice PROPOSAL belongs at the writer cast-contract phase (persist a per-char
   `voice_ref_id` proposal to meta), and CastLock VALIDATES + applies (or falls back
   to the scorer) -- no new LLM call in CastLock (it's an audio node, no model).
2. [C -- determinism] The proposal must be seed-keyed reproducible: when the LLM
   slot is OFF / fails, the seeded scorer is byte-identical to today. The LLM choice
   is recorded so a replay is stable. CONFIRMED: `stable_cast_seed` + the cast_seed
   replay already give the deterministic floor.
3. [B -- library bar] Define a coverage minimum per (engine x gender x age_band) and
   MEASURE it; the bank is workable (137 refs) but male-light (~14-15 M vs ~22 F per
   cloner). A deterministic build-time check (a test) should assert each approved
   engine meets the bar so casting never starves. bark's 10 `v2/` presets are its
   library; keep them coherent with the cloner banks.
4. [A -- robustness ordering] Keep the two-layer net but ADD a named mechanical-
   defect flag for a stage-direction-only line so it is COUNTED (telemetry) and can
   drive a reroll if the recompose fails -- then the silence guard is the last
   resort, not the first. Order: spine recompose -> mechanical floor flag -> silence.
5. [4 -- identity] The LLM chooses a `voice_ref_id` (bank identity) for cloner
   engines and a `v2/` preset for bark -- i.e. it chooses from the SELECTED engine's
   library entries, not a free string. `voice_preset` stays the universal fallback
   id every adapter maps from.

SHOULD-FIX:
- Fold the voice proposal into the EXISTING cast-contract LLM pass if possible (no
  extra per-episode call / latency); make LLM-casting opt-in/default with a clean
  $0 deterministic fallback.
- Announcer stays engine-pinned (`announcer_voice_ref`) -- the LLM does not cast the
  announcer (it is the host, one voice per engine).

CUT / GUARD:
- Do NOT make casting REQUIRE an LLM call (cost/latency/offline). Deterministic
  scorer is the always-on floor.
- Do NOT let the LLM emit a raw voice name / character name as the id (I-9) -- only a
  bank voice_ref_id (or v2 preset) from the active engine.
- No SceneArcContext-style new heavy structure; ride free-form meta (frozen ledger).

[ASSUMPTION] the writer cast-contract phase can see each approved engine's library
(it needs the candidate list to let the LLM choose). VERIFY: load_voice_bank is
pure/dependency-free and importable at the writer; expose the per-engine candidate
list (id, gender, timbre, age) to the cast LLM prompt.
