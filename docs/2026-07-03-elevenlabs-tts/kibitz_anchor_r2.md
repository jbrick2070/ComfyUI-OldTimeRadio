# Kibitz r2 (coding) — Claude anchor review of BUILD_PLAN.md

VERDICT: yes-with-fixes. The build contracts C1-C7 are code-grounded and the
sprint slicing is sound. Coding-level risks to pressure-test with codex:

MUST-FIX (coding):
1. **Engine-conditional schema branch (C2).** Making ref_path/ref_sha256
   conditional must not break the 51 existing local bank entries or the
   VALIDATE_INPUTS disk check for local engines. The cleanest shape: keep the
   required fields but allow a reserved cloud sentinel (e.g. ref_path=
   "cloud:elevenlabs/<voice_id>", ref_sha256="cloud") AND branch the disk-
   presence check on runtime==cloud. Confirm where VALIDATE_INPUTS lives and
   that it reads the engine/runtime at check time.
2. **invoke_partner_node audio contract (C1/S1).** Verify cloud_media_invoke
   returns AUDIO in the exact dict/tensor shape the assembler expects, and that
   the built backend already handles async classes ElevenLabsTextToSpeech /
   SoniloTextToMusic (the memory says audio dicts save via soundfile). If the
   canonicalizer must convert SR/channels, own that in one place.
3. **Admission gate placement (C4).** The gate must run AFTER
   _resolve_character_voices_fail_soft (cast_lock.py:187, unconditional) but
   BEFORE the first cloud invoke. Decide whether it's a new node in the graph or
   a guard inside the cloud adapter's assert_usable. A node is more auditable.
4. **Delivery vector -> stability mapping (S1).** Only `stability` + `seed` are
   exposed. Define the numeric map and which lines are byte-stable (high
   stability, fixed seed) vs expressive (low stability).

SHOULD-FIX:
5. Music trim must live inside the frozen assembler (C6/S5) — confirm the exact
   function and that it does not touch the ripped credits-music loop.
6. Durable stamp path (S4): confirm stamp_durable writes cast_voice_slots +
   meta.music_engine and that OTR_CreditsRoll reads them (the cast_voice_slots
   gap is real per the scoping doc).

CONFIRMED anchors: cast_lock.py:386 fail-soft; voice_bank_entry_schema required
ref_path/ref_sha256; audio_engine_profiles runtime in_graph|oop_venv + role
music singular; partner_nodes combo_options_excluded. UNVERIFIABLE: ELEVENLABS_
VOICE payload shape + selector id-vs-label (live install); Sonilo short-duration.
