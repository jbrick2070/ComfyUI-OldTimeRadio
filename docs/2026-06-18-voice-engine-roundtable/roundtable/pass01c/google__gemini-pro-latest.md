<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The proposed multi-engine cast mix (Chatterbox + Kokoro) is structurally blocked by the current CastLock implementation, and Kokoro violates the offline invariant.

MUST-FIX BEFORE BUILD:
1. [eng_kokoro.py / Invariants] Offline violation. `load()` initializes `KPipeline(..., repo_id="hexgrad/Kokoro-82M")`. If the model JSON/weights are not cached in the HuggingFace cache, this attempts a network fetch mid-render, violating the "100% local/offline" invariant.
   Fix: Download the base model to `_kokoro_model_dir()` and point `KPipeline` to that local path instead of a Hub repo ID.
2. [cast_lock.py / The decision] Engine monoculture blocks the Chatterbox+Kokoro pairing. `_auto_registry` resolves exactly ONE `target_engine` for the entire cast via `_resolve_char_engine(voice_bank, bank_entries)`. You cannot mix Chatterbox for main characters and Kokoro for bit parts; the entire cast gets forced into whichever engine resolves first.
   Fix: Modify `_auto_registry` to resolve the engine per-character (e.g., based on a bit-part tag or tier) or allow `assign_voice_for_slot` to select the engine dynamically from the bank.
3. [eng_kokoro.py / Candidate engines] Kokoro `begin_episode` hardcodes announcer logic. If Kokoro is promoted to `char_voice`, `begin_episode` will still execute `_pick_announcer_voice(episode_seed)` and raise `EngineUnusable(..., "announcer_voice", ...)` if the announcer file is missing, even if Kokoro is only being used for characters.
   Fix: Gate the announcer preflight in `begin_episode` to only run if `"announcer_voice"` is actually present in the episode's cast/meta.

SHOULD-FIX:
1. [eng_kokoro.py / Candidate engines] Inappropriate fallback. In `generate_voice`, if a `voice_ref` is missing its `.pt` file, it falls back to `_episode_voice` or `_pick_announcer_voice("")`. If Kokoro is rendering a character line, falling back to the authoritative BBC announcer voice destroys the scene.
   Fix: If the role is `char_voice`, fallback to a default character preset or fail loudly.
2. [cast_lock.py / OTR context] Bark replay silently overwrites manual assignments. `_assign_bark_voices` does `row["voice_preset"] = voices[cid]`. If a writer manually stamped a `voice_preset` to override the seed, CastLock destroys it.
   Fix: Change to `if not row.get("voice_preset"): row["voice_preset"] = voices[cid]`.
3. [eng_indextts2.py / Invariants] Unsafe temp file handling. `generate_voice` uses `tempfile.mktemp()`, which is deprecated and vulnerable to race conditions, and `_load_wav` blindly does `os.remove(path)` which can fail if the worker holds a lock.
   Fix: Use `tempfile.NamedTemporaryFile(delete=False)` and ensure robust cleanup in a `finally` block.

OPTIONAL / NICE-TO-HAVE:
- [eng_bark.py] `_resolve_stage_temps` suppresses all exceptions during profile read (`except Exception: params = {}`). This hides syntax errors in the YAML profile. Catch specifically `FileNotFoundError` or `ValueError`.
- [_otr_delivery_vector.py] The regex `[a-z']+` strips punctuation before keyword matching, meaning cues like "how dare" won't match if written as "how dare!". [ASSUMPTION: `prepare_text` doesn't strip internal punctuation].

CUT THESE (over-engineering):
1. [Open questions #2] "Mirroring `_assign_bark_voices`" for Kokoro. Do not build a second deterministic preset assigner. Kokoro voices are just `.pt` files. Register them in the standard `_otr_voice_bank` as `voice_ref_id`s. `assign_voice_for_slot` already handles deterministic, seed-keyed assignment. Treat Kokoro presets exactly like clone references.