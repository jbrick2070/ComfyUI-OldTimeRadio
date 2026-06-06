<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Refactor (A) + new sidecars (B/C) cannot be built from this spec without code changes that contradict the provided grounding excerpts and leave multiple C-7 / PD1 paths underspecified.

MUST-FIX BEFORE BUILD:
1. [Part A] The dispatch rewrite rule `engine in _OTR_CLONE_ENGINES -> getattr(adapter,"requires_voice_ref",False)` is incomplete. `_otr_voice_node_common.py:272` and `:292` both test the tuple (first for `_resolve_clone_ref_path`, second for the char_voice bark fallback block). The second test also hard-codes `self.ROLE == "char_voice"`. Concrete fix: replace both sites with the getattr form plus an explicit `and adapter.roles` guard so non-char roles never enter the fallback; add the three metadata attrs to `AudioEngineAdapter` in `base.py:58` with the documented defaults.
2. [Part B] `eng_chatterbox.py:58` still performs the in-process `from chatterbox.tts import ...` + `ChatterboxTTS.from_pretrained` inside `load()`. The plan requires a Path-B worker identical to `eng_indextts2.py:79` (Popen + fd dance + JSON protocol). No such worker file exists in the grounding; the current adapter must be deleted/rewritten before any chatterbox registration can pass `assert_usable`.
3. [Part C] Dia worker and adapter are specified only as prose. No `eng_dia.py`, no `_otr_dia_worker.py`, and no equivalent of `IndexTTS2Engine._resolve_ref` or `_load_wav` (soundfile path) are present. The `[S1]` transcript-prepending logic is also absent; cannot ship without the concrete files.
4. [Part D + _otr_voice_node_common.py:185] The bank-mirroring script is described but the caster path still reads `e.engine == engine` (via `assign_voice_for_slot`). Adding `cb_*` / `dia_*` rows with the same `ref_path` values will silently produce duplicate `voice_ref_id` collisions unless the mirroring script also rewrites `voice_ref_id` prefixes and the lookup in `_resolve_clone_ref_path:200` is updated to filter on both engine and prefix.
5. [Part A + C-7] The fallback line `get_engine(getattr(adapter,"missing_ref_fallback",None))` can return None. `_render_per_line:292` then does `_get_engine("bark")` unconditionally. If `missing_ref_fallback=None` on a future engine, the code still falls back; the spec must define the exact guard ("skip if None or falsy") and add it to the two sites.

SHOULD-FIX:
1. [Part C wrinkle] The "audio_prompt-only vs bark" decision for missing Dia transcripts is left as verify-at-build. This directly affects whether a missing `dia_ref_transcripts.json` entry raises (C-7) or silently degrades; the policy must be written into the adapter before the worker is coded.
2. [registry.py:140] `assert_usable` only checks `requires_flag`; the new metadata fields (`requires_voice_ref` etc.) are not validated at registration time. Add a one-line duck-type check so an adapter declaring `requires_voice_ref=True` without `voice_ref_kind` fails at import rather than at render.
3. [_otr_voice_node_common.py:310] The bark fallback path does `resample_audio` only when rates differ, but never records which engine produced the clip. Add the engine name to the log line so operators can see mixed-engine renders.

OPTIONAL / NICE-TO-HAVE:
- Make `sample_rate` on the adapter a `@property` that the worker can override in the JSON reply (Dia 44100, chatterbox dynamic) instead of a class attr.
- Add a unit-test stub for the exact fd-dup dance in the two new workers so protocol bugs are caught before any venv install.

CUT THESE (over-engineering):
1. The entire "one lazily-spawned worker per engine" warm-across-lines discussion (open question 5) -- I-7 already does `finally: unload`, so keeping the proc alive adds no new code and can be measured later.
2. `supports_external_generator` flag and all generator= kwarg handling in the chatterbox adapter -- current `eng_chatterbox.py:78` already uses `supported_kwargs` and the plan defers the bit-exact decision; remove until F pilot passes.
3. The proposed `scripts/_otr_mirror_refs.py` as a separate file -- extend the existing `otr_dl_indextts2_refs.py --engine` path instead; one script already knows the bank schema.