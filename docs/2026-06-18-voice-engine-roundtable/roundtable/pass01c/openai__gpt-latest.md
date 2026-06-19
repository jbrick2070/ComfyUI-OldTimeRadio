<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The recommendation is not build-ready because it has unresolved product choices and proposes kokoro char_voice promotion that is not wired in the grounded code, while current CastLock still imposes bark-only invariants on all casts.

MUST-FIX BEFORE BUILD:
1. [The decision] The document says to pick “1-2 voice engines,” but [Starting recommendation] selects indextts2 + chatterbox + kokoro, with bark retained as fallback. That is not a hardened decision. Concrete fix: name the actual supported pair for this build, e.g. “non-commercial profile: indextts2 + chatterbox” or “commercial-clean profile: chatterbox + kokoro,” and move all others to fallback/future-work.

2. [Starting recommendation] The “indextts2 gives unbounded variety” claim is contradicted by grounded CastLock behavior. In `cast_lock.py`, `lock()` always calls `_assign_bark_voices(cast, meta, report)` before `_auto_registry()`, and `_assign_bark_voices()` then calls `_assert_unique_bark_voices(cast)` / `_assert_voice_preset_invariant(cast)` for every non-announcer row when `cast_seed` exists. That means clone-engine casts are still gated by bark preset uniqueness/invariants before indextts2/chatterbox assignment. Concrete fix: make bark replay + bark uniqueness assertions conditional on bark rendering / bark voice_bank, or split the invariant so clone-engine casts are not required to fit the bark preset pool.

3. [Starting recommendation: “Add kokoro as … PRESET cast option”] Kokoro is not currently a character engine in the grounded adapter. `eng_kokoro.py` has `roles = ("announcer_voice",)` and `default_roles = ("announcer_voice",)`. Concrete fix: add `char_voice` explicitly only if the dispatch, CastLock assignment, and validation path are implemented; otherwise keep kokoro announcer-only and remove it from the cast-engine hardening decision.

4. [Open questions #2 / kokoro-as-char_voice wiring] The proposed kokoro CastLock path is not defined enough to build. Grounded `cast_lock.py` has `_VOICE_BANKS = ("default", "bark_legacy", "kokoro_builtin")`, but `_resolve_char_engine()` returns an engine only if there are bank entries for that engine and the profile allows the selected voice_bank; otherwise kokoro_builtin yields no character refs. Concrete fix: choose one path and specify it completely:
   - bank path: add kokoro char_voice refs to `_otr_voice_bank`, add engine profile support for `kokoro_builtin`, and let `assign_voice_for_slot()` stamp `voice_ref_id` + `voice_engine=kokoro`; or
   - preset path: implement `_assign_kokoro_voices()` analogous to bark, with deterministic seed-keyed assignment and uniqueness checks.
   Do not leave this as an open design question in a build plan.

5. [Candidate engines: kokoro] The table says kokoro has a “fixed preset pool (~dozens am_/af_/bm_/bf_),” but grounded `eng_kokoro.py` only defines `ANNOUNCER_VOICE_POOL = ["bm_george", "bm_fable", "bf_emma", "bf_lily"]` and hardcodes `KPipeline(lang_code="b", device="cuda", repo_id="hexgrad/Kokoro-82M")`. American `am_/af_` voices are not compatible with the adapter as written unless the pipeline language is selected per voice. Concrete fix: either restrict the kokoro char pool to local `bm_/bf_` voices for this build, or implement per-voice lang_code selection and preflight every assigned `.pt`.

6. [Invariants: determinism/per-character unique] Kokoro char promotion would violate uniqueness if a char line reaches `generate_voice()` without a voice_ref. Grounded `eng_kokoro.py` resolves `voice_id = voice_ref or self._episode_voice`, so missing character refs would collapse to the single episode announcer voice. Concrete fix: for `char_voice`, fail closed on missing/invalid kokoro voice_ref instead of falling back to `_episode_voice`; keep fallback behavior only for announcer_voice.

7. [Open questions #3 / license] The plan does not resolve whether a non-commercial model may be the cast default. Grounded `eng_indextts2.py` sets `commercial_clean = False`. Concrete fix: define build modes and enforcement: non-commercial/local-research may default to indextts2; commercial-clean builds must default to chatterbox or another `commercial_clean=True` engine and must not silently select indextts2.

SHOULD-FIX:
1. [OTR context: Render interface] The document says engines declare `voice_ref_field`, but grounded `eng_indextts2.py` does not; it declares `requires_voice_ref = True`, `voice_ref_kind = "wav_path"`, and `missing_ref_fallback = "bark"`. Concrete fix: either update the spec to reflect the actual dispatch contract or add the missing metadata consistently before using it as the basis for new kokoro/chatterbox wiring.

2. [Candidate engines: chatterbox / dia] The table says chatterbox and dia are wired but “re-smoke needed”; no grounding excerpt for either adapter is provided. [ASSUMPTION] If either is a final pick, add explicit smoke gates before build: local/offline startup, one-line render, missing-ref failure mode, deterministic seed behavior, delivery_vector handling, unload/VRAM release, and commercial_clean metadata.

3. [Invariants: 100% local/offline] Kokoro only preflights voice `.pt` files in `begin_episode()`. `load()` still constructs `KPipeline(... repo_id="hexgrad/Kokoro-82M")`. [ASSUMPTION] If KPipeline downloads missing model assets by default, this violates the offline invariant. Concrete fix: add a local model-weight preflight and use whatever kokoro library option prevents network access, or document the exact local cache requirement and fail before render.

4. [Starting recommendation: bark fallback] IndexTTS2 has `missing_ref_fallback = "bark"` in grounded code, but [Starting recommendation] demotes bark to “last-resort fallback” without stating whether bark remains installed/validated for clone-engine missing-ref fallback. Concrete fix: either keep bark as a tested fallback dependency or remove/disable `missing_ref_fallback = "bark"` and fail closed on missing clone refs.

5. [Candidate engines: bark] “no per-character variety” is inaccurate. Grounded CastLock has deterministic bark `voice_preset` assignment and uniqueness assertions; the real limitation is a small fixed pool and no delivery vector. Concrete fix: change the claim to “limited preset variety; no delivery/emotion control; license unconfirmed.”

6. [Open questions #4] The golden-test requirement is named but not converted into build steps. Concrete fix: require `test_audio_byte_identical` plus targeted tests for CastLock clone assignment, bark invariant isolation, kokoro char assignment, missing local kokoro voice, and commercial-clean engine selection.

OPTIONAL / NICE-TO-HAVE:
- Add a compatibility matrix showing each selected engine’s exact `voice_ref_field`/ref type, whether CastLock or the bank assigns it, and what happens on missing refs.
- Add a small “engine selection profiles” config instead of encoding the decision only in prose.
- Add a report line that clearly distinguishes “assigned bark compatibility preset” from “actual render engine voice ref” if bark presets remain stamped for fallback.

CUT THESE (over-engineering):
1. [Candidate engines: Qwen3-TTS] Cut from this hardening build. It is not installed, needs a new isolated venv, has stated Blackwell torch risk, and its emotion control does not apply to the clone path per the document. It does not help land the selected 1-2 engines.

2. [Candidate engines: dia] Cut dia re-smoke unless dia is one of the final 1-2 picks. The recommendation does not select it, so testing and stabilizing it is scope creep.

3. [Starting recommendation: kokoro char promotion] If the final pair is indextts2 + chatterbox, cut kokoro char_voice work from this build and leave kokoro announcer-only. Grounded kokoro char support requires roles, pool, assignment, validation, and possibly language-selection changes; that is not a small hardening task.