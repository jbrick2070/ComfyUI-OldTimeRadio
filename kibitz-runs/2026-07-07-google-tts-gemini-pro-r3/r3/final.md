# Google TTS R3 Wiring Judgment

Driver: Codex
External reviewer: Antigravity / gemini-3.1-pro-high
Round: r3 wiring / integration / sequencing

## Verdict

Yes-with-fixes. The plan is suitable to hand to the build window after the doc
edits in `docs/google_tts_ideas.md`, with the additional operator constraint
that `google_tts` has no local, cross-provider, Comfy Cloud, or Partner-node
fallback path. A bounded same-provider Google/Gemini TTS model retry is allowed
only when explicitly configured and logged.

## Accepted Findings

1. CONFIRMED: `direct_api` must be excluded from automatic fallback/rank chains.

   Grounding: `nodes/_otr_engine_profiles.py` currently excludes only
   `runtime == "cloud"` in `EngineProfileResolver.rank_chain(...)`. If
   `google_tts` adds `runtime: direct_api` without changing that predicate, it
   can enter the automatic fallback/default ladder. The doc now requires
   excluding `runtime in ("cloud", "direct_api")` and adds a regression test.
   This is not a "fallback to cloud" design; it is the guard that prevents any
   automatic fallback into direct API engines.

2. CONFIRMED: stage-direction preservation must happen in the adapter
   `prepare_text(...)` path before neutral cleanup.

   Grounding: `nodes/_otr_script_prep.py` strips parentheticals and bracket
   tags in the shared prep path. Since Gemini can use tags such as `[whispers]`
   and `[sighs]`, `GoogleTTSVoice.prepare_text(...)` needs to preserve
   allowlisted tags before calling the neutral cleaner and then restore them as
   Gemini bracket tags. The doc now states this explicitly.

3. CONFIRMED: API-key redaction should be concrete, not just aspirational.

   Grounding: the plan already said to redact key values, but the build handoff
   is safer if it names the error boundary. The doc now requires wrapping the
   network call and response parsing and sanitizing `HTTPError`, `URLError`, JSON
   parse, and malformed-response messages before raising.

4. ACCEPTED AS DEFENSIVE: raw JSON extraction should accept both
   `output_audio.data` and `outputAudio.data`.

   Grounding: current Google examples expose `interaction.output_audio.data`.
   The exact raw REST casing is worth tolerating defensively. The doc now
   requires both casings in response parsing and tests.

5. CONFIRMED: hardcoded tests must be intentionally updated.

   Grounding: `tests/test_engine_profiles.py` enumerates exact profile ids and
   `tests/test_announcer_voice.py` asserts an exact dropdown list. Adding
   `google_tts` should update those tests while preserving index-0 defaults.

## Rejected / Misread

1. REJECTED: replace the official Interactions REST request with protobuf-style
   `voice_config.prebuilt_voice_config`.

   Grounding: the current official Gemini TTS Interactions REST example uses
   endpoint `https://generativelanguage.googleapis.com/v1beta/interactions` and
   `generation_config.speech_config: [{"voice": "Kore"}]`. The doc now adds a
   warning not to replace that shape with older protobuf-style snippets unless a
   live probe proves the official REST example changed.

## Doc Updates Made

- `docs/google_tts_ideas.md`
  - Added concrete sanitized exception handling.
  - Added a note preserving the official Interactions REST shape.
  - Added `outputAudio` response-casing tolerance.
  - Added adapter-owned stage-direction prep requirement.
  - Added `rank_chain(...)` exclusion for `direct_api`.
  - Tightened the no-fallback contract: no local fallback, no Comfy Cloud or
    Partner fallback, and no substitute audio from any other engine. A bounded
    same-provider retry to an allowlisted Google/Gemini TTS model is allowed
    only when explicitly configured and logged as a Google TTS retry.
  - Added focused regression-test bullets for the above.

## Build-Window Status

No implementation has started in this strategy thread. The next build window
should code from `docs/google_tts_ideas.md` plus this judgment artifact.
