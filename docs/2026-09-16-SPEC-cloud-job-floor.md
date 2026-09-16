# SPEC 2026-09-16 -- the cloud job floor

Status: IMPLEMENTED, uncommitted, under review. Branch `main`, origin HEAD
`9509319b`. Every file:line below is the working tree, not origin.

## 0. The fault

A deluxe Foley 1-act ran 57m51s and died at beat 40.

```
[ERROR] shot shot_shot_001_b40 engine 'cloud_ltx25_foley_plus' failed to render;
  fallbacks are disabled (FailureKind.CRASH_BEFORE_LOAD)
  cloud media: provider_rejected -- cloud_ltx25_i2v: Polling aborted due to error:
  Task failed: {"error": {"type": "content_filtered_error",
  "message": "Content filtered due to policy restrictions"}}
```

LTX 2.5 refused ONE prompt, ~5 s after submit. Beats b01-b39 had already
rendered and muxed their Foley stems. The episode raised, nothing reached
`output/otr`, and every paid clip was discarded.

`classify_failure` (render_driver) has no branch for `CloudMediaError`, so the
refusal fell through to `CRASH_BEFORE_LOAD` -- "the engine died before it
loaded" -- about an engine that loaded, submitted, polled and got an answer.
That name then told the operator to "fix the engine or its inputs" and, had the
retry budget ever been wired, would have bought a free retry against a gate
that returns the same verdict every time.

This is the same DUMP as the spend-cap abort fixed earlier the same day, through
a different door. It is NOT the same CONTINUE policy: a 402 must halt the
remaining shots because the wallet is empty, while a policy reject is one
prompt and the siblings can still pass.

## 1. Operator rulings this obeys

- 2026-08-03: no violence/profanity guardrails on generated episode content.
  **The shot prompt is not rewritten to pass LTX.** Option C is banned.
- 2026-06-16 / 2026-07-02: NO FALLBACKS. No engine swap, no still floor, no
  chain. A local engine failure still fails LOUD.
- 2026-08-22: a model refusal degrades, an engine fault hard-fails.
- 2026-09-16: "any of them could easily get a failed output; a failed output on
  a cloud video or still should not break the system."

Flooring is not a fallback: no engine is substituted, nothing is silent, and the
beat keeps its id, role, family, position and frame budget. It has no clip --
a state the pipeline already has, and already counts.

## 2. The design

### 2.1 Name the verdict at the boundary

`cloud_media_backend.CloudErrorCode` gains `CONTENT_REFUSED`.
`cloud_media_invoke._map_exception` stamps it AFTER the wallet-empty (402) and
auth (401) checks and BEFORE the `PROVIDER_REJECTED` catch-all, while the
provider's own words are still in hand. It joins `_RELEASE_CODES`, so a
5-second verdict releases its reservation instead of billing the estimate.

### 2.2 Job-scoped vs run-scoped

```
JOB_SCOPED_CODES = {CONTENT_REFUSED, PROVIDER_REJECTED, TIMEOUT,
                    RETRYABLE_TRANSPORT, CORRUPT_OUTPUT, ORPHANED_JOB}
RUN_SCOPED_CODES = {AUTH, BUDGET, MALFORMED_CONFIG, UNSUPPORTED_SCHEMA,
                    INCOMPATIBLE_PROFILE, GATED_BY_FLAG}
```

Job-scoped says nothing about the next beat, so the beat floors. Run-scoped
means nothing will ever render, so flooring would publish an empty episode as
"degraded" -- those stay LOUD. `BUDGET` keeps its own door because it must also
HALT further submits. `INTERRUPTED` is in NEITHER set: a cancel is the operator
saying stop, and turning that into 40 floored beats would be obscene.

`cloud_job_failure_code(exc)` walks the cause chain and reads ONLY a stamped
code. No prose fallback -- "an exception happened during a cloud beat" is far
too wide a net to floor on.

### 2.3 `is_content_refusal` -- two passes, and the order is the contract

The still funnel's sanctioned gap (2026-08-28) is minted from a RECORDED FACT,
and `otr_video_render_batch` states the standard as "AN ABSENCE IS NOT A
SANCTION". A substring match on an exception string is an absence being read as
a sanction, so prose may only speak where no verdict was recorded.

- PASS 1: a stamped `CONTENT_REFUSED` anywhere in the chain wins. Any OTHER
  stamped `CloudErrorCode` **vetoes**. This is load-bearing: `render_shot` wraps
  every failure in a `RenderError` whose message CONCATENATES the whole inner
  chain, so without the veto a `TIMEOUT` quoting refusal words in its body would
  floor a beat the boundary called retryable.
- PASS 2: prose needles, only when nothing in the chain carries a code.

Needles are compared on an alphanumeric-only normalization, because five
providers spell one verdict five ways: LTX `content_filtered_error`, BFL
`Content Moderated` / `Request Moderated`, Gemini `IMAGE_PROHIBITED_CONTENT`,
ByteDance `OutputAudioSensitiveContentDetected`, Comfy
`image_content_policy_violation`.

Four needles were CUT on review for matching ordinary faults:
`policy restriction` (a corporate proxy's "blocked by policy restriction" --
systemic, would floor EVERY beat), `safety system` ("our safety system is
temporarily unavailable" -- a retryable outage), `safety filter` (a
`FileNotFoundError` for `safety_filter_v2.pth`), `sensitive content` (ordinary
English, reachable by a prompt echoed into an error body). A miss only restores
fail-loud; a false match launders a broken render into a publishable episode.

### 2.4 The video floor

`render_driver._cloud_floor_reason(sid, errors, shot)` returns the reason, or
`""`. THREE gates, all of which must hold: the shot's engine is a CLOUD engine
(`_is_cloud_video_engine`); the failure carries a stamped job-scoped code; it is
not run-scoped. `_stamp_cloud_floor_shot` rides the existing sanctioned-gap
ACCOUNTING channel (`STATUS_SANCTIONED_GAP`) and records `cloud_floor=<reason>`;
the content case additionally keeps `content_floor=True`, because that is the
one with an operator ruling attached.

Both commit paths floor: the fan-out walk and the serial walk. Neither sets
`cloud_spend_halt` -- only BUDGET halts.

`build_clip_manifest` projects the sanction onto `manifest["clips"]`, so node
92's predicate counts the beat: `ok=True`, `degraded=True`,
`sanctioned_gap_count` names it. An unexplained hole still fails the episode.

`_report_content_floors` logs LOUD when refusals exceed 25% of the episode
(`CONTENT_FLOOR_SYSTEMIC_SHARE`) and NEVER raises -- aborting there would
reinstate the exact defect the floor removes.

### 2.5 The still funnel had the identical bug

`_render_still_pixels` only tolerated a refusal when the exception carried
`is_model_refusal`, which is set by exactly two engines (`eng_google_image`,
`ideogram4_local`). Every PARTNER still refusal raises a `CloudMediaError` with
no such attribute, so it re-raised as "NO FALLBACK -- fix the engine" and took
the whole fan-out wave -- already submitted, already paid -- down with it.

`_still_engine_refused` now reads both dialects; `_cloud_still_job_failure`
applies the job-scoped rule on both the fan-out and serial paths.
`still_receipt` gains `CLOUD_JOB_SKIP_REASON = "cloud_job_failed"` and
`is_sanctionable_skip()` -- a timeout is not a refusal and the receipt must not
say it was. The tolerance gate stays narrow: a dead path, a historical-row-only
target and a no-engine skip remain unsanctionable.

### 2.6 Taxonomy

`FailureKind.CONTENT_REFUSED`, HARD, `same_seed_retries=0`. Stated honestly:
nothing on the render path reads `same_seed_retries` today, so the zero
DOCUMENTS the policy rather than enforcing it -- it is there so that the day a
retry budget is wired, this kind does not inherit `CRASH_BEFORE_LOAD`'s one free
retry against an immovable gate.

## 3. Evidence

- Focused: 124 tests pass across `test_cloud_media_backend.py`,
  `test_video_render_driver_additive.py`, `test_video_retry_taxonomy_additive.py`,
  `test_cloud_partner_fanout_order.py`, `test_model_refusal_degrades.py`.
- Full suite: 15,774 collected, ZERO new failures vs the pre-change snapshot.
- The EXACT live LTX bytes are a test, so no credits are spent proving the map.
- End-to-end `run_episode` tests on stub cloud engines prove: refused beat
  floors while siblings commit; a TIMEOUT floors the same way; an AUTH still
  fails loud; an ordinary crash still fails loud; both the fan-out and serial
  branches are exercised (each asserts `_should_fanout_cloud_episode`).
- All 22 workflow JSONs validate (canonical + 21 variants). The diff contains no
  `INPUT_TYPES`, `widgets_values`, `RETURN_TYPES` or `NODE_CLASS_MAPPINGS` lines.

## 4. The two questions for this review

1. **Can any floor launder a real crash into a publishable degraded episode?**
   Enumerate every route into `_cloud_floor_reason` and
   `_cloud_still_job_failure`. Can a LOCAL engine reach a floor? Can
   `_is_cloud_video_engine` return True for something local? Can the prose
   fallback fire where a code should have been stamped?
2. **Can a real per-job cloud failure still kill the run?** Check
   `cloud_fanout`'s `stuck_ids`, the "never rendered shot" raise, and
   `render_beat_coverage`'s own post-render checks (no segment path,
   frame-count mismatch, foley stem mismatch) -- do those carry a stamped code,
   and if not, is that right?

## 5. Known, NOT fixed here

A survey found the same shape in six other places, left alone deliberately
because each needs an operator decision rather than a mechanical fix:

- `_otr_voice_node_common.py` TTS fan-out raises the first line error and
  discards every other rendered line (largest per-item cloud spend after video).
- `_otr_voice_node_common.py` post-render receipt gate discards fully rendered
  audio on a local ledger-write failure.
- `stable_audio_theme.py` music fan-out raises the first cue error.
- `otr_silent_composite.py` `ClipUnderrunsItsBeat` is terminal at composite
  time although a floor exists three lines away.
- `otr_video_render_batch.py` `_stamp_render_engines_meta` raises after clips
  are persisted.
- `render_driver` still raises for a cloud failure that carries NO stamped code.
