VERDICT: no. The removal is not complete: a live smoke script still writes the deleted input, and checked-in artifacts still carry the supposedly removed field.

MUST-FIX BEFORE BUILD:
1. [QA Q1 / What was changed §4] `scripts/run_otr_30word_smoke.py` still requires and patches `OTR_VideoDirector.allow_auto_fallback` at `scripts/run_otr_30word_smoke.py:212-216`, while the live node schema no longer declares it in `nodes/otr_video_director.py:204-280`. This will fail in `_confirm_input` (`scripts/run_otr_30word_smoke.py:102-106`). Concrete fix: remove the `_confirm_input`, `dinputs[...]`, and log-change block for `allow_auto_fallback`; add or update a smoke-script test/guard so recipe patchers never reference removed director inputs.

2. [QA Q1 / What was changed §4] `tests/debug_prompt.json` still contains `"allow_auto_fallback": true` under node `87` at `tests/debug_prompt.json:272-289`, and also carries stale removed director inputs `other_beats_clip_mode` / `other_beats_n` at `tests/debug_prompt.json:281-282`. Concrete fix: regenerate this artifact from the post-removal workflow or delete it if it is only a transient debug dump; do not leave a checked-in `tests/` API prompt that cannot validate against current `OTR_VideoDirector.INPUT_TYPES()`.

SHOULD-FIX:
1. [Invariants / QA Q3] The document says “NO fallbacks” globally, but the repo still has non-video or compatibility fallback concepts, e.g. deterministic announcer fallback tests in `tests/test_announcer_passes.py:202-226` and retained empty ledger schema slot `runtime_fallback_decisions` in `nodes/_otr_video_engines/schemas.py:329-333`. Concrete fix: narrow the wording to “no video runtime engine fallback / no auto-default render substitution” so future cleanup does not accidentally target unrelated deterministic recovery paths.

2. [QA Q3] Removing `Policy.allow_auto_fallback` from a `_Forbid` model means any still-serialized `VideoRequest.policy` containing that key will now reject unknown input; `Policy` is now only `mute_generated_audio` and `strict_sync_required` at `nodes/_otr_video_engines/schemas.py:128-130`, and `_Forbid` rejects extras at `nodes/_otr_video_engines/schemas.py:78-81`. [ASSUMPTION] persisted episode/request payloads may exist outside tracked repo files. Concrete fix: explicitly state “old serialized video requests must be regenerated” or add a narrow migration that drops this one removed key before validation.

3. [QA Q1] Current architecture docs still assert the key exists in normalized `VideoRequest.policy`, e.g. `docs/2026-06-02-video-engine-architecture__consolidated-final.md:85`. Concrete fix: update current architecture docs or mark them historical; do not require rewriting every archived handoff unless a forbidden-symbol gate intentionally scans docs.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line audit to the workflow validator or a focused test that converts node 87 to API prompt and asserts no undeclared inputs are emitted.

CUT THESE (scope / over-engineering):
1. [QA Q1] Do not spend build time purging historical object-info captures such as `docs/2026-07-01-talking-radio/object_info_ltx_capture.json:19308` unless an explicit forbidden-symbol gate scans historical captures. Safe to cut because they are snapshots, not runtime loaders; update only live scripts, tests, canonical workflow, and current docs.