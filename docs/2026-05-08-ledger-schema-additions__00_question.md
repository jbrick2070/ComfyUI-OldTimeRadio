# Question -- 2026-05-08

# Design consult: OTR production ledger additions for BUG-126 telemetry + Cast Contract pre-wiring

## Background

OTR's production ledger (`<episode_id>_ledger.json`, schema
`l3-2026-05-02`) is the single source of truth that every
downstream node consumes: BatchBark, KokoroAnnouncer,
SceneSequencer, BatchLTXRender, BatchHumoRender, VideoComposite,
EpisodeAssembler, and the post-mortem audit script. It also
carries C7 byte-identity sha256 tripwires across audio phases
(`audio_gates[]`).

Two things just shipped that the ledger doesn't yet record:
1. **BUG-LOCAL-126** -- HuMo soak survives caught OOMs via a
   cleanup chain + per-process cap. The fix is in
   `nodes/batch_humo_render.py` (commit `146bf04`). Soak telemetry
   currently lives only in the ComfyUI process log; if the log
   rotates or the process dies hard, that telemetry is lost.
2. **Phase 0+ Cast Contract Extensions** -- `nodes/_otr_cast_contract.py`
   ships content-addressed sha-8 versioning, alias-aware lookup,
   per-episode lock. Orchestrator hooks at `story_orchestrator.py`
   L6423 / L640 / L920 are PENDING (waiting on next session).
   Without those, the contract module exists but no ledger entry
   carries the version stamp.

Concrete drift the current ledger is HIDING right now:
```
cast[0]: ANNOUNCER  voice_preset: "v2/en_speaker_4"   (Bark form)
lines[0]: char_id=c01  tts_engine: "kokoro"  voice_preset: "bm_fable"
```
The cast roster claims Bark; the actual line was rendered with
Kokoro. Cast Contract is the systemic fix; the ledger needs
fields to surface this kind of mismatch.

## What I want from you

For each numbered proposal, give me:
- **Field name + JSON location** (top-level / `meta.*` / `lines[].*` / etc.)
- **Schema-bump verdict**: ADDITIVE (no version bump), MINOR BUMP
  (`l3-2026-05-08`), or MAJOR (`l4-...`)
- **Default value** when not yet stamped (so back-compat readers
  using `dict.get(field, default)` don't crash)
- **Per-line vs aggregate** opinion where applicable
- **Risk** if we DON'T add this

Then a closing section with:
- **Stale fields you'd retire NOW** vs **defer**
- **One naming convention you'd disagree with** out of the proposals
- **Single biggest "gotcha" you see** if I ship all of this

NOT what I want: rewrites of the whole ledger schema, JSON Schema
spec generation, Pydantic models. The current schema is plain
dicts on purpose (diff-able on disk, no dep on validators). Keep
this consistent.

## The 11 proposals

### Tier 1 -- BUG-126 telemetry (would unblock self-diagnosing soaks)

1. **`lines[].oom_recovery_count: int`** -- bump when this line
   fails with a caught OOM and HuMo retries (today: line either
   succeeds or skips; tomorrow once retry-after-cleanup ships:
   tracks how many cleanup cycles burned on this line).
2. **`lines[].fallback_kind: str`** -- one of
   `humo_native` / `ltx_native` / `static_radio_fill` /
   `kokoro_announcer_only`. Right now you can only infer
   "this line fell through to static radio" from its absence
   in `clips[]`. Make absence explicit.
3. **`clips[].humo_oom_recovered: bool`** plus
   **`clips[].humo_oom_recovered_at_chunk: int | null`** -- so an
   outlier `humo_render_ms` (e.g., 646000ms vs typical 540000ms)
   is interpretable instead of mysterious.
4. **`meta.cuda_hard_reset_count: int`** -- bumped every time
   `_hard_reset_cuda_context()` runs in this process. Direct
   telemetry for whether the BUG-126 alarm plumbing is firing.
5. **`meta.soak_cap: {cap: int, lines_completed_when_hit: int, hit: bool}`** --
   stamped if `HumoSoakCapReached` raised this run. Pairs with
   `resume_from_ledger=True` so a multi-batch soak chain is
   provable from any single ledger.

### Tier 2 -- Cast Contract pre-wiring (lands fields NOW even though orchestrator hooks come later)

6. **`cast_contract_version: str`** (top-level) -- canonical
   `sha:HEX...` produced by `nodes/_otr_cast_contract.CastContract.stamp_version`.
   Stamp at the moment the orchestrator hooks land; until then
   the field stays absent (NOT empty string).
7. **`cast_contract: {version, characters: [{character_id, canonical_name, aliases, voice_spec}]}`** --
   the locked-roster JSON dump. Mirrors `_otr_cast_contract.CastContract.to_dict`.
   Orchestrator session populates this; pre-wire the field shape
   in the ledger schema docstring + `_otr_ledger.CURRENT_SCHEMA_VERSION`
   release notes so consumers know to look for it.
8. **`lines[].cast_contract_version: str`** (per-line) --
   stamped on every dialogue line at production-ledger merge time.
   Lets `production_ledger` reject merges in O(1) when the version
   on a line doesn't match the episode's locked version.
9. **`cast[].voice_spec: str`** -- canonical `"engine:preset"`
   form (e.g. `"bark:v2/en_speaker_5"`, `"kokoro:bm_fable"`)
   replacing the implicit binding via `cast.voice_preset`. The
   current cast/line drift surfaced above is exactly what this
   field eliminates: `lines[].tts_engine` + `lines[].voice_preset`
   gets validated against `cast[].voice_spec` at merge time.

### Tier 3 -- operational completeness (would help, not blocking)

10. **`meta.process_id: int` + `meta.process_started_at: str` (ISO-8601)** --
    so a multi-batch resume sequence (cap=6, run, resume, run, ...)
    is provable from any single ledger as a chain of restarts.
11. **`meta.audit_verdict: {pass: bool, ts: str, fail_patterns_hit: list[str]}`** --
    stamped by `scripts/audit_otr_full_run.py` after a run
    completes. Watcher's PASS/FAIL gets persisted into the ledger
    itself so post-hoc review doesn't need to also grep
    `outputs/soak_status.txt`.

## Stale field cleanup (separate question)

Should I retire any of these in the same change?

- `beats[]` -- length 0 in every ledger; schema docstring says
  "l2 adds beats[] hierarchy" but no producer populates it.
- `total_beats: 0` -- always 0.
- `lines[].boundary` / `shot_id` / `beat_id` -- always null.
- `lines[].bark_wav_path` -- only set when Bark renders; Kokoro
  lines never set it. Could rename `tts_wav_path` (engine-agnostic).
- `shots[].png_path` / `start_s` / `dur_s` -- always null since
  the FLUX-anchor era ended.
- `final_audio_path` -- empty string in this ledger; never gets
  stamped by EpisodeAssembler.

The risk of retiring NOW: any external script / tool that reads
these fields breaks silently. The risk of NOT retiring: schema
bloat continues, "is this actually used?" becomes harder to
answer over time.

## Constraints

- Plain dict JSON, no validator dep.
- All additions must be ADDITIVE (consumers using
  `dict.get(field, default)` keep working on older ledgers).
- Schema version is single string `l3-2026-05-DD` per
  `nodes/_otr_ledger.CURRENT_SCHEMA_VERSION`.
- We are NOT splitting the ledger across multiple files.
- Stamping happens at the writing node's hot path -- helper
  functions in `_otr_ledger.py` should be ~5-line additions, not
  a refactor.

Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
`v2.0-alpha`, head `146bf04`).
