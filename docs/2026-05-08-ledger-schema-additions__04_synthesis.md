# Synthesis -- 2026-05-08

**Question:** # Design consult: OTR production ledger additions for BUG-126 telemetry + Cast Contract pre-wiring

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

---

## ChatGPT (gpt-5.5)

## Executive recommendation

Ship these as **one additive minor schema bump**: `l3-2026-05-08`.

None of the 11 proposals require an `l4` major bump if you **only add fields** and keep old fields present. But because you are documenting new ledger semantics in `nodes/_otr_ledger.CURRENT_SCHEMA_VERSION`, I would not leave the schema string at `l3-2026-05-02`. Treat this as:

> **MINOR BUMP, JSON-compatible/additive.**

I cannot verify exact repo line numbers from here. The line references I can cite confidently are the ones you gave: pending orchestrator hooks at `story_orchestrator.py` **L640 / L920 / L6423**, and the relevant modules/files are `nodes/batch_humo_render.py`, `nodes/_otr_cast_contract.py`, and `nodes/_otr_ledger.py`.

---

# Tier 1 — BUG-LOCAL-126 telemetry

## 1. `lines[].oom_recovery_count: int`

### Field + location

```json
lines[i].oom_recovery_count
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
line.get("oom_recovery_count", 0)
```

Use `0` for old ledgers and for lines that never hit a caught HuMo OOM.

### Per-line vs aggregate

**Per-line is correct.**

A global count would tell you that BUG-126 recovery fired somewhere, but not which dialogue/video line consumed the cleanup cycles. For post-mortem analysis, the line-level count is the useful unit.

I would not add a separate aggregate yet. If you later need it, compute:

```python
sum(line.get("oom_recovery_count", 0) for line in ledger["lines"])
```

### Risk if you do not add it

You lose the causal link between a line and the recovery path. A run may “succeed” after cleanup, but the ledger will not explain why one line was slow, skipped, or later fell through to static fill.

This is especially important because BUG-LOCAL-126 currently only leaves durable evidence in the ComfyUI process log, which can rotate or vanish after a hard process death.

---

## 2. `lines[].fallback_kind: str`

### Field + location

```json
lines[i].fallback_kind
```

Proposed producer values:

```text
humo_native
ltx_native
static_radio_fill
kokoro_announcer_only
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

For readers:

```python
line.get("fallback_kind", "unknown")
```

I would not backfill old ledgers as `static_radio_fill` based only on absence from `clips[]`. Absence is not a reliable semantic signal.

### Per-line vs aggregate

**Per-line is correct.**

This is a line outcome, not a run outcome. A single episode can contain HuMo-native lines, LTX-native lines, announcer-only lines, and static radio fill.

### Risk if you do not add it

You continue relying on negative inference:

> “This line is absent from `clips[]`, therefore maybe it fell through to static radio.”

That is brittle. It makes audits and resume logic harder, and it hides the distinction between:

- intentional announcer-only line,
- static fallback,
- render skip,
- missing clip due to failure,
- old ledger format.

### Naming concern

This is the one proposed name I dislike most.

`fallback_kind` contains values like `humo_native` and `ltx_native`, which are not fallbacks. I would prefer:

```json
lines[].render_kind
```

or

```json
lines[].output_kind
```

If you keep `fallback_kind`, I would at least document that it really means “final render/output disposition,” not only fallback.

---

## 3. `clips[].humo_oom_recovered: bool` and `clips[].humo_oom_recovered_at_chunk: int | null`

### Field + location

```json
clips[i].humo_oom_recovered
clips[i].humo_oom_recovered_at_chunk
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
clip.get("humo_oom_recovered", False)
clip.get("humo_oom_recovered_at_chunk", None)
```

### Per-line vs aggregate

**Clip-level is correct**, because the timing anomaly is visible on the rendered clip.

However, this should pair with proposal 1:

- `lines[].oom_recovery_count` tells you the line had recovery activity.
- `clips[].humo_oom_recovered` tells you the final successful rendered clip survived recovery.
- `clips[].humo_oom_recovered_at_chunk` explains why one clip took much longer than similar clips.

### Implementation note

Define the chunk index convention explicitly. I would use whatever the internal HuMo chunk list uses, but document it. If logs are human-facing and use 1-based chunks, be careful not to mix 0-based ledger values with 1-based log messages.

### Risk if you do not add it

A suspicious clip duration remains mysterious.

Example:

```text
normal humo_render_ms: 540000
outlier humo_render_ms: 646000
```

Without this field, you cannot tell whether that outlier is:

- longer source material,
- scheduler stall,
- caught OOM + cleanup + retry,
- accidental CPU/GPU contention,
- malformed line metadata.

---

## 4. `meta.cuda_hard_reset_count: int`

### Field + location

```json
meta.cuda_hard_reset_count
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
ledger.get("meta", {}).get("cuda_hard_reset_count", 0)
```

### Per-line vs aggregate

**Aggregate/meta is correct.**

This is process/run telemetry, not line content. You can correlate it with per-line `oom_recovery_count` if needed.

### Risk if you do not add it

You cannot prove from the ledger whether the BUG-LOCAL-126 alarm plumbing fired. You would still need the volatile ComfyUI process log.

For a local-only workstation where logs may rotate and the owner does not want low-level VRAM archaeology, this is exactly the kind of small durable telemetry worth adding.

---

## 5. `meta.soak_cap: {cap, lines_completed_when_hit, hit}`

### Field + location

```json
meta.soak_cap
```

Shape:

```json
{
  "cap": 6,
  "lines_completed_when_hit": 6,
  "hit": true
}
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

For absent field:

```python
meta.get("soak_cap", {
    "cap": None,
    "lines_completed_when_hit": None,
    "hit": False
})
```

If you strongly want integer-only defaults:

```json
{
  "cap": 0,
  "lines_completed_when_hit": 0,
  "hit": false
}
```

But I prefer `null`/`None` for `cap` and `lines_completed_when_hit` when no cap was configured or stamped. `0` can look like a real configured cap.

### Per-line vs aggregate

**Aggregate/meta is correct.**

This describes the run boundary, not an individual line.

### Risk if you do not add it

A multi-batch resume sequence is hard to prove from the ledger alone. You may see that an episode is partially complete, but not whether it stopped because:

- soak cap intentionally fired,
- process crashed,
- operator cancelled,
- HuMo OOM escaped the cleanup path,
- ComfyUI died.

### Naming note

`meta.soak_cap` is acceptable, but slightly generic. If this is specifically HuMo, I would consider:

```json
meta.humo_soak_cap
```

That said, if the existing code and logs already say “soak cap,” do not churn the name unless you feel strongly.

---

# Tier 2 — Cast Contract pre-wiring

## 6. `cast_contract_version: str`

### Field + location

Top-level:

```json
cast_contract_version
```

Example:

```json
"cast_contract_version": "sha:3f8a91c2"
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

Per your note, keep it absent until the orchestrator hooks land.

Reader default:

```python
ledger.get("cast_contract_version", None)
```

Do **not** use empty string as the default semantic value.

### Per-line vs aggregate

**Aggregate/top-level is correct.**

This is the episode’s locked cast contract version.

### Risk if you do not add it

The ledger cannot expose whether it was produced under a locked cast contract. You will keep seeing hidden drift like:

```json
cast[0].voice_preset = "v2/en_speaker_4"
lines[0].tts_engine = "kokoro"
lines[0].voice_preset = "bm_fable"
```

The contract module in `nodes/_otr_cast_contract.py` can exist, but downstream nodes will have no durable version stamp to check.

### Relevant files

- `nodes/_otr_cast_contract.py`
- `nodes/_otr_ledger.py`
- pending orchestrator integration at `story_orchestrator.py` L640 / L920 / L6423, per your notes.

---

## 7. `cast_contract: {version, characters: [...]}`

### Field + location

Top-level:

```json
cast_contract
```

Shape mirroring `nodes/_otr_cast_contract.CastContract.to_dict`:

```json
{
  "version": "sha:3f8a91c2",
  "characters": [
    {
      "character_id": "c01",
      "canonical_name": "ANNOUNCER",
      "aliases": ["ANNOUNCER", "NARRATOR"],
      "voice_spec": "kokoro:bm_fable"
    }
  ]
}
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
ledger.get("cast_contract", None)
```

If a consumer wants dict-style access:

```python
contract = ledger.get("cast_contract") or {}
```

### Per-line vs aggregate

**Aggregate/top-level is correct.**

The locked roster is an episode-level object. Do not duplicate the full contract per line.

### Risk if you do not add it

You may have a version stamp but not the thing being versioned. That makes post-hoc review dependent on reconstructing or finding the external contract source.

Since the ledger is the single source of truth for downstream nodes, the locked roster should live in the ledger once the contract is active.

### Important gotcha

The version must be computed from a **stable canonical representation**:

- sorted keys,
- stable character ordering,
- stable alias ordering,
- no timestamps,
- no process IDs,
- no runtime telemetry,
- no ledger path,
- no machine-specific state.

Otherwise the content-addressed version will drift across otherwise identical runs.

---

## 8. `lines[].cast_contract_version: str`

### Field + location

```json
lines[i].cast_contract_version
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
line.get("cast_contract_version", None)
```

### Per-line vs aggregate

This is mildly redundant, but I agree with adding it.

The top-level version tells you the episode contract. The per-line version lets `production_ledger` reject mixed-version merges cheaply and locally:

```python
if line.get("cast_contract_version") != ledger.get("cast_contract_version"):
    reject
```

That is worth the small amount of duplication.

### Risk if you do not add it

A resumed or merged production ledger can accidentally combine lines produced under different cast contracts. You would only catch it by deeper inspection of character/voice fields, and possibly too late.

---

## 9. `cast[].voice_spec: str`

### Field + location

```json
cast[i].voice_spec
```

Canonical form:

```text
engine:preset
```

Examples:

```json
"voice_spec": "bark:v2/en_speaker_5"
```

```json
"voice_spec": "kokoro:bm_fable"
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

Do **not** remove `cast[].voice_preset` yet. Add `voice_spec` beside it.

### Default

```python
cast_member.get("voice_spec", None)
```

For legacy readers, continue falling back to:

```python
cast_member.get("voice_preset")
```

For validators, `None` should mean “legacy/uncontracted, cannot validate strictly.”

### Per-line vs aggregate

This belongs on `cast[]`.

The line still has:

```json
lines[i].tts_engine
lines[i].voice_preset
```

Then merge-time validation can compare:

```python
line_voice_spec = f"{line['tts_engine']}:{line['voice_preset']}"
cast_voice_spec = cast_member.get("voice_spec")
```

### Risk if you do not add it

You keep the exact hidden drift you showed:

```json
cast[0].voice_preset = "v2/en_speaker_4"
lines[0].tts_engine = "kokoro"
lines[0].voice_preset = "bm_fable"
```

The current `voice_preset` field is not enough because the preset namespace is engine-dependent. `"v2/en_speaker_4"` and `"bm_fable"` are not comparable without engine identity.

---

# Tier 3 — operational completeness

## 10. `meta.process_id: int` and `meta.process_started_at: str`

### Field + location

```json
meta.process_id
meta.process_started_at
```

Example:

```json
{
  "meta": {
    "process_id": 18420,
    "process_started_at": "2026-05-08T14:22:31Z"
  }
}
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
meta.get("process_id", None)
meta.get("process_started_at", None)
```

### Per-line vs aggregate

**Aggregate/meta is correct.**

This is run/process telemetry.

### Risk if you do not add it

You cannot tie a ledger state back to the specific ComfyUI process that produced it. That matters for BUG-126 because the current useful details live in the process log.

### Caveat

As proposed, scalar fields only identify the **last process that wrote the ledger**. They do not fully prove a multi-batch chain of restarts from a single ledger unless you preserve prior process info somewhere else.

If you truly want chain proof later, the eventual field would be something like:

```json
meta.process_runs[]
```

But I would not add that now unless you need it immediately. The scalar fields are a small and useful first step.

### C7 caution

`process_started_at` is intentionally nondeterministic. That is okay for the ledger, but do not let this field feed any downstream prompt, seed, scheduling, naming, or audio decision path.

---

## 11. `meta.audit_verdict: {pass, ts, fail_patterns_hit}`

### Field + location

```json
meta.audit_verdict
```

Shape:

```json
{
  "pass": true,
  "ts": "2026-05-08T15:10:03Z",
  "fail_patterns_hit": []
}
```

### Verdict

**MINOR BUMP** to `l3-2026-05-08`.

Additive and safe.

### Default

```python
meta.get("audit_verdict", {
    "pass": None,
    "ts": None,
    "fail_patterns_hit": []
})
```

### Per-line vs aggregate

**Aggregate/meta is correct.**

The audit script verdict is episode/run-level.

### Risk if you do not add it

The ledger cannot answer the basic question:

> “Did the post-mortem audit pass?”

You would still need to grep `outputs/soak_status.txt` or external logs.

### Implementation caution

Make sure `scripts/audit_otr_full_run.py` ignores its own prior `meta.audit_verdict` when auditing. Otherwise a second audit can differ from the first because the input ledger has changed.

Also, this mutates the ledger after the run. That should be fine for C7 audio byte identity if your C7 tripwires are over audio files, but be careful if any tooling hashes the full ledger file.

---

# Summary table

| # | Field | Location | Verdict | Default |
|---:|---|---|---|---|
| 1 | `oom_recovery_count` | `lines[]` | MINOR, additive | `0` |
| 2 | `fallback_kind` | `lines[]` | MINOR, additive | `"unknown"` |
| 3 | `humo_oom_recovered` | `clips[]` | MINOR, additive | `false` |
| 3 | `humo_oom_recovered_at_chunk` | `clips[]` | MINOR, additive | `null` |
| 4 | `cuda_hard_reset_count` | `meta` | MINOR, additive | `0` |
| 5 | `soak_cap` | `meta` | MINOR, additive | `{hit:false, cap:null, lines_completed_when_hit:null}` |
| 6 | `cast_contract_version` | top-level | MINOR, additive | absent / `None` |
| 7 | `cast_contract` | top-level | MINOR, additive | absent / `None` |
| 8 | `cast_contract_version` | `lines[]` | MINOR, additive | absent / `None` |
| 9 | `voice_spec` | `cast[]` | MINOR, additive | absent / `None` |
| 10 | `process_id` | `meta` | MINOR, additive | absent / `None` |
| 10 | `process_started_at` | `meta` | MINOR, additive | absent / `None` |
| 11 | `audit_verdict` | `meta` | MINOR, additive | `{pass:null, ts:null, fail_patterns_hit:[]}` |

---

# Stale field cleanup

## Retire now

I would physically retire **none** of these in the same change.

Reason: your stated constraint is additive compatibility. Removing fields, renaming fields, or stopping writes is where external scripts break silently.

For this change, I would only mark fields as deprecated in the schema docstring/release notes.

## Soft-deprecate now, remove later in `l4`

These look safe to mark as deprecated now:

### `beats[]`

If length `0` in every ledger and no producer populates it, mark deprecated.

Do not remove yet.

### `total_beats: 0`

Same as `beats[]`.

Mark deprecated.

### `lines[].boundary`, `lines[].shot_id`, `lines[].beat_id`

If always `null`, mark deprecated.

Do not remove until `l4`.

### `shots[].png_path`, `shots[].start_s`, `shots[].dur_s`

If these are artifacts of the FLUX-anchor era and now always `null`, mark deprecated.

Do not remove until `l4`.

## Defer

### `lines[].bark_wav_path`

Do not retire directly.

Add an engine-agnostic alias first:

```json
lines[].tts_wav_path
```

Then keep writing both for at least one schema generation:

```json
"bark_wav_path": "...",
"tts_wav_path": "..."
```

For Kokoro lines, `tts_wav_path` should be populated even when `bark_wav_path` is absent.

This one has real value, but the rename should be additive.

### `final_audio_path`

Do not retire yet.

If EpisodeAssembler is supposed to stamp it and does not, that is more likely an assembler integration gap than a stale schema field. I would either fix stamping or explicitly deprecate it later after confirming no downstream node expects it.

---

# Naming convention I disagree with

The main one is:

```json
lines[].fallback_kind
```

Because values like:

```text
humo_native
ltx_native
```

are not fallbacks.

I would prefer:

```json
lines[].render_kind
```

or:

```json
lines[].output_kind
```

If you already have code or docs using “fallback,” keep it to avoid churn, but the name is semantically leaky.

---

# Single biggest gotcha

The biggest gotcha is letting nondeterministic operational metadata leak into deterministic production behavior.

These fields are nondeterministic by design:

```json
meta.process_id
meta.process_started_at
meta.cuda_hard_reset_count
meta.audit_verdict.ts
meta.soak_cap
```

They are fine as ledger telemetry, but they must not affect:

- prompt text,
- seed derivation,
- cast contract hashing,
- line ordering,
- audio render decisions,
- C7 audio gate hashes,
- resume selection except where explicitly intended.

Especially for Cast Contract: the content-addressed `sha:...` must be computed only from the canonical locked roster, not from the whole ledger. If the contract hash accidentally includes process IDs, timestamps, audit verdicts, or BUG-126 telemetry, every run can produce a new contract version and your merge-time rejection logic will start rejecting valid resumes.

So the safe pattern is:

1. Generate or load deterministic creative state.
2. Lock and stamp cast contract from canonical cast data only.
3. Render audio/video.
4. Stamp operational telemetry.
5. Run audit and stamp `meta.audit_verdict`.
6. Never feed steps 4–5 back into creative generation.

---

## Gemini (gemini-3.1-pro-preview-customtools)

**1. Core Verdict**
PARTIALLY AGREE: The additive minor bump strategy is correct, but the proposed implementation for Proposal 10 actively defeats your stated goal, and the answer completely misses a critical file-corruption risk inherent to your VRAM constraints.

**2. Factual Errors in the ChatGPT Answer**
*   **Proposal 10 (Process ID) destroys the chain:** You explicitly requested that a multi-batch resume sequence be "provable from any single ledger as a chain of restarts." ChatGPT recommended scalar fields (`meta.process_id`, `meta.process_started_at`). If you resume, the new process overwrites the scalar fields, destroying the history. You need an append-only list (e.g., `meta.process_history: list[{pid, started_at}]`).
*   **Dangerous Default Types:** For `cast_contract_version` (Proposal 6 & 8), ChatGPT recommends `None` as the default. If your downstream nodes use string methods (e.g., `if line.get("cast_contract_version", None).startswith("sha:")`), `None` will throw a `TypeError` and crash the run. If downstream expects a string, default to `"unversioned"` or `"legacy"`, not `None`.

**3. Important Things That Were Omitted**
*   **Atomic Writes during Violent OOMs:** You are running HuMo/LTX on a 16 GB laptop with a 14.5 GB ceiling. CUDA OOMs and hard resets (`_hard_reset_cuda_context()`) are violent. If `nodes/_otr_ledger.py` writes directly to `<episode_id>_ledger.json` and the process dies mid-write during a BUG-126 recovery, your *single source of truth is truncated and destroyed*. 
*   **C7 Hash Contamination:** You mandate C7 byte-identity for audio. If your audio nodes (BatchBark, KokoroAnnouncer) calculate their cache/tripwire hashes by hashing the *entire* `line` dictionary, injecting `lines[].oom_recovery_count` or `lines[].fallback_kind` will alter the hash, trigger a re-render, and violate C7. Telemetry must be explicitly excluded from audio hashing.

**4. My Short Recommendation**
*   **Ship the minor bump (`l3-2026-05-08`)** with all 11 proposals, but make them strictly additive. Do not delete the stale fields yet; schema bloat is a future problem, breaking a solo dev's pipeline today is a present problem.
*   **Change Proposal 10 to a list:** Use `meta.process_history: list[{pid: int, ts: str}]` and append to it on startup in `story_orchestrator.py`.
*   **Rename Proposal 2:** I agree with ChatGPT's distaste for `fallback_kind`. Use `lines[].render_method`. It's accurate and requires no mental gymnastics.
*   **Enforce Atomic Writes:** In `nodes/_otr_ledger.py` (wherever the JSON dump happens), you *must* write to `<episode_id>_ledger.json.tmp` and then `os.replace()` to the final filename. This is non-negotiable for a system designed to soak through OOMs.
*   **Audit your C7 Tripwires:** Ensure the sha256 generation in your `audio_gates[]` only hashes creative fields (`text`, `voice_spec`, `tts_engine`), not the whole `lines[]` dict.

**5. Uncertainties to Verify**
*   **How does `_otr_ledger.py` currently save to disk?** I cannot see the file, but if it's doing a standard `with open(filepath, 'w') as f: json.dump(...)`, you are at extreme risk of file corruption during a hard CUDA crash.
*   **How are the `audio_gates[]` sha256 tripwires calculated?** I need to verify that adding keys to the `lines[]` dictionaries won't accidentally change the inputs to your audio hash functions.

---

## NVIDIA ()



---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
