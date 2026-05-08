# Round A -- ChatGPT (gpt-5.5) elapsed=142.5s

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
