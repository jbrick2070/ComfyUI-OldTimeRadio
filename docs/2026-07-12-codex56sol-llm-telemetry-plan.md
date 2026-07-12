# codex56sol LLM telemetry capture -- CODE-READY PLAN

2026-07-12. Hardened through a full kibitz r1->r4 arc (Codex `gpt-5.5` +
Antigravity, with Cowork Claude as grounded anchor panelist and sole judge).
Artifacts: `kibitz-runs/2026-07-12-p5-telemetry/`.

Supersedes `docs/2026-07-12-p5-telemetry-capture-problem-statement.md`.

**Renamed on purpose.** The problem statement was called "P5 telemetry", but `P5`
is a real `pass_id` in the very file being changed (the broadcast score,
`_otr_original_codex56sol.py:1883`) and the scope is ALL NINE passes. Anyone
reading "P5 telemetry" would reasonably build the wrong thing.

Operator-approved direction: *"add telemetry if it works and is not a big
performance hit."* It is not a big performance hit: ~0.005% of call wall time.

---

## 1. Why

`_call` (`nodes/_otr_original_codex56sol.py:699-818`) journals only a raw-response
sha256 plus pass/slot (:817). It does NOT journal the exact messages sent, the raw
rejected output, the exact validator error, which deterministic projection fired
and what it changed, or the accepted artifact.

Consequences today: prod-bug forensics reconstruct prompts by hand; seam tunes are
argued from memory instead of counted defects; no adapter or constrained-decoding
census is possible. **This pays for itself on bug forensics and seam tuning even if
no adapter is ever trained.**

## 2. What the arc found (four blockers the original plan would have shipped)

| # | defect | why it mattered |
|---|---|---|
| **B1** | The pending sweep `rmtree`s every ABORTED episode | Aborted episodes ARE the defect census. The plan's own retention model would have destroyed the population it exists to measure. |
| **B2** | There are FOUR Python projection sites, not two -- and the P5 *post-validator* MUTATES the score in place | Without them the census cannot tell "the model got it right" from "Python quietly fixed it". That distinction is the entire question. |
| **B3** | `_call.capture` normalizes BEFORE returning | A ladder-level observer would record Python-repaired text as the model's raw output -- silently corrupting the most important field in the schema. |
| **B4** | A held file handle inside the episode dir breaks the downstream `os.replace` rename | Self-inflicted P0 at the END of a multi-hour render, in a different node, with an error message pointing at Notepad. |

Plus three join defects (patched artifacts, hash-contract mismatch, `else`-block
never running) that would have produced quietly unjoinable data.

## 3. Architecture

### 3.1 The capture seam is the LADDER, not just `_call`

`_call` sees messages and raw output. It does **not** see the verdict. Parse /
schema-validate / post-validate all happen inside `structured_call`
(`nodes/_otr_structured_call.py:551-823`); each of its four rungs catches
`(json.JSONDecodeError, ValidationError, PostValidationError)` in a LOCAL `except`
(:683, :715, :769, :805) and the error never reaches `_call` unless the whole
ladder is exhausted (:819).

**Add two optional keyword-only callbacks to `structured_call`:**

```
on_attempt:    Callable[[dict], None] | None = None
on_projection: Callable[[dict], None] | None = None
```

`structured_call` is declared PURE (:30-31: *"no I/O, no GPU, no ComfyUI imports"*).
**Injecting callables preserves that** -- the ladder never imports the telemetry
module. Say so in the docstring in the same commit, or the next reader will
correctly read this as a purity violation and revert it.

No other lane passes a recorder, so every other `structured_call` caller is
byte-identical. The seam is not a generalization.

### 3.2 The four projection sites

Every one of these rewrites model output with Python before acceptance. All four
already compute a description of what they changed; they just have no channel.

| id | site | fires on | what it changes |
|---|---|---|---|
| **P-A** | `_call.normalize_attempt_output` (:713-747, inside `capture`) | P3, P5 | lifts nested collections out of the raw JSON |
| **P-B** | `_call.repair` factory (:753-789) | P3, P5 | same repair, on the pre-typed-repair failure |
| **P-C** | `validate_tolerant_data` -> `_clamp_overlong_strings` (`_otr_structured_call.py:422-439`, :489-543) | **ALL passes, all lanes** | clamps over-long strings at any depth |
| **P-D** | `_validate_score_attempt` (:1060-1102) | **P5 only** | **mutates `score.shots` / `score.beats` IN PLACE** (:1090-1091) |

**P-D is the most valuable record in the schema and was completely invisible.** It
is the `post_validator` for P5 (:1894) -- a "validator" that repairs, running inside
the ladder on every schema-valid P5 response. From the rung boundary the attempt
looks like a clean `accepted`. It has exactly ONE production caller, so it is safe
to change.

### 3.3 Raw-vs-projected: the `_pending` cells (fixes B3)

`_call.capture` (:749-752) IS the `slot_fn` handed to `structured_call`:

```
def capture(messages, **kwargs):
    raw = fn(messages, **kwargs)
    attempts.append(sha256(raw))
    return normalize_attempt_output(raw)     # <-- P-A projection, BEFORE return
```

So `_invoke_slot` -- and any observer inside the ladder -- receives the
POST-PROJECTION text.

**Fix: `capture`, `repair`, and the observer are all closures in `_call`'s scope.**
Use that. No `GenerateFn` signature change (an extra kwarg through `_invoke_slot`
would reach the real backend fns and break them).

```
_pending_call: dict = {}          # facts only `capture` can see
_pending_projections: list = []   # P-A (capture) + P-B (repair), awaiting ladder context

def capture(messages, **kwargs):
    _pending_call.clear()                   # BEFORE fn: a backend error must not
                                            # inherit the prior rung's raw output
    raw = fn(messages, **kwargs)            # the TRUE provider output
    attempts.append(hashlib.sha256(str(raw).encode()).hexdigest())   # unchanged
    projected = normalize_attempt_output(raw)                        # P-A -> queue
    _pending_call.update(messages=messages, raw=raw, projected=projected)
    return projected

def on_attempt(event):     # fired by structured_call AFTER parse/validate
    try:
        telemetry.record_attempt(event, **_pending_call)
        for proj in _pending_projections:
            telemetry.record_projection(
                {**proj, "attempt_idx": event["attempt_idx"], "rung": event["rung"]}
            )
    finally:
        _pending_call.clear()
        _pending_projections.clear()
```

Ordering is guaranteed: the ladder calls `_invoke_slot` (-> `capture`), THEN parses,
THEN fires the observer.

`_pending_call` is correctly EMPTY for the two rungs that make no slot call:

- **`deterministic_repair`** (`_otr_structured_call.py:750-761`) -- the repair
  factory returned a finished instance and no LLM call was made.
  `raw_sha256: null`, `llm_call: false`. Invisible today.
- **`backend_error`** -- `capture` raised before populating the cell. No stale raw.

`_pending_projections` is NOT cleared by `capture` (only `_pending_call` is), so a
P-B projection queued by `repair` -- which the ladder calls BEFORE `_invoke_slot` on
the typed-repair rung -- survives to be drained by the observer with the correct
rung and attempt index.

The attempt record therefore carries BOTH `raw_sha256` (true provider output) and
`projected_sha256` (what the ladder actually parsed; `null` when no projection
fired).

### 3.4 `on_projection` -- ladder context is INJECTED, not threaded

`structured_call` already knows `attempts_run` and the rung. It wraps the caller's
callback per rung:

```
_proj = lambda ev: on_projection({**ev, "attempt_idx": attempts_run, "rung": rung})
```

So `validate_tolerant_data` and `_validate_score_attempt` stay ignorant of the
ladder and each gains exactly ONE optional kwarg. (Rejected the panel's proposal to
thread `attempts_run` down through three functions: it puts ladder state inside a
pure validator for no gain.)

**Critical -- do NOT pass `on_projection` into `post_validator`.**
`validate_tolerant_data` calls `post_validator(instance)` (:436) and EVERY other
lane's post_validator is a 1-arg lambda. Passing a kwarg there is a `TypeError`
that crashes every other lane. P-D's callback is **closed over by the lambda at the
P5 call site** (:1894):

```
post_validator=lambda value: _validate_score_attempt(
    value, truth, None, story_rules, on_projection=_proj)
```

The lambda's arity stays 1. `validate_tolerant_data`'s call to `post_validator` is
UNCHANGED. The only place `validate_tolerant_data` itself gains an `on_projection`
kwarg is for its own P-C clamp.

**P-D must snapshot before it mutates.** `_validate_score_attempt` assigns
`score.shots` / `score.beats` in place (:1090-1091), so a callback fired afterwards
has already lost the "before". Snapshot `sha256_model(score)` before the first
assignment -- only paid on the repair path (the loop breaks immediately when
`repaired is None`), so the common case is free.

### 3.5 ONE canonical hash helper (join integrity)

`model_dump_json()` and `json.dumps(model_dump(mode="json"), sort_keys=True)`
produce DIFFERENT BYTES for an identical object. Using both would silently break
the accepted->terminal join.

```
def sha256_obj(value) -> str:
    """Canonical hash for a pydantic model OR an already-dumped dict."""
    data = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return sha256_text(json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
```

It must accept dicts: the `accepted_artifacts` entries at :2031-2041 are already
`model_dump(mode="json")` results. Same canonicalization for `pack_stages_sha256`
over `pack.prompt_stages` -- dict insertion order is not a hash contract.

### 3.6 Acceptance records, including the PATCH/MERGE paths

`_call` cannot know `artifact_key`:

- P1 returns a `PossibilitySlate`, but `accepted_artifacts` stores
  `"selected_possibility"` -- a `PossibilityCard` chosen AFTER the call (:1874).
- Passes supersede: `P7_rerun` (:1944), `P8_optional` (:1957), `P9_retake` (:1983),
  `P9_rerun` (:1999).
- **Patched artifacts never appear in any `_call` result.**
  `_repair_score_grounding_intents` (:1392-1419) `_call`s for a `ScoreIntentPatch`,
  then `_apply_score_intent_patch` -> `_merge_score_intent_patch` (:1389) returns a
  MERGED `BroadcastScore`. Same for `_repair_script_grounding_lines` (:1612-1634) ->
  `_merge_script_line_patch` -> merged `PerformanceScript`. The terminal map hashes
  the MERGED object, which no `accepted` record contains. The join would just miss.

So, three record shapes:

- **`_call` emits** `kind:"accepted"` at its return (:818): `pass_id`,
  `schema_name`, `accepted_sha256 = sha256_obj(result)`.
- **Each merge site emits** a DERIVED `kind:"accepted"`:
  `derivation:"merged"`, `origin_pass` (P5 / P6 / P8 / P9), `patch_pass`
  (`*_grounding_patch`), `base_sha256`, `patch_sha256`, and
  `accepted_sha256` of the merged object.
- **The terminal record carries** `accepted_artifacts_sha256`: the
  `{semantic_key -> sha256}` map built at :2031-2041, where the semantic names
  exist. Hash the payloads that already go to the ledger; do not copy them.

Every entry in the terminal map now has a matching `accepted` record. That is the
join, and it is testable.

### 3.7 Model identity: `model_meta` records, emitted on first use

The header CANNOT carry model identity. `_SlotScheduler` only learns the cache entry
inside `_account_and_get_entry` (`OTR_LedgerScriptWriter.py:477-518`) after
`request_slot`, and the technical slot is not requested until P2 (:1873). Eagerly
resolving both slots to build a header would trigger a real model load -- telemetry
must never cause a load.

- `_SlotScheduler.__init__`: `self.entry_meta_by_slot: dict[str, dict] = {}`.
- `_account_and_get_entry`, after `request_slot`: stash
  `{"model_id": resolved_id, "provider": cache_entry.get("provider") or "transformers",
    "context_cap": cache_entry.get("context_cap")}`.
- The recorder emits ONE `kind:"model_meta"` per slot, the FIRST time it observes an
  attempt on that slot -- by which point the entry exists. If `request_slot` itself
  fails, that attempt is a `backend_error` and **no `model_meta` is invented**.

Providers are a real enum, not a guess: `openrouter`, `comfy_credits`, `google_api`,
`gguf_native`, else local transformers (`OTR_LedgerScriptWriter.py:627-644`).

**`model_commit_sha`:** `resolve_snapshot_dir` (`nodes/_otr_hf_env.py:132`) returns
`<hf_home>/hub/models--<org>--<name>/snapshots/<commit_sha>/` -- **the PATH, not the
sha**. Writing it raw both misses the point and leaks the host filesystem layout.

```
snap = resolve_snapshot_dir(model_id)
model_commit_sha = Path(snap).name if snap else None
```

Resolve ONLY when the provider is absent/`transformers` (it is an HF-cache scan;
`gguf_native` and the three remotes have no snapshot dir). `null` otherwise. It
stays in the plan: the loader picks the snapshot by MTIME, so the catalog model id
is a moving pointer and the commit sha is the only stable weight identity.

## 4. Record model

`otr/episodes/<ep>/telemetry/llm_calls.jsonl` -- one JSON object per line, UTF-8 no
BOM, `telemetry_schema_version` from day one.

| kind | when | carries |
|---|---|---|
| `header` | once, at lane entry | `telemetry_schema_version`, `run_id` (uuid4), **`episode_id_at_capture`**, `source_bank_id`, `story_pipeline_id`, `pack_schema_version`, `pack_stages_sha256`, `OTR_CAST_SEED` / `OTR_STYLE_SEED` / `OTR_ORIGINAL_SEED` if set |
| `model_meta` | once per slot, on first attempt | `slot`, `model_id`, `provider`, `context_cap`, `llm_policy`, sampling knobs, `model_commit_sha` |
| `attempt` | one per rung, incl. the deterministic short-circuit | `pass_id`, `seam`, `seam_sha256`, `slot`, `rung`, `attempt_idx`, `llm_call`, `temperature`, `max_new_tokens`, `messages_sha256`, `raw_sha256`, `projected_sha256`, `outcome`, `error_type`, `error_str`, `latency_ms` |
| `projection` | one per projection that FIRED (P-A..P-D) | `site`, `repair_fn`, `changed`, `before_sha256`, `after_sha256`, `attempt_idx`, `rung` |
| `accepted` | one per `_call` return + one per merge | `pass_id`, `schema_name`, `accepted_sha256` [, `derivation`, `origin_pass`, `patch_pass`, `base_sha256`, `patch_sha256`] |
| `terminal` | once, in `finally` | `completed` \| `aborted`; on abort `error_type` + `error_str` + `pass_id`; on success `accepted_artifacts_sha256` |

`changed` is a sorted list of STRUCTURAL identifiers -- field paths (P-C) or
projection-kind names (P-D) -- **never a free-text diff**. The field exists to be
COUNTED. Full before/after already lives in the blobs.

Two distinct `schema_version`s exist (telemetry record format vs StoryPack format).
They are named apart -- `telemetry_schema_version` / `pack_schema_version` -- so no
one conflates them.

`error_type` and `error_str` are SEPARATE fields.
`OriginalCodex56SolContractError` vs `OriginalCodex56SolPassError` vs
`StructuredCallFailedError` is the first cut of any defect census; a free-text
reason is not countable.

**`episode_id_at_capture` is the PENDING id.** `new_ledger(episode_id=None)` yields
`pending_<YYYYMMDD_HHMMSS>` (`OTR_LedgerScriptWriter.py:3636`) and the dir is
renamed to the title slug LATER and DOWNSTREAM. The writer can only ever know the
pending id. The AUTHORITATIVE join is the telemetry dir's LOCATION: it rides inside
the episode dir, so `os.replace` carries it along and the folder name is always
current.

### Blob store

`otr/episodes/<ep>/telemetry/blobs/<sha256>.txt` -- content-addressed messages and
raw/projected outputs. Dedupe is within-episode only (an in-memory `set` of seen
hashes; no per-call `stat`). The seam system prompt is byte-identical across every
attempt of a pass, so dedupe collapses most of the volume.

**Keep the blob store** (antigravity proposed inlining). Tens of files and low
single-digit MB per episode is nothing next to a render writing thousands of frames,
and a small greppable jsonl plus `type blobs\<sha>.txt` is a far better forensic
surface than 200 KB lines.

## 5. Hard constraints

### F1. FAIL-OPEN, warn-ONCE, latching

`new_recorder()` is itself fail-open (including the `mkdir`) and returns
`_NullRecorder` if setup fails. Every public method is wrapped:

```
def _failopen(fn):
    def wrapped(self, *a, **kw):
        if self._dead:
            return None
        try:
            return fn(self, *a, **kw)
        except Exception as exc:                      # noqa: BLE001
            self._dead = True
            log.warning("[otr_telemetry] disabled for this episode after %s: %s",
                        type(exc).__name__, exc)
            return None
    return wrapped
```

First failure logs ONE line and latches dead for the episode. A sick filesystem
costs one failed write, not one per attempt, and does not bury the render log.
House pattern already exists (`OTR_LedgerScriptWriter.py:3162`, :3228-3229, :4489).

**The one exception:** a `backend_error` is RECORDED and then the ladder RE-RAISES
exactly as today. The render still fails loudly. A telemetry-caused lane abort is a
P0 self-inflicted bug.

### F2. NO PERSISTENT FILE HANDLE (fixes B4)

`Ledger.rename_episode` (`nodes/production_ledger.py:580-726`) does
`os.replace(old_ep_dir, new_ep_dir)` (:671) -- it moves the WHOLE episode directory.
On Windows that FAILS if any file inside is open; the code says so in its own error
text (:696-700). Three retries at 0.5 s, then `raise RuntimeError` (:692-701). And
it fires DOWNSTREAM, in **`video_engine.py:2430`**, once the title is final -- so a
leaked handle kills the run at the END of a multi-hour render, in a different node.
Recorder-side fail-open cannot catch it: the exception is raised by the LEDGER.

**Every write is open-append-close.** Blobs likewise. ~15-40 opens per episode, tens
of microseconds each, against 100-200 SECOND LLM calls. Unmeasurable.

The telemetry path NEVER opens the episode LEDGER files -- its own files only.

### F3. Null-object recorder -- `_call` carries ZERO `if telemetry:` branches

```
class _NullRecorder:
    enabled = False
    def observer(self, **_): return None    # -> on_attempt=None: today's exact branch
    def record_header(self, **_): pass
    def record_model_meta(self, **_): pass
    def record_attempt(self, *a, **kw): pass
    def record_projection(self, *a, **kw): pass
    def record_accepted(self, **_): pass
    def record_terminal(self, **_): pass
    def close(self): pass
```

It must mirror the FULL public API of `TelemetryRecorder` (a missing
`record_header` is an `AttributeError` inside a render). `_call`'s `telemetry`
kwarg defaults to the module `_NULL` singleton, so all 12 existing `_call` sites and
every direct-`_call` test stay green with no edit, and the telemetry-off path is
byte-identical **by type, not by discipline**.

### F4. Perf budget

Per attempt: `json.dumps(messages)` for the blob (P6/P8 payloads are 100-200 KB) ~1-3
ms; sha256 over it ~0.5-1 ms; `sha256_obj(result)` on accept ~1-3 ms; 2-4 file opens
~tens of microseconds. **< 10 ms worst case against a 100-200 SECOND NF4 decode:
~0.005%.** The stated budget (< 0.5%) has two orders of magnitude of headroom.

The one thing that WOULD blow it is buffering payloads in memory and flushing at the
end -- holding a dozen 200 KB payloads mid-render on a 16 GB card. **Forbidden.** The
observer hashes and streams each payload straight to the blob store.

`resolve_snapshot_dir` is a filesystem scan: called ONCE, in `model_meta`, inside
fail-open. Never per attempt.

### F5. Enablement -- one resolver, one table test

`telemetry_enabled()`: `OTR_TELEMETRY=1` -> on (force); `=0` -> off (force); else
`OTR_TEST_MODE=1` -> off; else on. House pattern is
`os.environ.get("OTR_TEST_MODE") == "1"` (`production_ledger.py:440`,
`_otr_ledger.py:329`). `tests/conftest.py:38` sets `OTR_TEST_MODE=1` suite-wide, so
telemetry is OFF for ~4200 tests by default; telemetry tests opt IN with
`monkeypatch.setenv("OTR_TELEMETRY", "1")` + a `tmp_path` episode root.

### F6. Retention -- the pending sweep must not eat the census (fixes B1)

`nodes/_otr_pending_cleanup.py:109-196` sweeps `otr/episodes/pending_*`.
`_ledger_has_lines` (:86-106) returns `True` (KEPT -- *"forensic evidence of a run
that got somewhere"*), `False` (ledger shaped, `lines == []`), or `None` (no
ledger). **Both `False` and `None` are `shutil.rmtree`'d** (:168, :190) once older
than 2 h.

A codex56sol lane that aborts on a contract error at ANY of P1-P9 never reaches
`led.set_lines(...)` (:2011). Its ledger has `lines == []`. **The sweep deletes it,
and its telemetry with it.** Failed episodes ARE the defect census.

Fix, same sprint, using the rule the file already applies to lines:

- `_has_telemetry(episode_dir) -> bool`: `True` iff
  `<dir>/telemetry/llm_calls.jsonl` exists and is non-empty.
- Check it **before `skipped_no_ledger.append(...)`** (:160-174) and before the
  `lines == []` delete arm (:183-195) -- otherwise the same dir is booked as both
  "no ledger" and "retained for telemetry".
- NOT checked ahead of the `ledger_state is True` arm, so `skipped_has_lines`
  statistics stay exactly as they are today.
- `PendingSweepReport.__init__` (:63-69): `self.skipped_has_telemetry: List[str] = []`;
  `as_dict` (:71-83): `skipped_has_telemetry_count` + `skipped_has_telemetry`.
- Truly-empty pending dirs (no ledger AND no telemetry) still get swept --
  BUG-LOCAL-290 stays fixed.

### F7. UTF-8, no BOM. SFW. No workflow JSON change.

No new node class -> no `NODE_CLASS_MAPPINGS`. No new widget -> no `INPUT_TYPES`
change -> no `widgets_values` positional shift (the BUG-LOCAL-097 class).
`_otr_llm_telemetry.py` is a private `_otr_*` module, not registered. Enablement is
an ENV VAR precisely so no widget is needed. **Run the section-0 ritual anyway**
(`OTR_WorkflowValidator` + JSON round-trip + link/widget audit) so "no JSON change"
is a CHECKED fact, not a comfortable assumption.

### F8. Scope: codex56sol lane ONLY

The `on_attempt` / `on_projection` seam in `structured_call` is lane-agnostic by
construction, but NO other lane passes a recorder in v1.

## 6. Change list

| # | file | change | ~lines |
|---|---|---|---|
| 1 | `nodes/_otr_llm_telemetry.py` | **NEW.** Stdlib only (no torch, no ComfyUI, no writer imports -- same purity class as the ladder). `TELEMETRY_SCHEMA_VERSION`, `telemetry_enabled()`, `sha256_text()`, `sha256_obj()`, `TelemetryRecorder` (open-append-close jsonl + content-addressed blob store + `_failopen` warn-once latch), `_NullRecorder`, `new_recorder()` | 250 |
| 2 | `nodes/_otr_structured_call.py` | `on_attempt` + `on_projection` kwargs (both default `None`); `_invoke_slot_observed()` helper (timing + backend-error emit + re-raise); 5 emit points; per-rung `_proj` wrapper injecting `attempt_idx`/`rung`; `on_projection` threaded to `validate_tolerant_data` for the P-C clamp ONLY; docstring note that **purity is preserved** | 70 |
| 3 | `nodes/_otr_original_codex56sol.py` | `_call(telemetry=_NULL)`; `_pending_call` + `_pending_projections` cells; observer built once after the seam read (:704); P-A/P-B queued projections; `accepted` at :818; `_validate_score_attempt(on_projection=...)` with a snapshot BEFORE :1090; derived `accepted` at both merge sites (:1412, :1634); thread `telemetry` through `_repair_score_grounding_intents` (:1392), `_repair_script_grounding_lines` (:1612), `_call_grounded_script` (:1637) + all 12 `_call` sites; `run_original_codex56sol_episode` **stops `del`-ing `episode_root`** (:1865), builds recorder + header, wraps the lane in `try/except/else/finally` | 120 |
| 4 | `nodes/OTR_LedgerScriptWriter.py` | `_SlotScheduler.__init__`: `entry_meta_by_slot = {}`; `_account_and_get_entry`: stash `{model_id, provider, context_cap}` | 8 |
| 5 | `nodes/_otr_pending_cleanup.py` | `_has_telemetry()`; `skipped_has_telemetry` bucket; checked inside both delete arms only | 20 |
| 6 | `tests/test_llm_telemetry.py` | **NEW** | 280 |
| 7 | `scripts/otr_telemetry_dump.py` | **NEW.** Given a `run_id` + `attempt_idx`, print messages / raw / projected / error from jsonl + blobs. It IS the offline-reader acceptance test's implementation -- write once, use twice | 60 |

**Five production files. If it grows a sixth, something went wrong.**

### The `else`-block trap (fixes a real Python bug in the draft plan)

`return` inside `try` **skips `else` entirely** -- the success `terminal` record
would never be written. Assign, then return AFTER the block:

```
try:
    ...P1..P9, ledger stamping, meta...
    parts = OriginalCodex56SolTailParts(...)
except Exception as exc:
    telemetry.record_terminal(status="aborted", error_type=type(exc).__name__,
                              error_str=str(exc))
    raise
else:
    telemetry.record_terminal(status="completed",
                              accepted_artifacts_sha256=_artifact_hashes)
finally:
    telemetry.close()
return parts
```

Terminal success is emitted **after the ledger stamping at :2005-2043 SUCCEEDED**,
from the `accepted_artifacts` map built at :2031-2041 -- the only place the semantic
keys exist. Returning from P9 is not the same as completing the lane; a
`stamp_receipt` failure is an abort.

### Already wired -- no plumbing needed (checked, not assumed)

- `_lane_runner(...)` (`OTR_LedgerScriptWriter.py:3739-3753`) **already passes**
  `episode_root=episode_root` and `episode_id=episode_id`. The lane just throws them
  away (:1865). The change is a DELETION. (Both panelists proposed stashing
  `episode_root` on the scheduler or the ledger; neither is needed.)
- `episode_root` EXISTS on disk before P1: `Ledger.__init__` does
  `os.makedirs(out_dir, exist_ok=True)` (`production_ledger.py:551-557`) on
  `otr/episodes/<ep>/audio`, so its parent exists by construction.
- **The ledger must not change by one byte.** `journal.append(...)` (:817) stays
  as-is; NOTHING is added to `meta["original_codex56sol"]` (:2018-2042).
  `stamp_receipt` / `validate_receipt` hash ledger content -- a new meta key would
  ripple into freeze-receipt tests for no benefit. Telemetry lives ONLY in its own
  files.
- Defensive reads throughout: `getattr(pack, "prompt_stages", {})`,
  `getattr(scheduler, "entry_meta_by_slot", {})` -- tests pass `SimpleNamespace`
  stubs, and the whole build is inside fail-open anyway.

## 7. Sequencing (commit AND push each green chunk to `v2.0-alpha`, same session)

1. **Module 5 first** (`_otr_pending_cleanup.py`). Imports nothing from the lane or
   telemetry; `_has_telemetry` is a pure filesystem predicate testable with a fake
   jsonl in `tmp_path`. It is the change that PROTECTS the data -- have it in place
   before any data exists.
2. **Module 1** (`_otr_llm_telemetry.py`) + unit tests. Pure, stdlib-only, no lane.
3. **Module 2** (`_otr_structured_call.py`) **ALONE**, with `on_attempt=None` /
   `on_projection=None` at every existing call site. **The full suite must be
   byte-identical green here.** Every lane routes through this module; if the suite
   moves, the seam is wrong and nothing downstream is worth building.
4. **Module 4** (scheduler meta) -- additive, 8 lines.
5. **Module 3** (lane wiring) -- the real change. Suite + Bug Bible.
6. Live 30-word smoke; then the fail-open, abort, and rename smokes.

## 8. Acceptance

**Regression suite + Bug Bible green after every chunk.**

New tests in `tests/test_llm_telemetry.py`:

- **`test_recorder_never_blocks_episode_dir_rename`** -- highest value-per-line test
  in the plan. Write a header, an attempt, and a blob, then `os.replace(ep_dir,
  final_dir)` MUST NOT raise. If anyone later "optimizes" back to a held handle,
  this fails loudly instead of a render dying six hours later.
- **`test_null_recorder_mirrors_full_api`** -- every public method of
  `TelemetryRecorder` exists on `_NullRecorder`. This is exactly how a null-object
  rots.
- **Raw-vs-projected:** a stubbed P5 response that trips P-A yields
  `raw_sha256 != projected_sha256`, both blobs resolve, and the dump tool shows them
  side by side.
- **Hash join, unpatched:** the P5 `accepted_sha256` equals the `broadcast_score`
  entry of `terminal.accepted_artifacts_sha256`.
- **Hash join, PATCHED:** force a P5 grounding patch and one script-line patch; the
  terminal hashes must match the DERIVED merged `accepted` records.
- **Projections:** one record each for P-A, P-B, P-C, P-D, each carrying `site`,
  `repair_fn`, `changed`, `before_sha256 != after_sha256`, `attempt_idx`, `rung`.
- **Fail-open (deterministic):** inject an `OSError` from the blob-store write ->
  ONE warning, green render, NO second warning. (A read-only DIRECTORY does not
  reliably block file writes on Windows -- do not build the automated test on that.)
- **Backend error:** `outcome="backend_error"`, `raw_sha256: null` (NOT the prior
  rung's raw), correct `latency_ms`, and the exception RE-RAISES.
- **Deterministic short-circuit:** the observer fires once with `llm_call: false`.
- **`telemetry_enabled()` precedence table**; record schema round-trip; blob dedupe
  hit path.
- **Provider shapes:** local HF, `gguf_native`, and one remote-shaped cache entry ->
  `model_commit_sha` is `Path(snap).name` for local HF and `null` otherwise, and no
  host path leaks into any record.
- **Pending sweep:** retains a telemetry-bearing dir, still deletes a truly-empty
  one, and does not double-book it in two report buckets.
- **Perf budget:** stub generate fn, assert telemetry overhead per call is under a
  generous ceiling (the real figure is < 10 ms; assert < 250 ms so the test is not
  timing-flaky on a loaded box). Both panelists wanted this test cut as flaky --
  keeping it, generously bounded, because the operator's constraint 3 explicitly
  asks for a perf assertion. The load-bearing invariants are the no-buffering, blob
  resolution, and rename tests.
- **Offline reader:** reconstruct one full attempt (messages + raw + projected +
  error) from the jsonl + blobs alone via `scripts/otr_telemetry_dump.py`. The stated
  goal is forensic replay -- prove it replays.

**Live smokes (Claude runs them; reset the box per CLAUDE.md section 4):**

- **30-word live smoke:** episode renders normally; `otr/episodes/<ep>/telemetry/`
  has a header, a `model_meta` per slot used, >= 1 attempt record per observed model
  call, an `accepted` per pass, and a terminal record; every referenced blob hash
  resolves. `Test-Path` the jsonl AND one blob before declaring success. Log the
  telemetry dir size. **Do NOT assert an exact attempt count against a live model** --
  the ladder is nondeterministic. Exact counts are asserted only against the stub.
- **Abort smoke:** force a contract error -> terminal says `aborted` with the type +
  pass_id, AND the pending sweep does not delete that episode dir.
- **Workflow check:** `OTR_WorkflowValidator` + JSON round-trip + link/widget audit
  against `workflows/otr_canonical.json` -- proving zero JSON change.

## 9. Cut from v1 (each proposed, each cut with a reason)

- **Token counts -- the FIELDS too, not just the values.** The slot fn returns a bare
  `str` (`_otr_structured_call.py:310-344`); the local generate fn computes prompt
  length and throws it away (`OTR_LedgerScriptWriter.py:664-707`); four remote
  providers are separate closures. Surfacing counts means changing the `GenerateFn`
  return contract across five backends -- its own change, not a rider. Always-`null`
  fields buy migration surface and no capture value. `telemetry_schema_version`
  exists precisely so they can be added later.
- **File locking.** The lane is a strict P1->P9 data-dependency chain and the refine
  loop is a sequential `for i in range(...)` calling `self.run`
  (`OTR_LedgerScriptWriter.py:3092-3131`). No threading, multiprocessing, or asyncio
  anywhere in the writer, and every refine candidate gets its own episode dir. A
  lock would guard against nothing and add a dependency.
- **Inlining payloads instead of a blob store.** See 4.
- **Cross-episode dedupe.** Forfeited by design; volume makes it pointless.
- **Any census / analytics / dashboard.** This change proves reliable capture.
  Nothing consumes the data yet. `otr_telemetry_dump.py` prints ONE attempt --
  the moment it grows aggregation it has become the census the non-goals cut, built
  on a schema that has never met real data.
- **`inspect.signature` sniffing on `post_validator`.** Proposed to guard a
  `TypeError` that this design never creates (3.4). A shim guarding a bug we do not
  have.

## 10. Non-goals, corrected

The problem statement said *"No capture of operator-personal data; payloads are
model prompts and outputs only."* **False as written:** P1's inputs include
`operator_hint` from `meta["source_meta"]` (:1872) -- text the operator typed -- and
it lands verbatim in the messages blob.

Reword: *captured payloads are the model prompts and outputs, which include any
`operator_hint` the operator supplied; no other operator or system data is
captured.* A labelling fix, not a redaction requirement -- the hint is part of the
prompt that produced the defect and belongs in the record.

## 11. One thing to expect

Four Python projections currently repair model output before acceptance. The moment
this telemetry exists, the honest question it raises is: *should any of these
projections exist at all, or are they masking a seam defect a better prompt would
fix?* That is the right question and it is why the operator wanted this -- but it
means the projection records are not neutral instrumentation, they are an
indictment list. **Expect the first real census to end in a seam-tuning sprint, not
an adapter.** That is a feature.

And it is why P-D is the most valuable record in the schema: a structural repair
hiding inside something named `_validate_*`.
