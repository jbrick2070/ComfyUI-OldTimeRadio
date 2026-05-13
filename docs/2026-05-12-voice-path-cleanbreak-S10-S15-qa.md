# Voice-Path-Cleanbreak — S10 / S11 / S12 / S13 / S14 / S15 QA Document

**Date:** 2026-05-12
**Branch:** `v2.0-alpha`
**Latest commit covered:** `f813b37`
**Reviewers requested:** ChatGPT (gpt-4.1) first, Gemini (gemini-2.5-pro) second, Claude synthesizes.

---

## 0. What this QA round is for

The S6-S8 batch had its own round-robin (`docs/2026-05-12-voice-path-cleanbreak-S6-S8-qa.md`).
This doc covers the next 20 commits — the S10-S15 batch — that
implemented every locked decision from that prior round-robin
(F-1 through F-9, IMP-1 through IMP-9, Q-D9 through Q-D11) plus
one new sprint item (S11.6) that the S11.4 audit scheduled.

QA scope: please scrutinize the post-S8 wiring + mechanics for
bugs, accuracy issues, and possible improvements. Two items are
intentionally **deferred** (S14.2, S15.3) and listed in §7 with
their gating conditions; please vote on whether the deferral
windows are sized right.

| #  | Commit  | Sprint | Subject                                                                          |
|----|---------|--------|----------------------------------------------------------------------------------|
| 1  | 3090007 | S10.1  | G7 constants flow into both consumers (no magic numbers)                         |
| 2  | 55f52f4 | S10.2  | _resolve_genre raises empty + unknown; _preview_genre helper isolated            |
| 3  | 5363966 | S10.3  | docs/conventions.md test enforcement                                             |
| 4  | 53ed966 | S11.1+2| story_orchestrator docstring + comment housekeeping                              |
| 5  | 6ed0fd8 | S11.3  | rename _director_from_script_json -> _visual_plan_from_script_json               |
| 6  | 5f10188 | S11.4  | dict-shape mirror audit (SCHEDULES S11.6 flatten)                                |
| 7  | cdc176a | S11.5  | QA doc §2 diagram + typo fixes (F-8)                                             |
| 8  | 1a23976 | S11.6  | flatten _visual_plan_from_script_json projection, retire 'director' as local-var |
| 9  | c4ab258 | S12.1  | ProcSFX filename includes 8-char perm hash (F-6 fix)                             |
| 10 | 74c1f9f | S12.2  | AST-based ProcSFX import isolation guard (F-7 fix)                               |
| 11 | 574038e | S12.3  | AudioGen cache key 12-char + JSON-canonical + complete dimensions (IMP-1)        |
| 12 | 7ea481e | S12.4  | pin AudioGen cache hash truncation length to 12 (IMP-2)                          |
| 13 | badcae5 | S13.1  | cast contract rejects structural tokens (IMP-4)                                  |
| 14 | 02ca26c | S13.2  | G8 line_id uniqueness invariant (IMP-8)                                          |
| 15 | 7a7607a | S13.3  | Python-based fixture dur_s audit (IMP-9)                                         |
| 16 | 5652c7c | S14.1  | workflow contract validator (commit A: validator + test-only) (IMP-7)            |
| 17 | f813b37 | S15.1+2| known-failures with nodeid tracking + schema rewrite (Q-D11)                     |

Plus this QA doc as a separate commit.

---

## 1. Standing directives (now extended through S15)

Every directive from the prior round-robin still in force. S10-S15
extended the list to 12 directives total (per the S10-S15
execution plan §"Standing directives carry forward — extended this
round"):

1. No silent fallbacks on production surfaces.
2. ffprobe is ground truth for media-derived metadata; ledger is
   ground truth for cast / style / dur_s / structural fields.
3. Gates G1-G8 are the only enforcement points. (G8 added in S13.2.)
4. Cast is the only source of truth for voice data.
5. Director is dead everywhere.
6. Phase 0 collect / Phase 10 raise.
7. Clean-break commit category discipline (replacement+deletion
   together, OR pure deletion).
8. G7 bounds are honest with the writer (no magic numbers in
   widgets / clamps).
9. Cache keys include every output-determining input.
10. Renamed-but-keep-history filenames carry both old + new names
    in their provenance comment.
11. Deleted symbols don't survive as words in active code.
12. Structural invariants get structural tests (AST-walk over
    grep, frozenset constants over hardcoded literals).

If any reviewer finds a violation of any directive in this batch,
flag it as a **standing-directive breach**, severity HIGH.

---

## 2. State of the world (post-S15)

### 2.1 Voice-path wiring

Unchanged from the post-S8 diagram in `docs/2026-05-12-voice-path-cleanbreak-S6-S8-qa.md` §2.
Seven consumers (Bark / AudioGen / ProcSFX / Sequencer / MusicGen /
KokoroAnnouncer / SignalLostVideo) hang off FreezeCascade fanout.
OTRVideoPlan reads through SignalLostVideo. video_engine is the
leaf.

### 2.2 Cast contract surface (post-S13.1)

```
lock_cast()
    ├── _assert_unique_bark_voices(cast)             # voice preset collisions
    ├── _assert_voice_preset_invariant(cast)         # Gate 1: every Bark row has v2/* preset
    └── _assert_no_structural_tokens_in_cast(cast)   # NEW: name shape gate
```

Three independent assertions; each surfaces its own diagnostic on
violation. The structural-token gate uses
`_NON_CHARACTER_CAST_PATTERNS` (ported from the deleted
`story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS`, plus 5 new
exact-match patterns for TITLE/NOTE/TARGET/STYLE/NARRATOR).

### 2.3 FreezeCascade gates (post-S13.2)

```
G1  cast.char_id uniqueness
G2  line.line_id format
G3  speaker_role enum
G4  text non-empty
G5  reserved link IDs (workflow-side, not ledger-side)
G6  voice_preset cross-check (Gate 2 of cast contract)
G7  SFX dur_s in [SFX_DUR_MIN_S, SFX_DUR_MAX_S]
G8  line_id uniqueness across ledger.lines[]    # NEW in S13.2
```

G8 fires at Phase 0 (collect) and Phase 10 (raise) like every
other gate. ProcSFX's content-addressed filename
(`proc_<sfx_type>_<line_id>_<perm>.wav` per S12.1) and every ledger
write-back path now have a structural guarantee that line_ids are
unique.

### 2.4 Cache surfaces

| Surface | Identity hash | Inputs | Layer |
|---|---|---|---|
| AudioGen `_cache_prefix` | SHA-256 [:12] over JSON-canonical payload | prompt, duration_sec (3 dp), episode_seed, model_id, guidance_scale (2 dp) | S12.3 |
| MusicGen `_cache_prefix` | SHA-256 [:8] over canonical payload | cue_id, prompt, duration_sec (int), episode_seed | unchanged |
| ProcSFX filename | SHA-256 [:8] over canonical payload | dur_s (3 dp), chosen_type, line_id | S12.1 |

Note: MusicGen still on 8-char hash — reviewer should consider
whether the same IMP-1 layer-1 collision-risk fix should propagate
there. See §6.

### 2.5 Validator surfaces

`nodes/_workflow_validation.py` defines:

- 5 typed exception classes (Q-D9 vote: ValueError-rooted)
- 3 reserved sets: `G5_RESERVED_LINK_IDS`, `DELETED_NODE_TYPES`,
  `FORBIDDEN_INPUT_SOCKETS`
- `validate_workflow_contract(workflow, mappings, *, strict_unknown_types=False)`
  with 6 independent checks

Test-only invocation today; auto-invoke on workflow load (S14.2)
deferred one week per Q-D10.

### 2.6 Test infrastructure

`tests/conftest.py::EXPECTED_FAILED_NODEIDS` is the source of truth
for what failures are quarantine-allowed. The
`pytest_sessionfinish` hook fails the suite if any nodeid not in
the set fails (`SystemExit(2)`); prints `[KNOWN-FAIL-GUARD]
PROMOTABLE` if any expected-fail starts passing.

`tests/_helpers.py::load_all_ledger_fixtures` walks JSONs +
`make_*_ledger` factories under `tests/fixtures/`. Used by S13.3
SFX dur_s audit; available for any future structural fixture
audit.

---

## 3. Mechanics — please scrutinize

### 3.1 G7 constants flow (S10.1, 3090007)

```python
# nodes/_otr_ledger_freeze.py
SFX_DUR_MIN_S = 0.5
SFX_DUR_MAX_S = 10.0
__all__ extended with both names
```

```python
# nodes/batch_procedural_sfx.py + nodes/batch_audiogen_generator.py
from ._otr_ledger_freeze import SFX_DUR_MIN_S, SFX_DUR_MAX_S
# Widget min/max + per-cue clamp both reference the constants
```

**Bug-hunt prompts**

- Widget specs use `SFX_DUR_MIN_S` / `SFX_DUR_MAX_S` for `min` /
  `max` keys. Comfy widget spec parsing: does it require
  build-time literal floats or accept module-level imports? (It
  accepts imports — confirmed via Comfy's INPUT_TYPES contract;
  please double-check.)
- AudioGen had a dead `if i < len(sfx_plan):` branch (lines
  315-332 pre-edit) that was unreachable because `sfx_plan` was
  hardcoded to `[]` at line 264. S10.1 deleted both the branch
  AND the placeholder. Verify nothing in the codebase calls or
  references the deleted `sfx_plan` local — it shouldn't, but a
  forensic grep would confirm.
- The `test_consumer_clamp_constants_are_identical` test uses
  `is` (object identity) not `==` (equality). Catches a
  local-shadow refactor that defines `SFX_DUR_MIN_S = 0.5` inside
  the consumer file (would silently divergence on later edit).
  Verify the `is` check is right — a Python rebind via `import as`
  shouldn't fool it, but please confirm.

### 3.2 _resolve_genre raises (S10.2, 55f52f4)

```python
def _resolve_genre(style: str) -> str:
    if not (style or "").strip():
        raise ValueError(...)  # picker contract violation
    if style not in _GENRE_BY_STYLE:
        raise ValueError(...)  # palette drift
    return _GENRE_BY_STYLE[style]


def _preview_genre(style: str) -> str:
    """UI/demo helper. NEVER on production writer or freeze paths."""
    if not style:
        return "audio drama"
    if style in _GENRE_BY_STYLE:
        return _GENRE_BY_STYLE[style]
    return style.replace("_", " ") + " audio drama"
```

The isolation between `_resolve_genre` (production, fail-loud) and
`_preview_genre` (UI, best-effort) is enforced by an AST-walk test
that rejects `Call` sites of `_preview_genre` inside both
`OTR_LedgerScriptWriter.py` and `_otr_ledger_freeze.py`.

**Bug-hunt prompts**

- The AST guard catches `name(...)` and `module.name(...)` shapes.
  What about `getattr(module, "_preview_genre")(...)`? Or a
  reflective dispatch table with `_preview_genre` as a value? Both
  are unlikely in this codebase but theoretically slip through.
  Worth pinning?
- `_resolve_genre("   ")` (whitespace-only) raises with "empty
  style" message. Is that the right message, or should it
  distinguish "empty" from "whitespace-only"? Practical impact:
  zero (both indicate the picker contract broke). Cosmetic only.
- The `_GENRE_BY_STYLE` table has 10 entries matching
  `_STYLE_PICKER_SEED_POOL` exactly. The strict-equality drift
  guard (`test_genre_table_strict_equality_with_style_pool`)
  catches both missing AND orphaned entries. Verify that's
  symmetric in practice.

### 3.3 Conventions test enforcement (S10.3, 5363966)

`tests/test_naming_conventions.py` AST-walks every `nodes/_otr_*_lib.py`
and asserts none of them define `NODE_CLASS_MAPPINGS`. Plus a
glob-based test that any module ending `_lib.py` starts with
`_otr_`.

**Bug-hunt prompts**

- The AST test catches `NODE_CLASS_MAPPINGS = ...` at any scope
  (module-level, inside conditionals, inside try/except). Does it
  also catch `globals()['NODE_CLASS_MAPPINGS'] = ...`? No — that
  would be a reflective bypass. Is that bypass worth pinning?
  (Probably not; nobody writes that pattern by accident.)
- The doc table in `docs/conventions.md` lists the 5 currently-
  conforming `_otr_*_lib.py` modules. If a 6th is added without
  updating the doc, the test still passes (the test reads the
  filesystem, not the doc). Worth a doc-freshness check?

### 3.4 _visual_plan_from_script_json + flatten (S11.3 + S11.6)

The 4-key projection `{visual_plan, voice_assignments, style, genre}`
became the 5-key flat shape `{characters, scenes, voice_assignments,
style, genre}`. The two helper functions (`resolve_character_portrait`,
`extract_scenes`) and three build_* call sites read flat keys
directly; the local var `director` is gone everywhere in active
code (only forensic-comment references remain).

**Bug-hunt prompts**

- The helper `resolve_character_portrait` has a chain of
  defensive coercions for "characters as list", "characters[NAME]
  as list-of-dicts", "characters[NAME] as bare string" — all
  inherited from BUG-LOCAL-080 when LLMDirector emitted shape
  drift. Post-Director, the writer is the only producer of
  `meta.visual_plan.characters`. Are these coercions still
  needed? (Conservative answer: yes, defensive cost is one if
  per shape variant; LLM regressions are real.) Worth confirming.
- The flatten dropped one `test_extract_scenes_empty` case
  (`{"visual_plan": {}}`) — that case became meaningless when
  `visual_plan` is no longer a key. The `{"scenes": []}` case
  retained. Verify the deletion doesn't lose coverage of any
  meaningful path.

### 3.5 ProcSFX perm hash (S12.1, c4ab258)

```python
perm = hashlib.sha256(
    f"{cue_duration:.3f}|{chosen_type}|{line_id}".encode("utf-8")
).hexdigest()[:8]
fname = f"proc_{type}_{line_id}_{perm}.wav"
```

**Bug-hunt prompts**

- 8 hex chars = 32 bits. Same collision-math concern as
  AudioGen's pre-S12.3 8-char hash. Why 8 here when AudioGen
  went to 12? Rationale: ProcSFX filenames are scoped to ONE
  episode dir (no cross-episode collision risk). Cross-cue
  collision within an episode requires same `(dur_s, type, line_id)`
  triple — and `line_id` is itself unique per G8. So the only
  collision mode is two cues with literally the same id +
  type + duration, which is a writer bug not a hash failure.
  Reviewer: confirm that reasoning, or push to extend to 12
  for symmetry.
- The `cue_duration:.3f` format matches AudioGen's canonical
  encoding. Good — same float-arithmetic-stable behavior.
- The S12.2 AST guard prevents ProcSFX from importing AudioGen's
  cache module. But what if a future patch adds a ProcSFX-side
  cache helper that imports its own `hashlib`? The S12.2 guard
  is import-based; it won't fire. Worth a separate "ProcSFX
  doesn't define a `_cache_*` helper" test? S8.2 had this; it
  was retired in S12.2 as too rigid. Reviewer call.

### 3.6 AudioGen cache key (S12.3, 574038e)

Keyword-only signature:
```python
_cache_prefix(*, prompt, duration_sec, episode_seed, model_id, guidance_scale)
```

JSON-canonical payload:
```python
json.dumps({
    "duration_sec":   f"{float(duration_sec):.3f}",
    "prompt":         prompt,
    "episode_seed":   str(episode_seed),
    "model_id":       str(model_id),
    "guidance_scale": f"{float(guidance_scale):.2f}",
}, sort_keys=True, separators=(",", ":")).encode("utf-8")
```

12-char hex truncation; pinned by `test_audiogen_cache_hash_length_is_12`
(S12.4).

**Bug-hunt prompts**

- The keyword-only signature is a hard contract break. Any
  external caller (outside this repo) doing
  `AG._cache_key(prompt, dur, seed)` positionally now raises
  `TypeError`. Is that acceptable? OTR is a custom node, not a
  library — likely yes; flag if reviewer disagrees.
- `f"{float(guidance_scale):.2f}"` truncates to 2 decimal
  places. If a future user-facing widget exposes 3 decimals
  (`guidance_scale=3.123`), `3.12` and `3.13` collapse. The
  widget today uses `step: 0.5` so this is theoretical, but
  worth flagging.
- `str(episode_seed)` — what if `episode_seed` is a complex
  object (some nodes pass dicts?). `str()` on a dict produces a
  Python-repr string that's stable across runs (insertion-ordered
  in Py 3.7+) but not across Python versions. Worth pinning to
  always-string upstream? (Today the writer always emits a
  string; defensive coercion is cheap insurance.)

### 3.7 Cast contract structural-token guard (S13.1, badcae5)

`_NON_CHARACTER_CAST_PATTERNS` ports the deleted
`story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS` plus 5 new
exact-match patterns. Bug fix during port: original
`r"\bV\.O\.\b"` and `r"\bO\.S\.\b"` had trailing `\b` after `.`
that never matched (regex word-boundary doesn't fire after a
non-word char at end-of-string). Fixed by dropping the trailing
`\b`.

**Bug-hunt prompts**

- The original story_orchestrator pattern was BROKEN — it never
  caught `JOHN V.O.`. Anywhere ELSE in the codebase that uses
  similar `\b<chars>\.<chars>\.\b` patterns? Worth a sweep.
- Risk asymmetry rationale: false-positive (real character
  "Style" rejected) is far cheaper than false-negative (LLM
  hallucination ships). Patterns anchored as `^TITLE$` etc to
  minimize false positives on real names containing the
  substring. Verify "Anna Title-Holder" passes. (The test
  `test_cast_contract_helper_passes_real_names` covers 8
  representative names; please add any other concerning name
  pattern you can think of.)
- If a future legit story needs a character literally named
  "Style", the current path is reroll-the-cast. The plan
  recommends a case-sensitive whitelist instead of pattern
  relaxation. Reviewer: confirm whitelist is the right escape
  hatch when (not if) it comes up.

### 3.8 G8 line_id uniqueness (S13.2, 02ca26c)

```python
def _check_g8_line_id_uniqueness(ledger_data, errors, warnings):
    seen = set()
    duplicates = []
    for line in ledger_data.get("lines") or []:
        lid = line.get("line_id")
        if lid and isinstance(lid, str):
            if lid in seen:
                duplicates.append(lid)
            seen.add(lid)
    if duplicates:
        sample = duplicates[:5]
        more = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        errors.append(f"G8: {len(duplicates)} duplicate line_id(s) ...")
```

**Bug-hunt prompts**

- The check skips lines without `line_id` (per-line type
  invariants catch those). Verify the per-line invariants
  actually catch missing/empty line_id — if they don't, G8
  silently passes a malformed ledger. Spot-checked the upstream
  invariants; please double-check.
- Duplicates list capped at first 5 + `"(+N more)"` suffix.
  When N is exactly 6 (so 1 hidden), the suffix says "+1 more"
  — slightly awkward but accurate. Cosmetic only.
- The diagnostic mentions ProcSFX filename + ledger write-back
  paths as the load-bearing consumers. If a future consumer
  also keys by line_id, no test is going to fire on its
  absence — the diagnostic is informational. Worth maintaining
  a `LINE_ID_CONSUMERS` registry? Probably overkill.

### 3.9 Workflow contract validator (S14.1, 5652c7c)

Six checks; each raises a typed subclass of
`WorkflowValidationError` (which subclasses `ValueError`). The
`strict_unknown_types: bool = False` kwarg gates check #1 so test
environments with partial OTR registry loading still validate the
rest of the contract.

**Bug-hunt prompts**

- Check #3 (unwired-required input drift) uses the heuristic
  "`widgets_values` is non-empty". A node with 5 widget slots
  and only 1 filled-in value passes the heuristic. That's a
  loose check. Tighter version: positional indexing — match
  declared.required ordering against `widgets_values` list
  positions. Drift candidate. (Acknowledged in code as "tighter
  positional pinning is a future enhancement".)
- The validator's deep-INPUT_TYPES check only runs for
  OTR_-prefixed types. If a non-OTR node has a forbidden socket
  in its inputs[], check #5 still catches it (forbidden-socket
  check runs against all types). Verify.
- The reserved-set drift guards (`test_g5_reserved_link_ids_set_pinned`,
  `test_deleted_node_types_includes_director`, etc.) pin specific
  contents but don't pin ABSENCE of new entries. A future
  contributor adding `OTR_DeletedToday` to `DELETED_NODE_TYPES`
  doesn't fire any test. That's deliberate (additions are
  expected) — but if a prior entry gets accidentally REMOVED,
  the explicit-presence test fires. Confirm that's the right
  asymmetry.

### 3.10 Known-failures hook (S15.1, f813b37)

`pytest_sessionfinish` hook in `tests/conftest.py` diffs actual
failed nodeids against `EXPECTED_FAILED_NODEIDS`. Subset-run
guard: only enforces the diff when ≥80% of expected nodeids were
collected.

**Bug-hunt prompts**

- The 80% threshold is a heuristic. With 6 expected entries,
  80% = 4.8 ≈ 5 entries collected. So `pytest tests/` (full
  run) clearly enforces; `pytest tests/test_save_to_episode_workspace.py`
  (4 of the 6 expected entries) gets `4 < 4.8`, doesn't enforce
  — correct. `pytest tests/test_video_composite.py` (1 entry)
  also doesn't enforce — correct. `pytest tests/test_save_to_episode_workspace.py
  tests/test_video_composite.py` (5 entries) — DOES enforce.
  Edge case: would expectedly produce a PROMOTABLE banner for
  the unrun KNOWN-FAIL-001 (production_ledger). Is that
  desired? Probably yes — the user knows they're running a
  subset and can ignore.
- `SystemExit(2)` exits the session with code 2 (pytest's own
  exit code 1 = test failures, 2 = test execution interrupted).
  Choosing 2 distinguishes "regression detected" from "tests
  failed". Confirm pytest-using CI configs (e.g. GitHub Actions)
  treat exit 2 as a build failure (they do — any non-zero exit
  is a failure).
- The hook reads `item.rep_call.failed`. If a test errors during
  setup or teardown (not call), `rep_call` may not exist or
  `.failed` may be False even though the test produced an error
  result. The hook's `getattr(it, "rep_call", None)` defaults
  cleanly. Confirm error-during-setup tests don't leak past the
  diff as silent-passes.

---

## 4. Numerical invariants — please verify

| Invariant | Source | Pin | Drift guard |
|---|---|---|---|
| G7 SFX dur_s lower bound | `_otr_ledger_freeze.SFX_DUR_MIN_S = 0.5` | `test_g7_dur_s_below_min_raises` | `test_consumer_clamp_constants_are_identical` (S10.1) |
| G7 SFX dur_s upper bound | `_otr_ledger_freeze.SFX_DUR_MAX_S = 10.0` | `test_g7_dur_s_above_max_raises` | same |
| AudioGen hash digest length | `_cache_prefix` `[:12]` | `test_audiogen_cache_hash_length_is_12` (S12.4) | yes |
| AudioGen duration float precision | `f"{x:.3f}"` | `test_audiogen_cache_prefix_float_canonical` (S12.3) | yes |
| AudioGen guidance_scale precision | `f"{x:.2f}"` | implicit; not pinned | NO — drift candidate |
| ProcSFX hash digest length | `:8` | implicit; not pinned | NO — drift candidate |
| ProcSFX duration float precision | `f"{x:.3f}"` | implicit; not pinned | NO — drift candidate |
| MusicGen hash digest length | `:8` | implicit; not pinned | NO — drift candidate (out of scope this batch) |
| `_GENRE_BY_STYLE` size + symmetry | 10 entries == `_STYLE_PICKER_SEED_POOL` | `test_genre_table_strict_equality_with_style_pool` | yes |
| G5 reserved link IDs | `frozenset({111, 112})` | `test_g5_reserved_link_ids_set_pinned` | yes |

The "NO" rows in the drift-guard column are improvement candidates
in §6.

---

## 5. Test counts

| Sprint | Pre | Post | Δ | Notes |
|---|---|---|---|---|
| S10.1 | 2047 | 2052 | +5 | new `test_g7_consumer_constants.py` (5 tests) |
| S10.2 | 2052 | 2055 | +3 | net add in `test_musicgen_style_palette.py` (4 new − 1 deleted) |
| S10.3 | 2055 | 2057 | +2 | new `test_naming_conventions.py` |
| S11.1+2 | 2057 | 2057 | 0 | comment-only |
| S11.3 | 2057 | 2057 | 0 | rename-only |
| S11.4 | 2057 | 2057 | 0 | doc-only |
| S11.5 | 2057 | 2057 | 0 | doc-only |
| S11.6 | 2057 | 2057 | 0 | flatten + 1 test consolidated |
| S12.1 | 2057 | 2059 | +2 | perm-hash + source-guard |
| S12.2 | 2059 | 2059 | 0 | 1 test deleted, 1 added |
| S12.3 | 2059 | 2062 | +3 | model_id, guidance_scale, float-canonical |
| S12.4 | 2062 | 2063 | +1 | hash-length pin |
| S13.1 | 2063 | 2073 | +10 | structural-token rejection (5 parametrized + 5 sanity) |
| S13.2 | 2073 | 2080 | +7 | G8 invariant tests |
| S13.3 | 2080 | 2082 | +2 | fixture audit |
| S14.1 | 2082 | 2096 | +14 | validator (canonical + per-class + drift guards) |
| S15.1+2 | 2096 | 2096 | 0 | conftest hook (no tests, just infra) |

Total: 2047 → 2096 = **+49 net new tests** + ProcSFX/AudioGen test
refactors that consolidated some count.

KNOWN-FAIL count: 6 throughout. Bug Bible: 23/1/2 throughout.

---

## 6. Possible sight improvements (Claude's read; please vote)

Listed for round-robin signoff. Severity: BLOCKER, HIGH, MEDIUM, LOW.

### IMP-10 — ProcSFX hash truncation length pin

**Severity:** LOW
**Where:** `tests/test_audiogen_cache_keys.py` (or sibling)
**Proposed:** new test `test_procsfx_perm_hash_length_is_8` mirroring
the AudioGen pin from S12.4.
**Why:** Same drift class — silent change to the truncation
length is a cache-invalidation event.
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-11 — Extend MusicGen cache key with model_id + guidance_scale

**Severity:** MEDIUM
**Where:** `nodes/musicgen_theme.py`
**Proposed:** Same treatment as AudioGen got in S12.3 (12-char
hash + JSON-canonical + complete dimensions).
**Why:** MusicGen has the same model-id-blind cache risk AudioGen
had pre-S12.3. The S12 sprint deliberately scoped to AudioGen +
ProcSFX; MusicGen is the natural next target.
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-12 — Pin AudioGen guidance_scale precision

**Severity:** LOW
**Where:** `tests/test_audiogen_cache_keys.py`
**Proposed:** New test asserting the cache key uses 2 decimal
places for guidance_scale (`3.12` and `3.13` distinct, `3.121`
and `3.124` collapse to same key).
**Why:** Drift guard; if a future change extends to 3 dp, every
existing cached wav becomes unreachable.
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-13 — `_resolve_genre` empty-vs-whitespace diagnostic

**Severity:** LOW (cosmetic)
**Where:** `nodes/OTR_LedgerScriptWriter.py`
**Proposed:** Distinguish "empty style" from "whitespace-only style"
in the raised ValueError message.
**Why:** Easier diagnosis when the picker contract breaks via a
whitespace edge (LLM output starting/ending with space).
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-14 — Tighter widget-drift check (positional pinning)

**Severity:** MEDIUM
**Where:** `nodes/_workflow_validation.py` check #3
**Current:** Heuristic "`widgets_values` is non-empty" — accepts a
node with 5 widget slots and only 1 filled-in value.
**Proposed:** Match declared.required ordering against
`widgets_values` list positions.
**Why:** Real drift-catch coverage instead of presence-only.
**Risk:** ComfyUI's widget storage is implicit-positional; the
positional mapping isn't documented in `INPUT_TYPES()` -- it
follows declaration order. Confirming the order is stable across
ComfyUI versions is a design call. The current heuristic is
intentionally loose; tightening means signing up to track ComfyUI
internals.
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-15 — Sweep codebase for other broken `\b<chars>\.<chars>\.\b` patterns

**Severity:** LOW
**Where:** any `re.compile` / `re.search` site
**Proposed:** Grep for `\\b\\.\\b` patterns in regex strings;
audit each for the same end-of-string word-boundary bug S13.1
discovered.
**Why:** The original `story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS`
had the bug. Anywhere ELSE? Single audit script + doc finding.
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-16 — Doc-freshness check on `docs/conventions.md` table

**Severity:** LOW
**Where:** `tests/test_naming_conventions.py` (extension)
**Proposed:** Add a test that asserts every `_otr_*_lib.py` module
is named in the `docs/conventions.md` "current modules" table.
**Why:** A 6th lib module added without doc update silently
de-syncs the doc.
**Round-robin:** MERGE / DEFER / REJECT.

### IMP-17 — `episode_seed` type pinning

**Severity:** LOW
**Where:** `nodes/batch_audiogen_generator.py`
**Proposed:** Cast `episode_seed` to `str` defensively at the cache
key construction site (or pin "always-string upstream" via a test).
**Why:** Today the writer emits string. A future change passing
something complex (dict, etc.) gets `str()` representation that's
stable in Py 3.7+ insertion order but not bullet-proof.
**Round-robin:** MERGE / DEFER / REJECT.

---

## 7. Deferred items (gating + recovery)

### S14.2 — Validator auto-invoke on workflow load

**Status:** SCHEDULED
**Earliest ship:** one week after S14.1 (commit `5652c7c`,
2026-05-12) → **2026-05-19**
**Gating condition:** test-only mode false-positive count stays at
zero through the observation window.

**What ships:**
1. Edit wherever the workflow is loaded at runtime (likely
   `__init__.py` or a top-level loader) to call
   `validate_workflow_contract(wf, NODE_CLASS_MAPPINGS)` after
   `json.loads`.
2. Opt-out kwarg `validate=False` for callers that legitimately
   need to load malformed JSON for diagnostic purposes.
3. Verify across at least 5 hand-edit save cycles in ComfyUI
   editor that no false positive fires.

**Risk:** false positives during the observation week would
indicate the strict_unknown_types=False default isn't permissive
enough, or one of the deleted-node / forbidden-socket sets is
overzealous.

**Reviewer ask:** Is one week the right observation window? If
the false-positive-mode hasn't surfaced in a week, is that strong
enough signal to flip auto-invoke? Or push to 2 weeks?

### S15.3 — Survival-guide promotion of known-failures hook

**Status:** SCHEDULED
**Earliest ship:** after 2-3 sprints of OTR-scoped use of the
S15.1 hook surfaces any false-positive modes.
**Gating condition:** zero unhandled false-positive modes in OTR
usage; schema is stable; the hook + nodeid pattern earns its
promotion.

**What ships:**
1. Doc-only commit in `comfyui-custom-node-survival-guide` repo
   adding the hook + nodeid-tracking pattern as a recommended
   pattern.
2. Cross-link from OTR's `docs/known-failures.md` Promotion
   section to the survival-guide entry.

**Reviewer ask:** Is "2-3 sprints" the right deferral? Could be
shortened to "one full sprint of clean enforcement" if S14-style
quick-batch sprints continue.

### Other deferred items inherited from prior plans

- **Backend-specific duration fields** (`dur_s_audio_gen`,
  `dur_s_proc_sfx`) — out of scope for both S6-S15 batches; no
  confirmed SFX-under-0.5s use cases. Status: indefinitely
  deferred.
- **Helper-flatten refactor in `otr_video_plan.py`** — closed in
  S11.6 (was the original S11.4 audit's open question).
- **Reanimating any of the deleted Director helpers** — not
  scheduled; retrieve from git history at commit `83d7f17` or
  earlier if a use case ever arises.
- **Mechanical fallback in `_resolve_genre`** — isolated to
  `_preview_genre`; never on writer path (S10.2). Closed.
- **ComfyUI canvas-editor integration of `validate_workflow_contract`**
  — runtime-load auto-invocation (S14.2) is the supported gate;
  editor-side hook is a future enhancement. Not scheduled.

---

## 8. In-flight Q-D items needing signoff

None new this round. The S6-S8 round-robin's Q-D9, Q-D10, Q-D11
were all locked in the S10-S15 plan and shipped:

- **Q-D9** (validator exception hierarchy) — locked: 5 typed
  children, all `ValueError` subclasses. Implemented in S14.1.
- **Q-D10** (validator auto-invoke timing) — locked: half-measure,
  test-only commit A; auto-invoke commit B one week later if
  zero false positives. Commit A shipped (S14.1); commit B
  scheduled (S14.2).
- **Q-D11** (known-failures schema scope) — locked: nodeid SET
  not just count; OTR-scoped first; survival-guide promotion
  deferred. Implemented in S15.1+S15.2.

Reviewer-introduced new Q-D items welcome.

---

## 9. What we explicitly are NOT asking

(Same as prior round-robin; restated for completeness.)

- Do not propose work that requires API keys, cloud GPUs, paid
  services. Platform is 100% local.
- Do not propose reintroducing Director-shape adapters,
  re-export shims, or rename aliases.
- Do not propose Flash Attention 2/3 chasing.
- Do not propose backend-specific duration fields without a
  concrete sub-0.5s SFX use case.

---

## 10. Reproduce locally

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git log --oneline 89c56da..HEAD   # the 17 S10-S15 batch commits + this QA doc
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest `
  C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -q
```

Expected:

- OTR regression: 6 failed (KNOWN-FAIL-001..006), 2096 passed, 6 skipped
- Bug Bible regression: 23 passed, 1 skipped, 2 xfailed
- No `[KNOWN-FAIL-GUARD]` banner in stderr (confirms hook is happy)

---

## 11. Acceptance signal

After ChatGPT + Gemini have voted on the IMP-* candidates and
weighed in on the deferred-item windows:

1. Claude synthesizes votes into a follow-up commit (or sprint
   plan).
2. S14.2 lands on schedule (2026-05-19) if the false-positive
   gate stays clean.
3. S15.3 lands when the 2-3 sprint observation completes.
4. Approved IMP-* items become an S16 (or next-batch) execution
   plan.

Reviewers, thank you. Bug-hunt with prejudice; the pre-approved
disagreements (especially on the loose widget-drift check and the
ProcSFX vs AudioGen hash-length asymmetry) are exactly the kind
of structural call where a second opinion catches what implementer
momentum overlooks.
