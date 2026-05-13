# Voice-Path-Cleanbreak — S6 / S7 / S8 QA Document

**Date:** 2026-05-12
**Branch:** `v2.0-alpha`
**Latest commit covered:** `89c56da`
**Reviewers requested:** ChatGPT (gpt-4.1) first, Gemini (gemini-2.5-pro)
second, Claude synthesizes.

---

## 0. What this QA round is for

The S1-S5 sprint already had its own round-robin (the prior
`docs/2026-05-12-voice-path-cleanbreak-followup-qa.md`). Since
then, three more sprints have shipped (S6, S7, S8.1+8.2). Plan
items S8.3 and S9.1 are still pending. This doc is the new QA
ask: please scrutinize the post-S5 wiring + mechanics for bugs,
accuracy issues, and possible improvements **before S8.3 and S9.1
ship**.

The QA scope is the seven commits below, plus the in-flight plan
for the two remaining items.

| # | Commit  | Sprint | Subject                                                                          |
|---|---------|--------|----------------------------------------------------------------------------------|
| 1 | 47eb644 | S6-A   | genre table + drop persisted voice_assignments + deconstruct inline plan + QA typo |
| 2 | 6093182 | S6-B   | tighten G7 to (0.5, 10.0) consumer intersection                                  |
| 3 | b0810df | S6-C   | rip OTRVideoPlan adapter, ledger-derived helpers                                 |
| 4 | b6fb314 | S7.1   | delete dead Director helpers from story_orchestrator.py                          |
| 5 | ee1de3d | S7.2   | rename _bark_lib / _sfx_lib to _otr_-prefixed names                              |
| 6 | 89c56da | S8.1+2 | fixture audit + AudioGen cache-key drift guards                                  |

Plus in-flight (not yet shipped, listed in §6):

- S8.3 — workflow link-id validation module + 2 new test cases
- S9.1 — extend `docs/known-failures.md` schema + CI count guard

---

## 1. Standing directive (do NOT relax)

Quoted from project memory, reaffirmed at the top of every sprint:

> If code is called "legacy" but still runs, it isn't legacy — it's
> the current implementation, and the directive forbids that state.
> DELETE every legacy surface. No re-export shims, no
> `_RENAME_ALIASES`, no "both names stamped", no secondary input
> sockets kept for transition, no Director-derived fallbacks, no
> hardcoded legacy defaults, no `legacy_archive/` JSONs. Workflow
> JSONs are rewritten clean.

Two commit categories only:

1. **Replace + delete in the same commit** (no partials)
2. **Pure deletion** (when there's no replacement needed)

If any reviewer finds a violation in this batch, flag it as a
**S6-S8 directive breach**, severity HIGH.

---

## 2. Current voice-path wiring (post-S8 state)

```
                                     ┌──────────────────────┐
                                     │ OTR_LedgerScriptWriter│  ← writer
                                     │  (LPL, cast-locked)  │
                                     └──────────┬───────────┘
                                                │ script_json (L3 ledger)
                                                │  meta.style
                                                │  meta.visual_plan
                                                │  cast[]
                                                │  lines[]
                                                ▼
                ┌────────────────────────────────────────────────────────────┐
                │  consumers (each loads the ledger via                       │
                │  _otr_ledger_consumers.load_ledger, derives what it needs)  │
                └────────────────────────────────────────────────────────────┘
                  │           │           │           │           │
                  ▼           ▼           ▼           ▼           ▼
            ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
            │ Bark TTS│ │AudioGen │ │ ProcSFX │ │Sequencer│ │ MusicGen    │
            │(speech) │ │ (SFX)   │ │ (SFX)   │ │ (mix)   │ │ (open/close)│
            └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │OTR_VideoPlan │  ← video
                                                          │ (FLUX prompts)│
                                                          └──────────────┘
                                                                  │
                                                                  ▼
                                                       ┌──────────────────┐
                                                       │ video_engine     │
                                                       │ (HUD + treatment)│
                                                       └──────────────────┘
```

**What's gone vs. pre-cleanbreak:**

- `LLMDirector` node class (Sprint 2.4) — deleted
- Director JSON contract — every consumer now reads the L3 ledger
  directly, no Director shape anywhere
- `OTRVideoPlan.plan()` Director-shape adapter (Sprint 6.5 / commit
  b0810df) — deleted; helpers derive what they need via
  `_director_from_script_json(script_json)` at call time
- `meta.voice_assignments` persisted-by-writer field (Sprint 6.2) —
  retired; consumers call `voice_assignments_from_cast(led)` at
  render time
- `notes` field inside voice_assignments (Sprint 6.2) — retired;
  three-tier portrait fallback collapsed from 3 → 2 tiers (Tier 1
  portrait_prompt; Tier 2 removed; Tier 3 generic; renumbered
  internally but the test renames pin the retired-Tier-2 invariant)
- 357 lines of dead Director helpers in `story_orchestrator.py`
  (Sprint 7.1) — deleted
- `_bark_lib.py` / `_sfx_lib.py` module names (Sprint 7.2) — renamed
  to `_otr_bark_lib.py` / `_otr_sfx_lib.py`

**What's new in S6-S8:**

- `_GENRE_BY_STYLE` table in `OTR_LedgerScriptWriter.py` (10 entries,
  one per style preset; replaces hardcoded `"audio drama"` strings)
- `_resolve_genre(style)` helper with empty-string fallback
- `voice_assignments_from_cast(led)` in `_otr_ledger_consumers.py`
- G7 SFX `dur_s` bounds tightened from `(0.25, 12.0)` → `(0.5, 10.0)`
- `_director_from_script_json(script_json)` single derivation seam
  in `otr_video_plan.py`
- `docs/conventions.md` — new living doc codifying
  `_otr_<name>_lib.py` naming
- `docs/fixtures-audit-S8.md` — audit report
- `tests/test_audiogen_cache_keys.py` — 10 new drift-guard tests

---

## 3. Mechanics — please scrutinize each

### 3.1 Genre table (S6-A, 47eb644)

```python
# nodes/OTR_LedgerScriptWriter.py
_GENRE_BY_STYLE = {
    "closed_room_suspense":      "thriller audio drama",
    "detective_case_file":       "detective audio drama",
    "pulp_serial_cliffhanger":   "pulp serial audio drama",
    "mission_control_procedural":"procedural audio drama",
    "deep_space_distress_call":  "sci-fi audio drama",
    "noir_interrogation":        "noir audio drama",
    "small_town_uncanny":        "uncanny audio drama",
    "radio_newsroom_emergency":  "newsroom audio drama",
    "haunted_broadcast_signal":  "horror audio drama",
    "laboratory_containment":    "containment audio drama",
}

def _resolve_genre(style: str) -> str:
    if style in _GENRE_BY_STYLE:
        return _GENRE_BY_STYLE[style]
    words = (style or "").replace("_", " ").strip()
    return f"{words} audio drama" if words else "audio drama"
```

**Bug-hunt prompts**

- Is the table set complete vs. the active style palette? (10 slugs
  in palette; 10 entries in table; tests pin this.)
- Does the mechanical fallback (`{slug-words} audio drama`) produce
  reasonable copy for slugs the table doesn't know? Or does it
  generate awkward genre strings the LLM finds confusing?
- The auto-derive sentinel (`"let the story decide"`) is NOT in the
  table. Is that intentional? (Yes — the two-pass picker resolves
  the sentinel to one of the 10 slugs **before** `_resolve_genre`
  is called. Verify: the writer never asks for genre with the
  sentinel still in place.)
- Empty-string handling: `_resolve_genre("")` returns `"audio drama"`
  — is that the right default, or should it raise?

### 3.2 G7 SFX `dur_s` bounds tightening (S6-B, 6093182)

```python
# nodes/_otr_ledger_freeze.py
SFX_DUR_MIN_S = 0.5   # was 0.25
SFX_DUR_MAX_S = 10.0  # was 12.0
```

**Rationale:** loose union of consumer clamps (the old bounds) let
writer-time samples pass G7 only to get silently clamped at render
time. The intersection makes the contract honest.

**Bug-hunt prompts**

- Are there any fixtures or production paths emitting SFX `dur_s`
  in the gap regions `[0.25, 0.5)` or `(10.0, 12.0]`? The S8.1 audit
  found **zero** SFX-line fixtures in those regions; verify the
  audit script's regex is sound.
- Are the "old bound now fails" pins (`test_g7_dur_s_at_old_lower_bound_now_raises`,
  `test_g7_dur_s_at_old_upper_bound_now_raises`) checking the right
  thing? (They assert the boundary values `0.25` and `12.0` raise
  `FreezeAssertionError`, not just `warn`.)
- The drift-guard test `test_g7_bounds_match_consumer_intersection`
  asserts the constants match a manually-derived intersection. Is
  the consumer-side clamp logic (in AudioGen + ProcSFX) actually
  the source of those intersection values? Or are the constants
  drifting independently?

### 3.3 OTRVideoPlan adapter rip (S6-C, b0810df)

The adapter used to live in `OTRVideoPlan.plan()` — it took the
ledger upstream and rebuilt a Director-shape dict before handing
off to `build_pass1_char_prompts` / `build_pass2_scene_prompts` /
`build_shot_plan`. Now each helper derives the shape it needs
internally:

```python
def _director_from_script_json(script_json: str) -> dict:
    """Single derivation seam. Loads the L3 ledger, projects
    meta.visual_plan + cast + lines into the per-character +
    per-scene shape the three build_* helpers consume."""
    ...
```

**Bug-hunt prompts**

- The seam is a one-way projection. Round-tripping (writer → ledger
  → helper → render → ledger update) is fine because the helpers
  don't write back through this function. Verify that's actually
  true — no helper mutates the projected dict and stamps it back.
- Tier-2 portrait fallback (notes-from-voice_assignments) was
  deleted in this commit. Is Tier-3 (generic) reachable for every
  character without `portrait_prompt`? Or is there a path that
  used to hit Tier-2 and now hits something weirder?
- `resolve_character_portrait` now has the test name
  `test_portrait_tier_2_notes_fallback_is_retired` asserting Tier-2
  is dead. Does the assertion actually exercise a character whose
  voice_assignments USED to have a `notes` field? Or is the test
  vacuously passing because no character has `notes`?

### 3.4 Story orchestrator pure deletion (S7.1, b6fb314)

Deleted from `nodes/story_orchestrator.py` lines 4313-4669:

- `_DIRECTOR_SCHEMA`
- `DIRECTOR_PROMPT` (~150-line LLM prompt)
- `DirectorJSONParseError`
- `_build_director_json_repair_prompt`
- `_validate_director_plan`
- `_randomize_character_names`
- `_generate_minimal_plan_from_voice_tags`
- `_build_fallback_director_plan`
- `_looks_like_non_character_cast_name`
- CLEANBREAK SENTINEL comment block

Also deleted: `tests/test_prompt_format_safety.py` (its only target
was the deleted `DIRECTOR_PROMPT`).

**Bug-hunt prompts**

- Pre-deletion external-reference check (`findstr`) showed zero
  hits. Did it miss any indirect callers — e.g., a `getattr(module,
  "DirectorJSONParseError")` lookup, a string-based `eval`, a
  test that imports the module-level namespace via `import nodes.story_orchestrator as so` and pokes at `so._DIRECTOR_SCHEMA`?
- The deleted `_looks_like_non_character_cast_name` heuristic has
  no equivalent surviving anywhere. Is that capability needed by
  the new cast contract / `_otr_casting.py`? Or was it a Director-
  specific filter we're now obsolete?
- `test_prompt_format_safety.py` only tested `DIRECTOR_PROMPT`.
  Are any **surviving** LLM prompt templates susceptible to the
  same BUG-LOCAL-026 bug (literal `{}` in prose treated as
  positional format slots)? If yes, S8 should ship a replacement
  test covering them.

### 3.5 Library-module rename (S7.2, ee1de3d)

- `nodes/_bark_lib.py` → `nodes/_otr_bark_lib.py`
- `nodes/_sfx_lib.py` → `nodes/_otr_sfx_lib.py`
- 5 importer sites + 1 test patch target updated atomically
- `docs/conventions.md` NEW — codifies `_otr_<name>_lib.py` naming

**Bug-hunt prompts**

- Did any workflow JSON or `__init__.py` registration reference
  the old module names? (Spot-checked; none surfaced.)
- The convention says "leading underscore = package-private".
  Are there any uses of `from nodes._otr_bark_lib import ...`
  from **outside** `nodes/`? If yes, that's a convention violation
  the doc says we should clean up.
- `docs/conventions.md` mentions `_otr_ledger_consumers.py`,
  `_otr_ledger_freeze.py`, `_otr_casting.py` as following the
  convention. Verify each is in fact library-only (no
  `NODE_CLASS_MAPPINGS` export).

### 3.6 AudioGen cache-key invariants (S8.2, 89c56da)

```python
# nodes/batch_audiogen_generator.py
def _cache_prefix(prompt: str, duration_sec: float, episode_seed: str) -> str:
    payload = f"{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:20]).lower()
    return f"sfx_{safe_name}_{digest}"
```

**Bug-hunt prompts**

- The hash is SHA-256 truncated to 8 hex chars = 32 bits. Birthday
  collision probability across (say) 100 SFX cues is ~1.2e-6. Across
  a 50-episode catalog with 100 SFX/episode = 5,000 keys, the
  probability climbs to ~3e-3. **Is 8 hex chars enough?** Or should
  the prefix include 12-16 hex chars to give us 5-6 more orders of
  magnitude of safety against accidental cache poisoning?
- The payload format `f"{duration_sec}|{prompt}|{episode_seed}"`
  serializes `duration_sec` via Python's `str()`. Floats round-trip
  through `str` in a stable way **except** at edges (`str(0.1+0.2)`
  → `"0.30000000000000004"`). Is `duration_sec` ever the result of
  arithmetic that could expose this? Or is it always a writer-stamped
  value with a clean decimal representation?
- `safe_name = prompt[:20]` — what if two cues share a 20-char
  prefix? They distinguish via the 8-char digest, so functionally
  fine; just confirming the cosmetic part isn't the identity.

### 3.7 ProcSFX no-cache invariant (S8.2, 89c56da)

ProcSFX has no cache layer; output filename is
`proc_<sfx_type>_<line_id>.wav`. The S8.2 test
`test_procsfx_module_does_not_define_cache_lookup_helpers` is a
source-inspection guard against adding one without dur_s awareness.

**Bug-hunt prompts**

- If `dur_s` changes between runs for the same `line_id`, the wav
  gets overwritten with the new duration. Is **overwriting**
  correct, or should we keep the old one too for diff purposes?
- The filename includes `line_id` but not `sfx_type` permutation
  (it's `proc_<type>_<id>.wav`, with one `type` per `id`). If a
  writer ever emits two SFX lines with the same `line_id` but
  different `sfx_type`, what happens? (Verify line_id is
  guaranteed unique.)
- The source-inspection guard hardcodes 4 forbidden symbol names
  copied from AudioGen. If AudioGen renames `_cache_prefix` to
  something else, the guard goes silently blind. **Is hardcoding
  the symbol list the right approach?** Or should we instead
  assert ProcSFX defines **nothing** that imports from AudioGen's
  cache helpers?

---

## 4. Numerical invariants — please verify

| Invariant | Source | Pin |
|-----------|--------|-----|
| G7 SFX `dur_s` lower bound | `nodes/_otr_ledger_freeze.py` `SFX_DUR_MIN_S = 0.5` | 3 tests in `test_per_cue_sfx_dur.py` |
| G7 SFX `dur_s` upper bound | `nodes/_otr_ledger_freeze.py` `SFX_DUR_MAX_S = 10.0` | 3 tests in `test_per_cue_sfx_dur.py` |
| AudioGen hash truncation | `_cache_prefix` `[:8]` | implicit; not pinned |
| AudioGen cache filename ext | `_cache_filename_for_write` returns `<prefix>.wav` | `test_audiogen_cache_filename_extension_is_wav` |
| `_GENRE_BY_STYLE` size | 10 entries | `test_genre_table_covers_writer_style_pool` |
| `_otr_ledger_freeze` constants drift | manual derivation from AudioGen + ProcSFX clamps | `test_g7_bounds_match_consumer_intersection` |

**Improvement candidate:** the hash truncation length (`[:8]`) is
not pinned. If a future developer changes it to `[:6]`, no test
fires. Consider adding a drift guard.

---

## 5. Possible sight improvements (Claude's read; please vote)

Listed here for round-robin signoff. Severity scale: BLOCKER,
HIGH, MEDIUM, LOW.

### IMP-1 — Extend AudioGen hash digest length

**Severity:** MEDIUM
**Where:** `nodes/batch_audiogen_generator.py:96`
**Current:** `hashlib.sha256(payload).hexdigest()[:8]` (32 bits)
**Proposed:** `hashlib.sha256(payload).hexdigest()[:12]` (48 bits)
**Why:** birthday collision at 5,000 cues is ~3e-3 with 8 chars;
drops to ~1.8e-8 with 12 chars. Cosmetic cost on the filename: 4
extra chars. No back-compat issue because the cache is per-episode
and ephemeral.
**Risk if we do it:** existing cached wavs from prior runs become
unreachable (cache MISS once, regenerate; not data loss).
**Risk if we don't:** in the limit, a long catalog could see a
quiet cross-cue cache hit serving the wrong wav.
**Round-robin asks for:** vote MERGE / DEFER / REJECT.

### IMP-2 — Pin AudioGen hash truncation length

**Severity:** LOW
**Where:** `tests/test_audiogen_cache_keys.py`
**Proposed:** new test
`test_audiogen_cache_prefix_hash_length_is_pinned` asserting the
8-hex (or 12-hex if IMP-1 accepted) portion length.
**Why:** prevents silent change.
**Round-robin asks for:** MERGE / DEFER / REJECT.

### IMP-3 — Sharpen S8.1 audit heuristic

**Severity:** LOW
**Where:** `_s8_1_dur_audit.py` (session output dir; not in repo)
**Proposed:** add intentional-OOB tags (`_excluded`, `_non_sfx`,
`music_`, `_shot`, `invalid_timing`) so the manual second-pass
shrinks to zero.
**Why:** future audits run cleaner.
**Round-robin asks for:** MERGE / DEFER / REJECT.

### IMP-4 — Replace deleted `_looks_like_non_character_cast_name`?

**Severity:** MEDIUM (need to confirm need)
**Where:** would land in `nodes/_otr_casting.py`
**Why:** the heuristic filtered "TITLE", "NOTE", "TARGET", "STYLE",
"NARRATOR", etc. out of cast lists. If those tokens ever appear in
the cast contract output, the new contract has no defense.
**Round-robin asks for:** verify the new cast contract handles
these tokens **before** S9 starts; if not, file a follow-up.

### IMP-5 — `_resolve_genre` empty-string semantics

**Severity:** LOW
**Where:** `nodes/OTR_LedgerScriptWriter.py`
**Current:** `_resolve_genre("")` returns `"audio drama"`
**Question:** is that right, or should it raise / log a warning?
The auto-derive sentinel is supposed to be resolved upstream;
hitting `""` at this point means upstream contract broke.
**Round-robin asks for:** signoff on the silent-fallback policy
vs. fail-loud.

### IMP-6 — `docs/conventions.md` is not test-enforced

**Severity:** LOW
**Where:** would land as a new test
`tests/test_module_naming_conventions.py`
**Proposed:** scan `nodes/*.py` and assert: any file matching
`_*_lib.py` matches `_otr_*_lib.py`; any file matching
`_otr_*_lib.py` does NOT export `NODE_CLASS_MAPPINGS`; any file
that DOES export `NODE_CLASS_MAPPINGS` does not have a leading
underscore.
**Why:** the convention doc is honored today; without a test, the
next person adding a library module won't see the rule.
**Round-robin asks for:** MERGE / DEFER / REJECT.

---

## 6. In-flight (not yet shipped) — please vet before we land them

### 6.1 S8.3 — workflow link-id validation

**Plan:**

1. Create `nodes/_workflow_validation.py` with:
   - `G5_RESERVED_LINK_IDS = frozenset({111, 112})` constant
   - `validate_workflow_link_ids(workflow_dict)` → raises
     `WorkflowReservedLinkIDError` on any link reusing a reserved ID
   - `WorkflowReservedLinkIDError` exception class
2. Update `tests/test_lfc_g5_*.py` to import the constant (today
   it's a hardcoded literal in 3 places)
3. New tests:
   - `test_last_link_id_matches_max` (the workflow's `last_link_id`
     field is the max of any link's `id` value)
   - `test_no_reserved_link_ids_in_workflow` (current
     `otr_scifi_16gb_full.json` has no reused reserved IDs)

**Q-D items for the round-robin:**

- **Q-D9** — should `WorkflowReservedLinkIDError` inherit from
  `ValueError`, or be its own root? (Current bias: `ValueError`
  subclass so existing exception handlers catch it; reviewer may
  prefer fail-loud unhandled.)
- **Q-D10** — should the validator be invoked **automatically** on
  workflow load (e.g., via a hook in `__init__.py`), or stay
  test-only? (Current bias: test-only first; auto-load opt-in
  later. Reviewer may push for auto.)

### 6.2 S9.1 — known-failures schema + CI count guard

**Plan:**

1. Extend `docs/known-failures.md` schema with `expected_pass_count`
   and `last_verified_commit` fields.
2. Add `conftest.py` `pytest_terminal_summary` hook that asserts the
   live failed-count matches `len(known_failures) ± 0` — if a known
   failure suddenly passes (good!) the test suite tells you so you
   can promote and remove from quarantine. If a NEW failure appears
   (bad!) it surfaces as a hard error rather than getting absorbed
   into the quarantine.

**Q-D items for the round-robin:**

- **Q-D11** — should the hook be in the OTR repo's `conftest.py`,
  or moved upstream to the survival-guide repo so the Bug Bible
  regression follows the same discipline? (Current bias: OTR repo
  first; survival-guide later when the schema settles.)

---

## 7. What we explicitly are NOT asking

- Do not propose work that requires API keys, cloud GPUs, paid
  services. The platform is 100% local.
- Do not propose reintroducing Director-shape adapters, re-export
  shims, or rename aliases. Standing directive forbids them.
- Do not propose Flash Attention 2/3 chasing. RTX 5080 Laptop uses
  SDPA + SageAttention per CLAUDE.md.

---

## 8. Reproduce locally

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git log --oneline v2.0-alpha b22e418..HEAD   # the 6 S6-S8 commits
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest `
  C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -q
```

Expected:

- OTR regression: 6 failed (KNOWN-FAIL-001..006), 2047 passed, 6 skipped
- Bug Bible regression: 23 passed, 1 skipped, 2 xfailed

---

## 9. Acceptance signal

After ChatGPT + Gemini have voted on the IMP-* items and the
Q-D9..Q-D11 questions:

1. Claude synthesizes the votes into a follow-up commit
2. S8.3 + S9.1 land with the QA-blessed configuration
3. New QA doc closes the S6-S9 batch

Reviewers, thank you. Bug-hunt with prejudice.
