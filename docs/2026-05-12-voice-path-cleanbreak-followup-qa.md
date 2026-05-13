# Voice-Path Cleanbreak Follow-Up (S1-S5) — Reviewer QA

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**HEAD:** `d8eef4f` (origin and local in sync)
**Pre-S1 base:** `aa52415` (Sprint 1 already shipped before this doc)
**Plan-of-record:** `docs/voice-path-cleanbreak-execution-plan.md`

Earlier QA doc (`docs/2026-05-12-voice-path-cleanbreak-qa.md`) covered
P1–P3. This doc covers **S1–S5** of the follow-up plan. Focus is on
**design decisions made during execution that the plan didn't
explicitly call out** — the "judgment calls" that QA should scrutinize.

Five commits land on this doc:

| Sprint | Hash | Title | Net |
|---|---|---|---|
| S1 | `aa52415` | Defensive guardrails + quarantine list | +215 / −4 |
| S2 | `249bc06` | Director retirement (class delete + video migration) | +221 / −1527 |
| S3 | `b1bfe90` | Per-cue SFX duration (G7 + AudioGen + ProcSFX) | +351 / −10 |
| S4 | `6218de7` | Library-stub rename | +35 / −17 |
| S5 | `d8eef4f` | Doc discipline (gates / retirement log / QA promotion) | +251 / −5 |

---

## 1. Design decisions made during execution (the QA territory)

These are the calls that the plan-of-record was silent on. Each one
is a judgment call worth checking.

### D1 — meta.visual_plan shape (Sprint 2.1)

The plan said "mirror field-for-field at the ledger layer." It did not
specify the *content* fields. Sprint 2.1 picked this shape:

```jsonc
"meta": {
  "visual_plan": {
    "characters": {
      "<name>": { "portrait_prompt": "<character_description>" }
    },
    "scenes": [],
    "style":  "<resolved style slug>",
    "genre":  "audio drama"
  },
  "voice_assignments": {
    "<name>": {
      "voice_preset": "<cast.voice_preset>",
      "notes":        "<character_description>"
    }
  },
  "style": "<resolved style slug>"
}
```

**Decisions inside this:**

- `portrait_prompt` = the raw `character_description` from the cast
  contract. No `era_tail` / `style_tail` concatenation at stamp time —
  the downstream `otr_video_plan.compose_shot_prompt` appends those at
  render time.
- `scenes` is hardcoded empty `[]`. Outline beats aren't translated to
  visual scenes; OTRVideoPlan's `extract_scenes()` handles empty list
  via the existing 3-tier fallback chain.
- `genre` is hardcoded `"audio drama"`. The Director used to emit
  episode-specific genre strings ("sci-fi radio drama", etc) but the
  voice path only consumed this for HUD overlay text in
  SignalLostVideo. The hardcoded value mirrors the legacy fallback at
  `video_engine.py:701` (`plan.get("style", plan.get("genre", "sci-fi"))`).
- `notes` in `voice_assignments` mirrors `portrait_prompt` (same
  string). The Director used to emit distinct notes; today both surface
  the same description.

**Reviewer question Q-D1:** Is the `"audio drama"` hardcoded genre
acceptable, or should the writer surface a more specific genre string
derived from style (e.g. `"noir audio drama"` for `noir_interrogation`)?
The 3-tier portrait fallback in `otr_video_plan` already handles empty
genre gracefully, but the HUD overlay in `video_engine` reads it directly.

### D2 — OTRVideoPlan adapter strategy (Sprint 2.2)

The plan said "drop `production_plan_json` input socket. Read
`meta.visual_plan` from `script_json` (L3 ledger) via
`_otr_ledger_consumers.load_ledger`. Pattern matches the voice-side
migration in P2."

The straight migration would have meant rewriting every helper
(`build_pass1_char_prompts`, `build_pass2_scene_prompts`, `build_shot_plan`)
to take `script_json` directly + 9 test cases updated. Sprint 2.2 took
an **adapter approach** instead:

- The **socket** rename happened: `director_json` → `script_json`.
- The **helper signatures** stayed: all three `build_*` helpers still
  accept a `director_json` parameter name.
- The **node.plan() entry point** loads the ledger, extracts
  `meta.visual_plan` + `meta.voice_assignments` + `meta.style`,
  builds a legacy-director-shape dict, serializes to JSON, and passes
  it through to the helpers.

**Trade-off:** smaller diff + helper tests stay unchanged + the helpers
remain testable against legacy fixtures. **But:** a "legacy-director
shape" intermediate representation still exists in memory at render
time. The migration is at the boundary, not deep.

**Reviewer question Q-D2:** Is the adapter shape the right end state,
or should a future sprint push the L3 ledger read deeper into the
helpers (rename their `director_json` params + update fixtures)?

### D3 — SignalLostVideo inline plan dict (Sprint 2.3)

Same shape as D2 but smaller: `video_engine.py::render_video` builds a
4-key plan dict at the top of the function from `led.meta` and the
existing `plan.get("voice_assignments", ...)` / `plan.get("style", ...)`
sites read from it unchanged.

```python
plan = {
    "voice_assignments": _meta.get("voice_assignments") or {},
    "style":             _meta.get("style") or "",
    "genre":             (_meta.get("visual_plan") or {}).get("genre") or "",
}
```

**Trade-off:** minimal diff + defensive `.get()` pattern preserved.
**But:** the `plan` dict shape still implies a Director-like surface,
just constructed locally.

**Reviewer question Q-D3:** Same as Q-D2. Is the inline-dict adapter
acceptable, or should the read sites be refactored to read from
`led.meta` directly?

### D4 — Dead Director helpers left in place (Sprint 2.4)

The plan said "delete `LLMDirector` class. Keep file for `_unload_llm`
if any consumer imports it; otherwise delete file."

Sprint 2.4 deleted **the class** (927 lines) but left the orphan helper
functions in place:

- `DIRECTOR_PROMPT` (large constant)
- `DirectorJSONParseError`
- `_build_director_json_repair_prompt`
- `_validate_director_plan`
- `_randomize_character_names`
- `_generate_minimal_plan_from_voice_tags`
- `_DIRECTOR_SCHEMA`

A `CLEANBREAK SENTINEL` comment block at the deletion site documents
the scope of the pending prune. The helpers are unreferenced (grep
confirms) but physically still in `story_orchestrator.py`.

**Reviewer question Q-D4:** Was leaving the helpers (with the
sentinel) the right call, or should they have been pruned in the same
commit? Pro of keeping: smaller diff, easier review. Con: dead code
lingers; the next refactor has to read past 600 lines of un-reachable
code.

### D5 — G7 dur_s bounds vs AudioGen / ProcSFX clamps (Sprint 3)

G7 invariant: `0.25 <= dur_s <= 12.0`.
AudioGen reader clamp: `0.5 <= dur_s <= 10.0`.
ProcSFX reader clamp: `0.1 <= dur_s <= 10.0`.

These three windows **don't match**. A `dur_s = 11.0` passes G7 but
gets clamped to 10.0 by both consumers at render time. A `dur_s = 0.2`
fails G7 (writer-side reject), but if it slipped through to ProcSFX
it would still render at 0.2s (within ProcSFX's lower bound).

**Reasoning behind the asymmetry:**
- G7 = writer contract surface. The 0.25 / 12.0 is the practical range
  the writer's outline LLM should emit. Setting the bound here lets
  the outline prompt say "0.25 to 12 seconds" without lying.
- AudioGen 0.5 / 10.0 = the model's *quality* range (under 0.5s
  produces clicks; over 10s produces artifacts).
- ProcSFX 0.1 / 10.0 = procedural-synth-friendly window (synthesis
  works at very short durations; over 10s the texture stagnates).

**Reviewer question Q-D5:** Is the three-window asymmetry the right
shape, or should they all match (e.g. clamp everyone to G7's range)?
The current setup means G7 is the contract surface and the consumer
clamps are belt-and-braces — but the divergence might surprise a
debugger.

### D6 — Stub ledger fixture had unexpected `dur_s=0.8` (Sprint 3)

Discovered during S3 regression: `tests/fixtures/ledger_stub.py` had
`dur_s=0.8` pre-populated on its SFX line. Pre-S3 nothing read
`line.dur_s`, so the value was inert. Post-S3 AudioGen honors it,
which broke `test_audiogen_cache_hit_path` (the test pre-seeded the
cache at `default_duration=3.0`, not 0.8).

**Fix shipped in S3 commit:** the test's pre-seed updated to match
the fixture's actual `dur_s`. The fixture itself was not changed
(other tests already work against the 0.8 value).

**Reviewer question Q-D6:** Are there other fixtures in the test
tree that have `dur_s` pre-populated and didn't surface this issue
because they don't assert cache hits? Recommend a fixture-level grep
sweep to confirm.

### D7 — Workflow link ID 113 (skipping 111 + 112) (Sprint 2)

When wiring `FreezeCascade.script_json → OTRVideoPlan.script_json`,
Sprint 2.4 used link ID 113. IDs 111 + 112 were reserved by G5
(`tests/test_lfc_g5_no_pysssss_dependency.py` asserts those IDs
**never** reappear — they were the dropped `ShowText|pysssss` preview
links).

**Reviewer question Q-D7:** Is the "G5-reserved range" pattern
documented anywhere besides the G5 test file? A future workflow edit
that reuses 111 or 112 without checking would silently fail the G5
gate. Recommend either documenting the reserved range explicitly in
`docs/gates.md` or extending the G5 test to assert the link ID
allocator skips reserved values.

### D8 — Library-stub rename (Sprint 4)

The plan rejected "inline-fold of library stubs into their callers"
in favor of "rename + underscore prefix." Sprint 4 executed that.

**Decision NOT in the plan:** which underscore-suffix convention?
The names chosen were `_bark_lib.py` and `_sfx_lib.py` (underscore
**prefix** + `_lib` **suffix**). Alternatives considered:
- `_bark_helpers.py` / `_sfx_helpers.py`
- `_bark_internal.py`
- `bark_loader.py` (no prefix; descriptive of contents)

**Reviewer question Q-D8:** Is `_<topic>_lib.py` the right naming
convention for OTR's library-only stubs? Worth standardizing if more
arise (the `_otr_*` prefix pattern in other internal modules is a
sibling convention).

---

## 2. Wiring snapshot — final state

### Workflow JSON `workflows/otr_scifi_16gb_full.json`

**33 nodes, 57 links** (was 34 / 62 pre-sprint, 34 / 59 post-P3).

Cuts in this sprint batch:
- Link 1 (FreezeCascade.script_text → Director.script_text input)
- Link 17 (Director → SignalLostVideo)
- Link 38 (Director → OTRVideoPlan)
- Node id=2 (OTR_LLMDirector)

Adds:
- Link 113 (FreezeCascade.script_json → OTRVideoPlan.script_json)

Renames:
- OTRVideoPlan input socket `director_json` → `script_json`

Drops:
- SignalLostVideo input socket `production_plan_json`

### Final `FreezeCascade.script_json` fanout

```
FreezeCascade(62).script_json  -> 7 destinations
  link   2 -> SceneSequencer(3).script_json
  link  12 -> BatchBarkGenerator(11).script_json
  link  16 -> SignalLostVideo(12).script_json
  link  19 -> KokoroAnnouncer(13).script_json
  link  21 -> MusicGenTheme(14).script_json
  link  24 -> BatchAudioGenGenerator(15).script_json
  link 113 -> OTRVideoPlan(20).script_json    (new, Sprint 2)
```

Director is gone. All seven downstream consumers (voice + video) read
from the same canonical surface.

---

## 3. Verification recipes (paste-and-run)

```bash
# 1. Sprint-batch ship-gate greps
python - <<'PY'
import pathlib, re, subprocess
ROOT = pathlib.Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")

# Director gone from nodes + workflow
director_hits = list((ROOT / "nodes").rglob("*.py"))
import re
class_hit = sum(
    1 for p in director_hits
    if re.search(r'^class LLMDirector', p.read_text(encoding='utf-8'), re.MULTILINE)
)
print(f"LLMDirector class definitions: {class_hit} (expect 0)")

# Sprint 4 rename ship gate
nodes_dir = ROOT / "nodes"
old_modules_exist = (
    (nodes_dir / "bark_tts.py").exists()
    or (nodes_dir / "sfx_generator.py").exists()
)
new_modules_exist = (
    (nodes_dir / "_bark_lib.py").exists()
    and (nodes_dir / "_sfx_lib.py").exists()
)
print(f"old module files present: {old_modules_exist} (expect False)")
print(f"new module files present: {new_modules_exist} (expect True)")

# Workflow JSON shape
import json
wf = json.load(open(ROOT / "workflows" / "otr_scifi_16gb_full.json", encoding="utf-8"))
print(f"workflow JSON: {len(wf['nodes'])} nodes, {len(wf['links'])} links "
      f"(expect 33 / 57)")
director_present = any(n.get('type') == 'OTR_LLMDirector' for n in wf['nodes'])
print(f"Director node in workflow: {director_present} (expect False)")
PY

# 2. Targeted test suites (Sprint 1-3 + S5 docs aren't tested)
python -m pytest \
  tests/test_workflow_json_guardrails.py \
  tests/test_musicgen_style_palette.py \
  tests/test_freeze_cascade_g6.py \
  tests/test_per_cue_sfx_dur.py \
  tests/test_otr_video_plan.py \
  tests/test_audiogen_ledger.py \
  tests/test_procsfx_ledger.py \
  tests/test_bark_ledger.py \
  -q

# 3. Full regression (expect 6 quarantine failures only)
python -m pytest tests/ --ignore=tests/integration -q

# 4. Bug Bible regression (expect 23p / 1s / 2xf)
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q

# 5. Manual workflow load in ComfyUI Desktop
#    - Open workflows/otr_scifi_16gb_full.json
#    - Expect zero missing-node placeholders
#    - Expect OTRVideoPlan input slot 0 to be named "script_json"
#    - Expect SignalLostVideo to NOT have a production_plan_json input
#    - Expect no OTR_LLMDirector node anywhere on canvas
```

Expected: every grep agrees with the comment; every test suite passes
except the 6 known-failure entries in `docs/known-failures.md`.

---

## 4. Open questions for reviewer (where disagreement is the signal)

Eight items in §1 — Q-D1 through Q-D8. Each is a judgment call worth
a second opinion before the next round of work builds on it.

**Highest-priority for early feedback:**
- **Q-D5** (G7 vs AudioGen vs ProcSFX clamp asymmetry) — this is the
  most likely to surprise a debugger; an explicit decision would
  document the intent.
- **Q-D6** (other fixtures with unexpected pre-populated `dur_s`) — a
  fixture sweep before further regression-test work would prevent
  another `test_audiogen_cache_hit_path` style surprise.
- **Q-D4** (dead helper prune) — small, mechanical follow-up but
  shouldn't sit indefinitely; the CLEANBREAK SENTINEL block is a
  reminder, not a deferral.

**Lower-priority but worth flagging:**
- **Q-D1** (visual_plan content shape) — review when the video soak
  starts producing portraits; if FLUX output quality degrades, the
  hardcoded `"audio drama"` genre + empty scenes list become the prime
  suspects.
- **Q-D2 / Q-D3** (adapter strategy depth) — fine as-is, but a future
  reader will wonder why the legacy director-shape exists. Worth a
  one-paragraph note in the consumer files if not refactored.

---

## 5. Acceptance gates met

Pulled from each sprint's commit message. All checked at HEAD `d8eef4f`:

| Gate | Status |
|---|---|
| S1: Director-to-voice-link guardrail + palette-vs-pool drift test | PASSED (replaced in S2 with hard "Director-not-in-workflow") |
| S2: `grep -rn "OTR_LLMDirector" workflows/` | zero hits |
| S2: `__init__.py` does not register `OTR_LLMDirector` | confirmed |
| S2: workflow JSON 33 nodes / 57 links | confirmed |
| S3: G7 invariant errors in Phase 0 report; Phase 10 hard-fails | tested 10/10 cases |
| S3: AudioGen + ProcSFX honor line.dur_s with fallback | tested |
| S4: `grep -r "from .bark_tts\|from .sfx_generator"` returns zero in active code | confirmed |
| S5: gates / retirement / QA-promotion docs in place | confirmed |
| All sprints: 6 failed / 2042 passed (KNOWN-FAIL-001..006 only) | confirmed |
| All sprints: Bug Bible 23p / 1s / 2xf baseline held | confirmed |

---

## 6. What to send to the reviewer

This doc only. ~250 lines. Doesn't include the workflow JSON (47kB,
unchanged in shape from the prior QA), doesn't include the writer or
video module code (covered by the per-sprint commit messages already
in `git log`).

If the reviewer wants code-level inspection on a specific decision:

- D1 visual_plan shape: `nodes/OTR_LedgerScriptWriter.py` K.5 block
- D2 OTRVideoPlan adapter: `nodes/otr_video_plan.py::plan` head
- D3 SignalLostVideo adapter: `nodes/video_engine.py::render_video` head
- D4 dead helpers: `nodes/story_orchestrator.py` CLEANBREAK SENTINEL block
- D5 dur_s clamps: `nodes/_otr_ledger_freeze.py::_check_g7_sfx_dur_invariant` + the AudioGen + ProcSFX reader loops
- D6 stub fixture: `tests/fixtures/ledger_stub.py`
- D7 workflow link IDs: `workflows/otr_scifi_16gb_full.json` last_link_id
- D8 module naming: `nodes/_bark_lib.py` + `nodes/_sfx_lib.py` headers

---

**End.**
