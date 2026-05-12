# Voice-Path Cleanbreak — Reviewer QA

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**HEAD:** `83d7f17` (origin and local are in sync)
**Pre-sprint base:** `1aed66d` (LFC commit 12.20 — last LFC-cleanbreak commit)
**Series:** 3 commits (`3efaed6` → `446ec81` → `83d7f17`)
**Net diff:** 33 files changed, **+1492 insertions / −2214 deletions = −722 lines**

**Premise:** The downstream voice path was retrofitted to the L3 ledger
during the consumer-rewrite sprint (read side), but the legacy
**secondary** paths — `production_plan_json` sockets, Director-derived
fallbacks, hardcoded era defaults in MusicGen, pre-L3 parser-list
readers, dead single-line node wrappers — were left in place "for
back-compat." This sprint deletes every one of them. Single canonical
data surface: the L3 ledger emitted by `OTR_LedgerFreezeCascade.script_json`.
Three-gate defense in depth for `cast.voice_preset`. No Director on the
voice side (Director itself is deferred — video-side still consumes
its `visual_plan`).

Paste this whole doc into ChatGPT + Gemini. Disagreements between them
are the signal.

---

## 1. What landed (one-line summary per commit)

| # | Hash | Title | Net |
|---|------|-------|-----|
| P1 | `3efaed6` | **MusicGen ledger-aware rewrite + Gate 1 + workflow rewire.** MusicGen reads `meta.gen_params_initial.style` + `meta.news.script_brief` from the L3 ledger via `_otr_ledger_consumers.load_ledger`. Deterministic `_STYLE_PALETTE` over the 10 active style slugs + `_MOOD_TAGS` keyword overlay + `_PROMPT_TAIL` instrumental-steerer. `CUE_DEFAULTS` deleted (no "1940s old time radio" anchors). Gate 1 (`_otr_casting._assert_voice_preset_invariant`) added at `lock_cast` exit. Workflow link 21 repointed Director→MusicGen ⟶ FreezeCascade.script_json→MusicGen.script_json. 21 new tests. | +901/−77 across 7 files |
| P2 | `446ec81` | **G6 Gate 2 + Bark Gate 3 + voice-side socket prune + workflow rewire.** G6 invariant in `_check_per_cast_invariants` (every non-ANNOUNCER cast row must carry a `v2/*` voice_preset; Phase 10 hard-fails on violation). Bark + SceneSequencer hard-raise on missing/non-v2 preset (Gate 3). `_voice_preset_for_character` helper + supporting state (`_BARK_VOICE_PRESETS`, `_FEMALE_PRESETS`, `_MALE_PRESETS`, `_CHARACTER_VOICE_CACHE`) deleted from Bark and Sequencer. `production_plan_json` socket + kwarg + parse deleted from Bark / Sequencer / AudioGen / ProcSFX. Workflow links 4 / 13 / 26 cut. 14 new tests. | +382/−395 across 14 files |
| P3 | `83d7f17` | **Legacy file removal + alias loop prune + workflow guardrails + test cleanup.** `bark_tts.py` → library-only (kept `_load_bark`, dropped `BarkTTSNode` class). `sfx_generator.py` → library-only (kept `SFX_GENERATORS` dict, dropped `SFXGenerator` class). `voice_render.py` deleted outright. `batch_kokoro_generator.py` deleted (pre-L3 parser-list reader; Bark is the production character path). `__init__.py`: drop 4 registrations + drop the `NodeName` bare-alias mirror loop. `tests/test_workflow_json_guardrails.py`: 3 new asserts (no production_plan_json wires to voice nodes / no production_plan_json sockets on voice nodes / MusicGen.script_json wired from FreezeCascade). 23 legacy-prune-residue tests retired in lockstep. | +209/−1742 across 14 files |

---

## 2. Wiring snapshot (post-`83d7f17`)

### Workflow JSON

`workflows/otr_scifi_16gb_full.json` — **34 nodes, 59 links** (down from 62).

Three Director-output links retired: link 4 (→ SceneSequencer), link 13 (→ Bark), link 26 (→ AudioGen). Link 21 repointed (was Director→MusicGen, now FreezeCascade→MusicGen). Two Director-output links intentionally retained pending the video sprint: link 17 (→ SignalLostVideo), link 38 (→ OTRVideoPlan).

### `OTR_LedgerFreezeCascade(id=62)` script_json output (slot 1)

```json
"name": "script_json",
"links": [2, 12, 16, 19, 21, 24]
```

**Six downstream consumers** all read the L3 ledger from the same source. Five of them are voice nodes; the sixth is `OTR_SignalLostVideo`, which dual-reads (see Director section below).

```
FreezeCascade(62).script_json --[2]--> SceneSequencer(3).script_json
FreezeCascade(62).script_json --[12]--> BatchBarkGenerator(11).script_json
FreezeCascade(62).script_json --[16]--> SignalLostVideo(12).script_json   (video-side; dual-read with link 17)
FreezeCascade(62).script_json --[19]--> KokoroAnnouncer(13).script_json
FreezeCascade(62).script_json --[21]--> MusicGenTheme(14).script_json   (new in P1)
FreezeCascade(62).script_json --[24]--> BatchAudioGenGenerator(15).script_json
```

**Amendment 2026-05-12 (Sprint 1.4):** earlier versions of this doc enumerated 5 consumers and missed link 16. SignalLostVideo's `script_json` input has been wired off FreezeCascade since the consumer-rewrite sprint; the `production_plan_json` input (link 17) is the surviving second source. SignalLostVideo therefore reads BOTH the L3 ledger AND the Director production plan today; the Sprint 2 plan reframes around collapsing this dual-read into a single ledger source by stamping `meta.visual_plan` + `meta.style` at writer/freeze time and dropping link 17.

### `OTR_LLMDirector(id=2)` production_plan_json output (slot 0)

```json
"name": "production_plan_json",
"links": [17, 38]
```

Voice side is fully disconnected from the Director output. The two remaining wires are both video-side and both targeted for retirement in Sprint 2:

```
Director(2).production_plan_json --[17]--> SignalLostVideo(12).production_plan_json   (dual-read; sibling of link 16)
Director(2).production_plan_json --[38]--> OTRVideoPlan(20).director_json
```

### Voice-node INPUT_TYPES (post-prune)

| Node | required | optional (kept) | optional (deleted in P2) |
|------|----------|-----------------|--------------------------|
| `OTR_BatchBarkGenerator` | `script_json` (L3 ledger) | `temperature` | ~~`production_plan_json`~~ |
| `OTR_SceneSequencer` | `script_json` (L3 ledger) | audio bus inputs + offsets + output_dir + default_tts | ~~`production_plan_json`~~ |
| `OTR_BatchAudioGenGenerator` | `script_json` (L3 ledger) | `episode_seed` + `model_id` + `guidance_scale` + `default_duration` | ~~`production_plan_json`~~ |
| `OTR_BatchProceduralSFX` | `script_json` (L3 ledger) | `default_duration` + `volume_db` | ~~`production_plan_json`~~ |
| `OTR_MusicGenTheme` | `script_json` (L3 ledger) **— new in P1** | `episode_seed` + `model_id` + `guidance_scale` | (never had a script_json socket; production_plan_json socket renamed + repointed in P1) |
| `OTR_KokoroAnnouncer` | `script_json` (L3 ledger) | `episode_seed` + `voice_override` + `speed` | (was already production_plan_json-free) |

---

## 3. Three-gate cast.voice_preset contract

Empty / non-`v2/*` `cast.voice_preset` on a non-ANNOUNCER row is a writer cast-lock contract violation. The cleanbreak adds three gates in defense-in-depth order. ANNOUNCER (Kokoro namespace, `bm_*` / `bf_*`) is excluded at every gate.

### Gate 1 — Writer (`nodes/_otr_casting.py::_assert_voice_preset_invariant`)

Called at `lock_cast` exit, right after `_assert_unique_bark_voices`. Raises `CastingFailedError` with attempt-list message identifying the offending `char_id`(s). Exported via `__all__` so Gates 2 + 3 can reuse the same logic.

### Gate 2 — FreezeCascade Phase 0 / Phase 10 (G6 invariant)

`nodes/_otr_ledger_freeze.py::_check_per_cast_invariants` walks `ledger.cast[]` and appends to `errors` (not `warnings`) on G6 violation. Phase 0 logs and advances; Phase 10 raises `FreezeAssertionError` on any critical-gap, including G6. The legacy WARN-fallback chain (`voice_preset` → `tts_model` → `speaker_role`) was deleted — a Kokoro-namespace id substituted in for a Bark-namespace preset is now a hard error.

### Gate 3 — Bark + Sequencer inline-Bark fallback

`nodes/batch_bark_generator.py` iterates ledger character lines and raises `ValueError` when `_OTRLC.voice_preset(led, line)` returns missing / non-v2/*. Error message identifies character_name + line_id + char_id + the bad value, plus a pointer that Gates 1 + 2 should have caught it upstream. `nodes/scene_sequencer.py` mirrors the same raise inside the inline-Bark fallback branch only (the announcer-Kokoro and pre-rendered-Bark paths don't read `voice_preset` so they don't fire the gate on Kokoro ids).

---

## 4. Six design questions for the reviewer (where disagreement is the signal)

### Q1 — Gate 3 in SceneSequencer: only on inline-Bark path?

The Sequencer's `voice_preset` lookup was moved INSIDE the inline-Bark fallback branch (lines ~723-748 of the post-P2 file). That keeps the announcer path (Kokoro pre-rendered) and the Bark pre-rendered path from tripping the gate on legitimately-Kokoro announcer presets. But it also means a misconfigured cast (e.g. character with empty `voice_preset`) doesn't fire Gate 3 in Sequencer **unless** the inline-Bark fallback is reached (which only happens when both pre-rendered audio buses are exhausted). Is that acceptable, or should Gate 3 in Sequencer fire eagerly at iteration time the same way it does in Bark?

### Q2 — MusicGen unknown-style hard-fail vs degrade

`_resolve_cue_from_style` raises `ValueError("unknown style slug 'X'. Add an entry to _STYLE_PALETTE. Known slugs: ...")` on any style slug not in the 10-entry `_STYLE_PALETTE`. The writer's style-picker today is constrained to the 10-slug `_STYLE_PICKER_SEED_POOL` so this should never fire in production; it's a defense against a future writer change that adds an 11th slug without updating the palette. **Is hard-fail correct here, or should it degrade to a documented fallback slug (e.g. `mission_control_procedural`) so MusicGen never blocks an episode?** The plan's §10.2 question on Bark's preset gate had the same shape; we landed on hard-fail for Bark. Same answer for MusicGen?

### Q3 — Deferred Director-class delete: scope creep risk

Director-class delete is deferred to a follow-up sprint because `SignalLostVideoRenderer` (`video_engine.py`) and `OTRVideoPlan` (`otr_video_plan.py`) are live readers of `production_plan_json.visual_plan / voice_assignments / style`. The voice path is structurally independent of this — link 17 + 38 still flow from Director to video-side, but no voice consumer touches `production_plan_json` anymore. **Two risks:** (a) Director still runs every workflow execution (one LLM call, ~2000 max_new_tokens, ~5–10s wallclock, ~2–4 GB VRAM peak); is that acceptable as ongoing cost for a deferred cleanup? (b) Director's `voice_assignments` is now generated and discarded — silent drift in the writer's cast contract could be masked by the fact that nothing fails when Director and cast disagree on a preset; is the G6 invariant + the Bark Gate 3 enough to catch divergence without any cross-check against Director output?

### Q4 — `production_plan_json` mentions in voice-file docstrings

The acceptance gate "grep -rn `production_plan_json` nodes/" returns 4 hits across voice files even post-P3. All four are explanatory comments saying "the legacy production_plan_json was deleted":

```
batch_audiogen_generator.py:248  # halts the run early. The legacy Director production_plan_json
batch_procedural_sfx.py:125      # halts the run early. The legacy Director production_plan_json
scene_sequencer.py:574           # halts the run early. The legacy Director production_plan_json
scene_sequencer.py:822           # inlined here after the production_plan_json socket deletion (P2).
```

These are documentation of the deletion, not active code. **Is the spirit of the no-back-compat directive satisfied, or should the comments be reworded to avoid leaving searchable references to the deleted surface?**

### Q5 — Per-cue SFX duration

P2 deleted the Director `sfx_plan[i].dur_s` per-cue override path in `batch_audiogen_generator.py`. Every SFX cue now renders at `default_duration` (default 3.0s, widget-overridable per workflow run but one value for the whole run). The plan-time per-beat duration that the Director used to provide is gone. **Is this an acceptable degrade, or does the writer's outline need to start stamping `lines[i].dur_s` (or a sibling field) on sfx-role beats so AudioGen can respect per-cue durations?** This is tracked as a deferred follow-up; reviewer agreement on the design path would unblock it.

### Q6 — Legacy node deletion symmetry

P3 deleted `voice_render.py` outright (no library functions) but kept `bark_tts.py` and `sfx_generator.py` as library-only modules because `batch_bark_generator.py` imports `_load_bark` and `batch_procedural_sfx.py` imports `SFX_GENERATORS`. **Is the "library-only stub" treatment acceptable, or should `_load_bark` and `SFX_GENERATORS` be moved into the consumer modules (or a dedicated `_voice_lib.py` / `_sfx_lib.py`) so the deleted node classes don't leave any module-shaped trace?** The current shape:

```
nodes/bark_tts.py        — 11,180 bytes, library-only (no class, no node registration)
nodes/sfx_generator.py   — 8,790 bytes, library-only (no class, no node registration)
```

---

## 5. Acceptance criteria

For the series to be accepted, both reviewers should agree:

1. **Acceptance gate greps**

   | Check | Expected |
   |---|---|
   | `production_plan_json` in voice-side `nodes/*.py` | 4 hits, all in explanatory comments (Q4 above) |
   | `_voice_preset_for_character` in `nodes/batch_bark_generator.py` + `nodes/scene_sequencer.py` | 2 hits, all in explanatory comments |
   | Era literals (`1940s`, `vintage`, `old time radio`, `warm brass`, `upright bass`) in `nodes/musicgen_theme.py` `_STYLE_PALETTE` + `_MOOD_TAGS` block | **zero hits in active data** (5 hits exist in the module docstring as a forbidden-list reference; verify those are not in the palette itself) |
   | `production_plan_json` wires to any voice node in `workflows/otr_scifi_16gb_full.json` | zero |
   | `OTR_BarkTTS` / `OTR_SFXGenerator` / `OTR_VoiceRender` / `OTR_BatchKokoroGenerator` registrations in `__init__.py` | all four absent |
   | `OTR_LLMDirector` registration in `__init__.py` | present (deferred) |

   Verification recipe at the end of this doc.

2. **Workflow JSON guardrail tests (P3)**

   ```bash
   python -m pytest tests/test_workflow_json_guardrails.py::TestVoicePathCleanbreakWiring -v
   ```

   Three asserts, all PASS:
   - `test_no_production_plan_json_wires_to_voice_nodes`
   - `test_voice_nodes_have_no_production_plan_json_input_socket`
   - `test_musicgen_script_json_wired_from_freeze_cascade`

3. **Voice-path tests across the sprint**

   ```bash
   python -m pytest tests/test_musicgen_style_palette.py \
                    tests/test_musicgen_news_brief_used.py \
                    tests/test_writer_cast_lock_voice_preset.py \
                    tests/test_freeze_cascade_g6.py \
                    tests/test_bark_cast_contract.py \
                    tests/test_bark_ledger.py \
                    tests/test_audiogen_ledger.py \
                    tests/test_procsfx_ledger.py \
                    tests/test_sequencer_ledger.py \
                    tests/test_workflow_json_guardrails.py \
                    tests/test_lfc_c3_phase_telemetry.py \
                    tests/test_lfc_commit_12_1_fixes.py
   ```

   Expected: ~140 passed / 5 skipped / 0 failed. The 5 skipped are pre-existing torch-conditional cases in `test_workflow_json_guardrails.py`.

4. **Full pytest baseline**

   ```bash
   python -m pytest tests/ --ignore=tests/integration
   ```

   Expected: **2033 passed / 7 skipped / 6 failed**. The 6 failures are all pre-existing, NONE introduced by this sprint:

   - `test_production_ledger.py::test_save_merges_schema_l3_fields_from_disk` (KeyError 'phase_ms')
   - `test_save_to_episode_workspace.py` × 4 (NaN tensor upstream regression)
   - `test_video_composite.py::test_default_canvas_is_native_832x480_at_25fps` (canvas-size assertion drift)

   Pre-sprint baseline was 29 failures (23 legacy-prune-residue + 6 unrelated). This sprint retired the 23.

5. **Bug Bible regression**

   ```bash
   python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v
   ```

   Expected: **23 passed / 1 skipped / 2 xfailed**. Baseline. Held across all three commits.

6. **HEAD parity + branch posture**

   ```bash
   git rev-parse HEAD             # → 83d7f17
   git rev-parse origin/v2.0-alpha # → 83d7f17 (matches local)
   git log --oneline 1aed66d..HEAD # → 3 commits: 3efaed6, 446ec81, 83d7f17
   ```

   No force-push. No `main` touched. Tag `v2.0-alpha-cleanbreak` still points at `f582a38` (its prior LFC-sprint anchor); a re-tag at `83d7f17` is a Jeffrey decision pending soak.

---

## 6. Status of follow-up items

Originally listed as "known deferrals" pre-cleanbreak. Updated 2026-05-12
end-of-sprint-batch:

| Item | Status | Tracking |
|---|---|---|
| Director-class delete + workflow Director-node removal | **SHIPPED Sprint 2** (commit `249bc06`). `meta.visual_plan` + `meta.voice_assignments` + `meta.style` stamped at writer cast-lock exit; `OTR_VideoPlan` + `OTR_SignalLostVideo` migrated to read from L3 ledger. Workflow now 33 nodes / 57 links. | `tests/test_workflow_json_guardrails.py::TestVoicePathCleanbreakWiring::test_no_llm_director_in_workflow` |
| Per-cue SFX duration via writer outline | **SHIPPED Sprint 3** (commit `b1bfe90`). G7 invariant in FreezeCascade enforces `[0.25, 12.0]` bounds; AudioGen + ProcSFX honor `line.dur_s` with default fallback. Writer-side outline-prompt extension (asking the LLM to emit per-beat dur_s) is the remaining piece -- still deferred. | `tests/test_per_cue_sfx_dur.py` (10 cases) |
| Library-stub rename | **SHIPPED Sprint 4** (commit `6218de7`). `bark_tts.py` -> `_bark_lib.py`; `sfx_generator.py` -> `_sfx_lib.py`. All 5 import sites + 1 mock.patch path updated. | grep -r "from .bark_tts\|from .sfx_generator" -> zero hits in active code |
| Doc discipline | **SHIPPED Sprint 5** (this sprint). `docs/gates.md` (Gate 1 + 2 (G6/G7) + 3A/3B naming convention), `docs/test-retirement-log.md` (23-test retirement table for P3 + Sprint 2 deletes), this Q5 promotion. | this doc |
| 6 unrelated pre-existing test failures | Quarantined in `docs/known-failures.md`. KNOWN-FAIL-001 through KNOWN-FAIL-006. Not voice-path-related (phase_ms KeyError, save-to-workspace NaN, video_composite canvas size); explicit non-blockers per the quarantine contract. | `docs/known-failures.md` |
| Writer outline prompt -- emit per-beat dur_s for SFX | **DEFERRED**. Sprint 3 enabled the consumer surface; the LLM-prompt change to populate it requires its own LLM-quality testing pass + outline-schema extension. The L3 ledger schema slot is open and back-compatible. | None until follow-up sprint scope. |

---

## 7. Verification recipe (paste-and-run)

```bash
# 1. Acceptance gate greps (excludes explanatory comments in voice-side files;
#    expects zero hits in active code/data and acknowledges the comment-only hits)
python - <<'PY'
import pathlib, re, json
ROOT = pathlib.Path("C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio")
VOICE_FILES = {
    "batch_bark_generator.py", "batch_audiogen_generator.py",
    "batch_procedural_sfx.py", "scene_sequencer.py",
    "musicgen_theme.py", "kokoro_announcer.py",
}
# Voice files: only comment refs to production_plan_json allowed.
for p in (ROOT / "nodes").glob("*.py"):
    if p.name not in VOICE_FILES: continue
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if "production_plan_json" in line and not line.strip().startswith("#"):
            print(f"FAIL non-comment hit: {p.name}:{i}: {line.strip()}")

# MusicGen palette+mood block: zero era literals in actual data.
src = (ROOT / "nodes" / "musicgen_theme.py").read_text(encoding="utf-8")
block = src[src.index("_STYLE_PALETTE"):src.index("MUSICGEN_MODEL_ID")].lower()
for term in ("1940s", "vintage", "old time radio", "warm brass", "upright bass"):
    if term in block:
        print(f"FAIL era literal in palette: {term}")

# Workflow JSON: no production_plan_json wires to voice nodes.
wf = json.load(open(ROOT / "workflows" / "otr_scifi_16gb_full.json", encoding="utf-8"))
nodes = {n["id"]: n for n in wf["nodes"]}
voice_types = {"OTR_BatchBarkGenerator","OTR_KokoroAnnouncer","OTR_BatchAudioGenGenerator",
               "OTR_BatchProceduralSFX","OTR_SceneSequencer","OTR_MusicGenTheme"}
for L in wf["links"]:
    if L[3] in nodes and nodes[L[3]].get("type") in voice_types:
        ins = nodes[L[3]].get("inputs", [])
        if 0 <= L[4] < len(ins) and ins[L[4]]["name"] == "production_plan_json":
            print(f"FAIL workflow has production_plan_json wire to voice node: link {L[0]}")

# __init__.py: deleted node names absent, Director kept.
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
for name in ("OTR_BarkTTS","OTR_SFXGenerator","OTR_VoiceRender","OTR_BatchKokoroGenerator"):
    if re.search(rf'^\s*"{name}"\s*:', init, re.MULTILINE):
        print(f"FAIL {name} still registered")
if not re.search(r'^\s*"OTR_LLMDirector"\s*:', init, re.MULTILINE):
    print("FAIL Director registration unexpectedly removed (P3 should keep it)")
print("acceptance-gate-grep DONE (any FAIL above is a real issue)")
PY

# 2. Voice-path tests
python -m pytest tests/test_musicgen_style_palette.py \
                 tests/test_musicgen_news_brief_used.py \
                 tests/test_writer_cast_lock_voice_preset.py \
                 tests/test_freeze_cascade_g6.py \
                 tests/test_bark_cast_contract.py \
                 tests/test_bark_ledger.py \
                 tests/test_audiogen_ledger.py \
                 tests/test_procsfx_ledger.py \
                 tests/test_sequencer_ledger.py \
                 tests/test_workflow_json_guardrails.py \
                 tests/test_lfc_c3_phase_telemetry.py \
                 tests/test_lfc_commit_12_1_fixes.py -q

# 3. Full pytest (expect 2033 passed / 7 skipped / 6 pre-existing failed)
python -m pytest tests/ --ignore=tests/integration -q

# 4. Bug Bible regression baseline (expect 23p / 1s / 2xf)
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q

# 5. Workflow JSON loads in ComfyUI Desktop (manual)
#    - Open workflows/otr_scifi_16gb_full.json
#    - Expect zero missing-node placeholders
#    - Expect MusicGen.script_json socket wired in (visible)
#    - Expect no production_plan_json sockets on Bark/Sequencer/AudioGen/ProcSFX
```

---

## 8. What to send

Just this doc. ~280 lines. The workflow JSON is 47 kB; most of it is unchanged HuMo / LTX / FLUX / Visual nodes that have nothing to do with this sprint.

If a reviewer pushes for code-level inspection on a specific commit, give them the commit hash plus one or more of:

- `nodes/musicgen_theme.py` — P1, palette + rewrite (the largest single behavior change)
- `nodes/_otr_casting.py::_assert_voice_preset_invariant` — P1 Gate 1
- `nodes/_otr_ledger_freeze.py::_check_per_cast_invariants` — P2 Gate 2 (G6)
- `nodes/batch_bark_generator.py` — P2 Gate 3 + helper deletion
- `nodes/scene_sequencer.py` — P2 mirror of Bark's deletion + inline-Bark gate
- `__init__.py` — P3 registration + alias loop pruning
- `tests/test_workflow_json_guardrails.py` — P3 new asserts
- `workflows/otr_scifi_16gb_full.json` — the canonical surface

---

## 9. Background reading (optional, for context)

- `docs/2026-05-12-voice-path-audit.md` — the audit that opened the sprint (read-only; the §0 standing-directive section and §6 fix-list table are the load-bearing parts)
- `docs/voice-path-cleanbreak-plan.md` — the round-robin plan that named the three phases + the three gates
- `docs/2026-05-12-lfc-cleanbreak-qa.md` — the prior sprint's QA doc (parallel structure, useful for tonal calibration if a reviewer is unfamiliar with the project's review format)

---

**End.**
