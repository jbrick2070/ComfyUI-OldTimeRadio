# Sprint C retrospective triage -- consolidated findings

**Triage branch:** `triage-sprint-c-retrospective-2026-05-15` (cut from `main@0aa6d6e`)
**Captured:** 2026-05-16
**Inputs verified against:** `workflows/otr_scifi_16gb_full.json`, `nodes/_otr_story_brief.py` (sprint-c blob), `nodes/*.py` INPUT_TYPES, `requirements.txt`, `pyproject.toml`, `BUG_LOG.md`, `SPRINT.md`, `docs/closed-sprints/2026-05-15-sprint-c-story-brief-v2.md`, `docs/AI_Production_Pipeline_Retrospective__Sprint_C.md`.
**Mode:** read-only on existing repo files. Only new files created under `docs/retrospectives/`. Specs only -- no source modifications, no test additions, no commits to sprint branches.

---

## TLDR

| § | Finding | Verdict | Lands at |
|---|---|---|---|
| 1 | Null-state padding violation (retrospective §6) | **REFUTED** -- the empty strings / `'[]'` / `'{}'` are the BUG-LOCAL-032 canonical preserved-mode fix, not dummy data. Retrospective inverts cause and effect. | Sprint A acceptance row SA-100 reframes this as a schema-positive canonical-shape gate, NOT a "reject empty strings" gate. |
| 2 | Silent temperature clamp in reflection module (retrospective §1 / §2 / §5 / §7) | **PARTIAL** -- the silent-clamp observation is REAL (no log line between clamp computation at line 487 and LLM call at line 494 in `nodes/_otr_story_brief.py`). The "clamps to 0.55" framing is mathematically incomplete (`min(temp+0.15, 0.55)`, not flat-0.55). The "CRITICAL exception class" framing is a prompt-string artifact, NOT an exception class. | Sprint A acceptance row SA-101 lands one `log.info` line + 2 pytest tests at the first runtime-verification commit. |
| 3 | b3sum hardware-determinism gap (retrospective §7) | **REAL and actionable** -- repo currently has zero machine-readable record of the floating-point-affecting environment. Cross-hardware risk is theoretical for this operator (single RTX 5080 Laptop), but time-axis env drift on the same machine is the real risk. | Sprint A acceptance row SA-102 lands `tools/capture_hardware_snapshot.py` + `tests/fixtures/hardware_snapshot.json` baseline. SA-104 lands the tier-2 perceptual audio hash supplement. |
| 4 | VRAM telemetry during multi-model regression (retrospective §7 Surface Metric Bias) | **REAL** -- existing S-A.4 spec uses standard timing metrics only; misses VRAM fragmentation accumulation. | Sprint A acceptance row SA-103. |
| 5 | NUL padding in workflow JSON | **FALSE ALARM** -- sandbox/mount read artifact, not on-disk corruption. Forensic record retained as `UNEXPECTED_FINDING_nul_padding.md`. Pre-Sprint-C size growth in `068bf54` and `af4e655` deferred to Sprint G commit-hygiene sweep. | Not a Sprint A blocker. |

**Top 3 critical actions for Sprint A planning:**

1. **Land the schema-positive `widgets_values` canonical-shape gate (SA-100) on the first Sprint A runtime-verification commit.** The retrospective's recommended "reject zero-length string arrays" would break BUG-LOCAL-032 and re-introduce the widget-drift class that produced the operator's 2026-05-15 manual ComfyUI symptoms. Sprint A must ship the inverse gate.
2. **Capture `hardware_snapshot.json` (SA-102) in the SAME commit that captures `audio_c7_baseline.wav.b3sum` + `audio_c7_baseline_pre_c5g.wav.b3sum`.** Without the env snapshot, both b3sum fixtures become "captured against unknown environment" artifacts and the deterministic-deferral trap from retrospective §1 gets baked into Sprint A's fixtures.
3. **Add `log.info` clamp visibility (SA-101) AND VRAM telemetry (SA-103) in the first Sprint A runtime-verification commit.** Both are purely additive (no-change-logs rule preserved), pure pytest, and they close the Blind Handoff and Surface Metric Bias gaps simultaneously.

**Anything that blocks Sprint A planning:** none. All four findings are recommendations, not blockers. Sprint A can be planned and cut whenever the operator decides; this triage doc fills in the empirical-verification spec gaps that closed-sprint §C-final.5 left for downstream sprints.

---

## §1. Null-state padding (verification of retrospective §6)

### Claim being verified

> "A severe, totally undetected standing-rule violation exists concerning the strict 'no-dummy' data mandate. ... the `widgets_values` arrays within Node 1 (Story Writer), Node 3 (Scene Sequencer), Node 12 (Signal Lost Video), Node 13 (Kokoro Announcer), and Node 14 (MusicGen Theme) contain numerous undocumented empty string indices."

Retrospective severity: HIGH. Recommended remediation: "Implement a strict JSON schema validator that explicitly rejects arrays containing zero-length strings."

Operator's 2026-05-15 manual ComfyUI symptoms: `temperature='{}'`, `start_line='{}'`, `default_tts=''`, `dialogue_offset_ms='bark'`, `resolution=24`, `fps='{}'`.

### Method

For each OTR custom node in `workflows/otr_scifi_16gb_full.json`, read `widgets_values` and cross-reference against the source `INPUT_TYPES` declarations. Node classes resolved via `__init__.py:84-98` `NODE_CLASS_MAPPINGS`.

### Executive summary

Nodes flagged by retrospective: 5 (Nodes 1, 3, 12, 13, 14). Widget violations confirmed by source verification: **0**. Every empty string, every `'[]'`, every `'{}'`, and every numeric default in `widgets_values` is the legitimate source-declared default or the BUG-LOCAL-032 canonical preserved-mode shape.

### Per-node detail

#### Node 1 -- `OTR_Gemma4ScriptWriter` (`nodes/story_orchestrator.py:2556-2641` :: `LLMScriptWriter`)

14 widget-backed inputs in declared order. Empty strings at indices 0 (`episode_title`, source `default: ""`) and 5 (`custom_premise`, source `default: ""`) are source defaults. All other widgets (genre_flavor, target_words, num_characters, model_id, include_act_breaks, self_critique, open_close, target_length, style_variant, creativity, arc_enhancer, optimization_profile) match source declarations exactly. `project_state` is the deliberate socket-only tail anchor required by BUG-LOCAL-027 (`story_orchestrator.py:2632-2636`). **Not dummy data.**

#### Node 2 -- `OTR_Gemma4Director` (`nodes/story_orchestrator.py:6887-6927` :: `LLMDirector`)

5 widget-backed inputs. Empty string at index 0 (`script_text`, source `default: ""` multiline). All other widgets match. `project_state` socket-only tail anchor at lines 6918-6925.

#### Node 3 -- `OTR_SceneSequencer` (`nodes/scene_sequencer.py:564-624`)

8 widget-backed inputs. Workflow values `['[]', '{}', 0, 999, '', 'bark', 0.0, 0.0]` match BUG-LOCAL-032 fix verbatim (`BUG_LOG.md:299`: "Node 3 (OTR_SceneSequencer): `['[]', '{}', 0, 999]` (4) -> `['[]', '{}', 0, 999, '', 'bark', 0.0, 0.0]` (8)"). The retrospective is mis-reading the BUG-LOCAL-032 fix as a violation.

#### Node 4 -- `OTR_AudioEnhance` (`nodes/audio_enhance.py:278-319`)

7 widget-backed inputs, all numeric / dropdown. No empty strings. Not flagged by retrospective; verified clean.

#### Node 7 -- `OTR_EpisodeAssembler` (`nodes/scene_sequencer.py:891-923`)

4 widget-backed inputs. Index 0 (`episode_title`) has workflow value `''` vs source default `"The Last Frequency"`. **DIVERGENT but legal** -- operator-cleared STRING widget value. Not cross-wiring, not placeholder. Worth a one-line comment in Sprint G cosmetic cleanup.

#### Node 11 -- `OTR_BatchBarkGenerator` (`nodes/batch_bark_generator.py:474-501`)

3 widget-backed inputs. Workflow `['[]', '{}', 0.7]` matches BUG-LOCAL-032 fix verbatim (`BUG_LOG.md:300`: "Node 11 (OTR_BatchBarkGenerator): `[0.7]` (1) -> `['[]', '{}', 0.7]` (3) [canonicalized from stripped to preserved]").

#### Node 12 -- `OTR_SignalLostVideo` (`nodes/video_engine.py:1167-1207` :: `SignalLostVideoRenderer`)

6 widget-backed inputs. Workflow `['[]', '{}', '[]', 24, '1920x1080', '']` matches BUG-LOCAL-032 fix (`BUG_LOG.md:301`). Index 5 (`episode_title`) is operator-cleared (same legal pattern as Node 7).

#### Node 13 -- `OTR_KokoroAnnouncer` (`nodes/kokoro_announcer.py:116-148`)

4 widget-backed inputs. Workflow `['[]', '', 'random', 0.95]` matches source defaults + BUG-LOCAL-031 speed=0.95 addition.

#### Node 14 -- `OTR_MusicGenTheme` (`nodes/musicgen_theme.py:147-181`)

4 widget-backed inputs. Workflow `['{}', '', 'facebook/musicgen-medium', 3.0]` matches source defaults.

#### Node 15 -- `OTR_BatchAudioGenGenerator` (`nodes/batch_audiogen_generator.py:85-110`)

6 widget-backed inputs. Workflow matches source defaults verbatim. The source comment at lines 102-106 is dispositive on the retrospective's framing: *"BUG-LOCAL-027: the '3'/'3.0'/3/3.0 entries were scar tissue from widget-drift hitting this node. With the mapper fix in `_workflow_to_api_prompt`, socket-only inputs no longer leak into widget slots, so the hack is no longer needed. Fail loudly on bad input instead of silently accepting garbage."*

### Cross-wiring vs absent-default classification

| Class | Count across all 10 nodes |
|---|---|
| True cross-wiring (value in wrong slot vs source `INPUT_TYPES`) | 0 |
| Absent-where-source-declares-non-empty-default (operator-cleared STRING widgets in Nodes 7, 12 `episode_title`) | 2 (legal STRING-widget values; not dummy padding) |
| Canonical preserved-mode placeholder (`''`, `'[]'`, `'{}'`) source-declared or BUG-LOCAL-032-canonical | 11 across Nodes 1, 2, 3, 7, 11, 12, 13, 14, 15 |

**Zero violations.** The retrospective's framing is factually wrong for this codebase.

### Root cause hypothesis

The retrospective reverses cause and effect. The sequence is:

1. **Defect class (BUG-LOCAL-027 / 029 / 030 / 031):** ComfyUI Web-UI workflow JSONs can omit trailing unlinked widget slots when those slots hold defaults. The mapper's auto-sensing heuristic could not always reconstruct the "preserved-truncated" shape, producing widget cross-wiring at runtime (`temperature='{}'`, `start_line='{}'`, etc.).
2. **Fix (BUG-LOCAL-032, commit `dabcebd`, 2026-04-14):** Compute the canonical preserved-mode shape from the live `/object_info` schema for every node and write back the canonical array. **The fix introduced** the empty strings, `'[]'`, `'{}'`, and numeric defaults the retrospective now misreads.
3. **Architectural reinforcement (BUG-LOCAL-027 socket-only-at-tail rule):** `nodes/story_orchestrator.py:2632-2636` + `:6918-6925`.

The retrospective appears to have pattern-matched on "empty string == dummy" without consulting source `INPUT_TYPES` declarations or BUG_LOG history. Classic deep-research lexical-scan-presented-as-structural-finding hallucination class.

### Operator's 2026-05-15 manual symptoms

`temperature='{}'`, `start_line='{}'`, `default_tts=''`, `dialogue_offset_ms='bark'`, `resolution=24`, `fps='{}'` are **consistent with the widget-drift class** described in BUG-LOCAL-027 / 029 / 030 / 031 / 032 lineage. They are **inconsistent** with the current contents of `workflows/otr_scifi_16gb_full.json` on this branch (verified row-by-row above).

Two scenarios could explain the manual observation:

1. **Stale workflow loaded in ComfyUI Desktop.** Browser session held a pre-BUG-LOCAL-032 cached copy.
2. **A subsequent regression on the same bug class** -- a commit after `dabcebd` could have lost the canonical shape on some other node. Worth a Sprint A confirmation step, but outside the scope of triage.

In neither scenario is the empty-string-as-dummy framing accurate.

### Recommended remediation

**NOT** the retrospective's "reject zero-length string arrays" -- that would break BUG-LOCAL-032 and re-introduce the widget-drift bug class.

**Correct direction:** schema-positive canonical-shape gate using the existing `scripts/_schema_sweep.py`. Compare each node's `widgets_values` against the live `/object_info` canonical preserved-mode shape; treat any divergence (length mismatch, type mismatch against declared input type) as hard fail. See SA-100 in §4.

---

## §2. Temperature clamp logging spec (verification of retrospective §1 / §2 / §5 / §7)

### Claim being verified

§1, §2, §5, §7 all reference the same alleged silent-clamp pattern in the reflection module. Most concrete from §7: *"Require the C5a1 exception block to emit a highly visible, persistent flag in the main generation log when temperature is clamped."*

### File path verified

`nodes/_otr_story_brief.py` -- **NOT** `nodes/_otr_story_brief_reflection.py` as the triage prompt phrased it. Exists only on Sprint C branches; introduced at C5a1 commit `87f01bd`. Read via `git cat-file -p sprint-c-story-brief-v2:nodes/_otr_story_brief.py` (blob `aeda67ee...`, 27491 bytes).

### Verified facts

**Three scoped try/except arms exist (E-17 / RR-B3 / L-6 confirmed).** `run_story_brief_reflection` body:

- Block 1 -- LLM call: `try` at line 603, `except Exception as exc` at line 609. Returns `_failure_sentinel(reason="technical_fn_exception")`.
- Block 2 -- JSON parse: `try` at line 623, `except json.JSONDecodeError as exc` at line 625. Returns `_failure_sentinel(reason=REJECT_JSON_PARSE)`.
- Block 3 -- pydantic schema: `try` at line 640, `except ValidationError as exc` at line 642. Invokes `_repair_pass(...)` at line 649.
- Content-validation block (line 673-714): post-pydantic content validation calls `_repair_pass(...)` at line 680.

**The temperature clamp lives at `nodes/_otr_story_brief.py:487-490`** inside `_repair_pass`:

```python
repair_temperature = min(
    reflection_temperature + _REPAIR_TEMPERATURE_BUMP,
    _REPAIR_TEMPERATURE_CEILING,
)
```

Constants:
- `_REPAIR_TEMPERATURE_BUMP: float = 0.15` (line 65)
- `_REPAIR_TEMPERATURE_CEILING: float = 0.55` (line 60)
- `_REFLECTION_TEMPERATURE` is the base; default in refinement section 3.2 range (0.2-0.4).

The retrospective's "clamps to 0.55" framing is **mathematically incomplete but directionally correct** -- the actual operation is `min(temp+0.15, 0.55)`, pinning at 0.55 only when `reflection_temperature >= 0.40`.

**The clamp triggers on validation-rejection paths, NOT on a "CRITICAL exception".** The retrospective's "CRITICAL" framing is a prompt-string artifact -- `"CRITICAL: You previously failed validation because: <reasons>..."` is the textual instruction prepended in `_build_repair_messages` (lines 459-470). It is NOT an exception class.

**The clamp emits NO log line at the call site (verified silent).** Inside `_repair_pass` (lines 473-498), between the clamp computation (line 487) and the LLM call (line 494), there is no `log.*` call. The surrounding "attempting repair pass" log messages at lines 643-646 and 675-678 mention WHY (validation rejection) but NOT:

- the resulting `repair_temperature` value,
- the base `reflection_temperature` value, or
- whether the clamp pinned at the 0.55 ceiling.

Sprint A inspectors performing S-A.3 motion-priority manual visual inspections will have no observability into whether a semantically-impoverished output came from a 0.4 + 0.15 = 0.55-ceilinged retry or a 0.2 + 0.15 = 0.35 retry.

### Minimal additive log patch spec (DO NOT APPLY)

**Target file:** `nodes/_otr_story_brief.py` on the Sprint C branch lineage. On a Sprint A working branch cut from `sprint-c-story-brief-v2` post-C5a1.

**Exact insertion site:** between line 490 (end of `min(...)`) and line 491 (`messages = _build_repair_messages(...)` call). One new line.

**Proposed log line text:**

```python
    repair_temperature = min(
        reflection_temperature + _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING,
    )
    log.info(
        "[OTR_StoryBrief] repair pass clamped: base=%.3f bump=%.3f "
        "ceiling=%.3f -> repair_temperature=%.3f reasons=%s",
        reflection_temperature, _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING, repair_temperature, rejection_reasons,
    )
    messages = _build_repair_messages(
        failed_output, rejection_reasons, base_user_message,
    )
```

`log.info` (not `log.warning`) because the clamp is designed pre-flight behavior, not unexpected. The repair pass's reason for firing is already logged at warning/info severity upstream.

### Why this is purely additive (no-change-logs rule preserved)

Sprint C standing directive (closed-sprint doc §5): existing runtime log strings stay byte-stable; new log lines added at C5a1 (and successors) follow neighboring format conventions; no surrounding existing line is modified.

The proposed line adds ONE new `log.info` call; does not modify any existing log string at lines 610-612, 626-629, 643-646, 660-662, 675-678, 694-697, 706-708; uses the same `[OTR_StoryBrief]` prefix and `%`-formatting style; uses an existing severity convention. Compliant.

### Why a Sprint A change, not a Sprint C amend

Sprint C is closed. Re-opening the closed branch for a log addition would violate closed-sprint discipline. The silent clamp does not fail the 2276-pytest gate; it only obscures Sprint A's S-A.3 manual inspections. Land at the first Sprint A runtime-verification commit alongside the b3sum fixture pair.

### Test coverage to add alongside the patch

`tests/test_story_brief_reflection_pure_c5a1.py` (or equivalent file added by C5a1):

1. `test_repair_pass_emits_clamp_log` -- monkeypatch `nodes._otr_story_brief.log`, force a schema-rejection path, assert exactly one `log.info` call with substring `"repair pass clamped"` and the resulting `repair_temperature` formatted to 3 decimals.
2. `test_repair_pass_clamp_log_does_not_break_no_change_logs_rule` -- AST parse the module, collect all `log.*` call sites, assert existing log strings are byte-identical to a pinned snapshot.

Pure pytest, no GPU, fits in the existing test envelope.

---

## §3. b3sum hardware determinism spec (verification of retrospective §7)

### Claim being verified

> "Sprint A is explicitly mandated to verify that the B3SUM of the legacy audio fixture is exactly byte-identical over five separate execution runs after blessing the new post-C5g render. However, the contract entirely fails to specify the hardware execution profile, the precision modes, or the pseudo-random number generator (PRNG) seed state required to achieve this determinism."

### Operator hardware envelope (fixed, single-machine)

From SPRINT.md §Hardware envelope + CLAUDE.md global rules: RTX 5080 Laptop, Blackwell sm_120, 16 GB VRAM, single workstation, no cloud, 100% local. **Cross-hardware risk is theoretical for this user.** The real risk is time-axis env drift on the same machine (OS updates, NVIDIA driver bumps, CUDA toolkit upgrades, PyTorch / transformers / bitsandbytes drift, ComfyUI-managed `.venv` mutations).

### Version-pinning state at Sprint C close

- `requirements.txt`: pins `transformers>=4.40,<6.0`, `soundfile>=0.12`, `numpy>=1.24`, `feedparser>=6.0`, `tokenizers>=0.15`, `sentencepiece>=0.1.99`, `bitsandbytes>=0.42.0`. **PyTorch is intentionally NOT pinned** -- line 2 comment: "IMPORTANT: Do NOT pin torch -- ComfyUI manages its own torch version."
- `pyproject.toml`: pins `requires-python = ">=3.10"` and `setuptools>=68.0`. No torch / CUDA / driver pin.
- No `setup.py`, `setup.cfg`, `environment.yml`, `conda-lock.yml`, `uv.lock`, `poetry.lock`, or Dockerfile.

The repo currently provides **ZERO machine-readable record** of the floating-point-affecting environment that produced any given fixture.

### Versions that affect floating-point determinism on the audio path

In approximate order of impact:

- **GPU device** -- name + compute capability (`sm_XYZ`). Tensor Core implementations differ across Blackwell / Ada / Ampere.
- **NVIDIA driver** -- host driver version.
- **CUDA toolkit (runtime)** -- `torch.version.cuda`.
- **cuDNN library** -- `torch.backends.cudnn.version()`.
- **PyTorch** -- `torch.__version__` + `torch.version.git_version`. Largest determinism axis.
- **TF32 / FP16 flags** -- `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `torch.backends.cudnn.benchmark`, `torch.backends.cudnn.deterministic`.
- **`use_deterministic_algorithms`** -- per-process flag.
- **`CUBLAS_WORKSPACE_CONFIG`** -- required for full determinism with `use_deterministic_algorithms`.
- **`PYTHONHASHSEED`** + `torch.manual_seed()` / `np.random.seed()` / `random.seed()`.
- **Python interpreter** -- `sys.version_info`, `platform.python_implementation()`.
- **transformers / bitsandbytes / tokenizers / sentencepiece / soundfile / numpy / scipy / librosa** -- packages on the generation + audio path.
- **ffmpeg** (out-of-process) -- `ffmpeg -version` first line.
- **OS** -- `platform.platform()` + `platform.win32_ver()`.

### `tools/capture_hardware_snapshot.py` spec (DO NOT WRITE)

**File:** `tools/capture_hardware_snapshot.py`, co-located with existing `tools/audit_workflow_schema.py` + `tools/validate_workflow_links.py`.

**Output:** `tests/fixtures/hardware_snapshot.json`. UTF-8, LF, no BOM.

**Schema (target shape):**

```json
{
  "captured_at": "2026-05-16T08:13:42-07:00",
  "captured_by_branch": "...",
  "captured_by_commit": "<sha>",
  "host": {"platform": "...", "win32_ver": [...], "machine": "AMD64", "node": "..."},
  "python": {"version": "3.11.7", "executable": "...", "implementation": "CPython"},
  "gpu": {
    "name": "NVIDIA GeForce RTX 5080 Laptop GPU",
    "compute_capability": "12.0",
    "total_memory_gb": 16.0,
    "driver_version": "560.XX",
    "cuda_toolkit_torch_built_with": "12.4"
  },
  "torch": {
    "version": "2.4.0+cu124",
    "git_version": "<sha>",
    "cuda_available": true,
    "cuda_runtime_version": "12.4",
    "cudnn_version": 90100,
    "backends_flags": {
      "cudnn.deterministic": false,
      "cudnn.benchmark": true,
      "cudnn.allow_tf32": true,
      "cuda.matmul.allow_tf32": true,
      "use_deterministic_algorithms": false
    }
  },
  "env_vars_of_interest": {"CUBLAS_WORKSPACE_CONFIG": null, "PYTHONHASHSEED": null, "OTR_REGRESSION_RUNTIME": "1"},
  "packages": {"transformers": "...", "tokenizers": "...", "sentencepiece": "...", "bitsandbytes": "...", "soundfile": "...", "numpy": "...", "scipy": "...", "librosa": null},
  "out_of_process": {"ffmpeg_first_line": "..."},
  "seeds_at_capture": {"torch_manual_seed": 1337, "numpy_random_seed": null, "python_random_seed": null}
}
```

**CLI:**

```
python tools/capture_hardware_snapshot.py                   # write to default path
python tools/capture_hardware_snapshot.py --out <path>      # alternate output
python tools/capture_hardware_snapshot.py --check <path>    # compare current env vs <path>; exit 0 if match-or-advisory, 2 if strict-fail
python tools/capture_hardware_snapshot.py --dry-run         # write to stdout only
```

**Test coverage (same Sprint A commit):**

- `tests/test_hardware_snapshot.py::test_snapshot_capture_runs_clean`
- `tests/test_hardware_snapshot.py::test_snapshot_required_keys_present`
- `tests/test_hardware_snapshot.py::test_snapshot_check_mode_passes_against_self`

### Strict vs advisory mode -- recommendation

ADVISORY at Sprint A's first runtime-verification session (CAPTURE mode), STRICT thereafter.

| Sprint A pass | Mode | Behavior |
|---|---|---|
| First runtime-verification commit (captures audio_c7 fixtures) | **CAPTURE** | Run `capture_hardware_snapshot.py` once. Output becomes Sprint A baseline-truth fixture. |
| Every later test pass | **ADVISORY** for: `driver_version`, `ffmpeg_first_line`, non-determinism-axis package versions | `log.info` advisory line, proceed. |
| Drift in `torch.version`, `gpu.compute_capability`, `cudnn_version`, `transformers.version` (major), `bitsandbytes.version` (major), or any `backends_flags` value | **STRICT FAIL** | Exit 2 from `--check`. Operator must either accept the new baseline (re-capture) or roll back the offending env change. |

### Perceptual audio hash supplement -- recommendation

Supplement B3SUM as a **tier-2 disambiguator**, not a primary gate. Keep b3sum primary per Prime Directive 1's byte-identical mandate.

| Tier | Method | Interpretation |
|---|---|---|
| 1 | `b3sum` byte-identical | PASS if equal. Done. |
| 2 | If tier 1 fails: `--check` hardware_snapshot | If env strict-fails, regression is environmental, not a Sprint C smuggled regression. ADVISORY + halt for operator review. |
| 3 | If tier 2 advisory-passes but tier 1 failed | STRICT FAIL. Real regression on identical env. Investigate. |
| 4 | If tier 2 strict-fails | Run perceptual hash. Chromaprint (`fpcalc` out-of-process or `pyacoustid`) on both fixtures. If similarity >= 95%, mark VERSION-DRIFT-TOLERANT PASS with diagnostic dump. Below threshold = STRICT FAIL. |

**Library choice:** Chromaprint via `fpcalc` subprocess (open source, offline, Windows-compatible, no API keys). Alternative fallback: `librosa.feature.chroma_cqt` + cosine similarity.

### Out of scope

LipDub IC-LoRA audio drift, MCP harness / `/mcp-builder` evaluation, Sprint G `_STYLE_WORLD_BLOCK` orphan sweep -- all deferred per SPRINT.md v2.1+ watch-list and closed-sprint §C-final.5.

---

## §4. Sprint A acceptance rows draft (additions to Sprint A's existing acceptance table)

Format matches Sprint C's closed-sprint table (closed-sprint doc §7). Each row: `# | Check | Target`. Numbered SA-100+ as additions to whatever Sprint A's existing rows already number.

| # | Check | Target |
|--:|---|---|
| SA-100 | Workflow JSON `widgets_values` schema-positive canonical-shape gate green. Every node's `widgets_values` matches the live `/object_info` canonical preserved-mode shape (linked placeholders + all unlinked defaults, in declared input order). Cross-wiring class regressions hard-fail. **Note:** retrospective §6 "Null-State Padding Violation" reframed -- empty strings / `'[]'` / `'{}'` are the BUG-LOCAL-032 fix, not a violation. See §1 of this triage doc. | First Sprint A runtime-verification commit; uses existing `scripts/_schema_sweep.py`. |
| SA-101 | Reflection-module `_repair_pass` clamp visibility: one new `log.info("[OTR_StoryBrief] repair pass clamped: ...")` line at the exact site between current lines 490 and 491 of `nodes/_otr_story_brief.py`. Two pytest tests staged (`test_repair_pass_emits_clamp_log`, `test_repair_pass_clamp_log_does_not_break_no_change_logs_rule`). Purely additive; no existing log string modified. See §2 of this triage doc. | First Sprint A runtime-verification commit. |
| SA-102 | `tools/capture_hardware_snapshot.py` lands. First Sprint A runtime-verification commit runs `capture_hardware_snapshot.py` once and commits the resulting `tests/fixtures/hardware_snapshot.json` alongside the `audio_c7_baseline.wav.b3sum` and `audio_c7_baseline_pre_c5g.wav.b3sum` fixture pair. Schema matches §3 of this triage doc. Three pytest tests staged. | First Sprint A runtime-verification commit (the same commit that closes acceptance rows 38, 39 from the Sprint C closed-sprint table). |
| SA-103 | VRAM telemetry in S-A.4 multi-model regression. After each generation cycle, log `torch.cuda.memory_summary()` output to a per-cycle artifact (`logs/sprint_a_vram_<cycle>.txt`). Aggregator script (or one-liner) extracts peak allocated, peak reserved, allocator-cached-but-unused, and fragmentation indicators. Strict fail if any cycle exceeds 14.5 GB peak (existing VRAM ceiling). Advisory fail if cached-but-unused fragmentation exceeds 20% of peak. Closes retrospective §7 Surface Metric Bias gap. | S-A.4 multi-model regression commit. |
| SA-104 | B3SUM tier-2 perceptual audio hash supplement wired in via `fpcalc` (Chromaprint) subprocess. Tier-1 b3sum stays the primary gate per Prime Directive 1; tier-4 perceptual hash runs only when tier-2 `--check` reports strict env drift. >= 95% similarity = VERSION-DRIFT-TOLERANT PASS with diagnostic dump; below threshold = STRICT FAIL. Two pytest tests staged (`test_perceptual_hash_runs_clean_against_self`, `test_perceptual_hash_tier_ordering`). | Sprint A acceptance row -- can land in the same commit as SA-102 or a follow-up. |

**No SA-105 / SA-106 added** -- the operator directive says "don't pad; only real actionable findings." SA-100..SA-104 cover every actionable item surfaced by Deliverables 1-3 plus the §7 Surface Metric Bias VRAM-telemetry gap; the rest of the retrospective's recommendations are framing arguments that the closed-sprint plan already accepted or that the triage refuted.

### Notes on numbering

- SA-100..SA-104 are additions, not replacements. Sprint A's existing acceptance rows from the Sprint C close (closed-sprint doc §C-final.5 post-state contract -- audio C7 baseline reset captures, empirical visual + audio render quality verification, empirical LTX motion fidelity verification) keep their existing scope and ordering.
- SA-100 is intentionally numbered high to avoid colliding with any earlier-numbered Sprint A row the operator may already have in flight.

---

## §5. NUL padding investigation (forensic note)

A side-finding surfaced during §1's workflow JSON inspection: the bash sandbox view of `workflows/otr_scifi_16gb_full.json` read it as 22314 bytes of valid JSON followed by 21735 NUL bytes -- while the git blob on `sprint-c-story-brief-v2` was 44049 bytes of clean text and the same file on `main` was 22314 bytes of clean text.

Captured in detail at `docs/retrospectives/UNEXPECTED_FINDING_nul_padding.md` (committed `4aab34d`; resolution appended at `eb7a7ae`).

**Operator resolution 2026-05-16: FALSE ALARM.** Sandbox NUL observation was a mount artifact, not on-disk corruption. Sprint C did not introduce the size delta. Pre-Sprint-C commits `068bf54` and `af4e655` contain the bulk of the size growth; their commit subjects do not clearly explain the JSON expansion. Commit-hygiene verification of those two commits deferred to Sprint G's broad cleanup sweep (10-minute diff inspection). **Not a Sprint A blocker.**

---

## Sources cited (in-repo, read-only)

- `workflows/otr_scifi_16gb_full.json`
- `__init__.py:84-98`
- `nodes/story_orchestrator.py:2556-2641, 2632-2636, 6887-6927, 6918-6925`
- `nodes/scene_sequencer.py:564-624, 891-923`
- `nodes/audio_enhance.py:278-319`
- `nodes/batch_bark_generator.py:474-501`
- `nodes/video_engine.py:1167-1207`
- `nodes/kokoro_announcer.py:116-148`
- `nodes/musicgen_theme.py:147-181`
- `nodes/batch_audiogen_generator.py:85-110, 102-106`
- `nodes/_otr_story_brief.py` on `sprint-c-story-brief-v2` (blob `aeda67ee...`); lines 49-65, 446-498, 580-720
- `BUG_LOG.md:294-304` (BUG-LOCAL-032)
- `requirements.txt`, `pyproject.toml`
- `SPRINT.md` §Hardware envelope + §Rules in force + Previous Sprint Handoff
- `CLAUDE.md` (repo root + global)
- `docs/closed-sprints/2026-05-15-sprint-c-story-brief-v2.md` §1.2 (E-17, E-18, R-06), §5 standing directives, §7 acceptance table, §C-final.5 post-state contract
- `docs/AI_Production_Pipeline_Retrospective__Sprint_C.md` (full)
- `tools/audit_workflow_schema.py`, `tools/validate_workflow_links.py` (co-location reference)
- `docs/retrospectives/UNEXPECTED_FINDING_nul_padding.md` (forensic record + resolution)
