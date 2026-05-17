# OTR distillation round-up -- fix plan -- 2026-05-16

**Source of truth:** sprint-d-period-llm @ 5b0d0ba.
**Inputs:** distillation MD cold-read + two adversarial reviews of `docs/2026-05-16-otr-workflow-distillation-v2-pre-sprint-a.md`. 24 unique findings after dedupe.
**Owner:** Jeffrey A. Brick.
**Branch:** cut `sprint-e-distillation-roundup` from `sprint-d-period-llm @ 5b0d0ba`. No work on main, no work on v2.0-alpha.

Structure of this file: review (what we found), code (source edits), wire (workflow JSON edits), regs (regression tests), commit (commit boundaries). Sprint A backlog at the bottom. No source attribution, no change log -- live work plan.

---

## §1  Review: 24 findings clustered by lane

### HIGH -- ship-stoppers

| # | Lane | Finding |
|---|---|---|
| H1 | C7 baseline | Canonical workflow JSON ships writer seed=0 + control=randomize. Audio C7 contract (`tests/fixtures/baseline_v1.5.wav`) requires seed=42 fixed + both writer slots Mistral-Nemo. Default Queue Prompt cannot reproduce baseline. Same drift on HuMo seed widget. |
| H2 | C7 / Sprint D | MusicGen mood-prefix is the C5g baseline shift. No unit test guards a future drift in `_MUSIC_MOOD_VOCAB` / `_resolve_cue_from_style` / `_mood_suffix`. |
| H3 | C7 / HuMo | HuMo audio passthrough is Prime Directive 1 load-bearing. No automated b3sum check that EpisodeAssembler audio == HuMo-muxed output audio. A future codec/sample-rate change would land silently. |
| H4 | Contract gap | Reflection failure (`status="failed_repair"` or `"missing"`) is handled per-consumer with no centralized dispatch. 5 consumers (FLUX env, FLUX portrait, LTX, HuMo, MusicGen) each invent their own fallback semantics. |
| H5 | Silent failure | Workflow JSON node 63 stores validator `workflow_json_path` widget as `""`. Validator's `_load_workflow("")` behavior is not explicit -- empty path either no-ops or errors. |
| H6 | Doc / work-by-accident | Distillation §1 + §5 describe L86 as HuMo `report` STRING -> LowVRAMCheckpointLoader `ckpt_name` STRING. Actual workflow JSON edge type is `dependencies` `*`. Distillation has it wrong. |
| H7 | Doc | Distillation §0 D0d narrative is wrong: claims "three rewires", calls `audio_gate` slot 3 and unwired. Reality: five wire changes; `audio_gate` is slot 1 and IS wired via L47. |
| H8 | Doc | Distillation §4 D2b claim is wrong: portrait prompt is deterministic Python composition, no LLM call, so D2b's creative prompt router does NOT wire there. The 4 actual D2b wire sites are writer-internal LLM phases (outline, line composer, polish character, polish announcer). |

### MEDIUM -- correctness + observability

| # | Lane | Finding |
|---|---|---|
| M1 | Source / silent failure | `_otr_creative_prompt_router.resolve()` selection is substring match against curated rows. A non-curated forked HF id containing the talkie substring dispatches the period system prompt against modern weights silently. |
| M2 | Source / contract gap | `meta.freeze_unload_ok` (S34 B2 stamp) is written but read by zero consumers. A leaked Mistral-Nemo cache passes silently into Bark / FLUX VRAM. |
| M3 | Source / naming + sequencing | `LowVRAMCheckpointLoader` accepts a STRING-named input `ckpt_name` that is used in the graph as a pure DAG-seq edge from HuMo. The name lies about the purpose. A future commit that starts honoring `ckpt_name` would break the graph. |
| M4 | Source / naming + sequencing | `OTR_VideoPlan` has an input named `audio_gate` that is wired (post-D0d) from FreezeCascade `script_json` -- value ignored, purpose is dependency edge. The name actively misleads. Same anti-pattern as M3. |
| M5 | Source / observability | VideoComposite `clips_dir` widget tooltip says "HuMo clips"; actual wire is from LTX (L92) and HuMo clips are resolved via `ledger.clips[]`. |
| M6 | Source / observability | HuMo per-clip wall-time (~10-12 min on RTX 5080) is not surfaced in the node tooltip or pre-run log. A 6-line script silently consumes ~1 hour. |
| M7 | Source / observability | BatchHumoRender `portraits_dir` auto-resolves to `output/otr/portraits/<ep_id>/` when the input is empty. With D0d wire-3 in place, a disconnect falls through silently. |
| M8 | Source / license gate | License audit framework (D0b) ships per-row audit files but no model loader consults `license_audit_status` before download. Today's curated rows are all `green` or `unaudited`, so the framework is forward-compat-only. |
| M9 | Source / ergonomics | MusicGen `allow_silence_fallback=False` raises `RuntimeError` on `ImportError` but the message does not name the missing package or install hint. |
| M10 | Naming | `OTR_ShotDurationCalculator` currently returns a fixed 8s stub. Name implies it computes durations; behavior is a placeholder. |
| M11 | Source / writer | Writer does not stamp `meta.episode_title` from the widget; SignalLost's Tier 1 title slot is dead, chain falls to Tier 4 (widget) or Tier 5 (TIMESTAMP_LASTRESORT). |

### LOW -- defensive polish

| # | Lane | Finding |
|---|---|---|
| L1 | Defensive alias | Node type `PathchSageAttentionKJ` is the upstream KJ-Nodes spelling. If upstream ever fixes the typo, workflow load breaks. |
| L2 | Distillation doc | §10 should note: the C7 baseline holds ONLY when both writer slots are Mistral-Nemo-Instruct-2407. Talkie on the creative slot drifts the baseline by construction. |
| L3 | Distillation doc | §11 should add: HuMo wall-time reference (10-12 min) was measured Sprint C era; Sprint D writer meta-stamping + portraits_dir wire may shift; re-time before quoting. |
| L4 | Distillation doc | §2 INPUT_TYPES table should annotate JSON-shipped values where they differ from schema defaults (seed=0+randomize vs schema 42; act_count=3 vs schema 0). |
| L5 | Distillation doc | §0 D0d narrative correction (folds into H7). Single line: "audio_gate is wired as a freeze dependency gate, not as an audio dependency." |

---

## §2  Code -- source edits

Order: ship-stoppers first, then ergonomics. Each commit is one bullet unless noted.

**C-1.** `nodes/_otr_creative_prompt_router.py` -- replace substring match with exact-match lookup against curated rows by `repo_id`; pick prompt_profile from `row.prompt_profile` (the actual CuratedModel field). Raise `RouterAmbiguityError` if zero or multiple matches. Adds `tests/test_router_exact_match.py`. Fixes M1.

**C-2.** New module `nodes/_otr_story_brief_fallback.py`. Single public helper:

```python
def get_story_brief_with_fallback(meta, consumer_id) -> StoryBriefResolved:
    """Centralized fallback dispatch. Reads meta.story_brief.status,
    returns a typed (status, prose, terms_by_kind) tuple, and logs
    the disposition exactly once per consumer per run. Consumer ids:
    'flux_env', 'flux_portrait', 'ltx', 'humo', 'musicgen'."""
```

Rewires all 5 consumers (`visual/batch_flux_render.py`, `visual/batch_flux_portrait_render.py`, `nodes/batch_ltx_render.py`, `nodes/batch_humo_render.py`, `nodes/musicgen_theme.py`) to call this helper instead of inventing per-site fallback. Fixes H4.

**C-3.** `nodes/_otr_workflow_validator.py` -- make `_load_workflow("")` explicitly fall back to `_DEFAULT_WORKFLOW_PATH` and log the resolved path. Adds explicit log line `OTR_WorkflowValidator: empty widget path; resolved to <canonical>`. Fixes H5.

**C-4.** `nodes/musicgen_theme.py` (and `nodes/batch_audiogen_generator.py` if same pattern) -- catch `ImportError` at MusicGen pipeline import and raise `RuntimeError("MusicGen requires 'transformers' and 'audiocraft'. Run: pip install audiocraft transformers")`. Fixes M9.

**C-5.** Rename `OTR_ShotDurationCalculator` -> `OTR_FixedShotDurationStub`. Add a one-release `_RENAME_ALIASES` entry in `__init__.py` so saved workflow JSONs keep loading, then delete the alias in the next sprint. Per CLAUDE.md no-back-compat directive: schedule the alias delete as the very next commit after the workflow JSON is rewritten clean. Fixes M10.

**C-6.** Rename `OTRVideoPlan.audio_gate` input -> `freeze_done_gate`. Add a fail-early guard: if input is unwired (empty string after coercion), log error and return a sentinel JSON `{"error": "VideoPlan: freeze_done_gate not wired -- run will be unusable downstream"}` for all 3 pass outputs. Fixes M4 + Sprint D regression risk follow-up.

**C-7.** Rename `LowVRAMCheckpointLoader.ckpt_name` consumer of L86 -> add a new input `sequence_gate` typed `*` (matches the existing JSON `dependencies` edge type from H6). Keep the original `ckpt_name` STRING input for the actual checkpoint name -- separate the two purposes. Updates workflow JSON to wire HuMo `report` -> `sequence_gate` instead of `ckpt_name`. Fixes M3 + H6.

**C-8.** `nodes/batch_bark_generator.py` + `visual/batch_flux_render.py` -- read `meta.freeze_unload_ok`; if False, log warning and attempt one defensive `_otr_model_loader.unload_llm()`. Fixes M2.

**C-9.** `nodes/batch_humo_render.py` -- when `portraits_dir` input arrives empty and auto-resolve fires, log:
`log.warning("[BatchHumoRender] portraits_dir input is empty; auto-resolved to %s. This is the D0d fallback path; wire FluxPortrait.portraits_dir output for explicit contract.", resolved_path)`. Fixes M7.

**C-10.** `nodes/OTR_LedgerScriptWriter.py` -- at K.5.7 (after K.5.6 stamps `meta.creative_model` + `meta.creative_prompt_profile`), stamp `meta.episode_title` from the widget value (or from outline title if widget empty). Fixes M11 + L (SignalLost Tier 1 dead slot).

**C-11.** `visual/batch_flux_portrait_render.py` (or `visual/batch_flux_render.py`, wherever the radio bookend skip placeholder lives) -- when `skip_env_stills=True` and the placeholder 16x16 IMAGE is returned, also stamp `meta.env_stills_skipped=True` so downstream nodes can branch deliberately. Fixes BatchFluxRender placeholder ambiguity (Cluster V).

**C-12.** New helper `nodes/_otr_model_inputs.py::require_license_audit(model_id)` -- read `licenses/audits/<safe_model_id>.json`; raise `LicenseAuditError` if file is missing OR `audit_status not in {"green", "yellow"}` OR `audit_date` is older than 180 days. Every LLM / FLUX / LTX / HuMo / MusicGen / AudioGen / Bark loader calls this guard before any HF download. Fixes M8.

**C-13.** `__init__.py` -- defensive alias `PatchSageAttentionKJ` -> `PathchSageAttentionKJ` (and vice versa) so workflow JSON keeps loading if KJ-Nodes upstream fixes the typo. Fixes L1.

---

## §3  Wire -- workflow JSON edits

Single canonical workflow JSON: `workflows/otr_scifi_16gb_full.json`. Smoke variants (`otr_humo_*_smoke.json`, `ltx_2_3_downstream_smoke.json`) get the corresponding subset.

**W-1.** Node 1 (`OTR_LedgerScriptWriter`) -- set widget seed = `42`, set the control widget (the value next to the seed in `widgets_values`) to `"fixed"`. Fixes H1 writer side.

**W-2.** Node 51 (`OTR_BatchHumoRender`) -- set widget seed = `7` (the schema default), set control to `"fixed"`. Fixes H1 HuMo side.

**W-3.** Node 63 (`OTR_WorkflowValidator`) -- set `workflow_json_path` widget to the absolute path string `"C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/otr_scifi_16gb_full.json"` (or the repo-relative form the validator accepts). Fixes H5 wire side. Note: even with C-3's explicit empty-string fallback, populate the JSON so a user reading the workflow surface sees the canonical path.

**W-4.** Node 23 (`OTR_BatchFluxRender`) -- verify `skip_env_stills = True` is stored in widgets_values. If not, fix.

**W-5.** Workflow JSON rewires from C-7 (L86 type change):
- Replace L86 with a `*`-typed edge from HuMo `report` (51.2) to LowVRAMCheckpointLoader `sequence_gate` (54.new). Keep `ckpt_name` widget at the canonical checkpoint filename.

**W-6.** Workflow JSON rewires from C-6 (VideoPlan rename):
- L47 source / target slot rename: `audio_gate` -> `freeze_done_gate`. No edge change; just the slot name on node 20.

**W-7.** Workflow JSON rewire from C-5 (ShotDurationCalculator rename):
- Replace node type `OTR_ShotDurationCalculator` with `OTR_FixedShotDurationStub` on node 21.

**W-8.** Validator widget realignment: confirm `widgets_values` on node 63 is `[canonical_path, true, true]` after W-3.

**W-9.** Tooltip update on node 52 (`OTR_VideoComposite`) `clips_dir` widget: "LTX clips directory (unused; HuMo clips resolved via ledger.clips[])." This is a code-side change in `nodes/video_composite.py` INPUT_TYPES tooltip; the workflow JSON does not store tooltips. Fixes M5.

---

## §4  Regs -- regression tests

Pytest only. Every test runs in CI; runtime-gated tests are flagged for Sprint A.

**R-1.** `tests/test_workflow_canonical_baseline.py` -- assert node 1 widget seed == 42 + control == "fixed"; node 51 widget seed == 7 + control == "fixed"; node 63 path widget non-empty. Drift guard for H1 + H5.

**R-2.** `tests/test_router_exact_match.py` -- feed a synthetic catalog with a forked talkie row (`mradermacher/Mistral-Nemo-Talkie-Forked`); assert the router does NOT dispatch the period profile. Drift guard for M1 + C-1.

**R-3.** `tests/test_story_brief_fallback_centralized.py` -- AST-walk the 5 consumer modules; assert each imports `get_story_brief_with_fallback` and contains no inline fallback path. Drift guard for H4 + C-2.

**R-4.** `tests/test_freeze_unload_ok_consumed.py` -- AST-walk `nodes/batch_bark_generator.py` and `visual/batch_flux_render.py`; assert each reads `meta.get("freeze_unload_ok")`. Drift guard for M2 + C-8.

**R-5.** `tests/test_humo_audio_passthrough_b3sum.py` -- using a fixture WAV + a mock HuMo mux call, assert b3sum of pre-mux audio == b3sum of post-mux audio. Runtime-gated test (`OTR_REGRESSION_RUNTIME=1`) runs the actual HuMo render path. Drift guard for H3.

**R-6.** `tests/test_musicgen_c7_b3sum.py` -- fixed-seed MusicGen render against a known script JSON fixture; b3sum stable against a committed baseline file. Drift guard for H2.

**R-7.** `tests/test_workflow_validator_empty_path_fallback.py` -- assert `WorkflowValidator.validate(workflow_json_path="", ...)` resolves to `_DEFAULT_WORKFLOW_PATH` and logs the resolved path. Drift guard for H5 + C-3.

**R-8.** `tests/test_videoplan_freeze_done_gate_required.py` -- assert VideoPlan's input is named `freeze_done_gate`; assert unwired input returns the sentinel-error JSON. Drift guard for M4 + C-6.

**R-9.** `tests/test_lowvram_loader_sequence_gate_typed.py` -- assert `LowVRAMCheckpointLoader.INPUT_TYPES` declares `sequence_gate` typed `("*",)` (or `("DEPENDENCY",)` per ComfyUI convention); assert L86 in workflow JSON is `*`-typed. Drift guard for M3 + C-7.

**R-10.** `tests/test_license_audit_required.py` -- assert every loader call site invokes `require_license_audit(model_id)`. AST-walk for the call. Drift guard for M8 + C-12.

**R-11.** `tests/test_episode_title_stamped.py` -- run the writer with widget `episode_title="The Last Frequency"`; assert `meta.episode_title` in the resulting ledger equals the widget value. Drift guard for M11 + C-10.

**R-12.** `tests/test_videocomposite_clips_dir_tooltip.py` -- assert tooltip on `clips_dir` contains "LTX clips" + "ledger.clips[]". Drift guard for M5 + W-9.

**R-13.** `tests/test_humo_wall_time_estimate_logged.py` -- assert HuMo's pre-batch log line contains an estimated total runtime string. Drift guard for M6 (companion to Sprint A A2).

**R-14.** `tests/test_humo_portraits_dir_fallback_logged.py` -- assert log.warning fires when `portraits_dir` arrives empty. Drift guard for M7 + C-9.

**R-15.** Forbidden-sweep markers (append to `docs/_s28_forbidden_sweep.py`): `OTR_ShotDurationCalculator` (post-rename guard), `audio_gate` (post-rename guard), `ckpt_name.*STRING.*forceInput.*from.*HuMo` regex guard. Zero runtime hits expected after Phase 3 lands.

**R-16.** Bug Bible regression baseline must hold 23 passed / 1 skipped / 2 xfailed at every commit boundary. No new entries; this is a behavior-preserving sprint at the contract layer.

---

## §5  Commit -- commit boundaries

Branch: `sprint-e-distillation-roundup` cut from `sprint-d-period-llm @ 5b0d0ba`.

| Commit | Subject | Scope | Tests gating |
|---|---|---|---|
| E0 | Sprint E branch cut + this plan landing | docs-only | Bug Bible 23/1/2 holds |
| E1 | Distillation doc fixes (§0 narrative + §1 + §4 + §5 + §8 + §10 + §11) | docs-only (`docs/2026-05-16-otr-workflow-distillation-v2-pre-sprint-a.md` edits) | Bug Bible 23/1/2 holds |
| E2 | Workflow JSON canonical-config widgets (H1 + H5 + skip_env_stills audit) | wire-only (W-1, W-2, W-3, W-4) | R-1 + Bug Bible |
| E3 | Centralized story_brief fallback helper + 5 consumer rewires | source (C-2) | R-3 + Bug Bible |
| E4 | Router exact-match + license audit guard | source (C-1, C-12) | R-2 + R-10 + Bug Bible |
| E5 | Validator empty-path fallback + MusicGen ImportError message | source (C-3, C-4) | R-7 + Bug Bible |
| E6 | L86 sequence_gate typed input + workflow rewire | source + wire (C-7, W-5) | R-9 + Bug Bible |
| E7 | VideoPlan freeze_done_gate rename + fail-early guard + workflow rewire | source + wire (C-6, W-6) | R-8 + Bug Bible |
| E8 | ShotDurationCalculator -> FixedShotDurationStub rename + workflow rewire | source + wire (C-5, W-7) | Bug Bible (rename is mechanical) |
| E9 | freeze_unload_ok consumed at Bark + FLUX | source (C-8) | R-4 + Bug Bible |
| E10 | HuMo portraits_dir fallback log + wall-time estimate log | source (C-9, M6 log line) | R-13 + R-14 + Bug Bible |
| E11 | Writer stamps meta.episode_title at K.5.7 | source (C-10) | R-11 + Bug Bible |
| E12 | VideoComposite clips_dir tooltip + skip_env_stills meta flag + Sage alias | source (C-11, C-13, W-9) | R-12 + Bug Bible |
| E13 | Forbidden-sweep markers (rename guards) | regs only (R-15) | R-15 + Bug Bible |
| E14 | Audio C7 b3sum guards (HuMo passthrough + MusicGen) | regs only (R-5, R-6) | R-5 + R-6 (pytest proxy mode) + Bug Bible |
| E15 | Sprint E close + Sprint A handoff note | docs-only | Bug Bible final |

Commit message convention per CLAUDE.md: cmd shell, `git commit -F .git\COMMIT_EDITMSG` (file tool writes the message; no `-m` to avoid cmd.exe quoting trap). Subject line `Sprint E E<N> <short summary>` plus 1-3 body bullets.

Push convention per CLAUDE.md: Desktop Commander cmd shell, `cd /d <repo> && git push origin sprint-e-distillation-roundup`. Verify HEAD match after push.

Round-robin per CLAUDE.md: any architecture choice inside an E<N> commit that has a real branching trade-off (centralized fallback shape in C-2, license audit semantics in C-12, sequence_gate type in C-7) goes through ChatGPT + Gemini + Claude synthesis before code lands.

---

## §6  Sprint A acceptance backlog (deferred)

Runtime gates that require ComfyUI Desktop + RTX 5080. Sprint E ships pytest + structural only; these flip live when Sprint A opens.

| A# | Item | Gate |
|---|---|---|
| A1 | Audio C7 runtime gate, post-C5g baseline capture | `OTR_REGRESSION_RUNTIME=1 pytest tests/test_story_brief_musicgen_c5g.py::TestRuntimeOnly` -- b3sum captured to `tests/fixtures/baseline_v1.6.wav` and committed |
| A2 | HuMo per-clip wall-time re-time | 6-line default-config script, log total HuMo time, compare to 10-12 min/line reference |
| A3 | VRAM full-episode soak | `OTR_SOAK_FULL_EPISODE=1` + nvidia-smi 5s polling; assert peak <= 14.5 GB |
| A4 | LTX 2.3 coherence smoke | Render 8s, then 14s, then 22s non-character clips; visual inspect for tear / repetition |
| A5 | Period-prose poisoning runtime test | Talkie in both writer slots; assert `_PERIOD_REGEX` clean AND indirect period vocabulary check ("wireless set", "victrola", "boys at the front") flags appropriately |
| A6 | PostUpscaleProcgenBlend node 58 documentation + wire verification | Document class signature; verify OUTPUT_NODE=True; if final output unwired, add explicit saver/display node |
| A7 | RGB/YUV blend visual hash | Visual-hash test for `lighten`, `screen`, `addition` blend modes |
| A8 | Workflow JSON drift validator rule | New `WorkflowValidator` check: assert C7-critical widgets (writer seed, both writer model slots, HuMo seed) at canonical baseline values; opt-out via widget for non-baseline runs |

---

## §7  What is explicitly NOT in scope for Sprint E

- LipDub IC-LoRA adoption (ROADMAP addendum -- Sprint A or later forward feature work).
- Writer LLM phase reshape beyond meta.episode_title stamping.
- Cast contract / casting architecture changes.
- ComfyUI custom node migration (e.g., adopting community VHS / SVI-Pro-FLF).
- v2.0-alpha umbrella label bump; sprint-e branches off sprint-d.
- BUG_LOG.md promotion to Bug Bible (batch-promote after v2.0 ships).
- ROADMAP rewrite (line up with sprint-e close note in E15).

---

## §8  Halt gates within Sprint E

Surface for operator review (do not auto-resolve) if any of these fire:

- Round-robin disagreement on C-2 centralized fallback shape, C-7 sequence_gate type, or C-12 license audit semantics.
- Bug Bible regression delta non-zero at any commit boundary.
- Forbidden-sweep runtime hits non-zero at any commit boundary.
- Audio C7 pytest proxy fails between E5 and E14.
- Workflow JSON link validator reports new violations after any W-* edit.

End.
