# Synthesis -- 2026-04-30

**Question:** # OTR Project + ROADMAP QA -- Round-Robin Consultation

You are doing an **internal QA pass** on the ComfyUI-OldTimeRadio project,
v2.0-alpha branch.  This is NOT for shipping.  Your job is to:

1. Spot priority misalignment in ROADMAP vs current open bugs.
2. Flag any node, workflow, or design choice that looks brittle, redundant,
   or out of sync with the rest of the stack.
3. Sanity-check the last few commits -- did the recent fixes actually close
   the bugs they claim to close?
4. Recommend the **next 3-5 things to do before the next episode test run**,
   in priority order, with one-sentence justifications.

Constraints to respect (do not relitigate):

- Single RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows.
- 100% local for the shipped pipeline.  No cloud weights.
- VRAM ceiling 14.5 GB.  No weight streaming, no Flash Attention chasing,
  no quantization heroics.
- Audio output must remain byte-identical to v1.5 baseline (rule C7).
- Default models: writer = Mistral-Nemo 12B (LOCAL); critic + reviser = a
  different family for two-immune-system gate.
- Default audio path: HuMo native audio + master_mix slices, ffmpeg concat.
- "Do not chase VRAM/DRAM dragons" -- VRAM work is alarm plumbing only.

Be candid.  Flag uncertainty.  Cite specific file paths and line numbers
when you can.  If a recent commit message claims a fix that the current
code does not actually contain, say so.

---

## Branch + git status

Branch: `v2.0-alpha`

### git status (short)

```
M config/episode_cast.txt
 M docs/2026-04-29-vram-context-test.md
 M docs/BUG_LOG.md
 M docs/OTR_PIPELINE_EXPLAINER.md
 M nodes/kokoro_announcer.py
 M nodes/project_state.py
 D otr_v2/schema/visual_plan.schema.json
 D otr_v2/subprocess_runners/__init__.py
 M scripts/_consult_round_robin.py
 M scripts/audit_video_stack_weights.ps1
 M scripts/soak_operator.py
 M scripts/supersoaker.py
 M tests/test_wedge_probe.py
 D tests/v2/__init__.py
 D tests/v2/_debug_submit.py
 D tests/v2/_extract_from_history.py
 D tests/v2/_fix_workflow.py
 D tests/v2/_git_push.py
 D tests/v2/_run_baseline.py
 D tests/v2/debug_prompt.json
 D tests/v2/fixtures/baseline_v1.5.sha256
 D tests/v2/fixtures/baseline_v1.5.wav
 D tests/v2/test_audio_byte_identical.py
 M workflows/otr_videoplan_TEST_humo.json
?? .claude_tail_anomalies.txt
?? .claude_tail_check_critic.ps1
?? .claude_tail_critic.txt
?? .claude_tail_critic_full.ps1
?? .claude_tail_critic_ran.txt
?? .claude_tail_critic_ran.txt.err
?? .claude_tail_critic_run.ps1
?? .claude_tail_ledger.txt
?? .claude_tail_pytest.ps1
?? .claude_tail_pytest.txt
?? .claude_tail_status.txt
?? .otr_critic_check.tmp.txt
?? .otr_inflight.tmp.txt
?? .otr_ledger.tmp.txt
?? .otr_log_scan.tmp.txt
?? .otr_status.tmp.txt
?? .otr_tailer_critic.ps1
?? .otr_tailer_inflight.ps1
?? .otr_tailer_ledger.ps1
?? .otr_tailer_scan.ps1
?? .tailer_anomaly.txt
?? .tailer_check.txt
?? .tailer_count_titlefail.ps1
?? .tailer_critic.txt
?? .tailer_critic_ledger.txt
?? .tailer_critic_run.ps1
?? .tailer_critic_run.txt
?? .tailer_errors.txt
?? .tailer_errors_run.ps1
?? .tailer_grep_critic.ps1
?? .tailer_ledger.cmd
?? .tailer_ledger.ps1
?? .tailer_ledger_check.ps1
?? .tailer_ledger_check.py
?? .tailer_ledger_out.txt
?? .tailer_ledger_run.txt
?? .tailer_ledgers.txt
?? .tailer_run.ps1
?? .tailer_status.txt
?? .tailer_status_run.txt
?? .tailer_step2.ps1
?? .tailer_step2.txt
?? .tailer_step3.ps1
?? .tailer_step3.txt
?? .tailer_step3b.ps1
?? .tailer_test.txt
?? .tailer_titlefail.txt
?? .tailer_traceback.ps1
?? .tailer_traceback.txt
?? 2]
?? _anomaly_scan.ps1
?? _episode_scan.ps1
?? _ledger_inspect.ps1
?? _progress.ps1
?? _scan_anomaly.ps1
?? _scan_critical.ps1
?? _tail_critic.ps1
?? _tail_grep.ps1
?? _tail_log.ps1
?? config/news_history.json
?? docs/2026-04-18-model-inventory.md
?? docs/2026-04-19-flux-bf16-thrashing-fix__00_question.md
?? docs/2026-04-19-flux-bf16-thrashing-fix__01_chatgpt.md
?? docs/2026-04-19-flux-bf16-thrashing-fix__02_gemini.md
?? docs/2026-04-19-flux-bf16-thrashing-fix__03_synthesis.md
?? docs/2026-04-19-flux-bf16-thrashing-fix__04_claude_synthesis.md
?? docs/2026-04-19-flux-bf16-thrashing-fix__transcript.json
?? docs/2026-04-19-qa-plan-lean-team-response.md
?? docs/2026-04-19-test-workflow-readiness__01_chatgpt.md
?? docs/2026-04-19-test-workflow-readiness__02_gemini.md
?? docs/2026-04-26-peer-review-v2-alpha/
?? docs/2026-04-26-peer-review-v2-alpha__00_question.md
?? docs/2026-04-26-peer-review-v2-alpha__01_chatgpt.md
?? docs/2026-04-26-peer-review-v2-alpha__02_gemini.md
?? docs/2026-04-26-peer-review-v2-alpha__03_synthesis.md
?? docs/2026-04-26-peer-review-v2-alpha__transcript.json
?? docs/2026-04-30-nvidia-smoke/
?? docs/2026-04-30-project-qa/
?? docs/internal/
?? led.json
?? outputs_scratch/
?? pytest_087.txt
?? pytest_out.txt
?? schema/
?? scripts/_anomaly_grep.tmp
?? scripts/_bug_titles.tmp
?? scripts/_critic_check.err
?? scripts/_critic_check.tmp
?? scripts/_critic_check_combined.tmp
?? scripts/_status_out.tmp
?? scripts/_test_py.tmp
?? scripts/dedupe_and_organize_scan.ps1
?? scripts/dedupe_progress.txt
?? scripts/dedupe_report.json
?? scripts/dedupe_report.txt
?? scripts/model_inventory.txt
?? scripts/qa_project_bundle.py
?? scripts/scan_models.ps1
?? scripts/verify_ltx_hybrid.py
?? subprocess_runners/
?? tests/_run_baseline.py
?? tests/fixtures/baseline_v1.5.sha256
?? tests/fixtures/baseline_v1.5.wav
?? tests/fixtures/concat_smoke/
?? tests/test_audio_byte_identical.py
?? video
?? workflows/otr_scifi_16gb_full.json.bak
```

### Last 20 commits

```
ef9989c BUG-LOCAL-118 fix: BatchHumoRender ledger lookup robust to filename underscore drift between SignalLostVideo and Ledger renamer
80bf6bf Future-proof writer prompt for E2B-only users (Patches A + B)
d6859ea BUG-LOCAL-116 + BUG-LOCAL-117 fixes: full audio palette restored
0be500b BUG_LOG: file BUG-LOCAL-116 + BUG-LOCAL-117 from overnight ledger audit
5392682 BUG-LOCAL-115 fix: third fallback for title resolution unblocks soak
c28376d VideoComposite: humo_concat is the default audio_source
830b323 VideoComposite: humo_concat audio mode + bookend skip diagnostic
70ab155 BUG_LOG: file BUG-LOCAL-111 / 112 / 113 / 114
db8a4b1 ROADMAP: lock LTX render strategy at 8s native + 2x slow-mo
2887ba2 ScriptCritic: revise_on_findings defaults to ON
fd02e37 ScriptCritic: add Reviser pass + reroute audio chain through critic
cc9e87a ScriptCritic: also inherit optimization_profile from ledger
8ea25dd ScriptCritic: inherit critic_model_id + genre_flavor from ledger
a0bf40d ROADMAP.md: 2026-04-29 PM session log + open items list
16294df BUG-LOCAL-112: news-history reset actually restores the unfiltered pool
61a85b3 BUG-LOCAL-112: news-history reset actually restores the unfiltered pool
5ec2fcd Add OTR QA tooling: per-clip frame extraction + per-line waveform plots
f520e12 Add INPUT_TYPES vs FUNCTION signature audit script
21aba38 Fix VideoComposite: add cleanup_clips_after_assembly to execute signature
7f37fe9 Anti-slop dynamic template: gate parser + ledger-driven filter
```

## ROADMAP.md (full)

```markdown
# OTR Roadmap

**Last updated:** 2026-04-29 PM (enhanced-narrative pipeline shipped end-to-end: ScriptCritic gate (Gemma-4 E4B as separate judge) wired into otr_scifi_16gb_full.json, OTR-CANON.md auto-included in writer's system prompt, dynamic anti-slop rubric with [applies_when] gates filtered per-run from ledger gen_params_initial, news curation Option B (parallel body-fetch + body-aware LLM re-rank), news-history dedup with full-wipe restoration BUG-LOCAL-112 fix, news_seed stamped in ledger.meta, VideoComposite cleanup_clips_after_assembly TypeError fixed, INPUT_TYPES vs FUNCTION audit script (66 classes, 0 HIGH-priority issues), QA tooling for episode review (qa_frames + qa_waveforms + qa_episode), per-model context_cap dict, in-workflow VRAMContextTest node, HF cache migration to C:\ComfyUI-Models, early ledger init at workflow phase 0, per-clip incremental ledger save during HuMo render, smoke preset target_words=100 + 1-act, Gemma-4 E2B/E4B added to dropdown + Gemma 2 family removed, Captain-Eris/Mag-Mell marked EXPERIMENTAL; 186/186 tests passing)
**Branch:** `v2.0-alpha`
**Owner:** Jeffrey A. Brick

**This file is the single source of truth.** Canonical going-forward plan. Three horizons: **v1.7 audio pipeline** (shipped, live-test cycle ongoing), **v2.0 video stack sprint** (14-day build, drives the next two weeks), and **v2.0 continuity layer** (Scene-Geometry-Vault + Style-Anchor cache, post-sprint). Everything shipped or discarded stays in source docs — this file is open items only.

**Canonical narrative hierarchy** (every script, ledger, workflow, and doc in this repo follows this):

```
Scene  >  Shot  >  Beat  >  Clip
```

- **Scene** — high-level narrative location (`AstroTech Research Facility`, `Control Room`, ...). One per scene_id.
- **Shot** — continuous visual unit. Same framing, same lighting. May contain multiple speakers.
- **Beat** — single-speaker continuous turn within a shot. The unit at which the 7 s clip-fill rule applies — beats never cross speakers, so HuMo audio windows align cleanly to one voice. Added 2026-04-25 PM (production_ledger schema bump l1 → l2).
- **Clip** — one HuMo render call. Length must be 4n + 1 frames (Wan VAE temporal compression of 4) and ≤ 177 (verified ceiling on 16 GB).

Every consumer of `ledger.json` must understand all four levels. The orchestrator's Goal 3 anchor mode reads `lines[].boundary` ∈ {`shot_start`, `beat_start`, `continue`} to decide between fresh anchor, clean speaker reset, or daisy-chain α-blend.

---

## v2.0-alpha session log — 2026-04-28 PM (paths + lip-sync alignment)

Three commits shipped today on `v2.0-alpha`:

- **`70f4a5c`** — Step 0 paths refactor. New `nodes/_otr_paths.py` helper module (resolution order: `OTR_OUTPUT_DIR` env → `folder_paths.get_output_directory()` → walk-up to ComfyUI root → cwd fallback). 14 files retouched; ~12-15 hardcoded `r"C:\Users\jeffr\..."` strings in nodes/scripts replaced with helper calls. Workflow JSON `otr_humo_smoke.json` widget defaults scrubbed.
- **`cce3472`** — BUG-LOCAL-100 + BUG-LOCAL-101. Stellar Shadows render exposed two upstream-of-HuMo bugs producing audible script garbage:
  - **BUG-100** — script parser captured third-person narration as the dialogue line (Stellar Shadows l002/l003: "Lev bursts onto catwalk deck...", "Stanley follows closely..."). Fix: new heuristic `_is_inline_narration(speaker, text)` + next-line look-ahead in v1/v2 inline match arms of `_parse_script`.
  - **BUG-101** — `_clean_text_for_bark` translated parentheticals to Bark non-verbal tokens (`(panting)` → `[pants]`); Bark's rendered tokens leaked breath audio before dialogue (Stellar Shadows 0:34 "Let's work I guess..."). Fix: drop ALL parentheticals in both `_clean_text_for_bark` implementations (batch_bark + scene_sequencer).
- **`84ffe1e`** — BUG-LOCAL-100b + BUG-LOCAL-102.
  - **BUG-100b** — schema-level dialogue contract added to `SCRIPT_SYSTEM_PROMPT`. Concrete WRONG/RIGHT examples mirroring the Stellar Shadows failure pattern. Belt-and-suspenders with the BUG-100 parser heuristic: prevent emission of the bad pattern, then catch at parse time if the LLM slips.
  - **BUG-102** — HuMo's intrinsic ~3-6 frame motion-onset freeze produced a constant ~150-200 ms audio-leads-lips perception. Round-robin consult (Gemini + ChatGPT 4.1) converged on Option A: pre-pad each per-line audio with leading silence (~200 ms zeros) so HuMo burns its warm-up freeze on silence rather than the first phoneme; trim symmetrically before save so VideoComposite placement math is unchanged. New optional widget `humo_warmup_pad_ms` (default 200, range 0-500). Fix lives entirely inside `BatchHumoRender` — master audio mix + VideoComposite untouched (CLAUDE.md C7 preserved).

Protective regression layer also shipped:
- `tests/test_inline_narration_heuristic.py` — 22 cases covering BUG-100 helper.
- 5 new BUG-101 paren-drop assertions in `tests/test_core.py::TestCleanTextForBark`.
- `tests/test_humo_warmup_pad.py` — 21 cases covering BUG-102 pad/trim helpers + round-trip.

All four bugs marked Bible candidates; promotion deferred until next render confirms behavioural fix.

### QA gate before Step 1

The Step 1-5 work below is **gated on the next FULL queue's QA pass.** Verify in the new ledger:

1. `lines[].text` contains NO third-person narration (no "Lev bursts onto deck...", no "Stanley follows closely..." style prose).
2. `lines[].text` contains NO parentheticals (no `(panting)`, no `(grabbing the mic)`, etc.).
3. `clips[].warmup_pad_ms = 200` stamped on every entry.
4. User perception of lip-sync: alignment within ~30 ms across whole episode (audio no longer reliably leads lips).

If any of (1)-(4) fail, triage / re-tune the relevant fix BEFORE starting Step 1. Do not stack new visual work on top of a still-broken audio stack.

---

## v2.0-alpha session log — 2026-04-29 PM (enhanced narrative pipeline)

Eight commits shipped today on `v2.0-alpha`. End-to-end enhanced narrative path is now active:

- **`62382dd`** — Option B news curation. `_llm_rerank_with_bodies` plus parallel `ThreadPoolExecutor` body-fetch of top 5 candidates. The old serial walk-the-list bailed at the first acceptable headline; the new flow body-fetches all five in parallel (~10s) then asks the LLM to re-rank by article content, not catchy title. Total ~50s, comfortably under the 65s news-curation budget. Side fix: `vram_context_test.py` writes report through `folder_paths.get_output_directory()` to satisfy BUG-01.02.
- **`b1b422b`** — `nodes/script_critic.py` (`OTR_LLMScriptCritic` node), `docs/OTR-ANTI-SLOP.md` (placeholder rubric), `docs/OTR-CANON.md` (SIGNAL LOST canon — tonal rules, period rules, motifs, used premises/twists/motifs sections). Critic node is OUTPUT_NODE=True, advisory by default (`block_on_reject=False`), default judge model is Gemma-4 E4B (different from Mistral-Nemo writer = two-immune-system gate). 27 unit tests. Cache invalidation between writer→critic→writer model swaps via the BUG-LOCAL-111 plumbing.
- **`7f37fe9`** — Anti-slop dynamic template. 46-rule rubric (A1-A46) plus 12 directives (B1-B12) with `[applies_when: ...]` gates and `{placeholder}` tokens. Loader (`_normalize_target_length`, `_evaluate_gate` sandboxed eval, `_coerce_params` no-defaults, `_filter_rubric`, `_SafeMissing` partial-tolerant placeholder substitution) filters per-run from ledger `gen_params_initial`. Drops "tiny" → "smoke", removes default-value masking. 28 new unit tests.
- **`21aba38`** — VideoComposite TypeError fix. `cleanup_clips_after_assembly` widget was added to INPUT_TYPES earlier this session but execute() signature was never updated; ComfyUI passes every INPUT_TYPES key as a kwarg, function rejected with TypeError. Added param with `default=False`, log-only when True (deletion logic still deferred per session plan).
- **`f520e12`** — `scripts/audit_input_types_vs_signatures.py`. AST walks every OTR node class, extracts INPUT_TYPES widget keys, finds the matching FUNCTION method, flags any widget the function won't accept. Audit on commit 21aba38: 66 classes, 0 HIGH-priority issues. Run before any push that touches a node's widgets.
- **`5ec2fcd`** — QA tooling. `scripts/qa_frames.py` (smart per-clip frame extraction at boundary stress points, hard-cap 60 frames), `scripts/qa_waveforms.py` (matplotlib per-line waveform PNGs + episode summary, with prepad/tail-silence markers, falls back to ffmpeg audio extraction when `final_audio_path` is null), `scripts/qa_episode.py` (one-shot wrapper). Live-tested on `signal_lost_viral_bloom_protocol_20260429_175715`: 10/10 frames + 2/2 line plots in <4s. Cowork's native Read tool consumes both formats cleanly.
- **`61a85b3`** — BUG-LOCAL-112: news-history reset actually restores the unfiltered pool. Symptom from `pending_20260429_204318`: news fetch returned 43 headlines, history-dedup filtered ALL 43, the warning fired but the "reset" was a no-op. Real fix: stash the unfiltered pool before filtering, restore on full wipe. Better to pick a recent repeat than starve the writer with zero context.
- **`16294df`** — Wire ScriptCritic into `workflows/otr_scifi_16gb_full.json` (id=53, link=84 from writer slot 0 → critic slot 0; writer's link to Director unchanged). Wire OTR-CANON.md into the writer's system prompt at `write_script` entry: new `_load_canon_for_writer()` helper reads tonal rules / period rules / recurring motifs / used premises/twists/motifs, escapes literal `{}` so `.format()` doesn't choke, prepends to `SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT`. Skipped on small models (Gemma-4 E2B and below) to avoid Model Collapse.

### Earlier in this same session (commits before the enhanced-narrative push)

- Per-model `_MODEL_CONTEXT_CAPS` dict replacing the primitive heuristic (Mistral-Nemo→16384, Gemma 4 E2B/E4B→16384, Qwen 14B→12288, EXPERIMENTAL→12288, default→8192).
- `nodes/vram_context_test.py` — in-workflow `OTR_VRAMContextTest` node (production-accurate VRAM measurement). Three-column reporting (VRAM nvml = nvidia-smi truth, VRAM torch = process allocator, CPU RAM = host-side, never mixed into VRAM accounting).
- `_run_with_timeout` BUG-LOCAL-111: invalidate `_LLM_CACHE` on `FuturesTimeout` so the next phase forces a fresh load and avoids `cudaErrorIllegalAddress` from orphan worker threads.
- HF cache architecture: `setx HF_HUB_CACHE / HF_HOME / OTR_MODELS_DIR` to `C:\ComfyUI-Models`, NTFS rename of `Documents\ComfyUI\models` → `C:\ComfyUI-Models` (~316 GB moved in 5 sec), `scripts/consolidate_hf_cache.py` recovered ~50 GB of deduped AppData shadow.
- Early ledger init at `write_script` entry — closes the 5+ minute observability gap, stamps `gen_params_initial` and `log_paths` from t=0.
- Per-clip incremental ledger save inside `BatchHumoRender` loop — crash resilience. New `resume_from_ledger` BOOLEAN widget (default ON) skips already-rendered clips on retry.
- Per-character voice priming + AISM negative-constraint bullets in `SCRIPT_SYSTEM_PROMPT` (animated-environment cliches, telegraphed emotion).
- `num_characters` widget min: 2 → 1 (monologue mode).
- `target_words` widget min: 350 → 100 (smoke step-down).
- `target_length` dropdown: added "smoke (1 act)" tier.
- 1-character monologue support (`PreFlight` char clamp respects user-explicit num_characters=1, `has_cast` assertion >= 1).
- `_VOICE_TRAITS` constant + `_check_voice_consistency()` helper stamps `ledger.voice_warnings`.
- LLM dropdown: Gemma 4 E2B/E4B added, Gemma 2 family removed (BUG-LOCAL-110 — bnb NF4 + Gemma 2 + Blackwell sm_120 = CUDA assert), Captain-Eris/Mag-Mell marked EXPERIMENTAL and reordered to bottom.

### Open items after today's session (testing-gated)

The next FULL workflow run is the QA gate. After it lands clean, prioritise in this order:

1. **Inspect `ledger.script_gates[]`** in the new ledger. Read the critic's verdict, score, fired rules. Tune the rubric thresholds (90/70/0) if calibration is off.
2. **Decide if `block_on_reject=True` is safe to flip** based on calibration data from 3-5 runs.
3. **Drop "tiny" label** from the workflow JSON / target_length dropdown UI display (rubric loader already normalizes; this is just UX polish).
4. **News-history fuzzy dedup for syndication edge case** — URL dedup catches direct repeats; same content with different URLs (Orion Flywheel duplicate yesterday) needs a fuzzy headline match. Filed.
5. **Empty-section pruning in filtered rubric** — a 1-character run keeps the `### Ensemble-voice collapse` heading even after all 3 rules are filtered out. Wastes a few tokens, doesn't break anything. Low priority.
6. **Diff 3 — spine ledger-stamping with bundled metadata + schema bump l3 → l4** (filed in `docs/2026-04-29-spine-ledger-stamping-ticket.md`, scheduled with explicit unblock conditions).
7. **LLM edge-case matrix sweep** (6 rows queued in `docs/2026-04-29-llm-edge-case-matrix.md`).
8. **VRAM measurement runs via `OTR_VRAMContextTest` node** to tune `_MODEL_CONTEXT_CAPS` from real data (currently conservative defaults).
9. **GGUF unblock check** — lazy poll for cu13 wheel OR Path Y install window (filed in `docs/2026-04-29-gguf-parked.md`).
10. **VideoComposite cleanup deletion logic** — widget shipped, no-op for now. Wire actual deletion when stable enough to trust.
11. **Auto-update `OTR-CANON.md` from passing critic verdicts** — `_canon_update()` helper exists in `script_critic.py` but is intentionally not called yet. Wire in once we have 3-5 runs of critic data.

### Acceptance gate for this session's work

- Critic stamps `ledger.script_gates[]` on every run (advisory mode).
- Body-aware news re-rank logs `[NewsFetcher] body re-rank chose #N: ...`.
- News history reset never produces a "Body-fetching top 0 candidate(s)" line again.
- Writer system prompt contains `<canon>...</canon>` block on every Mistral-Nemo run (skipped silently on small models).
- `qa_episode.py --latest` produces frames + waveforms for the next finished episode.

---

## v2.0-alpha continuation — Morning of 2026-04-29 order of operations (Jeffrey-locked)

Captured at the end of the 2026-04-28 marathon session. **Execute in this exact order** -- each step's outputs feed the next:

### Step 1 — Extra FLUX stills (radio bookend + per-scene backgrounds)

Extend `OTR_BatchFluxRender` to emit **two new sets of stills** beyond the existing PASS3 cast-shot composites:

- **One vintage radio bookend** per episode. Hard-coded prompt (canned in workflow widget, deterministic across episodes): `"1940s vintage console radio, glowing amber dials, walnut cabinet, vacuum tube halo, dimly lit listening room, cinematic 35mm film aesthetic, 1080p"`. Saved as `output/otr/stills/radio_bookend_<ep_id>.png`. Used during opening + closing music windows.
- **One per-scene environment still** per `ledger.scenes[]` entry. Prompt comes from Director's `visual_plan.scenes[i].visual_prompt`. Saved as `output/otr/stills/scene_bg_<scene_id>_<ep_id>.png`. Doubles as: (a) HuMo I2V portrait stand-in (per-scene flavor), (b) LTX img2vid seed (Step 3) OR ffmpeg-native zoompan source (Step 3 alternative).

### Step 2 — Ledger write-back so HuMo reads canonical paths

`OTR_BatchFluxRender` writes the rendered paths back to ledger:
- `ledger.shots[shot_id].png_path` for every PASS3 + scene_bg rendered
- New top-level field `ledger.radio_bookend_path` for the radio still (or use `ledger.shots["radio_bookend"].png_path`)

This closes the BUG-LOCAL-096-followup gap (Flux→shot_id flow). HuMo's `_resolve_cast_stills_from_ledger` (BUG-088) and `_find_composite` (BUG-087 family) get upgraded to **PREFER `ledger.shots[].png_path` lookup over filesystem-glob-by-mtime**. Glob fallback stays as last resort. HuMo lip-sync gets the right scene-flavored still as a portrait stand-in instead of mtime-guessing.

### Step 3 — Background animation: LTX OR ffmpeg-native, evaluate cost/look

Two paths to "animated background":

- **3a. LTX img2vid** (rich motion). Per-scene FLUX still feeds `OTR_BatchLTXRender`, produces 5-10s clip at 12 fps, looped via ffmpeg `-stream_loop -1` across scene duration. Render budget ~15-90s per episode (per the loop+12fps cuts already roadmap'd above). Layered as the BOTTOM of the 3-layer composite.
- **3b. ffmpeg-native pseudo-motion** (free, no second model). FLUX still + `zoompan` (slow zoom in/out) + `setpts=PTS*0.5` (slow down) + duplicate-frame loop = a "Ken Burns animated still" without any LTX render. Costs zero extra GPU time. Less rich than LTX but might look good enough for the broadcast-distress aesthetic.

Decision: **render BOTH on the next test episode**, A/B compare visually. If 3b looks acceptable, ship 3b for v2.0-alpha and keep LTX as v2.0-beta upgrade. If 3b feels too static, commit to LTX.

### Step 4 — Sequencing dependency edges

Per the BUG-086 dependency-gate pattern:

```
LLM/Bark/SceneSequencer/EpisodeAssembler/SignalLostVideo
        ↓ (existing)
VideoPlan + ShotDuration
        ↓
BatchFluxRender (ALL stills: PASS3 cast + scene_bg + radio_bookend)
        ↓ ledger write-back (Step 2)
        ↓
UnloadAll (frees FLUX VRAM)
        ↓
[OPTIONAL] BatchLTXRender (per-scene img2vid loops)
        ↓ ledger write-back (ledger.scenes[].ltx_path)
        ↓
UnloadAll (frees LTX VRAM)
        ↓
BatchHumoRender (reads ledger.shots[].png_path → BUG-088 cast-still binding)
        ↓
VideoComposite
   ┌─ assemble layers per timeline:
   │    bookend window:  radio_bookend full canvas + procgen lighten on top
   │    scene window:    scene_bg/LTX bottom + HuMo center pillarbox (during dialogue) + procgen lighten on top
   └─ output → episodes_for_obs/<ep>/<ep>.mp4
        ↓ (audio: -map 0:a from procgen, ALWAYS preserved)
        ↓
[OPTIONAL] SeedVR2 upscale pass
   ⚠ AUDIO MUST REMAIN INTACT through the upscale -- ffmpeg `-c:a copy` to passthrough,
     don't re-encode. Verify with ffprobe post-upscale.
```

### Step 5 — Audio integrity guard through upscale

Sanity check: at every composite/upscale stage, the final mp4 must have **both video AND audio streams**. ffprobe verification step in `OTR_VideoComposite`'s report output. If audio is missing, surface as a warning in the report STRING. Pin in regression test (`test_video_composite.py`): assert ffprobe shows audio stream codec_type=audio after composite.

### Acceptance / done

- Radio bookend visible at opening + closing of every episode
- Per-scene FLUX backgrounds populated to `ledger.shots[].png_path`
- HuMo prefers ledger lookup over mtime glob (logs `source=ledger-shot-canonical` for matched, `source=mtime-fallback` only when shot has no png_path)
- (3a OR 3b) animated backgrounds visible behind HuMo during dialogue
- Final mp4 has audio. Optional upscaled mp4 has audio.

This is the morning-of plan. Each step is small enough to ship as a focused commit (BUG-100 series).

---

## v2.0 release blocker — Generic / relative paths (no Windows-hardcoded absolutes)

**Owner:** Jeffrey | **Status:** open | **Added:** 2026-04-28 PM

Jeffrey's stance 2026-04-28: *"can't we make them more relative paths to the output folder before we ship — we need all generic relative paths"*. Every `r"C:\Users\jeffr\Documents\ComfyUI\output"` in the codebase is a release blocker for any non-Jeffrey user (Linux/Mac/RunPod/cloud) AND a portability blocker for the 8GB-tier work. v2.0 cannot ship while paths are user-and-OS-specific.

### Inventory of hardcoded Windows paths to refactor

Quick grep audit (2026-04-28 evening):

- `nodes/batch_humo_render.py` — 3-4 places: `_resolve_cast_stills_from_ledger`, `_load_ledger_with_path` auto-discover, `comfy_output` in `execute()`, `_extract_json` raw output dump
- `nodes/video_composite.py` — 2 places: output_dir computation, `_load_ledger_with_path` auto-discover
- `nodes/musicgen_theme.py` — BUG-095 audio_dir hardcoded for ledger write-back
- `nodes/batch_audiogen_generator.py` — same as MusicGen
- `nodes/batch_bark_generator.py` — BUG-096 audio_dir hardcoded
- `nodes/story_orchestrator.py` — Director BUG-090 raw output dump uses Windows path
- `scripts/render_episode_concat.py` — `--out-dir` default + comfy_output_dir CLI default
- `scripts/render_humo_batch.py` — episode dir resolver
- `workflows/otr_scifi_16gb_full.json` — `LoadAudio` widget hard-pins a `C:\...` mp4 path; some other node widgets carry default paths
- `workflows/otr_humo_smoke.json` — same widget pattern with the Resonance Chamber fixture path

Total: ~12-15 spots in code + 2-4 widget defaults in workflow JSONs. Mechanical refactor, no logic change.

### Refactor strategy

**Single `nodes/_otr_paths.py` helper module** that exposes typed accessors all nodes import:

```python
def comfy_output_dir() -> Path:
    """ComfyUI's main output directory. Resolution order:
       1. OTR_OUTPUT_DIR env var if set
       2. folder_paths.get_output_directory() (ComfyUI's API) if importable
       3. Walk up from this module: <repo>/../../../output (typical custom_nodes layout)
       4. Fallback: Path.cwd() / "output"
    """
    ...

def otr_audio_dir() -> Path:    return comfy_output_dir() / "otr" / "audio"
def otr_stills_dir() -> Path:   return comfy_output_dir() / "otr" / "stills"
def otr_portraits_dir() -> Path: return comfy_output_dir() / "otr" / "portraits"
def otr_videos_dir(episode_id: str) -> Path:
    return comfy_output_dir() / "otr" / "videos" / episode_id
def episodes_for_obs_dir(episode_id: str) -> Path:
    return comfy_output_dir() / "episodes_for_obs" / episode_id
def director_raw_dump_dir() -> Path:
    return otr_audio_dir()  # BUG-090 dumps live alongside ledgers
```

Every hardcoded `Path(r"C:\Users\jeffr\Documents\ComfyUI\output")` gets replaced with the appropriate helper. Future cloud-tier work (RunPod, 8GB tier, whatever) just sets `OTR_OUTPUT_DIR=/workspace/output` and the whole pipeline obeys.

### Workflow JSON path scrub

Widget defaults in `otr_scifi_16gb_full.json` and `otr_humo_smoke.json` need the Windows-specific path strings cleared OR replaced with a placeholder that's user-overridable on first load. Two options:

- **A. Empty defaults**: clear the LoadAudio path, force the user to drag a file in. Simple, but breaks the smoke's drag-and-queue UX.
- **B. Comment-marker defaults**: use a placeholder like `"<set_audio_path_here>"` that the node detects and either (a) auto-discovers the most recent ledger-paired mp4 in the canonical audio dir or (b) prints a clear "set the LoadAudio widget" message in the report. Preserves UX while flagging the missing input.

Lean toward B for the smoke (preserves drag-and-queue), A for the FULL workflow (audio comes from upstream nodes anyway, so no widget needed).

### Validation plan

After refactor:
- All AST + 200+ regression tests still pass on Windows (no regression of current behavior)
- `OTR_OUTPUT_DIR=/tmp/otr_test python -m pytest tests/` runs cleanly with the override
- Documented in `README.md` how to set `OTR_OUTPUT_DIR` for cloud / non-default installs
- Smoke + FULL workflows load and queue cleanly on a fresh ComfyUI install with no manual path edits

### Tradeoff vs. doing it tomorrow

If we ship v2.0-alpha as a portfolio piece WITHOUT this refactor: only Jeffrey can run it. Anyone else (collaborator, friend, RunPod template) hits immediate `Path("C:\\Users\\jeffr\\...")` failures. Path refactor is a 30-60 min mechanical task — earlier is better than later because future BUG fixes will keep adding new hardcoded paths if we don't lock down the helper module first.

**Recommended placement in tomorrow's morning plan: Step 0 (before Step 1's radio bookend work).** Lock down the path helper, refactor existing code, THEN start adding new Flux stills / LTX / etc with the helper from day one.

---

## v2.0 release blocker — 8GB-VRAM-class user experience

**Owner:** Jeffrey | **Status:** open | **Added:** 2026-04-28

Jeffrey's stance: *we don't release v2.0 until 8GB-class users get an enhanced visual output too*. Right now the pipeline is hard-locked at 16 GB (HuMo 14B fp8 = 16.5 GB staged, BUG-LOCAL-082 frame cap, Pro Ultra LLM profile). On an 8 GB card the HuMo branch OOMs immediately.

**Decision needed (NOT now -- after 16GB FULL run is green):** workflow strategy for 8GB users + experimentation. Three options on the table; **Option C is current leaning** per Jeffrey 2026-04-28 ("kinda like the idea you can toggle your video models in one jason to really experiment multiple image and vid models"):

- **Option A — Two workflow JSONs (`otr_scifi_16gb_full.json` + `otr_scifi_8gb_full.json`).** Each purpose-built; users pick the right one for their card. Pro: simple per-JSON, no runtime branching, easy to test. Con: two files to maintain, drift risk, doesn't help with model-A/B experimentation across stacks.
- **Option B — Single master JSON with a runtime VRAM switch widget.** One workflow with a `vram_mode` widget that gates HuMo vs. cheaper paths via OTR_VRAMSwitch node or `mode=4` script pass. Pro: one canonical JSON. Con: ComfyUI's executor isn't built for clean runtime conditional execution; both branches load at queue time defeating the VRAM gate; doesn't help with experimentation either.
- **Option C — Single master JSON with bypassable video-stack groups (LEANING).** One workflow with the shared audio chain → procgen, then MULTIPLE side-by-side video render groups (FLUX-fp8 stills branch, HuMo 14B fp8 video branch, LTX-Video 0.9 branch, CogVideoX-2B branch, SDXL+AnimateDiff branch, etc.) -- each group bypassable via ComfyUI's right-click → Bypass (Ctrl+B). All groups feed a final VideoComposite that takes whichever group is active. Pro: matches Jeffrey's experimentation use case (A/B test image AND video models in the same file without juggling JSONs); 8GB users bypass the HuMo+FLUX groups and enable a lightweight pair, no 16GB allocation; one canonical JSON; new experimental models add as new bypassable groups; audio+procgen path stays shared automatically. Con: visual clutter on canvas (mitigated by ComfyUI's Groups feature with collapse), graph load time slightly higher (parses more nodes even when bypassed). Bypassed nodes DO skip execution AND skip loader allocation per ComfyUI 0.20+ semantics, so the VRAM-saving promise holds.

**Acceptance criteria for the 8GB path (whichever option):**
- Full audio pipeline (LLM + Bark + AudioGen + MusicGen + SceneSequencer + EpisodeAssembler) — same as 16GB path.
- SignalLostVideo procgen base — same.
- **Visual layer must be MOTION VIDEO, not stills.** Jeffrey explicit 2026-04-28: *"i just don't want stills if we can find a 8gb vid model"*. The 8GB tier gets actual lip-sync or B-roll video, just at a smaller scale than HuMo 14B fp8.
- Final mp4 lands in `output/episodes_for_obs/<ep>/<ep>.mp4` same as 16GB path.

**Model research required (before designing the 8GB workflow):**
1. **Image model under 8 GB.** Current pipeline uses FLUX-fp8 (~12 GB staged for fast_batch). Replacement candidates worth benchmarking on 8 GB cards: SDXL-Turbo, SD 3 Medium, FLUX-schnell at lower precision, SD 1.5 (last resort if quality acceptable). Must produce 1024x1024 cinematic stills usable as portrait stand-ins for the video model below.
2. **Video model under 8 GB.** Current pipeline uses HuMo 14B fp8 (16.5 GB staged). Replacement candidates: HuMo's smaller siblings if they exist, Wan 2.1 1.3B if available, LTX-Video 0.9 (~5 GB), CogVideoX-2B fp8, AnimateDiff with a lightweight base. Lip-sync ideal but not required — even animated B-roll over the procgen base is a strict upgrade over stills. Must accept the same per-line audio segment input pattern HuMo uses (or a documented adapter).
3. **Pairing:** the chosen image + video models should share a tokenizer/conditioning style if possible to avoid double prompt-engineering surface. If they can't share, the BatchFluxRender / BatchHumoRender split is fine; just document the prompt-shape difference per workflow.

**Decision deferred until first clean 16GB FULL run ships.** Add to `docs/2026-04-28-8gb-strategy-decision.md` after the run lands so we have real data on what the 16GB path looks like end-to-end before designing the 8GB fallback.

**Decision deferred until first clean 16GB FULL run ships.** Add to `docs/2026-04-28-8gb-strategy-decision.md` after the run lands so we have real data on what the 16GB path looks like end-to-end before designing the 8GB fallback.

**Related thought (separate decision):** flip default `optimization_profile` from current default to `Pro (Ultra Quality)` once 16GB FULL has shipped clean — Jeffrey: "I almost feel we should default to Pro Ultra". Holding off until at least one clean Pro Ultra FULL run ships, in case Pro Ultra exposes new edge cases the BUG-085/090/091/094/095/096 safety nets don't yet cover.

**v2.0-beta candidate — LTX-Video animated backgrounds (3-layer composite).** Jeffrey 2026-04-28 PM, while Stellar Shadows ran: *"yes well LTX would be background behind all other layers ... maybe reactive at top in lighten mode ... we don't need that many LTX clips you know maybe 1 or two per scene looping for the whole scene"*. Architecture refinement that promotes the current 2-layer composite (procgen-base + HuMo-overlay, BUG-092) into a 3-layer composite when LTX-Video 0.9 is wired in:

```
TOP:    Procgen/CRT audio-reactive overlay -- `lighten` blend, ~0.3 opacity
MID:    HuMo lip-sync portrait -- center pillarbox during dialogue, opaque
BOTTOM: LTX animated background -- full canvas, opaque
```

**Why CRT-on-top in `lighten` mode is more truthful:** a failing broadcast's scanlines + audio-peak flicker should cover the WHOLE frame including the speaker's face -- the interference doesn't politely stop at the pillarbox edges. Lighten mode takes max(CRT, underlying) per channel so artifacts ride on top without erasing detail; ~0.3 opacity keeps HuMo lip-sync readable while audio-reactive peaks make the CRT flare across LTX + HuMo together for unified broadcast-medium feel.

**LTX render budget (Jeffrey-locked 2026-04-29 PM — render-native + slow-mo):**

**Decision:** render LTX at its trained native fps (24 fps), then slow-mo to 12 fps in ffmpeg post. The same N frames cover 2× the timeline duration, motion stays coherent (LTX was trained at 24 fps so its temporal layers expect that cadence), and the slow-mo IS the SIGNAL LOST broadcast-degraded aesthetic. Three wins from one trick.

- **1-2 LTX clips per SCENE** (NOT per shot). Loop across the scene's duration via ffmpeg `-stream_loop -1` with optional crossfade or ping-pong reverse to mask the loop tell on long scenes.
- **Render at 24 fps native, then slow to 12 fps via ffmpeg `setpts=PTS*2,fps=12`.** LTX's diffusion temporal layers were trained at this cadence; rendering at 12 fps native produces stuttery / wrong-pace motion on environmental content. Render-native + post-slow keeps motion coherent AND reinforces the broadcast-degraded aesthetic.
- **Frame budget: 193 frames per clip = 8 sec native = 16 sec apparent after 2× slow-mo.** Math: LTX uses 8x temporal VAE compression so frame counts must be `8n+1`. 193 = 24*8 + 1, sweet spot for 16 GB tier with FLUX/HuMo unloaded.
- Optional dial-up to 241 frames (10 sec native, 20 sec apparent) when scenes are long and loop-tells become visible. Documented LTX-2.3 max is 257 frames per clip.
- **Steps: distilled 4-8 steps** (default 6). Full-model 20-50 steps overkill for background loops -- HuMo is the hero, LTX is atmospheric.

**Per-episode LTX wall-clock:**
- Smoke run (1 scene): 1 x 50s LTX = ~50s per episode
- Short run (3 scenes): ~2.5 min total
- Medium run (5 scenes): ~4 min total

That's negligible vs HuMo (~10 min per dialogue line). Total pipeline cost goes up by ~2-4 min for a richer 3-layer composite vs 2-layer.

**Frame-count widget on `OTR_BatchLTXRender`:**
```
ltx_frames:    [97, 145, 193, 241]   (8n+1 dropdown, default 193)
ltx_steps:     [4, 6, 8]             (distilled, default 6)
slow_mo_factor: float (default 2.0)  (1.0 = no slow, 2.0 = half-speed)
target_fps:    int (default 12)      (post-slow display rate)
```

**Implementation sketch:**
- `OTR_VideoComposite` gains an optional `bottom_video` STRING input (path to LTX clip OR a list of one-per-scene clip paths). When set -> 3-layer mode, procgen flips to top-overlay-with-lighten. When unset -> current 2-layer architecture stands (BUG-092 unchanged).
- New `OTR_BatchLTXRender` node parallels `OTR_BatchFluxRender`'s pattern: per-scene loop seeded by the scene's PASS3 FLUX env still as I2V conditioning. Reuses BUG-088 cast-still binding work for visual continuity (the LTX clip's first frame ≈ the FLUX still HuMo characters use as portraits).
- ffmpeg loop handling in VideoComposite: for each scene's `[scene_start_s, scene_end_s]` window, layer the LTX clip with `setpts=PTS-STARTPTS+SCENE_START_S/TB,loop=loop=-1:size=N_FRAMES`.
- Sequencing follows BUG-086 dependency-edge pattern: FLUX -> UnloadAll (frees FLUX VRAM) -> LTX (fits in 5-6 GB) -> UnloadAll (frees LTX VRAM) -> HuMo. All three big models share the 16 GB budget by being mutually exclusive in time.

**Bonus for 8GB tier:** LTX 0.9 fp16 fits on 8 GB cards. Same `OTR_BatchLTXRender` node serves both 16 GB tier (as background layer) and 8 GB tier (as primary visual; HuMo bypassed). Single model, two roles via workflow toggle.

**Sub-candidate — Vintage radio bookend still.** Jeffrey 2026-04-28 PM: *"did we ever render the radio still for the talking radio?"* — confirmed grep across the whole repo: NO. Never rendered one. That's an actual identity gap given the project is called "Old Time Radio." Proposed shape:

- **One FLUX render per episode** of a vintage 1940s console radio: glowing amber dials, walnut cabinet, vacuum-tube halo, dimly lit listening room, cinematic 35mm film aesthetic. Saved as `output/otr/stills/radio_bookend_<ep_id>.png` for ledger traceability (and reuse if re-rendering the composite later).
- **Bookend overlay** in `OTR_VideoComposite`: during the opening music window (0 → opening_music.dur_s, ~10-12s) the radio still is the FULL CANVAS (LTX + HuMo branches muted). When opening music fades out, the radio still crossfades into the regular composite (LTX/procgen base + HuMo dialogue). At episode end, the closing music window (last 8s) crossfades BACK to the radio still + closing music. Procgen-CRT lighten layer rides on top of both bookend and middle phases — keeps the broadcast-failing aesthetic continuous.
- **Why this works:** every episode opens with the camera lingering on the radio while the brass fanfare plays, then fades INTO the actual scene drama, then returns to the radio for the closing sting. Anchors the "you're listening to an OTR broadcast — but with picture" identity that the project name implies. Currently the opening music plays over procgen-only (visually empty); this fills that with the iconic radio image.
- **Cost:** 1 extra FLUX render per episode (~30-60s). Cheap. The radio prompt can be canned in the workflow widget so it's deterministic across episodes (or randomized if user wants variety).
- **Ledger:** `radio_bookend_path` field in the ledger (new minor schema bump or reuse existing `shots[]` with `shot_id="radio_bookend"`). Use the BUG-088 freshness mechanism to ensure the bookend still is regenerated per episode rather than reused stale.

**Decision pending the Stellar Shadows landing at ~6 PM 2026-04-28:** if procgen-base + HuMo-pillarbox alone feels alive enough -> ship v2.0-alpha 2-layer canonical, LTX is v2.0-beta enhancement. If procgen-only background between dialogue feels static -> LTX promotes from "v2.0-beta opt-in" to "v2.0 must-have". Radio bookend is a separate decision -- low cost, high identity value, likely worth doing regardless of the LTX call.

**Related v2.0-beta candidate — LLM character normalize pass.** Jeffrey 2026-04-28: *"should an LLM character normalize pass be run too?"* Currently cast cleanup is two layers: (1) regex blocklist `_SFX_CAST_BLOCKLIST_PATTERNS` (BUG-091 + BUG-097), and (2) fuzzy `_consolidate_similar_cast_rows_with_aliases` (BUG-098 + earlier prefix/typo merges). Both are deterministic but limited to KNOWN patterns. An LLM-based normalize pass after fuzzy dedup could catch semantic aliases neither layer sees: `KEVIN VOICEOVER` → `KEVIN STENDAHL` (same person, narration mode), `(captain)` lowercase narration cue → `CAPTAIN` proper noun, `DR. AMELIA HARTFIELD` → `AMELIA` short form, `THE ANNOUNCER` → `ANNOUNCER`. **Design constraints:** (i) prompt must be conservative ("ONLY merge when names CLEARLY refer to the same character; when in doubt, do NOT merge"); (ii) hard-cap the merge-set size (never collapse >50% of cast in one pass — flags hallucination); (iii) only run when `optimization_profile = "Pro (Ultra Quality)"` (adds 2-5 min wall time per run); (iv) feed the LLM the cast list + first 1500 chars of script_text + first sentence of each character's first line for context. **Defer to v2.0-beta** — by then we'll have a real corpus of run logs showing common emission patterns, so the prompt can be data-informed instead of guesswork-driven.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, open source, no API keys.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack sprint only — audio stays at 14.5 GB).
- Audio is king. Full narrative output must never break, shorten, or degrade.

---

## What shipped 2026-04-26 (today's session)

- **`scripts/render_flux_batch.py` --mode bundled** (commit `d5772a3`) — one /prompt POST renders ALL targets in a single ComfyUI queue slot, sharing the FLUX model load and negative CLIP encode. Saved ~8 min on the astrotech 44-composite pass3 (47 → 37 min total). Default mode; `--mode serial` keeps the legacy one-at-a-time loop for granular retry.
- **BUG-LOCAL-064 fixed** (commit `e80f0fc`) — LLMDirector procedural cast no longer leaks the procedural last-name into `cast.description`. The leak was real on the astrotech 2026-04-24 ledger (MEREDITH → "PRIAM SMITHERS - …"). Caught at FLUX-pass1 setup before any image rendered. New regression `tests/test_director_cast_naming.py` + helper `scripts/fix_cast_descriptions.py` for cleaning already-shipped ledgers in place.
- **`scripts/verify_flux_coverage.py`** (commit `d5772a3`) — reads the ledger, walks expected portraits + per-(shot, speaker) composites, scans the output dir, prints PRESENT/MISSING per-target. RADIO ambient beats correctly skip the composite check. Exit non-zero if anything missing. Used to prove **7/7 portraits + 44/44 composites = 100% FLUX coverage of astrotech**.
- **FULL workflow merge — HuMo single-clip demo + trial widget defaults** (commit `d7028fb`) — added 16 HuMo-side nodes from `otr_videoplan_TEST_humo.json` to `otr_scifi_16gb_full.json` (16→32 nodes, 29→50 links). FULL canvas now goes audio → FLUX → ONE HuMo demo clip → mp4 in a single Run. Trial-friendly widget defaults baked in (350 words / 2 chars / no act-breaks / no critique / no open-close / no arc / Standard profile). Multi-clip episode HuMo still goes through `render_humo_batch.py`.
- **Auto-title from spine** (commits `594556e` + `06483ad`) — between the OpenClose evaluator and the ScriptWriter draft, a small LLM call generates a 2-5 word evocative title from the winning spine. Prompt v2 is genre-agnostic (the LLM picks up genre from the spine itself, no on-the-nose tropes). Activates only when user widget is empty / `auto` / a stuck-default. 9 regression tests covering wrappers, stuck-default rejection, overlong-leak rejection, LLM-failure swallowing.
- **`scripts/render_upscale_batch.py`** (commit `eb9152e`) — post-HuMo SeedVR2 1080p upscale orchestrator. Mirrors `external_examples/SeedVR2_HD_video_upscale.json` widget defaults (3B fp16 DiT + ema_vae_fp16, blocks_to_swap=32, batch_size=33, temporal_overlap=3, color_correction=lab). Pattern matches our other `render_*_batch.py` scripts: build one API graph, POST one /prompt, poll /history, confirm output. Estimated 30-50 min wall-clock for a 5-min episode. Decoupled from HuMo render so re-upscaling at different resolutions doesn't need a HuMo redo.
- **Output foldering** — outputs reorganised into `output/otr_stills/` (FLUX pass1 portraits + pass3 composites) and `output/otr_videos/<episode_id>/` (HuMo per-clip mp4s + concat episode mp4 + 1080p upscale). Lookup paths in `render_humo_batch.py` and `verify_flux_coverage.py` search both new and legacy locations for backwards compatibility. Legacy `output/old_time_radio/` retained for v1.5 audio episodes.

## What shipped 2026-04-26 PM (peer-review fixes + Captain-Eris-Violet promotion + cache-thrash fix)

- **Peer-review fixes from `cowork_full_workflow_audit.md`** (commit `733f0db`):
    - **ITU BS.1770 loudness normalization** in `render_episode_concat.py` mux step (`-af loudnorm=I=-16:TP=-1.5:LRA=11`). Prevents quiet/loud mismatch when mixing HuMo audio with proc-gen Bark/Kokoro/MusicGen layers — broadcast-grade target across the final master.
    - **`is_clip_readable()` ffprobe corrupt-clip guard** — every HuMo mp4 gets `ffprobe -show_entries format=duration` checked before joining the concat list. Corrupt clips drop into a `corrupt[]` list and are excluded from concat (surfaced alongside `missing[]` in the status print). Stops one busted HuMo clip from poisoning the whole episode master.
- **BUG-LOCAL-065 fixed — LLM cache-mismatch reload thrash** (commit `ddfa392`) — between LLM phases (title-gen → Open-Close → draft → critique → revise → arc-enhancer) the cache-mismatch check at `_load_llm` was firing false-positive evictions because the eviction probe used `next(model.parameters()).device` and bnb 4-bit's first parameter is non-deterministically a CPU-resident metadata buffer. Each false-positive triggered a full unload + cold-load (~15-17 s × 4-6 cycles per FULL run). Fix scans up to 8 parameters (any cuda → resident) and replaces the opaque boolean check with explicit per-field delta collection so the runtime log names the drifting field. New regression `tests/test_llm_cache_mismatch_diagnostics.py` — 12 tests covering no-drift across all phase boundaries, quantized-model false-positive guards, and legitimate-mismatch coverage. Bible candidate.
- **Captain-Eris-Violet promoted to default LLM** (commit `74e4e81`) — `Nitral-AI/Captain-Eris_Violet-V0.420-12B` is now the dropdown default in both `OTR_LLMScriptWriter` and `OTR_VisualLLMSelector`. Dialogue-first RP fine-tune of Mistral-Nemo: same architecture, same tokenizer, same VRAM footprint, but with explicit RP/dialogue training that holds character voice across long scenes — a better fit for the OTR `[CHARACTER, mood] dialogue` format than a pure narrative-prose fine-tune. Mistral-Nemo base kept as the validated fallback that cleared BUG-061/062/063. Gemma 2/4 variants kept as smaller alternates; Qwen-2.5-14B kept as alpha. Both shipped workflow JSONs (`otr_scifi_16gb_full.json` + `otr_scifi_16gb_TEST.json`) updated to match. Four function defaults aligned: `_load_llm`, `_generate_with_llm`, `_CURRENT_LLM_MODEL`, `write_script`. Test fixtures updated; full regression green (92 passed, AST OK, lockstep verified).

### LLM dropdown order (v2.0-alpha)

1. `Nitral-AI/Captain-Eris_Violet-V0.420-12B` ← default (dialogue-first RP fine-tune)
2. `mistralai/Mistral-Nemo-Instruct-2407` ← validated fallback
3. `google/gemma-2-2b-it`
4. `google/gemma-2-9b-it`
5. `google/gemma-4-E4B-it`
6. `Qwen/Qwen2.5-14B-Instruct [ALPHA]`

If Captain-Eris-Violet produces format-gate failures (BUG-061/062/063 family), rollback is to flip the default back to `mistralai/Mistral-Nemo-Instruct-2407` in the four function defaults + dropdown + the two workflow JSONs.

### FULL workflow trial outcome (Long Goodbye, 2026-04-26 12:25 PM)

`The Long Goodbye / cyberpunk / 2 chars / maximum chaos creativity / hard-sci-fi procedural style / Pro (Ultra Quality) profile / self_critique on / open_close on`. Edge-case loadout to stress the format-drift safety nets.

What worked:
- News fetch, OpenClose 3-spine competition + evaluator merge, draft + self-critique + revision pass, PARSE, Director, Bark + Kokoro + MusicGen + AudioGen, SceneSequencer + AudioEnhance + EpisodeAssembler, FLUX (BatchFluxRender → 2 envs), HuMo single-clip demo (16,531 MB Staged + 1053 patches attached on the lightx2v + ModelSamplingSD3 + HuMo 14B fp8 stack)
- BUG-LOCAL-061/062/063 hardening held: bracket-shorthand normalised, dialogue density preserved, no zero-line WORD_EXTEND rescue triggered

Key finding (real, not a regression):
- The FULL workflow's HuMo path and `OTR_SignalLostVideo` audio-only path run as **parallel sinks, not a chain**. The final `signal_lost_the_long_goodbye_20260426_125131.mp4` is the SignalLostVideo proc-gen audio episode. The HuMo demo clip landed at `output/video/ComfyUI_00002_.mp4` and was never stitched in. Multi-clip HuMo episodes need a separate concat-and-mux orchestrator (see P1 delivery chain below).

---

## v2.0-alpha P1 — End-to-end delivery chain (in-graph batch architecture, current focus)

Production HuMo + composite work runs as **visible nodes inside `otr_scifi_16gb_full.json`** -- no subprocess, no hidden orchestrator. Version target is **v2.0-alpha** (NOT v2.5 -- Jeffrey explicit). Architecture pivoted 2026-04-27 from the BUG-076 subprocess pattern (OTR_PostAudioVideoPipeline + test_humo_batch_concat.ps1 + render_humo_batch.py) to in-graph nodes after Jeffrey called the subprocess design "hidden": "humo batching should happen in node in workflow not some hidden thing".

### Architecture as locked 2026-04-27 (in-graph)

```
FULL workflow (otr_scifi_16gb_full.json -- 25 nodes, 41 links post-pivot)
  │
  ├─ Audio path (unchanged)
  │   Story → Director → SceneSequencer → AudioEnhance → EpisodeAssembler → SignalLostVideo
  │   Output: signal_lost_<id>.mp4 at 1920x1080, audio-reactive CRT proc gen + AAC 48 kHz audio embedded
  │
  ├─ FLUX environment stills (unchanged)
  │   VideoPlan → ShotDurationCalculator → BatchFluxRender → UnloadAll → SaveImage
  │   Output: full_env_*.png at output/otr/stills/  (used as PASS1 portrait stand-ins per BUG-078)
  │
  ├─ NEW HuMo loader chain (cold-loads HuMo model family)
  │   UNETLoader (Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors)
  │      → LoraLoaderModelOnly (lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16)
  │         → ModelSamplingSD3 (shift=8)                              ─┐
  │   CLIPLoader (umt5_xxl_fp8_e4m3fn_scaled, type=wan)               ─┤
  │   VAELoader (wan_2.1_vae.safetensors)                             ─┤
  │   AudioEncoderLoader (whisper_large_v3_fp16.safetensors)          ─┤
  │                                                                     ↓
  ├─ NEW OTR_BatchHumoRender                       [SHIPPED — BUG-078]
  │   Single execute() call. Loads HuMo + Lora + CLIP + VAE + Whisper once,
  │   loops every ledger line internally, renders per-line lip-sync clips
  │   via direct ComfyUI node calls (CLIPTextEncode / KSampler / VAEDecode /
  │   WanHuMoImageToVideo / AudioEncoderEncode / CreateVideo / SaveVideo).
  │   Per-line clip is renamed from SaveVideo's humo_<line_id>_NNNNN_.mp4
  │   to canonical <line_id>.mp4 in-place (BUG-074 / BUG-075 convention).
  │   • clip_length=7.0 (HuMo sweet spot at length=175 -> 177 frames @ 25 fps = 7.08s actual)
  │   • max_clips=0 (unlimited; smoke test caps with positive int)
  │   • OUTPUT_NODE=True per BUG-077 lesson (without it, executor prunes silently)
  │   Output: output/otr/videos/<ep_id>/<line_id>.mp4 (per ledger line)
  │
  └─ NEW OTR_VideoComposite                         [SHIPPED — BUG-078]
      Single ffmpeg invocation. Reads SignalLostVideo's mp4 as audio-reactive CRT base
      + audio source. Pillarboxes HuMo clips at 624x1080 center in 1920x1080 canvas,
      additive-blends proc gen on top at 50% opacity, maps audio from proc gen mp4.
      No separate audio mux step.
      Output: output/otr/episodes/<ep_id>/<ep_id>.mp4
```

### Composition geometry (locked)

- Final canvas: **1920x1080** at **25 fps**
- HuMo center pane: **624x1080** (HuMo native 480x832 portrait → lanczos-scale to 1080 height)
- Pillarbox sides: **648 + 648 = 1296px** filled by SignalLostVideo proc gen
- Blend over HuMo: proc gen at `all_mode=addition`, `all_opacity=0.5`
  - black pixels of proc gen contribute zero (additive identity), HuMo face stays clean
  - bright CRT waveform/spectrum bars glow over HuMo face during dialogue
  - and continue across pillarbox wings during silence (where HuMo isn't present)
- Audio: from SignalLostVideo proc gen mp4 (48 kHz AAC, full episode), do NOT separately mux
- blend_mode dropdown supports addition / screen / lighten / overlay / normal for A/B

### Per-line timing source

Ledger lines do not yet carry `dur_s` reliably (SceneSequencer-populated only). Fallback chain in `OTR_VideoComposite._build_clip_timeline`:
1. `ledger.lines[].dur_s` if positive
2. ffprobe the actual rendered `<line_id>.mp4` for true duration
3. fallback `clip_length` (default 7.0s)

Cumulative `start_s` is the sum of prior `dur_s` -- HuMo clips chain back-to-back from t=0. Real audio-aligned timing requires SceneSequencer to populate per-line `start_s` (separate work item).

### What's retired

- `OTR_PostAudioVideoPipeline` node (subprocess trigger, BUG-076) -- removed from FULL workflow JSON, class kept registered with title suffix "(retired)" for back-compat with old JSONs.
- `scripts/test_humo_batch_concat.ps1` (PowerShell wrapper) -- still works for ad-hoc smoke testing via CLI; not invoked from the FULL workflow.
- `scripts/render_humo_batch.py` (subprocess orchestrator) -- still works for ad-hoc smoke; not invoked from the FULL workflow. The 7s clip-length logic, BUG-074 fixes, daisy-chain mechanic all now live INSIDE `OTR_BatchHumoRender`.
- `scripts/render_episode_concat.py` (subprocess concat) -- ditto. Composite logic now in `OTR_VideoComposite`.

### Pending work (in build order)

1. **Real PASS1 portrait render path** -- currently `_find_portrait` falls through to `full_env_*.png` (FLUX env stills) as visual stand-ins because no node renders character portraits to disk. Requires either (a) generalizing BatchFluxRender to accept any token type filter, then wiring a second invocation to `VideoPlan.pass1_char_prompts_json` with a SaveImage prefix targeting `otr/stills/pass1_portrait_*.png`; or (b) a sibling `OTR_BatchFluxPortraitRender` node. Without this, faces in the composite are visually wrong (env stills as character placeholders).
2. **Real audio-aligned per-line timing** -- requires SceneSequencer to populate `lines[].start_s` and `lines[].dur_s` from the assembled audio timeline. Currently those fields are null in the ledger. With them populated, `OTR_VideoComposite` can position HuMo overlays at real speech windows instead of cumulative-from-zero.
3. **`upscale_clips: BOOLEAN` toggle on OTR_BatchHumoRender** -- when on, run SeedVR2 3B fp16 (`render_upscale_batch.py` already shipped) on each clip at `--resolution 624` before the move-rename step. ~22 min GPU surcharge for sharper face detail in the pillarbox center pane.
4. **Optional 4K final pass** -- second OTR_VideoComposite invocation OR a Phase 4 SeedVR2 pass on the 1920x1080 composite at `--resolution 2160`.

### Superseded designs (kept for context, do NOT build)

- The earlier `render_compose_frame.py` plan with vintage radio cabinet PNG + filament glow + analog VU needles is SUPERSEDED. SignalLostVideo's `_CRTRenderer` + `_TelemetryHUDRenderer` already produces the period-authentic audio-reactive layer; the additive blend in OTR_VideoComposite puts it on top of HuMo. Single source of truth.
- The subprocess pattern (BUG-076 OTR_PostAudioVideoPipeline + wrapper script + render_humo_batch.py orchestrator) is SUPERSEDED by the in-graph nodes (BUG-078). Subprocess script remains as a CLI smoke tool but the production path is in-graph.

### Cost estimates (Storms-Sentience-sized, 45 lines, 313s episode)

- Audio path: ~5-10 min CPU+GPU (LLM + Bark + Kokoro + MusicGen + AudioGen + assemble + SignalLostVideo encode)
- FLUX env stills: ~5-10 min GPU (currently 13 shots @ ~26s/shot batched)
- HuMo cold load: ~30-60s GPU one-time per workflow run
- HuMo per-line render: ~265s/clip × 45 = **~3h20m GPU**
- ffmpeg composite: **~1-2 min CPU** (single pass, no GPU)
- Optional Phase 4 4K upscale: ~10-15 min GPU

Total without optional upscale: **~4h end-to-end** for a 5-min episode.
With optional 4K final pass: ~4h15m.

---

## P0 — HuMo Full-Episode Coverage (current focus)

**Branch:** `v2.0-alpha`
**State:** Goal 1 scaffolded (orchestrator + concat scripts shipped 2026-04-25), not yet run end-to-end. Goal 2 not started.

This section supersedes the original 14-day video stack sprint as the active P0. The video stack sprint shipped on 2026-04-17 and is retained below as completed history (see "P0 [SHIPPED 2026-04-17] — Video Stack Sprint").

### Hardware floor (measured 2026-04-25)

After ~5 hours of HuMo configuration testing on the RTX 5080 Laptop 16 GB + 64 GB RAM, these numbers are locked. Do not relitigate without new hardware or a major upstream change (FA3 on Blackwell, ComfyUI memory manager rewrite, etc.).

- **Production weight set (locked 2026-04-25 PM, weighted scoring 37.5/45):** **HuMo 14B fp8 e4m3fn scaled (Kijai)** — `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` from [Kijai/WanVideo_comfy_fp8_scaled](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled). Loaded via stock `UNETLoader` (no GGUF wrapper). Validated in the verified RunningHub HuMo+InfiniteTalk A3 stack and tuned by Kijai specifically for 16 GB cards. Replaces previous Q5_K_M GGUF default. Decision details in `docs/2026-04-25-humo-daisy-chain-AB-split.md`. **Fallback ladder (kept on disk, do NOT delete):**
    1. `humo_17B_fp8_e4m3fn.safetensors` from `Comfy-Org/HuMo_ComfyUI` — 36/45, highest raw quality, slower (~6 min/clip at L=177 on this rig). Promote if 14B's quality on talking-head fails the Stage 1 visual inspection.
    2. `Wan2_1-HuMo-17B_Q5_K_M.gguf` from `VeryAladeen/Wan2_1-HuMo_17B-GGUF` — 30/45, speed-tuned. Loaded via city96's `UnetLoaderGGUF`. Keep for fast prompt iteration / smoke runs where wall-clock matters more than HQ output.
- **Stable shape:** `length=97` (3.88 s @ 25 fps), `width=480`, `height=832`, `batch_size=1`. `length=65` triggers a torch.compile RAM spike that pages the Python worker to disk swap (reproduced twice — normalvram + lowvram modes both hung). Do not retry length=65 without a fresh OS process and pinned-RAM config audit.
- **Per-step:** 42 s at length=97. Confirmed identical across fp8 / Q5_K_M GGUF / smart-memory-off / lowvram. ComfyUI is already evicting maximally — the partial-load split (~6.7 GB GPU + ~5.5 GB pinned RAM) is HuMo's own weights, not encoder squeeze.
- **Per-clip:** ~4:30 native (HuMo only), ~6:15 in TEST_humo (FLUX preroll + UnloadAll handoff + HuMo).
- **Cold load:** ~50 s for the first prompt; amortizes across N prompts via Pattern B sequential POSTs because ComfyUI's model cache stays warm.
- **NVIDIA Sysmem Fallback Policy:** "Prefer No Sysmem Fallback" set on `python.exe` for diagnostic clarity. Doesn't change `cudaMallocAsync` behavior.
- **Decisions documented:** `docs/2026-04-25-humo-batch-pipeline.md`.

**Update 2026-04-25 (afternoon, expert continuity brief):**

- **Single-call 7 s viable** at 640×640 fp8_e4m3fn / `length=177` / lightx2v 4-step / uni_pc 6 steps. Heavy PCIe spillover (peak ~12.9 GB VRAM + 17 GB D3D Shared) but stable. Wall-clock ~6 min cold. Verified by single live render against `humo_test_7s.wav`. The earlier "length=97 only" floor was specific to the Q5_K_M GGUF + 480×832 config above; it does not generalise to other config combinations.
- **Frame count must be 4n+1.** Wan 2.1 VAE temporal compression is 4-frames-into-1-latent, so HuMo's `length` widget only accepts `length = 4n + 1` (97, 113, 125, 153, 177 are valid; 175 errors). The `humo_length_for_dur(dur_s)` helper in `scripts/render_humo_batch.py` snaps any duration up to the nearest valid frame count.
- **Architecture finding (`docs/2026-04-25-humo-continuity-brief.md`):** HuMo's `ref_image` is VAE-encoded and **appended at the end** of the temporal latent sequence with `mask=1` on the anchor and `mask=0` on generated frames. Frame 0 is denoised from noise with attention to the anchor — it is **not** a first-frame seed. Naive daisy-chain (`next_ref = prev_last_frame`) will accumulate drift. The practical fix is hybrid blending (Goal 3 below).
- **Out-of-distribution risk:** HuMo was trained on 97-frame clips. 177-frame runs are already OOD before any chaining. Visual quality at 177 is empirically passable but may carry subtle artefacts inherent to the OOD shape, separate from any chaining issues. Fallback path if Stage 0/1 metrics fail unacceptably: drop the clip-fill rule's base from 7.0 s (177 frames) to 3.88 s (97 frames). Every render becomes in-distribution at the cost of more clips per shot.

### Goal 1 — TEST workflow renders every shot with HuMo

**Definition of done:** Queue a TEST-style HuMo workflow and end up with one HuMo MP4 per shot the OTR_VideoPlan would have rendered as a FLUX-only PASS3 composite. Audio is sliced from the master WAV per ledger timing (or `--auto-slice` when timing is missing).

**Pieces shipped 2026-04-25 (`c387e525`):**

- `scripts/render_humo_batch.py` — Pattern B orchestrator. Reads ledger, slices audio, copies portraits to ComfyUI `input/`, POSTs HuMo prompts to `:8000/prompt` sequentially. Scope flags: `all` / `first-per-scene` / `cold-open` / `custom:l001,l005,...`. `--auto-slice` fallback for ledgers without per-line timing. Pure Python stdlib + ffmpeg subprocess.
- `scripts/concat_humo_episode.py` — ffmpeg stitcher. Two modes: `concat` (back-to-back clips, master WAV replaces audio) and `overlay` (clips composite onto base track at line.start_s/dur_s positions).
- `workflows/otr_videoplan_TEST_humo.json` — TEST workflow wired with `UnetLoaderGGUF` → `Wan2_1-HuMo-17B_Q5_K_M.gguf`. Reverted to length=97 stable shape.
- Recipe + decision log: `docs/2026-04-25-humo-batch-pipeline.md`, `docs/2026-04-24-humo-poc-recipe.md` (corrected fp8 size + Q5_K_M / Q4_K_S guidance committed `2093a14`).

**Pieces shipped 2026-04-25 (Goal 1 prep, this session):**

- `scripts/build_test_ledger_from_director.py` — adapter that reads a TEST-style workflow's baked-in `director_json`, expands the pass3 shot plan via `nodes.otr_video_plan.build_shot_plan`, and emits a synthetic ledger with `cast[]` + `lines[]` (one line per shot, speaker rotated across cast). Bridges the gap between `OTR_VideoPlan.execute()` (which writes `shots[]` only) and `render_humo_batch.py` (which iterates `lines[]`). No workflow edit, no new OTR_* nodes, pure stdlib. 16 unit tests + dry-run verified against `workflows/otr_videoplan_TEST_humo.json` — produces 6 ledger lines (3 scenes × 2 shots/scene) with cycled portraits.
- `tests/test_build_test_ledger_from_director.py` — 16 tests covering workflow parsing, director expansion, speaker strategies, schema versioning, and round-trip with `render_humo_batch.filter_lines`. All green.

**Architecture decision (2026-04-25):** Picked path B (orchestrator + small adapter) over path A (mega-workflow with N HuMo subgraphs). Reasoning: the orchestrator already exists, scales to any N via `--scope all`, and keeps the workflow JSON small. The adapter is the smallest possible bridge — it does not modify the workflow, does not add new ComfyUI nodes, and reuses `build_shot_plan` for shot expansion so the TEST run lines up with the same plan a FULL run would produce. See `docs/2026-04-25-humo-batch-pipeline.md`.

**Remaining for Goal 1:**

- [x] **Adapter + dry-run** — `build_test_ledger_from_director.py` against `otr_videoplan_TEST_humo.json` writes a 6-line ledger; `render_humo_batch.py --scope all --dry-run` plans 6 prompts cleanly with portraits resolved per speaker. Verified 2026-04-25 in this session.
- [ ] **Smoke run (live ComfyUI)** — Jeffrey runs the two-step block below against a live ComfyUI server at :8000. Confirms warm-cache assumption + HTTP flow on real hardware. Expect ~6:15 for clip 1 (cold load), ~4:30 each for clips 2-6 → ~30 min total.
- [ ] **Scale up** — same flow on the FULL ledger once it lands; drop scope cap. For a 7-min episode at ~4:30/clip, ~6 h overnight.
- [ ] **Concat run** — `concat_humo_episode.py --mode concat` against the clip directory + master WAV. Verify the final MP4 plays end-to-end and audio aligns with the visible clips.

**Smoke-run command block (Jeffrey to execute when ComfyUI is up):**

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

# 1. Render the 3 character portraits via the existing TEST_humo workflow.
#    Open the workflow in ComfyUI Desktop (localhost:8000) and Queue Prompt
#    once. Confirms portraits land at:
#      C:\Users\jeffr\Documents\ComfyUI\output\otr_humo_pass1_portrait_*.png
#    The in-graph HuMo step also produces 1 smoke clip — that's expected.

# 2. Build the synthetic test ledger from the baked-in director_json.
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\build_test_ledger_from_director.py `
  --workflow workflows\otr_videoplan_TEST_humo.json `
  --out output\old_time_radio\test_humo_ledger.json

# 3. Render N HuMo clips against the ledger. With LEMMY/Saturn this gives
#    6 MP4s landing in output\old_time_radio\humo_test\.
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\render_humo_batch.py `
  --ledger output\old_time_radio\test_humo_ledger.json `
  --master-wav C:\Users\jeffr\Documents\ComfyUI\input\humo_test.wav `
  --out-dir output\old_time_radio\humo_test `
  --scope all
```

**Gates:**

| Gate | Threshold |
|---|---|
| Smoke run wall clock | First clip ≤ 6:00 (cold load), clip 2 ≤ 5:00 (warm) |
| Per-clip steady-state | ≤ 4:45 after warm |
| MP4 validity | ffprobe-parseable, > 100 KB, contains audio + video streams, duration ≈ 3.88 s |
| VRAM peak across run | ≤ 14 GB sustained (no pagefile thrash, no swap-out symptoms) |
| Failure recovery | If a single clip fails, orchestrator continues to next clip and reports failure count at end |

### Goal 2 — FULL pipeline covers every shot with HuMo (not yet started)

**Definition of done:** Run `otr_scifi_16gb_full.json` end-to-end, walk away, return to a finished episode MP4 where every line of dialogue is HuMo-animated. The Director's script is consumed as-is — no human-in-the-loop curation of which lines deserve HuMo treatment. The pipeline is fire-and-forget.

**Required upstream pieces:**

- [ ] **Ledger L2 (Task #20)** — SceneSequencer populates `ledger.lines[].start_s` and `dur_s`. Without per-line timing, the orchestrator falls back to `--auto-slice` which doesn't respect line boundaries (lip-sync drift accumulates across the episode). Goal 2 quality depends on Ledger L2 landing first.
- [ ] **FULL→HuMo handoff** — after the FULL pipeline writes the master WAV, the orchestrator must auto-trigger. Two options:
  - A new ComfyUI node (e.g. `OTR_HuMoBatch`) at the end of the FULL graph that shells out to `render_humo_batch.py` then `concat_humo_episode.py`. Per existing memory, prefer external script over OTR_* custom node — so:
  - A post-FULL script (`scripts/run_full_then_humo.py`) that watches for ledger completion, then chains the orchestrator + concat. Invokable as a single command.
- [ ] **Resumability** — if the long render dies at clip 50 of 80, restart should skip the first 50. Add `--resume` flag to `render_humo_batch.py` that skips `line_id`s whose MP4 already exists in `--out-dir`.
- [ ] **Episode meta.json** — total HuMo clips rendered, total wall clock, per-clip durations, lip-sync drift warnings, ffmpeg concat command logged for reproducibility.

**Gates:**

| Gate | Threshold |
|---|---|
| End-to-end unattended | A 7-min episode completes FULL + HuMo coverage in one overnight run, ≤ 10 hours total |
| Zero human edits | No prompts, no clicks, no manual file moves between FULL queue and final MP4 |
| Output validity | Final MP4 plays end-to-end, audio matches master WAV duration, every line is on-screen |
| Lip-sync drift | Visible only at HuMo's 3.88 s window boundaries (acceptable; not a regression vs Goal 1 single-clip lip-sync) |
| Resumability | Killing the orchestrator mid-render and restarting completes without re-rendering existing clips |

**Out of scope for Goal 2 (defer):**

- Mixing HuMo with Wan 2.2 + SVI Pro for atmospheric (non-talking) clips
- FLUX-still-with-pan-zoom as a non-HuMo fallback for narration / sound-design beats
- Director scoring of "headline" lines for higher-fidelity treatment
- FA3 / Blackwell flash-attention 3 (not yet available on this driver / torch combo)

### Goal 3 — Shot continuity via hybrid anchor (research-validated path)

**Definition of done:** Within a single shot (one speaker, one setting), N chained HuMo clips concatenate into a seamless ~28-second sequence with no visible cut every 7 seconds and no character-identity drift across hops. At shot boundaries (new speaker OR scene-background change — the only two cut conditions), a fresh FLUX portrait re-anchors the chain. Per-hop metrics (seam SSIM, ArcFace identity cosine) are written to a CSV alongside the clip set so the orchestrator can flag drift and propose anchor resets without manual review.

**Source of plan:** `docs/2026-04-25-humo-continuity-brief.md` — expert brief, model-source-verified, includes architecture finding (ref image appended at end of temporal latent sequence, attention anchor not first-frame seed) and the staged validation gates below.

**Architecture context (already in Hardware floor section above):**

The naive chain `next_ref = prev_last_frame` will drift because the ref image is an attention anchor, not a frame 0 seed. The practical fix is hybrid blending: `ref_image = α × clean_portrait + (1−α) × prev_last_frame` with α swept on a 2-clip seam test. Stage 2 below picks α; Stages 0 / 1 / 3 gate whether we even need it.

**Staged validation (sequential, gated — do not skip ahead):**

| Stage | Goal | Gate to advance |
|---|---|---|
| **0. Disambiguation** | One clip from a clean FLUX portrait. Compute SSIM(input_portrait, F1_0). | `> 0.97` → first-frame seeding (naive chain works, redesign Stage 2). `0.80–0.93` → end-of-sequence anchor (proceed). `< 0.80` → weak conditioning (raise scale_t, re-check VAE). |
| **1. Two-clip seam** | Render C1 with clean portrait, save F1_176. Render C2 with `ref = F1_176`, save F2_0. Measure `seam_ssim(F1_176, F2_0)` and `id_cosine(F1_0, F2_0)` via ArcFace. | `seam_ssim ≥ 0.95 AND id_cosine ≥ 0.85` → skip to Stage 4 (naive chain holds). Else → Stage 2. |
| **2. Hybrid anchor α sweep** | ComfyUI-native: insert `Image Blend` before VAE Encode. Sweep α ∈ {0.5, 0.7, 0.9} on a 2-clip seam test. Pick highest seam_ssim where id_cosine ≥ 0.85. | One α value selected with metrics that beat Stage 1 numbers. |
| **3. 5-hop validation** | Chain 5 clips with the chosen α. Per hop: `seam_ssim`, `id_vs_clean`, `id_vs_F1_0`. Drift trigger: `id_vs_clean < 0.80` → force clean re-anchor on next clip (α=1). Hard reset: `id_vs_clean < 0.70` → declare a cut, require motivated camera move from the writer. | All 5 hops complete without hitting the hard-reset threshold; drift trigger fires at most once. |
| **4. Cowork orchestration** | Pipe ComfyUI logs to the orchestrator. Per-hop metrics auto-emitted as JSON. Auto-generated stitch list with continuity flags for Resolve/Filmora. Error parser maps ComfyUI tracebacks to actionable fixes. Audio slice manager aligns 7 s windows to phoneme boundaries (avoid mid-syllable cuts). | Orchestrator reads a run log, computes metrics, proposes the next-clip anchor strategy without Jeffrey opening the ComfyUI UI between hops. |

**Orchestrator chain mechanic — schema l2 boundary state machine (MVP shipped 2026-04-25 PM):**

The chain happens inside `scripts/render_humo_batch.py` and reads each ledger line's `boundary` field (added by the L2 schema bump). The 4-state machine drives the per-clip reference image decision:

| `boundary` | When it fires | MVP anchor recipe (shipped) | Stage-2-tuned hybrid (deferred until metrics) |
|---|---|---|---|
| `shot_start` | First clip of a new shot (or first clip of run) | `ref = clean_portrait[speaker]` (or per-(shot, speaker) FLUX composite when present) | same |
| `beat_start` | Same shot, NEW speaker (never seen in this shot before) | `ref = clean_portrait[speaker]` — clean reset, no chain | same |
| `beat_resume` | Same shot, RETURNING speaker (was interrupted by another beat earlier in this shot) | `ref = speaker_last_frame_in_shot[speaker]` — chain back to that speaker's own last frame in this shot, NOT the interrupting speaker's frame | `ref = α × portrait + (1−α) × speaker_last_frame_in_shot[speaker]` |
| `continue` | Same beat, same speaker, next clip | `ref = prev_last_frame` — naive daisy chain from immediately preceding clip | `ref = α × portrait + (1−α) × prev_last_frame` |

State the orchestrator tracks across the run:

- `prev_last_frame_path` — single value, the last MP4's terminal frame (drives `continue`)
- `prev_shot_id` — used to detect shot transitions
- `speaker_last_frame_in_shot[speaker]` — per-speaker map, **cleared whenever shot_id changes**, so a returning speaker in a new shot doesn't chain from their prior shot's frame

After every successful render the orchestrator:

1. Moves the rendered MP4 to `--out-dir/<line_id>.mp4`
2. Extracts the last frame as PNG via `ffmpeg -sseof -0.04 -i <mp4> -frames:v 1 -update 1 <line_id>_last.png` (the `extract_last_frame` helper)
3. Updates `prev_last_frame_path` and `speaker_last_frame_in_shot[speaker]`

**MVP shipped 2026-04-25 (`scripts/render_humo_batch.py`):**

- All four boundary types honoured (naive — no alpha blend yet)
- `--no-chain` flag falls back to fresh-portrait-every-clip for A/B comparison
- Chain state cleared on shot transition
- ffmpeg `extract_last_frame` helper — last-frame PNG saved alongside each clip's MP4
- Per-clip log line shows the chain source: `clean portrait` / `prev clip last frame` / `speaker last frame in shot (<filename>)`

**Deferred until Stage 0 / 1 / 2 metrics land:**

- Alpha-blend hybrid anchor (`α × portrait + (1−α) × prev_frame`) — needs Stage 2 α sweep to pick the value
- ArcFace identity cosine + SSIM seam metrics → `metrics.csv` per run
- Drift triggers (`id_vs_clean < 0.80` → force clean re-anchor; `< 0.70` → declare a hard cut)

The MVP is enough to test whether naive chaining holds within HuMo's training distribution. If Stage 1 seam_ssim ≥ 0.95 with naive chain, we're done. If not, the alpha blend lands as a ~30-line incremental patch on top of the MVP.

**Alternative paths within Goal 3 (A-tier, current HuMo, fact-checked 2026-04-25 — `docs/2026-04-25-humo-daisy-chain-AB-split.md`):**

The hybrid-anchor mechanic above (Stages 0-3 + orchestrator integration) is one of three paths that deliver continuity on the *current* HuMo model. Pick after Stage 0/1 metrics land — the right choice depends on whether the seam test passes naively, marginally, or not at all.

| Path | What it does | When it's the right pick | Caveats |
|---|---|---|---|
| **A1. Single-window** | Stay inside HuMo's native 97-frame (3.88 s) training distribution. No chain, no continuity. Every shot is one HuMo call max. | Stage 0 SSIM is well below 0.97 AND we want zero artefacts. Smallest, safest baseline. | Hard cap on shot length at 3.88 s; multi-clip shots become hard cuts. |
| **A2. Hybrid-anchor RGB chain** (default plan above) | Extract clip N's last frame, ADAIN/mkl colour-match **back to the original FLUX portrait** (never to clip N's last frame), optional IP-Adapter Identity Wash anchored to the clean portrait, blend at α with the portrait, feed as clip N+1's `ref_image`. | Stage 1 seam_ssim ≥ 0.95 AND id_cosine ≥ 0.85 with α-blending. | Practical limit ~3-4 hops before drift becomes visible. Refresh portrait every 4 clips. |
| **A3. HuMo + InfiniteTalk stack** (verified [RunningHub workflow 1968348721056501761](https://www.runninghub.ai/post/1968348721056501761)) | Run HuMo for character/identity, hand off to InfiniteTalk for long-video lip-sync via its `motion_frame` mechanism. ~15 s seamless output per RunningHub's MV reference. | A2 fails or the seam is unacceptable AND we can spare disk for an extra ~18 GB of weights. | **Requires `Wan2_1-HuMo-14B` (Kijai fp8 scaled, NOT our current 17B GGUF) + InfiniteTalk weights from [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) + `whisper_large_v3_encoder_fp16.safetensors`.** Pipeline diverges from current single-call HuMo orchestrator — InfiniteTalk becomes the long-video driver, HuMo becomes the identity provider. |

**Skip list (locked — do not relitigate without new findings):**

| Tempting | Why not |
|---|---|
| Fork DiT for true frame-0 latent injection | 2-week detour, hybrid anchor gets ~80% of the gain |
| Switch to Wan 2.1 i2v (alone, without HuMo) | Loses audio-driven motion, the reason for HuMo |
| 30-second continuous-shot target on current HuMo | Architecture won't support it; plan cuts every 2 clips. A3 stack lifts the ceiling to ~15 s. |
| Increase frames beyond 177 on current HuMo | Already OOD vs 97-frame training |
| Feed HuMo into `WanVideo Long I2V Multi/InfiniteTalk` directly | Per kijai issue [#1941](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1941), that node is built for InfiniteTalk-as-primary, not HuMo-as-primary. The verified workflow is the **stacked** A3 path above. |

**Deliverable checklist:**

- [ ] Stage 0 SSIM result (one number, decides everything downstream)
- [ ] Stage 1 seam metrics (two numbers, pass/fail gate)
- [ ] Stage 2 α sweep table (three rows)
- [ ] Stage 3 5-hop metrics CSV
- [ ] Stage 4 log-reader → metrics → drift flag → reset proposal wired into the orchestrator

**Why Goal 3 sits between Goal 1 and Goal 2 (the critical-path argument):**

Goal 1 proves the orchestrator submits N HuMo prompts, gets N MP4s back, and stays alive. That's an *orchestration* test — the visual quality of those MP4s is not the gate. Goal 3 is the *quality* gate: without continuity, every shot >7 seconds has visible 7-second cuts inside it, and a 30-second narrative beat looks like four hard jump-cuts. Goal 2 (FULL pipeline unattended) cannot ship a deliverable-quality episode without Goal 3 already in place. So the dependency chain is **Goal 1 (mechanics) → Goal 3 (quality) → Goal 2 (production)**, and any attempt to do Goal 2 first would produce visibly broken episodes that need Goal 3 retrofitted anyway.

### Parking lot — long-video continuity options for the post-HuMo era (B-tier)

When HuMo's window-bounded architecture stops being the right tool — either because we hit a quality ceiling on Stages 0-3 we can't blend our way out of, or because a model with native long-video support lands and changes the trade-off — these are the candidates already fact-checked. Each requires a separate model swap and pipeline rebuild. Do NOT pull weights or rewrite the orchestrator for any of these until A-tier hits a wall on real metrics.

| Option | What it is | Why parked |
|---|---|---|
| **B1. InfiniteTalk + Wan 2.1 / 2.2 (alone, no HuMo)** — [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) | Audio-driven talking-head with native long-video via `WanVideo Long I2V Multi/InfiniteTalk` node. `motion_frame` mechanism with built-in colour correction. GGUF variants fit 16 GB. | Promote when HuMo's character preservation isn't the bottleneck and continuity is. Trade: weaker identity hold, much stronger seam quality. |
| **B2. Stable Video Infinity 2.0 (Wan 2.2 I2V A14B base, LoRA)** — [vita-epfl/Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity) (ICLR 26 Oral) | LoRA on Wan 2.2 I2V. 5-pass chained sampling with motion-latent forwarding + tail-frame blending. "Error Recycling Fine Tuning" approach removes drift. **No audio conditioning** — text/image only. | Promote for long non-talking shots: cutaways, environment beats, sound-design holds. Not a HuMo replacement; a sibling for clips where lip-sync isn't needed. Fits 16 GB with GGUF Wan 2.2 base. |
| **B3. LTX 2.3 (22B, native audio-video sync)** | Native audio-video sync at the model level. 4K / 50 fps capable. | **Does not fit 16 GB** — minimum 24 GB VRAM. Park until next hardware refresh or a quantised variant ships. Already aligned with existing memory `reference_ltx_keep_only_2b.md` (only 2B v0.9 retained on this rig). |
| **B4. HuMo "Longer Generation" official checkpoint** ([Phantom-video/HuMo](https://github.com/Phantom-video/HuMo)) | TODO in HuMo README, promised October 2025, vapor as of Apr 2026. If/when it ships it would supersede A1/A2/A3 entirely — native long-video on the same model. | Set GitHub watch on the repo. Revisit Goal 3 entirely if released. |

**Why these are parking-lot, not active P0:** the smoke run of the current HuMo orchestrator hasn't even produced its first end-to-end episode yet (Goal 1). Promoting any B-tier option now would mean rebuilding the orchestrator + downloading 18-50 GB of new weights before we have a deliverable to compare against. Goal 1 ships first, Stage 0/1 metrics tell us whether A2 (hybrid anchor) holds, A3 (HuMo+InfiniteTalk stack) is the in-family upgrade if it doesn't, and only THEN do B-tier alternatives become live decisions.

### Why these three goals in this order

Goal 1 proves the orchestrator runs end-to-end on real hardware with real ledgers. It validates Pattern B's warm-cache hypothesis at scale, exposes any bugs in the audio slicing or portrait selection, and gives a known-good output to compare against. Goal 3 then lifts the output from "55 disjoint clips with visible cuts every 7 s" to "a seamless episode with cuts only at intentional boundaries" — without it, the orchestrator's output is mechanically correct but visually unwatchable. Goal 2 wraps the result in unattended automation. Reversing any of these orderings — wiring auto-trigger before proving the orchestrator works, or shipping unattended runs before continuity is solved — risks 10-hour overnight runs that produce broken episodes we'd then have to throw away.

### Open prerequisites

- **Task #20** (Ledger L2 — per-line `start_s` / `dur_s` populated by SceneSequencer): blocks Goal 2 quality. Goal 1 can proceed with `--auto-slice` while Task #20 ships.
- **Task #25** (OTR_LoadLedger node): non-blocking. Lets TEST workflow replay an existing ledger for HuMo iteration without paying the LLM cost — useful if Goal 1 needs many iterations on prompt wording or portrait selection.
- **Task #34** (tail-end OTR_UnloadAll on TEST_humo): non-blocking but reduces zombie residual between runs. Apply opportunistically.

### Kill criteria (fail-fast)

| Trigger | Response |
|---|---|
| Per-step time > 60 s sustained on a known-good config | Stop. Re-validate hardware (LHM telemetry), driver, and that no other process is competing for VRAM. Do not chase per-step optimization. |
| Process paged to disk swap (negative `WorkingSet`) | Kill, restart ComfyUI, do not retry the same shape change. The 2026-04-25 length=65 hangs are the precedent — the system RAM ceiling is real. |
| Clip output is silent video / black frames | Whisper audio format mismatch. Verify mono 16 kHz WAV. |
| Lip-sync visibly drifts mid-clip | Audio slice longer than HuMo's 3.88 s native window. Cap at 3.88 s in orchestrator. |
| FULL pipeline regression on Goal 2 wiring | Revert immediately. Audio is king (C7) — never let HuMo break the audio path. |

---

## P0 [SHIPPED 2026-04-17] — Video Stack Sprint (14-day build)

Branch: `v2.0-alpha`. Tag target: `v2.0-alpha-video-full`.
Supersedes the retired Visual 2.0 Gate 0 probe. The VisualBridge → VisualPoll → VisualRenderer trio (shipped) stays as the harness; the backends swap.

### Locked stack

| # | Stage | Pick | Runtime | Peak VRAM | Canonical repo |
|---|---|---|---|---|---|
| 1 | Style anchors | FLUX.1-dev FP8 + ControlNet Union Pro 2.0 | diffusers | 12.5 GB | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` |
| 2 | Scene keyframes | FLUX.1-dev + Depth/Canny | diffusers | 13.5 GB | `XLabs-AI/x-flux` (weights) |
| 3 | Character lock | PuLID for FLUX | diffusers | 14.0 GB | `ToTheMoon/PuLID` *(verify Day 3)* |
| 4 | Hero motion | LTX-Video 2.3 | existing sidecar | 14.5 GB | `Lightricks/LTX-Video` |
| 5 | Long motion / VJ loops | Wan2.1 1.3B I2V | diffusers | 8-10 GB | `Wan-Video/Wan2.1` |
| 6 | Compositing | Florence-2 + SDXL Inpainting | diffusers | 8 GB | `microsoft/Florence-2-large` (HF) |
| 7 | Final mux | VisualRenderer (shipped `86bfeae`) | ffmpeg | — | in-repo |

Post-processing: ffmpeg + OpenCV VHS stylizer (scanlines, chroma bleed, HUDs, lower-thirds).

### Fallbacks (real, reserved — do not promote without cause)

- Stage 1: SDXL 1.0 + 1980s VHS LoRA stack
- Stage 3: InstantCharacter (`Tencent-Hunyuan/InstantCharacter`)
- Stage 4: HunyuanVideo via Nunchaku INT4 (`mit-han-lab/nunchaku`) if LTX quality ceiling hit
- Stage 5: FramePack (`lllyasviel/FramePack`)
- Stage 6: Insert Anything (`song-wensong/insert-anything`)
- FP8 spike escape across any FLUX stage: GGUF Q8/Q5 via `city96/ComfyUI-GGUF` logic ported into sidecar

### 14-day sprint

Every day ends with: `pytest tests/bug_bible_regression.py`, `pytest tests/test_dropdown_guardrails.py`, `pytest tests/test_audio_byte_identical.py`. No exceptions. C7 failure halts and reverts the day's work.

| Day | Task | Gate |
|---|---|---|
| 1 | **[DONE 2026-04-17]** `backends/` harness, `_base.py`, STATUS.json schema, `placeholder_test.py`. Wire Bridge `backend=` arg + LHM cooldown gate. Fixed bridge.py:296-299 PIPE deadlock (stdout/stderr → per-job log files). | ✅ 14/14 new dispatch tests green; 26/26 Bug Bible; 56/56 dropdown guardrails; 22/22 anchor_gen. C7 unchanged. Pre-existing BUG-LOCAL-042 vram_sentinel errors surviving (not caused by Day 1). |
| 2 | **[DONE 2026-04-17]** `flux_anchor.py` — FLUX.1-dev FP8 e4m3fn + enable_model_cpu_offload + VRAMCoordinator gate + deterministic per-shot SHA256 seeds + CI-safe stub fallback (OTR_FLUX_STUB=1 / model-missing / no-CUDA). `requirements.video.txt` pins torch 2.10.0+cu130 / diffusers 0.37.0 / transformers 5.5.0 / accelerate 1.13.0. Also repaired bridge.py (previously truncated mid-execute at line 269 → 446 lines, `_cooldown_gate` / `_spawn_sidecar` / `_write_status` restored; `backend=` arg in INPUT_TYPES + execute signature). | ✅ 10/10 new flux_anchor tests green; 14/14 backend dispatch; 77/77 dropdown+anchor_gen. C7 unchanged. Bug Bible sister repo not mounted in sandbox — Windows-side Bible regression still pending. 1024² real-mode render ≤ 12.5 GB gate deferred until FLUX weights land on disk. |
| 3 | **[DONE 2026-04-17]** `pulid_portrait.py` — PuLID-FLUX identity-locked portrait backend. Real mode: FluxPipeline FP8 + PuLID adapter try-import (`pulid.pipeline_flux` / `PuLID.pipeline_flux` / `comfyui_pulid_flux.pipeline_flux`), `enable_model_cpu_offload`, VRAMCoordinator gate, `id_images`+`id_weight`+`true_cfg` call kwargs. Stub mode (OTR_PULID_STUB=1 / weights missing / no CUDA): deterministic color keyed on `refs_hash` so identity-lock invariant is unit-testable pre-weights. Characters + ref filenames are per-episode emergent from the LLM script process — backend reads `shot.get("character")` and `refs` generically, no fixed roster. | ✅ 16/16 new pulid tests green (registry, stub, identity-lock same→same & diff→diff, helper round-trip); 117/117 combined regression (pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. Face-embedding SSIM identity gate deferred until real PuLID weights land on disk. |
| 4 | **[DONE 2026-04-17]** `flux_keyframe.py` — FLUX + ControlNet Union Pro 2.0 scene keyframe backend. Round-robin consult (`docs/2026-04-17-day4-controlnet__*`) locked: Row 1 Union Pro 2.0 single-mode, Row 2 depth only, Row 3 control image always derived from Day 2 anchor `render.png` (ignores `shot["control_image"]`), Row 4 strict preprocessor sequencing (depth → save → del + empty_cache → load FLUX), Row 5 `depth.png` cached to disk, Row 6 explicit bf16 cast on CN for FP8+bf16 casting safety, Row 7 dedicated Depth CN fallback if Union Pro fails, Row 8 stub mode (`OTR_FLUX_KEYFRAME_STUB=1` / `OTR_FLUX_STUB=1` / weights missing / no CUDA). Output: `keyframe.png` + `depth.png` per shot. Seed base 0x4B_45_59_46 ("KEYF") distinct from flux_anchor + pulid_portrait. | ✅ 28/28 new flux_keyframe tests green (registry, stub mode, layout-lock invariant across 3 prompt variations, Row 3 shotlist control_image ignore, stub-mode envvar permutations, helper determinism); 145/145 combined regression (flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. ≤ 13.5 GB real-mode gate deferred until FLUX + Union Pro 2.0 weights land on disk. |
| 5 | **[DONE 2026-04-17]** `ltx_motion.py` — LTX-Video 2.3 I2V motion sidecar + FLUX still → LTX handoff. Reads upstream still with priority `keyframe.png` (Day 4) > `render.png` (Day 2) > error; records `input_still_source` in meta.json. Real mode tries `LTXImageToVideoPipeline` (preferred) then falls back to `LTXPipeline` (older diffusers) at `torch.float8_e4m3fn` (C5) with `enable_model_cpu_offload`, VRAMCoordinator gate; exports to `motion.mp4` via `diffusers.utils.export_to_video`. C4 enforced: duration_s ≤ 10.0 @ 24 fps. Stub mode (`OTR_LTX_STUB=1` / weights missing): emits a minimal-but-valid MP4 (ftyp + mdat atoms, payload keyed on input-still hash) so handoff determinism is unit-testable without ffmpeg or weights. Seed base 0x4C_54_58_4D ("LTXM") distinct from all prior backends. VRAM isolation achieved structurally via the existing spawn subprocess pattern — FLUX fully releases before LTX loads. | ✅ 29/29 new ltx_motion tests green (registry, stub mode valid MP4 + duration cap, Day 5 handoff priority keyframe>anchor>missing, handoff determinism same-still→same-bytes, different-stills→different-bytes, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe); 174/174 combined regression (ltx + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown + anchor_gen). C7 unchanged. Real-mode ≤ 14.5 GB VRAM gate + clean FLUX→LTX handoff deferred until LTX-Video 2.3 weights land on disk. |
| 6 | **[DONE 2026-04-17]** `wan21_loop.py` — Wan2.1 1.3B I2V loop sidecar + FLUX still → Wan handoff. Inherits Day 5 upstream priority (`keyframe.png` > `render.png` > error) and records `input_still_source` in meta.json. Real mode tries `WanImageToVideoPipeline` first at `torch.float8_e4m3fn` then falls back to `torch.float16` (dtype choice recorded in meta.json) with `enable_model_cpu_offload` + VRAMCoordinator gate, and degrades cleanly to `WanPipeline` (T2V) on older diffusers; exports to `loop.mp4` (not `motion.mp4` — distinct from LTX) via `diffusers.utils.export_to_video`. C4 enforced: duration_s ≤ 10.0 @ 24 fps (240-frame single-call cap). Stub mode (`OTR_WAN_STUB=1` / weights missing / no CUDA): emits minimal-but-valid MP4 (ftyp + mdat atoms) with mdat payload salted `"wan21_loop"` so wan and ltx stubs are byte-distinguishable even for identical still hashes — prevents planner-routing bugs from hiding behind stub identity. Seed base 0x57_41_4E_32 ("WAN2") distinct from all 4 prior backends. Exposes `loop_prompt` (falls back to `motion_prompt` → `env_prompt`) with loopable-motion suffix "seamless loop, subtle cycling motion, 24fps". | ✅ 33/33 new wan21_loop tests green (registry including Days 1-6 roster, stub mode valid MP4 + duration cap + filename gate `loop.mp4` not `motion.mp4`, handoff priority keyframe>anchor>missing, handoff determinism same-still→same-bytes + different-stills→different-bytes, backend isolation: wan vs ltx stubs differ for identical still hash, envvar permutations, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe + ltx_motion); 130/130 combined video backend regression across Days 1-6 (backend dispatch + flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop); 403/403 broader suite pass (10 pre-existing workflow JSON errors flagged on Day 5, not caused by Day 6). C7 unchanged. Real-mode ≤ 10 GB VRAM gate deferred until Wan2.1-I2V-1.3B weights land on disk. |
| 7 | **[DONE 2026-04-17]** `florence2_sdxl_comp.py` — text-prompt mask via Florence-2 `<REFERRING_EXPRESSION_SEGMENTATION>` → SDXL inpaint insert. Inherits Days 5-6 upstream priority (`keyframe.png` > `render.png` > error) and records `input_still_source` in meta.json. Real mode runs in two phases with explicit VRAM handoff: (A) Florence-2 (transformers `AutoModelForCausalLM` + `AutoProcessor`, fp16, trust_remote_code, local_files_only) rasterises polygons/bboxes to `mask.png`, then gets `del`'d + `torch.cuda.empty_cache()` — Day 4 CN handoff discipline; (B) `StableDiffusionXLInpaintPipeline` loads at `torch.float16` (canonical SDXL) with fp8 opt-in via `OTR_SDXL_INPAINT_DTYPE`, `enable_model_cpu_offload` + VRAMCoordinator gate, runs inpaint with `mask_prompt` segmenting and `insert_prompt` painting. Two outputs per shot: `composite.png` (RGB, distinct from Day 4 `keyframe.png`) + `mask.png` (grayscale 8-bit). Stub mode (`OTR_FLORENCE_STUB=1` / either weight tree missing / no CUDA) emits three-way deterministic outputs: `composite.png` color keyed on SHA256(still, mask_prompt, insert_prompt), `mask.png` grayscale value keyed on mask_prompt alone (clamped 1-254 to avoid degenerate all-black/all-white masks), so composite and mask can be regression-tested independently. Seed base 0x46_32_53_44 ("F2SD") distinct from all 5 prior backends. mask_prompt missing triggers per-shot error in real mode (Day 7 requires explicit region naming). | ✅ 40/40 new florence2_sdxl_comp tests green (registry including Days 1-7 roster, stub mode valid PNGs with correct colour-type bytes 2/RGB and 0/grayscale, filename gate `composite.png` not `keyframe.png`, three-way composite invariant [same triple→same bytes; mask-change→shifts; insert-change→shifts], mask-png-depends-on-mask-alone invariant, Day 5-6 handoff priority, envvar permutations, helper determinism with cross-backend seed distinctness across flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop); 170/170 combined video backend regression across Days 1-7 (backend dispatch + flux_anchor + pulid + flux_keyframe + ltx_motion + wan21_loop + florence2_sdxl_comp); 443/443 broader suite pass (10 pre-existing workflow JSON errors flagged on Day 5, not caused by Day 7). C7 unchanged. Real-mode ≤ 8 GB VRAM gate + Florence-2 mask quality gate deferred until both weight trees land on disk. |
| 8 | **[DONE 2026-04-17]** `visual/postproc/vhs.py` — ffmpeg-based VHS aesthetic post-processor. Pure `build_vhs_filter_chain(params)` returns a deterministic `filter_complex` string with seven ordered stages: (1) `format=yuv420p` normalise, (2) `rgbashift=rh=-N:bh=N` chromatic aberration, (3) `gblur planes=6` chroma bleed (U/V only — luma detail preserved), (4) `geq` scanlines (luma-only alternating-row multiplier, density-configurable), (5) `noise=c0s=N:c0f=t+u` tape grain on luma, (6) `vignette=PI/X` soft edge, (7) `gblur` final tape softness. `apply_vhs_filter(input, output, params)` invokes ffmpeg with `-c:a copy` + `-map 0:a?` so audio streams pass through byte-identical when present (C7) or are absent-safely skipped when the input is video-only. Intensity presets low/medium/high scale all five visible knobs proportionally. Stub mode (`OTR_VHS_STUB=1` / ffmpeg missing / `force_stub=True`) is a byte-identical `shutil.copyfile` passthrough, so CI and weight-missing dev machines can unit-test the pipeline without ffmpeg. `apply_vhs_to_job_dir(job_dir)` batch-scans for `render.mp4` > `motion.mp4` > `loop.mp4` per shot, emits `*_vhs.mp4` siblings, skips still images (`composite.png`, `keyframe.png`, `mask.png`, `depth.png`, `anchor.png`, `render.png`), ignores internal `_cache/` and `.hidden/` dirs, and writes a `vhs_postproc_summary.json` meta. Per-clip meta.json alongside each output records mode, stub_reason, params_hash, filter_chain text, ffmpeg argv, duration_ms. Not registered as a backend — `test_postproc_does_not_pollute_backend_registry` asserts the Day 1-7 roster is unchanged. Default `fps=24` asserted equal to `renderer._FPS`. | ✅ 34/34 new vhs_postproc tests green (module imports torch-free; DEFAULT_VHS_PARAMS key coverage; public constants; filter chain deterministic + uses defaults when None + has all 7 structural stages + varies across low/medium/high intensity + unknown intensity → medium fallback + zero-strength knob drops stage + override lands in chain text + scanline density reflected in `mod(Y\\,N)` + vignette always on; stub mode byte-identical passthrough including audio-like trailing payload [C7 invariant] + force_stub overrides env + meta.json schema + env stub reason + ffmpeg-missing autodetect via monkeypatched find_ffmpeg + missing input raises FileNotFoundError + input==output no-clobber; batch finds render/motion/loop + skips still images + mixed shot with both still and video only touches video + renders `render.mp4` takes priority over `motion.mp4` when both exist + ignores internal dirs + empty job dir + missing job dir + batch summary file + params hash stable + params hash shifts with overrides + registry isolation + no shell metacharacters in chain + fps matches renderer._FPS); 281/281 combined video backend regression across Days 1-8 (vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen); 495/509 broader suite (14 pre-existing `test_core.py` BUG-LOCAL-042 `vram_sentinel` ImportError failures/errors from before Day 1 — not caused by Day 8). C7 unchanged (verified structurally: stub = byte-for-byte copy; real = `-c:a copy`). Real-mode wall-clock + CRT quality gate deferred until the Day 10-11 canary renders feed it actual LTX/Wan MP4s. |
| 9 | **[DONE 2026-04-17]** `visual/planner.py` — orchestration timeline planner. Given an outline (dict / JSON string / Path), emits a non-repeating sidecar job list covering full runtime. Each `PlannerJob` names one Day 1-7 backend (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `ltx_motion` / `wan21_loop` / `florence2_sdxl_comp` / `placeholder_test`) plus `shot_id`, `scene_id`, `prompt`, `duration_s`, `refs`, `handoff_from`, `mask_prompt`, `insert_prompt`, `prompt_hash`. Backend assignment: explicit `beat["backend"]` override wins (unknown name → ValueError), else `BEAT_KIND_TO_BACKENDS[kind]` priority list, else `flux_keyframe` fallback. Graceful degradation with warnings: pulid without character/refs → flux_keyframe; florence without mask/insert prompts → flux_keyframe. C4 enforced: `_clamp_duration` caps `ltx_motion` / `wan21_loop` at 10.0s; non-positive duration replaced with `DEFAULT_BEAT_DURATION_S=6.0`. Non-repetition sliding window (default 3 jobs, configurable via `nonrepeat_window`) rejects duplicate `(backend, prompt_hash)` tuples; `_nudge_prompt_for_uniqueness` appends ` [variant N]` suffix deterministically, max 32 nudge attempts before accept-and-warn. Handoff selection for motion/loop: reverse-iterates same-scene prior jobs, picks first still-producer (`flux_anchor` / `pulid_portrait` / `flux_keyframe` / `florence2_sdxl_comp`); warning + stub-mode routing if none. Scene rotation: if `sum(beats) < runtime`, re-enters scenes from top (safety cap at `len(scenes)*20` empty rotations). `plan_episode(outline, target_runtime_s=..., nonrepeat_window=..., default_beat_duration_s=...)` → `PlannerResult` with `jobs`, `total_duration_s`, `target_runtime_s`, `scenes_covered`, `warnings[]`, `repetition_window`. Outline coercion: dict passes through; `str` is JSON-fast-path when stripped starts with `{`/`[` (avoids `Path.exists()` "File name too long" on long JSON), else treated as path with `OSError`-guarded exists check, else raw JSON string. `emit_shotlist_json(result)` returns bridge-ready `{"shots":[...flat job dicts...], "target_runtime_s", "total_duration_s", "job_count", "warnings"}`. `write_shotlist(result, path)` writes JSON to disk. Pure stdlib — no torch, no diffusers — safe to import from tests and bridge. | ✅ 33/33 new planner tests green (module imports torch-free; public constants; backend assignment per kind incl. degrade paths; explicit override wins + unknown raises ValueError; C4 duration clamp for ltx+wan + non-clamp for stills + negative→default; non-repetition window 3 identical beats produce unique hashes after nudging + window=1 vs window=5 boundary behaviour + nudging determinism across runs; handoff selection picks prior still + warns when no upstream + scene boundary respected; runtime coverage respects target + repeats scenes when beats short + target override + empty outline warning; shotlist JSON schema with shots[] + job_count + target_runtime_s + per-shot shot_id/backend/prompt/duration_s/prompt_hash; write_shotlist to disk; coerce string JSON + Path; 3-min dry run gate ≥180s + ≥3 scene_ids + ≥4 backend diversity + window invariant; all emitted backends registered; PlannerJob.to_dict omits empty optional fields; PlannerResult.to_dict includes diagnostics); 314/314 combined regression across Days 1-9 (planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + dropdown guardrails + anchor_gen). C7 unchanged (planner is pure-stdlib, no audio path touched). Planner is not a backend — emits jobs that name Day 1-7 backends, does not register a new one. |
| 10 | **[DONE 2026-04-17]** `tests/test_cold_open_canary.py` — cold-open canary test drives a full Stage 1→7 pass in stub mode for SCENE 01 "Cockpit, Baba boots up the radio." Scene outline has 6 beats (b01 establishing→`flux_anchor`, b02 close_up→`pulid_portrait` with BABA character + refs, b03 keyframe→`flux_keyframe`, b04 motion→`ltx_motion` at 6.0s, b05 loop→`wan21_loop` at 10.0s, b06 insert→`florence2_sdxl_comp` with mask_prompt + insert_prompt) totalling ≥ 30s runtime. `_BACKEND_MATRIX` maps each backend to its stub envvar and expected per-shot outputs. Stubs all seven backends via `OTR_FLUX_STUB` / `OTR_PULID_STUB` / `OTR_FLUX_KEYFRAME_STUB` / `OTR_LTX_STUB` / `OTR_WAN_STUB` / `OTR_FLORENCE_STUB` / `OTR_VHS_STUB` so the canary runs CI-safe without GPU weights. VHS post-processor tested via `apply_vhs_to_job_dir(force_stub=True)` to sibling `*_vhs.mp4` files. Determinism test runs the full pass twice under the same tmp root (backends hash on absolute anchor path for layout-lock invariance, so the same absolute path must be reused between runs). | ✅ 15/15 new canary tests green (planner module torch-free; all 7 backends registered; scene_01 outline well-formed; planner covers runtime; planner emits every expected backend for scene_01; C4 honoured on motion + loop in scene_01; per-backend stub pass parametrized over 6 backends; VHS postproc over full canary emits summary + `*_vhs.mp4` siblings; no zero-byte outputs gate; determinism across two runs byte-identical); 276/276 combined video backend regression across Days 1-10 (cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode end-to-end render with GPU weights deferred to Day 11. |
| 11 | **[DONE 2026-04-17]** `visual/wall_clock.py` — per-backend wall-clock estimator (pure stdlib, torch-free). Point estimates per shot: `flux_anchor`=28s, `pulid_portrait`=32s, `flux_keyframe`=25s, `ltx_motion`=95s, `wan21_loop`=65s, `florence2_sdxl_comp`=18s (conservative upper bounds on RTX 5080 Laptop Blackwell sm_120 FP8 e4m3fn + SageAttention + SDPA path; catches regressions where FA chasing falls back to eager). Cold-load penalties charged once per distinct backend (`flux_anchor` 45s, `pulid` 50s, `keyframe` 40s, `ltx` 70s, `wan` 30s, `florence` 25s). VHS postproc charged at 5s per motion/loop clip (real) / 0.02s (stub). `WallClockEstimate` dataclass with `mode` / `total_s` / `render_s` / `cold_load_s` / `vhs_s` / `per_backend_s` / `per_backend_shots` / `unknown_backends` + `to_dict()`. `estimate(jobs, *, mode, include_vhs, include_cold_load)` accepts `PlannerJob` dataclass OR plain dict; mode=`real`/`stub`; cold-load auto-skipped in stub. `DAY_11_WALL_CLOCK_CEILING_S=2700` (45 min) and `DAY_11_STUB_CEILING_S=60.0` as ROADMAP bars. `tests/test_three_minute_continuous.py` — Day 11 ROADMAP gate. `_three_minute_cockpit_outline()` builds a 180s SCENE 01 with 8 beats spanning every backend kind (b01 establishing→`flux_anchor`, b02 close_up→`pulid_portrait` BABA+refs, b03 keyframe→`flux_keyframe`, b04 motion→`ltx_motion`, b05 loop→`wan21_loop`, b06 insert→`florence2_sdxl_comp` with mask+insert, b07 two_shot→`pulid_portrait` BOOEY, b08 ambient→`wan21_loop`) with scene rotation triggered by beats < target runtime. Stubs all 7 backends via `OTR_*_STUB=1` so the 3-min canary runs CI-safe without GPU weights. | ✅ 22/22 new wall_clock_estimator tests green (module torch-free import; all Day 1-7 backends covered in stub + real tables; cold-load table coverage; 45-min ceiling constant; accepts PlannerJob + dict + mixed iterable; stub << real cost invariant; render_s sum; cold-load charged once per distinct backend + scales with backend diversity + skipped in stub mode; VHS only charged for ltx_motion + wan21_loop + can be disabled; unknown backends recorded costing zero; empty jobs → zero total; invalid mode raises ValueError; to_dict schema; per-backend breakdown accumulates; representative 3-min mix [4 anchor + 3 pulid + 6 keyframe + 9 ltx + 6 wan + 2 florence = 30 jobs] fits under 45-min ceiling; stub 3-min scene fits well under 1-min ceiling); 10/10 new three_minute_continuous tests green (planner covers 180s runtime; ≥20 jobs to avoid stagnation; ≥4 distinct backends for diversity; non-repetition window invariant across full 3-min timeline; C4 duration clamp holds on motion + loop; projected real wall-clock ≤ 45 min; projected stub wall-clock ≤ 60s; stub end-to-end execution finishes in < 60s monotonic clock; no zero-byte outputs gate; emits `render.png` + `keyframe.png` + `motion.mp4` + `loop.mp4` + `composite.png`/`mask.png` mix); 308/308 combined video backend regression across Days 1-11 (three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode end-to-end 3-min render with GPU weights deferred (estimator is a conservative-upper-bound projection that catches catastrophic regressions, not a precise render-time predictor). |
| 12 | **[DONE 2026-04-17]** `visual/character_regression.py` — cross-scene character identity gate. Pure-stdlib SSIM computation for the Day 1-7 sidecar output tree. `_decode_stub_solid_rgb(png)` reverses the pulid stub PNG format (8-byte sig → IHDR → IDAT zlib decompress → first-pixel R,G,B triple) so the gate is unit-testable without Pillow/numpy. `ssim_solid(rgb_a, rgb_b, reduction={min,mean,product})` implements the Wang et al. SSIM formula simplified for solid-color images (σ=0 both sides, so SSIM reduces to the luminance term per channel); `min` reduction is default to punish any channel divergence. `compute_ssim(png_a, png_b, mode={auto,stub,real})` dispatches: auto tries stub decoder first + falls back to real SSIM on non-solid PNGs; real mode lazy-imports Pillow + numpy and raises a clear ImportError if missing. `SSIM_GATE = 0.85` constant strictly-greater-than semantics matches ROADMAP Day 12 bar. `find_portraits(out_dir, character)` walks `<out_dir>/<scene_id>/<shot_id>/{render.png,meta.json}` where `meta["backend"] == "pulid_portrait"` and `meta["character"]` matches, returns sorted `PortraitSample` list. `regress_character(out_dir, character, *, gate, mode)` computes pairwise SSIM across DISTINCT scene_ids only (within-scene pairs skipped — Day 12 bar is scene-1 vs scene-3, not shot-to-shot). Single-scene coverage → `gate_ok=True` with note (can't fail what isn't testable). `regress_cast(out_dir, cast)` aggregates per-character. `CharacterRegressionResult` dataclass with `character`, `gate`, `samples`, `pairs`, `min_ssim`, `mean_ssim`, `gate_ok`, `notes` + `to_dict()`. Torch-free + no audio imports (C7 preserved). | ✅ 26/26 new character_regression tests green (module torch-free import; SSIM_GATE == 0.85; ssim_solid identity → 1.0 + max divergence black-vs-white << 0.01; reduction modes agree on identity + differ on unbalanced divergence + unknown reduction raises ValueError + per-channel symmetry; stub decoder roundtrips known colors + minimum-channel floor + rejects non-PNG; compute_ssim auto + stub paths + auto detects divergence + invalid mode raises; find_portraits walks scene layout + ignores other characters + empty when missing; same refs across scenes locks identity [min_ssim == 1.0, gate_ok]; different refs break identity lock [min_ssim < gate, gate_ok == False]; full ROADMAP Day 12 BABA + BOOEY scene_01 vs scene_03 → both pass; within-scene pairs skipped; empty samples → gate_ok + note; single-scene → gate_ok + note "only one scene"; regress_cast aggregates; to_dict JSON-serialisable schema; real-mode SSIM raises ImportError with "Pillow" hint when PIL/numpy blocked); 334/334 combined video backend regression across Days 1-12 (character_regression + three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a). C7 unchanged. Real-mode Pillow+numpy SSIM on cropped-face regions path implemented but deferred to post-weights landing (stub path alone proves the gate's regression-detection behaviour; real-path Pillow+numpy is wired and will be exercised once PuLID-FLUX weights produce non-stub portraits). |
| 13 | **[DONE 2026-04-17]** `visual/lhm_monitor.py` — torch-free LibreHardwareMonitor sampler + summariser. Polls `http://localhost:8085/data.json` (env `OTR_LHM_URL`), walks the LHM JSON tree DFS for GPU temperature (hottest sensor under `GPU` path), VRAM used/total (GB or MB→GB normalised), system RAM used/total, and CPU package temperature. `LhmSample` dataclass with `t_monotonic`, `t_unix`, per-metric fields, `unreachable` + `reason` so network / parse failures land as countable samples instead of raised exceptions. `poll_once(url, timeout_s, fetcher, now_mono, now_unix)` — `fetcher` injectable for tests; urllib fallback wraps `URLError`/`TimeoutError`/`OSError`/`ValueError` into unreachable samples. `poll_loop(out_path, interval_s, duration_s, max_samples, stop_when, fetcher, sleep_fn, monotonic_fn, unix_fn)` streams NDJSON (one JSON line per sample) and returns the full list; clocks + sleep + stop_when all injectable so tests drive the loop deterministically. `LhmSummary` dataclass rolls up peak / mean / min / last per metric with three Day 13 ceiling-breach flags (`VRAM_CEILING_GB=14.5`, `RAM_CEILING_GB=28.0`, `GPU_TEMP_CEILING_C=85.0`); `summarize_ndjson(path)` loads a saved log and summarises. `scripts/lhm_poller.py` — CLI wrapper (`--out`, `--interval`, `--duration`, `--max-samples`, `--summary`, `--summarise-only`); writes `<stem>.summary.json` alongside the NDJSON; exits with code 2 when any ceiling is breached so Windows Task Scheduler flags the overnight run as failed automatically. Pure stdlib — no torch, no numpy, safe to import in the main venv. `tests/test_episode_dry_run.py` — Day 13 ROADMAP gate. `_twenty_minute_episode_outline()` builds a 1200-s six-scene outline (Cockpit + Corridor + Engine Room + Viewport + Galley + Airlock) with 30 beats spanning every Day 1-7 backend kind — scene rotation stress-tests the planner's rotate-from-top safety net over 20 minutes. Stubs all 7 backends via `OTR_*_STUB=1` so the dry run is CI-safe without GPU weights. Asserts: planner covers full 1200-s runtime; planner uses all six scenes; planner exercises every Day 1-7 backend at least once; ≥150 jobs emitted to avoid coalescing; non-repetition window invariant holds across 20 min; C4 10-s cap on motion + loop; projected real wall-clock fits under 8-hour overnight ceiling with cold loads + VHS; stub execution finishes under 120-s CI floor; no zero-byte outputs across 30-job run; every STATUS.json ends in `READY` (no `OOM` / `ERROR` / `RUNNING`); artifact mix gate (`render.png` + `keyframe.png` + `motion.mp4` + `loop.mp4`); LHM poller with injected nominal fake telemetry tree captures 18-22 samples across a simulated 20-min run at 60 s interval; summary shows no ceiling breach on nominal hardware values; inverse gate trips `vram_ceiling_breached=True` when tree reports >14.5 GB VRAM. Fixed `poll_loop` to check `stop_when` once per iteration at the top only (was double-checking before + after poll, making sample-count semantics non-deterministic). | ✅ 20/20 new lhm_monitor tests green (module torch-free import; Day 13 ceiling constants; poll_once extracts 4 metrics from fake LHM tree; poll_once records unreachable on network + parse errors; MB→GB normalisation; to_dict JSON-serialisable; poll_loop NDJSON + sample list with `max_samples=5`; poll_loop duration enforcement via state-advancing sleep_fn; poll_loop `stop_when` trips on 3rd call → 2 samples; non-positive interval raises ValueError; summarize empty note; peak/mean/min/last stats; VRAM + RAM + GPU temp ceiling breach flags; unreachable count; summarize_ndjson roundtrip + missing-file note; summary to_dict JSON-serialisable with ceiling constants). 15/15 new episode_dry_run tests green (full 20-min runtime + all six scenes used + every Day 1-7 backend hit + ≥150 jobs + non-repetition window + C4 clamp + real wall-clock ≤ 8 h + stub ≤ 120 s + stub execution < 120 s + no zero-byte outputs + all STATUS.json `READY` + artifact mix + LHM sampler 18-22 samples + no nominal breach + VRAM breach on thrash tree). 554/554 combined v2 Day 1-13 regression (lhm_monitor + episode_dry_run + character_regression + three_minute_continuous + wall_clock_estimator + cold_open_canary + planner + vhs_postproc + florence2_sdxl_comp + wan21_loop + ltx_motion + flux_keyframe + pulid + flux_anchor + backend dispatch + visual_phase_a + anchor_gen + arc_check + camera_path + dropdown_guardrails + obsidian_profile + p0_features + treatment_scanner + widget_drift + v2 audio byte-identical [skipped without GPU]). C7 unchanged. Pre-existing `BUG-LOCAL-042` (`vram_sentinel` ImportError cascading through `tests/test_core.py`) marked `[FIXED]` on 2026-04-17 — stale Windows `__pycache__` from mid-April Phase B churn self-resolved; `tests/test_core.py` now 103/103 green both warm and after full pycache purge. | No OOM, no pagefile thrash, no shared-memory fallback. |
| 14 | **[DONE 2026-04-17]** Stack frozen on `v2.0-alpha` at commit `2430064` (Day 13 ship). All Day 1-13 backends + harness + planner + wall-clock estimator + character regression gate + LHM telemetry poller + 20-min episode dry-run gate shipped and locked. Tag handoff to Jeffrey via `scripts/tag_v2.0-alpha-video-full.cmd` (per CLAUDE.md: only Jeffrey tags releases); script verifies branch + clean tree + lockstep with origin + creates annotated tag `v2.0-alpha-video-full` + pushes, failing fast on any mismatch. BUG_LOG already carries pre-existing `BUG-LOCAL-042` (`vram_sentinel` ImportError in `nodes/batch_bark_generator.py` import chain) as the only open regression-noise — not caused by the sprint, last touched `5cf338e` before Day 1. MEMORY refreshed with sprint-complete snapshot: `project_v20_alpha_video_stack_complete.md` notes canonical branch + tag + 14 rows of gates + deferred real-mode weight gates. All three test suites pass: 334/334 combined v2 Day 1-12 regression on Day 12, 554/554 combined Day 1-13 regression on Day 13; `tests/test_dropdown_guardrails.py` 56/56; `tests/v2/test_audio_byte_identical.py` 7/7+1 skipped without GPU (C7 unchanged across all 14 days). Zero `video-stack` blockers in BUG_LOG. | ✅ Stack feature-complete. Jeffrey runs the tag script at his convenience to cut `v2.0-alpha-video-full`. Real-mode weight-landing gates (FLUX ≤ 12.5 GB 1024², PuLID face-embedding SSIM, Union Pro 2.0 ≤ 13.5 GB, LTX-2.3 + Wan2.1 ≤ 10-14.5 GB, Florence-2 mask quality) cleanly deferred to the post-sprint weight-landing pass as designed. |

### Kill criteria (fail-fast; do not hero-fix)

| Stage | Kill trigger | Fallback |
|---|---|---|
| 1 Anchors | FLUX FP8 peak > 14 GB on 1024² OR `flash_attn` import detected | SDXL 1.0 + 1980s VHS LoRA |
| 2 Keyframes | Union Pro 2.0 fails on diffusers + torch 2.10 | FLUX + single Depth CN |
| 3 Portraits | PuLID no identity lock after 10 attempts × 3 ref packs | InstantCharacter |
| 4 Motion | LTX-2.3 quality regression vs current baseline | HunyuanVideo via Nunchaku INT4 |
| 5 Loops | Wan2.1 peak > 12 GB OR visible VAE temporal drift | FramePack |
| 6 Comp | SDXL inpaint seams > 5 px | Insert Anything |
| 7 Mux | Any C7 regression | Revert, hold day's work |

**Overarching kill rule:** audio degrades in any way → revert immediately. Audio is king.

### Video-stack risks

1. **VRAM fragmentation on Windows spawn.** PyTorch doesn't always release VRAM to OS. Mitigation: Bridge cooldown gate — `libre_tail.snapshot()` must show GPU free ≥ 2 GB before spawn, else 3-s wait + `WORKER_VRAM_BLOCKED` fail-fast; 2-s sleep + `torch.cuda.empty_cache()` post-exit.
2. **FP8 scaling bugs on sm_120 without FA3.** Mitigation: pre-pin GGUF Q8 variant as instant fallback; do not chase FA3.
3. **I2V temporal drift on chained gens.** Mitigation: planner NEVER chains video-to-video; always regenerates motion from a pristine FLUX still.
4. **diffusers + torch 2.10 + FLUX FP8 incompatibility.** Mitigation: pin exact diffusers version Day 2 in `requirements.video.txt`.
5. **Bark/audio interference from co-running video sidecars.** Mitigation: daily C7 gate, separate process trees.
6. **Windows PIPE backpressure deadlock** (already flagged in `bridge.py:296-299`). Mitigation: stderr → tempfile, never `stderr=PIPE` undrained.

### Sanity pass findings (2026-04-17)

1. PuLID upstream uncertain — Day 3 `WebFetch` verification before clone.
2. diffusers version must be pinned Day 2 in `requirements.video.txt`.
3. C7 audio regression runs at end of EVERY day, not only Day 10.
4. Bridge cooldown gate is non-negotiable (LHM free ≥ 2 GB).
5. Both consultants had errors — trust the verified repos above, not either consult raw output.
6. No `ComfyUI-*-Wrapper` as primary runtime (pulls flash_attn, wraps add overhead) — diffusers native or raw model code only.
7. Hotfixes on `v2.0-alpha` during sprint → rebase `v2.0-alpha` daily.

### Definition of Done (Day 14)

- `v2.0-alpha-video-full` tagged on origin.
- 20-min episode renders end-to-end with no manual steps.
- C7 audio byte-identical to v1.5 baseline.
- No `flash_attn` imports in venv trace.
- No `CheckpointLoaderSimple` in any live workflow (C2).
- All visual generation in subprocesses (C3).
- Character identity SSIM > 0.85 on face crops between Scene 1 and Scene 3.
- BUG_LOG has zero open `video-stack` blockers.
- ROADMAP.md updated; items 3/4/6/8 unblocked into P2.
- MEMORY.md gets a project memory summarizing the shipped stack.

### New backend module layout

```
visual/backends/
  _base.py               # write_status(), STATUS.json schema, cooldown helper
  placeholder_test.py    # Day 1 spawn/cleanup canary
  flux_anchor.py         # Stage 1 — diffusers FP8 FLUX + Shakker-Labs ControlNet
  flux_keyframe.py       # Stage 2 — FLUX + Depth/Canny
  pulid_portrait.py      # Stage 3 — FLUX + PuLID identity insertion
  ltx_motion.py          # Stage 4 — wraps existing LTX sidecar under uniform STATUS contract
  wan21_loop.py          # Stage 5 — Wan2.1 1.3B I2V
  florence2_sdxl_comp.py # Stage 6 — Florence-2 mask + SDXL inpaint
```

Bridge contract additions: `backend=<name>` arg; pre-spawn LHM cooldown gate; post-exit `empty_cache()` + 2 s sleep; STATUS.json adds `peak_vram_gb` field for learned ceilings.

---


---

## Session handoff &mdash; 2026-04-23 (hybrid refactor shipped, video-continuity decision pending)

Branch: `v2.0-alpha`. Do NOT touch `main`.

### What shipped tonight

- **Pivot from sidecar to ComfyUI-native FLUX.** After ~4 days of Diffusers+torchao+accelerate sidecar failures (BUG-LOCAL-046..057), ported FLUX generation into the main ComfyUI graph via `CheckpointLoaderSimple`. Renders FLUX.1-dev-fp8 clean in ~44 s on the 5080 at 11.35 GB peak. Validated in memory `feedback_comfyui_native_over_diffusers_torchao.md`.
- **Three new OTR custom nodes** (registered in `__init__.py`; 23 &rarr; 25 loaded nodes on startup):
  - `OTR_CheckpointLoaderGated` &mdash; wraps stock `CheckpointLoaderSimple` with a `trigger` forceInput + pre-load `llm_polish.unload()`. Forces FLUX load to wait until Mistral polish is done; releases Mistral-Nemo NF4 before FLUX occupies VRAM.
  - `OTR_VisualExtractFluxPrompt` &mdash; pure string adapter between PromptCoercion's JSON-stringified token list and CLIPTextEncode's STRING input. Zero torch / zero VRAM.
  - `OTR_UnloadAll` &mdash; IMAGE passthrough between VAEDecode and SaveImage that calls `comfy.model_management.unload_all_models()` + `soft_empty_cache(force=True)` + `llm_polish.unload()`. Fixes the 44% post-render VRAM retention Jeffrey flagged.
  - `OTR_BatchFluxRender` &mdash; lockstep multi-prompt renderer. Parses `cleaned_script_json`, pulls N env tokens, renders them under one MODEL/CLIP/VAE load. Two modes via the `fast_batch` BOOLEAN widget (default True): stacks N CONDITIONING tensors into one batched call + single KSampler for the whole batch (18% faster per shot); falls through to serial loop on any stacking error. Pre-pins MODEL via `load_models_gpu` so per-sample `load_models_gpu` calls in KSampler become no-ops.
- **Option A workflow wiring** in `workflows/otr_scifi_16gb_TEST.json`: deleted 6 single-shot sampler nodes (CLIPTextPos + CLIPTextNeg + EmptySD3Latent + FluxGuidance + KSampler + VAEDecode), replaced with `OTR_BatchFluxRender` + `OTR_UnloadAll` + `SaveImage`. Node count dropped 14 &rarr; 9. `batch_limit = 4` preset, changeable on the fly via widget.
- **First successful 4-shot lockstep run** 2026-04-23 at 14:50: 4 unique FLUX images (starship bridge, science officer, wide captain, climactic reveal) in **131.60 s total, 108 s sampling**, one KSampler progress bar, clean UnloadAll trio at the end.
- **BUG_LOG additions** &mdash; BUG-LOCAL-058 (OneDrive sync race truncates JSON tail during Edit roundtrip, environmental) and BUG-LOCAL-059 (ComfyUI auto-injects a hidden `control_after_generate` widget after any INT field named `seed` &mdash; positional `widgets_values` array must account for it, Bible candidate).
- **Two live Cowork artifacts**: `otr-test-workflow-walkthrough` (full node graph + live log tail + per-node AI explain + pass/fail tracking) and `otr-batch-render-progress` (simplified 4-shot visual grid + batch-progress bar + demo button).

### Pipeline state snapshot (v2.0-alpha HEAD: `df578db`)

```
Mistral-Nemo NF4 polish (6 GB VRAM, 11s)
    |  cleaned_script_json (N env tokens)
    v
OTR_CheckpointLoaderGated (unload Mistral, load FLUX fp8, 11.3 GB peak)
    |  MODEL + CLIP + VAE
    v
OTR_BatchFluxRender fast_batch (N images in one KSampler call, 108s for N=4)
    |  IMAGE batch [N, H, W, C]
    v
OTR_UnloadAll (evict MODEL/CLIP/VAE + empty_cache, VRAM -> baseline)
    |  IMAGE
    v
SaveImage -> N PNGs in ComfyUI/output/
```

Wall clock for a 4-shot 1024x1024 run: **131.60 s**, peak VRAM **11.35 GB**.

### Open video-continuity decision (for the next conversation)

Reference: `uploads/OTR_v2-alpha_VIDEO_CONSISTENCY_BUILD_DECISION.md` (full v3 brief, 199 lines). Three options evaluated:

| Option | Engine | Identity lock | Multi-clip continuity | VRAM | Risk |
|---|---|---|---|---|---|
| A (recommended in the brief) | Wan 2.2 I2V A14B fp8 + SVI 2.0 Pro LoRA | Single reference image anchors across clips | Built-in latent-level continuity via motion-frame overlap | ~12-14 GB | Blackwell sm_120 fp8 kernel path is the unknown |
| B | Wan 2.2 Animate single-shot I2V | None built-in | Manual last-frame-to-first-frame chain | ~12 GB | Character drift between clips expected |
| C | LTX-Video 2.3 + IPAdapter FaceID + ControlNet/Pose | FaceID conditioning on every frame | Per-clip only | <10 GB | No native inter-clip continuity; manual stitching |

**Existing workflows surveyed (all free + local):**
1. `vita-epfl/Stable-Video-Infinity` &mdash; paper authors' own ComfyUI graph
2. `Well-Made/ComfyUI-Wan-SVI2Pro-FLF` &mdash; pre-wired 7-clip FLF + SVI Pro, closest match to OTR scene-graph
3. Kijai's native Wan 2.2 SVI Pro JSON &mdash; fallback if FLF repo doesn't cooperate with Blackwell
4. DaSiWa Wan 2.2 pack on Civitai &mdash; reference buffet
5. Wan 2.2 FLF2V native ComfyUI template &mdash; lighter-weight chain option

**The Goofer discovery (2026-04-23 late):** Jeffrey's own `jbrick2070/ComfyUI-Goofer` repo ships a working **LTX-Video 0.9.5** batch-video + video-concat + music-gen pipeline already validated on the 5080 (RTX 5080 / 4090 is the listed recommended GPU). The `GooferBatchVideo` + `GooferVideoConcat` + `GooferPromptGen` node pattern maps 1:1 onto what OTR-video needs. Saved as reference memory `reference_goofer_pipeline.md`. **Likely cheapest path to video:** port the Goofer node pattern into OTR as `OTR_BatchLtxVideo` + `OTR_VideoConcat`, feed FLUX batch stills as LTX starting frames for I2V. That's Option C in the brief, except the pattern's already written.

**Strategic tension to resolve next session:**
- Character continuity is NOT yet solved in either the Goofer path or the SVI-Pro path. Goofer doesn't care (each clip is its own scene); SVI solves inter-clip drift but not "is this the same character." Before committing to a video engine, **test character lock in stills first** &mdash; wire PuLID (already in repo at `visual/backends/pulid_portrait.py`) into BatchFluxRender so all N shots share one character identity. If PuLID works on Blackwell fp8, identity-locked FLUX anchors feed either video path cleanly.
- Video-vs-stills repositioning question: OTR is a radio drama with visual companion. Moving to SVI Pro V2 repositions the project around video-as-main-artifact. C7 "Audio is king" stays invariant regardless.

### Open items carried forward

| Item | Why | Who |
|---|---|---|
| Wire PuLID into BatchFluxRender for character continuity in stills | Answers "can OTR make a coherent episode" before video engine choice | Next session |
| Port `goofer_batch_video.py` + `goofer_video_concat.py` patterns into OTR as `OTR_BatchLtxVideo` + `OTR_VideoConcat` | Reuses working LTX 0.9.5 batch-video + stitch pattern Jeffrey already shipped in Goofer | Next session |
| Evaluate `Well-Made/ComfyUI-Wan-SVI2Pro-FLF` 7-clip workflow on Blackwell fp8 | Validate Option A is actually reachable on sm_120 before committing to it | Next session |
| Add `tests/test_widget_value_alignment.py` regression | Cross-check every workflow JSON's `widgets_values` against node `INPUT_TYPES()` + auto-injected hidden widgets (BUG-LOCAL-059 class of bugs) | When stills work lands |
| Fix `[PromptCoercion] cleaned 49 tokens (env=0 dlg=0 sfx=0)` counter | Reports zero env tokens when 4 are clearly produced; cosmetic, not blocking | Low priority |
| Flip `OTR_FLUX_ALL_SHOTS=1` default | Get Mistral polish on all N env prompts (currently only shot 1 is polished; shots 2..N pass through raw) | When character continuity is wired |
| Promote BUG-LOCAL-059 to Bug Bible | Bible candidate, needs one more clean live-run verification first | After widget-alignment test lands |
| Save BUG-LOCAL-058 (OneDrive JSON truncation) monitor note | Environmental; recurs if ComfyUI repo stays under OneDrive-synced Documents. Options: pause sync / move repo / use Desktop Commander `edit_block` for JSON writes | Already logged |

### Handoff prompt &mdash; drop into a new Claude conversation

> I'm continuing OTR v2-alpha on branch `v2.0-alpha` (do NOT touch main). Tonight we shipped the hybrid FLUX refactor + 4-shot batched rendering and now need to decide the video-continuity path. Read `ROADMAP.md` "Session handoff &mdash; 2026-04-23" section first, then `uploads/OTR_v2-alpha_VIDEO_CONSISTENCY_BUILD_DECISION.md` for the full options brief.
>
> First move: wire PuLID (already shipped at `visual/backends/pulid_portrait.py`) into `OTR_BatchFluxRender` so all N shots lock to one character reference. This is the prerequisite test before committing to Option A (SVI Pro / Wan 2.2) or the Goofer-pattern LTX 0.9.5 path. If character lock works on Blackwell fp8 in stills, identity-anchored FLUX stills feed either video engine cleanly.
>
> Context:
> - 16 GB VRAM hard ceiling (video stack lifted to 15.5 GB during sprint).
> - Audio is king. C7 byte-identical audio gate unchanged.
> - `OTR_BatchFluxRender.fast_batch=True` is the production path. Serial loop is fallback.
> - The `seed` widget auto-injects a hidden `control_after_generate` slot &mdash; any JSON edit to that node must include `"randomize"` at `widgets_values[3]` (BUG-LOCAL-059).
> - Repo lives under OneDrive-synced Documents. Prefer Desktop Commander `edit_block` + `copy /Y` over multi-step sandbox Edits for JSON writes (BUG-LOCAL-058).
> - `ComfyUI-Goofer` (Jeffrey's own repo) has a working `GooferBatchVideo` + `GooferVideoConcat` + `GooferPromptGen` pipeline on LTX-Video 0.9.5 2B &mdash; copy the pattern when porting OTR-video, don't reinvent.
>
> Build order:
> 1. `OTR_CharacterAnchor` node or widget on BatchFluxRender that accepts one reference image + writes PuLID-conditioning into the batched CONDITIONING. Smoke-test identity consistency across 4 FLUX shots at 1024x1024 on Blackwell fp8.
> 2. If (1) works: port Goofer's LTX 0.9.5 batch-video + video-concat patterns as `OTR_BatchLtxVideo` + `OTR_VideoConcat`, feeding BatchFluxRender's identity-locked IMAGE batch as I2V starting frames. Validate on Blackwell before committing.
> 3. If (1) fails on Blackwell: fall back to InstantCharacter (`Tencent-Hunyuan/InstantCharacter`) as Stage 3 per the ROADMAP fallback list.
> 4. Evaluate `Well-Made/ComfyUI-Wan-SVI2Pro-FLF` 7-clip workflow only AFTER still-image identity lock is proven; SVI Pro is meaningful only if FLUX anchors are already character-consistent.
>
> Constraints:
> - Use `placeholder` / `stub` / descriptive name &mdash; never `dummy` &mdash; in code and comments.
> - UTF-8 no BOM always.
> - Run three regression suites after every code change: `tests/test_core.py`, `tests/test_dropdown_guardrails.py`, `tests/test_workflow_json_guardrails.py`. Bug Bible regression if the sister repo is mounted.
> - 16 GB VRAM hard ceiling. LHM @ `http://localhost:8085/data.json` is always on.


## Session handoff &mdash; 2026-04-23b (MIT video-consistency pivot)

Branch: `v2.0-alpha`. Do NOT touch `main`.

### Decisions locked this session

1. **OTR stays MIT licensed.** Considered and rejected a shift to GPL-3.0 (one-way door; preserves future optionality). Never vendor GPL-3 packs (`Kosinkadink/ComfyUI-VideoHelperSuite`, `Well-Made/ComfyUI-Wan-SVI2Pro-FLF`) into the OTR tree &mdash; reimplement under MIT or use native ComfyUI templates (Apache via core) instead.
2. **Don't vendor community packs as `OTR_*` wrappers.** Past OTR custom video nodes have bottlenecked. Where a pack is MIT/Apache-compatible and the license permits, write OTR-native MIT code instead; prefer native ComfyUI core nodes for anything they can do. Community-pack vendoring only when (a) license is MIT/Apache, (b) no native equivalent, (c) OTR doesn't need pipeline-specific coordination.
3. **Character consistency without faces.** Jeffrey rejected face-identity anchors (PuLID). Path forward is style/environment anchoring via IP-Adapter (per C6 &mdash; environments only, never characters) using XLabs FLUX IP-Adapter weights (Apache 2.0, MIT-compatible reimplementation path).
4. **First-Last-Frame chained video** is the consistency mechanism for audio-length coverage. N+1 FLUX stills produce N video clips where each clip's last frame equals the next clip's first frame by construction &mdash; concat is seamless without crossfades.

### Shipped this session

- **`nodes/otr_video_concat.py`** (new, ~280 lines, MIT) &mdash; ffmpeg-based seamless clip concatenation. Pure subprocess wrapper (no Python pixel processing); `-c copy` fast path with auto-fallback to re-encode on codec mismatch; stub mode (`OTR_VIDEO_CONCAT_STUB=1` / ffmpeg-missing / `force_stub=True`); C7 audio passthrough via `-c:a copy -map 0:a?`. Replaces the need to vendor `VideoHelperSuite`.
- **`tests/test_otr_video_concat.py`** (new, ~320 lines) &mdash; 24 unit tests covering path parsing, filelist writer, argv builder, stub mode, node surface area. Torch-free, ffmpeg-free.
- **`__init__.py`** &mdash; OTR_VideoConcat registered. Loaded node count: 25 &rarr; 26.
- **`.gitignore`** &mdash; `nodes/vendor/` and `/tmp/otr_vendor_stage/` blacklisted so failed clone fragments can never enter git (OneDrive held sandbox locks during the MIT-pivot cleanup).
- **`docs/2026-04-23-MIT-video-consistency-plan.md`** (new) &mdash; full implementation spec for OTR_FluxIpAdapter (phase 2) and OTR_WanFlfVideo (phase 3). Architecture notes, kill criteria, test lists, path A vs B analysis for the FLF engine.
- **Memory**: `feedback_otr_stays_mit.md` saved (never vendor GPL into MIT); `feedback_use_community_nodes_not_custom.md` saved (community over OTR wrappers).

### Pending handoff (Jeffrey, on return)

Full verification + commit script queued at `scripts/_claude_handoff_2026-04-23b.ps1`. Runs: OneDrive cleanup of `nodes/vendor/` fragments, AST parse of both new files, regression suites, test run on new tests, local commit. Jeffrey reviews + pushes.

### Phase 2 &mdash; OTR_FluxIpAdapter (next session)

See `docs/2026-04-23-MIT-video-consistency-plan.md` for the full spec. Estimate 2-3 sessions. Read `XLabs-AI/x-flux-comfyui` (Apache 2.0) staged at `/tmp/otr_vendor_stage/x-flux-comfyui/` during this session; MIT reimplementation via ComfyUI `ModelPatcher.set_model_attn2_patch()`; stub mode first; Blackwell fp8 kernel check before claiming done.

### Phase 3 &mdash; OTR_WanFlfVideo (next-next session)

Two candidate paths. Path A: native ComfyUI Wan 2.2 FLF2V template (Apache) wired in via a thin `OTR_WanFlfShotList` helper, ~60 lines. Path B: OTR-native Wan 2.2 I2V sidecar, adapted from existing `visual/backends/wan21_loop.py` with `last_image` conditioning added. 30-min Blackwell fp8 smoke test decides A vs B.

### Environmental issues to watch

- **BUG-LOCAL-058 (OneDrive sync race)** hit hard this session. Sandbox `rm` on fresh git clones failed with "Operation not permitted" on `.git/` internals; sandbox-mount view of `__init__.py` stayed stale for minutes after the Write tool updated the Windows file. Mitigation used: file tool Writes for deliverables, bash heredoc for any sandbox-verified writes, all verification deferred to Windows-side script. Consider moving the repo off OneDrive-synced `Documents\` to clear this class of bug permanently.

---

## Session handoff &mdash; 2026-04-23c (video branch shipped; FLUX.2 + HuMo rollout planned)

Branch: `v2.0-alpha`. 14 commits tonight. Full audio+video pipeline runs end-to-end on 5080 16GB.

### Shipped this session (cumulative, commits `3cee09d`..`ce15063`)

- `OTR_VideoConcat` &mdash; MIT ffmpeg concat node, 28 unit tests + real smoke passed. Replaces need for VideoHelperSuite (GPL-3).
- `OTR_VideoPlan` &mdash; read-only Director/script adapter, 3-pass outputs (pass1 char portraits, pass2 scene envs, pass3 composite shots). Multi-character mode default. `audio_gate` optional STRING input for execution-order sequencing. 52 unit tests.
- `OTR_ShotDurationCalculator` &mdash; expands 1-clip-per-shot to N-clips-per-shot from shot durations. FLF shared-boundary invariant preserved. `clips_per_shot(dur) = 1 if dur<=10 else ceil(dur/9)`. 25 unit tests.
- `workflows/otr_scifi_16gb_full.json` &mdash; full audio pipeline + bolt-on video branch wired via `audio_gate`. Execution order guaranteed: Director → audio → POC video → ffmpeg mux → VideoPlan → Calculator → FLUX → UnloadAll → SaveImage. **No separate `_with_video` variant** (consolidated per minimum-JSON discipline).
- `workflows/otr_videoplan_TEST.json` &mdash; standalone 10-frame video-branch test. Validated end-to-end on 5080, 10 PNGs on disk at `output/otr_videoplan_pass3_000{03..12}.png`.
- `tests/fixtures/sample_director_lemmy.json` &mdash; realistic 3-scene Director JSON with LEMMY + KENJI CROSS for manual testing without running the full ScriptWriter+Director chain.
- **258 tests green** (cumulative: 52 plan + 28 concat + 25 calculator + 50 dropdown + 103 core).

### Architectural direction &mdash; SUPERSEDED and new (FLUX.2 + HuMo)

**Previously:** FLUX.1-dev + Wan 2.2 + VACE (First-Last-Frame) + Lightning LoRA &mdash; locked earlier on 2026-04-23.

**Now (as of 2026-04-23c):** FLUX.2-klein + HuMo 17B for audio-driven character animation. Research confirmed both fit on 5080 16GB Blackwell via GGUF quantization. HuMo is a direct replacement for Wan 2.2 in the video position, with the critical difference that HuMo is **audio-driven** &mdash; characters visibly speak their Bark-rendered dialogue with real lip-sync. Aligns perfectly with OTR's "audio is king" principle.

Wan 2.2 + VACE plan retained as fallback if HuMo fails to fit (unlikely given the quantized variants available).

### FLUX.2-klein + HuMo 4-stage rollout plan (next sessions)

| Stage | Scope | Est. time | Success signal |
|---|---|---|---|
| **1. FLUX.2-klein in TEST** | Download FLUX.2-klein Q5_K_M GGUF (~10-13 GB). Swap `CheckpointLoaderSimple` widget in `otr_videoplan_TEST.json`. Re-queue. Compare new 10 PASS 3 PNGs against the FLUX.1 baseline (on disk from tonight). | 30-60 min | Better multi-character composites (LEMMY + KENJI + ANNOUNCER in one frame). Rollback = one widget change. |
| **2. Add HuMo to TEST with pre-baked audio** | Grab 3-4 per-line Bark WAVs from tonight's full-run output. Add `LoadAudio` node to TEST. Download HuMo 17B GGUF Q6 from `calcuis/humo-gguf`. Wire: FLUX.2-klein portrait + WAV + prompt → HuMo node. | 2-3 hr first time | One ~3.9s clip where character's mouth moves with the audio. |
| **3. "Creative ffmpeg" proof-of-life .mp4** | HuMo emits IMAGE batch. Mux with audio via either VHS_VideoCombine (install as runtime-only, GPL-3 not vendored) or extend OTR_VideoConcat to take IMAGE batch + audio. | 30 min | One `.mp4` where a character speaks a real OTR line. Concrete deliverable from TEST. |
| **4. Full integration** | Swap FLUX.2-klein into `otr_scifi_16gb_full.json` CheckpointLoaderSimple. Insert HuMo nodes between Calculator and VideoConcat. Wire real per-shot Bark WAVs into HuMo's audio input (this is the audio-timeline wiring we've been deferring). Flip `SEGMENT_TARGET_DURATION_S` 9.0 → 3.5 and `SEGMENT_HARD_CAP_S` 10.0 → 4.0 in `otr_shot_duration_calculator.py` (matches HuMo's 97-frames-at-25fps = ~3.9s cap). | 3-5 hr | Full episode rendered with characters lip-synced to their Bark dialogue. |

Full plan details preserved in memory `project_flux2_humo_rollout_2026-04-23.md`. Do NOT start Stage 2 until Stage 1 is green.

### VRAM stack &mdash; projected production pipeline (fits on 5080 16 GB)

| Stage | Model | Est. peak | Native Blackwell? |
|---|---|---|---|
| LLM Script + Director | Mistral-Nemo NF4 | ~6 GB | yes |
| Dialogue TTS | Bark bf16 | ~4 GB | yes |
| Music | MusicGen medium | ~5 GB | yes |
| Announcer TTS | Kokoro (local) | ~1 GB | yes |
| PASS 1/2/3 image | FLUX.2-klein Q5_K_M GGUF | ~10-13 GB | yes (FP8 tensor cores) |
| Video (audio-driven) | HuMo 17B GGUF Q6 | ~10-15 GB | yes (FP8 tensor cores) |
| Mux | ffmpeg | 0 GB (CPU) | n/a |

All 100% local, no cloud, no API keys. Full 2026-era audio-drama-with-video pipeline on consumer hardware.

### Constants that flip in Stage 4 (the whole diff)

```python
# nodes/otr_shot_duration_calculator.py
SEGMENT_TARGET_DURATION_S = 9.0   # -> 3.5 for HuMo
SEGMENT_HARD_CAP_S        = 10.0  # -> 4.0 for HuMo
SEGMENT_TARGET_FPS        = 16    # -> 25 for HuMo
SEGMENT_MAX_FRAMES        = 161   # -> 97 for HuMo (97 @ 25fps = 3.88s)
```

Math is unchanged. Only the constants move.

### Honest gaps still open after Stage 4

- **Multi-character in one shot when both speak.** HuMo is designed for one speaking character at a time. For two-shot dialogue, need either (a) composite non-speaker into background + animate speaker, or (b) run HuMo twice per shot and composite. Deferred until after single-character quality is proven.
- **Scene-geometry consistency across episodes** (Scene-Geometry-Vault from P2). Still deferred, same as before.
- **Still no IP-Adapter / Kontext for character identity lock.** Text-composite remains the floor; upgrading to image-reference PASS 3 compose (FLUX.2-Kontext klein) is a Stage 5 item.

### Parallel consideration &mdash; TTS upgrade path (Bark &rarr; Fish Speech / CosyVoice)

Independent of the FLUX.2 + HuMo rollout. Bark is shipping and stable, but both Fish Speech and CosyVoice are legitimate 2026-era upgrades worth evaluating once the video stack is green. Do NOT start this before Stage 4 is done &mdash; audio is king, and replacing the TTS backbone while video is still in flux would break our baseline reference.

**Candidates (all fit on 5080 16 GB):**

| Engine | Est. VRAM | License | Strengths | Weaknesses |
|---|---|---|---|---|
| **Bark** (current) | 8-12 GB | MIT | Natural laughter / sighs / non-verbal; great character colour; proven in OTR pipeline | Slower per-token; no per-speaker cloning without retraining; English-centric |
| **Fish Speech S2-Pro** | 16 GB (BNB NF4 4-bit); 24 GB+ full | Non-commercial research license &mdash; verify before use | Best-in-class audio quality among 2026 open TTS; clean zero-shot voice cloning; has ComfyUI node (`Saganaki22/ComfyUI-FishAudioS2`) | License profile is the blocker &mdash; OTR is MIT and we do not vendor restrictive code. Worth evaluating for personal use only unless upstream relaxes. |
| **CosyVoice 2.0** | ~6-8 GB | Apache-2.0 | Ultra-low latency (~150 ms first-chunk); pronunciation error rate 30-50% lower than v1; multilingual; streaming; Apache-2.0 is safe to vendor | No built-in non-verbal sounds as expressive as Bark; would lose some of the 1940s-radio character colour we rely on |
| **CosyVoice 3.0** | ~8-10 GB | Apache-2.0 | Quality improvements over 2.0; same license profile | Still maturing; double-check upstream stability before committing |
| **Qwen3-TTS** | ~6 GB | Apache-2.0 | Apache-2.0; Alibaba quality; strong multilingual | Newer &mdash; less community tooling than CosyVoice; ComfyUI node coverage thin as of 2026-04-23 |

**Criteria to evaluate when we pick this up:**

1. **License compatibility first.** OTR stays MIT. Fish Speech's non-commercial research license means we can listen and evaluate on Jeffrey's machine, but cannot ship it vendored in the repo. CosyVoice 2/3 and Qwen3-TTS are Apache-2.0 &mdash; safe to use and recommend.
2. **Does it keep the "1940s radio voice" character?** Bark's strength is non-verbal expressivity (sighs, laughter, uh-huhs). A pure-quality win on neutral speech is a net loss if the period-drama colour drains out of the announcer / character reads.
3. **Does it fit into `batch_bark_generator.py` without rewriting the orchestrator?** The sequencer (length-sorted batching, VRAM-Sentinel decorator, per-call snapshots) is proven; an ideal swap is a drop-in TTS backend behind the same interface.
4. **Per-character voice consistency across episodes.** Currently Bark uses preset matching. CosyVoice's zero-shot cloning from a 3-10s reference could actually improve cross-episode consistency (feed the same reference WAV every time).
5. **Streaming vs batch.** CosyVoice 2's streaming 150 ms could let us interleave TTS with FLUX &mdash; probably overkill for OTR's batch pipeline, but worth noting.

**Recommended first look (when we get here):** CosyVoice 2.0 Apache-2.0, sideload a ComfyUI node, A/B against Bark on the LEMMY + KENJI CROSS + ANNOUNCER test fixture. No commitment, no rip-and-replace until we hear it in context.

Deferred. Captured here so it doesn't drop on the floor.

### Quick-start for next session

1. Read `memory/project_flux2_humo_rollout_2026-04-23.md`
2. Download FLUX.2-klein Q5_K_M GGUF to `C:\Users\jeffr\Documents\ComfyUI\models\checkpoints\`
3. Load `workflows/otr_videoplan_TEST.json`, swap checkpoint widget, queue
4. Compare outputs against `C:\Users\jeffr\Documents\ComfyUI\output\otr_videoplan_pass3_00003..12.png` (FLUX.1 baseline from this session)

---

## Session handoff &mdash; 2026-04-23d (FULL workflow cleanup + reference fixture)

Branch: `v2.0-alpha`. Three small commits to end the night: `c2067ed`, `7c5415f`, `2dea4b1`. All ready for tomorrow.

### What shipped after the 23:04:18 full run

- **`c2067ed` &mdash; ROADMAP TTS upgrade section.** Parallel consideration for Bark &rarr; Fish Speech / CosyVoice / Qwen3-TTS. Deferred until FLUX.2+HuMo Stage 4 green. License-first filter keeps OTR MIT. Bark stays shipping for now.
- **`7c5415f` &mdash; FULL workflow cleanup.** Removed the dead sidecar trio (`OTR_VisualBridge`, `OTR_VisualPoll`, `OTR_VisualRenderer`) from `otr_scifi_16gb_full.json` &mdash; they were hitting `BUG-046` meta-tensor every run and burning ~4 min on procedural-video fallback. Rewired `OTR_VideoPlan.audio_gate` &rarr; `OTR_SignalLostVideo.video_path` for sequencing. Also bumped `LLMDirector` token budget from `min(1700, 550+len//10)` to `min(2500, 700+len//6)` &mdash; tonight's 6180-char script got 1168 tokens and truncated mid-`visual_plan.characters`, losing `visual_plan.scenes` entirely (PASS2=0 PASS3=0 downstream). Workflow went 19&rarr;16 nodes, 36&rarr;29 links.
- **`2dea4b1` &mdash; Satellites Collide reference fixture.** Hand-built from the run's `_treatment.txt` + Director log output. Five real characters (DUANE VOSS, PARRY MARTIN, ALAN SIRIKIT, REGINALD HAYES, ANNOUNCER), phantom cast stripped (CAPTAIN JOHNSON / ENSIGN PARKER / CONTROL were a critique-revise bleed-through), full `visual_plan.characters` + `visual_plan.scenes` for all 3 scenes, 7 SFX, 3 music cues. Saves ~37 min per TEST iteration.

### Bugs this closes

- Sidecar FLUX meta-tensor error (BUG-046 family) no longer in the FULL graph.
- `[VisualRenderer] No shot assets found. Falling back to procedural video.` no longer happens.
- `OTR_VideoPlan READY: PASS1=N PASS2=0 PASS3=0` should flip to nonzero PASS2/PASS3 with the new token budget.
- `OTR_ShotDurationCalculator READY: shots=0` should flip to real shot counts.

### What's NOT yet fixed (known + deferred)

- **Phantom cast bleed-through** &mdash; critique/revise still pastes in unrelated scenes (the SPACE STATION / CAPTAIN JOHNSON scene in tonight's run). Needs a guard in `_critique_and_revise()` or a post-revise scrubbing pass. Separate task.
- **Reference POC video is pointer-only** &mdash; 483 MB `.mp4` stays in `output/old_time_radio/`, not in git. Fixture README documents the path. If it ever gets deleted, regenerate from the fixture's `director.json` + script text.
- **Per-line Bark WAVs** &mdash; not yet extracted to the fixture. When Stage 2 (HuMo insertion) lands, write `scripts/extract_reference_bark_wavs.py` that reads the baked audio from the POC `.mp4` and splits it into per-line WAVs keyed off the canonical 1.0 script timing.

### Confirmed architecture for video compositing (Stage 3)

The "creative ffmpeg" proof-of-life deliverable uses a base+overlay model:

- **Base layer:** POC proc-gen `.mp4` (already has the full episode audio baked in with crossfades, waveform visualizer, treatment/title splash). Acts as the scaffold.
- **Foreground overlays:** HuMo clips (97 frames @ 25fps = 3.88s each, one per dialogue line or per shot).
- **Composite pass:** `ffmpeg` overlays HuMo clips onto the scaffold at timecodes that match the character's dialogue in the audio timeline. Moments where no character is on-screen (ANNOUNCER, SFX beats, music bridges) keep showing the POC base.

The math works because both layers derive from the same audio timeline.

### Quick-start for tomorrow (2026-04-24)

1. Pull `v2.0-alpha`, confirm HEAD is `2dea4b1`.
2. Open ComfyUI Desktop &rarr; reload workflows &rarr; load `workflows/otr_scifi_16gb_full.json`. Confirm only **16 nodes** render, with VideoPlan fed from `OTR_SignalLostVideo.video_path` on its audio_gate.
3. Queue one FULL run. Paste the console log into chat when it finishes (or errors). **Expected green signals:**
  - No `[VisualBridge]` / `[sidecar:]` / `[flux_anchor]` noise.
  - No `Falling back to procedural video.`
  - `[LLMDirector] max_new_tokens=...` is ~1700 for a 6k script and there's no `+3 braces` JSON-repair warning, OR the warning is smaller than tonight's.
  - `OTR_VideoPlan READY: PASS1=N PASS2=M PASS3=K` with **nonzero** M and K.
  - `OTR_ShotDurationCalculator READY: shots=X` where X equals the Director's scene count &times; shots_per_scene.
  - `BatchFluxRender` renders **all** shots, not just 1.
4. If green: diff the fresh `production_plan_json` against `tests/fixtures/reference_episode/director_satellites_collide.json`. If structurally similar, call the fixture canonical; if the fresh one is better, promote it and update the fixture.
5. If green: start Stage 1 of `memory/project_flux2_humo_rollout_2026-04-23.md` &mdash; download FLUX.2-klein Q5_K_M GGUF, swap into TEST, compare against tonight's `otr_videoplan_pass3_00003..12.png` baselines.

### Ready-to-paste pickup prompt (copy this into tomorrow's first message)

```
Continuing OTR v2.0-alpha on branch v2.0-alpha. Read in order:

1. ROADMAP.md "Session handoff - 2026-04-23d" (latest section, post-TTS table).
2. memory/project_flux2_humo_rollout_2026-04-23.md.
3. tests/fixtures/reference_episode/README.md.

Last three commits (verify `git log --oneline -3` == c2067ed, 7c5415f, 2dea4b1):
- c2067ed: roadmap TTS upgrade section (Bark -> Fish Speech / CosyVoice / Qwen3-TTS, deferred until Stage 4 green)
- 7c5415f: FULL workflow - removed dead sidecar trio (VisualBridge/Poll/Renderer), bumped Director tokens 1700->2500, rewired VideoPlan.audio_gate to OTR_SignalLostVideo.video_path
- 2dea4b1: Satellites Collide reference fixture (clean director.json with 5 chars + 3 scenes, no phantom cast)

Tonight's goal: run the patched FULL workflow and verify the four green signals
in the ROADMAP handoff. Specifically confirm:
- no sidecar errors
- PASS2/PASS3 nonzero in OTR_VideoPlan
- Calculator sees real shot durations
- BatchFluxRender processes all shots not just 1

If green, diff fresh Director output against the fixture, promote if better.
If green, begin Stage 1 of FLUX.2+HuMo rollout: download FLUX.2-klein
Q5_K_M GGUF, swap into otr_videoplan_TEST.json, re-queue.

Do NOT touch main. Do NOT start Stage 2 (HuMo) before Stage 1 is green.
Do NOT re-run the full pipeline just to get a script - use the fixture.
```

---

## P1 — Audio pipeline (shipped, live-test cycle)

All items code-complete and on `v2.0-alpha`; awaiting real-soak verification as episodes run.

| Item | Summary | Status |
|---|---|---|
| `min_line_count_per_character` self-critique guard | Injected floor=2 into `_critique_and_revise()`; rejects revision if any character drops below. Falls back to pre-critique draft. | Shipped, needs live test |
| Director JSON schema + validator | `_DIRECTOR_SCHEMA` + `_validate_director_plan()` in LLMDirector; repairs missing entries, validates voice_preset strings, filters broken sfx, clamps duration. Wired in `direct()`. | Shipped, needs live test |
| Length-sorted Bark batching | Sort by line length within preset group; script order restored at assembly. Pure throughput win. | Shipped, needs live test |
| VRAM-Sentinel decorator | `vram_sentinel(phase_label, max_entry_gb)` on `BatchBarkGenerator.generate_batch()` at 6 GB ceiling. CUDA-absent safe. | Shipped, needs live test |
| High-creativity soak profile | `"maximum chaos"` re-added to CREATIVITIES pool (~10% weighted). Catches temperature-sensitive regressions. | Shipped, needs live test |
| Per-LLM-call VRAM snapshots | `vram_snapshot("llm_generate_entry"/"exit")` inside `_generate_with_llm()`. Logs tokens + inference time. | Shipped, needs live test |

---

## P2 — Continuity layer (unblocks after video stack sprint ships)

Previously blocked on the retired Gate 0. Now blocked on video stack sprint Day 14. Design begins once stack empirics exist.

| Item | Summary |
|---|---|
| Scene-Geometry-Vault | Series-scale persistent geometry vault so Act 3's bridge matches Act 1's bridge across episodes. Seeded by FLUX anchor outputs from Stage 1. |
| Style-Anchor cache (World Seed + Lighting/Mood split) | Reuse engine over the vault. Same geometry, N relight passes. `style_anchor_hash` in Director schema keys the split. |
| Head-Start async pre-bake (Phase B.5) | Kick off VisualBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability. |
| ASCII sanitizer in prompt_compiler | Strip non-ASCII before Tencent text encoders. Preserve case. Collapse whitespace. Fold into `flux_anchor.py` prompt compiler on video-stack Day 2. |

---

## P3 — Experiments & polish

| Item | Summary |
|---|---|
| `torch.compile` on Bark sub-models | `mode="reduce-overhead"` on semantic, coarse, fine acoustic. Needs isolated A/B timing; variable-length loops may fight the compiler. |
| Skip/shorten Bark fine acoustic pass | Fine pass detail that AudioEnhance destroys via tape emu / LPF / Haas. Needs listening test, not spectrogram. |
| `episode_title` socket input on OTR_SignalLostVideo | Replace implicit `script_json` title-token read with explicit socket from ScriptWriter. v2.1 cleanup. |
| Rename `workflows/soak_target_api.json` → `workflows/helpers/antigrav_api_scratch.json` | Antigravity API-conversion helper; keep but move out of top-level workflows to reduce confusion. |

---

## Recently shipped

| Item | Summary | Status |
|---|---|---|
| v1.7 | Tagged and merged to `main` (`0aa6d6e`) | Shipped |
| BUG-LOCAL-034–040 | Parser resilience, title fixes, JSON repair | Shipped with v1.7 |
| Visual sidecar trio | VisualBridge + VisualPoll + VisualRenderer wired into `workflows/otr_scifi_16gb_full.json` | Shipped |
| VisualRenderer audio-length exact-match | `-t audio_duration` + `tpad` for C7 safety; stderr → tempfile | Shipped (`86bfeae`) |
| Phase A race-free sidecar contract | Atomic writes + Windows `os.replace` retry (`_atomic.py`) | Shipped (`ed4c44f` + `5e795a0`) |
| Phase B v0 SD 1.5 anchor generator | `anchor_gen.py` behind `OTR_VISUAL_ANCHOR=sd15` flag; 27 unit tests | Shipped (`c46a013`) |
| Round-robin consult infrastructure | `scripts/_consult_round_robin.py` (ChatGPT → Gemini → Claude synth) | Shipped |

---

## Discarded (do not revisit)

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10 (stale by multiple minor versions)
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased Visual unified latent space
- **Visual 2.0 Gate 0 probe** (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. VisualBridge + Poll + Renderer harness stays; the backends are the P0 video stack above.
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — did not read as 1980s VHS (pivoted to SDXL + period LoRA, now superseded by FLUX-native anchors under P0)

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/HANDOFF_2026-04-16.md` — last handoff (Phase A + Phase B v0)
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-04-14-otr-v2.1-spec.md` — v2.1 spec
- `docs/2026-04-14-green-zone-guardrail-decision.md` — guardrail decision
- Survival guide: `https://github.com/jbrick2070/comfyui-custom-node-survival-guide`

---

## Daily operating cadence

- First thing: read this file, `CLAUDE.md`, `BUG_LOG.md` header, `git log --oneline -5` on current branch.
- LHM is always on — poll `http://localhost:8085/data.json` (or `outputs/libre_tail.py`) before asking Jeffrey for system status.
- After every code change: AST parse + three regression suites. Do not report "done" until green.
- One `git push` attempt max — if it fails, hand Jeffrey a cmd block with `cd /d` included.
- Verify every push: local HEAD == origin HEAD, no 0-byte files, no BOM, workflow JSONs valid.
- Log bugs the moment they surface. Don't batch. Promote `Bible candidate: yes` to the survival guide only after the fix is verified.

---

## v2.0-alpha session log -- 2026-04-29 (LLM dropdown trim + prompt-engineering audit + GGUF parked)

Four commits shipped today on `v2.0-alpha`:

- **`b123ade`** -- Mark Captain-Eris-Violet and MN-12B-Mag-Mell-R1 as `(EXPERIMENTAL)` in both `model_id` and `cleanup_model_id` dropdowns (`nodes/story_orchestrator.py` and `visual/llm_selector.py`). VisualLLMSelector.select() strips suffix tags before broadcasting downstream so HF lookup still gets a clean ID. Drove off the empirical evidence: wormhole_swallowing_phobos and echo_chamber runs both produced 6-19% of target word count under Captain-Eris + maximum chaos.
- **`db145bd`** -- Reorder dropdown so EXPERIMENTAL models drop to the bottom, validated models surface first. Add `google/gemma-4-E2B-it` (Gemma 4 effective-2B featherweight, edge-targeted). Adds `docs/2026-04-29-llm-edge-case-matrix.md` (8-row test plan; later trimmed to 6 rows after Gemma 2 family removal).
- **`fa83ee2`** -- Remove Gemma 2 family (`gemma-2-2b-it` + `gemma-2-9b-it`). gemma-2-2b proved CUDA-incompatible on Blackwell sm_120 + bnb 4-bit NF4 + torch 2.10.0 + CUDA 13.0 (BUG-LOCAL-110, Bible candidate). gemma-2-9b would route through the same NF4 quantization path so the entire family is treated as not viable on this hardware. Edge-case matrix doc renumbered to 6 rows.
- **`4fa1ec5`** -- Per-character voice priming + AISM bullets + voice consistency soft-warnings. Cast pre-roll now uses `_VOICE_TRAITS` pool so each character carries fixed (gender, age, tone, energy, register, signature) tuples across all scenes. AISM Filter list gains two bullets (animated-environment cliches + telegraphed emotion in dialogue). New `_check_voice_consistency()` walks every `[VOICE:]` tag in the final script and stamps drift mismatches to `ledger.voice_warnings[]` (no schema bump; Ledger.save() merge from BUG-108 preserves arbitrary keys).

Tests across all four commits: 200/200 passed Windows-side venv Python 3.12, run time ~108-110s each (widget_drift_guard 27, two_llm_split 15, dropdown_guardrails 50, test_core 108).

GGUF loader (TheDrummer Rocinante / DavidAU Gemma-The-Writer creative fine-tunes) was attempted via Path X (prebuilt llama-cpp-python wheel) and parked. Latest cp312 Windows wheel on abetlen's index is 0.3.4 built against CUDA 12.4; host runtime is CUDA 13.0; `llama.dll` failed to load due to missing `cudart64_12.dll` / `cublas64_12.dll`. Path Y (full source build with VS Build Tools 2022 + CUDA 13 Toolkit + CMake) is the deterministic path; reopens when Jeffrey schedules the 90-min install window OR when a native cu13 Windows wheel with sm_120 PTX ships. Filed in `docs/2026-04-29-gguf-parked.md` with explicit unblock conditions.

### Next session priorities (CANNOT do this session)

In execution order:

1. **Diff 3 -- spine ledger-stamping with bundled metadata expansion + schema bump** (`docs/2026-04-29-spine-ledger-stamping-ticket.md`).
   - New ledger fields: `outline` (string), `beats[]` (array of `{beat_id, scene_id, summary, characters_present, expected_dialogue_lines}`), `spine_meta` (`{open_hook, close_payoff, character_arcs}`).
   - Bundled metadata additions in same commit (rationale: one regression validates entire ledger-shape expansion):
     - **Items 1-5** (single touch site in `story_orchestrator.write_script()`): `episode_title` top-level, `meta.gen_params` (model_id + cleanup_model_id + target_words + num_characters + target_length + style_variant + creativity + genre_flavor + optimization_profile), `meta.news_seed` (headline + source + url + fetched_at), `meta.bug_109_retries` (count + initial_ratio + final_ratio + fired), `meta.word_ratio_pct`.
     - **Items 6-7** (separate node touches): `meta.title_source` ("user" / "auto_from_spine" / "llm_derived" / "stuck_default") in story_orchestrator, `meta.episode_breakdown_s` (`{opening_s, scene_audio_s, closing_s, total_s}`) in EpisodeAssembler.
   - Schema version bump `l3-2026-04-28` -> `l4-YYYY-MM-DD`.
   - SceneSequencer validates parsed scenes against `ledger.beats[]` as soft warnings to `ledger.structural_warnings[]` (same pattern as today's `voice_warnings[]`).
   - Round-robin consult before kickoff (per CLAUDE.md, schema bumps qualify).
   - Estimated: half a day to a day.
   - **Unblock conditions explicit in the ticket:** (1) 2-3 real-episode runs of `voice_warnings[]` data accumulated, (2) Mistral-Nemo + Gemma 4 E4B both have at least one PASS in the LLM edge-case matrix, (3) v2.0-alpha video stack feature-complete and renders end-to-end without surfacing new BUG-LOCAL-1xx in the audio path, (4) the seven metadata fields are designed and stub-tested alongside the spine fields so the schema bump validates everything at once.

2. **LLM edge-case matrix sweep** (`docs/2026-04-29-llm-edge-case-matrix.md`). Six rows queued. Run them to populate the unblock data Diff 3 needs:
   - Row 1: Mag-Mell (EXPERIMENTAL) / maximum chaos / 350 / short -- does it short-output the way Captain-Eris did?
   - Row 2: gemma-4-E2B-it / balanced / 350 / short -- featherweight kernel-path validation
   - Row 3: gemma-4-E4B-it / balanced / 350 / short -- edge sweet spot for 16 GB
   - Row 4: gemma-4-E4B-it / maximum chaos / 350 / short -- BUG-109 retry on a base model
   - Row 5: Qwen 2.5 14B [ALPHA] / balanced / 700 / medium -- larger model + alpha tag suffix-strip + format gates
   - Row 6: Mistral-Nemo / maximum chaos / 350 / short -- baseline confirmation that today's commits didn't break the validated path
   - Per-run capture template lives in the matrix doc.

3. **GGUF unblock check (lazy poll, do not actively work)** -- on the off chance abetlen ships a cu13 Windows cp312 wheel with sm_120 PTX before Jeffrey schedules the Path Y install window, this becomes a 5-min commit. Curl the `cu130/llama-cpp-python/` index occasionally; otherwise the parked-doc unblock conditions hold.

4. **Voice consistency data analysis** -- after 2-3 real-episode runs accumulate `ledger.voice_warnings[]` entries, summarise the drift pattern (which fields drift most -- gender/age vs tone/energy; which models are worst). The summary is one of the inputs that sizes Diff 3's structural validation work.

### What remains stable after today

- `nodes/_otr_paths.py` portable-paths refactor (committed earlier in the 2026-04-28 marathon)
- Schema l3 (audio_gates with sha256, transitions, master-mix offset shift, dual-ledger atomic-rename fix)
- BUG-100 / BUG-101 / BUG-102 fixes (narration filter, paren-strip, HuMo motion-onset pad)
- BUG-108 dual-ledger fix (atomic rename + on-disk merge in Ledger.save)
- BUG-109 WORD_EXTEND retry loop with no-progress guard
- BUG-110 dropdown trim (Gemma 2 removal -- documented; do not re-add)
- Per-character voice profile pool (`_VOICE_TRAITS` + `pre_rolled_cast_traits`) -- voice_warnings collection is the data-foundation for Diff 3

### Branch & tag plan

`v2.0-alpha` HEAD at session close: **`4fa1ec5`**. Origin lockstep verified. Next session continues on this same branch -- do NOT re-base or merge to main. Tag cut for v2.0-alpha-video-stack happens only after Diff 3 lands AND a clean end-to-end episode renders with the new schema.

```

## BUG_LOG.md -- last ~400 lines

```markdown
- **Date:** 2026-04-17 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Running `pytest tests/test_dropdown_guardrails.py tests/test_core.py` (or combined regression that orders dropdown_guardrails before test_core) reproduces the exact same failure signature BUG-LOCAL-042 had originally: `3 failed, 139 passed, 11 errors in 116.10s`. `ImportError: cannot import name 'vram_sentinel' from 'nodes._vram_log' (unknown location)` at `nodes/batch_bark_generator.py:32`. Test_core alone: 103/103 pass. Bug Bible + test_core: 127 + 2 xfailed. wedge_probe + test_core: 111 passed. The cascade is test-order dependent, not pycache-dependent.
- **Cause:** `tests/test_dropdown_guardrails.py` lines 52-58 builds a `types.ModuleType("nodes._vram_log")` stub with `vram_snapshot`, `vram_reset_peak`, `force_vram_offload`, `register_vram_cleanup` — but **omits `vram_sentinel`**. The stub is assigned to `sys.modules["nodes._vram_log"]` and never restored. When a later test suite (e.g. `test_core.py`) imports `nodes.batch_bark_generator`, the `from ._vram_log import force_vram_offload, vram_sentinel` statement resolves against the polluted stub and raises `ImportError` because the stub doesn't have `vram_sentinel`. The `(unknown location)` phrasing in the error came from `ModuleType` having no `__file__`, not from a stale `.pyc`. Earlier "fixed-by-time" diagnosis was wrong: the original fix only happened to validate isolated `test_core.py` runs, which don't trigger the pollution chain. A second stubbed attribute (`vram_snapshot` returning `None`) also caused `VRAMGuardian.flush()` to fail at `vram_guardian.py:58` with `TypeError: 'NoneType' object is not subscriptable` on `before["current_gb"]`.
- **Fix:** In `tests/test_dropdown_guardrails.py`, two changes: (1) added `_vram_mod.vram_sentinel = lambda *a, **kw: (lambda fn: fn)` to the stub — real `vram_sentinel(label, max_entry_gb)` is a decorator factory, so a pass-through decorator stub is shape-correct for import-time resolution; (2) upgraded `_fake_vram_snapshot` from `pass` (returns `None`) to `return {"phase": label, "current_gb": 0.0, "peak_gb": 0.0}` so callers subscripting the result don't crash. Both changes are test-scaffold fixes, not production code changes. No restoration-on-teardown added because the pattern is deliberate (allows other test files in the same session to import `story_orchestrator` without invoking the real VRAM machinery).
- **Verify:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\test_dropdown_guardrails.py tests\test_core.py tests\test_wedge_probe.py -q` returns `161 passed in 109.40s`. Bug Bible regression: `24 passed, 2 xfailed in 0.87s`.
- **Tags:** test-order-pollution, sys-modules-leak, vram-sentinel, stub-incomplete, batch-bark-generator, test-scaffold, supersedes-bug-local-042

### BUG-LOCAL-043: SD 1.5 `.ckpt` loading through diffusers fails with 4-layer offline/Windows stack [FIXED]
- **Date:** 2026-04-16 | **Phase:** B | **Bible candidate:** yes
- **Symptom:** Phase B smoketest (`scripts/phase_b_smoketest.py`) repeatedly returned `STATUS=READY detail=... SD15 anchors: 0 ok, 2 failed` with `cache_index.json` reporting `"error": "OSError: [Errno 22] Invalid argument"` for every shot, and `io/visual_in/<job>/anchor_error.log` capturing a different failure each fix pass. Layer-by-layer: (1) `_pickle.UnpicklingError: Weights only load failed ... pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint was not an allowed global`; (2) `ModuleNotFoundError: No module named 'pytorch_lightning'` during full unpickle; (3) `[Errno 22]` deep inside `snapshot_download → thread_map → tqdm.fp_write`; (4) `[Errno 22]` inside `tqdm.std.print_status` during pipeline inference step progress.
- **Cause:** Four stacked issues all triggered when `diffusers.StableDiffusionPipeline.from_single_file()` tried to load `v1-5-pruned-emaonly.ckpt` from an OTR sidecar subprocess (`stdout=PIPE, stderr=PIPE`) with 100%-local/offline rules: (1) PyTorch 2.6+ changed `torch.load` default to `weights_only=True`, blocking legacy `.ckpt` pickles that embed `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` metadata; (2) `pytorch_lightning` is not installed in the ComfyUI venv, so even with `weights_only=False` the unpickler needs the module to resolve the class; (3) diffusers calls `snapshot_download` to fetch the diffusers-format model config from HF Hub — violates offline-first constraint and fails on Windows WinError 22 in restricted environments; (4) both the loading-phase and inference-phase tqdm progress bars write to PIPE'd stdout that ComfyUI never drains, triggering `[Errno 22]` once the pipe backs up.
- **Fix:** Four coordinated fixes in `otr_v2/visual/anchor_gen.py::_default_sd15_loader()`: (1) monkey-patch `torch.load` to force `weights_only=False` (kwargs override, not setdefault — diffusers passes `weights_only=True` explicitly); (2) inject 3-level `pytorch_lightning` shim into `sys.modules` with `_ShimModelCheckpoint` placeholder class before load, remove after; (3) vendor `configs/v1-inference.yaml` from CompVis (MIT) and pass `original_config=<local path>` + `local_files_only=True` to skip HF Hub entirely; (4) call `diffusers.utils.logging.disable_progress_bar()` + set env vars (`HF_HUB_DISABLE_PROGRESS_BARS=1`, `TRANSFORMERS_VERBOSITY=error`, `DIFFUSERS_VERBOSITY=error`) before load, and `pipe.set_progress_bar_config(disable=True)` after `pipe.to(device)` to silence both tqdm paths. Also: hardened `otr_v2/visual/worker.py` anchor exception handler to write full traceback to `io/visual_in/<job>/anchor_error.log` (was only recording `type(exc).__name__`, losing all debug info).
- **Verify:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -u scripts\phase_b_smoketest.py` returns `STATUS=READY detail=... SD15 anchors: 2 ok, 0 failed`. Per-shot meta: `anchor_used=True cache_hit=False render.png` ~800KB–1MB. Cold-load first shot ~12s, warm subsequent shots ~3.9s (Job `vs_12cdd0ef1d94`). `cache_index.json` error fields empty strings. Dropdown guardrails 50/50, test_core 103/103, audio byte-identical 7/7+1 skipped.
- **Tags:** sd15, ckpt-loading, diffusers, weights-only, pytorch-lightning-shim, hf-hub-offline, tqdm-pipe-winerror22, windows-only, sidecar-subprocess, anchor-gen, phase-b, four-layer

### BUG-LOCAL-042: `vram_sentinel` import-chain failure cascades into 14 test_core.py failures on Windows [FIXED]
- **Date:** 2026-04-16 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Windows regression run after Phase B v0 push (`c46a013`, `308dde0`) reports `3 failed, 260 passed, 1 skipped, 11 errors`. Errors all read `ImportError: cannot import name 'vram_sentinel' from 'nodes._vram_log' (unknown location)` traced to `nodes/batch_bark_generator.py:32: from ._vram_log import force_vram_offload, vram_sentinel`. Cascades into 11 errors across `TestCleanTextForBark::*`, 1 failure in `TestCleanTextForBark::test_scene_sequencer_clean_matches_batcher`, and 2 `TestVRAMGuardianNode::test_passthrough_*` failures (`vram_guardian.py:58 TypeError: 'NoneType' object is not subscriptable`).
- **Cause:** Stale Windows `__pycache__` from the mid-April Phase B churn. `vram_sentinel` IS and was defined in `nodes/_vram_log.py:150` per grep, and `from ._vram_log import force_vram_offload, vram_sentinel` at `nodes/batch_bark_generator.py:32` is correct source-level — but the `(unknown location)` phrasing in the ImportError indicated the `nodes._vram_log` module spec was `None` when `batch_bark_generator` ran its import, which is the hallmark of a bytecode cache that references a moved/renamed symbol in an older `.pyc`. Neither Phase B (`c46a013`) nor any Day 1-14 sprint commit touched `nodes/_vram_log.py`, `nodes/batch_bark_generator.py`, or `nodes/__init__.py` — last touch on all three is pre-sprint `5cf338e` ("P0+P1+P2 ROADMAP features"). Not surfaced earlier because the Linux sandbox skips these torch-dependent tests, so the cache staleness only ever lived on Windows.
- **Fix:** No code change required — normal dev churn since `5cf338e` invalidated the stale `.pyc` entries organically. Verified on 2026-04-17 by running `tests/test_core.py` against the current Windows venv Python both with warm pycache (103/103 passed in 3.60s) and after a `for /d /r %d in (__pycache__) do rmdir /s /q "%d"` full purge (103/103 passed in 4.38s). All previously-failing classes now green: `TestCleanTextForBark` 12/12, `TestVRAMGuardianNode::test_passthrough_*` 2/2. The 2026-04-17 ROADMAP Day 13 + Day 14 rows carried the caveat "4 failures + 11 errors in v1 test_core.py are pre-existing BUG-LOCAL-042, not caused by Day N" — that caveat is now stale and removed.
- **Verify:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/test_core.py -v` returns `103 passed`. Rerun after a `__pycache__` purge to confirm the fix is structural, not cached.
- **Tags:** vram-sentinel, import-chain, batch-bark-generator, vram-guardian, windows-only, pycache-stale, pre-existing, fixed-by-time, phase-b-non-cause

### BUG-LOCAL-041: ffmpeg zoompan multiplies frames, producing 1880-second clips for 8.7-second shots [FIXED]
- **Date:** 2026-04-16 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** First cut of `_make_motion_clip()` in `otr_v2/visual/worker.py` produced a `render.mp4` with `duration=1898.416667s` for a shot whose `duration_sec=8.7`. Every motion-stub clip ballooned to 30+ minutes long. Renderer then concatenated them and produced a 90-minute "8-minute episode" video. ffprobe confirmed: `nb_frames=45136 r_frame_rate=24/1`.
- **Cause:** `zoompan` is a per-input-frame filter — it emits `d` output frames for EACH input frame. The first implementation fed the still through `-loop 1 -t 8.7` at the default 25 fps, producing 217 input frames; each one was multiplied by `d=208` (8.7s * 24fps), giving 45,136 output frames at 24 fps = ~1880 seconds. The `-t` flag on the input side did not cap the output as I had assumed.
- **Fix:** Switched `_make_motion_clip()` in `otr_v2/visual/worker.py` to the canonical Ken Burns ffmpeg pattern: feed exactly one input frame using `-loop 1 -framerate 1 -t 1 -i still.png`, then cap the zoompan output explicitly with `-frames:v N` where `N = int(round(duration_sec * 24))`. This guarantees `nb_frames == N` regardless of zoompan's internal `d` value.
- **Verify:** Run `scripts/visual_smoketest.py` — Test 7 (renderer with stub assets) PASS. ffprobe each shot's `render.mp4`: `duration` must equal `duration_sec` from `shotlist.json` to within ±0.05s. Verified post-fix: 8.7s expected -> 8.709s actual; 3.5s expected -> 3.500s actual.
- **Tags:** ffmpeg, zoompan, ken-burns, visual, worker, motion-stub, frame-multiplication

### BUG-LOCAL-040: Director JSON parse fails on JS-style comments in LLM output [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `ValueError: Failed to parse production plan JSON. Aborting run to prevent silent audio failure.` Director produced 3491-char output starting with valid-looking JSON but `json.loads()` rejected it. Repair pass logged `Expecting property name enclosed in double quotes: line 13 column 42`.
- **Cause:** Mistral emitted JavaScript-style `//` line comments inside JSON values, e.g. `"v2/en_speaker_8", // Reserved for LEMMY`. Python `json.loads()` does not accept comments. The existing repair pass stripped trailing commas and tried brace-closure but never stripped comments, so the parse always failed at the first `//`. The truncation (open braces) was a secondary issue masked by the comment failure.
- **Fix:** Added `_strip_json_comments()` static method to `LLMDirector` — a state-machine parser that removes `// ...` to end-of-line only outside of quoted strings (preserving URL-like values such as `v2/en_speaker_8`). Wired into `_extract_json` at three points: (1) after the first raw `json.loads()` fails, strip comments + trailing commas and retry, (2) before the truncation-repair brace closure, (3) in the last-resort brace-scan path. Comment stripping runs before trailing-comma stripping so `value, // comment\n}` collapses cleanly.
- **Verify:** Next run should show `[LLMDirector] Plan: N voices, N SFX cues, N music cues` instead of the FATAL. Runs where the LLM emits clean JSON (no comments) hit the first `json.loads()` and skip the stripper entirely — zero perf cost on the clean path.
- **Tags:** director, json-parse, llm-output, comments, mistral, truncation

### BUG-LOCAL-039: Leading markdown bold wrapper leaks into extracted title [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** BUG-LOCAL-038 verification tail showed `TITLE_TRACE | source=script_json | resolved='** Bioluminal Tide' | widget='' | script_json='** Bioluminal Tide'`. Title leaked leading `** ` into the resolved value -- every downstream consumer (filename, video overlay, log lines) carried the cosmetic garbage.
- **Cause:** Mistral emitted `TITLE: **Bioluminal Tide**` (markdown bold wrapping the value, not the whole line). `_RE_TITLE_LINE` only strips `**` as an optional prefix BEFORE the word `TITLE` and as an optional suffix AFTER a trailing quote. The lazy capture group `(.+?)` grabbed the leading `**` of the value and retained it. `_extract_title_from_script_text` only post-processed the capture with `strip()` and quote-strip, no markdown-wrapper strip.
- **Fix:** In `_extract_title_from_script_text` (around line 1586 in `nodes/story_orchestrator.py`), add two regex substitutions right after the quote-strip to peel leading/trailing `*`/`_` runs (1-3 chars) plus surrounding whitespace, then re-run the quote-strip so nested cases like `**"Title"**` still land clean. Empty-result guard added so a `TITLE: ****` residue returns `""` instead of an empty string that later stages might treat as valid.
- **Verify:** Next run's `otr_runtime.log` should show `TITLE_TRACE ... resolved='Bioluminal Tide'` (no leading `**`) when Mistral emits markdown-bold titles. Existing Gemma/Nemo runs with unwrapped titles must remain unchanged. Filename must still vary per episode (BUG-LOCAL-035 regression guard).
- **Tags:** title-extraction, regex, markdown, mistral, cosmetic, post-processing

### BUG-LOCAL-038: BatchBark sees 0 dialogue lines despite Grammarian reporting 21 -- Bark bus renders only ANNOUNCER [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Out-of-box QA run (Mistral, defaults, seed=Bacterial Echo at 13:23-13:36) shipped `signal_lost_bacterial_echo_20260415_133600.mp4` with the correct title (BUG-LOCAL-037 fix verified) but the audio bus had no character dialogue. Heartbeat trail:
  * `ScriptWriter DONE: 1749 tokens ... | 4 scenes | 21 dialogue lines | Characters: ANNOUNCER, DRACULA MALONE, JOHN VANCE, KELLY ECKELS, SOM HALLOWAY` (streaming detector post-grammarian)
  * `SCENE_TRACK: 06_AFTER_GRAMMARIAN | count=4`
  * `TITLE_STRIP | extracted='Bacterial Echo'`
  * `SCENE_TRACK: 07_AFTER_PARSE | count=4` (no dialogue-count checkpoint)
  * `[BatchBark] Found 0 dialogue lines in Canonical 1.0 format (skipped 1 ANNOUNCER lines - routed to Kokoro bus)`
  Upstream word-extend path also failed on this run (WORD_ENFORCEMENT 0 words / 700 target on the primary pass, recovered to 33% only via pre-rolled cast fallback), confirming the final `script_json` had dialogue strings somewhere in the text but `_parse_script` did not emit canonical `{"type":"dialogue"}` tokens for them.
- **Cause:** Chain-of-custody break between FORMAT_NORM / GRAMMARIAN and `_parse_script` for Mistral's bare `NAME: text` dialogue style. Three compounding gaps:
  1. `_normalize_script_format` skip heuristic (line 5289) treated `canonical_count >= 5` raw `NAME:` matches as "already canonical" and skipped the LLM rewrite to `[VOICE: NAME, traits]`. Mistral's native output is bare `NAME: text` -- that format made the skip-counter happy but was NOT a format `_parse_script`'s four VOICE-tag patterns accepted.
  2. `_parse_script` had no first-class pattern for bare `NAME: dialogue`. v1-v4 all required `[VOICE: ...]` or `[NAME, traits]` bracket tags. Anything bare hit the "treat as structural direction" fallback (line ~6488) and was lost as a `{"type": "direction"}` token.
  3. The permissive 2B-fallback inside `_parse_script` (line 6506) only fired when the strict pass produced `dialogue_count == 0 AND len(lines) > 0`. If GRAMMARIAN or any upstream pass injected even one malformed VOICE tag that registered as a dialogue token, the guard short-circuited and the bare `NAME:` recovery pass never ran -- leaving 20+ legitimate dialogue lines stranded as direction tokens.
  No dialogue-count checkpoint existed between 06_AFTER_GRAMMARIAN and BatchBark input, so the exact loss point was invisible.
- **Fix:** Four-part defense in `nodes/story_orchestrator.py`:
  * **Diagnostic** (around line 3848): added `DIALOGUE_TRACK: 07_AFTER_PARSE | count=N | characters=[...]` runtime log line right after the existing SCENE_TRACK checkpoint. Makes the silent drop visible in one grep on any future run.
  * **FORMAT_NORM skip tightened** (line 5289): changed `has_dialogue = (voice_tag_count >= 3 or canonical_count >= 5)` to `has_dialogue = (voice_tag_count >= 3)`. Bare `NAME:` scripts must now always go through the LLM rewrite pass; they no longer look canonical to the skip heuristic.
  * **2B-fallback guard loosened** (line ~6506): replaced single `dialogue_count == 0 AND len(lines) > 0` trigger with an OR of that original trigger and a new `dialogue_count < 3 AND raw-text has 5+ NAME: shape matches`. Now any handful of stray malformed VOICE tags at the top cannot short-circuit recovery of a genuinely bare-NAME script. A raw-text pre-check prevents false firings on narration-only treatments.
  * **v5 VOICE pattern added** (before the direction fallback around line 6487): first-class regex for bare `NAME: dialogue` (accepts 0-2 asterisks, optional `(emotion)` parenthetical after the name, structural-token blacklist covering `TITLE`/`ENV`/`SFX`/etc but explicitly allowing `ANNOUNCER`). Registers the token as `{"type":"dialogue", "character_name": ..., "voice_traits": ..., "line": ...}` directly from the strict parse, so FORMAT_NORM becomes nice-to-have (adds voice traits) rather than load-bearing.
- **Verify:** Next run's `otr_runtime.log` should show `DIALOGUE_TRACK: 07_AFTER_PARSE | count=N | characters=[...]` with N matching or beating the streaming heartbeat's dialogue count. `[BatchBark] Found N dialogue lines` in `comfyui_8000.log` should be N >= 15 for a normal 12-minute Mistral episode (was 0 before). Full MP4 audio should contain character dialogue audible in VLC, not announcer-only. Scripts emitted natively in `[VOICE: ...]` format (Gemma) or bracket-shorthand `[NAME, traits]` (Nemo) must still parse through their existing paths unchanged.
- **Tags:** format-norm, parser, canonical-1.0, batch-bark, mistral, skip-heuristic, permissive-fallback, silent-drop

### BUG-LOCAL-037: BUG-LOCAL-035 fix made the parser see "TITLE" as a speaking character [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Out-of-box-settings test (Mistral, defaults) showed `Characters: ANNOUNCER, LEMMY, QUINN MARTIN, TITLE, VICTOR WELLS, WATSON BERNARD` in the ScriptWriter heartbeat. Dialogue collapsed across the self-critique pass (28 dialogue lines -> 2). Final WORD_ENFORCEMENT logged 2 words / 700 target (0%) and WORD_EXTEND could only recover 1 valid line. The MP4 still rendered with the correct title (`signal_lost_magnetic_echo_*.mp4`) but had almost no dialogue.
- **Cause:** BUG-LOCAL-035's writer-prompt addition required Mistral/Gemma to emit `TITLE: <name>` as the very first line of output. Both the streaming token detector (line ~1885 inline tuple) and the post-stream parser (`_DIALOGUE_FALSE_POSITIVES` frozenset, line 1545) match a `NAME: text` shape as a dialogue line, and neither blacklist contained `TITLE`. Result: every `TITLE: Magnetic Echo` line got booked as character `TITLE` speaking the title text, polluting the cast roster, eating the word budget, and almost certainly nudging the self-critique into deleting "redundant" dialogue.
- **Fix:** Three-part defense in `nodes/story_orchestrator.py`:
  * Added `"TITLE"` to `_DIALOGUE_FALSE_POSITIVES` frozenset (line 1545) -- catches it in the canonical post-stream parser, the dialogue-extension cast filter, and any other consumer of the shared blacklist.
  * Added `"TITLE"` to the streaming heartbeat's inline false-positive tuple (line ~1885) -- removes the noisy `Characters: ..., TITLE` log lines and stops the running cast count from inflating mid-stream.
  * Belt-and-suspenders: just before `_parse_script` (line ~3798), capture the LLM's `TITLE:` line via `_extract_title_from_script_text` into `_early_llm_title`, then strip all `TITLE:` lines from `script_text` with `_RE_TITLE_LINE.sub("", text)`. Wired the captured value into the title-resolution block so `source=llm` resolution still works after the strip. Logs `TITLE_STRIP | extracted=...` so future runs can confirm the strip happened.
- **Verify:** Next run's `otr_runtime.log` should show `TITLE_STRIP | extracted='...'` AND a `ScriptWriter DONE` line whose `Characters: ...` list does NOT contain `TITLE`. Filename should still vary (BUG-LOCAL-035 must remain fixed). `TITLE_TRACE source=llm` should still resolve to the correct title.
- **Tags:** title-stuck, regression, dialogue-parser, false-positive, cast-roster, self-critique

### BUG-LOCAL-036: WordExtend NameError `_false_positives is not defined` — 100% fail on every run [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `[WordExtend] Extension pass failed: name '_false_positives' is not defined` on every single overnight soak run (40/40 failures, SHORT_DURATION scan flag). Extension pass was silently short-circuiting; every episode shipped at the un-extended word count.
- **Cause:** Literal one-character typo inside the WordExtend node body. The module-level constant is `_DIALOGUE_FALSE_POSITIVES` (defined near line 1545 of `nodes/story_orchestrator.py`), but the extension-pass code path referenced an abbreviated local name `_false_positives` that was never bound. Python raised NameError, the surrounding try/except logged it and returned the script unchanged.
- **Fix:** Changed the reference to the correct module-level constant `_DIALOGUE_FALSE_POSITIVES` (`nodes/story_orchestrator.py` around line 6065). Added a short comment noting the old name for future grep-archaeology.
- **Verify:** `grep -rn "_false_positives" nodes/` shows only the comment line — no live code reference. Next real-workflow run should emit target word counts; SHORT_DURATION scan flag should clear.
- **Tags:** typo, word-extend, short-duration, scan-flag, name-error

### BUG-LOCAL-035: TITLE_STUCK — every episode filename locked to "The Last Frequency" regardless of LLM output [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Overnight soak produced 40 consecutive `The_Last_Frequency_*.mp4` files. Treatment scanner raised TITLE_STUCK on every run. Writer was asked for a title, but the filename path ignored it.
- **Cause:** Two independent gaps.
  1. `workflows/otr_scifi_16gb_full.json` hardcoded the widget default `"The Last Frequency"` on three nodes: Node 1 `OTR_Gemma4ScriptWriter` widget #0, Node 7 `OTR_EpisodeAssembler` widget #0, Node 12 `OTR_SignalLostVideo` widget #5.
  2. The writer's prompt asked for `EPISODE TITLE: ...` but the output was never parsed back. `OTR_SignalLostVideo`'s title-extraction code handled only the dict-format `script_json`; the actual Canonical 1.0 list-format fell through to the widget default on every run. No catch, no error — silent lock-in.
- **Fix:** Four-part fix across three files.
  * `workflows/otr_scifi_16gb_full.json`: cleared all three hardcoded widget defaults to `""` so the writer/video nodes fall through to real resolution.
  * `nodes/story_orchestrator.py`: added module-level `_STUCK_TITLE_DEFAULTS` frozenset + `_RE_TITLE_LINE` + `_extract_title_from_script_text()` + `_derive_title_from_script_lines()`. Updated the Gemma user_prompt (both OC-mode and non-OC branches) to require `TITLE: <...>` as the **very first** line of output, with an anti-stuck-default instruction. In `write_script`, added four-tier title resolution (user override → LLM-parsed → derived-from-first-environment → timestamp), prepended a `{"type": "title", "value": _resolved_title}` token to `script_lines`, and log one `TITLE_TRACE` line per run with raw/parsed/final values.
  * `nodes/video_engine.py`: rewrote the title-extraction block in `render_video` to (a) scan list-format `script_json` for the new `title` token, (b) fall back to dict-format top-level `title`, (c) fall back to the widget, (d) hard-fail with `RuntimeError("TITLE_RESOLVE_FAIL: ...")` if all paths yield a stuck default. Cleared the default widget value in `INPUT_TYPES` and the `render_video` signature.
- **Verify:** Next run should emit a `TITLE_TRACE | raw="..." | parsed="..." | final="..."` line. Filename stem should vary across 5 consecutive runs. `TITLE_RESOLVE_FAIL` firing is the desired loud behavior if resolution ever breaks again.
- **Tags:** title-stuck, widget-default, workflow-json, writer-prompt, video-render, scan-flag, fail-loud

### BUG-LOCAL-034: Fatal-streak auto-halt + user-controlled STOP_FILE pause for soak operator [FIXED]
- **Date:** 2026-04-15 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** Overnight soak churned through 40 identical TITLE_STUCK + WordExtend failures because there was no auto-halt on repeated fatals. Jeffrey had no clean pause-and-ask contract.
- **Cause:** Soak loop in `scripts/soak_operator.py` flagged failures reactively but never counted streaks or respected a user stop signal between runs.
- **Fix:** Added module-level constants (`STOP_FILE`, `STOP_POLL_S`, `FATAL_STREAK_LIMIT=3`), a sliding window `_recent_fatal_tags`, and four helpers: `classify_fatal(result, error_msg, scan_flags)`, `check_fatal_streak(tag)`, `trigger_fatal_halt(run_num, tag)`, `wait_for_stop_clear(run_num)`. Wired `classify_fatal` + `check_fatal_streak` into `run_iteration` right after the treatment-scan block so both the run outcome and scan flags feed tagging. Wired `wait_for_stop_clear` at the top of the main `while True:` loop so each iteration honors a live stop before spinning up the next run. Clean SUCCESS with no streakable scan flags resets the window. Three identical fatal tags in a row writes STOP_FILE, sends an urgent ntfy, and blocks the next iteration until Jeffrey removes the file.
- **Verify:** Touch `scripts/.soak_stop` between runs — next iteration should pause and poll every 30s until removed. Trigger three identical fatals (e.g. three TITLE_RESOLVE_FAIL raises) and confirm STOP_FILE is created automatically.
- **Tags:** soak, operator, streak, stop-file, user-control

### BUG-LOCAL-001: v2_preview.py placeholder nodes flagged as output nodes [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Bug Bible regression BUG-01.02 fails: "Output nodes without folder_paths usage: v2_preview.py"
- **Cause:** All four v2 placeholder nodes (CharacterForge, ScenePainter, VisualCompositor, ProductionBus) had `OUTPUT_NODE = True` despite not writing any files to disk. They only return in-memory tensors or strings.
- **Fix:** Removed `OUTPUT_NODE = True` from all four placeholder classes. These nodes are data-flow nodes, not file-output nodes.
- **Verify:** `python -m pytest bug_bible_regression.py -v --pack-dir .` passes BUG-01.02
- **Tags:** widget-drift, registration, bug-bible

### BUG-LOCAL-002: Stale TestWorkflowJSONLite references deleted workflow [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** 8 test errors + 1 failure in test_core.py: FileNotFoundError for otr_scifi_16gb_lite.json
- **Cause:** The lite workflow was removed in commit 44cbdec ("chore: remove lite workflow") but the TestWorkflowJSONLite test class was not cleaned up.
- **Fix:** Removed the entire TestWorkflowJSONLite class from test_core.py with a comment noting the removal reason.
- **Verify:** `pytest tests/test_core.py -v` shows 83 passed, 0 errors
- **Tags:** stale-test, cleanup

### BUG-LOCAL-003: Widget-value drift in workflow-to-API prompt conversion [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ComfyUI rejects POST /prompt with HTTP 400. Node #15 (BatchAudioGenGenerator) gets `episode_seed: 3.0, model_id: 3.0` instead of `episode_seed: "", model_id: "facebook/audiogen-medium"`.
- **Cause:** Workflow-to-API conversion mapped `widgets_values` positionally to ALL widget-capable params. But ComfyUI's workflow JSON excludes linked inputs from `widgets_values`, so linked params (script_json, production_plan_json) consumed slots 0-1, shifting all downstream values by 2 positions.
- **Fix:** Filter widget-capable params to only UNLINKED ones before positional mapping. `unlinked_widgets = [p for p in widget_capable if p not in linked]`.
- **Verify:** Regenerate debug_prompt.json and check node #15 values are correct.
- **Tags:** widget-drift, api, baseline-capture

### BUG-LOCAL-004: v2 placeholder nodes cause API 400 from missing required inputs [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** ComfyUI rejects prompt because CharacterForge, ScenePainter require MODEL/CLIP/VAE inputs that are not connected in the audio-only workflow.
- **Fix:** Strip v2 placeholder nodes from the API prompt before submission. They are not part of the audio pipeline.
- **Verify:** Prompt submits successfully with only audio-pipeline nodes + PreviewAudio capture node.
- **Tags:** api, baseline-capture, placeholder

### BUG-LOCAL-005: Emoji vs [EMOJI] placeholder mismatch in dropdown values [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ComfyUI 400 error: `runtime_preset: '\ud83d\udcfb standard (12 min)' not in ['[EMOJI] standard (12 min)', ...]`
- **Cause:** Workflow JSON stored real Unicode emoji (e.g. U+1F4FB) in dropdown values, but the running ComfyUI node code uses `[EMOJI]` as a text placeholder. The API prompt validation does exact string matching.
- **Fix:** Added `_dropdown_text_match()` that strips leading emoji or `[TAG]` prefixes before comparing, and remaps to the schema's expected value.
- **Verify:** Regenerate debug_prompt.json and check node #1 runtime_preset matches schema.
- **Tags:** encoding, widget-drift, api, baseline-capture

### BUG-LOCAL-007: PARSE_FATAL when target_length=short (3 acts) + runtime_preset=[FAST] quick (5 min) [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ScriptWriter generates 4 scenes, 48 lines, but 0 parseable dialogue lines. PARSE_FATAL fires, execution aborts. Episode never reaches TTS stage.
- **Cause:** `short (3 acts)` compresses the arc so aggressively that Mistral-Nemo produces narration/outline-style content instead of `CHARACTER: dialogue` format. The parser finds no dialogue tags and hard-aborts.
- **Fix:** Keep `[FAST] quick (5 min)` runtime target but use `medium (5 acts)` for `target_length`. Five acts requires 45 minimum dialogue lines, forcing proper dialogue structure. workflow updated: `target_length` = `medium (5 acts)`.
- **Verify:** Run `test_audio_byte_identical.py --capture-baseline` and confirm ScriptWriter log shows `dialogue lines > 0`.
- **Tags:** script-writer, parse-fatal, episode-length

### BUG-LOCAL-008: Node 15 (OTR_BatchAudioGenGenerator) widget drift recurrence [FIXED-WORKAROUND]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** API prompt has `episode_seed: 3.0, model_id: 3.0` (both float) instead of `episode_seed: "", model_id: "facebook/audiogen-medium"`. Positional mapping shifted by 2.
- **Cause:** ComfyUI `/object_info` schema returns `optional` params in a different order than `INPUT_TYPES` defines them. The `_workflow_to_api_prompt` positional mapper uses schema order for `params_with_wv_slot`, but `widgets_values` are stored in `INPUT_TYPES` order. When the schema omits or reorders optional params, the wv indices are wrong. Root cause: schema ordering vs INPUT_TYPES ordering mismatch for this node specifically. `debug_audiogen_schema.json` is dumped on each baseline run for diagnosis.
- **Fix (workaround):** `_fix_known_widget_drift()` in `_run_baseline.py` hardcodes correct values for `OTR_BatchAudioGenGenerator` after prompt conversion. Real fix requires aligning schema ordering â€” see `debug_audiogen_schema.json` output.
- **Verify:** Check `debug_prompt.json` after run â€” node #15 should show `episode_seed: "", model_id: "facebook/audiogen-medium"`.
- **Tags:** widget-drift, api, baseline-capture, schema-ordering

### BUG-LOCAL-009: Preset/target_length mismatch causes wrong dialogue line targets
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `runtime_preset=[FAST] quick (5 min)` paired with `target_length=medium (5 acts)` tells the LLM "Target 8-minute runtime, MINIMUM 45 dialogue lines" even though the actual runtime is 5 minutes. LLM overshoots or gets confused by conflicting length signals.
- **Cause:** `length_instruction` dict was hardcoded per `target_length` with fixed runtime targets and dialogue line minimums that did not scale with `target_minutes`. Also, the 1-min test preset was prone to PARSE_FATAL (see BUG-LOCAL-007).
- **Fix:** (1) Removed 1-min test preset, set minimum to 3 minutes. (2) Added `_safe_length_for_preset` auto-clamp: each runtime_preset forces the safe `target_length` (e.g. quick->medium, long->long 7-8 acts, epic->epic 10+ acts). (3) Made `length_instruction` dynamic: dialogue line floor = `max(18, target_minutes * 8)`, act label from `target_length`, runtime target from actual `target_minutes`.
- **Verify:** Run with each preset. Check runtime log for "PREFLIGHT: Auto-clamped target_length" when mismatch detected. Verify `length_instruction` shows correct minute target and proportional line count.
- **Tags:** preset, length-scaling, parse-fatal-prevention

### BUG-LOCAL-011: Obsidian profile string mismatch - all guardrails dead [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Selecting "Obsidian (UNSTABLE/4GB)" in ComfyUI had zero effect - no one-shot mode, no token cap, no runtime clamp. Obsidian users got full Pro behavior, then OOM on 4GB cards.
- **Cause:** Code checked for `"Obsidian (Low VRAM/Fast)"` (6 locations in story_orchestrator.py) but INPUT_TYPES dropdown value is `"Obsidian (UNSTABLE/4GB)"`. String never matched. Likely a rename in the UI that was never propagated to the runtime code.
- **Fix:** Replace all 6 occurrences of `"Obsidian (Low VRAM/Fast)"` with `"Obsidian (UNSTABLE/4GB)"` to match INPUT_TYPES. Caught by new `test_dropdown_guardrails.py` regression suite (59 tests).
- **Verify:** Run `pytest tests/test_dropdown_guardrails.py -v` â€” TestGuardrails::test_obsidian_disables_multipass and test_obsidian_caps_runtime must pass.
- **Tags:** string-mismatch, obsidian, guardrails, dead-code

### BUG-LOCAL-010: Full pre-flight guardrail sweep [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** Multiple unguarded parameter combos could cause silent failures or PARSE_FATAL: (a) 2 characters + 7+ acts = dialogue starvation, (b) 8 characters + 5 min = too many voices for runtime, (c) "maximum chaos" + chunked outline pushes temp above model max, (d) Obsidian + 20 min = 2500 token cap truncates 60% of script, (e) `news_headlines` widget has zero effect, (f) `temperature` widget silently overridden by `creativity`.
- **Cause:** Pre-flight validation only checked 1-min edge case. No guardrails for character count vs episode length, no profile-aware runtime cap, no temp ceiling in outline gen.
- **Fix:** (a) Clamp chars to 4 if <=5 min, to 3 if <=3 min. Floor chars to 3 if >=7 acts. (b) Obsidian profile caps target_minutes at 10. (c) Outline gen temp no longer adds +0.1 when already >= 1.0. (d) Deprecated tooltips on news_headlines and temperature widgets.
- **Verify:** AST parse clean. Check PREFLIGHT log lines for each clamp scenario.
- **Tags:** guardrails, pre-flight, parameter-validation

### BUG-LOCAL-006: Converted widget alignment in widgets_values mapping [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Node #2 (Gemma4Director) gets `tts_engine: 0.4` (should be dropdown string). Widget values shifted by 1.
- **Cause:** The BUG-LOCAL-003 fix skipped ALL linked inputs from positional mapping, but linked inputs with a `"widget"` flag in the workflow JSON ("converted widgets") still keep their slot in `widgets_values`. Only linked inputs WITHOUT the widget flag should be skipped.
- **Fix:** Check `inp.get("widget")` on each linked input. Include converted widgets in the positional mapping, skip non-widget links.
- **Verify:** Regenerate debug_prompt.json and check node #2 values: temperature=0.4, tts_engine='bark (standard 8GB)', vintage_intensity='subtle'.
- **Tags:** widget-drift, api, baseline-capture

### BUG-LOCAL-012: Episode duration significantly undershoots target_minutes
- **Date:** 2026-04-12 | **Phase:** 0-1 | **Bible candidate:** yes
- **Symptom:** Test run "The Last Frequency" with target_minutes=3, 2 characters, 3 acts, Standard profile generated a 2-minute episode (vs 3-minute target). ~33% duration shortfall.
- **Cause:** Dialogue scaling formula enforces **line count minimum** (floor = max(18, target_minutes * 8)) but not **dialogue density**. For 3 min with 2 chars: floor = 24 lines total (12 per char). LLM hit the minimum and stopped, natural pacing resulted in ~1 min audio runtime. The 41 total generated lines (39 dialogue + 2 ANNOUNCER) meet the **count** requirement but not the **duration** requirement.
- **Fix:** (Phase 0.5) Relabel target_minutes dropdown to reflect realistic output range: "Target 3 (actual 2-3 min)" instead of exact promise. No code change â€” UI expectation mismatch only.
- **Verify:** Added UI warning labels to INPUT_TYPES. User sees "2-3 min" as the expected range when they select "3 min".
- **Tags:** duration, dialogue-scaling, episode-length, ui-expectation

### BUG-LOCAL-014: Maximum chaos creativity produces unparseable dialogue format
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** 8 chars + 3 min + maximum chaos: ScriptWriter generated 1065 tokens across 3 scenes but parser found 0 dialogue lines. LLM used `*NAME*(emotion): dialogue` format with single asterisks. Characters detected as garbage: "ENVSIRENS WAIL", "OPENING THEME". Episode proceeded with near-silent output.
- **Cause:** (1) Maximum chaos (temperature=1.35) pushes Mistral-Nemo into non-standard formatting. (2) Parser Pass 1 regex only accepted 0 or 2 asterisks around names, not 1. (3) Permissive fallback matched structural tags as "characters", so dialogue_count > 0 and PARSE_FATAL never fired.
- **Fix:** Four-layer defense: (a) Clamped maximum chaos temp from 1.35 to 0.95, wild & rough from 1.1 to 0.92 - LLM stays creative but follows structural rules. (b) Hardened Pass 1 regex to accept 0-2 asterisks and filter structural tag names. (c) Added Format Normalizer pass (Creative-to-Strict): same LLM, low temperature, rewrites any dialogue format into strict Canonical 1.0 BEFORE parser runs. (d) Structural name blocklist prevents ENV/SFX/MUSIC tags from being misidentified as characters.
- **Verify:** Run 8 chars + 3 min + maximum chaos again. Check runtime log for "CREATIVITY maximum chaos - temp=0.95" and "FORMAT_NORM: Success". Verify dialogue_count > 0 in ScriptWriter DONE line.
- **Tags:** parse-fatal, creativity, format-drift, dialogue-parser, temperature, phase-0.5

### BUG-LOCAL-015: System cascades to Director crash on 0-dialogue script
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** When ScriptWriter produced a script with garbage "dialogue" (structural tags misidentified as characters), the run continued to Director and Bark. Director attempted to generate voice assignments for "ENVSIRENS WAIL" and "OPENING THEME". Bark generated near-silent audio.
- **Cause:** Permissive fallback matched structural tags, returning dialogue_count > 0, which bypassed PARSE_FATAL. No quality gate between ScriptWriter output and downstream nodes.
- **Fix:** (a) Structural name blocklist in permissive fallback prevents false positive matches. (b) Format Normalizer pass gives the parser clean input. (c) PARSE_FATAL still fires as last resort if both normalizer and fallback fail.
- **Verify:** Same test as BUG-014. If normalizer fails gracefully, PARSE_FATAL should fire with clear error instead of silent garbage propagation.
- **Tags:** cascade-failure, parse-fatal, quality-gate, phase-0.5

### BUG-LOCAL-013: UI doesn't warn user when guardrails clamp parameters [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** User selected "8 characters + 3 minutes" but the guardrail silently clamped to 3 chars in the logs. No warning visible in ComfyUI UI â€” user had no idea the setting was changed.
- **Cause:** Guardrail warnings were logged internally but not returned to the UI. ComfyUI only shows what the nodes return.
- **Fix:** ~~(v1) Prepend as comment block to script_json~~ Caused BUG-016. **(v2)** Guardrail clamp warnings logged to otr_runtime.log as `GUARDRAIL_UI:` lines alongside existing `PREFLIGHT:` lines. script_json stays pure valid JSON.
- **Verify:** âœ… VALIDATED 2026-04-12 19:50:39: Test ran with 8 chars + 3 min. PREFLIGHT fired and logged clamps. âœ… REVISED: BUG-016 fix confirmed guardrail_warnings log via `_runtime_log()` without corrupting JSON.
- **Tags:** ui, guardrails, feedback, phase-0.5

### PHASE 0.5 QA SUMMARY [VALIDATED 2026-04-12]
- **All fixes deployed and tested together**
- **Test case:** 8 characters + 3 minutes + maximum chaos creativity
- **Run 1 result (old code, temp=1.35):**
  - PREFLIGHT guardrails fired: clamped 8â†’3 chars, disabled act breaks
  - FORMAT_NORM activated but reported "No improvement" (both counts 0)
  - Parser recovered 6 dialogue lines via permissive fallback
  - QA_REPAIR auto-injected ANNOUNCER bookends (generic canned text)
  - **KokoroAnnouncer crashed (BUG-016):** JSON comment prefix broke `json.loads()`
- **Post-crash fixes applied:**
  - BUG-016: âœ… Guardrail warnings now log-only, script_json stays pure JSON
  - BUG-014 (updated): âœ… Temperature clamped: maximum chaos 1.35â†’0.95, wild & rough 1.1â†’0.92
  - BUG-017: âœ… Story-aware ANNOUNCER via LLM micro-pass replaces canned placeholders
  - BUG-018: âœ… Test suite updated for runtime_preset removal
- **Test suite status:**
  - test_core.py: 83 passed, 21 skipped
  - test_dropdown_guardrails.py: 133 passed, 0 failed
  - AST parse: âœ… Clean
- **Code changes validated:**
  - No BOM: âœ… Confirmed
  - Obsidian strings: âœ… All 8 updated correctly
  - runtime_preset dropdown: âœ… Removed entirely, target_minutes is now sole control
  - Workflow JSON: âœ… Updated to remove runtime_preset widget index
- **Next phase:** Reload ComfyUI and retest with all Phase 0.5 changes live

### BUG-LOCAL-016: Guardrail warning comments break downstream JSON parsing [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Node #13 (OTR_KokoroAnnouncer) crashes with `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. The `script_json` string starts with `// GUARDRAIL WARNINGS:` instead of valid JSON.
- **Cause:** BUG-013 fix prepended `// comment` lines to the `script_json` output from ScriptWriter. JSON does not support comments. KokoroAnnouncer (and any other downstream node) calls `json.loads(script_json)` which fails immediately on the `//` prefix.
- **Fix:** Remove comment-prefix injection from script_json. Guardrail warnings are already logged via PREFLIGHT log lines visible in otr_runtime.log. Instead, store warnings in a separate `guardrail_warnings` string and log them, but keep script_json as pure valid JSON.
- **Verify:** Run 8 chars + 3 min + maximum chaos. KokoroAnnouncer should receive valid JSON and not crash. Check otr_runtime.log for PREFLIGHT warnings still present.
- **Tags:** json-parse, guardrails, downstream-crash, phase-0.5

### BUG-LOCAL-017: QA_REPAIR ANNOUNCER bookends are generic canned text [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** When the LLM fails to generate ANNOUNCER bookends (e.g. at high creativity), QA_REPAIR auto-injects canned placeholder text: "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown." and "And so the transmission ends. This has been Signal Lost. Stay safe." These are completely generic with no story context - no date, no location, no character names, no science hook.
- **Cause:** QA_REPAIR in `_parse_script()` had no access to episode context (title, news, characters). It could only insert hardcoded strings.
- **Fix:** (a) QA_REPAIR now flags missing ANNOUNCER with `__NEEDS_LLM_OPENING/CLOSING__` sentinels. (b) New `_generate_announcer_bookends()` method does a quick LLM micro-pass (temp 0.4, max 200 tokens, ~3-5s) at the `write_script` call site where full context is available. The LLM reads episode_title, genre, news headline, character names, and a dialogue preview to generate story-specific bookends. (c) Falls back to canned text if LLM call fails.
- **Verify:** Run any episode where ANNOUNCER is missing from LLM output. Check otr_runtime.log for "ANNOUNCER_GEN: Generated opening (N chars) + closing (N chars)". ANNOUNCER lines should reference actual story content.
- **Tags:** announcer, qa-repair, llm-micro-pass, story-context, phase-0.5

### BUG-LOCAL-019: FORMAT_NORM times out generating runaway filler tokens [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** FORMAT_NORM LLM pass exceeds 120s wall-clock budget. The LLM generates 1700+ tokens at ~14 tok/s but dialogue count plateaus at 22 lines around token 700 â€” the remaining 1000+ tokens are stage-direction prose, scene descriptions, and padding that the streaming counter never recognizes as dialogue. Fires on every 8-min target run tested so far. Pipeline falls back to original script text and relies on permissive 2B-fallback parser.
- **Cause:** FORMAT_NORM has no early-stop heuristic. The `max_new_tokens` budget is too generous relative to the input script length, and the LLM drifts into narrative prose after exhausting the dialogue content. The 120s timeout is a blunt wall-clock kill, not a quality gate.
- **Fix:** (1) Token budget reduced from `min(2048, len//3+500)` to `min(1024, len//4)` â€” prevents runaway filler. (2) Timeout reduced from 120s to 75s. For a 10k-char script: old budget=2048 tokens, new budget=1024.
- **Verify:** Run 8-min target with maximum chaos. FORMAT_NORM should complete in <75s or bail faster, not generate 1700+ filler tokens.
- **Tags:** format-norm, timeout, runaway-tokens, early-stop, phase-0.5

### BUG-LOCAL-020: Episode duration significantly undershoots target_minutes (systemic)
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** 8-minute target produces 3.7-minute output (46%). 3-minute target produces ~2-minute output (67%). The LLM prompt says "MINIMUM 64 dialogue lines" but only 25 are generated. At maximum chaos, Mistral-Nemo ignores word/line count instructions. The word-per-minute estimator used 130 wpm but Bark TTS actually paces at ~67 wpm, making estimates doubly wrong.
- **Cause:** Three compounding issues: (a) LLM instructions used minutes (not measurable) instead of words (countable). (b) No post-generation enforcement â€” the pipeline accepted whatever the LLM produced. (c) Duration estimator used 130 wpm instead of the measured 67 wpm.
- **Fix:** Word-count enforcement system with raw-text-first pipeline reorder: (1) Convert `target_minutes` to `target_words` using measured Bark rate of 67 wpm. (2) LLM prompt now asks for specific word count ("write at least 536 words of dialogue") instead of minutes. (3) Post-generation pipeline reordered to: **WORD_EXTEND â†’ ANNOUNCER â†’ FORMAT_NORM â†’ Parse**. All four stages operate on raw text before a single final parse. (4) `_extend_script_dialogue()` counts dialogue words via regex on raw text, generates additional dialogue lines via LLM if under 70% target, appends to raw text. (5) ANNOUNCER bookends generated on raw text (sees full extended script). (6) FORMAT_NORM normalizes the complete text (original + extensions + announcer) in one pass. (7) Parser runs once on clean text. (8) Duration estimator fixed to use 67 wpm.
- **Verify:** Run 8-min target. Check runtime log for `WORD_ENFORCEMENT:` lines showing word count vs target, and `WORD_EXTEND:` if extension fires. Final output should be closer to 8 min than 3.7 min.
- **Tags:** duration, word-count, enforcement, extension-pass, bark-wpm, pipeline-reorder, phase-0.5

### BUG-LOCAL-019: Gender assignment inversion in LLMDirector procedural cast
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** LLMDirector JSON plan specifies correct genders (e.g. COMMANDER_MC: "Male, 50s") but procedural cast assigns the opposite gender voice preset. In soak Run 77 (maximum chaos, post_apocalyptic, 2100w): COMMANDER_MC (male) got FLETCHER HUDSON (female, 60s), TARKON_TS (male) got GULLIVER KAPOOR (female, 50s), PALMER_PR (female) got RASHIDA CORBEN (male, 20s). All 4 non-announcer characters had inverted gender assignments.
- **Cause:** Pending investigation. The Director JSON `gender_hints` parse returned 0 hints (`Parsed 0 gender hints from script: {}`), causing procedural cast to ignore the Director's own voice_assignments and assign randomly from the pool. Likely the gender hint regex does not match the maximum-chaos script format.
- **Fix:** pending
- **Verify:** pending
- **Tags:** gender, llm-director, procedural-cast, maximum-chaos, soak

### BUG-LOCAL-020: Name squish and character drift under maximum chaos
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Character "Nemo Sirikit" appears in LLMDirector plan as `NEIMO_NEMEO_SIRIKIT` (hallucinated spelling). In the script body the same character appears as `NS`, `NEMO`, and `NEMO SIRIKIT`. BatchBark cannot map `NS` or `NEMO` to the Director plan, so both fall back to `v2/en_speaker_9` (already assigned to COMMANDER_MC). Result: 3 characters share one voice, 2 voices unused.
- **Cause:** Maximum chaos creativity (highest temperature/top_p) causes the LLM to hallucinate variant spellings of character names. The Director name-matching is exact-match only and cannot reconcile `NEIMO_NEMEO_SIRIKIT` with `NEMO` or `NS`.
- **Fix:** pending
- **Verify:** pending
- **Tags:** name-squish, character-drift, maximum-chaos, batch-bark, voice-collapse, soak

### BUG-LOCAL-021: Act count exceeds target_length ceiling under maximum chaos
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Config specified `medium (5 acts)` but the generated script contains 8 acts (ACT 1 through ACT 8). The act-by-act generation loop did not enforce the target ceiling.
- **Cause:** Pending investigation. The act-by-act chunked generation may not hard-cap the number of iterations, relying on the LLM to self-terminate. Under maximum chaos temperature the LLM keeps generating new acts instead of concluding.
- **Fix:** pending
- **Verify:** pending
- **Tags:** act-count, target-length, maximum-chaos, chunked-generation, soak

### BUG-LOCAL-033: Treatment scanner false-positive storm: U+2500 separator and U+2192 arrow mismatch [FIXED]
- **Date:** 2026-04-14 | **Phase:** v2.0-alpha | **Bible candidate:** no (data-format/regex fix, not a reusable code pattern)
- **Symptom:** Every completed soak run fired five flags simultaneously: EMPTY_CAST, NO_SCENE_ARC, EMPTY_SCRIPT, TITLE_STUCK, NEWS_SEED_MISSING. This happened even on clean successful episodes (e.g. RUN 237: 87% RT, 44 dialogue lines, 9.6GB VRAM). Five flags firing together on every run was the diagnostic signature of a systematic regex failure, not real content problems.
- **Cause:** Treatment files use U+2500 BOX DRAWINGS LIGHT HORIZONTAL (─, repeated ~64 times) as section separators. Both soak_operator.py scan_treatment() and scripts/treatment_scanner.py parse_treatment() used [-]+ in their regex character classes, which only matches ASCII hyphen (U+002D). The U+2500 chars never matched, causing all four separator-dependent sections (CAST & VOICES, SCENE ARC, FULL SCRIPT, NEWS SEED) to fail extraction. Cast entries also use U+2192 RIGHT ARROW (→) which neither script accepted (only -> and --> were in the alternation). TITLE_STUCK was a genuine positive (LLM defaulting title to the show's name -- separate writer-prompt issue).
- **Fix:** In soak_operator.py scan_treatment() and scripts/treatment_scanner.py parse_treatment(): (1) replaced [-]+ with [-\u2500]+ in the CAST, SCENE ARC, FULL SCRIPT, and NEWS SEED section separator regexes; (2) added \u2192 to cast arrow alternation (?:->|-->|\u2192); (3) tightened NO_SCENE_ARC terminator from (?:\n\nFULL SCRIPT) to (?:\nFULL SCRIPT\b) to match the actual single-newline boundary with trailing content. Both files already had encoding='utf-8' on their open() calls; .gitattributes *.txt eol=lf preserves encoding of treatment fixture.
- **Verify:** Smell-check against 3 real treatments (141936, 140330, 134843): all five false-positive flags cleared. TITLE_STUCK remains (intentionally -- real positive). New pytest suite 	ests/test_treatment_scanner_unicode.py (7 tests) added with real fixture 	ests/fixtures/treatment_141936.txt. Pre-existing TestVRAMGuardianNode interaction-with-torch failure confirmed at baseline (dabcebd) and unrelated to this fix.
- **Tags:** scanner, regex, unicode, separator, U+2500, U+2192, false-positive, soak

### BUG-LOCAL-032: Four workflow nodes had preserved-truncated widgets_values shapes; canonicalized to full preserved mode [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** no (data, not code)
- **Symptom:** RUN 227 of the soak operator failed on Node 3 (OTR_SceneSequencer) with ComfyUI HTTP 400: `end_line, {}, invalid literal for int()`. A post-fix schema sweep against live `/object_info` revealed four nodes whose committed `widgets_values` arrays were shorter than the declared widget-backed input count, producing the same preserved-vs-stripped auto-sense ambiguity class as BUG-LOCAL-029 / 031.
- **Cause:** Web-UI workflow JSON can omit trailing unlinked widget slots when they hold defaults. The mapper's auto-sensing heuristic (b3d33bf) handles unambiguous cases (wv_len matches either widget_backed_count for preserved or unlinked_count for stripped) but any wv_len strictly less than widget_backed_count is a preserved-truncated shape the heuristic cannot always reconstruct correctly.
- **Fix:** Use live `/object_info` schema to compute the canonical preserved-mode shape (linked placeholders + all unlinked defaults, in declared input order) for every node and write back the canonical array. Fixed nodes:
  - Node 3 (OTR_SceneSequencer): `['[]', '{}', 0, 999]` (4) -> `['[]', '{}', 0, 999, '', 'bark', 0.0, 0.0]` (8)
  - Node 11 (OTR_BatchBarkGenerator): `[0.7]` (1) -> `['[]', '{}', 0.7]` (3) [canonicalized from stripped to preserved]
  - Node 12 (OTR_SignalLostVideo): `[24, '1920x1080', 'The Last Frequency']` (3) -> `['[]', '{}', '[]', 24, '1920x1080', 'The Last Frequency']` (6)
  - Node 15 (OTR_BatchAudioGenGenerator): `['', 'facebook/audiogen-medium', 3.0, 3.0]` (4) -> `['[]', '{}', '', 'facebook/audiogen-medium', 3.0, 3.0]` (6)
- **Verify:** `scripts/_schema_sweep.py` confirms every node's widgets_values matches its canonical preserved shape (user-tuned Nodes 1/2/4 intentionally diverge in values but match in length). Full sandbox regression: widget_drift 27, dropdown_guardrails 50, core 89, v2/audio 7 = 166 passed. Next soak run expected to clear Node 3 and reach the LLM phase.
- **Tags:** widget-drift, data-corruption, workflow-json, preserved-truncated, scene-sequencer, video-engine, audiogen, batch-bark

### BUG-LOCAL-031: Node 13 (OTR_KokoroAnnouncer) widgets_values truncated to 3 slots, `speed` FLOAT received 'random' [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** no (data bug, not code)
- **Symptom:** RUN 127 of the soak operator failed with ComfyUI HTTP 400: `node_errors: {"13": {"errors": [{"type": "invalid_input_type", "message": "Failed to convert an input value to a FLOAT value", "details": "speed, random, could no..."}]}}`. The `speed` FLOAT param on Node 13 was receiving the string `'random'` (which is the default value for `voice_override`, not `speed`).
- **Cause:** Node 13's `widgets_values` in the committed workflow JSON was `['[]', '', 'random']` â€” only 3 slots. `OTR_KokoroAnnouncer` declares 4 widget-backed params: `script_json` (STRING, linked), `episode_seed` (STRING), `voice_override` (dropdown, default `'random'`), `speed` (FLOAT, default 0.95). The shape was ambiguous between preserved-truncated (linked placeholder + 2 of 3 unlinked values) and pure-stripped (no link slot + 3 unlinked values). The auto-sensing heuristic picked pure-stripped because `wv_len(3) == unlinked_count(3)`, which pushed `voice_override='random'` into the `speed` slot. Pre-existing data corruption, not a mapper bug.
- **Fix:** Set Node 13 `widgets_values` to `['[]', '', 'random', 0.95]` â€” the canonical preserved-mode shape: linked placeholder for `script_json`, then all three unlinked defaults. Auto-sensing now cleanly reads preserved mode (`wv_len(4) == widget_backed_count(4)`).
- **Verify:** Direct mapper trace confirms Node 13 resolves to `script_json='[]'` (overridden by link at runtime), `episode_seed=''`, `voice_override='random'`, `speed=0.95`. Full regression green. Next soak run should clear Node 13 validation.
- **Tags:** widget-drift, data-corruption, workflow-json, kokoro, preserved-truncated, ambiguity

### BUG-LOCAL-030: Node 11 (OTR_BatchBarkGenerator) widgets_values corrupted to ['[]'] in workflow JSON [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** no (data bug, not code)
- **Symptom:** After BUG-LOCAL-029 shipped and auto-sensing mapper was verified against Nodes 2, 13, 15, a direct mapper trace against `workflows/otr_scifi_16gb_full.json` revealed Node 11 resolves to `temperature='[]'` â€” a literal string where a FLOAT is expected. ComfyUI would reject with a type validation error. AntiGravity incorrectly blamed this on the mapper being "blind to dropdowns"; in reality the mapper is correct and the workflow JSON itself is malformed.
- **Cause:** Node 11's widgets_values was literally `['[]']` â€” a single-element list containing the string `'[]'`. The schema for `OTR_BatchBarkGenerator` declares three widget-backed params: `script_json` (STRING, default "[]", linked), `production_plan_json` (STRING, default "{}", linked), `temperature` (FLOAT, default 0.7, unlinked). Auto-sensing correctly treated the shape as stripped mode (len(wv)=1 == unlinked_count=1) and assigned `wv[0]` to the only unlinked widget-backed param, which is `temperature`. The resulting value `'[]'` is the wrong type. Root origin: hand-editing or stale web UI state left a placeholder string in the temperature slot. Predates all recent commits (was present in HEAD before BUG-LOCAL-027).
- **Fix:** Set Node 11 `widgets_values` to `[0.7]` â€” the schema default for temperature. Auto-sensing now produces `temperature=0.7` (correct FLOAT). Also reverted AntiGravity's unauthorized placeholder-strip edits to Nodes 2 and 13 (both shapes the auto-sensing mapper handles, but the committed source of truth should reflect the canonical web-UI-emitted shape).
- **Verify:** `python scripts/_verify_mapper.py` (one-off trace helper) shows Node 11 `temperature=0.7` and all of Nodes 2/13/15 mapping cleanly. Full regression green: widget_drift 27, dropdown_guardrails 50, core 103, bug_bible 22 passed + 2 xfailed, v2/audio_byte_identical 7 passed + 1 skipped.
- **Tags:** widget-drift, data-corruption, workflow-json, hand-edit, bark-generator

### BUG-LOCAL-029: ComfyUI workflow JSON uses two shapes for linked converted widgets; mapper must auto-sense [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** After b6c610e fixed Node 15, Node 2 (OTR_Gemma4Director) started failing with bad dropdown values: `temperature` received `""` instead of `0.4`, `tts_engine` got the wrong slot, and `optimization_profile` shifted down by one. Validation rejected the run. The fix for Node 15 had regressed Node 2.
- **Cause:** ComfyUI's web UI saves `widgets_values` in two inconsistent shapes depending on when and how a widget was converted to a socket. Inspecting `workflows/otr_scifi_16gb_full.json` directly: Node 15 has 6 widget-backed params, 2 linked, and `len(wv) == 4` ("stripped" mode â€” linked converted widgets have NO slot). Node 2 has 5 widget-backed params, 1 linked (`script_text`), and `len(wv) == 5` with an empty-string placeholder at slot 0 ("preserved" mode â€” linked converted widgets keep a placeholder slot). Both shapes are valid; neither is universal.
- **Fix:** `_workflow_to_api_prompt` now auto-senses per-node mode from slot-count arithmetic: if `len(wv) == total widget-backed param count` and there is at least one linked widget-backed input, the node is in preserved mode and linked params consume a placeholder slot. If `len(wv) == unlinked widget-backed param count`, stripped mode â€” linked params consume zero slots. Ambiguous cases (trailing unset optionals, manual JSON edits) default to stripped mode, which errs on the side of omitting bad placeholder values rather than letting them land in real widget keys.
- **Verify:** `pytest tests/test_widget_drift_guard.py` â€” 27 tests pass. New class `TestPreservedSlotMode` covers Node 2's shape end-to-end: `temperature == 0.4`, `optimization_profile == "Pro (Ultra Quality)"`, `script_text` retains its link, socket-only `project_state` absent from inputs. `TestLinkedConvertedWidgetSlots` continues to cover Node 15's stripped mode. On live soak, `API_PAYLOAD node=1` and `node=2` lines should show correct optimization_profile values and no DRIFT_DETECTED output.
- **Tags:** widget-drift, socket-only, linked-converted-widget, api, auto-sensing, mode-detection, bug-bible

### BUG-LOCAL-028: BUG-LOCAL-027 shipfix regressed Node 15: linked converted widgets eat widgets_values slots [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** HTTP 400 value_not_in_list on Node 15 (OTR_BatchAudioGenGenerator) immediately after deploying 2b52ebe. ComfyUI rejected the API submission because `model_id` received `3.0` instead of `"facebook/audiogen-medium"`. Same positional-shift class of bug that BUG-LOCAL-003 and BUG-LOCAL-027 addressed, surfaced on a different param by the new mapper.
- **Cause:** The rewritten `_workflow_to_api_prompt` in 2b52ebe kept a "consume-and-skip" branch for linked params that carried a `"widget": {"name": ...}` metadata block (a converted widget). The reasoning was that a converted widget still reserves its widgets_values slot. In practice ComfyUI's web UI does NOT keep widgets_values slots for inputs that have been converted to sockets â€” it saves slots only for inputs still displayed as widgets. Node 15's `script_json` and `production_plan_json` are linked + carry the `widget` metadata, but have no slots in `widgets_values`. The mapper consumed `wv[0]` (episode_seed's slot) and `wv[1]` (model_id's slot) for nothing, shifting every subsequent value down by two. `model_id` ended up with `wv[3] = 3.0`.
- **Fix:** `_workflow_to_api_prompt` now treats any linked param as consuming zero widgets_values slots, regardless of whether the input has converted-widget metadata. The walk is: start with linked names already populated from the link map, then iterate declared params and only the widget-backed + not-linked ones consume a slot. This is the original BUG-LOCAL-003 contract, restored.
- **Verify:** `pytest tests/test_widget_drift_guard.py::TestLinkedConvertedWidgetSlots -v` (4 tests) locks down the Node 15 case explicitly: `model_id` must stay a string, `episode_seed` must be empty, `guidance_scale` + `default_duration` must land as 3.0 each, and the link tables for `script_json` / `production_plan_json` must survive intact.
- **Tags:** widget-drift, socket-only, linked-converted-widget, api, hotfix, bug-bible

### BUG-LOCAL-027: Widget-drift in soak API mapper emits project_state as string, drops optimization_profile [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Wire-level capture of `soak_target_api.json` showed node #1 (Gemma4ScriptWriter) and node #2 (Gemma4Director) were being submitted with `"project_state": "Standard"` and `"project_state": "Pro (Ultra Quality)"` respectively, while `optimization_profile` was missing entirely. ProjectState loader silently failed to parse the string as a dict, producing an empty preamble; optimization_profile silently defaulted to "Standard" (only by luck matching the intended widget default). Ghost runs (empty CAST / SCENE ARC / FULL SCRIPT) traced back to this corrupted input.
- **Cause:** `_workflow_to_api_prompt` in `scripts/soak_operator.py` walked `widgets_values` positionally against `INPUT_TYPES` order without filtering socket-only params (types like `PROJECT_STATE` that have no widget in the UI). Because `project_state` was declared between `arc_enhancer` and `optimization_profile` in the optional block, every widget after it shifted up by one slot. `optimization_profile`'s value landed in the `project_state` key, and the true `optimization_profile` key was never emitted at all. Same bug class as BUG-LOCAL-003; the fix there addressed linked inputs but not socket-only inputs.
- **Fix:** Added `_is_widget_backed(spec)` helper that returns True for `STRING/INT/FLOAT/BOOLEAN` primitives and for dropdowns (list-typed specs), and False for socket-only custom types. Mapper now walks params in declaration order but only widget-backed params consume a `widgets_values` slot. Socket-only params are either filled via the link map or omitted from `inputs`. Defense in depth: moved `project_state` to the LAST entry in `optional` for both `Gemma4ScriptWriter` (`nodes/story_orchestrator.py`:2484-2534) and `LLMDirector` (`nodes/story_orchestrator.py`:6649-6670) so any future mapper regression cannot shift widget slots. Also stripped the `"3"/"3.0"/3/3.0` back-compat hack from `BatchAudioGenGenerator.model_id` (`nodes/batch_audiogen_generator.py`:102) â€” scar tissue from widget drift that's no longer needed. Added `API_PAYLOAD` and `DRIFT_DETECTED` instrumentation lines in the soak operator just before the POST. Tightened `_RE_SCENE_MARKER` to numeric-only and added a `_RE_SCENE_TERMINATOR` for `=== SCENE FINAL ===` (kills BUG-LOCAL-026 confound).
- **Verify:** `pytest tests/test_widget_drift_guard.py -v` (18 tests) passes. Assertions lock down: (1) `project_state` is never emitted as a string, (2) `optimization_profile` always survives with its correct string value, (3) mapper stays correct even if `project_state` is interleaved before `optimization_profile` in INPUT_TYPES, (4) scene regex no longer captures `FINAL` as a scene number. On next live soak run, runtime log must show `API_PAYLOAD node=1 ... optimization_profile='Standard' project_state=None` and no `DRIFT_DETECTED` lines.
- **Tags:** widget-drift, socket-only, api, soak, ghost-run, input-types, regression-test

### BUG-LOCAL-026: Scene regex matches "FINAL" as a scene number, inflates scene counts [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** `SCENE_TRACK: count=6 | tokens=['1', '2', 'FINAL', '3', '4', 'FINAL']`. FORMAT_NORM's `has_scenes` signal was fooled by pseudo-scenes, and the chunked FORMAT_NORM split treated `SCENE FINAL` blocks as real scene boundaries.
- **Cause:** `_RE_SCENE_MARKER` used `\S+?` for the scene token, which matched any non-whitespace including the literal "FINAL" that the creative LLM emits as a closing-scene marker.
- **Fix:** Tightened `_RE_SCENE_MARKER` to `===\s*SCENE\s+(\d+)(?:\s*:\s*[^=]*?)?\s*===` (numeric only). Added separate `_RE_SCENE_TERMINATOR` for `=== SCENE FINAL ===`. `_scene_inventory` returns numeric tokens followed by `'END'` when a terminator is present, so downstream counts are honest.
- **Verify:** `pytest tests/test_widget_drift_guard.py::TestSceneRegex -v` (5 tests) passes.
- **Tags:** regex, scene-marker, parser, soak

### BUG-LOCAL-024: FORMAT_NORM ghost-run bypass and silent bailout on long scripts [FIXED]
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Runs 011 and 012 under maximum chaos produced scripts with no CAST section, no `=== SCENE N ===` markers, and TITLE_STUCK downstream. Runtime log showed FORMAT_NORM was skipped entirely, despite the script clearly being malformed. On longer scripts FORMAT_NORM also logged `Output too short - keeping original` and silently bailed out.
- **Cause:** Two independent blindspots in `_normalize_script_format`: (1) pre-flight skip heuristic only checked dialogue line count (`voice_tag_count >= 3 OR canonical_count >= 5`), so a ghost run with voice tags but no CAST or scene markers bypassed normalization entirely; (2) token budget was capped at 1024 regardless of script length, so a reformatted output of a 10k-char script could not fit in the budget and triggered the `< 0.3 * input` bailout.
- **Fix:** (1) Tightened skip heuristic to require ALL THREE signals present: `has_dialogue AND has_scenes AND has_cast` (scene marker count >= 1 AND unique character count >= 2). Missing any signal forces FORMAT_NORM to run. (2) Added `_normalize_chunked` following the `_grammarian_chunked` pattern: split by `=== SCENE N ===` markers, reformat each scene independently with a full per-chunk 1024-token budget, 75s per-chunk timeout, reassemble with 80% dialogue-count floor. Single-pass retained for scripts with <=50 dialogue lines or <2 scenes. Also hoisted class constant `_FORMAT_NORM_NON_CHARS` so canonical-name regex excludes SCENE/ACT/SFX/ENV/NARRATOR etc. from the skip count.
- **Verify:** Run soak with a creative-pass script that lacks CAST + scene markers. Runtime log should show `FORMAT_NORM: Running (dialogue=X+YV, scenes=0, cast=0) - missing: scenes cast`, followed by single-pass or chunked flow. For 50+ line scripts, log should show `FORMAT_NORM: Chunked mode` with per-chunk progress.
- **Tags:** format-norm, ghost-run, skip-heuristic, chunked-generation, token-budget, maximum-chaos, soak

### BUG-LOCAL-023: Grammarian timeout on long scripts (60+ dialogue lines) [FIXED]
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** `GRAMMARIAN: Failed (Grammarian exceeded 75s) - keeping original` on a 67-line space opera script. Token budget of 2048 at ~15 tok/s needs ~136s, but timeout was 75s. Grammarian silently falls back to original script, losing all grammar polish.
- **Cause:** Single-pass grammarian with fixed 75s timeout cannot handle scripts with 50+ dialogue lines. The prompt + full script exceeds what the LLM can process within the timeout window.
- **Fix:** Implemented chunked grammarian in `_grammarian_pass()`. Scripts with >50 dialogue lines are split by `=== SCENE N ===` markers, each scene polished independently (90s timeout per chunk, 1024 token budget), then reassembled. Falls back to 40-line raw chunking if no scene markers exist. Single-pass timeout increased from 75s to 150s as safety net. Each chunk has its own dialogue-line safety check; failed chunks keep original text without blocking the rest.
- **Verify:** Run soak with 60+ line episode config. Runtime log should show `GRAMMARIAN: Chunked mode` followed by per-chunk progress, ending with `GRAMMARIAN: Chunked complete`.
- **Tags:** grammarian, timeout, chunked-generation, long-scripts, soak

### BUG-LOCAL-018: test_dropdown_guardrails.py references removed runtime_preset [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** `pytest tests/test_dropdown_guardrails.py` fails with `KeyError: 'runtime_preset'` during collection. 1 additional NameError at runtime for `RUNTIME_PRESETS` variable.
- **Cause:** runtime_preset was removed from INPUT_TYPES but the test file still extracted it from `_REQUIRED` and used it in 12 test locations.
- **Fix:** Removed all runtime_preset references from tests. Replaced `runtime_preset="[FAST] quick (5 min)"` with `target_minutes=5`, etc. Added `test_runtime_preset_removed` assertion alongside existing dead-param checks. Removed obsolete `test_no_1min_test_preset` and `test_runtime_presets_produce_different_target_minutes`.
- **Verify:** `pytest tests/test_dropdown_guardrails.py -v` shows 133 passed, 0 failed.
- **Tags:** test-suite, runtime-preset, cleanup, phase-0.5
```

## Node inventory

### nodes/

- `nodes/__init__.py` (0 KB): -
- `nodes/_otr_ledger.py` (11 KB): -
- `nodes/_otr_paths.py` (13 KB): -
- `nodes/_vram_log.py` (7 KB): -
- `nodes/audio_enhance.py` (18 KB): AudioEnhance
- `nodes/bark_tts.py` (18 KB): BarkTTSNode
- `nodes/batch_audiogen_generator.py` (18 KB): BatchAudioGenGenerator
- `nodes/batch_bark_generator.py` (37 KB): BatchBarkGenerator
- `nodes/batch_humo_render.py` (73 KB): BatchHumoRender
- `nodes/batch_kokoro_generator.py` (8 KB): BatchKokoroGenerator
- `nodes/batch_procedural_sfx.py` (4 KB): BatchProceduralSFX
- `nodes/kokoro_announcer.py` (10 KB): KokoroAnnouncer
- `nodes/musicgen_theme.py` (16 KB): MusicGenTheme
- `nodes/otr_shot_duration_calculator.py` (16 KB): OTRShotDurationCalculator
- `nodes/otr_video_concat.py` (12 KB): OTRVideoConcat
- `nodes/otr_video_plan.py` (31 KB): OTRVideoPlan
- `nodes/post_audio_video_pipeline.py` (16 KB): PostAudioVideoPipeline
- `nodes/production_ledger.py` (25 KB): Ledger
- `nodes/project_state.py` (8 KB): ProjectState, ProjectStateLoader
- `nodes/scene_sequencer.py` (63 KB): SceneSequencer, EpisodeAssembler
- `nodes/script_critic.py` (42 KB): _SafeMissing, LLMScriptCritic
- `nodes/sfx_generator.py` (11 KB): SFXGenerator
- `nodes/story_orchestrator.py` (511 KB): _LLMTimeout, GemmaHeartbeatStreamer, LLMScriptWriter, DirectorJSONParseError, LLMDirector
- `nodes/video_composite.py` (34 KB): VideoComposite
- `nodes/video_engine.py` (60 KB): _CRTRenderer, _TelemetryHUDRenderer, SignalLostVideoRenderer
- `nodes/vram_context_test.py` (15 KB): VRAMContextTest
- `nodes/vram_guardian.py` (2 KB): VRAMGuardian

### visual/

- `visual/__init__.py` (0 KB): -
- `visual/_atomic.py` (4 KB): -
- `visual/_hf_token.py` (3 KB): -
- `visual/anchor_gen.py` (23 KB): AnchorRequest, AnchorResult
- `visual/backends/__init__.py` (3 KB): -
- `visual/backends/_base.py` (6 KB): Backend
- `visual/backends/florence2_sdxl_comp.py` (33 KB): Florence2SdxlCompBackend
- `visual/backends/flux_anchor.py` (37 KB): FluxAnchorBackend
- `visual/backends/flux_keyframe.py` (27 KB): FluxKeyframeBackend
- `visual/backends/ltx_motion.py` (23 KB): LtxMotionBackend
- `visual/backends/placeholder_test.py` (3 KB): PlaceholderTestBackend
- `visual/backends/pulid_portrait.py` (21 KB): PulidPortraitBackend
- `visual/backends/video_stack.py` (12 KB): VideoStackBackend
- `visual/backends/wan21_loop.py` (26 KB): Wan21LoopBackend
- `visual/batch_flux_render.py` (23 KB): BatchFluxRender
- `visual/bridge.py` (26 KB): VisualBridge
- `visual/camera_path.py` (11 KB): Trajectory
- `visual/character_regression.py` (15 KB): PortraitSample, CharacterRegressionResult
- `visual/checkpoint_loader_gated.py` (5 KB): CheckpointLoaderGated
- `visual/flux_prompt_extractor.py` (6 KB): VisualExtractFluxPrompt
- `visual/lhm_monitor.py` (15 KB): LhmSample, LhmSummary
- `visual/llm_polish.py` (14 KB): -
- `visual/llm_selector.py` (4 KB): VisualLLMSelector
- `visual/planner.py` (17 KB): PlannerJob, PlannerResult
- `visual/poll.py` (12 KB): VisualPoll
- `visual/postproc/__init__.py` (1 KB): -
- `visual/postproc/vhs.py` (18 KB): -
- `visual/prompt_coercion.py` (14 KB): VisualPromptCoercion
- `visual/renderer.py` (19 KB): VisualRenderer
- `visual/shotlist.py` (10 KB): -
- `visual/unload_all.py` (6 KB): UnloadAll
- `visual/vram_coordinator.py` (11 KB): LockInfo, VRAMCoordinator
- `visual/wall_clock.py` (9 KB): WallClockEstimate
- `visual/wedge_probe.py` (13 KB): ProbeEvent, ProbeSpan, _NoopSpan, WedgeProbe
- `visual/worker.py` (18 KB): -

### otr_v2/

- `otr_v2/__init__.py` (0 KB): -

### subprocess_runners/

- `subprocess_runners/__init__.py` (0 KB): -

## __init__.py registrations

NODE_CLASS_MAPPINGS keys (33):
- OTR_LLMScriptWriter
- OTR_LLMDirector
- OTR_BarkTTS
- OTR_SFXGenerator
- OTR_SceneSequencer
- OTR_EpisodeAssembler
- OTR_AudioEnhance
- OTR_BatchBarkGenerator
- OTR_BatchKokoroGenerator
- OTR_BatchAudioGenGenerator
- OTR_BatchProceduralSFX
- OTR_SignalLostVideo
- OTR_ProjectStateLoader
- OTR_KokoroAnnouncer
- OTR_MusicGenTheme
- OTR_VRAMGuardian
- OTR_VRAMContextTest
- OTR_LLMScriptCritic
- OTR_VisualBridge
- OTR_VisualPoll
- OTR_VisualRenderer
- OTR_VisualPromptCoercion
- OTR_VisualLLMSelector
- OTR_VisualExtractFluxPrompt
- OTR_CheckpointLoaderGated
- OTR_UnloadAll
- OTR_BatchFluxRender
- OTR_VideoConcat
- OTR_VideoPlan
- OTR_ShotDurationCalculator
- OTR_PostAudioVideoPipeline
- OTR_BatchHumoRender
- OTR_VideoComposite

## Workflow JSON shapes

**otr_scifi_16gb_full.json**: 26 nodes, 44 links

  - id=  1 type=OTR_LLMScriptWriter widgets=15 title='1. Story Writer (LLM)'
  - id=  2 type=OTR_LLMDirector widgets=5 title='2. Director (LLM)'
  - id=  3 type=OTR_SceneSequencer widgets=8 title='4. Scene Sequencer'
  - id=  4 type=OTR_AudioEnhance widgets=7 title='5. Audio Enhance'
  - id=  7 type=OTR_EpisodeAssembler widgets=4 title='6. Episode Assembler'
  - id= 11 type=OTR_BatchBarkGenerator widgets=3 title='3a. Bark Voices (batch)'
  - id= 12 type=OTR_SignalLostVideo widgets=6 title='8. Signal Lost Video (audio-only fallback)'
  - id= 13 type=OTR_KokoroAnnouncer widgets=4 title='3b. Kokoro Announcer'
  - id= 14 type=OTR_MusicGenTheme widgets=4 title='3c. MusicGen Theme'
  - id= 15 type=OTR_BatchAudioGenGenerator widgets=6 title='3d. AudioGen SFX (batch)'
  - id= 20 type=OTR_VideoPlan widgets=6 title='Video Plan (wired from Director)'
  - id= 21 type=OTR_ShotDurationCalculator widgets=4 title='Shot Duration Calc (stub 8s each)'
  - id= 22 type=CheckpointLoaderSimple widgets=1 title='Load FLUX.1-dev-fp8 (video branch)'
  - id= 23 type=OTR_BatchFluxRender widgets=17 title='Batch FLUX Render (full pipeline)'
  - id= 24 type=OTR_UnloadAll widgets=3 title='Unload All (video branch)'
  - id= 25 type=SaveImage widgets=1 title='Save FLUX environment stills (otr/stills/full_env_*)'
  - id= 42 type=PathchSageAttentionKJ widgets=2 title='Patch Sage Attention (FLUX) -- DISABLED, BUG-LOCAL-070'
  - id= 45 type=UNETLoader widgets=2 title='Load HuMo 14B fp8 e4m3fn scaled (Kijai)'
  - id= 46 type=LoraLoaderModelOnly widgets=2 title='lightx2v 4-step distill'
  - id= 47 type=ModelSamplingSD3 widgets=1 title='ModelSamplingSD3 shift=8'
  - id= 48 type=CLIPLoader widgets=3 title='Load umt5_xxl text encoder'
  - id= 49 type=VAELoader widgets=1 title='Load wan_2.1 VAE'
  - id= 50 type=AudioEncoderLoader widgets=1 title='Load Whisper Large v3'
  - id= 51 type=OTR_BatchHumoRender widgets=13 title='Batch HuMo Render (in-graph batch)'
  - id= 52 type=OTR_VideoComposite widgets=13 title='Video Composite (1080p pillarbox + additive proc gen)'
  - id= 53 type=OTR_LLMScriptCritic widgets=3 title='Script Critic (gate)'

**otr_scifi_16gb_TEST.json**: 8 nodes, 9 links

  - id=  9 type=SaveImage widgets=1
  - id= 30 type=OTR_CheckpointLoaderGated widgets=1
  - id= 34 type=Note widgets=1
  - id=101 type=OTR_VisualPromptCoercion widgets=2 title='Prompt Coercion (rule-based)'
  - id=102 type=OTR_VisualLLMSelector widgets=1 title='Visual LLM Selector'
  - id=103 type=OTR_VisualExtractFluxPrompt widgets=4 title='Extract FLUX Prompt (shot 0)'
  - id=104 type=OTR_UnloadAll widgets=3
  - id=105 type=OTR_BatchFluxRender widgets=15 title='Batch FLUX Render (4 shots)'

---

## ChatGPT (gpt-5.4)

Below is an internal QA pass against the material you provided. I’m being candid and I’ll flag uncertainty where I can’t verify code directly from the snippets.

---

# Executive read

## Biggest takeaway

**ROADMAP priority is currently a bit out of sync with the actual bug surface.**  
The roadmap still gives a lot of oxygen to **next-step visual expansion** and medium-horizon schema work, while the **highest-risk pre-run issues are now narrower and more operational**:

1. **Audio byte-identity regression coverage is in flux** because the old `tests/v2/test_audio_byte_identical.py` was deleted and a new `tests/test_audio_byte_identical.py` is untracked. That is a direct C7 confidence hit until stabilized.
2. **Recent “fixes” for BUG-116/117 and BUG-118 look directionally right, but they need one real end-to-end ledger-backed verification run before you should trust them.**
3. **Repo/worktree hygiene is poor right now**: lots of temp tailer scripts, moved tests, deleted schema/subprocess files, and untracked fixtures. For an internal branch that’s survivable, but it increases false positives and makes QA harder than it needs to be.
4. **The roadmap still lists some now-secondary items ahead of immediate run blockers** like verifying the restored audio palette, verifying byte-identical audio tests are actually still wired, and validating the ledger/title/filename lookup chain after the underscore-drift fix.

If you do only a few things before the next episode test run, they should be **QA hardening and acceptance checks**, not new feature work.

---

# 1) Priority misalignment: ROADMAP vs current open bug reality

## Misalignment A: ROADMAP still overweights future visual work vs immediate audio/run confidence

The roadmap’s “Open items after today’s session” prioritizes:

1. inspect critic calibration
2. maybe flip block_on_reject
3. tiny-label UX polish
4. fuzzy news dedup
5. empty-section pruning
6. schema bump / spine stamping
7. LLM matrix
8. VRAM measurement
9. GGUF unblock
10. cleanup deletion logic
11. canon auto-update

That ordering is not ideal for the next episode test run.

### Why
Your last few commits show active bugfixing around:
- **BUG-LOCAL-115** title resolution fallback
- **BUG-LOCAL-116 / 117** full audio palette restored
- **BUG-LOCAL-118** ledger lookup robust to filename underscore drift
- default audio source changes in `VideoComposite`

Those are **closer to run-blockers** than critic-threshold tuning or fuzzy news dedup.

### Recommendation
For the next run, the practical priority should be:

1. **Reconfirm C7 guardrail coverage is intact**  
2. **Verify BUG-116/117 actually restored all intended audio layers/sources**  
3. **Verify BUG-118 on a real ledger with renamed files**  
4. **Only then** inspect critic calibration / schema / roadmap polish

---

## Misalignment B: schema bump / Diff 3 is still too early relative to current branch churn

ROADMAP still carries **Diff 3 / schema bump** as a major next-session item. I think that is premature **for this branch state**.

### Why
You currently have:
- `otr_v2/schema/visual_plan.schema.json` deleted
- `subprocess_runners/__init__.py` deleted in one place, new `subprocess_runners/` untracked in another
- tests moved from `tests/v2/` into top-level `tests/`
- baseline fixtures moved and re-added as untracked files
- workflow and node behavior still changing around audio source selection and ledger path resolution

That’s not a good moment to add another schema bump unless it is absolutely required. It increases moving parts without helping the next run.

### Recommendation
**Defer schema bump until after one clean episode run** with:
- restored audio palette confirmed
- byte-identical test path stabilized
- ledger/title/audio-source lookup chain validated

---

## Misalignment C: “VRAM measurement runs” are too high in roadmap order

Given your explicit constraint set:
- single 5080 laptop
- no VRAM heroics
- no low-level optimization chasing
- “alarm plumbing only”

The roadmap item:
> “VRAM measurement runs via OTR_VRAMContextTest node to tune _MODEL_CONTEXT_CAPS”

is useful, but **not urgent** unless you’re seeing active OOMs or allocator instability.

### Recommendation
Demote this below:
- audio regression verification
- workflow acceptance checks
- test fixture stabilization

---

# 2) Brittle / redundant / out-of-sync design choices

## A. Audio byte-identical regression coverage is currently brittle

From `git status`:

- deleted: `tests/v2/test_audio_byte_identical.py`
- untracked: `tests/test_audio_byte_identical.py`
- deleted: `tests/v2/fixtures/baseline_v1.5.*`
- untracked: `tests/fixtures/baseline_v1.5.*`

This looks like a test relocation in progress, but **right now it weakens confidence**.

### Why brittle
If the branch is between old-path deletion and new-path adoption, then:
- CI / local commands documented in ROADMAP may still point at old paths
- developers may think C7 is covered when the test is merely untracked
- fixture path assumptions may be stale in scripts/docs

### Risk
This is the single biggest QA concern in your current status because **C7 is non-negotiable**.

### Recommendation
Treat this as a pre-run blocker:
- ensure the new test file is committed
- ensure the new fixture paths are committed
- update any scripts/docs still referencing `tests/v2/...`

---

## B. `VideoComposite` audio-source behavior is drifting and needs one source-of-truth check

Recent commits:
- `c28376d` VideoComposite: `humo_concat` is default audio_source
- `830b323` VideoComposite: humo_concat audio mode + bookend skip diagnostic

This is a sensible direction given your stated default path:
> HuMo native audio + master_mix slices, ffmpeg concat

But it also means the audio-source contract has changed recently.

### Why brittle
The project has historically had:
- procgen mp4 audio
- master mix slices
- HuMo native audio
- ffmpeg concat
- optional upscale / composite stages

That’s a lot of places for audio-source drift.

### What I’d worry about
- `VideoComposite` defaulting to `humo_concat` may be correct now, but if any workflow or node still assumes procgen audio is canonical, you can get silent mismatches.
- “bookend skip diagnostic” suggests there are conditional branches around opening/closing windows; those are exactly where accidental source switching can happen.

### Recommendation
Add/keep one explicit acceptance test:
- final mp4 audio stream source is what you think it is
- waveform/hash of final extracted audio matches expected concat source for a smoke fixture

Not byte-identical to v1.5 if video path differs—but at least deterministic and source-correct.

---

## C. Ledger/path/title resolution remains a fragile chain

Recent commits:
- `5392682` BUG-LOCAL-115 fix: third fallback for title resolution unblocks soak
- `ef9989c` BUG-LOCAL-118 fix: BatchHumoRender ledger lookup robust to filename underscore drift

These are both “fallback-on-fallback” style fixes. Necessary, but they also signal fragility.

### Why brittle
You now have multiple resolution chains for:
- title
- ledger path
- clip filename
- renamed output files
- workflow-generated vs ledger-canonical names

Every extra fallback helps soak survive, but also makes it harder to know which path is actually canonical.

### Recommendation
For the next run, log the chosen branch explicitly:
- title source used
- ledger path source used
- clip lookup source used (`ledger canonical`, `filename fallback`, etc.)

If those logs already exist, great—make them part of the acceptance checklist.

---

## D. `OTR_PostAudioVideoPipeline` is retired but still registered: acceptable, but confusing

Inventory still shows:
- `nodes/post_audio_video_pipeline.py`
- registered `OTR_PostAudioVideoPipeline`

ROADMAP says it is retired and kept for back-compat.

### My take
This is fine short-term, but it’s **out of sync with the current stack** and can confuse future QA or workflow audits.

### Recommendation
Not urgent before next run, but:
- rename title clearly to include `[RETIRED]` if not already
- ensure no live workflow references it
- maybe add a warning in execute()

---

## E. Repo hygiene is currently poor enough to hide real issues

There are dozens of untracked temp files:
- `.claude_tail_*`
- `.tailer_*`
- `_tail_*`
- `_scan_*`
- temp txt/json/ps1 files
- scratch outputs

### Why this matters
For internal QA, this creates noise:
- makes it harder to spot real new files
- increases chance of accidentally depending on a local helper
- obscures whether a “fix” is actually in tracked code

### Recommendation
Before the next serious run, do a cleanup pass or at least gitignore the recurring temp patterns.

This is not glamorous, but it’s high leverage.

---

# 3) Sanity-check of recent commits: did they actually close what they claim?

I can only partially verify from the evidence you gave. Where I can’t inspect code, I’ll say so.

---

## `ef9989c` — BUG-LOCAL-118 fix: BatchHumoRender ledger lookup robust to filename underscore drift

### Claimed fix
Robust ledger lookup when filename underscores drift between `SignalLostVideo` and ledger renamer.

### Likely status
**Plausible but not fully proven from provided material.**

### Why
You explicitly mention the bug and the commit, but I don’t see:
- updated BUG_LOG entry text for 118
- code excerpt from `nodes/batch_humo_render.py`
- test added for underscore drift

Given the branch status shows:
- `tests/test_wedge_probe.py` modified
- no obvious new `test_batch_humo_render_*` listed

I do **not** have evidence of a dedicated regression test for this exact bug.

### QA verdict
**Probably fixed in code, not yet convincingly closed in QA.**  
Needs one real ledger-backed smoke test with a deliberately underscore-drifted filename.

---

## `d6859ea` — BUG-LOCAL-116 + BUG-LOCAL-117 fixes: full audio palette restored

### Claimed fix
“full audio palette restored”

### Likely status
**High-value fix, but currently under-verified.**

### Why
This sounds like a behavior fix, not just a crash fix. Those are easy to “mostly fix” while still missing one layer:
- announcer
- music
- SFX
- Bark dialogue
- HuMo native audio / concat source
- bookends

I don’t see from your provided material:
- a new regression test specifically asserting all expected audio layers are present
- a fixture-based waveform/source check
- a BUG_LOG verification note

### QA verdict
**Do not consider this fully closed until one episode run confirms all intended buses/layers are audible and correctly sourced.**

---

## `5392682` — BUG-LOCAL-115 fix: third fallback for title resolution unblocks soak

### Claimed fix
Third fallback for title resolution.

### Likely status
**Probably closed enough for soak, but it’s a survivability fix, not a cleanliness fix.**

### Why
This is exactly the kind of fix that can unblock runs while leaving the architecture messy. That’s acceptable for alpha/internal QA.

### QA verdict
**Likely effective, but title resolution chain is still fragile and should be logged explicitly.**

---

## `c28376d` — VideoComposite: `humo_concat` is the default audio_source

### Claimed change
Default audio source switched.

### Likely status
**Code likely does what it says. Whether it is correct for all workflows is the real question.**

### QA verdict
This is not “closed” by existing evidence; it needs acceptance verification against the shipped default audio path.

---

## `830b323` — VideoComposite: humo_concat audio mode + bookend skip diagnostic

### Claimed change
Adds mode + diagnostic.

### Likely status
**Probably implemented, but diagnostics are not the same as correctness.**

### QA verdict
Useful instrumentation, but not sufficient proof of correctness.

---

## `2887ba2` — ScriptCritic: revise_on_findings defaults to ON

### Claimed change
Default behavior changed.

### QA verdict
This likely landed, but I’d question whether it’s the right default **before calibration is complete**. Advisory critic + automatic reviser can be helpful, but it also changes script generation behavior in subtle ways. Not a bug, but a policy choice that may be ahead of your validation.

---

## `fd02e37` — ScriptCritic: add Reviser pass + reroute audio chain through critic

### QA concern
This is a bigger architectural change than the commit message sounds like.

### Why
“reroute audio chain through critic” means the critic is now in the critical path for the script that feeds audio. That’s okay if stable, but it raises the cost of critic miscalibration.

### QA verdict
Feature likely landed, but **I would not prioritize flipping `block_on_reject=True` yet.**  
The roadmap already says to wait for 3–5 runs; I agree.

---

## `21aba38` — VideoComposite execute signature fix

### Claimed fix
Added `cleanup_clips_after_assembly` to execute signature.

### QA verdict
This one is very likely actually closed. The failure mode was concrete (`TypeError` from ComfyUI kwarg passing), and the fix is straightforward.

### Caveat
ROADMAP itself admits deletion logic is still no-op. So:
- **signature mismatch bug:** fixed
- **actual cleanup behavior:** not implemented

That distinction is correctly documented.

---

## `61a85b3` and `16294df` — BUG-LOCAL-112 news-history reset

### QA note
You have two commits with nearly identical messages:
- `61a85b3`
- `16294df`

That suggests either:
- duplicate commit / amended follow-up
- first attempt incomplete, second actual wire-up

### QA verdict
Not necessarily bad, but it’s a smell. I’d trust the later one more than the earlier one.

---

# 4) Specific brittle/redundant/out-of-sync files to watch

I can’t cite exact line numbers without file contents, but here are the hotspots by path.

## `nodes/batch_humo_render.py`
High-risk node right now because it sits at the intersection of:
- ledger lookup
- clip naming
- warmup pad
- resume behavior
- canonical path preference vs fallback

Given BUG-118 and earlier warmup-pad work, this is one of the top files to inspect before the run.

### What to verify
- prefers ledger canonical path before glob/mtime fallback
- logs which resolution path won
- resume logic doesn’t silently skip mismatched files
- underscore normalization doesn’t over-match wrong clips

---

## `nodes/video_composite.py`
Another top hotspot because recent commits changed audio-source defaults.

### What to verify
- default `audio_source` really is `humo_concat`
- opening/closing bookend logic does not accidentally drop or remap audio
- diagnostics are informative enough to debug source selection

---

## `nodes/story_orchestrator.py`
Still the giant risk surface. At 511 KB, this is a “works until it doesn’t” file.

### My candid view
This file is functionally central but architecturally brittle. You’ve done a lot of good defensive work in it, but it is still carrying too many responsibilities:
- title resolution
- script generation
- normalization
- critique/revise
- canon loading
- voice consistency checks
- ledger init
- likely more

### Recommendation
Not for this run, but medium-term: peel off pure helpers into smaller modules. Biggest payoff would be maintainability, not performance.

---

## `tests/test_audio_byte_identical.py`
Because it is untracked right now, this is a must-fix.

---

## `workflows/otr_videoplan_TEST_humo.json`
Modified in status. Since this is your practical smoke harness, any drift here matters more than roadmap prose.

### What to verify
- still points at the intended default audio path
- still exercises the current ledger/path assumptions
- not carrying stale widget values from old audio-source behavior

---

## `nodes/project_state.py`
Modified in status. Since BUG-LOCAL-027 was about `project_state` socket/widget drift historically, any current edits here deserve scrutiny.

---

## `nodes/kokoro_announcer.py`
Modified in status. Since announcer routing is part of the “full audio palette,” this file is directly relevant to BUG-116/117 verification.

---

# 5) Did anything claim a fix that current code may not actually contain?

I can’t prove a false claim from the snippets alone, but I can flag two areas where commit-message confidence exceeds current evidence.

## A. BUG-116/117 “full audio palette restored”
**Possible overclaim.**  
I believe the code probably changed, but I do not see enough evidence that the branch has a regression test or acceptance proof covering the whole palette. I would phrase this as:

> “likely addressed in code, not yet QA-closed”

## B. BUG-118 “robust to filename underscore drift”
**Possible overclaim.**  
Again, likely implemented, but I don’t see a dedicated regression test or acceptance note. Same verdict:

> “likely addressed in code, not yet QA-closed”

---

# 6) Recommended next 3–5 things before the next episode test run

Priority order, one-sentence justifications as requested.

1. **Commit and re-run the relocated `tests/test_audio_byte_identical.py` with its new `tests/fixtures/baseline_v1.5.*` paths.**  
   C7 is your hardest constraint, and right now the test/fixture move appears incomplete from `git status`.

2. **Run one targeted acceptance smoke for BUG-116/117 using the current default audio path (`HuMo native audio + master_mix slices, ffmpeg concat`) and explicitly verify every expected audio layer is present.**  
   “Full audio palette restored” is too important to trust from commit message alone.

3. **Run one ledger-backed `BatchHumoRender` smoke with intentionally underscore-drifted filenames to validate BUG-118 in reality, not just in code.**  
   This is a classic fix that can look robust until a real renamed file hits the fallback chain.

4. **Freeze and clean the branch’s test/fixture/workflow surface: commit the moved fixtures/tests, remove or gitignore the tailer/temp script clutter, and ensure docs/scripts no longer reference `tests/v2/...`.**  
   Small hygiene work here buys a disproportionate reduction in QA ambiguity.

5. **Do one end-to-end episode smoke and inspect the logs for explicit source-selection traces (title source, ledger path source, audio source, clip lookup source).**  
   The recent fixes are mostly fallback-chain fixes, so the fastest way to catch drift is to verify which branch actually won at runtime.

---

# Bottom line

If I were steering this branch for the next internal run, I would **not** spend the next session on schema bumping, critic threshold tuning, or VRAM cap refinement.

I would spend it on:
- **C7 regression coverage stabilization**
- **audio palette acceptance verification**
- **ledger/filename fallback verification**
- **repo hygiene enough to trust what’s actually under test**

That’s the smallest change set with the biggest payoff.

If you want, I can do a second pass focused just on:
1. likely hotspots in `nodes/video_composite.py`, `nodes/batch_humo_render.py`, and `nodes/kokoro_announcer.py`, or  
2. a concrete pre-run QA checklist you can execute in 15–20 minutes on Windows.

---

## Gemini (gemini-3-pro-preview)

**1. Core Agreement**
I AGREE with the core recommendation to halt feature work (like the Diff 3 schema bump) and immediately secure the C7 audio regression gate and repo hygiene, as the untracked test files represent a critical blind spot.

**2. Factual Errors in ChatGPT's Answer**
*   **Misdiagnosed Commit Duplication:** ChatGPT flagged the duplicate commit messages for `16294df` and `61a85b3` (`BUG-LOCAL-112: news-history reset...`) as a "smell", but missed the actual discrepancy: `ROADMAP.md` explicitly documents `16294df` as *"Wire ScriptCritic into workflows/otr_scifi_16gb_full.json"*. Your git log and roadmap are out of sync (likely due to a bad `git commit --amend` or rebase dropping a message).
*   **Misunderstanding ComfyUI Node Deprecation:** ChatGPT called keeping the retired `OTR_PostAudioVideoPipeline` registered "confusing." In ComfyUI, keeping deprecated nodes registered (with a `(Retired)` title, as you did) is the standard best practice; deleting the class entirely causes older workflow JSONs to throw red-box load errors. 

**3. Important Omissions**
*   **The Source of the Repo Clutter:** ChatGPT noted the poor repo hygiene but missed *why* it's happening. The dozens of `.claude_tail_*`, `.tailer_*`, and `*.tmp.txt` files are operational exhaust from your AI coding agent (Desktop Commander). You don't just need to "clean them up"—you need to update `.gitignore` immediately so the agent stops blinding your `git status`.
*   **The `ffmpeg` Re-encode Risk (C7):** Commits `c28376d` and `830b323` switched `VideoComposite` to `humo_concat` audio mode. If this new mode uses `ffmpeg` to concatenate audio streams rather than muxing the pre-assembled master mix with `-c:a copy`, you risk floating-point drift in the audio bytes, instantly violating Rule C7.
*   **VRAM Telemetry is NOT "Low Priority":** ChatGPT advised demoting VRAM measurement. On a 16GB laptop running Mistral + FLUX + HuMo, VRAM telemetry isn't "heroics", it's your only warning before a hard crash. Keep the `VRAMContextTest` node active and visible.

**4. My Short Recommendation**
*   **Secure the C7 Gate:** Run `git add tests/test_audio_byte_identical.py tests/fixtures/baseline_v1.5.*` immediately. Do not run another test or write another line of code until your byte-identical safety net is tracked in version control.
*   **Silence the AI Exhaust:** Add `.*tail*`, `*.tmp*`, and `*.ps1` (if they are just scratch scripts) to `.gitignore`. A clean `git status` is a requirement for a solo dev moving this fast.
*   **Audit `VideoComposite` for C7 Compliance:** Check the `ffmpeg` arguments in `nodes/video_composite.py` for the new `humo_concat` mode. Ensure it uses `-c:a copy` and does not re-encode the audio stream.
*   **Defer Diff 3 (Schema Bump):** Hold off on the `l3 -> l4` schema bump. You have too much unverified audio/ledger routing churn (BUG-115, 116, 117, 118) to introduce a data structure change right now.
*   **Run a "Dirty" Smoke Test:** Force an underscore-drifted filename in your test directory and run `BatchHumoRender` to prove BUG-118 actually falls back correctly before trusting it in a 10-hour overnight soak.

**5. Uncertainties to Verify**
*   I am uncertain if `humo_concat` actually re-encodes audio or just muxes it. If it re-encodes, C7 is broken.
*   I am uncertain why `otr_v2/schema/visual_plan.schema.json` was deleted but not replaced in the git status. Did it move, or is JSON validation currently disabled?
*   I am uncertain if the `tests/v2/` to `tests/` migration broke any hardcoded paths in your `pytest` CI scripts (like `.claude_tail_pytest.ps1`). Verify your test runner is actually finding the newly moved files.

---

## NVIDIA (mistralai/mistral-nemotron)

You are a senior systems architect advising a solo developer on a ComfyUI radio-drama generator (OTR "SIGNAL LOST") running on a single RTX 5080 Laptop / 16 GB VRAM Windows workstation. 100% local, no cloud. Audio output must remain byte-identical between runs (rule C7). VRAM ceiling is 14.5 GB. The owner does NOT want low-level VRAM optimization work (no weight streaming, no Flash Attention chasing). Prefer the smallest change with the largest payoff. Cite specific files / line numbers when relevant. Be candid; flag uncertainty rather than bluffing.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
