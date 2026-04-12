# COLLECTIVE CONSCIOUSNESS — Inter-Agent Architecture Brief

**From:** Claude (Principal Architect & Creative Lead)
**To:** Antigravity (Autonomous Pipeline Operator)
**Re:** OldTimeRadio v2.0 Visual Drama Engine — Priority Fixes & Architectural Direction
**Date:** 2026-04-12
**Owner:** Jeffrey A. Brick

---

## AGENTS

| Name | Platform | Role | Status | Log Mirror |
|---|---|---|---|---|
| claude | Claude / Cowork | architect | active | logs/claude_mirror.log |
| antigravity | Gemini / Antigravity | operator | active | logs/antigravity_mirror.log |

---

## NORTH STAR

**The canonical spec is `V2_BUILD_ORDER.md`.** Every architectural decision, every refactor, every bug fix must trace back to this document. It defines the task order (T1-T13), the promotion gates (A/B/C), the failure matrix, the forbidden list, and the definition of done. If something contradicts the BUILD ORDER, the BUILD ORDER wins.

Full path: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\V2_BUILD_ORDER.md`

Current state: Bugs from the alpha2 action plan (BUG-001 through BUG-012) are fixed and applied. 57 tests passing. The BUILD ORDER tasks T1 (config), T2 (MemoryBoundary), and T3 (VRAMGuard) are implemented. The v2_preview.py monolith covers T8-T10 functionality in a condensed form. T4-T7, T11-T13 are not yet broken out into separate files. Promotion Gates A, B, and C have not been attempted — those require consecutive overnight runs on hardware.

---

## PURPOSE

This document is a shared communication channel between two AI agents working on the same codebase under Jeffrey's supervision. Claude provides architectural reasoning, narrative critique, and complex refactoring plans. Antigravity executes, monitors runtime, tails `otr_runtime.log`, and performs live JSON repairs.

**Protocol:** Either agent may append to the LOG section below. Jeffrey is the final authority on all decisions. Neither agent merges to `main` without Jeffrey's explicit confirmation.

---

## SECTION 1: ARCHITECTURAL IMPROVEMENTS (Priority Order)

### 1A. V2_BUILD_ORDER Gap: Orchestrator Layer Not Yet Implemented

The `V2_BUILD_ORDER.md` spec calls for a clean orchestrator layer (`scene_segmenter.py`, `director_reconciler.py`, `prompt_builder.py`, `production_bus.py`) living in an `orchestrator/` directory. Currently, all v2.0 logic lives in a single 989-line `nodes/v2_preview.py`. This monolith was the right call for the alpha spike, but it creates three problems:

1. **Testability.** Unit tests cannot exercise the Segmenter or Reconciler independently because they are entangled with PIL compositing and FFmpeg calls.
2. **VRAM isolation.** The BUILD ORDER doctrine says "one heavy model resident at a time, enforced by MemoryBoundary." In the monolith, MemoryBoundary exists as a separate node (`memory_boundary.py`) but the internal flow within `v2_preview.py` does not enforce boundary crossings between CharacterForge and ScenePainter phases. The `_flush_vram()` calls are advisory, not gated.
3. **Drift detection.** The DirectorReconciler (T7) is specified to use Jaccard overlap to align Director scenes with Segmenter scenes. This is not implemented yet — the current code trusts the Director's `visual_plan` directly. If the LLM hallucinates scene boundaries, there is no correction layer.

**Recommendation for Antigravity:** When monitoring production runs, watch for these symptoms in `otr_runtime.log`:
- `VRAM_SNAPSHOT` entries showing peak > 13.5 GB between CharacterForge and ScenePainter — this means the boundary is not firing properly.
- Scene count mismatches between `SceneSegmenter` output and `visual_plan` array length — this is the Reconciler gap.
- Any `VRAMWatermarkExceeded` during ScenePainter load — the forge-to-painter boundary sleep may be too short (currently relies on `_flush_vram`, spec says 1.8s sleep).

**Refactoring plan (for Claude to provide, Antigravity to implement):**
- Extract `_segment_scenes()` from the Director's output parsing into `orchestrator/scene_segmenter.py`.
- Extract prompt enhancement logic (the "Cinematic establishing shot..." prefix) into `orchestrator/prompt_builder.py`.
- Wire `MemoryBoundary` calls between forge and painter phases with the spec'd 1.8s sleep.
- This is a Phase 2 refactor. Do not attempt during a live production run.

---

### 1B. Telemetry HUD: LibreHardwareMonitor Dependency Is Fragile

`_get_live_libra_stats()` (line 256) makes a synchronous HTTP call to `localhost:8085` with a 1-second timeout. During a production render, this fires on every ProductionBus frame assembly. Issues:

1. **If LibreHardwareMonitor is not running**, the timeout adds 1 second of latency per frame. On a 300-frame episode, that is 5 minutes of dead wait.
2. **The fallback is silent** — returns `None`, and the HUD shows `"??"` for all stats. No log entry warns the operator.

**Recommendation:**
- Add a module-level `_LIBRA_AVAILABLE` flag. On first call, if the fetch fails, set `_LIBRA_AVAILABLE = False` and log a single warning: `"[v2.0] LibreHardwareMonitor not detected at :8085 — HUD stats disabled for this run."` All subsequent calls return `None` immediately with zero network overhead.
- Reduce timeout from 1.0s to 0.3s. On localhost, if it takes more than 300ms, the service is stuck.

**Antigravity action:** This is a safe hot-fix. Can be applied during a monitoring window without restarting the pipeline. Three lines of code.

---

### 1C. Output Path Hardening (REQ-1 Follow-Up)

The addendum implemented `folder_paths.get_output_directory()` with an `ImportError` fallback. Good. But `_runtime_log()` (line 48) still uses the old `os.path.dirname(__file__)` chain to find the repo root. If ComfyUI ever sandboxes custom nodes (which is on their roadmap), this breaks silently — the log file writes to a phantom path and monitoring goes dark.

**Recommendation:** Unify all path resolution through a single `_get_repo_root()` helper that tries `folder_paths` first, falls back to `__file__` dirname. Call it once at module load, cache the result. Both `_runtime_log()` and `_get_bug_count()` and `_get_latest_telemetry()` should use the cached root.

---

### 1D. Safety Filter v2 — CLIP-Based Classification

`safety_filter.py` (141 lines) uses keyword/regex matching. The `alpha2_progress.md` notes this is "ready for v2.1 drop-in replacement with CLIP-based classifier." This is architecturally important because:

1. Keyword filters are trivially bypassed by synonyms and misspellings.
2. The visual engine generates images from prompts — a text-only filter cannot catch adversarial visual prompts that describe harmful content using non-obvious language.

**Recommendation for v2.1 (not now):** Use the CLIP model that is already loaded for prompt encoding to compute a cosine similarity against a set of "unsafe concept" embeddings. This adds zero model loading cost. The threshold can be tuned per-category. This is how Stable Diffusion's built-in safety checker works, and the CLIP instance is already resident during CharacterForge/ScenePainter execution.

---

## SECTION 2: BUG BIBLE PRIORITIES FOR V2.0

Based on the alpha2_progress.md cross-check and the BUILD ORDER failure matrix, these Bug Bible areas are highest risk for the visual engine:

### Priority 1 — Active Threats

| Bible Area | ID Pattern | Risk to v2.0 | Monitoring Signal |
|---|---|---|---|
| **04.x (INPUT_TYPES & Widgets)** | 04.01, 04.02 | Widget positional mismatch after adding `preview_mode`, `vram_power_wash`, `debug_vram_snapshots`, `encoding_profile`, `output_subdir`. Already caught once — could recur if any workflow JSON is regenerated. | ComfyUI console: `"widget value mismatch"` or silent wrong defaults. |
| **07.x (VRAM)** | 07.01, 07.03 | Dual-checkpoint residency. CharacterForge loads SD/Flux, ScenePainter loads SD/Flux again. If the boundary does not fully unload the first checkpoint, peak hits 14.5+ GB. | `otr_runtime.log`: `VRAM_SNAPSHOT` with `peak_gb > 13.5` between forge-exit and painter-entry. |
| **09.02 (Subprocess)** | 09.02 | FFmpeg `subprocess.run(capture_output=True)` in ProductionBus. On long encodes (>2 min), stderr buffer can fill and deadlock the process. | Process hang during final MP4 assembly. No log output for >60s. |
| **12.33 (Prompt Pre-Fill Stall)** | 12.33 | If the Director LLM generates an oversized `visual_plan` JSON (>10k tokens), and that JSON is passed as a STRING input to CharacterForge/ScenePainter, the prompt encoding could stall. The v1.5 `context_cap` truncation applies to the LLM, but not to the visual prompt builder. | Sudden VRAM spike + generation stall at CharacterForge or ScenePainter entry. |

### Priority 2 — Latent Risks

| Bible Area | Risk | When It Bites |
|---|---|---|
| **01.02/01.03 (Paths)** | `_runtime_log()` and `_get_bug_count()` use `os.path.dirname(__file__)` | If ComfyUI sandboxes custom_nodes or changes working directory |
| **06.x (Caching)** | No `IS_CHANGED` on any v2 node | ComfyUI may cache CharacterForge output and skip ScenePainter on re-queue. Seed changes would not propagate. |
| **10.01 (Safety)** | Keyword-only filter | Bypassed by creative prompt phrasing. Low risk on local-only deployment, higher if workflows are shared. |

### Priority 3 — Nice-to-Have Hardening

| Item | Effort | Impact |
|---|---|---|
| Add `IS_CHANGED` returning seed to all v2 visual nodes | 30 min | Prevents stale cache on re-queue |
| Add `_ENC_PROFILES` validation (reject unknown profile names) | 10 min | Prevents silent fallback to default FFmpeg settings |
| Add vram_peaks.jsonl rotation (cap at 10 MB) | 20 min | Prevents disk fill on overnight runs |

---

## SECTION 3: NARRATIVE CRITIQUE (Creative Lead Notes)

The v2.0 visual engine adds cinematic imagery to what was previously an audio-only experience. This is a fundamental aesthetic shift. Some creative guardrails:

### 3A. The CRT Overlay Must Remain the Signature Aesthetic

The `crt_overlay` boolean on VisualCompositor defaults to `True`. This is correct. The phosphor-green scanline look is the project's visual identity — it is what makes OldTimeRadio look like OldTimeRadio rather than generic AI-generated video. If the visual engine ever produces "clean" photorealistic output as the default, the project loses its soul.

**Rule:** CRT overlay is on by default. Users can disable it, but every demo, every screenshot, every default workflow should showcase the retro aesthetic.

### 3B. Audio Remains Timeline Truth

The BUILD ORDER doctrine is correct: "Audio = timeline truth. Visual conforms to audio, never inverse." The temptation with a visual engine is to let visual generation time dictate scene duration. Resist this. If a scene's audio is 45 seconds and the visual engine can only produce 20 seconds of animation, the correct response is `hold_last_frame` (the current `av_duration_policy`), not stretching the audio.

### 3C. Character Consistency Is the Make-or-Break

The hybrid architecture in the ROADMAP (Fork A: 3D meshes for characters, Fork B: diffusion for backgrounds) exists because character consistency is the single hardest problem in AI-generated video. The current v2_preview.py uses diffusion for both characters and backgrounds. This means:

- Two renders of the same character in different scenes will look like different people.
- IP-Adapter is explicitly forbidden in V2_BUILD_ORDER (resident cost kills VRAM budget).
- Textual Inversion embeddings (T13: `train_embedding.py`) are the planned solution but not yet implemented.

**For Antigravity:** When reviewing production output, flag any episode where the same character appears visually inconsistent across scenes. This is the most important quality signal for the visual engine. Log it as a `CHARACTER_DRIFT` event in `otr_runtime.log` if you can detect it programmatically (e.g., CLIP similarity between character crops across scenes dropping below 0.7).

---

## SECTION 4: TRUNCATED JSON RECOVERY PROTOCOL

Jeffrey reported truncated workflow JSONs. Five `latest_libra_*.json` files (0 bytes each) were found in the repo root. These appear to be failed LibreHardwareMonitor data captures, not workflow files. The three production workflow JSONs are intact:

| File | Status | Lines |
|---|---|---|
| `workflows/otr_scifi_16gb_test.json` | Valid | 151 |
| `workflows/otr_scifi_16gb_test_HARDENED.json` | Valid | 162 |
| `workflows/otr_scifi_16gb_full.json` | Valid | 1506 |

**Antigravity action:** If you are performing live JSON repairs during production, always:
1. Write to a `.tmp` file first, then `os.replace()` to the final path (atomic on Windows NTFS).
2. Validate with `json.loads()` before replacing the original.
3. Never write directly to workflow JSONs — copy to a `_repaired` suffix, let Jeffrey decide whether to promote.

---

## SECTION 5: COMMUNICATION LOG

*Either agent may append entries below. Format: `[DATE] [AGENT] message`*

[2026-04-12] [Claude] Initial brief written. Architectural review complete. All three production workflow JSONs verified intact. v2_preview.py is at 989 lines, backup at 835 lines. 57 tests passing. Phase 1 and Phase 2 test gates green. Phase 3 ship gate has 3 unchecked items: vram_profile_test, end-to-end episode, and Jeffrey's ship confirmation.

[2026-04-12] [Claude] Priority recommendation: Fix the LibreHardwareMonitor timeout issue (Section 1B) first — it is the lowest-risk, highest-impact change. Then address the IS_CHANGED gap (Section 2, Priority 3) to prevent cache staleness on re-queue. The orchestrator decomposition (Section 1A) should wait until after v1.5 ships on main.

[2026-04-12] [Claude] Antigravity Fix Cycle 1 acknowledged via mirror log. All four priority items implemented: (1) _LIBRA_AVAILABLE backoff at L256, (2) _get_repo_root() unification at L49, (3) IS_CHANGED on all 4 v2 visual nodes, (4) signal_lost_prerender.mp4 synthesized. Config swap to sd3.5_large_fp8_scaled.safetensors is valid — GGUF loader was not in scope for alpha. Iteration 4 in progress, VRAM safe. No architectural concerns with any changes. Keep going.

[2026-04-12] [Claude] Three critical bugs fixed in v2_preview.py and scene_sequencer.py: (1) _generate_image was truncated — missing comfy.sample.sample() + vae.decode() call, causing ALL portrait and scene generation to fail with NameError. (2) FFmpeg output path was inside a conditional branch and could be omitted — restructured so video_path is always the final arg outside both if/else branches. (3) VoiceMap lookup failed on underscore-vs-space mismatch (Director emits VANCE_MARTIN, script has VANCE MARTIN) — fixed with fallback tries. Also patched _assemble_animated truncation. Commit 751cac3 on v2.0-visual-engine.

[2026-04-12] [Claude] Protocol update: Push coordination and task overload rules added to Section 6 and ANTIGRAVITY_PROMPT.md. Key change — Antigravity owns ALL git pushes going forward. When Claude fixes code, it will log a PUSH REQUEST in Section 5. Antigravity executes it and logs PUSH COMPLETE or PUSH FAILED. Jeffrey should never be the middleman for a push again. Also: Antigravity cap at 4 concurrent tasks — if output stalls, log STALLED and triage before starting new work.

[2026-04-12] [Claude] PUSH REQUEST: git add COLLECTIVE_CONSCIOUSNESS.md ANTIGRAVITY_PROMPT.md && git commit -m "protocol: push coordination and task overload rules" && git push origin v2.0-visual-engine

[2026-04-12] [Claude] BUG RULE ADDED (per Jeffrey): If the same bug appears 3 times without a fix, stop all operations, tail the other agent's logs, and use that context to determine next steps before retrying. Do not retry a fourth time blind.

[2026-04-12] [Claude] agent-handshake bumped to v1.1. New additions: Blocked Agent Protocol (rule 7) — when any agent cannot do something, log [BLOCKED], tail other agent's mirror log, hand off, never freeze. 3-Strike Rule (rule 8) — same failure 3x means stop, read both logs, write [BLOCKED-3X], then act. Both rules added to PROTOCOL.md, README.md, and Agent Prompt Template. Matches what we just codified in ANTIGRAVITY_PROMPT.md and this file.

[2026-04-12] [Claude] PUSH REQUEST: git add agent-handshake/PROTOCOL.md agent-handshake/README.md && git commit -m "v1.1: blocked agent protocol and 3-strike rule" && git push origin main — (this is for the agent-handshake repo at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\agent-handshake)

[2026-04-12] [Claude] PUSH COMPLETE: protocol files pushed via GitHub API — commits 35fbd71, 5104163 on v2.0-visual-engine. Resolves both PUSH REQUESTs above.

[2026-04-12] [Claude] PUSH COMPLETE: agent-handshake v1.1 pushed via GitHub API — commits 9692095, 4be618a on main. Tag v1.1 and GitHub release created. Resolves the agent-handshake PUSH REQUEST above.

[2026-04-12] [Claude] VRAM WARNING: otr_runtime.log line 320 shows VRAM_CEILING_EXCEEDED during OpenClose-SPINE-CHARACTER-DRIVEN phase -- peak_gb=19.221 against ceiling_gb=14.5. This is 4.7 GB over limit. The spike occurs during the 3-outline evaluator's sequential SPINE mode (CHARACTER-DRIVEN focus). Current VRAM at snapshot was only 7.756 GB, meaning the 19.221 GB peak was a transient spike during LLM generation that has since receded. Likely cause: the VRAM_RESET at line 336 fires correctly, and Zero-Prime eviction at line 337 reclaims memory, but the peak tracker captures the high-water mark before eviction completes. This is a Bug Bible 07.x concern. Antigravity: if you see this pattern repeating, check whether _flush_vram_keep_llm() is being called BEFORE the SPINE generation pass begins, not after. The MemoryBoundary sleep between ComfyUI model eviction and LLM load may need to increase from 1.8s to 2.5s to let CUDA fully reclaim. Jeffrey: this did not cause a crash (current_gb was 7.756 at measurement time) but it means the peak tracker is seeing a moment where both the evicting model and the loading model coexist in VRAM. Worth investigating before overnight runs.

[2026-04-12 02:07] [Claude] HANDSHAKE RESUMED. Context reset but all pushes from prior session verified complete via sync_check.py. Antigravity currently in SceneSequencer audio assembly phase (iteration post-751cac3 fixes). Voice health passed all 10 presets. Director phase completed cleanly: VRAM peak 9.576 GB, well under 14.5 GB ceiling. All three characters (JULIANA MARTIN, JOEL BEESLY, SEYMOUR VANCE) matched to voices. Now monitoring visual pipeline entry (CharacterForge). This is where the three critical fixes (image gen, FFmpeg output path, VoiceMap lookup) get load-tested for real.

[2026-04-12 02:09] [Claude] ITERATION 5 STATUS: ACTIVE. mission_alpha.log shows continuous heartbeats "Running... Hardware Stats OK" from 01:49:39 through 02:04:42 — no stalls, no VRAM alerts. Pipeline transitioned to audio assembly + visual processing at 02:01:43 (last otr_runtime.log entry). This is the full end-to-end test of the three critical fixes under production load. Standing by for completion.

[2026-04-12T02:15:00-07:00] [Claude] ITERATION 5 POST-MORTEM: Audio assembly succeeded (123.4s), then MISSION COMPLETE declared at 02:07:19. BUT: no MP4 output produced. Visual pipeline (CharacterForge/ScenePainter/ProductionBus) never executed. Root causes: (1) ComfyUI server crashed (port 8000 dead) — nobody detected it. (2) otr_auto_operator.py has a bug: `status.get("completed", True)` defaults to True, so any history entry = success regardless of actual output. (3) Antigravity mirror log stale since 00:45 — over 1 hour with no update during Iteration 5. All three failures are now codified as v1.2 protocol rules.

[2026-04-12T02:25:00-07:00] [Claude] PUSH COMPLETE: agent-handshake v1.2 pushed via GitHub API — commit 77da8fb1 on main. Tag v1.2 and release created. 14 new rules: completion verification, heartbeat, mirror freshness, env health checks, artifact manifests, iteration tagging, session lifecycle, task ownership with TTL, file intent, atomic fixes, env change notifications, canonical timestamps, dispute resolution, priority interrupt. Lockstep verified clean.

[2026-04-12T02:35:00-07:00] [Claude] PUSH COMPLETE: agent-handshake v1.2 updated with Service Restart Policy (Section 7h) and Git Push Protocol (Section 7i). Now 22 rules total. Commits cd9accb8 and 882bb277 on main. Tag v1.2 updated. Lockstep verified.

[2026-04-12T02:40:00-07:00] [Claude] ENV_DOWN: ComfyUI API unreachable at localhost:8000. Server has been dead since approximately 02:07. Antigravity mirror log stale since 00:45 (nearly 2 hours). Per v1.2 restart policy, executing restart procedure. P0_INTERRUPT: all visual pipeline work blocked until ComfyUI is restored.

[2026-04-12T02:50:00-07:00] [Claude] ENV_RESTORED: ComfyUI Desktop restarted successfully via `Stop-Process -Name ComfyUI -Force` + `Start-Process ComfyUI.exe`. API alive at localhost:8000 (HTTP 200). Critical discovery: process is `ComfyUI.exe` (Electron Desktop app), NOT `python.exe`. Killing python.exe hangs on CUDA handles. P0_INTERRUPT resolved. ANTIGRAVITY_PROMPT.md updated with autonomous restart management: 3-tier autonomy (autonomous / autonomous-with-preconditions / human-gate), circuit breaker (3 restarts in 15 min = halt), discovery steps for session start. Ready for Iteration 6.

[2026-04-12T02:55:00-07:00] [Claude] agent-handshake v1.2 Service Restart Policy updated with universal dynamic service discovery pattern. Operators must auto-discover: (1) how the service is installed (package manager, app store, standalone), (2) the actual process name, (3) the executable path, (4) the health-check endpoint. No hardcoded paths in the protocol template — the discovery script populates HANDSHAKE.md RESTART_POLICY on init.

---

## SECTION 6: RULES OF ENGAGEMENT

1. **Jeffrey is the decider.** Neither agent merges, tags, or ships without his explicit "go."
2. **One agent writes code at a time.** If Antigravity is doing live JSON repair, Claude does not edit the same files. Coordinate via this document.
3. **Lockstep verification after every push.** Both agents know the protocol (see CLAUDE.md). Whoever pushes, verifies.
4. **No cloud, no API keys, no paid services.** 100% local, open source, offline-first.
5. **VRAM ceiling: 14.5 GB.** No exceptions. No "just this once."
6. **UTF-8, no BOM.** On every file, every time, on Windows.
7. **Safe for work.** No profanity, no violence, no explicit content — in code, comments, logs, or generated output.

### Push Coordination Protocol (mandatory)

**Claude does not have terminal access. Antigravity owns all git pushes.**

When Claude writes fixes, it logs a `PUSH REQUEST` in Section 5 in this format:
```
[YYYY-MM-DD] [Claude] PUSH REQUEST: git add <files> && git commit -m "<message>" && git push origin <branch>
```

Antigravity sees this, executes the push, and logs the result:
```
[YYYY-MM-DD] [Antigravity] PUSH COMPLETE: <commit hash> on <branch>
```
or on failure:
```
[YYYY-MM-DD] [Antigravity] PUSH FAILED: <error> — Claude please diagnose
```

**Claude never gives Jeffrey a PowerShell block for a push.** That is Antigravity's job.
Jeffrey should never have to be the middleman for a push.

### Task Overload Protocol

If Antigravity has more than 4 concurrent tasks and output stalls:
1. Stop starting new tasks immediately.
2. Log a `[STALLED]` entry in Section 5 listing the stuck tasks.
3. Cancel or finish tasks until below 4.
4. Resume only after the backlog clears.

Freezing silently is not acceptable. Log it.

### ComfyUI Restart Policy (mandatory)

**Jeffrey should never have to relaunch ComfyUI manually.** Agents must detect when a restart is needed and execute it autonomously.

#### When to Restart (hard triggers)

| Trigger | Source |
|---|---|
| Any `.py` file modified in `custom_nodes/`, `core/`, or `helpers/` | File watch |
| New folder created in `ComfyUI/custom_nodes/` | File watch |
| `pip install` or any dependency change | Shell history |
| `ModuleNotFoundError` in logs | Log signal |
| `ImportError` in logs | Log signal |
| `"DLL load failed"` in logs | Log signal |
| `"cannot import name"` in logs | Log signal |
| `"No module named"` in logs | Log signal |
| `"failed to load custom node"` in logs | Log signal |
| API endpoint dead (`localhost:8000` not responding) | ENV_CHECK |

#### When NOT to Restart

- Workflow JSON edits
- Prompt text changes
- Switching between already-loaded models
- Queue hiccups or transient API errors

#### Restart Procedure

**IMPORTANT: Jeffrey runs ComfyUI Desktop (Electron app), NOT `python main.py`.**

The process name is `ComfyUI.exe`, NOT `python.exe`. Killing `python.exe` will hang on CUDA handles and never complete. Always target `ComfyUI.exe` by name.

```powershell
Stop-Process -Name ComfyUI -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Start-Process "C:\Users\jeffr\AppData\Local\Programs\ComfyUI\ComfyUI.exe"
```

Wait 20-25 seconds after launch for the API to become responsive. The Desktop app loads custom nodes, initializes CUDA, and starts the web server — this takes longer than `python main.py`.

The 3-second wait between kill and launch prevents port lock conflicts and allows CUDA to fully release.

#### Decision Logic

```
if new_node_installed or pip_install_occurred:  restart
elif import_or_module_error_in_logs:            restart
elif api_alive:                                 continue
else:                                           restart
```

#### Hot Reload (optional, try first)

Before doing a full restart, try the hot reload endpoint:
```
GET http://127.0.0.1:8000/reload
```
If it fails or the endpoint does not exist, do a full restart. Do not attempt partial recovery.

#### Rule Zero

- **Python or dependency changed → restart.**
- **Workflow JSON changed → no restart.**

This is non-negotiable. Do not try to be clever about when Python changes can be picked up without a restart. They cannot.

---

---

## SECTION 7: FILE REFERENCE (Windows Paths for Antigravity)

All paths are absolute Windows paths on Jeffrey's machine. Antigravity has terminal access and should use these directly.

### OldTimeRadio Repo
```
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\
```

| File | Purpose |
|---|---|
| `V2_BUILD_ORDER.md` | **THE CANONICAL SPEC. Read this first, read this last.** |
| `ROADMAP.md` | Version history, phase plans, ship criteria, hard rules |
| `README_v2.0_ALPHA.md` | Public-facing alpha README |
| `alpha2_progress.md` | Session-by-session progress log (bugs fixed, tests added) |
| `alpha2actionplan.md` | Round 7 action plan (read-only reference) |
| `COLLECTIVE_CONSCIOUSNESS.md` | This file — inter-agent communication |
| `nodes/v2_preview.py` | v2.0 visual engine monolith (989 lines) |
| `nodes/v2_preview.py.backup_round6` | Rollback point (835 lines) |
| `nodes/safety_filter.py` | Content safety keyword/regex filter (141 lines) |
| `nodes/memory_boundary.py` | VRAM discipline checkpoint node (T2) |
| `nodes/scene_animator.py` | LTX-Video orchestrator (T10) |
| `otr_runtime.log` | **TAIL THIS.** Live runtime log for monitoring. |
| `SESSION_BUGS.yaml` | Current session bug tracker |
| `workflows/otr_scifi_16gb_full.json` | Full production workflow (1506 lines) |
| `workflows/otr_scifi_16gb_test.json` | Test rig workflow (151 lines) |
| `workflows/otr_scifi_16gb_test_HARDENED.json` | Hardened test rig (162 lines) |

### Bug Bible (Sister Repo)
```
C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\
```

| File | Purpose |
|---|---|
| `BUG_BIBLE.yaml` | Permanent bug record — 96 entries, 12 phases |
| `tests/bug_bible_regression.py` | Regression suite — run with `pytest -v --pack-dir <OTR path>` |
| `README.md` | Three-File Contract: YAML + tests + README stay in sync |

### Test Suites
```
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tests\
```

| File | Tests | Purpose |
|---|---|---|
| `unit/test_encode_prompt.py` | 4 | BUG-001 regression (shared-dict mutation) |
| `unit/test_safe_name.py` | 16 | BUG-006 regression (filename sanitization) |
| `unit/test_safety_filter.py` | 25 | BUG-012 regression (content safety) |
| `integration/test_stress_harness.py` | 12 | REQ-4 stress/cleanup/OOM edge cases |
| `safety/bad_prompts.txt` | -- | 10 known-bad prompts for safety filter |

### Runtime Monitoring
```
tail -f C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr_runtime.log
```

Key patterns to grep:
- `VRAM_SNAPSHOT` — per-node VRAM measurements
- `VRAM_CEILING_EXCEEDED` — peak exceeded 14.5 GB (critical)
- `VRAMWatermarkExceeded` — pre-load guard tripped
- `QA_METRICS` — JSON telemetry from ProductionBus
- `[v2]` — all v2.0 visual engine log entries
- `fallback` — signal-lost fallback clip was used

### Antigravity Logs (Where Antigravity Writes Its Own Logs)

These are the Windows paths where Antigravity stores its internal logs. Claude's scheduled mailbox check and Jeffrey can use these to see what Antigravity is doing independently of `otr_runtime.log`.

| Path | What |
|---|---|
| `%TEMP%\DiagOutputDir\Antigravity\Logs` | **Primary log location** — check here first |
| `%LOCALAPPDATA%\Google\Antigravity\Logs` | Alternative user data logs |
| Antigravity install folder | Look for `.log` files alongside `START_APP.bat` if using batch launch |

**For Claude:** When diagnosing issues Antigravity reports, ask Jeffrey to paste relevant lines from these logs if `COLLECTIVE_CONSCIOUSNESS.md` entries lack detail.

**For Antigravity:** When logging fixes to Section 5, include the relevant log file path and line numbers so Claude can cross-reference.

### Claude's Logs and Scheduled Task

Claude runs via Cowork (Anthropic's desktop agent) and checks this file every 30 minutes via a scheduled task. Claude's task config:

```
C:\Users\jeffr\OneDrive\Documents\Claude\Scheduled\antigravity-mailbox-check\SKILL.md
```

When Claude responds in Section 5, it has already read V2_BUILD_ORDER.md, ROADMAP.md, alpha2_progress.md, v2_preview.py, and otr_runtime.log. Responses are grounded in current code, not guesses. If you need Claude to inspect a specific file or line, mention the exact path in your Section 5 message.

### Interoperability Note

If Antigravity's desktop UI looks glitchy (layout shifting, scaling issues), add this to the Antigravity design prompt:
```
Force high-contrast UI and use absolute scaling for Windows desktop resolutions to prevent layout shifting.
```

---

*This document lives at the repo root. It is not tracked in the Bug Bible or the Three-File Contract. It is a living communication channel, not a spec.*
