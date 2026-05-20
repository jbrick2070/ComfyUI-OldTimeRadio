# Round-robin -- Sprint H bug-hunt next steps (post-overnight failure)

**Branch:** v2.0-alpha @ dc0774b (uncommitted local patch + workflow normalize)
**Date:** 2026-05-17 14:00 PT
**Author:** Jeffrey Brick (operator) + Claude Cowork (session)
**Reviewers:** ChatGPT (gpt-4.1), Gemini (gemini-2.5-pro)
**Synthesis:** Claude
**Format:** lean -- 250 lines or under in final synthesis
**Constraints:** ASCII only, no em-dashes (use `--`), no "dummy" word, default-config audio C7 byte-identity is non-negotiable

---

## 1. The question, in one paragraph

Sprint H pivoted from a prompt-tester gate to a headless whole-workflow bug-hunt: run `workflows/otr_scifi_16gb_full.json` end-to-end via ComfyUI's HTTP API, tail logs, fix bugs in a loop until 3 consecutive clean runs. Yesterday evening caught 7 distinct bugs and fixed 6 in three iters; the seventh (Mistral-Nemo OOM with `Currently allocated: 29.97 GiB` on a 16 GB card) was diagnosed as a model-loader orphan after `load_llm` raises post-`from_pretrained`. A patch was applied to `nodes/_otr_model_loader.py request_slot()` wrapping `load_llm` in try/except + `unload_llm()` on failure. An overnight unattended loop (`scripts/overnight_bug_hunt.py`, 12 iters) was launched detached at 03:37 AM to validate the patch -- **and self-killed within 3 seconds because `kill_all_python.bat` killed the supervisor's own python process at iter-1 pre-flight.** 0 iters ran. No verification of the OOM patch. Operator is back, asking for round-robin guidance on how to recover and rearchitect the bug-hunt loop so this class of self-inflicted failure can't recur.

---

## 2. Session timeline (factual, lead in for the consultants)

### 2.1 Session start
- Branch consolidation: v2.0-alpha forced onto sprint-e-distillation-roundup tip + H1 outline rewrite cherry-picked on top (commit dc0774b)
- 8 docs-only commits from triage-sprint-c-retrospective-2026-05-15 + sprint-c-story-brief-v2 absorbed via `git checkout <branch> -- <path>` rollup commit
- Regression: 2370 passed / 20 skipped / 0 failed + Bug Bible 23/1/2
- All 12 prompt entrypoint modules confirmed importable in plain venv
- §0 verification of `artokun/comfyui-mcp` package: real, MIT, current npm 0.1.6, package name `comfyui-mcp` (no scope)

### 2.2 Iter 0 -- baseline
- ComfyUI Desktop NOT running. Launched headless via adapted `start_comfy_h0_baseline.bat` (Blackwell flags: `--port 8000 --highvram --force-fp16 --cuda-malloc --user-directory <docs>`)
- HF_HOME set explicitly in bat after iter 2 surfaced env-inheritance bug
- 1561 ComfyUI nodes loaded, 34 OTR_* classes present
- VRAM idle 1.24 GB / 15.92 GB

### 2.3 Iters 1-3 -- bugs caught
| # | Bug | Class | Site | Fix |
|---|-----|-------|------|-----|
| 1 | Plan's `main.py` path wrong | Environment | start command | Used existing `start_comfy.bat` paths |
| 2 | Workflow widget drift (orphan `"fixed"` on node 1) | Graph wiring | `workflows/otr_scifi_16gb_full.json` node 1 | Removed `"fixed"` at index 4 |
| 3 | `scripts/queue_smoke.py` references retired `target_length` widget | Stale tooling | `queue_smoke.py:55` | Replaced with `act_count=1` |
| 4 | All LLM models `[NOT DOWNLOADED]` | Environment | HF_HOME not inherited by ComfyUI child | Set HF_HOME explicitly in start bat |
| 5 | Node 12 (SignalLostVideo) `fps="[]"`, `resolution="[]"` | Graph wiring | widgets_values stripped-mode misalignment | Manual rewrite to `[24, '1920x1080', '']` |
| 6 | Node 55 (BatchLTXRender) `seed=""`, `clip_length="fixed"` | Graph wiring | same | Manual rewrite to `[1, 'ffmpeg', '', 22.0]` |
| 7 | Mistral-Nemo OOM `Currently allocated 29.97 GiB` | Model lifecycle | `_otr_model_loader.request_slot()` orphan after raise | Patch applied (unverified) |

Plus one workflow-wide drift sweep: `scripts/normalize_workflow_widgets.py` realigned 7 nodes' widget arrays against live `/object_info`.

### 2.4 Iter 3 (post-OOM-patch retry)
- Did NOT run -- overnight supervisor self-killed at 03:37 AM before iter 1 fired

### 2.5 Overnight supervisor design (the failure)
- `scripts/overnight_bug_hunt.py` launched detached via PowerShell Start-Process at 03:37:07
- Iter loop calls `scripts/kill_all_python.bat` as pre-flight cleanup
- `kill_all_python.bat` uses `Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"` and stops every match
- This INCLUDES the supervisor's own python.exe (PID 51560 was overnight_bug_hunt.py running)
- Supervisor died mid-pre-flight, 3 log lines written, no iters
- 10 hours 20 minutes of idle GPU before operator returned

---

## 3. The OOM patch (unverified)

In `nodes/_otr_model_loader.py request_slot()`, the call site around line 741 was:

```python
cache_entry = load_llm(normalized, context_cap=ctx_verdict.value)
LLM_CACHE["model_id"] = normalized
LLM_CACHE["slot"] = slot
LLM_CACHE["cache_entry"] = cache_entry
```

If `load_llm` raises AFTER `AutoModelForCausalLM.from_pretrained` succeeds (BNB quant step, warmup pass, BUG-098 tripwire, accelerate device-map conflict), the orphan ~10 GB of Mistral weights stay resident on GPU AND `LLM_CACHE` never gets populated. The next retry inside `_otr_style_picker._run_inventor`'s 3-attempt loop cache-misses, loads a SECOND copy on top, OOMs at 20+ GB.

Patch wraps the load in try/except, calls `unload_llm()` on any exception, re-raises:

```python
try:
    cache_entry = load_llm(normalized, context_cap=ctx_verdict.value)
except Exception:
    log.warning("[Selector] load_llm raised for %s; running unload_llm() "
                "to drop any orphan VRAM before retry", normalized)
    try:
        unload_llm()
    except Exception:
        log.exception("[Selector] unload_llm() also raised; continuing")
    raise
```

**Unverified:** this is a code-side hypothesis. The actual failure inside `load_llm` (the thing that raises after `from_pretrained`) is NOT diagnosed. The orphan-cleanup is defensive at the wrong level if the real bug is e.g. that `load_llm` is BUILDING the model twice itself, or that the 16 GB card simply cannot hold Mistral-Nemo + KV-cache + activations + accelerate's CPU staging.

---

## 4. The supervisor self-kill (the immediate blocker)

`scripts/overnight_bug_hunt.py` calls `scripts/kill_all_python.bat` at three points:
- pre-flight (line "kill_all_python()" right after START log)
- AFTER each iter (Jeffrey directive 2026-05-17)
- implicitly via launch_comfyui_iter's startup race

`kill_all_python.bat`:
```powershell
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force }
```

No PID exclusion. The supervisor's own python.exe matches the filter. Suicide on first call.

Fix options:
A) **PID exclude.** Pass `$env:OWN_PID` (or `os.getpid()`) into the bat and skip that PID. Smallest change. Risk: misses any python the supervisor spawns as a child (`subprocess.run`-spawned pythons are caught -- that's the intent).
B) **Cmd-line filter.** Skip processes whose CommandLine contains "overnight_bug_hunt.py". Specific to this script. Risk: brittle to rename.
C) **Filtered version only.** Use `scripts/kill_comfyui_pythons.bat` (the filtered one I built before the nuclear one) which only kills processes whose CommandLine matches `main.py` / `ComfyUI\resources`. Safer default. Risk: misses orphan-via-other-name pythons.
D) **Two-process design.** Supervisor launches a separate `worker.py` per iter via Start-Process detached. Supervisor only kills the worker's PID tree, never blanket-kills. Cleanest architecture, ~50 lines extra code.
E) **Skip iter-1 pre-flight.** Operator manually kills before invoking. Simplest, but fragile.

Recommended by operator-historian: (D) two-process. Cleanest separation, lowest collateral risk.

---

## 5. What we want from the round-robin

1. **Rank options A-E above** for the supervisor self-kill fix. One winner with reasoning.
2. **Critique the OOM patch.** Is wrapping `load_llm` with try/except + `unload_llm()` correct, OR is the right fix inside `load_llm` itself (a `finally` that releases on partial failure), OR does the writer need a different upstream fix (single load + share cache_entry across pick_style)?
3. **Mistral-Nemo on 16 GB sanity.** Mistral-Nemo Instruct 2407 is 12B params, ~24 GB at FP16. BNB 4-bit gets it to ~7 GB, plus KV cache + accelerate staging. Can this realistically run on a 16 GB Blackwell card with `--highvram --force-fp16 --cuda-malloc`, or is the OOM a real "model too big" problem masquerading as a cache bug? If the answer is "switch default to Gemma-4-E4B for the writer", say so.
4. **Bug-hunt loop architecture.** With (D) two-process design, what's the right per-iter recipe so unattended overnight runs actually produce useful data instead of a wall of identical OOMs? Specifically: should the loop re-queue after a FAIL, or move on and report? Should it auto-bisect by reducing target_words on each retry? Should it kill ComfyUI between iters or keep one ComfyUI alive across iters?
5. **Risk to v2.0-alpha.** The applied OOM patch in `_otr_model_loader.py` is UNCOMMITTED, untested. The 9 workflow-JSON realignments (normalizer + manual nodes 12/55) are UNCOMMITTED. Should we commit + push now even though the patch is unverified, or commit only after one clean iter?

---

## 6. Decision criteria

1. **Audio C7 baseline non-negotiable.** Default-config audio bytes remain identical or stamped as a new baseline. The OOM-patch site is in the model loader -- it does NOT change generation; only error-recovery. C7 should not regress.
2. **Solo + chronic foot pain.** No "build another tester first" answer. Whatever lands has to be the smallest viable thing that gets one green iter.
3. **Local-only.** No cloud fallback, no API key services. RTX 5080 Laptop, 16 GB VRAM, peak ceiling 14.5 GB.
4. **No new branches.** v2.0-alpha is the only working branch. No sprint-c, sprint-d, sprint-e, etc.
5. **Lean docs.** This md is the round-robin packet. Synthesis comes back as one new md, not three. After synthesis lands, BUG_LOG.md + ROADMAP.md only.

---

## 7. What I want from the consultants in their answer

1. Pick option A-E for the supervisor self-kill, defend.
2. Verdict on the OOM patch: ship it / fix in load_llm instead / both / something else.
3. One paragraph on Mistral-Nemo viability at 16 GB with current OTR loader. If switch model recommended, name the swap.
4. Per-iter recipe: 5-8 numbered steps the supervisor runs. Include kill scope.
5. Riskiest assumption you're making.

---

## 8. Constraints again

- ASCII only in any inline code
- No em-dashes (`--` only)
- No "dummy" word -- use "placeholder" or "stub"
- Keep synthesis at 250 lines or under
- v2.0-alpha is the only branch; do not propose new sprint branches
- Audio C7 byte-identity is non-negotiable for default config

---

## 9. Appendix A -- file inventory created/modified this session (uncommitted)

```
nodes/_otr_outline.py                                    H1 lean rewrite (committed)
workflows/otr_scifi_16gb_full.json                       7-node normalizer + 2-node manual fix
workflows/otr_scifi_16gb_full.json.bak-normalize-*       backup
nodes/_otr_model_loader.py                               OOM patch in request_slot
scripts/start_comfy_h0_baseline.bat                      Blackwell launcher with HF_HOME
scripts/kill_all_python.bat                              nuclear python killer
scripts/kill_comfyui_pythons.bat                         filtered python killer
scripts/normalize_workflow_widgets.py                    drift realigner
scripts/test_prompt_import_isolation.py                  isolation guard (shelved)
scripts/overnight_bug_hunt.py                            supervisor (self-kill bug)
docs/2026-05-17-sprint-h-go-forward-plan.md              earlier plan (still valid)
docs/2026-05-17-headless-tester-rr.md                    earlier round-robin (closed)
docs/2026-05-17-headless-tester-rr__04_synthesis.md      earlier synthesis
docs/2026-05-17-sprint-h-bug-hunt-roundrobin.md          this file
docs/2026-05-16-sprint-d-review-packet.md                absorbed from sprint-c
docs/sprint-d-action-plan-v3-2026-05-16.md               absorbed from sprint-c
docs/retrospectives/*.md (5 files)                       absorbed from triage
docs/SOAK_PROMPT_ANTIGRAVITY.md                          absorbed from triage
```

---

## 10. Appendix B -- the OOM traceback (verbatim, for the consultants)

```
inventor failed after 3 attempts; errors:
[
  "generate_fn raised: ModelLoaderError: load_llm failed for
   model_id='mistralai/Mistral-Nemo-Instruct-2407':
   Allocation on device 0 would exceed allowed memory. (out of memory)
   Currently allocated     : 29.97 GiB
   Requested               : 7.17 GiB
   Device limit            : 15.92 GiB
   Free (according to CUDA): 0 bytes
   PyTorch limit (set by user-supplied memory fraction)
                           : 17179869184.00 GiB"
   ... same shape across all 3 attempts ...
]

Stack:
  OTR_LedgerScriptWriter.run() line 1778
   -> _OTRSP.pick_style(creative_fn=creative_generate_fn, ...)
  _otr_style_picker.pick_style() line 612
   -> _run_inventor(generate_fn, ...)
  _otr_style_picker._run_inventor() line 494
   -> raise StyleGenerationFailedError(...)
```

---

## 11. Appendix C -- the supervisor self-kill smoking gun

```
logs/overnight_supervisor.log (3 lines, all timestamped 2026-05-17T03:37:07):

[2026-05-17T03:37:07] === overnight bug-hunt START ===
[2026-05-17T03:37:07] iters=12 inter_iter_sec=30 until_iso=None
[2026-05-17T03:37:07] running kill_all_python.bat...
```

logs/overnight_results.jsonl: does not exist
ComfyUI process: not running
Supervisor PID 51560: not running

Tasklist confirms no tracked python process from the night.
