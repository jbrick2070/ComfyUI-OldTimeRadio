# HuMo 88× v2 — Decision Point: ship Commit B preemptively, or wait for alien_whispers verdict?

> **Round-robin consultation request, second pass.** Companion to `2026-05-08-humo-phase-c-slowdown-problem-statement.md` and `HuMo_88x_path_forward.md` (the response Claude synthesized to ship Commit A). Read those first if you haven't.
>
> **TEMPORARY** — `git rm` once decision is made and Commit B is shipped (or proven unnecessary).

---

## TL;DR

Commit A (Fix 1, removal of section-5 HuMo pre-pin) is shipped at HEAD `601ae35` and validated non-regressive on the 4-HuMo smoke (4/4 clips, 22:54 wall, step time 62.85 → 66.23 → 64.82 → 62.82 — bouncing in noise band, no fragmentation creep).

Open question: **before queueing alien_whispers (which costs ~7 hours if Fix 1 didn't help), do we ship Commit B (Fix 2 + Fix 3) now and test once, or do we test Fix 1 alone first and ship Commit B only if needed?**

The path-forward author's staged plan said "test Fix 1 alone first." Jeffrey accepted that yesterday. New question is whether the 23 min of 4-HuMo smoke we just ran has changed the math.

## What changed since v1

**Shipped at HEAD `601ae35`:**

1. Removed the section-5 HuMo pre-pin. Code path that previously did
   ```python
   mm.load_models_gpu([model], force_full_load=True)
   ```
   *before* Phase A is now a comment block. Phase A → Phase B → Phase C now run without a doomed pre-pin attempt that the offloader silently violated and that left the cudaMallocAsync pool fragmented.

**Verified non-regressive (4-HuMo smoke, 4 character clones of l002, ledger `synthetic_4humo_ledger.json`):**

| metric | pre-Fix-1 (commit 9c6353d, 6 steps) | post-Fix-1 (commit 601ae35, 5 steps) |
|---|---|---|
| clip 1 wall | 6:17 | 5:34 |
| clip 2 wall | 6:14 | 5:51 |
| clip 3 wall | 6:09 | 5:44 |
| clip 4 wall | (cap=3 fired, not rendered) | 5:36 |
| step time | 60 s/it | 63-66 s/it |
| total wall | 18:49 (3 clips) | 22:54 (4 clips) |
| pin log line | `pinned MODEL via load_models_gpu` present | absent |

**Critical observation:** in the post-Fix-1 smoke, clip 4 was *faster* than clip 2 (62.82 vs 66.23 s/it). Fragmentation does not accumulate at this scale. The smoke envelope was healthy under HEAD even before Fix 1; the smoke does not exercise the ~30-line floor where BUG-LOCAL-081's fix runs out of envelope.

**Alien_whispers (60+ chunk, 7-cast) test NOT yet run on Fix 1.** Cost of running it: ~7 hours wall if Fix 1 didn't close the gap. ~30-90 min if it did.

## The pivotal decision

Two paths:

### Path A — staged (current plan)
1. Queue alien_whispers on HEAD `601ae35` (Fix 1 only)
2. Watch first HuMo Phase C step time
3. Three branches based on per-step time observed in clip 1:
   - ~60 s/it → done. Ship as-is.
   - 200-500 s/it → ship Commit B (Fix 2 + Fix 3) and rerun
   - >5000 s/it → kill within 5 min, ship Commit B, rerun
4. Cost worst-case: 5 min wasted to detect failure + ~30 min to ship Commit B + ~30-90 min to rerun = ~70-130 min total
5. Cost best-case: ~30-90 min (single successful run)

### Path B — preemptive (ship Commit B now)
1. Implement Commit B (Fix 2 = swap inter-phase cleanup to `_hard_reset_cuda_context()`; Fix 3 = pin encoders around Phase A and Phase B loops)
2. Run AST + Bug Bible regression + 4-HuMo smoke
3. Push Commit B
4. Queue alien_whispers
5. Cost: ~30 min Commit B work + ~30-90 min rerun = ~60-120 min total
6. Risk: introduces two more code changes whose impact we can't isolate from Fix 1's. If alien_whispers passes, we don't know which fix carried the load. If it fails, we have more places to debug.

### Cost-benefit summary

|  | Path A (staged) | Path B (preemptive) |
|---|---|---|
| Best-case wall time | ~30-90 min | ~60-120 min |
| Worst-case wall time | ~70-130 min | ~60-120 min |
| Variance | high (depends on Fix 1 sufficiency) | low |
| Causal data on which fix did the work | clean | muddied (3 fixes stacked) |
| GPU-time risk | up to 7 hr if we miss the kill window | bounded |
| Code complexity introduced | 1 deletion (Fix 1) | 1 deletion + ~30 lines added (Fix 1+2+3) |

**Jeffrey's stated bias:** lean toward Path A for clean causal data. The path-forward doc says "If Fix 1 alone moves step time below 500 s/it: H2 was secondary, Fix 3 priority drops." That measurement matters.

**Counter:** if Path A's worst case is monitored carefully (5-min kill window if first step shows >500 s/it), the GPU-time risk shrinks to the same as Path B's, while preserving the causal data.

## Updated hypothesis confidence (since v1)

| hypothesis | v1 confidence | v2 confidence | basis for update |
|---|---|---|---|
| H1 (allocator fragmentation past 081 envelope) | most likely | unchanged | smoke didn't reach the envelope; no new evidence |
| H2 (encoder re-staging is real PCIe per call) | secondary | leaning weaker | smoke shows step time stable across 4 clips with 4 encoder rounds; if H2 were dominant, even smoke should drift |
| H3 (cumulative tensor pin growth) | unlikely | unchanged | smoke step time bounces, doesn't climb |
| H4 (cudaMallocAsync regression under sustained pressure) | unlikely | unchanged |  |
| H5 (Mistral-Nemo residency) | unlikely | unchanged |  |

The smoke can rule OUT H2/H3 dominance at small scale. It cannot rule them out at large scale. Only alien_whispers can.

## Commit B scope (if shipped)

**Fix 2 — replace `batch_humo_render.py` lines 2109-2115 with `_hard_reset_cuda_context()` call.**

```python
# Before:
try:
    import comfy.model_management as mm  # type: ignore
    log.info("[BatchHumoRender] Inter-phase VRAM cleanup: unload_all_models + soft_empty_cache")
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
except Exception as exc:
    log.warning("[BatchHumoRender] inter-phase VRAM cleanup failed: %s", exc)

# After:
log.info("[BatchHumoRender] Inter-phase VRAM cleanup via _hard_reset_cuda_context() (BUG-126 chain)")
_hard_reset_cuda_context()
```

`_hard_reset_cuda_context()` is at line 155, already imports torch + mm internally, includes Item H telemetry (before/after `memory_allocated()` + `memory_reserved()`), already battle-tested on the OOM path. The function name is misleading (it's a soft chain, not a context destroy) but the chain itself is what we want at the inter-phase boundary.

**Fix 3 — pin encoders around Phase A loop (lines 2018-2038) and Phase B loop (lines 2040-2059).**

Phase A pattern (mirror for Phase B with `audio_encoder` and chunk loop):
```python
import comfy.model_management as mm
try:
    try:
        mm.load_models_gpu([clip], force_full_load=True)
    except TypeError:
        mm.load_models_gpu([clip])
    log.info("[BatchHumoRender] Phase A: pinned umt5 via load_models_gpu")
    negative = _call(text_enc, clip=clip, text=_CHINESE_NEGATIVE)[0]
    for entry in plan:
        try:
            entry["positive"] = _call(text_enc, clip=clip, text=entry["pos_text"])[0]
        except Exception as exc:
            log.warning("[BatchHumoRender] %s: text encode failed: %s",
                        entry["line_id"], exc)
            entry["positive"] = None
finally:
    try:
        mm.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        log.warning("[BatchHumoRender] Phase A unpin failed: %s", exc)
```

**Open code-design questions inside Fix 3:**
- Does `force_full_load=True` exist as a kwarg on the running ComfyUI 0.20.x's `load_models_gpu`? The `try/except TypeError` is defensive but we don't know if it's needed.
- Phase A produces conditioning tensors that flow into Phase C. If we `unload_all_models()` at the end of Phase A, do those conditioning tensors get evicted too? The current `_to_cpu` walk (lines 2082-2107) handles that for the CPU offload, but if the unpin happens BEFORE the CPU offload we may double-evict. Order of operations matters: pin → encode all → CPU-offload conditioning → unpin → reset.

## Round-robin questions (v2)

1. **Path A vs Path B:** which would you choose, given the cost/causal-data tradeoff above? Is the 5-min kill window a sufficient bound on Path A's worst case to make it dominant?

2. **Hypothesis update:** does the 4-HuMo smoke's step-time stability (62-66 s/it across 4 clips, last clip fastest) update your prior on H1 vs H2 dominance? It rules nothing out at the alien_whispers scale, but does it shift the relative likelihoods?

3. **Fix 2 risk:** wiring `_hard_reset_cuda_context()` into the unconditional inter-phase boundary is what was already proposed. Any concern with the function being called outside an `except` block? The Element 3 traceback-retention catch (line 191 in the docstring) says "MUST be called from OUTSIDE the active `except` block." We are outside. But is there any other context-of-call concern (active autograd graphs, in-flight async streams from Phase B's last Whisper call, etc.)?

4. **Fix 3 ordering:** in Phase A's `finally` block, should we
   - (a) unpin first, then CPU-offload (current proposal)
   - (b) CPU-offload first, then unpin
   - (c) unpin → empty_cache → CPU-offload → empty_cache again
   ? The conditioning tensors need to land somewhere safe before umt5 is evicted. Current `_to_cpu` walk runs at the existing 2082-2107 boundary, which is *after* both Phase A and Phase B complete. If we add per-phase unpin, the CPU-offload boundary shifts.

5. **Should Fix 2 and Fix 3 ship together or separately?** The path-forward doc said "they're complementary fragmentation hardening; splitting them gives you noisier signal, not cleaner." Agreed for Path B. But if Path A is chosen and Fix 1 alone is insufficient, is there value in shipping Fix 3 alone before Fix 2 (since Fix 3 directly attacks the 150 `prepared` log lines, which is the most visible symptom)?

6. **`load_models_gpu([clip], force_full_load=True)` API stability:** is this kwarg safe to depend on across ComfyUI 0.20.x patch versions? Should we pin to a specific ComfyUI version in `pyproject.toml` to avoid the `try/except TypeError` becoming load-bearing?

7. **Alternative I might be missing:** is there a simpler structural fix we haven't considered that subsumes Fix 2 + Fix 3? For example, restructuring `BatchHumoRender.execute()` so that Phase A and Phase B are subprocess-isolated (spawn a child process per phase, child returns CPU tensors, child exits → CUDA pool resets cleanly when child dies)? OTR's v2.0 visual stack already uses subprocess isolation for FLUX (per `CLAUDE.md` C3 constraint). Would that pattern fit here, and is it overkill for a 30-line fix?

## Disposition

- **Status:** open, second-pass round-robin requested
- **Owner:** Claude (synthesizer) → Jeffrey (decision)
- **Decision deadline:** before queuing alien_whispers
- **Rollback plan:** unchanged — `git reset --hard 467969a` reverts Fix 1; tag `v2.0-alpha-pre-humo-fix-2026-05-08` is the pre-Fix-1 known-good
- **Delete this doc** when: (1) Path A or Path B is chosen, (2) alien_whispers completes, (3) BUG_LOG entry written

---

## Appendix: full repro recipe for round-robin AIs

**Repo:** `https://github.com/jbrick2070/ComfyUI-OldTimeRadio` branch `v2.0-alpha`
**HEAD:** `601ae35` (Fix 1 shipped)
**Tag for rollback:** `v2.0-alpha-pre-humo-fix-2026-05-08`
**File of interest:** `nodes/batch_humo_render.py` (now 2972 lines after Fix 1)
**Hardware:** RTX 5080 Laptop, 16 GB VRAM, 64 GB RAM, Windows, torch 2.10.0+cu130, CUDA 13, SDPA + SageAttention only

**Reproduction (smoke, 23 min, no GPU risk):**
1. `git checkout v2.0-alpha`
2. ComfyUI → Load → `workflows/otr_humo_4x_smoke.json`
3. Queue. Confirm `pinned MODEL via load_models_gpu` log line is absent.
4. 4 clips render in 22-23 min wall, step time bounces 62-66 s/it.

**Reproduction (alien_whispers, 30 min - 7 hr depending on Fix 1 sufficiency):**
1. `git checkout v2.0-alpha`
2. ComfyUI → Load → `workflows/otr_scifi_16gb_full.json`
3. Queue, watch first HuMo Phase C step time
4. Document per-step time on first character clip
5. If >500 s/it → kill, ship Commit B
6. If <500 s/it → let complete, document total wall time
