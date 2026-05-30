# Lean Go-Forward Cleanup Sprint

**Date:** 2026-05-29 | **Branch:** `v2.0-alpha` | **HEAD:** `d893e56` (8 commits ahead of origin, UNPUSHED)
**Inputs synthesized:** `session_handoff.md` + `docs/DEAD_CODE_AUDIT_2026-05-29_round3.md` + external consult feedback
**Verification:** 3 parallel read-only subagents re-ran every claim against the live tree (`rg -a`, byte scans, JSON parse). Findings below supersede the consult where they differ.

---

## 0. Execution outcome (2026-05-29, executed)

**Status: EXECUTED. Phase A + B1 + C1–C3 done; 3 new commits gated green and UNPUSHED; D1 deferred.**

| Phase | Outcome |
|---|---|
| A — push existing 8 | Already on origin (`git ls-remote` = local HEAD `d893e56`). No action needed. |
| B1 — NUL strip | **NO-OP / phantom.** On the real Windows disk the strip script reported `Stripped 0 file(s)`. Only 3 tracked files contain NULs and they are legitimate binaries (`baseline_v1.5.wav`, 2 `workflows/*.png`). The "9–10 source files with trailing NULs" in §1/§2 below was a **stale Cowork bash-mount artifact** that fooled both the round-3 audit and the verification subagents. No source file was ever NUL-padded. No commit. |
| C1 — `_otr_critic_rubric.py` + rubric md | Deleted. Commit `5fc49d5`, −426 lines. Re-verified dead via `git grep` on the clean real tree. |
| C2 — `visual/planner.py` | Deleted. Commit `6c80597`, −513 lines. |
| C3 — `visual/postproc/` | Deleted. Commit `b5c1e93`, −579 lines. |
| D1 — LoRA 60/61 | **Deferred / untouched** per operator (test current vs 0.7 before changing). Verified-present and accurate; workflow JSON not modified. |

**Gates after every deletion:** Bug Bible 23 passed / 1 skipped / 2 xfailed; `test_core` + `test_audio_byte_identical` 68 passed / 1 skipped. Full suite final gate: only the 2 documented baseline reds (`test_bark_freeze_halt_bypass…node_11`, `test_llm_slot_sweep`), zero new failures. Audio byte-identical throughout. Net `−1,518` lines.

**Remaining:** final `git push origin v2.0-alpha` (3 commits) — gated on operator OK.

> Note: §§1–10 below are the original pre-execution plan. The NUL-strip content (the "10 files" table, the strip script) is retained for the record but proved unnecessary on the real disk — see this banner.

---

## 1. Synthesis — consult vs. verified ground truth

The consult was directionally right on all four findings. Verification corrected two numbers and caught one safety trap. **Trust the right-hand column.**

| Finding | Consult / audit said | Verified ground truth | Delta |
|---|---|---|---|
| NUL-byte files | 9 files (4 prod, 5 test) | **10 files (5 prod, 5 test)** — audit missed `session_handoff.md` (324 trailing NULs) | +1 prod file |
| NUL strip safety | "strip trailing `\x00`" | Safe for all 10 text files (trailing-only, all `.py` re-parse). **But 3 tracked binaries have INTERIOR NULs** (`workflows/*.png` ×2, `tests/fixtures/baseline_v1.5.wav`) — a naive "strip every tracked file" would corrupt them | Script must self-guard |
| Dead code | ~1,330 LOC, 3 items | **Confirmed dead, 3 items.** 1,329 LOC of `.py` + 189-line rubric `.md` = **1,518 total lines** | "1,330" was Python-only; +189 doc |
| Double LoRA stack | nodes 60/61, same file, 0.5 + 0.2 | **CONFIRMED exactly.** `54 → 60(0.5) → 61(0.2) → 55`, identical file `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | None — consult was precise |

Net: the cleanup target is **3 file deletions (1,518 lines) + 1 NUL-strip sweep (10 files) + 1 optional workflow JSON consolidation (decision gate)**. None of the 3 dead modules appears in any of the 10 workflow JSONs, so deletions need **zero re-wiring**.

---

## 2. Verified removable inventory

### Dead code — TIER 1 (one deletion per commit)

| # | Item | LOC | Why dead | Re-wire? |
|---|---|---|---|---|
| C1 | `nodes/_otr_critic_rubric.py` + `docs/2026-05-26-sprint-10a-whole-episode-critic-rubric.md` | 237 + 189 | Orphaned by `aad4cfb` (whole-episode critic teardown). Zero importers; `Rubric`/`load_rubric`/`RubricAxis`/`ShipThreshold` referenced nowhere live | No |
| C2 | `visual/planner.py` | 513 | Live sidecar (`worker.py`) dispatches via `backends.resolve()` directly; `plan_episode` never called | No |
| C3 | `visual/postproc/` (`vhs.py` 549 + `__init__.py` 30) | 579 | Unused VHS filter; nothing imports `postproc`/`vhs` | No |

**Combined: 1,329 LOC code + 189 LOC doc = 1,518 lines.**
Only surviving mention after deletion: a harmless design-intent comment at `visual/backends/wan21_loop.py:58` ("...so the planner can mix and match..."). Not an import. Trim it in C2 or leave it.

### NUL-byte sweep (single commit, B1) — verified 10 files

| File | Trailing NULs | Class |
|---|---|---|
| `__init__.py` | 862 | prod |
| `nodes/_otr_freeze_cascade.py` | 3696 | prod |
| `nodes/_otr_legacy_to_stage1_adapter.py` | 100 | prod |
| `visual/unload_all.py` | 10 | prod |
| `session_handoff.md` | 324 | prod (audit missed) |
| `tests/test_audiogen_cache_keys.py` | 4110 | test |
| `tests/test_core.py` | 65 | test |
| `tests/test_g7_consumer_constants.py` | 1147 | test |
| `tests/test_per_cue_sfx_dur.py` | 140 | test |
| `tests/test_stage7_shadow_critic_wiring.py` | 4336 | test |

**EXCLUDE (interior NULs — real binaries):** `workflows/flux_dev_checkpoint_example.png`, `workflows/flux_schnell_checkpoint_example.png`, `tests/fixtures/baseline_v1.5.wav`. The script in §6 excludes these automatically.

### Workflow JSON anomaly — DECISION GATE (D1, not auto-applied)

`workflows/otr_scifi_16gb_full.json` chains two `LoraLoaderModelOnly` nodes loading the **identical** file:

```
node 54 (LowVRAMCheckpointLoader)
   --MODEL--> node 60  LoraLoaderModelOnly  ltx-2.3-22b-distilled-lora-384-1.1.safetensors  strength 0.5
   --MODEL--> node 61  LoraLoaderModelOnly  ltx-2.3-22b-distilled-lora-384-1.1.safetensors  strength 0.2
   --MODEL--> node 55  OTR_BatchLTXRender
```

`LoraLoaderModelOnly` patches weights additively, so two sequential loads of the same adapter are mathematically identical to one: `W + 0.5·Δ + 0.2·Δ = W + 0.7·Δ`. **Consolidating node 60→0.7 and deleting node 61 is weight-equivalent** and removes one node + one load. Keep the split only if you intend to later swap one slot for a *different* LoRA. **This is your call — see §5 gate D1. Do not touch the JSON without a yes.**

---

## 3. KEEP guards — do NOT delete (over-deletion tripwires)

The round-3 audit cleared these as live or KEEP-by-design. Listed so the executor never widens scope:

- **8 unwired-but-KEEP nodes:** `OTR_ProjectStateLoader`, `OTR_VRAMGuardian`, `OTR_VRAMContextTest`, `OTR_VisualBridge`, `OTR_VisualPoll`, `OTR_VisualRenderer`, `OTR_VisualPromptCoercion`, `OTR_VisualExtractFluxPrompt` (sidecar subprocess + VRAM topology).
- **2 test-only modules HELD BACK** (model/provider plumbing KEEP zone): `nodes/_otr_model_runtime.py`, `nodes/_voice_backends/`. Retire only in a deliberate sprint with their test suites.
- **node-63 `OTR_WorkflowValidator`:** test-pinned present in the workflow by `TestValidatorPathFallback` + `TestValidatorEmptyPathFallback`. Removing it reds 3 tests. Out of scope.
- **Intentional mirrors:** `_resolve_radio_still_path` (BUG-LOCAL-121), `_load_ledger` (BUG-LOCAL-076). Do not consolidate.
- **NUL-padded files are LIVE** (`_otr_freeze_cascade.py`, `_otr_legacy_to_stage1_adapter.py`, `__init__.py`, `unload_all.py`) — strip NULs, never delete.

---

## 4. The sprint — review → code → wire → regress → commit

Each commit follows the identical loop. **Serial, one change per commit** (see §8 on why deletions can't be parallelized). Order is deliberate: NUL-strip first so every later `rg` audit reads clean files.

| Phase | Commit | Change | Wire impact |
|---|---|---|---|
| **A** | (push) | Land the 8 existing commits + docs to origin first — de-risk before adding more | — |
| **B1** | 1 | NUL-strip sweep (10 files via §6 script) | `__init__.py` touched → re-confirm 35-node load |
| **C1** | 2 | Delete `_otr_critic_rubric.py` + rubric `.md` | none |
| **C2** | 3 | Delete `visual/planner.py` (optionally trim `wan21_loop.py:58` comment) | none |
| **C3** | 4 | Delete `visual/postproc/` subtree | none |
| **D1** | 5 | *(gated)* Consolidate LoRA 60→0.7, delete node 61, re-wire 60→55 | **JSON re-wire — workflow validator must pass** |
| **E** | (push) | One push attempt → else PowerShell block; full verify | — |

### Per-commit loop (run every time, no exceptions)

1. **Review** — `rg -a` the symbols once more on the now-clean tree; confirm still zero live refs.
2. **Code** — delete/edit the file(s).
3. **Wire** — C1–C3 need none (verified). B1 touches `__init__.py`; D1 touches the JSON. For any node-surface or JSON change, confirm class names, input names, widget defaults, output sockets still match (CLAUDE.md Prime Directive 3).
4. **Regress** — run the gate block in §7. Must be green except the 2 known pre-existing baseline reds (`test_bark_freeze_halt_bypass…node_11…bypass_default`, `test_llm_slot_sweep::test_every_llm_call_site_has_slot_tag`). Audio stays byte-identical.
5. **Commit** — `.git\COMMIT_EDITMSG` + `-F` (templates §7). One change per commit.

---

## 5. Decision gates (your call before the executor proceeds)

- **Gate A — push order:** push the 8 landed commits + 3 untracked docs to origin *before* the new cleanup, or batch everything into one push at the end? Recommended: push first (smaller blast radius, matches handoff next-steps).
- **Gate D1 — LoRA consolidation:** consolidate 60/61 → single 0.7 loader (weight-equivalent, −1 node), or leave the split intentional? Recommended only if you confirm 0.7 was the intent and you won't repurpose node 61.
- **In-flight render (from handoff):** FLUX stalled at 66 s/step on a VRAM thrash unrelated to cleanup; ComfyUI PID 17104 may still be up. Stop it (`taskkill /F /T /PID 17104`) before launching a fresh gated load test, or leave it. Out of cleanup scope but noted so the gate runs clean.

---

## 6. NUL-strip script (UTF-8, no BOM, self-guarding)

Save as `scripts/_otr_strip_trailing_nuls.py`. Only rewrites files whose NULs are **trailing-only**; any file with interior NULs (the 2 PNGs + the WAV) is skipped untouched. `.py` files must re-parse or they're skipped.

```python
#!/usr/bin/env python3
"""Strip trailing NUL (\\x00) padding from tracked text/source files.

Self-guarding: rewrites a file only when every NUL is trailing. Any file with
interior NULs (real binaries: PNG, WAV) is left untouched. .py files must still
AST-parse after the strip or they are skipped. Run from repo root with the venv
python. Reports what changed; makes no commit.
"""
import subprocess
import sys
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT_EXT = {".py", ".md", ".json", ".txt", ".yaml", ".yml", ".cfg", ".ini"}


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO,
        capture_output=True, text=True, check=True,
    )
    return [REPO / line for line in out.stdout.splitlines() if line]


def main():
    changed, skipped_binary, skipped_parse = [], [], []
    for path in tracked_files():
        if path.suffix.lower() not in TEXT_EXT:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" not in data:
            continue
        stripped = data.rstrip(b"\x00")
        if b"\x00" in stripped:           # interior NULs -> never touch
            skipped_binary.append(path)
            continue
        if path.suffix.lower() == ".py":  # must remain valid Python
            try:
                ast.parse(stripped.decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError) as exc:
                skipped_parse.append((path, exc))
                continue
        path.write_bytes(stripped)
        changed.append((path, len(data) - len(stripped)))

    print(f"Stripped {len(changed)} file(s):")
    for path, removed in changed:
        print(f"  -{removed:>5} NUL  {path.relative_to(REPO)}")
    if skipped_binary:
        print(f"\nSkipped {len(skipped_binary)} file(s) with interior NULs (untouched):")
        for path in skipped_binary:
            print(f"  {path.relative_to(REPO)}")
    if skipped_parse:
        print(f"\nSkipped {len(skipped_parse)} .py file(s) that failed re-parse:")
        for path, exc in skipped_parse:
            print(f"  {path.relative_to(REPO)} :: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Run (Desktop Commander cmd, repo root):**

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\_otr_strip_trailing_nuls.py
```

Expect: 10 files stripped, 3 skipped (2 PNG + 1 WAV). If the count differs, stop and investigate before committing.

---

## 7. Gate block + commit templates

### Regression gate (run after EVERY change — CLAUDE.md mandate)

Windows pytest rejects forward-slash paths and `findstr | pytest` eats the summary line — use backslashes and tee to a log, then read the tail (handoff lesson):

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

REM Bug Bible regression
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\tests\bug_bible_regression.py" -v > .git\_gate_bible.log 2>&1

REM Core + audio byte-identical
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\test_core.py tests\test_audio_byte_identical.py -v > .git\_gate_core.log 2>&1

REM Full suite (the lean-down baseline ran 3350 tests green)
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests -q > .git\_gate_full.log 2>&1
```

Then read the tails of `.git\_gate_*.log`. **Green = only the 2 known baseline reds remain** and `test_audio_byte_identical` passes. After B1 and D1 also run the API load test (catches node-surface / registration drift):

```
cd /d C:\Users\jeffr\Documents\ComfyUI
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe _otr_build_api.py
```

Expect zero drift, all 18 type-sanity PASS, and `[OldTimeRadio] OK - All 35 nodes loaded successfully`.

### Commit (no `-m`; write message file then `-F`)

Single ASCII line:

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
echo Strip trailing NUL padding from 10 tracked files (clean-file fix)> .git\COMMIT_EDITMSG
git add -A
git commit -F .git\COMMIT_EDITMSG
```

Structured / multi-line message: write `.git\COMMIT_EDITMSG` with the file tool (read-before-write), then `git commit -F .git\COMMIT_EDITMSG`. Never `-m`, never PowerShell, never `( echo & echo )` blocks, never inline `python -c` heredocs (all corrupt the message through Desktop Commander cmd — CLAUDE.md anti-patterns).

**Suggested subjects (one per commit):**

- B1: `Strip trailing NUL padding from 10 tracked files (clean-file fix)`
- C1: `Remove orphaned _otr_critic_rubric.py + rubric doc (dead since aad4cfb)`
- C2: `Remove unused visual/planner.py (no live caller)`
- C3: `Remove unused visual/postproc/ VHS filter subtree`
- D1: `Consolidate duplicate LTX LoRA loaders 60/61 into single 0.7 loader`

---

## 8. Push + final verify

One push attempt via Desktop Commander cmd; if it fails, hand a PowerShell block (never push from PowerShell directly):

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio && git push origin v2.0-alpha
```

If `.git\HEAD.lock` blocks: `del .git\HEAD.lock` then retry once.

**Verify after push (all must hold):**

- `git rev-parse HEAD` == `git rev-parse origin/v2.0-alpha` (local HEAD == origin HEAD)
- No 0-byte files: `git ls-files | ...` size check
- No BOM, no remaining trailing NULs (re-run §6 script — expect "Stripped 0 files")
- All node classes still registered in `__init__.py`; API load test = 35 nodes
- Every workflow JSON parses and is wired to current node surfaces (D1: validator passes)

---

## 9. Parallelization — what's parallel, what's serial

You asked for subagents in parallel. Honest split:

- **Parallel (done):** the three verification lanes above (NUL inventory / dead-code zero-refs / workflow LoRA) ran concurrently as read-only subagents — independent, no shared writes.
- **Serial (required):** the commit loop. One deletion per commit with a full gate between each is a hard CLAUDE.md rule, and concurrent commits to the same `v2.0-alpha` ref would race the index and `.git\HEAD.lock`. You cannot parallelize gated commits safely.
- **Optional parallel within a phase:** a subagent can pre-stage the next commit's `rg -a` review while the current commit's gate runs — but the `git add/commit` step stays single-threaded.

So: parallel verification → serial gated commits → single push.

---

## 10. One-glance checklist

```
[ ] Gate A answered (push existing 8 now, or batch at end)
[ ] B1  strip NULs (10 files)        -> gate + API load (35 nodes) -> commit
[ ] C1  del _otr_critic_rubric +md   -> gate -> commit
[ ] C2  del visual/planner.py        -> gate -> commit
[ ] C3  del visual/postproc/         -> gate -> commit
[ ] D1  LoRA consolidate (if YES)    -> gate + API load + JSON validator -> commit
[ ] E   one push attempt -> verify HEAD==origin, no 0-byte, no BOM, nodes registered, JSON wired
[ ] Audio byte-identical held at every gate
[ ] Only the 2 known baseline reds remain
```
