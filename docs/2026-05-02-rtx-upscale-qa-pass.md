# rtx_upscale spacesaver QA pass — go-forward plan

**Branch:** `v2.0-alpha`
**Owner:** Jeffrey A. Brick
**Date:** 2026-05-02
**Status:** Phase A diff drafted; Phases B–E pending
**Scope:** Findings 1–7 from the `rtx_upscale.py` review. Non-Sprint-1 hygiene track. Becomes load-bearing the moment two episodes are ever in flight (soak runs, Sprint 3 acceptance, any successor to `supersoaker`).

---

## Position relative to the roadmap

This sits **adjacent to** Sprint 1 (BUG-LOCAL-003/004/005/006), not inside it. Sprint 1 is "make the 30-word smoke green" — single episode, no concurrent runs, so Finding 1's wrong-episode wipe is invisible to it. Sprint 1 keeps top priority. The QA track interleaves where there is spare context, with **Phase A landing first** because it is destructive and the diff is already written.

---

## Canonical episode layout (reference)

```
output/otr/
├── obs/
│   └── <episode_id>.mp4                      # FINAL deliverable, FLAT, OBS watches
└── episodes/
    └── <episode_id>/                         # everything for this one episode
        ├── audio/
        │   ├── <episode_id>.mp4              # procgen base
        │   ├── <episode_id>_ledger.json      # canonical ledger
        │   ├── opening_<sha>_<ts>.wav        # MusicGen
        │   ├── closing_<sha>_<ts>.wav        # MusicGen
        │   ├── interstitial_<sha>_<ts>.wav   # MusicGen
        │   ├── sfx_<prompt>_<sha>_<ts>.wav   # AudioGen, one per cue
        │   └── director_dump_<ts>.txt
        ├── stills/                           # FLUX environments + radio bookend
        ├── portraits/                        # PASS1 character portraits
        ├── videos/                           # per-line clip pieces
        └── composited/                       # 832x480 intermediate
```

**Two structural facts the plan leans on:**

1. `src` from VideoComposite resolves to `output/otr/episodes/<ep>/composited/<ep>.mp4`, so `src.parent.parent` is the episode root. This is the basis of Phase A.
2. `output/otr/obs/<ep>.mp4` is a sibling of `episodes/`, not a child. Spacesaver's `len(rel.parts) != 1` guard cannot reach it.

---

## Phase A — Land Finding 1 + OBS-existence guard

**File:** `nodes/rtx_upscale.py` · function `_spacesaver_cleanup_if_flagged`.

**Core fix (drafted diff):** derive `ep_dir = src.resolve().parent.parent` directly from the composited mp4. Drop the global mtime-based ledger scan. Two consequences:

- Two episodes in flight are isolated by construction. A's RTXUpscale physically cannot reach B's tree.
- Keep-list is built from real filenames via `audio_dir.glob("*_treatment.txt")`, not from slug-reconstructed names. This absorbs most of Finding 4 as a side effect.

**Amendment — OBS-existence precondition.** Before merge, confirm the run order: (1) RTXUpscale writes the upscaled mp4, (2) copy/move to `output/otr/obs/<ep>.mp4`, (3) `_spacesaver_cleanup_if_flagged`. If 2 currently happens after 3, the wipe takes the upscaled mp4 with it before it can land in `obs/`. Defensive guard regardless — add at the top of the function after the existing safety guards:

```python
obs_final = ep_dir.parent.parent / "obs" / f"{ep_id}.mp4"
if not obs_final.exists():
    log.warning(
        "[OTR_RTXUpscale] spacesaver: OBS deliverable %s not on disk; "
        "refusing cleanup until final lands",
        obs_final,
    )
    return
```

Spacesaver must never wipe intermediates if the final deliverable has not landed.

**Acceptance:**
- Queue Episode A, queue Episode B before A's RTXUpscale fires. Inspect log lines from `[OTR_RTXUpscale] spacesaver:` — `ep_dir` resolves to A's path, not B's.
- Bypass safety guard: pass an `src` outside `output/otr/episodes/`. Expect `refusing destructive cleanup` warning, no deletion.
- Legacy flat-layout episodes under `output/audio/` are silently skipped — matches "leave existing as is" intent.
- Delete `output/otr/obs/<ep>.mp4` before cleanup fires, expect the new precondition to abort.

**Bug log entry:** `BUG-LOCAL-014 — Spacesaver wrong-episode wipe via global mtime ledger scan`. Bible candidate: yes, after the two-episode test confirms behavior on a real run.

**Risk gate before Phase B:** if the two-episode test surfaces any other concurrent-episode artifact, stop and inventory before continuing. There may be a sibling bug in another node that also assumes single-episode-at-a-time.

---

## Phase B — Findings 2 + 3 in `production_ledger.py`

Single commit. Both touch the same pending → finalized rename path.

- **Finding 2 — treatment rename gap.** `pending_<ts>_treatment.txt` does not always get renamed to its final form. Stale pending files accumulate and confuse Phase A's keep-list (which currently catches *any* `*_treatment.txt` for safety — that safety is paid for by Phase B closing the gap).
- **Finding 3 — `os.replace` fallback.** Atomic ledger swap on Windows. The current path can leave a half-written ledger if the process is killed between write and rename.

**Acceptance:**
- Kill a run mid-rename (Ctrl-C between treatment write and rename), restart, confirm clean recovery and no orphan `pending_*_treatment.txt` after the next successful run.
- After a normal end-to-end run: `audio_dir.glob("pending_*")` returns empty.
- Post-fix, Phase A's keep-list shrinks naturally to one ledger + one treatment per episode.

---

## Phase C — Finding 4 residual audit

Phase A handles the slug-mismatch trap inside `rtx_upscale.py`. Phase C extends the same discipline to every other consumer of the audio dir.

**Filename pattern audit table:**

| On-disk pattern | Producer | Correct lookup |
|---|---|---|
| `<ep>.mp4` | EpisodeAssembler procgen | `audio_dir / f"{ep_id}.mp4"` (slug-derivable, OK) |
| `<ep>_ledger.json` | production_ledger | `audio_dir.glob("*_ledger.json")` |
| `opening_<sha>_<ts>.wav` | musicgen_theme | `audio_dir.glob("opening_*.wav")` |
| `closing_<sha>_<ts>.wav` | musicgen_theme | `audio_dir.glob("closing_*.wav")` |
| `interstitial_<sha>_<ts>.wav` | musicgen_theme | `audio_dir.glob("interstitial_*.wav")` |
| `sfx_<prompt>_<sha>_<ts>.wav` | batch_audiogen_generator | `audio_dir.glob("sfx_*.wav")` |
| `director_dump_<ts>.txt` | story_orchestrator | `audio_dir.glob("director_dump_*.txt")` |
| `pending_<ts>_treatment.txt` / `*_treatment.txt` | story_orchestrator | `audio_dir.glob("*_treatment.txt")` |

**Audit rule:** any code path that constructs `audio_dir / f"opening_{ep_id}.wav"` (or any equivalent slug-prefix attempt) is wrong — there is no such filename in this tree. Replace with the right glob and pick by sha or by mtime depending on intent. Computing log strings from `ep_id` is fine. The rule is: do not compute on-disk paths from slugs.

**Likely hit list (verify):** `story_orchestrator.py`, `production_ledger.py`, `video_composite.py`, `episode_assembler.py`. Single PR, one commit per file touched.

**Acceptance:** no `f"{ep_id}_..."` filename construction remains in any path that reads or deletes from disk.

---

## Phase D — Finding 6 cache-key timestamp suffix

**Files:** `musicgen_theme.py`, `batch_audiogen_generator.py`.

**Symptom:** cache keys include a timestamp suffix, so identical prompts on a re-run produce a different key and miss the cache. Wasted GPU cycles, no correctness impact.

**Fix:** the `<sha>` in the on-disk filename **is** the cache key. The `<ts>` is purely an on-disk uniqueness suffix for the case where multiple identical-content files coexist (e.g., a re-run that fired before a prior cleanup). Lookup logic:

```python
sha = hash(prompt + seed + sample_rate + model_revision + decode_mode)
matches = list(audio_dir.glob(f"{role}_{sha}_*.wav"))
if matches:
    return max(matches, key=lambda p: p.stat().st_mtime)  # cache hit
# else generate, save as f"{role}_{sha}_{ts}.wav" with fresh ts
```

This matches the C7 protocol locked in the roadmap: same prompt + same seed + same model rev → byte-identical output → cache hit is safe.

**Acceptance:**
- Two consecutive identical-input runs produce one file on disk for that `(role, sha)` pair. Second run logs `cache_hit=True`.
- Five mutation tests, one per dimension {seed, model_revision, decode_mode, sample_rate, prompt}. Each mutation produces a fresh sha and a second file alongside the first.
- The `<ts>` suffix never proliferates extra files for identical content. If it does, the cache lookup is wrong.

---

## Phase E — Finding 7 schema additions + `meta.paths` block

Additive only. New `meta.*` fields. Lowest risk in the series.

**Action:**
- Bump `ledger.meta.schema_version`.
- Document new fields in `docs/ledger_schema.md` (create if missing — the QA pass is the right moment).
- Codify the layout in `meta.paths` so downstream nodes stop reconstructing paths from `ep_id`:

```json
"meta": {
  "schema_version": "<bumped>",
  "paths": {
    "episode_root":    "<abs>/output/otr/episodes/<ep>",
    "audio_dir":       "<abs>/output/otr/episodes/<ep>/audio",
    "stills_dir":      "<abs>/output/otr/episodes/<ep>/stills",
    "portraits_dir":   "<abs>/output/otr/episodes/<ep>/portraits",
    "videos_dir":      "<abs>/output/otr/episodes/<ep>/videos",
    "composited_dir":  "<abs>/output/otr/episodes/<ep>/composited",
    "obs_final":       "<abs>/output/otr/obs/<ep>.mp4"
  }
}
```

Resolved absolute paths at write time. Relative paths are fragile across machines and across Desktop-vs-CLI launches. Readers can compute workspace-relative if they want.

Backward-compat: readers tolerate missing fields via `meta.get(key, default)`. Verify that pattern is used everywhere; no `meta[key]` direct subscripts on the new fields.

**Follow-up (additive, separate commit):** grep across the codebase for path reconstruction from `ep_id`; replace each consumer with `meta.paths[...]` lookup. Removes the slug-reconstruction temptation at its source.

**Acceptance:** an old ledger from before the bump still loads cleanly; a new ledger has the new fields populated; no downstream node throws `KeyError` on either shape.

---

## Operating cadence (per `CLAUDE.md`)

- After each phase: AST parse + the three regression suites (Bug Bible regression in survival-guide repo, `tests/test_dropdown_guardrails.py`, `tests/test_core.py`). Don't mark a phase done until green.
- Round-robin consult before Phase B and Phase D. Both touch determinism guarantees.
- Bug Bible promotion only after a real run confirms each fix, not after AST + tests pass.
- One `git push` attempt max per phase. cmd shell, not PowerShell.
- Save consult transcripts under `docs/<date>-<topic>/`.

---

## Load-bearing decision to confirm before Phase A merges

**Phase A's behavior change for legacy flat-layout episodes.** The drafted diff silently skips them. The earlier "existing leave as is, the code change is for new runs" position matches this, but it is worth one explicit confirmation before merge — once Phase A lands, no one can ever re-enable spacesaver for the legacy layout without reintroducing the global-scan code that caused the original bug. Any cleanup of a legacy-layout episode becomes a manual operation by design.

---

## References

- `nodes/rtx_upscale.py` — Finding 1 fix site
- `nodes/production_ledger.py` — Findings 2, 3 fix site
- `nodes/musicgen_theme.py` · `nodes/batch_audiogen_generator.py` — Finding 6 fix site
- `ROADMAP.md` — canonical going-forward plan; Sprint 1 priority
- `docs/BUG_LOG.md` — live bug tracking; new `BUG-LOCAL-NNN` entry per phase
- `CLAUDE.md` — project rules, regression contract, git pattern

---

## Shipped — appendix (2026-05-02 EVENING)

Stack landed autonomously per Jeffrey's mode change after Phase A. All five phases pushed to `origin/v2.0-alpha`, lockstep verified.

| Phase | Commit | BUG_LOG | New tests | Live regression | Consult |
|---|---|---|---|---|---|
| A — spacesaver wrong-episode + OBS guard | `d2c2df8` | BUG-LOCAL-014 | (covered by audit + regression suites) | 178 / 1 / 2 | round-A consult landed earlier in QA pass |
| B — production_ledger rename invariant | `29295c9` | BUG-LOCAL-015 | `tests/test_ledger_rename.py` (10) | 178 / 1 / 2 | `docs/2026-05-02-phase-b-rename-consult__*` |
| C — filename pattern audit guard | `3e1d995` | BUG-LOCAL-016 | `tests/test_filename_pattern_audit.py` (3) | 191 / 1 / 2 | none — mechanical audit |
| D — musicgen + audiogen cache key | `e43695d` | BUG-LOCAL-017 | `tests/test_cache_key_mutations.py` (30) | 221 / 1 / 2 | `docs/2026-05-02-phase-d-cache-key-consult__*` |
| E — schema bump + meta.paths block | `7c84ee8` | BUG-LOCAL-018 | `tests/test_meta_paths.py` (13) | 234 / 1 / 2 | none — additive only |

**Cumulative:** 5 commits, 56 new tests, 234 passing tests in the live regression suite, 5 BUG_LOG entries (`[FIXED]`, all "Bible candidate: yes (after end-of-stack soak)"), 5 consult bundles checked in under `docs/`, 1 schema doc (`docs/ledger_schema.md`), schema version bumped `l3-2026-04-28` → `l3-2026-05-02`.

**Deferred to soak:** Bible promotion for 014–018 waits on the two-episode-in-flight acceptance run. Bisect surface if soak fails: B → C → D → E (in suspect order; A is oldest and most-tested).

**What stood out:**

- **Phase C** found 0 substantive code changes — A and B had already absorbed the dangerous slug-reconstruction pattern. Codified the rule as a regression guard so future drift can't reintroduce it.
- **Phase D** consult earned its keep twice. Gemini caught the atomic-write requirement (`.tmp` + `os.replace` to prevent corrupt cache hits if killed mid-write). Mutation tests then caught two real bugs the consult missed: `soundfile` can't infer WAV format from `.tmp` extension (needed explicit `format='WAV'`), and a test premise bug putting `*` in a Windows filename (illegal). Both fixed before commit.
- **Phase E** was as predicted — additive, all readers already use `meta.get()`, no behavior change. 13 tests green first try.

The consult → mutation-test → ship loop worked as designed: consult surfaces what solo work misses; tests surface what the consult misses; the shipped commit is cleaner than any single layer would have produced.

---

## End-of-stack acceptance gate

Single two-episode soak. Queue Episode A in ComfyUI Desktop, queue Episode B before A reaches RTXUpscale, let both finish, paste console output. Per-phase grep targets:

| Phase | What to look for |
|---|---|
| A | `[OTR_RTXUpscale] spacesaver:` lines pointing at A's `ep_dir` for A's run, B's for B's. OBS-existence guard fires if either `obs/<ep>.mp4` is missing. |
| B | `[Ledger] per-episode dir moved ... (attempt 1)` once per episode. No orphan `<old>_*.txt` in any episode dir. No `[Ledger] meta.paths stamp failed` warnings. |
| C | No `f"{ep_id}_..."` paths in any logged disk-touching line. |
| D | `[MusicGenTheme] CACHE HIT` on second run with canonical `.wav` (no `_<ts>` suffix). Same for AudioGen. |
| E | New ledgers carry `meta.paths.layout: per-episode-workspace`. `meta.paths.audio_dir` matches actual on-disk location after rename. Old ledgers (if any) load via `dict.get` defaults without KeyError. |

Once green, BUG-LOCAL-014/015/016/017/018 promote to the Bible together.
