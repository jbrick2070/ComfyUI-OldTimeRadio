# CLEANUP_PROPOSALS -- 2026-09-20 registry close-out (v2.3.0)

Lane: isolated worktree `otr-cleanup-lane` on `cleanup-20260920`.
Default was propose, not delete. Reachability is not permission. A symbol
nothing reaches may be correct code waiting to be wired.

`dead_code_closure.py` (read-only, its own contract): 41 candidates over 3
rounds; tests/ counted as ROOTS. Five printed as clean; the rest are named
in a protective doc. This lane deleted one clean leftover. Everything else
is below with a grep receipt.

## Done on this branch

| hash | files | why |
|---|---|---|
| `84a1fbc8` | `.gitignore`; 171 `*.log` under `docs/` and `kibitz-runs/`; `tmp/GPT_SHAKESPEARE_HUNT_3_RESULTS_VERIFIED.md` | Tracked scratch by shape. Already ignored (`*.log`, `/tmp/`) and still in the index. `git rm --cached`. Added `*.bak` and `*.orig` so those shapes cannot return. Did not touch `config/`, `nodes/`, `workflows/`, `tests/`, or `scripts/`. |
| `03e4b1f6` | `.gitignore` | `.comfyignore` already drops `.claude/` from the registry zip (agent exhaust). Now gitignored too. |
| `94a5a98a` | `nodes/_otr_line_composer.py` | `WORK_LINE_PREFIX` -- English leftover after the prefix moved to `spoken_chrome` / `config/episode_languages.json`. Grep hits: definition only. No ruling, inventory, or `PROD_BUG_LOG` names it. Wiring it back would re-pin English on every language. Tests that import the module: 412 passed, 1 skipped in 8.92s (20 modules). |
| `a1057d23` | `docs/2026-09-20-closeout/CLEANUP_PROPOSALS.md` | This receipt. Force-added because `docs/2026-*/` is already gitignored. |
| `fce42c04` | `docs/2026-09-20-closeout/CLEANUP_PROPOSALS.md` | Replaced the `REPORT_HASH` placeholder with `a1057d23`. |

The branch tip after these five is the commit that added the `fce42c04` row to this table.

## Proposed, not done

path | reason | the exact command
---|---|---
`tests/fixtures/baseline_v1.5.wav` (12,886,366 bytes; last commit 2026-04-30 `b09b3ced`) | Only tracked file over 5 MB. Versioned C7 byte-identical baseline; `.gitignore` already has `!tests/fixtures/baseline_v1.5.wav`. Not scratch. | leave it. If someone still wants it out: `git rm --cached -- tests/fixtures/baseline_v1.5.wav` then drop the `!` exception in `.gitignore`. That breaks the C7 suite.
`kibitz-runs/` (613 tracked files still in the index after the log untrack) | Directory is already gitignored and already `.comfyignore`d. Remaining files are panel transcripts, not `*.log`. | `git rm --cached -r -- kibitz-runs`
`docs/2026-*` (779 tracked files still in the index) | `docs/2026-*/` is already gitignored. The `docs-consolidation-20260920` lane owns these folders. This lane only untracked the `*.log` files inside them. | other lane. Do not `git rm --cached -r docs/2026-*` from here.
`docs/2026-*/**/*.xml` (41 tracked junit dumps; several ~2.3 MB) | Pytest XML under dated folders. Scratch by role, not by the Class A shape list. Already covered by `docs/2026-*/` once untracked. | `git ls-files "*.xml" | grep '^docs/' | git rm --cached --pathspec-from-file=-`
`nodes/cast_lock.py:_episode_language_iso` (line 49) | Clean candidate. Grep `nodes/ scripts/ tests/ config/ workflows/ docs/OTR_STANDING_RULINGS.md docs/PROD_BUG_LOG.md docs/2026-08-22-dead-symbol-inventory.md docs/DEAD_CODE_EXECUTION_PLAN.md docs/DEAD_CODE_HUNT_PROMPT_V5.md`: definition only. `_require_language_engines` at line 58 inlines the same `_EPLANG.iso_from_meta(...)` call. That is the waiting-to-be-wired shape. | wire: replace the inline `iso_from_meta` in `_require_language_engines` with `_episode_language_iso(meta)`. Or delete: remove lines 49-51 only after that call site is not the twin.
`nodes/cast_lock.py:_is_bark_namespace_preset` (line 205) | Clean candidate. Grep of the same set: definition only. Live twin at line 1971 is `leftover.startswith("v2/")`. Helper docstring: "Bark's live identity namespace." | wire: `if _is_bark_namespace_preset(leftover):` at `cast_lock.py:1971`. Or delete the unused def after that site uses it.
`scripts/otr_machine_matrix.py:headline_block` (line 615) | Clean candidate by the sweep. Grep: definition plus `inject_readme` comment at line 650, "See headline_block for why." That is a reference outside the definition. The function is the written reason README no longer carries the class table (now points at `apple/MACHINES.md`). This lane must not touch `README.md` or `apple/`. | leave it. If a later lane rips it: delete `headline_block` and the "See headline_block for why" sentence together, then run `tests/test_machine_matrix_drift.py`.
`nodes/_otr_image_engines/schemas.py:CanonicalImage` (line 57) | Round-2 orphan of the protected schema classes. Grep: defined at 57; used at 86 as `ImageLedgerSection.images: list[CanonicalImage]`. Also named in `docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md:48`. `ImageLedgerSection` is protected. | leave it. Deleting it breaks the protected class. No command.
`nodes/_otr_image_engines/schemas.py:ImageEngineConfig` (line 45) | Protective-doc hit. Grep: definition in `schemas.py`. Named in `docs/OTR_STANDING_RULINGS.md:1657` ("DELIBERATELY KEPT ... declared contract shapes") and `docs/2026-08-22-dead-symbol-inventory.md:61,121`. | leave it. No command.
`nodes/_otr_image_engines/schemas.py:ImageLedgerSection` (line 81) | Same ruling pair as `ImageEngineConfig`. Grep: definition plus the `CanonicalImage` field. | leave it. No command.
`nodes/_otr_scifi_news_pro.py:_validate_scene_envelope` (line 760) | Protective-doc hit. Grep in `nodes/`: definition only. Named in `docs/OTR_STANDING_RULINGS.md:1632` as a fail-closed validator that must stay. | leave it. No command.
`nodes/_otr_audio_cache.py:LEDGER_SCHEMA_VERSION_TARGET` (line 187) | Protective-doc hit. Grep in `nodes/` and `tests/`: definition only. Named in `docs/2026-08-22-dead-symbol-inventory.md:120` under "Still not cleared". | leave it. No command.
`nodes/_otr_shared/role_slots.py:PER_ROLE_VIDEO_SLOTS` (line 48) | Protective-doc hit. Grep in `nodes/` and `tests/`: definition only. Named in the inventory at lines 62 and 123 as a public-looking API that may still be a declared contract. | leave it. No command.
`nodes/_otr_shared/role_slots.py:NEW_ROUTE_A_VIDEO_SLOTS` (line 56) | Same inventory row as `PER_ROLE_VIDEO_SLOTS`. Grep in `nodes/` and `tests/`: definition only. | leave it. No command.
`nodes/_otr_scifi_p0_contract.py` (whole module) | `PROTECTED_MODULES` in `dead_code_closure.py`: every symbol stays -- it is the evidence for OPEN PBUG-20260729-03. Sweep named: `MAX_CLAIM_CHARS`, `MAX_ENTITY_NAME_CHARS`, `MAX_ENTITY_ROWS`, `MAX_FACT_ROWS`, `MAX_FAILED_ARTIFACT_ECHO_CHARS`, `MAX_NUMBER_ROWS`, `MAX_NUMERIC_TOKEN_CHARS`, `MAX_PROVENANCE_NOTE_CHARS`, `MAX_SPANS_PER_EVIDENCE_ROW`, `MAX_TONE_CHARS`, `MIN_EVIDENCE_FIELD_CHARS`, `P0RepairTrimReceipt`, `P0_REPAIR_CONTEXT_MAX_BYTES`, `_P0_FIT_MARGIN_TOKENS`, `_P0_LOCAL_CONTEXT_CAP`, `_P0_TOKENS_PER_CHAR`, `_TRIM_MARK`, `_render_p0_repair_context`, `_trim_evidence_to_fit`, `compact_p0_repair_context`, `p0_contract_instruction`, `p0_contract_receipt`, `p0_source_char_budget`. | leave the module. No command. Do not delete a symbol from this file.
`BUG_LOG.md` (last commit 2026-06-14 `2991d142`) | Archived v1 bug log. Marked archive in that commit. Live log is `docs/PROD_BUG_LOG.md`. | `git rm -- BUG_LOG.md` only if the operator wants the archive off `main`. Prefer leave; GitHub history already has it.
`BUG_LOG_2026-06.md` (last commit 2026-07-05 `8da76394`) | Next archive slice. Same story. `.comfyignore` already drops both from the zip. | `git rm -- BUG_LOG_2026-06.md` only on an operator call.
`api_key.location.example` (last commit 2026-09-19 `5c898ec3`) | Template for the local key-path file. The renamed copies are gitignored (`*_api_key.location`). Needed on a fresh clone. | leave it. No command.
`cb_contract.json` (last commit 2026-09-09 `9abdebf0`) | Tiny decl/widget contract snapshot (`video_path`, caption widgets, ledger/output paths). Not imported by `nodes/` (grep the filename before ripping). | confirm zero runtime readers, then `git rm -- cb_contract.json`
`prestartup_script.py` (last commit 2026-09-17 `259d1faa`) | Live ComfyUI prestartup hook. Comfy loads it by filename before any node import. Not stale. | leave it. No command.
`session_handoff.md` (last commit 2026-09-11 `424738d5`) | Stale 2026-09-11 Opus handoff. Pins HEAD `9367f2fd` on retired `v2.0-alpha`. `.comfyignore` already drops it from the zip. | `git rm -- session_handoff.md`
`scripts/_otr_dia_worker.py` | Ignore-consistency DEFECT, not a delete. `.gitignore` has `scripts/_*.py` and does NOT list `!scripts/_otr_dia_worker.py`. `nodes/_otr_audio_engines/eng_dia.py:86` joins that path at runtime. File is currently tracked, so clones work; a recreate cannot be `git add`ed without `-f`. Chatterbox and IndexTTS2 workers already have `!` exceptions. `.comfyignore` already re-includes this worker. | report only. Suggested fix (not done): add `!scripts/_otr_dia_worker.py` under the existing worker exceptions in `.gitignore`.

## Untracked files at the root

Count: 0

`git ls-files --others` (including ignored) at the worktree root returned no root-level files. `git status --porcelain` showed no `??` entries. No action. Operator untracked files elsewhere were not touched.

## Ignore consistency

### Comfyignored and also junk -- should be gitignored

| pattern | before this branch | after |
|---|---|---|
| `kibitz-runs/` | already in both | unchanged |
| `tmp/` / `/tmp/` | already in both | unchanged |
| `secrets/` | already in both | unchanged |
| `_tmp_*` / `scripts/_tmp_*` | already in both | unchanged |
| `CLAUDE.md` | already in both (still tracked; ignore does not untrack) | unchanged |
| `/AGENTS.md` | already gitignored; file is not on disk in this worktree | unchanged |
| `.claude/` | in `.comfyignore` only | **gitignored this branch** (`03e4b1f6`) |

Not junk (comfyignored so the zip stays small; belong in git): `tests/`, `.github/`, `.cursor/`, `scripts/*` (dev harness), `docs/`, `tools/`, `viewer/`, `assets/`, `kibitz-plugin/`, `nodes/_otr_audio_engines/eng_indextts2.py`, `.gitignore`, `.comfyignore`, `.gitattributes`, `workflows/variants/*.md`.

### Gitignored and imported at runtime -- defects, not fixed

| path | who imports it | note |
|---|---|---|
| `scripts/_otr_dia_worker.py` | `nodes/_otr_audio_engines/eng_dia.py:86` (`os.path.join(_REPO_ROOT, "scripts", "_otr_dia_worker.py")`) | Covered by `scripts/_*.py`. No `!` exception. Tracked today, so a clone is fine. Report only. |
| `scripts/_otr_chatterbox_worker.py` | `nodes/_otr_audio_engines/eng_chatterbox.py:84` | Has `!scripts/_otr_chatterbox_worker.py`. Not a defect. |
| `scripts/_otr_indextts2_worker.py` | `nodes/_otr_audio_engines/eng_indextts2.py:184` | Has `!scripts/_otr_indextts2_worker.py`. Not a defect. `.comfyignore` excludes this engine from the zip on purpose. |
| `config/news_history.json` | `nodes/story_orchestrator.py` reads it as a legacy fallback path | Intentional per-machine state (BUG-LOCAL-090). Missing file is handled. Not a defect. |

No test module imports a gitignored module name. Tests that `import scripts.otr_*` hit files the `scripts/_*.py` rule does not cover.

## Worktrees and branches

No action. Captured from this worktree after the class commits, before this file's own commit.

```
C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio                                              f1f6a840 [main]
C:/Users/jeffr/Documents/ComfyUI/_worktrees/otr-cleanup-lane                                                    94a5a98a [cleanup-20260920]
C:/Users/jeffr/Documents/ComfyUI/_worktrees/otr-docs-lane                                                       0b0bcd0c [docs-consolidation-20260920]
C:/Users/jeffr/Documents/ComfyUI/_worktrees/otr-promptv3                                                        811ee0b4 [prompt-v3]
C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/.claude/worktrees/awesome-brahmagupta-a509b4 acb09719 (detached HEAD)
C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/.claude/worktrees/ecstatic-diffie-ec1cfb     0aa6d6e1 (detached HEAD)
```

```
  claude/awesome-brahmagupta-a509b4
  claude/ecstatic-diffie-ec1cfb
* cleanup-20260920
+ docs-consolidation-20260920
  feat/otr-video-platform
+ main
+ prompt-v3
  v2.0-alpha
  remotes/origin/HEAD -> origin/main
  remotes/origin/archive/main-v1.7
  remotes/origin/claude/new-session-bq0dsx
  remotes/origin/main
  remotes/origin/s25-musicgen-parity
  remotes/origin/s26-cleanbreak
  remotes/origin/s27-cleanbreak-tail
  remotes/origin/s28-cleaner-break
  remotes/origin/s29-clean-slate-gate
  remotes/origin/s30-two-model-selector
  remotes/origin/s31-loader-clean-break
  remotes/origin/s31p5-legacy-residue-cleanup
  remotes/origin/s32-helper-per-subpass-routing
  remotes/origin/s33-editor-only-cleanup
  remotes/origin/s34-p0-p1-hotfix
  remotes/origin/sprint-c-story-brief-v2
  remotes/origin/sprint-d-period-llm
  remotes/origin/sprint-e-distillation-roundup
  remotes/origin/triage-sprint-c-retrospective-2026-05-15
  remotes/origin/v2.0-alpha
  remotes/origin/v2.0-alpha-stable-2026-05-05
```
