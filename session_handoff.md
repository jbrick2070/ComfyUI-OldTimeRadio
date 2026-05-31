# Session Handoff -- OldTimeRadio: OpenRouter Remote LLM -- 2026-05-31

## Core goal
Add OpenRouter as an **optional, non-local** LLM provider selectable on **both** writer slots (creative + technical). A go-forward plan is **locked and verified against live code**; this session produced the plan, not the code. The next session's job is to **execute that plan**: `docs/openrouter-remote-llm-go-forward-plan.md` is the single source of truth — read it first. (Supersedes the earlier "dual-route LLM" design handoff.)

## Tech stack & constraints
Existing repo (Python, ComfyUI custom node, RTX 5080 / 16 GB, Windows, `v2.0-alpha` branch). CLAUDE.md auto-loads — its git flow (Desktop Commander cmd, `.git\COMMIT_EDITMSG` + `-F`, one push attempt), VRAM ceiling, audio-king, "no dummy", and Bug-Bible-after-every-change rules are in force and are **not** repeated here. Work-specific hard rules (full detail in the plan): remote is **default-off, env-gated** (`OPENROUTER_API_KEY` + `OTR_ENABLE_OPENROUTER=1`); remote technical JSON is **fail-closed**; **no half-remote episodes** (abort, never mid-run fall-back); the remote branch **must not evict the resident local model** (C2); no new nodes, no new writer widgets, no `model_id` widget anywhere.

## Operator activation (Windows env vars)
`OPENROUTER_API_KEY` is **already set** in the User environment (2026-05-31); its value is not stored in any file. To enable remote and choose the A/B models, run via `setx`, then restart ComfyUI in a fresh terminal:

```
setx OTR_ENABLE_OPENROUTER 1
setx OPENROUTER_MODEL_A "anthropic/claude-3.5-sonnet"
setx OPENROUTER_MODEL_B "openai/gpt-4o"
```

Unset `OTR_ENABLE_OPENROUTER` (or set it to `0`) to disable remote entirely. Slugs are the operator's choice and swappable; verify current ids at openrouter.ai/models. None of S0–S3 or the mocked tests require these — only the enabled smoke run (S6 / W4) does.

## What's done & decided
- **Architecture locked: Option A.** Two virtual catalog rows `openrouter:slot-a` / `openrouter:slot-b` (`loader_backend="openrouter_http"`, new `provider` field), bound to real slugs by env (`OPENROUTER_MODEL_A/B`). They appear in both dropdowns only when enabled. No graph surgery.
- **Technical JSON: controlled T1, fail-closed.** Remote technical calls reuse the **existing** `structured_call` validate + bounded-repair ladder — zero new validation logic.
- **Three code-review refinements baked in** (marked `[code-review refinement]` in the plan): (1) the `LoaderBackend`/`BACKENDS_BY_KEY` dispatch table is **dormant** — remote must be wired into two live seams (`request_slot` + the generate-fn factory), not just "registered"; (2) **no `validate_model_id` surgery** — virtual rows join the curated set when enabled, so Path 1 admits them; (3) **no-evict rule** for the single-resident model cache.
- **Verified against live code this session** (2 subagents, all-PASS): dormant dispatch, skippable-for-remote steps, clean curated-set injection, generate-fn seam signatures, safe `provider` field, and the full S4 `structured_call` / `_parse_and_validate` / `make_constrained_generate_fn` surface.
- **Rejected:** Option B (writer config widgets — deferred; clean future upgrade, needs *no* B6 change), Option C (profile nodes — violates PD6 intent), Option D (raw slugs in dropdown), default-on cloud, mid-episode local fall-back, streaming.
- **No code written yet.** Nothing committed for this feature.

## State of the art
- `docs/openrouter-remote-llm-go-forward-plan.md` — **the plan to execute.** Decision, hard constraints C1–C9, frozen contracts FC1–FC5, the autonomous wave map (W0–W4), sprints S0–S6 with checkbox acceptance criteria, and a verified file:line anchor appendix.
- `docs/2026-05-31-openrouter-remote-llm__round-robin-locked.md` — raw round-robin output (provenance).
- `docs/2026-05-31-openrouter-remote-llm-architecture-options.md` — options analysis (superseded; background only).
- `docs/openrouter-setup.md` — **end-user** guide (account → key → `setx` → enable → use) with honest "unproven quality" framing. README carries an experimental pointer to it; promoting that pointer + adding an in-app hint (error + dropdown tooltip pointing users to the guide) is an **S6 deliverable** in the plan. **Also tracked in S6:** the README is stale overall and needs a full newbie-oriented refresh — audience is **ComfyUI beginners using AI coding assistants**, so low-jargon and copy-paste-first. The OpenRouter section is only one part of that refresh.
- Key live-code seams (verified; full list in plan appendix): `request_slot` `nodes/_otr_model_loader.py:712` (calls `load_llm` directly at `:812`); generate-fn factory `OTR_LedgerScriptWriter.py:586` + `_otr_model_loader.py:864`/`:939`; catalog `nodes/_otr_model_catalog.py` (`CuratedModel:53`, `CURATED_LLM_MODELS:96`, `_by_repo_id:195`, `validate_model_id:452`); structured calls `nodes/_otr_structured_call.py:293`; meta stamp `OTR_LedgerScriptWriter.py:3732`.

## Immediate next steps
1. Read `docs/openrouter-remote-llm-go-forward-plan.md` end to end; treat FC1–FC5 + C1–C9 as fixed.
2. Execute **W0 / S0 — baseline lock**: run the full regression set (`bug_bible_regression.py`, core tests, `test_audio_byte_identical.py`) + workflow JSON audits, confirm all green, freeze contracts, and make a clean checkpoint commit as the rollback point. Do **not** write feature code until the baseline is green.
3. **W1 (2 parallel subagents):** S1 `nodes/_otr_openrouter_backend.py` (mocked HTTP; prove the cost-ceiling abort with a mocked token counter) ∥ S2 catalog rows + `provider` field (enabled-gated). Merge → full regress → gate.
4. **W2 (solo):** S3 wire the remote branch into `request_slot` + generate-fn factory; enforce C2 no-evict. **W3 (parallel):** S4 fail-closed JSON ∥ S5 meta stamp. **W4 (solo):** S6 smoke proofs + final regress + Bug Bible.
5. **Run autonomously — no operator-approval gates.** Drive W0 → W3, plus S5 and the S6 *disabled* byte-identical proof, plus every mocked/unit test and the mocked cost-abort proof, committing per wave via the CLAUDE.md git flow (Desktop Commander cmd, `.git\COMMIT_EDITMSG` + `-F`, verify after each push). Halt only on a red regression you can't fix, a real ambiguity, or a C1–C9 breach.
6. **One human gate — leave for Jeffrey:** the S6 *enabled* live smoke (a real OpenRouter call) needs env vars not yet set (`OPENROUTER_API_KEY` is set; `OTR_ENABLE_OPENROUTER=1` + `OPENROUTER_MODEL_A/B` are not) and spends credits on a GPU episode run. Do everything up to it, then stop and report what's green; Jeffrey runs the enabled smoke when awake.

## Open questions
None blocking. The actual slugs `OPENROUTER_MODEL_A/B` point to are the operator's runtime choice and do not block any sprint. Option B remains an explicitly deferred future upgrade.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff and `docs/openrouter-remote-llm-go-forward-plan.md`, then execute W0–W3 (plus S5, the S6 *disabled* proof, and all regressions) autonomously, committing per wave. Do not wait for my approval between sprints — only stop on a red regression you can't fix, a real ambiguity, a C1–C9 breach, or the S6 *enabled* live smoke (which needs me). Post progress as you go and start now."
