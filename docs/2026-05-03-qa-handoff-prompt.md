# QA handoff prompt — paste this into a fresh Claude Code conversation

**Author:** Jeffrey Brick
**Date:** 2026-05-03
**Stack head:** `03dfbfa` on `v2.0-alpha`
**Purpose:** Hand the OTR v2.0-alpha codebase off to a fresh Claude session for QA acceptance testing without losing context.

---

## Copy everything between the "BEGIN PROMPT" and "END PROMPT" markers below into your new conversation.

---

### BEGIN PROMPT

You're picking up the OTR (ComfyUI-OldTimeRadio) v2.0-alpha branch in the middle of a QA acceptance cycle. The previous Claude session shipped 17 bug fixes in two weeks of context. All code is committed and pushed to `origin/v2.0-alpha`. Your job is **acceptance verification** — confirm the fixes hold up under a real ComfyUI Desktop run, sift the runtime log for the expected signatures, and only ship NEW code if a real bug surfaces that wasn't caught.

**Project root:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
**Stack head:** `03dfbfa` (verify with `git log --oneline -1` from a cmd shell with `set PATH=C:\Program Files\Git\cmd;%PATH%`)
**Standing rules:** see `CLAUDE.md` at the project root and `~/.claude/CLAUDE.md` for global rules. Read both before doing anything.

**First moves this session:**

1. Read `CLAUDE.md` (project standing rules).
2. Read `ROADMAP.md` — the "Status snapshot — 2026-05-03" section at the top is the current state.
3. Read `docs/BUG_LOG.md` entries BUG-LOCAL-014 through BUG-LOCAL-026 (the 13 most recent). Skim earlier entries for context.
4. Run `git log --oneline -20` in cmd to see the commit shape.
5. **Do not start coding until Jeffrey gives a target.** This is a QA pass, not a fresh sprint.

**The 17 fixes shipped this cycle (all `[FIXED]` in BUG_LOG, all need real-run confirmation):**

| Bug | Phase | Commit | Quick description |
|---|---|---|---|
| 003 | Sprint 1 | (mega) | `scripts/run_comfyui.cmd` HF_HOME inheritance |
| 004 | Sprint 1 | (mega) | LLM script-writer OOM fix |
| 005 | Sprint 1 | (mega) | 30-word preset CHARACTER:/SCENE: enforcement |
| 006 | Sprint 1 | (mega) | `tests/conftest.py` CUDA mask |
| 014 | A | `d2c2df8` | Spacesaver wrong-episode wipe |
| 015 | B | `29295c9` | production_ledger rename invariant |
| 016 | C | `3e1d995` | Filename pattern audit guard |
| 017 | D | `e43695d` | MusicGen/AudioGen cache key fix |
| 018 | E | `7c84ee8` | Schema bump + meta.paths block |
| 019 | (cleanup) | `ca85a01` | Test-suite cleanup, `[PARTIAL]`→`[FIXED]` for 006 |
| 020 | G | `1fabd5c` | video_engine procgen path (SOAK BLOCKER from prior run) |
| 021 | G | `1fabd5c` | Audio-side singleton sweep |
| 022 | G | `1fabd5c` | BatchHumoRender layout-aware ledger lookup |
| 023 | H | `5075b9e` | Skip ANNOUNCER from FLUX portraits |
| 024 | H | `5075b9e` | Radio bookend story-arc fallback chain |
| 025 | H | `5075b9e` | LTX role prompts get scene + style enrichment |
| **026** | **G/H hotfix** | **`03dfbfa`** | **DIRECTOR_PROMPT.format crash from Phase H unescaped braces** |

**Real-run acceptance signatures to grep for in `otr_runtime.log` after Jeffrey's next soak run:**

- `[Video] Saved: ...output/otr/episodes/pending_<ts>/audio/<file>.mp4` (NOT legacy `output/otr/audio/`) — confirms BUG-020 fix
- `[Ledger] per-episode dir moved pending_<ts> -> <ep_id> (attempt 1)` — confirms Phase B rename
- `[Video] post-rename mp4 path: <pending> -> <final>` — confirms BUG-020 post-rename recompute (only logs if path actually changes)
- `OTR_VideoPlan: skipped non-visual role(s) from portrait composition: ANNOUNCER` — confirms BUG-023
- `[BatchFluxRender] radio prompt: branch=gen_params_initial.style -> <style> radio broadcast unit ...` — confirms BUG-024 + Phase G singleton
- `[BatchFluxRender] radio bookend stage: ledger=<current-episode>_ledger.json episode_id=<current-episode>` — same `episode_id` as the in-flight run, NOT a stale leftover
- `[BatchHumoRender] Phase G layout-aware ledger lookup: <ep_id>_ledger.json` — confirms BUG-022
- `[BatchLTXRender]` prompt log lines containing `, scene context:` and `, <style> broadcast tone` suffixes — confirms BUG-025
- `MAX_PARSE_RETRIES_EXCEEDED` (NOT `torch.OutOfMemoryError`) on parse-fail edge cases — confirms BUG-004
- `LLM tokenizer loaded from cache (no HTTP checks)` early in the run — confirms BUG-003
- Peak VRAM `<14.5 GB` across the LLM ladder — confirms BUG-004

**What to do when Jeffrey pastes a runtime log tail:**

1. Search for the signatures above. Note which fired and which didn't.
2. If everything green, propose Bug Bible promotion of 003-006 + 014-026 as a single batch (17 entries). The Bug Bible repo is at `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`.
3. If a NEW bug surfaces, file BUG-LOCAL-027+ in `docs/BUG_LOG.md` and ask Jeffrey before fixing.
4. Don't re-ship existing fixes — they're committed.

**Hard rules from CLAUDE.md you MUST follow:**

- **No PowerShell for git** — use cmd shell only. PowerShell mangles `&&` and commit message quoting.
- **Use temp file for commit messages:** `echo Subject> .git\COMMIT_EDITMSG && git commit -F .git\COMMIT_EDITMSG`. Cmd block parens crash.
- **Never use the word "dummy"** — use "placeholder" or "stub" or a descriptive name.
- **Safe for work, no profanity** in code/comments/output.
- **Run regression after every code change** before reporting done. The three suites are: Bug Bible regression in survival-guide repo, `tests/test_dropdown_guardrails.py`, `tests/test_core.py`. **AND `pytest tests/ -q` for the full directory** (BUG-LOCAL-019 lesson — the three named suites missed 5 latent failures).
- **Round-robin consult required for:** anything touching VRAM determinism, audio C7, or LLM prompt templates. Skip for mechanical edits and obvious typo fixes.
- **One push attempt max per phase.** Then hand a cmd block to Jeffrey.
- **`scripts/_consult_round_robin.py`** is the round-robin driver. Save consult transcripts under `docs/<date>-<topic>/`.
- **No new commits to `main`.** All v2.0 work stays on `v2.0-alpha`.
- **template `str.format()` footgun (lesson from BUG-026):** when adding prose to any `*_PROMPT` constant that's later passed to `.format()`, run `tests/test_prompt_format_safety.py` BEFORE pushing. Literal `{}` in prose breaks `.format()` with `IndexError`.

**The critical real-run sequence Jeffrey will likely run:**

1. **Restart ComfyUI Desktop** — custom node code changes don't hot-reload. Quit completely (close window, kill process), re-launch via `scripts\run_comfyui.cmd`.
2. Pick a richer episode title than "Test" (e.g. "Cold Circuit", "Deep Signal", "The Last Broadcast"). Empty title also works — pulls from news headlines.
3. Queue. Wait.
4. Paste runtime log tail when ready.

**Cadence reminder:** the previous session ran in autonomous mode for the 17 fixes. After this real-run, autonomous mode is OFF — Jeffrey will explicitly opt back into it before the next sprint.

### END PROMPT

---

## Notes for Jeffrey on using this handoff

- This file is itself committed at `03dfbfa+` (or whatever commit you push the handoff at).
- If you start a new session and don't want to paste the whole prompt, just tell the new Claude: **"read `docs/2026-05-03-qa-handoff-prompt.md` and follow the BEGIN/END PROMPT block."** Same effect.
- The cumulative 17-bug list is canonical. Anything else the new Claude tries to "remember" should be checked against `git log` and `BUG_LOG.md` first.
