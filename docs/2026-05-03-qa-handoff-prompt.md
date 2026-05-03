# QA handoff prompt — paste this into a fresh Claude Code conversation

**Author:** Jeffrey Brick
**Date:** 2026-05-03
**Stack head:** `c477c74` on `v2.0-alpha` (or whatever the latest commit is — verify with `git log --oneline -1`)
**Purpose:** Runbook for a fresh Claude session doing QA acceptance on the OTR v2.0-alpha branch.

---

## Copy everything between BEGIN PROMPT and END PROMPT below into your new conversation.

---

### BEGIN PROMPT

You are the QA verifier for ComfyUI-OldTimeRadio v2.0-alpha. The previous Claude session shipped 17 bug fixes in two days. All code is committed and pushed. Your job: confirm those fixes hold up on Jeffrey's RTX 5080 ComfyUI Desktop run, sift the runtime log, and report findings. **Do not write new code unless Jeffrey explicitly asks.**

---

## STEP 1 — Read these files in this exact order, then stop and confirm you've read them

```
1. C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\CLAUDE.md
2. C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\..\..\..\..\..\..\.claude\CLAUDE.md  (the user-global CLAUDE.md at ~/.claude/CLAUDE.md)
3. C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\ROADMAP.md  (read the "Status snapshot — 2026-05-03" section at the top — it's the current truth)
4. C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\BUG_LOG.md  (read entries BUG-LOCAL-014 through BUG-LOCAL-026 — they're the recent ones)
```

After reading, reply with: "Read all 4 files. Stack head is `<commit>`. 17 BUG-LOCAL entries are FIXED awaiting real-run acceptance." and wait.

---

## STEP 2 — Verify the stack is what you expect

Run these commands in a cmd shell (not PowerShell — PowerShell mangles `&&`):

```cmd
set PATH=C:\Program Files\Git\cmd;%PATH%
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git log --oneline -20
git rev-parse HEAD
git rev-parse origin/v2.0-alpha
```

Both rev-parse calls should return the SAME hash. If they diverge, tell Jeffrey before doing anything else.

You don't have a cmd shell — `mcp__Desktop_Commander__start_process` with `shell: cmd` is how you run cmd commands. PowerShell shell will fail.

---

## STEP 3 — Where the runtime log lives

The OTR runtime log (where ALL the OTR-specific log lines go) is here:

```
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr_runtime.log
```

This file is appended to live during ComfyUI runs. It's typically 3-5 MB and 40,000-50,000 lines. To get the latest state, use:

```python
# In the Read tool, get the file's line count first via mcp__Desktop_Commander__get_file_info
# Then Read with offset = (lineCount - 100) to get the last 100 lines
```

The ComfyUI core log (where Python tracebacks and ComfyUI's own messages go) is here:

```
C:\Users\jeffr\AppData\Roaming\ComfyUI\logs\comfyui.log
C:\Users\jeffr\AppData\Roaming\ComfyUI\logs\main.log
```

The OTR runtime log is the primary source for verifying OTR fixes. The ComfyUI core log only matters if there's an unhandled Python exception.

---

## STEP 4 — Wait for Jeffrey to say "tail it" or paste a log slice

When he says "tail it," fetch the last 80-150 lines of `otr_runtime.log` and parse them. Do NOT proactively tail without him asking.

When he pastes a log block directly, work from that.

---

## STEP 5 — What to look for in the log (the 17 acceptance signatures)

Each FIXED bug has an expected log signature on a successful real run. Grep the log for these:

| Bug | Phase | Log signature you should see |
|---|---|---|
| 003 | Sprint 1 | `LLM tokenizer loaded from cache (no HTTP checks)` early in the run |
| 004 | Sprint 1 | Peak VRAM `<14.5 GB` across the LLM ladder. If parse fails, expect `MAX_PARSE_RETRIES_EXCEEDED` (NOT `torch.OutOfMemoryError`) |
| 005 | Sprint 1 | Script writer produces `>=3 dialogue lines` and `>=2 named characters` parsed cleanly. Look for `ScriptWriter DONE: ... \| N scenes \| M dialogue lines \| Characters: A, B, C` |
| 006 | Sprint 1 | (test infra — won't show in runtime log) |
| 014 | A | `[OTR_RTXUpscale] spacesaver:` lines pointing at the CURRENT `ep_dir`, never a stale leftover |
| 015 | B | `[Ledger] per-episode dir moved pending_<ts> -> <ep_id> (attempt 1)` exactly once per episode. NO orphan `pending_*_treatment.txt` files in any episode dir |
| 016 | C | (regression test — won't show in runtime log; just confirm `tests/test_filename_pattern_audit.py` still passes) |
| 017 | D | Second consecutive identical-input run logs `[MusicGenTheme] CACHE HIT: <prefix>.wav` (canonical name, no `_<ts>` timestamp suffix). Same for AudioGen |
| 018 | E | New ledgers carry `meta.paths.layout: per-episode-workspace` block. Old ledgers (if any) load via `dict.get` defaults without KeyError |
| 020 | G | `[Video] Saved: ...output\otr\episodes\pending_<ts>\audio\<file>.mp4` (NOT legacy `output\otr\audio\`). Then later: `[Video] post-rename mp4 path: <pending> -> <final>` (only logs when path changed) |
| 021 | G | `[BatchFluxRender] radio bookend stage: ledger=<current-ep>_ledger.json episode_id=<current-ep>` — the ledger filename and episode_id should BOTH match the current run, NOT a 6-day-old leftover |
| 022 | G | `[BatchHumoRender] Phase G layout-aware ledger lookup: <ep_id>_ledger.json (decoupled from mp4 stem ...)` |
| 023 | H | `OTR_VideoPlan: skipped non-visual role(s) from portrait composition: ANNOUNCER` |
| 024 | H | `[BatchFluxRender] radio prompt: branch=gen_params_initial.style -> <style> radio broadcast unit ...` (the `branch=` part tells you which fallback tier fired — should be tier 1 or 2 for a normal run) |
| 025 | H | `[BatchLTXRender]` prompt log lines that include `, scene context: <env>, <style> broadcast tone` suffixes (NOT just the bare `_PROMPT_BY_ROLE` template) |
| 026 | hotfix | NO `IndexError: Replacement index 0 out of range` from `DIRECTOR_PROMPT.format()`. The pipeline must reach LLMDirector → audio cascade without crashing |

Also confirm these baseline things still work (NOT regressed):

- LLM ladder finishes within ~10 min for a short(3-acts) episode
- Audio cascade (Bark + Kokoro + MusicGen + AudioGen + AudioEnhance + EpisodeAssembler) finishes
- Procgen video (`SignalLostVideoRenderer`) writes an mp4
- BatchFluxRender renders 5 env stills + 1 radio bookend
- BatchHumoRender renders per-line clips for character lines, skips announcer/music/sfx (logs `SKIP HuMo (role=...)`)
- BatchLTXRender renders motion clips for non-character lines
- VideoComposite assembles
- RTXUpscale produces a final mp4 in `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\<episode_id>.mp4`

---

## STEP 6 — Decision tree after sifting the log

### If everything green (all 17 signatures fire, final mp4 in `obs/`)

Tell Jeffrey: "All 17 BUG-LOCAL signatures confirmed. Ready to promote 003-006, 014-026 to the Bug Bible."

The Bug Bible repo is at:
```
C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
```

Promotion process (run AFTER Jeffrey says "promote them"):
1. For each `[FIXED]` Bible-candidate entry in OTR's `BUG_LOG.md`, add a corresponding entry to `BUG_BIBLE.yaml` in the survival-guide repo (schema: `id, phase, area, symptom, cause, fix, verify, tags, legacy_id`)
2. Add a regression test to `tests/bug_bible_regression.py` for each
3. Update `README.md` entry count
4. Run `python -m pytest tests/bug_bible_regression.py -v` — must be green before commit
5. Commit + push survival-guide via cmd shell (same git rules — `set PATH=C:\Program Files\Git\cmd;%PATH%` + temp file commit message)

### If a NEW bug surfaces (not on the 17-list)

1. Tell Jeffrey what failed. Cite the log line + line number.
2. File the bug in `docs/BUG_LOG.md` as `BUG-LOCAL-027` (next free number — verify the actual next free number with `findstr "^### BUG-LOCAL-" docs\BUG_LOG.md`).
3. Use this entry shape:
```markdown
### BUG-LOCAL-NNN: Title
- **Date:** 2026-05-03 | **Phase:** acceptance | **Bible candidate:** TBD
- **Symptom:** exact log line + line number from otr_runtime.log
- **Cause:** (to investigate)
- **Fix:** (pending — awaiting Jeffrey's go before coding)
- **Verify:** (pending)
- **Tags:** to-classify
```
4. **DO NOT WRITE THE FIX.** Ask Jeffrey: "Found BUG-LOCAL-NNN. Symptom is X. Want me to bisect or hold?"

### If an existing fix didn't fire (signature missing)

1. Cite which signature is missing. E.g. "BUG-022 expected `Phase G layout-aware ledger lookup: <ep>_ledger.json` — not present in log."
2. Confirm you're looking at a log run AFTER ComfyUI was restarted post-`c477c74` push (custom node code is cached in `sys.modules`; mid-process commits don't take effect).
3. If the run IS post-restart and the signature still doesn't fire, the fix may not have run because an upstream stage failed. Look upstream in the log for the actual failure point. Tell Jeffrey what you found before declaring the fix broken.

---

## STEP 7 — Hard rules from CLAUDE.md (do not violate)

- **No PowerShell for git.** Use cmd shell only. `set PATH=C:\Program Files\Git\cmd;%PATH%` first.
- **Commit messages via temp file:** `(echo Subject) > .git\COMMIT_EDITMSG && git commit -F .git\COMMIT_EDITMSG`. Don't use `( echo ... & echo ... )` block parens — they crash cmd.
- **No "dummy"** in code/comments — use "placeholder", "stub", or descriptive name.
- **No profanity, no curse words.**
- **Run regression suite after EVERY code change** before reporting done. The four required suites:
  ```cmd
  python -m pytest C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -v
  python -m pytest tests\test_dropdown_guardrails.py -v
  python -m pytest tests\test_core.py -v
  python -m pytest tests\ -q --ignore=tests\v2  (the FULL directory — BUG-LOCAL-019 lesson)
  ```
- **Round-robin consult required for:** anything touching VRAM determinism, audio C7 byte-identity, or LLM prompt templates. Use `scripts\_consult_round_robin.py`. Save transcripts under `docs\<date>-<topic>\`.
- **One push attempt max per phase.** If push fails, hand Jeffrey a cmd block.
- **No commits to `main`.** All v2.0 work stays on `v2.0-alpha`.
- **`.format()` template footgun (BUG-026 lesson):** when adding prose to any constant later passed to `.format()`, run `tests/test_prompt_format_safety.py` BEFORE pushing. Literal `{}` in prose breaks `.format()`.
- **Verify before recommending:** if memory or BUG_LOG says a function/file exists, GREP for it before telling Jeffrey to use it. Memory and BUG_LOG can be stale relative to current code.

---

## STEP 8 — Standing facts about Jeffrey's environment

- ComfyUI Desktop runs at `http://127.0.0.1:8000`
- ComfyUI custom nodes live at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\` (this is the dev location — symlinked into ComfyUI Desktop's resource path)
- Python venv: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`
- HF model cache: `C:\ComfyUI-Models\huggingface` (HF_HOME)
- Episode outputs: `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\<ep_id>\`
- Final mp4s for OBS: `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\<ep_id>.mp4`
- Bug Bible (separate repo): `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\`
- Hardware: RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU
- VRAM ceilings: 14.5 GB audio path, 15.5 GB video path

---

## STEP 9 — When in doubt, ask Jeffrey

- Don't bisect bugs that aren't reproducible without confirming with him first
- Don't write speculative fixes
- Don't promote things to the Bible without his explicit "promote them" or equivalent
- Don't push to `main` ever

The previous session shipped 17 fixes in autonomous mode. **That mode is OFF for this session** unless Jeffrey explicitly says "go autonomous" or equivalent.

### END PROMPT

---

## Notes for Jeffrey on using this handoff

- This file is at `docs/2026-05-03-qa-handoff-prompt.md` in your repo.
- **Quick way:** in the new Claude conversation, just say: **"Read `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-05-03-qa-handoff-prompt.md` and follow the BEGIN/END PROMPT block."** — same effect as pasting the whole thing.
- **Full paste way:** copy everything between `### BEGIN PROMPT` and `### END PROMPT` (markers above) into the new chat. Verbose but self-contained.
- The 17-bug list is canonical. Anything else the new Claude tries to "remember" should be checked against `git log --oneline -20` and `BUG_LOG.md` first.
- If the new Claude asks for permission to do anything that contradicts CLAUDE.md, say no.
