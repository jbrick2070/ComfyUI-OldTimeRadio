# OldTimeRadio — Rules of the Road

These are the permanent operating rules for any AI assistant working on this project.
Violating these rules is grounds for immediate rollback.

---

## 1. Branching and Versioning

- **All v1.5 work lives on a beta branch** (`v1.5-audiogen-dsp`), never on `main`, until Jeffrey says ship.
- **No point releases.** Stay at v1.5 — no 1.5.1, no 1.5.2. Just keep iterating.
- **No changelogs.** Keep the `ROADMAP.md` updated with current status instead.
- **Only Jeffrey ships.** Merge to `main` and tag a release only when Jeffrey personally confirms.

---

## 2. Git Push Protocol

Jeffrey must never be the middleman for a push. Exhaust every autonomous option first.

**Push escalation order — stop at the first one that works:**

1. **GitHub API via Python** (most reliable from sandbox):
   ```python
   import json, urllib.request
   TOKEN = "<token>"  # ask Jeffrey once per session if not in context
   # Create tree → create commit → PATCH refs/heads/<branch>
   ```
2. **Windows MCP PowerShell** (`mcp__Windows-MCP__PowerShell`) — run the git commands directly on Jeffrey's machine. Set `GIT_TERMINAL_PROMPT=0` and embed the token in the remote URL to avoid credential prompts.
3. **Bash sandbox git** — works if the repo is mounted and credentials are cached.
4. **Last resort only:** hand Jeffrey a PowerShell block. This should be rare. If it happens, note in COLLECTIVE_CONSCIOUSNESS.md why all three methods failed so the next session can fix the root cause.

**After every push, always run the integrity checklist:**
   - Verify local HEAD matches `origin/main` (or the target branch) — lockstep check.
   - Scan for **0-byte files** in the repo.
   - Scan for **BOM signatures** (UTF-8 only, no BOM on Windows).
   - Check for **truncated files** (compare expected line counts).
   - Verify **all node classes are registered** in `__init__.py` `NODE_CLASS_MAPPINGS`.
   - Check for **missing widget mapping blocks** in workflow JSON files.
4. **If any check fails**, fix it and repeat from step 1.
5. **Do not declare done until lockstep is confirmed.**

---

## 3. Regression Testing (Non-Negotiable)

Every code change must pass a regression sweep before being considered complete:

- **Run the full test suite** (`tests/test_core.py`, `tests/vram_profile_test.py`).
- **Widget error testing**: Verify all nodes load without widget errors in ComfyUI. Check that all `INPUT_TYPES` return valid type specs and that workflow JSONs have explicit `{"widget": "string"}` mapping blocks for string inputs.
- **Bug Bible regression**: Cross-reference changes against the Bug Bible:
  - Repo: https://github.com/jbrick2070/comfyui-custom-node-survival-guide
  - File: `BUG_BIBLE.yaml`
  - Key bugs to always check: BUG-010 (0-dialogue hard abort), BUG-12.33 (Oversized prompt pre-fill stall/VRAM spike), widget connection drops, VRAM spills, cache deadlocks.
- **VRAM ceiling**: Peak VRAM must stay at or below 14.5 GB. Run `vram_profile_test.py` to confirm.
- **VRAM & Context Etiquette**:
  - **Never use `force_vram_offload()` between LLM phases** within the same script-writing pass. Use `_flush_vram_keep_llm()` to retain weights while clearing KV cache.
  - **Always enforce Prompt Truncation** against `context_cap` before calling `model.generate()`. If prompt > cap, truncate head.
  - **Warmup Mandatory**: All LLM loaders must perform a 1-token warmup pass to front-load CUDA kernel JIT compilation.
- Create **scratch scripts** (in `scratch/` directory) to help validate edge cases. These are disposable — delete when done.

---

## 4. Bug Log and QA Guide Lifecycle

- **No QA guides or bug logs exist unless there are active bugs.** Zero stale docs.
- If the **same bug appears 3 times**, create a formal bug log file (e.g., `BUG_LOG_v1.5.md`) in the repo root.
- A QA guide may be created alongside the bug log to document testing protocols for that specific bug.
- The bug log exists for **peer review** — Jeffrey may share it with reviewers.
- **Delete both the bug log AND the QA guide** once the bug is resolved and the fix is verified.
- The Bug Bible (`BUG_BIBLE.yaml` in the survival guide repo) is the permanent record. Local QA docs are temporary.

---

## 5. Code Quality Standards

- **Clean code.** Every function, variable, and file should make sense to a reader on first pass.
- **Clean logs.** Log messages should be informative, structured, and greppable. No noise, no spam.
- **Meaningful comments.** Explain the *why*, not the *what*.
- **Perfection is the target.** If something looks half-done, finish it.

---

## 6. Content Standards

Generated stories and narrative content must follow these rules:

- **Safe for work.** No explicit, violent, or disturbing content.
- **No profanity.** Zero tolerance for curse words in code, comments, logs, or generated output.
- **Good narrative structure.** Every story should have a clear beginning, middle, and end with a satisfying arc.
- **Non-violent.** Drama and tension are fine, but no graphic violence.

---

## 7. Security Awareness

- **Assume compromise.** Always verify that the right files are updated and that nothing unexpected has changed.
- **After every push**, visually confirm on GitHub (via web) that the pushed files match what was intended.
- **Check file integrity** — 0-byte files, unexpected modifications, or missing files could indicate a problem.

---

## 8. Active Agents on This Project

This project runs a multi-agent setup. Any AI assistant reading this file is one of these agents.

| Agent | Platform | Role | Prompt File | Mirror Log |
|-------|----------|------|-------------|------------|
| **Claude** | Anthropic / Cowork | Principal Architect — design, refactors, bug triage, narrative critique | This file (`CLAUDE.md`) + `COLLECTIVE_CONSCIOUSNESS.md` | `logs/claude_mirror.log` |
| **AntiGravity** | Google / Gemini | Autonomous Pipeline Operator — runtime monitoring, live repair, QA, execution | `ANTIGRAVITY_PROMPT.md` | `logs/antigravity_mirror.log` |

**Shared communication file:** `COLLECTIVE_CONSCIOUSNESS.md` — read it before doing anything. Append to Section 5 after every fix cycle.

**Protocol spec:** `agent-handshake/PROTOCOL.md` (v1.1) — the universal rules both agents follow.

### Updating Agent Prompts

Prompt files (`CLAUDE.md`, `ANTIGRAVITY_PROMPT.md`) are **human-controlled**. Agents do not edit each other's prompt files directly — too risky.

If an agent needs a prompt update (new path, new rule, new capability), it logs a suggestion in `COLLECTIVE_CONSCIOUSNESS.md` Section 5:
```
[DATE] [agent] PROMPT UPDATE SUGGESTION: <file> — <what to change and why>
```
Jeffrey reviews and applies it. This keeps prompt changes intentional and auditable.

### Adding a New Agent

If a third agent joins (Cursor, Codex, Windsurf, etc.):
1. Add it to the AGENTS table above.
2. Create `<agent_name>_PROMPT.md` using `agent-handshake/templates/prompt_operator.md` as the base.
3. Add its mirror log path to the table.
4. Log the addition in `COLLECTIVE_CONSCIOUSNESS.md` Section 5.
5. Run `agent-handshake/scripts/discover_agents.py --init .` to update `HANDSHAKE.md`.

---

## 9. Reference Links

| Resource | URL |
|----------|-----|
| **OldTimeRadio Repo** | https://github.com/jbrick2070/ComfyUI-OldTimeRadio |
| **agent-handshake** | https://github.com/jbrick2070/agent-handshake |
| **Bug Bible (Survival Guide)** | https://github.com/jbrick2070/comfyui-custom-node-survival-guide |
| **Bug Bible YAML** | https://github.com/jbrick2070/comfyui-custom-node-survival-guide/blob/main/BUG_BIBLE.yaml |
| **ROADMAP** | `ROADMAP.md` (in this repo) |

---

## 9. Integrity Checklist (Run After Every Push)

```
[ ] Local HEAD == origin HEAD (lockstep)
[ ] No 0-byte files in repo
[ ] No BOM signatures (UTF-8 only)
[ ] No truncated files
[ ] All node classes registered in __init__.py NODE_CLASS_MAPPINGS
[ ] All workflow JSONs have widget mapping blocks for string inputs
[ ] vram_profile_test.py passes
[ ] Full test suite passes (89+ tests)
```