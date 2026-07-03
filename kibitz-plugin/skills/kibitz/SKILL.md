---
name: kibitz
description: >-
  Kibitz gets you a second opinion on a plan, sprint plan, spec, or architecture doc
  using local file-reading CLI agents (Codex and Claude Code) when the Antigravity UI
  is the driver. The Antigravity lane is the live UI session, never an `agy` CLI
  subprocess launched from the base system.
  Use when the user wants a second opinion / to pressure-test / harden / "round-robin" /
  "make bulletproof" a doc with their LOCAL agents, or says "kibitz", "/kibitz",
  "kibitz this", "get a second opinion", "run the local panel", or "use Codex, Claude,
  and Antigravity to review this".
---

# Kibitz Roundtable (Antigravity UI-driven)

When you (Antigravity UI) are the active driver and are asked to run a Kibitz round, you write the grounded anchor review and act as the judge/synthesizer.

## Core Rules

1. **Active Driver Role**: You (Antigravity UI) write the initial grounded anchor review and serve as the final judge/synthesizer.
2. **UI-only Antigravity lane**: Do NOT run a base-system CLI roundtable for Antigravity. Do NOT spawn `agy`, `--only agy`, `--all-agents`, or any second Antigravity CLI reviewer from this plugin. The live Antigravity UI is the Antigravity panelist.
3. **External Reviewers**: Codex (`codex exec`) and Claude Code (`claude -p`) run as the only external CLI reviewers.
4. **No Duplicate Driver**: Do NOT duplicate the active Antigravity UI driver through `agy` CLI. If the user says "all three", treat that as Antigravity UI anchor + Codex + Claude Code.
5. **Command Pattern**: Run the following command for r1:
   ```bash
   python C:\Users\jeffr\.codex\skills\kibitz\scripts\kibitz.py --doc <plan.md> --round r1 --repo <repo> --driver agy
   ```
6. **Round Selection**:
   - For `r2`, `r3`, and `r4`, change only the `--round` value (e.g., `--round r2`).
   - Run all 4 rounds in order: r1 (high-level) -> r2 (coding plan) -> r3 (wiring) -> r4 (convergence).
7. **If the user asks for an Antigravity CLI reviewer**:
   - Do not launch it from this plugin. Explain that Antigravity is already present through the UI driver, then continue with Codex + Claude Code unless the user explicitly moves the work to a non-UI Kibitz context.

## Step-by-Step Loop

For each round `r1` to `r4`:

1. **Write Anchor Review**: Read the real source files in the repo. Write your own grounded anchor review of the current plan (following the structure in the corresponding round prompt file under `C:\Users\jeffr\.codex\skills\kibitz\references\review-prompt-r<N>.md`). Label every claim as CONFIRMED, MISREAD, or UNVERIFIABLE against the actual files.
2. **Fan Out**: Execute the `kibitz.py --driver agy` command above to generate reviews from Codex and Claude Code only. Do not invoke `agy` CLI from the base system.
3. **Ground and Verify**: Read the generated reviews at `<repo>/kibitz-runs/<date>-<topic>/<round>/<agent>.md`. Ground and verify every claim made by Codex and Claude Code against the actual files. Discard any hallucinations or misreads.
4. **Synthesize**: Merge your anchor review with the verified claims from the other agents. Save the resulting updated plan as `r<N>_plan.md` (or `final.md` for `r4`). Keep a short judgment log of accepted, rejected, and verify-at-build items.
