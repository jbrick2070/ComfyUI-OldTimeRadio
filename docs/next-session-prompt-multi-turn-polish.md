# Next-session prompt — implement multi-turn polish

Branch `v2.0-alpha`. Follow CLAUDE.md + ROADMAP.md.

Inputs:
- `docs/2026-05-11-multi-turn-polish-problem-statement.md` (the spec)
- `docs/multi-turn-polish-adr.md` (the round-robin ADR — locked decisions)

For each phase the ADR defines, in order:

1. Code the phase.
2. Wire into `workflows/otr_scifi_16gb_full.json` (new widgets default OFF unless the ADR says otherwise).
3. Unit + regression tests in the existing test files.
4. Bug Bible regression must hold baseline (16/7/3xf) — stop and ask if it breaks.
5. AST parse clean across touched modules.
6. Commit via `.git\COMMIT_EDITMSG` file. Cmd shell only. Never PowerShell for commit messages.

Loop until all phases land. Push the full series to `origin/v2.0-alpha` at the end — one push, not per commit.

Don't flip any widget to default ON. Don't promote version labels — branch stays `v2.0-alpha` until I bump it.

After the series lands clean, write a QA handoff at `docs/2026-05-XX-multi-turn-polish-qa-handoff.md` modeled on `docs/2026-05-11-v4-composer-round-robin-handoff.md`. Include:

- per-commit change summary
- new widget surface (positions in `widgets_values`)
- pitfalls flagged for my QA
- 5-step manual QA recipe in ComfyUI Desktop
- file-list-to-serve guide for the next round-robin
- 4-5 concrete review prompts I can paste into ChatGPT + Gemini for the post-implementation review

Hand me the punch-list when done.
