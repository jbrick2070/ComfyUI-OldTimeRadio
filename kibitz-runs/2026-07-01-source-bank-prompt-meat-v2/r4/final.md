# R4 Final: Focused Prompt/Python Convergence

Status: Codex-grounded R4. Use the companion update map as the coding handoff.

## Final Decision

Phase 2 is content-and-routing surgery:

- Same broad current story-builder for first transplant.
- Different prompt meat per source/story model.
- Different Python source routing where sci-fi/news assumptions are hardcoded.
- Exposed planned lanes, no hidden fallback.
- `simple_4_prompt_experimental` is included as a visible lab/selector
  experiment and tested against the same ledger contract.

## Code-Ready Output

See:

`docs/2026-07-01-source-bank-visual-style-code-ready/PHASE2_PROMPT_PY_UPDATE_MAP.md`

## Final Risks

- Current writer style picker falls back internally to first candidate after
  chooser failure. That behavior may remain only for current science/default,
  not for non-science source/story lanes.
- Render-driver visual fallback prompts are deep and should be staged after the
  story/source prompt pack work.
- Public-domain fidelity needs manifest/source loader tests before it is useful.

## Final Gate

No edit to `workflows/otr_scifi_16gb_full.json` in this phase.
