# R3 Final: Wiring And Transplant Convergence

Status: Codex-grounded R3.

## Verdict

Do not wire the canonical workflow yet. Prepare the transplant by making every
future widget/control explicit and append-only.

## Future Widget/Selector Surface

Append, never insert:

- `source_bank`
  - `science_news`
  - `media_archive`
  - `public_domain_story`
  - `custom_source_bank`
- `story_model`
  - source-scoped choices; invalid pairs fail loudly
- `story_pipeline`
  - `legacy_many_pass`
  - `simple_4_prompt_experimental`
  - later `lean_5_prompt` / custom lab pipelines
- `visual_style`
  - `sci_fi_radio`
  - `archival_documentary`
  - `cinematic_35mm`
  - `noir`
  - `anime`
  - `cartoon`
  - `paper_origami`

## Wiring Rules

- Source/story/pipeline controls feed the writer upstream only.
- Visual style feeds ledger/meta visual policy and visual prompt seams.
- `media_archive` source bank is not the same thing as
  `archival_documentary` visual style.
- Whitelists must be updated together:
  - `nodes/_otr_workflow_apply.py`
  - `scripts/otr_api.py`
  - `tests/test_workflow_apply.py`
- Workflow JSON update happens only after:
  - prompt-pack tests pass
  - no-fallback resolver tests pass
  - leakage previews pass
  - widget append plan is reviewed

## Reject

- Hidden selectors.
- Silent science fallback.
- Duplicate whole downstream workflow per story model as the default design.
- Runtime reads from `upstream_story_lab`.
