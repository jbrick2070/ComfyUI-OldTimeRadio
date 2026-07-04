# R1 Final: Architecture And Top-Level Schema

Status: high-level arc synthesis for R2 coding review.

## Verdict

Architecture is sound and now clear:

```
ledger_writing_spec -> existing multi-stage writer -> production ledger
visual_style_policy -> meta.visual_style + visual ledger direction
```

There is one production ledger. The new upper layer is a ledger-writing spec,
not a second downstream ledger. It selects source intent, prompt profile,
source packet/provenance, adaptation mode, visual bible, and validation
expectations.

Build this upper layer as fresh upstream code. Do not surgically mutate every
legacy prompt path first. The clean path is:

```
fresh source brain / ledger-writing spec -> translator/adapter -> existing production ledger
```

When the new upstream path is ready, it translates into the existing ledger
contract and downstream pipeline.

## Core Invariant

Reuse the existing multi-stage story/ledger structure wherever possible.

Source banks change:

- source material
- source packet/provenance
- source intent
- source-specific prompt variables
- source-fidelity expectations

Source banks do not create separate downstream ledger schemas.

Visual styles change:

- `meta.visual_style`
- visual ledger direction
- still prompt language
- video prompt language

Visual styles do not rewrite source facts, dialogue contracts, or source-story
fidelity.

## Accepted Design Corrections

1. All ledger-filling prompts should become profile-aware variables where they
   contain source-specific language.
   - Keep source-neutral radio-drama craft shared.
   - Move only science/news/archive/PD-specific wording behind the profile.
   - Evidence artifact:
     `docs/2026-07-01-source-bank-visual-style-code-ready/LEDGER_PROMPT_AUDIT.md`

2. Do not bypass reusable story-quality machinery just because its current
   prompt text says science fiction.
   - Parameterize pitch-room, story-select, refine grading, dramatic-state, and
     outline prompts where practical.
   - If any module is bypassed in V1, the bypass must be explicit, tested, and
     justified as temporary.

3. `source_bank` plumbing must be explicit.
   - Append widget.
   - Add `run(..., source_bank="science_news")`.
   - Forward into `_resolve_inputs`.
   - Build/select the active `ledger_writing_spec`.
   - Use that spec/profile before outline generation.

4. Visual style is a ledger-level visual bible.
   - Add `VisualStylePolicy.ledger_directives`.
   - ShotLock stamps parsed policy into returned patched ledger meta.
   - MetaBrief reads same policy while composing still prompts.
   - `finish_visual_prompt` applies policy at the shared prompt seam.
   - Render-driver inherits style through patched ledger meta.

5. `OTR_VisualStyleDirector` must have full ComfyUI node contract.
   - `INPUT_TYPES` with style combo.
   - `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`.
   - Return `(visual_style_policy_json,)`.
   - Register in root `__init__.py` `_NODE_MODULES`.
   - Keep style definitions in a catalog module, not inline in the node.

6. Compatibility mapping must be exact.
   - `StoryInputPacket.close_brief` -> `meta.news.news_close_brief`.
   - `script_brief` -> `meta.news.script_brief`.
   - `casting_brief` -> `meta.news.casting_brief`.
   - `key_terms` -> `meta.news.key_terms`.
   - source title/url/hash must mirror legacy fields where consumers expect
     them.

7. Visual style sockets must be real sockets.
   - `visual_style_policy_json` in MetaBrief/ShotLock `INPUT_TYPES` must use
     `"forceInput": True`.
   - Method signatures must append the same kwarg.

8. Workflow and widget guardrails are hard.
   - Writer optional count changes from 16 to 17 for `source_bank`.
   - Workflow writer `widgets_values` count changes from 25 to 26.
   - `source_bank` goes into both headless creative whitelists.
   - Workflow JSON changes land in the same chunk as node/socket changes.

## Rejected Or Modified

- Rejected: append `source_text_path` in C1. It is an unused public-domain
  widget until C7, and two append-only migrations are acceptable when each is
  validated against the real workflow.
- Rejected: "offline fallback" for archive RSS. No silent fallback. A later
  explicit `local_archive_index` mode is fine, but it must be selected and
  tested, not hidden.
- Rejected: blanket `ConfigDict(extra="ignore")`. Canonical contracts should
  stay strict. If ledger-load tolerance is needed, implement it in load helpers,
  not by weakening the schema everywhere.
- Modified: `base_tail_strategy="suppress"` must be defined precisely in C0.
  Prefer naming it as base-tail behavior only, so policy ledger directives still
  apply.

## R2 Coding Targets

R2 should turn this into exact code chunks:

1. `LedgerWritingSpec` / prompt profile contracts.
2. Fresh upstream source-brain modules and translator into existing ledger
   inputs.
3. Prompt audit -> profile variables, used by the new upstream layer first.
4. Source selector plumbing through writer and `_resolve_inputs`.
5. Compatibility mirror into `meta.news`.
6. Visual style catalog/director node.
7. Visual style propagation into patched ledger, MetaBrief, ShotLock, and
   `finish_visual_prompt`.
8. Canonical workflow deltas.
9. Tests and validation commands.

## R1 Judgment Log

Accepted:

- Codex anchor: same ledger, different prompt packs; visual style is ledger
  visual bible.
- User correction: `ledger_writing_spec` is the upper-level control plane for
  ledger-writing logic.
- User correction: prompts are swappable variables; hardcoded source-specific
  Python strings must move behind profiles.
- Antigravity: node registration and node input contract must be explicit.
- Antigravity: compatibility mapping must include `close_brief` ->
  `news_close_brief`.
- Claude: `source_bank` plumbing from widget to `_resolve_inputs` was missing.
- Claude: `forceInput` detail is required for visual-style wiring.
- Claude: render-driver propagation assumption must be verified.

Rejected:

- Antigravity: add `source_text_path` in C1.
- Antigravity: hidden offline fallback.
- Antigravity: weaken schemas to `extra="ignore"` everywhere.
