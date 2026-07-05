# Multi-Modal Story Schema -- IN-REPO BUILD PLAN (plan of record)

Date: 2026-07-04. Branch: `v2.0-alpha`. Repo: `ComfyUI-OldTimeRadio` (ONE repo).

This is the go-forward for Sprint-3 item-1 + the full multi-modal story/visual vision.
The design was prototyped in the sibling `ComfyUI-OTR-UpstreamStoryLab` repo; the
kibitz gut-check (unanimous, HIGH) ruled that shape wrong (a separate repo +
`production_mirror/` + a bridge cannot honor CLAUDE.md same-commit / one-repo rules).
So the IDEAS come here; the lab's parallel package/registry/bridge scaffolding does NOT.
We build it into the EXISTING `nodes/`, clean-break, no new parallel package.

- Design reference (the "very well thought out" schema): `design-reference/`
  (R1 architecture, the JSON-owns-content law, the code-ready brief, pack-author
  checklist, story-engine-map brief). READ these for intent.
- Schema by example (the lab's real JSON blueprints -- 12 story packs, 5 visual
  styles, banks/pipelines): `schema-examples/`. These are the TARGET shapes to adapt.

## The vision (operator, 2026-07-04)
Multiple STORY PATHS that all populate the SAME ledger but with different logic +
content, PLUS a VISUAL STYLE switch that only rewrites downstream visual prompts:
- Story paths (source_bank / story_model): **public-domain**, **media-archive RSS**,
  **simple-4-LLM (all original)**, plus the existing **science/sci-fi** path. Same
  ledger contract; different pack content + pass logic.
- Visual styles: **OTR-scifi** (today's look), **anime**, **origami**, ... Same ledger
  content; the style pack rewrites the DOWNSTREAM still/video prompt language only.

## Core law (from the design, adopted verbatim)
```
JSON owns content + configuration (prompt/seam text, tone rules, forbidden terms,
  visual style tails, source examples, pack/style defaults).
Python owns validation, routing, execution, and FAIL-LOUD errors.
No fallbacks. No hidden models. No hidden engines. Unknown id = hard error.
```

## Four orthogonal axes (each is DATA; none silently implies another)
- `source_bank` -- where material comes from + how it is interpreted.
- `story_model` -- dramatic/tonal writing lane (source-scoped).
- `story_pipeline` -- the LLM pass structure (named passes + slot roles + budgets).
- `visual_style` -- render language (role-keyed motion/still prompt tails).

## Clean-break rules (operator: breakage-in-progress is OK)
- No back-compat shims, no fallbacks, no tracebacks papering over the transition.
- BUT tests move WITH the schema in the SAME commit -- the suite stays GREEN per chunk
  (CLAUDE.md), even though the end-to-end FEATURE is incomplete until the last stage.
- Every node/widget change lands in `workflows/otr_scifi_16gb_full.json` in the same
  commit + re-validate (CLAUDE.md invariant 6). Commit AND push per green chunk.

## STAGE ORDER (each stage = its own hardened sub-plan before coding it)

### Stage 1 -- Content -> JSON foundation (START HERE; the "py->JSON" work)
- Define the in-repo pack SCHEMA (adapt `schema-examples/story_packs/` +
  `banks.json` / `pipelines.json`); decide where packs live IN this repo (JSON under
  `nodes/` -- NO new top-level package).
- Add a small LOADER the EXISTING story nodes call (direct `json.load`, fail-loud on
  unknown/missing seam id). One canonical seam list (from R1 section 4:
  interpret / outline_system / pitch_room_system / story_select_system /
  dramatic_state_system / line_grounding / casting_brief_seam / coda / title_system /
  style_pick_inventor / style_pick_chooser / labels).
- Extract the CURRENT science/sci-fi prompt constants into the FIRST pack. Start
  BYTE-IDENTICAL (prompt bytes the LLM sees unchanged) so audio/story regressions
  stay green; the schema is then free to diverge in later stages.
- NOTE: this is the same work the lab already did once -- re-do it in-repo (faster
  the 2nd time). `line_grounding` is a conditional f-string; keep it Python until the
  loader supports parameterized seams (defer, don't force it).

### Stage 2 -- Story paths (source_bank / story_model routing)
- Add the selector + Python routing (fail-loud on unknown id); author packs for
  public-domain, media-archive, simple-4-LLM (adapt `schema-examples/story_packs/*`).
- The `simple_4_prompt_experimental` pipeline stays visible: pass1 creative story,
  pass2 ledger fill, pass3 technical schema cleanup, pass4 technical ledger audit,
  optional adaptive cleanup under a hard deterministic cap.
- All paths write the SAME ledger contract.

### Stage 3 -- Visual style schema
- Add the `visual_style` selector + JSON style packs (adapt
  `schema-examples/visual_styles/*`: sci_fi_radio / anime / cartoon / paper_origami /
  archival_documentary). Style rewrites ONLY downstream still/video prompt tails --
  same ledger content. Integrate with the Sprint 1 radio_object + Sprint 2 still_word
  prompt work already landed (do not clobber them).

### Stage 4 -- Asserts -> JSON (LAST; has a real prerequisite)
- Moving story-CONTENT validation asserts into JSON needs a NEW declarative-rule
  ENFORCER node first (today's `_otr_workflow_validator.py` only audits litegraph
  structure, not story-content rules). Build the enforcer, THEN move the rules.

## Process per stage
Before coding each stage, write its hardened sub-plan (roundtable/kibitz if torn),
then execute per CLAUDE.md: regression + Bug Bible after every change, commit+push per
green chunk, workflow JSON edits in the same commit + re-validate.
