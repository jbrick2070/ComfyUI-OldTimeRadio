# Fable Final Review Prompt - Before Transplant Coding

Use this after the upstream story/source/visual lab is content-complete, but
before coding the bridge/transplant into `ComfyUI-OldTimeRadio`.

## Prompt

You are reviewing the upstream rewrite plan for ComfyUI Old Time Radio before
we begin the production transplant. Please do a grounded final review and revise
the plan only where the actual files justify it.

Core law for this rewrite:

```text
JSON owns content and configuration.
Python owns validation, routing, execution, and fail-loud errors.
```

Do not propose hidden fallbacks. Do not preserve legacy paths merely for old
tests. By the time this review runs, the old SFX surface is expected to have
been removed because it did not produce a real material instance consumed by
the ledger or downstream pipeline. Treat that removal as the expected new
baseline and verify that no upstream/transplant plan accidentally depends on
the deleted SFX path.

## Files To Review First

Standalone upstream lab:

- `ComfyUI-OTR-UpstreamStoryLab/docs/JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md`
- `ComfyUI-OTR-UpstreamStoryLab/kibitz-runs/2026-07-01-upstream-story-lab-code-ready/r4/final.md`
- `ComfyUI-OTR-UpstreamStoryLab/PROMPT_SURGERY_CHECKLIST.md`
- `ComfyUI-OTR-UpstreamStoryLab/TRANSPLANT_MANIFEST.md`
- `ComfyUI-OTR-UpstreamStoryLab/README.md`
- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/contracts.py`
- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/catalogs.py`
- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/preview.py`
- `ComfyUI-OTR-UpstreamStoryLab/nodes.py`
- `ComfyUI-OTR-UpstreamStoryLab/tests/test_lab_contracts.py`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/story_packs/**/*.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/visual_styles/*.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/source_packets/*.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/public_domain_sources/**/manifest.json`

Original sci-fi-remnant and prompt/visual audit docs:

- `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/STORY_AND_VISUAL_SCI_FI_REMNANTS_ARTIFACT.html`
- `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/PHASE2_PROMPT_PY_UPDATE_MAP.md`
- `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/LEDGER_PROMPT_AUDIT.md`
- `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/VISUAL_PROMPT_AUDIT.md`
- `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/VISUAL_SCI_FI_REMNANTS_ARTIFACT.html`

Production code to inspect before approving transplant:

- `ComfyUI-OldTimeRadio/workflows/otr_scifi_16gb_full.json`
- `ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py`
- `ComfyUI-OldTimeRadio/nodes/OTR_LedgerFreezeCascade.py`
- `ComfyUI-OldTimeRadio/nodes/_otr_line_composer.py`
- `ComfyUI-OldTimeRadio/nodes/_otr_style_picker.py`
- `ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py`
- any remaining production references to the removed SFX/material path

## Specific Questions

1. Does the upstream lab obey the rule that JSON owns content/config while
   Python owns validation/routing/execution?

2. Are any prompts, tone rules, visual tails, forbidden terms, or examples still
   buried in Python when they should be JSON fixtures?

3. Are any Python files treating `science_news`, `media_archive`, or
   `public_domain_story` through scattered hardcoded conditionals instead of a
   validated spec?

4. Does the plan prevent hidden fallback from media archive or public domain
   back to sci-fi/news defaults?

5. Does the visual-style path correctly separate visual content from execution?
   In particular, do archive/anime/cartoon/origami visual policies live in JSON
   while Python only validates and routes them?

6. Is the experimental `simple_4_prompt_experimental` path correctly treated as
   visible experimental behavior, not a hidden fallback?

7. Before transplant, what exact production prompt/code sites must be changed
   so coda, style picker, meta-brief image prompts, still prompts, and video
   prompts consume `StoryPromptProfile` and `VisualStylePolicy` instead of
   sci-fi hardcoded text?

8. Given that the old SFX surface should already be removed by this review,
   does the transplant plan avoid depending on that deleted path? Are there any
   remaining tests, imports, prompts, schema fields, or workflow assumptions
   that still expect SFX output even though no real material instance exists?

9. What is the smallest safe bridge artifact production should consume before
   editing `otr_scifi_16gb_full.json`?

10. What must be true before the real workflow JSON is touched?

## Output Format

Please answer in this structure:

```text
VERDICT:

MUST-FIX BEFORE TRANSPLANT:
- file/path: concrete issue, why it matters, exact fix

SHOULD-FIX BEFORE TRANSPLANT:
- file/path: concrete issue, why it matters, exact fix

DEFER UNTIL AFTER TRANSPLANT:
- item, why safe to defer

DELETE / RIP OUT:
- any remaining dead legacy item after the SFX removal, evidence, replacement
  or reason no replacement is needed

KEEP:
- parts of the current plan that are correct and should not be overworked

REVISED TRANSPLANT PLAN:
1. ...
2. ...
3. ...

TEST / VALIDATION GATES:
- exact checks that should pass before workflow JSON edits

OPEN QUESTIONS:
- only questions that cannot be answered from the files
```

Ground every claim in the actual files. If you cannot verify a claim from the
files, label it `UNVERIFIED` and do not treat it as a finding.
