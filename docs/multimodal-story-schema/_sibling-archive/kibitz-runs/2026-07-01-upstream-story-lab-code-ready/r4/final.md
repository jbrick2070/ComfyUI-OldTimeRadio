# R4 Final - Convergence

Verdict: converged. The standalone upstream lab is live, code-ready as a
fixture/prototype custom node, and validated. Production transplant remains a
separate gated project.

R4 residuals accepted and fixed:

- Added top-level `__init__.py` to the lab state digest so entrypoint edits can
  invalidate ComfyUI cache.
- Added `LedgerWritingSpec` consistency checks for top-level source/story ids
  against `story_input` and `prompt_profile`.
- Added public-domain manifest path safety: absolute paths and `..` components
  now fail loudly.
- Changed media-archive fixture text construction to tolerate an untitled
  source value cleanly.
- Added `source_packet_note` to preview JSON so shared source-packet fixtures
  are explicit.
- Removed `simple_4_prompt_experimental` from the narrative `story_model_id`
  dropdown while keeping it visible in the `story_pack` dropdown.
- Re-exported `PublicDomainSourceManifest` and `StoryPack` from the package
  interface.
- Added one more pytest case for mismatched story ids.

Final green checks:

```text
scripts\validate_lab.py
OK upstream_story_lab
science story_model=science_news_default
archive story_model=media_restoration_adventure
public_domain story_model=faithful_radio_adaptation
archive visual_style=archival_documentary
public_domain_manifests=3
story_packs=12
```

```text
pytest
5 passed in 0.04s
```

```text
ComfyUI import smoke
loaded nodes: ['OTR_StoryPackPreview', 'OTR_UpstreamStoryLabValidator']
model choices exclude simple_4_prompt_experimental from narrative story models
style choices: ['anime', 'archival_documentary', 'cartoon', 'paper_origami', 'sci_fi_radio']
has shared note: True
```

```text
production workflow check
git diff --name-only -- workflows\otr_scifi_16gb_full.json
<no output>
```

Agent calls made:

- R1: Antigravity + Claude
- R2: Antigravity + Claude
- R3: Antigravity + Claude
- R4: Antigravity + Claude
- Total external reviewer calls: 8

Remaining verify-at-transplant gates:

- Confirm `build_legacy_news_mirror()` keys against the real production
  `meta.news` consumers before writing a bridge.
- Parameterize production coda/style-picker/meta-brief prompt code from
  `StoryPromptProfile` and `VisualStylePolicy`.
- Add production tests before touching `otr_scifi_16gb_full.json`, especially
  widget-vector and canonical-workflow tests.
- Append any new widgets only at the end of existing production nodes.
- Do not transplant until the upstream lab emits a frozen bridge artifact that
  can be validated independently.
