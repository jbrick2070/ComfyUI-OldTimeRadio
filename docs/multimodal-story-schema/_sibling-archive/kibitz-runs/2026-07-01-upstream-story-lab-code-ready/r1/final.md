# R1 Final - Upstream Story Lab Architecture

Verdict: yes, with R1 fixes applied before advancing.

Grounded findings accepted:

- CONFIRMED: the first live version had media/science spec paths but no
  public-domain `LedgerWritingSpec` path. Fixed by adding
  `fixtures/source_packets/public_domain_book_chapter.json`, a
  `public_domain_story` branch in `preview.py`, and validator coverage.
- CONFIRMED: public-domain source-folder manifests used their own fields and
  were not validated by any contract. Fixed by adding
  `PublicDomainSourceManifest` and manifest/file-reference checks.
- CONFIRMED: visual-style discovery was split; the node injected `sci_fi_radio`
  while only `archival_documentary.json` existed as a fixture. Fixed by adding
  JSON fixtures for all five styles, loading fixture overrides into the
  `VISUAL_STYLES` catalog, and making node choices come from the catalog.
- CONFIRMED: `custom_source_bank` preview succeeded without a real custom
  schema/source packet. Fixed by making it fail loudly with a pointer to
  `CUSTOM_SOURCE_BANK_GUIDE.md`.
- CONFIRMED: `IS_CHANGED` content hashing is the right cache policy for this
  fixture-driven lab. Kept.

Claims rejected or deferred:

- DEFERRED: removing separate source/story/pipeline widgets from
  `OTR_StoryPackPreview`. The mismatch error is not ideal UX, but the explicit
  dimensions are useful for architecture inspection right now. Revisit when the
  UI graduates from preview node to production controls.
- DEFERRED: source-bank/visual-style compatibility matrix. Needed before real
  prompt meat lands, but not required for the live scaffold after all styles are
  selectable and fail-closed by id.
- DEFERRED: making `simple_4_prompt_experimental` portable across real source
  banks. Current truth: it is a visible custom-source experiment only until
  per-lane packs exist.

Post-fix validation:

```text
OK upstream_story_lab
science story_model=science_news_default
archive story_model=media_restoration_adventure
public_domain story_model=faithful_radio_adaptation
archive visual_style=archival_documentary
public_domain_manifests=3
story_packs=12
```

Post-fix import smoke:

```text
loaded nodes: ['OTR_StoryPackPreview', 'OTR_UpstreamStoryLabValidator']
style choices: ['anime', 'archival_documentary', 'cartoon', 'paper_origami', 'sci_fi_radio']
media anime preview chars: 10675
pd preview chars: 10712
custom hard error: custom_source_bank is visible but not runnable yet
```

R2 prompt-forward:

Review the corrected live folder as a code-ready scaffold for upstream story
generation. Focus on implementability of the contracts, validation surface,
ComfyUI node API, and smallest next coding step for media-archive and
public-domain prompt meat. Do not review downstream production transplant or
`ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json`.
