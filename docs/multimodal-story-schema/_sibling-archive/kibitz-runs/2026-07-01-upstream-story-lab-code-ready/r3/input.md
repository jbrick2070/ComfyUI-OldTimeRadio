# R2 Final - Coding Plan and Implementability

Verdict: code-ready scaffold, with R2 hardening applied.

Grounded findings accepted and fixed:

- CONFIRMED: `_lab_state_digest()` read full file contents under `fixtures/`,
  which would become expensive once public-domain source folders contain large
  books or comic images. Fixed by hashing path, size, and `mtime_ns` metadata.
- CONFIRMED: visual-style fixture parsing happened during `catalogs.py` import.
  Fixed by moving style fixture loading behind lazy `get_visual_styles()` /
  `get_visual_style_ids()` access.
- CONFIRMED: `LedgerWritingSpec` allowed `visual_style_id` and `visual_policy`
  to diverge if hand-constructed. Fixed by requiring `visual_policy` and adding
  a Pydantic validator that checks `visual_policy.style_id == visual_style_id`.
- CONFIRMED: `build_spec_from_material()` defaulted visual style to
  `sci_fi_radio`, which was wrong for media/public-domain source banks. Fixed
  with source-scoped `"auto"` resolution:
  `science_news -> sci_fi_radio`, `media_archive -> archival_documentary`,
  `public_domain_story -> archival_documentary`.
- CONFIRMED: `_find_source_packet()` had a silent `None` path that could fall
  back to raw prompt stages for new source banks. Fixed by raising a hard
  `RuntimeError` for unregistered source banks.
- CONFIRMED: the live validator did not validate public-domain source-folder
  manifests. Fixed by validating `PublicDomainSourceManifest` and referenced
  files inside `OTR_UpstreamStoryLabValidator`.
- CONFIRMED: raw story-pack choice loading bypassed `StoryPack` validation and
  could overwrite duplicate keys. Fixed by validating with `StoryPack(**data)`
  and raising on duplicate pack keys.
- CONFIRMED: empty system-prompt fields were ambiguous. Fixed the contract to
  use `str | None` and added concrete style-picker defaults for media archive
  and public-domain profiles.

Rejected or deferred:

- REJECTED: replacing all startup/`INPUT_TYPES` hard errors with warning
  dropdowns. This lab is intentionally fail-closed; if its fixtures are corrupt,
  the right behavior is a visible error, not a quiet placeholder path.
- DEFERRED: caching parsed story-pack maps process-wide. Current 12-pack scan is
  acceptable for a lab. Revisit if the fixture count grows or UI refresh becomes
  noisy.
- DEFERRED: source-bank/visual-style compatibility matrix. Still needed before
  real prompt meat, but R2’s `auto` default removes the biggest footgun.

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
pack default: media_archive / media_restoration_adventure / legacy_many_pass
media anime preview chars: 11061
pd preview chars: 11344
custom hard error: custom_source_bank is visible but not runnable yet
```

Smallest next coding step:

Implement real prompt-meat packs for `media_archive` and `public_domain_story`
against the existing `StoryPack.prompt_stages` contract, then add validator
coverage that renders each stage through a `LedgerWritingSpec` and checks:

- no forbidden leakage terms in source-specific prompt text
- non-empty required prompt stages per pack
- source attribution/close mode matches source bank
- public-domain fidelity requirements survive into the generated spec
- visual-style defaults remain source-appropriate

R3 prompt-forward:

Review wiring, sequencing, and transplant boundaries. This lab is live as
`ComfyUI-OTR-UpstreamStoryLab`, but production OTR workflow JSON is still
untouched. Focus on what must be true before a bridge/transplant node is allowed
to touch `ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json`.
