# R3 Final - Wiring, Bridge, and Transplant Boundaries

Verdict: standalone lab wiring is green; production bridge/transplant remains
gated.

Grounded findings accepted and fixed inside the lab:

- CONFIRMED: `get_visual_styles()` cache could go stale after fixture edits in
  the same ComfyUI process. Fixed with a visual-style fixture stamp that rebuilds
  the cache when style JSON size/mtime changes.
- CONFIRMED: `_lab_state_digest()` included `__pycache__` and `.pyc` files.
  Fixed by excluding those paths from the digest.
- CONFIRMED: `story_pipeline_id` was accepted by the preview node but dropped
  before `LedgerWritingSpec`. Fixed by adding `story_pipeline_id` to
  `LedgerWritingSpec` and threading it through `build_spec_from_material()` and
  `OTR_StoryPackPreview`.
- CONFIRMED: the lab lacked an install dependency declaration. Fixed with
  `requirements.txt` containing `pydantic>=2.0,<3.0`.
- CONFIRMED: validation was only a script. Added `tests/test_lab_contracts.py`
  with pytest coverage for public-domain spec build, visual-style registration,
  visual-policy mismatch rejection, and custom-source fail-loud behavior.
- CONFIRMED: pytest imported the hyphenated custom-node folder as top-level
  `__init__`, breaking relative import. Fixed `__init__.py` to support both
  ComfyUI package import and pytest top-level import.

Rejected or deferred:

- REJECTED for current lab: replacing fixture-load errors with warning dropdown
  sentinels. This package is a fail-closed lab; corrupt fixtures should surface
  clearly during validation instead of appearing to run.
- DEFERRED to transplant: production dependency edits in `ComfyUI-OldTimeRadio`.
  The standalone lab has its own `requirements.txt`; production should not be
  touched until the bridge/transplant chunk.
- DEFERRED to transplant: `compose_news_coda`, style picker, meta-brief image
  prompt, and canonical workflow widget-index tests. These are real production
  risks, but out of scope for the standalone folder.
- DEFERRED: proving `build_legacy_news_mirror()` against the production
  `meta.news` contract. This is a bridge gate before transplant.

Bridge/transplant gates before touching production workflow JSON:

- Add a bridge node or bridge script in the lab that emits a frozen
  `LedgerWritingSpec` JSON artifact; do not patch production first.
- Read the production ledger consumer keys and make
  `build_legacy_news_mirror()` compare against the authoritative `meta.news`
  contract.
- Parameterize production coda/style-picker/meta-brief prompt code from
  `StoryPromptProfile` and `VisualStylePolicy`, not hardcoded sci-fi/news text.
- Add production tests before workflow JSON edits, especially for widget vector
  positions and any new optional input sockets.
- Only after tests are green should the transplant touch
  `ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json`.

Validation after R3 fixes:

```text
OK upstream_story_lab
science story_model=science_news_default
archive story_model=media_restoration_adventure
public_domain story_model=faithful_radio_adaptation
archive visual_style=archival_documentary
public_domain_manifests=3
story_packs=12
```

Pytest:

```text
4 passed in 0.04s
```

Production workflow check:

```text
git diff --name-only -- workflows\otr_scifi_16gb_full.json
<no output>
```

R4 prompt-forward:

Converge on residual defects only. Treat the standalone lab as live and green.
Do not reopen broad architecture unless a concrete code defect remains in
`ComfyUI-OTR-UpstreamStoryLab`.
