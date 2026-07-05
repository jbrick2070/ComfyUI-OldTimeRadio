# NotebookLM Source List - Real Prompt And Code Content

Purpose: build a comparison presentation about the real sci-fi-specific prompt
and Python remnants, plus the proposed media-archive and public-domain prompt
content. This is intentionally focused on content and code changes, not broad
architecture theory.

## Load These First

1. `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/STORY_AND_VISUAL_SCI_FI_REMNANTS_ARTIFACT.html`

   Best visual map of current sci-fi remnants across story prompts, visual
   prompts, and Python code.

2. `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/PHASE2_PROMPT_PY_UPDATE_MAP.md`

   Best focused map of which prompts and Python logic need replacement or
   repurposing for media archive and public-domain models.

3. `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/LEDGER_PROMPT_AUDIT.md`

   Story/ledger prompt audit: useful for isolating sci-fi/news assumptions in
   the story-writing path.

4. `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/VISUAL_PROMPT_AUDIT.md`

   Visual prompt audit: useful for isolating sci-fi/cinematic/radio remnants in
   image, still, and video prompt text.

5. `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/VISUAL_SCI_FI_REMNANTS_ARTIFACT.html`

   Visual-only artifact for sci-fi remnants in still, character, scene, and
   video prompt wording.

6. `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/BRAINSTORM_MEDIA_ARCHIVE_PUBLIC_DOMAIN_UPDATES.md`

   Brainstorm notes for how media archive/RSS and public-domain books, comics,
   plays, and source folders should change prompts and code.

## Load The Actual Proposed Prompt Content

Use these fixture folders as the real proposed content examples:

- `ComfyUI-OTR-UpstreamStoryLab/fixtures/story_packs/science_news/science_news_default.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/story_packs/media_archive/*.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/story_packs/public_domain/*.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/story_packs/experimental/simple_4_prompt_experimental.json`

These are the concrete prompt-stage packs for science news, media archive,
public domain, and the experimental 4-pass pipeline.

## Load The Proposed Visual Content

Use these as the concrete visual-style content examples:

- `ComfyUI-OTR-UpstreamStoryLab/fixtures/visual_styles/sci_fi_radio.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/visual_styles/archival_documentary.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/visual_styles/anime.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/visual_styles/cartoon.json`
- `ComfyUI-OTR-UpstreamStoryLab/fixtures/visual_styles/paper_origami.json`

These files show the actual visual prompt tails, forbidden terms, motion prompt
language, ledger visual directives, and style-specific replacements. They are
the best sources for comparing:

- sci-fi radio/cinematic film-grain assumptions
- media archive/documentary restoration visuals
- anime visual prompt language
- cartoon visual prompt language
- paper origami/papercraft visual prompt language

Also load:

- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/catalogs.py`

That file shows how visual styles are registered and how source banks default
to visual styles, such as science news defaulting to `sci_fi_radio` while media
archive and public-domain default to `archival_documentary`.

## Load The Python Logic That Turns The Content Into Specs

- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/contracts.py`
- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/catalogs.py`
- `ComfyUI-OTR-UpstreamStoryLab/src/upstream_story_lab/preview.py`
- `ComfyUI-OTR-UpstreamStoryLab/nodes.py`

These files show the actual source-bank, story-model, story-pipeline, visual
style, and fail-loud validation logic.

## Load The Proposed Migration Checklist

- `ComfyUI-OTR-UpstreamStoryLab/PROMPT_SURGERY_CHECKLIST.md`
- `ComfyUI-OTR-UpstreamStoryLab/TRANSPLANT_MANIFEST.md`
- `ComfyUI-OTR-UpstreamStoryLab/kibitz-runs/2026-07-01-upstream-story-lab-code-ready/r4/final.md`

These explain what remains before transplanting the upstream model into the
production OTR workflow.

## Suggested NotebookLM Question

Create a comparison presentation that focuses on real prompt and Python content,
not abstract architecture. Compare:

- current sci-fi science-news story generation assumptions
- current sci-fi visual prompt assumptions
- proposed media archive/RSS story packs and code changes
- proposed public-domain source-folder story packs and code changes
- proposed visual-style content changes for media archive, anime, cartoon, and
  paper origami
- the experimental 4-pass pipeline

For each category, identify:

- which current prompts or Python logic are sci-fi/news-specific
- what exact replacement prompt content is proposed
- what exact visual prompt/style content is proposed
- what exact Python logic must change
- what content is already represented in story-pack fixtures
- what content is already represented in visual-style fixtures
- what remains as transplant or bridge work before production
