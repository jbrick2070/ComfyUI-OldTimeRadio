# Gap-Hunt Briefing: OTR Upstream Multi-Source Story Engine

You are reviewing work that transplants a multi-source story architecture
into ComfyUI-OldTimeRadio (OTR) - a fully local, offline pipeline on one
Windows machine (RTX 5080 laptop, 16GB VRAM) that turns source material into
complete radio-drama episodes: LLM-written script ledger -> TTS voices ->
music -> FLUX stills -> video clips -> muxed episode. One operator (Jeffrey).
No cloud, no APIs at runtime.

## The architecture under review

Production today is hardwired to ONE lane: science-news RSS -> sci-fi radio
drama. The upstream rewrite makes source/story/visuals swappable via four
orthogonal, JSON-declared axes:

- source_bank: science_news | media_archive | public_domain_story |
  custom_source_bank (guided stub). Declared in fixtures/banks.json with
  defaults + a named interpreter binding (Python allowlist).
- story_model: dramatic/tonal lane; one JSON "story pack" per
  (bank, model, pipeline) - 12 packs total (attached samples). Packs own ALL
  prompt prose (seams: outline/pitch/select/dramatic-state/line-grounding/
  coda/title/style-pick) + tone guardrails + forbidden patterns/leakage
  terms. Python never authors prose.
- story_pipeline: legacy_many_pass (production writer's ~12-pass flow,
  declared DESCRIPTIVELY for audit only) | simple_4_prompt_experimental
  (lab-executable: story -> ledger-fill -> schema-cleanup -> audit; loud
  per-pass failure, never falls back).
- visual_style: 5 JSON policies (sci_fi_radio reproduces production's tail
  constants byte-identically; archival_documentary/anime/cartoon/
  paper_origami are new looks). Motion prompts keyed to the 4 surviving
  production video roles.

Law: JSON owns content and configuration; Python owns validation, routing,
execution, fail-loud errors. No lane may ever silently substitute another's
content ("no hidden fallbacks"); LLM engines are declared slot roles
(creative/technical), engine ids resolved at runtime from node widgets.

Flow: registry loads/validates all fixtures (duplicate/unknown/malformed =
hard error, per-seam template-variable validation at load) -> resolve() with
auditable per-axis decisions -> interpreter binding builds StoryInputPacket
(generated fields: casting_brief/script_brief/close_brief/key_terms) ->
resolved StoryPromptProfile (pack+bank-default merge, no Python prose) ->
LedgerWritingSpec (cross-ID validated) -> BRIDGE ARTIFACT: one frozen JSON
{spec, meta_mirrors{news, news_seed}, adapter_news_article, provenance
hashes incl. pinned production commit}. Production (tomorrow) consumes the
artifact through a validating adapter; science lane keeps its live RSS +
news_interpreter brain untouched.

Compatibility strategy: production's meta.news is the 13-field NewsBriefs
shape; meta.news_seed is a 7-key dict; both are PINNED in compat.py and
drift-tested by AST-PARSING a mirrored copy of production (production_mirror/
@ commit d48a9d76, hash-manifested). The mirror is never imported, only
parsed. news_used is derived downstream from the outline - never faked.

Status: 46 pytest + full-matrix headless validator + ComfyUI smoke green on
the real venv. Production repo untouched. Transplant chunk T1 scheduled for
tomorrow per a module-level PATCH_PLAN (attached).

## Ledger of findings the LOCAL panel already caught (do not re-report)

Structural: prompt content moved out of Python into packs (two diverged
copies reconciled); silent Python visual-style fallback removed; per-seam
template variables incl. runtime vars {n_required, seed_sample_block,
article_excerpt, candidates_block}; missed production LLM pass
(story_brief_reflection) declared; slot-plan vs engine-id split;
meta.news shape corrected (no title/headline/link - those live in
news_seed/article shapes; url not link); adapter file-loader + e2e
emit->validate test; artifact filenames carry all four axes + content hash;
ComfyUI cache digests include the mirror; registry instance cached;
production_baseline provenance read live from the mirror manifest;
science facade maps article dict onto build_news_briefs' REAL kwargs
(headline/summary/full_text/outlet/pub_date); non-science lanes set
resolved["style"] = story_model id deterministically and never read the
legacy style widget; canonical appended-widget contract (source_bank/
story_model/story_pipeline/visual_style COMBOs + bridge_artifact_path
STRING forceInput; empty path valid only for science); atomic callee-first
transplant order; writer self-test widget-count update in same chunk;
whitelist parity keys named; fallback taxonomy with six inventoried
production quality-floor sites (style-picker padding + first-candidate
chooser grandfathered science-only; RSS slug substitution science-only;
title -> outline.title floor kept; announcer outro deterministic floor kept;
briefs-degrade n/a via bridge); PD manifest path safety incl. symlink
containment; adaptive-cleanup experiment cut to documentation.

## Where the driver (Claude, the judge) suspects blind spots

- Nobody with dramaturgical taste has read the 12 packs' PROSE as prose.
- The simple_4 experimental pipeline is tested with a FakeLLM; whether a
  real local model can satisfy a production ledger schema in 4 passes is
  untested by design (it is an experiment) - but the pack prose for those 4
  passes may be under-specified for a real attempt.
- The archival_documentary motion prompts were authored tonight by the
  driver (music_open/close/inter variants); no one has challenged their
  craft or their fit to LTX-video motion-prompt best practice (motion verbs,
  no set-dressing nouns - production's own table follows that rule).
- Operator-scale drift: what breaks when Jeffrey adds pack #13 by
  hand-copying pack #7 at 1am.
- The mirror strategy assumes production refactors keep NewsBriefs and the
  motion table AST-extractable (class with annotated fields, dict literal).
  A refactor to dynamic construction breaks extraction - the tests fail
  loudly (good) but re-pinning guidance may rot.

## Your task

Per the review prompt: hunt for NEW findings in the blind-spot zones.
Attached grounding files are the real code and fixtures - trust them over
this briefing if they disagree, and say so if they do.
