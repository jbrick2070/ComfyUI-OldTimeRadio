# Vibe-Coder Extensibility -- R1 Scoper

- **Date:** 2026-07-12
- **Status:** R1 scope only. NO code. Architecture deliberately left OPEN -- this doc frames the problem and the option space for an R1 panel pass (arc routing: cloud roundtable for ideas, or kibitz for economy).
- **Operator ask:** make it easy for users -- "vibe coders" (README's target audience: ComfyUI newbies working with an LLM assistant) -- to add (A) new models to the dropdowns, (B) new source banks, (C) new visual styles. Keep the architecture open; scope first.

---

## 1. Problem statement

OTR's three main creative surfaces are extensible today, but only by an expert who reads the
source. A vibe coder -- someone who can edit a JSON file with an LLM assistant's help but will
not trace Python module contracts -- currently hits a wall on each surface at a different
depth. The goal of this effort is a documented, validated, no-Python-edit path for the common
case on all three surfaces, without weakening any of the fail-loud / no-fallback / preflight
laws the pipeline depends on.

Key reframe from grounding: **this is NOT a re-architecture.** All three surfaces already
follow the converged "JSON owns content, Python owns behavior" law. The work is closing the
last Python-edit seams, adding scaffolding + a friendly validation loop, and writing the
5-minute recipes. The registries themselves are sound.

## 2. Current state (grounded 2026-07-12 against v2.0-alpha HEAD)

### Surface A -- Model dropdowns (LLM writer slots)
- `nodes/_otr_model_catalog.py`: `CURATED_LLM_MODELS` = frozen dataclass rows with honesty
  fields (`repo_id`, `requires_auth`, `loader_backend` [7 backends incl. `gguf_native`,
  `openrouter_http`], `vram_fit_tier` PASS/WARN/UNKNOWN/FAIL, `approx_safetensors_gb`,
  `prompt_profile`, license audit fields, chat-template dispatch hints).
- Dropdown = curated set + **local HF cache scan** + **local GGUF scan**, with label suffixes
  (`[LOCAL HF]`, `[LOCAL GGUF]`, `[NOT DOWNLOADED]`); validator strips suffixes; three
  admit-paths (curated / locally-scanned / valid org-name when auto-download enabled).
- S30 two-model-selector: only `creative_writing_model` + `technical_model` widgets exist;
  every consumer node receives its model id via a STRING socket broadcast (fail-loud
  `MissingModelInputError` / `UnknownModelError` at the socket boundary).
- **Vibe-coder reality:** dropping a GGUF/HF model into the local cache ALREADY surfaces it in
  the dropdown. The wall: (1) nobody documents this; (2) a *curated* row (with VRAM honesty,
  license fields, backend dispatch) requires a Python edit; (3) non-LLM model pins (video/
  image/TTS engine checkpoints) live in engine code + the canonical workflow JSON, out of
  scope of the catalog entirely.

### Surface B -- Source banks
- `nodes/story_packs/banks.json` (schema v2.0): one row per bank -- `source_bank_id`, `label`,
  `source_kind`, `interpreter`, `fetcher`, `default_story_model`, `default_story_pipeline`,
  `defaults` (prompt-label dict), `required_seams` (the pack seam list), `runnable`,
  `guide_ref`. Companion `pipelines.json`. Per-bank folders hold the story-model packs
  (7 banks live today: science_news, media_archive, public_domain_story, shakespeare,
  original_radio, custom_source_bank, original_codex56sol + the scifi_* bakeoff lanes).
- `_otr_story_routing.list_bank_ids()` supplies the `source_bank` dropdown at INPUT_TYPES;
  `get_bank()` gates at run.
- **Vibe-coder reality:** a new bank that REUSES an existing `interpreter` + `fetcher` is
  in principle a banks.json row + a pack folder -- no Python. The walls: (1) `interpreter` /
  `fetcher` are bare Python identifiers with no documented contract or inventory; (2) the
  seam-pack schema (`required_seams` content) is undocumented outside the code; (3) the
  SOURCE_BANK_PREFLIGHT hard gate (docs/SOURCE_BANK_PREFLIGHT.md, hashed receipt -- operator
  directive 2026-07-11) is a manual expert checklist; (4) `custom_source_bank` exists as an
  experimental lane but is not a documented template.

### Surface C -- Visual styles
- `nodes/visual_styles/<style_id>.json`, 9 packs shipped. `_otr_visual_styles.py` is pure
  behavior: lazy directory sweep on first resolve, strict v2 schema validation, unknown id =
  hard error, no fallbacks. JSON owns look/subject deltas; Python owns geometry contracts.
- **Vibe-coder reality:** this is ALREADY drop-a-JSON. The walls: (1) the v2 schema is heavy
  (~15 string fields + 4 dict fields, exact-placeholder template rules `{form}`/`{base}`,
  mouth-prominence vocabulary requirement, 240-char motion-register budget, forbidden-terms
  lint) -- fail-loud errors are correct but arrive one at a time against a schema the user has
  never seen; (2) no annotated template or schema doc exists; (3) the sci_fi_radio pack is
  pinned byte-identical to extraction fixtures -- an outsider copying it as a starter must not
  "fix" it.

### Cross-cutting laws that any design MUST preserve
- Fail-loud, no fallbacks, unknown id = hard error (Stage 2C/3C converged law).
- Source-bank preflight hard gate + hashed receipt stays a gate (automate the mechanics, never
  waive the gate).
- LLM-first: banks/styles feed the LLM; Python judges, never rewrites story text.
- `widgets_values` is positional: combo widgets store the selected STRING, so ADDING dropdown
  entries is save-safe; REMOVING/renaming ids breaks saved workflows silently (BUG-LOCAL-097
  family). Additions must never reorder or insert widgets.
- New entries must not require touching `workflows/otr_canonical.json` -- dropdown CONTENTS
  are data; wiring is structure. (Contracts never freeze -- adapters PARKED; do not smuggle a
  versioning/adapter layer in through this effort.)
- still_word lettering: per-episode locked font/lettering, backdrop varies.

## 3. Proposed shape (directions, all OPEN for the panel)

**Thesis: one pattern, three surfaces.** Each surface gets the same four artifacts:

1. **A drop-in data path with zero Python** for the common case
   (already true for styles + local models; banks need the interpreter/fetcher seam closed
   or documented around).
2. **An annotated template / scaffold** the user copies
   (`_TEMPLATE.json.example` per directory, or a scaffold script, or -- the vibe-coder-native
   option -- a per-surface `SCHEMA.md` written to be PASTED INTO AN LLM ASSISTANT, i.e.
   docs-as-prompts: "give this contract to Claude/ChatGPT with your idea; it emits a valid
   pack").
3. **A one-command validator with batch, friendly errors**
   (`--check`-style CLI per surface or one `otr_check.py` covering all three; today's loaders
   validate correctly but fail one error at a time at run time -- the vibe loop needs ALL
   errors at once, at authoring time, with fix hints).
4. **A 5-minute recipe in the README** (newbie audience refresh is already pending -- these
   three recipes are its spine).

Per-surface sketches (NOT decisions):
- **A (models):** document the local-scan path as THE vibe path. Optionally add a
  `models.d/*.json` overlay for user-curated rows, marked UNVERIFIED and machine-derivable
  fields (size, maybe vram tier) auto-filled -- honesty fields must not be vibe-guessed.
  Engine/video/TTS checkpoint pins stay OUT of scope for R1 (canonical-JSON coupling).
- **B (banks):** define the Tier-1 bank = "new bank row + pack folder reusing a documented
  existing interpreter/fetcher (rss / static-text / custom)" with zero Python; Tier-2 (new
  interpreter/fetcher) stays expert. Ship a preflight SCRIPT that runs the mechanical checks
  and emits the hashed receipt.
- **C (styles):** template + validator + schema doc; possibly a "lint all packs" test that
  already exists extended into the authoring CLI. Byte-identity pins and geometry-vs-look law
  untouched.

## 4. Non-goals (R1)
- No marketplace / plugin system / pack manager; no in-ComfyUI GUI editor.
- No schema freeze or adapter/versioning layer (contract adapters are PARKED with a
  stable-defect revive trigger; respect it).
- No relaxation of fail-loud, preflight, or SFW laws.
- No new engine lanes, no new interpreters/fetchers as part of THIS effort.
- No touching the 720w bake-off lanes (scifi_* banks are frozen race artifacts).

## 5. Acceptance criteria (for the eventual build, testable)
1. A user with no Python knowledge adds a **visual style** from a template, runs one check
   command, gets EITHER a complete actionable error list OR a pass, and sees it in the
   dropdown after restart -- without editing any .py or the canonical JSON.
2. Same for a **Tier-1 source bank** (reusing a documented fetcher/interpreter), including a
   script-emitted preflight receipt.
3. A **local GGUF/HF model** drop is documented and surfaces in both writer dropdowns with
   correct suffix labels (behavior exists; recipe + test coverage for the recipe).
4. Saved workflows survive additions (no widget reorder; combo string semantics pinned by a
   guardrail test).
5. Every failure a vibe coder can trigger names the file, the field, and the fix in plain
   language.

## 6. Open questions for the R1 panel
1. Scaffold delivery: template files vs scaffold CLI vs docs-as-prompts (SCHEMA.md written for
   LLM consumption)? Which does a vibe coder actually succeed with?
2. One `otr_check.py` for all three surfaces vs per-surface validators? Relationship to the
   existing variants `--check` and OTR_WorkflowValidator?
3. User packs in-repo (git-dirty, clobbered by updates) vs a `user_packs/` overlay dir merged
   at scan time (extra_model_paths-style)? Update-survival matters for non-git users.
4. Dropdown refresh semantics: lazy first-resolve cache + restart (today) vs re-scan per
   INPUT_TYPES call so drops appear on browser refresh -- cost/staleness trade.
5. models.d overlay: worth it, or is local-scan + curated-stays-expert the honest scope?
6. Which interpreter/fetcher pairs are safe to document as Tier-1 today, and does
   `custom_source_bank` become the blessed template lane?
7. How much of SOURCE_BANK_PREFLIGHT is mechanizable without diluting the gate (receipt
   emission yes -- but which checks stay human)?

## 7. Next step
Run R1 on this doc (panel per CLAUDE.md section 8: cloud roundtable for the ideas round, or
kibitz r1 for economy), then R2 coding plan only after the option space above converges.
