# Vibe-Coder Extensibility -- R1 Scoper (r1-HARDENED)

- **Date:** 2026-07-12. **Revision:** r1 kibitz pass folded (panel: codex @ gpt-5.6-sol,
  antigravity @ gemini-3.5-pro; Claude anchor + judge). Run artifacts + judgment log:
  `kibitz-runs/2026-07-12-vibe-coder-extensibility/r1/final.md`.
- **Status:** R1 scope only. NO code. Operator directed r1-only -- next round waits for go.
  Architecture stays OPEN where marked; facts below are code-verified.
- **Operator ask:** make it easy for vibe coders (README's target audience) to add (A) new
  models to the dropdowns, (B) new source banks, (C) new visual styles.

---

## 1. Problem statement

OTR's three creative surfaces are extensible today, but only by an expert who reads the
source. A vibe coder -- ComfyUI newbie working with an LLM assistant -- hits a wall on each
surface at a different depth. Goal: a documented, validated, no-Python-edit path for the
common case on all three surfaces, without weakening fail-loud / no-fallback / preflight law.

Reframe #1: this is NOT a re-architecture -- all three surfaces already follow "JSON owns
content, Python owns behavior."

Reframe #2 (r1 panel, adopted): "one pattern, three surfaces" holds only at the UX layer.
The three surfaces have materially DIFFERENT admission contracts -- styles are self-contained
content packs; models are discovered binaries with unknown compatibility; banks are
multi-artifact EXECUTABLE lanes with rights/authorship/publication gates. One user-facing
command + recipe shape, three explicitly distinct qualification paths. Do not promise
equivalent drop-in extensibility.

## 2. Current state (all claims code-verified 2026-07-12, v2.0-alpha)

### Surface A -- Model dropdowns (LLM writer slots)
- `_otr_model_catalog.py`: `CURATED_LLM_MODELS` dataclass rows (honesty fields: backend,
  vram_fit_tier, license audit, chat-template hints) + HF-hub-cache scan
  (`scan_local_llm_cache` walks `HF_HOME/hub/models--*/snapshots/*` -- offline, HF-only).
- Uncurated cache hits are admitted ONLY if their config.json declares a `*ForCausalLM`
  architecture (BUG-LOCAL-257 guard) and get a BARE label. Suffixes are narrow:
  `[LOCAL HF]` only on `google/gemma*` local rows, `[LOCAL GGUF]` only on the single curated
  gguf row, `[NOT DOWNLOADED]` on absent curated rows.
- **GGUF is NOT generically scanned.** `_otr_gguf_backend.py` is hardcoded to one family:
  `ROW_ID = "unsloth/gemma-4-12b-it-GGUF"`, `DEFAULT_GGUF_FILENAME = "gemma-4-12b-it-Q8_0.gguf"`
  (+ Q6_K/Q4_K_M artifact table, `GEMMA4_12B_GGUF_PATH` escape hatch). The draft's "local
  GGUF drop already surfaces" was FALSE.
- **Broadcast is technical-only.** The 2026-05-29 lean-down removed the zero-consumer
  `creative_writing_model` output (contract audit at `OTR_LedgerScriptWriter.py:7076-7087`;
  RETURN_NAMES ends `technical_model`). Creative model id rides the ledger meta. The
  `_otr_model_inputs.py` S30 docstring predates this -- do not trust it on this point.
- **Refresh differs by surface:** `dropdown_choices()` re-scans on every call, and
  INPUT_TYPES consumes it -- model drops appear without restart. Banks and styles are lazy
  in-memory singletons -- restart required (see B/C).
- **Serialization hazard (new defect-class, design against it):** suffixed labels ARE the
  serialized `widgets_values` strings; the writer defines no `VALIDATE_INPUTS`; stock
  ComfyUI combo validation rejects a queued value not in the current choices list. A saved
  workflow holding `X [NOT DOWNLOADED]` breaks after X is downloaded (label changes to bare
  `X`). run() strips suffixes, so the fix direction exists -- but queue-time validation
  precedes run(). Verify-at-build with a repro test.

### Surface B -- Source banks
- Registry: `nodes/story_packs/banks.json` (v2.0 rows) + `pipelines.json`;
  `_otr_story_routing.list_bank_ids()` feeds the dropdown; `get_bank()` gates at run.
- **Contracts ARE documented** (draft was wrong): `docs/SOURCE_BANK_GUIDE.md` is normative --
  three routing coordinates (§3), runner interface + two-slot law (§4), fetcher envelope
  `fetch(*, bank, technical_model, source_ref="")` returning the writer's exact seven
  `SOURCE_PAYLOAD_KEYS` (§5), `legacy_many_pass` interpreter contract, rights/provenance
  sidecars. The real gap = newbie-safe inventory + templates + automated conformance
  feedback, not absent documentation.
- **Closed registries, no discovery:** 5 fetchers (`science_rss`, `media_archive_rss`,
  `public_domain_source`, `shakespeare_folger`, `original_codex56sol_local_seed` --
  bank-locked synthetic), 4 interpreters (`news_interpreter`, `media_archive_interpreter`,
  `public_domain_interpreter`, `shakespeare_interpreter`) (`_otr_source_payload.py:512-532`);
  runners registered explicitly in `_RUNNER_BY_PIPELINE` (preflight Gate 5 forbids
  plugin-style discovery).
- **The complete runnable-bank bundle** (draft understated it): bank row + story pack
  (`story_packs/<bank>/<model>.json`) + **`nodes/story_rules/<source_bank_id>.json`** (the
  loader HARD-FAILS any runnable bank missing its rules pack, `_otr_story_rules.py:274-280`;
  stray files or unregistered stems in `story_rules/` fail the WHOLE load -- a vibe-coder
  landmine) + registered pipeline/runner pair + qualification state.
- `custom_source_bank` is `runnable: false` (banks.json:263-272) because its runner does not
  exist -- it is NOT a blessed template lane as-is.
- **Preflight has two natures.** Gate 1 is an independent-DESIGN gate: it hard-fails
  "existing lane plus different prompts" architectures -- which is EXACTLY the Tier-1 vibe
  bank. Gates 5-6 are production qualification: live 30w smokes across >=2 local LLM
  families + a frontier lane, then 120w with ledger + published-asset receipts, full
  regression + Bug Bible. No authoring script can truthfully emit that PASS.

### Surface C -- Visual styles
- `nodes/visual_styles/<style_id>.json`, 9 packs, v2-only schema, lazy dir sweep on first
  resolve then cached, unknown id = hard error, forbidden-terms lint sweeps all string
  leaves, sci_fi_radio is byte-identity-pinned to extraction fixtures (copying it as a
  starter must not "fix" it).
- Fixed in-repo root (`_VISUAL_STYLES_ROOT = <pkg>/visual_styles`) -- user drops are
  git-dirty and at risk across custom-node updates. Same for `story_packs/`. This is an
  architectural boundary, not a UX preference.

### Cross-cutting laws (any design MUST preserve)
- Fail-loud, no fallbacks, unknown id = hard error. LLM-first (Python judges, never writes).
- No runtime source I/O at module import, pack discovery, or INPUT_TYPES evaluation
  (preflight Gate 2) -- constrains any refresh design.
- Combo widgets serialize the selected STRING; additions are save-safe, removals/renames are
  not; additions must never reorder `widgets_values`.
- New entries must not require touching `workflows/otr_canonical.json` (registry-driven
  selection; preflight Gate 5 pins the shipped science_news default).
- SFW: preflight bans guns/blood/violence/swearing as a creative constraint, not a censor.
- still_word lettering: per-episode locked font/lettering.

## 3. Proposed shape (r1-hardened; OPEN where marked)

**Per surface, four artifacts:** a drop-in data path for the common case; an annotated
CHECKED template; one check command; a 5-minute README recipe. Plus, panel-adopted:

1. **Two-state admission model for banks (adopted, ends the "5-minute bank vs preflight"
   contradiction):**
   - *authoring-validated* -- structural, script-checkable: schema, duplicate keys, seam
     cross-refs, registry coordinates resolve, SFW/forbidden lint, story_rules pack present.
     The check command can assert this state and emit a PARTIAL receipt that LISTS the
     unresolved hard gates by ID.
   - *production-qualified* -- only via the existing human-signed preflight with live
     evidence (smokes, 120w receipts, published asset). The script NEVER emits a production
     PASS. Qualitative gates stay human.
2. **Tier-1 bank, redefined honestly:** the complete bundle (bank row + pack + story_rules
   pack) reusing ONE blessed, proven fetcher/interpreter/pipeline triple from the closed
   registries. OPEN: which triple gets blessed (R2 selects from the live 5x4 inventory;
   `public_domain_source` lane is the natural candidate), or whether the missing
   `custom_source_bank` runner gets built as the one code item (that is Tier-2 work).
3. **Gate-1 vs Tier-1 ruling needed (operator decision, R1's biggest fork):**
   (a) a distinct lightweight gate for declared-derivative Tier-1 content banks (honest
   lane-reuse declaration replaces independence fingerprints; Gate 2 source/rights/safety
   checks kept), or (b) Tier-1 "banks" are formally CONTENT PACKS within an existing lane
   (naming change sidesteps Gate 1, which continues to govern new ENGINE lanes), or
   (c) vibe banks stay Tier-2-only (kills most of Surface B's value). Anchor + both
   panelists lean (a)/(b); (b) is cleanest against the directive's letter.
4. **Validator: ONE implementation, two entry points.** A single public command with
   subcommands (`check style | check bank | check model`) that CALLS the production
   validators (`_otr_visual_styles` load path, `_otr_story_routing`/`_otr_story_rules`
   sweeps, catalog validator) -- never a parallel schema authority. Batch output, ALL errors
   at once, file+field+fix per error, plus machine-readable JSON diagnostics so an LLM
   assistant can repair in one pass. Windows `.bat`/`.ps1` wrapper that locates the venv
   python (portable installs have non-standard interpreters).
5. **Authoring aid = annotated templates + a DERIVED LLM-ready schema doc.** Templates are
   checked fixtures (validator-green by test). The paste-to-your-assistant SCHEMA doc is
   generated/derived from the same validator+template source -- never hand-maintained
   (drift law). NO interactive scaffold CLI (both panelists cut it).
6. **Overlay decision is decide-FIRST (before any template/recipe work):** whether
   user-owned packs live in `user_packs/<surface>/` merged at scan (update-survival,
   git-clean) or stay in-repo (simpler, git-dirty). If overlay: define merge order,
   duplicate-id rejection, built-in-id protection, and an update-survival acceptance test.
   OPEN as to shape; NOT open as to timing -- the boundary is chosen in R1/R2, not after
   templates exist. Note: user packs in an overlay are invisible to repo tests -- the check
   command becomes the ONLY lint they ever see (its SFW/forbidden coverage is therefore
   mandatory, not optional).
7. **Suffix decoupling (Surface A design constraint, resolve in R2):** serialized combo
   values should be canonical ids; status decoration must not change identity. Options:
   UI-only decoration, or `VALIDATE_INPUTS` accepting suffix-mismatched-but-strippable
   values. Either way: an ID-stability test (pin selected ids across additions +
   download-state changes; reject duplicate/renamed/removed ids).
8. **Refresh loop (OPEN, per-surface facts locked):** models already live-rescan; banks +
   styles cache until restart. R1 default = keep the restart contract for banks/styles
   (lean; cache-invalidation complexity avoided). Candidate upgrade if the operator wants
   the hotter loop = stat/mtime-based invalidation in the two `_ensure_loaded` singletons.
   Any design honors the no-I/O-at-INPUT_TYPES law (a directory stat at INPUT_TYPES is the
   boundary case to rule on explicitly).

### CUT from R1 (panel-converged)
- `models.d/*.json` curated-row overlay (both panelists; HF-cache discovery + curated-stays-
  expert is the honest scope).
- Interactive scaffold CLI (both).
- Docs-as-prompts as an INDEPENDENT hand-written contract (drifts; the derived doc in #5
  is the surviving form).
- Live no-restart rescan for banks/styles as an R1 requirement (kept as the open upgrade
  path in #8).
- Generic GGUF walker (real backend work: parameterize `_otr_gguf_backend`, arbitrary
  filenames/sizes/context; explicitly OUT unless the operator opts it in as its own item).

## 4. Non-goals (R1)
- No marketplace / plugin system / GUI editor; no schema freeze or adapter layer (parked);
  no relaxation of fail-loud / preflight / SFW; no new interpreters/fetchers/engine lanes
  (unless the operator picks the custom-lane-runner option in §3.2); no touching the 720w
  bake-off lanes; video/TTS/image engine pins stay out (canonical-JSON coupling) -- operator
  ratifies this exclusion.

## 5. Acceptance criteria (testable, r1-hardened)
1. **Style:** template copy -> one check command (same code as the loader; ALL errors at
   once with file+field+fix; SFW/forbidden lint included) -> restart -> appears in dropdown.
   No .py or canonical-JSON edits.
2. **Tier-1 bank:** complete bundle from template (bank row + pack + story_rules pack,
   blessed triple) reaches *authoring-validated* + partial receipt via the check command.
   *Production-qualified* remains the human/live-evidence preflight -- explicitly NOT
   satisfiable by the script.
3. **Model (discovery, not usability):** documented recipe; a dropped HF causal-LM snapshot
   appears (bare label) in both writer dropdowns without restart; docs state UNKNOWN-tier
   treatment honestly. Optional R2 add-on: a lightweight load/generate compatibility probe;
   without it, acceptance = discovery only.
4. **Saved-workflow safety:** additions never invalidate saved workflows; ID-stability test
   pins selected ids before/after additions AND across download-state changes (the suffix
   repro); duplicate/renamed/removed ids are rejected by the check command.
5. **Failure quality:** every AUTHORING/ADMISSION failure on the three supported paths names
   the file, the field, and the fix in plain language (network outages, model
   incompatibility, and render failures are out of this promise).
6. **Update survival:** the overlay decision is made and either implemented (drops survive a
   custom-node update in an acceptance test) or explicitly deferred with operator sign-off.

## 6. Open questions (operator / R2)
1. Gate-1 vs Tier-1 ruling: (a) lightweight derivative gate, (b) "content pack" reframing,
   or (c) Tier-2-only. (Biggest fork; blocks Surface B.)
2. Overlay: `user_packs/` merge-at-scan vs in-repo, and the merge/duplicate/protection rules.
3. Which fetcher/interpreter/pipeline triple is blessed for Tier-1 -- or build the
   custom-lane runner instead?
4. Suffix decoupling: clean-id serialization vs VALIDATE_INPUTS tolerance.
5. Refresh: accept restart contract for banks/styles (default) vs stat-based invalidation?
6. Generic GGUF walker: stay out, or become its own scoped item?
7. Model compatibility probe (load/generate smoke): R2 add-on or cut?

## 7. Next step
Operator directed r1 only -- STOP here. When cleared: r2 coding plan via kibitz (local
panel), folding the operator's rulings on §6.1-3 first, since they shape everything else.
