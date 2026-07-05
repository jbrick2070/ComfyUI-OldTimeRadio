# KIBITZ R2 INPUT -- PHASE A JSON PROMPT EXTRACTION (coding plan)

## Round focus: r2 -- CODING PLAN / IMPLEMENTABILITY

r1 converged GO-WITH-FIXES. All 8 MUST-FIX items must be resolved in r2's
coding plan. r2 output must be per-chunk, exact-line-diff, code-ready.

Panel: Codex + Fable + Sonnet. Antigravity dropped (was stalled).

For each of the 8 MUST-FIX from r1, r2 must produce:

- MF-C1 audio-C7 object identity: pick the fix (module-level rebind vs
  merged chunk vs Phase-B-defer) with concrete before/after diff.
- MF-C2 line_composer_system 16th site: add to seam list + specify diff.
- MF-C3 real 14-seam vocabulary + labels + interpret + casting: publish
  the canonical seam table with lab-name mapping.
- MF-C4 extractor helper signature: exact Python signature + return
  contract + failure modes.
- MF-C5 baseline SHA pinning: refresh sibling production_mirror to
  a7bdc42d or accept drift; specify commands.
- MF-C6 empty-science-overrides pattern: publish per-pack JSON diffs.
- MF-C7 scope surgery: rewrite anchor sections that pull Phase B in.
- MF-C8 spec self-inconsistency: publish canonical 14-seam schema.

Plus SF-C1..SF-C6 folded.

## Repos to grep

- Production OTR: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio` at `a7bdc42d` (docs tip `c98a67ab`).
- Lab: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab` at `7df7c80`.

## What r2 must produce

Per-chunk plan with, for each of the (recommend 6-8) chunks:

- Chunk name + one-line intent
- Files touched (repo, path, before/after content or line-level diff)
- Test additions (file, name, exact assertion)
- Regression command (verbatim PowerShell + pytest)
- Commit + push discipline (per-chunk, byte-identical audio green)

Include a concrete seam table (site key + Python source file:line at
a7bdc42d + JSON destination in the lab + byte-identity strategy per
site: empty-override vs literal-move vs merged-chunk).

Return your review in VERDICT / MUST-FIX / SHOULD-FIX format with a
grounding table. Every claim CONFIRMED / MISREAD / UNVERIFIABLE.

---

# CURRENT PLAN STATE (input to r2 review)

Source: `docs\2026-07-04-json-prompt-transplant\kibitz\pass01_plan.md`

# pass01_plan.md -- r1 (ARC) synthesis

**Round:** r1 -- high-level arc / creative coherence.
**Panel:** Codex (`codex exec`) + Fable (via Agent) + Sonnet (via Agent) +
Claude anchor. Antigravity dropped -- stalled at 0-CPU / 0-log for 8+ min,
killed per operator directive.
**Grounding:** verified against ``ComfyUI-OldTimeRadio`` @ ``a7bdc42d``
(production) and ``ComfyUI-OTR-UpstreamStoryLab`` @ ``7df7c80`` (lab).
**Docs commit tip:** ``6d793d40`` (SUPERSEDED note + real-anchor r1 input).

## Convergent verdict

All three review panelists + Claude anchor: **GO-WITH-FIXES**.

Consensus points across the panel:

- Phase A / Phase B carve is architecturally right, but the Phase A doc as
  written pulls in Phase B machinery it shouldn't.
- Sci-fi correctly treated as a profile (not ripped) via the empty-science-
  overrides pattern already shipped as
  ``tests/test_transplant_modules.py:70-77``
  ``test_science_profile_leaves_style_picker_constants``.
- Fable's 2026-07-02 four MUST-FIX items are all resolved at ``7df7c80``
  (catalogs.py deleted; profiles.py routes packs; bridge.py emits dual
  mirrors; archival_documentary re-keyed to four production roles).
- The lab's ``registry.py`` + ``profiles.py`` + ``bridge.py`` are the
  correct architecture for Phase B; Phase A should adopt a subset, not
  invent a parallel ``get_prompt()`` helper.

## MUST-FIX (block r2 convergence) -- CONVERGED

### MF-C1. Audio-C7 object-identity contract (Fable MF2, load-bearing)

Verified at ``nodes/_otr_outline.py:1846-1847``:

```python
# If the resolver returned the legacy _SYSTEM_PROMPT verbatim (object
# identity), no overlay -- modern profile.
if resolved is _SYSTEM_PROMPT:
    period_system_overlay = None
else:
    period_system_overlay = resolved
```

And at ``nodes/_otr_creative_prompt_router.py:43-64``:

- Line 45: ``from ._otr_line_composer import (_SYSTEM_PROMPT as _MODERN_LINE_COMPOSER_SYSTEM,)``
- Line 47: ``from ._otr_outline import _SYSTEM_PROMPT as _MODERN_OUTLINE_SYSTEM``
- Line 55: ``Phase = Literal["outline", "line_composer_system"]``
- Lines 60-63: ``_MODERN_BY_PHASE: dict[str, str] = {"outline": _MODERN_OUTLINE_SYSTEM, "line_composer_system": _MODERN_LINE_COMPOSER_SYSTEM}``
- Comment at :57-60: *"Built at module-import time from the four per-phase
  constants so the returned references are object-identity stable across
  calls (preserves the Sprint D audio C7 contract under default config)."*

Any Phase A chunk that deletes or rebinds ``_SYSTEM_PROMPT`` from
``_otr_outline.py`` or ``_otr_line_composer.py`` BREAKS this identity
check silently. The router will import a stale/rebound object; the
outline's ``resolved is _SYSTEM_PROMPT`` check will fail; the modern
prompt gets prepended as a period overlay; prompt bytes drift; audio
changes.

**Fix (locked for r2 chunks):**

- Extraction MUST keep ``_SYSTEM_PROMPT`` bound at module level as a
  singleton reference to the loader's returned string, or
- Merge extraction of the outline system + router refactor into one
  atomic chunk (a "load-and-rebind" chunk that updates both files
  together), or
- Do NOT extract these two sites in Phase A -- defer to Phase B where
  the router is rebuilt end-to-end.

r2 MUST pick one. Ship a byte-equality pytest that stamps the returned
string ID + full byte comparison against a pre-Phase-A snapshot for
both ``outline`` and ``line_composer_system`` phases.

### MF-C2. `line_composer_system` is the 16th site (Fable MF1 + Codex MF3)

Verified at ``nodes/_otr_line_composer.py:1174``:
``_SYSTEM_PROMPT = """\ You write one spoken line for a character in a
radio drama..."`` and at router
``_otr_creative_prompt_router.py:55``: ``Phase = Literal["outline",
"line_composer_system"]``.

The plan's 15-site table treats the line-composer's :1621 grounding rider
and :3275 news_coda_system, but MISSES the :1174 creative system prompt
that the router already routes. Fixing MF-C1 without adding this site is
incomplete.

**Fix:** add ``line_composer_system`` to the extraction scope (16th
site). Same audio-C7 identity guarantee as MF-C1.

### MF-C3. Real seam vocabulary vs plan's loose "12" (Sonnet MF1)

Verified at ``contracts.py:25-42`` -- ``TEMPLATE_SEAMS`` is 14 entries:

```python
TEMPLATE_SEAMS = (
    "outline_system", "pitch_room_system", "story_select_system",
    "dramatic_state_system", "line_grounding", "coda_system",
    "title_system", "style_pick_inventor", "style_pick_chooser",
    "style_pick_chooser_user_template",
    # experimental adaptive-cleanup CUT to docs-only:
    "pass_1_creative_story", "pass_2_creative_ledger_fill",
    "pass_3_technical_schema_cleanup", "pass_4_technical_ledger_audit",
)
```

Plan's "12 seams" is looser than the real code shape. Real disjoint
Phase A vocabulary is:

- **10 template seams** (14 - 4 experimental) from ``TEMPLATE_SEAMS``:
  outline_system, pitch_room_system, story_select_system,
  dramatic_state_system, line_grounding, coda_system, title_system,
  style_pick_inventor, style_pick_chooser,
  style_pick_chooser_user_template
- **Plus 4 seams to ADD to ``TEMPLATE_SEAMS`` in r2** (from Codex MF3 +
  Fable MF1 + Claude anchor grounding): ``outline_macro_system``,
  ``outline_phase_system``, ``outline_beat_system``,
  ``line_composer_system``
- **Plus `labels` = ``LABEL_TEMPLATE_VARIABLES`` / ``BankDefaults``
  fields** (not a template seam; per Sonnet MF1)
- **Plus `interpret` = per-bank interpreter binding** (not a template
  seam; per Fable step 6 + Sonnet MF1)
- **Plus `casting_brief` = ``StoryInputPacket.casting_brief``** (content
  field emitted by interpreter; per Sonnet grounding; per lab
  ``contracts.py:23`` comment)

**Total Phase A: 14 template seams + labels + interpret + casting_brief
(the last three via non-template mechanisms).** r2 MUST rewrite the site
table to this vocabulary.

### MF-C4. Phase A adopts a SUBSET of the lab architecture, not a new flat helper (Codex MF5 + Sonnet MF2)

Verified at ``src/upstream_story_lab/registry.py:245-307`` and
``src/upstream_story_lab/profiles.py:31-96`` -- the lab already has
``Registry.resolve()`` (4-axis) + ``profiles.resolve_profile()`` (bank-
default merge, fail-loud missing-label/coda checks) + per-seam
``string.Formatter`` template-variable validation at load
(``registry.py:47-63``).

Phase A production-side API MUST adopt a subset of that (name TBD in
r2). Recommend a read-only per-seam extractor:

```python
def get_pack_prompt_or_none(bank_id: str, seam_key: str) -> str | None
```

where ``None`` means "use the current Python literal" (empty-science-
overrides pattern). NOT a parallel ``get_prompt()``; NOT the full
resolver. r2 defines the exact signature.

### MF-C5. Baseline SHA pinning (Codex MF1)

Verified: OTR at ``a7bdc42d``; lab at ``7df7c80``; my docs tip
``6d793d40`` (docs-only after ``a7bdc42d``). No
``PRODUCTION_MIRROR_MANIFEST.md`` located at ``7df7c80`` for me to check
the mirror's pinned SHA; Fable 2026-07-02 review cites ``d48a9d76``.

**Fix (r2):** pin Phase A to production ``a7bdc42d``. If sibling's
``production_mirror/`` is at a different SHA, refresh it before r2
elaborates line-level diffs.

### MF-C6. Empty-science-overrides pattern (Codex MF2, resolved-by-pattern)

The sibling has a working test at
``tests/test_transplant_modules.py:70-77``:

```python
overrides = spp.style_picker_overrides(profile)
assert overrides == {
    "inventor_system_prompt": "",
    "chooser_system_prompt": "",
    "chooser_user_template": "",
}  # empty = production module constants stay byte-identical
```

Phase A extends this pattern to ALL 14 template seams:

- ``science_news`` pack: empty-string overrides everywhere -- production
  Python literals stay authoritative, byte-identical (satisfies MF-C1
  audio invariant).
- ``media_archive`` / ``public_domain_story`` packs: carry actual
  content (already exists at ``7df7c80``, needs r2 audit).
- ``custom_source_bank``: schema-only stub, fail-loud on unknown seam
  reference.

The ``science_news_default.json`` currently contains PARAPHRASES
(verified in prior grounding). Under MF-C6 those become empty strings
in the science-lane packs, and the paraphrase content is dropped.

### MF-C7. Scope surgery: cut Phase B machinery from Phase A doc (Codex MF4)

**CUT from Phase A** (all move to Phase B):

- Compat mirrors (``NEWS_BRIEFS_FIELDS``, ``NEWS_SEED_KEYS``,
  ``MOTION_ROLE_KEYS``, ``PRODUCTION_VISUAL_TAILS``).
- Visual policy (``VisualStylePolicy``, tail constants).
- Provenance stamping with sha256.
- Cross-product invariant tests (bank x model x pipeline x style).
- Pipeline simulation with failure injection.
- Adaptive cleanup pipeline (already docs-only in
  ``fixtures/pipelines.json:39-42``).
- Bridge artifact emit in production.
- ``_otr_ledger_input_adapter.py``.
- Runtime routing widgets on the ledger writer.
- ``workflows/otr_scifi_16gb_full.json`` edits.

### MF-C8. Spec self-inconsistency in the anchor doc (Fable MF3)

Fable flagged: the schema example includes ``news_grounding_rider`` as
a site key absent from the 15-site table; ``unknown keys are a load-
time error`` (self-contradictory); section 5 lists site 2 among "no
existing profile routing" contradicting table row 2.

**Fix (r2):** publish a canonical 14-key list; correct the section 5
routing table; drop the self-rejecting schema example.

## SHOULD-FIX (fold before r2 elaborates chunks)

### SF-C1. Vocabulary alignment lab <-> production (Fable SF1)

Names differ: lab ``coda_system`` vs plan ``coda``; lab
``line_grounding`` vs plan ``line_grounding_rider``; ``labels`` is
different at the two sides. r2 uses lab vocabulary end-to-end.

### SF-C2. `_INVENTOR_SYSTEM` variable binding (Sonnet SF1)

Verified at ``nodes/_otr_style_picker.py:296``: ``_INVENTOR_SYSTEM = ("You
are a sci-fi radio drama showrunner.")`` -- zero ``{}`` placeholders.
The runtime variables ``n_required``, ``seed_sample_block``,
``article_excerpt`` bind to ``_INVENTOR_USER_TEMPLATE``, not the system
prompt. Lab's ``SEAM_RUNTIME_VARIABLES["style_pick_inventor"]`` may
mis-attribute them. r2 audits seam-to-variable map per site.

### SF-C3. `interpret` f-string interpolation (Fable SF2 + Claude anchor MF4)

``news_interpreter.py:704-712`` interpolates ``{_MAX_CASTING_BRIEF_CHARS}``,
etc. at runtime. Two paths: keep as Python-owned formatter OR promote
caps to profile-declared template variables. r2 picks one.

### SF-C4. "No production code touched" wording (Fable SF3)

Contradicts "replace the Python literal with a loader call." Pin to
"**behavior-preserving mechanical edits only**" -- MF-C6 (empty-science-
overrides) means most production sites do not get a loader call at
Phase A; only the non-science lanes wire in.

### SF-C5. Compat mirrors are Phase B (Sonnet SF2)

State this explicitly in the Phase A chunk list so no one imports
``compat.py`` drift tests into a Phase A PR by habit.

### SF-C6. Baseline manifest visibility

If ``PRODUCTION_MIRROR_MANIFEST.md`` exists somewhere in the sibling
repo I didn't find, r2 locates it and confirms the pinned SHA against
MF-C5.

## Grounding table (this pass; all CONFIRMED unless marked)

| claim | source file:line | status |
|---|---|---|
| OTR HEAD ``a7bdc42d`` on ``v2.0-alpha`` (before my docs commits) | git rev-parse | CONFIRMED |
| Lab HEAD ``7df7c80`` on ``main`` | git rev-parse | CONFIRMED |
| Router phase list ``["outline", "line_composer_system"]`` | ``_otr_creative_prompt_router.py:55`` | CONFIRMED |
| Object identity check ``resolved is _SYSTEM_PROMPT`` | ``_otr_outline.py:1846`` | CONFIRMED |
| Sprint D audio C7 contract cited in code comment | ``_otr_creative_prompt_router.py:57-60`` | CONFIRMED |
| ``TEMPLATE_SEAMS`` has 14 entries (10 template + 4 experimental) | ``contracts.py:25-42`` | CONFIRMED |
| line-composer :1174 ``_SYSTEM_PROMPT`` | ``_otr_line_composer.py:1174`` | CONFIRMED |
| story-critic :266 ``_CRITIC_SYSTEM_PROMPT`` (out-of-scope) | ``_otr_story_critic.py:266`` | CONFIRMED |
| Empty-science-overrides pattern | ``tests/test_transplant_modules.py:70-77`` | CONFIRMED |
| Fable 2026-07-02 MF1 (catalogs.py) resolved | catalogs.py deleted at 7df7c80 | CONFIRMED |
| Fable 2026-07-02 MF2 (_BASE_VISUAL_STYLES) resolved | catalogs.py deleted | CONFIRMED |
| Fable 2026-07-02 MF3 (mirror shape) resolved | ``bridge.py:120-166`` | CONFIRMED (via Sonnet) |
| Fable 2026-07-02 MF4 (archival scene_broll) resolved | ``archival_documentary.json:20-24`` | CONFIRMED (via Sonnet + Fable panel) |
| ``PRODUCTION_MIRROR_MANIFEST.md`` presence | not found in ``production_mirror/`` at ``7df7c80`` | UNVERIFIABLE (may exist under different name; r2 hunts) |

## Panel judgment log

**Accepted from Codex:** MF1 baseline (folded to MF-C5), MF2 paraphrase
risk (folded to MF-C6), MF3 seams (folded to MF-C2 + MF-C3), MF4 scope
(folded to MF-C7), MF5 API (folded to MF-C4), SF1 Fable resolved
(status recorded), SF2 chooser template (folded to SF-C1), SF3 real
vocab (folded to MF-C3 + SF-C1), CUTs 1-4 (folded to MF-C7).

**Accepted from Fable:** MF1 16th site (folded to MF-C2), MF2
object-identity (folded to MF-C1 -- SINGLE MOST LOAD-BEARING), MF3
schema self-inconsistency (folded to MF-C8), SF1 vocab alignment
(folded to SF-C1), SF2 f-string interpret (folded to SF-C3), SF3
wording (folded to SF-C4).

**Accepted from Sonnet:** MF1 real 14-entry TEMPLATE_SEAMS (folded to
MF-C3), MF2 adopt registry subset (folded to MF-C4), SF1
``_INVENTOR_SYSTEM`` variable binding (folded to SF-C2), SF2 compat
mirrors Phase B (folded to SF-C5).

**Accepted from Claude anchor:** everything already reflected; my
initial MF1 (seam coverage) and MF4 (interpret) resolved by panel
grounding.

**Rejected / deferred:**

- None. Every claim ground-truthed to real files. Zero hallucinations.

## Delta to feed into r2

r2 input = ``pass01_plan.md`` (this file) + operator scope + the two
real anchor docs. r2 focus = coding plan: per-chunk file:line diffs
against ``a7bdc42d``, exact JSON schema deltas at ``7df7c80``,
extractor helper signature, byte-identity harness spec.

r2 explicitly must:

- Pick one of the three MF-C1 fixes (module-level rebind vs merged
  chunk vs Phase-B-defer).
- Name the 14-seam final list + 4 seams to add to ``TEMPLATE_SEAMS``.
- Name the extractor helper signature (MF-C4).
- Refresh sibling ``production_mirror/`` to ``a7bdc42d`` or accept
  drift (MF-C5).
- Publish the empty-overrides science pack diffs (MF-C6).
- Rewrite anchor sections that pull Phase B machinery in (MF-C7 +
  MF-C8).

Codex + Fable + Sonnet all re-review at r2. Panel structure unchanged.


---

# ANCHOR 1 (lab) -- R1 Architecture v2

# R1 Architecture + Coding Plan v2 - Upstream Multi-Source Story Engine

Date: 2026-07-02 (overnight run). Author: Claude Fable R1 pass.
Workspace: ComfyUI-OTR-UpstreamStoryLab (transplant workspace).
Baseline: production_mirror @ ComfyUI-OldTimeRadio d48a9d76 (SFX-free).
Prior art: v1 lab (git 41c6512), FABLE_FINAL_REVIEW_2026-07-02.md.
Write scope: THIS FOLDER ONLY. Production is not edited until the explicit
transplant chunk.

Core law (unchanged, now enforced structurally):

```text
JSON owns content and configuration.
Python owns validation, routing, execution, and fail-loud errors.
No fallbacks. No hidden models. No hidden engines.
```

## 1. The architecture in one picture

```text
banks.json ---------+
story packs (JSON) -+-> REGISTRY (loaded, validated, fail-loud)
visual styles (JSON)+        |
pipelines (JSON) ---+        v
                     resolve(source_bank, story_model, story_pipeline, visual_style)
                             |            [every id explicit or declared default;
                             |             unknown/ambiguous = hard error]
source packet (JSON/fetch) --+
                             v
                     SourceInterpreter[bank]   <- declared binding, allowlisted
                             |                    (science = news_interpreter at
                             v                     transplant; fixtures in lab)
                     StoryInputPacket
                             v
                     LedgerWritingSpec  (spec = ids + material + story input
                             |           + prompt profile + visual policy +
                             |           model plan; cross-id validated)
                             v
                     BRIDGE ARTIFACT (one frozen JSON file)
                       - ledger_writing_spec
                       - meta_mirrors: news (NewsBriefs shape), news_seed
                       - lab_state_digest + schema_version + baseline hash
                             v
                     production adapter (tomorrow's transplant chunk)
```

## 2. Axes, precisely separated

Four orthogonal axes; each is data, none may imply another silently:

- `source_bank` - where material comes from and how it is interpreted.
  Declared in `fixtures/banks.json`. Adding a bank touches zero routing code.
- `story_model` - dramatic/tonal writing lane, source-scoped. One JSON pack
  per (bank, model, pipeline). Adding a model = dropping a pack file.
- `story_pipeline` - the LLM pass structure. Declared in
  `fixtures/pipelines.json` as a named sequence of passes with per-pass slot
  roles (creative/technical), seam references, and hard budgets. Python owns
  sequencing, stop conditions, and failure reporting; JSON owns the sequence
  and prompts.
- `visual_style` - render language. One JSON policy per style. Role-keyed
  motion prompts validated against the production role vocabulary.

Bank/model/style defaults are configuration -> they live in `banks.json`
(`default_story_model`, `default_visual_style`), not in Python dicts.

## 3. No hidden models, no hidden engines - made structural

- Every LLM-touching pass in a pipeline declares its slot role
  (`creative` | `technical`) in `pipelines.json`. The spec carries the
  resolved `model_plan` so the ledger can stamp what ran. No pass may call a
  model without a declaration.
- Every Python behavior a bank needs (fetcher, interpreter) is a NAMED
  binding string in `banks.json` (e.g. `"interpreter": "fixture_media_archive"`,
  at transplant `"science_rss_news_interpreter"`). Python resolves bindings
  through one explicit allowlist registry; an undeclared or unknown binding
  is a hard error with the bank id in the message.
- The adaptive-cleanup experiment's approved technical models are a JSON
  allowlist (`pipelines.json`), enforced by Python; the cap
  (`max_cleanup_passes`) is JSON config, the stop condition is Python.
- Ledger stamps at transplant: `meta.source_bank`, `meta.story_model`,
  `meta.story_pipeline`, `meta.visual_style` (+ mirrors). Nothing runs
  unlabeled.

## 4. One prompt vocabulary: seams

v1 had two overlapping content vocabularies (pack `prompt_stages` dict AND
`StoryPromptProfile` fields). v2 defines ONE canonical seam list, matching
the production prompt sites the transplant will parameterize:

```text
interpret            (source brain -> briefs; science keeps news_interpreter)
outline_system
pitch_room_system
story_select_system
dramatic_state_system
line_grounding       (per-line instruction)
casting_brief_seam   (source/casting brief text path; craft stays shared)
coda                 (coda_mode + coda_system + examples)
title_system
style_pick_inventor
style_pick_chooser   (+ chooser_user_template)
labels               (story_form_label, source_material_label,
                      source_develop_verb, source_grounding_label,
                      key_terms_label, close_brief_label, title_form_label)
```

A story pack supplies seam content + guardrails + forbidden lists.
`StoryPromptProfile` is now a RESOLVED VIEW built by validating and merging
(pack + bank defaults). It contains no Python-authored prose. If a seam is
missing and the bank declares it required, loading fails loudly - no default
prose is invented.

Forbidden-term handling stays metadata (leakage tests scan rendered
previews; forbidden phrases are never rendered into live prompts - models
copy negated terms).

## 5. Compatibility mirrors as pinned, drift-proof contracts

`src/upstream_story_lab/compat.py` pins the production shapes:

- `NEWS_BRIEFS_FIELDS` = exact NewsBriefs field list (casting_brief,
  script_brief, news_close_brief, key_terms, source_hash, source_chars,
  prompt_version, schema_version, model_id, decoder_profile, seed, attempts,
  attempt_failures) - cited to production_mirror/nodes/news_interpreter.py.
- `NEWS_SEED_KEYS` = {headline, source, url, date, body_chars, style,
  selected_at} - cited to production_mirror/nodes/_otr_legacy_to_stage1_adapter.py.
- `MOTION_ROLE_KEYS` = {announcer, music_open, music_close, music_inter} -
  cited to production_mirror/nodes/_otr_video_engines/render_driver.py.
- `PRODUCTION_VISUAL_TAILS` = STYLE_TAIL_DEFAULT / IMAGE_GRADE_TAIL /
  RADIO_BROADCAST_TAIL / ERA_TAIL_DEFAULT strings - cited to
  production_mirror/nodes/_otr_story_brief_helpers.py. (Renamed from
  SCI_FI_TAILS in kibitz r2: the constants are genre-neutral; sci_fi_radio
  is the policy that reproduces them. Where this doc and the round finals
  disagree, the latest kibitz final supersedes.)

Drift-proofing (the advanced part): tests AST-parse the mirrored production
files and EXTRACT these shapes (NewsBriefs class fields, the news_seed dict
literal keys, `_LTX_MOTION_PROMPT_BY_ROLE` keys, the tail constants), then
assert the pinned copies match. Re-mirroring after production moves makes
any shape drift a test failure, not a silent bug. `key_terms` is always a
list (freeze invariant, _otr_ledger_freeze.py).

The bridge emits `meta_mirrors = {news: <NewsBriefs shape>, news_seed:
<seed shape>}`. `meta.news = None` degrade semantics stay a production
decision; the bridge always emits a complete mirror.

## 6. Visual policy, production-shaped

`VisualStylePolicy` v2:

- `positive_tail`, `image_grade_tail`, `broadcast_tail`, `era_tail` -
  policy-owned replacements for the four production constants.
- `allow_radio_tails`, `forbidden_terms` (leakage-test fodder, never
  rendered).
- subjects: announcer / music / scene_open / character_portrait /
  character_scene.
- `motion_prompts` keyed ONLY by `MOTION_ROLE_KEYS` (validator rejects
  unknown keys - scene_broll/background_abstract/sfx are dead roles and must
  stay dead).
- `sci_fi_radio` policy must reproduce the production tails byte-identically
  (test compares against the AST-extracted constants) - this IS the science
  visual baseline pin.

## 7. What gets CODED tonight (lab-only)

C1 `src/upstream_story_lab/` v2 package:
   - `contracts.py` - pydantic v2 models, extra=forbid: SourceMaterialPacket,
     PublicDomainSourceManifest, StoryPack (seam-complete), SourceBankSpec,
     PipelineSpec, StoryPromptProfile (resolved view), StoryInputPacket,
     VisualStylePolicy, LedgerWritingSpec (cross-id validators), BridgeArtifact.
   - `compat.py` - pinned production shapes + citations.
   - `registry.py` - loads banks/packs/styles/pipelines from fixtures/,
     duplicate/unknown/missing = hard error, zero content literals,
     binding allowlist for interpreters.
   - `interpreters.py` - fixture interpreters per bank (deterministic,
     network-free), same protocol production brains will implement.
   - `profiles.py` - pack -> profile resolution (validation only).
   - `bridge.py` - spec assembly + bridge artifact emit/refuse + mirrors.
   - `preview.py` - prompt preview rendering + pack-driven leakage scan
     (every non-science bank, not just media archive).
C2 `fixtures/` - banks.json, pipelines.json, 12 story packs (recovered from
   git v1 and extended to seam-complete; diverged v1 Python/JSON lists
   reconciled by union), 5 visual styles (motion keys re-keyed), source
   packets (fixture briefs moved INTO packets), PD source folders,
   custom-bank schema template.
C3 `tests/` - contracts, registry fail-loud/no-fallback, leakage (story +
   visual), mirror drift-proof (AST vs pinned), sci-fi tail byte-pin,
   motion-key validation, bridge emit/refuse, PD manifest safety
   (absolute/.. paths), pipeline sequencing + loud pass failure.
C4 `nodes.py` + `scripts/validate_lab.py` - ComfyUI validator/preview nodes
   and CLI runner rebuilt on v2 (choices discovered from JSON, never
   hardcoded lists).
C5 `transplant_work/` - staged production edits, lab-only:
   - NEW modules as real files (production-ready, no imports from lab):
     `_otr_source_interpreter.py` (facade; science delegates to
     news_interpreter unchanged), `_otr_story_prompt_profile.py`,
     `_otr_visual_style_policy.py`, `_otr_ledger_input_adapter.py`
     (bridge-artifact validator).
   - PATCH SPECS (exact before/after hunks, file+line cited against
     production_mirror) for the big files: OTR_LedgerScriptWriter (routing,
     title, coda call sites, RSS gate to science_news, meta stamps,
     append-only widgets), _otr_line_composer (compose_source_coda facade,
     line grounding), _otr_style_picker (override kwargs, loud non-science
     failure), _otr_story_brief_helpers + otr_meta_brief_image_prompt +
     otr_shot_lock (policy seam reads), news_interpreter (science-only
     confinement), whitelists (scripts/otr_api.py, _otr_workflow_apply.py).
   - `workflows/otr_scifi_16gb_full.json` working copy: widget-append plan
     documented, NOT applied until production tests exist (tomorrow).

Priority order tonight: C1 -> C2 -> C3 green -> C4 -> C5 as far as context
allows. Every chunk committed; single push at end.

## 7b. Fable R1 delta - upgrades the prior rounds did not have

These five are structural, cheap, and compose with the registry design.
They are what makes this architecture verifiable rather than merely tidy.

1. Template-variable validation at load time. Every seam prompt is a
   template with a DECLARED variable set per seam (e.g. line_grounding may
   reference {source_grounding_label}, {scene_premise}). The registry
   string.Formatter-parses every template in every pack and fails loudly on
   an undeclared or misspelled variable at LOAD, not mid-episode. A JSON
   content system without this ships prompt typos silently.
2. Auditable resolution. `resolve()` returns a Resolution record: requested
   ids, resolved ids, which defaults applied, and the source file of every
   decision. "auto" stops being invisible behavior and becomes data the
   bridge artifact carries. Kills the hidden-default class structurally.
3. Provenance stamping with content hashes. The spec/bridge carries
   `provenance = {bank, model, pipeline, visual_style, pack_sha256,
   style_sha256, banks_sha256, pipelines_sha256, lab_state_digest,
   production_baseline}`. Any episode is reproducible and auditable back to
   the exact JSON bytes that shaped it. At transplant this lands in ledger
   meta beside the four id stamps.
4. Cross-product invariant tests. The registry makes (bank x model x
   pipeline x style) enumerable, so tests assert over ALL combos: resolution
   succeeds or raises a typed error (never substitutes); non-science
   previews never contain science/news terms; every template formats
   cleanly against a fixture context; every declared binding exists in the
   allowlist. This is the architectural proof of "no hidden fallback" -
   not example-based, exhaustive.
5. Pipeline simulation with failure injection. A FakeLLM runner executes
   any declared pipeline end-to-end (network-free) and tests inject a
   failure at each pass to assert the exact failing pass is reported and
   nothing falls back. Plus `schema_version` on every fixture; the registry
   refuses versions it does not know. No silent migration.

Considered and rejected (deliberately, for this system's scale): agent-graph
DSLs, content-addressed prompt stores, effect-system-style capability
tokens, pack inheritance/merge trees. Each adds machinery a one-operator
local pipeline does not need; rejection keeps "clean" honest. Single-level
`extends` for packs is noted as a future option if pack count grows past ~30.

## 8. Gates (unchanged from the final review, plus two)

All gates in FABLE_FINAL_REVIEW_2026-07-02.md TEST/VALIDATION GATES, plus:

- AST drift-proof tests pass against production_mirror (new).
- sci_fi_radio visual policy byte-identical to production tails (new).

## 9. Non-goals tonight

- No production writes. No workflow JSON widget edits (plan only).
- No LLM calls in lab tests (fixture interpreters are deterministic).
- No custom_source_bank implementation beyond schema + fail-loud stub.
- No adaptive-cleanup implementation (declared in pipelines.json as
  documented-but-disabled experiment; loading it runs nothing).


---

# ANCHOR 2 (lab) -- R1-R4 Rewrite

# R1-R4 Rewrite - JSON Content, Python Behavior

Core law:

```text
JSON owns content and configuration.
Python owns validation, routing, execution, and fail-loud errors.
```

This replaces the fuzzier architecture discussion. The upstream rewrite should
be judged by whether it keeps prompt/story/visual content in editable JSON packs
while Python stays a strict loader, validator, router, and executor.

## R1 - Architecture Rule

The system should treat source/story/visual material as data, not hardcoded
behavior.

JSON owns:

- story prompt stage text
- source-bank examples and manifests
- story model tone rules
- forbidden leakage terms
- visual style prompt tails
- visual forbidden terms
- visual motion prompt language
- default source/story/style configuration when it is pure configuration

Python owns:

- schema contracts
- JSON loading
- source-bank routing
- story-model routing
- visual-style routing
- default resolution
- ledger-writing spec construction
- validation and clear errors
- execution of story/prompt passes

Rule of thumb:

If changing a tone, prompt, style, source pack, forbidden term, or example needs
a Python edit, the design is drifting wrong. If changing the schema or execution
flow needs only JSON edits, the design is also drifting wrong.

## R2 - Coding Plan

Keep the current standalone lab shape and make it stricter.

Content/config files:

- `fixtures/story_packs/**/*.json`
- `fixtures/visual_styles/*.json`
- `fixtures/source_packets/*.json`
- `fixtures/public_domain_sources/**/manifest.json`
- future custom source-bank schema JSON files

Python behavior files:

- `contracts.py`: Pydantic models and consistency validators
- `catalogs.py`: lookup, default resolution, source/style registration
- `preview.py`: source packet -> interpreted story input -> ledger-writing spec
- `nodes.py`: ComfyUI preview/validation nodes and fail-loud UI surface
- `scripts/validate_lab.py`: command-line validation
- `tests/*.py`: automated proof that JSON content routes correctly

Concrete coding rules:

- No prompt meat should be buried inside Python unless it is genuinely generated
  by algorithm.
- No hidden fallback from media archive or public domain back to sci-fi.
- Unknown source bank, story model, story pipeline, or visual style must error.
- New JSON style packs should appear through the loader without editing node UI
  code.
- New JSON story packs should validate through the same `StoryPack` contract.
- The experimental 4-pass pipeline may be JSON-described, but Python must own
  pass sequencing, pass status, and pass failure reporting.

## R3 - Wiring And Transplant Plan

Before touching production OTR, the standalone lab should emit a bridge artifact
that production can consume.

Bridge artifact should include:

- `source_bank_id`
- `story_model_id`
- `story_pipeline_id`
- `visual_style_id`
- validated `source_material`
- validated `story_input`
- validated `prompt_profile`
- validated `visual_policy`
- compatibility mirror for current `meta.news` consumers
- explicit error if any required content/config JSON is missing or invalid

Transplant rule:

Production code should not ask, “Is this sci-fi, media archive, or public
domain?” in scattered conditionals. It should consume the validated spec and
route by declared ids and contracts.

Production edits should focus on:

- replacing hardcoded sci-fi/news prompt strings with profile fields
- replacing hardcoded cinematic/radio visual tails with visual policy fields
- proving `meta.news` compatibility against the real downstream ledger consumer
- adding any new widgets only at the end of production node widget lists
- updating the canonical workflow JSON only after code and validation are green

## R4 - Convergence Gates

The rewrite is ready to transplant only when these are true:

- JSON story packs contain the actual media archive and public-domain prompt
  content.
- JSON visual styles contain the actual archive/anime/cartoon/origami visual
  content.
- Python validates every JSON pack through strict contracts.
- Python can build a ledger-writing spec for science news, media archive, and
  public domain.
- Media archive and public domain default to non-sci-fi visual policy unless
  explicitly overridden.
- The experimental 4-pass path reports the exact failing pass.
- Tests prove media archive/public-domain do not silently fall back to sci-fi.
- Tests prove forbidden sci-fi/news terms do not leak into non-sci-fi prompt
  previews.
- Production `meta.news` compatibility is verified against the actual consumer
  code, not guessed.
- `otr_scifi_16gb_full.json` remains untouched until the transplant chunk.

## One-Sentence Version

Put the creative material in JSON, make Python enforce the contract, and let
production consume one validated ledger-writing spec instead of inheriting more
hidden sci-fi conditionals.
